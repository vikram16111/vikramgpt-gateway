# ingest_s3.py
# S3 -> (approved-only) -> chunk -> embed (text-embedding-3-small, 1536) -> Pinecone
# - Skips files not explicitly approved via S3 object tags (approved=yes by default)
# - Skips already-indexed chunks (deterministic IDs with ETag)
# - Supports PDF/TXT/MD (DOCX optional)
# - Progress via tqdm

import os
import io
import hashlib
import logging
from typing import List, Dict, Iterable, Tuple

from dotenv import load_dotenv
import boto3
from tqdm import tqdm
from pypdf import PdfReader

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # text-embedding-3-small

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt-1536")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # optional

# APPROVAL GATE (S3 object tags)
APPROVAL_TAG_KEY = os.getenv("APPROVAL_TAG_KEY", "approved")
REQUIRED_APPROVAL_VALUE = os.getenv("REQUIRED_APPROVAL_VALUE", "yes").lower()

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))      # chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200")) # overlap

# Batching
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

ALLOWED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger("ingest")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def file_ext(key: str) -> str:
    key = key.lower()
    for ext in ALLOWED_EXTS:
        if key.endswith(ext):
            return ext
    return ""


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    out: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return out


def read_pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def read_s3_object_to_text(s3, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    ext = file_ext(key)
    if ext == ".pdf":
        return read_pdf_bytes_to_text(body)
    if ext in {".txt", ".md", ".markdown"}:
        return body.decode("utf-8", errors="ignore")
    if ext == ".docx":
        try:
            import docx  # optional
            doc = docx.Document(io.BytesIO(body))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            log.warning(f"DOCX parse failed or python-docx missing: {key}")
            return ""
    return ""


def get_object_tags(s3, bucket: str, key: str) -> dict:
    try:
        resp = s3.get_object_tagging(Bucket=bucket, Key=key)
        return {t["Key"]: t["Value"] for t in resp.get("TagSet", [])}
    except Exception:
        return {}


def ensure_index(pc: Pinecone, name: str, dim: int, cloud: str, region: str) -> None:
    names = [i["name"] for i in pc.list_indexes()]
    if name not in names:
        log.info(f"Creating Pinecone index '{name}' (dim={dim}, metric=cosine)...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        # Wait until ready
        while True:
            meta = pc.describe_index(name)
            if meta.status.get("ready"):
                break


def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_existing_ids(index, ids: List[str], namespace: str) -> set:
    existing = set()
    for batch in batched(ids, 1000):
        res = index.fetch(ids=batch, namespace=namespace)
        vectors = res.get("vectors", {}) if isinstance(res, dict) else res.vectors
        existing.update(vectors.keys())
    return existing


def embed_texts(oai: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def upsert_vectors(index, vectors: List[Dict], namespace: str):
    for batch in batched(vectors, UPSERT_BATCH_SIZE):
        index.upsert(vectors=batch, namespace=namespace)


# ──────────────────────────────────────────────────────────────────────────────
# Ingest
# ──────────────────────────────────────────────────────────────────────────────
def ingest_bucket():
    # Sanity
    missing = [k for k, v in [
        ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
        ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
        ("AWS_BUCKET_NAME", AWS_BUCKET_NAME),
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("PINECONE_API_KEY", PINECONE_API_KEY),
    ] if not v]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    # Clients
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    oai = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, PINECONE_INDEX, EMBED_DIM, PINECONE_CLOUD, PINECONE_REGION)
    index = pc.Index(PINECONE_INDEX)

    # List bucket keys
    log.info(f"Listing objects in s3://{AWS_BUCKET_NAME} ...")
    paginator = s3.get_paginator("list_objects_v2")
    page_it = paginator.paginate(Bucket=AWS_BUCKET_NAME)

    candidates: List[Tuple[str, str]] = []  # (key, etag)
    for page in page_it:
        for it in page.get("Contents", []):
            key = it["Key"]
            if key.endswith("/"):
                continue
            if not file_ext(key):
                continue
            etag = (it.get("ETag") or "").strip('"')
            candidates.append((key, etag))

    if not candidates:
        log.info("No supported files found.")
        return

    processed_files = skipped_files = 0
    new_chunks = skipped_chunks = 0

    with tqdm(total=len(candidates), desc="Files", unit="file") as pbar:
        for key, etag in candidates:
            # 1) Approval gate via S3 tags
            tags = get_object_tags(s3, AWS_BUCKET_NAME, key)
            approved = tags.get(APPROVAL_TAG_KEY, "").lower() == REQUIRED_APPROVAL_VALUE
            if not approved:
                tqdm.write(f"⏭️  Skipping (not approved) {key} tags={tags}")
                skipped_files += 1
                pbar.update(1)
                continue

            # 2) Read & chunk
            try:
                text = read_s3_object_to_text(s3, AWS_BUCKET_NAME, key)
            except Exception as e:
                tqdm.write(f"❌ Read failed {key}: {e}")
                skipped_files += 1
                pbar.update(1)
                continue

            chunks = chunk_text(text)
            if not chunks:
                tqdm.write(f"⚠️  No text extracted {key}")
                skipped_files += 1
                pbar.update(1)
                continue

            # 3) Deterministic IDs = sha1('bucket/key:etag') + ':i'
            base = sha1(f"{AWS_BUCKET_NAME}/{key}:{etag}")
            ids = [f"{base}:{i}" for i in range(len(chunks))]

            # 4) Skip duplicates already in Pinecone
            existing = fetch_existing_ids(index, ids, PINECONE_NAMESPACE)
            to_embed, to_ids = [], []
            for cid, ctext in zip(ids, chunks):
                if cid in existing:
                    skipped_chunks += 1
                    continue
                to_embed.append(ctext)
                to_ids.append(cid)

            if not to_embed:
                processed_files += 1
                pbar.update(1)
                continue

            # 5) Embed + upsert
            vectors: List[Dict] = []
            batches = (len(to_embed) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
            for bi in tqdm(range(batches), desc=f"Indexing {os.path.basename(key)}", leave=False, unit="batch"):
                start = bi * EMBED_BATCH_SIZE
                end = start + EMBED_BATCH_SIZE
                batch_texts = to_embed[start:end]
                batch_ids = to_ids[start:end]

                embs = embed_texts(oai, batch_texts)
                for vid, vec, txt in zip(batch_ids, embs, batch_texts):
                    vectors.append({
                        "id": vid,
                        "values": vec,
                        "metadata": {
                            "doc_id": key,
                            "s3_path": f"s3://{AWS_BUCKET_NAME}/{key}",
                            "etag": etag,
                            "chunk_chars": len(txt),
                        }
                    })
                if len(vectors) >= UPSERT_BATCH_SIZE:
                    upsert_vectors(index, vectors, PINECONE_NAMESPACE)
                    vectors.clear()

            if vectors:
                upsert_vectors(index, vectors, PINECONE_NAMESPACE)

            new_chunks += len(to_embed)
            processed_files += 1
            pbar.update(1)

    log.info(
        f"Done. Files processed: {processed_files}, skipped files: {skipped_files}, "
        f"new chunks upserted: {new_chunks}, duplicate chunks skipped: {skipped_chunks}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(
        f"Starting ingest: bucket={AWS_BUCKET_NAME}, index={PINECONE_INDEX}, "
        f"namespace='{PINECONE_NAMESPACE}', model={EMBED_MODEL} (dim={EMBED_DIM}), "
        f"approval_tag={APPROVAL_TAG_KEY}=={REQUIRED_APPROVAL_VALUE}"
    )
    ingest_bucket()
