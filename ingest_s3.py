# ingest_s3.py
# Ingests documents from S3 into Pinecone using OpenAI text-embedding-3-small (1536-dim).
# - Skips unchanged content (deterministic IDs include the S3 ETag)
# - Skips already-indexed chunks via Pinecone fetch()
# - Shows progress with tqdm
# Supported types: .pdf, .txt, .md (optionally .docx if python-docx is available)

import os
import io
import hashlib
import logging
from typing import List, Dict, Iterable, Optional, Tuple

from dotenv import load_dotenv
import boto3
from tqdm import tqdm

# PDF
from pypdf import PdfReader

# OpenAI (>=1.0 style client)
from openai import OpenAI

# Pinecone (new SDK)
from pinecone import Pinecone, ServerlessSpec

# --------------------------
# Config & constants
# --------------------------
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")  # REQUIRED

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # REQUIRED
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536  # text-embedding-3-small outputs 1536-d vectors

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # REQUIRED
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt-1536")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")  # optional

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))     # ~chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Batching
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "96"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

# Allowed file types
ALLOWED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("ingest")


# --------------------------
# Helpers
# --------------------------
def file_extension(key: str) -> str:
    key_low = key.lower()
    for ext in ALLOWED_EXTS:
        if key_low.endswith(ext):
            return ext
    return ""


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple character-based chunker, overlap at boundaries.
    Keeps it robust & dependency-light.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def read_pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # If any page fails, continue
            continue
    return "\n".join(parts)


def read_s3_object_to_text(s3_client, bucket: str, key: str) -> str:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    ext = file_extension(key)

    if ext == ".pdf":
        return read_pdf_bytes_to_text(body)
    elif ext in {".txt", ".md", ".markdown"}:
        # let decoding errors pass-through
        return body.decode("utf-8", errors="ignore")
    elif ext == ".docx":
        # Optional: only if python-docx is available
        try:
            import docx  # type: ignore
            doc = docx.Document(io.BytesIO(body))
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            logger.warning(f"python-docx not installed or failed to parse: {key}")
            return ""
    else:
        return ""


def ensure_index(pc: Pinecone, name: str, dim: int, cloud: str, region: str) -> None:
    names = [idx["name"] for idx in pc.list_indexes()]
    if name not in names:
        logger.info(f"Creating Pinecone index '{name}' (dim={dim})...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        # Wait for it to be ready
        while True:
            status = pc.describe_index(name).status["ready"]
            if status:
                break


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------
# Core ingestion
# --------------------------
def fetch_existing_ids(index, ids: List[str], namespace: str) -> set:
    """
    Fetch a set of ids that already exist in Pinecone (to skip duplicates).
    """
    existing = set()
    for batch in batched(ids, 1000):
        res = index.fetch(ids=batch, namespace=namespace)
        # res is a dict-like: {'vectors': {'id': {...}}}
        vectors = res.get("vectors", {}) if isinstance(res, dict) else res.vectors
        existing.update(vectors.keys())
    return existing


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """
    Embed texts (batch) using OpenAI embeddings.
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # Respect output ordering
    return [d.embedding for d in resp.data]


def upsert_vectors(index, vectors: List[Dict], namespace: str) -> None:
    for batch in batched(vectors, UPSERT_BATCH_SIZE):
        index.upsert(vectors=batch, namespace=namespace)


def ingest_bucket():
    # ---------- sanity checks
    missing = [name for name, val in [
        ("AWS_BUCKET_NAME", AWS_BUCKET_NAME),
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("PINECONE_API_KEY", PINECONE_API_KEY),
    ] if not val]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    # ---------- clients
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, PINECONE_INDEX, EMBED_DIM, PINECONE_CLOUD, PINECONE_REGION)
    index = pc.Index(PINECONE_INDEX)

    oai = OpenAI(api_key=OPENAI_API_KEY)

    # ---------- list objects (paginated)
    logger.info(f"Listing objects in s3://{AWS_BUCKET_NAME} ...")
    paginator = s3.get_paginator("list_objects_v2")
    page_iter = paginator.paginate(Bucket=AWS_BUCKET_NAME)

    total_keys = 0
    candidate_keys: List[Tuple[str, str]] = []  # (key, etag)

    for page in page_iter:
        contents = page.get("Contents", [])
        for it in contents:
            key = it["Key"]
            ext = file_extension(key)
            if not ext:
                continue
            etag = (it.get("ETag") or "").strip('"')
            candidate_keys.append((key, etag))
            total_keys += 1

    if not candidate_keys:
        logger.info("No supported files found in the bucket.")
        return

    logger.info(f"Found {len(candidate_keys)} candidate files to process.")
    processed_files = 0
    skipped_files = 0
    total_chunks = 0
    skipped_chunks = 0

    with tqdm(total=len(candidate_keys), desc="Files", unit="file") as pbar_files:
        for key, etag in candidate_keys:
            try:
                # Read & chunk
                text = read_s3_object_to_text(s3, AWS_BUCKET_NAME, key)
                if not text.strip():
                    skipped_files += 1
                    pbar_files.update(1)
                    continue

                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                if not chunks:
                    skipped_files += 1
                    pbar_files.update(1)
                    continue

                # Deterministic chunk IDs: hash(bucket/key:etag) + :chunkNo
                base_id = sha1(f"{AWS_BUCKET_NAME}/{key}:{etag}")
                chunk_ids = [f"{base_id}:{i}" for i in range(len(chunks))]

                # Skip already indexed chunks
                existing = fetch_existing_ids(index, chunk_ids, PINECONE_NAMESPACE)
                to_embed = []
                to_ids = []
                for cid, ctext in zip(chunk_ids, chunks):
                    if cid in existing:
                        skipped_chunks += 1
                        continue
                    to_embed.append(ctext)
                    to_ids.append(cid)

                # Embed & upsert (only the new ones)
                if to_embed:
                    vectors: List[Dict] = []
                    for batch_texts, batch_ids in tqdm(
                        zip(batched(to_embed, EMBED_BATCH_SIZE), batched(to_ids, EMBED_BATCH_SIZE)),
                        total=(len(to_embed) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE,
                        desc=f"Indexing {os.path.basename(key)}",
                        leave=False,
                        unit="batch"
                    ):
                        # Flatten batched() generator zips; unwrap
                        bt = list(batch_texts)
                        bi = list(batch_ids)
                        if not bt:
                            continue

                        embs = embed_texts(oai, bt)
                        for vid, vec, text_chunk in zip(bi, embs, bt):
                            vectors.append({
                                "id": vid,
                                "values": vec,
                                "metadata": {
                                    "doc_id": key,
                                    "s3_path": f"s3://{AWS_BUCKET_NAME}/{key}",
                                    "etag": etag,
                                    "chunk_chars": len(text_chunk),
                                }
                            })

                        if len(vectors) >= UPSERT_BATCH_SIZE:
                            upsert_vectors(index, vectors, PINECONE_NAMESPACE)
                            vectors.clear()

                    if vectors:
                        upsert_vectors(index, vectors, PINECONE_NAMESPACE)

                    total_chunks += len(to_embed)

                processed_files += 1

            except Exception as e:
                logger.exception(f"Failed to process {key}: {e}")

            pbar_files.update(1)

    logger.info(
        f"Done. Files processed: {processed_files}, skipped: {skipped_files}. "
        f"New chunks upserted: {total_chunks}, duplicates skipped: {skipped_chunks}."
    )


# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    logger.info(
        f"Starting ingest: bucket={AWS_BUCKET_NAME}, index={PINECONE_INDEX}, "
        f"namespace='{PINECONE_NAMESPACE}', model={EMBED_MODEL} (dim={EMBED_DIM})"
    )
    ingest_bucket()
