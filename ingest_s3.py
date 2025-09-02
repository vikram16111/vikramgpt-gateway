# ingest_s3.py
import os
import hashlib
from typing import List, Tuple

import boto3
from botocore.client import Config
from dotenv import load_dotenv
from tqdm import tqdm

from openai import OpenAI
from pinecone import Pinecone

# ──────────────────────────────────────────────────────────────────────────────
# Config & Clients
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))  # text-embedding-3-large -> 3072

# Chunking
MAX_CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

SUPPORTED_TEXT_EXTS = {".txt", ".md"}  # simple & robust; skip binary formats

# sanity check
required = [
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("PINECONE_API_KEY", PINECONE_API_KEY),
    ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
    ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
    ("AWS_BUCKET_NAME", AWS_BUCKET_NAME),
]
missing = [k for k, v in required if not v]
if missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
oc = OpenAI(api_key=OPENAI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def ext_of(key: str) -> str:
    key = key.lower()
    i = key.rfind(".")
    return key[i:] if i != -1 else ""

def is_supported(key: str) -> bool:
    return ext_of(key) in SUPPORTED_TEXT_EXTS

def list_all_objects(bucket: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        for item in page.get("Contents", []):
            yield item

def read_s3_text(key: str) -> Tuple[str, str]:
    """Return (text, etag) for a text object in S3."""
    obj = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
    etag = obj.get("ETag", "").strip('"')
    body = obj["Body"].read()
    try:
        text = body.decode("utf-8", errors="ignore")
    except Exception:
        text = body.decode("latin-1", errors="ignore")
    return text, etag

def chunk_text(text: str, size: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    text = text.replace("\r", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_batch(snippets: List[str]) -> List[List[float]]:
    resp = oc.embeddings.create(model=EMBED_MODEL, input=snippets)
    return [d.embedding for d in resp.data]

def already_indexed_version(doc_id: str, etag: str) -> bool:
    """
    Use a zero-vector query with a metadata filter `{doc_id, doc_etag}`.
    If any match exists, this exact S3 version was already ingested.
    """
    zero_vec = [0.0] * EMBED_DIM
    try:
        r = index.query(
            vector=zero_vec,
            top_k=1,
            include_metadata=False,
            filter={"doc_id": {"$eq": doc_id}, "doc_etag": {"$eq": etag}},
        )
        return len(r.get("matches", [])) > 0
    except Exception:
        # If index empty or filter unavailable, treat as not indexed
        return False

def vector_id(doc_id: str, etag: str, chunk_idx: int, chunk_text_val: str) -> str:
    h = hashlib.md5(chunk_text_val.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}::v:{etag[:8]}::c:{chunk_idx}::h:{h}"

# ──────────────────────────────────────────────────────────────────────────────
# Ingestion
# ──────────────────────────────────────────────────────────────────────────────
def ingest():
    objs = list(list_all_objects(AWS_BUCKET_NAME))
    if not objs:
        print("No objects found in the bucket.")
        return

    total = len(objs)
    ingested_files = 0
    skipped_files = 0
    ignored_files = 0

    print(f"Found {total} objects in s3://{AWS_BUCKET_NAME}")
    for obj in tqdm(objs, desc="Scanning S3 objects"):
        key = obj["Key"]

        # Ignore "folders" and unsupported extensions
        if key.endswith("/") or not is_supported(key):
            ignored_files += 1
            tqdm.write(f"⏭️  Ignoring (unsupported) {key}")
            continue

        # Read & version info
        try:
            text, etag = read_s3_text(key)
        except Exception as e:
            tqdm.write(f"❌ Failed to read {key}: {e}")
            continue

        # Skip if this exact version is already in the index
        if already_indexed_version(key, etag):
            skipped_files += 1
            tqdm.write(f"✅ Skipping (already indexed) {key} @ {etag}")
            continue

        # Chunk and embed
        chunks = chunk_text(text)
        if not chunks:
            skipped_files += 1
            tqdm.write(f"⚠️  No text to ingest for {key}")
            continue

        tqdm.write(f"📥 Ingesting {key} ({len(chunks)} chunks) …")
        BATCH = 64
        pending = []
        failed = False

        for i in tqdm(range(0, len(chunks), BATCH), desc=f"Embedding {key}", leave=False):
            batch = chunks[i:i + BATCH]
            try:
                vectors = embed_batch(batch)
            except Exception as e:
                tqdm.write(f"❌ Embedding failed for {key} (batch {i//BATCH}): {e}")
                failed = True
                break

            for j, (chunk, vec) in enumerate(zip(batch, vectors), start=i):
                vid = vector_id(key, etag, j, chunk)
                pending.append({
                    "id": vid,
                    "values": vec,
                    "metadata": {
                        "doc_id": key,
                        "doc_etag": etag,
                        "chunk_index": j,
                        "source": f"s3://{AWS_BUCKET_NAME}/{key}",
                    }
                })

            # Upsert per-batch to keep payloads small
            try:
                index.upsert(vectors=pending)
                pending = []
            except Exception as e:
                tqdm.write(f"❌ Upsert failed for {key} (batch {i//BATCH}): {e}")
                failed = True
                break

        if not failed:
            ingested_files += 1
            tqdm.write(f"✅ Done: {key} @ {etag}")
        else:
            tqdm.write(f"⚠️  Partial/failed ingestion for {key}")

    # Summary
    print("\n──── Ingest run summary ─────────────────────────")
    print(f"Total S3 objects:          {total}")
    print(f"Ingested (new/updated):    {ingested_files}")
    print(f"Skipped (already indexed): {skipped_files}")
    print(f"Ignored (unsupported):     {ignored_files}")
    print("Done ✅")

if __name__ == "__main__":
    ingest()
