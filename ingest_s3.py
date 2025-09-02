import os
import io
import hashlib
from typing import List, Tuple

import boto3
from botocore.client import Config
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# -------------------------
# Env & clients
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Embedding model & dim (text-embedding-3-large = 3072)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

SUPPORTED_TEXT_EXTS = {".txt", ".md"}   # keep simple & reliable; skip binaries (jpg/png/pdf/docx etc.)
MAX_CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

assert OPENAI_API_KEY and PINECONE_API_KEY and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_BUCKET_NAME, \
    "One or more required environment variables are missing."

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

# -------------------------
# Helpers
# -------------------------
def ext_of(key: str) -> str:
    key = key.lower()
    dot = key.rfind(".")
    return key[dot:] if dot != -1 else ""

def is_supported(key: str) -> bool:
    return ext_of(key) in SUPPORTED_TEXT_EXTS

def chunk_text(text: str, size: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple character-based chunking (safe for embeddings)."""
    if not text:
        return []
    text = text.replace("\r", "")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def embed_batch(snippets: List[str]) -> List[List[float]]:
    """Embed a list of snippets with OpenAI."""
    resp = oc.embeddings.create(model=EMBED_MODEL, input=snippets)
    return [d.embedding for d in resp.data]

def already_indexed_version(doc_id: str, etag: str) -> bool:
    """
    Fast metadata filter check.
    We query with a zero vector and filter by doc_id + etag.
    If any match exists, this exact version is already ingested.
    """
    zero_vec = [0.0] * EMBED_DIM
    try:
        r = index.query(
            vector=zero_vec,
            top_k=1,
            include_metadata=False,
            filter={"doc_id": {"$eq": doc_id}, "doc_etag": {"$eq": etag}},
        )
        return len(r["matches"]) > 0
    except Exception:
        # If the index is empty or filter not supported yet, treat as not indexed
        return False

def id_for(doc_id: str, etag: str, chunk_idx: int, chunk_text_val: str) -> str:
    # Stable ID so repeated runs do no harm; also allows overwrite if text changed.
    h = hashlib.md5(chunk_text_val.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}::v:{etag[:8]}::c:{chunk_idx}::h:{h}"

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

def list_all_objects(bucket: str) -> List[dict]:
    """List all objects in a bucket using pagination."""
    paginator = s3.get_paginator("list_objects_v2")
    page_iter = paginator.paginate(Bucket=bucket)
    out = []
    for page in page_iter:
        for item in page.get("Contents", []):
            out.append(item)
    return out

# -------------------------
# Main ingestion
# -------------------------
def ingest():
    objects = list_all_objects(AWS_BUCKET_NAME)
    if not objects:
        print("No objects found in the bucket.")
        return

    total = len(objects)
    ingested = 0
    skipped = 0
    ignored = 0

    for obj in objects:
        key = obj["Key"]

        # ignore "folders" and unsupported types
        if key.endswith("/") or not is_supported(key):
            ignored += 1
            print(f"⏭️  Ignoring (unsupported) {key}")
            continue

        # read and get etag
        try:
            text, etag = read_s3_text(key)
        except Exception as e:
            print(f"❌ Failed to read {key}: {e}")
            continue

        # skip if exactly this version is already indexed
        if already_indexed_version(doc_id=key, etag=etag):
            skipped += 1
            print(f"✅ Skipping (already indexed) {key} @ {etag}")
            continue

        # chunk and embed
        chunks = chunk_text(text, MAX_CHUNK_CHARS, CHUNK_OVERLAP)
        if not chunks:
            skipped += 1
            print(f"⚠️  No text to ingest for {key}")
            continue

        # embed in sensible batch sizes to avoid payload limits
        BATCH = 64
        upserts = []
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            try:
                vectors = embed_batch(batch)
            except Exception as e:
                print(f"❌ Embedding failed for {key} (batch {i//BATCH}): {e}")
                break

            for j, (chunk, vec) in enumerate(zip(batch, vectors), start=i):
                vid = id_for(key, etag, j, chunk)
                upserts.append({
                    "id": vid,
                    "values": vec,
                    "metadata": {
                        "doc_id": key,
                        "doc_etag": etag,
                        "chunk_index": j,
                        "source": f"s3://{AWS_BUCKET_NAME}/{key}",
                    }
                })

            # Upsert per batch to keep payloads small
            try:
                index.upsert(vectors=upserts)
                upserts = []
            except Exception as e:
                print(f"❌ Upsert failed for {key} (batch {i//BATCH}): {e}")
                break
        else:
            # only increments if we didn't break the batching loop
            ingested += 1
            print(f"📥 Ingested {key} @ {etag} ({len(chunks)} chunks)")

    print("\n──── Ingest run summary ─────────────────────")
    print(f"Total S3 objects: {total}")
    print(f"Ingested (new/updated): {ingested}")
    print(f"Skipped (already indexed): {skipped}")
    print(f"Ignored (unsupported): {ignored}")
    print("Done ✅")

if __name__ == "__main__":
    ingest()
