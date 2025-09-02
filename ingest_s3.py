import os, io, json, hashlib, time
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from tqdm import tqdm

import boto3
from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# --- Config ---
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX", "vikramgpt")

AWS_REGION  = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
S3_BUCKET   = os.getenv("S3_BUCKET")
S3_PREFIX   = os.getenv("S3_PREFIX", "")
MANIFEST_PREFIX = os.getenv("MANIFEST_PREFIX", "_vikramgpt")
MANIFEST_KEY    = f"{MANIFEST_PREFIX.rstrip('/')}/manifest.json"

DEFAULT_ACL = [a.strip() for a in os.getenv("DEFAULT_ACL", "vikram,public").split(",") if a.strip()]

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
CHUNK_SIZE  = 3000
CHUNK_OVERLAP = 300
BATCH = 64

assert OPENAI_API_KEY and PINECONE_API_KEY and S3_BUCKET, "Missing required env vars"

# --- Clients ---
s3 = boto3.client("s3", region_name=AWS_REGION)
oc = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in [it.name for it in pc.list_indexes()]:
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
    while True:
        if pc.describe_index(INDEX_NAME).status.get("ready", False):
            break
        time.sleep(2)

index = pc.Index(INDEX_NAME)

# --- Helpers ---
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_manifest() -> Dict[str, str]:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MANIFEST_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return {}

def save_manifest(m: Dict[str, str]):
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=MANIFEST_KEY,
        Body=json.dumps(m).encode("utf-8"),
        ContentType="application/json"
    )

def list_objects() -> List[Dict]:
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf") or key.lower().endswith(".txt"):
                out.append({"Key": key, "ETag": obj["ETag"].strip("\"")})
    return out

def get_bytes(key: str) -> bytes:
    return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()

def pdf_pages(data: bytes) -> List[Tuple[int, str]]:
    r = PdfReader(io.BytesIO(data))
    pages = []
    for i, p in enumerate(r.pages, start=1):
        text = p.extract_text() or ""
        if text.strip():
            pages.append((i, " ".join(text.split())))
    return pages

def chunk_text(t: str, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    t = " ".join(t.split())
    chunks, i, n = [], 0, len(t)
    while i < n:
        j = min(i + max_chars, n)
        chunks.append(t[i:j])
        if j == n: break
        i = max(0, j - overlap)
    return chunks

def embed_batch(snippets: List[str]) -> List[List[float]]:
    resp = oc.embeddings.create(model=EMBED_MODEL, input=snippets)
    return [d.embedding for d in resp.data]

# --- Ingest ---
def ingest():
    manifest = load_manifest()
    to_process = []

    objs = list_objects()
    for o in objs:
        key, etag = o["Key"], o["ETag"]
        if manifest.get(key) == etag:
            continue   # unchanged
        to_process.append(o)

    if not to_process:
        print("✅ No changes detected. Manifest is up to date.")
        return

    upserts = []
    total_chunks = 0

    for o in to_process:
        key, etag = o["Key"], o["ETag"]
        s3_path = f"s3://{S3_BUCKET}/{key}"
        title   = os.path.basename(key)

        try:
            raw = get_bytes(key)

            if key.lower().endswith(".pdf"):
                for page_num, text in pdf_pages(raw):
                    for chunk in chunk_text(text):
                        total_chunks += 1
                        vec_id = f"{md5(s3_path)}-{page_num}-{md5(chunk)[:8]}"
                        upserts.append((vec_id, chunk, {
                            "doc_id": vec_id, "title": title, "s3_path": s3_path,
                            "page": page_num, "text": chunk, "acl": DEFAULT_ACL
                        }))

            elif key.lower().endswith(".txt"):
                text = raw.decode("utf-8", errors="ignore")
                for chunk in chunk_text(text):
                    total_chunks += 1
                    vec_id = f"{md5(s3_path)}-txt-{md5(chunk)[:8]}"
                    upserts.append((vec_id, chunk, {
                        "doc_id": vec_id, "title": title, "s3_path": s3_path,
                            "page": None, "text": chunk, "acl": DEFAULT_ACL
                    }))

            manifest[key] = etag

        except Exception as e:
            print(f"⚠️ Failed {key}: {e}")

    if not upserts:
        print("No new chunks to upsert.")
        save_manifest(manifest)
        return

    for i in tqdm(range(0, len(upserts), BATCH), desc="Upserting"):
        batch = upserts[i:i+BATCH]
        ids   = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        vecs  = embed_batch(texts)
        index.upsert(vectors=[(ids[k], vecs[k], metas[k]) for k in range(len(batch))])

    save_manifest(manifest)
    print(f"✅ Upserted {total_chunks} chunks from {len(to_process)} changed files")

if __name__ == "__main__":
    ingest()
