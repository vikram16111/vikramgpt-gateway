import os
import hashlib
from typing import Optional, List, Annotated

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI

# Pinecone (new client style)
from pinecone import Pinecone

# =======================
# ENV
# =======================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
TOP_K = int(os.getenv("TOP_K", "8"))

# Optional (used by /ingest/s3-object)
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
INGEST_TOKEN = os.getenv("INGEST_TOKEN")  # set this in Render (Settings → Environment)

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")

# Embedding dimension for text-embedding-3-large
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))

# =======================
# APP & CLIENTS
# =======================
app = FastAPI(title="VikramGPT Gateway", version="1.0")

# OpenAI client (safe to construct at import time)
oc = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone: lazy init to avoid startup crashes
_pc: Optional[Pinecone] = None
_pc_index = None

def get_index():
    """Get (and cache) a Pinecone Index handle. Never crash import/startup."""
    global _pc, _pc_index
    if _pc_index is not None:
        return _pc_index
    try:
        if _pc is None:
            _pc = Pinecone(api_key=PINECONE_API_KEY)
        _pc_index = _pc.Index(PINECONE_INDEX)  # assumes index already exists
        return _pc_index
    except Exception as e:
        # Defer the error to the endpoint call, with clear message
        raise HTTPException(status_code=500, detail=f"Pinecone index init failed: {e}")

# =======================
# MODELS
# =======================
class AskRequest(BaseModel):
    query: str
    mode: Optional[str] = "concise"  # "concise" | "detailed"

class Cite(BaseModel):
    doc_id: Optional[str] = None
    title: Optional[str] = None
    s3_path: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Cite] = []

# =======================
# HELPERS
# =======================
def chunk_text(text: str, size: int = 1800, overlap: int = 200) -> List[str]:
    text = (text or "").replace("\r", "")
    out, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        out.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return out

def vector_id(doc_id: str, etag: str, chunk_idx: int, chunk: str) -> str:
    h = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}::v:{etag[:8]}::c:{chunk_idx}::h:{h}"

def already_indexed_version(index, doc_id: str, etag: str) -> bool:
    """Fast 'did we already index this exact version?' check."""
    try:
        zero = [0.0] * EMBED_DIM
        r = index.query(
            vector=zero, top_k=1,
            filter={"doc_id": {"$eq": doc_id}, "doc_etag": {"$eq": etag}},
        )
        return len(r.get("matches", [])) > 0
    except Exception:
        return False  # fail-open

# =======================
# ROOT / HEALTH
# =======================
@app.get("/")
def root():
    return {
        "status": "VikramGPT Gateway is live 💥",
        "index": PINECONE_INDEX,
        "endpoints": ["/healthz", "/ask", "/ingest/s3-object"],
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

# =======================
# /ASK
# =======================
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # 1) Embed the query
    emb = oc.embeddings.create(
        model="text-embedding-3-large",
        input=[query]
    ).data[0].embedding

    # 2) Retrieve from Pinecone
    index = get_index()
    results = index.query(vector=emb, top_k=TOP_K, include_metadata=True)

    # 3) Build context & citations
    snippets: List[str] = []
    citations: List[Cite] = []
    for m in results.get("matches", []):
        meta = m.get("metadata", {}) or {}
        snippet = meta.get("text") or meta.get("content") or ""
        if snippet:
            snippets.append(snippet)
        citations.append(
            Cite(
                doc_id=meta.get("doc_id"),
                title=meta.get("title"),
                s3_path=meta.get("source"),
                page=meta.get("page"),
                score=m.get("score"),
            )
        )

    if not snippets:
        return AskResponse(
            answer=(
                "The provided context does not contain information for your query. "
                "Please rephrase or upload relevant material to your S3 bucket."
            ),
            citations=citations,
        )

    context = "\n\n---\n\n".join(snippets[:TOP_K])

    # 4) Ask the LLM
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    style = (
        "Keep it concise and cite specific sources if helpful."
        if req.mode == "concise"
        else "Be thorough and structured. Cite sources."
    )

    prompt = f"""You are VikramGPT. Answer the user's question using ONLY the context.
If something is not covered in the context, say you don't know.
Write clearly. {style}

Context:
{context}

Question: {query}
"""

    chat = oc.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful assistant that only uses provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = chat.choices[0].message.content.strip()
    return AskResponse(answer=answer, citations=citations)

# =======================
# EVENT-DRIVEN S3 INGEST (secured)
# =======================
@app.post("/ingest/s3-object")
def ingest_single_s3_object(
    payload: dict,
    x_ingest_token: Annotated[Optional[str], Header(None, alias="X-INGEST-TOKEN")],
):
    """
    Body:   {"bucket":"<bucket>", "key":"path/to/file.txt"}
    Header: X-INGEST-TOKEN: <your secret>  (must match INGEST_TOKEN env var)
    Supports .txt and .md for now.
    """
    if not INGEST_TOKEN or x_ingest_token != INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    bucket = payload.get("bucket")
    key = payload.get("key")
    if not bucket or not key:
        raise HTTPException(status_code=400, detail="bucket and key required")

    if not key.lower().endswith((".txt", ".md")):
        return {"ingested": False, "reason": "unsupported extension", "key": key}

    # Read object from S3
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    etag = (obj.get("ETag") or "").strip('"')
    body = obj["Body"].read()
    try:
        text = body.decode("utf-8", errors="ignore")
    except Exception:
        text = body.decode("latin-1", errors="ignore")

    # Skip if same version already indexed
    index = get_index()
    if already_indexed_version(index, key, etag):
        return {"ingested": False, "reason": "already indexed", "key": key, "etag": etag}

    # Chunk -> embed -> upsert
    chunks = chunk_text(text)
    if not chunks:
        return {"ingested": False, "reason": "no text", "key": key}

    BATCH = 64
    pending = []
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        r = oc.embeddings.create(model="text-embedding-3-large", input=batch)
        embs = [d.embedding for d in r.data]

        for j, (chunk, vec) in enumerate(zip(batch, embs), start=i):
            vid = vector_id(key, etag, j, chunk)
            pending.append({
                "id": vid,
                "values": vec,
                "metadata": {
                    "doc_id": key,
                    "doc_etag": etag,
                    "chunk_index": j,
                    "source": f"s3://{bucket}/{key}",
                    "text": chunk,
                }
            })

        index.upsert(vectors=pending)
        pending = []

    return {"ingested": True, "chunks": len(chunks), "key": key, "etag": etag}
