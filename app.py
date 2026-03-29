import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ---------- Env & clients ----------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
TOP_K = int(os.getenv("TOP_K", "8"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

app = FastAPI(title="VikramGPT Gateway", version="1.0")

oc = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create serverless index if missing (safe no-op if it exists already)
def ensure_index(name: str):
    existing = {i["name"] for i in pc.list_indexes()}
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=1536,  # text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2"),
        )
ensure_index(PINECONE_INDEX)
index = pc.Index(PINECONE_INDEX)

# ---------- Models ----------
class Cit(BaseModel):
    doc_id: Optional[str] = None
    title: Optional[str] = None
    s3_path: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class AskRequest(BaseModel):
    query: str
    mode: Optional[str] = "concise"

class AskResponse(BaseModel):
    answer: str
    citations: List[Cit] = []

# ---------- Helpers ----------
EMBED_MODEL = "text-embedding-3-small"

def embed(text: str) -> List[float]:
    """
    Create an embedding with the OpenAI SDK and return the vector.
    Raises a ValueError if anything looks off.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty query")

    resp = oc.embeddings.create(model=EMBED_MODEL, input=text)
    vec = resp.data[0].embedding if resp and resp.data else None
    if not vec or not isinstance(vec, list) or len(vec) == 0:
        raise ValueError("Failed to create embedding vector")
    return vec

def build_answer(query: str, contexts: List[dict], mode: str) -> str:
    """
    Very simple answer composer; swap with an LLM call if you want.
    """
    if not contexts:
        return (
            "I couldn’t find anything relevant in your private index for that question. "
            "If you add more docs or provide more detail, I’ll try again."
        )
    bullets = []
    for m in contexts[:3]:
        md = m.get("metadata", {}) or {}
        title = md.get("title") or md.get("doc_id") or "Untitled"
        bullets.append(f"- {title} (score: {m.get('score'):.3f})")
    prefix = "Here’s a concise answer based on your private docs:\n" if mode == "concise" else "Here’s what I found:\n"
    return prefix + "\n".join(bullets)

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "VikramGPT Gateway is live 💥", "index": PINECONE_INDEX, "endpoints": ["/healthz", "/ask"]}

@app.get("/healthz")
def healthz():
    try:
        _ = pc.list_indexes()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    # 1) Make an embedding for the user query
    try:
        vec = embed(body.query)
    except Exception as e:
        return AskResponse(
            answer=f"Embedding failed: {e}. Please try a different query.",
            citations=[]
        )

    # 2) Query Pinecone with that vector
    try:
        res = index.query(
            vector=vec,
            top_k=TOP_K,
            include_metadata=True,
        )
    except Exception as e:
        # This is the error you saw earlier; surfacing it cleanly helps
        return AskResponse(
            answer=f"Pinecone query failed: {e}",
            citations=[]
        )

    matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    # 3) Build citations
    citations: List[Cit] = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        citations.append(
            Cit(
                doc_id=md.get("doc_id"),
                title=md.get("title"),
                s3_path=md.get("s3_path"),
                page=md.get("page"),
                score=float(m.get("score")) if isinstance(m, dict) else float(getattr(m, "score", 0.0)),
            )
        )

    # 4) Compose an answer (replace with an LLM call if you want richer responses)
    answer = build_answer(body.query, matches, body.mode or "concise")
    return AskResponse(answer=answer, citations=citations)

# ---------- Uvicorn entry ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
