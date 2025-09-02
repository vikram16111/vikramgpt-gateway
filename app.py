import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

# Pinecone v3 client (package is "pinecone-client")
from pinecone import Pinecone
# OpenAI 1.x client
from openai import OpenAI

# ---------- env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
TOP_K = int(os.getenv("TOP_K", "8"))

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY environment variables.")

# ---------- clients ----------
oc = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# ---------- models ----------
class AskRequest(BaseModel):
    query: str
    mode: Optional[str] = "concise"

class Cit(BaseModel):
    doc_id: Optional[str] = None
    title: Optional[str] = None
    s3_path: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Cit]

# ---------- app ----------
app = FastAPI(title="VikramGPT Gateway", version="1.0")

@app.get("/")
def root():
    return {
        "status": "VikramGPT Gateway is live 💥",
        "index": PINECONE_INDEX,
        "endpoints": ["/healthz", "/ask"]
    }

@app.get("/healthz")
def healthz():
    try:
        _ = index.describe_index_stats()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    # NOTE: Header default must be set with '=' when using FastAPI; alias maps header name
    x_ingest_token: Optional[str] = Header(default=None, alias="x-ingest-token"),
):
    """
    Simple RAG: query Pinecone -> build context -> ask OpenAI.
    'x-ingest-token' reserved for future ACL hooks; not used yet.
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # 1) retrieve from Pinecone
    # Pinecone v3 typically expects a vector; some deployments expose text search.
    # We'll try text first; if the SDK complains, you can switch to vector search
    # by embedding 'query' with your embed model.
    try:
        res = index.query(
            top_k=TOP_K,
            include_metadata=True,
            text=query,  # works in Pinecone text-enabled indexes; harmless to try first
        )
    except TypeError:
        # fallback (for SDKs that do not accept 'text' kw): just raise a helpful error
        raise HTTPException(
            status_code=500,
            detail=(
                "This Pinecone index/API does not support text queries. "
                "Use vector search: create an embedding for the query and call index.query(vector=[...])."
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

    citations: List[Cit] = []
    context_chunks: List[str] = []

    matches = getattr(res, "matches", []) or getattr(res, "data", [])
    for m in matches or []:
        md = getattr(m, "metadata", {}) or {}
        score = getattr(m, "score", None)
        doc_id = md.get("doc_id") or md.get("id") or None
        title = md.get("title")
        s3_path = md.get("s3_path")
        page = md.get("page")
        text = md.get("text") or md.get("chunk") or ""
        if text:
            context_chunks.append(text)
        citations.append(Cit(doc_id=doc_id, title=title, s3_path=s3_path, page=page, score=score))

    if not context_chunks:
        return AskResponse(
            answer=(
                "The provided context does not contain relevant information for your query. "
                "Please refine the question or ingest more documents."
            ),
            citations=citations
        )

    # 2) build prompt
    context = "\n\n---\n\n".join(context_chunks[:TOP_K])
    system = (
        "You are VikramGPT, a precise assistant. Answer only from the provided context. "
        "If the answer is not in the context, say you don’t know."
    )
    user = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"

    # 3) generate
    try:
        completion = oc.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return AskResponse(answer=answer, citations=citations)

# For local debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
