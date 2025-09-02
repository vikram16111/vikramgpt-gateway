import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# ---------------------- Config ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
TOP_K = int(os.getenv("TOP_K", "8"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # <— set this on Render

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY")

# ---------------------- Clients ----------------------
app = FastAPI(title="VikramGPT Gateway", version="1.1")
oc = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)  # assumes exists

# ---------------------- Security ----------------------
def require_api_key(x_api_key: Optional[str] = Header(None)):
    # if ADMIN_TOKEN is set, enforce it; if not set, allow (dev mode)
    if ADMIN_TOKEN and x_api_key != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ---------------------- Models ----------------------
class AskRequest(BaseModel):
    query: str
    mode: Optional[str] = "concise"
    top_k: Optional[int] = None

class Cit(BaseModel):
    doc_id: str = ""
    title: Optional[str] = None
    s3_path: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Cit] = []

# ---------------------- Helpers ----------------------
SYSTEM = "You are VikramGPT. Use ONLY the provided context. Cite like [1], [2]."

def embed(text: str) -> List[float]:
    e = oc.embeddings.create(model="text-embedding-3-small", input=text)
    return e.data[0].embedding

def style_for(mode: str) -> str:
    return {
        "concise": "Answer in 5–10 sentences.",
        "executive": "Bullet points for an executive.",
        "legal": "Quote exact lines; be neutral and precise."
    }.get((mode or "concise").lower(), "Answer in 5–10 sentences.")

def build_prompt(q: str, matches: List[dict], mode: str) -> str:
    ctx_lines = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        snippet = (md.get("text", "") or "")[:2000]
        ref = f'[{i}] {md.get("title","")} {md.get("s3_path","")} p.{md.get("page","")}'
        ctx_lines.append(ref + "\n" + snippet)

    return (
        f"{SYSTEM}\n\n"
        f"# Query\n{q}\n\n"
        f"# Style\n{style_for(mode)}\n\n"
        f"# Context (ranked)\n" + "\n".join(ctx_lines)
    )

def to_citations(matches: List[dict]) -> List[Cit]:
    out: List[Cit] = []
    for m in matches[:5]:
        md = m.get("metadata", {}) or {}
        out.append(Cit(
            doc_id=md.get("doc_id", m.get("id", "")),
            title=md.get("title"),
            s3_path=md.get("s3_path"),
            page=md.get("page"),
            score=m.get("score"),
        ))
    return out

# ---------------------- Routes ----------------------
@app.get("/")
def root():
    return {"status": "VikramGPT Gateway is live 💫", "index": PINECONE_INDEX, "endpoints": ["/healthz", "/ask"]}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(
    req: AskRequest,
    x_user_id: Optional[str] = Header(None)  # used for ACL filtering
):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(400, "Empty query")

    try:
        qvec = embed(q)
        top_k = req.top_k or TOP_K

        pc_filter = None
        if x_user_id:
            pc_filter = {"acl": {"$in": [x_user_id, "public"]}}

        res = index.query(
            vector=qvec,
            top_k=top_k,
            include_metadata=True,
            filter=pc_filter
        )

        matches = [
            {"id": m.id, "score": m.score, "metadata": (getattr(m, "metadata", {}) or {})}
            for m in (res.matches or [])
        ]

        if not matches:
            return AskResponse(answer="No relevant context found in your index.", citations=[])

        prompt = build_prompt(q, matches, req.mode or "concise")
        chat = oc.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = chat.choices[0].message.content.strip()
        return AskResponse(answer=answer, citations=to_citations(matches))

    except Exception as e:
        raise HTTPException(500, f"Query failed: {type(e).__name__}: {e}")
