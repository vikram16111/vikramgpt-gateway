import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
TOP_K = int(os.getenv("TOP_K", "8"))

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY")

# Clients
app = FastAPI(title="VikramGPT Gateway", version="1.0")
oc = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)  # your index must already exist

# Models
class AskRequest(BaseModel):
    query: str
    mode: Optional[str] = "concise"

class Cit(BaseModel):
    doc_id: str = ""
    title: Optional[str] = None
    s3_path: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Cit]

# Embed
def embed(text: str) -> List[float]:
    e = oc.embeddings.create(model="text-embedding-3-large", input=text)
    return e.data[0].embedding

SYSTEM = "You are VikramGPT. Use ONLY the provided context. Cite like [1], [2]."

def build_prompt(q, matches, mode):
    style = {
        "concise": "Answer in 5–10 sentences.",
        "executive": "Bullet points for an executive.",
        "legal": "Quote exact lines; be neutral."
    }.get(mode, "Answer in 5–10 sentences.")

    ctx_lines = []
    for i, m in enumerate(matches, start=1):
        md = m.get("metadata", {}) or {}
        snippet = (md.get("text", "") or "")[:2000]
        ref = f'[{i}] {md.get("title","")} {md.get("s3_path","")} p.{md.get("page","")}'
        ctx_lines.append(ref + "\n" + snippet)

    return f"{SYSTEM}\n\n# Query\n{q}\n\n# Style\n{style}\n\n# Context\n" + "\n".join(ctx_lines)

def to_citations(matches):
    out = []
    for m in matches[:5]:
        md = m.get("metadata", {}) or {}
        out.append(Cit(
            doc_id=md.get("doc_id", ""),
            title=md.get("title"),
            s3_path=md.get("s3_path"),
            page=md.get("page"),
            score=m.get("score"),
        ))
    return out

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Empty query")

    qvec = embed(q)
    res = index.query(vector=qvec, top_k=TOP_K, include_metadata=True)

    matches = [{"score": m.score, "metadata": getattr(m, "metadata", {}) or {}} for m in res.matches]
    if not matches:
        return AskResponse(answer="No context found.", citations=[])

    prompt = build_prompt(q, matches, req.mode)
    chat = oc.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = chat.choices[0].message.content.strip()
    return AskResponse(answer=answer, citations=to_citations(matches))
