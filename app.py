import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI
import pinecone

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
TOP_K = int(os.getenv("TOP_K", "8"))

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY in environment")

# Initialize FastAPI
app = FastAPI(title="VikramGPT Gateway", version="1.0")

# OpenAI client
oc = OpenAI(api_key=OPENAI_API_KEY)

# Pinecone client
pinecone.init(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX)

# Request/response models
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
    citations: List[Cit] = []

# Root endpoint
@app.get("/")
async def root():
    return {"message": "VikramGPT Gateway is live 🚀"}

# Ask endpoint
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        # Step 1: Embed query
        emb = oc.embeddings.create(
            model="text-embedding-3-small",
            input=req.query
        )
        query_emb = emb.data[0].embedding

        # Step 2: Query Pinecone
        res = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)

        # Step 3: Build context
        context = []
        citations = []
        for m in res["matches"]:
            meta = m["metadata"]
            text = meta.get("text", "")
            context.append(text)
            citations.append(Cit(
                doc_id=m["id"],
                title=meta.get("title"),
                s3_path=meta.get("s3_path"),
                page=meta.get("page"),
                score=m["score"]
            ))
        context_str = "\n\n".join(context)

        # Step 4: Ask OpenAI
        completion = oc.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant. Mode={req.mode}"},
                {"role": "user", "content": f"Answer based on the following context:\n\n{context_str}\n\nQuestion: {req.query}"}
            ]
        )

        answer = completion.choices[0].message.content.strip()
        return AskResponse(answer=answer, citations=citations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
