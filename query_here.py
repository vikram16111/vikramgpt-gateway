import sys, os
from dotenv import load_dotenv
import pinecone, openai

# Load env
load_dotenv()

# Setup Pinecone & OpenAI
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index = pinecone.Index("vikramgpt-empire")
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_memory(query, top_k=5):
    """Empire Query - searches docs + chats seamlessly"""
    emb = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small"
    )["data"][0]["embedding"]

    # Search docs + chats together
    results_docs = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace="docs")
    results_chats = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace="chats")

    merged = []
    for r in results_docs["matches"] + results_chats["matches"]:
        merged.append(r["metadata"]["text"])

    # Final reasoning: blend memory + LLM
    context = "\n\n".join(merged[:top_k])
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are VikramGPT Empire, gold-medalist world-class assistant."},
            {"role": "user", "content": f"Answer using context below if relevant:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.4
    )
    return completion["choices"][0]["message"]["content"]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_here.py \"your question\"")
        sys.exit(1)

    question = sys.argv[1]
    answer = query_memory(question)
    print("\n[Empire Answer]:\n", answer)
