import os, pinecone, openai
from dotenv import load_dotenv

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index = pinecone.Index("vikramgpt-empire")

def query_memory(q, namespace="docs"):
    emb = openai.Embedding.create(input=q, model="text-embedding-3-small")["data"][0]["embedding"]
    res = index.query(vector=emb, top_k=3, include_metadata=True, namespace=namespace)
    print("🔍 Results:")
    for match in res["matches"]:
        print("-", match["metadata"]["text"])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
    else:
        query_memory(sys.argv[1])
