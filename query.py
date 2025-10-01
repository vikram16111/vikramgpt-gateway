import os, sys
from dotenv import load_dotenv
import pinecone, openai

load_dotenv()

# Pinecone setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index = pinecone.Index("vikramgpt-empire")

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_memory(query, namespace="docs"):
    emb = openai.Embedding.create(input=query, model="text-embedding-3-small")["data"][0]["embedding"]
    res = index.query(vector=emb, top_k=3, include_metadata=True, namespace=namespace)
    print("🔍 Query Results:")
    for match in res["matches"]:
        print("-", match["metadata"]["text"][:200])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query.py \"your question\"")
        sys.exit(1)
    query_memory(sys.argv[1])
