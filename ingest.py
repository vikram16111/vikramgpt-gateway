import os, sys, boto3
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pinecone, openai

load_dotenv()

# Pinecone setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index_name = "vikramgpt-empire"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
index = pinecone.Index(index_name)

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")

def embed_and_store(text, namespace="docs"):
    emb = openai.Embedding.create(input=text, model="text-embedding-3-small")["data"][0]["embedding"]
    index.upsert(vectors=[{"id": str(hash(text)), "values": emb, "metadata": {"text": text}}],
                 namespace=namespace)
    print(f"[OK] Stored vector for text: {text[:60]}...")

def ingest_file(bucket, key, namespace="docs"):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    if key.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(body))
        for page in reader.pages:
            embed_and_store(page.extract_text(), namespace)
    else:
        embed_and_store(body.decode("utf-8"), namespace)

if __name__ == "__main__":
    bucket = os.getenv("AWS_BUCKET")
    key = sys.argv[1] if len(sys.argv) > 1 else None
    if not key:
        print("Usage: python ingest.py <s3_key>")
        sys.exit(1)
    ingest_file(bucket, key)
