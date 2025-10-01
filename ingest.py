# ingest.py (Lambda trigger equivalent for local testing)
import os, pinecone, openai, boto3
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")

index_name = "vikramgpt-empire"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
index = pinecone.Index(index_name)

def vectorize_text(text, namespace="docs"):
    embedding = openai.Embedding.create(input=text, model="text-embedding-3-small")["data"][0]["embedding"]
    index.upsert([(namespace + "-" + str(hash(text)), embedding, {"text": text})], namespace=namespace)
    print("✅ Vector stored in Pinecone")

if __name__ == "__main__":
    print("Run inside Lambda via S3 trigger")
