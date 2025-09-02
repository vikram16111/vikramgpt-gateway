import os
import boto3
import tempfile
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# Initialize clients
s3 = boto3.client("s3",
                  aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                  aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                  region_name=AWS_REGION)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

def embed_text(text: str):
    """Generate embeddings with OpenAI"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def process_file(file_path, s3_key):
    """Read file and create embeddings"""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        embedding = embed_text(text)

        index.upsert([
            {
                "id": s3_key,
                "values": embedding,
                "metadata": {"source": s3_key}
            }
        ])
        print(f"✅ Ingested {s3_key}")
    except Exception as e:
        print(f"❌ Failed {s3_key}: {e}")

def ingest_bucket():
    """Go through S3 bucket and ingest new/updated files"""
    objects = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME).get("Contents", [])

    for obj in tqdm(objects, desc="Ingesting S3 files"):
        s3_key = obj["Key"]

        # Temporary download
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3.download_file(Bucket=AWS_BUCKET_NAME, Key=s3_key, Filename=tmp.name)
            process_file(tmp.name, s3_key)

if __name__ == "__main__":
    ingest_bucket()
