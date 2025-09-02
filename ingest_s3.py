import os
import boto3
import pinecone
from dotenv import load_dotenv
from openai import OpenAI

# Load env vars
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "vikramgpt")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Init clients
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)
pinecone.init(api_key=PINECONE_API_KEY)
index = pinecone.Index(PINECONE_INDEX)
client = OpenAI(api_key=OPENAI_API_KEY)

def file_already_indexed(file_key: str) -> bool:
    """Check Pinecone if file is already indexed by file_key metadata."""
    query = index.query(vector=[0.0]*1536, top_k=1, include_metadata=True, filter={"file_key": file_key})
    return len(query.matches) > 0

def embed_and_upsert(file_key: str, content: str):
    """Embed file content and upsert into Pinecone."""
    embedding = client.embeddings.create(model="text-embedding-3-small", input=content)
    vector = embedding.data[0].embedding

    index.upsert([
        {
            "id": file_key,
            "values": vector,
            "metadata": {"file_key": file_key, "text": content[:500]}
        }
    ])

def process_s3_files():
    """Ingest new/modified files from S3 into Pinecone."""
    response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME)

    if "Contents" not in response:
        print("No files in S3 bucket.")
        return

    for obj in response["Contents"]:
        file_key = obj["Key"]

        if file_already_indexed(file_key):
            print(f"Skipping {file_key} (already indexed)")
            continue

        # Download file
        s3_object = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=file_key)
        content = s3_object["Body"].read().decode("utf-8")

        print(f"Ingesting {file_key}...")
        embed_and_upsert(file_key, content)

if __name__ == "__main__":
    process_s3_files()
    print("Ingestion completed ✅")
