import os, sys, boto3
from dotenv import load_dotenv
from ingest import ingest_file

load_dotenv()

s3 = boto3.client("s3", region_name="us-east-2")
bucket = os.getenv("AWS_BUCKET", "vikramgpt-memory-vault")

def upload_and_ingest_chat_file(local_path, namespace="docs"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")
    
    key = os.path.basename(local_path)
    s3.upload_file(local_path, bucket, key)
    print(f"[OK] Uploaded {local_path} to s3://{bucket}/{key}")

    ingest_file(bucket, key, namespace=namespace)
    print(f"[OK] Ingested {local_path} into Pinecone index under namespace '{namespace}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_here.py <local_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    upload_and_ingest_chat_file(file_path)
