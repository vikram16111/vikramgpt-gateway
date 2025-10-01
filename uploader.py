import os, boto3, requests
from dotenv import load_dotenv
load_dotenv()

s3 = boto3.client("s3", region_name="us-east-2")
bucket = os.getenv("AWS_BUCKET", "vikramgpt-memory-vault")

def upload_file(filepath, namespace="docs"):
    filename = os.path.basename(filepath)
    key = f"{namespace}/{filename}"
    s3.upload_file(filepath, bucket, key)
    print(f"✅ Uploaded {filename} to s3://{bucket}/{key}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python uploader.py <file>")
    else:
        upload_file(sys.argv[1])
