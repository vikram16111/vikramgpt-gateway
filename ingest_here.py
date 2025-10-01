import os, sys, boto3
from dotenv import load_dotenv
from ingest import ingest_file
import pinecone, openai

load_dotenv()

# AWS + S3
s3 = boto3.client("s3", region_name="us-east-2")
bucket = os.getenv("AWS_BUCKET", "vikramgpt-memory-vault")

# Pinecone + OpenAI setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index = pinecone.Index("vikramgpt-empire")
openai.api_key = os.getenv("OPENAI_API_KEY")

def upload_and_ingest_file(local_path, namespace="docs"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")
    
    key = os.path.basename(local_path)
    s3.upload_file(local_path, bucket, key)
    print(f"[OK] Uploaded {local_path} to s3://{bucket}/{key}")

    ingest_file(bucket, key, namespace=namespace)
    print(f"[OK] Ingested {local_path} into Pinecone index under namespace '{namespace}'")

def ingest_chat(message, namespace="chats"):
    emb = openai.Embedding.create(input=message, model="text-embedding-3-small")["data"][0]["embedding"]
    index.upsert(vectors=[{"id": str(hash(message)), "values": emb, "metadata": {"text": message}}],
                 namespace=namespace)
    print(f"[OK] Chat stored: {message[:80]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ingest_here.py <local_file_path> --mode file")
        print("  python ingest_here.py \"your chat text here\" --mode chat")
        sys.exit(1)

    mode = "file"
    if "--mode" in sys.argv:
        mode_idx = sys.argv.index("--mode")
        if len(sys.argv) > mode_idx + 1:
            mode = sys.argv[mode_idx + 1].lower()

    if mode == "file":
        file_path = sys.argv[1]
        upload_and_ingest_file(file_path)
    elif mode == "chat":
        chat_text = sys.argv[1]
        ingest_chat(chat_text)
    else:
        print(f"[ERROR] Unknown mode: {mode}. Use 'file' or 'chat'.")
        sys.exit(1)
