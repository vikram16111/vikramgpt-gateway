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

# -------------------------------
# File ingestion
# -------------------------------
def upload_and_ingest_file(local_path, namespace="docs"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"[ERROR] File not found: {local_path}")
    
    key = os.path.basename(local_path)
    s3.upload_file(local_path, bucket, key)
    print(f"[OK] Uploaded {local_path} to s3://{bucket}/{key}")

    ingest_file(bucket, key, namespace=namespace)
    print(f"[OK] Ingested {local_path} into Pinecone index under namespace '{namespace}'")

# -------------------------------
# Chat ingestion
# -------------------------------
def ingest_chat(message, namespace="chats"):
    emb = openai.Embedding.create(input=message, model="text-embedding-3-small")["data"][0]["embedding"]
    index.upsert(vectors=[{
        "id": str(hash(message)),
        "values": emb,
        "metadata": {"text": message}
    }], namespace=namespace)
    print(f"[OK] Chat stored in namespace '{namespace}': {message[:80]}...")

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ingest_here.py <local_file_path> --mode file [--namespace myspace]")
        print("  python ingest_here.py \"your chat text here\" --mode chat [--namespace myspace]")
        sys.exit(1)

    # Defaults
    mode = "file"
    namespace = None

    # Parse args
    if "--mode" in sys.argv:
        mode_idx = sys.argv.index("--mode")
        if len(sys.argv) > mode_idx + 1:
            mode = sys.argv[mode_idx + 1].lower()

    if "--namespace" in sys.argv:
        ns_idx = sys.argv.index("--namespace")
        if len(sys.argv) > ns_idx + 1:
            namespace = sys.argv[ns_idx + 1]

    # Apply defaults if not provided
    if namespace is None:
        namespace = "docs" if mode == "file" else "chats"

    # Execute
    if mode == "file":
        file_path = sys.argv[1]
        upload_and_ingest_file(file_path, namespace=namespace)
    elif mode == "chat":
        chat_text = sys.argv[1]
        ingest_chat(chat_text, namespace=namespace)
    else:
        print(f"[ERROR] Unknown mode: {mode}. Use 'file' or 'chat'.")
        sys.exit(1)
