import os, sys, boto3, hashlib
from dotenv import load_dotenv
import pinecone, openai

load_dotenv()

# AWS Clients
s3 = boto3.client("s3", region_name="us-east-2")
textract = boto3.client("textract", region_name="us-east-2")
bucket = os.getenv("AWS_BUCKET", "vikramgpt-memory-vault")

# Pinecone + OpenAI setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index_name = "vikramgpt-1536"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")
index = pinecone.Index(index_name)
openai.api_key = os.getenv("OPENAI_API_KEY")

# File ingestion (PDF, Images with OCR, Text)
def upload_and_ingest_file(local_path, namespace="docs"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"[ERROR] File not found: {local_path}")

    key = os.path.basename(local_path)
    s3.upload_file(local_path, bucket, key)
    print(f"[OK] Uploaded {local_path} to s3://{bucket}/{key}")

    text_chunks = []

    if local_path.lower().endswith(".pdf"):
        # Use simple text extractor for PDF
        from PyPDF2 import PdfReader
        reader = PdfReader(local_path)
        for page in reader.pages:
            text_chunks.append(page.extract_text())

    elif local_path.lower().endswith((".png", ".jpg", ".jpeg")):
        # OCR via Textract
        response = textract.analyze_document(
            Document={'S3Object': {'Bucket': bucket, 'Name': key}},
            FeatureTypes=["TABLES", "FORMS"]
        )
        extracted_text = " ".join([item["Text"] for block in response["Blocks"] if block["BlockType"] == "LINE" for item in [block]])
        text_chunks.append(extracted_text)

    else:
        with open(local_path, "r", encoding="utf-8") as f:
            text_chunks.append(f.read())

    for text in text_chunks:
        if text.strip():
            emb = openai.Embedding.create(input=text, model="text-embedding-3-small")["data"][0]["embedding"]
            index.upsert(vectors=[{"id": str(hash(text)), "values": emb, "metadata": {"text": text}}],
                         namespace=namespace)
    print(f"[OK] Ingested {local_path} into Pinecone index under namespace '{namespace}'")
    return True

# Chat ingestion
def ingest_chat(message, namespace="chats"):
    emb = openai.Embedding.create(input=message, model="text-embedding-3-small")["data"][0]["embedding"]
    index.upsert(vectors=[{"id": str(hash(message)), "values": emb, "metadata": {"text": message}}],
                 namespace=namespace)
    print(f"[OK] Chat stored in namespace '{namespace}': {message[:80]}...")
    return True

# Entrypoint
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest_here.py <mode:file|chat> <input> [--namespace myspace]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    arg = sys.argv[2]
    namespace = "docs" if mode == "file" else "chats"
    if "--namespace" in sys.argv:
        ns_idx = sys.argv.index("--namespace")
        namespace = sys.argv[ns_idx + 1]

    if mode == "file":
        upload_and_ingest_file(arg, namespace=namespace)
    elif mode == "chat":
        ingest_chat(arg, namespace=namespace)
    else:
        print(f"[ERROR] Unknown mode: {mode}. Use 'file' or 'chat'.")
