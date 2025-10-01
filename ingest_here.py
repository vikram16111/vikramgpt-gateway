import os, sys, boto3
from dotenv import load_dotenv
import pinecone, openai
from ingest import ingest_file

# Load env
load_dotenv()

# AWS + S3
s3 = boto3.client("s3", region_name="us-east-2")
bucket = os.getenv("AWS_BUCKET", "vikramgpt-memory-vault")

# Pinecone + OpenAI setup
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-gcp")
index = pinecone.Index("vikramgpt-empire")
openai.api_key = os.getenv("OPENAI_API_KEY")

# File ingestion
def upload_and_ingest_file(local_path, namespace="docs"):
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"[ERROR] File not found: {local_path}")

    key = os.path.basename(local_path)
    s3.upload_file(local_path, bucket, key)
    print(f"[OK] Uploaded {local_path} to s3://{bucket}/{key}")
    ingest_file(bucket, key, namespace=namespace)
    print(f"[OK] Ingested {local_path} into Pinecone under namespace '{namespace}'")
    return f"File {key} ingested."

# Chat ingestion
def ingest_chat(message, namespace="chats"):
    emb = openai.Embedding.create(input=message, model="text-embedding-3-small")["data"][0]["embedding"]
    index.upsert(vectors=[{"id": str(hash(message)), "values": emb, "metadata": {"text": message}}], namespace=namespace)
    print(f"[OK] Chat stored in namespace '{namespace}': {message[:80]}...")
    return f"Chat message ingested: {message[:60]}..."

# Unified query
def query_memory(query, top_k=5):
    emb = openai.Embedding.create(input=query, model="text-embedding-3-small")["data"][0]["embedding"]
    results_docs = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace="docs")
    results_chats = index.query(vector=emb, top_k=top_k, include_metadata=True, namespace="chats")

    merged = []
    for r in results_docs["matches"] + results_chats["matches"]:
        merged.append(r["metadata"]["text"])

    context = "\n\n".join(merged[:top_k])
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are VikramGPT Empire, world-class assistant."},
            {"role": "user", "content": f"Answer using context:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.4
    )
    return completion["choices"][0]["message"]["content"]

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python ingest_here.py <local_file_path> --mode file --ask \"Your follow-up question\"")
        print("  python ingest_here.py \"Your chat text\" --mode chat --ask \"Your follow-up question\"")
        sys.exit(1)

    mode = "file"
    ask_query = None
    args = sys.argv[1:]

    if "--mode" in args:
        midx = args.index("--mode")
        mode = args[midx + 1].lower()
        args.pop(midx); args.pop(midx)

    if "--ask" in args:
        qidx = args.index("--ask")
        ask_query = args[qidx + 1]
        args.pop(qidx); args.pop(qidx)

    if mode == "file":
        file_path = args[0]
        status = upload_and_ingest_file(file_path)
    elif mode == "chat":
        chat_text = args[0]
        status = ingest_chat(chat_text)
    else:
        print("[ERROR] Unknown mode. Use 'file' or 'chat'.")
        sys.exit(1)

    print(f"\n✅ Ingestion complete: {status}")

    if ask_query:
        print(f"\n🔎 Running follow-up query: {ask_query}")
        answer = query_memory(ask_query)
        print("\n[Empire Answer]:\n", answer)
