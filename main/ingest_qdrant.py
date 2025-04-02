import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import os
import pymupdf
import time
import psutil

# Initialize Qdrant connection (local)
qdrant_client = QdrantClient("localhost", port=6333)

VECTOR_DIM = 768
COLLECTION_NAME = "embedding_index"

def clear_qdrant_store():
    print("Clearing existing Qdrant store...")
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    print("Qdrant store cleared.")

# Create collection in Qdrant
def create_qdrant_collection():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )
    print("Collection created successfully.")

# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store the embedding in Qdrant
def store_embedding(file: str, page: str, chunk: str, embedding: list, point_id: int):
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={"file": file, "page": page, "chunk": chunk}
            )
        ]
    )
    print(f"Stored embedding for: {chunk}")

# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

# Split text into chunks with overlap
def split_text_into_chunks(text, chunk_size=400, overlap=50):
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# Process all PDF files in a given directory
def process_pdfs(data_dir):
    point_id = 0  # Unique ID for each stored chunk
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file_name, str(page_num), chunk, embedding, point_id)
                    point_id += 1
            print(f" -----> Processed {file_name}")

# Query Qdrant
def query_qdrant(query_text: str):
    embedding = get_embedding(query_text)
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=5
    )
    print("\nQuery Results:")
    for result in search_result:
        print(f"File: {result.payload['file']}, Page: {result.payload['page']}, Chunk: {result.payload['chunk']}, Score: {result.score}\n")

# Memory profiling
def process_memory():
    return psutil.Process(os.getpid()).memory_info().rss

def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print(f"{func.__name__}: consumed memory: {mem_after - mem_before:,}")
        return result
    return wrapper

@profile
def main():
    clear_qdrant_store()
    start_time = time.perf_counter()
    create_qdrant_collection()
    process_pdfs("main/PDFs")
    end_time = time.perf_counter()
    print("\n---Done processing PDFs---\n")
    query_qdrant("What is the capital of France?")
    print(f"TIME TO RUN: {(end_time - start_time):.4f} seconds")

if __name__ == "__main__":
    main()