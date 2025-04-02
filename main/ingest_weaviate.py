import weaviate
from weaviate.connect import ConnectionParams  # Correct import
import os
import numpy as np
import pymupdf
import time
import psutil
import ollama

# Set up the connection parameters
connection_params = ConnectionParams(
    http={
        "host": "localhost",        # Host for HTTP connection
        "port": 8080,               # Port for HTTP connection
        "secure": False             # Whether the HTTP connection is secure (True for HTTPS)
    },
    grpc={
        "host": "localhost",        # Host for GRPC connection
        "port": 50051,              # Port for GRPC connection
        "secure": False             # Whether the GRPC connection is secure (True for gRPC with SSL)
    }
)

# Initialize Weaviate client (v4)
client = weaviate.WeaviateClient(connection_params)

VECTOR_DIM = 768
CLASS_NAME = "Document"
DISTANCE_METRIC = "cosine"

# Define Weaviate schema
def create_weaviate_schema():
    try:
        # Check if class already exists
        client.schema.delete_class(CLASS_NAME)
    except weaviate.exceptions.WeaviateException:
        pass

    # Create schema for storing documents and embeddings
    client.schema.create_class({
        "class": CLASS_NAME,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"]
            },
            {
                "name": "embedding",
                "dataType": ["blob"]
            }
        ]
    })
    print("Weaviate schema created.")

# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store the embedding in Weaviate
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    data_object = {
        "text": chunk,
        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
    }
    client.data_object.create(data_object, class_name=CLASS_NAME)
    print(f"Stored embedding for chunk: {chunk}")

# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Process all PDF files in a given directory
def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

# Query Weaviate for similar embeddings
def query_weaviate(query_text: str):
    embedding = get_embedding(query_text)
    query_result = client.query.get(CLASS_NAME, ["text", "_additional {distance}"]) \
        .with_near_vector({"vector": embedding, "certainty": 0.7}) \
        .with_limit(5) \
        .do()
    
    print("")
    for result in query_result["data"]["Get"][CLASS_NAME]:
        print(f"Text: {result['text']}\nDistance: {result['_additional']['distance']}\n")

# Process memory usage
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# Decorator for profiling memory usage
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print(f"{func.__name__}: consumed memory: {mem_after - mem_before:,} bytes")
        return result
    return wrapper

@profile
def main():
    create_weaviate_schema()

    start_time = time.perf_counter()
    process_pdfs("main/PDFs")
    end_time = time.perf_counter()

    print("\n---Done processing PDFs---\n")
    query_weaviate("What is the capital of France?")
    print(f"TIME TO RUN: {(end_time - start_time):.4f} seconds")

if __name__ == "__main__":
    main()