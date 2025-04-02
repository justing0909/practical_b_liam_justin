## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import pymupdf
import time
import psutil
import chromadb

# Initialize Redis connection
client = chromadb.HttpClient(host="localhost", port=8000)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the chroma vector store
def clear_chroma_store():
    print("Clearing existing Chroma store...")

    # List collection names and delete each collection
    for collection_name in client.list_collections():
        # Retrieve the collection object by name
        collection = client.get_collection(collection_name)
        client.delete_collection(collection.name)

    print("Chroma store cleared.")


# Create an HNSW index in ChromaDB
def create_chroma_collection():
    try:
        # Check if collection exists, delete if necessary
        existing_collections = [col.name for col in client.list_collections()]
        if INDEX_NAME in existing_collections:
            client.delete_collection(INDEX_NAME)

        # Create a new collection (Chroma handles vector indexing automatically)
        client.get_or_create_collection(name=INDEX_NAME)
        print(f"Collection '{INDEX_NAME}' created successfully.")

    except Exception as e:
        print(f"Error creating collection: {e}")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in ChromaDB
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    collection = client.get_or_create_collection(name=INDEX_NAME)  # Ensure collection exists
    
    # Use a unique ID for each chunk (e.g., file name, page, chunk index)
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    
    collection.add(
        ids=[doc_id],  # Unique identifier for the entry
        embeddings=[embedding],  # Store the vector embedding
        metadatas=[{"file": file, "page": page, "chunk": chunk}],  # Store metadata
    )
    
    print(f"Stored embedding for: {doc_id}")

# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir: str):
    """Processes all PDFs in a directory, extracts text, generates embeddings, and stores them in ChromaDB."""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            print(f"Processing: {file_name}...")

            try:
                text_by_page = extract_text_from_pdf(pdf_path)

                for page_num, text in text_by_page:
                    chunks = split_text_into_chunks(text)

                    for chunk_index, chunk in enumerate(chunks):
                        if not chunk.strip():  # Skip empty chunks
                            continue
                        
                        embedding = get_embedding(chunk)

                        store_embedding(
                            file=file_name,
                            page=str(page_num),
                            chunk=str(chunk_index),  # Store numeric index instead of full text
                            embedding=embedding,
                        )
                
                print(f" ✅ Processed {file_name}")

            except Exception as e:
                print(f" ❌ Error processing {file_name}: {e}")


def query_chroma(query_text: str, n_results: int = 5):
    """Queries ChromaDB using a text embedding and retrieves the most similar results."""
    collection = client.get_or_create_collection(name=INDEX_NAME)  # Ensure collection exists
    
    # Generate the embedding for the query text
    embedding = get_embedding(query_text)
    
    # Perform similarity search in Chroma
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    
    print("\nQuery Results:")
    if results["ids"]:
        for i in range(len(results["ids"][0])):  # Iterate over the retrieved documents
            doc_id = results["ids"][0][i]
            text_chunk = results["metadatas"][0][i]["chunk"]
            distance = results["distances"][0][i]
            print(f"ID: {doc_id}\nText: {text_chunk}\nDistance: {distance}\n")
    else:
        print("No results found.")


# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
def profile(func):
    def wrapper(*args, **kwargs):

        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))

        return result
    return wrapper

# instantiation of decorator function
@profile

def main():
    clear_chroma_store()

    start_time = time.perf_counter()
    create_chroma_collection()
    process_pdfs("main/PDFs")
    end_time = time.perf_counter()

    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")
    print("--------------------------------")
    print(f"TIME TO RUN: {(end_time - start_time):.4f} seconds")

    # List all collections in the ChromaDB instance
    collection_names = client.list_collections()
    print("Collections in ChromaDB:")
    for collection_name in collection_names:
        print(collection_name)

    # Query for the first few stored documents in the embedding collection
    collection = client.get_collection(INDEX_NAME)

    # Get first few documents (n=5) from the collection
    results = collection.query(
        query_embeddings=[[0] * VECTOR_DIM],  # dummy query embedding
        n_results=5  # retrieve top 5 results
    )

    print("\nTop 5 results in collection:")
    for result in zip(results["ids"][0], results["metadatas"][0], results["distances"][0]):
        doc_id, metadata, distance = result
        print(f"ID: {doc_id}, Metadata: {metadata}, Distance: {distance}")


if __name__ == "__main__":
    main()