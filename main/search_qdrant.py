import json
import numpy as np
import ollama
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# Define collection name
COLLECTION_NAME = "embeddings"

# Check if collection exists, if not create it
def initialize_qdrant():
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

initialize_qdrant()


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generate embeddings using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    """Search for the most similar embeddings in Qdrant."""
    query_embedding = get_embedding(query)

    try:
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )

        top_results = [
            {
                "file": result.payload["file"],
                "page": result.payload["page"],
                "chunk": result.payload["chunk"],
                "similarity": result.score,
            }
            for result in search_results
        ]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Text: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    """Generate response using retrieved context from Qdrant."""
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, text: {result.get('chunk', 'Unknown text')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    print(f"context_str: {context_str}")

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="gemma:2b", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        start_time = time.perf_counter()
        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        end_time = time.perf_counter()

        print("\n--- Response ---")
        print(response)
        print("--------------------------------")
        print(f"TIME TO RUN: {(end_time - start_time):.4f} seconds")


def store_embedding(file, page, chunk, embedding):
    """Store an embedding in Qdrant."""
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=int(time.time() * 1000),  # Unique ID
                vector=embedding,
                payload={"file": file, "page": page, "chunk": chunk},
            )
        ],
    )


if __name__ == "__main__":
    interactive_search()