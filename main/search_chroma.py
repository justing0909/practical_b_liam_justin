import redis
import json
import numpy as np
# from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import time
import chromadb


# Initialize Chroma client
client = chromadb.HttpClient(host="localhost", port=8000)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"

# Function to get embedding using ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Search embeddings in Chroma
def search_embeddings(query, top_k=3):
    """Search for the most similar embeddings in ChromaDB."""
    # Get the embedding for the query
    query_embedding = get_embedding(query)
    
    # Retrieve the collection from Chroma
    collection = client.get_or_create_collection(name=INDEX_NAME)  # Ensure collection exists
    
    try:
        # Perform the query to Chroma for the top K results
        results = collection.query(
            query_embeddings=[query_embedding],  # Embedding of the query
            n_results=top_k  # Number of top results to retrieve
        )

        # Process results
        top_results = [
            {
                "file": result_metadata["file"],
                "page": result_metadata["page"],
                "chunk": result_metadata["chunk"],
                "similarity": result_distance,  # Distance (similarity)
            }
            for result_metadata, result_distance in zip(results["metadatas"][0], results["distances"][0])
        ]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Text: {result['chunk']}, Similarity: {result['similarity']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    # Prepare context string using the 'chunk' key and 'similarity'
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, text: {result.get('chunk', 'Unknown text')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
    
    try:
        # Generate response using Ollama
        response = ollama.chat(
            model="gemma:2b", messages=[{"role": "user", "content": prompt}]
        )  # TODO: Change model here

        return response["message"]["content"]
    
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "Sorry, I could not generate a response."


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            print("Exiting the search interface. Goodbye!")
            break

        # Search for relevant embeddings
        start_time = time.perf_counter()
        context_results = search_embeddings(query)

        # If no relevant context is found, inform the user
        if not context_results:
            print("\nNo relevant information found.")
            print("--------------------------------")
            continue

        # Generate RAG response
        response = generate_rag_response(query, context_results)
        end_time = time.perf_counter()

        print("\n--- Response ---")
        print(response)
        print("--------------------------------")
        print(f"TIME TO RUN: {(end_time - start_time):.4f} seconds")


# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )


if __name__ == "__main__":
    interactive_search()
