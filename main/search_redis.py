import redis
import json
import numpy as np
# from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import time


# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_embedding(text: str, model: str = "snowflake-arctic-embed:137m") -> list:  # TODO: change embedding model
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    query_embedding = get_embedding(query)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

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
    # Generate response using Ollama
    response = ollama.chat(
        model="gemma:2b", messages=[{"role": "user", "content": prompt}]
    )   # TODO: change model here

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        start_time = time.perf_counter()
        context_results = search_embeddings(query)

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
