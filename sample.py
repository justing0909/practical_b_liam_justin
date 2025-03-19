## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768 
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:" # prepining key with "doc:"
DISTANCE_METRIC = "COSINE"


# Create an index in Redis
def create_hnsw_index(): # Indexing strategy
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD") # FT = Full Text
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} 
        TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """ # Asking Redis client to run this command, uses hash set
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the calculated embedding in Redis
def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {text}")


if __name__ == "__main__":
    create_hnsw_index()

    # Example texts to encode and store
    texts = [
        "Redis is an in-memory key-value database.",
        "Ollama provides efficient LLM inference on local machines.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "HNSW indexing enables fast vector search in Redis.",
        "Ollama can generate embeddings for RAG applications.",
    ] # Sample data, imagine they are individual documents. Will talk about chunking later

    for i, text in enumerate(texts): # Interate over documents then calculate embeding
        embedding = get_embedding(text)
        store_embedding(str(i), text, embedding)

    # Perform KNN
    query_text = "Efficient search in vector databases" # What you would type into GPT

    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]") #
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    ) # Query in database
    
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    print(res.docs)
