import os
import sys
from typing import Dict, Any
import importlib.util
import subprocess

def get_user_choice(prompt: str, options: Dict[str, str]) -> str:
    """Get user choice from a set of options."""
    print(f"\n{prompt}")
    for key, value in options.items():
        print(f"[{key}] {value}")
    
    while True:
        choice = input("\nEnter your choice (letter): ").upper()
        if choice in options:
            return options[choice]
        print("Invalid choice. Please try again.")

def configure_environment() -> Dict[str, Any]:
    """Get user configuration for the system."""
    config = {}
    
    # Ollama model selection
    ollama_options = {
        "A": "llama3.2:latest",
        "B": "gemma:2b"
    }
    config["ollama_model"] = get_user_choice("Select Ollama model:", ollama_options)
    
    # Embedding model selection
    embedding_options = {
        "A": "nomic-embed-text",
        "B": "all-minilm",
        "C": "snowflake-arctic-embed:137m"
    }
    config["embedding_model"] = get_user_choice("Select embedding model:", embedding_options)
    
    # Database selection
    db_options = {
        "A": "redis",
        "B": "chroma",
        "C": "qdrant"
    }
    config["database"] = get_user_choice("Select database:", db_options)
    
    # Chunk size selection
    chunk_options = {
        "A": "200",
        "B": "250",
        "C": "300",
        "D": "350",
        "E": "400"
    }
    config["chunk_size"] = int(get_user_choice("Select chunk size:", chunk_options))
    
    # Overlap selection
    overlap_options = {
        "A": "10",
        "B": "30",
        "C": "40",
        "D": "50",
        "E": "60",
        "F": "70"
    }
    config["overlap"] = int(get_user_choice("Select overlap:", overlap_options))
    
    return config

def update_config_files(config: Dict[str, Any]):
    """Update the configuration in the database-specific ingest and search files."""
    # Update ingest file
    ingest_path = os.path.join(os.path.dirname(__file__), f"ingest_{config['database']}.py")
    with open(ingest_path, "r") as f:
        ingest_content = f.read()
    
    # Update VECTOR_DIM based on embedding model
    vector_dims = {
        "nomic-embed-text": 768,
        "all-minilm": 384,
        "snowflake-arctic-embed:137m": 768
    }
    vector_dim = vector_dims.get(config["embedding_model"], 384)
    
    # Update the VECTOR_DIM line
    ingest_content = ingest_content.replace(
        "VECTOR_DIM = 384",
        f"VECTOR_DIM = {vector_dim}"
    )
    
    # Update the get_embedding function
    ingest_content = ingest_content.replace(
        'def get_embedding(text: str, model: str = "all-minilm")',
        f'def get_embedding(text: str, model: str = "{config["embedding_model"]}")'
    )
    
    # Update chunk size and overlap in split_text_into_chunks
    ingest_content = ingest_content.replace(
        "def split_text_into_chunks(text, chunk_size=300, overlap=50):",
        f'def split_text_into_chunks(text, chunk_size={config["chunk_size"]}, overlap={config["overlap"]}):'
    )
    
    with open(ingest_path, "w") as f:
        f.write(ingest_content)
    
    # Update search file
    search_path = os.path.join(os.path.dirname(__file__), f"search_{config['database']}.py")
    with open(search_path, "w",encoding="utf-8") as f:
        search_content = f.read()
    
    # Update VECTOR_DIM
    search_content = search_content.replace(
        "VECTOR_DIM = 384",
        f"VECTOR_DIM = {vector_dim}"
    )
    
    # Update get_embedding function
    search_content = search_content.replace(
        'def get_embedding(text: str, model: str = "all-minilm")',
        f'def get_embedding(text: str, model: str = "{config["embedding_model"]}")'
    )
    
    # Update Ollama model in generate_rag_response
    search_content = search_content.replace(
        'model="llama3.2:latest"',
        f'model="{config["ollama_model"]}"'
    )
    
    with open(search_path, "w") as f:
        f.write(search_content)

def main():
    print("\nWelcome to the RAG System Configuration!")
    print("This will help you set up your RAG system with your preferred settings. Press ^C to quit at any time.")
    
    # Get user configuration
    config = configure_environment()
    
    # Update configuration files
    print("\nUpdating configuration files...")
    update_config_files(config)
    
    print("\nConfiguration complete! You can now:")
    print(f"1. Run ingest_{config['database']}.py to process your PDFs")
    print(f"2. Run search_{config['database']}.py to query your documents")
    
    while True:
        choice = input("\nWhat would you like to do? (1: Ingest, 2: Search, c: Change Parameters, q: Quit): ")
        if choice.lower() == "q":
            break
        elif choice == "1":
            subprocess.run([sys.executable, f"main/ingest_{config['database']}.py"])
        elif choice == "2":
            subprocess.run([sys.executable, f"main/search_{config['database']}.py"])
        elif choice.lower() == "c":
            config = configure_environment()
            update_config_files(config)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 