import os
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import ConfigLoader
from src.ingest.loader import get_loader
from src.chunking.splitter import TextSplitter
from src.indexing.vector_store import VectorStoreWrapper

def main():
    parser = argparse.ArgumentParser(description="RAG ChatBot: Data Ingestion Pipeline")
    parser.add_argument("--config", type=str, default="config/local.yaml", help="Path to configuration file")
    parser.add_argument("--step", type=str, default="all", choices=["all", "load", "chunk", "embed"], help="Pipeline step to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config}...")
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    # 2. Setup Components
    # vector_store = VectorStoreWrapper(config)
    # vector_store.initialize(embedding_function=None) # Mock embedding for now
    
    splitter = TextSplitter(
        chunk_size=config.get("chunking", {}).get("chunk_size", 1000),
        chunk_overlap=config.get("chunking", {}).get("chunk_overlap", 200)
    )

    # 3. Load Data
    raw_data_path = config.get("paths.raw_data", "data/files")
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path, exist_ok=True)
        print(f"Created data directory: {raw_data_path}")
    
    files = [os.path.join(raw_data_path, f) for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]
    if args.limit:
        files = files[:args.limit]
        print(f"Limiting to first {args.limit} files.")
    
    if not files:
        print(f"No files found in {raw_data_path}. Please add some PDF/HWP files.")
        return

    all_chunks = []
    
    for file_path in files:
        try:
            # Load
            loader = get_loader(file_path)
            docs = loader.load()
            print(f"Loaded {len(docs)} documents from {os.path.basename(file_path)}")
            
            # Chunk
            chunks = splitter.split_documents(docs)
            print(f"  -> {len(chunks)} chunks generated.")
            all_chunks.extend(chunks)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 4. Store (Indexing)
    if all_chunks:
        print(f"Storing {len(all_chunks)} total chunks to Vector DB...")
        vector_store = VectorStoreWrapper(config)
        vector_store.initialize(embedding_function=None)
        vector_store.add_documents(all_chunks)
        print("Ingestion complete.")
    else:
        print("No chunks to store.")

if __name__ == "__main__":
    main()
