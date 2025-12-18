import os
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import ConfigLoader
from src.indexing.vector_store import VectorStoreWrapper
from src.retrieval.search import Retriever
from src.generation.rag import RAGChain

def main():
    parser = argparse.ArgumentParser(description="RAG ChatBot: Inference / QA")
    parser.add_argument("--config", type=str, default="config/local.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="qa", choices=["qa", "search"], help="Inference mode")
    parser.add_argument("--query", type=str, required=True, help="User query question")
    args = parser.parse_args()

    # 1. Load Config
    config_loader = ConfigLoader(args.config)
    config = config_loader.config

    # 2. Setup Vector Store & Retriever
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize(embedding_function=None) # Mock
    
    # Initialize RAG Chain (includes Re-ranking)
    rag_chain = RAGChain(config, vector_store)
    
    # 3. Retrieve
    print(f"Retrieving documents (Hybrid + Rerank) for query: '{args.query}'")
    retrieved_docs = rag_chain.retriever.invoke(args.query)
    
    if args.mode == "search":
        print(f"Top {len(retrieved_docs)} Search Results:")
        for idx, doc in enumerate(retrieved_docs):
            print(f"[{idx+1}] {doc.page_content[:100]}... (Source: {doc.metadata.get('source', 'unknown')})")
        return

    # 4. Generate Answer (QA Mode)
    answer = rag_chain.generate_answer(args.query, retrieved_docs)
    
    print("\n=== Question ===")
    print(args.query)
    print("\n=== Answer ===")
    print(answer)
    print("\n=== Sources ===")
    for doc in retrieved_docs:
        print(f"- {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    main()
