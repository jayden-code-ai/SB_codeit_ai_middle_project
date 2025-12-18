import argparse
import json
import os
import sys
from tqdm import tqdm
from rouge_score import rouge_scorer

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.config import config
from src.indexing.vector_store import VectorStoreWrapper
from src.generation.rag import RAGChain

def main():
    parser = argparse.ArgumentParser(description="RAG ChatBot: Evaluation")
    parser.add_argument("--input", type=str, default="data/test_set.json", help="Input JSON file with questions and ground truths")
    parser.add_argument("--output", type=str, default="data/eval_results.json", help="Output results file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        # Create a dummy test set if not exists, for demonstration
        dummy_data = [
            {"question": "이 사업의 예산은 얼마야?", "ground_truth": "예산 정보는 문서에 따라 다르지만 보통 금액으로 표시됩니다."},
        ]
        with open(args.input, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=2)
        print(f"Created dummy test set at {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    # Initialize RAG
    print("Initializing RAG...")
    vector_store = VectorStoreWrapper(config)
    vector_store.initialize()
    rag_chain = RAGChain(config=config, vector_store_wrapper=vector_store)

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    total_score = 0

    print(f"Evaluating {len(test_set)} items...")
    for item in tqdm(test_set):
        question = item.get("question")
        ground_truth = item.get("ground_truth")

        if not question:
            continue

        # Retrieve & Generate
        # We need to manually invoke retrieve+generate pending RAGChain refactor
        # Or assumes generate_answer handles it if we pass docs.
        # Let's do the same as API
        docs = rag_chain.retriever.invoke(question)
        answer = rag_chain.generate_answer(question, docs)

        score = 0.0
        if ground_truth:
            scores = scorer.score(ground_truth, answer)
            score = scores['rougeL'].fmeasure
        
        total_score += score
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "rouge_l": score
        })

    avg_score = total_score / len(test_set) if test_set else 0
    print(f"Average ROUGE-L Score: {avg_score:.4f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
