"""
Filter out hypothetical questions from diverse evaluation dataset.

Hypothetical questions ask "what if" scenarios that require inference
beyond source documents - unfair for RAG evaluation.
"""

import json
from pathlib import Path


def filter_hypothetical_questions():
    # Load diverse dataset
    data_path = Path("evaluation/data/synthetic/deepeval_goldens_korean_100.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))

    # Filter out hypothetical questions
    filtered = []
    removed = []

    for item in data:
        evolutions = item.get("additional_metadata", {}).get("evolutions", [])
        if "Hypothetical" in evolutions:
            removed.append({
                "input": item["input"][:80],
                "evolutions": evolutions
            })
        else:
            filtered.append(item)

    print(f"Original: {len(data)} questions")
    print(f"Removed: {len(removed)} hypothetical questions")
    print(f"Remaining: {len(filtered)} questions")

    print("\n--- Removed Questions ---")
    for r in removed:
        print(f"  - {r['input']}...")
        print(f"    Evolutions: {r['evolutions']}")

    # Save filtered dataset
    output_path = Path("evaluation/data/synthetic/deepeval_goldens_korean_no_hypothetical.json")
    output_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\nSaved to: {output_path}")

    # Also filter the retrieval results
    results_path = Path("evaluation/experiments/results/llm_rerank_binary_deepeval_diverse.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Get list of filtered inputs
    filtered_inputs = {item["input"] for item in filtered}

    # Filter results
    filtered_results = [r for r in results["results"] if r["input"] in filtered_inputs]

    results["results"] = filtered_results
    results["total_cases"] = len(filtered_results)
    results["success"] = len([r for r in filtered_results if r.get("status") == "success"])

    output_results_path = Path("evaluation/experiments/results/llm_rerank_binary_diverse_no_hypothetical.json")
    output_results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved filtered results to: {output_results_path}")

    return len(filtered)


if __name__ == "__main__":
    filter_hypothetical_questions()
