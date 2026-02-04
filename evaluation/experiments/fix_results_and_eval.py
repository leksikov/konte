"""
Fix ground truth errors in retrieval results and re-run DeepEval.

Fixes:
1. 비스코스 레이온: 5403.10 -> 5403.31
2. 합성스테이플섬유사: 5511.20 -> 5511.10
3. 롱파일 면 메리야스: 6001.10 -> 6001.21
"""

import json
from pathlib import Path


def fix_retrieval_results():
    """Fix expected outputs in retrieval results."""
    results_path = Path("evaluation/experiments/results/llm_rerank_binary_100.json")
    data = json.loads(results_path.read_text(encoding="utf-8"))

    fixed_count = 0

    for r in data["results"]:
        # Fix 1: 비스코스 레이온 필라멘트사
        if "비스코스 레이온으로 만든 꼬임이 없거나" in r["input"]:
            r["expected_output"] = r["expected_output"].replace("5403.10", "5403.31")
            r["expected_output"] = r["expected_output"].replace(
                "강력사에 한정하며",
                "비스코스 레이온 필라멘트사에 해당하며"
            )
            print(f"Fixed: 비스코스 레이온 (5403.10 -> 5403.31)")
            fixed_count += 1

        # Fix 2: 합성스테이플섬유사 85% (with expected 5511.20)
        if ("합성스테이플섬유의 함유량이 전 중량의 85%" in r["input"] and
            "5511.20" in r["expected_output"]):
            r["expected_output"] = r["expected_output"].replace("5511.20", "5511.10")
            print(f"Fixed: 합성스테이플섬유사 (5511.20 -> 5511.10)")
            fixed_count += 1

        # Fix 3: 롱파일 면 메리야스 편물
        if "롱파일(looped pile) 편물 중 면으로 만든" in r["input"]:
            r["expected_output"] = r["expected_output"].replace("6001.10", "6001.21")
            print(f"Fixed: 롱파일 면 메리야스 (6001.10 -> 6001.21)")
            fixed_count += 1

    # Save fixed results
    output_path = Path("evaluation/experiments/results/llm_rerank_binary_100_fixed.json")
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nTotal results: {len(data['results'])}")
    print(f"Fixed: {fixed_count}")
    print(f"Saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    fix_retrieval_results()
