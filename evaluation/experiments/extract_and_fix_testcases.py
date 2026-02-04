"""
Extract test cases from retrieval results and fix ground truth errors.

Fixes identified:
1. 비스코스 레이온 필라멘트사: 5403.10 -> 5403.31 (question doesn't mention 강력사)
2. 합성스테이플섬유사 85%: 5511.20 -> 5511.10 (matches document evidence)
3. 롱파일 면 메리야스: 6001.10 -> 6001.21 (면으로 만든 것 = 6001.21)
"""

import json
from pathlib import Path


def extract_and_fix_test_cases():
    # Load retrieval results
    results_path = Path("evaluation/experiments/results/llm_rerank_binary_100.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    # Extract test cases
    test_cases = []
    fixed_count = 0

    for r in results["results"]:
        tc = {
            "input": r["input"],
            "expected_output": r["expected_output"],
        }

        # Fix 1: 비스코스 레이온 필라멘트사
        if "비스코스 레이온으로 만든 꼬임이 없거나" in tc["input"]:
            old_expected = tc["expected_output"]
            tc["expected_output"] = tc["expected_output"].replace("5403.10", "5403.31")
            tc["expected_output"] = tc["expected_output"].replace(
                "강력사에 한정하며",
                "비스코스 레이온 필라멘트사에 해당하며"
            )
            print(f"Fixed case 1: 비스코스 레이온")
            print(f"  Old: 5403.10 -> New: 5403.31")
            fixed_count += 1

        # Fix 2: 합성스테이플섬유사 85% (the one with expected 5511.20)
        if ("합성스테이플섬유의 함유량이 전 중량의 85%" in tc["input"] and
            "5511.20" in tc["expected_output"]):
            tc["expected_output"] = tc["expected_output"].replace("5511.20", "5511.10")
            print(f"Fixed case 2: 합성스테이플섬유사")
            print(f"  Old: 5511.20 -> New: 5511.10")
            fixed_count += 1

        # Fix 3: 롱파일 면 메리야스 편물
        if "롱파일(looped pile) 편물 중 면으로 만든" in tc["input"]:
            tc["expected_output"] = tc["expected_output"].replace("6001.10", "6001.21")
            # Also fix the question itself (looped pile should be 루프파일)
            tc["input"] = tc["input"].replace("롱파일(looped pile)", "루프파일(looped pile)")
            print(f"Fixed case 3: 롱파일 면 메리야스")
            print(f"  Old: 6001.10 -> New: 6001.21")
            print(f"  Also fixed question: 롱파일 -> 루프파일")
            fixed_count += 1

        test_cases.append(tc)

    # Save fixed test cases
    output_path = Path("evaluation/data/synthetic/synthetic_goldens_100_fixed.json")
    output_path.write_text(
        json.dumps(test_cases, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nTotal test cases: {len(test_cases)}")
    print(f"Fixed cases: {fixed_count}")
    print(f"Saved to: {output_path}")

    return test_cases


if __name__ == "__main__":
    extract_and_fix_test_cases()
