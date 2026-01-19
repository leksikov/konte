"""Read detailed failure examples for manual analysis."""

import json
import re
from pathlib import Path


def extract_hs_codes(text: str) -> set[str]:
    """Extract HS codes from text."""
    patterns = [
        r'\b(\d{4}\.\d{2}(?:\.\d{2})?)\b',
        r'\b제?(\d{4})호\b',
        r'\b(\d{4})(?:\s|$|,|\)|\.)',
    ]
    codes = set()
    for pattern in patterns:
        for match in re.findall(pattern, text):
            code = match.replace('.', '')[:4]
            if code.isdigit() and len(code) == 4:
                codes.add(code)
    return codes


def main():
    results_dir = Path("evaluation/experiments/results")
    path = results_dir / "llm_rerank_binary.json"

    with open(path) as f:
        data = json.load(f)

    results = data["results"]

    # Find failures
    failures = []
    for r in results:
        if r.get("status") != "success":
            continue

        expected_codes = extract_hs_codes(r["expected_output"])
        actual_codes = extract_hs_codes(r["actual_output"])

        if expected_codes and not (expected_codes & actual_codes):
            failures.append(r)

    print(f"Total failures: {len(failures)}")
    print("=" * 100)

    for i, f in enumerate(failures, 1):
        print(f"\n{'='*100}")
        print(f"FAILURE #{i}")
        print("=" * 100)

        print(f"\n[QUERY]")
        print(f"{f['input']}")

        print(f"\n[EXPECTED OUTPUT]")
        print(f"{f['expected_output']}")

        print(f"\n[ACTUAL OUTPUT]")
        print(f"{f['actual_output']}")

        print(f"\n[RETRIEVED CHUNKS ({len(f.get('retrieval_context', []))} total)]")
        for j, chunk in enumerate(f.get("retrieval_context", [])[:5], 1):
            print(f"\n  --- Chunk #{j} ---")
            print(f"  {chunk[:500]}...")

        print(f"\n[HS CODE ANALYSIS]")
        expected_codes = extract_hs_codes(f["expected_output"])
        actual_codes = extract_hs_codes(f["actual_output"])
        context_codes = set()
        for chunk in f.get("retrieval_context", []):
            context_codes.update(extract_hs_codes(chunk))

        print(f"  Expected codes: {sorted(expected_codes)}")
        print(f"  Actual codes: {sorted(actual_codes)}")
        print(f"  Codes in context: {sorted(context_codes)[:20]}...")
        print(f"  Expected in context: {sorted(expected_codes & context_codes)}")
        print(f"  Missing from context: {sorted(expected_codes - context_codes)}")

        # Determine failure type
        if expected_codes - context_codes == expected_codes:
            print(f"  >> FAILURE TYPE: RETRIEVAL (expected codes not in any chunk)")
        elif expected_codes & context_codes:
            print(f"  >> FAILURE TYPE: LLM GENERATION (codes in context but not cited)")
        else:
            print(f"  >> FAILURE TYPE: UNKNOWN")

        print("\n" + "-" * 100)


if __name__ == "__main__":
    main()
