"""Generate synthetic test dataset in KOREAN from wco_hs_explanatory_notes.

The source document is in Korean, so test cases must also be in Korean.
This script extracts HS codes from chunks first, then generates questions.
"""

import asyncio
import json
import random
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from konte import Project
from konte.config.settings import settings


class KoreanTestCase(BaseModel):
    """Korean test case with question and expected answer."""

    question: str = Field(..., description="한국어로 작성된 질문")
    expected_answer: str = Field(..., description="한국어로 작성된 예상 답변")
    hs_code: str = Field(..., description="문서에서 추출한 HS 코드")


# Pattern to find HS codes like "8540.20" or "3201.10 - 품목명"
HS_CODE_PATTERN = re.compile(r'\b(\d{4}\.\d{2})\s*[-–—]\s*([^\n\r]{3,50})')


def extract_hs_codes_from_chunk(content: str) -> list[tuple[str, str, int]]:
    """Extract HS codes and their descriptions from chunk content.

    Returns list of (hs_code, item_name, position) tuples.
    Filters out ambiguous items like "기타", "그 밖의 것" etc.
    """
    # Ambiguous item names to skip
    ambiguous_patterns = [
        "기타",
        "그 밖의",
        "그밖의",
        "기타의",
        "그 외",
    ]

    # Deduplicate and filter
    seen = set()
    result = []
    for match in HS_CODE_PATTERN.finditer(content):
        code = match.group(1)
        name = match.group(2)
        position = match.start()

        if code not in seen:
            seen.add(code)
            # Clean up the item name
            name = name.strip()

            # Skip ambiguous names
            if not name or len(name) < 3:
                continue
            if any(name.startswith(p) for p in ambiguous_patterns):
                continue
            if name.startswith("-"):  # Skip incomplete items like "-- 면으로"
                continue

            result.append((code, name, position))
    return result


def extract_context_around_hs_code(
    content: str,
    hs_code: str,
    position: int,
    window_before: int = 500,
    window_after: int = 1000,
) -> str:
    """Extract context window around the HS code position.

    Ensures the HS code and its description are included in the context.
    """
    # Find a good start position (try to start at a paragraph or sentence)
    start = max(0, position - window_before)

    # Try to find paragraph break before
    para_break = content.rfind("\n\n", start, position)
    if para_break > start:
        start = para_break + 2

    # Find end position
    end = min(len(content), position + window_after)

    # Try to find paragraph break after
    para_break = content.find("\n\n", position + len(hs_code), end)
    if para_break > 0:
        end = para_break

    extracted = content[start:end].strip()

    # Verify HS code is in extracted context
    if hs_code not in extracted:
        # Fallback: just use position-based window
        start = max(0, position - 200)
        end = min(len(content), position + 800)
        extracted = content[start:end].strip()

    return extracted


KOREAN_SYNTHESIS_PROMPT = """당신은 관세 및 HS 코드 분류 전문가입니다.

다음 문서와 HS 코드 정보를 바탕으로 질문과 예상 답변을 생성하세요.

## 문서:
{context}

## 이 문서에 있는 HS 코드:
{hs_code} - {item_name}

## 요구사항:
1. 위 HS 코드({hs_code})에 대한 구체적이고 명확한 질문을 만드세요
2. 예상 답변은 "HS 코드 {hs_code}에 분류됩니다" 형식으로 시작하세요
3. 문서의 추가 정보를 답변에 포함하세요

## 중요 - 피해야 할 질문 유형:
- "기타"로 시작하는 질문 (예: "기타 반도체는?") - 너무 모호함
- 단순히 소호 번호만 언급하는 질문 - 구체적인 품목명을 사용하세요
- "-- 로 만든 것"처럼 불완전한 품목명

## 좋은 질문 예시:
- "퀘브라쵸 추출물은 어느 HS 코드에 분류되나요?" (구체적 품목명)
- "DDR5 메모리 모듈은 어느 HS 코드에 분류되나요?" (구체적 제품)
- "신선한 포도로 만든 포도주는 어느 HS 코드에 분류되나요?" (구체적 상태 명시)

## 나쁜 질문 예시 (사용하지 마세요):
- "기타은(는) 어느 HS 코드에 분류되나요?" - 너무 모호함
- "-- 면으로 만든 것은?" - 불완전함
- "그 밖의 것은?" - 불명확함

## 응답 형식 (JSON):
{{
    "question": "구체적인 품목명은 어느 HS 코드에 분류되나요?",
    "expected_answer": "HS 코드 {hs_code}에 분류됩니다. (추가 설명)",
    "hs_code": "{hs_code}"
}}
"""


async def generate_korean_test_case(
    llm: ChatOpenAI,
    context: str,
    hs_code: str,
    item_name: str,
) -> KoreanTestCase | None:
    """Generate a single Korean test case from context with known HS code."""
    prompt = KOREAN_SYNTHESIS_PROMPT.format(
        context=context[:2000],  # Limit context length
        hs_code=hs_code,
        item_name=item_name,
    )

    try:
        structured_llm = llm.with_structured_output(KoreanTestCase, method="json_mode")
        result = await structured_llm.ainvoke(prompt)

        # Validate that the generated HS code matches
        if result and result.hs_code == hs_code:
            return result
        elif result:
            # Fix the HS code if LLM changed it
            result.hs_code = hs_code
            return result
        return None
    except Exception as e:
        print(f"Error generating test case: {e}")
        return None


def validate_test_case(
    project: Project,
    question: str,
    expected_hs_code: str,
    top_k: int = 5,
) -> bool:
    """Validate that the expected HS code can be retrieved."""
    response = project.query(question, mode="hybrid", top_k=top_k)

    # Check if expected HS code appears in any retrieved chunk
    code_4digit = expected_hs_code[:4]
    for result in response.results:
        if code_4digit in result.content or code_4digit in (result.context or ""):
            return True
    return False


async def synthesize_korean_dataset(
    project_name: str,
    output_path: Path,
    num_goldens: int = 30,
    seed: int = 42,
    batch_size: int = 5,
) -> None:
    """Generate validated Korean test cases from Konte project chunks.

    Args:
        project_name: Name of Konte project.
        output_path: Path to save generated goldens as JSON.
        num_goldens: Target number of valid test cases.
        seed: Random seed for reproducibility.
        batch_size: Number of concurrent LLM calls.
    """
    random.seed(seed)

    # Load project
    print(f"Loading project: {project_name}")
    project = Project.open(project_name)

    # Load chunks
    chunks_path = project.project_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.json found in {project.project_dir}")

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)

    print(f"Loaded {len(chunks_data)} chunks from project")

    # Extract chunks with HS codes
    chunks_with_codes = []
    for chunk_item in chunks_data:
        chunk = chunk_item.get("chunk", {})
        context = chunk_item.get("context", "")
        content = chunk.get("content", "")

        full_text = f"{context}\n\n{content}" if context else content

        # Extract HS codes from this chunk (now includes position)
        hs_codes = extract_hs_codes_from_chunk(content)

        if hs_codes and len(full_text) > 200:
            for hs_code, item_name, position in hs_codes:
                # Extract focused context around this specific HS code
                focused_context = extract_context_around_hs_code(
                    full_text, hs_code, position
                )
                chunks_with_codes.append({
                    "content": full_text,
                    "focused_context": focused_context,
                    "hs_code": hs_code,
                    "item_name": item_name,
                    "position": position,
                })

    print(f"Found {len(chunks_with_codes)} chunks with extractable HS codes")

    # Shuffle and prepare for generation
    random.shuffle(chunks_with_codes)

    # Initialize LLM
    if settings.use_backendai:
        llm = ChatOpenAI(
            model=settings.BACKENDAI_MODEL_NAME,
            api_key=settings.BACKENDAI_API_KEY or "placeholder",
            base_url=settings.BACKENDAI_ENDPOINT,
            temperature=0.3,
            max_tokens=1000,
        )
    else:
        llm = ChatOpenAI(
            model=settings.CONTEXT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1000,
        )

    # Generate and validate test cases
    golden_dicts = []
    used_hs_codes = set()
    idx = 0
    generation_errors = 0
    validation_failures = 0

    print(f"\nGenerating {num_goldens} validated Korean test cases...")

    with tqdm(total=num_goldens, desc="Generating validated goldens") as pbar:
        while len(golden_dicts) < num_goldens and idx < len(chunks_with_codes):
            # Process in batches
            batch = []
            batch_indices = []

            while len(batch) < batch_size and idx < len(chunks_with_codes):
                chunk_info = chunks_with_codes[idx]
                # Skip if we already have a question for this HS code
                if chunk_info["hs_code"] not in used_hs_codes:
                    batch.append(chunk_info)
                    batch_indices.append(idx)
                idx += 1

            if not batch:
                continue

            # Generate test cases concurrently
            tasks = [
                generate_korean_test_case(
                    llm,
                    chunk_info["content"],
                    chunk_info["hs_code"],
                    chunk_info["item_name"],
                )
                for chunk_info in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Validate and add successful results
            for chunk_info, result in zip(batch, results):
                if isinstance(result, Exception):
                    generation_errors += 1
                    continue

                if result is None:
                    generation_errors += 1
                    continue

                # Validate retrieval
                is_valid = validate_test_case(
                    project,
                    result.question,
                    result.hs_code,
                )

                if is_valid:
                    # Use focused_context that contains the specific HS code
                    retrieval_ctx = chunk_info.get("focused_context", chunk_info["content"][:1000])

                    # Verify the HS code is actually in the retrieval context
                    if result.hs_code not in retrieval_ctx:
                        # Try to find and include the HS code
                        if result.hs_code in chunk_info["content"]:
                            retrieval_ctx = extract_context_around_hs_code(
                                chunk_info["content"],
                                result.hs_code,
                                chunk_info.get("position", 0),
                            )

                    golden_dicts.append({
                        "input": result.question,
                        "expected_output": result.expected_answer,
                        "retrieval_context": [retrieval_ctx],
                    })
                    used_hs_codes.add(result.hs_code)
                    pbar.update(1)
                    pbar.set_postfix({
                        "valid": len(golden_dicts),
                        "gen_err": generation_errors,
                        "val_fail": validation_failures,
                    })

                    if len(golden_dicts) >= num_goldens:
                        break
                else:
                    validation_failures += 1

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_dicts, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(golden_dicts)} validated Korean test cases")
    print(f"Generation errors: {generation_errors}")
    print(f"Validation failures: {validation_failures}")
    print(f"Unique HS codes used: {len(used_hs_codes)}")
    print(f"Saved to: {output_path}")


async def main():
    """Generate Korean test dataset from wco_hs_explanatory_notes project."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Korean test cases")
    parser.add_argument("--project", default="wco_hs_explanatory_notes_korean",
                        help="Project name to use for synthesis")
    parser.add_argument("--output", default="evaluation/data/synthetic/synthetic_goldens_30.json",
                        help="Output path for generated test cases")
    parser.add_argument("--num", type=int, default=30, help="Number of test cases to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    output_path = Path(args.output)

    await synthesize_korean_dataset(
        project_name=args.project,
        output_path=output_path,
        num_goldens=args.num,
        seed=args.seed,
        batch_size=5,
    )


if __name__ == "__main__":
    asyncio.run(main())
