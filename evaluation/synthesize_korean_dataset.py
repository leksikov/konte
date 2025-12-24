"""Generate synthetic test dataset in KOREAN from wco_hs_explanatory_notes.

The source document is in Korean, so test cases must also be in Korean
to enable fair comparison between context-embedded and context-metadata approaches.
"""

import asyncio
import json
import random
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


class KoreanTestCaseList(BaseModel):
    """List of Korean test cases."""

    test_cases: list[KoreanTestCase] = Field(default_factory=list)


KOREAN_SYNTHESIS_PROMPT = """당신은 관세 및 HS 코드 분류 전문가입니다.

다음 문서를 읽고, 이 문서에서 **직접 추출한 정보**로만 답변할 수 있는 질문과 예상 답변을 한국어로 생성하세요.

## 문서 (Document):
{context}

## 핵심 원칙 (매우 중요):
**예상 답변에 포함되는 모든 정보(HS 코드, 분류 기준, 제품명 등)는 반드시 위 문서에서 직접 추출해야 합니다.**
문서에 없는 정보를 추가하거나 자신의 지식을 사용하지 마세요.

## 질문 작성 규칙:
- 문서에 나오는 구체적인 제품명, 기술 용어를 포함하세요
- 하나의 명확한 답변을 요구하는 질문을 작성하세요

## 예상 답변 작성 규칙:
- 문서에 명시된 HS 코드를 **그대로** 인용하세요 (예: "제3921호", "8540.20" 등 문서에 나온 형식 그대로)
- 문서에서 직접 인용할 수 있는 근거를 포함하세요
- 2-3문장으로 간결하게 작성하세요
- 제외되는 코드나 관련 코드는 언급하지 마세요

## 피해야 할 패턴:
❌ 문서에 없는 HS 코드를 언급
❌ 자신의 관세 지식을 추가
❌ 제외 코드 언급 (예: "~는 제외됩니다")

## 응답 형식 (JSON):
{{
    "question": "문서 내용에 기반한 구체적인 한국어 질문",
    "expected_answer": "문서에서 직접 추출한 정보만으로 구성된 답변 (HS 코드는 문서 형식 그대로)"
}}
"""


async def generate_korean_test_case(
    llm: ChatOpenAI,
    context: str,
) -> KoreanTestCase | None:
    """Generate a single Korean test case from context."""
    prompt = KOREAN_SYNTHESIS_PROMPT.format(context=context)

    try:
        structured_llm = llm.with_structured_output(KoreanTestCase, method="json_mode")
        result = await structured_llm.ainvoke(prompt)
        return result
    except Exception as e:
        print(f"Error generating test case: {e}")
        return None


async def synthesize_korean_dataset(
    project_name: str,
    output_path: Path,
    num_goldens: int = 120,
    use_segments: bool = True,
    seed: int = 42,
    batch_size: int = 10,
) -> None:
    """Generate Korean test cases from Konte project segments.

    Args:
        project_name: Name of Konte project to extract segments from.
        output_path: Path to save generated goldens as JSON.
        num_goldens: Number of golden test cases to generate.
        use_segments: If True, use full segments (~8000 tokens). If False, use chunks.
        seed: Random seed for reproducibility.
        batch_size: Number of concurrent LLM calls.
    """
    random.seed(seed)

    # Load project
    print(f"Loading project: {project_name}")
    project = Project.open(project_name)

    if use_segments:
        # Load segments (full ~8000 token documents)
        segments_path = project.project_dir / "segments.json"
        if not segments_path.exists():
            raise FileNotFoundError(f"No segments.json found in {project.project_dir}")

        with open(segments_path, encoding="utf-8") as f:
            segments_data = json.load(f)

        print(f"Loaded {len(segments_data)} segments from project")

        # Extract segment content (segments are stored as dict with string keys)
        contexts = []
        for key, content in segments_data.items():
            if content and len(content) > 500:  # Filter too short segments
                contexts.append(content)

        print(f"Using {len(contexts)} quality segments for synthesis")
    else:
        # Load chunks (fallback)
        chunks_path = project.project_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(f"No chunks.json found in {project.project_dir}")

        with open(chunks_path, encoding="utf-8") as f:
            chunks_data = json.load(f)

        print(f"Loaded {len(chunks_data)} chunks from project")

        contexts = []
        for chunk_item in chunks_data:
            chunk = chunk_item.get("chunk", {})
            context = chunk_item.get("context", "")
            content = chunk.get("content", "")

            if context and content:
                text = f"{context}\n\n{content}"
            else:
                text = content

            if text and len(text) > 100:
                contexts.append(text)

        print(f"Using {len(contexts)} quality chunks for synthesis")

    # Randomly sample
    if len(contexts) < num_goldens:
        sampled_indices = [random.randint(0, len(contexts) - 1) for _ in range(num_goldens)]
    else:
        sampled_indices = random.sample(range(len(contexts)), num_goldens)

    contexts_to_process = [contexts[i] for i in sampled_indices]
    print(f"Selected {len(contexts_to_process)} contexts for test case generation")

    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.BACKENDAI_MODEL_NAME,
        api_key="placeholder",
        base_url=settings.BACKENDAI_ENDPOINT,
        temperature=0.7,
        max_tokens=2000,
    )

    # Generate test cases with progress bar
    golden_dicts = []
    errors = 0

    print(f"Generating {len(contexts_to_process)} Korean test cases...")

    with tqdm(total=len(contexts_to_process), desc="Generating Korean goldens") as pbar:
        for batch_start in range(0, len(contexts_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(contexts_to_process))
            batch_contexts = contexts_to_process[batch_start:batch_end]

            # Create tasks for concurrent execution
            tasks = [
                generate_korean_test_case(llm, ctx)
                for ctx in batch_contexts
            ]

            # Execute batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                ctx_idx = batch_start + i
                if isinstance(result, Exception):
                    errors += 1
                    print(f"\nError at index {ctx_idx}: {result}")
                elif result is not None:
                    golden_dicts.append({
                        "input": result.question,
                        "expected_output": result.expected_answer,
                        "retrieval_context": [contexts_to_process[ctx_idx]],
                    })

            pbar.update(len(batch_contexts))
            pbar.set_postfix({"generated": len(golden_dicts), "errors": errors})

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_dicts, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(golden_dicts)} Korean test cases")
    print(f"Errors: {errors}")
    print(f"Saved to: {output_path}")


async def main():
    """Generate Korean test dataset from wco_hs_explanatory_notes project."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Korean test cases")
    parser.add_argument("--project", default="wco_hs_explanatory_notes_korean",
                        help="Project name to use for synthesis")
    parser.add_argument("--output", default="evaluation/data/synthetic/synthetic_goldens_korean_v4.json",
                        help="Output path for generated test cases")
    parser.add_argument("--num", type=int, default=120, help="Number of test cases to generate")
    parser.add_argument("--use-chunks", action="store_true",
                        help="Use chunks instead of segments (default: use segments)")
    args = parser.parse_args()

    output_path = Path(args.output)

    await synthesize_korean_dataset(
        project_name=args.project,
        output_path=output_path,
        num_goldens=args.num,
        use_segments=not args.use_chunks,
        batch_size=10,
    )


if __name__ == "__main__":
    asyncio.run(main())
