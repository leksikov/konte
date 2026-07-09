"""Generate synthetic English HS-code lookup test dataset for Singapore project.

Mirrors evaluation/synthesize_korean_dataset.py but adapted for English source
documents and Singapore/AHTN HS code formats (XXXX.XX and XXXX.XX.XX).
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


class EnglishTestCase(BaseModel):
    question: str = Field(..., description="A question in English about the HS code item")
    expected_answer: str = Field(..., description="An English expected answer")
    hs_code: str = Field(..., description="HS code extracted from the document")


# Match XXXX.XX optionally followed by .XX (AHTN 8-digit), then a separator and a description.
# Accept dash, em-dash, or whitespace+text. Description 3..60 chars (cut at line break).
HS_CODE_PATTERN = re.compile(
    r"\b(\d{4}\.\d{2}(?:\.\d{2})?)\s*[-–—:\t ]+\s*([A-Za-z][^\n\r]{2,80})"
)


def extract_hs_codes_from_chunk(content: str) -> list[tuple[str, str, int]]:
    ambiguous = ("other", "others", "n.e.s", "not elsewhere", "etc.", "etc")
    seen = set()
    result = []
    for match in HS_CODE_PATTERN.finditer(content):
        code, name = match.group(1), match.group(2).strip()
        if code in seen:
            continue
        if len(name) < 3:
            continue
        if name.lower().startswith(ambiguous):
            continue
        if name.startswith("-"):
            continue
        seen.add(code)
        result.append((code, name, match.start()))
    return result


def extract_context_around_hs_code(
    content: str, hs_code: str, position: int,
    window_before: int = 500, window_after: int = 1000,
) -> str:
    start = max(0, position - window_before)
    para = content.rfind("\n\n", start, position)
    if para > start:
        start = para + 2
    end = min(len(content), position + window_after)
    para = content.find("\n\n", position + len(hs_code), end)
    if para > 0:
        end = para
    extracted = content[start:end].strip()
    if hs_code not in extracted:
        start = max(0, position - 200)
        end = min(len(content), position + 800)
        extracted = content[start:end].strip()
    return extracted


ENGLISH_SYNTHESIS_PROMPT = """You are an expert in customs tariff classification and the Harmonized System (HS).

Generate a clear, specific question and an expected answer based on the document and the given HS code.

## Document:
{context}

## HS code from this document:
{hs_code} - {item_name}

## Requirements:
1. Write a specific, unambiguous English question about classifying the item ({item_name}).
2. The expected answer must start with: "The HS code is {hs_code}".
3. Include relevant context from the document in the expected answer.
4. Avoid vague items like "other", "n.e.s.", or partial descriptions.

## Good examples:
- "What is the HS code for fresh table grapes?"
- "Under which HS code are DDR5 memory modules classified?"
- "Which HS code applies to wine made from fresh grapes?"

## Bad examples (do not use):
- "What is the HS code for other?" — too vague
- "What is XXXX classified as?" — non-specific

## Response format (JSON):
{{
    "question": "...",
    "expected_answer": "The HS code is {hs_code}. ...",
    "hs_code": "{hs_code}"
}}
"""


async def generate_test_case(llm, context: str, hs_code: str, item_name: str):
    prompt = ENGLISH_SYNTHESIS_PROMPT.format(
        context=context[:2000], hs_code=hs_code, item_name=item_name,
    )
    try:
        structured = llm.with_structured_output(EnglishTestCase, method="json_mode")
        result = await structured.ainvoke(prompt)
        if result:
            result.hs_code = hs_code
            return result
        return None
    except Exception as e:
        print(f"Error generating test case: {e}")
        return None


def validate_test_case(project: Project, question: str, expected_hs_code: str, top_k: int = 5) -> bool:
    response = project.query(question, mode="hybrid", top_k=top_k)
    code_4digit = expected_hs_code[:4]
    for r in response.results:
        if code_4digit in r.content or code_4digit in (r.context or ""):
            return True
    return False


async def synthesize(project_name: str, output_path: Path, num_goldens: int, seed: int, batch_size: int = 5):
    random.seed(seed)
    print(f"Loading project: {project_name}")
    project = Project.open(project_name)

    chunks_path = project.project_dir / "chunks.json"
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks_data)} chunks")

    chunks_with_codes = []
    for item in chunks_data:
        chunk = item.get("chunk", {})
        ctx = item.get("context", "")
        content = chunk.get("content", "")
        full = f"{ctx}\n\n{content}" if ctx else content
        codes = extract_hs_codes_from_chunk(content)
        if codes and len(full) > 200:
            for hs_code, name, position in codes:
                focused = extract_context_around_hs_code(full, hs_code, position)
                chunks_with_codes.append({
                    "content": full, "focused_context": focused,
                    "hs_code": hs_code, "item_name": name, "position": position,
                })

    print(f"Found {len(chunks_with_codes)} chunks with extractable HS codes")
    random.shuffle(chunks_with_codes)

    if settings.use_backendai:
        llm = ChatOpenAI(
            model=settings.BACKENDAI_MODEL_NAME,
            api_key=settings.BACKENDAI_API_KEY or "placeholder",
            base_url=settings.BACKENDAI_ENDPOINT,
            temperature=0.3, max_tokens=4000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    else:
        llm = ChatOpenAI(
            model=settings.CONTEXT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3, max_tokens=4000,
        )

    goldens, used = [], set()
    idx = gen_err = val_fail = 0
    print(f"\nGenerating {num_goldens} validated test cases...")

    with tqdm(total=num_goldens, desc="Generating goldens") as pbar:
        while len(goldens) < num_goldens and idx < len(chunks_with_codes):
            batch = []
            while len(batch) < batch_size and idx < len(chunks_with_codes):
                ci = chunks_with_codes[idx]
                if ci["hs_code"] not in used:
                    batch.append(ci)
                idx += 1
            if not batch:
                continue

            tasks = [generate_test_case(llm, ci["content"], ci["hs_code"], ci["item_name"]) for ci in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for ci, result in zip(batch, results):
                if isinstance(result, Exception) or result is None:
                    gen_err += 1
                    continue
                if validate_test_case(project, result.question, result.hs_code):
                    retrieval_ctx = ci.get("focused_context") or ci["content"][:1000]
                    if result.hs_code not in retrieval_ctx and result.hs_code in ci["content"]:
                        retrieval_ctx = extract_context_around_hs_code(
                            ci["content"], result.hs_code, ci.get("position", 0),
                        )
                    goldens.append({
                        "input": result.question,
                        "expected_output": result.expected_answer,
                        "retrieval_context": [retrieval_ctx],
                    })
                    used.add(result.hs_code)
                    pbar.update(1)
                    pbar.set_postfix({"valid": len(goldens), "gen_err": gen_err, "val_fail": val_fail})
                    if len(goldens) >= num_goldens:
                        break
                else:
                    val_fail += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(goldens, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGenerated {len(goldens)} validated test cases")
    print(f"Gen errors: {gen_err} | Validation failures: {val_fail}")
    print(f"Saved to: {output_path}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="singapore_hs_code")
    parser.add_argument("--output", default="evaluation/data/synthetic/singapore_hs_goldens_100.json")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    await synthesize(args.project, Path(args.output), args.num, args.seed)


if __name__ == "__main__":
    asyncio.run(main())
