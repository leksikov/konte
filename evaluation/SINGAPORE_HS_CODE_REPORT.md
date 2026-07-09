# Singapore HS Code — Evaluation Report

Date: 2026-04-28

## Project

`singapore_hs_code` — Konte project built from 9 Singapore Customs PDFs:

- `ahtn 2022 changes.pdf`
- `chemicalguide.pdf`
- `customs-ruling-database.pdf`
- `generalrulesfortheinterpretation.pdf`
- `how-to-determine-hs-code.pdf`
- `how-to-read-the-hs.pdf`
- `Infographics-9-Find-my-HS-Code-V4.pdf`
- `productguide.pdf`
- `stcced2022.pdf`

Source location: `example_knowledge_base/singapore_hs_code/`.

Build script: `scripts/build_singapore_hs_code.py`.

| Metric | Value |
|---|---|
| Total chunks | 901 |
| Context-generation model | Qwen3.6-35B-A3B (`https://qwen36_35b.asia03.app.backend.ai/v1`) |
| Embedding | text-embedding-3-small |
| Index | FAISS (semantic) + BM25 (lexical) |

## Test datasets

Both stored in `evaluation/data/synthetic/`.

| Dataset | File | Size | Generator |
|---|---|---|---|
| HS-100 lookup | `singapore_hs_goldens_100.json` | 100 questions | `evaluation/synthesize_singapore_dataset.py` (English variant of the Korean HS synthesizer; Qwen3.6-35B-A3B with thinking disabled) |
| Diverse-100 | `singapore_deepeval_goldens_100.json` | 100 questions | `evaluation/deepeval_synthesizer.py --language en` (DeepEval Synthesizer with 7 evolution types; Qwen3.6-35B-A3B) |

The diverse set covers Reasoning, Multi-context, Concretizing, Constrained, Comparative, Hypothetical, In-Breadth.

## Pipeline

For every benchmark row:

1. Hybrid retrieve top-100 chunks (FAISS semantic + BM25 lexical, reciprocal rank fusion).
2. Binary LLM rerank with the answer model (filter relevant + keep top-15 by score).
3. Generate answer from the 15 chunks via `konte.generator.generate_answer`.
4. Score with DeepEval `GEval` — `HSCodeCorrectness` for HS-100, `AnswerCorrectness` for Diverse-100.

## Results

### Headline

| Test set | Answer model | Judge | Avg score | Pass rate |
|---|---|---|---|---|
| HS-100 | Qwen3.6-35B-A3B | Qwen3.6-27B-FP8 | 0.825 | 84.0% |
| HS-100 | Qwen3.6-27B-FP8 | Qwen3.6-27B-FP8 | 0.833 | **87.0%** |
| HS-100 | gpt-4.1 | gpt-4.1 | 0.837 | 84.0% |
| HS-100 | gemma-4-26B-A4B-it | gpt-4.1 | 0.791 | 80.0% |
| Diverse-100 | Qwen3.6-35B-A3B | Qwen3.6-27B-FP8 | 0.809 | 81.0% |
| Diverse-100 | gpt-4.1 | gpt-4.1 | 0.905 | 92.0% |
| Diverse-100 | **gemma-4-26B-A4B-it** | gpt-4.1 | **0.912** | **95.0%** |

### Per-task winner

- **HS-code lookup (precise classification):** Qwen3.6-27B-FP8 — 87% pass.
- **Diverse RAG (open-ended Q&A):** gemma-4-26B-A4B-it — 95% pass.

Gemma is split: weakest on HS lookup (-7 pts vs the best Qwen) but strongest on diverse Q&A (+3 pts vs gpt-4.1). It is good at synthesising prose from context but less precise at picking the exact AHTN 8-digit leaf.

## Failure analysis — HS-100 (Qwen3.6-35B-A3B answer model)

16 failures out of 100.

| Cause | Count |
|---|---|
| Wrong AHTN sibling (same 4-6 digit parent, wrong 8-digit leaf) | 12 |
| Wrong code, correct one in retrieved context | 4 |
| Retrieval miss (correct HS code absent from retrieved chunks) | 0 |
| Judge too strict on format | 0 |

The bottleneck is **answer-selection at the leaf level**, not retrieval and not judging. Two contributing factors:

1. **Question under-specification.** Many synthesised questions omit the leaf-level discriminator (engine displacement, hybrid-vehicle wattage, gross tonnage band). Even a perfect LLM cannot pick the right sibling without it.
2. **LLM reasoning gaps.** In 4 cases the model picked an unrelated heading despite having the right context.

## Failure analysis — Diverse-100 (Qwen3.6-35B-A3B answer model)

19 failures out of 100, split:

| Category | Count | Notes |
|---|---|---|
| Judge wrapper instability ("unknown" reason, score=0) | 6 | `BackendAIModel._create_default_schema` fell back to defaults when the Qwen judge produced non-JSON output. The actual model answers in these cases match the expected. **Not real failures.** |
| Factual error in supporting prose (correct HS code but wrong surrounding facts) | 7 | |
| Hypothetical / unanswerable (expected answer is "context doesn't say") | 4 | DeepEval Hypothetical evolution generates Qs the docs cannot answer. |
| Wrong HS code | 2 | Genuine LLM error. |

Effective pass rate when ignoring judge instability and unanswerable hypotheticals: **87%** — in line with HS-100. Switching the judge to gpt-4.1 lifts the headline to 92% (and 95% with Gemma as answerer).

Excluding the Hypothetical evolution entirely (75 questions remain): 82.7% pass rate / 0.827 avg under the Qwen judge, vs. 92% / 0.905 under gpt-4.1.

## Operational notes

- `konte/loader.py` already supports PDFs natively (`pypdf`). One Singapore PDF is AES-encrypted; install `cryptography>=3.1` for `pypdf` to handle it.
- BackendAI Qwen3.6 is a reasoning model. For non-reasoning workloads (binary rerank, structured-output synthesis), set `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` and raise `max_tokens` (>= 4000 for synthesis, 256 for binary rerank). Without this Qwen exhausts the token budget on its `<think>` channel and returns `content=None`.
- `evaluation/custom_llm.py BackendAIModel` had a silent fallback that filled missing string fields with the literal "unknown". Patched to return the raw LLM text in single-string-field schemas (the DeepEval `Response{response: str}` shape) so genuine answers are not masked as empty.
- For long-running jobs, launch as `nohup caffeinate -dis bash -c '...' &; disown`. `caffeinate -i` does not prevent system or lid-close sleep on macOS and will silently kill multi-hour jobs.

## Reproduction

```bash
# Build the project
caffeinate -dis python scripts/build_singapore_hs_code.py

# Generate datasets
BACKENDAI_ENDPOINT="https://qwen36_35b.asia03.app.backend.ai/v1" \
  BACKENDAI_MODEL_NAME="Qwen3.6-35B-A3B" \
  python evaluation/synthesize_singapore_dataset.py \
    --project singapore_hs_code \
    --output evaluation/data/synthetic/singapore_hs_goldens_100.json \
    --num 100 --seed 42

BACKENDAI_ENDPOINT="https://qwen36_35b.asia03.app.backend.ai/v1" \
  BACKENDAI_MODEL_NAME="Qwen3.6-35B-A3B" \
  python -m evaluation.deepeval_synthesizer \
    --project singapore_hs_code \
    --output evaluation/data/synthetic/singapore_deepeval_goldens_100.json \
    --num 100 --seed 42 --language en

# Rerank + answer (example: gpt-4.1)
BACKENDAI_ENDPOINT="" CONTEXT_MODEL="gpt-4.1" \
  python -m evaluation.experiments.llm_reranking \
    --project singapore_hs_code \
    --test-cases evaluation/data/synthetic/singapore_hs_goldens_100.json \
    --method binary --initial-k 100 --final-k 15 --max-cases 0 \
    --output evaluation/experiments/results/llm_rerank_binary_singapore_hs100_gpt41.json

# Score
BACKENDAI_ENDPOINT="" CONTEXT_MODEL="gpt-4.1" \
  python -m evaluation.experiments.run_deepeval_full binary singapore_hs100_gpt41 hs_code
```

Swap `BACKENDAI_ENDPOINT` / `BACKENDAI_MODEL_NAME` for Qwen3.6-27B-FP8 (`https://qwen_36_27b_fp8.asia03.app.backend.ai/v1`) or gemma-4-26B-A4B-it (`https://gemma_4_26b.asia03.app.backend.ai/v1`) to reproduce the other rows.
