# RAG Evaluation Report

## Overview

This report documents the evaluation of the Konte RAG system using DeepEval with LLM-as-judge metrics.

**Important**: Konte is a RAG (Retrieval-Augmented Generation) system for answering questions based on retrieved context. It is NOT a classification system.

---

## Evaluation Results Summary

### Contextual RAG vs Baseline Comparison

| Configuration | HS Code (100q) | Diverse RAG (100q) |
|---------------|----------------|---------------------|
| **Contextual + Reranking** | **94% (0.920)** | **94% (0.828)** |
| Baseline (no context, no rerank) | 85% (0.822) | 74% (0.613) |
| **Improvement** | **+9% (+0.098)** | **+20% (+0.215)** |

**Key Findings**:
1. Context generation provides significant improvement - particularly for diverse questions (+20% pass rate)
2. HS code lookups benefit moderately from context (+9% pass rate)
3. The combined effect of contextual chunks + LLM reranking is most pronounced on complex, multi-context questions

### Two Evaluation Approaches

| Evaluation | Dataset | Questions | Pass Rate | Avg Score |
|------------|---------|-----------|-----------|-----------|
| **DeepEval Synthesizer (Diverse)** | `deepeval_goldens_korean_100.json` | 100 diverse types | **94.0%** | **0.828** |
| HS Code Lookup (Legacy) | `synthetic_goldens_100.json` | 100 classification | 94.0% | 0.918 |

---

## Evaluation 1: DeepEval Synthesizer (Diverse Questions)

Generated using DeepEval Synthesizer with 7 Evolution types for question diversity.

### Results by Question Type

| Evolution Type | Description | Pass Rate | Avg Score | Count |
|----------------|-------------|-----------|-----------|-------|
| **Reasoning** | Logical complexity questions | **100.0%** | 0.861 | 23 |
| Comparative | Comparison questions | 96.9% | 0.841 | 32 |
| Constrained | Questions with constraints | 96.7% | 0.857 | 30 |
| In-Breadth | Scope expansion questions | 95.7% | 0.852 | 23 |
| Multi-context | Questions requiring multiple sources | 92.9% | 0.814 | 28 |
| Concretizing | Specific detail questions | 91.2% | 0.809 | 34 |
| Hypothetical | "What if" scenarios | 86.7% | 0.777 | 30 |
| **TOTAL** | | **94.0%** | **0.828** | **100** |

### Sample Questions by Type

**Reasoning**:
- "세포 조작 정도와 생물학적 특성 변경이 인체 치료 및 예방 목적 범위 내에서 어떻게 정의되고 적용되는지 설명해 주실 수 있나요?"

**Comparative**:
- "플라티늄과 팔라듐의 내식성 및 반응성 차이는 무엇이며, 이 두 금속의 단조와 압연 가공 형태에 따른 산업적 활용은 어떻게 비교할 수 있나요?"

**Hypothetical**:
- "만약 두 가지 이상의 성분이 혼합된 의약품이 투여량 고정 없이 대량 포장된다면, HS 분류는 어떻게 달라질까요?"

**Concretizing**:
- "비오틴이 난황, 간, 효모 등 천연 함유원에서 어떻게 발견되며, 합성 제조법과 효소 보조 메커니즘은 무엇인지 설명해 주실 수 있나요?"

### Failure Analysis (6/100)

| # | Question Type | Question (truncated) | Score | Root Cause |
|---|---------------|---------------------|-------|------------|
| 1 | Hypothetical | 인조 커런덤 용융법과 무수 알루미나 소성법... | 0.30 | Context lacks combined process info |
| 2 | Comparative | 복사지용 종이와 등사원지의 방수 처리... | 0.20 | Insufficient detail in context |
| 3 | Hypothetical | 가동코일형 대신 가동철편형을 사용할 경우... | 0.20 | Technical comparison not in context |
| 4 | Comparative | 케이폭과 코이어 섬유의 가구용 충전재... | 0.20 | Missing HS code info |
| 5 | Hypothetical | PHEV가 충전 불가능한 장거리 주행 시... | 0.20 | Hypothetical scenario not covered |
| 6 | Hypothetical | 오배자 탄닌과 제외된 유도체가 혼합된다면... | 0.30 | Edge case classification |

**Key Finding**: Most failures (4/6) are Hypothetical questions where the retrieved context doesn't contain information about the specific "what if" scenario.

---

## Evaluation 2: HS Code Lookup Questions (Legacy)

Previous evaluation using manually created HS code classification questions.

### Question Pattern

All 100 questions follow the same pattern:
```
"[Product description]은/는 어느 HS 코드에 분류되나요?"
(Which HS code is [product description] classified under?)
```

This is equivalent to the **Constrained** question type in DeepEval Synthesizer terminology - direct lookup questions with specific constraints.

### Results

| Metric | Value |
|--------|-------|
| Pass Rate | **94.0%** |
| Avg Score | 0.918 |
| Passed | 94/100 |
| Failed | 6/100 |

### Failure Analysis (6/100)

| # | Question | Expected | Actual | Score | Root Cause |
|---|----------|----------|--------|-------|------------|
| 1 | 인조섬유로 만든 저지 풀오버 카디건 | 6110.30 | 6101/6102 | 0.2 | Knitwear vs coat chapter confusion |
| 2 | 비스코스 레이온 꼬임 없는 필라멘트사 | 5403.10 | 5403.31 | 0.2 | Twist threshold subcode error |
| 3 | 시아노겐 클로라이드(클로르시안) | 2853.10 | 2812 | 0.2 | Wrong chapter (inorganic compounds) |
| 4 | 합성스테이플섬유 85%+ 인조스테이플섬유사 | 5511.20 | 5511.10 | 0.0 | Fiber content threshold ambiguity |
| 5 | 제5602/5603호 직물 여성용 실내복 | 6210.10 | 6208 | 0.2 | Special fabric vs woven chapter |
| 6 | 롱파일 면 메리야스 편물 | 6001.10 | 6001.21 | 0.0 | Looped vs long pile distinction |

**Key Finding**: Most failures (4/6) involve textile products (HS chapters 54-62) with complex subcode distinctions based on fiber content percentage, construction method, and end-use.

---

## Methodology

### GEval Metric (LLM-as-Judge)

**AnswerCorrectness** criteria:
```
Evaluate if the actual output correctly answers the question based on the expected output.

Evaluation criteria:
- Does the actual output contain the KEY FACTS from the expected output?
- Is the information semantically equivalent (same meaning, different wording is OK)?
- Are technical terms, codes, or specific details accurate?
- Ignore format differences, language mixing (Korean/English), or extra explanation

Scoring:
- Score 1.0: All key facts match, answer is complete and accurate
- Score 0.7-0.9: Most key facts match, minor omissions or variations
- Score 0.5-0.6: Partially correct, some key facts present but incomplete
- Score 0.3-0.4: Few key facts match, significant information missing
- Score 0.0-0.2: Wrong information or contradicts expected output
```

### RAG Pipeline

1. **Retrieval**: Hybrid (FAISS semantic + BM25 lexical) with RRF fusion
2. **Initial Retrieval**: top-k=100 candidates
3. **LLM Reranking**: Binary filter to select relevant chunks
4. **Final Context**: top-k=15 chunks passed to answer generation
5. **Answer Generation**: Qwen3-VL-8B-Instruct (BackendAI)

---

## Test Case Generation

### DeepEval Synthesizer (Recommended)

```bash
python -m evaluation.deepeval_synthesizer \
  --project wco_hs_explanatory_notes_korean \
  --output evaluation/data/synthetic/deepeval_goldens_korean_100.json \
  --num 100 --model gpt-4.1-mini
```

Evolution types for question diversity:
- **Reasoning**: Logical complexity questions
- **Multi-context**: Questions requiring multiple sources
- **Concretizing**: Specific detail questions
- **Constrained**: Questions with constraints (similar to HS code lookup)
- **Comparative**: Comparison questions
- **Hypothetical**: "What if" scenarios
- **In-Breadth**: Scope expansion questions

### Manual HS Code Extraction (Legacy)

Used for `synthetic_goldens_100.json`:
1. Extract HS codes from WCO explanatory notes chunks
2. Generate "Which HS code?" questions from context
3. Validate via retrieval test

---

## Quick Start

### Run Evaluation (DeepEval Synthesizer Dataset)

```bash
# Run LLM reranking
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/deepeval_goldens_korean_100.json \
  --method binary --initial-k 100 --final-k 15 --max-cases 0 \
  --output evaluation/experiments/results/llm_rerank_binary_deepeval_diverse.json

# Run DeepEval answer correctness evaluation
python -m evaluation.experiments.run_deepeval_full binary deepeval_diverse
```

### Run Evaluation (Legacy HS Code Dataset)

```bash
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/synthetic_goldens_100.json \
  --method binary --initial-k 100 --final-k 15 --max-cases 0

python -m evaluation.experiments.run_deepeval_full binary 100
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/synthetic/deepeval_goldens_korean_100.json` | 100 diverse questions (DeepEval Synthesizer) |
| `data/synthetic/synthetic_goldens_100.json` | 100 HS code lookup questions (legacy) |
| `experiments/results/llm_rerank_binary_deepeval_diverse.json` | Reranking results (diverse) |
| `experiments/results/binary_deepeval_diverse_deepeval_correctness.json` | DeepEval scores (diverse) |
| `experiments/results/llm_rerank_baseline_hs_code.json` | Baseline answers for HS code |
| `experiments/results/llm_rerank_baseline_diverse.json` | Baseline answers for diverse RAG |
| `experiments/results/baseline_hs_code_deepeval_correctness.json` | Baseline DeepEval results (85% pass) |
| `experiments/results/baseline_diverse_deepeval_correctness.json` | Baseline DeepEval results (74% pass) |

---

## Baseline Comparison

### What is Baseline?

Baseline configuration removes both key innovations of Konte:
1. **No contextual chunks** - raw chunks without LLM-generated context prepended
2. **No LLM reranking** - top-k chunks from hybrid retrieval used directly

### Baseline vs Contextual Pipeline

| Step | Contextual (Full) | Baseline |
|------|-------------------|----------|
| Chunking | Segment → Chunk | Segment → Chunk |
| Context Generation | ✅ LLM prepends context | ❌ Skipped |
| Embedding | Contextual chunk | Raw chunk |
| Retrieval | Hybrid (FAISS + BM25) | Hybrid (FAISS + BM25) |
| Reranking | ✅ LLM binary filter | ❌ Skipped |
| Answer Generation | Top-k reranked | Top-k direct |

### Scripts for Baseline Evaluation

```bash
# Build baseline project (no context)
python scripts/build_baseline_project.py

# Run baseline evaluation
python -m evaluation.experiments.run_baseline_eval \
  --test-cases evaluation/data/synthetic/deepeval_goldens_korean_100.json \
  --eval-type answer
```

---

## Key Learnings

### 1. Question Type Matters

- **Reasoning questions (100%)**: RAG excels at logical inference from context
- **Hypothetical questions (86.7%)**: Most challenging - "what if" scenarios often not covered in context
- **Constrained/Lookup questions (96.7%)**: Direct fact lookup performs well

### 2. Evaluation Metric Must Match Use Case

- Old metric focused on HS code matching only
- New AnswerCorrectness metric evaluates factual accuracy for all question types
- Same 94% pass rate, but different failure patterns

### 3. Context Coverage is Critical

- Hypothetical failures: context doesn't cover the specific scenario
- The RAG system correctly identifies when information is not available
- Consider: should "I don't have that information" be counted as correct?

---

## Historical Experiments (Archived)

| Version | Test Cases | Source | Pass Rate | Notes |
|---------|------------|--------|-----------|-------|
| v2 | 120 | Chunks | 90.0% | LLM hallucination in some cases |
| v3 | 120 | Chunks | 74.2% | Bad prompt caused hallucination |
| v4 | 120 | Segments | 78.3% | Different questions |
| v5 | 120 | Segments | 78.4% | Gemma-3-27b generated |
| 30_v2 | 30 | Validated | 96.7% | First validated dataset |
| 100 | 100 | Validated | 94.0% | HS code lookup questions |
| **DeepEval Diverse** | **100** | **Synthesizer** | **94.0%** | **Diverse question types** |

Old datasets archived to `data/synthetic/archive/`.
