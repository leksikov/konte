# RAG Evaluation Report

## Final Results: 94.0% Accuracy (94/100)

| Metric | Value |
|--------|-------|
| Test Cases | 100 (validated) |
| Pass Rate | **94.0%** |
| Passed | 94/100 |
| Failed | 6 |
| Avg Score | 0.918 |

### Results by Dataset Size

| Dataset | Cases | Pass Rate | Avg Score | Date |
|---------|-------|-----------|-----------|------|
| **100 validated** | **100** | **94.0%** | **0.918** | **Jan 2025** |
| 30 validated | 30 | 96.7% | 0.933 | Jan 2025 |

---

## Methodology

### Evaluation Framework
- **Metric**: DeepEval G-Eval FactualCorrectness
- **Judge Model**: gpt-4.1-mini (OpenAI)
- **Threshold**: Score >= 0.5 = PASS

### FactualCorrectness Criteria
Evaluates if the actual RAG output contains the same key factual information as the expected output:
- Checks if key HS codes and facts from expected output appear in actual output
- Ignores: length differences, format differences, language mixing, ordering
- Score 1.0: All key facts present (same HS code)
- Score 0.7-0.9: Most facts present
- Score 0.4-0.6: Some facts missing
- Score 0.0-0.3: Wrong or missing key facts

### RAG Pipeline
1. **Retrieval**: Hybrid (FAISS semantic + BM25 lexical) with RRF fusion
2. **Initial Retrieval**: top-k=100 candidates
3. **LLM Reranking**: Binary filter to select relevant chunks
4. **Final Context**: top-k=15 chunks passed to answer generation
5. **Answer Generation**: Qwen3-VL-8B-Instruct (BackendAI)

---

## Failure Analysis (6/100)

### Summary by Category

| Category | Count | Failure Rate |
|----------|-------|--------------|
| Textile subcode ambiguity | 4 | 4.0% |
| Chemical classification | 1 | 1.0% |
| Fabric type distinction | 1 | 1.0% |

### Detailed Failures

| # | Question | Expected | Actual | Score | Root Cause |
|---|----------|----------|--------|-------|------------|
| 1 | 인조섬유로 만든 저지 풀오버 카디건 | 6110.30 | 6101/6102 | 0.2 | Knitwear vs coat chapter confusion |
| 2 | 비스코스 레이온 꼬임 없는 필라멘트사 | 5403.10 | 5403.31 | 0.2 | Twist threshold subcode error |
| 3 | 시아노겐 클로라이드(클로르시안) | 2853.10 | 2812 | 0.2 | Wrong chapter (inorganic compounds) |
| 4 | 합성스테이플섬유 85%+ 인조스테이플섬유사 | 5511.20 | 5511.10 | 0.0 | Fiber content threshold ambiguity |
| 5 | 제5602/5603호 직물 여성용 실내복 | 6210.10 | 6208 | 0.2 | Special fabric vs woven chapter |
| 6 | 롱파일 면 메리야스 편물 | 6001.10 | 6001.21 | 0.0 | Looped vs long pile distinction |

### Analysis

**Textile Classification Complexity (4/6 failures)**:
- HS chapters 54-62 cover textiles with complex subcode distinctions
- Subcodes depend on: fiber content percentage, construction method, end-use
- Similar products can fall into adjacent codes based on subtle criteria

**Chemical Classification (1/6 failures)**:
- Inorganic compounds have specific placement rules
- 시아노겐 클로라이드 could be interpreted as halogen compound (2812) or other inorganic (2853)

---

## Score Distribution (100 cases)

| Score Range | Count | Percentage |
|-------------|-------|------------|
| 1.0 (perfect) | 58 | 58.0% |
| 0.8-0.99 | 25 | 25.0% |
| 0.7-0.79 | 6 | 6.0% |
| 0.5-0.69 | 5 | 5.0% |
| < 0.5 (fail) | 6 | 6.0% |

---

## Configuration

### Best Configuration (94.0% Accuracy)

| Setting | Value |
|---------|-------|
| Test Cases | `synthetic_goldens_100.json` |
| Test Generation | HS code extraction + retrieval validation |
| Reranking | Binary filter with fallback |
| Initial K | 100 |
| Final K | 15 |
| Model | gpt-4.1-mini (eval), Qwen3-VL-8B-Instruct (answer) |

### Settings (`konte/config/settings.py`)
```python
EMBEDDING_MODEL = "text-embedding-3-small"
CONTEXT_MODEL = "gpt-4.1"
DEFAULT_TOP_K = 20
SEGMENT_SIZE = 8000  # tokens
CHUNK_SIZE = 800     # tokens
BACKENDAI_MODEL_NAME = "Qwen3-VL-8B-Instruct"
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/synthetic/synthetic_goldens_100.json` | 100 validated test cases |
| `data/synthetic/synthetic_goldens_30.json` | 30 validated test cases |
| `experiments/results/llm_rerank_binary_100.json` | Reranking results (100) |
| `experiments/results/binary_100_deepeval_correctness.json` | DeepEval scores (100) |

---

## Key Learnings

### 1. Scale Reveals Edge Cases
- 30-case dataset: 96.7% accuracy (1 failure)
- 100-case dataset: 94.0% accuracy (6 failures)
- Larger datasets expose more challenging classification boundaries

### 2. Textile Classification is Hardest
- 4/6 failures involve textile products (chapters 54-62)
- HS code subcodes have subtle distinctions based on fiber content, construction
- Consider domain-specific fine-tuning for textile questions

### 3. Test Data Quality Remains Critical
- Validated test generation (HS code extraction + retrieval validation) essential
- Previous LLM-generated test data had hallucination issues
- Ground truth must come from source documents

---

## Historical Experiments (Archived)

| Version | Test Cases | Source | Pass Rate | Notes |
|---------|------------|--------|-----------|-------|
| v2 | 120 | Chunks | 90.0% | LLM hallucination in some cases |
| v3 | 120 | Chunks | 74.2% | Bad prompt caused hallucination |
| v4 | 120 | Segments | 78.3% | Different questions |
| v5 | 120 | Segments | 78.4% | Gemma-3-27b generated |
| 30_v2 | 30 | Validated | 96.7% | First validated dataset |
| **100** | **100** | **Validated** | **94.0%** | **Current best** |

Old datasets archived to `data/synthetic/archive/`.
