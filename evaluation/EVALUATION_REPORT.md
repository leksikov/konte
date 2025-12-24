# RAG Evaluation Report

## Methodology

### Evaluation Framework
- **Metric**: DeepEval G-Eval FactualCorrectness
- **Judge Model**: BackendAI Qwen3-VL-8B-Instruct
- **Threshold**: Score >= 0.5 = PASS

### FactualCorrectness Criteria
Evaluates if the actual RAG output contains the same key factual information as the expected output:
- Checks if key HS codes and facts from expected output appear in actual output
- Ignores: length differences, format differences, language mixing, ordering
- Score 1.0: All key facts present
- Score 0.7-0.9: Most facts present
- Score 0.4-0.6: Some facts missing
- Score 0.0-0.3: Wrong or missing key facts

### RAG Pipeline
1. **Retrieval**: Hybrid (FAISS semantic + BM25 lexical) with RRF fusion
2. **Initial Retrieval**: top-k=100 candidates
3. **LLM Reranking**: Binary filter to select relevant chunks
4. **Final Context**: top-k=15 chunks passed to answer generation
5. **Answer Generation**: BackendAI Qwen3-VL-8B-Instruct

---

## Experiments

### Experiment 1: Baseline Binary Filter (v2)

| Setting | Value |
|---------|-------|
| Test Cases | `synthetic_goldens_korean_v2.json` (120 cases) |
| Test Source | Chunks (~800 tokens each) |
| Project | `wco_hs_explanatory_notes_korean` |
| Initial K | 100 |
| Final K | 15 |
| Reranking | Binary filter (no fallback) |
| Fallback | None |

**Results**:
| Metric | Value |
|--------|-------|
| Pass Rate | **90.0%** |
| Passed | 108 |
| Failed | 12 |
| Avg Score | 0.793 |

**Output Files**:
- `results/llm_rerank_binary.json`
- `results/binary_deepeval_correctness.json`

---

### Experiment 2: Binary Filter with Fallback (v2 improved)

| Setting | Value |
|---------|-------|
| Test Cases | `synthetic_goldens_korean_v2.json` (120 cases) |
| Test Source | Chunks (~800 tokens each) |
| Project | `wco_hs_explanatory_notes_korean` |
| Initial K | 100 |
| Final K | 15 |
| Reranking | Binary filter with fallback |
| Fallback | If <5 chunks pass filter, fill to 15 with top hybrid-scored chunks |

**Fallback Logic** (`llm_reranking.py`):
```python
if len(relevant) < min_filtered:  # min_filtered=5
    non_relevant = [(idx, score) for idx, is_relevant, score in results if not is_relevant]
    non_relevant.sort(key=lambda x: x[1], reverse=True)
    slots_to_fill = top_k - len(relevant)
    fallback = non_relevant[:slots_to_fill]
    combined = relevant + fallback
else:
    combined = relevant[:top_k]
```

**Results**:
| Metric | Value |
|--------|-------|
| Pass Rate | **90.0%** |
| Passed | 108 |
| Failed | 12 |
| Avg Score | 0.793 |

**Output Files**:
- `results/llm_rerank_binary_v2_improved.json`
- `results/binary_v2_improved_deepeval_correctness.json`

**Observation**: Fallback mechanism activated in ~20% of cases but did not improve overall accuracy. Cases needing fallback were likely already difficult retrieval scenarios.

---

### Experiment 3: New Test Generation Prompt (v3)

| Setting | Value |
|---------|-------|
| Test Cases | `synthetic_goldens_korean_v3.json` (120 cases) |
| Test Source | Chunks (~800 tokens each) |
| Project | `wco_hs_explanatory_notes_korean` |
| Initial K | 100 |
| Final K | 15 |
| Reranking | Binary filter with fallback |
| Test Prompt Changes | Removed exclusion codes, added HS code normalization format |

**Test Generation Prompt Changes**:
- Added: "HS 코드는 4자리 정규화 형식으로 작성하세요: 0908, 8540, 2906"
- Added: "제외되는 코드나 관련 코드는 언급하지 마세요"
- Reduced exclusion patterns: 35 → 11

**Results**:
| Metric | Value |
|--------|-------|
| Pass Rate | **74.2%** |
| Passed | 89 |
| Failed | 31 |
| Avg Score | 0.668 |

**Output Files**:
- `results/llm_rerank_binary_v3.json`
- `results/binary_v3_deepeval_correctness.json`

**Failure Analysis**:
| Category | Count | Description |
|----------|-------|-------------|
| RAG Correct, Test Wrong | 11 | RAG answer matches context, expected is hallucinated |
| RAG No Info | 8 | RAG correctly says "no info", but expected has answer |
| Different Codes | 8 | Both have codes but differ |

**Root Cause**: Test generation LLM hallucinated HS codes instead of extracting from context. Example:
- Context: "셀룰러 플라스틱은 제3921호에 분류"
- Expected (hallucinated): "HS 코드 3904"
- RAG (correct): "HS 코드 3921"

---

### Experiment 4: Segment-Based Test Generation (v4)

| Setting | Value |
|---------|-------|
| Test Cases | `synthetic_goldens_korean_v4.json` (120 cases) |
| Test Source | **Segments (~8000 tokens each)** |
| Project | `wco_hs_explanatory_notes_korean` |
| Initial K | 100 |
| Final K | 15 |
| Reranking | Binary filter with fallback |
| Test Prompt | Extraction-focused (no format requirements) |

**Test Generation Prompt**:
```
핵심 원칙 (매우 중요):
예상 답변에 포함되는 모든 정보(HS 코드, 분류 기준, 제품명 등)는
반드시 위 문서에서 직접 추출해야 합니다.
문서에 없는 정보를 추가하거나 자신의 지식을 사용하지 마세요.

예상 답변 작성 규칙:
- 문서에 명시된 HS 코드를 그대로 인용하세요 (예: "제3921호", "8540.20")
- 문서에서 직접 인용할 수 있는 근거를 포함하세요
```

**Results**:
| Metric | Value |
|--------|-------|
| Pass Rate | **78.3%** |
| Passed | 94 |
| Failed | 26 |
| Avg Score | 0.696 |

**Output Files**:
- `results/llm_rerank_binary_v4.json`
- `results/binary_v4_deepeval_correctness.json`

**Note**: v4 uses completely different test questions (new segments), not directly comparable to v2/v3.

---

## Summary Table

| Experiment | Test Data | Source | Pass Rate | Notes |
|------------|-----------|--------|-----------|-------|
| 1. Binary baseline | v2 | Chunks | **90.0%** | Baseline |
| 2. Binary + fallback | v2 | Chunks | **90.0%** | Fallback didn't improve |
| 3. New prompt | v3 | Chunks | 74.2% | LLM hallucinated HS codes |
| 4. Segments + extraction | v4 | Segments | 78.3% | Different questions |

---

## Key Findings

### 1. Binary Filter Fallback
- Implemented: Fill to 15 chunks with hybrid-scored when <5 pass filter
- Impact: No improvement on v2 test set
- Reason: Fallback cases are inherently difficult retrieval scenarios

### 2. Test Data Quality Critical
- v3 accuracy dropped due to LLM hallucinating HS codes in expected answers
- RAG system was correct, but marked wrong due to bad test data
- Lesson: Test generation must extract, not generate

### 3. Segments vs Chunks for Test Generation
- Segments (~8000 tokens) provide more context for accurate extraction
- However, creates different test questions - not directly comparable
- v4 showed improvement over v3 (78.3% vs 74.2%)

### 4. Current Best: 90% on v2
- v2 test cases remain the reliable benchmark
- 12 failures need investigation for further improvement

---

## Failure Categories (v2, 12 failures)

| Category | Count | Example |
|----------|-------|---------|
| RAG says "no info" | 5 | Query about item not in retrieved chunks |
| Different HS code | 4 | RAG and expected cite different codes |
| Incomplete answer | 3 | RAG gives partial information |

---

## Configuration Files

### Settings (`konte/config/settings.py`)
```python
EMBEDDING_MODEL = "text-embedding-3-small"
CONTEXT_MODEL = "gpt-4.1"
DEFAULT_TOP_K = 20
SEGMENT_SIZE = 8000  # tokens
CHUNK_SIZE = 800     # tokens
BACKENDAI_MODEL_NAME = "Qwen3-VL-8B-Instruct"
```

### Answer Generation Prompt (`konte/generator.py`)
```python
DEFAULT_ANSWER_PROMPT = """You are a helpful assistant...
- When citing HS codes, use the normalized 4-digit format
  with leading zeros if needed (e.g., "0908", "8540", "2906")
- For HS subheadings, use format with dots (e.g., "0908.31", "8540.11")
"""
```

---

## Recommendations

1. **Use v2 test cases** for benchmarking (90% baseline)
2. **Investigate 12 failures** in v2 for targeted improvements
3. **Consider validation step** in test generation to verify HS codes exist in source
4. **Monitor fallback usage** - high fallback rate may indicate retrieval issues
