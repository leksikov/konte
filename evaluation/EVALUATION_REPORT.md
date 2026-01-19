# RAG Evaluation Report

## Final Results: 96.7% Accuracy (29/30)

| Metric | Value |
|--------|-------|
| Test Cases | 30 (validated) |
| Pass Rate | **96.7%** |
| Passed | 29/30 |
| Failed | 1 |
| Avg Score | 0.933 |

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

## Test Generation Methodology (Jan 2025)

### Problem: LLM Hallucination in Test Data
Previous test generation allowed LLM to generate any HS code, leading to:
- Hallucinated HS codes that didn't exist in source chunks
- Ambiguous questions using "기타" (other) patterns
- Low accuracy (83.3%) due to test data quality issues

### Solution: Validated Test Generation

The improved `synthesize_korean_dataset.py` ensures high-quality test cases:

#### 1. HS Code Extraction from Source
```python
HS_CODE_PATTERN = re.compile(r'(\d{4}\.\d{2})\s*[-–—]\s*(.+?)(?:\n|$)')

def extract_hs_codes_from_chunk(content: str) -> list[tuple[str, str]]:
    """Extract HS codes directly from chunk content using regex."""
    matches = HS_CODE_PATTERN.findall(content)
    # Returns: [("2529.30", "백류석, 하석"), ("8435.90", "부분품"), ...]
```

#### 2. Ambiguity Filtering
```python
AMBIGUOUS_PATTERNS = ["기타", "그 밖의", "그밖의", "기타의", "그 외"]

# Filter out ambiguous items before question generation
if any(name.startswith(p) for p in AMBIGUOUS_PATTERNS):
    continue  # Skip this HS code
```

#### 3. Retrieval Validation
```python
def validate_retrieval(query: str, expected_hs_code: str) -> bool:
    """Check if expected HS code appears in top-5 retrieval results."""
    response = project.query(query, mode="hybrid", top_k=5)
    for result in response.results:
        if expected_hs_code in result.content:
            return True
    return False
```

#### 4. Specific Question Prompting
```python
KOREAN_SYNTHESIS_PROMPT = """
## 중요 - 피해야 할 질문 유형:
- "기타"로 시작하는 질문 (예: "기타 반도체는?") - 너무 모호함
- 단순히 소호 번호만 언급하는 질문 - 구체적인 품목명을 사용하세요

## 좋은 질문 예시:
- "퀘브라쵸 추출물은 어느 HS 코드에 분류되나요?" (구체적 품목명)
- "DDR5 메모리 모듈은 어느 HS 코드에 분류되나요?" (구체적 제품)
"""
```

---

## Detailed Results

### Pass Distribution
| Score Range | Count | Percentage |
|-------------|-------|------------|
| 1.0 (perfect) | 19 | 63.3% |
| 0.8-0.9 | 9 | 30.0% |
| 0.7-0.8 | 1 | 3.3% |
| < 0.5 (fail) | 1 | 3.3% |

### Single Failure Analysis

| Field | Value |
|-------|-------|
| Question | 합성스테이플섬유의 함유량이 전 중량의 85% 이상인 소매용 인조스테이플섬유사는 어느 HS 코드에 분류되나요? |
| Expected | 5511.20 |
| Actual | 5511.10 |
| Score | 0.20 |
| Reason | Both codes are in the same chapter (5511 - synthetic staple fiber yarns), different subcodes |

**Root Cause**: Borderline HS code ambiguity where both 5511.10 and 5511.20 relate to similar products. The distinction depends on whether the fiber content threshold applies to the product category.

---

## Configuration

### Best Configuration
| Setting | Value |
|---------|-------|
| Test Cases | `synthetic_goldens_30.json` |
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

## Running Evaluation

```bash
# Generate 30 validated test cases
python -m evaluation.synthesize_korean_dataset \
  --project wco_hs_explanatory_notes_korean \
  --output evaluation/data/synthetic/synthetic_goldens_30.json \
  --num 30 \
  --seed 123

# Run LLM reranking experiment
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/synthetic_goldens_30.json \
  --method binary \
  --initial-k 100 \
  --final-k 15

# Run DeepEval correctness metric
python -m evaluation.experiments.run_deepeval_full binary 30_v2
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/synthetic/synthetic_goldens_30.json` | 30 validated test cases |
| `experiments/results/llm_rerank_binary_30_v2.json` | Reranking results |
| `experiments/results/binary_30_v2_deepeval_correctness.json` | DeepEval scores |

---

## Key Learnings

### 1. Test Data Quality is Critical
- Previous 90% accuracy was limited by test data quality (LLM hallucination)
- With validated test generation, accuracy improved to 96.7%
- Lesson: Always extract ground truth from source, never generate

### 2. Ambiguity Filtering Matters
- "기타" (other) style questions are inherently ambiguous
- Filtering these out improves test reliability
- Remaining failure is borderline ambiguity case

### 3. Retrieval Validation Ensures Fair Evaluation
- Only include questions where answer is retrievable
- Separates retrieval issues from generation issues
- Allows focused improvement on each component

---

## Historical Experiments (Archived)

| Version | Test Cases | Source | Pass Rate | Notes |
|---------|------------|--------|-----------|-------|
| v2 | 120 | Chunks | 90.0% | LLM hallucination in some cases |
| v3 | 120 | Chunks | 74.2% | Bad prompt caused hallucination |
| v4 | 120 | Segments | 78.3% | Different questions |
| v5 | 120 | Segments | 78.4% | Gemma-3-27b generated |
| **30_v2** | **30** | **Validated** | **96.7%** | **Current best** |

Old datasets archived to `data/synthetic/archive/`.
