# Evaluation Guide

Step-by-step guide to run the RAG evaluation pipeline: test generation, LLM reranking, and DeepEval metrics.

**Important**: Konte is a RAG (Retrieval-Augmented Generation) system for answering questions based on retrieved context. It is NOT a classification system.

## Prerequisites

```bash
# Install dependencies
pip install -e .
pip install deepeval tqdm

# Set environment variables
export OPENAI_API_KEY=sk-...
export BACKENDAI_ENDPOINT=https://qwen3vl.asia03.app.backend.ai/v1
```

---

## Two Evaluation Types

The system supports two evaluation types with switchable LLM judge prompts:

| Type | Dataset | Metric | Description |
|------|---------|--------|-------------|
| `answer` | `deepeval_goldens_korean_no_hypothetical.json` | AnswerCorrectness | 70 diverse RAG questions (recommended) |
| `hs_code` | `synthetic_goldens_100_fixed.json` | HSCodeCorrectness | 100 HS code lookup questions |

**Notes on datasets:**
- Diverse dataset filtered to 70 questions - removed 30 hypothetical questions that require inference beyond source documents
- HS Code dataset has 3 ground truth errors fixed (5403.10→5403.31, 5511.20→5511.10, 6001.10→6001.21)

Prompts are defined in `evaluation/prompts/eval_prompts.py`.

---

## Quick Start (Use Existing Test Cases)

### Option 1: Diverse RAG Questions (Recommended)

Run evaluation on DeepEval Synthesizer generated questions (6 Evolution types, hypothetical removed):

```bash
# Step 1: Run LLM reranking (generates answers)
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/deepeval_goldens_korean_no_hypothetical.json \
  --method binary \
  --initial-k 100 \
  --final-k 15 \
  --max-cases 0 \
  --output evaluation/experiments/results/llm_rerank_binary_deepeval_diverse.json

# Step 2: Run DeepEval evaluation (answer correctness)
python -m evaluation.experiments.run_deepeval_full binary deepeval_diverse answer
```

**Expected results**: 98.6% pass rate (69/70)

Results saved to:
- `evaluation/experiments/results/llm_rerank_binary_deepeval_diverse.json` (answers)
- `evaluation/experiments/results/binary_deepeval_diverse_deepeval_correctness.json` (scores)

### Option 2: HS Code Lookup Questions

Run evaluation on HS code classification questions (ground truth fixed):

```bash
# Step 1: Run LLM reranking (generates answers)
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/synthetic_goldens_100_fixed.json \
  --method binary \
  --initial-k 100 \
  --final-k 15 \
  --max-cases 0

# Step 2: Run DeepEval evaluation (HS code correctness)
python -m evaluation.experiments.run_deepeval_full binary 100 hs_code
```

**Expected results**: 97% pass rate (97/100)

Results saved to:
- `evaluation/experiments/results/llm_rerank_binary_100.json` (answers)
- `evaluation/experiments/results/binary_100_deepeval_correctness.json` (scores)

---

## Full Pipeline: Generate New Test Cases

### Step 1: Generate Validated Test Cases

```bash
python -m evaluation.synthesize_korean_dataset \
  --project wco_hs_explanatory_notes_korean \
  --output evaluation/data/synthetic/synthetic_goldens_100.json \
  --num 100 \
  --seed 42
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--project` | Konte project name | Required |
| `--output` | Output JSON file path | Required |
| `--num` | Number of test cases to generate | 30 |
| `--seed` | Random seed for reproducibility | 123 |

**What it does:**
1. Loads chunks from the Konte project
2. Extracts HS codes using regex pattern: `(\d{4}\.\d{2})\s*[-–—]\s*(.+?)(?:\n|$)`
3. Filters out ambiguous items ("기타", "그 밖의", etc.)
4. Generates Korean questions using LLM with specific HS code context
5. Validates each question by checking if HS code appears in top-5 retrieval
6. Saves validated questions to JSON

**Output format:**
```json
[
  {
    "input": "포도주 제조용 프레스 기계의 부품은 어느 HS 코드에 분류되나요?",
    "expected_output": "8435.90",
    "context": "...chunk content..."
  }
]
```

### Step 2: Run LLM Reranking Experiment

```bash
python -m evaluation.experiments.llm_reranking \
  --project wco_hs_explanatory_notes_korean \
  --test-cases evaluation/data/synthetic/synthetic_goldens_100.json \
  --method binary \
  --initial-k 100 \
  --final-k 15 \
  --max-cases 0
```

**Arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--project` | Konte project name | Required |
| `--test-cases` | Path to test cases JSON | Required |
| `--method` | Reranking method (`binary`, `score`) | `binary` |
| `--initial-k` | Initial retrieval candidates | 100 |
| `--final-k` | Final chunks after reranking | 15 |
| `--max-cases` | Limit test cases (0 = all) | 10 |

**What it does:**
1. For each test case:
   - Retrieves top-k candidates using hybrid search
   - Applies LLM binary filter to select relevant chunks
   - Uses fallback if too few chunks selected
   - Generates answer using Qwen3-VL-8B-Instruct
2. Saves results with actual outputs

**Output file:** `evaluation/experiments/llm_rerank_results.json`

Copy to versioned location:
```bash
cp evaluation/experiments/llm_rerank_results.json \
   evaluation/experiments/results/llm_rerank_binary_100.json
```

### Step 3: Run DeepEval Correctness Evaluation

```bash
# For diverse RAG questions (default)
python -m evaluation.experiments.run_deepeval_full binary deepeval_diverse answer

# For HS code lookup questions
python -m evaluation.experiments.run_deepeval_full binary 100 hs_code
```

**Arguments:**
- `method`: Reranking method name (e.g., `binary`)
- `version`: Dataset version suffix (e.g., `100`, `deepeval_diverse`)
- `eval_type`: Evaluation type - `answer` (default) or `hs_code`

**What it does:**
1. Loads reranking results from `results/llm_rerank_{method}_{version}.json`
2. Selects appropriate prompt based on eval_type:
   - `answer`: Uses AnswerCorrectness criteria (for diverse RAG questions)
   - `hs_code`: Uses HSCodeCorrectness criteria (for HS code lookup)
3. For each case, creates DeepEval test case with:
   - `input`: Original question
   - `actual_output`: Generated answer
   - `expected_output`: Ground truth answer/HS code
4. Evaluates using G-Eval metric
5. Calculates pass rate (score >= 0.5 = PASS)

**Output file:** `evaluation/experiments/results/{method}_{version}_deepeval_correctness.json`

---

## Metrics Explained

Two GEval metrics are available in `evaluation/prompts/eval_prompts.py`:

### AnswerCorrectness (for Diverse RAG Questions)

Evaluates if the actual output correctly answers the question based on the expected output:

```python
metric = GEval(
    name="AnswerCorrectness",
    criteria="""Evaluate if the actual output correctly answers the question based on the expected output.

This is a RAG (Retrieval-Augmented Generation) system evaluation. The system retrieves relevant context and generates answers to user questions.

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
- Score 0.0-0.2: Wrong information or contradicts expected output""",
    threshold=0.5,
)
```

### HSCodeCorrectness (for HS Code Lookup Questions)

Evaluates if the actual output contains the same HS code as expected:

```python
metric = GEval(
    name="HSCodeCorrectness",
    criteria="""Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on HS code accuracy and semantic equivalence:
- The key information is the HS CODE (e.g., 2523.21, 제8540호, 8540.20)
- Ignore format differences: "제2523.21호" = "2523.21" = "제2523호의 21" (all equivalent)
- Ignore language mixing (Korean/English)
- Ignore length differences or extra explanation

Scoring:
- Score 1.0 if the SAME HS CODE is mentioned (regardless of format)
- Score 0.7-0.9 if mostly correct with minor code variations
- Score 0.4-0.6 if partially correct (related but not exact code)
- Score 0.0-0.3 if wrong HS code or contradictory information""",
    threshold=0.5,
)
```

### Pass/Fail Threshold
- **PASS**: Score >= 0.5
- **FAIL**: Score < 0.5

### Score Interpretation (AnswerCorrectness)
| Score | Meaning |
|-------|---------|
| 1.0 | All key facts match, complete and accurate |
| 0.7-0.9 | Most key facts match, minor omissions |
| 0.5-0.6 | Partially correct, some key facts present |
| 0.3-0.4 | Few key facts match, significant info missing |
| < 0.3 | Wrong or contradictory information |

### Score Interpretation (HSCodeCorrectness)
| Score | Meaning |
|-------|---------|
| 1.0 | Exact HS code match |
| 0.8-0.9 | Correct chapter, minor subcode variation |
| 0.7 | Related code, same product family |
| 0.5-0.6 | Partially correct |
| < 0.5 | Wrong HS code or missing |

---

## Test Generation Methodology

### HS Code Extraction

Uses regex to extract HS codes directly from chunk content:

```python
HS_CODE_PATTERN = re.compile(r'(\d{4}\.\d{2})\s*[-–—]\s*(.+?)(?:\n|$)')

# Example matches:
# "2529.30 - 백류석, 하석" → ("2529.30", "백류석, 하석")
# "8435.90 – 부분품" → ("8435.90", "부분품")
```

### Ambiguity Filtering

Filters out items with ambiguous patterns:

```python
AMBIGUOUS_PATTERNS = ["기타", "그 밖의", "그밖의", "기타의", "그 외"]

# Skipped: "기타 반도체", "그 밖의 플라스틱"
# Included: "퀘브라쵸 추출물", "DDR5 메모리 모듈"
```

### Retrieval Validation

Only includes questions where expected HS code is retrievable:

```python
def validate_retrieval(query: str, expected_hs_code: str) -> bool:
    response = project.query(query, mode="hybrid", top_k=5)
    for result in response.results:
        if expected_hs_code in result.content:
            return True
    return False
```

---

## Output File Formats

### Test Cases (`synthetic_goldens_*.json`)
```json
[
  {
    "input": "면 린터 펄프(cotton linters pulp)는 어느 HS 코드에 분류되나요?",
    "expected_output": "4706.10",
    "context": "제4706호 펄프..."
  }
]
```

### Reranking Results (`llm_rerank_*.json`)
```json
{
  "method": "binary",
  "initial_k": 100,
  "final_k": 15,
  "results": [
    {
      "input": "...",
      "expected_output": "4706.10",
      "actual_output": "면 린터 펄프는 HS 코드 4706.10에 분류됩니다...",
      "status": "success",
      "chunks_used": 15,
      "retrieval_time": 0.5,
      "generation_time": 2.1
    }
  ]
}
```

### DeepEval Results (`*_deepeval_correctness.json`)
```json
{
  "method": "binary_100",
  "metric": "deepeval_correctness",
  "avg_score": 0.918,
  "pass_rate": 0.94,
  "passed": 94,
  "failed": 6,
  "total": 100,
  "detailed_results": [
    {
      "input": "...",
      "score": 1.0,
      "reason": "Both outputs reference the same HS code...",
      "passed": true
    }
  ]
}
```

---

## Troubleshooting

### "503 Service Unavailable" during reranking
- OpenAI API rate limit or temporary outage
- Script handles this gracefully with fallback
- Re-run if too many failures

### Low pass rate on new test cases
1. Check if HS codes exist in source chunks
2. Verify retrieval validation is working
3. Check for ambiguous questions (filter more patterns)

### DeepEval evaluation errors
- Ensure `llm_rerank_results.json` exists
- Check JSON format is valid
- Verify OpenAI API key is set

---

## Best Practices

1. **Use `caffeinate`** for long-running processes:
   ```bash
   caffeinate -i python -m evaluation.experiments.llm_reranking ...
   ```

2. **Set `--max-cases 0`** to run all test cases (default is 10)

3. **Use consistent seeds** for reproducible test generation:
   ```bash
   --seed 42  # or any fixed number
   ```

4. **Monitor progress** with tqdm output showing:
   - `valid`: Validated questions generated
   - `gen_err`: Question generation errors
   - `val_fail`: Retrieval validation failures

5. **Logging visibility**: The pipeline logs progress at info level during evaluation runs. For troubleshooting, enable debug logging to see granular events (token counts, individual segments):
   ```python
   import structlog
   structlog.configure(
       wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
   )
   ```
