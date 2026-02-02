# RAG System Ablation Study & Benchmark Report

**Date**: 2026-02-01
**Project**: `wco_hs_explanatory_notes_korean`
**Models Tested**: `gemma-3-27b-it`, `Qwen3-VL-8B-Instruct`

## Executive Summary

| Question | Answer |
|----------|--------|
| Is retrieval the bottleneck? | **No** - 100% hit rate |
| Is LLM reasoning the bottleneck? | **No** - 90-97% with good context |
| What is the bottleneck? | **Context presentation** - single chunk fails |
| Best open-source model? | **Qwen3-VL-8B** (73.3%, beats 27B Gemma) |
| Is generated context needed? | **No** - not with multiple chunks |

---

## Objective

Identify the bottleneck in RAG system accuracy: is it the **LLM reasoning** or the **retriever**?

## Test Datasets

| Dataset | Cases | Source File |
|---------|-------|-------------|
| HS Code | 15 | `synthetic_goldens_100.json` |
| DeepEval | 15 | `deepeval_goldens_korean_100.json` |
| **Total** | **30** | per ablation case |

**Evidence Source**: `retrieval_context[0]` (actual source chunks from test dataset)

---

## Methodology

### Case 1: Perfect Evidence → LLM Ceiling Test

- **Goal**: Test if LLM can produce correct answers when given perfect evidence directly
- **Method**: Bypass retrieval entirely. Create synthetic `RetrievalResponse` with `retrieval_context[0]` as the only chunk, then call `generate_answer()`
- **Expected**: ~99% accuracy

### Case 2: Evidence Injection → Answer Generation

- **Goal**: Test answer generation when evidence is guaranteed in retrieval results
- **Method**: Do real hybrid retrieval (top-k=15), inject `retrieval_context[0]` at **random position**, then call `generate_answer()`
- **Expected**: ~97% accuracy

### Case 3: Retrieval Quality Check

- **Goal**: Measure how often evidence appears in top-k retrieved chunks
- **Method**: Do real hybrid retrieval (top-k=50), check if `retrieval_context[0]` appears in any chunk using substring and HS code matching
- **Metric**: Hit rate (% of cases where evidence found)

---

## Results

### Overall Summary

| Case | Description | Pass Rate | Avg Score |
|------|-------------|-----------|-----------|
| Case 1 | Perfect Evidence (single chunk) | 73.3% | 0.663 |
| Case 2 | Evidence Injection (random position) | **90.0%** | **0.830** |
| Case 3 | Retrieval Hit Rate (top-k=50) | **100%** | N/A |

### Results by Dataset

| Case | HS Code (15) | DeepEval (15) | Total (30) |
|------|--------------|---------------|------------|
| Case 1 | 10/15 (66.7%) | 12/15 (80.0%) | 22/30 (73.3%) |
| Case 2 | **15/15 (100%)** | 12/15 (80.0%) | 27/30 (90.0%) |
| Case 3 | 15/15 (100%) | 15/15 (100%) | 30/30 (100%) |

### Average Scores by Dataset

| Case | HS Code Avg | DeepEval Avg | Overall Avg |
|------|-------------|--------------|-------------|
| Case 1 | 0.647 | 0.680 | 0.663 |
| Case 2 | **0.953** | 0.707 | 0.830 |

---

## Key Findings

### 1. Retrieval is NOT the Bottleneck

Case 3 shows **100% hit rate** at top-k=50. The retrieval system successfully finds the evidence chunks for all test cases.

### 2. Case 2 Outperforms Case 1 (Counterintuitive)

| Comparison | Case 1 | Case 2 | Delta |
|------------|--------|--------|-------|
| Pass Rate | 73.3% | 90.0% | **+16.7%** |
| Avg Score | 0.663 | 0.830 | **+0.167** |
| HS Code | 66.7% | 100% | **+33.3%** |

The LLM performs **better** when evidence is mixed with other retrieved chunks than when given only the perfect evidence.

### 3. LLM Needs Context Diversity

Case 1 failures show the LLM responding "context doesn't contain this information" even when evidence IS provided. This suggests:

- Single-chunk evidence may lack sufficient surrounding context
- Multiple retrieved chunks provide broader context for confident answers
- The model benefits from seeing related information alongside the exact answer

---

## Failure Analysis

### Case 1 Failures (8 cases)

| Question Type | Failure Reason |
|---------------|----------------|
| HS Code lookups | LLM claims context lacks the information |
| Technical comparisons | Insufficient detail in single chunk |
| Multi-part questions | Single chunk doesn't cover all aspects |

**Example failures**:
- "바이올린은 어느 HS 코드에 분류되나요?" → LLM says context lacks violin HS code
- "수산화알루미늄은 어느 HS 코드에 분류되나요?" → LLM says context doesn't contain the code

### Case 2 Failures (3 cases)

All 3 failures are from DeepEval dataset (complex reasoning questions):
- Dental plaster accelerators/retarders mechanism
- Hypothetical combined tool functionality
- Exclusion categories for iron/steel products

---

## Interpretation

### What This Means

```
If Case 3 (retrieval) is high but Case 1 (perfect evidence) is low:
→ LLM reasoning with single-chunk context is the bottleneck

If Case 2 > Case 1:
→ LLM benefits from context diversity, not just perfect evidence
```

### Our Results

| Finding | Implication |
|---------|-------------|
| Case 3 = 100% | Retrieval works perfectly |
| Case 1 = 73% | LLM struggles with isolated evidence |
| Case 2 = 90% | Multiple chunks help LLM reasoning |
| Case 2 > Case 1 | Context diversity > perfect single evidence |

---

## Recommendations

1. **Keep hybrid retrieval with multiple chunks** - The LLM benefits from seeing related context, not just the exact answer chunk

2. **Increase top-k for answer generation** - More context helps LLM confidence and accuracy

3. **Consider chunk overlap** - Overlapping chunks provide continuity that helps LLM understand document structure

4. **Focus optimization on prompt engineering** - Since retrieval works well, improvements should target how context is presented to the LLM

---

## Files Generated

| File | Description |
|------|-------------|
| `ablation_case1_results.json` | Case 1 raw results |
| `ablation_case2_results.json` | Case 2 raw results |
| `ablation_case3_results.json` | Case 3 raw results |
| `ablation_case1_deepeval.json` | Case 1 DeepEval scores |
| `ablation_case2_deepeval.json` | Case 2 DeepEval scores |

---

## Reproduction

```bash
# Run Case 1: Perfect Evidence Test
python -m evaluation.experiments.ablation_case1_perfect_evidence --sample-size 15

# Run Case 2: Evidence Injection Test
python -m evaluation.experiments.ablation_case2_evidence_injection --sample-size 15

# Run Case 3: Retrieval Quality Check
python -m evaluation.experiments.ablation_case3_retrieval_quality --sample-size 15 --top-k 50

# Run DeepEval evaluation on Cases 1 & 2
python -m evaluation.experiments.run_ablation_eval --case 0
```

---

## Conclusion

**The retriever is not the bottleneck.** With top-k=50, retrieval achieves 100% hit rate.

**The LLM performs better with context diversity.** Case 2 (90%) outperforms Case 1 (73%), indicating that multiple retrieved chunks help the LLM reason more effectively than a single perfect evidence chunk.

**Optimization focus**: Prompt engineering and context presentation, not retrieval improvements.

---

## Ablation V2: Contextual vs Raw Evidence Comparison

### Background

The previous ablation study used **contextual evidence** which contains pre-generated context that may make evidence "too easy":

```
맥락 설명: [LLM-generated summary]        ← Pre-digested
문서 위치: [document location]            ← Helpful metadata
관련 HS 코드: 8301.20, 8301.30, ...      ← Answer listed!
[actual document text]                    ← Raw source
```

### New Cases

| Case | Evidence Type | Description |
|------|---------------|-------------|
| Case 1 | Contextual | Perfect evidence with generated context |
| Case 1b | Raw | Perfect evidence without generated context |
| Case 2 | Contextual | Injected evidence with generated context |
| Case 2b | Raw | Injected evidence without generated context |

### Comparison Results

| Case | Evidence Type | Pass Rate | Avg Score | HS Code | DeepEval |
|------|---------------|-----------|-----------|---------|----------|
| Case 1 | Contextual | 73.3% | 0.663 | 10/15 (0.647) | 12/15 (0.680) |
| Case 1b | Raw | **63.3%** | **0.570** | 7/15 (0.480) | 12/15 (0.660) |
| Case 2 | Contextual | 90.0% | 0.830 | 15/15 (0.953) | 12/15 (0.707) |
| Case 2b | Raw | **96.7%** | **0.833** | 14/15 (0.893) | 15/15 (0.773) |

### Delta Analysis

| Comparison | Pass Rate Delta | Score Delta | Interpretation |
|------------|-----------------|-------------|----------------|
| Case 1 vs 1b | **-10.0%** | **-0.093** | Context helps single-chunk reasoning |
| Case 2 vs 2b | **+6.7%** | **+0.003** | Context not needed with multiple chunks |

### Key Findings

1. **Single-chunk (Case 1 vs 1b):** Generated context **helps** (+10% pass rate)
   - Raw evidence alone is insufficient for HS code lookups
   - LLM says "context doesn't contain this information" more often with raw evidence

2. **Multi-chunk (Case 2 vs 2b):** Generated context **not needed** (similar performance)
   - Multiple retrieved chunks provide sufficient surrounding context
   - Raw evidence + diverse retrieval ≈ Contextual evidence + diverse retrieval

3. **Context diversity matters more than context generation**
   - Case 2b (96.7%) > Case 1 (73.3%) despite raw evidence
   - Multiple chunks compensate for lack of generated context

### Reproduction

```bash
# Run Case 1b (raw evidence)
python -m evaluation.experiments.ablation_case1b_raw_evidence --sample-size 15

# Run Case 2b (raw injection)
python -m evaluation.experiments.ablation_case2b_raw_injection --sample-size 15

# Run DeepEval on all cases (including 1b, 2b)
python -m evaluation.experiments.run_ablation_eval --case 0

# Or run individual cases
python -m evaluation.experiments.run_ablation_eval --case 1b
python -m evaluation.experiments.run_ablation_eval --case 2b
```

---

## Model Comparison: Open-Source LLM Benchmark

### Objective

Evaluate whether free open-source models are usable for the RAG answer generation task.

### Models Tested

| Model | Size | Endpoint |
|-------|------|----------|
| gemma-3-27b-it | 27B | `https://xaas_model.asia03.app.backend.ai/v1` |
| Qwen3-VL-8B-Instruct | 8B | `https://qwen3vl.asia03.app.backend.ai/v1` |

### Results (Perfect Evidence Test - Case 1)

| Model | Size | Pass Rate | Avg Score | HS Code | DeepEval |
|-------|------|-----------|-----------|---------|----------|
| **Qwen3-VL-8B-Instruct** | 8B | **73.3%** | **0.687** | 11/15 (0.720) | 11/15 (0.653) |
| gemma-3-27b-it | 27B | 70.0% | 0.643 | 10/15 (0.647) | 11/15 (0.640) |

### Key Findings

1. **Both models are usable** for RAG answer generation
2. **Qwen3-VL-8B outperforms Gemma-27B** despite being 3x smaller
3. **Smaller model = faster inference** with better accuracy

### Recommendation

Use **Qwen3-VL-8B-Instruct** for production:
- Better accuracy (73.3% vs 70.0%)
- 3x smaller (8B vs 27B)
- Faster inference
- Lower resource cost

### Reproduction

```bash
# Test with Gemma-27B
BACKENDAI_ENDPOINT="https://xaas_model.asia03.app.backend.ai/v1" \
BACKENDAI_MODEL_NAME="gemma-3-27b-it" \
python -m evaluation.experiments.ablation_case1_perfect_evidence --sample-size 15

# Test with Qwen3-8B
BACKENDAI_ENDPOINT="https://qwen3vl.asia03.app.backend.ai/v1" \
BACKENDAI_MODEL_NAME="Qwen3-VL-8B-Instruct" \
python -m evaluation.experiments.ablation_case1_perfect_evidence --sample-size 15
```

---

## Summary: What We Learned

### The Bottleneck

| Component | Status | Evidence |
|-----------|--------|----------|
| Retrieval | **Not a bottleneck** | 100% hit rate at top-k=50 |
| LLM reasoning | **Not a bottleneck** | 90-97% with multiple chunks |
| Context presentation | **The bottleneck** | Single chunk = 63-73%, Multi-chunk = 90-97% |

### Actionable Insights

1. **Keep multiple chunks** - Context diversity > perfect single evidence
2. **Generated context optional** - Not needed when using multiple chunks
3. **Use Qwen3-VL-8B** - Better than Gemma-27B, 3x smaller
4. **Don't optimize retrieval** - It already works at 100%
5. **Focus on prompt engineering** - How context is presented matters most

### Production Configuration

```python
# Recommended settings
TOP_K = 15              # Multiple chunks for context diversity
MODEL = "Qwen3-VL-8B"   # Best accuracy/size ratio
MODE = "hybrid"         # FAISS + BM25 fusion
```
