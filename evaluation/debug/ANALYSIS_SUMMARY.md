# Faithfulness Metric Failure Analysis

> **Note**: This analysis was conducted during early experiments with English test cases against Korean knowledge base. The current best configuration uses Korean test cases (v2) with FactualCorrectness metric, achieving **90% accuracy**. See `EVALUATION_REPORT.md` for latest results.

## Root Cause Identified

The Faithfulness metric failure is **NOT primarily a token limit issue**. It's a combination of:
1. **Language mismatch** between questions and knowledge base
2. **Retrieval quality** issues causing cascading failures

### The Chain of Problems:

1. **Knowledge Base Language Mismatch**
   - Test questions: English (e.g., "What is GRI 1?")
   - Knowledge base (`wco_hs_explanatory_notes`): **Korean translations** of WCO HS Explanatory Notes
   - Retrieved content contains: 통칙 제2호, 통칙 제3호 (Korean for GRI 2, GRI 3)
   - For Test Case 1: RAG answer says "context does not contain information" because the semantic search doesn't find exact matches

2. **Truths Extraction Fails or Is Irrelevant**
   - Korean retrieval context → Model extracts Korean truths (or none)
   - English claims from RAG answer → Cannot match against Korean truths
   - Model returns empty truths list: `{"truths": []}`

3. **Verdicts Generation Explodes**
   - With empty truths, the model must generate "idk" verdicts for each claim
   - The model generates very verbose explanations trying to justify why it can't verify
   - This hits the token limit (4000+ tokens)

### Evidence from Trace:

```
STAGE: 1. Claims Extraction
  Input: 335 tokens
  Output: 27 tokens
  Result: 1 claim extracted ("The context does not contain information...")

STAGE: 2. Truths Extraction
  Input: 1,138 tokens (Korean text about GRI 2)
  Output: 6 tokens
  Result: 0 truths extracted ← PROBLEM!

STAGE: 3. Verdicts Generation
  Input: 386 tokens
  ERROR: Token limit reached (4000 completion tokens used)
```

### Data from enriched_test_cases.json:

| Test Case | Question (English) | Retrieved Content (Korean) | RAG Answer Quality |
|-----------|-------------------|---------------------------|-------------------|
| 1 | GRI 1 | 통칙 제2호가목 (GRI 2a) | ❌ "no information" |
| 2 | GRI 2(a) | GRI 2가목 content | ✓ Good answer |
| 3 | GRI 2(b) | GRI 2나목, GRI 3호 content | ✓ Good answer |
| 4 | GRI 3 | GRI 2나목, GRI 3호 content | ✓ Partial answer |
| 5 | GRI 4 | GRI 2호, 3827호 content | ⚠️ Answers GRI 2 instead |

## Root Cause Analysis

### Why Token Limit Is Hit:

1. **Verdicts stage with empty truths** triggers verbose "idk" responses
2. The model tries to explain WHY it can't verify claims against nothing
3. Each verdict generates ~500-1000 tokens of explanation
4. With multiple claims, this exceeds the output limit

### Why Truths Extraction Returns Empty:

1. Korean context text doesn't match English-language claim extraction patterns
2. Model may not recognize Korean legal/tariff terminology
3. The truths extraction prompt expects similar language between context and output

## Solutions

### Option A: Use Korean Test Cases (Best)
Since the knowledge base is in Korean, create Korean test cases:
```python
EvalTestCase(
    input="통칙 제1호란 무엇이며 어떻게 적용해야 합니까?",
    expected_output="통칙 제1호는 분류는 호의 용어와 관련 부 또는 류의 주에 따라 결정되어야 한다고 규정합니다...",
)
```

### Option B: Use Different Knowledge Base
Use an English language knowledge base for English test cases.

### Option C: Use truths_extraction_limit Parameter
```python
FaithfulnessMetric(
    truths_extraction_limit=10,  # Limit truths to 10 most important
    ...
)
```
This won't fix the language mismatch but will prevent token explosion.

### Option D: Skip Faithfulness for Failed Retrievals
Add pre-check: if RAG answer contains "does not contain information" or similar, skip Faithfulness evaluation.

### Option E: Use Multilingual Model
Use a model that handles Korean-English cross-lingual evaluation better.

## Conclusion

The token limit error is a **symptom**, not the root cause. The real issues are:

1. **Language mismatch**: English test cases vs Korean knowledge base
2. **Retrieval gaps**: GRI 1 and GRI 4 content not being retrieved (possibly not in KB)
3. **Cross-lingual evaluation**: Faithfulness metric can't compare English claims against Korean truths

## Recommended Next Steps

1. **Verify knowledge base content**: Check what languages and content exist in `wco_hs_explanatory_notes`
2. **Create matching test cases**: Either Korean questions for Korean KB, or find/create English KB
3. **Add safeguards**: Use `truths_extraction_limit=10` and skip evaluation for "no context" answers
4. **Consider**: The evaluation shows the RAG system works (4/5 questions got relevant context), but test design needs alignment with actual KB content

## Files Created

- `evaluation/debug/debug_faithfulness.py` - Token analysis for each stage
- `evaluation/debug/trace_faithfulness.py` - Live tracing of LLM calls
- `evaluation/debug/debug_truths_prompt.py` - Shows exact prompts sent
- `evaluation/debug/faithfulness_analysis.json` - Detailed analysis data
- `evaluation/debug/ANALYSIS_SUMMARY.md` - This summary
