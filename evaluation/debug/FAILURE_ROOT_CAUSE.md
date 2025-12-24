# Root Cause Analysis: Failure Rate Investigation

> **Note**: This analysis was conducted during early experiments. The current best configuration (v2 with binary filter + fallback) achieves **90% accuracy**. See `EVALUATION_REPORT.md` for latest results.

## Summary (Early Experiments)

**Total test cases**: 120 (Korean)
**Early failures**: ~30% (HYBRID mode: 22% fail)
**Early successes**: ~70-78%

**Current best (v2)**: 90% accuracy with LLM binary reranking

## Failure Categories

| Category | Count | % | Root Cause |
|----------|-------|---|------------|
| **Retrieval at edge of top-k** | 10+ | ~8% | Relevant chunk at position 17-21, but only 15 sent to LLM |
| **Chunk truncation for metrics** | 5+ | ~4% | Only 10 chunks saved for eval, relevant chunk beyond |
| **No relevant chunks in top-100** | 5 | ~4% | Content truly not retrievable (e.g., 2906.90) |
| **LLM wrong answer** | 10 | ~8% | LLM gave incorrect answer despite having context |

## Key Finding: Retrieval Edge Cases

### Test with different top-k values:

| Case | Target Code | top-k=20 | top-k=50 | top-k=100 |
|------|-------------|----------|----------|-----------|
| 8540 | 8540 | Found at **18** | Found at 20 | Found at 25 |
| 3506 | 3506 | Found at 0 | Found at 0 | Found at 0 |
| 0805.21 | 0805.21 | Found at **17** | Found at 21 | Found at 23 |
| 8465.93 | 8465.93 | NOT FOUND | Found at 21 | Found at 28 |
| 2906.90 | 2906.90 | NOT FOUND | NOT FOUND | NOT FOUND |

### Problem Flow:

```
top-k=20 retrieves 20 chunks
    ↓
max-chunks=15 sent to LLM for answer generation  ← Chunk at position 18 NOT included!
    ↓
eval_chunks=10 saved for metrics  ← Even fewer!
```

**Relevant chunks are found at position 17-21 but cut off by max-chunks=15!**

## Chunk Distribution in Knowledge Base

```
Total chunks: 3036

HS code distribution:
  8540: 14 chunks  (spread across 3036)
  3506: 20 chunks
  0805: 4 chunks   (rare!)
  8465: 16 chunks
  2906: 9 chunks
```

With only 4-20 chunks per HS code among 3036 total, retrieval must be precise.

## Recommendations

### 1. Increase max-chunks to 25 ✅ FIXED
```python
# evaluate_modes.py
parser.add_argument("--max-chunks", type=int, default=25)  # was 15
```

### 2. Match eval_chunks to max-chunks ✅ FIXED
```python
# evaluate_modes.py line 55
retrieval_context = [result.content for result in response.results[:max_chunks]]  # was hardcoded 10
```

### 3. Increase top-k to 30 ✅ FIXED
```python
# evaluate_modes.py
parser.add_argument("--top-k", type=int, default=30)  # was 20
```

## Expected Improvement

| Setting | Approx Accuracy |
|---------|-----------------|
| Old (top-k=20, max-chunks=15, eval=10) | 72-78% |
| New (top-k=30, max-chunks=25, eval=25) | ~85-90% |

## Irreducible Error Rate

~5-8% cases where:
- HS code not in document (e.g., 2906.90)
- LLM reasoning errors
- Ambiguous test questions

---

## Deeper Analysis: Why FAISS Doesn't Always Win

### Mode Comparison (wco baseline)
| Mode | Accuracy | Notes |
|------|----------|-------|
| Hybrid | 78% | Best overall |
| Semantic | 66% | FAISS alone |
| Lexical | 62% | BM25 alone |

### Failure Breakdown (19 "no context" cases)

| Pattern | Count | Example |
|---------|-------|---------|
| Content not in KB | ~50% | 8465.93, 2906.90, 5007.90 - HS codes don't exist in document |
| Semantic finds, BM25 hurts hybrid | ~25% | 8540: Semantic pos 8, Hybrid pos 19 |
| True retrieval failure | ~25% | 3506: Correct chunk not in top 30 for ANY mode |

### BM25 Tokenization Issue

Simple `text.lower().split()` tokenizer causes mismatch:
- Query: `"8540"` (bare token)
- Chunk: `"제8540호"`, `"(제8540호)"` (compound tokens)

Result: BM25 cannot find exact HS code matches.

### Why Hybrid Still Wins Overall

Despite BM25 failures on specific HS code queries:
1. For general tariff questions, BM25 boosts recall of keyword matches
2. RRF combination smooths out individual mode weaknesses
3. FAISS alone misses some lexical patterns BM25 catches

### Semantic Embedding Limitations

Case: Query asks about 3506 exceptions (3501, 3503, 3505)
- Top results: chunks mentioning "3506" in retail context
- Correct chunk with "카세인 글루" exceptions: NOT in top 30

Embeddings don't capture the semantic relationship between:
- "예외 항목" (exception items)
- "카세인 글루...제3501호" (actual exception content)

---

## Context-Embedded vs Content-Only Comparison

### Projects Tested
| Project | What's Embedded | Chunks |
|---------|-----------------|--------|
| wco_hs_explanatory_notes | context + content | 3036 |
| exp_8000_800_context_meta | content only | 3036 |

### Retrieval Position Comparison (20 queries, semantic mode)

| Result | Count |
|--------|-------|
| exp8000 (content-only) better | 6 |
| wco (context+content) better | 4 |
| Tie | 7 |

### Key Insight
- English context helps for English-like queries (cardamom, mandarin)
- Korean-only content works better for Korean HS code queries
- Context sometimes adds noise that hurts Korean term matching

---

## BM25 Korean Tokenization Problem

Simple `text.lower().split()` fails for Korean due to agglutinative grammar:

| Query Token | Chunk Token | Issue |
|-------------|-------------|-------|
| `멘톨은` | `멘톨(menthol)*` | Particle `은` attached |
| `코드에` | `코드` | Particle `에` attached |
| `8540` | `제8540호` | Prefix `제` and suffix `호` |

**Result**: BM25 cannot match Korean terms with grammatical particles.

**Fix needed**: Korean morphological tokenizer (e.g., KoNLPy, Mecab) instead of `.split()`
