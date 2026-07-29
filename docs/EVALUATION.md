# Evaluation Results

Konte's contextual retrieval pipeline was evaluated end-to-end as a RAG system (retrieve → generate → LLM-as-judge scoring) using [DeepEval](https://github.com/confident-ai/deepeval) GEval metrics.

> **Note on reproducibility**: These results were measured on a private Korean customs corpus (WCO HS explanatory notes, ~3,000 chunks) that is not distributed with this repository. The evaluation harness in `evaluation/` is generic and can be pointed at your own Konte project and test set.

## Headline Results

### Contextual RAG vs Baseline (December 2025)

| Configuration | HS Code lookup (100 q) | Diverse RAG (70 q)* |
|---------------|------------------------|---------------------|
| **Contextual chunks + reranking** | **97% (0.940)** | **98.6% (0.831)** |
| Baseline (no context, no reranking) | 85% (0.822) | 74% (0.613) |
| **Improvement** | **+12% (+0.118)** | **+25% (+0.218)** |

*After removing 30 hypothetical questions that require inference beyond the source documents.

Key findings:

- LLM-generated chunk context provides the largest gains on complex, multi-context questions (+25%).
- Exact-lookup questions (HS code retrieval) still benefit substantially (+12%).
- Ground-truth validation matters: 3 labeling errors were found and fixed in the test set during evaluation.

### Answer-Model Comparison (February 2026)

Same retrieval pipeline, different answer-generation models:

| Model | HS Code (100 q) | Diverse RAG (70 q) |
|-------|-----------------|---------------------|
| **gpt-4.1** | **95.0% (0.950)** | **91.4% (0.816)** |
| Qwen3-30B-A3B-Instruct | 89.0% (0.910) | 80.0% (0.797) |

Key finding: the answer model matters more than prompt tuning — gpt-4.1 outperforms Qwen3-30B by +6% (HS code) and +11% (diverse questions) with an identical retrieval stack.

### Results by Question Type (Diverse RAG, 70 questions)

| Question type | Pass rate | Avg score |
|---------------|-----------|-----------|
| Reasoning | 100.0% | 0.861 |
| Comparative | 96.9% | 0.841 |
| Constrained | 96.7% | 0.857 |
| In-breadth | 95.7% | 0.852 |
| Multi-context | 92.9% | 0.814 |
| Concretizing | 91.2% | 0.809 |

## Methodology

1. **Test set generation**: DeepEval Synthesizer produced diverse question types (reasoning, multi-context, comparative, constrained, concretizing, in-breadth) from the source corpus; a separate 100-question exact-lookup set was extracted manually and validated.
2. **Retrieval**: hybrid FAISS + BM25 with reciprocal rank fusion, initial k=100, LLM-reranked to a final k=15.
3. **Answer generation**: retrieved chunks passed to the answer model via `query_with_answer()`.
4. **Scoring**: GEval LLM-as-judge with two metrics — *AnswerCorrectness* (key facts present, semantic equivalence, technical accuracy; format/language differences ignored) and *HSCodeCorrectness* (code accuracy with format normalization, partial credit for parent codes).
5. Hypothetical "what if" questions were excluded, since they require inference beyond the source documents and unfairly penalize any retrieval system.

## Running Your Own Evaluation

Install the evaluation dependencies first:

```bash
uv sync --group eval
```

The reusable harness lives in `evaluation/`:

- `deepeval_synthesizer.py` — generate a diverse test set from any Konte project (`python -m evaluation.deepeval_synthesizer --help`)
- `custom_metrics.py` / `prompts/eval_prompts.py` — GEval judge metrics (AnswerCorrectness, HSCodeCorrectness)
- `custom_llm.py` — DeepEval model wrapper for any OpenAI-compatible endpoint

A typical loop: synthesize goldens from your project's chunks, answer them with `project.query_with_answer()`, then score actual vs expected answers with the GEval metrics.
