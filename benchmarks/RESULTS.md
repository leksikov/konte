# Performance: before and after

Measured comparison of the two revisions. Every row is a claim from the
changelog, the two numbers behind it, and whether it held up.

| Claim | Stated | Before | After | Change | Verdict |
|---|---|---|---|---|---|
| Tokenizer resolved on first use, not at import | import no longer builds the BPE table | 2,310.73 ms | 594.76 ms | -74.3% | confirmed |
| BPE files fetched during import | none | 1 | 0 | -100.0% | confirmed |
| Tokenizer calls per build | konte stopped encoding each string twice | 19,439 | 19,388 | -0.3% | confirmed |
| Text splitters constructed per build | 2 instead of one per segment | 51 | 2 | -96.1% | confirmed |
| Checkpoint bytes written during a build | ~1.7 GB -> ~7 MB at 500 segments | 511.6 MB | 4.6 MB | -99.1% | confirmed |
| Project directory size | 4.97x -> 2.94x the size of the text | 49.7 MB | 30.2 MB | -39.4% | confirmed |
| Lexical index file | ~25 MB -> ~2 MB at 20k chunks | 13.4 MB | 3.1 MB | -76.6% | confirmed |
| Duplicated corpus file (bm25_chunks.json) | removed | 9.2 MB | - | - | confirmed, gone |
| Project directory as a multiple of its text | 4.97x -> 2.94x | 6.27x | 3.80x | -39.4% | confirmed |
| Project directory as a multiple of its text, non-ASCII | 4.97x -> 2.94x | 10.07x | 5.05x | -49.9% | confirmed |
| Project directory size, non-ASCII corpus | not claimed; JSON stopped escaping non-ASCII | 47.8 MB | 24.0 MB | -49.9% | confirmed |
| Corpus JSON, non-ASCII text | not claimed; was pretty-printed and escaped | 9.8 MB | 5.8 MB | -40.4% | confirmed |
| Opening a project | indexes only; corpus parsed on first use | 1,023.56 ms | 515.97 ms | -49.6% | confirmed |
| Per-request project lookup | ~330 ms reopen -> a dictionary lookup | 1,033.67 ms | 0.03 ms | -100.0% | confirmed |
| Unfiltered lexical query, steady state | 84 ms -> 68 ms at 100k chunks | 23.12 ms | 14.71 ms | -36.4% | confirmed |
| First lexical query after open | pays the corpus parse that open no longer does | 20.00 ms | 233.53 ms | +1067.9% | one-off cost, by design |
| Peak memory holding a lexical project | not claimed; a consequence of storing the corpus once | 7,333 MB | 1,678 MB | -77.1% | confirmed |
| Filtered vector query, steady state | 16-42 ms -> 0.05-0.4 ms of id selection at 100k chunks | 2.36 ms | 0.91 ms | -61.4% | confirmed |
| First filtered query | builds the postings once | 2.58 ms | 46.89 ms | +1718.9% | one-off cost, by design |
| Query cost against a stalled endpoint | ~360 s -> 5 s | 361.4 s | 5.1 s | -98.6% | confirmed |
| Requests for 8 repeats of one query | cached after the first | 8 | 1 | -87.5% | confirmed |
| Requests resent by a single rate limit | the whole segment -> the one chunk | 10 | 0 | -100.0% | confirmed |
| Chunks losing their context when one chunk exhausts its retries | the whole segment -> the one chunk | 0 | 0 | n/a | no change |
| Context generation wall-clock | segments overlap under one ceiling | 3.8 s | 2.8 s | -26.7% | confirmed |
| Index build wall-clock | embedding batches overlap in a window | 3.1 s | 1.1 s | -65.9% | confirmed |
| End-to-end build against the configured endpoint | a build costs its request volume divided by the ceiling | 84.1 s | 66.6 s | -20.9% | confirmed |
| End-to-end build, second endpoint | same change, an endpoint with stronger prefix caching | 28.7 s | 21.8 s | -24.2% | confirmed |
| Chunks that actually received generated context | every chunk | 74 | 74 | +0.0% | no change |

## Do both revisions return the same results?

**Identical.** All 8 queries returned the same chunks in the same order with the same scores on both revisions. The speed numbers above compare the same operation.

## Prefix caching

Every context prompt is `[segment ~8000 tokens][chunk ~800]`, so all chunks of a segment share a long prefix. Whether that is worth anything depends on when the requests arrive, not only on what they contain.

| Endpoint | Sequential, shared | Concurrent, shared | Concurrent, distinct | Sharing helps |
|---|---|---|---|---|
| hosted | 2.04s cold -> 0.93s warm | 1.16s | 2.81s | yes, 2.4x |
| selfhosted_a | 0.75s cold -> 0.10s warm | 0.58s | 2.77s | yes, 4.8x |
| selfhosted_b | 0.79s cold -> 0.19s warm | 0.86s | 2.84s | yes, 3.3x |
| selfhosted_b_repeat | 0.79s cold -> 0.19s warm | 0.79s | 2.85s | yes, 3.6x |

**Sharing a prefix pays off on 4 of 4 endpoints measured, including for requests issued simultaneously.** That is what makes segment scheduling matter: the older revision held one segment's prefix in flight at a time, while the newer one admits up to `MAX_CONCURRENT_CALLS` segments at once (`project.py`, `segment_gate`), so the server sees interleaved prefixes. The end-to-end builds above still came out faster after the change, so on these endpoints overlapping segments outweighs the lost cache locality on wall-clock - but it is bought by recomputing prefixes, which a provider that bills cached input at a discount would charge for.

## Accuracy

Golden set: `deepeval_goldens_korean_no_hypothetical.json` - 70 questions, 70 with ground-truth context.

### Retrieval accuracy

Deterministic: a question counts as found when a retrieved chunk contains a fragment of the context the golden says it should be answered from.

| Metric | Before | After | Change |
|---|---|---|---|
| recall@1 | 0.214 | 0.214 | +0.0% |
| recall@5 | 0.471 | 0.471 | +0.0% |
| recall@10 | 0.543 | 0.543 | +0.0% |
| recall@k | 0.643 | 0.643 | +0.0% |
| MRR | 0.332 | 0.333 | +0.1% |

Rank changed for 1 of 70 questions: 0 found before but not after, 0 found after but not before.

| # | Question | Rank before | Rank after |
|---|---|---|---|
| 5 | 기계식 스토커 내 Archimedean 스크루와 자동 슬라이딩 트레이 연동 메커... | 9 | 8 |

### Answer correctness

Both revisions' answers graded in one pass by the same judge, using the `answer` criteria from `evaluation/prompts/eval_prompts.py`.

| Metric | Before | After | Change |
|---|---|---|---|
| answers graded | 20 | 20 | +0.0% |
| mean score | 0.580 | 0.530 | -8.6% |
| median score | 0.700 | 0.650 | -7.1% |
| pass rate (>=0.7) | 60.0% | 50.0% | -16.7% |
| scored 1.0 | 40.0% | 50.0% | +25.0% |

5 of 20 answers scored differently, and 5 of those 5 were generated from **identical retrieved evidence**. Where the evidence is the same and the answer is not, the difference is the generator sampling, not anything under test.

**Is the gap real?** Pairing each question's score across revisions gives 40 paired observations, of which 11 differ (7 lower after, 4 higher). Mean paired difference -0.0425, bootstrap 95% CI [-0.1050, +0.0150], sign test p = 0.549. The interval includes zero and the sign test is far from significant, so this is sampling in the generator, not a measured change in answer quality.

**3 answers scored materially lower after.**

| Question | Before | After | Judge's reason (after) |
|---|---|---|---|
| 자전거용 직류 발전기 조명기구와 자동차용 전기식 조명기구는 HS 8512... | 0.70 | 0.30 | The actual output incorrectly states that there is no difference in excluded items between |
| 복사지용 종이와 등사원지의 방수 처리 방법, 압력 구멍 형성 방식, 그리... | 0.50 | 0.00 | The actual output incorrectly swaps the definitions of 'copy paper' (복사지) and 'mimeograph  |
| 치과용 플라스터에 첨가된 응결촉진제와 지연제는 각각 어떤 역할을 하며, ... | 0.70 | 0.00 | The actual output fails to answer the question by claiming the information is missing, whe |


## Segments, chunks, retrieval and answers

### Segmentation and chunking

Document: `excerpt-40-chunks.md` (46,191 characters)

| | Before | After |
|---|---|---|
| segments | 4 | 4 |
| chunks | 47 | 47 |
| segment text identical | yes | |
| chunk text identical | yes | |

First chunks, identical on both revisions:

| chunk | tokens | starts |
|---|---|---|
| `excerpt-40-chunks.md_s0_c0` | 122 | # HS해석에 관한 통칙  이 표의 품목분류는 다음 원칙에 따른다.  # 통칙 제1호  # 이 표의 부(部)... |
| `excerpt-40-chunks.md_s0_c1` | 798 | #### [해설]  - (I) 이 표는 국제무역에서 취급되고 있는 상품을 체계적인 형식으로 표시하고 있다. ... |
| `excerpt-40-chunks.md_s0_c2` | 715 | 통칙 제2호가목의 조건이 충족되고 그 호나 주에서 따로 규정한 것이 없는 한, 완전한 물품이나 완성한 물품과... |
| `excerpt-40-chunks.md_s0_c3` | 788 | 완성된 물품의 중요한 모양을 갖추고 있지 않은 반제품(semi-manufacture)(일반적으로 봉ㆍ디스크ㆍ... |
| `excerpt-40-chunks.md_s0_c4` | 545 | 부(部)나 류(類)의 주(註)나 호(號)의 본문에서 규정한 조제 혼합물은 통칙 제1호에 따라 분류하여야 한다... |

### Retrieval, keyword extraction pinned

Extraction replaced by identical local tokenization on both sides, so this compares the ranking code alone.

| Query | Mode | Top-k identical | Overlap |
|---|---|---|---|
| 기계류의 부분품은 어느 호에 분류되나요?... | lexical | yes | 5/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | semantic | yes | 5/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | hybrid | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | lexical | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | semantic | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | hybrid | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | lexical | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | semantic | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | hybrid | yes | 5/5 |

Top-5 hybrid results for `기계류의 부분품은 어느 호에 분류되나요?`:

| # | Before | After |
|---|---|---|
| 1 | `wco_hs_explanatory_notes.md_s180_c6` (1.0) | `wco_hs_explanatory_notes.md_s180_c6` (1.0) |
| 2 | `wco_hs_explanatory_notes.md_s181_c9` (0.987342) | `wco_hs_explanatory_notes.md_s181_c9` (0.987342) |
| 3 | `wco_hs_explanatory_notes.md_s181_c0` (0.954882) | `wco_hs_explanatory_notes.md_s181_c0` (0.954882) |
| 4 | `wco_hs_explanatory_notes.md_s182_c1` (0.52046) | `wco_hs_explanatory_notes.md_s182_c1` (0.52046) |
| 5 | `wco_hs_explanatory_notes.md_s180_c12` (0.512065) | `wco_hs_explanatory_notes.md_s180_c12` (0.512065) |

### Retrieval, each revision on its own defaults

What a caller actually gets. Differences here can come from extraction returning different keywords, not from the ranking code.

| Query | Mode | Top-k identical | Overlap |
|---|---|---|---|
| 기계류의 부분품은 어느 호에 분류되나요?... | lexical | yes | 5/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | semantic | NO | 5/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | hybrid | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | lexical | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | semantic | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | hybrid | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | lexical | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | semantic | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | hybrid | yes | 5/5 |

Top-5 hybrid results for `기계류의 부분품은 어느 호에 분류되나요?`:

| # | Before | After |
|---|---|---|
| 1 | `wco_hs_explanatory_notes.md_s180_c10` (1.0) | `wco_hs_explanatory_notes.md_s180_c10` (1.0) |
| 2 | `wco_hs_explanatory_notes.md_s181_c9` (0.549058) | `wco_hs_explanatory_notes.md_s181_c9` (0.549058) |
| 3 | `wco_hs_explanatory_notes.md_s207_c2` (0.549058) | `wco_hs_explanatory_notes.md_s207_c2` (0.549058) |
| 4 | `wco_hs_explanatory_notes.md_s180_c12` (0.540202) | `wco_hs_explanatory_notes.md_s180_c12` (0.540202) |
| 5 | `wco_hs_explanatory_notes.md_s210_c5` (0.540202) | `wco_hs_explanatory_notes.md_s210_c5` (0.540202) |

**Self-control:** the same revision returned a different `semantic` ordering across its own two runs of the same query (1 case(s), e.g. on the *before* side). Embedding calls are not bit-identical between requests, and adjacent results here are separated by less than that variance. A cross-revision ordering difference of this size is noise, not a behaviour change; only a difference in the *set* of retrieved chunks would be.

### Generated answers

Answer generation is sampled, so identical wording is not expected and not required; what matters is whether both revisions answer the same question from the same evidence.

**기계류의 부분품은 어느 호에 분류되나요?**

| | Before | After |
|---|---|---|
| model | model A | model A |
| chunks used | 5 | 5 |
| same evidence | yes | |

_Before:_

> 기계류 부분품의 분류는 해당 기계와의 전용 관계에 따라 다음과 같이 구분됩니다.
> 
> 1. **특정 기계 전용/주요 사용 부분품**: 원칙적으로 해당 기계와 동일한 호에 분류되나, 다음 호에 독립적으로 분류됩니다: `8409`, `8431`, `8448`, `8466`, `8473`, `8503`, `8522`, `8529`, `8538`.
> 2. **다수 기계 공통 부분품**: 특정 기계에 전용되지 않고 여러 기계에 공통적으로 사용되는 부분품은 `8487`(비전기식) 또는 `8548`(전기식)에 분류됩니다.
> 3. **재료별 분류 예외**: `8484`, `8544`, `8545`, `8546`, `8547`에 해당하는 부분품은 해당 재료가 속하는 류로 분류됩니다.
> 4. **일반 원칙**: 완성가공 여부(완성품 또는 조단조품 등)와 상관없이 이 부에 분류되나, 철강으로 만든 조단조품(rough forging)은 `7207`에 분류됩니다.
> 
> *(참고: 위 호는 제공된 문맥에서 명시된 가장 구체적인 분류 기준입니다.)*

_After:_

> 제공된 문맥에 따르면 기계류 부분품의 분류는 다음과 같습니다:
> 
> * **일반 원칙**: 특정 기계나 동일한 호에 해당하는 기계에 전용되거나 주로 사용하는 부분품은, 해당 기계와 **동일한 호**에 분류합니다.
> * **별도 호에 분류되는 경우 (제16부 주 제2호)**:
>   * 엔진 부분품 → 제8409호
>   * 제8425~8430호 기계류 부분품 → 제8431호
>   * 섬유기계 부분품 → 제8448호
>   * 제8456~8465호 기계 부분품 → 제8466호
>   * 사무용기계 부분품 → 제8473호
>   * 제8501/8502호 기계 부분품 → 제8503호
>   * 제8519/8521호 기기 부분품 → 제8522호
>   * 제8525~8528호 기기 부분품 → 제8529호
>   * 제8535~8537호 기기 부분품 → 제8538호
> * **공통 부분품**: 여러 기계에 공통적으로 사용되는 부분품은 전기식 여부(비전기식: 제8487호, 전기식: 제8548호)에 따라 분류됩니다.
> * **재료별 예외**: 개스킷(제8484호), 절연전선(제8544호), 전기용 탄소제품(제8545호), 애자(제8546호), 전선용 도관(제8547호) 등은 해당 재료가 속하는 류의 호에 분류됩니다.
> 
> (출처: 제16부 총설 및 관련 호 설명 기준)

**소매용으로 포장된 세트의 분류 기준은 무엇인가요?**

| | Before | After |
|---|---|---|
| model | model A | model A |
| chunks used | 5 | 5 |
| same evidence | yes | |

_Before:_

> 제공된 문헌(통칙 제3호나목 해설)에 따르면, '소매용으로 하기 위하여 세트로 된 물품'의 분류 기준은 다음과 같습니다.
> 
> **1. 세트로 인정되는 요건**
> - **가.** 일견 서로 다른 호에 분류될 수 있을 것으로 보이는 최소한 둘 이상의 서로 다른 물품으로 구성되어야 함
> - **나.** 특정 요구를 충족하거나 특정 활동을 수행하기 위해 함께 조합된 제품/물품이어야 함
> - **다.** 재포장 없이 최종 사용자에게 직접 판매하기에 적합한 방법(상자, 케이스, 판 등)으로 조합되어 있어야 함
> 
> **2. 분류 원칙**
> - 세트를 구성하는 요소 중 **본질적인 특성(essential character)을 부여하는 구성요소**에 따라 분류합니다.
> - 예시: 샌드위치(**1602**)와 포테이토 칩(**2004**)을 함께 포장한 세트는 **1602**에 분류됩니다. 스파게티(**1902**), 잘게 갈은 치즈(**0406**), 토마토 소스(**2103**)를 카톤에 넣은 세트는 **1902**에 분류됩니다.
> 
> **3. 적용 제외 사항**
> - 통칙 제3호나목은 함께 조합된 물품 중 특정 품목에는 적용되지 않습니다. (예: 새우통조림(**1605**), 패테(**1602**), 치즈통조림(**0406**), 엷게 썬 베이컨통조림(**1602**), 칵테일소시지(**1601**), 제2208호의 증류주 1병과 제2204호의 포도주 1병 등)

_After:_

> 제공된 문헌(특히 Chunk [3])에 따르면, '소매용으로 하기 위하여 세트로 된 물품'의 분류 기준은 다음과 같습니다.
> 
> **1. 세트 구성 요건**
> - **다수 물품 구성**: 일견 서로 다른 HS 호(號)에 분류될 수 있는 최소 2개 이상의 서로 다른 물품으로 구성되어야 함.
> - **특정 용도 조합**: 특정 요구를 충족하거나 특정 활동을 수행하기 위해 함께 조합되어야 함.
> - **소매용 포장**: 재포장 없이 최종 사용자에게 직접 판매하기에 적합한 방법(예: 상자, 케이스, 판 위 등)으로 조합되어야 함.
> 
> **2. 분류 원칙 (통칙 제3호나목)**
> - 위 요건을 충족하는 세트는 **세트에 본질적인 특성을 부여하는 구성요소**에 따라 분류됩니다.
> - 예시: 샌드위치(제1602호)와 포테이토 칩(제2004호)을 함께 포장한 세트는 본질적 특성을 주는 샌드위치로 분류됨.
> 
> **참고 사항**
> - 단순 다수 개수의 집합(예: 폰듀 포크 6개)이나 재포장·재판매를 목적으로 한 물품은 이 기준이 적용되지 않습니다.
> - 소매용으로 포장되지 않은 세트는 구성요소별로 구분하여 분류해야 합니다. (Chunk [1], [3] 참조)


## Reading these numbers

- **Tokenizer calls barely moved (19,439 to 19,388), and that is expected.** konte's own double-encoding is gone - exactly the 51 calls it accounted for - but it was a rounding error against the text splitter's internal per-piece encoding, which neither revision changes. The chunking win is splitter reuse, not tokenizer calls.
- **The filter index pays for itself after about 31 filtered queries.** Building the postings costs 44 ms once; each filtered query then saves 1.45 ms. A process that issues a handful of filtered queries and exits is slower, not faster.
- **Work moved out of open and into the first query; it did not disappear.** A process that opens a project and issues one lexical query does roughly the same total work on both revisions. The gain is for a process that opens many projects, or answers only vector queries, and never touches the lexical corpus at all.
- **Scales differ from the changelog's.** Several claims were stated at 20k or 100k chunks; the runs here are at the scale each case records above. Directions and orders of magnitude are comparable, absolute milliseconds are not.

## Run detail

| Case | Scale | Status |
|---|---|---|
| accuracy | - | both revisions ok |
| accuracy_control | - | both revisions ok |
| build_concurrency | chunks=21 | both revisions ok |
| build_concurrency_live | chunks=21 | both revisions ok |
| build_concurrency_live_alt | chunks=21 | both revisions ok |
| checkpoint_io | requested_segments=100 | both revisions ok |
| chunking | document=synthetic-500-chunks.md, characters=2083263, chunks_produced=553, segments_produced=50 | both revisions ok |
| import_cost | - | both revisions ok |
| keyword_extraction | - | both revisions ok |
| open_cache | project=all_tariff_documents | both revisions ok |
| output_parity | project=wco_korean_feb2026 | both revisions ok |
| query_bm25 | requested_chunks=20000, queries=200 | both revisions ok |
| query_faiss_filter | project=all_tariff_documents | both revisions ok |
| retrieval_parity | requested_chunks=2000 | both revisions ok |
| storage_size | document=synthetic-2000-chunks.md, requested_chunks=2000, source_text_bytes=8319205 | both revisions ok |
| storage_size_korean | document=excerpt-2000-chunks.md, requested_chunks=2000, source_text_bytes=4982686 | both revisions ok |

## Measurements in full

### build_concurrency_live

| Measurement | Before | After |
|---|---|---|
| `context.build_seconds` | 3.783 | 2.771 |
| `context.chunks` | 132 | 132 |
| `context.effective_parallelism` | 8.724 | 11.91 |
| `context.requests` | 132 | 132 |
| `context.round_trip_seconds` | 0.25 | 0.25 |
| `index.build_seconds` | 3.097 | 1.057 |
| `index.chunks` | 889 | 889 |
| `index.embed_round_trip_seconds` | 0.2 | 0.2 |
| `live.build_seconds` | 84.11 | 66.57 |
| `live.chunks` | 74 | 74 |
| `live.chunks_with_context` | 74 | 74 |
| `live.requested_chunks` | 60 | 60 |
| `retry.chunks` | 21 | 21 |
| `retry.chunks_with_context` | 21 | 21 |
| `retry.distinct_prompts` | 21 | 21 |
| `retry.rate_limit_responses` | 3 | 3 |
| `retry.requests_resent` | 10 | 0 |
| `retry.total_requests` | 34 | 24 |

### build_concurrency_live_alt

| Measurement | Before | After |
|---|---|---|
| `context.build_seconds` | 3.792 | 2.667 |
| `context.chunks` | 132 | 132 |
| `context.effective_parallelism` | 8.702 | 12.37 |
| `context.requests` | 132 | 132 |
| `context.round_trip_seconds` | 0.25 | 0.25 |
| `index.build_seconds` | 3.037 | 1.018 |
| `index.chunks` | 889 | 889 |
| `index.embed_round_trip_seconds` | 0.2 | 0.2 |
| `live.build_seconds` | 28.73 | 21.77 |
| `live.chunks` | 74 | 74 |
| `live.chunks_with_context` | 74 | 74 |
| `live.context_model` | model A | model A |
| `live.endpoint` | endpoint B | endpoint B |
| `live.requested_chunks` | 60 | 60 |
| `retry.chunks` | 21 | 21 |
| `retry.chunks_with_context` | 21 | 21 |
| `retry.distinct_prompts` | 21 | 21 |
| `retry.rate_limit_responses` | 3 | 3 |
| `retry.requests_resent` | 10 | 0 |
| `retry.total_requests` | 34 | 24 |

### checkpoint_io

| Measurement | Before | After |
|---|---|---|
| `build_seconds` | 5.658 | 2.265 |
| `checkpoint_bytes_per_segment` | 5.364e+06 | 4.84e+04 |
| `checkpoint_bytes_written` | 536,446,936 | 4,839,576 |
| `chunks_contextualized` | 1,110 | 1,110 |
| `context_requests` | 1,110 | 1,110 |
| `requested_segments` | 100 | 100 |

### chunking

| Measurement | Before | After |
|---|---|---|
| `corpus.characters` | 2,083,263 | 2,083,263 |
| `corpus.chunks_produced` | 553 | 553 |
| `corpus.document` | synthetic-500-chunks.md | synthetic-500-chunks.md |
| `corpus.segments_produced` | 50 | 50 |
| `first_build.seconds` | 0.4059 | 0.4168 |
| `first_build.splitter_constructions` | 51 | 2 |
| `first_build.tokenizer_encode_calls` | 19,439 | 19,388 |
| `repeat_build_ms.iqr` | - | - |
| `repeat_build_ms.max` | 416.5 | 345.9 |
| `repeat_build_ms.median` | 410 | 326.5 |
| `repeat_build_ms.min` | 401.2 | 318.6 |
| `repeat_build_ms.n` | 3 | 3 |
| `repeat_build_ms.p95` | 416.5 | 345.9 |

### import_cost

| Measurement | Before | After |
|---|---|---|
| `cold_cache_files_written` | 1 | 0 |
| `cold_cache_import_ms.iqr` | - | - |
| `cold_cache_import_ms.max` | 3,987 | 617.7 |
| `cold_cache_import_ms.median` | 2,311 | 594.8 |
| `cold_cache_import_ms.min` | 634 | 571.8 |
| `cold_cache_import_ms.n` | 2 | 2 |
| `cold_cache_import_ms.p95` | 3,987 | 617.7 |
| `encodings_registered_after_import` | ["o200k_base"] | [] |
| `warm_import_ms.iqr` | 6.45 | 19.25 |
| `warm_import_ms.max` | 659 | 567.6 |
| `warm_import_ms.median` | 641.7 | 556.2 |
| `warm_import_ms.min` | 639 | 539.9 |
| `warm_import_ms.n` | 7 | 7 |
| `warm_import_ms.p95` | 659 | 567.6 |

### keyword_extraction

| Measurement | Before | After |
|---|---|---|
| `cache.distinct_queries.calls` | 8 | 8 |
| `cache.distinct_queries.requests_sent` | 8 | 8 |
| `cache.repeated_query.calls` | 8 | 8 |
| `cache.repeated_query.latency_ms.iqr` | 1.148 | 0.001052 |
| `cache.repeated_query.latency_ms.max` | 41.06 | 40.74 |
| `cache.repeated_query.latency_ms.median` | 2.919 | 0.000312 |
| `cache.repeated_query.latency_ms.min` | 2.252 | 0.0001669 |
| `cache.repeated_query.latency_ms.n` | 8 | 8 |
| `cache.repeated_query.latency_ms.p95` | 41.06 | 40.74 |
| `cache.repeated_query.requests_sent` | 8 | 1 |
| `has_extraction_cache` | False | True |
| `stalled_endpoint.fell_back_to_tokenization` | True | True |
| `stalled_endpoint.requests_sent` | 3 | 1 |
| `stalled_endpoint.seconds_until_fallback` | 361.4 | 5.147 |

### open_cache

| Measurement | Before | After |
|---|---|---|
| `corpus_parsed_during_open._chunks` | True | False |
| `corpus_parsed_during_open._contextualized_chunks` | True | False |
| `corpus_parsed_during_open._segments` | True | False |
| `first_lexical_query_ms` | 9.41 | 242.7 |
| `get_project_ms.iqr` | 9.579 | 96.63 |
| `get_project_ms.max` | 1,040 | 523.4 |
| `get_project_ms.median` | 1,034 | 446 |
| `get_project_ms.min` | 1,026 | 408.7 |
| `get_project_ms.n` | 5 | 5 |
| `get_project_ms.p95` | 1,040 | 523.4 |
| `get_shared_project_ms.iqr` | - | 0.001667 |
| `get_shared_project_ms.max` | - | 0.1109 |
| `get_shared_project_ms.median` | - | 0.03425 |
| `get_shared_project_ms.min` | - | 0.03362 |
| `get_shared_project_ms.n` | - | 20 |
| `get_shared_project_ms.p95` | - | 0.03917 |
| `get_shared_project_ms.status` | unsupported | - |
| `open_ms.iqr` | 194.9 | 144.1 |
| `open_ms.max` | 1,124 | 587.6 |
| `open_ms.median` | 1,024 | 516 |
| `open_ms.min` | 851.7 | 410.5 |
| `open_ms.n` | 5 | 5 |
| `open_ms.p95` | 1,124 | 587.6 |
| `project` | all_tariff_documents | all_tariff_documents |
| `project_dir_bytes` | 347,065,058 | 347,065,058 |
| `rss_mb.after_first_query` | 1,892 | 1,585 |
| `rss_mb.after_opens` | 1,892 | 1,585 |
| `rss_mb.start` | 194 | 124.2 |

### output_parity

| Measurement | Before | After |
|---|---|---|
| `project` | wco_korean_feb2026 | wco_korean_feb2026 |

### query_bm25

| Measurement | Before | After |
|---|---|---|
| `corpus.queries` | 200 | 200 |
| `corpus.requested_chunks` | 20,000 | 20,000 |
| `first_query_results` | 20 | 20 |
| `first_query_seconds` | 0.02 | 0.2335 |
| `open_seconds` | 1.184 | 0.1334 |
| `rss_mb.after_first_query` | 7,333 | 1,678 |
| `rss_mb.after_open` | 7,333 | 1,678 |
| `rss_mb.before_open` | 7,333 | 1,678 |
| `warm_query_ms.iqr` | 5.607 | 7.396 |
| `warm_query_ms.max` | 29.57 | 24.78 |
| `warm_query_ms.median` | 23.12 | 14.71 |
| `warm_query_ms.min` | 13.98 | 8.597 |
| `warm_query_ms.n` | 199 | 199 |
| `warm_query_ms.p95` | 28.72 | 23.41 |

### query_faiss_filter

| Measurement | Before | After |
|---|---|---|
| `filter.matching_chunks` | 6,270 | 6,270 |
| `filter.source` | source C | source C |
| `filter.total_chunks` | 10,660 | 10,660 |
| `filtered_query_ms.iqr` | 0.05202 | 0.01731 |
| `filtered_query_ms.max` | 2.421 | 0.9647 |
| `filtered_query_ms.median` | 2.362 | 0.9109 |
| `filtered_query_ms.min` | 2.309 | 0.898 |
| `filtered_query_ms.n` | 40 | 40 |
| `filtered_query_ms.p95` | 2.404 | 0.951 |
| `first_filtered_query_ms` | 2.578 | 46.89 |
| `first_filtered_results` | 20 | 20 |
| `project` | all_tariff_documents | all_tariff_documents |
| `rss_mb.after_first_filter` | 1,129 | 771.3 |
| `rss_mb.before_first_filter` | 1,129 | 770.6 |
| `rss_mb.postings_mb` | 0.04688 | 0.6875 |
| `unfiltered_query_ms.iqr` | 0.01537 | 0.01896 |
| `unfiltered_query_ms.max` | 1.349 | 1.216 |
| `unfiltered_query_ms.median` | 1.171 | 1.141 |
| `unfiltered_query_ms.min` | 1.136 | 1.087 |
| `unfiltered_query_ms.n` | 40 | 40 |
| `unfiltered_query_ms.p95` | 1.204 | 1.176 |

### retrieval_parity

| Measurement | Before | After |
|---|---|---|
| `corpus.requested_chunks` | 2,000 | 2,000 |
| `top_k` | 20 | 20 |

### storage_size

| Measurement | Before | After |
|---|---|---|
| `corpus.document` | synthetic-2000-chunks.md | synthetic-2000-chunks.md |
| `corpus.requested_chunks` | 2,000 | 2,000 |
| `corpus.source_text_bytes` | 8,319,205 | 8,319,205 |
| `files.bm25.pkl` | 14,018,552 | 3,286,436 |
| `files.bm25_chunks.json` | 9,600,286 | - |
| `files.chunks.json` | 9,724,776 | 9,600,286 |
| `files.config.json` | 432 | 432 |
| `files.raw_chunks.json` | 9,609,180 | 9,542,488 |
| `files.segments.json` | 9,196,301 | 9,195,899 |
| `project_dir_bytes` | 52,149,527 | 31,625,541 |
| `project_dir_multiple_of_text` | 6.269 | 3.802 |

### storage_size_korean

| Measurement | Before | After |
|---|---|---|
| `corpus.document` | excerpt-2000-chunks.md | excerpt-2000-chunks.md |
| `corpus.requested_chunks` | 2,000 | 2,000 |
| `corpus.source_text_bytes` | 4,982,686 | 4,982,686 |
| `files.bm25.pkl` | 10,274,769 | 7,477,706 |
| `files.bm25_chunks.json` | 10,132,960 | - |
| `files.chunks.json` | 10,270,834 | 6,124,658 |
| `files.config.json` | 432 | 432 |
| `files.raw_chunks.json` | 10,142,810 | 6,060,646 |
| `files.segments.json` | 9,337,998 | 5,487,104 |
| `project_dir_bytes` | 50,159,803 | 25,150,546 |
| `project_dir_multiple_of_text` | 10.07 | 5.048 |

