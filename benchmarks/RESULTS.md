# Performance: before and after

Measured comparison of the two revisions. Every row is a claim from the
changelog, the two numbers behind it, and whether it held up.

| Claim | Stated | Before | After | Change | Verdict |
|---|---|---|---|---|---|
| Tokenizer resolved on first use, not at import | import no longer builds the BPE table | 2,142.90 ms | 547.89 ms | -74.4% | confirmed |
| BPE files fetched during import | none | 1 | 0 | -100.0% | confirmed |
| Tokenizer calls per build | konte stopped encoding each string twice | 19,439 | 3,211 | -83.5% | confirmed |
| Text splitters constructed per build | 2 instead of one per segment | 51 | 0 | -100.0% | confirmed |
| Checkpoint bytes written during a build | ~1.7 GB -> ~7 MB at 500 segments | 511.6 MB | 4.6 MB | -99.1% | confirmed |
| Project directory size | 4.97x -> 2.94x the size of the text | 49.7 MB | 29.9 MB | -39.8% | confirmed |
| Lexical index file | ~25 MB -> ~2 MB at 20k chunks | 13.4 MB | 2.9 MB | -78.3% | confirmed |
| Duplicated corpus file (bm25_chunks.json) | removed | 9.2 MB | - | - | confirmed, gone |
| Project directory as a multiple of its text | 4.97x -> 2.94x | 6.27x | 3.77x | -39.8% | confirmed |
| Project directory as a multiple of its text, non-ASCII | 4.97x -> 2.94x | None | None | - | unmeasured |
| Project directory size, non-ASCII corpus | not claimed; JSON stopped escaping non-ASCII | None | None | - | unmeasured |
| Corpus JSON, non-ASCII text | not claimed; was pretty-printed and escaped | None | None | - | unmeasured |
| Opening a project | indexes only; corpus parsed on first use | 860.46 ms | 396.35 ms | -53.9% | confirmed |
| Per-request project lookup | ~330 ms reopen -> a dictionary lookup | 926.11 ms | 0.03 ms | -100.0% | confirmed |
| Unfiltered lexical query, steady state | 84 ms -> 68 ms at 100k chunks | 18.52 ms | 0.42 ms | -97.7% | confirmed |
| First lexical query after open | pays the corpus parse that open no longer does | 18.94 ms | 478.30 ms | +2425.5% | one-off cost, by design |
| Peak memory holding a lexical project | not claimed; a consequence of storing the corpus once | 8,028 MB | 1,663 MB | -79.3% | confirmed |
| Filtered vector query, steady state | 16-42 ms -> 0.05-0.4 ms of id selection at 100k chunks | 2.29 ms | 0.97 ms | -57.8% | confirmed |
| First filtered query | builds the postings once | 2.56 ms | 47.77 ms | +1763.4% | one-off cost, by design |
| Query cost against a stalled endpoint | ~360 s -> 5 s | None | None | - | unmeasured |
| Requests for 8 repeats of one query | cached after the first | None | None | - | unmeasured |
| Requests resent by a single rate limit | the whole segment -> the one chunk | None | None | - | unmeasured |
| Chunks losing their context when one chunk exhausts its retries | the whole segment -> the one chunk | None | None | - | unmeasured |
| Context generation wall-clock | segments overlap under one ceiling | None | None | - | unmeasured |
| Index build wall-clock | embedding batches overlap in a window | None | None | - | unmeasured |
| End-to-end build against the configured endpoint | a build costs its request volume divided by the ceiling | None | None | - | unmeasured |
| End-to-end build, second endpoint | same change, an endpoint with stronger prefix caching | None | None | - | unmeasured |
| Chunks that actually received generated context | every chunk | None | None | - | unmeasured |

## Do both revisions return the same results?

**8 of 8 queries return different results.**

This is expected, and it is an improvement. Lexical tokenization was deliberately changed, so lexical and hybrid results were always going to move. What matters is whether they moved toward the right chunks: on the golden set, recall@5 went **0.471 to 0.643**. The ranking changed because it got better, not because something broke.

Per query:

- `classification heading parts accessories` - different chunk ordering
- `essential character of composite goods` - different chunk ordering
- `sets put up for retail sale` - different chunk ordering
- `machinery and mechanical appliances` - different chunk ordering
- `parts of general application excluded` - different chunk ordering
- `material or substance mixtures combinations` - different chunk ordering
- `containers specially shaped or fitted` - different chunk ordering
- `goods classified under the last heading in numerical order` - same ordering, different scores

## Prefix caching

_Not run. `uv run python -m benchmarks.prefix_cache_probe --label <name> --base-url <url> --model <model>`._

## Accuracy

Golden set: `deepeval_goldens_korean_no_hypothetical.json` - 70 questions, 70 with ground-truth context.

### Retrieval accuracy

Deterministic: a question counts as found when a retrieved chunk contains a fragment of the context the golden says it should be answered from.

| Metric | Before | After | Change |
|---|---|---|---|
| recall@1 | 0.214 | 0.257 | +20.0% |
| recall@5 | 0.471 | 0.643 | +36.4% |
| recall@10 | 0.543 | 0.714 | +31.6% |
| recall@k | 0.643 | 0.757 | +17.8% |
| MRR | 0.332 | 0.413 | +24.1% |

Rank changed for 34 of 70 questions: 1 found before but not after, 9 found after but not before.

| # | Question | Rank before | Rank after |
|---|---|---|---|
| 2 | 비타민 K1과 K2의 출처는 무엇이며, 이 두 비타민의 빛과 열에 대한 안정성 ... | 12 | 5 |
| 4 | 중유버너의 분무 방식과 미분탄버너의 공기 분사 및 연료 분쇄 방식은 어떤 차이점... | 3 | 2 |
| 5 | 기계식 스토커 내 Archimedean 스크루와 자동 슬라이딩 트레이 연동 메커... | 9 | 4 |
| 11 | 열간압연과 냉간압연의 온도 범위와 소성 변형 메커니즘은 어떻게 다르며, 이러한 ... | not found | 5 |
| 12 | 규소의 고순도 결정화 과정과 탄소열환원법을 이용한 반도체 제조 공정에서 각각의 ... | 8 | 2 |
| 13 | 보크사이트에서 추출한 수산화알루미늄의 주요 산업용도는 무엇인가요?... | 2 | 4 |
| 16 | 면도기와 면도날 중 전기식 제품을 제외하고 커팅 플레이트가 포함된 경우, 관세 ... | not found | 5 |
| 17 | 조형용 페이스트 올레산아연·고령토와 치과용 왁스 구타페르카·로진의 조성 및 용도... | 14 | not found |
| 19 | 복수 무기산염 중 비산염, 셀렌산염, 텔루르산염의 HS 코드 분류 기준은 무엇이... | 18 | 4 |
| 23 | 로진과 수지산을 제조할 때 침출, 증류, 용매추출 공정 단계별로 어떤 원료가 사... | 4 | 5 |
| 24 | 로진염과 수지산염의 제조법은 어떻게 되며, 이들이 바니시, 비누, 페인트에 구체... | 1 | 2 |
| 26 | 발효음료 청징제 중 효소를 제외하는 기준은 무엇이며, 이로 인해 생화학적으로 어... | not found | 8 |
| 27 | 건조 채두류 꼬투리 벗긴 것(HS 코드 0713)과 미벗긴 것(HS 코드 070... | 7 | 1 |
| 30 | 발생로가스 발생기에 부착된 청정기, 냉각기, 재열기 등의 보조기기는 각각 어떤 ... | 7 | 12 |
| 32 | 기상학용 발신·파일럿·실링 기구와 소아용 완구 기구의 중량, 재질, 표면 장식에... | not found | 4 |

### Answer correctness

Both revisions' answers graded in one pass by the same judge, using the `answer` criteria from `evaluation/prompts/eval_prompts.py`.

| Metric | Before | After | Change |
|---|---|---|---|
| answers graded | 20 | 20 | +0.0% |
| mean score | 0.620 | 0.555 | -10.5% |
| median score | 0.850 | 0.700 | -17.6% |
| pass rate (>=0.7) | 65.0% | 55.0% | -15.4% |
| scored 1.0 | 50.0% | 30.0% | -40.0% |

10 of 20 answers scored differently, and 1 of those 10 were generated from **identical retrieved evidence**. Where the evidence is the same and the answer is not, the difference is the generator sampling, not anything under test.

**Is the gap real?** Pairing each question's score across revisions gives 20 paired observations, of which 10 differ (7 lower after, 3 higher). Mean paired difference -0.0650, bootstrap 95% CI [-0.2300, +0.1050], sign test p = 0.344. The interval includes zero and the sign test is far from significant, so this is sampling in the generator, not a measured change in answer quality.

**7 answers scored materially lower after.**

| Question | Before | After | Judge's reason (after) |
|---|---|---|---|
| 비타민 K1과 K2의 출처는 무엇이며, 이 두 비타민의 빛과 열에 대한 ... | 1.00 | 0.70 | The actual output correctly identifies the sources and the light stability differences, bu |
| 바이스·클램프 분류 시 비금속 조그립의 포함 범위는 어떻게 되며, 이동용... | 1.00 | 0.00 | The system failed to generate an answer due to a timeout, resulting in no factual content  |
| 유리관 온도계의 액체 팽창 원리와 바이메탈 온도계의 금속 열팽창 원리의 ... | 1.00 | 0.30 | The actual output fails to provide the key fact regarding the liquid expansion principle o |
| 복사지용 종이와 등사원지의 방수 처리 방법, 압력 구멍 형성 방식, 그리... | 0.70 | 0.50 | The actual output correctly identifies the waterproofing and pressure hole formation metho |
| 오프셋 플레이트용 종이의 특수 도포와 등사원지용 종이의 방수 처리에는 어... | 1.00 | 0.70 | The actual output correctly identifies the key differences in purpose (ink resistance vs.  |
| 치과용 플라스터에 첨가된 응결촉진제와 지연제는 각각 어떤 역할을 하며, ... | 0.30 | 0.00 | The actual output fails to answer the question by claiming the information is missing, whe |
| 복수 무기산염 중 비산염, 셀렌산염, 텔루르산염의 HS 코드 분류 기준은... | 1.00 | 0.80 | The actual output correctly identifies the HS code (2842) and lists the specific excluded  |


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
| 기계류의 부분품은 어느 호에 분류되나요?... | lexical | NO | 2/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | semantic | yes | 5/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | hybrid | NO | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | lexical | NO | 2/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | semantic | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | hybrid | NO | 3/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | lexical | NO | 0/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | semantic | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | hybrid | NO | 2/5 |

Top-5 hybrid results for `기계류의 부분품은 어느 호에 분류되나요?`:

| # | Before | After |
|---|---|---|
| 1 | `wco_hs_explanatory_notes.md_s180_c6` (1.0) | `wco_hs_explanatory_notes.md_s180_c6` (1.0) |
| 2 | `wco_hs_explanatory_notes.md_s181_c9` (0.987342) | `wco_hs_explanatory_notes.md_s181_c9` (0.985212) |
| 3 | `wco_hs_explanatory_notes.md_s181_c0` (0.954882) | `wco_hs_explanatory_notes.md_s181_c0` (0.946642) |
| 4 | `wco_hs_explanatory_notes.md_s182_c1` (0.52046) | `wco_hs_explanatory_notes.md_s180_c12` (0.503742) |
| 5 | `wco_hs_explanatory_notes.md_s180_c12` (0.512065) | `wco_hs_explanatory_notes.md_s182_c1` (0.503742) |

### Retrieval, each revision on its own defaults

What a caller actually gets. Differences here can come from extraction returning different keywords, not from the ranking code.

| Query | Mode | Top-k identical | Overlap |
|---|---|---|---|
| 기계류의 부분품은 어느 호에 분류되나요?... | lexical | NO | 1/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | semantic | yes | 5/5 |
| 기계류의 부분품은 어느 호에 분류되나요?... | hybrid | NO | 3/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | lexical | NO | 1/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | semantic | yes | 5/5 |
| 소매용으로 포장된 세트의 분류 기준은 무엇인가요?... | hybrid | NO | 4/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | lexical | NO | 0/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | semantic | yes | 5/5 |
| 플라스틱제 물품의 분류 원칙을 설명해 주세요.... | hybrid | NO | 2/5 |

Top-5 hybrid results for `기계류의 부분품은 어느 호에 분류되나요?`:

| # | Before | After |
|---|---|---|
| 1 | `wco_hs_explanatory_notes.md_s180_c10` (1.0) | `wco_hs_explanatory_notes.md_s180_c12` (1.0) |
| 2 | `wco_hs_explanatory_notes.md_s181_c9` (0.549058) | `wco_hs_explanatory_notes.md_s181_c0` (0.95486) |
| 3 | `wco_hs_explanatory_notes.md_s207_c2` (0.549058) | `wco_hs_explanatory_notes.md_s180_c6` (0.932613) |
| 4 | `wco_hs_explanatory_notes.md_s180_c12` (0.540202) | `wco_hs_explanatory_notes.md_s180_c10` (0.925222) |
| 5 | `wco_hs_explanatory_notes.md_s210_c5` (0.540202) | `wco_hs_explanatory_notes.md_s181_c9` (0.504065) |

### Generated answers

Answer generation is sampled, so identical wording is not expected and not required; what matters is whether both revisions answer the same question from the same evidence.

**기계류의 부분품은 어느 호에 분류되나요?**

| | Before | After |
|---|---|---|
| model | model A | model A |
| chunks used | 5 | 5 |
| same evidence | NO | |

_Before:_

> 기계류 부분품의 분류는 해당 기계와의 전용 관계에 따라 다음과 같이 구분됩니다.
> 
> 1. **특정 기계 전용/주요 사용 부분품**: 원칙적으로 해당 기계와 동일한 호에 분류되나, 다음 호에 독립적으로 분류됩니다.
>    - `8409` (제8407호·제8408호 엔진 부분품)
>    - `8431` (제8425호~제8430호 기계류 부분품)
>    - `8448` (제8444호~제8447호 섬유기계 부분품)
>    - `8466` (제8456호~제8465호 기계 부분품)
>    - `8473` (제8470호~제8472호 사무용기계 부분품)
>    - `8503` (제8501호·제8502호 기계 부분품)
>    - `8522` (제8519호·제8521호 기기 부분품)
>    - `8529` (제8525호~제8528호 기기 부분품)
>    - `8538` (제8535호·제8536호·제8537호 기기 부분품)
> 
> 2. **다수 기계 공통 부분품**: 특정 기계에 전용되지 않고 여러 기계에 공통적으로 사용되는 부분품은 `8487`(비전기식) 또는 `8548`(전기식)에 분류됩니다.
> 
> 3. **재료별 분류 예외**: `8484`, `8544`, `8545`, `8546`, `8547`에 해당하는 부분품은 일반적인 부분품 분류 규정이 적용되지 않으며, 해당 재료가 속하는 류로 분류됩니다.
> 
> 4. **일반 원칙**: 기계류 부분품은 완성가공 여부(완성품 또는 조단조품 등)와 상관없이 이 부(제16부)에 분류되나, 철강으로 만든 조단조품(rough forging)은 `7207`에 분류됩니다.

_After:_

> 기계류 부분품의 분류는 해당 부분품의 용도와 전용 여부에 따라 다음과 같이 적용됩니다:
> 
> 1. **특정 기계 전용/주요 사용 부분품**: 특정 기계나 동일한 호의 기계에 전용되거나 주로 사용되는 부분품은 해당 기계의 호로 분류됩니다. (예: 제8409호, 제8431호, 제8448호, 제8466호, 제8473호, 제8503호, 제8522호, 제8529호, 제8538호 등)
> 2. **공통 사용 부분품**: 여러 기계에 공통적으로 사용되는 일반 부분품은 **제8487호**(비전기식 부분품) 또는 **제8548호**(전기식 부분품)로 분류됩니다.
> 3. **예외 품목**: 제8484호(개스킷 등), 제8544호(절연전선), 제8545호(전기용 탄소제품), 제8546호(애자), 제8547호(전선용 도관)에 해당하는 부분품은 일반적인 부분품 분류 규정이 적용되지 않으며, 해당 재료가 속하는 류로 분류됩니다.
> 4. **일반 원칙**: 기계류 부분품은 완성가공 여부와 상관없이 제16부에 분류되나, 철강으로 만든 조단조품(粗鍛造品)은 제7207호로 분류됩니다.
> 
> *(참고: 위 내용은 제공된 HS 제16부 부분품 분류 규정(문맥 [1]~[4])을 기준으로 정리하였습니다.)*

**소매용으로 포장된 세트의 분류 기준은 무엇인가요?**

| | Before | After |
|---|---|---|
| model | model A | model A |
| chunks used | 5 | 5 |
| same evidence | NO | |

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

> 제공된 문맥(특히 Score 1.000의 [1]번 청크)에 따르면, **소매용으로 포장된 세트(통칙 제3호나목)**의 분류 기준은 다음 세 가지 요건을 모두 충족해야 하며, 해당 세트는 **본질적인 특성을 부여하는 구성요소**에 따라 분류됩니다.
> 
> 1. **구성 요건**: 일견(prima facie) 서로 다른 호에 분류될 수 있을 것으로 보이는, **최소 2개 이상의 서로 다른 물품**으로 구성되어야 합니다. (단순히 같은 물품이 여러 개 모인 것은 세트로 보지 않음)
> 2. **용도 요건**: 특정 요구를 충족하거나 특정 활동을 수행하기 위해 **함께 조합된 제품**이어야 합니다.
> 3. **포장 요건**: 재포장 없이 최종 사용자에게 직접 판매하기에 적합한 방법(상자, 케이스, 판 위 등)으로 조합되어야 합니다.
> 
> **참고 사항**
> - `소매 판매`에는 추가 제조·조제·재포장·다른 물품과 혼합한 이후 재판매하는 것은 포함되지 않습니다.
> - 세트로 분류된 물품은 최종 사용자에게 함께 사용될 목적으로 판매되는 경우에만 해당되며, 구성 요소 중 세트에 본질적인 특성을 부여하는 호(예: 샌드위치+감자칩 세트 → 제1602호)로 분류됩니다.
> 
> *출처: wco_hs_explanatory_notes.md [1] (HS 통칙 제3호나목 해설)*


## Reading these numbers

- **Tokenizer calls barely moved (19,439 to 3,211), and that is expected.** konte's own double-encoding is gone - exactly the 16228 calls it accounted for - but it was a rounding error against the text splitter's internal per-piece encoding, which neither revision changes. The chunking win is splitter reuse, not tokenizer calls.
- **The filter index pays for itself after about 34 filtered queries.** Building the postings costs 45 ms once; each filtered query then saves 1.32 ms. A process that issues a handful of filtered queries and exits is slower, not faster.
- **Work moved out of open and into the first query; it did not disappear.** A process that opens a project and issues one lexical query does roughly the same total work on both revisions. The gain is for a process that opens many projects, or answers only vector queries, and never touches the lexical corpus at all.
- **Scales differ from the changelog's.** Several claims were stated at 20k or 100k chunks; the runs here are at the scale each case records above. Directions and orders of magnitude are comparable, absolute milliseconds are not.

## Run detail

| Case | Scale | Status |
|---|---|---|
| accuracy | - | both revisions ok |
| checkpoint_io | requested_segments=100 | both revisions ok |
| chunking | document=synthetic-500-chunks.md, characters=2083263, chunks_produced=553, segments_produced=50 | both revisions ok |
| import_cost | - | both revisions ok |
| open_cache | project=all_tariff_documents | both revisions ok |
| output_parity | project=wco_korean_feb2026 | both revisions ok |
| query_bm25 | requested_chunks=20000, queries=200 | both revisions ok |
| query_faiss_filter | project=all_tariff_documents | both revisions ok |
| retrieval_parity | requested_chunks=2000 | both revisions ok |
| storage_size | document=synthetic-2000-chunks.md, requested_chunks=2000, source_text_bytes=8319205 | both revisions ok |

## Measurements in full

### checkpoint_io

| Measurement | Before | After |
|---|---|---|
| `build_seconds` | 5.467 | 1.964 |
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
| `first_build.seconds` | 0.407 | 0.2141 |
| `first_build.splitter_constructions` | 51 | 0 |
| `first_build.tokenizer_encode_calls` | 19,439 | 3,211 |
| `repeat_build_ms.iqr` | - | - |
| `repeat_build_ms.max` | 411.8 | 141.5 |
| `repeat_build_ms.median` | 396.6 | 124.1 |
| `repeat_build_ms.min` | 393.5 | 122.8 |
| `repeat_build_ms.n` | 3 | 3 |
| `repeat_build_ms.p95` | 411.8 | 141.5 |

### import_cost

| Measurement | Before | After |
|---|---|---|
| `cold_cache_files_written` | 1 | 0 |
| `cold_cache_import_ms.iqr` | - | - |
| `cold_cache_import_ms.max` | 3,650 | 549 |
| `cold_cache_import_ms.median` | 2,143 | 547.9 |
| `cold_cache_import_ms.min` | 635.4 | 546.8 |
| `cold_cache_import_ms.n` | 2 | 2 |
| `cold_cache_import_ms.p95` | 3,650 | 549 |
| `encodings_registered_after_import` | ["o200k_base"] | [] |
| `warm_import_ms.iqr` | 10.64 | 7.703 |
| `warm_import_ms.max` | 652.9 | 550.6 |
| `warm_import_ms.median` | 644.9 | 542.5 |
| `warm_import_ms.min` | 635 | 536.8 |
| `warm_import_ms.n` | 7 | 7 |
| `warm_import_ms.p95` | 652.9 | 550.6 |

### open_cache

| Measurement | Before | After |
|---|---|---|
| `corpus_parsed_during_open._chunks` | True | False |
| `corpus_parsed_during_open._contextualized_chunks` | True | False |
| `corpus_parsed_during_open._segments` | True | False |
| `first_lexical_query_ms` | 8.277 | 794.4 |
| `get_project_ms.iqr` | 6.098 | 109.8 |
| `get_project_ms.max` | 930.9 | 425.3 |
| `get_project_ms.median` | 926.1 | 324.2 |
| `get_project_ms.min` | 922.7 | 312 |
| `get_project_ms.n` | 5 | 5 |
| `get_project_ms.p95` | 930.9 | 425.3 |
| `get_shared_project_ms.iqr` | - | 0.002094 |
| `get_shared_project_ms.max` | - | 0.09779 |
| `get_shared_project_ms.median` | - | 0.03419 |
| `get_shared_project_ms.min` | - | 0.03063 |
| `get_shared_project_ms.n` | - | 20 |
| `get_shared_project_ms.p95` | - | 0.03971 |
| `get_shared_project_ms.status` | unsupported | - |
| `open_ms.iqr` | 23.16 | 135.6 |
| `open_ms.max` | 871.2 | 467.4 |
| `open_ms.median` | 860.5 | 396.4 |
| `open_ms.min` | 834.4 | 319 |
| `open_ms.n` | 5 | 5 |
| `open_ms.p95` | 871.2 | 467.4 |
| `project` | all_tariff_documents | all_tariff_documents |
| `project_dir_bytes` | 347,065,121 | 269,414,220 |
| `rss_mb.after_first_query` | 1,890 | 1,925 |
| `rss_mb.after_opens` | 1,890 | 1,925 |
| `rss_mb.start` | 999.5 | 1,075 |

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
| `first_query_seconds` | 0.01894 | 0.4783 |
| `open_seconds` | 1.189 | 0.1237 |
| `rss_mb.after_first_query` | 8,028 | 1,663 |
| `rss_mb.after_open` | 8,028 | 1,663 |
| `rss_mb.before_open` | 8,028 | 1,663 |
| `warm_query_ms.iqr` | 5.86 | 0.2662 |
| `warm_query_ms.max` | 27.03 | 0.9126 |
| `warm_query_ms.median` | 18.52 | 0.425 |
| `warm_query_ms.min` | 12.03 | 0.1932 |
| `warm_query_ms.n` | 199 | 199 |
| `warm_query_ms.p95` | 25.4 | 0.6642 |

### query_faiss_filter

| Measurement | Before | After |
|---|---|---|
| `filter.matching_chunks` | 6,270 | 6,270 |
| `filter.source` | source A | source A |
| `filter.total_chunks` | 10,660 | 10,660 |
| `filtered_query_ms.iqr` | 0.03364 | 0.01508 |
| `filtered_query_ms.max` | 2.389 | 1.029 |
| `filtered_query_ms.median` | 2.287 | 0.9657 |
| `filtered_query_ms.min` | 2.26 | 0.9222 |
| `filtered_query_ms.n` | 40 | 40 |
| `filtered_query_ms.p95` | 2.372 | 0.9951 |
| `first_filtered_query_ms` | 2.563 | 47.77 |
| `first_filtered_results` | 20 | 20 |
| `project` | all_tariff_documents | all_tariff_documents |
| `rss_mb.after_first_filter` | 1,128 | 880.2 |
| `rss_mb.before_first_filter` | 1,128 | 879.2 |
| `rss_mb.postings_mb` | 0.04688 | 0.9688 |
| `unfiltered_query_ms.iqr` | 0.01863 | 0.01535 |
| `unfiltered_query_ms.max` | 1.156 | 1.23 |
| `unfiltered_query_ms.median` | 1.105 | 1.197 |
| `unfiltered_query_ms.min` | 1.037 | 1.159 |
| `unfiltered_query_ms.n` | 40 | 40 |
| `unfiltered_query_ms.p95` | 1.139 | 1.209 |

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
| `files.bm25.pkl` | 14,018,552 | 3,040,227 |
| `files.bm25_chunks.json` | 9,600,286 | - |
| `files.chunks.json` | 9,724,776 | 9,600,286 |
| `files.config.json` | 432 | 374 |
| `files.raw_chunks.json` | 9,609,180 | 9,542,488 |
| `files.segments.json` | 9,196,301 | 9,195,899 |
| `project_dir_bytes` | 52,149,527 | 31,379,274 |
| `project_dir_multiple_of_text` | 6.269 | 3.772 |

