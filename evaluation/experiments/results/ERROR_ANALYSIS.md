# Error Analysis Report

## Summary

| Dataset | Total Failures | Recoverable | Not Recoverable |
|---------|---------------|-------------|-----------------|
| HS Code | 6 | 4 | 2 |
| Diverse | 6 | 6 | 0 |
| **Total** | **12** | **10** | **2** |

### Failure Type Distribution

**HS Code Dataset:**
- context_quality: 1
- hs_code_confusion: 3
- knowledge_gap: 2

**Diverse Dataset:**
- retrieval_miss: 6

---

## HS Code Failures (Detailed)

### Case 1: 인조섬유로 만든 저지 풀오버 카디건은 어느 HS 코드에 분류되나요?...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | knowledge_gap |
| **Evidence in Chunks** | False |
| **Recoverable** | False |

**Expected:** HS 코드 6110.30에 분류됩니다. 이 코드는 메리야스 편물이나 뜨개질 편물로 만든 상반신용 의류 중 인조섬유로 제작된 저지, 풀오버, 카디건, 웨이스트코트 및 이와 유사한 의류에 적용됩니다. 남녀 구분 없이 분류되며, 팔꿈치 보호용 패드가 부착된 스포츠용 저지도 포함됩니다....

**Actual:** 인조섬유로 만든 저지 풀오버 카디건은 메리야스 편물이나 뜨개질 편물로 만든 의류에 해당하며, 제61류(의류와 그 부속품, 메리야스 편물이나 뜨개질 편물로 한정)에서 분류됩니다.

제61류는 메리야스 편물이나 뜨개질 편물로 만든 의류를 다루고 있으며, 저지 풀오버 카디건은 이 범주에 포함됩니다. 인조섬유로 만든 경우에도 이 분류가 적용됩니다.

따라서, 인조...

**Evaluation Reason:** The actual output identifies HS codes 6101 and 6102, focusing on male and female knitwear categories, while the expected output specifies HS code 6110.30 for similar garments without gender distinctio...

**Recommendation:** Information may not exist in corpus. Verify if the expected HS code is covered in source documents.

---

### Case 2: 비스코스 레이온으로 만든 꼬임이 없거나 미터당 120회 이하인 재생 필라멘트사는 어느 HS...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | hs_code_confusion |
| **Evidence in Chunks** | True |
| **Recoverable** | True |

**Expected:** HS 코드 5403.10에 분류됩니다. 이 코드는 비스코스 레이온(viscose rayon)으로 만든 강력사에 한정하며, 꼬임이 없거나 꼬임이 미터당 120회 이하인 재생·반합성 필라멘트사를 포함합니다....

**Actual:** 비스코스 레이온으로 만든 꼬임이 없거나 미터당 120회 이하인 재생 필라멘트사는 HS 코드 5403.31에 분류됩니다.  
(출처: [1] "5403.31 -- 비스코스레이온(viscose rayon)의 것(꼬임이 없거나, 꼬임이 미터당 120회 이하인 것으로 한정한다)")...

**Evaluation Reason:** The actual output references HS code 5403.31, while the expected output references 5403.10; these are distinct classifications despite both relating to viscose rayon filaments. The evaluation steps re...

**Recommendation:** LLM confused similar codes: expected 5403.10, got 5403.31. Improve context prompt to emphasize subheading distinctions or add few-shot examples.

---

### Case 3: 시아노겐 클로라이드(클로르시안)는 어느 HS 코드에 분류되나요?...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | knowledge_gap |
| **Evidence in Chunks** | False |
| **Recoverable** | False |

**Expected:** HS 코드 2853.10에 분류됩니다. 시아노겐 클로라이드(클로르시안)는 제28류 무기화합물 중 인화물 및 그 밖의 무기화합물에 포함되며, 특히 2853.10호에 명확히 규정되어 있습니다....

**Actual:** 시아노겐 클로라이드(클로르시안)의 HS 코드 분류에 관한 직접적인 정보는 제공된 문서 내에 명확히 나타나 있지 않습니다. 

다만, 관련 문서 [5]에서는 비금속 할로겐화합물과 산화할로겐화물의 분류 기준을 다루고 있으며, 이 범주에 시아노겐 클로라이드가 포함될 가능성이 있습니다. 비금속 할로겐화합물은 제28류(무기화학제품) 내 2812호(비금속 할로겐화합물...

**Evaluation Reason:** The actual output identifies a related HS code (2812) within the same chapter (28) but does not match the expected specific code 2853.10. It acknowledges the lack of explicit HS code information, show...

**Recommendation:** Information may not exist in corpus. Verify if the expected HS code is covered in source documents.

---

### Case 4: 합성스테이플섬유의 함유량이 전 중량의 85% 이상인 소매용 인조스테이플섬유사는 어느 HS ...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | hs_code_confusion |
| **Evidence in Chunks** | True |
| **Recoverable** | True |

**Expected:** HS 코드 5511.20에 분류됩니다. 이 코드는 합성스테이플섬유의 함유량이 전 중량의 100분의 85 이상인 소매용 인조스테이플섬유사에 적용되며, 재봉사는 제외됩니다....

**Actual:** 합성스테이플섬유의 함유량이 전 중량의 85% 이상인 소매용 인조스테이플섬유사는 HS 코드 5511.10에 분류됩니다.

근거: 문서 [1], [3]에 따르면, "인조스테이플섬유사(재봉사는 제외하며, 소매용으로 한정한다)" 중 합성스테이플섬유 함유량이 85% 이상인 것은 5511.10호에 해당합니다....

**Evaluation Reason:** The actual output cites HS code 5511.10 while the expected output specifies 5511.20; despite both referring to synthetic staple fibers with at least 85% content, the codes differ and normalization doe...

**Recommendation:** LLM confused similar codes: expected 5511.20, got 5511.10. Improve context prompt to emphasize subheading distinctions or add few-shot examples.

---

### Case 5: 제5602호 또는 제5603호의 직물로 만든 여성용 실내복(슬립, 페티코트, 나이트드레스 ...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | context_quality |
| **Evidence in Chunks** | True |
| **Recoverable** | True |

**Expected:** HS 코드 6210.10에 분류됩니다. 이 코드는 제5602호나 제5603호의 직물로 만든 의류에 적용되며, 여성용 실내복과 같은 의류가 포함됩니다. 다만, 메리야스 편물이나 뜨개질 편물로 만든 경우에는 제6108호나 제6109호로 분류해야 하며, 브래지어, 거들, 코르셋 등은 제6212호로 분류됩니다....

**Actual:** 제5602호 또는 제5603호의 직물로 만든 여성용 실내복(슬립, 페티코트, 나이트드레스 등)은 제6208호에 분류됩니다. 

근거:  
- 문서 [1], [2], [3], [5]에 따르면 여성용 슬립, 페티코트, 나이트드레스, 파자마, 네그리제, 목욕용 가운 등은 메리야스 편물이나 뜨개질 편물이 아닌 방직용 섬유로 만든 경우 제6208호에 분류됩니다.  ...

**Evaluation Reason:** The actual output classifies the women's indoor clothing made from fabrics of HS codes 5602 or 5603 under HS code 6208, whereas the expected output classifies them under HS code 6210.10. Despite both ...

**Recommendation:** Evidence present but context quality insufficient. Review context generation prompt.

---

### Case 6: 롱파일(looped pile) 편물 중 면으로 만든 메리야스 편물은 어느 HS 코드에 분류...

| Field | Value |
|-------|-------|
| **Score** | 0.0 |
| **Failure Type** | hs_code_confusion |
| **Evidence in Chunks** | True |
| **Recoverable** | True |

**Expected:** HS 코드 6001.10에 분류됩니다. 이 코드는 롱파일 편물, 특히 메리야스 편물이나 뜨개질 편물 중 루프파일 형태로 면으로 만든 제품에 적용됩니다. 금속드리사를 사용한 편물이라도 의류나 실내용품 등 특정 용도에 해당하면 이 류에 포함되며, 원단 상태나 단순 재단된 형태의 편물도 포함됩니다....

**Actual:** 롱파일(long pile) 편물 중 면으로 만든 메리야스 편물은 HS 코드 6001.21에 분류됩니다.

(출처: [1], "6001.10 - 롱파일(long pile) 편물 - 루프파일(looped pile) 편물 - 6001.21 -- 면으로 만든 것")...

**Evaluation Reason:** The actual output references HS code 6001.21, while the expected output references HS code 6001.10. Despite format normalization, these are distinct codes and do not match, indicating no alignment in ...

**Recommendation:** LLM confused similar codes: expected 6001.10, got 6001.21. Improve context prompt to emphasize subheading distinctions or add few-shot examples.

---

## Diverse Failures (Detailed)

### Case 1: 인조 커런덤 용융법과 무수 알루미나 소성법이 결합될 경우, 미세구조와 내화성은 어떻게 변화...

| Field | Value |
|-------|-------|
| **Score** | 0.3 |
| **Failure Type** | retrieval_miss |
| **Evidence in Chunks** | False |
| **Recoverable** | True |

**Expected:** 인조 커런덤 용융법은 산화알루미늄을 전기로에서 용융하여 단단하고 내산성·내열성이 뛰어난 미세구조를 형성하며, 무수 알루미나 소성법은 수산화알루미늄이나 암모늄 명반을 소성하여 가벼운 백색 분말 형태의 미세구조를 만든다. 이 두 방법이 결합되면, 용융법의 고밀도·고경도 특성과 소성법의 미세한 분말 특성이 조합되어 내화성이 향상되고 미세구조가 더욱 균일하며 치밀...

**Actual:** 제공된 맥락 내에는 인조 커런덤(산화알루미늄을 전기로에서 용융하여 제조)과 무수 알루미나(수산화알루미늄을 소성하여 얻는 백색 분말)의 제조법이 각각 설명되어 있으나, 이 두 방법이 결합될 경우 미세구조와 내화성의 변화에 관한 구체적인 내용은 포함되어 있지 않습니다.

따라서, 인조 커런덤 용융법과 무수 알루미나 소성법이 결합될 경우 미세구조와 내화성이 어떻...

**Evaluation Reason:** The actual output correctly identifies the manufacturing methods of 인조 커런덤 and 무수 알루미나 and their individual properties, aligning with key facts from the expected output. However, it explicitly states ...

**Recommendation:** Relevant information not retrieved. Adjust retrieval parameters or query expansion.

---

### Case 2: 복사지용 종이와 등사원지의 방수 처리 방법, 압력 구멍 형성 방식, 그리고 복사기 고정 방...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | retrieval_miss |
| **Evidence in Chunks** | False |
| **Recoverable** | True |

**Expected:** 복사지용 종이는 파라핀, 왁스, 콜로디온 등으로 방수 처리되며, 타자기나 철필 등으로 압력을 가해 도포된 표면에 구멍을 뚫어 문자를 형성합니다. 반면 등사원지는 두꺼운 이장지를 상단에 부착하고 복사기에 고정되도록 특별히 구멍이 뚫려 있으며, 때로 카본지가 삽입되어 복사기에 고정됩니다....

**Actual:** 복사지용 종이와 등사원지의 차이는 다음과 같습니다.

1. **방수 처리 방법**  
   - **등사원지용 종이**는 파라핀(paraffin)이나 왁스(wax), 콜로디온(collodion) 등으로 도포·침투시켜 방수 처리되어 있다. (문서 [2])  
   - **복사지용 종이**에 대한 방수 처리에 관한 구체적 언급은 없으나, 복사지용 종이는 도포하거...

**Evaluation Reason:** The actual output correctly distinguishes between 복사지용 종이 and 등사원지 in terms of waterproofing, pressure application, and copier fixation, but it reverses key facts: it states that 등사원지 is waterproofed ...

**Recommendation:** Relevant information not retrieved. Adjust retrieval parameters or query expansion.

---

### Case 3: 가동코일형 대신 가동철편형을 사용할 경우 전류 측정의 정확도와 응답 속도에는 어떤 차이가 ...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | retrieval_miss |
| **Evidence in Chunks** | False |
| **Recoverable** | True |

**Expected:** 가동철편형은 가동코일형에 비해 구조가 간단하고 내구성이 높지만, 전류 측정의 정확도와 응답 속도는 상대적으로 낮다. 가동코일형은 코일이 영구자석 자계 내에서 자유롭게 움직여 높은 정확도와 빠른 응답 속도를 제공하는 반면, 가동철편형은 솔레노이드가 철편을 편위시키는 방식으로 정확도와 응답 속도가 다소 떨어진다....

**Actual:** 제공된 문헌에서는 가동코일형(moving-coil instrument)과 가동철편형(moving-iron instrument) 기기의 작동 원리와 구분에 대해 설명하고 있으나, 두 기기 간 전류 측정의 정확도나 응답 속도의 차이에 대한 구체적인 비교나 언급은 포함되어 있지 않습니다.

따라서, 가동코일형 대신 가동철편형을 사용할 경우 전류 측정의 정확도와 ...

**Evaluation Reason:** The actual output correctly identifies the types of instruments (moving-coil and moving-iron) and notes the absence of specific information on accuracy and response speed differences, aligning with th...

**Recommendation:** Relevant information not retrieved. Adjust retrieval parameters or query expansion.

---

### Case 4: 케이폭과 코이어 섬유의 가구용 충전재 분류 기준과 HS 코드의 차이점은 무엇인가요?...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | retrieval_miss |
| **Evidence in Chunks** | False |
| **Recoverable** | True |

**Expected:** 케이폭은 주로 가구용 충전재로 제1404호에 분류되며, 신축성이 있고 가벼우나 부서지기 쉬운 식물성 솜털입니다. 반면 코이어 섬유(코코넛 섬유)는 충전용이더라도 제5305호에 분류되어 케이폭과는 다른 HS 코드를 가집니다....

**Actual:** 제공된 문서 내에는 케이폭(coir)과 코이어 섬유의 가구용 충전재 분류 기준 및 HS 코드의 차이에 관한 구체적인 설명이 포함되어 있지 않습니다.

다만, 문서 [4]에서는 코이어(coir)의 실(yarn)에 대해 다음과 같이 분류하고 있습니다.

- 코이어의 실은 한 가닥이나 두 가닥의 것, 또는 세 가닥 이상의 것으로 구분되며, 관련 HS 코드는 제5...

**Evaluation Reason:** The actual output correctly states that the provided documents do not contain specific information about the classification criteria and HS code differences between 케이폭 and 코이어 섬유 for furniture stuffi...

**Recommendation:** Relevant information not retrieved. Adjust retrieval parameters or query expansion.

---

### Case 5: PHEV가 충전이 불가능한 장거리 주행 시 HEV와 추진용 전동기 및 총중량별로 어떤 성능...

| Field | Value |
|-------|-------|
| **Score** | 0.2 |
| **Failure Type** | retrieval_miss |
| **Evidence in Chunks** | False |
| **Recoverable** | True |

**Expected:** PHEV는 충전이 불가능한 장거리 주행 시 내연기관과 전동기를 함께 사용하여 HEV와 유사하게 작동하지만, HEV보다 배터리 용량이 크고 전동기 성능이 더 우수할 수 있습니다. 총중량에 따라 5톤 이하, 5톤 초과 20톤 이하, 20톤 초과로 분류되며, 중량이 클수록 내연기관과 전동기 출력 조합이 달라져 주행 성능에 차이가 발생합니다....

**Actual:** 제공된 문헌에서는 PHEV(플러그-인 하이브리드 전기 자동차)가 충전이 불가능한 장거리 주행 시 HEV(하이브리드 전기 자동차)와 추진용 전동기 및 총중량별 성능 차이에 대한 구체적인 기술이나 비교 내용은 포함되어 있지 않습니다.

다만, 문헌 [1]에 따르면 PHEV는 외부 전원(전력 그리드 아웃렛이나 충전소)에서 축전지를 충전할 수 있는 자동차이며, H...

**Evaluation Reason:** The actual output correctly defines PHEV and HEV and notes the lack of specific information on performance differences during long-distance driving without charging. However, it fails to mention key e...

**Recommendation:** Relevant information not retrieved. Adjust retrieval parameters or query expansion.

---

### Case 6: 만약 오배자 탄닌과 제외된 유도체가 혼합된다면, 이 혼합물의 분류 기준은 어떻게 달라질까요...

| Field | Value |
|-------|-------|
| **Score** | 0.3 |
| **Failure Type** | retrieval_miss |
| **Evidence in Chunks** | False |
| **Recoverable** | True |

**Expected:** 오배자 탄닌과 제외된 유도체가 혼합된 경우, 해당 혼합물은 제외된 유도체의 분류 기준에 따라 제2843호~제2852호, 제2918호, 제2936호~제2941호, 또는 제3501호~제3504호 등 관련 HS 코드에 따라 분류되며, 제3202호(합성 유연제 등)에는 포함되지 않습니다....

**Actual:** 오배자 탄닌(갈로탄닌산)과 제외된 유도체가 혼합된 경우, 이 혼합물은 합성유연제(제3202호) 또는 제외된 유도체가 속하는 해당 호로 분류됩니다.

근거:
- 오배자 탄닌과 그 유도체는 제3201호(식물성 유연용 추출물, 탄닌과 그 염ㆍ에테르ㆍ에스테르ㆍ그 밖의 유도체)에 분류됩니다(문서 [1], [2], [5]).
- 그러나 제외되는 유도체(예: 탄닌산 귀...

**Evaluation Reason:** The actual output correctly identifies that mixtures of 오배자 탄닌 and excluded derivatives are classified according to the HS codes of the excluded derivatives and specifically mentions classification un...

**Recommendation:** Relevant information not retrieved. Adjust retrieval parameters or query expansion.

---

## Recommendations to Reach 96% Accuracy

Based on the analysis, here are the prioritized recommendations:

### High Priority (Quick Wins)

1. LLM confused similar codes: expected 6001.10, got 6001.21. Improve context prompt to emphasize subheading distinctions or add few-shot examples.

2. LLM confused similar codes: expected 5403.10, got 5403.31. Improve context prompt to emphasize subheading distinctions or add few-shot examples.

3. Evidence present but context quality insufficient. Review context generation prompt.

4. Relevant information not retrieved. Adjust retrieval parameters or query expansion.

5. LLM confused similar codes: expected 5511.20, got 5511.10. Improve context prompt to emphasize subheading distinctions or add few-shot examples.


### Expected Impact

- **HS Code Dataset:** 4/6 failures are recoverable
  - Current: 94% → Potential: 98% (if all recoverable fixed)

- **Diverse Dataset:** 6/6 failures are recoverable
  - Current: 94% → Potential: 100% (if all recoverable fixed)

### Key Patterns Identified

1. **HS Code Confusion**: Similar subheadings within same heading cause confusion
2. **Evidence Present but Misinterpreted**: LLM sometimes reverses or misunderstands facts
3. **Knowledge Gap**: Some questions require information not in the corpus

### Next Steps

1. Fix top 2 recoverable issues to reach 96% target
2. Consider adding few-shot examples for HS code disambiguation
3. Evaluate chain-of-thought prompting for complex questions
