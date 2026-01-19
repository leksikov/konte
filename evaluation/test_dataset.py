"""Test dataset for RAG evaluation on wco_hs_explanatory_notes knowledge base.

This dataset contains question-answer pairs based on WCO HS Explanatory Notes
covering General Rules for Interpretation (GRI) and HS classification concepts.

NOTE: Knowledge base is in Korean, so test cases must be in Korean.
"""

from pydantic import BaseModel


class EvalTestCase(BaseModel):
    """A single evaluation test case."""

    input: str  # User question
    expected_output: str  # Expected/ideal answer


# Test cases for WCO HS Explanatory Notes evaluation (Korean)
TEST_CASES: list[EvalTestCase] = [
    # 통칙 제1호 - 호의 용어와 부/류 주에 따른 분류
    EvalTestCase(
        input="통칙 제1호란 무엇이며 어떻게 적용해야 합니까?",
        expected_output="통칙 제1호는 품목분류가 호의 용어와 관련 부 또는 류의 주에 따라 결정되어야 한다고 규정합니다. 부, 류, 소류의 표제는 참조의 편의를 위한 것일 뿐 법적 효력이 없습니다. 분류는 호의 용어와 주에 따라 법적으로 결정되어야 합니다.",
    ),
    # 통칙 제2호가목 - 불완전한 물품이나 미완성 물품
    EvalTestCase(
        input="통칙 제2호가목에 따라 불완전한 물품이나 미완성 물품은 어떻게 분류합니까?",
        expected_output="통칙 제2호가목에 따르면, 불완전한 물품이나 미완성된 물품이 완전한 물품이나 완성된 물품의 본질적인 특성을 지니고 있으면 완전한 물품이나 완성된 물품과 같이 분류합니다. 조립되지 않거나 분해된 상태로 제시된 물품도 마찬가지입니다. 예를 들어, 안장과 페달이 없는 자전거도 본질적인 특성을 갖추고 있으면 자전거로 분류할 수 있습니다.",
    ),
    # 통칙 제2호나목 - 재료나 물질의 혼합물과 복합물
    EvalTestCase(
        input="통칙 제2호나목은 재료나 물질의 혼합물과 복합물을 어떻게 취급합니까?",
        expected_output="통칙 제2호나목은 특정 재료나 물질을 규정하는 호의 범위를 해당 재료나 물질과 다른 재료나 물질의 혼합물 또는 복합물까지 확대합니다. 마찬가지로 특정 재료나 물질로 구성된 물품에는 전부 또는 일부가 해당 재료나 물질로 구성된 물품이 포함됩니다. 두 가지 이상의 재료나 물질로 구성된 물품의 분류는 통칙 제3호에 따라 결정합니다.",
    ),
    # 통칙 제3호 - 둘 이상의 호에 분류 가능한 물품
    EvalTestCase(
        input="물품이 둘 이상의 호에 분류될 수 있는 경우 어떤 통칙이 어떤 순서로 적용됩니까?",
        expected_output="통칙 제3호는 일견 둘 이상의 호에 해당되는 것으로 보이는 물품에 적용됩니다. 규칙은 순서대로 적용됩니다: (가) 가장 구체적으로 표현된 호가 일반적으로 표현된 호에 우선합니다; (나) 제3호가목으로 분류할 수 없는 경우 혼합물, 복합물, 세트는 본질적인 특성을 부여하는 재료나 구성요소에 따라 분류합니다; (다) 가목과 나목으로 분류할 수 없는 경우 순서상 가장 마지막 호로 분류합니다.",
    ),
    # 통칙 제4호 - 어떤 호에도 해당하지 않는 물품
    EvalTestCase(
        input="어떤 호에도 해당하지 않는 물품은 어떻게 분류해야 합니까?",
        expected_output="통칙 제4호는 통칙 제1호부터 제3호까지의 규정에 따라 분류할 수 없는 물품은 가장 유사한 물품이 해당하는 호에 분류한다고 규정합니다. 이는 유사한 물품과 비교하여 기능, 구성, 거래 관행을 고려하여 결정합니다.",
    ),
    # 통칙 제5호 - 용기와 포장재료
    EvalTestCase(
        input="통칙 제5호에 따른 용기와 포장재료의 분류 규칙은 무엇입니까?",
        expected_output="통칙 제5호가목은 특정 물품을 넣기 위해 특별히 만들어지거나 갖추어진 용기로서 장기간 사용하기에 적합하고 해당 물품과 함께 제시되는 경우 해당 물품과 같이 분류한다고 규정합니다. 통칙 제5호나목은 포장재료와 포장용기가 해당 물품의 포장에 통상적으로 사용되는 것이면 해당 물품과 같이 분류한다고 규정합니다. 다만, 반복 사용에 적합한 것이 명백한 경우에는 적용하지 않습니다.",
    ),
    # 통칙 제6호 - 소호 분류
    EvalTestCase(
        input="통칙 제6호는 소호 수준의 분류에 대해 무엇을 규정합니까?",
        expected_output="통칙 제6호는 법적 목적상 소호 수준의 분류는 해당 소호의 용어와 관련 소호 주에 따라, 그리고 준용하여 통칙 제1호부터 제5호까지의 규정에 따라 결정한다고 규정합니다. 같은 수준의 소호만 비교할 수 있습니다. 이 통칙에서 문맥상 달리 해석되지 않는 한 관련 부 및 류의 주도 적용됩니다.",
    ),
    # 본질적인 특성 개념
    EvalTestCase(
        input="복합물이나 세트의 본질적인 특성은 어떻게 결정합니까?",
        expected_output="복합물의 본질적인 특성은 재료의 성질, 용적, 수량, 중량, 가액, 물품의 사용에 있어서 구성요소의 역할 등의 요소를 고려하여 결정합니다. 세트의 경우 본질적인 특성은 세트의 주된 기능이나 목적을 부여하는 구성요소에 의해 결정됩니다. 모든 경우에 일률적으로 적용되는 단일 요소는 없습니다.",
    ),
    # 부분품과 부속품 분류
    EvalTestCase(
        input="부분품과 부속품의 일반적인 분류 원칙은 무엇입니까?",
        expected_output="부분품은 호의 용어나 주에서 부분품을 명시적으로 규정하는 경우 해당 기계나 물품과 같이 분류합니다. 범용성 부분품(나사, 볼트 등)은 해당하는 호에 분류합니다. 부속품은 주된 물품의 사용을 향상시키는 물품입니다. 특정 기계에 전용되거나 주로 사용되는 부분품은 주에서 특별히 제외하지 않는 한 해당 기계와 같은 호에 분류합니다.",
    ),
    # 반가공품(blank) 분류
    EvalTestCase(
        input="반가공품(blank)이란 무엇이며 어떻게 분류합니까?",
        expected_output="반가공품(blank)이란 직접 사용할 수 있는 물품이 아니라 완성한 물품이나 부분품의 대체적인 모양이나 윤곽을 갖추고 있는 물품으로서 예외적인 경우를 제외하고는 오직 완성한 물품이나 부분품으로 완성하기 위하여만 사용될 수 있는 물품을 말합니다. 통칙 제2호가목에 따라 완성품과 같이 분류합니다.",
    ),
]


def get_test_cases() -> list[EvalTestCase]:
    """Return all test cases for evaluation."""
    return TEST_CASES
