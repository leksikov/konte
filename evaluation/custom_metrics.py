"""Custom DeepEval metrics for RAG evaluation.

These metrics focus on factual correctness rather than format/length matching.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from evaluation.custom_llm import BackendAIModel


def create_factual_correctness_metric(
    threshold: float = 0.5,
    model: BackendAIModel | None = None,
) -> GEval:
    """Create a factual correctness metric that ignores format differences.

    This metric focuses on:
    - Are the key facts present?
    - Are HS codes correct?
    - Is the core information accurate?

    It does NOT penalize:
    - Longer/more detailed answers
    - Different formatting
    - Additional context or citations
    - Language mixing (Korean/English)
    """
    criteria = """Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on:
1. Are the main HS codes or classification codes mentioned in expected output ALSO present in actual output?
2. Are the key facts, categories, or classifications from expected output covered in actual output?
3. Is the core answer semantically equivalent, even if worded differently or more detailed?

IGNORE these differences:
- Length differences (actual may be longer with more detail - this is OK)
- Format differences (bullet points vs paragraphs, etc.)
- Additional context, explanations, or citations in actual output
- Language mixing (Korean text alongside English)
- Different ordering of information

Score 1.0 if: Actual contains all key facts from expected (even if it has more)
Score 0.7-0.9 if: Actual contains most key facts with minor omissions
Score 0.4-0.6 if: Actual contains some key facts but misses important ones
Score 0.0-0.3 if: Actual is missing most key facts or contains wrong information"""

    evaluation_steps = [
        "Extract the key HS codes, classification codes, or category names from the expected output",
        "Check if these same codes/categories appear in the actual output",
        "Extract the key factual claims from the expected output",
        "Verify these facts are present in the actual output (can be worded differently)",
        "Ignore length, format, and stylistic differences",
        "Score based on factual coverage, not format matching",
    ]

    return GEval(
        name="FactualCorrectness",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model or BackendAIModel(),
        threshold=threshold,
    )


def create_hs_code_accuracy_metric(
    threshold: float = 0.5,
    model: BackendAIModel | None = None,
) -> GEval:
    """Create a metric specifically for HS code accuracy.

    This metric checks if the correct HS codes are present in the answer.
    """
    criteria = """Evaluate if the actual output contains the correct HS (Harmonized System) codes.

Focus ONLY on HS codes:
1. Extract all HS codes from the expected output (format: XXXX or XXXX.XX)
2. Check if these codes appear in the actual output
3. A code is correct if it matches exactly OR if a more specific subcode is given
   (e.g., expected "2523" matches actual "2523.10" or "2523.21")

Scoring:
- 1.0: All expected HS codes are present in actual
- 0.8: Most HS codes present (1 minor code missing)
- 0.5: About half of HS codes present
- 0.2: Few HS codes present
- 0.0: No matching HS codes or completely wrong codes"""

    evaluation_steps = [
        "Extract all HS codes (XXXX or XXXX.XX format) from expected output",
        "Extract all HS codes from actual output",
        "For each expected code, check if it or a more specific version exists in actual",
        "Calculate the match ratio",
        "Ignore all other content - focus only on HS codes",
    ]

    return GEval(
        name="HSCodeAccuracy",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model or BackendAIModel(),
        threshold=threshold,
    )


def create_information_coverage_metric(
    threshold: float = 0.5,
    model: BackendAIModel | None = None,
) -> GEval:
    """Create a metric for information coverage (recall).

    Checks if actual output covers the key information from expected,
    regardless of additional content.
    """
    criteria = """Evaluate the INFORMATION COVERAGE: Does the actual output cover the key information from expected?

This is a RECALL metric - we check if expected information is CONTAINED in actual.
It's OK if actual has MORE information than expected.

Scoring:
- 1.0: All key information from expected is present in actual
- 0.7-0.9: Most key information present, minor details missing
- 0.4-0.6: Some key information missing
- 0.0-0.3: Most key information missing

DO NOT penalize:
- Additional information in actual (this is fine)
- Different wording or phrasing
- Different structure or format
- More detailed explanations"""

    evaluation_steps = [
        "Identify the KEY information points in the expected output",
        "For each key point, check if it is covered in the actual output",
        "A point is covered if the same fact is stated, even if worded differently",
        "Calculate coverage ratio (covered points / total points)",
        "Do not penalize for additional information in actual",
    ]

    return GEval(
        name="InformationCoverage",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model or BackendAIModel(),
        threshold=threshold,
    )
