"""GEval prompt configurations for RAG evaluation.

Two evaluation types:
1. HS Code Classification - For direct HS code lookup questions
2. Answer Correctness - For diverse RAG question types
"""

# HS Code Classification prompt (for direct lookup questions)
# Use with: synthetic_goldens_100.json
HS_CODE_CRITERIA = """Evaluate if the actual output contains the same KEY FACTUAL INFORMATION as the expected output.

Focus on HS code accuracy and semantic equivalence:
- The key information is the HS CODE (e.g., 2523.21, 제8540호, 8540.20)
- Ignore format differences: "제2523.21호" = "2523.21" = "제2523호의 21" (all equivalent)
- Ignore language mixing (Korean/English)
- Ignore length differences or extra explanation

Scoring:
- Score 1.0: EXACT HS code match (regardless of format)
- Score 0.7-0.9: Parent code given when subheading expected (e.g., "7114" when "7114.20" expected, or "8446호" when "8446.10" expected) - this is a PARTIAL MATCH, not a failure
- Score 0.5-0.6: Related but different subheading under same parent (e.g., "7114.11" when "7114.20" expected)
- Score 0.3-0.4: Wrong parent code but same chapter
- Score 0.0-0.2: Completely wrong HS code or contradictory information

IMPORTANT:
- Parent code matching the expected subheading's parent should score 0.7+, NOT zero
- If actual output provides a DIFFERENT but MORE CORRECT HS code based on the question context, score 0.7+ (the expected output may be wrong)"""

HS_CODE_STEPS = [
    "Extract the HS code(s) from both expected and actual outputs",
    "Normalize format differences (제2523호 = 2523 = 제2523.00호)",
    "Compare if they refer to the same classification",
    "Score based on code match, ignoring format/language differences",
]


# RAG Answer Correctness prompt (for diverse question types)
# Use with: deepeval_goldens_korean_100.json
ANSWER_CORRECTNESS_CRITERIA = """Evaluate if the actual output correctly answers the question based on the expected output.

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
- Score 0.0-0.2: Wrong information or contradicts expected output

IMPORTANT: Focus on factual correctness, not style or verbosity. A shorter but accurate answer scores higher than a verbose but inaccurate one."""

ANSWER_CORRECTNESS_STEPS = [
    "Identify the key facts in the expected output",
    "Check if those key facts are present in the actual output",
    "Verify technical accuracy (codes, terms, numbers, etc.)",
    "Score based on factual alignment, ignoring format/style differences",
]
