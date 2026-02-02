"""Utility functions for extracting raw evidence from retrieval context.

The retrieval_context in test datasets contains pre-generated context that may
make evidence "too easy" for the LLM:

```
맥락 설명: [LLM-generated summary]        ← Pre-digested
문서 위치: [document location]            ← Helpful metadata
관련 HS 코드: 8301.20, 8301.30, ...      ← Answer listed!
[actual document text]                    ← Raw source
```

This module provides functions to strip the generated context and extract
only the raw document text for fair comparison experiments.
"""


def extract_raw_evidence(retrieval_context: str) -> str:
    """Strip generated context metadata, keep raw document text.

    The retrieval_context format (when metadata is present):
    1. 맥락 설명: [context explanation]
    2. 문서 위치: [document location]
    3. 관련 HS 코드: [HS codes]
    4. [raw document text]

    This function:
    - If metadata markers are found at the TOP, strips them and returns the rest
    - If no metadata markers at top, returns the entire content (it's already raw)

    Args:
        retrieval_context: Full retrieval context with or without metadata.

    Returns:
        Raw document text without generated context/metadata.
    """
    if not retrieval_context:
        return retrieval_context

    # Check if contextual metadata exists at the START of the document
    # Metadata markers that indicate LLM-generated context
    metadata_start_markers = ["맥락 설명:", "문서 위치:"]

    has_metadata_at_start = any(
        retrieval_context.strip().startswith(marker) or
        f"\n{marker}" in retrieval_context[:500]  # Check first 500 chars
        for marker in metadata_start_markers
    )

    if not has_metadata_at_start:
        # No contextual metadata - return as-is (already raw document text)
        return retrieval_context

    # Has metadata - try to find the raw text after "관련 HS 코드:" marker
    markers = ["관련 HS 코드:", "관련 HS"]
    for marker in markers:
        if marker in retrieval_context:
            parts = retrieval_context.split(marker, 1)
            if len(parts) > 1:
                # Skip the HS code line, get the rest
                lines = parts[1].split("\n", 2)
                if len(lines) > 2:
                    return lines[2].strip()
                elif len(lines) == 2:
                    return lines[1].strip()

    # Has metadata but no "관련 HS 코드:" - find raw text after last metadata line
    # Look for the pattern: metadata lines are followed by empty line then raw text
    lines = retrieval_context.split("\n")
    raw_start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("맥락 설명:") or stripped.startswith("문서 위치:") or stripped.startswith("관련 HS"):
            raw_start_idx = i + 1
            # Skip the next line if it's empty
            if raw_start_idx < len(lines) and not lines[raw_start_idx].strip():
                raw_start_idx += 1

    if raw_start_idx > 0 and raw_start_idx < len(lines):
        return "\n".join(lines[raw_start_idx:]).strip()

    # If no markers found, return as-is
    return retrieval_context


def has_context_metadata(retrieval_context: str) -> bool:
    """Check if retrieval context contains generated metadata.

    Args:
        retrieval_context: Retrieval context string.

    Returns:
        True if context contains generated metadata markers.
    """
    if not retrieval_context:
        return False

    metadata_markers = ["맥락 설명:", "문서 위치:", "관련 HS 코드:"]
    return any(marker in retrieval_context for marker in metadata_markers)


def get_context_metadata(retrieval_context: str) -> dict:
    """Extract metadata from retrieval context.

    Args:
        retrieval_context: Full retrieval context.

    Returns:
        Dict with extracted metadata (context_explanation, document_location, hs_codes).
    """
    result = {
        "context_explanation": None,
        "document_location": None,
        "hs_codes": None,
        "raw_text": None,
    }

    if not retrieval_context:
        return result

    lines = retrieval_context.split("\n")
    current_key = None
    current_value = []

    for line in lines:
        if line.startswith("맥락 설명:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "context_explanation"
            current_value = [line.replace("맥락 설명:", "").strip()]
        elif line.startswith("문서 위치:"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "document_location"
            current_value = [line.replace("문서 위치:", "").strip()]
        elif line.startswith("관련 HS 코드:") or line.startswith("관련 HS"):
            if current_key and current_value:
                result[current_key] = "\n".join(current_value).strip()
            current_key = "hs_codes"
            # Extract just the codes
            code_part = line.replace("관련 HS 코드:", "").replace("관련 HS", "").strip()
            current_value = [code_part] if code_part else []
        elif current_key:
            current_value.append(line)

    # Save last accumulated value
    if current_key and current_value:
        result[current_key] = "\n".join(current_value).strip()

    # Extract raw text
    result["raw_text"] = extract_raw_evidence(retrieval_context)

    return result
