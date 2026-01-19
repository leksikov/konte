"""Generate test cases using DeepEval Synthesizer.

Uses DeepEval's Synthesizer to generate test cases from pre-extracted contexts
(Konte project chunks) with various evolution types for question complexity.
"""

import argparse
import json
import random
from pathlib import Path

from deepeval.synthesizer import Evolution, Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig
from tqdm import tqdm

# Evolution configuration - equal distribution for balanced question types
EVOLUTIONS = {
    Evolution.REASONING: 1 / 7,  # Logical complexity
    Evolution.MULTICONTEXT: 1 / 7,  # Uses multiple contexts (context-faithful)
    Evolution.CONCRETIZING: 1 / 7,  # Adds specific details (context-faithful)
    Evolution.CONSTRAINED: 1 / 7,  # Adds constraints (context-faithful)
    Evolution.COMPARATIVE: 1 / 7,  # Compares concepts (context-faithful)
    Evolution.HYPOTHETICAL: 1 / 7,  # What-if scenarios
    Evolution.IN_BREADTH: 1 / 7,  # Expands scope
}


def load_contexts_from_project(project_name: str) -> list[list[str]]:
    """Load contexts from a Konte project's chunks.

    Args:
        project_name: Name of the Konte project.

    Returns:
        List of context lists (each context is a list of strings for multi-context).
    """
    from konte.config.settings import settings

    project_dir = settings.STORAGE_PATH / project_name
    chunks_path = project_dir / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.json found in {project_dir}")

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)

    print(f"Loaded {len(chunks_data)} chunks from project {project_name}")

    # Extract contexts - combine context + content for each chunk
    contexts = []
    for chunk_item in chunks_data:
        chunk = chunk_item.get("chunk", {})
        context = chunk_item.get("context", "")
        content = chunk.get("content", "")

        # Combine context and content
        full_text = f"{context}\n\n{content}" if context else content

        # Only include chunks with substantial content
        if len(full_text) > 200:
            contexts.append([full_text])  # Each context is a list for multi-context support

    return contexts


def load_contexts_from_document(document_path: str, chunk_size: int = 2000) -> list[list[str]]:
    """Load contexts by chunking a document directly.

    Args:
        document_path: Path to the source document.
        chunk_size: Size of each chunk in characters.

    Returns:
        List of context lists.
    """
    doc_path = Path(document_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    content = doc_path.read_text(encoding="utf-8")
    print(f"Document size: {len(content)} chars")

    # Simple chunking by splitting on section markers
    # For markdown, split on headers
    sections = []
    current_section = []

    for line in content.split("\n"):
        if line.startswith("# ") or line.startswith("## "):
            # New section starts
            if current_section:
                section_text = "\n".join(current_section)
                if len(section_text) > 200:
                    sections.append(section_text)
            current_section = [line]
        else:
            current_section.append(line)

    # Don't forget the last section
    if current_section:
        section_text = "\n".join(current_section)
        if len(section_text) > 200:
            sections.append(section_text)

    print(f"Extracted {len(sections)} sections from document")

    # If sections are too long, split them further
    contexts = []
    for section in sections:
        if len(section) > chunk_size:
            # Split into smaller chunks
            for i in range(0, len(section), chunk_size - 200):  # Overlap of 200
                chunk = section[i : i + chunk_size]
                if len(chunk) > 200:
                    contexts.append([chunk])
        else:
            contexts.append([section])

    return contexts


def generate_goldens(
    project_name: str | None = None,
    document_path: str | None = None,
    output_path: str = "evaluation/data/synthetic/deepeval_goldens_100.json",
    num_goldens: int = 100,
    model: str = "gpt-4.1-mini",
    seed: int = 42,
) -> list:
    """Generate test cases using DeepEval Synthesizer with pre-extracted contexts.

    Args:
        project_name: Konte project name to load chunks from.
        document_path: Path to source document (if not using project).
        output_path: Path to save generated goldens as JSON.
        num_goldens: Target number of test cases to generate.
        model: LLM model to use for synthesis.
        seed: Random seed for reproducibility.

    Returns:
        List of generated Golden objects.
    """
    random.seed(seed)

    # Load contexts
    if project_name:
        print(f"Loading contexts from project: {project_name}")
        contexts = load_contexts_from_project(project_name)
    elif document_path:
        print(f"Loading contexts from document: {document_path}")
        contexts = load_contexts_from_document(document_path)
    else:
        raise ValueError("Either project_name or document_path must be provided")

    print(f"Total contexts available: {len(contexts)}")

    # Shuffle and select contexts for generation
    random.shuffle(contexts)

    # Calculate how many contexts we need
    # With max_goldens_per_context=2, we need num_goldens/2 contexts
    goldens_per_context = 2
    num_contexts_needed = (num_goldens // goldens_per_context) + 10  # Extra buffer

    selected_contexts = contexts[:min(num_contexts_needed, len(contexts))]
    print(f"Using {len(selected_contexts)} contexts for generation")

    # Configure evolutions for diverse question types
    evolution_config = EvolutionConfig(
        num_evolutions=2,  # Apply 2 evolution types per golden
        evolutions=EVOLUTIONS,
    )

    # Configure styling for Korean language (but diverse question types)
    styling_config = StylingConfig(
        scenario="한국어 WCO HS 해설서 기반 RAG 시스템 평가",
        task="문서 내용에 기반한 다양한 유형의 한국어 질문을 생성합니다. "
             "추론, 비교, 가설적 시나리오, 구체화 등 다양한 질문 유형을 포함하세요.",
        input_format="한국어로 작성된 질문",
        expected_output_format="한국어로 작성된 답변",
    )

    print(f"\nInitializing Synthesizer with model: {model}")
    print(f"Evolution types: {list(EVOLUTIONS.keys())}")
    print(f"Language: Korean")

    # Initialize Synthesizer with Korean styling
    synthesizer = Synthesizer(
        model=model,
        evolution_config=evolution_config,
        styling_config=styling_config,
    )

    print(f"\nGenerating goldens from {len(selected_contexts)} contexts...")
    print(f"Target: {num_goldens} test cases")

    # Generate goldens from contexts
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=selected_contexts,
        include_expected_output=True,
        max_goldens_per_context=goldens_per_context,
    )

    print(f"\nGenerated {len(goldens)} goldens")

    # Limit to requested number
    if len(goldens) > num_goldens:
        print(f"Limiting to {num_goldens} goldens")
        goldens = goldens[:num_goldens]

    # Convert to JSON format compatible with existing pipeline
    output_data = []
    for golden in goldens:
        output_data.append({
            "input": golden.input,
            "expected_output": golden.expected_output,
            "retrieval_context": golden.context or [],
            "additional_metadata": golden.additional_metadata,
        })

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(output_data)} goldens to: {output_path}")

    # Print samples
    if output_data:
        print("\n--- Sample Goldens ---")
        for i, sample in enumerate(output_data[:3]):
            print(f"\n[{i+1}] Input: {sample['input'][:80]}...")
            print(f"    Expected: {sample['expected_output'][:80]}...")
            if sample.get("additional_metadata"):
                evolutions = sample["additional_metadata"].get("evolutions", [])
                print(f"    Evolutions: {evolutions}")

    return goldens


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate test cases using DeepEval Synthesizer"
    )
    parser.add_argument(
        "--project",
        default="wco_hs_explanatory_notes_korean",
        help="Konte project name to load chunks from",
    )
    parser.add_argument(
        "--document",
        default=None,
        help="Path to source document (alternative to --project)",
    )
    parser.add_argument(
        "--output",
        default="evaluation/data/synthetic/deepeval_goldens_100.json",
        help="Output path for generated goldens",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of test cases to generate",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="LLM model for synthesis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    generate_goldens(
        project_name=args.project if not args.document else None,
        document_path=args.document,
        output_path=args.output,
        num_goldens=args.num,
        model=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
