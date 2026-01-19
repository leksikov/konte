"""Generate synthetic test dataset from wco_hs_explanatory_notes using DeepEval Synthesizer.

Uses Konte's existing chunks as contexts for golden generation.
Chunks are randomly sampled across the entire chunk space to ensure diversity.
"""

import json
import random
import sys
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from tqdm import tqdm

from evaluation.custom_llm import BackendAIModel
from konte import Project


def synthesize_from_konte_chunks(
    project_name: str,
    output_path: Path,
    num_goldens: int = 20,
    chunks_per_context: int = 2,
    seed: int = 42,
) -> None:
    """Generate synthetic test cases from Konte project chunks.

    Randomly samples chunks across the entire chunk space to ensure diversity.
    Each context group contains randomly selected chunks, not sequential ones.

    Args:
        project_name: Name of Konte project to extract chunks from.
        output_path: Path to save generated goldens as JSON.
        num_goldens: Number of golden test cases to generate.
        chunks_per_context: Number of chunks to group per context.
        seed: Random seed for reproducibility.
    """
    random.seed(seed)
    # Load project
    print(f"Loading project: {project_name}")
    project = Project.open(project_name)

    # Load chunks from project
    chunks_path = project.project_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"No chunks.json found in {project.project_dir}")

    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)

    print(f"Loaded {len(chunks_data)} chunks from project")

    # Extract contextualized content (context + original content)
    chunk_texts = []
    for chunk_item in chunks_data:
        # Structure is {"chunk": {...}, "context": "..."}
        chunk = chunk_item.get("chunk", {})
        context = chunk_item.get("context", "")
        content = chunk.get("content", "")

        # Combine context + content
        if context and content:
            text = f"{context}\n\n{content}"
        else:
            text = content

        if text and len(text) > 100:  # Skip very short chunks
            chunk_texts.append(text)

    print(f"Using {len(chunk_texts)} quality chunks for synthesis")

    # Randomly sample chunks across the entire chunk space
    # Each context group contains randomly selected chunks, not sequential ones
    total_chunks_needed = num_goldens * chunks_per_context

    if len(chunk_texts) < total_chunks_needed:
        print(f"Warning: Not enough chunks ({len(chunk_texts)}) for {num_goldens} goldens with {chunks_per_context} chunks each")
        # Sample with replacement if not enough chunks
        sampled_indices = [random.randint(0, len(chunk_texts) - 1) for _ in range(total_chunks_needed)]
    else:
        # Sample without replacement for diversity
        sampled_indices = random.sample(range(len(chunk_texts)), total_chunks_needed)

    # Group sampled chunks into contexts
    contexts = []
    for i in range(0, len(sampled_indices), chunks_per_context):
        group_indices = sampled_indices[i : i + chunks_per_context]
        context_group = [chunk_texts[idx] for idx in group_indices]
        if context_group:
            contexts.append(context_group)

    print(f"Created {len(contexts)} context groups from randomly sampled chunks")
    print(f"Sample indices (first 10): {sampled_indices[:10]}")

    if not contexts:
        print("No contexts to synthesize from!")
        return

    # Initialize custom LLM for BackendAI
    custom_model = BackendAIModel()

    # Initialize synthesizer with custom model
    synthesizer = Synthesizer(
        model=custom_model,
        async_mode=True,
    )

    # Process in batches with progress bar
    # Smaller batch size to avoid token limit errors
    batch_size = 5
    contexts_to_process = contexts[:num_goldens]
    golden_dicts = []

    print(f"Generating {len(contexts_to_process)} goldens in batches of {batch_size}...")
    sys.stdout.flush()

    with tqdm(total=len(contexts_to_process), desc="Generating goldens", unit="golden") as pbar:
        for batch_start in range(0, len(contexts_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(contexts_to_process))
            batch_contexts = contexts_to_process[batch_start:batch_end]

            try:
                goldens = synthesizer.generate_goldens_from_contexts(
                    contexts=batch_contexts,
                    include_expected_output=True,
                    max_goldens_per_context=1,
                )

                for golden in goldens:
                    golden_dicts.append(
                        {
                            "input": golden.input,
                            "expected_output": golden.expected_output,
                            "retrieval_context": golden.context,
                        }
                    )

                pbar.update(len(batch_contexts))
                pbar.set_postfix({"generated": len(golden_dicts)})

            except Exception as e:
                print(f"\nError in batch {batch_start}-{batch_end}: {e}")
                pbar.update(len(batch_contexts))
                continue

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_dicts, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(golden_dicts)} test cases, saved to {output_path}")


def main():
    """Generate test dataset from wco_hs_explanatory_notes project."""
    output_path = Path("evaluation/data/synthetic/synthetic_goldens.json")

    synthesize_from_konte_chunks(
        project_name="wco_hs_explanatory_notes",
        output_path=output_path,
        num_goldens=120,
        chunks_per_context=2,
    )


if __name__ == "__main__":
    main()
