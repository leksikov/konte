"""Build experimental project variants for context embedding comparison.

Experiments:
1) exp_8000_800_context_meta - 8000/800 with context in metadata (not embedded)
2) exp_4000_400_context_meta - 4000/400 with context in metadata (not embedded)

Both variants store context in FAISS metadata but only embed original chunk content.
"""

import asyncio
import json
from pathlib import Path

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from konte.chunker import create_chunks
from konte.config import settings
from konte.context import generate_contexts_batch
from konte.loader import load_document
from konte.models import Chunk, ContextualizedChunk, ProjectConfig
from konte.stores import BM25Store

logger = structlog.get_logger()

# Source document
SOURCE_FILE = Path("/Users/sergeyleksikov/Documents/GitHub/contextual_rag/example_knowledge_base/wco_hs_explanatory_notes.md")


async def build_experimental_project(
    name: str,
    segment_size: int,
    segment_overlap: int,
    chunk_size: int,
    chunk_overlap: int,
    embed_context: bool = False,  # If False, context only in metadata
) -> int:
    """Build an experimental project variant.

    Args:
        name: Project name
        segment_size: Segment size in tokens
        segment_overlap: Segment overlap in tokens
        chunk_size: Chunk size in tokens
        chunk_overlap: Chunk overlap in tokens
        embed_context: If True, embed context+chunk. If False, embed only chunk.

    Returns:
        Number of chunks created.
    """
    storage_path = settings.STORAGE_PATH
    project_dir = storage_path / name
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Building: {name}")
    print(f"  Segment: {segment_size} tokens, overlap: {segment_overlap}")
    print(f"  Chunk: {chunk_size} tokens, overlap: {chunk_overlap}")
    print(f"  Embed context: {embed_context}")
    print(f"{'='*60}\n")

    # Load document
    print("Loading document...", flush=True)
    content = load_document(SOURCE_FILE)

    # Create chunks
    print("Creating chunks...", flush=True)
    chunks, segments_map = create_chunks(
        text=content,
        source=SOURCE_FILE.name,
        segment_size=segment_size,
        segment_overlap=segment_overlap,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"  Created {len(chunks)} chunks from {len(segments_map)} segments", flush=True)

    # Group chunks by segment
    chunks_by_segment: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        seg_idx = chunk.segment_idx
        if seg_idx not in chunks_by_segment:
            chunks_by_segment[seg_idx] = []
        chunks_by_segment[seg_idx].append(chunk)

    # Generate context for each segment's chunks
    print("Generating context (this may take a while)...", flush=True)
    contextualized_chunks: list[ContextualizedChunk] = []

    for seg_idx in tqdm(sorted(chunks_by_segment.keys()), desc="Segments"):
        segment_chunks = chunks_by_segment[seg_idx]
        segment_text = segments_map.get(seg_idx, "")

        ctx_chunks = await generate_contexts_batch(
            segment=segment_text,
            chunks=segment_chunks,
            model=None,  # Use default (BackendAI if configured)
            skip_context=False,
        )
        contextualized_chunks.extend(ctx_chunks)

    print(f"  Generated context for {len(contextualized_chunks)} chunks", flush=True)

    # Build FAISS index with context in metadata only (not embedded)
    print("Building FAISS index...", flush=True)
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

    documents = []
    for ctx_chunk in tqdm(contextualized_chunks, desc="Preparing docs"):
        # KEY DIFFERENCE: If not embed_context, use only original content for embedding
        if embed_context:
            page_content = ctx_chunk.contextualized_content  # context + chunk
        else:
            page_content = ctx_chunk.chunk.content  # original chunk only

        doc = Document(
            page_content=page_content,
            metadata={
                "chunk_id": ctx_chunk.chunk.chunk_id,
                "source": ctx_chunk.chunk.source,
                "segment_idx": ctx_chunk.chunk.segment_idx,
                "chunk_idx": ctx_chunk.chunk.chunk_idx,
                "context": ctx_chunk.context,  # Store context in metadata
                "original_content": ctx_chunk.chunk.content,
            },
        )
        documents.append(doc)

    # Build FAISS in batches
    batch_size = 200
    if len(documents) <= batch_size:
        vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    else:
        print(f"  Building FAISS in batches of {batch_size}...", flush=True)
        vectorstore = FAISS.from_documents(documents=documents[:batch_size], embedding=embeddings)
        for i in tqdm(range(batch_size, len(documents), batch_size), desc="FAISS batches"):
            batch_docs = documents[i:i + batch_size]
            vectorstore.add_documents(batch_docs)

    # Save FAISS
    vectorstore.save_local(str(project_dir), index_name="faiss")
    print(f"  Saved FAISS index", flush=True)

    # Build BM25 index
    print("Building BM25 index...", flush=True)
    bm25 = BM25Store()
    bm25.build_index(contextualized_chunks)
    bm25.save(project_dir)
    print(f"  Saved BM25 index", flush=True)

    # Save config
    config = ProjectConfig(
        name=name,
        storage_path=storage_path,
        segment_size=segment_size,
        segment_overlap=segment_overlap,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        context_min_tokens=100,
        context_max_tokens=200,
        embedding_model=settings.EMBEDDING_MODEL,
        context_model=settings.CONTEXT_MODEL,
        enable_faiss=True,
        enable_bm25=True,
    )
    config_path = project_dir / "config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    # Save raw chunks
    raw_chunks_path = project_dir / "raw_chunks.json"
    raw_chunks_data = [c.model_dump() for c in chunks]
    raw_chunks_path.write_text(json.dumps(raw_chunks_data, indent=2), encoding="utf-8")

    # Save segments
    segments_path = project_dir / "segments.json"
    segments_path.write_text(json.dumps(segments_map, indent=2), encoding="utf-8")

    # Save contextualized chunks
    chunks_path = project_dir / "chunks.json"
    chunks_data = [
        {"chunk": c.chunk.model_dump(), "context": c.context}
        for c in contextualized_chunks
    ]
    chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

    print(f"\nProject '{name}' built successfully with {len(contextualized_chunks)} chunks!")
    return len(contextualized_chunks)


async def main():
    """Build all experimental projects."""
    print("="*60)
    print("EXPERIMENTAL PROJECT BUILDER")
    print("="*60)
    print(f"\nSource: {SOURCE_FILE}")
    print(f"Storage: {settings.STORAGE_PATH}")

    # BASELINE: 8000/800 with context EMBEDDED (same source as experiments)
    await build_experimental_project(
        name="exp_8000_800_context_embed",
        segment_size=8000,
        segment_overlap=800,
        chunk_size=800,
        chunk_overlap=80,
        embed_context=True,  # Context EMBEDDED into chunk
    )

    # Experiment 1: 8000/800 with context in metadata only
    await build_experimental_project(
        name="exp_8000_800_context_meta",
        segment_size=8000,
        segment_overlap=800,
        chunk_size=800,
        chunk_overlap=80,
        embed_context=False,  # Context in metadata only
    )

    # Experiment 2: 4000/400 with context in metadata only
    await build_experimental_project(
        name="exp_4000_400_context_meta",
        segment_size=4000,
        segment_overlap=400,
        chunk_size=400,
        chunk_overlap=40,
        embed_context=False,  # Context in metadata only
    )

    print("\n" + "="*60)
    print("ALL EXPERIMENTAL PROJECTS BUILT SUCCESSFULLY")
    print("="*60)

    # Summary
    print("\nProjects created:")
    print("1) exp_8000_800_context_embed - 8000/800 with context EMBEDDED (baseline)")
    print("2) exp_8000_800_context_meta - 8000/800 with context in metadata")
    print("3) exp_4000_400_context_meta - 4000/400 with context in metadata")


if __name__ == "__main__":
    asyncio.run(main())
