"""E2E tests for processing large documents (>100k tokens).

These tests validate the system handles large documents correctly,
with proper segmentation and without memory issues.
"""

import os
from pathlib import Path

import pytest

from konte.chunker import count_tokens

# Skip all tests if OPENAI_API_KEY is not set
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
    pytest.mark.e2e,
]


def generate_large_document(num_tokens: int) -> str:
    """Generate a synthetic document with approximately num_tokens tokens.

    Creates a document with varied content to simulate real text.
    """
    paragraphs = [
        "The Harmonized System (HS) is an international nomenclature for the classification of products. "
        "It allows participating countries to classify traded goods on a common basis for customs purposes. "
        "At the international level, the HS for classifying goods is a six-digit code system.",

        "Import duties are taxes collected on imports by a country's customs authorities. "
        "A duty levied on goods based on their value is called an ad valorem duty. "
        "Alternatively, duty may be levied on the quantity of goods, known as a specific duty.",

        "Free trade agreements (FTAs) reduce or eliminate tariffs between member countries. "
        "Rules of origin determine which goods qualify for preferential treatment under these agreements. "
        "Certificate of origin documentation is required to claim preferential tariff rates.",

        "Customs valuation is the process of determining the economic value of imported goods. "
        "The transaction value method is the primary basis for customs valuation under WTO rules. "
        "When transaction value cannot be used, alternative methods include deductive and computed value.",

        "Tariff rate quotas (TRQs) allow a certain quantity of imports at a lower duty rate. "
        "Once the quota is filled, higher duties apply to additional imports. "
        "TRQs are commonly used for agricultural products in international trade.",
    ]

    # Repeat paragraphs to reach target token count
    document = []
    current_tokens = 0
    para_idx = 0

    while current_tokens < num_tokens:
        para = paragraphs[para_idx % len(paragraphs)]
        document.append(para)
        current_tokens += count_tokens(para)
        para_idx += 1

        # Add section headers periodically for variety
        if para_idx % 10 == 0:
            header = f"\n## Section {para_idx // 10}: Trade Regulations\n"
            document.append(header)
            current_tokens += count_tokens(header)

    return "\n\n".join(document)


@pytest.mark.e2e
class TestLargeDocumentProcessing:
    """Test processing of large documents."""

    async def test_process_50k_token_document(self, tmp_path):
        """Test processing a 50k token document (skip context for speed)."""
        from konte.project import Project

        # Generate large document
        large_doc_path = tmp_path / "large_doc.txt"
        doc_content = generate_large_document(50000)
        large_doc_path.write_text(doc_content)

        doc_tokens = count_tokens(doc_content)
        assert doc_tokens >= 45000, f"Document only has {doc_tokens} tokens"

        # Process document
        project = Project.create(
            name="e2e_large_doc_50k",
            storage_path=tmp_path,
        )

        num_chunks = project.add_documents([large_doc_path])

        # Should create many chunks
        assert num_chunks > 50, f"Expected >50 chunks, got {num_chunks}"

        # Should create multiple segments
        assert len(project._segments) > 5, f"Expected >5 segments, got {len(project._segments)}"

        # Build without context (skip LLM calls for speed)
        await project.build(skip_context=False)

        # Query should work
        response = project.query("tariff classification customs")
        assert response.total_found > 0

    async def test_segments_bounded_for_large_document(self, tmp_path):
        """Verify segments are correctly bounded for large documents."""
        from konte.project import Project

        # Generate large document
        large_doc_path = tmp_path / "large_doc_segments.txt"
        doc_content = generate_large_document(30000)
        large_doc_path.write_text(doc_content)

        doc_tokens = count_tokens(doc_content)

        project = Project.create(
            name="e2e_large_segments",
            storage_path=tmp_path,
            segment_size=8000,  # Default
            segment_overlap=800,
        )

        project.add_documents([large_doc_path])

        # All segments should be bounded
        for seg_idx, segment_text in project._segments.items():
            segment_tokens = count_tokens(segment_text)

            # Segment should be much smaller than full document
            assert segment_tokens < doc_tokens / 2, (
                f"Segment {seg_idx} has {segment_tokens} tokens, "
                f"should be < {doc_tokens / 2} (half of document)"
            )

            # Segment should be close to segment_size
            assert segment_tokens <= 8000 * 1.5, (
                f"Segment {seg_idx} has {segment_tokens} tokens, "
                f"expected <= 12000 (8000 * 1.5)"
            )

    async def test_all_chunks_have_valid_segment_reference(self, tmp_path):
        """Verify all chunks reference valid segments for large docs."""
        from konte.project import Project

        # Generate document
        large_doc_path = tmp_path / "large_doc_refs.txt"
        doc_content = generate_large_document(20000)
        large_doc_path.write_text(doc_content)

        project = Project.create(
            name="e2e_large_refs",
            storage_path=tmp_path,
        )

        project.add_documents([large_doc_path])

        # Every chunk must have a valid segment reference
        # Segment keys are (source, segment_idx) tuples
        for chunk in project._chunks:
            segment_key = (chunk.source, chunk.segment_idx)
            assert segment_key in project._segments, (
                f"Chunk {chunk.chunk_id} references segment {segment_key} "
                f"which doesn't exist in segments map"
            )


@pytest.mark.e2e
class TestLargeDocumentWithContext:
    """Test large document processing with context generation.

    These tests are slow due to API calls - use sparingly.
    """

    async def test_process_10k_document_with_context(self, tmp_path):
        """Test processing a 10k token document with context generation."""
        from konte.project import Project

        # Generate smaller document for context test (to avoid rate limits)
        large_doc_path = tmp_path / "doc_with_context.txt"
        doc_content = generate_large_document(10000)
        large_doc_path.write_text(doc_content)

        project = Project.create(
            name="e2e_context_large",
            storage_path=tmp_path,
            chunk_size=400,  # Smaller chunks = fewer API calls
            chunk_overlap=40,
        )

        project.add_documents([large_doc_path])

        # Build with context (real API calls)
        await project.build(skip_context=False)

        # Verify some chunks got context
        chunks_with_context = [
            c for c in project._contextualized_chunks
            if c.context and len(c.context) > 10
        ]

        # At least some should have context (allow for rate limiting)
        assert len(chunks_with_context) > 0, "No chunks received context"

        # Query should return results
        response = project.query("import duty tariff")
        assert response.total_found > 0


@pytest.mark.e2e
class TestLargeDocumentSaveLoad:
    """Test save/load for large documents."""

    async def test_save_load_large_document(self, tmp_path):
        """Test that large document project saves and loads correctly."""
        from konte.project import Project

        # Generate document
        large_doc_path = tmp_path / "large_save_load.txt"
        doc_content = generate_large_document(20000)
        large_doc_path.write_text(doc_content)

        # Create and build
        project1 = Project.create(
            name="e2e_large_persist",
            storage_path=tmp_path,
        )
        project1.add_documents([large_doc_path])
        await project1.build(skip_context=False)

        num_chunks = len(project1._contextualized_chunks)
        num_segments = len(project1._segments)

        # Save
        project1.save()

        # Verify files created
        project_dir = tmp_path / "e2e_large_persist"
        assert (project_dir / "faiss.faiss").exists()
        assert (project_dir / "bm25.pkl").exists()
        assert (project_dir / "chunks.json").exists()

        # Reload
        project2 = Project.open(name="e2e_large_persist", storage_path=tmp_path)

        # Verify data preserved
        assert len(project2._contextualized_chunks) == num_chunks

        # Query should work
        response = project2.query("customs valuation")
        assert response.total_found > 0
