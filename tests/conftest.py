"""Pytest configuration and fixtures."""

import os

# Fix OpenMP conflict on macOS (FAISS + other libs linking to libomp)
# This must be set before any FAISS imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_query_caches():
    """Empty the process-wide query caches around each test.

    Neither key distinguishes one test's stub client from the next one's.
    """
    from konte.index.embeddings import clear_query_embedding_cache
    from konte.retrieval.query_processor import clear_keyword_cache

    clear_query_embedding_cache()
    clear_keyword_cache()
    yield
    clear_query_embedding_cache()
    clear_keyword_cache()


def pytest_sessionfinish(session, exitstatus):
    """Print message after all tests complete."""
    print("Running teardown with pytest sessionfinish...")
