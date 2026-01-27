"""Pytest configuration and fixtures."""

import os

# Fix OpenMP conflict on macOS (FAISS + other libs linking to libomp)
# This must be set before any FAISS imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pytest_sessionfinish(session, exitstatus):
    """Print message after all tests complete."""
    print("Running teardown with pytest sessionfinish...")
