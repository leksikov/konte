"""Unit tests for settings module."""

import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.mark.unit
class TestSettings:
    """Test settings validation and defaults."""

    def test_default_values(self):
        """Test that default settings are applied correctly."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            from konte.config import Settings

            s = Settings()
            assert s.EMBEDDING_MODEL == "text-embedding-3-small"
            assert s.CONTEXT_MODEL == "gpt-4.1-mini"
            assert s.SEGMENT_SIZE == 8000
            assert s.SEGMENT_OVERLAP == 800
            assert s.CHUNK_SIZE == 800
            assert s.CHUNK_OVERLAP == 80
            assert s.CONTEXT_MIN_TOKENS == 100
            assert s.CONTEXT_MAX_TOKENS == 200
            assert s.DEFAULT_TOP_K == 20
            assert s.MAX_CONCURRENT_CALLS == 1  # Sequential to avoid rate limits
            assert s.CONFIDENCE_THRESHOLD == 0.7

    def test_storage_path_expansion(self):
        """Test that ~ is expanded in STORAGE_PATH."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key", "STORAGE_PATH": "~/.konte"},
            clear=False,
        ):
            from konte.config import Settings

            s = Settings()
            assert "~" not in str(s.STORAGE_PATH)
            assert isinstance(s.STORAGE_PATH, Path)

    def test_prompt_path_is_path(self):
        """Test that PROMPT_PATH is a Path object."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            from konte.config import Settings

            s = Settings()
            assert isinstance(s.PROMPT_PATH, Path)
            assert s.PROMPT_PATH.name == "context_prompt.txt"

    def test_openai_api_key_optional(self):
        """Test that OPENAI_API_KEY is optional (Backend.AI is default)."""
        from konte.config import Settings

        # Verify the field has default=None (making it optional)
        field_info = Settings.model_fields["OPENAI_API_KEY"]
        assert field_info.default is None, "OPENAI_API_KEY should default to None"

    def test_overlap_less_than_size(self):
        """Test segment overlap is less than segment size."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            from konte.config import Settings

            s = Settings()
            assert s.SEGMENT_OVERLAP < s.SEGMENT_SIZE
            assert s.CHUNK_OVERLAP < s.CHUNK_SIZE
