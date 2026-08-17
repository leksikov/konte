"""Unit tests for crash-safe file primitives."""

import json

import pytest

from konte.persistence.storage import atomic_writer, read_json, read_jsonl, write_json


@pytest.mark.unit
class TestAtomicWriter:
    """Test atomic_writer."""

    def test_writes_file(self, tmp_path):
        """Test that a clean block produces the file."""
        path = tmp_path / "out.txt"

        with atomic_writer(path, encoding="utf-8") as handle:
            handle.write("hello")

        assert path.read_text(encoding="utf-8") == "hello"

    def test_creates_parent_directory(self, tmp_path):
        """Test that a missing parent directory is created."""
        path = tmp_path / "nested" / "deeper" / "out.txt"

        with atomic_writer(path, encoding="utf-8") as handle:
            handle.write("hello")

        assert path.exists()

    def test_binary_mode(self, tmp_path):
        """Test that omitting the encoding writes bytes."""
        path = tmp_path / "out.bin"

        with atomic_writer(path) as handle:
            handle.write(b"\x00\x01\x02")

        assert path.read_bytes() == b"\x00\x01\x02"

    def test_failure_leaves_original_intact(self, tmp_path):
        """Test that a raise mid-write does not damage the file being replaced."""
        path = tmp_path / "out.txt"
        path.write_text("original", encoding="utf-8")

        with pytest.raises(RuntimeError):
            with atomic_writer(path, encoding="utf-8") as handle:
                handle.write("half a rep")
                raise RuntimeError("killed mid-write")

        assert path.read_text(encoding="utf-8") == "original"

    def test_failure_leaves_no_temporary_file(self, tmp_path):
        """Test that a failed write cleans up after itself."""
        path = tmp_path / "out.txt"

        with pytest.raises(RuntimeError):
            with atomic_writer(path, encoding="utf-8") as handle:
                handle.write("half a rep")
                raise RuntimeError("killed mid-write")

        assert list(tmp_path.iterdir()) == []

    def test_success_leaves_no_temporary_file(self, tmp_path):
        """Test that the temporary file is renamed, not left behind."""
        path = tmp_path / "out.txt"

        with atomic_writer(path, encoding="utf-8") as handle:
            handle.write("hello")

        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_preserves_existing_permissions(self, tmp_path):
        """Test that rewriting a file does not silently change its mode."""
        path = tmp_path / "out.txt"
        path.write_text("original", encoding="utf-8")
        path.chmod(0o640)

        with atomic_writer(path, encoding="utf-8") as handle:
            handle.write("replacement")

        assert path.stat().st_mode & 0o777 == 0o640


@pytest.mark.unit
class TestWriteJson:
    """Test write_json."""

    def test_round_trip(self, tmp_path):
        """Test that what is written is what is read back."""
        path = tmp_path / "data.json"
        data = {"a": [1, 2, 3], "b": "값"}

        write_json(path, data)

        assert read_json(path) == data

    def test_compact_by_default(self, tmp_path):
        """Test that machine-read artifacts are not indented."""
        path = tmp_path / "data.json"

        write_json(path, {"a": 1})

        assert "\n" not in path.read_text(encoding="utf-8")

    def test_indent_is_available(self, tmp_path):
        """Test that human-facing files can still be indented."""
        path = tmp_path / "config.json"

        write_json(path, {"a": 1}, indent=2)

        assert "\n" in path.read_text(encoding="utf-8")

    def test_non_ascii_is_not_escaped(self, tmp_path):
        """Test that Korean text is stored as UTF-8, not as escapes."""
        path = tmp_path / "data.json"

        write_json(path, {"q": "관세"})

        assert "관세" in path.read_text(encoding="utf-8")


@pytest.mark.unit
class TestReadJson:
    """Test read_json."""

    def test_missing_file_returns_none(self, tmp_path):
        """Test that an absent file reads as None."""
        assert read_json(tmp_path / "absent.json") is None


@pytest.mark.unit
class TestReadJsonl:
    """Test read_jsonl."""

    def test_reads_all_records(self, tmp_path):
        """Test that whole lines are returned in order."""
        path = tmp_path / "log.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")

        records, offset = read_jsonl(path)

        assert records == [{"a": 1}, {"a": 2}]
        assert offset == path.stat().st_size

    def test_unterminated_tail_is_excluded(self, tmp_path):
        """Test that a line with no newline is treated as never written."""
        path = tmp_path / "log.jsonl"
        intact = '{"a": 1}\n'
        path.write_text(intact + '{"a": 2', encoding="utf-8")

        records, offset = read_jsonl(path)

        assert records == [{"a": 1}]
        assert offset == len(intact)

    def test_invalid_tail_is_excluded(self, tmp_path):
        """Test that a terminated but unparseable line stops the read."""
        path = tmp_path / "log.jsonl"
        intact = '{"a": 1}\n'
        path.write_text(intact + '{"a":\n', encoding="utf-8")

        records, offset = read_jsonl(path)

        assert records == [{"a": 1}]
        assert offset == len(intact)

    def test_empty_file(self, tmp_path):
        """Test that an empty log reads as no records."""
        path = tmp_path / "log.jsonl"
        path.write_text("", encoding="utf-8")

        assert read_jsonl(path) == ([], 0)

    def test_embedded_newlines_do_not_split_records(self, tmp_path):
        """Test that text containing newlines still occupies one line."""
        path = tmp_path / "log.jsonl"
        path.write_text(json.dumps({"text": "one\ntwo"}) + "\n", encoding="utf-8")

        records, _ = read_jsonl(path)

        assert records == [{"text": "one\ntwo"}]
