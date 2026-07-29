# Contributing to Konte

Thanks for your interest in contributing!

## Development Setup

Konte uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/leksikov/konte.git
cd konte
uv sync   # creates .venv with the library, CLI, and dev tools from uv.lock
```

If you prefer plain pip: `pip install -e .` plus `pip install --group dev` (pip ≥ 25.1).

## Running Tests

Tests are organized in three layers with pytest markers:

| Layer | Location | Requirements |
|-------|----------|--------------|
| Unit | `tests/unit/` | None (mocks allowed here only) |
| Integration | `tests/integration/` | `OPENAI_API_KEY` (real API calls) |
| End-to-end | `tests/e2e/` | `OPENAI_API_KEY` (slow, full workflows) |

```bash
uv run pytest tests/unit          # fast, no API keys — run these before every PR
uv run pytest tests/integration   # requires OPENAI_API_KEY
uv run pytest                     # everything
```

On macOS, if FAISS crashes with an OpenMP error, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Code Style

- Python 3.10+, fully type-hinted
- Pydantic V2 for all models (no dataclasses)
- Async-first: no threads, no mixing sync into async paths
- `pathlib` for paths; configuration through `konte/config/settings.py` (env vars / `.env`)
- `structlog` for logging
- Lint with `uv run ruff check .` before submitting

## Commit Messages

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```
type(scope): Description
```

- `feat` (minor), `fix`/`perf` (patch); `!` or a `BREAKING CHANGE:` footer for major
- Non-bumping types: `chore`, `refactor`, `build`, `test`, `ci`, `docs`, `style`
- Subject: imperative mood, capitalized, ≤50 chars, no trailing period

## Pull Requests

1. Fork and create a branch (`feat/...` or `fix/...`)
2. Add or update tests for your change (unit tests must pass without API keys)
3. Run `uv run pytest tests/unit` and `uv run ruff check .`
4. Open a PR with a clear description of the motivation and the change

## Releasing (maintainers)

Releases are published to PyPI by `.github/workflows/release.yml` via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API tokens.

One-time setup on [pypi.org](https://pypi.org/manage/account/publishing/): add a
**pending trusted publisher** for project `konte` with owner `leksikov`,
repository `konte`, workflow `release.yml`, environment `pypi`.

Per release:

1. Bump `version` in `pyproject.toml` and add a dated CHANGELOG entry (with its link reference)
2. Commit, tag `vX.Y.Z`, and push the tag
3. Create a GitHub release from the tag — publishing the release triggers the
   workflow, which builds the sdist/wheel, validates metadata with
   `twine check`, and uploads to PyPI
4. After the first publish, add the PyPI version badge to the README:
   `[![PyPI](https://img.shields.io/pypi/v/konte)](https://pypi.org/project/konte/)`

To dry-run locally without uploading: `uv build && uvx twine check dist/*`
