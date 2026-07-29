# Contributing to Konte

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/leksikov/konte.git
cd konte
pip install -e ".[dev]"
```

## Running Tests

Tests are organized in three layers with pytest markers:

| Layer | Location | Requirements |
|-------|----------|--------------|
| Unit | `tests/unit/` | None (mocks allowed here only) |
| Integration | `tests/integration/` | `OPENAI_API_KEY` (real API calls) |
| End-to-end | `tests/e2e/` | `OPENAI_API_KEY` (slow, full workflows) |

```bash
pytest tests/unit          # fast, no API keys — run these before every PR
pytest tests/integration   # requires OPENAI_API_KEY
pytest                     # everything
```

On macOS, if FAISS crashes with an OpenMP error, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Code Style

- Python 3.10+, fully type-hinted
- Pydantic V2 for all models (no dataclasses)
- Async-first: no threads, no mixing sync into async paths
- `pathlib` for paths; configuration through `konte/config/settings.py` (env vars / `.env`)
- `structlog` for logging
- Lint with `ruff check .` before submitting

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
3. Run `pytest tests/unit` and `ruff check .`
4. Open a PR with a clear description of the motivation and the change
