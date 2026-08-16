# Maintainers

This file records who maintains Konte and what that carries. It is the list
`.mailmap` refers to; one-off contributors are credited by git history, not here.

## Current maintainers

| Name | GitHub | Role |
|---|---|---|
| Sergey Leksikov | [@leksikov](https://github.com/leksikov) | Author |
| Минже | [@urdekcah](https://github.com/urdekcah) | Maintainer, security contact |

Reach either of them through a [GitHub issue](https://github.com/leksikov/konte/issues)
or a pull request. **Do not report a vulnerability that way** — see
[SECURITY.md](SECURITY.md) for the private channel.

## What a maintainer does

- **Review and merge.** Contributed changes arrive as pull requests and need a
  maintainer's approval. No area of the codebase is reserved to one person;
  whoever knows the surface reviews it.
- **Keep the gates honest.** A change merges only with `uv run ruff check .` and
  `uv run pytest tests/unit` green. The integration and end-to-end layers need an
  `OPENAI_API_KEY`, so CI never runs them — someone has to before a release.
- **Cut releases.** The tag-and-publish procedure lives in
  [CONTRIBUTING.md](CONTRIBUTING.md#releasing-maintainers). Publishing runs
  through PyPI trusted publishing, which only maintainers can configure.
- **Answer security reports.** Triage, fix, and disclosure are described in
  [SECURITY.md](SECURITY.md).

## Decisions

Ordinary changes need one approving review. Anything that changes the on-disk
format, the public API, or the storage trust model needs agreement from both
maintainers, because those cannot be walked back for anyone who already
depends on them — a released format is a promise, not a default.

Disagreement that a review cannot settle is worked out in the pull request
thread. There is no tie-breaker, so an unresolved objection means the change
does not land in that form.

## Becoming a maintainer

There is no application. A contributor who has landed a body of non-trivial
work, reviewed other people's changes usefully, and stayed around long enough
to maintain what they wrote may be invited by the current maintainers. The
invitation adds a row to the table above and an entry to `.mailmap`.

## Stepping down

Say so in an issue and open a pull request removing the row. Retired
maintainers keep their `.mailmap` entry — it exists to keep authorship in the
history consistent, not to mark who is currently on duty.
