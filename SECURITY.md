# Security Policy

## Reporting a vulnerability

Email **ya@urdekcah.ru**. That address is for security reports only; anything
else belongs in a [GitHub issue](https://github.com/leksikov/konte/issues).

Please do not open a public issue, pull request, or discussion for a
vulnerability until a fix is available.

A useful report says which version or commit you tested, what an attacker
gains, and the shortest path to reproducing it. A proof of concept is welcome
but not required — a clear description of the flawed code path is enough.

You will get an acknowledgement within three working days. If none arrives,
assume the mail was lost and open a GitHub issue asking for a security contact
without describing the problem.

We will tell you whether the report is accepted, agree a disclosure date with
you, credit you in the CHANGELOG unless you would rather stay anonymous, and
publish the fix and the advisory together.

## Supported versions

No release has been published yet. The `main` branch is the only supported
version, and fixes land there.

Once versioned releases exist, this section will name the ones that receive
security fixes.

## Scope

Konte is a library. It runs inside your process, reads a storage directory you
choose, and talks to model endpoints you configure. That shapes what counts as
a vulnerability.

**In scope** — anything that lets one of those inputs exceed what it was
granted:

- A project name, document path, or `config.json` field that reaches outside
  `STORAGE_PATH`
- An artifact under `STORAGE_PATH` that causes code execution when a project
  is opened, queried, or rebuilt
- An index file that passes the integrity check without having been recorded
  by the installation that opened it, or a way to bypass that check while
  `INDEX_INTEGRITY=enforce`
- A document that causes something worse than a parse error when loaded
- Leaking `OPENAI_API_KEY`, `LLM_API_KEY`, or the contents of `.signing-key`
  into logs, artifacts, or error messages

**Out of scope** — documented behavior working as intended:

- Anything reachable only by an attacker who can already write into
  `STORAGE_PATH` *and* holds the signing key, or who runs as the same user
- `INDEX_INTEGRITY=warn` or `off` loading an index that fails its check. Both
  are opt-in settings whose cost is described in the README
- Content returned by an LLM or reranker endpoint you pointed Konte at,
  including prompt injection carried in your own documents. Retrieved text is
  data your application is responsible for handling
- `RERANKER_VERIFY_SSL=false`, which exists for self-signed internal
  endpoints and disables certificate verification by design
- Denial of service through a corpus you supplied to your own build

## Hardening

The storage directory is the trust boundary. [Storage
trust](https://github.com/leksikov/konte/blob/main/README.md#storage-trust) in
the README covers what the integrity record proves, how to choose between a
local key and a committed manifest, and what each option costs. Two points are
worth repeating here:

- Konte stores no pickles. Index files are parsed as data, so no index file can
  make the process that reads it run code
- Keep `INDEX_INTEGRITY` at `enforce`, and treat `konte trust` as a statement
  that you know where the files came from. Where that is in doubt, rebuild
  instead
