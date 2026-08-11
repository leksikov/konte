"""Corpora the benchmark cases run against.

Three tiers, for three different jobs:

``real``
    Copies of already-built projects. Real prose, real index sizes, real
    distribution of chunk lengths. Read-only: a copy is taken because saving a
    project with the newer revision rewrites its layout past what the older one
    can read.

``document``
    An excerpt of a real source document, sized in tokens, for cases that build
    a project end to end against a live endpoint.

``synthetic``
    Generated text, sized in tokens, for the scale tiers. Deterministic from a
    fixed seed and materialized to disk once, so both revisions read byte-identical
    input - the corpus must not be a variable in a comparison of the code.

Nothing here imports konte. Corpus construction has to be identical on both
sides of the comparison, so it cannot go through the code under test.
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import tiktoken

from benchmarks.harness import PROTECTED_STORAGE, REPO_ROOT, assert_scratch_storage, scratch_root

#: Matches konte's own tokenizer. Fixed here rather than read from konte so the
#: corpus does not change when the code under test does.
ENCODING_NAME = "o200k_base"

#: Tokens a chunk actually contributes once overlap is accounted for, at konte's
#: defaults (800-token chunks, 80-token overlap). Used to size a corpus by the
#: chunk count a claim is stated at.
EFFECTIVE_TOKENS_PER_CHUNK = 720

SOURCE_DOCUMENT = REPO_ROOT / "example_knowledge_base" / "wco_hs_explanatory_notes.md"

_SENTENCES = [
    "This heading covers goods presented in sets put up for retail sale, provided the "
    "components are put up together to meet a particular need or carry out a specific activity.",
    "Parts suitable for use solely or principally with the machines of this heading are "
    "classified in the same heading, subject to the general provisions governing parts.",
    "The classification of composite goods is determined by the material or component "
    "which gives them their essential character, so far as this criterion is applicable.",
    "Goods which cannot be classified by reference to the preceding provisions are "
    "classified under the heading which occurs last in numerical order among those which "
    "equally merit consideration.",
    "This heading excludes articles of a kind used as parts of general application, as "
    "well as similar articles of plastics or of other materials.",
    "Where reference is made to a material or substance, that reference extends to mixtures "
    "or combinations of that material or substance with other materials or substances.",
    "Products obtained by a process of manufacture which alters the essential character of "
    "the input material are treated as originating in the country where that process occurs.",
    "Containers specially shaped or fitted to hold a specific article, presented with the "
    "articles for which they are intended, are classified with those articles.",
]

_TOPICS = [
    "machinery and mechanical appliances",
    "electrical equipment and parts thereof",
    "optical and precision instruments",
    "base metals and articles of base metal",
    "plastics and articles thereof",
    "textile fabrics and made-up articles",
    "vehicles and associated transport equipment",
    "chemical and allied industry products",
]


def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def corpora_dir() -> Path:
    path = scratch_root() / "corpora"
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_dir(tier: str) -> Path:
    """Scratch storage for built projects, guarded against the real store."""
    path = assert_scratch_storage(scratch_root() / "projects" / tier)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# synthetic
# --------------------------------------------------------------------------


def _generate(target_tokens: int, seed: int) -> str:
    """Build prose of roughly ``target_tokens`` tokens, deterministically."""
    rng = random.Random(seed)
    encoder = _encoder()
    parts: list[str] = []
    tokens = 0
    section = 0
    # Grow in blocks and re-measure, rather than counting per sentence: encoding
    # a whole block at once is far cheaper than encoding each line.
    while tokens < target_tokens:
        section += 1
        block = [f"\n## Section {section}. Notes on {rng.choice(_TOPICS)}\n"]
        for _paragraph in range(6):
            sentences = [rng.choice(_SENTENCES) for _ in range(rng.randint(3, 6))]
            heading_ref = f"{rng.randint(28, 97):02d}.{rng.randint(1, 99):02d}"
            block.append(f"Heading {heading_ref}. " + " ".join(sentences))
        text = "\n\n".join(block)
        parts.append(text)
        tokens += len(encoder.encode(text))
    return "\n".join(parts)


def synthetic_document(chunks: int, *, seed: int = 20260811) -> Path:
    """Return a generated document sized for roughly ``chunks`` chunks.

    Written once and reused. Both revisions read the same file, so the corpus
    cannot drift between the two halves of a comparison.
    """
    target_tokens = chunks * EFFECTIVE_TOKENS_PER_CHUNK
    path = corpora_dir() / f"synthetic-{chunks}-chunks.md"
    stamp = path.with_suffix(".stamp")
    signature = f"{target_tokens}:{seed}:{len(_SENTENCES)}:{len(_TOPICS)}"
    digest = hashlib.sha256(signature.encode()).hexdigest()[:16]

    if path.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return path

    path.write_text(_generate(target_tokens, seed))
    stamp.write_text(digest)
    return path


# --------------------------------------------------------------------------
# real source document
# --------------------------------------------------------------------------


def document_excerpt(chunks: int) -> Path:
    """Return an excerpt of a real source document sized for ``chunks`` chunks.

    Used by the live-endpoint cases: context generation should see real prose,
    because generation latency depends on what the model is asked to read.
    """
    if not SOURCE_DOCUMENT.exists():
        raise FileNotFoundError(
            f"{SOURCE_DOCUMENT} is missing; it is untracked local data. "
            "Point KONTE_BENCH_SOURCE_DOC at another document, or use the "
            "synthetic tier."
        )
    target_tokens = chunks * EFFECTIVE_TOKENS_PER_CHUNK
    path = corpora_dir() / f"excerpt-{chunks}-chunks.md"
    if path.exists():
        return path

    encoder = _encoder()
    tokens = encoder.encode(SOURCE_DOCUMENT.read_text())
    if len(tokens) < target_tokens:
        raise ValueError(
            f"{SOURCE_DOCUMENT.name} holds {len(tokens)} tokens, too few for "
            f"{chunks} chunks ({target_tokens} tokens)"
        )
    path.write_text(encoder.decode(tokens[:target_tokens]))
    return path


# --------------------------------------------------------------------------
# real built projects
# --------------------------------------------------------------------------


def real_project(name: str) -> tuple[Path, str]:
    """Copy an already-built project into scratch storage and return it.

    Returns ``(storage_path, project_name)`` so a case can open it the ordinary
    way. The copy exists because the newer revision's ``save()`` deletes the
    legacy lexical corpus file and slims the lexical index, neither of which the
    older revision can read back - measuring against the original would destroy
    the baseline halfway through the run.
    """
    source = PROTECTED_STORAGE / name
    if not source.is_dir():
        available = sorted(p.name for p in PROTECTED_STORAGE.iterdir() if p.is_dir())
        raise FileNotFoundError(f"no built project {name!r}; found: {available}")

    storage = projects_dir("real")
    target = storage / name
    if not target.exists():
        shutil.copytree(source, target)
    return storage, name


def available_real_projects() -> list[str]:
    if not PROTECTED_STORAGE.is_dir():
        return []
    return sorted(p.name for p in PROTECTED_STORAGE.iterdir() if (p / "config.json").exists())
