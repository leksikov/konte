"""Cross-revision benchmark harness.

Runs a benchmark case against two revisions of konte and reports their
measurements side by side.

Revision selection is by working directory. For both ``python -m pkg`` and
``python -c``, ``sys.path[0]`` is the process cwd, so running from the baseline
worktree makes ``import konte`` resolve there while the repository root stays on
PYTHONPATH so ``benchmarks`` itself remains importable. PYTHONPATH alone does not
work: the cwd entry precedes it and would win.

Both revisions share one virtualenv. That is deliberate and sound here - the
dependency set is byte-identical across the range under test, so konte's own
source is the only variable.

Nothing in this module imports konte. It has to load in a process that has not
yet chosen a revision.
"""

from __future__ import annotations

import json
import os
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = REPO_ROOT / ".benchmark-baseline"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

#: Revision the baseline worktree is pinned to - the commit before the work
#: under test. Overridable so the harness outlives this particular comparison.
BASELINE_REF = os.environ.get("KONTE_BENCH_BASELINE_REF", "de54b54")

REVISIONS = ("baseline", "head")

#: Real project storage. Cases must never write here: saving a project with the
#: newer revision rewrites its on-disk format irreversibly for the older one.
PROTECTED_STORAGE = Path("~/.konte").expanduser().resolve()


# --------------------------------------------------------------------------
# process placement
# --------------------------------------------------------------------------


def isolate_revision() -> list[str]:
    """Remove the editable-install finder so a revision cannot leak submodules.

    ``konte`` is installed in editable mode, which appends a finder to
    ``sys.meta_path``. Top-level ``import konte`` still resolves from the cwd,
    because ``PathFinder`` is consulted first and wins. A *submodule* the cwd
    lacks does not: ``PathFinder`` searches ``konte.__path__``, finds nothing,
    and the editable finder then supplies the file from the repository - so the
    baseline process would silently import modules the baseline never had, and
    run them against the baseline's own classes.

    Returns the names of the finders that were dropped.
    """
    dropped = []
    kept = []
    for finder in sys.meta_path:
        name = getattr(finder, "__module__", "") or type(finder).__module__
        if name.startswith("__editable__"):
            dropped.append(name)
        else:
            kept.append(finder)
    sys.meta_path[:] = kept
    sys.path[:] = [entry for entry in sys.path if "__editable__" not in entry]
    return dropped


def revision_root(revision: str) -> Path:
    """Return the source tree a revision runs from."""
    if revision == "head":
        return REPO_ROOT
    if revision == "baseline":
        return BASELINE_ROOT
    raise ValueError(f"unknown revision {revision!r}, expected one of {REVISIONS}")


def ensure_baseline() -> Path:
    """Create the baseline worktree if it is not already checked out."""
    if (BASELINE_ROOT / "konte" / "__init__.py").exists():
        return BASELINE_ROOT
    subprocess.run(
        ["git", "worktree", "add", str(BASELINE_ROOT), BASELINE_REF],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return BASELINE_ROOT


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a dotenv file into a plain mapping.

    The baseline worktree has no dotenv of its own (it is untracked), so its
    subprocess only sees configuration if the harness passes it explicitly.
    Endpoint and credential values therefore live in the developer's local
    dotenv and travel through here; they are never hard-coded in this package.
    """
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def subprocess_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the environment a case subprocess runs under.

    Precedence matches pydantic-settings: real environment beats dotenv, and
    explicit case overrides beat both. Settings are passed as environment
    variables rather than relying on a dotenv, because ``konte.settings`` is
    instantiated at import and the baseline's cwd has no dotenv to read.
    """
    env = {**read_dotenv(REPO_ROOT / ".env"), **os.environ}
    env.update(overrides or {})
    env["PYTHONPATH"] = str(REPO_ROOT)
    # FAISS and libomp both link OpenMP on macOS; tests/conftest.py does the same.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def run_case(
    case: str,
    revision: str,
    *,
    overrides: Mapping[str, str] | None = None,
    options: Mapping[str, str] | None = None,
    timeout: float = 7200.0,
) -> dict:
    """Run one case against one revision, in its own process.

    A fresh process per measurement is what makes cold-import and first-touch
    numbers honest - module-level caches in konte would otherwise leak between
    trials.
    """
    root = revision_root(revision)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = Path(handle.name)
    option_args = []
    for key, value in (options or {}).items():
        option_args += ["--option", f"{key}={value}"]
    try:
        proc = subprocess.run(
            [
                str(VENV_PYTHON),
                "-m",
                "benchmarks.run",
                case,
                "--revision",
                revision,
                "--out",
                str(out_path),
                *option_args,
            ],
            cwd=root,
            env=subprocess_env(overrides),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not out_path.stat().st_size:
            return {
                "case": case,
                "revision": revision,
                "status": "error",
                "returncode": proc.returncode,
                "stderr": proc.stderr[-4000:],
            }
        payload = json.loads(out_path.read_text())
        # Kept even on success: konte logs retries and backoff to stderr, and a
        # build that succeeded slowly is a different story from one that
        # succeeded quickly.
        payload["stderr_tail"] = proc.stderr[-8000:]
        return payload
    finally:
        out_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# case context
# --------------------------------------------------------------------------


@dataclass
class Context:
    """What a case is handed when it runs."""

    revision: str
    root: Path
    scratch: Path
    options: dict = field(default_factory=dict)

    @property
    def is_baseline(self) -> bool:
        return self.revision == "baseline"


def scratch_root() -> Path:
    """Directory holding every corpus and project a benchmark builds.

    Deliberately outside the repository and outside real project storage; the
    scale tiers run to tens of gigabytes.
    """
    root = Path(
        os.environ.get("KONTE_BENCH_SCRATCH", Path(tempfile.gettempdir()) / "konte-bench")
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def assert_scratch_storage(path: Path | str) -> Path:
    """Refuse to let a case write inside real project storage.

    ``Project.save()`` on the newer revision deletes the legacy lexical corpus
    file and rewrites the lexical index without its tokenized copy, which the
    older revision then cannot read. One careless save would destroy the very
    baseline the comparison rests on, so this is enforced rather than trusted.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved == PROTECTED_STORAGE or PROTECTED_STORAGE in resolved.parents:
        raise RuntimeError(
            f"benchmark storage {resolved} is inside protected project storage "
            f"{PROTECTED_STORAGE}; benchmarks must work on copies"
        )
    return resolved


#: Settings fields are discovered by shape rather than named, because the two
#: revisions call the chat endpoint's settings different things. Discovery keeps
#: the harness working across the rename without encoding either name.
_ENDPOINT_SUFFIXES = ("_BASE_URL", "_ENDPOINT")
_MODEL_SUFFIXES = ("_MODEL", "_MODEL_NAME")
_NOT_CHAT = ("RERANKER", "EMBEDDING", "OPENAI")


@contextmanager
def point_llm_at(base_url: str, model: str) -> Iterator[list[str]]:
    """Redirect chat traffic at ``base_url`` for the block, then put it back.

    ``konte.settings`` is a module-level singleton built at import, so a case
    that needs a runtime-allocated endpoint has to assign to it rather than set
    an environment variable. Restoring on exit is not tidiness: a case that
    redirects at a stub and then does real work would otherwise send that work
    to a port nobody is listening on any more, and konte answers a failed
    context call with empty context rather than an error - so the run still
    looks like it succeeded, just slowly and with nothing generated.

    Yields the field names that were changed.
    """
    from konte.config import settings

    previous: dict[str, object] = {}
    changed = []
    for name in type(settings).model_fields:
        if name.startswith(_NOT_CHAT):
            continue
        if name.endswith(_ENDPOINT_SUFFIXES):
            value = base_url
        elif name.endswith(_MODEL_SUFFIXES):
            value = model
        elif name.endswith("_API_KEY"):
            value = "not-needed"
        else:
            continue
        previous[name] = getattr(settings, name)
        setattr(settings, name, value)
        changed.append(name)

    clear_llm_clients()
    try:
        yield changed
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)
        clear_llm_clients()


def active_chat_endpoint() -> str | None:
    """The chat endpoint konte would use right now, on either revision."""
    from konte.config import settings

    for name in type(settings).model_fields:
        if name.startswith(_NOT_CHAT) or not name.endswith(_ENDPOINT_SUFFIXES):
            continue
        value = getattr(settings, name, None)
        if value:
            return str(value)
    return None


def assert_live_endpoint() -> str:
    """Fail unless chat traffic is pointed at a real, remote endpoint.

    Guards the case this harness already got wrong once: a stub redirect left
    in place sends a live measurement at a closed local port, and konte answers
    a failed context call with empty context instead of an error - so the build
    reports success having generated nothing, and the wall-clock silently
    becomes a measurement of the retry budget.
    """
    endpoint = active_chat_endpoint()
    if not endpoint:
        raise RuntimeError(
            "no chat endpoint configured; a live measurement would fall back to "
            "the default provider or fail every call"
        )
    if any(host in endpoint for host in ("127.0.0.1", "localhost", "0.0.0.0")):
        raise RuntimeError(
            f"chat endpoint is {endpoint}, which is a local stub. A live "
            f"measurement must run against the configured remote endpoint."
        )
    return endpoint


def clear_llm_clients() -> None:
    """Drop cached chat clients so the next call rebuilds against new settings.

    Both revisions memoize constructed clients; they just keep the cache in
    different modules.
    """
    import importlib

    for module_name, attr in (("konte.llm", "_client_cache"), ("konte.context", "_llm_cache")):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cache = getattr(module, attr, None)
        if isinstance(cache, dict):
            cache.clear()


def pin_keyword_extraction() -> None:
    """Replace LLM keyword extraction with local tokenization, on either revision.

    The older revision has no way to turn extraction off - it runs on every
    lexical and hybrid query - so a like-for-like comparison of the ranking code
    has to neutralize it on both sides. Both revisions import the function into
    the retriever's namespace, which is what gets replaced here, leaving each
    revision's own call sites and control flow intact.
    """
    import importlib

    retriever = importlib.import_module("konte.stores.retriever")

    def local(query: str) -> list[str]:
        return [token for token in query.replace("?", " ").split() if len(token) > 1]

    async def local_async(query: str) -> list[str]:
        return local(query)

    retriever.extract_search_keywords = local
    if hasattr(retriever, "extract_search_keywords_async"):
        retriever.extract_search_keywords_async = local_async


class _CountingHandle:
    """File proxy that tallies everything written through it."""

    def __init__(self, handle, tally: list[int]):
        self._handle = handle
        self._tally = tally

    def write(self, data):
        self._tally[0] += len(data.encode()) if isinstance(data, str) else len(data)
        return self._handle.write(data)

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc):
        return self._handle.__exit__(*exc)


@contextmanager
def counting_writes(match: str) -> Iterator[list[int]]:
    """Tally bytes written to paths containing ``match``, for the block.

    Total bytes *written* is the interesting quantity for a file that is
    rewritten repeatedly - its final size says nothing about what the build
    cost. The two revisions write through different calls (one replaces the
    file, one appends to an open handle), so both routes are counted.
    """
    original_write_text = Path.write_text
    original_open = Path.open
    tally = [0]

    def write_text(self, data, *args, **kwargs):
        if match in str(self):
            tally[0] += len(data.encode()) if isinstance(data, str) else len(data)
        return original_write_text(self, data, *args, **kwargs)

    def opener(self, mode="r", *args, **kwargs):
        handle = original_open(self, mode, *args, **kwargs)
        if match in str(self) and any(flag in mode for flag in ("w", "a", "+")):
            return _CountingHandle(handle, tally)
        return handle

    Path.write_text = write_text
    Path.open = opener
    try:
        yield tally
    finally:
        Path.write_text = original_write_text
        Path.open = original_open


@contextmanager
def stub_embeddings(dim: int = 1536, latency: float = 0.0) -> Iterator[None]:
    """Answer embedding requests locally, deterministically, for the block.

    The vector-search cases measure how matching ids are selected, which is
    tens of microseconds to tens of milliseconds. A real embedding round trip
    is hundreds of milliseconds with a long tail, so leaving it inside the
    timed region would bury the effect under network variance. The vector's
    content does not matter to what is being measured - only that both
    revisions search with the same one.

    Patched on the class, before the index is loaded, so a vector store that
    captures the embeddings object still sees the stub.
    """
    import hashlib

    from langchain_openai import OpenAIEmbeddings

    def vector_for(text: str) -> list[float]:
        seed = hashlib.blake2b(text.encode(), digest_size=8).digest()
        rng = random.Random(int.from_bytes(seed, "big"))
        return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

    originals = {
        name: getattr(OpenAIEmbeddings, name)
        for name in ("embed_query", "embed_documents", "aembed_documents")
        if hasattr(OpenAIEmbeddings, name)
    }

    def embed_documents(self, texts):
        if latency:
            time.sleep(latency)
        return [vector_for(text) for text in texts]

    async def aembed_documents(self, texts):
        # asyncio.sleep, not time.sleep: a blocking sleep here would serialize
        # the very overlap the index-build case exists to measure.
        if latency:
            import asyncio

            await asyncio.sleep(latency)
        return [vector_for(text) for text in texts]

    OpenAIEmbeddings.embed_query = lambda self, text: vector_for(text)
    OpenAIEmbeddings.embed_documents = embed_documents
    if "aembed_documents" in originals:
        OpenAIEmbeddings.aembed_documents = aembed_documents
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(OpenAIEmbeddings, name, original)


def capabilities() -> dict[str, bool]:
    """Report which of the newer APIs the loaded revision actually has.

    An absent feature is a result, not a crash: the baseline predates several
    of the modules and entry points under test.
    """
    import importlib

    def has_module(name: str) -> bool:
        try:
            importlib.import_module(name)
        except ImportError:
            return False
        return True

    def has_attr(module: str, attr: str) -> bool:
        try:
            return hasattr(importlib.import_module(module), attr)
        except ImportError:
            return False

    return {
        "project_cache": has_module("konte.cache"),
        "atomic_storage": has_module("konte.storage"),
        "checkpoint_log": has_module("konte.checkpoint"),
        "llm_factory": has_module("konte.llm"),
        "retrieve_async": has_attr("konte.stores.retriever", "Retriever")
        and hasattr(
            importlib.import_module("konte.stores.retriever").Retriever,
            "retrieve_async",
        ),
        "async_faiss_build": has_attr("konte.stores.faiss_store", "FAISSStore")
        and hasattr(
            importlib.import_module("konte.stores.faiss_store").FAISSStore,
            "abuild_index",
        ),
        "keyword_cache": has_attr("konte", "clear_keyword_cache"),
        "shared_project": has_attr("konte", "get_shared_project"),
    }


# --------------------------------------------------------------------------
# measurement primitives
# --------------------------------------------------------------------------


@contextmanager
def timed(sink: dict, key: str) -> Iterator[None]:
    """Record the wall-clock seconds a block takes into ``sink[key]``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = time.perf_counter() - start


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB.

    ``ru_maxrss`` is bytes on macOS and kilobytes on Linux.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def dir_bytes(path: Path) -> int:
    """Total size of every file under a directory."""
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def dir_breakdown(path: Path) -> dict[str, int]:
    """Per-file sizes under a directory, largest first."""
    sizes = {p.name: p.stat().st_size for p in Path(path).iterdir() if p.is_file()}
    return dict(sorted(sizes.items(), key=lambda kv: kv[1], reverse=True))


@contextmanager
def counting(target: object, attr: str) -> Iterator[list[int]]:
    """Count calls to ``target.attr`` for the duration of the block.

    Yields a one-element list holding the running count, so a case can read the
    total after the block closes.
    """
    original = getattr(target, attr)
    tally = [0]

    def wrapper(*args, **kwargs):
        tally[0] += 1
        return original(*args, **kwargs)

    setattr(target, attr, wrapper)
    try:
        yield tally
    finally:
        setattr(target, attr, original)


def repeat(fn: Callable[[], object], trials: int) -> list[float]:
    """Time ``fn`` ``trials`` times and return the individual durations."""
    durations = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return durations


def summarize(durations: Sequence[float], scale: float = 1000.0) -> dict:
    """Reduce a sample of durations to the shape the report needs.

    Median and IQR rather than mean and standard deviation: latency against a
    live endpoint is long-tailed, and a mean is dragged around by the tail.
    """
    if not durations:
        return {"n": 0}
    values = sorted(d * scale for d in durations)
    quantiles = statistics.quantiles(values, n=4) if len(values) > 3 else None
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": values[0],
        "max": values[-1],
        "p95": values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))],
        "iqr": (quantiles[2] - quantiles[0]) if quantiles else None,
    }
