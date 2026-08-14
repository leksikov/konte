"""Do both revisions return the same results?

The ranking rewrite claims results are unchanged, ties included. That claim
gates every speed number in this suite: a faster ranking that returns something
different is not the same operation, and the comparison would be meaningless.

This case does not time anything. It records the ranking each revision produces
for a fixed query set over the same corpus, and the report compares the two.
Scores are rounded before recording - the NumPy path and the Python path can
differ in the last bits of a float without ranking anything differently, and a
bit-exact comparison would report that as a behaviour change.
"""

from __future__ import annotations

from benchmarks.build import ensure_project, open_project
from benchmarks.corpus import projects_dir, synthetic_document
from benchmarks.harness import Context, pin_keyword_extraction

DEFAULT_CHUNKS = 2000
TOP_K = 20

QUERIES = [
    "classification heading parts accessories",
    "essential character of composite goods",
    "sets put up for retail sale",
    "machinery and mechanical appliances",
    "parts of general application excluded",
    "material or substance mixtures combinations",
    "containers specially shaped or fitted",
    "goods classified under the last heading in numerical order",
]

#: Enough precision to catch a real scoring change, loose enough to ignore
#: float noise between a Python loop and a vectorized reduction.
SCORE_PRECISION = 6


def run(ctx: Context) -> dict:
    size = int(ctx.options.get("chunks", DEFAULT_CHUNKS))
    document = synthetic_document(size)
    storage = projects_dir(f"parity-{size}-{ctx.revision}")
    ensure_project(f"parity_{size}", storage, document, enable_faiss=False, skip_context=True)

    pin_keyword_extraction()
    project = open_project(f"parity_{size}", storage)

    rankings = {}
    for query in QUERIES:
        response = project.query(query, mode="lexical", top_k=TOP_K)
        rankings[query] = [
            {"chunk_id": result.chunk_id, "score": round(result.score, SCORE_PRECISION)}
            for result in response.results
        ]

    return {
        "corpus": {"requested_chunks": size},
        "top_k": TOP_K,
        "rankings": rankings,
    }
