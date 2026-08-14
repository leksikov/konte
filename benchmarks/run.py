"""Run one benchmark case inside an already-selected revision.

This module is the subprocess entry point. By the time it executes, the working
directory has decided which konte ``import konte`` will find, so the case can
import konte freely.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import traceback
from pathlib import Path

from benchmarks.harness import (
    Context,
    capabilities,
    isolate_revision,
    revision_root,
    scratch_root,
)


def _assert_revision(root: Path, revision: str) -> str:
    """Fail loudly if konte did not come from the revision this run asked for.

    Checked against the requested revision's tree, not just against the working
    directory. Revision selection *is* the working directory, so comparing the
    two to each other always agrees - and a run launched by hand from the repo
    root with `--revision baseline` would measure head twice and say baseline.
    """
    import konte

    resolved = Path(konte.__file__).resolve()
    expected = revision_root(revision).resolve()
    if expected not in resolved.parents:
        raise RuntimeError(
            f"revision isolation failed: konte loaded from {resolved}, expected a path "
            f"under {expected}. Run cases through `benchmarks.compare`, or from within "
            f"the revision's own tree - the working directory is what selects it."
        )
    if root.resolve() != expected:
        raise RuntimeError(
            f"working directory {root.resolve()} is not the {revision} tree at {expected}"
        )
    return str(resolved)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="module name under benchmarks.cases")
    parser.add_argument("--revision", required=True, choices=("baseline", "head"))
    parser.add_argument("--out", type=Path, help="write the result JSON here")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="case-specific option, repeatable",
    )
    args = parser.parse_args(argv)

    dropped = isolate_revision()
    options = dict(item.partition("=")[::2] for item in args.option)
    ctx = Context(
        revision=args.revision,
        root=Path.cwd(),
        scratch=scratch_root(),
        options=options,
    )

    payload: dict = {
        "case": args.case,
        "revision": args.revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "editable_finders_dropped": dropped,
    }
    try:
        module = importlib.import_module(f"benchmarks.cases.{args.case}")
        measurements = module.run(ctx)
        payload["measurements"] = measurements
        # A case that cannot find the corpus or project it needs says so and
        # returns nothing to measure. Reporting that as ok would let the report
        # print "both revisions ok" over a run that measured nothing.
        if isinstance(measurements, dict) and measurements.get("status") == "skipped":
            payload["status"] = "skipped"
            payload["reason"] = measurements.get("reason")
        else:
            payload["status"] = "ok"
    except Exception:
        payload["status"] = "error"
        payload["traceback"] = traceback.format_exc()

    # After the case, so a case measuring cold import is not polluted by the
    # probe importing konte first.
    try:
        payload["konte_path"] = _assert_revision(ctx.root, args.revision)
        payload["capabilities"] = capabilities()
    except Exception:
        payload["status"] = "error"
        payload.setdefault("traceback", traceback.format_exc())
        payload["konte_path"] = None

    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    return 0 if payload["status"] in ("ok", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
