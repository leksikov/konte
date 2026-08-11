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

from benchmarks.harness import Context, capabilities, isolate_revision, scratch_root


def _assert_revision(root: Path) -> str:
    """Fail loudly if konte did not come from the tree this process selected."""
    import konte

    resolved = Path(konte.__file__).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError(
            f"revision isolation failed: konte loaded from {resolved}, expected a "
            f"path under {root.resolve()}"
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
        payload["measurements"] = module.run(ctx)
        payload["status"] = "ok"
    except Exception:
        payload["status"] = "error"
        payload["traceback"] = traceback.format_exc()

    # After the case, so a case measuring cold import is not polluted by the
    # probe importing konte first.
    try:
        payload["konte_path"] = _assert_revision(ctx.root)
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
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
