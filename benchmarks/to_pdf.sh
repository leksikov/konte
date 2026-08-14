#!/usr/bin/env bash
# Render the results report to a PDF suitable for sending to a reviewer.
#
# pandoc for Markdown -> HTML, Chrome for HTML -> PDF. Chrome rather than a
# LaTeX engine because the report contains Korean throughout, and Chrome picks
# up the system CJK fonts without a font-configuration step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT/benchmarks/RESULTS.md}"
OUT="${2:-$ROOT/benchmarks/results/konte-performance-before-after.pdf}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The revisions the stored results were produced by, not whatever the working
# tree and environment say now: rendering an older report, or rendering after
# the baseline ref moved, would otherwise stamp measurements with two commits
# that never ran them.
read -r BASELINE_REF HEAD_REF <<<"$(
  "$ROOT/.venv/bin/python" - "$ROOT/benchmarks/results" <<'PY'
import json
import sys
from pathlib import Path

pair = ("unknown", "unknown")
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    try:
        revisions = json.loads(path.read_text()).get("revisions") or {}
    except (json.JSONDecodeError, OSError):
        continue
    if revisions.get("baseline") and revisions.get("head"):
        pair = (revisions["baseline"][:12], revisions["head"][:12])
        break
print(*pair)
PY
)"
GENERATED="$(date '+%Y-%m-%d %H:%M')"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# A title block, so the document states what it compared without needing the
# conversation it came from.
cat > "$work/doc.md" <<EOF
---
title: "konte: performance before and after"
---

<div class="frontmatter">

**Compared:** \`$BASELINE_REF\` (before) against \`$HEAD_REF\` (after)  ·
**Generated:** $GENERATED  ·
**Host:** $(uname -s) $(uname -m), Python $(cd "$ROOT" && .venv/bin/python -V 2>&1 | cut -d' ' -f2)

Both revisions run from one virtualenv, selected per subprocess by working
directory; the dependency set is identical across the range, so konte's source
is the only variable. Timings are medians. Every number below is reproducible
with \`uv run python -m benchmarks.compare\`.

</div>

EOF
tail -n +2 "$SRC" >> "$work/doc.md"

cat > "$work/style.css" <<'EOF'
@page { size: A4 landscape; margin: 14mm 12mm; }
body {
  font-family: -apple-system, "Helvetica Neue", "Apple SD Gothic Neo",
               "Noto Sans KR", sans-serif;
  font-size: 9.5pt; line-height: 1.45; color: #1a1a1a; max-width: none;
}
h1 { font-size: 20pt; margin: 0 0 4pt; border-bottom: 2px solid #333; padding-bottom: 4pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #bbb;
     padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
.frontmatter { font-size: 9pt; color: #444; background: #f6f6f6;
               border-left: 3px solid #999; padding: 8pt 10pt; margin-bottom: 12pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt;
        font-size: 8.5pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3pt 5pt; text-align: left;
         vertical-align: top; word-break: break-word; }
th { background: #ececec; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafafa; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.5pt;
       background: #f0f0f0; padding: 0 2px; border-radius: 2px; }
blockquote { border-left: 3px solid #ccc; margin: 4pt 0; padding: 2pt 10pt;
             color: #333; background: #fbfbfb; }
strong { color: #000; }
ul { margin: 4pt 0; padding-left: 16pt; }
li { margin-bottom: 3pt; }
EOF

pandoc "$work/doc.md" \
  --from=gfm --to=html5 --standalone \
  --css=style.css --embed-resources \
  --resource-path="$work" \
  --output="$work/doc.html"

mkdir -p "$(dirname "$OUT")"
"$CHROME" --headless --disable-gpu --no-sandbox \
  --no-pdf-header-footer --print-to-pdf-no-header \
  --print-to-pdf="$OUT" "file://$work/doc.html" 2>/dev/null

echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
