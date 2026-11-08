#!/usr/bin/env bash
# Rebuild the paper's figures and the submitted PDF (paper_apart/main.pdf).
# Needs pandoc and a TeX distribution with latexmk. Run from the repository root:
#   bash scripts/build_paper.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

for f in make_network_big make_gtg1002 make_network_real make_allocator_inputs; do
  $PY "paper_v2/figures/$f.py" >/dev/null
done
(cd paper_apart && $PY build.py)
