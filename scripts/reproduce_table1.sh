#!/usr/bin/env bash
# Regenerate Table 1 of the paper (paper/analysis/table_summary.tex) from the stored
# monitor outputs in this repository. No model calls, no API keys, no local catalog.
# Run from the repository root:  bash scripts/reproduce_table1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

$PY scripts/oracle_allocation_all20.py >/dev/null          # customer service (results/mesa_pairings_all20)
$PY paper/analysis/rotation_game.py >/dev/null           # maximin rotation (research/completed_grid_review)
$PY paper/analysis/table_k7.py >/dev/null                # 13-channel workflow (results/keep/topology_workload_development_v0_1)
$PY paper/analysis/table_summary.py                      # + constructed violations, workflow shapes

if git diff --quiet -- results/keep/oracle_allocation_all20_fixed0.5.json paper/analysis/*.json paper/analysis/*.tex; then
  echo "Table 1 reproduced: paper/analysis/table_summary.tex is identical to the committed version."
else
  echo "Regenerated files differ from the committed versions:"; git diff --stat -- results/keep/oracle_allocation_all20_fixed0.5.json paper/analysis/*.json paper/analysis/*.tex
  exit 1
fi
