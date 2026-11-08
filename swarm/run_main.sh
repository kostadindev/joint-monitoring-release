#!/usr/bin/env bash
# Main swarm generation run: 3 families x 2 assurance variants x 4 topology seeds x 6 cases x 5 conditions.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
for assurance in line none; do
  for t in 0 1 2 3; do
    cs=$(python3 -c "print(','.join(str(100*$t+20+i) for i in range(6)))")
    $PY -m swarm.llm_run --assurance $assurance --tseed $t --cseeds $cs --run-cap 1.2 \
        --parallel-runs 4 --threads 6 --out swarm/out/llm_main/${assurance}_t$t > swarm/out/llm_main_${assurance}_t$t.log 2>&1
    echo "done $assurance t$t: $(tail -1 swarm/out/llm_main_${assurance}_t$t.log)"
  done
done
