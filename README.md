# Which Channels Should a Monitor Read Together?

Code and data for **Which Channels Should a Monitor Read Together? Graph-Based Allocation
for Budgeted Monitoring of Multi-Agent Systems** (Kostadin Devedzhiev, Apart Research Swarm
Alignment Hackathon, November 2026). Paper: [`paper_apart/main.pdf`](paper_apart/main.pdf).

A monitor of an LLM multi-agent system cannot read every message at once. We choose, from
the communication graph alone, which channels to inspect (a monitoring budget *k*) and
which to read together in one monitor call (a context budget *s*), and compare individual,
random and graph-based grouping on recorded monitor outputs:

- **F1.** Graph-based joint monitoring beats independent monitoring, and beats random
  grouping when evidence follows the graph.
- **F2.** Mixing and rotating grouping strategies improves worst-case detection across
  evidence routes.

Every model response is stored, so reproducing the paper needs **no API key and no model
calls**.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[test,analysis]'   # Python >= 3.10
bash scripts/reproduce_table1.sh          # recompute Table 1 from the stored monitor outputs
.venv/bin/python -m pytest -q             # deterministic test suite (mock models only)
```

`reproduce_table1.sh` regenerates every cell of Table 1 and checks it against the committed
[`paper_v2/analysis/table_summary.tex`](paper_v2/analysis/table_summary.tex). The two
larger releases carry their own verifiers (byte hashes, replayed metrics, no model calls):

```bash
.venv/bin/python scripts/verify_topology_release.py results/keep/topology_workload_development_v0_1 --replay
.venv/bin/python scripts/build_topology_causal_release.py --verify results/keep/topology_causal_development_v0_1
```

To rebuild the figures and the PDF (needs pandoc and a TeX distribution with latexmk):
`bash scripts/build_paper.sh`.

## Dataset

660 recorded episodes and over 15,000 stored monitor calls across four testbeds
(paper Section 3.2 and Appendix C). All content is simulated with fictitious records.

| Testbed | Size | Location |
|---|---|---|
| Customer service (MESA tasks; templated and dynamic rewrites) | 240 episodes, 20 tasks | episodes: [`results/keep/pilot_2026-11-06/episodes.jsonl.gz`](results/keep/pilot_2026-11-06) (templated), [`results/mesa_pilot_followup_gemma/episodes.jsonl`](results/mesa_pilot_followup_gemma) (dynamic); monitor calls on every channel subset: [`results/mesa_pairings_all20/`](results/mesa_pairings_all20); Table 1 allocations: [`results/keep/oracle_allocation_all20_fixed0.5.json`](results/keep/oracle_allocation_all20_fixed0.5.json) |
| Constructed decomposed violations | 72 episodes (40 test, 32 calibration) | [`results/situations_smoke/`](results/situations_smoke) |
| 13-channel workflow | 288 records, 13,303 monitor calls | [`results/keep/topology_workload_development_v0_1/`](results/keep/topology_workload_development_v0_1) (own README) |
| Workflow-shape panel (flat, sequential, fork/join) | 60 records, 1,415 monitor calls | [`results/keep/topology_causal_development_v0_1/`](results/keep/topology_causal_development_v0_1) (own README) |

**Loading.** Every file is JSON or JSON Lines (some gzipped):

```python
import gzip, json

# Customer service: one line per episode (messages, attack edges, decision, scores)
episodes = [json.loads(l) for l in gzip.open("results/keep/pilot_2026-11-06/episodes.jsonl.gz", "rt")]

# Customer service and decomposed violations: cases, then monitor calls per case
cases = [json.loads(l) for l in open("results/situations_smoke/situations.jsonl")]   # id, family, label, messages, ...
calls = [json.loads(l) for l in open("results/situations_smoke/completed.jsonl")]    # id, rows (one per view), calls

# 13-channel workflow and workflow shapes: what the allocator sees (anonymous graph, k, s)
inputs = [json.loads(l) for l in gzip.open(
    "results/keep/topology_workload_development_v0_1/allocator_inputs.jsonl.gz", "rt")]
```

In the two topology releases, `judge_data/observations.jsonl[.gz]` holds the monitor's
inputs and outputs, `evaluation/` the labels, and `private/generation/` the ground truth
that is hidden from allocators and the monitor (native records and the attacking agents'
intervention prompts). "Private" refers to that separation; the content is synthetic.
Each release README documents the remaining files. Both releases are hashed, so do not
edit them in place.

## Repository layout

```
paper_apart/   submitted PDF and its LaTeX wrapper (build.py converts paper_v2/report.md)
paper_v2/      report.md (paper source), references, analysis/ (Table 1), figures/
results/       the released dataset (see above)
scripts/       experiment runners, release builders and verifiers, reproduce_table1.sh
src/jmon/      library: graph, allocation (selection and grouping rules), monitors, MESA bridge
tests/         deterministic test suite (mock models only)
configs/       run configurations for the customer-service episodes
research/      inputs read by the code (strategy grid, frozen learned-pair models, protocols)
swarm/         additional executed swarm testbed (720 LLM runs; not reported in the paper; see swarm/README.md)
talk/          slides
vendor/        where MESA's code goes if you re-run the customer-service testbed
```

## Re-running the experiments

Re-running needs model access. The paper uses Gemma-4-26B agents and a Gemma-4-31B monitor
through the Adaption Labs API: copy `.env.example` to `.env` and set `ADAPTION_API_KEY`.
The customer-service testbed also needs MESA's code, which is not redistributed; see
[`vendor/README.md`](vendor/README.md). Entry points:

- Customer service: `scripts/run_mesa_pilot.py`, `scripts/run_mesa_monitors.py` (configs in
  `configs/`), then `scripts/compare_mesa_pairings_all20.py` and
  `scripts/oracle_allocation_all20.py`.
- Decomposed violations: `scripts/compare_situations.py`.
- 13-channel workflow: `scripts/run_topology_workload.py`, `scripts/build_topology_release.py`.
- Workflow shapes: `scripts/run_topology_causal_workflows.py`,
  `scripts/run_topology_causal_monitor.py`, `scripts/build_topology_causal_release.py`.

Each runner caps spending and caches every model response.

## Prior work

The customer-service testbed runs MESA's released benchmark, agent runner, topology,
rewrite attacks and grader (Li et al., 2026) without editing its code; we changed the model
backend, applied the rewrites to the policy channel, added two split attacks at MESA's
message hook and replaced oracle interception with an offline monitor. All original
experiment code, newly generated data and analysis were produced during the sprint (6 to 8
November 2026); third-party materials are identified in [`vendor/README.md`](vendor/README.md).

## License

MIT (see [`LICENSE`](LICENSE)). MESA's code and data are not included.
