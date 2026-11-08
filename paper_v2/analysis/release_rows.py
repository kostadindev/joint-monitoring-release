"""Scored allocation rows read directly from the tracked releases in results/keep/.

The analysis catalog (results/graph_catalog_v1/graph_runs.sqlite, about 800 MB) is not
in the repository. The paper's tables use only two of its datasets, and both are stored
row for row in tracked releases, so a fresh clone can regenerate the tables from these:

  topology_workload_development_v0_1  analysis/metrics.jsonl.gz        (13-channel workflow)
  topology_causal_development_v0_1    audit/monitor/per_record_metrics.json  (workflow shapes)

`rows(dataset)` yields the metric dictionaries that the catalog stores in
benchmark_rows.metrics_zlib.
"""
import collections, gzip, json
from pathlib import Path

KEEP = Path(__file__).resolve().parents[2] / "results/keep"
SOURCES = {
    "topology_workload_development_v0_1": "analysis/metrics.jsonl.gz",
    "topology_causal_development_v0_1": "audit/monitor/per_record_metrics.json",
}


def rows(dataset):
    path = KEEP / dataset / SOURCES[dataset]
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            for line in f:
                yield json.loads(line)
    else:
        yield from json.loads(path.read_text())


def columns(m):
    """The catalog's benchmark_rows columns for one metric row (as in scripts/build_graph_catalog.py)."""
    return dict(selector=m.get("selector"), grouper=m.get("grouper"), k=m.get("k"), s=m.get("s"),
                seed=m.get("seed"), input_byte_allowance=m.get("input_byte_allowance"),
                condition=m.get("condition"), environment=m.get("environment"),
                status=m.get("status", m.get("evaluation_status")),
                tp=m.get("tp", m.get("source_tp")), fp=m.get("fp", m.get("source_fp")), n_truth=m.get("n_truth"),
                known_negative=m.get("selected_known_benign", m.get("inspected_negative")))


def catalog_path(root):
    """The development catalog if JMON_CATALOG names one; otherwise None, so callers read
    the released data in results/keep/ (the default for anyone reproducing the paper)."""
    import os
    db = os.environ.get("JMON_CATALOG")
    return Path(db) if db and Path(db).exists() else None


def _pct(a, b):
    return None if not b else round(100 * a / b, 1)


def native_panel(metrics):
    """Per-policy summary of the workflow-shape panel (Table 1's workflow-shape rows).
    `metrics` are rows of topology_causal_development_v0_1."""
    N = collections.defaultdict(lambda: collections.defaultdict(float))
    for m in metrics:
        a = N[(m["selector"], m["grouper"], m["k"], m["s"], m["environment"])]
        cond = m["condition"]
        a[cond + ":TP"] += m.get("tp") or 0
        a[cond + ":NT"] += m.get("n_truth") or 0
        a[cond + ":FP"] += m.get("fp") or 0
        a[cond + ":NB"] += m.get("selected_known_benign") or 0
        a[cond + ":any"] += m.get("source_any_recovery") or 0
        a[cond + ":both"] += m.get("source_both_recovery") or 0
        a[cond + ":n"] += 1
        a[cond + ":calls"] += m.get("judge_calls") or 0
        a[cond + ":tok"] += m.get("input_tokens") or 0
    nat = []
    for key in sorted(N):
        a = N[key]
        sel, grp, k, s, env = key
        nat.append(dict(selector=sel, grouper=grp, k=k, s=s, shape=env,
                        recall_AB=_pct(a["combined:TP"], a["combined:NT"]),
                        any_AB=_pct(a["combined:any"], a["combined:n"]), both_AB=_pct(a["combined:both"], a["combined:n"]),
                        recall_single_origin=_pct(a["a_only:TP"] + a["b_only:TP"], a["a_only:NT"] + a["b_only:NT"]),
                        honest_twin_flagged=_pct(a["honest_error_combined:TP"], a["honest_error_combined:NT"]),
                        fpr_AB=_pct(a["combined:FP"], a["combined:NB"]), fpr_clean=_pct(a["clean:FP"], a["clean:NB"]),
                        calls=round(a["combined:calls"] / a["combined:n"], 2), tokens=round(a["combined:tok"] / a["combined:n"])))
    return nat
