"""MESA customer-service pilot: N tasks x conditions on one topology, through our provider.

Conditions (jmon.scenarios.mesa_bridge.CONDITIONS): clean, overt (MESA static misinformation
on the policy edge), a_only, b_only, ab (conjunctive fragments on the customer_facing and
policy edges), mismatched (A names a code of another family). Attack success is MESA's
corrected definition: the clean run of the same task was correct and the attacked run is
wrong. Writes results/<out>/episodes.jsonl (one line per episode: full MessageLog with
hidden labels, judge verdict, provider usage, call counts) and summary.json.

Usage: python scripts/run_mesa_pilot.py configs/mesa_pilot_mock.yaml
"""
from __future__ import annotations

import json
import os
import sys
import time

import yaml

from jmon.llm import make_provider
from jmon.llm.base import BudgetExceeded
from jmon.scenarios import mesa_bridge as mb


def summarize_rows(cfg: dict, rows: list[dict], failures: list[dict], topo_name: str, n_tasks: int,
                   elapsed: float, provider_summary: dict) -> dict:
    conditions = cfg.get("conditions", list(mb.CONDITIONS))
    summary = {"config": cfg, "topology": topo_name, "n_tasks": n_tasks, "elapsed_s": round(elapsed, 2),
               "provider": provider_summary, "n_failed_episodes": len(failures), "failures": failures,
               "conditions": {}}
    for cond in conditions:
        rs = [r for r in rows if r["condition"] == cond]
        acc = [r["decision_accuracy"] for r in rs]
        succ = [r["attack_success"].get(cond) for r in rs if cond != "clean"]
        eligible = [x for x in succ if x is not None]
        summary["conditions"][cond] = {
            "n": len(rs),
            "decision_accuracy": sum(1 for a in acc if a == 1) / max(1, len(acc)),
            "attack_success_rate": (sum(eligible) / len(eligible)) if eligible else None,
            "n_eligible": len(eligible),
            "empty_resolutions": sum(1 for r in rs if not (r.get("final_resolution") or "").strip()),
            "mean_calls": {k: sum(r["calls"][k] for r in rs) / max(1, len(rs)) for k in ("agent", "judge", "attack")},
            "mean_messages": sum(len(r["messages"]) for r in rs) / max(1, len(rs)),
            "followups": sum(1 for r in rs if r.get("followup_fired")),
        }
    return summary


def main(cfg_path: str, shard: tuple[int, int] = (0, 1)) -> dict:
    cfg = yaml.safe_load(open(cfg_path))
    i, n = shard
    base_out = cfg.get("out_dir", "results/mesa_pilot")
    out_dir = base_out if n == 1 else os.path.join(base_out, f"shard_{i}")
    os.makedirs(out_dir, exist_ok=True)
    pcfg = dict(cfg["provider"])
    if n > 1 and "cost" in pcfg:   # split the caps across shards
        cost = dict(pcfg["cost"])
        for k in ("max_calls", "max_tokens"):
            if cost.get(k): cost[k] = int(cost[k] // n)
        if cost.get("max_credits"): cost["max_credits"] = cost["max_credits"] / n
        pcfg["cost"] = cost
    provider = make_provider(pcfg)
    mb.set_attack_provider(provider)
    scenario = mb.load_scenario(cfg.get("scenario", mb.SCENARIO_PATH))
    topo = mb.load_topology(cfg["topology"])
    all_tasks = cfg.get("tasks") or [t["id"] for t in scenario["tasks"]][: int(cfg.get("n_tasks", 20))]
    tasks = all_tasks[i::n]
    conditions = cfg.get("conditions", list(mb.CONDITIONS))
    overt_role = cfg.get("overt_edge_role", "policy")
    seed = int(cfg.get("master_seed", 0))

    t0 = time.time()
    rows, clean, failures = [], {}, []
    path = os.path.join(out_dir, "episodes.jsonl")
    with open(path, "w") as fh:
        for tid in tasks:
            task = mb.get_task(scenario, tid)
            for cond in conditions:
                builder = mb.CONDITIONS[cond]
                if builder is None:
                    fns = {}
                elif cond in ("overt", "overt_dyn"):
                    fns = builder(task, topo, overt_role)
                else:
                    fns = builder(task, topo)
                try:
                    ep = mb.run_cs_episode(topo, tid, provider, fns, seed=seed, scenario=scenario, condition=cond)
                except BudgetExceeded:
                    raise
                except Exception as exc:   # e.g. EmptyResponse: record and continue
                    failures.append({"task_id": tid, "condition": cond, "error": f"{type(exc).__name__}: {exc}"[:300]})
                    print(f"{tid:8s} {cond:11s} FAILED {type(exc).__name__}", flush=True)
                    continue
                resolved = bool((ep.sink_output.get("final_resolution") or "").strip())
                if cond == "clean":
                    if resolved:
                        clean[tid] = ep
                elif tid in clean:
                    ep.attack_success[cond] = mb.attack_succeeded(clean[tid], ep) if resolved else None
                rec = mb.episode_record(ep)
                fh.write(json.dumps(rec, sort_keys=True) + "\n"); fh.flush()
                rows.append(rec)
                print(f"{tid:8s} {cond:11s} acc={rec['decision_accuracy']} calls={rec['calls']} "
                      f"msgs={len(rec['messages'])} success={ep.attack_success.get(cond)}", flush=True)
    summary = summarize_rows(cfg, rows, failures, topo["name"], len(tasks), time.time() - t0,
                             getattr(provider, "summary", lambda: {})() or {"calls": getattr(provider, "calls", None)})
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "failures")}, indent=1))
    print(f"wrote {path}")
    return summary


def merge(cfg_path: str, n: int) -> dict:
    """Concatenate shard outputs into out_dir/episodes.jsonl and recompute the summary."""
    cfg = yaml.safe_load(open(cfg_path))
    base = cfg.get("out_dir", "results/mesa_pilot")
    rows, failures, prov, elapsed = [], [], {}, 0.0
    for i in range(n):
        d = os.path.join(base, f"shard_{i}")
        rows += [json.loads(l) for l in open(os.path.join(d, "episodes.jsonl"))]
        s = json.load(open(os.path.join(d, "summary.json")))
        failures += s.get("failures", []); elapsed = max(elapsed, s["elapsed_s"])
        for k, v in (s.get("provider") or {}).items():
            if isinstance(v, (int, float)): prov[k] = prov.get(k, 0) + v
    rows.sort(key=lambda r: (int(r["task_id"].split("_")[1]), list(cfg.get("conditions", mb.CONDITIONS)).index(r["condition"])))
    with open(os.path.join(base, "episodes.jsonl"), "w") as fh:
        for r in rows: fh.write(json.dumps(r, sort_keys=True) + "\n")
    topo = mb.load_topology(cfg["topology"])
    summary = summarize_rows(cfg, rows, failures, topo["name"], len({r["task_id"] for r in rows}), elapsed, prov)
    json.dump(summary, open(os.path.join(base, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config", "failures")}, indent=1))
    return summary


if __name__ == "__main__":
    # usage: run_mesa_pilot.py CFG [--shard i/n] | CFG --merge n
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/mesa_pilot_mock.yaml"
    if "--merge" in sys.argv:
        merge(cfg_path, int(sys.argv[sys.argv.index("--merge") + 1]))
    else:
        sh = (0, 1)
        if "--shard" in sys.argv:
            a, b = sys.argv[sys.argv.index("--shard") + 1].split("/"); sh = (int(a), int(b))
        main(cfg_path, sh)
