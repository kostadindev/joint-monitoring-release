"""Table 1: individual, random and graph-based joint monitoring on every testbed.
Reads results/keep/oracle_allocation_all20_fixed0.5.json (customer service, 20 tasks),
results/situations_smoke/summary.json (constructed decomposed violations),
table_k7.json (13-channel workflow), rotation_game.json and the workflow-shape release
results/keep/topology_causal_development_v0_1. Writes table_summary.tex. No model calls."""
import json, re
from pathlib import Path

import os
from release_rows import native_panel, rows as release_rows
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
def src(rel):
    return ROOT / rel
ORA = json.load(open(src("results/keep/oracle_allocation_all20_fixed0.5.json")))
SIT = json.load(open(src("results/situations_smoke/summary.json")))
K7 = json.load(open(HERE / "table_k7.json"))
NAT = native_panel(release_rows("topology_causal_development_v0_1"))
ROT = json.load(open(HERE / "rotation_game.json"))["k7_s2"]

def sit(strat):
    return sum(v[strat]["tp"] for v in SIT.values()), sum(v[strat]["n_positive"] for v in SIT.values())
def mesa(att):
    d = ORA[att]
    return (d["k4_s1"]["per_allocation"]["c1 | c2 | c3 | c4"]["detected"], d["k4_s2"]["random"],
            d["k4_s4"]["per_allocation"]["c1+c2+c3+c4"]["detected"], d["k1_s1"]["per_allocation"]["c3"]["detected"])
def k7cell(sel, grp, s, allow=None):
    for b in K7["rows"]:
        if b["sel_key"] == sel:
            if grp == "singleton": return b["s1"]
            for g in b["groupings"]:
                if g["grouping"].startswith(grp): return g[s]
def nat(sel, grp, shape):
    return next(r["recall_AB"] for r in NAT if r["selector"] == sel and r["grouper"] == grp and r["shape"] == shape)

rows = []
for att, lab in (("static", "Customer service: templated rewrite"), ("dynamic", "Customer service: dynamic rewrite")):
    ind, rnd, one, alone = mesa(att)
    extra = "$^{c}$" if att == "static" else ""
    rows.append((lab, "4, 2", f"{ind}/20{extra}", f"{rnd:.1f}/20", f"{rnd:.1f}/20$^{{a}}$", f"{one}/20"))
i, n = sit("independent"); s0 = [sit(f"pairing_{j}")[0] for j in range(3)]; st, _ = sit("structural"); fu, _ = sit("full")
rows.append(("Constructed decomposed violations", "4, 2", f"{i}/{n}", f"{sum(s0)/3:.1f}/{n}", f"\\textbf{{{st}/{n}}}", f"{fu}/{n}"))
refs = K7["refs"]["all13"]
ind = k7cell("structural_path_pairs", "singleton", "s1"); exr = k7cell("structural_path_pairs", "Random groups, exact", "s2")
dp = k7cell("structural_path_pairs", "Directed path", "s2")
rows.append(("13-channel workflow: aligned", "7, 2", f"{ind[0]:.1f}\\%", f"{exr[0]:.1f}\\%", f"\\textbf{{{dp[0]:.1f}\\%}}", f"{refs[0]:.1f}\\%$^{{b}}$"))
rows.append(("13-channel workflow: shuffled", "7, 2", f"{ind[1]:.1f}\\%", f"\\textbf{{{exr[1]:.1f}\\%}}", f"{dp[1]:.1f}\\%", f"{refs[1]:.1f}\\%$^{{b}}$"))
rows.append(("\\quad worst case of the two", "7, 2", f"{min(ind):.1f}\\%", f"{min(exr):.1f}\\%",
             f"{min(dp):.1f}\\%", f"{min(refs):.1f}\\%$^{{b}}$"))
BEST = max(min(v.values()) for v in ROT["strategies"].values())
rows.append(("SPAN", "\\quad best fixed combination, worst case", f"{100*BEST:.1f}\\%"))
rows.append(("SPAN", "\\quad fitted mixture, expected worst case", f"{100*ROT['maximin_value']:.1f}\\%"))
for shape, lab in (("flat", "flat"), ("fork_join", "fork/join"), ("sequential", "sequential")):
    v = [nat("structural_path_pairs", g, shape) for g in ("singleton", "exact_uniform_matched_random", "directed_path")]
    full = nat("all_edges", "full_network", shape)
    rows.append((f"Workflow shape: {lab}", "7, 2", f"{v[0]:.1f}\\%", f"{v[1]:.1f}\\%", f"{v[2]:.1f}\\%", f"{full:.1f}\\%"))
L = [r"\begin{table}[!t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
     r"\caption{\textbf{Detection by grouping rule at a fixed budget.} Graph-based grouping wins where the evidence follows the graph (aligned) and loses where it does not (shuffled). $k$ channels inspected, at most $s$ per call; $n$: tasks, violations, or false-origin opportunities across three allocation seeds (not independent episodes). Counts are detections out of $n$, percentages recall of false origins; random entries are averages; bold marks the best budgeted entry. $^{a}$All four channels report to one supervisor, so a graph rule cannot choose among pairings; the entry is their mean. $^{b}$Larger input limit. $^{c}$Any channel read alone; the attacked channel alone catches 2/20.}\label{tab:summary}",
     r"\begin{tabular}{@{}lllllll@{}}", r"\toprule",
     r"Testbed & $k$, $s$ & $n$ & Individual ($s{=}1$) & Random grouping & Graph-based grouping & One call, all channels \\", r"\midrule"]
def nof(label):
    # sample sizes, verified against the stored runs: 20 attacked tasks; 8 violations;
    # 24 attacked records x 3 seeds (13-channel); 4 tasks x 2 origins x 3 seeds (shapes)
    if label.startswith("Customer service"): return "20"
    if label.startswith("Constructed"): return "8"
    if label.startswith("13-channel") or label.startswith("\\quad"): return "72"
    return "24"
def num(c):
    m = re.match(r"(?:\\textbf\{)?([0-9.]+)", c); return float(m.group(1)) if m else -1
for r in rows:
    r = list(r)
    if r[0] == "SPAN":
        L.append(f"{r[1]} & 7, 2 & 72 & \\multicolumn{{3}}{{c}}{{{r[2]}}} & \\\\")
        if "fitted mixture" in r[1]: L.append(r"\addlinespace[2pt]")
        continue
    if "textbf" not in "".join(r[2:5]):
        vals = [num(c) for c in r[2:5]]; best = max(vals)
        for j, v in enumerate(vals):
            if v == best and best > 0:
                c = r[2 + j]; head = re.match(r"[0-9.]+(?:/[0-9]+|\\%)?", c).group(0)
                r[2 + j] = c.replace(head, "\\textbf{" + head + "}", 1)
    L.append(" & ".join(r[:2] + [nof(r[0])] + r[2:]) + r" \\")
    if r[0].startswith("Customer service: dynamic") or r[0].startswith("Constructed"):
        L.append(r"\addlinespace[2pt]")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(HERE / "table_summary.tex").write_text("\n".join(L) + "\n")
print("\n".join(" | ".join(r) for r in rows))
