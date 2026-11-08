"""Table 1: every complete strategy at a fixed monitoring budget k = 7, varying the
group size s. Reads the scored rows of results/keep/topology_workload_development_v0_1 (or, if
present, the same rows in the local analysis catalog) and rotation_game.json; writes table_k7.json and
table_k7.tex next to this file. No model calls.

Cells are recall of false-origin channels (%) in the aligned / shuffled environments,
sixteen-case workload, 72 false-report runs per environment."""
import collections, json, os, sqlite3, zlib
from pathlib import Path

from release_rows import catalog_path, columns, rows as release_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DS = "topology_workload_development_v0_1"
db = catalog_path(ROOT)


def benchmark_rows():
    """From the local catalog if present, else from the tracked release (identical rows)."""
    if db is None:
        for m in release_rows(DS):
            c = columns(m)
            yield (c["selector"], c["grouper"], c["k"], c["s"], c["condition"], c["environment"], c["status"],
                   c["tp"], c["fp"], c["n_truth"], c["known_negative"], m)
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    for *r, mz in con.execute(
            "select selector,grouper,k,s,condition,environment,status,tp,fp,n_truth,known_negative,metrics_zlib "
            "from benchmark_rows where dataset_id=?", (DS,)):
        yield (*r, json.loads(zlib.decompress(mz)))


A = collections.defaultdict(lambda: collections.defaultdict(float))
for sel, grp, k, s, cond, env, st, tp, fp, nt, kn, m in benchmark_rows():
    if m.get("workload_size") != 16 or st != "scored":
        continue
    a = A[(sel, grp, k, s, m.get("input_byte_allowance"))]
    if cond == "false_report":
        a[env + "tp"] += tp or 0; a[env + "nt"] += nt or 0
    elif cond == "clean":
        a["fp"] += fp or 0; a["kn"] += kn or 0

def fpr(sel, grp, k, s, allow=24000):
    a = A.get((sel, grp, k, s, allow))
    return None if not a or not a["kn"] else round(100 * a["fp"] / a["kn"], 1)

def cell(sel, grp, k, s, allow=24000):
    a = A.get((sel, grp, k, s, allow))
    if not a or not a["alignednt"]:
        return None
    return (round(100 * a["alignedtp"] / a["alignednt"], 1), round(100 * a["shuffledtp"] / a["shufflednt"], 1))

SEL = [("structural_path_pairs", "Path pairs"), ("random", "Random$^\\dagger$"), ("degree", "Degree"),
       ("downstream_coverage", "Downstream coverage"), ("pagerank", "PageRank"), ("betweenness", "Betweenness"),
       ("mesa_static", "edge ranking")]
GRP = [("directed_path", "Directed path"), ("spectral", "Spectral clusters"),
       ("random_matched_sizes", "Random groups$^\\dagger$"), ("line_community", "Line-graph communities"),
       ("shared_receiver", "Shared receiver")]
rows = []
for sel, sname in SEL:
    block = {"sel_key": sel, "selection": sname, "s1": cell(sel, "singleton", 7, 1),
             "s7": cell(sel, "full_selected_relaxed", 7, 7, 60000), "groupings": []}
    for grp, gname in GRP:
        block["groupings"].append({"grouping": gname, "s2": cell(sel, grp, 7, 2), "s4": cell(sel, grp, 7, 4)})
    if sel == "structural_path_pairs":
        block["groupings"].insert(3, {"grouping": "Random groups, exact expectation",
                                      "s2": cell(sel, "exact_matched_random", 7, 2), "s4": None})
    rows.append(block)
learned = cell("learned_pairs", "learned_pairs", 7, 2)
R = json.load(open(HERE / "rotation_game.json"))
refs = {"rotation": (round(100 * R["k7_s2"]["maximin_value"], 1), round(100 * R["k7_s4"]["maximin_value"], 1)),
        "location_oracle": cell("target_location_oracle", "diagnostic", 1, 1),
        "location_evidence_oracle": cell("reference_location_oracle", "diagnostic", 2, 2),
        "all13": cell("all_edges", "full_selected_relaxed", 13, 13, 60000), "learned_pairs_s2": learned}
json.dump({"rows": rows, "refs": refs}, open(HERE / "table_k7.json", "w"), indent=1)

FPR = [fpr(sel, g, 7, sz) for sel, _ in SEL for g, _ in GRP for sz in (2, 4)]
FPR = [x for x in FPR if x is not None]
DIR = [fpr(sel, "directed_path", 7, sz) for sel in ("structural_path_pairs", "random") for sz in (2, 4)]
json.dump({"rows": rows, "refs": refs, "clean_fpr_range_s2_s4": [min(FPR), max(FPR)], "clean_fpr_directed_path": DIR},
          open(HERE / "table_k7.json", "w"), indent=1)

ZERO = ("pagerank", "betweenness", "mesa_static")
def allzero(sel):
    b = next(b for b in rows if b["sel_key"] == sel)
    vals = [b["s1"], b["s7"]] + [g[c] for g in b["groupings"] for c in ("s2", "s4")]
    return all(v is None or v == (0.0, 0.0) for v in vals)
cols = {}
for b in rows:
    if b["sel_key"] in ZERO:
        continue
    for g in b["groupings"]:
        for c in ("s2", "s4"):
            if g[c]:
                cols.setdefault(c, [[], []]); cols[c][0].append(g[c][0]); cols[c][1].append(g[c][1])
    if b["s7"]:
        cols.setdefault("s7", [[], []]); cols["s7"][0].append(b["s7"][0]); cols["s7"][1].append(b["s7"][1])
best = {c: (max(v[0]), max(v[1])) for c, v in cols.items()}
def num(x, c, j):
    t = f"{x:.1f}"
    return f"\\textbf{{{t}}}" if c in best and x == best[c][j] and x > 0 else t
def pair(v, c):
    return "-- & --" if v is None else f"{num(v[0], c, 0)} & {num(v[1], c, 1)}"
L = [r"\begin{table}[H]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
     r"\caption{\textbf{Recall of false-origin channels (\%) on the 13-channel workflow at a fixed monitoring budget of $k{=}7$ channels, varying the group size $s$.} Each row is a complete strategy (selection rule and grouping rule); Al.\ and Sh.\ are the aligned and shuffled environments, 24 attacked records each under three allocation seeds. Within one selection rule, grouping alone moves recall from 0\% to 47\%, larger groups help some rules (spectral clusters), and for $s\le 4$ no strategy is best in both environments. Bold: column maximum among allocator strategies. Merged $s{=}1$ and $s{=}7$ cells apply to every grouping in the block; dashes mark configurations not evaluated. Clean false-positive rates over selected channels range from " + f"{min(FPR):.1f}" + r"\% to " + f"{max(FPR):.1f}" + r"\% at $s{=}2$ and $s{=}4$ (pooled over both environments) and are at most " + f"{max(DIR):.1f}" + r"\% for directed-path grouping with random or path-pair selection.}\label{tab:strategies}",
     r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
     r" & & \multicolumn{2}{c}{$s{=}1$} & \multicolumn{2}{c}{$s{=}2$} & \multicolumn{2}{c}{$s{=}4$} & \multicolumn{2}{c}{$s{=}7^{\ddagger}$} \\",
     r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
     r"Selection & Grouping & Al. & Sh. & Al. & Sh. & Al. & Sh. & Al. & Sh. \\", r"\midrule"]
order = ["random", "structural_path_pairs", "degree", "downstream_coverage"]
for key in order:
    b = next(b for b in rows if b["sel_key"] == key)
    n = len(b["groupings"])
    for i, g in enumerate(b["groupings"]):
        lab = f"\\multirow{{{n}}}{{*}}{{{b['selection']}}}" if i == 0 else ""
        s1 = f"\\multirow{{{n}}}{{*}}{{{pair(b['s1'], 's1').split(' & ')[0]}}} & \\multirow{{{n}}}{{*}}{{{pair(b['s1'], 's1').split(' & ')[1]}}}" if i == 0 else " & "
        s7 = f"\\multirow{{{n}}}{{*}}{{{pair(b['s7'], 's7').split(' & ')[0]}}} & \\multirow{{{n}}}{{*}}{{{pair(b['s7'], 's7').split(' & ')[1]}}}" if i == 0 else " & "
        L.append(f"{lab} & {g['grouping']} & {s1} & {pair(g['s2'], 's2')} & {pair(g['s4'], 's4')} & {s7} \\\\")
    L.append(r"\addlinespace[3pt]")
assert all(allzero(z) for z in ZERO)
L.append(r"\makecell[l]{PageRank, betweenness,\\ edge ranking} & all five groupings & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\")
L.append(r"\addlinespace[3pt]")
L.append(f"Learned pairs & learned grouping & -- & -- & {pair(learned, 's2')} & -- & -- & -- & -- \\\\")
L.append(r"\midrule")
rot = refs["rotation"]
L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Maximin rotation (Section 4.5)}} & -- & -- & {rot[0]:.1f} & {rot[0]:.1f} & {rot[1]:.1f} & {rot[1]:.1f} & -- & -- \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{6pt}",
      r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
      r"Reference (uses attack labels or a larger budget) & $k$ & $s$ & Al. & Sh. \\", r"\midrule",
      f"Location oracle: attacked channel alone & 1 & 1 & {refs['location_oracle'][0]:.1f} & {refs['location_oracle'][1]:.1f} \\\\",
      f"Location and evidence oracle & 2 & 2 & {refs['location_evidence_oracle'][0]:.1f} & {refs['location_evidence_oracle'][1]:.1f} \\\\",
      f"All channels in one call$^{{\\ddagger}}$ & 13 & 13 & {refs['all13'][0]:.1f} & {refs['all13'][1]:.1f} \\\\",
      r"\bottomrule", r"\end{tabular}", r"\par\vspace{3pt}",
      r"\begin{minipage}{0.97\linewidth}\footnotesize $^\dagger$Random selection and random groups are means over three paired seeds; the exact expectation is computed over all uniform partitions. $^\ddagger$One call over all selected channels exceeded the harness's input limit and was scored with a larger limit.\end{minipage}",
      r"\end{table}"]
(HERE / "table_k7.tex").write_text("\n".join(L) + "\n")
print("wrote table_k7.tex; clean FPR range", min(FPR), max(FPR), "directed", DIR, "best", best)
