"""Rotation as a mixed strategy: reads Codex's completed topology-only grid
(research/completed_grid_review/full_grid.csv) and treats the aligned and shuffled
environments as two moves of an adversary who decides whether the corroborating
record travels on the graph path the defender expects. For each topology-only
strategy (selector x grouper) at a fixed budget it reports recall under both moves,
the worst case, and the maximin mixture (with two adversary moves an optimal mixture
needs at most two strategies, so pairs are enumerated exactly). Oracles are shown as
references and excluded from the mixture because they use attack labels.
Also writes the exact co-location floor of uniform random partitions."""
import csv, itertools, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "research/completed_grid_review/full_grid.csv"
OUT = Path(__file__).resolve().parent
ORACLES = {"target_location_oracle", "reference_location_oracle"}


def load(k, s, size="16", allowance="24000"):
    rows = [r for r in csv.DictReader(open(GRID)) if r["workload_size"] == size and r["k"] == str(k)
            and r["s"] == str(s) and r["condition"] == "false_report" and r["input_byte_allowance"] == allowance
            and r["source_recall"] != ""]
    pay = {}
    for r in rows:
        pay.setdefault((r["selector"], r["grouper"]), {})[r["environment"]] = float(r["source_recall"])
    return {k_: v for k_, v in pay.items() if set(v) == {"aligned", "shuffled"}}


def maximin(pay):
    best = None
    names = [n for n in pay if n[0] not in ORACLES]
    for a in names:  # pure strategies
        v = min(pay[a].values())
        if best is None or v > best[0]:
            best = (v, {a: 1.0})
    for a, b in itertools.combinations(names, 2):  # two-strategy mixtures
        (a1, a2), (b1, b2) = (pay[a]["aligned"], pay[a]["shuffled"]), (pay[b]["aligned"], pay[b]["shuffled"])
        den = (a1 - a2) - (b1 - b2)
        if abs(den) < 1e-12:
            continue
        p = (b2 - b1) / den  # p*a + (1-p)*b equalizes the two moves
        if 0 < p < 1:
            v = p * a1 + (1 - p) * b1
            if v > best[0] + 1e-12:
                best = (v, {a: p, b: 1 - p})
    return best


def colocation_floor(sizes):
    k = sum(sizes)
    return sum(math.comb(g, 2) for g in sizes) / math.comb(k, 2)


res = {}
for k, s in [(7, 2), (4, 2), (7, 4)]:
    pay = load(k, s)
    if not pay:
        continue
    v, mix = maximin(pay)
    res[f"k{k}_s{s}"] = {"strategies": {f"{a}/{b}": p for (a, b), p in sorted(pay.items())},
                         "maximin_value": v, "maximin_mixture": {f"{a}/{b}": w for (a, b), w in mix.items()}}
res["colocation_floor_uniform_partition_sizes_1_2_2_2"] = colocation_floor([1, 2, 2, 2])
json.dump(res, open(OUT / "rotation_game.json", "w"), indent=1)

r = res["k7_s2"]
print("k=7,s=2 strategies (aligned, shuffled, worst):")
for n, p in sorted(r["strategies"].items(), key=lambda x: -min(x[1].values())):
    print(f"  {n:55s} {100*p['aligned']:6.2f} {100*p['shuffled']:6.2f} {100*min(p.values()):6.2f}")
print("maximin", round(100 * r["maximin_value"], 2), {n: round(w, 3) for n, w in r["maximin_mixture"].items()})
for key in ("k4_s2", "k7_s4"):
    if key in res:
        print(key, "maximin", round(100 * res[key]["maximin_value"], 2), {n: round(w, 3) for n, w in res[key]["maximin_mixture"].items()})
print("co-location floor, uniform partition [1,2,2,2]:", res["colocation_floor_uniform_partition_sizes_1_2_2_2"])

# Robustness: the same game using only exact expectations for random grouping
# (no three-seed realizations): directed-path grouping vs exact matched-random grouping
# on the same structural acquisition.
pay = load(7, 2)
sub = {n: pay[n] for n in [("structural_path_pairs", "directed_path"), ("structural_path_pairs", "exact_matched_random")]}
v, mix = maximin(sub)
res["k7_s2_exact_only"] = {"maximin_value": v, "maximin_mixture": {f"{a}/{b}": w for (a, b), w in mix.items()}}
json.dump(res, open(OUT / "rotation_game.json", "w"), indent=1)
print("exact-only maximin", round(100 * v, 2), {f"{a}/{b}": round(w, 3) for (a, b), w in mix.items()})
