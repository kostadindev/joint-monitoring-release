"""Per-fragment analysis (after the third review).

Primary metric: exposure (a judge group holds the false and true versions of the attacked
check, or a channel carrying the false version is flagged). Strata: liar role x evidence
geometry (closest honest contradiction: shared receiver / path-adjacent / distant), pooled to at
least MIN cells' fragments. For each budget:
  - per-policy exposure per stratum and worst case (random policies averaged over seeds),
  - held-out comparison over case-cluster splits and leave-one-environment-out: maximin mixture
    fitted on train vs best fixed policy chosen on train, uniform mixture, and a null mixture of
    seeds of one random policy; worst case on the held-out cases,
  - bootstrap intervals over case clusters for the full-sample worst cases."""
import collections, json, random, sys
import numpy as np
from swarm.analyze import maximin

MIN = 10


def strata_of(r):
    return f"{r['mode']}|{r['geometry']}"


def table(rows, key="exposed"):
    """policy -> stratum -> mean over fragments (random seeds averaged first per fragment)."""
    per = collections.defaultdict(list)
    for r in rows: per[(r["policy"], r["record"], r["mode"], r["geometry"], r.get("check_idx", 0))].append(r[key])
    cell = collections.defaultdict(list)
    for (p, rec, m, g, _), v in per.items(): cell[(p, f"{m}|{g}")].append(np.mean(v))
    return cell


def worst(cell, pols, strata):
    return {p: min(np.mean(cell[(p, s)]) for s in strata) for p in pols}


def run(path, kf, s, key="exposed"):
    rows = [r for r in json.load(open(path)) if r["kfrac"] == kf and r["s"] == s]
    counts = collections.Counter()
    for r in rows:
        if r["seed"] == 0: counts[strata_of(r)] += 1
    npol = len({r["policy"] for r in rows})
    strata = sorted(st for st, n in counts.items() if n / npol >= MIN)
    rows = [r for r in rows if strata_of(r) in strata]
    pols = sorted({r["policy"] for r in rows})
    cell = table(rows, key)
    M = [[np.mean(cell[(p, st)]) for st in strata] for p in pols]
    w = worst(cell, pols, strata)
    v, q = maximin(M)
    print(f"\n=== k={kf}E s={s} metric={key}; strata (>= {MIN} fragments): " + ", ".join(f"{st} ({counts[st]//npol})" for st in strata))
    for p, row in zip(pols, M):
        print(f"  {p:36s} mean {100*np.mean(row):5.1f}%  worst {100*w[p]:5.1f}%  " + " ".join(f"{100*x:4.0f}" for x in row))
    print(f"  in-sample maximin worst {100*v:.1f}%  mix " + ", ".join(f"{p.split('/')[0]}:{x:.2f}" for p, x in zip(pols, q) if x > 0.01))
    # held-out evaluation over case-cluster splits
    clusters = sorted({r["case_cluster"] for r in rows}); rng = random.Random(0)
    res = collections.defaultdict(list)
    def evaluate(train, test):
        tr = [r for r in rows if r["case_cluster"] in train]; te = [r for r in rows if r["case_cluster"] in test]
        ctr, cte = table(tr, key), table(te, key)
        ok = [st for st in strata if all(cte[(p, st)] for p in pols) and all(ctr[(p, st)] for p in pols)]
        if not ok: return None
        Mtr = [[np.mean(ctr[(p, st)]) for st in ok] for p in pols]; Mte = np.array([[np.mean(cte[(p, st)]) for st in ok] for p in pols])
        _, qq = maximin(Mtr)
        best = int(np.argmax([min(r_) for r_ in Mtr]))
        uni = np.ones(len(pols)) / len(pols)
        rnd = [i for i, p in enumerate(pols) if p.startswith("random/directed_path")]
        return {"maximin": min(qq @ Mte), "best_fixed": min(Mte[best]), "uniform": min(uni @ Mte),
                "single_random_policy": min(Mte[rnd[0]]) if rnd else np.nan}
    for _ in range(200):
        sh = clusters[:]; rng.shuffle(sh); h = len(sh) // 2
        out = evaluate(set(sh[:h]), set(sh[h:]))
        if out:
            for k_, x in out.items(): res[k_].append(x)
    if res:
        wins = np.mean(np.array(res["maximin"]) > np.array(res["best_fixed"]))
        print("  held-out (200 case-cluster splits) worst case: " + ", ".join(f"{k_} {100*np.mean(x):.1f}%" for k_, x in res.items())
              + f"; maximin beats best-fixed in {100*wins:.0f}% of splits")
    envs = sorted({r["environment"] for r in rows}); loo = collections.defaultdict(list)
    for e in envs:
        tr = {r["case_cluster"] for r in rows if r["environment"] != e}; te = {r["case_cluster"] for r in rows if r["environment"] == e}
        out = evaluate(tr, te)
        if out:
            for k_, x in out.items(): loo[k_].append(x)
    if loo:
        print("  leave-one-environment-out worst case (mean over envs): " + ", ".join(f"{k_} {100*np.nanmean(x):.1f}%" for k_, x in loo.items()))
    # bootstrap CI of in-sample worst case for best fixed and maximin
    bs = collections.defaultdict(list)
    for _ in range(200):
        samp = [rng.choice(clusters) for _ in clusters]; cnt = collections.Counter(samp)
        rb = [r for r in rows for _ in range(cnt.get(r["case_cluster"], 0))]
        cb = table(rb, key)
        if not all(cb[(p, st)] for p in pols for st in strata): continue
        Mb = np.array([[np.mean(cb[(p, st)]) for st in strata] for p in pols])
        bs["best_fixed"].append(max(Mb.min(axis=1))); bs["maximin_fixed_q"].append(min(q @ Mb))
    for k_, x in bs.items(): print(f"  bootstrap worst case {k_}: {100*np.percentile(x,5):.1f}-{100*np.percentile(x,95):.1f}%")


if __name__ == "__main__":
    path = sys.argv[1]
    for b in sys.argv[2:]:
        kf, s = b.split(":"); run(path, float(kf), int(s), "exposed"); run(path, float(kf), int(s), "origin_flagged")
