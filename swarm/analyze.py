"""Analyse judge results: per policy recall by environment and by attacker placement stratum,
each policy's worst case over strata, and the maximin randomized mixture of policies (LP)."""
import collections, json, sys
import numpy as np
from scipy.optimize import linprog


def maximin(M):
    """M[i][j] = recall of policy i under attacker move j. Returns (value, weights)."""
    M = np.asarray(M, float); n, m = M.shape
    c = np.zeros(n + 1); c[-1] = -1                      # maximize v
    A = np.hstack([-M.T, np.ones((m, 1))])                # v - sum_i q_i M[i,j] <= 0
    res = linprog(c, A_ub=A, b_ub=np.zeros(m), A_eq=[[1] * n + [0]], b_eq=[1], bounds=[(0, 1)] * n + [(None, None)])
    return res.x[-1], res.x[:-1]


def main(path, kf=0.25, s=4, stratum="modes"):
    rows = [r for r in json.load(open(path)) if r["recall"] is not None and r["kfrac"] == kf and r["s"] == s]
    pol = sorted({r["policy"] for r in rows}); strata = sorted({(r["environment"], r[stratum]) for r in rows})
    cell = collections.defaultdict(list)
    for r in rows: cell[(r["policy"], (r["environment"], r[stratum]))].append(r["recall"])
    M = [[np.mean(cell[(p, st)]) if cell[(p, st)] else 0.0 for st in strata] for p in pol]
    print(f"k={kf}E, s={s}; columns = (environment, {stratum})")
    for p, row in zip(pol, M):
        print(f"{p:38s} mean {100*np.mean(row):5.1f}%  worst {100*min(row):5.1f}%")
    v, q = maximin(M)
    print(f"maximin rotation: worst-case {100*v:.1f}% with", {p: round(w, 2) for p, w in zip(pol, q) if w > 0.01})
    return pol, strata, M, v, q


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:4]))
