"""Grouping policies: partition the selected edges into monitor groups.

Every policy produces the SAME group-size vector for a given (k, s): ceil(k/s) groups,
sizes as equal as possible (e.g. k=6, s=3 -> [3,3]; k=7, s=3 -> [3,2,2]). Only the
membership differs between policies, so the number of calls and the context per call are
matched across conditions (Codex review 2026-11-06, item 2).

- singleton: every edge alone (independent monitoring; sizes all 1)
- random
- by_receiver: edges into the same agent kept together as far as sizes allow
- proximity: affinity = -graph distance between edges
- convergence: affinity = Jaccard of downstream agent sets (graph-only)
- semantic: affinity supplied (e.g. embedding cosine of clean messages)
- interaction: affinity = clean-execution interaction J(e_i, e_j)
- oracle: affinity = co-fragment frequency on TRAIN attacks (a supervised heuristic,
  not an upper bound)

Affinity policies use ``constrained_partition``: a greedy fill of the fixed size vector
by total within-group affinity, ties broken by a seeded jitter that is separate from the
stopping rule (there is no stopping rule: groups are always filled).
"""
from __future__ import annotations

import math

import numpy as np

from jmon.graphs import CommGraph
from jmon.messages import Edge
from .assignment import Assignment

GROUPERS = ["singleton", "random", "by_receiver", "proximity", "convergence", "semantic", "interaction", "oracle"]


def size_vector(k: int, s: int) -> list[int]:
    if k <= 0:
        return []
    m = math.ceil(k / s)
    base, extra = divmod(k, m)
    return [base + 1 if i < extra else base for i in range(m)]


def group_edges(policy: str, edges: list[Edge], max_group_size: int, rng: np.random.Generator,
                graph: CommGraph | None = None, affinity: dict[tuple[Edge, Edge], float] | None = None) -> Assignment:
    edges = list(edges)
    if policy == "singleton":
        return Assignment([[e] for e in edges], policy)
    sizes = size_vector(len(edges), max_group_size)
    if policy == "random":
        perm = [edges[i] for i in rng.permutation(len(edges))]
        return Assignment(_fill(perm, sizes), policy)
    if policy == "by_receiver":
        aff = {(a, b): (1.0 if a[1] == b[1] else 0.0) for a in edges for b in edges if a != b}
        return Assignment(constrained_partition(edges, aff, sizes, rng), policy)
    if policy == "proximity":
        if graph is None:
            raise ValueError("proximity needs graph")
        aff = {(a, b): -float(graph.edge_distance(a, b)) for a in edges for b in edges if a != b}
        return Assignment(constrained_partition(edges, aff, sizes, rng), policy)
    if policy == "convergence":
        if graph is None:
            raise ValueError("convergence needs graph")
        return Assignment(constrained_partition(edges, convergence_affinity(graph, edges), sizes, rng), policy)
    if policy in ("semantic", "interaction", "oracle"):
        if affinity is None:
            raise ValueError(f"{policy} needs affinity")
        return Assignment(constrained_partition(edges, affinity, sizes, rng), policy)
    raise ValueError(policy)


def _fill(order: list[Edge], sizes: list[int]) -> list[list[Edge]]:
    out, i = [], 0
    for s in sizes:
        out.append(order[i:i + s]); i += s
    return out


def constrained_partition(edges: list[Edge], affinity: dict[tuple[Edge, Edge], float], sizes: list[int],
                          rng: np.random.Generator) -> list[list[Edge]]:
    """Greedy: seed each group with the unassigned edge of highest total affinity to the
    remaining pool, then add the edge with the highest mean affinity to the group until
    the group's size is reached. Ties broken by seeded jitter. Always returns exactly
    ``sizes``."""
    def a(x, y):
        return affinity.get((x, y), affinity.get((y, x), 0.0))
    pool = list(edges)
    jit = {e: rng.random() * 1e-9 for e in pool}
    groups = []
    for s in sizes:
        if not pool:
            break
        seed = max(pool, key=lambda e: sum(a(e, f) for f in pool if f != e) + jit[e])
        g = [seed]; pool.remove(seed)
        while len(g) < s and pool:
            nxt = max(pool, key=lambda e: float(np.mean([a(e, f) for f in g])) + jit[e])
            g.append(nxt); pool.remove(nxt)
        groups.append(g)
    assert [len(g) for g in groups] == sizes[:len(groups)] and not pool
    return groups


def enumerate_pair_partitions(edges: list[Edge]) -> list[list[list[Edge]]]:
    """All perfect matchings of an even edge list (15 for six edges): a diagnostic that
    lets Experiment B evaluate every grouping instead of a heuristic oracle."""
    if not edges:
        return [[]]
    first, rest = edges[0], edges[1:]
    out = []
    for i, e in enumerate(rest):
        for sub in enumerate_pair_partitions(rest[:i] + rest[i + 1:]):
            out.append([[first, e]] + sub)
    return out


def convergence_affinity(graph: CommGraph, edges: list[Edge]) -> dict[tuple[Edge, Edge], float]:
    """Jaccard similarity of the sets of agents reachable from each edge's head (including
    the head). On strongly connected graphs this is identically one and carries no
    information; callers should check the spread of the affinity."""
    import networkx as nx
    desc = {e: {e[1]} | nx.descendants(graph.g, e[1]) for e in edges}
    out = {}
    for a in edges:
        for b in edges:
            if a != b:
                out[(a, b)] = len(desc[a] & desc[b]) / len(desc[a] | desc[b])
    return out


def oracle_affinity(train_fragment_sets: list[set[Edge]], edges: list[Edge]) -> dict[tuple[Edge, Edge], float]:
    """Fraction of train attacks in which both edges carried fragments. Uses attack labels,
    so it may only be computed on the train split (callers must guard with Splits)."""
    n = max(1, len(train_fragment_sets))
    out = {}
    for a in edges:
        for b in edges:
            if a != b:
                out[(a, b)] = sum(1 for F in train_fragment_sets if a in F and b in F) / n
    return out
