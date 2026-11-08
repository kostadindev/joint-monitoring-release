"""Reimplementation of MESA's edge ranking (Li et al. 2026, Table I and Eq. 1).

Eight per-edge features, each normalized to [0, 1] over the graph's edges (min-max) and
summed with fixed signs:
  + edge betweenness, + information bottleneck, + bridge indicator,
  - endpoint centrality, - source out-degree, - target in-degree,
  + ablation drop (Acc(G) - Acc(G \\ e)), + masking drop (Acc(G) - Acc(G^mask(e))).
``mesa_static`` uses the six static features; ``mesa_combined`` all eight. The paper does
not specify the normalization beyond "normalized"; min-max is used here and stated.
The dynamic probes are single-edge clean-execution probes, as in the paper.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from jmon.graphs import CommGraph
from jmon.messages import Edge
from jmon.simulate import run_episode, EdgeMask, EdgeFiller


def static_features(graph) -> dict[str, dict[Edge, float]]:
    """``graph`` is a CommGraph or any networkx DiGraph (MESA's topologies are cyclic)."""
    g = graph.g if isinstance(graph, CommGraph) else graph
    edges = graph.edges if isinstance(graph, CommGraph) else list(g.edges())
    n = max(1, g.number_of_nodes() - 1)
    betw = nx.edge_betweenness_centrality(g, normalized=False)
    f = {"betweenness": {}, "bottleneck": {}, "bridge": {}, "endpoint_centrality": {},
         "source_outdeg": {}, "target_indeg": {}}
    for (u, v) in edges:
        f["betweenness"][(u, v)] = float(betw.get((u, v), 0.0))
        h = g.copy(); h.remove_edge(u, v)
        try:
            alt = nx.shortest_path_length(h, u, v)
            f["bottleneck"][(u, v)] = 1.0 - 1.0 / alt
            f["bridge"][(u, v)] = 0.0
        except nx.NetworkXNoPath:
            f["bottleneck"][(u, v)] = 1.0
            f["bridge"][(u, v)] = 1.0
        f["endpoint_centrality"][(u, v)] = max(g.degree(u), g.degree(v)) / n
        f["source_outdeg"][(u, v)] = g.out_degree(u) / n
        f["target_indeg"][(u, v)] = g.in_degree(v) / n
    return f


def dynamic_features(graph: CommGraph, agents, truth, seeds: list[int], edges: list[Edge] | None = None) -> dict[str, dict[Edge, float]]:
    """Ablation and masking accuracy drops on clean episodes (train seeds only)."""
    edges = edges or graph.edges
    acc = lambda hooks: float(np.mean([1.0 - run_episode(graph, agents, truth, s, hooks=hooks).task_loss for s in seeds]))
    base = acc([])
    return {"ablation": {e: base - acc([EdgeMask([e])]) for e in edges},
            "masking": {e: base - acc([EdgeFiller([e])]) for e in edges}}


SIGNS = {"betweenness": 1, "bottleneck": 1, "bridge": 1, "endpoint_centrality": -1, "source_outdeg": -1,
         "target_indeg": -1, "ablation": 1, "masking": 1}


def _minmax(d: dict[Edge, float]) -> dict[Edge, float]:
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {e: 0.0 for e in d}
    return {e: (x - lo) / (hi - lo) for e, x in d.items()}


def mesa_scores(graph, agents=None, truth=None, seeds: list[int] | None = None, combined: bool = True) -> dict[Edge, float]:
    feats = static_features(graph)
    edges = graph.edges if isinstance(graph, CommGraph) else list(graph.edges())
    if combined:
        if agents is None or seeds is None:
            raise ValueError("combined MESA needs agents, truth and train seeds for the probes")
        feats.update(dynamic_features(graph, agents, truth, seeds))
    score = {e: 0.0 for e in edges}
    for name, d in feats.items():
        nd = _minmax(d)
        for e in edges:
            score[e] += SIGNS[name] * nd[e]
    return score
