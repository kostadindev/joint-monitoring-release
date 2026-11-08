"""Clean-execution probes: single-edge masking loss and the pairwise interaction

    J(e_i, e_j) = L_ij - L_i - L_j + L_0

computed on clean episodes only (no attack hooks). Sign convention: J > 0 means the two
edges are redundant for the task (masking both hurts more than the sum, e.g. two paths
carrying the same fact); J < 0 means sub-additive. This is the second-order Mobius
(Harsanyi dividend) term of the masking-loss set function.

Whether J predicts joint *monitoring* value is hypothesis H3, not an assumption.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from jmon.graphs import CommGraph
from jmon.messages import Edge
from jmon.simulate import run_episode, EdgeMask


def clean_loss_matrix(graph: CommGraph, agents, truth, seeds: list[int], edges: list[Edge] | None = None,
                      pairs: bool = True) -> tuple[float, dict[Edge, float], dict[tuple[Edge, Edge], float]]:
    """Average clean task loss over ``seeds`` for: no mask (L0), each single edge, each pair."""
    edges = edges or graph.edges
    L0 = float(np.mean([run_episode(graph, agents, truth, s).task_loss for s in seeds]))
    L1 = {e: float(np.mean([run_episode(graph, agents, truth, s, hooks=[EdgeMask([e])]).task_loss for s in seeds]))
          for e in edges}
    L2 = {}
    if pairs:
        for a, b in combinations(edges, 2):
            L2[(a, b)] = float(np.mean([run_episode(graph, agents, truth, s, hooks=[EdgeMask([a, b])]).task_loss
                                        for s in seeds]))
    return L0, L1, L2


def interaction_matrix(L0: float, L1: dict[Edge, float], L2: dict[tuple[Edge, Edge], float]) -> dict[tuple[Edge, Edge], float]:
    return {(a, b): L2[(a, b)] - L1[a] - L1[b] + L0 for (a, b) in L2}
