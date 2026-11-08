"""Edge-selection policies (which channels to observe). All take only clean-execution
information or graph structure; never attack labels.

- random
- betweenness: structural proxy for MESA's graph metrics
- clean_ablation: MESA-style dynamic probe, rank by single-edge masking loss L_i - L_0
- score: an externally supplied per-edge score dict (e.g. a full MESA ranking)
"""
from __future__ import annotations

import numpy as np

from jmon.graphs import CommGraph
from jmon.messages import Edge


def select_edges(policy: str, graph: CommGraph, k: int, rng: np.random.Generator,
                 single_loss: dict[Edge, float] | None = None, scores: dict[Edge, float] | None = None) -> list[Edge]:
    E = graph.edges
    k = min(k, len(E))
    if policy == "random":
        idx = rng.choice(len(E), size=k, replace=False)
        return [E[i] for i in sorted(idx)]
    if policy == "betweenness":
        b = graph.betweenness()
        return _top(E, b, k, rng)
    if policy == "clean_ablation":
        if single_loss is None:
            raise ValueError("clean_ablation needs single_loss")
        return _top(E, single_loss, k, rng)
    if policy == "score":
        if scores is None:
            raise ValueError("score policy needs scores")
        return _top(E, scores, k, rng)
    raise ValueError(policy)


def _top(E, score: dict[Edge, float], k: int, rng) -> list[Edge]:
    # deterministic tie-break by a seeded jitter
    jitter = {e: rng.random() * 1e-9 for e in E}
    return sorted(E, key=lambda e: -(score.get(e, 0.0) + jitter[e]))[:k]


SELECTORS = ["random", "betweenness", "clean_ablation", "score"]
