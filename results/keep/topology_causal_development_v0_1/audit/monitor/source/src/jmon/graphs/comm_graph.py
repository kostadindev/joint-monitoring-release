"""Directed multi-agent communication graph."""
from __future__ import annotations

import networkx as nx
import numpy as np

from jmon.messages import Edge


class CommGraph:
    """Thin wrapper over ``networkx.DiGraph`` with a fixed edge order.

    Edge order is the order of insertion, so that allocation code can index edges
    reproducibly. ``sources`` are agents without predecessors; ``sink`` is the agent
    whose final output is the task output (must be unique for the synthetic task).
    """

    def __init__(self, nodes: list[str], edges: list[Edge], sink: str | None = None) -> None:
        self.g = nx.DiGraph()
        self.g.add_nodes_from(nodes)
        self.g.add_edges_from(edges)
        self.nodes: list[str] = list(nodes)
        self.edges: list[Edge] = list(edges)
        sinks = [n for n in nodes if self.g.out_degree(n) == 0]
        if sink is None:
            if len(sinks) != 1:
                raise ValueError(f"need exactly one sink, found {sinks}")
            sink = sinks[0]
        self.sink = sink

    def predecessors(self, n: str) -> list[str]:
        return list(self.g.predecessors(n))

    def successors(self, n: str) -> list[str]:
        return list(self.g.successors(n))

    def in_edges(self, n: str) -> list[Edge]:
        return [e for e in self.edges if e[1] == n]

    def out_edges(self, n: str) -> list[Edge]:
        return [e for e in self.edges if e[0] == n]

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.g))

    def edge_index(self, e: Edge) -> int:
        return self.edges.index(e)

    def betweenness(self) -> dict[Edge, float]:
        return nx.edge_betweenness_centrality(self.g)

    def edge_distance(self, a: Edge, b: Edge) -> int:
        """Distance between edges = shortest undirected path between their midpoints,
        approximated by min node distance between endpoints (0 if they share a node)."""
        ug = self.g.to_undirected()
        best = np.inf
        for u in a:
            for v in b:
                try:
                    best = min(best, nx.shortest_path_length(ug, u, v))
                except nx.NetworkXNoPath:
                    pass
        return int(best) if np.isfinite(best) else 10**6

    def to_dict(self) -> dict:
        return {"nodes": self.nodes, "edges": [list(e) for e in self.edges], "sink": self.sink}


def make_graph(kind: str, n_agents: int, seed: int = 0, **kw) -> CommGraph:
    """Generators for acyclic workflow graphs. All deterministic in ``seed``.

    kinds: ``chain``, ``star`` (sources -> hub -> sink), ``tree`` (binary aggregation),
    ``layered`` (random layered DAG, parameters ``width``, ``p``), ``diamond``
    (two parallel paths, used for redundancy tests).
    """
    rng = np.random.default_rng(seed)
    names = [f"a{i}" for i in range(n_agents)]
    if kind == "chain":
        edges = [(names[i], names[i + 1]) for i in range(n_agents - 1)]
        return CommGraph(names, edges)
    if kind == "star":
        hub, sink = names[-2], names[-1]
        edges = [(s, hub) for s in names[:-2]] + [(hub, sink)]
        return CommGraph(names, edges)
    if kind == "diamond":
        # a0 -> a1 -> a3, a0 -> a2 -> a3, plus extra sources into a1/a2
        if n_agents < 4:
            raise ValueError("diamond needs >= 4 agents")
        edges = [(names[0], names[1]), (names[0], names[2]), (names[1], names[3]), (names[2], names[3])]
        for i in range(4, n_agents):
            edges.append((names[i], names[1 + (i % 2)]))
        return CommGraph(names, edges, sink=names[3])
    if kind == "tree":
        edges = []
        for i in range(n_agents - 1, 0, -1):
            parent = (i - 1) // 2
            edges.append((names[i], names[parent]))
        # sink is a0: relabel so sink is last for readability
        return CommGraph(names, edges, sink=names[0])
    if kind == "layered":
        width = int(kw.get("width", 3))
        p = float(kw.get("p", 0.6))
        layers: list[list[str]] = []
        inner = names[:-1]
        for i in range(0, len(inner), width):
            layers.append(inner[i:i + width])
        layers.append([names[-1]])
        edges: list[Edge] = []
        for l in range(len(layers) - 1):
            for u in layers[l]:
                targets = [v for v in layers[l + 1] if rng.random() < p]
                if not targets:
                    targets = [layers[l + 1][rng.integers(len(layers[l + 1]))]]
                for v in targets:
                    edges.append((u, v))
        # ensure every non-final layer-(l+1) node has an in-edge
        for l in range(1, len(layers)):
            for v in layers[l]:
                if not any(e[1] == v for e in edges):
                    u = layers[l - 1][rng.integers(len(layers[l - 1]))]
                    edges.append((u, v))
        return CommGraph(names, edges, sink=names[-1])
    raise ValueError(f"unknown graph kind {kind}")
