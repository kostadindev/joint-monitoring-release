"""Figure: the anonymous directed graphs that are the allocator's only input.

Reads figures/data/anonymous_<shape>.json (the exact strict graph objects given to
allocators, exported from results/keep/topology_causal_development_v0_1) and writes figures/allocator_inputs.pdf.
Layout is display-only: nodes are layered by longest path from the source.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
SRC = HERE / "data"
SHAPES = [("flat", "flat"), ("sequential", "sequential"), ("fork_join", "fork/join")]
INK, EDGE, FILL = "#24384a", "#7a8a99", "#eef3f7"


def layout(nodes, edges):
    succ = {n: [v for u, v in edges if u == n] for n in nodes}
    pred = {n: [u for u, v in edges if v == n] for n in nodes}
    depth = {}
    def d(n):
        if n not in depth:
            depth[n] = 0 if not pred[n] else 1 + max(d(p) for p in pred[n])
        return depth[n]
    for n in nodes:
        d(n)
    layers = {}
    for n in nodes:
        layers.setdefault(depth[n], []).append(n)
    order = {n: i for L in layers.values() for i, n in enumerate(sorted(L))}
    for _ in range(4):  # barycenter sweeps to reduce crossings
        for k in sorted(layers)[1:]:
            layers[k].sort(key=lambda n: sum(order[p] for p in pred[n]) / max(len(pred[n]), 1))
            for i, n in enumerate(layers[k]):
                order[n] = i
    pos = {}
    for k, L in layers.items():
        for i, n in enumerate(L):
            pos[n] = (k, ((len(L) - 1) / 2 - i) * 0.55)
    # a source that feeds a chain spanning several layers is lifted above it
    for n in nodes:
        if not pred[n] and len({depth[v] for v in succ[n]}) > 2:
            xs = [pos[v][0] for v in succ[n]]
            pos[n] = (sum(xs) / len(xs), 1.0)
    return pos


def draw(ax, nodes, edges, label):
    pos = layout(nodes, edges)
    for u, v in edges:
        ax.add_patch(FancyArrowPatch(pos[u], pos[v], arrowstyle="-|>", mutation_scale=7,
                                     color=EDGE, lw=0.8, shrinkA=6, shrinkB=6))
    xs, ys = zip(*pos.values())
    ax.scatter(xs, ys, s=70, facecolor=FILL, edgecolor=INK, linewidth=0.9, zorder=3)
    ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4)
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(label, fontsize=9, color=INK, loc="left", pad=2)


def main():
    plt.rcParams["font.family"] = "serif"
    g = {k: json.loads((SRC / f"anonymous_{k}.json").read_text()) for k, _ in SHAPES}
    fig = plt.figure(figsize=(6.3, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.25, wspace=0.12)
    panels = [(gs[0, 0], "flat", "(a) flat"), (gs[0, 1], "fork_join", "(b) fork/join"),
              (gs[1, :], "sequential", "(c) sequential")]
    for spec, key, name in panels:
        ax = fig.add_subplot(spec)
        draw(ax, g[key]["nodes"], [tuple(e) for e in g[key]["edges"]],
             f"{name} · {len(g[key]['nodes'])} nodes, {len(g[key]['edges'])} edges")
    fig.savefig(HERE / "allocator_inputs.pdf", bbox_inches="tight")
    print("wrote", HERE / "allocator_inputs.pdf")


if __name__ == "__main__":
    main()
