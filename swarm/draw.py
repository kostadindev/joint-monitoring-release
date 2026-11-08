"""Draw a swarm: layered left-to-right by topological depth, nodes coloured by team,
sources as grey squares, the decider black. Optional highlights: compromised agents
(red ring), attacked channels (red), corroborating channels (gold)."""
from __future__ import annotations
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#4e79a7", "#59a14f", "#b07aa1", "#f28e2b", "#edc948", "#9c755f", "#ff9da7", "#17becf", "#bab0ac", "#e15759", "#2ca02c", "#9467bd"]


def layout(G):
    depth = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        depth[n] = 0 if not preds else 1 + max(depth[p] for p in preds)
    cols = {}
    for n, d in depth.items():
        cols.setdefault(d, []).append(n)
    order = {}
    for d in sorted(cols):                               # barycentre ordering reduces crossings
        ns = cols[d]
        if d == 0:
            ns.sort(key=lambda n: (G.nodes[n].get("team", ""), n))
        else:
            def bary(n):
                ps = [order[p] for p in G.predecessors(n) if p in order]
                return sum(ps) / len(ps) if ps else 0.5
            ns.sort(key=lambda n: (bary(n), G.nodes[n].get("team", ""), n))
        for i, n in enumerate(ns):
            order[n] = i / max(len(ns) - 1, 1)
    pos = {}
    for d, ns in cols.items():
        for i, n in enumerate(ns):
            pos[n] = (d, (i - (len(ns) - 1) / 2) * (12.0 / max(len(ns), 6)))
    return pos


def draw(G, path, title="", compromised=(), attacked=(), corroborating=(), groups=None):
    pos = layout(G)
    teams = sorted({d.get("team", "") for _, d in G.nodes(data=True) if d["kind"] == "agent"})
    tcol = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(teams)}
    width = 2.0 + 1.6 * (max(x for x, _ in pos.values()) + 1)
    fig, ax = plt.subplots(figsize=(min(width, 16), 9))
    attacked, corroborating = set(attacked), set(corroborating)
    for u, v in G.edges():
        c, lw, a, z = "#bbbbbb", 0.5, 0.55, 1
        if (u, v) in corroborating: c, lw, a, z = "#c9930c", 2.2, 1, 3
        if (u, v) in attacked: c, lw, a, z = "#c0392b", 2.6, 1, 4
        ax.annotate("", xy=pos[v], xytext=pos[u], zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, alpha=a, shrinkA=6, shrinkB=6, mutation_scale=7))
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        if d["kind"] == "source":
            ax.scatter([x], [y], marker="s", s=180, color="#d5d8dc", edgecolor="#555", zorder=5)
            ax.text(x - 0.12, y, d["role"], fontsize=6.5, ha="right", va="center")
        elif d["kind"] == "decider":
            ax.scatter([x], [y], s=320, color="#222", zorder=5)
            ax.text(x + 0.12, y, "decider\n(" + d["role"] + ")", fontsize=6.5, va="center")
        else:
            ax.scatter([x], [y], s=70, color=tcol[d["team"]], edgecolor="#333", lw=0.5, zorder=5)
            if n in compromised:
                ax.scatter([x], [y], s=190, facecolor="none", edgecolor="#c0392b", lw=2, zorder=6)
    for t in teams:
        ax.scatter([], [], s=40, color=tcol[t], label=t)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=6, frameon=False, title="teams", title_fontsize=7)
    import textwrap
    ax.set_title("\n".join(textwrap.wrap(title, 150)), fontsize=9, loc="left")
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)
