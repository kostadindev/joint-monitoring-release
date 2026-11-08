"""Shared drawing helper: a 'judge card' shows what one judge call reads and its outcome."""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED, GOLD, GREEN, GREYTXT = "#c0392b", "#c9930c", "#1e8449", "#555555"


def card(ax, x, y, w, h, title, chips, verdict, ok, arrows=False, chip_fs=6.2, split=False):
    """chips: list of (text, kind) with kind in {'lie','honest','task'}; drawn top to bottom."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc="#fcfcfc",
                                ec="#cfd8dc", lw=1.0))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=6.8, weight="bold")
    n = len(chips)
    top, bottom = y + h - 0.75, y + 1.05
    step = (top - bottom) / max(n, 1)
    ch = min(0.62, step * 0.78)
    for i, (txt, kind) in enumerate(chips):
        cy = top - step * (i + 0.5)
        fc, ec, col = {"lie": ("#fdecea", RED, RED), "honest": ("#fff7e0", GOLD, "#8a6400"),
                       "contam": ("#fdf2e9", "#e67e22", "#a04000"),
                       "task": ("#f4f6f7", "#7f8c8d", "#2c3e50")}[kind]
        ax.add_patch(FancyBboxPatch((x + 0.25, cy - ch / 2), w - 0.5, ch, boxstyle="round,pad=0.01,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, cy, txt, ha="center", va="center", fontsize=chip_fs, color=col, weight="bold")
        if split and i < n - 1:
            ax.plot([x + 0.35, x + w - 0.35], [cy - step / 2, cy - step / 2], color="#90a4ae", lw=0.9, ls=(0, (3, 2)))
        if arrows and i < n - 1:
            ax.annotate("", (x + w / 2, cy - step + ch / 2 + 0.02), (x + w / 2, cy - ch / 2 - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=0.8, mutation_scale=7))
    col = GREEN if ok else RED
    bw = min(w - 0.3, 0.6 + 0.105 * len(verdict))
    ax.add_patch(FancyBboxPatch((x + w / 2 - bw / 2, y + 0.22), bw, 0.5, boxstyle="round,pad=0.01,rounding_size=0.2",
                                fc=col, ec=col, lw=0))
    ax.text(x + w / 2, y + 0.47, ("\u2714 " if ok else "\u2718 ") + verdict, ha="center", va="center", fontsize=6.8,
            weight="bold", color="white")
