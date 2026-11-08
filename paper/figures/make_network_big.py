"""Figure 1 (illustrative, one graph): a larger customer-service network built on MESA's task 3
facts (Bob Chen bought headphones 20 days ago; 30-day return window). One compromised agent sends a
false date and an honest agent repeats it; the true date reaches the decision along parallel routes. No channel is
wrong on its own: the lie shows only when one judge call reads two channels at once. The
groupings follow from the topology (pair the channels where two routes from the same source
meet). Not executed; the measured analogue is the MESA case in Section 5."""
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

RED, GOLD, DARK, GREEN, GREY, ORANGE = "#c0392b", "#c9930c", "#222222", "#1e8449", "#a6a6a6", "#e67e22"
CALLC = {1: "#16a085", 2: "#7d3c98"}
N = {  # x, y, label, kind
    "cust": (0.9, 14.4, "Customer", "ext"), "orders": (0.9, 8.8, "Order\nrecords", "ext"),
    "desk": (4.2, 14.4, "Customer\ndesk", "w"), "db": (4.2, 11.4, "Database", "w"), "bill": (4.2, 8.8, "Billing", "w"),
    "ship": (4.2, 5.3, "Shipping", "w"),
    "intake": (8.0, 14.4, "Intake", "w"), "kb": (8.0, 12.55, "Policy\nhandbook", "ext"), "pol": (8.0, 10.1, "Policy", "w"),
    "res": (11.4, 8.6, "Resolution", "w"), "audit": (11.4, 12.3, "Audit", "w"),
    "sup": (14.6, 8.6, "Supervisor\n(decides)", "hub"), "fin": (14.6, 12.3, "Finance", "w"), "notify": (14.6, 4.9, "Notifi-\ncations", "w"),
}
R = {"hub": 0.85, "w": 0.62, "ext": 0.58}
E = [("cust", "desk"), ("orders", "db"), ("orders", "bill"), ("orders", "ship"), ("kb", "pol"),
     ("desk", "intake"), ("intake", "res"), ("db", "pol"), ("bill", "pol"), ("pol", "res"), ("res", "pol"),
     ("ship", "res"), ("res", "sup"), ("res", "audit"), ("audit", "sup"), ("sup", "fin"), ("sup", "notify"),
     ("sup", "intake"), ("db", "intake")]
TAMPER = {("db", "pol"): "MALICIOUS (adversary)\n“Purchased 2 Jan:\n65 days ago”"}
CONTAM = {("pol", "res"): "CONTAMINATED (honest Policy agent\nrepeats the lie)\n“65 days old: outside the\n30-day window. Deny.”"}
HONEST = {("bill", "pol"): "“Payment taken\n20 days ago”",
          ("ship", "res"): "“Shipped the day it was\nbought, 20 days ago”",
          ("desk", "intake"): "“I bought them\n20 days ago”"}
LABEL_AT = {("db", "pol"): (5.95, 10.12), ("bill", "pol"): (6.05, 8.35), ("pol", "res"): (10.05, 11.05),
            ("ship", "res"): (8.9, 5.55), ("desk", "intake"): (6.1, 15.05)}
CALLS = {1: [("db", "pol"), ("bill", "pol")], 2: [("pol", "res"), ("ship", "res")]}
ROUTES = {1: [("orders", "db"), ("orders", "bill")], 2: [("orders", "ship")]}
member = {e: c for c, es in CALLS.items() for e in es}
CURVE = {("sup", "intake"): 0.32, ("res", "pol"): 0.3, ("pol", "res"): 0.0}


def ends(a, b):
    (x1, y1, _, k1), (x2, y2, _, k2) = N[a], N[b]
    dx, dy = x2 - x1, y2 - y1; L = math.hypot(dx, dy); ux, uy = dx / L, dy / L
    return (x1 + ux * (R[k1] + 0.06), y1 + uy * (R[k1] + 0.06)), (x2 - ux * (R[k2] + 0.11), y2 - uy * (R[k2] + 0.11))


fig = plt.figure(figsize=(7.5, 10.6))
ax = fig.add_axes([0.0, 3.3 / 10.6, 1.0, 7.3 / 10.6]); ax.set_xlim(-0.2, 15.8); ax.set_ylim(2.75, 15.9); ax.set_aspect("equal"); ax.axis("off")

for c, es in ROUTES.items():   # the parallel routes feeding each meeting point (faint)
    for e in es:
        s, t = ends(*e)
        ax.add_patch(FancyArrowPatch(s, t, arrowstyle="-", lw=9, color=CALLC[c], alpha=0.12, shrinkA=0, shrinkB=0,
                                     capstyle="round", zorder=0))
for e in E:
    s, t = ends(*e)
    cs = f"arc3,rad={CURVE.get(e, 0.0)}"
    c = member.get(e)
    if c:
        ax.add_patch(FancyArrowPatch(s, t, arrowstyle="-", connectionstyle=cs, lw=14, color=CALLC[c], alpha=0.28,
                                     shrinkA=0, shrinkB=0, capstyle="round", zorder=1))
    if e in TAMPER:
        col, lw, ls = RED, 3.2, "-"
    elif e in CONTAM:
        col, lw, ls = ORANGE, 3.0, "-"
    elif e in HONEST:
        col, lw, ls = GOLD, 3.0, "-"
    else:
        col, lw, ls = GREY, 1.0, (0, (3, 2))
    ax.add_patch(FancyArrowPatch(s, t, arrowstyle="-|>", connectionstyle=cs, mutation_scale=12 if lw > 1.5 else 8,
                                 lw=lw, color=col, ls=ls, shrinkA=0, shrinkB=0, zorder=2))
    if c:
        bx, by = s[0] + 0.62 * (t[0] - s[0]), s[1] + 0.62 * (t[1] - s[1])
        ax.add_patch(Circle((bx, by), 0.33, fc=CALLC[c], ec="white", lw=1.2, zorder=6))
        ax.text(bx, by, str(c), color="white", fontsize=7.5, weight="bold", ha="center", va="center", zorder=7)
for e, (x, y) in LABEL_AT.items():
    if e in TAMPER:
        txt, col, fc, ec = TAMPER[e], RED, "#fdecea", RED
    elif e in CONTAM:
        txt, col, fc, ec = CONTAM[e], "#a04000", "#fdf2e9", ORANGE
    else:
        txt, col, fc, ec = HONEST[e], "#8a6400", "#fff7e0", GOLD
    ax.text(x, y, txt, fontsize=5.8, ha="center", va="center", weight="bold", zorder=8, color=col,
            bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=0.9))
for k, (x, y, lab, kind) in N.items():
    fc = DARK if kind == "hub" else ("#eceff1" if kind == "ext" else "white")
    ax.add_patch(Circle((x, y), R[kind], fc=fc, ec=DARK, lw=1.0, ls="--" if kind == "ext" else "-", zorder=4))
    ax.text(x, y, lab, ha="center", va="center", fontsize=6.0, color="white" if kind == "hub" else DARK, zorder=5)
xdb, ydb = N["db"][0], N["db"][1]
ax.add_patch(Circle((xdb, ydb), R["w"] + 0.12, fc="none", ec=RED, lw=2.6, zorder=5))
ax.text(xdb - 1.95, ydb + 0.15, "ADVERSARY\ncompromised\nagent", fontsize=6.0, ha="center", va="center", color="white",
        weight="bold", zorder=9, bbox=dict(boxstyle="round,pad=0.25", fc=RED, ec=RED))
ax.text(0.9, 7.85, "true purchase date:\n20 days ago", fontsize=5.6, ha="center", va="top", style="italic", color="#555555")
# call captions under the graph (one line each)
ax.text(4.5, 3.0, "Call 1: Database→Policy + Billing→Policy", fontsize=6.6, ha="center", va="center",
        color=CALLC[1], weight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CALLC[1], lw=1.0))
ax.text(11.9, 3.0, "Call 2: Policy→Resolution + Shipping→Resolution", fontsize=6.6, ha="center", va="center",
        color=CALLC[2], weight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CALLC[2], lw=1.0))

# ---------------- bottom: what one judge call sees (same budget, same judge)
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from cards import card
bx = fig.add_axes([0.0, 0.0, 1.0, 3.3 / 10.6]); bx.set_xlim(0, 16); bx.set_ylim(0, 7.0); bx.axis("off")
# legend: three aligned rows that fit the panel width
ROW = {"Nodes": 6.72, "Channels": 6.18, "Monitor": 5.64}
for lab, y in ROW.items():
    bx.text(0.2, y, lab, fontsize=6.6, weight="bold", va="center", color=DARK)
fig.canvas.draw(); _r = fig.canvas.get_renderer()


def _w(t):  # width of a text object in data units of bx
    bb = t.get_window_extent(renderer=_r); inv = bx.transData.inverted()
    return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]


x0, y = 1.75, ROW["Nodes"]
for (fc, ec, ls, lw, lab) in [("white", DARK, "-", 1.0, "LLM agent"), ("white", RED, "-", 2.4, "compromised agent"),
                              ("#eceff1", DARK, "--", 1.0, "data source (not an agent)"),
                              (DARK, DARK, "-", 1.0, "supervisor (decides)")]:
    bx.scatter([x0], [y], s=80, facecolor=fc, edgecolor=ec, linewidths=lw, linestyle=ls, zorder=3)
    t = bx.text(x0 + 0.28, y, lab, fontsize=6.2, va="center"); x0 += 0.28 + _w(t) + 0.6
x0, y = 1.6, ROW["Channels"]
for (col, lw, ls, lab) in [(RED, 3.0, "-", "malicious"), (ORANGE, 3.0, "-", "contaminated (repeats the lie)"),
                           (GOLD, 3.0, "-", "honest (true date)"), (GREY, 1.0, (0, (3, 2)), "not read")]:
    bx.plot([x0, x0 + 0.42], [y, y], color=col, lw=lw, ls=ls, solid_capstyle="round")
    t = bx.text(x0 + 0.52, y, lab, fontsize=6.2, va="center"); x0 += 0.52 + _w(t) + 0.6
y = ROW["Monitor"]
for k, c in enumerate((1, 2)):
    bx.scatter([1.85 + 0.42 * k], [y], s=110, color=CALLC[c], edgecolors="white", zorder=3)
    bx.text(1.85 + 0.42 * k, y, str(c), color="white", fontsize=6.0, weight="bold", ha="center", va="center", zorder=4)
bx.text(2.5, y, "channels read together in one monitor call; the monitor does not know which agent is compromised",
        fontsize=6.2, va="center")
bx.plot([0.2, 15.8], [5.2, 5.2], color="#dddddd", lw=0.8)
bx.text(0.2, 4.78, "What the monitor sees with the same budget of two channels", fontsize=7.8, weight="bold", va="center", color=DARK)
for i, (x, title, chips, verdict, ok, split) in enumerate([
        (0.4, "Malicious and honest, separate calls", [("call A: \u201cPurchased 65 days ago\u201d", "lie"),
                                                      ("call B: \u201cPayment taken 20 days ago\u201d", "honest")],
         "each looks normal: missed", False, True),
        (5.55, "Malicious + contaminated, one call", [("\u201cPurchased 65 days ago\u201d", "lie"), ("\u201c65 days old: deny\u201d", "contam")],
         "they agree: missed", False, False),
        (10.7, "Malicious + honest, one call (call 1)", [("\u201cPurchased 65 days ago\u201d", "lie"),
                                                ("\u201cPayment taken 20 days ago\u201d", "honest")], "65 \u2260 20: flagged", True, False)]):
    card(bx, x, 0.1, 4.9, 4.25, f"({'abc'[i]})  {title}", chips, verdict, ok, split=split)
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).with_name(f"network_big.{ext}"), dpi=220)
print("wrote network_big", len(E))
