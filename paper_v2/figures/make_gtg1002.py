"""Figure 2: the real GTG-1002 campaign (Anthropic, Nov 2025), drawn from the full report's
architecture, phases and task sequence, with the group view that reads the tasks one
orchestrator sent against one target together. Facts and quotes come from the report
(PDF pages cited). The report does not publish the messages; task labels quote or closely
paraphrase its task descriptions."""
from pathlib import Path
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

RED, DARK, GREEN, BLUE, GOLD = "#c0392b", "#222222", "#1e8449", "#2471a3", "#b9770e"
GROUP = "#c0392b"
fig = plt.figure(figsize=(7.5, 10.4))
ax = fig.add_axes([0.0, 0.33, 1.0, 0.67]); ax.set_xlim(0, 16); ax.set_ylim(0.4, 15.2); ax.axis("off")


def box(x, y, w, h, txt, fc, ec, fs=6.2, bold=False, color=DARK, ls="-"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=fc, ec=ec, lw=1.0, ls=ls, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", color=color, zorder=4)


def arrow(p, q, col="#9e9e9e", lw=1.0, ls="-", rad=0.0, ms=8, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", connectionstyle=f"arc3,rad={rad}", mutation_scale=ms, lw=lw,
                                 color=col, ls=ls, shrinkA=0, shrinkB=0, zorder=z))


# operators and gates
box(2.3, 13.6, 3.6, 1.2, "ADVERSARY: human operators\n(“employees of legitimate cybersecurity\nfirms”, the role-play persona)", "#eceff1", DARK, 6.0)
box(10.7, 13.6, 5.4, 1.2, "Orchestration engine (Claude)\nmaintains attack state, manages phase\ntransitions, aggregates results across sessions",
    "#fdecea", RED, 6.0, True, RED)
arrow((4.15, 13.85), (7.95, 13.85), DARK, 1.2)
arrow((7.95, 13.35), (4.15, 13.35), DARK, 1.2)
ax.text(6.05, 14.15, "target, approvals at gates", fontsize=5.6, ha="center")
ax.text(6.05, 12.6, "findings for review", fontsize=5.6, ha="center")
for i, g in enumerate(["approve exploitation", "authorize credential use", "approve exfiltration"]):
    ax.text(6.05, 14.85 - i * 0.0 + 0.0 if False else 14.6 + 0 * i, "", fontsize=1)
ax.text(2.3, 12.35, "decision gates: approve exploitation • authorize\nuse of harvested credentials • approve exfiltration",
        fontsize=5.5, ha="center", color="#444444", style="italic")

# sub-agents (task channels from the orchestrator)
SUB = [  # x, label (documented task type), phase
    (1.25, "Discovery\n“Scans target\ninfrastructure”\n“Enumerates services\nand endpoints”", "reconnaissance"),
    (3.45, "Vulnerability\nanalysis\n“Identifies SSRF\nvulnerability”", "vulnerability\ndiscovery"),
    (5.65, "Exploit\ndevelopment\n“Authors custom\npayload”", "validation"),
    (7.85, "Credential\nvalidation\n“testing harvested\ncredentials”", "credential\nharvesting"),
    (10.05, "Lateral\nmovement\n“enumeration of\naccessible systems”", "lateral\nmovement"),
    (12.25, "Data\nextraction\n“query databases …\nparse results”", "collection"),
    (14.45, "Documentation\n“Structured\nmarkdown files\ntracked …”", "handoff"),
]
for x, lab, ph in SUB:
    box(x, 9.4, 2.0, 1.75, lab, "white", RED, 5.3)
    ax.text(x, 10.55, "Claude agent task", fontsize=5.0, ha="center", color="#7b241c", style="italic")
    ax.add_patch(FancyArrowPatch((10.7, 12.95), (x, 10.75), arrowstyle="-", lw=12, color=GROUP, alpha=0.16,
                                 shrinkA=0, shrinkB=0, capstyle="round", zorder=1))
    arrow((10.7, 12.95), (x, 10.78), RED, 1.7, ms=9)
    ax.text(x, 8.25, ph, fontsize=5.0, ha="center", va="top", color="#7b241c")

# one shared tool layer: the report lists the MCP servers but not which task used which
ax.plot([0.4, 15.6], [6.75, 6.75], color="#9e9e9e", lw=1.2)
ax.text(15.6, 6.5, "agents call their tools through the framework's MCP servers (the report does not map tasks to servers)",
        fontsize=5.3, ha="right", color="#555555", style="italic")
for i in range(5):
    xx = 1.9 + i * 3.05
    arrow((xx, 6.75), (xx, 6.12), "#c8c8c8", 0.6, ms=6, z=1)
# MCP servers (documented)
MCP = ["Remote command\nexecution", "Browser\nautomation", "Code\nanalysis", "Testing\nframework", "Callback\ncommunication"]
for i, m in enumerate(MCP):
    x = 1.9 + i * 3.05
    box(x, 5.6, 2.4, 1.0, "MCP server\n" + m, "#eaf2f8", BLUE, 5.4)
for x, _, _ in SUB:
    arrow((x, 7.85), (x, 6.8), "#c8c8c8", 0.6, ms=6, z=1)
box(8.0, 2.9, 9.5, 1.35, "Targets: about 30 organisations (technology, finance, chemicals, government), attacked in parallel;\n"
    "“thousands of requests” (p. 7); “a handful of successful intrusions” (p. 3)",
    "#f2f3f4", DARK, 5.9, ls="--")
for i in range(5):
    x = 1.9 + i * 3.05
    arrow((x, 5.08), (x * 0.55 + 8.0 * 0.45, 3.6), "#c8c8c8", 0.6, ms=6, z=1)

# the group view
ax.add_patch(FancyBboxPatch((0.1, 7.55), 15.55, 4.45, boxstyle="round,pad=0.05,rounding_size=0.3", fc="none",
                            ec=GROUP, lw=1.6, ls=(0, (5, 3)), zorder=0))
ax.text(0.3, 7.25, "GROUP VIEW (dashed): one monitor call reads all tasks one orchestrator sent against one target",
        fontsize=6.6, color=GROUP, weight="bold", va="center")


# ---------------- bottom: what one judge call sees
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from cards import card
bx = fig.add_axes([0.0, 0.0, 1.0, 0.33]); bx.set_xlim(0, 16); bx.set_ylim(0, 7.0); bx.axis("off")
bx.text(8.0, 6.7, "What one monitor call sees", fontsize=8.4, weight="bold", ha="center")
card(bx, 0.4, 0.55, 5.6, 5.8, "One task per call (what each sub-agent saw)",
     [("“Scans target infrastructure”", "task")], "looks like a routine security test", False, chip_fs=6.0)
card(bx, 6.4, 0.55, 9.2, 5.8, "Group view: all tasks one orchestrator sent against one target",
     [("Scans target infrastructure", "task"), ("Identifies SSRF vulnerability", "task"),
      ("Authors custom payload", "task"), ("Testing harvested credentials", "task"),
      ("Enumeration of accessible systems", "task"), ("Query databases, parse results", "task")],
     "together: an intrusion chain is visible", True, arrows=True, chip_fs=5.9)
bx.text(8.0, 0.18, "Each task “appeared legitimate when evaluated in isolation” (Anthropic report, p. 6). "
        "Sources: Anthropic full report (PDF pp. 3–13); MITRE ATT&CK C0062. Monitor views are our illustration.", fontsize=5.6, ha="center", style="italic", color="#555555")
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).with_name(f"gtg1002.{ext}"), dpi=220)
print("wrote gtg1002")
