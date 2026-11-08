"""Monitor views: the only path by which a monitor receives messages.

``build_view(log, edges)`` returns the public copies of exactly the messages on
``edges``, ordered by (round, timestamp), with provenance (sender, recipient, edge, round,
timestamp) attached. Hidden ground truth is stripped. Tests assert that no message from
an unassigned edge can appear in a view.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy

from jmon.messages import Message, MessageLog, Edge


@dataclass(frozen=True)
class Observation:
    msg_id: str
    sender: str
    recipient: str
    round: int
    timestamp: float
    content: str
    fields: dict = field(default_factory=dict)

    @property
    def edge(self) -> Edge:
        return (self.sender, self.recipient)


@dataclass
class View:
    monitor_id: str
    edges: list[Edge]
    observations: list[Observation]

    @property
    def n_messages(self) -> int:
        return len(self.observations)

    def text(self, with_provenance: bool = True) -> str:
        lines = []
        for o in self.observations:
            if with_provenance:
                lines.append(f"[round {o.round}] {o.sender} -> {o.recipient}: {o.content}")
            else:
                lines.append(o.content)
        return "\n".join(lines)


def build_view(log: MessageLog, edges: list[Edge], monitor_id: str = "m0") -> View:
    es = set(edges)
    obs = []
    for m in sorted(log.on_edges(es), key=lambda m: (m.round, m.timestamp)):
        assert m.edge in es
        pm = m.public()
        obs.append(Observation(pm.msg_id, pm.sender, pm.recipient, pm.round, pm.timestamp, pm.content, copy.deepcopy(pm.fields)))
    return View(monitor_id, list(edges), obs)
