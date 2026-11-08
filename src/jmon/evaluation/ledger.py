from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class BudgetLedger:
    monitored_messages: int = 0
    monitor_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    monitored_edges: int = 0
    max_view_messages: int = 0
    by_monitor: dict = field(default_factory=dict)

    def charge(self, monitor_id: str, n_messages: int, verdict) -> None:
        self.monitored_messages += n_messages
        self.monitor_calls += verdict.calls
        self.tokens_in += verdict.tokens_in
        self.tokens_out += verdict.tokens_out
        self.latency_s += verdict.latency_s
        self.max_view_messages = max(self.max_view_messages, n_messages)
        d = self.by_monitor.setdefault(monitor_id, {"calls": 0, "messages": 0, "tokens_in": 0})
        d["calls"] += verdict.calls; d["messages"] += n_messages; d["tokens_in"] += verdict.tokens_in

    def to_dict(self) -> dict:
        return asdict(self)
