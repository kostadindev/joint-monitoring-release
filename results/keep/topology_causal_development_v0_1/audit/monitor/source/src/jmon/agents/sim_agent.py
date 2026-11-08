"""Deterministic synthetic agent for the fact-aggregation task.

Each agent holds ``facts`` (key -> value). On its turn it merges facts received on
inbound messages (first value wins per key, by message order) and forwards every fact
it knows to each successor. Messages also carry benign ``ctx``/``act`` tokens drawn from
a seeded stream so that the conjunctive attack's fragments are not distinguishable from
benign traffic by their mere presence (see attacks.conjunctive).
"""
from __future__ import annotations

import numpy as np

from jmon.messages import Message, Edge


class SimAgent:
    def __init__(self, name: str, facts: dict[str, int] | None = None, rng: np.random.Generator | None = None,
                 benign_token_rate: float = 0.5, n_tokens: int = 8) -> None:
        self.name = name
        self.facts: dict[str, int] = dict(facts or {})
        self._rng_seed = rng.bit_generator.state if rng is not None else None
        self.rng = rng or np.random.default_rng(0)
        self.benign_token_rate = benign_token_rate
        self.n_tokens = n_tokens
        self.known: dict[str, int] = dict(self.facts)
        self.ctx_seen: list[str] = []
        self.act_seen: list[str] = []

    def reset(self) -> None:
        """Restore facts and the token stream, so every episode with the same seed is identical."""
        self.known = dict(self.facts)
        if self._rng_seed is not None:
            self.rng.bit_generator.state = self._rng_seed
        self.ctx_seen, self.act_seen = [], []

    def ingest(self, inbox: list[Message]) -> None:
        for m in inbox:
            for k, v in m.fields.get("facts", {}).items():
                self.known.setdefault(k, v)
            if "ctx" in m.fields:
                self.ctx_seen.append(m.fields["ctx"])
            if "act" in m.fields:
                self.act_seen.append(m.fields["act"])

    def act(self, inbox: list[Message], out_edges: list[Edge], round_idx: int, make_msg) -> list[Message]:
        self.ingest(inbox)
        out = []
        for (u, v) in out_edges:
            fields: dict = {"facts": dict(self.known)}
            # benign context/action tokens, forwarded or freshly drawn
            if self.ctx_seen:
                fields["ctx"] = self.ctx_seen[-1]
            elif self.rng.random() < self.benign_token_rate:
                fields["ctx"] = f"c{self.rng.integers(self.n_tokens)}"
            if self.act_seen:
                fields["act"] = self.act_seen[-1]
            elif self.rng.random() < self.benign_token_rate:
                fields["act"] = f"x{self.rng.integers(self.n_tokens)}"
            content = render(fields)
            out.append(make_msg(u, v, round_idx, content, fields))
        return out


def render(fields: dict) -> str:
    """Human-readable rendering used as ``content`` (what an LLM monitor would read)."""
    parts = []
    facts = fields.get("facts", {})
    if facts:
        parts.append("Report: " + ", ".join(f"{k}={v}" for k, v in sorted(facts.items())))
    if "ctx" in fields:
        parts.append(f"Context tag {fields['ctx']}.")
    if "act" in fields:
        parts.append(f"Suggested action {fields['act']}.")
    if "note" in fields:
        parts.append(fields["note"])
    return " ".join(parts) if parts else "(empty)"
