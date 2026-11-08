"""Message and log types shared by the simulator, attacks and monitors."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable
import json

Edge = tuple[str, str]


@dataclass(frozen=True)
class Message:
    """One message on one edge in one round.

    ``hidden`` holds ground truth that monitors must never see (e.g. whether the message
    was injected by an attack, which attack, which fragment). Views strip it.
    """

    msg_id: str
    sender: str
    recipient: str
    round: int
    timestamp: float
    content: str
    fields: dict[str, Any] = field(default_factory=dict)  # structured payload (synthetic sim)
    hidden: dict[str, Any] = field(default_factory=dict)

    @property
    def edge(self) -> Edge:
        return (self.sender, self.recipient)

    def public(self) -> "Message":
        """Copy without hidden ground truth."""
        return Message(self.msg_id, self.sender, self.recipient, self.round, self.timestamp,
                       self.content, dict(self.fields), {})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["edge"] = list(self.edge)
        return d


class MessageLog:
    """Append-only log of messages with lookups by edge and round."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def append(self, m: Message) -> None:
        self._messages.append(m)

    def extend(self, ms: Iterable[Message]) -> None:
        for m in ms:
            self.append(m)

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def all(self) -> list[Message]:
        return list(self._messages)

    def on_edges(self, edges: Iterable[Edge]) -> list[Message]:
        es = set(edges)
        return [m for m in self._messages if m.edge in es]

    def in_round(self, r: int) -> list[Message]:
        return [m for m in self._messages if m.round == r]

    def edges(self) -> list[Edge]:
        seen: dict[Edge, None] = {}
        for m in self._messages:
            seen.setdefault(m.edge, None)
        return list(seen)

    def fingerprint(self) -> str:
        """Stable hash of the public content, used to check that monitoring never
        changed the communication."""
        import hashlib
        h = hashlib.sha256()
        for m in self._messages:
            h.update(json.dumps(m.public().to_dict(), sort_keys=True).encode())
        return h.hexdigest()

    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for m in self._messages:
                f.write(json.dumps(m.to_dict(), sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "MessageLog":
        log = cls()
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                d.pop("edge", None)
                log.append(Message(**d))
        return log
