from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .views import View


@dataclass
class Verdict:
    score: float                     # suspicion in [0, 1]; higher = more suspicious
    flagged_msg_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    calls: int = 1


class Monitor(Protocol):
    """A monitor maps one view (one or more messages) to a verdict.

    Independent monitoring = one call per single-message view.
    Joint monitoring = one call per multi-message view.
    """

    name: str

    def inspect(self, view: View) -> Verdict: ...
