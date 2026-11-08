from __future__ import annotations

from typing import Protocol

from jmon.messages import Message


class Agent(Protocol):
    """An agent reads the messages addressed to it and produces one message per out-edge.

    Implementations: ``SimAgent`` (deterministic synthetic forwarding) and
    ``LLMAgent`` (provider-backed). The simulator is agnostic to which is used.
    """

    name: str

    def act(self, inbox: list[Message], out_edges: list[tuple[str, str]], round_idx: int,
            make_msg) -> list[Message]: ...
