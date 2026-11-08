"""Provider-backed agent: a role prompt, the inbox rendered as text, one reply per out-edge.

Used for real-LLM runs; the simulator calls ``act`` exactly as for ``SimAgent``. Facts are
not parsed from LLM output; ``fields`` carries only the raw text so that task scoring is
done by a scenario-specific judge (``scenarios/``, to be added with the first LLM run).
"""
from __future__ import annotations

from jmon.llm.base import Provider
from jmon.messages import Message, Edge


class LLMAgent:
    def __init__(self, name: str, role_prompt: str, provider: Provider, task: str = "") -> None:
        self.name, self.role_prompt, self.provider, self.task = name, role_prompt, provider, task
        self.inbox_text: list[str] = []

    def reset(self) -> None:
        self.inbox_text = []

    def act(self, inbox: list[Message], out_edges: list[Edge], round_idx: int, make_msg) -> list[Message]:
        incoming = "\n".join(f"From {m.sender}: {m.content}" for m in inbox) or "(no messages yet)"
        out = []
        for (u, v) in out_edges:
            user = (f"Task: {self.task}\nRound {round_idx}. Messages you received:\n{incoming}\n\n"
                    f"Write your message to {v}.")
            text, _ = self.provider.complete(self.role_prompt, user)
            out.append(make_msg(u, v, round_idx, text.strip(), {"text": text.strip()}))
        return out
