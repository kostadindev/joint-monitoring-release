from __future__ import annotations

import json
import re
import time

from jmon.llm.base import Provider
from .base import Verdict
from .views import View
from . import prompts


class LLMMonitor:
    """Provider-backed monitor. ``style`` in {independent, joint, concat}.

    independent: exactly one message per view; joint: any number, with provenance;
    concat: any number, provenance stripped (ablation). Same system prompt and background
    in all three. Every call's prompt, raw output and usage are kept in ``calls`` for
    auditing.
    """

    def __init__(self, provider: Provider, style: str = "joint", task_description: str = "",
                 name: str | None = None, max_input_tokens: int | None = None) -> None:
        self.provider, self.style, self.task_description = provider, style, task_description
        self.name = name or f"llm-{style}"
        self.max_input_tokens = max_input_tokens
        self.calls: list[dict] = []

    def build_prompt(self, view: View) -> str:
        if self.style == "independent" and view.n_messages != 1:
            raise ValueError("independent monitor takes exactly one message per view")
        if self.style not in ("independent", "joint", "concat"):
            raise ValueError(self.style)
        return prompts.render(view, self.task_description, with_provenance=(self.style != "concat"))

    def inspect(self, view: View) -> Verdict:
        prompt = self.build_prompt(view)
        truncated = False
        if self.max_input_tokens is not None:
            words = prompt.split()
            if len(words) > self.max_input_tokens:
                prompt = " ".join(words[: self.max_input_tokens]); truncated = True
        t0 = time.time()
        text, usage = self.provider.complete(prompts.SYSTEM, prompt)
        lat = time.time() - t0
        score, flagged, reason = parse_verdict(text)
        self.calls.append({"monitor": view.monitor_id, "edges": [list(e) for e in view.edges],
                           "msg_ids": [o.msg_id for o in view.observations], "prompt": prompt, "raw": text,
                           "score": score, "usage": usage, "latency_s": lat, "truncated": truncated})
        return Verdict(score, flagged, reason, usage.get("input_tokens", 0), usage.get("output_tokens", 0), lat)


def parse_verdict(text: str) -> tuple[float, list[str], str]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return float(d.get("score", 0)) / 100.0, list(d.get("flagged", [])), str(d.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    nums = re.findall(r"\d+", text)
    return (float(nums[0]) / 100.0 if nums else 0.0), [], text[:200]
