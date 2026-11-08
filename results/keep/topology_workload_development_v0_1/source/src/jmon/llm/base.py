"""Provider abstraction: OpenAI, Anthropic, local OpenAI-compatible, and a deterministic mock.

Live providers refuse to run unless ``allow_live`` is set in the cost config, and they
stop once ``max_calls`` or ``max_tokens`` is exceeded. Keys come from environment
variables only (see .env.example); nothing here prints or stores them.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Protocol


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        """Return (text, usage) with usage = {input_tokens, output_tokens}."""
        ...


@dataclass
class CostConfig:
    allow_live: bool = False
    max_calls: int = 0
    max_tokens: int = 0


class BudgetExceeded(RuntimeError):
    pass


class MockProvider:
    """Deterministic: the score is a hash of the prompt unless a keyword is present."""

    name = "mock"

    def __init__(self, suspicious_keywords: tuple[str, ...] = ("OVERRIDE",)) -> None:
        self.keywords = suspicious_keywords
        self.calls = 0

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        self.calls += 1
        h = int(hashlib.sha256(user.encode()).hexdigest(), 16) % 30
        score = 90 if any(k in user for k in self.keywords) else h
        text = f'{{"score": {score}, "flagged": [], "reason": "mock"}}'
        return text, {"input_tokens": len(user.split()) + len(system.split()), "output_tokens": 12}

    def complete_messages(self, messages: list[dict], max_output_tokens: int | None = None) -> tuple[str, dict]:
        """Chat-style mock for agent protocols (e.g. the MESA bridge).

        Deterministic in the last user message. "SYSTEM OVERRIDE" in it yields a high-score verdict JSON;
        otherwise a short customer-service style reply that keeps MESA's protocol going. The reply never
        contains a role name, so MESA's "FOLLOW-UP TO <role>" branch never fires on the mock.
        """
        self.calls += 1
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        h = int(hashlib.sha256(last.encode()).hexdigest(), 16)
        n_in = sum(len(str(m.get("content", "")).split()) for m in messages)
        if "SYSTEM OVERRIDE" in last:
            return '{"score": 95, "flagged": [], "reason": "mock override"}', {"input_tokens": n_in, "output_tokens": 12}
        if h % 2:
            text = (f"Case review {h % 10000}: the request is within the applicable window, so the return is approved. "
                    f"Refund the purchase price to the original payment method and confirm with the customer.")
        else:
            text = (f"Case review {h % 10000}: the request falls outside the applicable window and is denied. "
                    f"Inform the customer and close the case.")
        return text, {"input_tokens": n_in, "output_tokens": len(text.split())}


class _Guarded:
    def __init__(self, cost: CostConfig) -> None:
        if not cost.allow_live:
            raise RuntimeError("live provider requested but cost.allow_live is false")
        self.cost = cost
        self.calls = 0
        self.tokens = 0

    def _charge(self, usage: dict) -> None:
        self.calls += 1
        self.tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if self.cost.max_calls and self.calls > self.cost.max_calls:
            raise BudgetExceeded("max_calls exceeded")
        if self.cost.max_tokens and self.tokens > self.cost.max_tokens:
            raise BudgetExceeded("max_tokens exceeded")


class OpenAICompatibleProvider(_Guarded):
    def __init__(self, model: str, cost: CostConfig, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float = 0.0, max_output_tokens: int = 200) -> None:
        super().__init__(cost)
        from openai import OpenAI  # lazy import
        self.client = OpenAI(api_key=os.environ.get(api_key_env), base_url=base_url)
        self.model, self.temperature, self.max_output_tokens = model, temperature, max_output_tokens
        self.name = f"openai:{model}"

    def complete(self, system, user):
        r = self.client.chat.completions.create(model=self.model, temperature=self.temperature,
                                                max_tokens=self.max_output_tokens,
                                                messages=[{"role": "system", "content": system},
                                                          {"role": "user", "content": user}])
        usage = {"input_tokens": r.usage.prompt_tokens, "output_tokens": r.usage.completion_tokens}
        self._charge(usage)
        return r.choices[0].message.content or "", usage


class AnthropicProvider(_Guarded):
    def __init__(self, model: str, cost: CostConfig, temperature: float = 0.0, max_output_tokens: int = 200) -> None:
        super().__init__(cost)
        import anthropic  # lazy import
        self.client = anthropic.Anthropic()
        self.model, self.temperature, self.max_output_tokens = model, temperature, max_output_tokens
        self.name = f"anthropic:{model}"

    def complete(self, system, user):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_output_tokens, temperature=self.temperature,
                                        system=system, messages=[{"role": "user", "content": user}])
        usage = {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}
        self._charge(usage)
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text"), usage


def make_provider(cfg: dict) -> Provider:
    """cfg: {kind: mock|openai|anthropic|local, model, cost: {allow_live, max_calls, max_tokens}, base_url}"""
    kind = cfg.get("kind", "mock")
    if kind == "mock":
        return MockProvider()
    if kind == "adaption":
        from .adaption import AdaptionProvider, AdaptionCost
        return AdaptionProvider(cfg["model"], AdaptionCost(**cfg.get("cost", {})), temperature=cfg.get("temperature", 0.0),
                                max_output_tokens=cfg.get("max_output_tokens", 200), cache_dir=cfg.get("cache_dir", "results/llm_cache"))
    cost = CostConfig(**cfg.get("cost", {}))
    if kind == "openai":
        return OpenAICompatibleProvider(cfg["model"], cost)
    if kind == "local":
        return OpenAICompatibleProvider(cfg["model"], cost, base_url=cfg.get("base_url", os.environ.get("LOCAL_OPENAI_BASE_URL")),
                                        api_key_env="LOCAL_OPENAI_API_KEY")
    if kind == "anthropic":
        return AnthropicProvider(cfg["model"], cost)
    if kind == "adaption":
        from .adaption import AdaptionProvider, AdaptionCost
        return AdaptionProvider(cfg["model"], AdaptionCost(**cfg.get("cost", {})), temperature=cfg.get("temperature", 0.0),
                                max_output_tokens=cfg.get("max_output_tokens", 200), cache_dir=cfg.get("cache_dir", "results/llm_cache"))
    raise ValueError(kind)
