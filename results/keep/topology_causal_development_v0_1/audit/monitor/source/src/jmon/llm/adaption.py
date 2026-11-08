"""Adaption Labs hosted inference (Responses-style API) with a response cache and caps.

Request/response shape follows the adapter reviewed in the collective-vulnerability repo:
POST {API}/responses with {model, input: [{role, content}], temperature,
max_output_tokens, agentic_memory: false, tools: [], tool_choice: "none", store: false};
response.output[].content[].text for type "message"/"output_text"; usage.input_tokens,
usage.output_tokens and optionally usage.cost (credits).

The key is read from ADAPTION_API_KEY (optionally loaded from the file named by
JMON_ENV_FILE, reading only that variable). It is never logged or stored; cached
responses hold prompt hashes, text and usage only.
"""
from __future__ import annotations

import hashlib
import threading
import json
import os
import time
import urllib.request
from dataclasses import dataclass

from .base import CostConfig, BudgetExceeded


class EmptyResponse(RuntimeError):
    """The model returned no output text (e.g. a reasoning model exhausted its output
    budget). Never cached; the call is charged."""

API = "https://api.prod.adaptionlabs.ai/api/v1"


def load_key(var: str = "ADAPTION_API_KEY") -> str:
    if os.environ.get(var):
        return os.environ[var]
    path = os.environ.get("JMON_ENV_FILE") or ".env"
    if os.path.exists(path):
        for line in open(path):
            if line.startswith(var + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not set (export it or point JMON_ENV_FILE at a .env file)")


@dataclass
class AdaptionCost(CostConfig):
    max_credits: float = 0.0   # 0 = no credit cap (token/call caps still apply)


class AdaptionProvider:
    def __init__(self, model: str, cost: AdaptionCost | CostConfig, temperature: float = 0.0,
                 max_output_tokens: int = 200, cache_dir: str = "results/llm_cache", http=None) -> None:
        if not cost.allow_live:
            raise RuntimeError("live provider requested but cost.allow_live is false")
        self.model, self.cost, self.temperature, self.max_output_tokens = model, cost, temperature, max_output_tokens
        self.name = f"adaption:{model}"
        self.cache_dir = os.path.join(cache_dir, model.replace("/", "_"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.http = http or self._http
        self.calls = self.tokens = 0
        self.credits = 0.0
        self.cache_hits = 0
        self.empty = 0
        self._lock = threading.Lock()
        self._key = None

    def _http(self, payload: dict) -> dict:
        if self._key is None:
            self._key = load_key()
        req = urllib.request.Request(API + "/responses", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        return self.complete_messages([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def complete_messages(self, messages: list[dict], max_output_tokens: int | None = None) -> tuple[str, dict]:
        """Chat-style call: ``messages`` is a list of {role, content} passed as the Responses ``input``.

        Same payload, cache and caps as ``complete``; ``complete(system, user)`` is the two-message case,
        so its cache keys are unchanged. ``max_output_tokens`` overrides the provider default per call.
        """
        payload = dict(model=self.model, input=[{"role": m["role"], "content": m["content"]} for m in messages],
                       temperature=self.temperature, max_output_tokens=int(max_output_tokens or self.max_output_tokens),
                       agentic_memory=False, tools=[], tool_choice="none", store=False, stream=False, background=False)
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cpath = os.path.join(self.cache_dir, h + ".json")
        if os.path.exists(cpath):
            self.cache_hits += 1
            d = json.load(open(cpath))
            return d["text"], d["usage"]
        self._precheck()
        result = self.http(payload)
        text = "".join(part["text"] for item in result.get("output", []) if item.get("type") == "message"
                       for part in item.get("content", []) if part.get("type") == "output_text")
        u = result.get("usage") or {}
        if "input_tokens" not in u or "output_tokens" not in u:
            raise RuntimeError("provider omitted usage; refusing further paid requests")
        usage = {"input_tokens": int(u["input_tokens"]), "output_tokens": int(u["output_tokens"]),
                 "cost": float(u["cost"]) if u.get("cost") is not None else None}
        self._charge(usage)
        if not text.strip():
            self.empty += 1
            raise EmptyResponse(f"empty output text ({usage['output_tokens']} output tokens) from {self.model}")
        tmp = cpath + f".tmp{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        json.dump({"text": text, "usage": usage, "model": result.get("model"), "time": time.time()}, open(tmp, "w"))
        os.replace(tmp, cpath)   # atomic, safe with parallel shards
        return text, usage

    def _precheck(self):
        with self._lock:
            if self.cost.max_calls and self.calls >= self.cost.max_calls:
                raise BudgetExceeded("max_calls reached")

    def _charge(self, usage):
        with self._lock:
            self._charge_locked(usage)

    def _charge_locked(self, usage):
        self.calls += 1
        self.tokens += usage["input_tokens"] + usage["output_tokens"]
        if usage.get("cost") is not None:
            self.credits += usage["cost"]
        if self.cost.max_tokens and self.tokens > self.cost.max_tokens:
            raise BudgetExceeded("max_tokens exceeded")
        mc = getattr(self.cost, "max_credits", 0.0)
        if mc and self.credits > mc:
            raise BudgetExceeded("max_credits exceeded")

    def summary(self) -> dict:
        return {"calls": self.calls, "tokens": self.tokens, "credits": self.credits, "cache_hits": self.cache_hits,
                "empty_responses": self.empty}
