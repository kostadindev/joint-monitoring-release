import pytest
from jmon.llm.adaption import AdaptionProvider, AdaptionCost
from jmon.llm.base import BudgetExceeded


def fake_http_factory(log):
    def http(payload):
        log.append(payload)
        assert payload["tools"] == [] and payload["store"] is False and payload["agentic_memory"] is False
        return {"model": payload["model"], "output": [{"type": "reasoning"}, {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": '{"score": 12, "flagged": [], "reason": "ok"}'}]}],
                "usage": {"input_tokens": 50, "output_tokens": 10, "cost": 0.0001}}
    return http


def test_adaption_provider_caches_and_caps(tmp_path):
    log = []
    p = AdaptionProvider("qwen3.8-27b", AdaptionCost(allow_live=True, max_calls=2, max_credits=1.0), cache_dir=str(tmp_path),
                         http=fake_http_factory(log))
    t, u = p.complete("sys", "hello")
    assert "score" in t and u["input_tokens"] == 50 and p.calls == 1
    t2, _ = p.complete("sys", "hello")          # cached: no HTTP call
    assert t2 == t and len(log) == 1 and p.cache_hits == 1
    p.complete("sys", "other")
    with pytest.raises(BudgetExceeded):
        p.complete("sys", "third")
    assert len(log) == 2
    assert "Bearer" not in open(next(tmp_path.rglob("*.json"))).read()


def test_adaption_refuses_without_allow_live(tmp_path):
    with pytest.raises(RuntimeError):
        AdaptionProvider("x", AdaptionCost(allow_live=False), cache_dir=str(tmp_path))


def test_make_provider_accepts_credit_cap(tmp_path):
    from jmon.llm import make_provider
    p = make_provider({"kind": "adaption", "model": "m", "cache_dir": str(tmp_path),
                       "cost": {"allow_live": True, "max_calls": 1, "max_credits": 0.1}})
    assert p.cost.max_credits == 0.1


def test_empty_response_raises_and_is_not_cached(tmp_path):
    from jmon.llm.adaption import EmptyResponse
    def http(payload):
        return {"model": payload["model"], "output": [{"type": "reasoning"}],
                "usage": {"input_tokens": 10, "output_tokens": 1024, "cost": 0.001}}
    p = AdaptionProvider("m", AdaptionCost(allow_live=True), cache_dir=str(tmp_path), http=http)
    with pytest.raises(EmptyResponse):
        p.complete("s", "u")
    assert not list(tmp_path.rglob("*.json")) and p.calls == 1 and p.empty == 1
