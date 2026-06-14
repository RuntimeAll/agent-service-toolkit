# -*- coding: utf-8 -*-
"""Unit tests for relay_pool failover semantics (PRD-C-011 Block B hardening):

- single-relay deployments must NEVER be fail-fasted by their own breaker
  (an open breaker with no backup only reduces availability: 30s of guaranteed
  RuntimeError("no relay available") while the relay may already be healthy);
- ainvoke_failover returns the *transacted* relay's per-relay model so
  conv_trace / llm_trace / cost attribution never misreport under failover.

Zero LLM / zero network: _chat is monkeypatched.
"""

import asyncio
import time

import core.relay_pool as rp
from core.relay_pool import Relay, _Breaker, ainvoke_failover


class _FakeChat:
    def __init__(self, resp):
        self._resp = resp

    def bind(self, **_kw):
        return self

    async def ainvoke(self, _messages):
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _wire(monkeypatch, relays, responses):
    """responses: dict relay_name -> resp object or Exception."""
    monkeypatch.setattr(rp, "_relays", lambda: relays)
    monkeypatch.setattr(rp, "_chat", lambda relay: _FakeChat(responses[relay.name]))
    rp._breakers.clear()


def test_single_relay_open_breaker_still_attempts(monkeypatch):
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    _wire(monkeypatch, [relay], {"main": "RESP"})
    # trip the breaker wide open
    rp._breakers["main"] = _Breaker(fails=9, open_until=time.monotonic() + 999)

    resp, name, model, fallback = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert resp == "RESP" and name == "main" and model == "m-main"
    assert fallback == 0
    assert rp._breakers["main"].open_until == 0.0  # success resets the breaker


def test_multi_relay_failover_returns_backup_per_relay_model(monkeypatch):
    main = Relay(name="main", base_url="http://a", api_key="k", model="m-main")
    backup = Relay(name="backup", base_url="http://b", api_key="k", model="m-backup")
    _wire(monkeypatch, [main, backup], {"main": RuntimeError("down"), "backup": "RESP"})

    resp, name, model, fallback = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert resp == "RESP"
    assert name == "backup" and model == "m-backup"  # cost/trace attribution source
    assert fallback == 1


def test_multi_relay_open_main_is_skipped(monkeypatch):
    main = Relay(name="main", base_url="http://a", api_key="k", model="m-main")
    backup = Relay(name="backup", base_url="http://b", api_key="k", model="m-backup")
    calls = []

    def chat(relay):
        calls.append(relay.name)
        return _FakeChat("RESP")

    monkeypatch.setattr(rp, "_relays", lambda: [main, backup])
    monkeypatch.setattr(rp, "_chat", chat)
    rp._breakers.clear()
    rp._breakers["main"] = _Breaker(fails=3, open_until=time.monotonic() + 999)

    resp, name, model, fallback = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert calls == ["backup"]  # open main skipped (breaker semantics intact with a backup)
    assert name == "backup" and model == "m-backup" and fallback == 1


def test_per_call_model_override_swaps_model_only(monkeypatch):
    """S1.1: per-call model 覆盖只换 model 字段，站点不变；relay_model 归因到覆盖模型；
    走 _chat_override（不动整站 _chat 缓存）。model=None 时行为完全不变（旧测已覆盖）。"""
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    monkeypatch.setattr(rp, "_relays", lambda: [relay])
    seen = []

    def chat_override(r, m, temperature=0.5):
        seen.append((r.name, r.base_url, m))
        return _FakeChat("RESP")

    # 若误走整站 _chat（不该），让它炸出来
    monkeypatch.setattr(rp, "_chat", lambda r: (_ for _ in ()).throw(AssertionError("should use override")))
    monkeypatch.setattr(rp, "_chat_override", chat_override)
    rp._breakers.clear()

    resp, name, model, fallback = asyncio.run(
        ainvoke_failover([], max_tokens=10, model="gpt-5.4-nano")
    )
    assert resp == "RESP" and name == "main" and fallback == 0
    assert model == "gpt-5.4-nano"  # 归因到覆盖模型，不再是 relay.model
    assert seen == [("main", "http://x", "gpt-5.4-nano")]  # 站点不变，只换 model


def test_per_call_temperature_override(monkeypatch):
    """PRD-C-017 M9: per-call temperature 覆盖只在 model 覆盖时生效，传给 _chat_override；
    不传 temperature 时默认 0.5（旧行为不变）。母题 opus 档用低温稳 JSON/解题。"""
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    monkeypatch.setattr(rp, "_relays", lambda: [relay])
    seen_temp = []

    def chat_override(r, m, temperature=0.5):
        seen_temp.append(temperature)
        return _FakeChat("RESP")

    monkeypatch.setattr(rp, "_chat_override", chat_override)
    rp._breakers.clear()

    # 显式低温
    asyncio.run(ainvoke_failover([], max_tokens=10, model="claude-opus-4-8", temperature=0.1))
    # 不传温度 → 默认 0.5
    asyncio.run(ainvoke_failover([], max_tokens=10, model="claude-opus-4-8"))
    assert seen_temp == [0.1, 0.5]


def test_chat_override_caches_per_temperature(monkeypatch):
    """温度进缓存键：同 (站|model) 不同温度构出独立实例，不互相覆盖。"""
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    rp._chat_cache.clear()
    c_low = rp._chat_override(relay, "claude-opus-4-8", 0.1)
    c_hi = rp._chat_override(relay, "claude-opus-4-8", 0.5)
    c_low2 = rp._chat_override(relay, "claude-opus-4-8", 0.1)
    assert c_low is c_low2  # 同温度命中缓存
    assert c_low is not c_hi  # 不同温度独立实例
    assert c_low.temperature == 0.1 and c_hi.temperature == 0.5
    rp._chat_cache.clear()
