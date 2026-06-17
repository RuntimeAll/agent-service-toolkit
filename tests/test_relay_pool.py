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
from langchain_core.messages import AIMessage


class _FakeChat:
    def __init__(self, resp):
        self._resp = resp

    def bind(self, **_kw):
        return self

    async def ainvoke(self, _messages):
        if isinstance(self._resp, Exception):
            raise self._resp
        # 🔴 包成 AIMessage（有 .content）：relay_pool 空白返回闸用 resp.content 判真空，裸 str 会被误判
        return AIMessage(content=self._resp) if isinstance(self._resp, str) else self._resp


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

    resp, name, model, fallback, _d = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert resp.content == "RESP" and name == "main" and model == "m-main"
    assert fallback == 0
    assert rp._breakers["main"].open_until == 0.0  # success resets the breaker


def test_multi_relay_failover_returns_backup_per_relay_model(monkeypatch):
    main = Relay(name="main", base_url="http://a", api_key="k", model="m-main")
    backup = Relay(name="backup", base_url="http://b", api_key="k", model="m-backup")
    _wire(monkeypatch, [main, backup], {"main": RuntimeError("down"), "backup": "RESP"})

    resp, name, model, fallback, _d = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert resp.content == "RESP"
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

    resp, name, model, fallback, _d = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert calls == ["backup"]  # open main skipped (breaker semantics intact with a backup)
    assert name == "backup" and model == "m-backup" and fallback == 1


def test_per_call_model_override_swaps_model_only(monkeypatch):
    """S1.1: per-call model 覆盖只换 model 字段，站点不变；relay_model 归因到覆盖模型；
    走 _chat_override（不动整站 _chat 缓存）。model=None 时行为完全不变（旧测已覆盖）。"""
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    monkeypatch.setattr(rp, "_relays", lambda: [relay])
    seen = []

    def chat_override(r, m, temperature=0.5, timeout=None):
        seen.append((r.name, r.base_url, m))
        return _FakeChat("RESP")

    # 若误走整站 _chat（不该），让它炸出来
    monkeypatch.setattr(rp, "_chat", lambda r: (_ for _ in ()).throw(AssertionError("should use override")))
    monkeypatch.setattr(rp, "_chat_override", chat_override)
    rp._breakers.clear()

    resp, name, model, fallback, _d = asyncio.run(
        ainvoke_failover([], max_tokens=10, model="gpt-5.4-nano")
    )
    assert resp.content == "RESP" and name == "main" and fallback == 0
    assert model == "gpt-5.4-nano"  # 归因到覆盖模型，不再是 relay.model
    assert seen == [("main", "http://x", "gpt-5.4-nano")]  # 站点不变，只换 model


def test_per_call_temperature_override(monkeypatch):
    """PRD-C-017 M9: per-call temperature 覆盖只在 model 覆盖时生效，传给 _chat_override；
    不传 temperature 时默认 0.5（旧行为不变）。母题 opus 档用低温稳 JSON/解题。"""
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    monkeypatch.setattr(rp, "_relays", lambda: [relay])
    seen_temp = []

    def chat_override(r, m, temperature=0.5, timeout=None):
        seen_temp.append(temperature)
        return _FakeChat("RESP")

    monkeypatch.setattr(rp, "_chat_override", chat_override)
    rp._breakers.clear()

    # 显式低温
    asyncio.run(ainvoke_failover([], max_tokens=10, model="claude-opus-4-8", temperature=0.1))
    # 不传温度 → 默认 0.5
    asyncio.run(ainvoke_failover([], max_tokens=10, model="claude-opus-4-8"))
    assert seen_temp == [0.1, 0.5]


def test_response_format_and_timeout_threaded(monkeypatch):
    """PRD-C-017 B1·F3/H4：response_format 经 chat.bind 进请求；timeout 经 _chat_override
    传到重建 chat（母题 opus 合并调用硬锁 10 维 schema + ≤180s 防挂死）。"""
    relay = Relay(name="main", base_url="http://x", api_key="k", model="m-main")
    monkeypatch.setattr(rp, "_relays", lambda: [relay])
    seen_timeout = []
    bind_kw_seen = {}

    class _Rec:
        def bind(self, **kw):
            bind_kw_seen.update(kw)
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content="RESP")

    def chat_override(r, m, temperature=0.5, timeout=None):
        seen_timeout.append(timeout)
        return _Rec()

    monkeypatch.setattr(rp, "_chat_override", chat_override)
    rp._breakers.clear()
    rf = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    asyncio.run(ainvoke_failover(
        [], max_tokens=10, model="claude-opus-4-8", temperature=0.1,
        response_format=rf, timeout=180,
    ))
    assert seen_timeout == [180]
    assert bind_kw_seen.get("response_format") == rf
    assert bind_kw_seen.get("max_tokens") == 10


class _Chunk:
    """伪 langchain 流式 chunk：content + additional_kwargs（+ 可选 usage）。"""

    def __init__(self, content="", usage=False):
        self.content = content
        self.additional_kwargs = {}
        if usage:
            self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}

    def __add__(self, other):
        return _Chunk(self.content + (other.content or ""))


def test_wall_clock_cap_catches_slow_dribble_and_failsover(monkeypatch):
    """🔴 round0 真因回归：sui-xiang「慢吐 reasoning 永不收尾」会持续重置 httpx read-gap 计时器
    （read 闸不触发），必须靠 asyncio 墙钟硬闸砍掉 → 切备用站。验证 fallback_detail 标
    WallClockTimeout（区别于 read 闸的 ReadTimeout）。timeout=1 让墙钟 1s 触发，测试秒级跑完。"""
    main = Relay(name="main", base_url="http://a", api_key="k", model="m-main")
    backup = Relay(name="backup", base_url="http://b", api_key="k", model="m-backup")

    class _SlowDribble:  # 主站：吐一帧后睡死（模拟慢吐，read-gap 永不触发）
        def bind(self, **_kw):
            return self

        async def astream(self, _messages, config=None):
            yield _Chunk("partial")
            await asyncio.sleep(30)  # > total_cap(1s) → 墙钟闸砍
            yield _Chunk("never")

    class _OkStream:  # 备站：正常吐
        def bind(self, **_kw):
            return self

        async def astream(self, _messages, config=None):
            yield _Chunk("OK", usage=True)

    monkeypatch.setattr(rp, "_relays", lambda: [main, backup])
    monkeypatch.setattr(
        rp, "_chat", lambda relay: _SlowDribble() if relay.name == "main" else _OkStream()
    )
    rp._breakers.clear()

    # on_delta 给了 → 走 astream（慢吐挂起只在流式路径出现）；timeout=1 = 墙钟 1s
    resp, name, model, fallback, detail = asyncio.run(
        ainvoke_failover([], max_tokens=10, on_delta=lambda _t: None, timeout=1)
    )
    assert name == "backup" and model == "m-backup"  # 慢吐被墙钟砍 → 切备用
    assert resp.content == "OK"
    assert fallback == 1
    assert detail and "main:WallClockTimeout" in detail  # 回溯证据：主站慢吐被硬闸砍


def test_truncated_finish_length_failsover(monkeypatch):
    """🔴 H3（审计补）截断哨兵回归：主站返「非空但 finish_reason=length」（截断坏 JSON）
    现过 blank 检测被当成功 → 必须当失败切备用站；fallback_detail 标 Truncated（区别于 blank/timeout）。"""
    main = Relay(name="main", base_url="http://a", api_key="k", model="m-main")
    backup = Relay(name="backup", base_url="http://b", api_key="k", model="m-backup")

    class _TruncChat:  # 主站：内容非空但被截断（finish_reason=length）
        def bind(self, **_kw):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content='{"stem":"半截 JSON 没收', response_metadata={"finish_reason": "length"})

    class _OkChat:  # 备站：完整收尾
        def bind(self, **_kw):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content='{"stem":"完整"}', response_metadata={"finish_reason": "stop"})

    monkeypatch.setattr(rp, "_relays", lambda: [main, backup])
    monkeypatch.setattr(rp, "_chat", lambda relay: _TruncChat() if relay.name == "main" else _OkChat())
    rp._breakers.clear()

    resp, name, model, fallback, detail = asyncio.run(ainvoke_failover([], max_tokens=10))
    assert name == "backup" and model == "m-backup"  # 截断被判失败 → 切备用
    assert resp.content == '{"stem":"完整"}'
    assert fallback == 1
    assert detail and "main:Truncated" in detail  # 回溯标注：主站截断


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
