# -*- coding: utf-8 -*-
"""PRD-C-100 B-converge roundE · 节点级单一墙钟回归（修压轴 perf 长尾根因）。

root cause：relay_pool.ainvoke_failover 的 asyncio.timeout 包在 `for relay` 循环【内】=
每站各 timeout 秒；叠上 _ainvoke_text 的空返重试（再调一次 failover）→ 节点级有效上限
≈ timeout × 站数 × (1+重试)，无单一墙钟 → generate 的 except TimeoutError 降级分支触发不到，
压轴几何变式 generate 拖到 9-11min 不收尾。

修：_ainvoke_text 用【一个】asyncio.timeout(timeout) 把「整个 failover + 空返重试」包成
节点级总预算（option ②）。本测断言节点 wall-clock ≤ timeout（不是 ×站数×重试），
且超时抛 TimeoutError（让 generate 既有降级分支接住）。

零网络：monkeypatch relay_pool.ainvoke_failover 模拟「每站都慢/空返」。
"""

import asyncio
import time

import pytest

import agents.variant as variant_mod
from agents.variant import _ainvoke_text
from langchain_core.messages import AIMessage, HumanMessage


def test_node_wall_clock_caps_total_not_per_station(monkeypatch):
    """🔴 核心断言：每站都慢（每次 failover 调用睡 timeout 秒）+ 空返会触发重试 →
    旧实现节点级有效上限 ≈ timeout×重试；新实现【一个】墙钟把总时长砍到 ≤ timeout。
    timeout=1（秒级跑完）。每次 ainvoke_failover 模拟「睡满预算后空返」（最坏情况：
    既慢又空返，旧实现会再调一次 failover 又睡 1s = 2s）。"""
    calls = {"n": 0}

    async def slow_empty_failover(*_a, **_kw):
        # 每次调用都睡很久（远超节点预算）→ 模拟「每站都慢」；返回空文本本会触发空返重试
        calls["n"] += 1
        await asyncio.sleep(30)
        return AIMessage(content=""), "main", "m-main", 0, None

    monkeypatch.setattr(variant_mod.relay_pool, "ainvoke_failover", slow_empty_failover)

    t0 = time.monotonic()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        asyncio.run(_ainvoke_text([HumanMessage(content="x")], timeout=1))
    elapsed = time.monotonic() - t0

    # 节点级单一墙钟：总耗时 ≤ timeout(1s) + 余量；绝不是 ×重试(=2s) 也不是 ×站数×重试
    assert elapsed < 1.6, f"node wall-clock {elapsed:.2f}s exceeded single budget (timeout=1s)"
    # 第二次 failover（空返重试）压根没机会跑——第一次就把整个节点预算耗尽并抛超时
    assert calls["n"] == 1, f"retry should be cut by node wall-clock, got {calls['n']} failover calls"


def test_node_wall_clock_timeout_raises_timeouterror(monkeypatch):
    """超时必须抛 TimeoutError（而非其它异常）→ generate 的 `except (TimeoutError,
    asyncio.TimeoutError)` 降级分支才接得住（eager 已出题先出 / 零题给可读文案）。"""

    async def hang_failover(*_a, **_kw):
        await asyncio.sleep(30)
        return AIMessage(content="late"), "main", "m-main", 0, None

    monkeypatch.setattr(variant_mod.relay_pool, "ainvoke_failover", hang_failover)

    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        asyncio.run(_ainvoke_text([HumanMessage(content="x")], timeout=1))


def test_no_timeout_keeps_unbounded_legacy_behavior(monkeypatch):
    """timeout=None（绝大多数调用）→ 不包墙钟，旧无界行为不变：慢吐照样等到返回，不被砍。"""

    async def slowish_ok(*_a, **_kw):
        await asyncio.sleep(0.3)  # 比任何 per-call 默认都短，但若误加了小墙钟会被砍
        return AIMessage(content="OK"), "main", "m-main", 0, None

    monkeypatch.setattr(variant_mod.relay_pool, "ainvoke_failover", slowish_ok)

    text = asyncio.run(_ainvoke_text([HumanMessage(content="x")], timeout=None))
    assert text == "OK"


def test_empty_return_retry_still_runs_within_budget(monkeypatch):
    """空返重试路径未被破坏：节点预算够时（每次调用很快），空返 → 重试一次 → 第二次拿到正文。
    断言两次 failover 都跑了、最终返回非空（option ② 只加总墙钟，不改重试语义）。"""
    calls = {"n": 0}

    async def empty_then_ok(*_a, **_kw):
        calls["n"] += 1
        # 第一次空返（触发重试），第二次返回正文
        return (AIMessage(content="" if calls["n"] == 1 else "RECOVERED"),
                "main", "m-main", 0, None)

    monkeypatch.setattr(variant_mod.relay_pool, "ainvoke_failover", empty_then_ok)

    text = asyncio.run(_ainvoke_text([HumanMessage(content="x")], timeout=10))
    assert text == "RECOVERED"
    assert calls["n"] == 2, f"empty-return retry should fire once, got {calls['n']} calls"
