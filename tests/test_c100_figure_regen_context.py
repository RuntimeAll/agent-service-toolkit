# -*- coding: utf-8 -*-
"""PRD-C-100 C：图片重生带上下文 —— prev_commands 必须拼进 opus prompt（在上一版命令基础上增量改图）。

桩注入 invoke（捕获发给 opus 的 messages）+ parse_json（恒定降级，避免打 mathfig/node），
只验「prompt 构造」这一层：带 prev_commands+correction → 增量分支；无 prev_commands → 原行为。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_capture():
    """返回 (invoke, captured) —— invoke 记下 HumanMessage 文本，返回触发降级的 JSON。"""
    captured: dict[str, str] = {}

    async def fake_invoke(messages, **kw):
        # messages = [SystemMessage, HumanMessage]；抓人类段落原文
        captured["human"] = str(messages[-1].content)
        captured["system"] = str(messages[0].content)
        # 返回 needs_figure=True → 提前降级返回，不触达 mathfig 渲染（本测只验 prompt）
        return '{"needs_figure": true, "commands": []}'

    return fake_invoke, captured


def _parse_json(text):
    import json
    return json.loads(text)


def _no_budget(monkeypatch):
    """护栏旁路：当日预算未超（否则 compose 早返、根本不构造 prompt）。"""
    from agents import cost_guard

    async def _ok():
        return False

    monkeypatch.setattr(cost_guard, "is_budget_exceeded_async", _ok)


def test_regen_with_prev_commands_injects_basis_and_incremental_instruction(monkeypatch) -> None:
    _no_budget(monkeypatch)
    from agents.figure import compose

    invoke, cap = _make_capture()
    prev = ["A=(0,0)", "B=(4,0)", "C=(2,3)", "t=Polygon(A,B,C)"]
    _run(
        compose.compose_variant_figure(
            stem="把三角形 ABC 顶点 C 上移",
            answer="略",
            invoke=invoke,
            parse_json=_parse_json,
            correction_prompt="顶点 C 再往上挪一点，标出高",
            prev_commands=prev,
            item_id="2",
        )
    )
    human = cap["human"]
    # 上一版命令逐条进了 prompt（增量基准）
    for c in prev:
        assert c in human, f"上一版命令 {c!r} 未拼进 prompt"
    # 修正词在
    assert "顶点 C 再往上挪一点，标出高" in human
    # 增量指令措辞在（在上一版基础上调整，而非从零重画）
    assert "基础上" in human and "从零" in human


def test_regen_without_prev_commands_keeps_original_behavior(monkeypatch) -> None:
    _no_budget(monkeypatch)
    from agents.figure import compose

    invoke, cap = _make_capture()
    _run(
        compose.compose_variant_figure(
            stem="画一个等腰三角形",
            invoke=invoke,
            parse_json=_parse_json,
            correction_prompt="改成等边",
            prev_commands=None,  # 无上一版 → 原行为
            item_id="1",
        )
    )
    human = cap["human"]
    assert "改成等边" in human
    # 原行为分支不喂上一版命令、不出现增量措辞
    assert "上一版配图 GeoGebra 命令" not in human
    assert "基础上" not in human


def test_first_compose_no_correction_has_no_regen_block(monkeypatch) -> None:
    _no_budget(monkeypatch)
    from agents.figure import compose

    invoke, cap = _make_capture()
    _run(
        compose.compose_variant_figure(
            stem="画一个正方形",
            invoke=invoke,
            parse_json=_parse_json,
            item_id="3",
        )
    )
    human = cap["human"]
    assert "画一个正方形" in human
    assert "修正" not in human and "上一版" not in human
