# -*- coding: utf-8 -*-
"""PRD-C-100 B-converge Round-D 单测：修 B3-配图（item_id int 不再 422 / 失败→needs_figure）
+ B3-perf（generate 套墙钟超时 → 有界降级，绝不拖到 11min / 绝不无界）。

全部 LLM 调用 monkeypatch → 零网络。覆盖：
- B3-配图: VariantFigureInput.item_id 接受 int（FE 传 number）→ pydantic v2 不再 422，
           coerce 成 str 落地（compose 仅用作 stem 后缀，str 安全）。None/str 原样。
- B3-perf: generate 首调超时（TimeoutError）→ ① eager 已稳收若干题 → 有界收尾用 eager 题
           （「先出的先出」D14）；② 一道都没出 → 可读「超时降级」文案 + items=[]，绝不裸抛。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import variant as variant_mod  # noqa: E402
from agents.variant import generate  # noqa: E402

# 复用 pipeline 测试的共享桩 state/items（同一事实源，避免重复造）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_variant_pipeline import (  # noqa: E402
    _FACTS_STATE,
    _GEN_ITEM,
    _patch_gate_chain,
)


def _gen_state(knobs=None):
    state = dict(_FACTS_STATE, messages=[HumanMessage(content="出几道变式")])
    state["knobs"] = {} if knobs is None else knobs
    return state


# ===========================================================================
# B3-配图 · VariantFigureInput.item_id 接受 int（不再 422）
# ===========================================================================
class TestFigureInputItemIdCoerce:
    def test_int_item_id_accepted_and_coerced_to_str(self):
        from service.service import VariantFigureInput

        inp = VariantFigureInput(
            mode="compose_variant", thread_id="t1", ruoyi_token="tok",
            stem="几何题", item_id=3,  # 🔴 FE 传 number —— 旧 str 死锁会 422
        )
        # coerce 成 str（compose 用作 stem 后缀；回显也是 str）
        assert inp.item_id == "3" and isinstance(inp.item_id, str)

    def test_str_item_id_unchanged(self):
        from service.service import VariantFigureInput

        inp = VariantFigureInput(
            mode="compose_variant", thread_id="t1", ruoyi_token="tok",
            stem="几何题", item_id="g1",
        )
        assert inp.item_id == "g1"

    def test_none_item_id_stays_none(self):
        from service.service import VariantFigureInput

        inp = VariantFigureInput(
            mode="crop_mother", thread_id="t1", ruoyi_token="tok",
            image_url="https://o.ss/m.png",
        )
        assert inp.item_id is None


# ===========================================================================
# B3-perf · generate 超时 → 有界降级（绝不无界 / 绝不裸抛）
# ===========================================================================
class TestGenerateTimeoutBounded:
    def test_timeout_with_no_items_emits_readable_degrade(self, monkeypatch):
        """首调超时且一道未出 → items=[] + 可读「超时」文案，绝不抛到上层裸 error。"""
        async def boom(messages, retry=True, *, on_delta=None, **kw):
            raise TimeoutError("wall-clock cap hit")

        monkeypatch.setattr(variant_mod, "_ainvoke_text", boom)
        out = asyncio.run(generate(_gen_state(), {}))
        assert out["items"] == []
        msg = out["messages"][-1].content
        assert "超时" in msg  # 可读降级文案

    def test_timeout_after_eager_keeps_already_emitted(self, monkeypatch):
        """流内已稳收若干完整题后才超时 → 有界收尾用 eager 题（D14 先出的先出），不丢正文。"""
        _patch_gate_chain(monkeypatch)
        parts = [json.dumps(dict(_GEN_ITEM, stem=f"s{i}"), ensure_ascii=False) for i in range(2)]

        async def feed_then_timeout(messages, retry=True, *, on_delta=None, **kw):
            # 先把 2 道完整题喂给 on_delta（驱动 eager 解析 + 派发），再墙钟超时
            if on_delta is not None:
                acc = "["
                for p in parts:
                    acc += p + ","
                    on_delta(acc)
            raise TimeoutError("wall-clock cap hit after stream")

        monkeypatch.setattr(variant_mod, "_ainvoke_text", feed_then_timeout)
        out = asyncio.run(generate(_gen_state(), {}))
        # 有界收尾：已流内稳收的 2 道题保留（不被长尾埋掉），不返回空
        assert [it["stem"] for it in out["items"]] == ["s0", "s1"]

    def test_generate_passes_timeout_kwarg(self, monkeypatch):
        """断言 generate 把 VARIANT_TIMEOUT_GENERATE 作为 timeout= 传进 _ainvoke_text
        （防回归：去掉 timeout= 又退回无界长尾）。"""
        from core import settings as settings_obj  # = settings 实例（from core.settings import settings）

        seen = {"timeout": None}
        parts = [json.dumps(dict(_GEN_ITEM, stem="s0"), ensure_ascii=False)]
        text = "[" + ",".join(parts) + "]"

        async def fake(messages, retry=True, *, on_delta=None, timeout=None, **kw):
            seen["timeout"] = timeout
            return text

        _patch_gate_chain(monkeypatch)
        monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
        asyncio.run(generate(_gen_state(), {}))
        assert seen["timeout"] == settings_obj.VARIANT_TIMEOUT_GENERATE
