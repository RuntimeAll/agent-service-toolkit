# -*- coding: utf-8 -*-
"""PRD-A-021 R2a·闸1（B5）·变式入口预设年级/章注入 端到端单测。

老师在 FE 已选好范围（preset_grade_book + preset_chapter_id，经 config.configurable 传入）→
mother_opus_entry **跳过确认闸**（needs_confirm 强制 False），直接走高置信 finalize，用预设章前缀
圈池 + 锚定。验证：① 不发 needConfirm；② 用预设章前 4 位当 grade_code 圈池；③ confirmed_chapter_id
落进返回（= 预设章）。

全部 LLM/HTTP monkeypatch → 零网络。复用真实 mother_opus / model_anchor 纯函数。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents import variant_entry as VE  # noqa: E402


# opus 一把输出：故意低置信（confidence=0.0 + gradeBook 空）→ **若无预设必弹确认**。
# 有预设时应被预设覆盖、跳过确认。dna 数据齐全，主考点名能在预设章池里锚到叶子。
_ENTRY = {
    "gradeBook": "",
    "chapter": "",
    "confidence": 0.0,
    "has_figure": False,
    "richText": {"stem": "解方程 $x^2-5x+6=0$", "answer": "$x=2$ 或 $x=3$", "analysis": "因式分解"},
    "solvedAnswer": "x=2 或 x=3",
    "dna": {
        "primaryKp": {"id": "", "name": "一元二次方程的解法"},
        "secondaryKps": [], "qtype": "解答", "assessmentType": "公式套用",
        "solutionSkeleton": ["移项", "【因式分解】"], "hardPointCount": 0,
        "breakthroughPoints": [], "scenario": "纯代数", "difficulty": 2,
        "tags": ["一元二次方程"], "modelCandidates": [],
    },
}

# 预设章 3082002 收窄池里有主考点叶子（id 以 3082002 为前缀）。
_LEAF_POOL = [("3082002001", "一元二次方程的解法"), ("3082002005", "配方法")]


async def _noop_aclose():
    return None


def _make_fake_V():
    emitted: dict = {"need_confirm": [], "stages": [], "errors": [], "mother_card": 0, "figure": 0,
                     "leaf_pool_grade_codes": []}

    async def _ainvoke_text(messages, **kw):
        import json
        return json.dumps(_ENTRY, ensure_ascii=False)

    async def _leaf_pool_for_grade(grade_code, client, include_review_books=False):
        emitted["leaf_pool_grade_codes"].append(grade_code)
        return list(_LEAF_POOL)

    async def _resolve_grade_code(analysis):
        return None  # 预设路径不应走到这里（用预设章前缀）

    fakeV = SimpleNamespace(
        STAGE_AWAIT="await", CONF_GATE=0.6,
        _extract_image_url=lambda t: "https://oss.example/q.png",
        _latest_human_text=lambda msgs: "出3道",
        _strip_urls=lambda t: t,
        _wants_review_books=lambda t: False,
        _grade_to_code=lambda v: ("3082" if "八年级下" in str(v or "") else ""),
        _resolve_grade_code=_resolve_grade_code,
        _emit_stage=lambda *a, **k: emitted["stages"].append(a),
        _emit_need_confirm=lambda p: emitted["need_confirm"].append(p),
        _emit_error=lambda *a, **k: emitted["errors"].append(a),
        _emit_reasoning=lambda *a, **k: None,
        _emit_mother_card=lambda st: emitted.__setitem__("mother_card", emitted["mother_card"] + 1),
        _emit_figure_stage=lambda st: emitted.__setitem__("figure", emitted["figure"] + 1),
        _sanitize_rich_text=lambda s: s,
        join_skeleton=lambda lines: "\n".join(str(s) for s in (lines or [])),
        knobs_desc=lambda k: "默认配方",
        _conf_ok=lambda a: True,
        _parse_json=lambda t: __import__("json").loads(t),
        _extract_knobs=lambda st: _async_ret({}),
        _ainvoke_text=_ainvoke_text,
        leaf_pool_for_grade=_leaf_pool_for_grade,
        build_mother_confirm=lambda st: {"needs_confirm": False},
        RuoyiClient=lambda token=None: SimpleNamespace(aclose=_noop_aclose),
        settings=SimpleNamespace(variant_model=lambda key: "m", MOTHER_OPUS_MAX_TOKENS=4096),
    )
    return fakeV, emitted


def _async_ret(v):
    async def _f():
        return v
    return _f()


def _run_entry(monkeypatch, *, configurable):
    fakeV, emitted = _make_fake_V()
    import agents
    monkeypatch.setattr(agents, "variant", fakeV)
    monkeypatch.setitem(sys.modules, "agents.variant", fakeV)

    from agents import cost_guard

    async def _budget_async():
        return False
    monkeypatch.setattr(cost_guard, "is_budget_exceeded_async", _budget_async)

    async def _b64(url):
        return url
    monkeypatch.setattr(VE, "_to_b64_data_url", _b64)

    async def _fetch_memory_block(client):
        return None
    _fake_tm = SimpleNamespace(fetch_memory_block=_fetch_memory_block)
    monkeypatch.setattr(agents, "teacher_memory", _fake_tm, raising=False)
    monkeypatch.setitem(sys.modules, "agents.teacher_memory", _fake_tm)

    state = {"messages": [HumanMessage(content="出3道 https://oss.example/q.png")]}
    out = asyncio.run(VE.mother_opus_entry(state, {"configurable": configurable}))
    return out, emitted


def test_no_preset_low_conf_pops_confirm(monkeypatch):
    """对照：无预设 + 低置信 → 照常弹确认（needs_confirm 路径）。"""
    out, emitted = _run_entry(monkeypatch, configurable={})
    assert out["awaiting_mother_confirm"] is True
    assert len(emitted["need_confirm"]) == 1


def test_preset_skips_confirm_and_finalizes(monkeypatch):
    """🔴 闸1：预设年级册 + 章 → 跳过确认闸（不发 needConfirm）→ 走 finalize 定死出母题卡。"""
    out, emitted = _run_entry(monkeypatch, configurable={
        "preset_grade_book": "八年级下册", "preset_chapter_id": "3082002",
    })
    # 不弹确认（以老师选择为准）
    assert len(emitted["need_confirm"]) == 0
    assert out["awaiting_mother_confirm"] is False
    # 用预设章前 4 位 3082 圈池（不走 opus 判的空 gradeBook）
    assert "3082" in emitted["leaf_pool_grade_codes"]
    # confirmed_chapter_id 落预设章
    assert out["confirmed_chapter_id"] == "3082002"
    # 母题卡先出
    assert emitted["mother_card"] >= 1
    # 高置信定死（锚到预设章内叶子）
    assert out["mother_confirmed"] is True


def test_preset_chapter_only_uses_prefix(monkeypatch):
    """只给 preset_chapter_id（无 grade_book 名）→ 仍用章前 4 位圈池、跳过确认。"""
    out, emitted = _run_entry(monkeypatch, configurable={"preset_chapter_id": "3082002"})
    assert len(emitted["need_confirm"]) == 0
    assert "3082" in emitted["leaf_pool_grade_codes"]
    assert out["confirmed_chapter_id"] == "3082002"


def test_preset_writes_solve_range_fp(monkeypatch):
    """闸2 配合：预设高置信 finalize 也落首解范围指纹 = 圈池 grade_code（册变重解判据要它）。"""
    out, _ = _run_entry(monkeypatch, configurable={"preset_chapter_id": "3082002"})
    assert out["mother_dna"].get("_solve_range_fp") == "3082"
