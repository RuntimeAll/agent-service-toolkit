# -*- coding: utf-8 -*-
"""PRD-C-017 B2 · nano 前置判 年级章必停确认 + 确认章接闸B + 聚合章排除 + 带图打回。

逐 gate 覆盖（零网络，纯函数 + monkeypatch LLM）：
- G1：母题每次必停弹窗——mother_precheck 发 needConfirm 停下（候选为空也停）；带图不发 needConfirm。
- G2：确认章 confirmed_chapter_id 接 anchor_to_chapter（前缀断言含越界）；聚合/复习章排除（M7）。
- G13：带图 → reject 终止流程（后续 opus/classify 零执行）；纯文本不误打回。
"""

import asyncio
import json

import pytest

import agents.variant as variant_mod
from agents import mother_opus, mother_precheck
from agents.variant import classify, mother_precheck_node, route_entry


# ===========================================================================
# 公共桩
# ===========================================================================
class _FakeClient:
    def __init__(self, token=None):
        pass

    async def aclose(self):
        pass


_POOL = [
    ("3071001001001", "一元一次方程"),
    ("3071001001002", "合并同类项"),
    ("3071002005003", "七上别章叶子(同册不同章)"),
    ("3081002003004", "八上别章叶子"),
]

_OPUS_GOOD = {
    "has_figure": False,
    "richText": {"stem": "解方程 $2x+3=7$", "answer": "$x=2$", "analysis": "移项得 $2x=4$"},
    "solvedAnswer": "x=2",
    "dna": {
        "primaryKp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondaryKps": [],
        "qtype": "解答", "assessmentType": "直接计算",
        "solutionSkeleton": ["移项", "【解一元一次方程】"],
        "hardPointCount": 0, "breakthroughPoints": [],
        "scenario": "纯代数", "difficulty": 2,
        "tags": ["解方程"], "modelCandidates": [],
    },
}

_BASE = {
    "image_url": "https://x/q.png",
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.4},
        "qtype": {"value": "解答", "confidence": 0.5},
    },
    "mother_dna": {"stem": "2x+3=7", "answer": "x=2", "solution_skeleton": "抄图骨架(旧)"},
}


def _patch_classify(monkeypatch, *, pool, opus_text, chapter_name=None, capture=None):
    """打桩 classify 的 IO/LLM。chapter_name=chapter_name_for_id 反查结果（M7 用）。
    capture: dict 收集 anchor_to_chapter 收到的 chapter_id（G2 前缀断言）。"""
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_error", lambda *a, **k: None)

    async def fake_leaf_pool(grade_code, client, **kw):
        return pool

    async def fake_solve(**kw):
        if isinstance(opus_text, Exception):
            raise opus_text
        return opus_text

    async def fake_anchor_models(dna, **kw):
        return {"models": [dict(variant_mod.model_anchor.M00)], "model_overflow": [],
                "model_warn": False, "model_flag": "m00_fallback"}

    async def fake_chapter_name(chapter_id, client):
        return chapter_name

    real_anchor = mother_opus.anchor_to_chapter

    def spy_anchor(dna, **kw):
        if capture is not None:
            capture["chapter_id"] = kw.get("chapter_id")
        return real_anchor(dna, **kw)

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.mother_opus, "solve_and_label", fake_solve)
    monkeypatch.setattr(variant_mod.mother_opus, "anchor_to_chapter", spy_anchor)
    monkeypatch.setattr(variant_mod.model_anchor, "anchor_models", fake_anchor_models)
    monkeypatch.setattr(variant_mod, "chapter_name_for_id", fake_chapter_name)


def _patch_precheck(monkeypatch, *, pre_result, calls=None):
    """打桩 mother_precheck_node 的 nano 判定。pre_result = precheck_judge 返回的归一 dict
    （或 Exception 模拟失败）。calls: list 记录 needConfirm/reject 发射（断言只发其一）。"""
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)

    async def fake_judge(**kw):
        if isinstance(pre_result, Exception):
            raise pre_result
        return pre_result

    monkeypatch.setattr(variant_mod.mother_precheck, "precheck_judge", fake_judge)
    if calls is not None:
        monkeypatch.setattr(variant_mod, "_emit_need_confirm",
                            lambda payload: calls.append(("needConfirm", payload)))
        monkeypatch.setattr(variant_mod, "_emit_reject",
                            lambda reason, msg: calls.append(("reject", reason)))


def _pre(grade="七年级上册", chapter="第1章 一元一次方程", has_figure=False,
         gc=None, cc=None, conf=0.8):
    return {
        "grade_book": grade, "chapter": chapter,
        "grade_candidates": gc or [], "chapter_candidates": cc or [],
        "has_figure": has_figure, "confidence": conf,
    }


# ===========================================================================
# G1 · 母题每次必停弹窗（needConfirm；候选空也停）
# ===========================================================================
def test_g1_precheck_emits_need_confirm_and_waits(monkeypatch):
    calls = []
    _patch_precheck(monkeypatch, pre_result=_pre(), calls=calls)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    # 发了 needConfirm、没发 reject、停下等确认
    assert ("needConfirm", calls[0][1]) == calls[0] or calls[0][0] == "needConfirm"
    assert any(c[0] == "needConfirm" for c in calls)
    assert not any(c[0] == "reject" for c in calls)
    assert out["awaiting_mother_confirm"] is True
    assert out["mother_rejected"] is False
    assert out["mother_confirmed"] is False


def test_g1_need_confirm_even_when_candidates_empty(monkeypatch):
    """🔴 决策表反性自检：候选为空（甚至年级/章都判不出）也发 needConfirm，让老师全手选。"""
    calls = []
    _patch_precheck(monkeypatch,
                    pre_result=_pre(grade="", chapter="", gc=[], cc=[], conf=0.0),
                    calls=calls)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    assert any(c[0] == "needConfirm" for c in calls)
    assert out["awaiting_mother_confirm"] is True


def test_g1_precheck_failure_still_stops_for_manual_confirm(monkeypatch):
    """前置判 nano 失败 → 不静默进 opus；退化为全手选 needConfirm（候选空），不卡死。"""
    calls = []
    _patch_precheck(monkeypatch, pre_result=TimeoutError("nano 超时"), calls=calls)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    assert any(c[0] == "needConfirm" for c in calls)
    assert out["awaiting_mother_confirm"] is True
    assert "_precheck_error" in out["analysis"]


def test_g1_route_resume_to_classify_when_confirmed_id_present():
    """needConfirm 停下后，老师经 config 回传确认章 id → route 直奔 classify（chat-resume，无 interrupt）。"""
    state = {**_BASE, "awaiting_mother_confirm": True, "mother_dna": {"stem": "x"}}
    cfg = {"configurable": {"ruoyi_token": "t", "confirmed_chapter_id": "3071001"}}
    # teacher_id_from_token 桩：让 auth 闸过
    import agents.conv_trace as ct
    orig = ct.teacher_id_from_token
    ct.teacher_id_from_token = lambda tok: 5 if tok else None
    try:
        assert route_entry(state, cfg) == "classify"
    finally:
        ct.teacher_id_from_token = orig


def test_g1_route_text_correction_goes_parse_not_classify():
    """老师没回 id、改纯文字纠正 → 不直奔 classify，落 parse（既有在途母题 patch 重锚路径）。"""
    state = {**_BASE, "awaiting_mother_confirm": True, "mother_dna": {"stem": "x"}}
    cfg = {"configurable": {"ruoyi_token": "t"}}  # 无 confirmed_chapter_id
    import agents.conv_trace as ct
    orig = ct.teacher_id_from_token
    ct.teacher_id_from_token = lambda tok: 5 if tok else None
    try:
        assert route_entry(state, cfg) == "parse"
    finally:
        ct.teacher_id_from_token = orig


# ===========================================================================
# G2 · 确认章接闸B（前缀断言含越界） + 聚合章排除（M7）
# ===========================================================================
def test_g2_confirmed_chapter_id_passed_to_anchor(monkeypatch):
    """老师确认章 id 经 config 回传 → classify 把它接到 anchor_to_chapter(chapter_id=...)。"""
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False), capture=cap)
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert cap["chapter_id"] == "3071001"   # 确认章 id（非年级册前缀 3071）接到闸B
    assert out["mother_confirmed"] is True   # opus 主 kp 3071001001001 以 3071001 为前缀 → 锚上
    assert out["confirmed_chapter_id"] == "3071001"
    assert out["awaiting_mother_confirm"] is False


def test_g2_anchor_rejects_out_of_confirmed_chapter_prefix(monkeypatch):
    """确认章=3071001，但 opus 选的叶子是同册别章 3071002... → 越界拒（need_anchor_review），不放行。"""
    bad = json.loads(json.dumps(_OPUS_GOOD))
    bad["dna"]["primaryKp"] = {"id": "3071002005003", "name": "同册别章考点"}
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(bad, ensure_ascii=False), capture=cap)
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert cap["chapter_id"] == "3071001"
    assert out["mother_confirmed"] is False        # 越界 → 不放行
    assert not (out["analysis"]["kp"].get("anchored"))


def test_g2_prefix_locked_not_masked_by_full_pool(monkeypatch):
    """前缀锁到本次确认章——同册别章叶子在池内但前缀不符也不被「全量池兜底」掩盖。"""
    # opus 选 3081...（别册，更明显越界），确认章 3071001 → 必拒
    bad = json.loads(json.dumps(_OPUS_GOOD))
    bad["dna"]["primaryKp"] = {"id": "3081002003004", "name": "八上别册叶子"}
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(bad, ensure_ascii=False), capture=cap)
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert out["mother_confirmed"] is False


def test_g2_aggregation_chapter_excluded(monkeypatch):
    """🔴 M7：确认章是「中考一轮复习」聚合章 → 排除其作锚定范围（降回年级册前缀 3071），标记。"""
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    chapter_name="中考一轮复习", capture=cap)
    cfg = {"configurable": {"confirmed_chapter_id": "3071007"}}  # 聚合章 id
    out = asyncio.run(classify(dict(_BASE), cfg))
    # 闸B 收到的 chapter_id 不是聚合章 id 3071007，而是降回年级册前缀 3071
    assert cap["chapter_id"] == "3071"
    assert "_aggregation_chapter_excluded" in out["analysis"]
    # 主 kp 3071001001001 以 3071 为前缀 → 仍锚上（按年级册范围）
    assert out["mother_confirmed"] is True


def test_g2_non_aggregation_chapter_not_excluded(monkeypatch):
    """普通教学章（非聚合）→ 不排除，chapter_id 收窄到确认章。"""
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    chapter_name="第1章 一元一次方程", capture=cap)
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert cap["chapter_id"] == "3071001"
    assert "_aggregation_chapter_excluded" not in out["analysis"]


def test_g2_no_confirmed_id_falls_back_to_grade_prefix(monkeypatch):
    """没有确认章 id（旧线程/降级）→ 回退年级册 4 位前缀（B1 行为，不回归）。"""
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False), capture=cap)
    out = asyncio.run(classify(dict(_BASE), {}))
    assert cap["chapter_id"] == "3071"  # grade_code 兜底


# M7 纯函数
def test_m7_is_aggregation_chapter_name():
    assert mother_precheck.is_aggregation_chapter_name("中考一轮复习") is True
    assert mother_precheck.is_aggregation_chapter_name("期末专题") is True
    assert mother_precheck.is_aggregation_chapter_name("专题训练") is True
    assert mother_precheck.is_aggregation_chapter_name("第2章 一元二次方程") is False
    assert mother_precheck.is_aggregation_chapter_name("二次根式") is False
    assert mother_precheck.is_aggregation_chapter_name("") is False
    assert mother_precheck.is_aggregation_chapter_name(None) is False


# ===========================================================================
# G13 · 带图打回（reject + 后续零执行） + 纯文本不误杀
# ===========================================================================
def test_g13_with_figure_rejects(monkeypatch):
    """含图 → 发 reject，不发 needConfirm，mother_rejected=True 终止。"""
    calls = []
    _patch_precheck(monkeypatch, pre_result=_pre(has_figure=True), calls=calls)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    assert any(c == ("reject", "with_figure") for c in calls)
    assert not any(c[0] == "needConfirm" for c in calls)
    assert out["mother_rejected"] is True
    assert out["awaiting_mother_confirm"] is False
    assert out["mother_confirmed"] is False


def test_g13_with_figure_does_not_call_opus_or_classify():
    """🔴 带图输入 → 后续节点零执行：图在 mother_precheck 就打回（END），opus/classify 不被调。

    用真 graph 跑：mock precheck 判带图 + spy opus/classify 计数（应为 0）。"""
    import agents.variant as v
    from langchain_core.messages import HumanMessage

    opus_calls = {"n": 0}
    classify_calls = {"n": 0}

    async def fake_judge(**kw):
        return _pre(has_figure=True)

    async def spy_solve(**kw):
        opus_calls["n"] += 1
        return json.dumps(_OPUS_GOOD, ensure_ascii=False)

    real_classify = v.classify

    async def spy_classify(state, config):
        classify_calls["n"] += 1
        return await real_classify(state, config)

    import agents.conv_trace as ct
    orig_tid = ct.teacher_id_from_token

    async def fake_analyze(state, config):
        # 桩 analyze：直接产出 analysis + image_url（不调真 LLM 读图），驱动到 mother_precheck
        return {
            "image_url": "https://x/q.png",
            "analysis": dict(_BASE["analysis"]),
            "mother_dna": dict(_BASE["mother_dna"]),
            "knobs": {}, "shape_defects": [],
            "mother_precheck": None, "awaiting_mother_confirm": False,
            "mother_rejected": False, "confirmed_chapter_id": None,
            "confirmed_grade_book_id": None, "messages": [],
        }

    # 重建一张图（节点替换后重 compile），避免污染模块级 variant
    import importlib
    from langgraph.graph import END, StateGraph

    g = StateGraph(v.VariantState)
    g.add_node("analyze", fake_analyze)
    g.add_node("mother_precheck", v.mother_precheck_node)
    g.add_node("classify", spy_classify)
    g.add_node("generate", lambda s: {"messages": []})

    def route(state, config):
        return "analyze"

    g.set_conditional_entry_point(route, {"analyze": "analyze"})

    def after_analyze(state):
        return "mother_precheck" if state.get("analysis") else "done"

    g.add_conditional_edges("analyze", after_analyze,
                            {"mother_precheck": "mother_precheck", "done": END})
    g.add_edge("mother_precheck", END)
    g.add_edge("classify", "generate")
    g.add_edge("generate", END)
    compiled = g.compile()

    v.mother_precheck.precheck_judge = fake_judge  # type: ignore
    v.mother_opus.solve_and_label = spy_solve  # type: ignore
    ct.teacher_id_from_token = lambda tok: 5
    try:
        res = asyncio.run(compiled.ainvoke(
            {"messages": [HumanMessage(content="https://x/q.png 出几道")]},
            {"configurable": {"ruoyi_token": "t", "thread_id": "tt"}},
        ))
    finally:
        ct.teacher_id_from_token = orig_tid
    assert res["mother_rejected"] is True
    assert opus_calls["n"] == 0       # opus 零执行
    assert classify_calls["n"] == 0   # classify 零执行


def test_g13_plain_text_not_rejected(monkeypatch):
    """纯文本题（has_figure=False）→ 不触发 reject，正常发 needConfirm 进确认流程。"""
    calls = []
    _patch_precheck(monkeypatch, pre_result=_pre(has_figure=False), calls=calls)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    assert not any(c[0] == "reject" for c in calls)
    assert any(c[0] == "needConfirm" for c in calls)
    assert out["mother_rejected"] is False


# ===========================================================================
# normalize_precheck 纯函数（false-positive 防护：has_figure 兜底 False）
# ===========================================================================
def test_normalize_has_figure_defaults_false_when_missing():
    assert mother_precheck.normalize_precheck({})["has_figure"] is False
    assert mother_precheck.normalize_precheck(None)["has_figure"] is False
    # 非布尔 / 字符串 "true" 也不算 true（只认真正的布尔 True，偏保守）
    assert mother_precheck.normalize_precheck({"has_figure": "true"})["has_figure"] is False
    assert mother_precheck.normalize_precheck({"has_figure": 1})["has_figure"] is False
    assert mother_precheck.normalize_precheck({"has_figure": True})["has_figure"] is True


def test_normalize_candidates_and_conf_clamp():
    r = mother_precheck.normalize_precheck({
        "grade_book": " 八年级下册 ", "chapter": "第2章",
        "grade_candidates": ["七下", "", "  "], "chapter_candidates": None,
        "has_figure": False, "confidence": 2.5,
    })
    assert r["grade_book"] == "八年级下册"
    assert r["grade_candidates"] == ["七下"]
    assert r["chapter_candidates"] == []
    assert r["confidence"] == 1.0  # clamp 到 [0,1]
