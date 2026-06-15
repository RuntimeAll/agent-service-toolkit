# -*- coding: utf-8 -*-
"""PRD-C-017 B5-toolkit · 母题卡硬停闸 + 阶段灯中性态 + 母题卡帧答案/解析/难度核齐 + 确认后年级名同步。

四个收尾补丁（零网络，纯函数 + monkeypatch LLM/writer）：
- 问题2·硬停闸：classify 定死 → gate_after_classify=await_review（不直通 generate）；await_review
  节点置 awaiting_mother_review + 不流向 generate；带 start_variants 信号 resume → route_entry 直奔
  generate（不重跑 classify）；无信号停在 review → route 落 parse（不被自动 generate 架空）。
- 问题1·阶段灯中性态：mother_precheck needConfirm 发 STAGE_AWAIT（"await"），不发 warn/已中断；
  await_review 同发 await；带图打回 reject 仍 warn（保留）。
- 问题3·母题卡帧答案/解析/难度核齐：solved_answer 兜底回退 answer；顶层 answer 外显；
  analysis/difficulty 非空。
- 问题2附带·确认后年级名同步：确认章驱动 grade_code 时，grade.value 同步成年级册人话名。
"""

import asyncio
import json

import agents.variant as variant_mod
from agents import mother_opus
from agents.variant import (
    STAGE_AWAIT,
    _build_mother_card,
    await_mother_review,
    classify,
    gate_after_classify,
    mother_precheck_node,
    route_entry,
)


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
]

_OPUS_GOOD = {
    "has_figure": False,
    "richText": {
        "stem": "解方程 $2x+3=7$",
        "answer": "标准答案：$x=2$",
        "analysis": "移项得 $2x=4$，故 $x=2$",
    },
    "solvedAnswer": "x=2",
    "dna": {
        "primaryKp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondaryKps": [],
        "qtype": "解答", "assessmentType": "直接计算",
        "solutionSkeleton": ["移项", "【解一元一次方程】"],
        "hardPointCount": 0, "breakthroughPoints": [],
        "scenario": "纯代数", "difficulty": 3,
        "tags": ["解方程"], "modelCandidates": [],
    },
}

_BASE = {
    "image_url": "https://x/q.png",
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {"value": "一元一次方程", "confidence": 0.4},
        "qtype": {"value": "解答", "confidence": 0.5},
    },
    "mother_dna": {"stem": "2x+3=7", "answer": "x=2", "solution_skeleton": "抄图骨架(旧)"},
}


def _patch_classify(monkeypatch, *, opus_text, chapter_name=None, grade_book_name=None,
                    gc_seen=None):
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_error", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_mother_card", lambda *a, **k: None)

    async def fake_leaf_pool(grade_code, client, **kw):
        if gc_seen is not None:
            gc_seen["gc"] = grade_code
        return list(_POOL)

    async def fake_solve(**kw):
        return opus_text

    async def fake_anchor_models(dna, **kw):
        return {"models": [dict(variant_mod.model_anchor.M00)], "model_overflow": [],
                "model_warn": False, "model_flag": "m00_fallback"}

    async def fake_chapter_name(chapter_id, client):
        # 4 位 = 年级册名；7 位 = 章名
        cid = str(chapter_id or "")
        if len(cid) == 4:
            return grade_book_name
        return chapter_name

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.mother_opus, "solve_and_label", fake_solve)
    monkeypatch.setattr(variant_mod.model_anchor, "anchor_models", fake_anchor_models)
    monkeypatch.setattr(variant_mod, "chapter_name_for_id", fake_chapter_name)


def _capture_stages(monkeypatch):
    """打桩 _emit_stage → 捕获 (key, title, status, detail) 元组列表。"""
    stages: list = []
    monkeypatch.setattr(
        variant_mod, "_emit_stage",
        lambda key, title, status, detail=None: stages.append((key, title, status, detail)),
    )
    return stages


def _patch_auth(monkeypatch):
    import agents.conv_trace as ct
    monkeypatch.setattr(ct, "teacher_id_from_token", lambda tok: 5 if tok else None)


# ===========================================================================
# 问题2·硬停闸：gate_after_classify=await_review，await_review 不流 generate
# ===========================================================================
def test_gate_after_classify_pinned_goes_await_review(monkeypatch):
    _patch_classify(monkeypatch, opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    chapter_name="第1章", grade_book_name="七年级上册")
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert out["mother_confirmed"] is True
    # 🔴 定死后不再 generate，改 await_review（母题卡硬停闸）
    assert gate_after_classify(out) == "await_review"


def test_await_review_node_sets_flag_and_stops(monkeypatch):
    """await_review 节点置 awaiting_mother_review=True，发中性态 await，不出 items（不流 generate）。"""
    stages = _capture_stages(monkeypatch)
    out = asyncio.run(await_mother_review(dict(_BASE), {}))
    assert out["awaiting_mother_review"] is True
    assert out["awaiting_mother_confirm"] is False
    assert "items" not in out  # 不产 items（变式由 generate 出，本节点只停）
    # 阶段灯中性态：await，不是 warn、不含「已中断」
    assert any(s[2] == STAGE_AWAIT for s in stages)
    assert all(s[2] != "warn" for s in stages)


# ===========================================================================
# 问题2·resume：带 start_variants → route_entry 直奔 generate（不重跑 classify）
# ===========================================================================
def test_route_resume_start_variants_goes_generate(monkeypatch):
    _patch_auth(monkeypatch)
    state = {
        **_BASE,
        "awaiting_mother_review": True,
        "mother_confirmed": True,
        "mother_dna": {"stem": "x", "dna": {"main_kp": {"id": "3071001001001"}}},
        # 还没出题
    }
    cfg = {"configurable": {"ruoyi_token": "t", "start_variants": True}}
    assert route_entry(state, cfg) == "generate"


def test_route_review_without_signal_goes_parse_not_generate(monkeypatch):
    """停在 review 但老师没点开始（发别的话）→ 落 parse，**绝不**自动 generate（硬停闸不被架空）。"""
    _patch_auth(monkeypatch)
    state = {
        **_BASE,
        "awaiting_mother_review": True,
        "mother_confirmed": True,
        "mother_dna": {"stem": "x"},
    }
    cfg = {"configurable": {"ruoyi_token": "t"}}  # 无 start_variants
    assert route_entry(state, cfg) == "parse"


def test_route_review_with_items_not_regen_auto(monkeypatch):
    """已出过题（items 非空）→ 不再走 review resume 分支（落既有 parse 编辑路径）。"""
    _patch_auth(monkeypatch)
    state = {
        **_BASE,
        "awaiting_mother_review": True,
        "mother_confirmed": True,
        "mother_dna": {"stem": "x"},
        "items": [{"stem": "v1"}],
    }
    cfg = {"configurable": {"ruoyi_token": "t", "start_variants": True}}
    assert route_entry(state, cfg) == "parse"


# ===========================================================================
# 问题1·阶段灯中性态：needConfirm 发 await（非 warn / 非已中断）；带图 reject 仍 warn
# ===========================================================================
def _patch_precheck(monkeypatch, *, pre_result):
    from agents import mother_precheck

    async def fake_judge(**kw):
        if isinstance(pre_result, Exception):
            raise pre_result
        return pre_result

    monkeypatch.setattr(mother_precheck, "precheck_judge", fake_judge)
    monkeypatch.setattr(variant_mod, "_emit_need_confirm", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_reject", lambda *a, **k: None)


def test_needconfirm_stage_is_await_not_warn(monkeypatch):
    """等老师确认年级章 → 阶段灯发 await（中性），不发 warn、不含「已中断」。"""
    _patch_precheck(monkeypatch, pre_result={
        "grade_book": "七年级上册", "chapter": "第1章",
        "grade_candidates": [], "chapter_candidates": [],
        "has_figure": False, "confidence": 0.8,
    })
    stages = _capture_stages(monkeypatch)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    assert out["awaiting_mother_confirm"] is True
    # 锚定考点灯：await（中性），不是 warn
    classify_stages = [s for s in stages if s[0] == "classify"]
    assert classify_stages, "应发了锚定考点阶段灯"
    assert all(s[2] != "warn" for s in classify_stages), "等确认不该发 warn"
    assert any(s[2] == STAGE_AWAIT for s in classify_stages)
    assert all("已中断" not in (s[3] or "") for s in classify_stages)
    # 🔴 B5-fix3：解析配方灯（knobs）也必须在 needConfirm 暂停时发 await（中性），不是 warn、
    #   不含「已中断」——否则 analyze 残留的 running 会被 FE settleStages 渲成 warn+「已中断」误告警。
    knobs_stages = [s for s in stages if s[0] == "knobs"]
    assert knobs_stages, "needConfirm 暂停应补发解析配方阶段灯（与锚定考点 await 配对）"
    assert all(s[2] != "warn" for s in knobs_stages), "等确认不该让解析配方发 warn"
    assert any(s[2] == STAGE_AWAIT for s in knobs_stages), "解析配方应发 await 中性态"
    assert all("已中断" not in (s[3] or "") for s in knobs_stages)


def test_with_figure_reject_keeps_warn(monkeypatch):
    """带图打回是真要拦 → 仍发 warn（保留）。"""
    _patch_precheck(monkeypatch, pre_result={
        "grade_book": "七上", "chapter": "第1章",
        "grade_candidates": [], "chapter_candidates": [],
        "has_figure": True, "confidence": 0.9,
    })
    stages = _capture_stages(monkeypatch)
    out = asyncio.run(mother_precheck_node(dict(_BASE), {}))
    assert out["mother_rejected"] is True
    assert any(s[2] == "warn" for s in stages if s[0] == "classify")


# ===========================================================================
# 问题3·母题卡帧答案/解析/难度核齐
# ===========================================================================
def _full_state():
    dna = mother_opus.opus_to_dna(_OPUS_GOOD)
    dna = mother_opus.anchor_to_chapter(dna, chapter_id="3071001", leaf_pool=_POOL)
    dna["models"] = [{"id": "M01", "name": "方程通解"}]
    mdna = {
        "stem": "解方程 $2x+3=7$",
        "answer": "标准答案：$x=2$",
        "analysis": "移项得 $2x=4$",
        "solved_answer": "x=2",
        "solution_skeleton": "移项\n【解一元一次方程】",
        "difficulty": 3,
        "dna": dna,
    }
    return {**_BASE, "mother_dna": mdna, "confirmed_chapter_id": "3071001"}


def test_card_answer_analysis_difficulty_present():
    card = _build_mother_card(_full_state())
    assert card["answer"] == "标准答案：$x=2$"      # 顶层标准答案外显
    assert card["solved_answer"]                     # solvedAnswer
    assert card["analysis"] == "移项得 $2x=4$"       # 解析非空
    assert card["dna"]["difficulty"] == 3            # 难度在 dna
    assert card["difficulty"] == 3                   # 难度顶层也在


def test_card_solved_answer_falls_back_to_answer():
    """🔴 问题3 根因修：opus 把标准答案放 richText.answer、solvedAnswer 留空时，
    solved_answer 兜底回退 answer（母题卡答案区永不空）。"""
    st = _full_state()
    st["mother_dna"]["solved_answer"] = ""  # opus solvedAnswer 空（真机症状）
    card = _build_mother_card(st)
    assert card["answer"] == "标准答案：$x=2$"
    assert card["solved_answer"] == "标准答案：$x=2$"  # 回退 answer，不再空


def test_card_difficulty_from_mdna_when_dna_missing():
    """dna 无 difficulty → 回退 mdna.difficulty（不空）。"""
    st = _full_state()
    st["mother_dna"]["dna"].pop("difficulty", None)
    card = _build_mother_card(st)
    assert card["difficulty"] == 3


# ===========================================================================
# 问题2附带·确认后年级名同步：grade.value = 老师确认的年级册名
# ===========================================================================
def test_confirmed_grade_book_name_synced(monkeypatch):
    """确认八上（3081）→ grade.value 同步成「八年级上册」（lazyTree 反查册名），
    不再是 analyze 误读的「七年级上学期」。"""
    gc_seen = {}
    _patch_classify(monkeypatch, opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    chapter_name="第1章 一元一次方程", grade_book_name="八年级上册",
                    gc_seen=gc_seen)
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert out["analysis"]["grade"]["value"] == "八年级上册"  # 同步成确认册名
    assert out["analysis"]["grade"]["code"] == "3071"          # code 仍来自确认章前缀


def test_confirmed_grade_book_name_from_config(monkeypatch):
    """FE 回传 grade_book_name 时优先用它（免一次树查）。"""
    _patch_classify(monkeypatch, opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    chapter_name="第1章", grade_book_name="树查册名(不该用)")
    cfg = {"configurable": {"confirmed_chapter_id": "3071001",
                            "grade_book_name": "FE回传册名"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert out["analysis"]["grade"]["value"] == "FE回传册名"
