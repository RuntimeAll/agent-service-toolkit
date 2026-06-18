# -*- coding: utf-8 -*-
"""PRD-C-017 B1 · 母题 classify 改 opus 合并解题打标 + _mother_facts 接 opus + 闸A/闸B。

逐 gate 覆盖（零网络，纯函数 + monkeypatch LLM）：
- G3：classify 母题调用实际命中 model==claude-opus-4-8（mother_solve_label 档），
      + opus 超时/失败 → SSE error 不静默退 gpt-5.4（fail-fast 已在 settings 启动期断言）。
- G4：opus 解答骨架进 _mother_facts（skeleton 来源 = opus 解答，非 analyze 抄图）。
- G10：闸A 富文本机器验证——坏 LaTeX / 缺表 → 标问题（不直接放行）。
- G11：闸B 锚定·宁空不凑——锚不到确认章内叶子 → 留空 + need_anchor_review（不硬塞）；
      有匹配 → 锚上；越界叶子（前缀不符）→ 拒。
"""

import asyncio
import json

import pytest

import agents.variant as variant_mod
from agents import mother_opus
from agents.variant import classify
from core.settings import assert_mother_solve_hits_opus, settings

_POOL = [
    ("3071001001001", "一元一次方程"),
    ("3071001001002", "合并同类项"),
    ("3081002003004", "八上别章叶子"),  # 不同章前缀，用于越界用例
]

_OPUS_GOOD = {
    "has_figure": False,
    "richText": {"stem": "解方程 $2x+3=7$", "answer": "$x=2$", "analysis": "移项得 $2x=4$"},
    "solvedAnswer": "x=2",
    "dna": {
        "primaryKp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondaryKps": [{"id": "3071001001002", "name": "合并同类项"}],
        "qtype": "解答", "assessmentType": "直接计算",
        "solutionSkeleton": ["移项", "【解一元一次方程】"],
        "hardPointCount": 0, "breakthroughPoints": [],
        "scenario": "纯代数", "difficulty": 2,
        "tags": ["解方程", "移项变号"], "modelCandidates": [],
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


class _FakeClient:
    def __init__(self, token=None):
        pass

    async def aclose(self):
        pass


def _patch_classify(monkeypatch, *, pool, opus_text, capture=None):
    """打桩 classify 的 IO/LLM。capture: dict 收集 solve_and_label 收到的 model（G3 断言）。"""
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_error", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_need_confirm", lambda *a, **k: None)

    async def _no_llm(messages, **kw):  # B2 自愈网 LLM 修复兜底层断网（坏 JSON 用例触达，返回仍坏）
        return "still not json"

    monkeypatch.setattr(variant_mod, "_ainvoke_text", _no_llm)

    async def fake_leaf_pool(grade_code, client, **kw):
        return pool

    async def fake_solve(**kw):
        if capture is not None:
            capture["model"] = kw.get("model")
        if isinstance(opus_text, Exception):
            raise opus_text
        return opus_text

    async def fake_anchor(dna, **kw):
        return {"models": [dict(variant_mod.model_anchor.M00)], "model_overflow": [],
                "model_warn": False, "model_flag": "m00_fallback"}

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.mother_opus, "solve_and_label", fake_solve)
    monkeypatch.setattr(variant_mod.model_anchor, "anchor_models", fake_anchor)


# ===========================================================================
# G3 · 母题调用命中 opus + fail-fast
# ===========================================================================
def test_g3_settings_fail_fast_resolves_opus():
    """启动期 fail-fast：mother_solve_label 档必解析为 claude-opus-4-8（B0 已落，B1 复核）。"""
    assert settings.variant_model("mother_solve_label") == "claude-opus-4-8"
    assert assert_mother_solve_hits_opus(settings) == "claude-opus-4-8"


def test_g3_classify_calls_opus_model(monkeypatch):
    """classify 实际把 mother_solve_label 档（opus）传给 solve_and_label（不靠人眼查日志）。"""
    cap = {}
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False), capture=cap)
    out = asyncio.run(classify(dict(_BASE), {}))
    assert cap["model"] == "claude-opus-4-8"
    assert out["mother_confirmed"] is True


def test_g3_opus_failure_routes_to_needs_confirm_not_silent_fallback(monkeypatch):
    """🔴 PRD-C-100 B2 改契约：opus 超时/异常 → 自愈网（重试≤2）耗尽后**不静默退 gpt-5.4**，
    且**不卡 clarify 死胡同**，而是导向 needs_confirm（弹真章树 picker，老师定章后重锚再来）。
    保留不变量：mother_confirmed=False + _mother_opus_error 记录 + awaiting_mother_confirm=True。"""
    needs = []
    _patch_classify(monkeypatch, pool=_POOL, opus_text=TimeoutError("opus 超时"))
    monkeypatch.setattr(variant_mod, "_emit_need_confirm", lambda p: needs.append(p))
    out = asyncio.run(classify(dict(_BASE), {}))
    assert out["mother_confirmed"] is False
    assert "_mother_opus_error" in out["analysis"]          # 记录错因，不静默
    assert out["awaiting_mother_confirm"] is True           # 可前进的确认态，非死胡同
    assert out.get("awaiting_mother_review") is False       # 清 stale review
    assert len(needs) == 1                                  # 真发了 picker（needConfirm）


def test_g3_opus_returns_non_json_routes_to_needs_confirm(monkeypatch):
    """坏 JSON：确定性引号修复 + LLM 兜底 + 重读一次都救不活 → 自愈耗尽 → needs_confirm（不死循环）。"""
    needs = []
    _patch_classify(monkeypatch, pool=_POOL, opus_text="这不是 JSON，是一段废话")
    monkeypatch.setattr(variant_mod, "_emit_need_confirm", lambda p: needs.append(p))
    out = asyncio.run(classify(dict(_BASE), {}))
    assert out["mother_confirmed"] is False
    assert out["awaiting_mother_confirm"] is True
    assert len(needs) == 1


# ===========================================================================
# G4 · _mother_facts 来源 = opus 解答骨架（非抄图）
# ===========================================================================
def test_g4_mother_facts_skeleton_from_opus(monkeypatch):
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False))
    out = asyncio.run(classify(dict(_BASE), {}))
    # mother_dna 标记来源 = opus，solution_skeleton 被 opus 骨架替换（不再是「抄图骨架(旧)」）
    assert out["mother_dna"]["mother_solve_source"] == "opus"
    assert out["mother_dna"]["solution_skeleton"] == "移项\n【解一元一次方程】"
    assert out["mother_dna"]["solution_skeleton"] != "抄图骨架(旧)"
    assert out["mother_dna"].get("solved_answer") == "x=2"
    # _mother_facts 取的 skeleton = opus 骨架（变式守恒注入引用它）
    facts = variant_mod._mother_facts(out)
    assert facts["skeleton"] == "移项\n【解一元一次方程】"


# ===========================================================================
# G10 · 闸A 富文本机器验证（纯函数）
# ===========================================================================
def test_g10_clean_richtext_passes():
    r = {"stem": "解方程 $2x+3=7$", "answer": "$x=2$", "analysis": "移项 $2x=4$"}
    res = mother_opus.validate_rich_text(r)
    assert res["ok"] is True
    assert res["issues"] == []


def test_g10_unbalanced_dollar_flagged():
    r = {"stem": "坏公式 $2x+3=7", "answer": "x=2", "analysis": ""}
    res = mother_opus.validate_rich_text(r)
    assert res["ok"] is False
    assert any("$" in i for i in res["issues"])


def test_g10_unbalanced_braces_flagged():
    r = {"stem": "$\\frac{1}{2$", "answer": "", "analysis": ""}
    res = mother_opus.validate_rich_text(r)
    assert res["ok"] is False


def test_g10_frac_missing_arg_flagged():
    r = {"stem": "$\\frac x$", "answer": "", "analysis": ""}
    res = mother_opus.validate_rich_text(r)
    assert res["ok"] is False
    assert any("frac" in i for i in res["issues"])


def test_g10_residual_paren_delim_flagged():
    r = {"stem": "残留 \\(x+1\\) 定界", "answer": "", "analysis": ""}
    res = mother_opus.validate_rich_text(r)
    assert res["ok"] is False


def test_g10_missing_table_flagged():
    r = {"stem": "下表数据……（无 table 标签）", "answer": "", "analysis": ""}
    res = mother_opus.validate_rich_text(r, has_table=True)
    assert res["ok"] is False
    assert any("table" in i.lower() for i in res["issues"])


def test_g10_table_present_passes():
    r = {"stem": "数据 <table><tr><td>1</td></tr></table>", "answer": "", "analysis": ""}
    res = mother_opus.validate_rich_text(r, has_table=True)
    assert res["ok"] is True


def test_g10_classify_marks_richtext_review_on_bad_latex(monkeypatch):
    bad = json.loads(json.dumps(_OPUS_GOOD))
    bad["richText"]["stem"] = "坏公式 $2x+3=7"  # $ 不配对
    _patch_classify(monkeypatch, pool=_POOL, opus_text=json.dumps(bad, ensure_ascii=False))
    out = asyncio.run(classify(dict(_BASE), {}))
    assert out["mother_dna"].get("need_richtext_review") is True
    assert "_richtext_issues" in out["analysis"]


# ===========================================================================
# G11 · 闸B 锚定·宁空不凑（纯函数 + 集成）
# ===========================================================================
def _dna(main_id, main_name="考点"):
    return {
        "main_kp": {"id": main_id, "name": main_name}, "secondary_kps": [],
        "qtype": "解答", "exam_type": "直接计算", "skeleton": ["s"], "hard_points": [],
        "hard_point_count": 0, "tags": ["t"], "scene": "纯代数", "difficulty": 2, "flags": [],
    }


def test_g11_anchor_hits_when_in_chapter():
    dna = _dna("3071001001001", "一元一次方程")
    out = mother_opus.anchor_to_chapter(dna, chapter_id="3071", leaf_pool=_POOL)
    assert out["main_kp"]["id"] == "3071001001001"
    assert out["need_anchor_review"] is False


def test_g11_anchor_empty_when_no_match_keeps_real_kp_name():
    """锚不到（池外 id）→ id 留空、保留真实考点名 + need_anchor_review（宁空不凑，不硬塞）。"""
    dna = _dna("9999999999999", "真实但池外的考点")
    out = mother_opus.anchor_to_chapter(dna, chapter_id="3071", leaf_pool=_POOL)
    assert out["main_kp"] == {"id": "", "name": "真实但池外的考点"}
    assert out["need_anchor_review"] is True
    assert variant_mod.dna_extract.FLAG_MAIN_KP_OOB in out["flags"]


def test_g11_anchor_rejects_out_of_chapter_prefix():
    """opus 选的叶子在池内但章前缀不符（八上叶子 vs 七上确认章）→ 拒（不锚），need_anchor_review。"""
    dna = _dna("3081002003004", "八上别章叶子")
    out = mother_opus.anchor_to_chapter(dna, chapter_id="3071", leaf_pool=_POOL)
    assert (out["main_kp"] or {}).get("id") in ("", None)
    assert out["need_anchor_review"] is True


def test_g11_anchor_review_id_rejected_when_not_opened():
    """opus 锚到复习册 id（未开放复习册）→ 拒 + review_oob flag。"""
    pool = _POOL + [("3010001", "复习册叶子")]
    dna = _dna("3010001", "复习考点")
    out = mother_opus.anchor_to_chapter(dna, chapter_id=None, leaf_pool=pool,
                                        include_review_books=False)
    assert out["need_anchor_review"] is True
    assert variant_mod.dna_extract.FLAG_MAIN_KP_REVIEW_OOB in out["flags"]


def test_g11_secondary_out_of_chapter_dropped():
    dna = _dna("3071001001001", "一元一次方程")
    dna["secondary_kps"] = [
        {"id": "3071001001002", "name": "合并同类项"},   # 同章 → 留
        {"id": "3081002003004", "name": "八上别章"},      # 越界 → 丢
    ]
    out = mother_opus.anchor_to_chapter(dna, chapter_id="3071", leaf_pool=_POOL)
    ids = [s["id"] for s in out["secondary_kps"]]
    assert ids == ["3071001001002"]


def test_g11_classify_integration_no_match_goes_clarify(monkeypatch):
    """集成：opus 选池外 kp → classify 闸B 留空 → mother_confirmed=False（不放行硬塞）。"""
    bad = json.loads(json.dumps(_OPUS_GOOD))
    bad["dna"]["primaryKp"] = {"id": "9999999999999", "name": "池外考点"}
    _patch_classify(monkeypatch, pool=_POOL, opus_text=json.dumps(bad, ensure_ascii=False))
    out = asyncio.run(classify(dict(_BASE), {}))
    assert out["mother_confirmed"] is False
    assert not (out["analysis"]["kp"].get("anchored"))


# ===========================================================================
# opus_to_dna 形态归一（克制重算）
# ===========================================================================
def test_opus_to_dna_hard_point_count_recomputed():
    raw = {"dna": {
        "primaryKp": {"id": "x", "name": "n"}, "secondaryKps": [],
        "qtype": "解答", "assessmentType": "直接计算",
        "solutionSkeleton": ["a"], "hardPointCount": 99,  # 谎报
        "breakthroughPoints": ["构造", "分类"], "scenario": "s",
        "difficulty": 9, "tags": ["t"], "modelCandidates": ["套路A"],
    }}
    dna = mother_opus.opus_to_dna(raw)
    assert dna["hard_point_count"] == 2  # 代码重算 = len(breakthroughPoints)，不信自报
    assert dna["difficulty"] == 4        # clamp 到 1~4
    assert dna["model_candidates"] == ["套路A"]


def test_opus_to_dna_exam_type_out_of_set_nulled():
    raw = {"dna": {
        "primaryKp": {"id": "x", "name": "n"}, "secondaryKps": [],
        "qtype": "解答", "assessmentType": "瞎编类型",
        "solutionSkeleton": ["a"], "hardPointCount": 0, "breakthroughPoints": [],
        "scenario": "s", "difficulty": 2, "tags": ["t"], "modelCandidates": [],
    }}
    dna = mother_opus.opus_to_dna(raw)
    assert dna["exam_type"] is None
