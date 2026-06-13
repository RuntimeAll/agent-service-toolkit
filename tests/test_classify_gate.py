# -*- coding: utf-8 -*-
"""PRD-C-014 B1 · classify 两步锚定 + DNA 闸（mock LLM/IO，零网络）。

🔴 核心判据（根治 C-013 凭 LLM 置信裸放行落 0）：
- 池内锚到主 kp → anchored.code 有值 → mother_confirmed=True → gate_after_classify=generate；
- 池内无匹配（main_kp=None）→ anchored 缺失 → mother_confirmed=False → clarify（不放行出题）；
- 库/网络故障（叶子池拉空）→ 锚定不可用 → mother_confirmed=False → clarify，不 silent-fail
  误判为成功；
- subjectId 科目锚 level1（年级册 code）；dim1KpId = 主 kp 叶子 code（DNA 锚定产物）；
- 全维 DNA 穿进 BO（secondaryKpIds/tags/skeleton/scene/examType/hardPoints/anchorId/
  needAnchorReview/reasoning），删除 dim3Skill/auxTags/freeTag。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import classify, gate_after_classify
from agents.variant_support import build_create_bo

_BASE_STATE = {
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.4},  # 低置信，待锚
        "qtype": {"value": "解答", "confidence": 0.5},
    },
    "mother_dna": {"stem": "2x+3=7", "answer": "x=2", "solution_skeleton": "移项"},
}

_POOL = [("3071001001001", "一元一次方程"), ("3071001001002", "合并同类项")]

_GOOD_DNA = {
    "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
    "secondary_kps": [{"id": "3071001001002", "name": "合并同类项"}],
    "qtype": "解答", "exam_type": "直接计算", "skeleton": ["移项", "求解"],
    "hard_points": [], "hard_point_count": 0, "tags": ["解方程", "移项变号"],
    "tag_reused_count": 1, "scene": "纯代数", "difficulty": 2, "flags": [],
}

_FAIL_DNA = {
    "main_kp": None, "secondary_kps": [], "qtype": None, "exam_type": None,
    "skeleton": [], "hard_points": [], "hard_point_count": 0, "tags": [],
    "tag_reused_count": 0, "scene": "", "difficulty": 2,
    "flags": [variant_mod.dna_extract.FLAG_MAIN_KP_OOB],
}


class _FakeClient:
    def __init__(self, token=None):
        pass

    async def aclose(self):
        pass


def _patch(monkeypatch, *, pool, dna):
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)

    async def fake_leaf_pool(grade_code, client, **kw):
        return pool

    async def fake_extract(**kw):
        return dna

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.dna_extract, "extract_dna", fake_extract)

    # 🔴 PRD-C-015 批2：classify 现会跑 model_anchor（连库反查 + LLM 确认）。单测桩成确定性 M00 兜底
    #   （零库零 LLM），避免连真库；models 维非空契约由 batch2 专测覆盖，这里只保 classify 主链不破。
    async def fake_anchor(dna, **kw):
        return {"models": [dict(variant_mod.model_anchor.M00)], "model_overflow": [],
                "model_warn": False, "model_flag": "m00_fallback"}

    monkeypatch.setattr(variant_mod.model_anchor, "anchor_models", fake_anchor)


# ---------------------------------------------------------------------------
# 锚定成功 → generate
# ---------------------------------------------------------------------------
def test_classify_anchors_then_generate(monkeypatch):
    _patch(monkeypatch, pool=_POOL, dna=_GOOD_DNA)
    out = asyncio.run(classify(dict(_BASE_STATE), {}))
    assert out["mother_confirmed"] is True
    assert out["analysis"]["kp"]["anchored"]["code"] == "3071001001001"
    assert gate_after_classify(out) == "generate"
    # DNA 穿进 mother_dna.dna，供 _mother_facts → BO
    assert out["mother_dna"]["dna"]["main_kp"]["id"] == "3071001001001"


# ---------------------------------------------------------------------------
# 🔴 池内无匹配 → clarify（不放行出题）
# ---------------------------------------------------------------------------
def test_classify_no_pool_match_goes_clarify(monkeypatch):
    _patch(monkeypatch, pool=_POOL, dna=_FAIL_DNA)
    out = asyncio.run(classify(dict(_BASE_STATE), {}))
    assert out["mother_confirmed"] is False
    assert not (out["analysis"]["kp"].get("anchored"))  # 锚定缺失
    assert gate_after_classify(out) == "clarify"  # 不出题


# ---------------------------------------------------------------------------
# 🔴 库/网络故障（空池）→ clarify，不 silent-fail 误判成功
# ---------------------------------------------------------------------------
def test_classify_empty_pool_degrades_to_clarify(monkeypatch):
    _patch(monkeypatch, pool=[], dna=_GOOD_DNA)  # 空池：即便 DNA 桩好，也不该锚定
    called = {"extract": False}

    async def fake_extract(**kw):
        called["extract"] = True
        return _GOOD_DNA

    monkeypatch.setattr(variant_mod.dna_extract, "extract_dna", fake_extract)
    out = asyncio.run(classify(dict(_BASE_STATE), {}))
    assert out["mother_confirmed"] is False
    assert gate_after_classify(out) == "clarify"
    assert called["extract"] is False  # 空池直接降级，连 LLM 都不调（不 silent-fail）
    assert "_anchor_error" in out["analysis"]


def test_classify_leaf_pool_fetch_raises_degrades_to_clarify(monkeypatch):
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)

    async def boom(grade_code, client, **kw):
        raise RuntimeError("lazyTree 连不上")

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", boom)
    out = asyncio.run(classify(dict(_BASE_STATE), {}))
    assert out["mother_confirmed"] is False  # 故障不误判为成功
    assert gate_after_classify(out) == "clarify"
    assert "_anchor_error" in out["analysis"]


# ---------------------------------------------------------------------------
# BO 对准新 8 表：新增键 + subjectId/dim1KpId 分级 + 删除三件套
# ---------------------------------------------------------------------------
def test_bo_carries_new_dna_keys_and_split_anchors(monkeypatch):
    _patch(monkeypatch, pool=_POOL, dna=_GOOD_DNA)
    out = asyncio.run(classify(dict(_BASE_STATE), {}))
    facts = variant_mod._mother_facts(out)
    # subjectId = 科目锚 level1（年级册 3071），dim1KpId = 主 kp 叶子 code
    assert facts["subject_id"] == "3071"
    assert facts["dim1_kp_id"] == "3071001001001"

    item = {"stem": "变式", "answer": "x=5", "solution": "解", "qtype": "解答", "difficulty": 2}
    bo = build_create_bo(item, facts)
    assert bo["subjectId"] == "3071"          # 科目锚 level1（非知识点）
    assert bo["dim1KpId"] == "3071001001001"  # 主 kp 叶子
    # B1 新增键（键名钉死）
    assert bo["secondaryKpIds"] == [3071001001002]
    # 🔴 批2：models 维（桩为 M00 兜底）走 `模型:` 标签三轨，续接在普通标签后（零 DDL）。
    assert bo["tags"] == ["解方程", "移项变号", "模型:概念直用"]
    assert bo["skeleton"] == "移项\n求解"
    assert bo["scene"] == "纯代数"
    assert bo["examType"] == "直接计算"
    assert bo["anchorId"] == "3071001001001"
    assert bo["needAnchorReview"] is False
    # 删除的三件套不得出现
    assert "dim3Skill" not in bo and "auxTags" not in bo and "freeTag" not in bo


def test_bo_need_anchor_review_when_dna_oob():
    # 锚定失败的 DNA（主 kp 越界）→ needAnchorReview=True
    facts = {
        "qtype": "解答", "subject_id": "3071", "dim1_kp_id": None,
        "dna": _FAIL_DNA, "mother_difficulty": 2,
    }
    bo = build_create_bo({"stem": "s", "answer": "a", "difficulty": 2}, facts)
    assert bo["needAnchorReview"] is True
    assert bo["subjectId"] == "3071"  # subjectId 仍兜底科目锚（不漏列）
    assert "dim1KpId" not in bo  # 主 kp 没锚到 → 不绑


def test_bo_hard_points_carried():
    facts = {
        "qtype": "解答", "subject_id": "3091", "dim1_kp_id": "3091001",
        "dna": dict(_GOOD_DNA, hard_points=["分类讨论", "构造辅助线"]),
        "mother_difficulty": 4,
    }
    bo = build_create_bo({"stem": "s", "answer": "a", "difficulty": 4}, facts)
    assert bo["hardPoints"] == ["分类讨论", "构造辅助线"]
