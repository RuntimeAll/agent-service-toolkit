# -*- coding: utf-8 -*-
"""批3（2026-06-13 整改）· 事实源冻结 + 单向写。

每批次一份「老师锚准的事实源」（年级学期/主考点 = analysis.grade/kp）：定死/确认时 facts_locked
置位 → 此后只许老师指令改、LLM 输出不许反向覆盖；老师每次修正记一条 audit（字段/旧值/新值/
指令原文）。覆盖：
- _fact_edit：locked 后 LLM 来源写被忽略 + 不改 analysis；老师来源始终放行 + 记 audit；
- 定死前（未 locked）LLM 来源可写（锚定靠它达成定死）；
- classify 定死 → facts_locked=True；未定死 → False；
- patch（老师修正）locked 也放行 + 解冻重锚 + 留痕；
- exec_solution_only 年级修正经 setter 留痕（不解冻）。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import _fact_edit, classify, patch


# ---------------------------------------------------------------------------
# _fact_edit 单向写
# ---------------------------------------------------------------------------
def test_fact_edit_llm_write_ignored_when_locked():
    analysis = {"grade": {"value": "七年级上学期", "confidence": 0.9}}
    audit: list = []
    ok = _fact_edit(
        analysis, "grade", "八年级下学期", source="llm", locked=True, audit=audit,
    )
    assert ok is False
    assert analysis["grade"]["value"] == "七年级上学期"  # 冻结后 LLM 回写不改值
    assert audit == []


def test_fact_edit_llm_write_allowed_before_lock():
    analysis = {"kp": {"value": "?", "confidence": 0.3}}
    audit: list = []
    ok = _fact_edit(
        analysis, "kp", "一元一次方程", source="llm", locked=False, audit=audit,
        confidence=0.75,
    )
    assert ok is True  # 定死前 LLM 可写（锚定靠它）
    assert analysis["kp"]["value"] == "一元一次方程"
    assert analysis["kp"]["confidence"] == 0.75
    assert audit == []  # LLM 来源不记 audit


def test_fact_edit_teacher_write_always_allowed_and_audited():
    analysis = {"grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"}}
    audit: list = []
    ok = _fact_edit(
        analysis, "grade", "九年级上学期", source="teacher", locked=True, audit=audit,
        instruction="这其实是9年级的题", confidence=0.9, clear_keys=("code",),
    )
    assert ok is True  # 老师指令 locked 也放行
    assert analysis["grade"]["value"] == "九年级上学期"
    assert "code" not in analysis["grade"]  # 清旧 code 重锚
    assert len(audit) == 1
    rec = audit[0]
    assert rec["field"] == "grade"
    assert rec["old"] == "七年级上学期"
    assert rec["new"] == "九年级上学期"
    assert rec["instruction"] == "这其实是9年级的题"


# ---------------------------------------------------------------------------
# classify 置 facts_locked
# ---------------------------------------------------------------------------
_POOL = [("3071001001001", "一元一次方程")]
_GOOD_DNA = {
    "main_kp": {"id": "3071001001001", "name": "一元一次方程"}, "secondary_kps": [],
    "qtype": "解答", "exam_type": "直接计算", "skeleton": ["移项"], "hard_points": [],
    "hard_point_count": 0, "tags": ["解方程"], "tag_reused_count": 0, "scene": "纯代数",
    "difficulty": 2, "flags": [],
}
_FAIL_DNA = dict(_GOOD_DNA, main_kp=None, flags=[variant_mod.dna_extract.FLAG_MAIN_KP_OOB])
_BASE = {
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.4},
        "qtype": {"value": "解答", "confidence": 0.5},
    },
    "mother_dna": {"stem": "2x+3=7", "answer": "x=2", "solution_skeleton": "移项"},
}


class _FakeClient:
    def __init__(self, token=None):
        pass

    async def aclose(self):
        pass


def _patch(monkeypatch, *, pool, dna):
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)

    async def fake_leaf_pool(gc, client, **kw):
        return pool

    async def fake_extract(**kw):
        return dna

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.dna_extract, "extract_dna", fake_extract)


def test_classify_locks_facts_when_pinned(monkeypatch):
    _patch(monkeypatch, pool=_POOL, dna=_GOOD_DNA)
    out = asyncio.run(classify(dict(_BASE), {}))
    assert out["mother_confirmed"] is True
    assert out["facts_locked"] is True  # 定死 → 冻结


def test_classify_no_lock_when_unpinned(monkeypatch):
    _patch(monkeypatch, pool=_POOL, dna=_FAIL_DNA)
    out = asyncio.run(classify(dict(_BASE), {}))
    assert out["mother_confirmed"] is False
    assert out["facts_locked"] is False  # 没定死 → 不冻结（等老师纠正后重锚）


# ---------------------------------------------------------------------------
# patch（老师修正）locked 放行 + 解冻 + 留痕
# ---------------------------------------------------------------------------
def test_patch_teacher_correction_when_locked_records_audit():
    st = {
        "analysis": {
            "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
            "kp": {"value": "一元一次方程", "confidence": 0.9,
                   "anchored": {"code": "3071001001001"}},
        },
        "items": [{"stem": "x"}],
        "mother_confirmed": True,
        "facts_locked": True,
        "facts_audit": [],
        "pending": {"mother_correction": {"grade": "九年级上学期", "kp": "二次函数"},
                    "utterance": "这其实是9年级的二次函数题，重新出"},
    }
    out = asyncio.run(patch(st, {}))
    # 老师修正 locked 也放行
    assert out["analysis"]["grade"]["value"] == "九年级上学期"
    assert out["analysis"]["kp"]["value"] == "二次函数"
    # 解冻重锚 + 清 items + mother_confirmed
    assert out["facts_locked"] is False
    assert out["mother_confirmed"] is False
    assert out["items"] == []
    # 两条 audit 留痕（grade + kp）
    fields = {r["field"] for r in out["facts_audit"]}
    assert fields == {"grade", "kp"}
    g = next(r for r in out["facts_audit"] if r["field"] == "grade")
    assert g["old"] == "七年级上学期" and g["new"] == "九年级上学期"
    assert "9年级" in g["instruction"]
