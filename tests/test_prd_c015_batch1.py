# -*- coding: utf-8 -*-
"""PRD-C-015 批1·地基层单测（DNA 契约 v2 + 守恒维确定性异常 + 合并确认闸 + facts_locked 扩维）。

覆盖 gate：G10/G11（合并确认闸/确定性异常门控）、G16（契约 v2 字段）、G18（facts_locked 扩 4 维）。

铁律对照（PRD §3.4 / §10.1 / D-merge7 / 缺口5/6）：
- 守恒维门控只看**确定性异常**（副考点越界/考察类型出闭集/骨架为空），绝不引 LLM 自报置信。
- 合并确认闸：needs_confirm = (有确定性异常) ∨ (年级+主考点三锚没定死)；无异常+三锚达标 → 直接放行。
- facts_locked 后 LLM 来源对 4 守恒维（副考点/考察类型/骨架/难点）回写一律忽略 + warn + audit；
  老师来源放行 + 留痕。
"""

import copy

from agents import dna_extract
from agents.variant import (
    REGEN_CLASS,
    _dna_fact_edit,
    build_mother_confirm,
    edit_dna_state,
    mother_confirm_flags,
)

# 一份「定死」（pinned）的干净基线 state：年级/考点/题型三锚置信达标 + 教材册 code + 守恒维无异常。
_PINNED_STATE = {
    "mother_confirmed": True,
    "facts_locked": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {
            "value": "一元一次方程",
            "confidence": 0.9,
            "anchored": {"id": "30710101", "code": "30710101", "name": "一元一次方程"},
        },
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干",
        "answer": "x=1",
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [{"id": "30710102", "name": "等式性质"}],
            "qtype": "解答",
            "exam_type": "直接计算",
            "skeleton": ["移项", "【合并同类项】"],
            "hard_points": [],
            "tags": ["解方程"],
            "scene": "纯代数",
            "difficulty": 2,
            "flags": [],
        },
    },
}


def _clean_dna():
    return copy.deepcopy(_PINNED_STATE["mother_dna"]["dna"])


# ===========================================================================
# 守恒维确定性异常（mother_confirm_flags · 纯函数 · 零 LLM · G11）
# ===========================================================================


def test_flags_clean_dna_no_anomaly():
    assert mother_confirm_flags(_clean_dna()) == []


def test_flag_skeleton_empty_when_skeleton_blank():
    dna = _clean_dna()
    dna["skeleton"] = []
    assert dna_extract.FLAG_SKELETON_EMPTY in mother_confirm_flags(dna)


def test_flag_skeleton_empty_when_only_whitespace_steps():
    dna = _clean_dna()
    dna["skeleton"] = ["   ", ""]
    assert dna_extract.FLAG_SKELETON_EMPTY in mother_confirm_flags(dna)


def test_flag_exam_type_oob_when_out_of_closed_set():
    dna = _clean_dna()
    dna["exam_type"] = "瞎编类型"  # 不在闭集
    flags = mother_confirm_flags(dna)
    assert dna_extract.FLAG_EXAM_TYPE_OOB in flags


def test_flag_exam_type_oob_from_extract_flag():
    # 抽取期已打 FLAG_EXAM_TYPE_OOB（exam_type 已被置 None）→ 仍透出异常
    dna = _clean_dna()
    dna["exam_type"] = None
    dna["flags"] = [dna_extract.FLAG_EXAM_TYPE_OOB]
    assert dna_extract.FLAG_EXAM_TYPE_OOB in mother_confirm_flags(dna)


def test_flag_secondary_kp_oob_透出抽取期事实():
    dna = _clean_dna()
    dna["flags"] = [dna_extract.FLAG_SECONDARY_KP_OOB]
    assert dna_extract.FLAG_SECONDARY_KP_OOB in mother_confirm_flags(dna)


def test_flags_none_dna_视同骨架空():
    assert mother_confirm_flags(None) == [dna_extract.FLAG_SKELETON_EMPTY]
    assert mother_confirm_flags({}) == [dna_extract.FLAG_SKELETON_EMPTY]


def test_flags_只产三类无逐维置信():
    # 多异常并发：三类都在；不混入任何非确定性异常 flag
    dna = _clean_dna()
    dna["skeleton"] = []
    dna["exam_type"] = "瞎编"
    dna["flags"] = [dna_extract.FLAG_SECONDARY_KP_OOB, "main_kp_oob", "difficulty_fallback"]
    flags = set(mother_confirm_flags(dna))
    assert flags == {
        dna_extract.FLAG_SECONDARY_KP_OOB,
        dna_extract.FLAG_EXAM_TYPE_OOB,
        dna_extract.FLAG_SKELETON_EMPTY,
    }
    # 非守恒维 flag（main_kp_oob/difficulty_fallback）不混进守恒维门控
    assert "main_kp_oob" not in flags
    assert "difficulty_fallback" not in flags


# ===========================================================================
# 合并确认闸（build_mother_confirm · 缺口5 + D-merge7 · G10/G11）
# ===========================================================================


def test_confirm_clean_pinned_直接放行():
    # 无确定性异常 + 三锚达标 → needs_confirm=False（直接出变式不打扰）
    mc = build_mother_confirm(copy.deepcopy(_PINNED_STATE))
    assert mc["needs_confirm"] is False
    assert mc["flags"] == []


def test_confirm_anomaly_停一道确认():
    # 守恒维有确定性异常（骨架空）→ needs_confirm=True（停下弹合并确认）
    state = copy.deepcopy(_PINNED_STATE)
    state["mother_dna"]["dna"]["skeleton"] = []
    mc = build_mother_confirm(state)
    assert mc["needs_confirm"] is True
    assert dna_extract.FLAG_SKELETON_EMPTY in mc["flags"]


def test_confirm_三锚没定死也停():
    # 守恒维无异常，但三锚没定死（kp 未锚）→ needs_confirm=True（合并闸：异常 ∨ 没定死）
    state = copy.deepcopy(_PINNED_STATE)
    state["analysis"]["kp"]["anchored"] = {}  # kp 没锚到真叶子
    mc = build_mother_confirm(state)
    assert mc["needs_confirm"] is True
    assert mc["flags"] == []  # 守恒维本身无异常，停是因三锚没定死


def test_confirm_异常与没定死合并一道():
    # 既有守恒维异常又三锚没定死 → 仍只一份 mother_confirm（合并不分两段）
    state = copy.deepcopy(_PINNED_STATE)
    state["mother_dna"]["dna"]["exam_type"] = "瞎编"
    state["analysis"]["grade"]["code"] = ""  # 年级没定死
    mc = build_mother_confirm(state)
    assert mc["needs_confirm"] is True
    assert dna_extract.FLAG_EXAM_TYPE_OOB in mc["flags"]


# ===========================================================================
# DNA 契约 v2 字段（G16）：REGEN_CLASS 四分流映射就位
# ===========================================================================


def test_regen_class_四分流映射():
    # 🔴 A-2/契约C4（PRD-A-018）：main_kp 由 hard_anchor → soft_regen（单一真相，与 FE 逐字一致）。
    assert REGEN_CLASS["main_kp"] == "soft_regen"
    assert REGEN_CLASS["grade"] == "hard_anchor"
    assert REGEN_CLASS["qtype"] == "soft_regen"
    assert REGEN_CLASS["difficulty"] == "soft_regen"
    assert REGEN_CLASS["exam_type"] == "soft_regen"
    assert REGEN_CLASS["scene"] == "soft_regen"
    assert REGEN_CLASS["skeleton"] == "rewrite_solve"
    assert REGEN_CLASS["models"] == "rewrite_solve"
    assert REGEN_CLASS["tags"] == "meta"
    assert REGEN_CLASS["secondary_kps"] == "meta"


def test_mother_confirm_契约形状():
    mc = build_mother_confirm(copy.deepcopy(_PINNED_STATE))
    # 契约 v2 (b)：flags / needs_confirm / confirmed_dims / audit_ref（无逐维置信分）
    assert set(mc.keys()) == {"flags", "needs_confirm", "confirmed_dims", "audit_ref"}
    assert "confidence" not in mc  # D-merge7：去掉 confidence
    assert isinstance(mc["flags"], list)
    assert isinstance(mc["needs_confirm"], bool)


# ===========================================================================
# facts_locked 扩 4 守恒维（_dna_fact_edit · 缺口6 · G18）
# ===========================================================================


def test_dna_fact_edit_teacher_放行并留痕():
    md = copy.deepcopy(_PINNED_STATE["mother_dna"])
    audit = []
    ok = _dna_fact_edit(
        md, "exam_type", "证明推理",
        source="teacher", locked=True, audit=audit, instruction="老师改",
    )
    assert ok is True
    assert md["dna"]["exam_type"] == "证明推理"  # 老师来源放行
    assert len(audit) == 1
    assert audit[0]["field"] == "exam_type"
    assert audit[0]["old"] == "直接计算" and audit[0]["new"] == "证明推理"
    assert audit[0]["source"] == "teacher"


def test_dna_fact_edit_llm_locked_被忽略并留痕():
    md = copy.deepcopy(_PINNED_STATE["mother_dna"])
    audit = []
    ok = _dna_fact_edit(
        md, "exam_type", "LLM想改的值",
        source="llm", locked=True, audit=audit,
    )
    assert ok is False  # 冻结 + LLM 来源 → 忽略
    assert md["dna"]["exam_type"] == "直接计算"  # 值未被改
    assert len(audit) == 1 and audit[0].get("ignored") is True  # 留痕（ignored）


def test_dna_fact_edit_llm_unlocked_正常写():
    # 定死前（未冻结）LLM 来源正常写（锚定就是靠它达成定死的）
    md = copy.deepcopy(_PINNED_STATE["mother_dna"])
    audit = []
    ok = _dna_fact_edit(
        md, "skeleton", ["新骨架步"],
        source="llm", locked=False, audit=audit,
    )
    assert ok is True
    assert md["dna"]["skeleton"] == ["新骨架步"]


def test_dna_fact_edit_覆盖四守恒维():
    for field, val in [
        ("secondary_kps", [{"id": "x", "name": "y"}]),
        ("exam_type", "性质判定"),
        ("skeleton", ["a", "b"]),
        ("hard_points", ["分类讨论"]),
    ]:
        md = copy.deepcopy(_PINNED_STATE["mother_dna"])
        ok = _dna_fact_edit(md, field, val, source="teacher", locked=True, audit=[])
        assert ok is True, field
        assert md["dna"][field] == val


def test_dna_fact_edit_非守恒维拒绝():
    md = copy.deepcopy(_PINNED_STATE["mother_dna"])
    ok = _dna_fact_edit(md, "tags", ["x"], source="teacher", locked=False, audit=[])
    assert ok is False  # tags 是元数据维，不归本 setter


# ===========================================================================
# edit_dna_state 守恒维改 → facts_audit 留痕（缺口6 端到端，老师手动路径）
# ===========================================================================


def _no_llm(monkeypatch):
    import agents.variant as variant_mod

    async def _boom(*a, **k):
        raise AssertionError("edit-dna 不该调用 LLM")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", _boom)


def test_edit_dna_exam_type_落facts_audit(monkeypatch):
    _no_llm(monkeypatch)
    state = copy.deepcopy(_PINNED_STATE)
    state["items"] = [{"stem": "q1", "check": {"tier": "verified"}}]
    update, item, err = edit_dna_state(state, 1, "exam_type", "纠错")
    assert err is None
    assert update["mother_dna"]["dna"]["exam_type"] == "纠错"
    # 老师手动改守恒维 → facts_audit 多一条留痕
    audit = update.get("facts_audit") or []
    assert any(a["field"] == "exam_type" and a["new"] == "纠错" for a in audit)


def test_edit_dna_secondary_kps_落facts_audit(monkeypatch):
    _no_llm(monkeypatch)
    state = copy.deepcopy(_PINNED_STATE)
    state["items"] = [{"stem": "q1"}]
    update, item, err = edit_dna_state(
        state, 1, "secondary_kps", [{"code": "30710103", "name": "去括号"}]
    )
    assert err is None
    sec = update["mother_dna"]["dna"]["secondary_kps"]
    assert [s["id"] for s in sec] == ["30710103"]
    audit = update.get("facts_audit") or []
    assert any(a["field"] == "secondary_kps" for a in audit)


def test_edit_dna_非守恒维不落facts_audit(monkeypatch):
    # 改 scene（非本批次守恒 setter 维）→ 不追加 facts_audit（仅守恒维走留痕）
    _no_llm(monkeypatch)
    state = copy.deepcopy(_PINNED_STATE)
    state["items"] = [{"stem": "q1"}]
    update, item, err = edit_dna_state(state, 1, "scene", "购物场景")
    assert err is None
    assert "facts_audit" not in update
