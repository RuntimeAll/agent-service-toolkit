# -*- coding: utf-8 -*-
"""W2 守恒注入单测（PRD-C-014 B2·T1）。

数据源 = 母题 DNA（facts.dna，B1 dna_extract 产物）：
  - _kp_whitelist：主 kp + 副 kp → [(id, name)]（去重去空）。
  - _conservation_blocked：DNA 已抽却无任何 kp（白名单空集）→ 不放行生成；无 DNA（{}）→ 放行（库内母题路径）。
  - _conservation_clause：白名单 / 考察类型 / 解法骨架最难步 / 表皮必换 / 难度档 五段硬约束。
  - generate 节点：白名单空集 → 降级 clarify（不裸出）。
"""

import asyncio

from langchain_core.messages import HumanMessage

import agents.variant as variant_mod
from agents.variant import (
    _conservation_blocked,
    _conservation_clause,
    _kp_whitelist,
    generate,
)

_DNA = {
    "main_kp": {"id": "100", "name": "一元一次方程"},
    "secondary_kps": [{"id": "101", "name": "移项"}, {"id": "102", "name": "合并同类项"}],
    "exam_type": "直接计算",
    "skeleton": ["设未知数", "【移项合并，系数化为1】", "检验"],
}


# ---------------------------------------------------------------------------
# _kp_whitelist
# ---------------------------------------------------------------------------


def test_whitelist_collects_main_and_secondary():
    wl = _kp_whitelist(_DNA)
    ids = [kid for kid, _ in wl]
    assert ids == ["100", "101", "102"]


def test_whitelist_dedups_and_drops_empty():
    dna = {
        "main_kp": {"id": "100", "name": "kp"},
        "secondary_kps": [{"id": "100", "name": "kp"}, {"id": "", "name": ""}, {"id": "200", "name": "b"}],
    }
    assert [kid for kid, _ in _kp_whitelist(dna)] == ["100", "200"]


def test_whitelist_empty_when_no_kp():
    assert _kp_whitelist({}) == []
    assert _kp_whitelist({"main_kp": None, "secondary_kps": []}) == []
    assert _kp_whitelist(None) == []


# ---------------------------------------------------------------------------
# _conservation_blocked
# ---------------------------------------------------------------------------


def test_blocked_when_dna_extracted_but_no_kp():
    # DNA 抽过（dict 非空）却没锚到 kp → 真·守恒失守 → 拦
    dna = {"exam_type": "直接计算", "skeleton": ["x"], "main_kp": None, "secondary_kps": []}
    assert _conservation_blocked(dna) is not None


def test_not_blocked_when_dna_absent():
    # 无 DNA（库内母题直进 generate 合法路径）→ 不归本闸管
    assert _conservation_blocked({}) is None
    assert _conservation_blocked(None) is None


def test_not_blocked_when_whitelist_present():
    assert _conservation_blocked(_DNA) is None


# ---------------------------------------------------------------------------
# _conservation_clause
# ---------------------------------------------------------------------------


def test_clause_lists_whitelist_ids_and_names():
    c = _conservation_clause(_DNA)
    assert "知识点守恒" in c
    assert "一元一次方程（100）" in c
    assert "移项（101）" in c


def test_clause_pins_exam_type_and_hardest_step():
    c = _conservation_clause(_DNA)
    assert "考察类型守恒" in c and "直接计算" in c
    assert "最难步" in c and "移项合并，系数化为1" in c


def test_clause_always_has_surface_swap_and_difficulty_rules():
    c = _conservation_clause(_DNA)
    assert "表皮必换" in c and "数字必须" in c
    assert "母题档 +1" in c  # T3 难度规则随 W2 段


def test_clause_degrades_missing_fields_gracefully():
    # exam_type / skeleton 缺 → 那两条不注入，但白名单 + 表皮 + 难度仍在（不输出半截占位）
    c = _conservation_clause({"main_kp": {"id": "1", "name": "kp"}})
    assert "知识点守恒" in c
    assert "考察类型守恒" not in c
    assert "最难步" not in c
    assert "表皮必换" in c


# ---------------------------------------------------------------------------
# generate 节点：白名单空集 → 降级 clarify（不放行生成、不裸出）
# ---------------------------------------------------------------------------

_STATE_BLOCKED = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "?", "confidence": 0.9},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    # DNA 抽过却没锚到 kp（白名单空集）→ generate 必须拒造
    "mother_dna": {
        "stem": "母题题干",
        "answer": "x=1",
        "difficulty": 3,
        "dna": {"exam_type": "直接计算", "skeleton": ["x"], "main_kp": None, "secondary_kps": []},
    },
}


def test_generate_blocks_on_empty_whitelist(monkeypatch):
    called = {"n": 0}

    async def spy_llm(*a, **k):
        called["n"] += 1
        return "[]"

    monkeypatch.setattr(variant_mod, "_ainvoke_text", spy_llm)
    state = dict(_STATE_BLOCKED, messages=[HumanMessage(content="https://oss/q.png")], knobs={})
    out = asyncio.run(generate(state, {}))
    # 不放行生成：无 items 产出，回降级消息（不裸出），且没调出题 LLM
    assert not out.get("items")
    assert out["messages"]  # 有降级回话
    assert "知识点" in str(out["messages"][-1].content)
    assert called["n"] == 0  # 守门在 LLM 调用之前
