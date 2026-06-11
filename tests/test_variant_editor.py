# -*- coding: utf-8 -*-
"""PRD-C-009 二期 · 题组编辑器三动作单测（reorder / edit-item / reverify）。

铁律对照（CLAUDE.md §4/§5）：
- reorder / edit-item = 零 LLM（纯代码重排 / patch+净化+标记）；reverify = 单题 LLM+sympy，
  但判决只读 sympy verdict（_check_one_item 单一事实源，不另写判决）。
- 字段白名单：manual_edited/from_edit 内部键不入库（build_create_bo/_artifact_payload 挡）。
- 全部 LLM/IO monkeypatch → 零网络。
"""

import asyncio

import agents.variant as variant_mod
from agents import math_verify
from agents.variant import (
    TIER_MANUAL,
    _artifact_payload,
    _is_full_permutation,
    _reorder_items,
    edit_item_state,
    reorder_items_state,
    reverify_item_state,
)
from agents.variant_support import build_create_bo

_FACTS_STATE = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.9, "anchored": {"code": "100200300"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {"stem": "母题题干", "answer": "x=1", "difficulty": 3},
}


# ===========================================================================
# _is_full_permutation 纯函数
# ===========================================================================


def test_is_full_permutation_valid():
    assert _is_full_permutation([3, 1, 2], 3) is True
    assert _is_full_permutation([1], 1) is True


def test_is_full_permutation_rejects_partial_dup_oob_type():
    assert _is_full_permutation([1, 2], 3) is False  # 给不全
    assert _is_full_permutation([1, 1, 2], 3) is False  # 重复
    assert _is_full_permutation([1, 2, 4], 3) is False  # 越界
    assert _is_full_permutation("1 2 3", 3) is False  # 非 list
    assert _is_full_permutation([1, 2, "3"], 3) is False  # 含非 int
    assert _is_full_permutation([0, 1, 2], 3) is False  # 0-based 误传


# ===========================================================================
# _reorder_items 纯重排（簿记字段随题搬位，不错位）
# ===========================================================================


def test_reorder_items_carries_bookkeeping():
    items = [
        {"stem": "q1", "persisted": True, "check": {"tier": "verified"}, "_seq": 1},
        {"stem": "q2", "persisted": False, "_seq": 2},
        {"stem": "q3", "gene": {"gate": "pass"}, "_seq": 3},
    ]
    out = _reorder_items(items, [3, 1, 2])
    assert [it["stem"] for it in out] == ["q3", "q1", "q2"]
    # q1 的 persisted/check/_seq 整体跟到第 2 位、不错位
    assert out[1]["persisted"] is True
    assert out[1]["check"]["tier"] == "verified"
    assert out[1]["_seq"] == 1
    assert out[0]["gene"]["gate"] == "pass"  # q3 的 gene 跟到第 1 位


# ===========================================================================
# reorder_items_state（端点逻辑）：合法重排 + 非法拒
# ===========================================================================


def test_reorder_items_state_ok_sets_manual_order():
    items = [{"stem": "q1"}, {"stem": "q2"}, {"stem": "q3"}]
    state = dict(_FACTS_STATE, items=items)
    update, err = reorder_items_state(state, [2, 3, 1])
    assert err is None
    assert [it["stem"] for it in update["items"]] == ["q2", "q3", "q1"]
    assert update["manual_order"] is True


def test_reorder_items_state_rejects_illegal_permutation():
    items = [{"stem": "q1"}, {"stem": "q2"}, {"stem": "q3"}]
    state = dict(_FACTS_STATE, items=items)
    for bad in ([1, 2], [1, 1, 3], [1, 2, 4]):
        update, err = reorder_items_state(state, bad)
        assert err is not None and "全排列" in err
        assert update == {}  # 非法不动 state


def test_reorder_items_state_artifact_renumbers_seq():
    # 重排后过 _artifact_payload：index 按新序 1,2,3 现编（_seq 跟题不变）
    items = [
        {"stem": "q1", "_seq": 1, "check": {"tier": "verified"}},
        {"stem": "q2", "_seq": 2, "check": {"tier": "silent"}},
    ]
    state = dict(_FACTS_STATE, items=items)
    update, _ = reorder_items_state(state, [2, 1])
    art = _artifact_payload({**state, **update})
    assert [(c["index"], c["stem"], c["seq"]) for c in art["items"]] == [
        (1, "q2", 2), (2, "q1", 1)
    ]


# ===========================================================================
# edit_item_state（零 LLM）：patch / 净化 / 标记 / check 置 manual
# ===========================================================================


def test_edit_item_patches_only_given_fields_and_sanitizes():
    items = [{"stem": "old", "answer": "x=1", "solution": "old sol", "check": {"tier": "verified"}}]
    state = dict(_FACTS_STATE, items=items)
    update, item, err = edit_item_state(state, 1, stem=r"解 \(x^2=4\) 得", solution=r"第一步\n第二步")
    assert err is None
    # 净化生效：\( \) → $ $；字面 \n → 换行
    assert item["stem"] == "解 $x^2=4$ 得"
    assert item["solution"] == "第一步\n第二步"
    # answer 未传 → 不动
    assert item["answer"] == "x=1"
    # 标手动编辑 + check 置中性 manual（清旧 verified）
    assert item["manual_edited"] is True
    assert item["from_edit"] is True
    assert item["check"] == {"tier": TIER_MANUAL}


def test_edit_item_does_not_mutate_input_state():
    items = [{"stem": "old", "check": {"tier": "verified"}}]
    state = dict(_FACTS_STATE, items=items)
    edit_item_state(state, 1, stem="new")
    # 原 state.items 不被改（端点回写走 update.items）
    assert state["items"][0]["stem"] == "old"
    assert "manual_edited" not in state["items"][0]


def test_edit_item_index_out_of_range_rejected():
    items = [{"stem": "q1"}, {"stem": "q2"}]
    state = dict(_FACTS_STATE, items=items)
    for bad in (0, 3, -1):
        update, item, err = edit_item_state(state, bad, stem="x")
        assert err is not None and "越界" in err
        assert update == {} and item is None


def test_edit_item_manual_keys_not_leaked_to_create_bo():
    # 🔴 manual_edited/from_edit 内部键不入库（build_create_bo 显式白名单挡）
    items = [{"stem": "题干", "answer": "x=2", "solution": "解析"}]
    state = dict(_FACTS_STATE, items=items)
    _, item, _ = edit_item_state(state, 1, stem="新题干")
    facts = variant_mod._mother_facts(state)
    bo = build_create_bo(item, facts)
    assert "manual_edited" not in bo
    assert "from_edit" not in bo
    # auxTags 也不透传这俩内部键
    assert "manual_edited" not in bo.get("auxTags", {})
    assert "from_edit" not in bo.get("auxTags", {})


def test_edit_item_artifact_passes_tier_manual():
    # artifact 透传 tier='manual' 给 FE 渲染中性徽章
    items = [{"stem": "old", "check": {"tier": "verified", "verify": "sympy_pass"}}]
    state = dict(_FACTS_STATE, items=items)
    update, _, _ = edit_item_state(state, 1, stem="new")
    art = _artifact_payload({**state, **update})
    assert art["items"][0]["tier"] == TIER_MANUAL
    # 旧 verify 误导被清（check 整体换成 {'tier':'manual'} → verify=None）
    assert art["items"][0]["verify"] is None


# ===========================================================================
# reverify_item_state（单题 LLM+sympy，mock LLM）：回写真实 check + 越界拒
# ===========================================================================


def test_reverify_sympy_pass_writes_verified_tier(monkeypatch):
    # sympy PASS → check.verify=sympy_pass + tier=verified，洗掉 manual 待验算语义
    async def solve_stub(stem):
        return {"solved_answer": "x=2", "solution": "解析步骤"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=2"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)

    items = [
        {"stem": "2x=4", "answer": "x=2", "qtype": "解答",
         "manual_edited": True, "from_edit": True, "check": {"tier": TIER_MANUAL}},
    ]
    state = dict(_FACTS_STATE, items=items)
    update, item, err = asyncio.run(reverify_item_state(state, 1))
    assert err is None
    chk = item["check"]
    assert chk["verify"] == variant_mod.VERIFY_SYMPY_PASS
    assert chk["tier"] == variant_mod.TIER_VERIFIED  # 真实验算结果，不再是 manual
    assert chk["tier"] != TIER_MANUAL


def test_reverify_from_edit_fail_kept_warn_not_dropped(monkeypatch):
    # 编辑后重验：sympy FAIL → from_edit 短路保留原题打 ⚠（fail_after_regen），不剔除/不回炉
    regen_called = {"n": 0}

    async def solve_stub(stem):
        return {"solved_answer": "x=9", "solution": "解析"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "mismatch", "computed": "x=9"}

    async def regen_spy(item, facts, feedback=None):
        regen_called["n"] += 1
        return {"stem": "should-not-happen"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_spy)

    items = [{"stem": "老师编辑过的题", "answer": "x=2", "qtype": "解答",
              "check": {"tier": TIER_MANUAL}}]
    state = dict(_FACTS_STATE, items=items)
    update, item, err = asyncio.run(reverify_item_state(state, 1))
    assert err is None
    assert item is not None  # 不剔除
    assert item["stem"] == "老师编辑过的题"  # 老师原题逐字保留
    assert item["check"]["verify"] == variant_mod.VERIFY_FAIL_AFTER_REGEN
    assert item["check"]["tier"] in (variant_mod.TIER_BOTH_LOW, variant_mod.TIER_SILENT)
    assert regen_called["n"] == 0  # from_edit 永不回炉换题


def test_reverify_clears_old_check_before_rejudge(monkeypatch):
    # 旧 check（manual）不该让 _check_one_item 原样跳过 → 必须重判（solve 被调到）
    solve_called = {"n": 0}

    async def solve_stub(stem):
        solve_called["n"] += 1
        return {"solved_answer": "x=2", "solution": "s"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=2"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)

    items = [{"stem": "2x=4", "answer": "x=2", "qtype": "解答", "check": {"tier": TIER_MANUAL}}]
    state = dict(_FACTS_STATE, items=items)
    asyncio.run(reverify_item_state(state, 1))
    assert solve_called["n"] == 1  # 旧 check 被清 → 真的重跑了验算


def test_reverify_index_out_of_range_rejected():
    items = [{"stem": "q1"}]
    state = dict(_FACTS_STATE, items=items)
    for bad in (0, 2, -1):
        update, item, err = asyncio.run(reverify_item_state(state, bad))
        assert err is not None and "越界" in err
        assert update == {} and item is None


def test_reverify_does_not_mutate_input_state(monkeypatch):
    async def solve_stub(stem):
        return {"solved_answer": "x=2", "solution": "s"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=2"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)

    items = [{"stem": "2x=4", "answer": "x=2", "qtype": "解答", "check": {"tier": TIER_MANUAL}}]
    state = dict(_FACTS_STATE, items=items)
    asyncio.run(reverify_item_state(state, 1))
    # 原 state.items 不被改（端点回写走 update.items）
    assert state["items"][0]["check"] == {"tier": TIER_MANUAL}
