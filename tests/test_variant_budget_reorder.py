# -*- coding: utf-8 -*-
"""PRD-C-013 S4 单测：P13 预算闸 / P9 排序 / P2b-BE 逐题上屏帧序。

铁律对照（CLAUDE.md §4/§5）：
- P13 是「超限跳过增强类调用」不是「LLM 决定流程」——核心链不跳，增强类（闸A rework /
  闸B heal / replenish 补题 / extract 兜底）超限即跳走 G5 降级（标 ⚠ / 保留原题，不卡死）。
- P9 默认序 = 纯函数难度升序稳定排序；指令排序走受约束枚举 reorder op（越界/混类→clarify），
  执行 = 纯代码 list 重排 + seq 重编，零 LLM 改题。
- 全部 LLM/IO monkeypatch → 零网络。
"""

import asyncio
import json

from langchain_core.messages import ChatMessage, HumanMessage

import agents.variant as variant_mod
from agents import math_verify
from agents.variant import (
    _budget_begin,
    _budget_bind,
    _budget_exhausted,
    _budget_tick,
    _sort_by_difficulty,
    exec_reorder,
    generate,
    validate_instruction,
)

# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

_FACTS_STATE = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.9, "anchored": {"code": "100200300"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干",
        "answer": "x=1",
        "difficulty": 3,
        "solution_skeleton": "移项合并",
    },
}

_FACTS = {"kp_name": "kp-x", "grade": "g7", "qtype": "解答"}

_GEN_ITEM = {
    "stem": "",
    "answer": "x=2",
    "solution": "略",
    "qtype": "解答",
    "difficulty": 3,
    "level": "normal",
    "injected_kp": None,
}


def _capture_frames(monkeypatch):
    captured: list = []
    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: captured.append)
    return captured


def _artifact_frames(captured):
    out = []
    for msg in captured:
        if isinstance(msg, ChatMessage) and msg.role == "custom" and "artifact" in msg.content[0]:
            out.append(msg.content[0]["artifact"])
    return out


# ===========================================================================
# P13 预算闸：counter 基本语义（contextvar）
# ===========================================================================


def test_budget_tick_and_exhaust_basic():
    async def go():
        _budget_begin(2)
        assert _budget_exhausted() is False
        _budget_tick()
        assert _budget_exhausted() is False
        _budget_tick()
        assert _budget_exhausted() is True  # used==limit -> exhausted
        _budget_tick()
        assert _budget_exhausted() is True

    asyncio.run(go())


def test_budget_disabled_when_limit_non_positive():
    async def go():
        _budget_begin(0)  # 关闸
        for _ in range(50):
            _budget_tick()
        assert _budget_exhausted() is False  # 无预算 → 永不超限

    asyncio.run(go())


def test_budget_bind_carries_used_across_nodes():
    # 模拟跨节点：入口 reset → tick → 把 budget 放回 state → 下游 bind 携带累计 used
    async def go():
        b = _budget_bind({}, reset_limit=3)
        assert b == {"used": 0, "limit": 3}
        _budget_tick()
        _budget_tick()
        # 下游节点从 state 携带（reset_limit=None）
        b2 = _budget_bind({"llm_call_budget": b})
        assert b2["used"] == 2 and b2["limit"] == 3
        assert _budget_exhausted() is False
        _budget_tick()
        assert _budget_exhausted() is True

    asyncio.run(go())


def test_budget_bind_none_when_state_has_no_budget():
    async def go():
        b = _budget_bind({})  # 无 reset、state 无簿记 → None（关闸，回退老逻辑）
        assert b is None
        for _ in range(99):
            _budget_tick()
        assert _budget_exhausted() is False

    asyncio.run(go())


def test_ainvoke_text_ticks_budget_on_success(monkeypatch):
    # _ainvoke_text 成功返回 → _budget_tick()+1（核心记账点）
    async def fake_failover(messages, **kw):
        return (variant_mod.AIMessage(content="ok"), "relay", "model", 0, None)

    monkeypatch.setattr(variant_mod.relay_pool, "ainvoke_failover", fake_failover)
    monkeypatch.setattr(variant_mod.relay_pool, "usage_tokens", lambda r: (0, 0))
    monkeypatch.setattr(variant_mod.relay_pool, "cost_yuan", lambda *a: None)
    monkeypatch.setattr(variant_mod, "_LLM_TRACE_ENABLED", False)
    monkeypatch.setattr(variant_mod.conv_trace, "write", lambda **kw: None)
    monkeypatch.setattr(variant_mod.conv_trace, "teacher_id_from_token", lambda t: None)

    async def go():
        _budget_begin(5)
        before = variant_mod._budget_ctx.get()["used"]
        await variant_mod._ainvoke_text([HumanMessage(content="hi")])
        assert variant_mod._budget_ctx.get()["used"] == before + 1

    asyncio.run(go())


# ===========================================================================
# G6：预算耗尽后增强类调用被跳过，流程正常收尾（闸A rework / 闸B heal / extract）
# ===========================================================================


def test_gene_gate_pure_code_never_reworks_or_spends_budget(monkeypatch):
    # 🔴 B2·T2 换血后：闸A = 纯代码三检，无 LLM judge / 无回炉 / 不花预算。
    #   旧「judge=rework → 预算耗尽跳回炉」语义整段退役（regen 永不被闸A 触达）。
    #   即便预算耗尽，闸A 仍判得出（纯代码），且绝不调 _regen_once。
    regen_called = {"n": 0}

    async def regen_spy(item, facts, feedback=None):
        regen_called["n"] += 1
        return None

    monkeypatch.setattr(variant_mod, "_regen_once", regen_spy)

    async def go():
        _budget_begin(1)
        _budget_tick()  # 耗尽——对纯代码闸A 无影响
        # 干净平行题（题型守恒、表皮已换）→ 三检全过 → pass
        return await variant_mod._gene_one_item(
            {"stem": "全新题面 5x=10", "qtype": "解答"}, _FACTS, 0, 1
        )

    out = asyncio.run(go())
    assert out["gene"]["gate"] == "pass"
    assert regen_called["n"] == 0  # 闸A 纯代码，永不回炉


def test_gene_gate_pure_code_flags_warn_without_rework(monkeypatch):
    # 三检命中（题型守恒破：变式选择题 ≠ 母题解答）→ warn + flags，仍不回炉。
    regen_called = {"n": 0}

    async def regen_spy(item, facts, feedback=None):
        regen_called["n"] += 1
        return None

    monkeypatch.setattr(variant_mod, "_regen_once", regen_spy)

    async def go():
        _budget_begin(10)  # 充足也不调回炉
        return await variant_mod._gene_one_item(
            {"stem": "全新题面 5x=10", "qtype": "选择"}, _FACTS, 0, 1
        )

    out = asyncio.run(go())
    assert out["gene"]["gate"] == "warn"
    assert "qtype_conservation" in out["gene"]["flags"]
    assert regen_called["n"] == 0


def test_check_heal_skipped_when_budget_exhausted_item_kept_warn(monkeypatch):
    # 🔴 整改4（2026-06-12）：闸B sympy FAIL + 预算耗尽 → 跳过 heal → 保留打 ⚠ 放行
    #   （不再剔除、不再补题）。不卡死、不抛。
    heal_called = {"n": 0}

    async def solve_stub(stem):
        return {"solved_answer": "x=9", "solution": "解析"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "mismatch", "computed": "x=9"}

    async def regen_spy(item, facts, feedback=None):
        heal_called["n"] += 1
        return {"stem": "healed"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_spy)

    async def go():
        _budget_begin(1)
        _budget_tick()  # 耗尽
        return await variant_mod._check_one_item(
            {"stem": "a", "answer": "x=2", "qtype": "解答"}, _FACTS, 0, 1
        )

    kept, note = asyncio.run(go())
    assert kept is not None  # 整改4：保留打 ⚠（不剔除）
    assert note is None  # 不进 dropped_notes
    assert kept["check"]["verify"] == variant_mod.VERIFY_FAIL_AFTER_REGEN
    assert heal_called["n"] == 0  # 预算耗尽 → 回炉跳过


# ===========================================================================
# 对抗审② — 编辑轮 from_edit 在闸B 失守：老师点名改造的题被 sympy 判 FAIL 不应静默吞
# ===========================================================================


def test_check_from_edit_fail_kept_not_dropped(monkeypatch):
    # 🔴 RC2「老师意志优先」补齐闸B：from_edit 题 sympy FAIL → 保留原题 + 标 ⚠ 注记
    #   （verify=fail_after_regen），不回炉/不换题/不剔除（与闸A from_edit 短路对称）。
    regen_called = {"n": 0}

    async def solve_stub(stem):
        return {"solved_answer": "x=9", "solution": "解析"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "mismatch", "computed": "x=9"}

    async def regen_spy(item, facts, feedback=None):
        regen_called["n"] += 1
        return {"stem": "regen-should-not-happen"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_spy)

    item = {
        "stem": "teacher-edited",
        "answer": "x=2",
        "qtype": "解答",
        "from_edit": True,
        "edit_note": "老师要求改造的",
    }
    kept, note = asyncio.run(variant_mod._check_one_item(item, _FACTS, 0, 1))
    assert kept is not None  # 不剔除
    assert note is None  # 不进 dropped_notes
    assert kept["stem"] == "teacher-edited"  # 老师编辑的原题保留逐字
    assert kept.get("edit_note") == "老师要求改造的"  # edit_note 没被丢
    assert kept["check"]["verify"] == variant_mod.VERIFY_FAIL_AFTER_REGEN
    assert regen_called["n"] == 0  # 永不回炉/换题（老师意志优先）
    # 4d 外显：verify 侧低 → both_low(⚠) 或 silent（不外显强正面）；绝不标 verified
    assert kept["check"]["tier"] in (variant_mod.TIER_BOTH_LOW, variant_mod.TIER_SILENT)


def test_check_non_edit_fail_kept_warn_after_regen(monkeypatch):
    # 🔴 整改4（2026-06-12）：非 from_edit 题 sympy FAIL + 回炉 1 次失败 → 保留打 ⚠ 放行
    #   （不再剔除/补题）。对照 from_edit 短路（那个连回炉都不走），这个走过一次回炉。
    regen_called = {"n": 0}

    async def solve_stub(stem):
        return {"solved_answer": "x=9", "solution": "解析"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "mismatch", "computed": "x=9"}

    async def regen_none(item, facts, feedback=None):
        regen_called["n"] += 1
        return None  # 回炉失败 → 整改4：标 ⚠ 放行

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_none)

    kept, note = asyncio.run(
        variant_mod._check_one_item(
            {"stem": "plain", "answer": "x=2", "qtype": "解答"}, _FACTS, 0, 1
        )
    )
    assert kept is not None and note is None  # 保留、不剔除
    assert kept["check"]["verify"] == variant_mod.VERIFY_FAIL_AFTER_REGEN
    assert regen_called["n"] == 1  # 整改4：只一次回炉，不再补题


def test_machine_verify_skips_extraction_when_budget_exhausted(monkeypatch):
    # 载荷不可用 + 预算耗尽 → 跳过 extract 兜底，按 degrade 降级（不调 LLM）
    extract_called = {"n": 0}

    async def fake_extract(*a, **k):
        extract_called["n"] += 1
        return {"kind": "numeric", "expr": "1+1", "claimed": "2"}

    monkeypatch.setattr(variant_mod, "_extract_payload", fake_extract)

    async def go():
        _budget_begin(1)
        _budget_tick()  # 耗尽
        # item 无可用 verify_payload → 本该 extract 兜底
        return await variant_mod._machine_verify({"stem": "s", "answer": "2", "qtype": "解答"}, "2")

    res = asyncio.run(go())
    assert res["verdict"] == math_verify.DEGRADE
    assert extract_called["n"] == 0


def test_generate_round_budget_exhaustion_still_finishes(monkeypatch):
    # 端到端：出题轮预算极小（1）→ eager 闸链增强类调用全跳过，但 generate 仍正常收尾，
    #         items 齐全（核心链不跳，宏观 DAG 不破）。
    _capture_frames(monkeypatch)
    monkeypatch.setattr(variant_mod.settings, "VARIANT_BUDGET_GENERATE", 1)

    async def solve_stub(stem):
        return {"solved_answer": "x=2", "solution": "s"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "m", "computed": "x=9"}

    regen_calls = {"n": 0}

    async def regen_spy(item, facts, feedback=None):
        regen_calls["n"] += 1
        return None

    # B2·T2: Gate-A is pure code now (no _gene_judge_one). 闸B sympy FAIL + 预算耗尽 → heal 跳过。
    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_spy)

    parts = [json.dumps(dict(_GEN_ITEM, stem=f"s{i}"), ensure_ascii=False) for i in range(3)]
    text = "[" + ",".join(parts) + "]"

    async def fake(messages, retry=True, *, on_delta=None, **kw):
        _budget_tick()  # 模拟核心调用记账（_ainvoke_text 真身会 tick）
        if on_delta is not None:
            acc = "["
            for p in parts:
                acc += p + ","
                on_delta(acc)
        return text

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
    state = dict(_FACTS_STATE, messages=[HumanMessage(content="https://o.ss/q.png")], knobs={})
    out = asyncio.run(generate(state, {}))
    # 流程收尾、items 在（剔除题转哨兵留位，下游 solve_explain 收口）
    assert len(out["items"]) == 3
    assert out["llm_call_budget"]["limit"] == 1
    # 预算闸生效：闸B heal 被跳过（regen 一次没调）；闸A 纯代码本就不回炉
    assert regen_calls["n"] == 0


# ===========================================================================
# P9 默认序：_sort_by_difficulty 纯函数（升序稳定 + stem 集合不变）
# ===========================================================================


def test_sort_by_difficulty_ascending_stable():
    items = [
        {"stem": "a", "difficulty": 3},
        {"stem": "b", "difficulty": 1},
        {"stem": "c", "difficulty": 3},  # 与 a 同难度 → 保持生成序（a 在 c 前）
        {"stem": "d", "difficulty": 2},
    ]
    out = _sort_by_difficulty(items)
    assert [it["stem"] for it in out] == ["b", "d", "a", "c"]
    # stem 集合不变（不丢/不增）
    assert {it["stem"] for it in out} == {"a", "b", "c", "d"}


def test_sort_by_difficulty_missing_difficulty_clamped_to_zero():
    items = [
        {"stem": "a", "difficulty": 2},
        {"stem": "b"},  # 缺难度 → 0，排最前
        {"stem": "c", "difficulty": "garbage"},  # 不可解析 → 0
    ]
    out = _sort_by_difficulty(items)
    assert [it["stem"] for it in out][:2] == ["b", "c"]  # 两个 0 保持生成序
    assert out[2]["stem"] == "a"


def test_sort_by_difficulty_does_not_mutate_input():
    items = [{"stem": "a", "difficulty": 3}, {"stem": "b", "difficulty": 1}]
    _sort_by_difficulty(items)
    assert [it["stem"] for it in items] == ["a", "b"]  # 原 list 不动


def test_assemble_default_sorts_by_difficulty(monkeypatch):
    # assemble 默认序：_grade_difficulty 桩成 identity，验排序后 artifact 序按难度升序
    captured = _capture_frames(monkeypatch)

    async def identity(items):
        return items

    monkeypatch.setattr(variant_mod, "_grade_difficulty", identity)
    items = [
        {"stem": "hard", "difficulty": 4, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
        {"stem": "easy", "difficulty": 1, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
        {"stem": "mid", "difficulty": 2, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
    ]
    out = asyncio.run(variant_mod.assemble(dict(_FACTS_STATE, items=items), {}))
    assert [it["stem"] for it in out["items"]] == ["easy", "mid", "hard"]
    art = _artifact_frames(captured)[0]
    # seq 重编（index=1,2,3 按新序）
    assert [(it["index"], it["stem"]) for it in art["items"]] == [
        (1, "easy"), (2, "mid"), (3, "hard")
    ]


# ===========================================================================
# P9 指令排序：reorder op 解析（含越界/混类→clarify）+ exec_reorder 执行
# ===========================================================================


def test_reorder_full_permutation_passes():
    out = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "reorder", "order": [3, 1, 2]}]}, 3
    )
    assert out["intent"] == "编辑"
    assert out["ops"] == [{"action": "reorder", "order": [3, 1, 2]}]


def test_reorder_partial_order_downgrades_to_clarify():
    # 长度不等于 N
    out = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "reorder", "order": [1, 2]}]}, 3
    )
    assert out["intent"] == "clarify" and out["ops"] == []


def test_reorder_out_of_range_index_downgrades_to_clarify():
    out = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "reorder", "order": [1, 2, 4]}]}, 3
    )
    assert out["intent"] == "clarify"


def test_reorder_duplicate_index_downgrades_to_clarify():
    out = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "reorder", "order": [1, 1, 2]}]}, 3
    )
    assert out["intent"] == "clarify"


def test_reorder_string_order_coerced_then_validated():
    out = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "reorder", "order": ["2", "1", "3"]}]}, 3
    )
    assert out["intent"] == "编辑"
    assert out["ops"][0]["order"] == [2, 1, 3]


def test_reorder_non_list_order_downgrades_to_clarify():
    out = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "reorder", "order": "1 2 3"}]}, 3
    )
    assert out["intent"] == "clarify"


def test_reorder_mixed_with_remove_downgrades_to_clarify():
    # 混类（reorder + remove）→ R7 整体降级 clarify
    out = validate_instruction(
        {
            "intent": "编辑",
            "ops": [{"action": "reorder", "order": [2, 1, 3]}, {"action": "remove", "index": 1}],
        },
        3,
    )
    assert out["intent"] == "clarify"


def test_exec_reorder_reorders_and_renumbers(monkeypatch):
    captured = _capture_frames(monkeypatch)
    items = [
        {"stem": "q1", "difficulty": 1, "persisted": True, "check": {"badge": "ok"}},
        {"stem": "q2", "difficulty": 2, "persisted": False},
        {"stem": "q3", "difficulty": 3, "persisted": False},
    ]
    state = dict(_FACTS_STATE, items=items, pending={"ops": [{"action": "reorder", "order": [3, 1, 2]}]})
    out = asyncio.run(exec_reorder(state, {}))
    # 纯代码重排，stem 集合不变
    assert [it["stem"] for it in out["items"]] == ["q3", "q1", "q2"]
    assert {it["stem"] for it in out["items"]} == {"q1", "q2", "q3"}
    # 簿记跟题走（q1 的 persisted/check 不错位）
    q1 = next(it for it in out["items"] if it["stem"] == "q1")
    assert q1["persisted"] is True and q1["check"]["badge"] == "ok"
    assert out["pending"] is None
    # 整帧重发：seq 按新序现编
    art = _artifact_frames(captured)[-1]
    assert [(it["index"], it["stem"]) for it in art["items"]] == [
        (1, "q3"), (2, "q1"), (3, "q2")
    ]


def test_exec_reorder_zero_llm(monkeypatch):
    # reorder 绝不调 LLM
    async def boom(*a, **k):
        raise AssertionError("reorder must not call any LLM")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", boom)
    _capture_frames(monkeypatch)
    items = [{"stem": "q1"}, {"stem": "q2"}]
    state = dict(_FACTS_STATE, items=items, pending={"ops": [{"action": "reorder", "order": [2, 1]}]})
    out = asyncio.run(exec_reorder(state, {}))
    assert [it["stem"] for it in out["items"]] == ["q2", "q1"]


def test_exec_reorder_guardrail_fallback_keeps_order(monkeypatch):
    # 罕见兜底：order 缺失（护栏失守）→ 原序不动、友好提示，不抛、不丢题
    _capture_frames(monkeypatch)
    items = [{"stem": "q1"}, {"stem": "q2"}]
    state = dict(_FACTS_STATE, items=items, pending={"ops": [{"action": "reorder"}]})
    out = asyncio.run(exec_reorder(state, {}))
    assert [it["stem"] for it in out["items"]] == ["q1", "q2"]  # 原序
    assert "没拿准" in out["messages"][0].content


# ===========================================================================
# 对抗审③ — 手排被后续编辑静默重排：manual_order sticky 跨轮保护
# ===========================================================================


def test_exec_reorder_sets_manual_order_sticky(monkeypatch):
    # exec_reorder 成功重排 → 置 manual_order=True（跨轮保护标记）
    _capture_frames(monkeypatch)
    items = [{"stem": "q1"}, {"stem": "q2"}, {"stem": "q3"}]
    state = dict(_FACTS_STATE, items=items, pending={"ops": [{"action": "reorder", "order": [3, 1, 2]}]})
    out = asyncio.run(exec_reorder(state, {}))
    assert out["manual_order"] is True


def test_assemble_skips_sort_when_manual_order(monkeypatch):
    # 🔴 手排 sticky：manual_order=True 时 assemble 不重排（保留老师手排序），即使难度乱序
    captured = _capture_frames(monkeypatch)

    async def identity(items):
        return items

    monkeypatch.setattr(variant_mod, "_grade_difficulty", identity)
    items = [
        {"stem": "hard", "difficulty": 4, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
        {"stem": "easy", "difficulty": 1, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
        {"stem": "mid", "difficulty": 2, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
    ]
    state = dict(_FACTS_STATE, items=items, manual_order=True)
    out = asyncio.run(variant_mod.assemble(state, {}))
    # 手排序原样保留（不被难度升序覆盖）
    assert [it["stem"] for it in out["items"]] == ["hard", "easy", "mid"]
    art = _artifact_frames(captured)[-1]
    assert [it["stem"] for it in art["items"]] == ["hard", "easy", "mid"]


def test_exec_remove_clears_manual_order():
    # 改变题集（删题）→ 清手排标记（次序失效，回默认序）
    items = [{"stem": "q1", "check": {"badge": "ok"}}, {"stem": "q2", "check": {"badge": "ok"}}]
    state = dict(
        _FACTS_STATE, items=items, manual_order=True,
        pending={"ops": [{"action": "remove", "index": 2}]},
    )
    out = asyncio.run(variant_mod.exec_remove(state, {}))
    assert out["manual_order"] is False
    assert [it["stem"] for it in out["items"]] == ["q1"]


def test_assemble_default_sort_returns_after_remove_clears_sticky(monkeypatch):
    # 端到端：手排 → 删题清 sticky → assemble 回默认难度升序排序（手排不再黏着）
    captured = _capture_frames(monkeypatch)

    async def identity(items):
        return items

    monkeypatch.setattr(variant_mod, "_grade_difficulty", identity)
    items = [
        {"stem": "hard", "difficulty": 4, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
        {"stem": "easy", "difficulty": 1, "check": {"badge": "ok"}, "gene": {"gate": "pass"}},
    ]
    # manual_order 被 remove 清成 False（模拟删题轮后）→ assemble 重新按难度排
    state = dict(_FACTS_STATE, items=items, manual_order=False)
    out = asyncio.run(variant_mod.assemble(state, {}))
    assert [it["stem"] for it in out["items"]] == ["easy", "hard"]


def test_generate_resets_manual_order(monkeypatch):
    # 新一组题 → manual_order 复位 False（上一组的手排不泄漏到新母题/新出题轮）
    captured = _capture_frames(monkeypatch)

    parts = [json.dumps(dict(_GEN_ITEM, stem=f"s{i}"), ensure_ascii=False) for i in range(2)]
    text = "[" + ",".join(parts) + "]"

    async def fake(messages, retry=True, *, on_delta=None, **kw):
        if on_delta is not None:
            acc = "["
            for p in parts:
                acc += p + ","
                on_delta(acc)
        return text

    async def fake_gene(item, facts, idx, total):
        item["gene"] = {"gate": "pass"}
        return item

    async def fake_check(item, facts, idx, total):
        item["check"] = {"badge": "ok", "tier": "verified"}
        return item, None

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
    monkeypatch.setattr(variant_mod, "_gene_one_item", fake_gene)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)
    state = dict(_FACTS_STATE, messages=[HumanMessage(content="出2道")], knobs={}, manual_order=True)
    out = asyncio.run(generate(state, {}))
    assert out["manual_order"] is False
