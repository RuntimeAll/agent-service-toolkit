# -*- coding: utf-8 -*-
"""PRD-C-014 B4·T2 · 有界 LLM 锚定重做单测（revise_item）。

铁律对照（CLAUDE.md §4/§5 / 22-SSOT）：
- skeleton/scene：有界 LLM 只重写该文本维，**diff 锁 target**（G12：不动 qtype/answer/difficulty/kp）→ 标 manual。
- whole：复用 REGEN 路径重出 → 闸B sympy 重验（判决只读 verdict，_check_one_item from_edit 模式）→ 标 manual。
- LLM 失败/解析不出 → 降级回 {ok:False, error}，绝不崩。
- 全部 LLM/sympy monkeypatch → 零网络。
"""

import asyncio
import copy
import json

import agents.variant as variant_mod
from agents import math_verify
from agents.variant import TIER_MANUAL, revise_item

_BASE_STATE = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {"value": "一元一次方程", "confidence": 0.9,
               "anchored": {"id": "30710101", "code": "30710101", "name": "一元一次方程"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干", "answer": "x=1", "difficulty": 3,
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
            "skeleton": ["移项"], "hard_points": [], "tags": ["解方程"],
            "scene": "纯代数", "difficulty": 3, "flags": [],
        },
    },
}


def _state(items):
    s = copy.deepcopy(_BASE_STATE)
    s["items"] = items
    return s


# ===========================================================================
# skeleton / scene：只动 target，其余维不漂移（G12）
# ===========================================================================


def test_revise_skeleton_only_changes_solution(monkeypatch):
    captured = {}

    async def fake_llm(messages, *a, **k):
        captured["called"] = True
        return json.dumps({"skeleton": "第一步移项；第二步【合并同类项】；第三步解出 x"})

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)

    items = [{"stem": "2x+1=5", "answer": "x=2", "qtype": "解答",
              "difficulty": 3, "level": "normal", "solution": "旧骨架"}]
    state = _state(items)
    update, result, err = asyncio.run(revise_item(state, 1, "skeleton", "讲清合并同类项那步"))
    assert err is None and result["ok"] is True
    it = update["items"][0]
    # solution（骨架载体）改了
    assert "合并同类项" in it["solution"]
    # diff 锁 target：其余维不动
    assert it["stem"] == "2x+1=5"
    assert it["answer"] == "x=2"
    assert it["qtype"] == "解答"
    assert it["difficulty"] == 3
    # 标 manual 中性
    assert it["manual_edited"] is True
    assert it["check"] == {"tier": TIER_MANUAL}


def test_revise_scene_changes_mother_dna_only(monkeypatch):
    async def fake_llm(messages, *a, **k):
        return json.dumps({"scene": "篮球比赛得分场景"})

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)

    items = [{"stem": "2x=4", "answer": "x=2", "qtype": "解答", "difficulty": 2}]
    state = _state(items)
    update, result, err = asyncio.run(revise_item(state, 1, "scene", "换成体育场景"))
    assert err is None and result["ok"] is True
    # scene 落 mother_dna.dna.scene（母题级表皮维）
    assert update["mother_dna"]["dna"]["scene"] == "篮球比赛得分场景"
    # 题本身不动（diff 锁）
    it = update["items"][0]
    assert it["stem"] == "2x=4"
    assert it["answer"] == "x=2"
    assert it["check"] == {"tier": TIER_MANUAL}


def test_revise_text_sanitizes_output(monkeypatch):
    async def fake_llm(messages, *a, **k):
        return json.dumps({"skeleton": r"先算 \(x^2\) 再开方\n得解"})

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _state([{"stem": "q", "answer": "a", "qtype": "解答", "solution": "old"}])
    update, result, err = asyncio.run(revise_item(state, 1, "skeleton", "改"))
    assert result["ok"] is True
    # 富文本净化：\( \) → $ $；字面 \n → 换行
    assert update["items"][0]["solution"] == "先算 $x^2$ 再开方\n得解"


def test_revise_text_llm_failure_degrades(monkeypatch):
    async def boom(messages, *a, **k):
        raise RuntimeError("网关抖动")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", boom)
    state = _state([{"stem": "q", "answer": "a", "qtype": "解答", "solution": "old"}])
    update, result, err = asyncio.run(revise_item(state, 1, "scene", "改"))
    assert err is None  # 不抛、不 400
    assert result["ok"] is False
    assert "失败" in result["error"]
    assert update == {}  # 降级不改 state


def test_revise_text_unparseable_degrades(monkeypatch):
    async def fake_llm(messages, *a, **k):
        return "这不是 JSON，模型跑偏了"

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _state([{"stem": "q", "answer": "a", "qtype": "解答", "solution": "old"}])
    update, result, err = asyncio.run(revise_item(state, 1, "skeleton", "改"))
    assert err is None
    assert result["ok"] is False
    assert update == {}


# ===========================================================================
# G12a：skeleton 重做 → _item_dna 反映 item 级新值，母题 dna.skeleton 未动
# ===========================================================================


def test_revise_skeleton_reflected_in_item_dna(monkeypatch):
    async def fake_llm(messages, *a, **k):
        return json.dumps({"skeleton": "第一步移项；第二步【合并同类项】；第三步解出 x"})

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)

    items = [{"stem": "2x+1=5", "answer": "x=2", "qtype": "解答",
              "difficulty": 3, "level": "normal", "solution": "旧骨架"}]
    state = _state(items)
    update, result, err = asyncio.run(revise_item(state, 1, "skeleton", "讲清合并同类项那步"))
    assert err is None and result["ok"] is True
    it = update["items"][0]

    # _item_dna(item) 的 skeleton 反映 item 级新值（FE DNA 面板数据源走的就是这个）
    facts = variant_mod._mother_facts(state)
    dna_panel = variant_mod._item_dna(it, facts)
    assert "合并同类项" in (dna_panel["skeleton"] or "")
    # item 级骨架字段落了
    assert "合并同类项" in it["skeleton"]
    # 🔴 母题组级 dna.skeleton（守恒基因）未动
    assert state["mother_dna"]["dna"]["skeleton"] == ["移项"]
    assert update.get("mother_dna") is None  # skeleton 路径不改母题级


def test_item_dna_skeleton_falls_back_to_mother_when_no_item_override():
    """item 无 skeleton override → _item_dna 回退母题 dna.skeleton（与 hard_points 同模式）。"""
    state = _state([{"stem": "q", "answer": "a", "qtype": "解答"}])
    facts = variant_mod._mother_facts(state)
    it = {"stem": "q"}  # 无 item 级 skeleton
    dna_panel = variant_mod._item_dna(it, facts)
    assert dna_panel["skeleton"] == "移项"  # 母题 ["移项"] → 拼成 str


# ===========================================================================
# whole：REGEN 重出 → 闸B sympy 重验
# ===========================================================================


def test_revise_whole_regen_then_reverify_pass(monkeypatch):
    regen_seen = {}

    async def regen_stub(item, facts, feedback=None):
        regen_seen["edit_note"] = item.get("edit_note")
        regen_seen["from_edit"] = item.get("from_edit")
        return {"stem": "3x=9", "answer": "x=3", "qtype": "解答",
                "difficulty": 4, "level": "hard"}

    async def solve_stub(stem):
        return {"solved_answer": "x=3", "solution": "解析"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=3"}

    grade_seen = {}

    async def grade_stub(items):
        grade_seen["called"] = True
        grade_seen["items"] = items
        # rubric 对新题断言为 4（压轴）→ 难度更新
        return [{**items[0], "difficulty": 4}]

    monkeypatch.setattr(variant_mod, "_regen_once", regen_stub)
    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)
    monkeypatch.setattr(variant_mod, "_grade_difficulty", grade_stub)

    items = [{"stem": "2x=4", "answer": "x=2", "qtype": "解答", "difficulty": 2, "_seq": 1}]
    state = _state(items)
    update, result, err = asyncio.run(revise_item(state, 1, "whole", "难一点"))
    assert err is None and result["ok"] is True
    it = update["items"][0]
    assert it["stem"] == "3x=9"  # 重出题顶位
    # instruction 注入成 edit_note → REGEN 收到（老师意志注回）
    assert regen_seen["edit_note"] == "难一点"
    assert regen_seen["from_edit"] is True
    # 闸B sympy PASS → tier verified（判决只读 verdict）
    assert it["check"]["verify"] == variant_mod.VERIFY_SYMPY_PASS
    assert it["check"]["tier"] == variant_mod.TIER_VERIFIED
    assert it["manual_edited"] is True
    assert it["_seq"] == 1  # 簿记跟题走
    # G12b：whole 重做后调用了难度重判（rubric 绝对调用），difficulty 更新、level 同步 hard
    assert grade_seen["called"] is True
    assert it["difficulty"] == 4
    assert it["level"] == "hard"


def test_revise_whole_fail_kept_warn_not_dropped(monkeypatch):
    # 重出后 sympy FAIL → from_edit 短路：保留打 ⚠（不剔除、不二次回炉换题）
    inner_regen = {"n": 0}

    async def regen_stub(item, facts, feedback=None):
        inner_regen["n"] += 1
        if inner_regen["n"] == 1:
            return {"stem": "重出题", "answer": "x=9", "qtype": "解答", "difficulty": 3}
        return {"stem": "不该二次回炉", "answer": "?"}

    async def solve_stub(stem):
        return {"solved_answer": "x=9", "solution": "解析"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "mismatch", "computed": "x=9"}

    async def grade_stub(items):
        return [{**items[0], "difficulty": 3}]

    monkeypatch.setattr(variant_mod, "_regen_once", regen_stub)
    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_grade_difficulty", grade_stub)

    state = _state([{"stem": "2x=4", "answer": "x=2", "qtype": "解答", "difficulty": 2}])
    update, result, err = asyncio.run(revise_item(state, 1, "whole", "改一下"))
    assert err is None and result["ok"] is True
    it = update["items"][0]
    assert it["stem"] == "重出题"  # 保留重出题、不剔除
    assert it["check"]["verify"] == variant_mod.VERIFY_FAIL_AFTER_REGEN
    assert it["check"]["tier"] in (variant_mod.TIER_BOTH_LOW, variant_mod.TIER_SILENT)
    assert inner_regen["n"] == 1  # from_edit 短路：闸B 不再二次回炉换题


def test_revise_whole_regen_returns_none_degrades(monkeypatch):
    async def regen_none(item, facts, feedback=None):
        return None  # 重出失败

    monkeypatch.setattr(variant_mod, "_regen_once", regen_none)
    state = _state([{"stem": "2x=4", "answer": "x=2", "qtype": "解答", "difficulty": 2}])
    update, result, err = asyncio.run(revise_item(state, 1, "whole", "改"))
    assert err is None
    assert result["ok"] is False
    assert "重出失败" in result["error"]
    assert update == {}  # 保留原题不动


def test_revise_whole_grade_failure_keeps_old_difficulty(monkeypatch):
    """G12b 降级：难度重判失败（_grade_difficulty 内部降级返回原值）→ 保留原难度，不崩。"""
    async def regen_stub(item, facts, feedback=None):
        return {"stem": "新题", "answer": "x=3", "qtype": "解答", "difficulty": 2}

    async def solve_stub(stem):
        return {"solved_answer": "x=3", "solution": "解析"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=3"}

    async def grade_degrade(items):
        # _grade_difficulty 真实降级语义：LLM 异常/解析失败 → 原样返回（difficulty 不动）
        return [dict(it) for it in items]

    monkeypatch.setattr(variant_mod, "_regen_once", regen_stub)
    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)
    monkeypatch.setattr(variant_mod, "_grade_difficulty", grade_degrade)

    items = [{"stem": "2x=4", "answer": "x=2", "qtype": "解答",
              "difficulty": 2, "level": "normal"}]
    state = _state(items)
    update, result, err = asyncio.run(revise_item(state, 1, "whole", "难一点"))
    assert err is None and result["ok"] is True
    it = update["items"][0]
    # 重判降级保留原难度（draft 给的 2，未被升档），level 不被误标 hard
    assert it["difficulty"] == 2
    assert it.get("level") != "hard"  # 未升档 → 不强标 hard
    assert it["manual_edited"] is True


def test_revise_whole_grade_levels_up_when_difficulty_rises(monkeypatch):
    """G12b：rubric 把新题判得比原值高（2→3）→ level 升 hard（不靠关键词 hack，由 rubric 断言）。"""
    async def regen_stub(item, facts, feedback=None):
        return {"stem": "新题", "answer": "x=3", "qtype": "解答", "difficulty": 2}

    async def solve_stub(stem):
        return {"solved_answer": "x=3", "solution": "解析"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=3"}

    async def grade_up(items):
        return [{**items[0], "difficulty": 3}]  # rubric 判 3，高于原 2

    monkeypatch.setattr(variant_mod, "_regen_once", regen_stub)
    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)
    monkeypatch.setattr(variant_mod, "_grade_difficulty", grade_up)

    items = [{"stem": "2x=4", "answer": "x=2", "qtype": "解答",
              "difficulty": 2, "level": "normal"}]
    state = _state(items)
    update, result, err = asyncio.run(revise_item(state, 1, "whole", "难一点"))
    assert err is None and result["ok"] is True
    it = update["items"][0]
    assert it["difficulty"] == 3
    assert it["level"] == "hard"  # 较原值升档 → 同步 hard


# ===========================================================================
# 越界 / 非法 target → 400
# ===========================================================================


def test_revise_index_out_of_range_rejected():
    state = _state([{"stem": "q1"}])
    for bad in (0, 2, -1):
        update, result, err = asyncio.run(revise_item(state, bad, "scene", "x"))
        assert err is not None and "越界" in err
        assert update == {} and result == {}


def test_revise_illegal_target_rejected():
    state = _state([{"stem": "q1"}])
    update, result, err = asyncio.run(revise_item(state, 1, "nonsense", "x"))
    assert err is not None and "非法 target" in err
    assert update == {} and result == {}


def test_revise_does_not_mutate_input_state(monkeypatch):
    async def fake_llm(messages, *a, **k):
        return json.dumps({"scene": "新场景"})

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _state([{"stem": "q", "answer": "a", "qtype": "解答", "solution": "old"}])
    asyncio.run(revise_item(state, 1, "scene", "改"))
    # 原 state 不被改
    assert state["mother_dna"]["dna"]["scene"] == "纯代数"
    assert state["items"][0].get("manual_edited") is None
