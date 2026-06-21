# -*- coding: utf-8 -*-
"""今晚（2026-06-12）四项整改单测：
  整改1：生题 prompt 注入确定上下文硬约束块（_context_block：考点/进度/教材版本）。
  整改2：难度判定并入生题（GENERATE/REGEN/ADD prompt 嵌入 rubric；assemble 不再独立复评）。
  整改3：编辑指令解析分粒度——「解法修正」scope（题面留只改解析），不整组重写。
  整改4：闸B 回炉松绑（regen 1 次仍 FAIL → 标 ⚠ 放行，不补题不二次回炉）。
"""

import asyncio

from langchain_core.messages import HumanMessage

import agents.variant as variant_mod
from agents.variant import (
    ADD_PROMPT,
    GENERATE_PROMPT,
    INTENT_CLARIFY,
    INTENT_SOLUTION_ONLY,
    REGEN_PROMPT,
    _context_block,
    _progress_phrase,
    _regen_max_tokens,
    validate_instruction,
)
from core import settings


# ===========================================================================
# 整改1：确定上下文硬约束块
# ===========================================================================
def _facts(**over):
    base = {
        "kp_name": "一元一次方程",
        "grade": "七年级上学期",
        "subject_id": "3071",
        "qtype": "解答",
        "stem": "解方程 2x+1=5",
        "skeleton": "移项；系数化为1",
        "dna": {
            "main_kp": {"id": "100", "name": "一元一次方程"},
            "secondary_kps": [{"id": "101", "name": "移项"}],
        },
    }
    base.update(over)
    return base


def test_context_block_has_kp_and_progress_and_method_guard():
    c = _context_block(_facts())
    assert "确定上下文" in c
    assert "一元一次方程" in c  # 主考点
    assert "移项" in c  # 连带知识点（副 kp 当路径补充）
    assert "七年级上学期" in c  # 进度
    assert "严禁使用该进度之后才学的内容" in c  # 硬约束解题方法不越界


def test_context_block_progress_from_code_when_grade_lacks_term():
    facts = _facts(grade="七年级", subject_id="3071")  # grade 缺学期 → 用 code 第4位补
    assert _progress_phrase(facts) == "七年级上学期"
    facts2 = _facts(grade="八年级", subject_id="3082")
    assert _progress_phrase(facts2) == "八年级下学期"


def test_context_block_textbook_version_only_when_set(monkeypatch):
    # 默认空 → 不注入教材版本行（绝不编造）
    monkeypatch.setattr(settings, "TEXTBOOK_VERSION", "")
    assert "教材版本" not in _context_block(_facts())
    # 配了真实来源 → 注入
    monkeypatch.setattr(settings, "TEXTBOOK_VERSION", "浙教版")
    assert "教材版本：浙教版" in _context_block(_facts())


def test_context_block_does_not_displace_conservation():
    """整改1 块与守恒白名单段并存、正交——守恒段不被新块挤失效。"""
    facts = _facts()
    prompt = (
        GENERATE_PROMPT.format(n=3, n_normal=2, n_hard=1, **facts)
        + "\n\n"
        + variant_mod._context_block(facts)
        + "\n\n"
        + variant_mod._conservation_clause(facts.get("dna"))
    )
    assert "确定上下文" in prompt  # 整改1 块在
    assert "守恒硬约束" in prompt  # W2 守恒段也在
    assert "知识点守恒" in prompt  # 守恒白名单段未被挤掉


# ===========================================================================
# 整改2：难度 rubric 已嵌入生题 prompt（不再独立复评）
# ===========================================================================
def test_difficulty_rubric_embedded_in_generate_regen_add():
    facts = _facts()
    g = GENERATE_PROMPT.format(n=3, n_normal=2, n_hard=1, **facts)
    assert "难度四档 rubric" in g
    assert "difficulty\":1~4" in g  # 难度档对齐 1~4（非 1~5）
    r = REGEN_PROMPT.format(
        kp_name="kp", grade="七年级", stem="s", level="normal", qtype="解答",
        difficulty=2, injected_kp="null",
    )
    assert "难度四档 rubric" in r
    a = ADD_PROMPT.format(
        n=2, kp_name="kp", grade="七年级", qtype="解答", stem="s", skeleton="sk", extra="无",
    )
    assert "难度四档 rubric" in a
    assert "difficulty\":1~4" in a


# ===========================================================================
# 整改3：解法修正 scope（validate + 路由）
# ===========================================================================
def test_solution_only_intent_normalized_with_method_constraint():
    parsed = {
        "intent": "解法修正",
        "method_constraint": "只能用一元一次方程，不用二元方程",
        "grade_correction": "七年级",
    }
    out = validate_instruction(parsed, 3)
    assert out["intent"] == INTENT_SOLUTION_ONLY
    assert out["method_constraint"] == "只能用一元一次方程，不用二元方程"
    assert out["grade_correction"] == "七年级"
    assert out["ops"] == []  # 物理带不动编辑 op


def test_solution_only_without_method_constraint_downgrades_clarify():
    parsed = {"intent": "解法修正", "method_constraint": None}
    out = validate_instruction(parsed, 3)
    assert out["intent"] == INTENT_CLARIFY  # 缺 method_constraint → 降级 clarify


def test_solution_only_routes_to_dedicated_node():
    state = {"pending": {"intent": INTENT_SOLUTION_ONLY, "method_constraint": "x"}}
    assert variant_mod.route_after_parse(state) == "solution_only"


def test_real_world_sample_classified_as_solution_only():
    """实测误判样本：「这里是7年级的题目，没学二元方程，只能用一元一次去解题」
    护栏层：只要 LLM 给出 intent=解法修正 + method_constraint，护栏放行（不降级）。"""
    parsed = {
        "intent": "解法修正",
        "method_constraint": "只能用一元一次方程解，七年级未学二元方程",
        "grade_correction": "七年级",
    }
    out = validate_instruction(parsed, 5)
    assert out["intent"] == INTENT_SOLUTION_ONLY
    assert out["ops"] == []


def test_exec_solution_only_node_registered():
    nodes = list(variant_mod.variant.get_graph().nodes)
    assert "exec_solution_only" in nodes


def test_exec_solution_only_rewrites_solution_keeps_stem(monkeypatch):
    """整改3 核心：可解题 → 题面不动，只换解析；重跑闸B。"""
    async def rewrite_stub(prompt_msgs, **kwargs):
        return '{"solvable": true, "solution": "只用一元一次方程：$2x=4$，$x=2$"}'

    async def check_stub(item, facts, idx, total):
        item["check"] = {"badge": "ok", "verify": variant_mod.VERIFY_SYMPY_PASS, "tier": "verified"}
        return item, None

    monkeypatch.setattr(variant_mod, "_ainvoke_text", rewrite_stub)
    monkeypatch.setattr(variant_mod, "_check_one_item", check_stub)
    monkeypatch.setattr(variant_mod, "_emit_artifact", lambda *a, **k: None)

    state = {
        "analysis": {"grade": {"value": "七年级上学期"}, "kp": {"value": "一元一次方程"}},
        "mother_dna": {"dna": {"main_kp": {"id": "100", "name": "一元一次方程"}}},
        "items": [{"stem": "2x=4 求 x", "answer": "x=2", "solution": "旧解析用二元", "qtype": "解答"}],
        "pending": {
            "intent": INTENT_SOLUTION_ONLY,
            "method_constraint": "只能用一元一次方程",
            "grade_correction": None,
        },
    }
    out = asyncio.run(variant_mod.exec_solution_only(state, {}))
    it = out["items"][0]
    assert it["stem"] == "2x=4 求 x"  # 题面保留不动
    assert "一元一次方程" in it["solution"]  # 解析按新约束重写
    assert it["from_edit"] is True  # 编辑印记（闸B FAIL 不回炉换题）


def test_exec_solution_only_resets_stale_dropped_notes(monkeypatch):
    """🔴 PRD-A-021 R4·F8：解法修正直连 assemble（绕过 solve_explain 的 dropped_notes 复位），
    须自行清空上一轮遗留的 dropped_notes，否则 assemble 头部把陈旧「剔除 N 道」误渲染进本轮。"""
    async def rewrite_stub(prompt_msgs, **kwargs):
        return '{"solvable": true, "solution": "只用一元一次方程：$x=2$"}'

    async def check_stub(item, facts, idx, total):
        item["check"] = {"badge": "ok", "verify": variant_mod.VERIFY_SYMPY_PASS, "tier": "verified"}
        return item, None

    monkeypatch.setattr(variant_mod, "_ainvoke_text", rewrite_stub)
    monkeypatch.setattr(variant_mod, "_check_one_item", check_stub)
    monkeypatch.setattr(variant_mod, "_emit_artifact", lambda *a, **k: None)

    state = {
        "analysis": {"grade": {"value": "七年级上学期"}, "kp": {"value": "一元一次方程"}},
        "mother_dna": {"dna": {"main_kp": {"id": "100", "name": "一元一次方程"}}},
        "items": [{"stem": "2x=4 求 x", "answer": "x=2", "solution": "旧", "qtype": "解答"}],
        # 上一轮 generate/验算遗留的剔除叙事（本轮解法修正不剔题，须被清掉）
        "dropped_notes": ["第3题验算失败已剔除"],
        "pending": {
            "intent": INTENT_SOLUTION_ONLY,
            "method_constraint": "只能用一元一次方程",
            "grade_correction": None,
        },
    }
    out = asyncio.run(variant_mod.exec_solution_only(state, {}))
    assert out["dropped_notes"] == []  # 🔴 F8：陈旧剔除叙事已复位，不泄漏到本轮头部


def test_exec_solution_only_unsolvable_regens_single_stem(monkeypatch):
    """整改3：某题新约束下不可解 → 单题重出题面（_regen_once），其余不动。"""
    async def rewrite_stub(prompt_msgs, **kwargs):
        return '{"solvable": false, "reason": "必须用二元才能解"}'

    regen_calls = {"n": 0}

    async def regen_stub(item, facts, feedback=None):
        regen_calls["n"] += 1
        return {"stem": "新题面（可用一元一次解）", "answer": "x=3", "solution": "s", "qtype": "解答"}

    async def check_stub(item, facts, idx, total):
        item["check"] = {"badge": "ok", "verify": variant_mod.VERIFY_SYMPY_PASS, "tier": "verified"}
        return item, None

    monkeypatch.setattr(variant_mod, "_ainvoke_text", rewrite_stub)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_stub)
    monkeypatch.setattr(variant_mod, "_check_one_item", check_stub)
    monkeypatch.setattr(variant_mod, "_emit_artifact", lambda *a, **k: None)

    state = {
        "analysis": {"grade": {"value": "七年级上学期"}, "kp": {"value": "一元一次方程"}},
        "mother_dna": {"dna": {"main_kp": {"id": "100", "name": "一元一次方程"}}},
        "items": [{"stem": "需二元方程的题", "answer": "?", "solution": "二元解", "qtype": "解答"}],
        "pending": {
            "intent": INTENT_SOLUTION_ONLY,
            "method_constraint": "只能用一元一次方程",
            "grade_correction": None,
        },
    }
    out = asyncio.run(variant_mod.exec_solution_only(state, {}))
    assert regen_calls["n"] == 1  # 单题重出题面
    assert out["items"][0]["stem"] == "新题面（可用一元一次解）"


def test_exec_solution_only_updates_grade_anchor(monkeypatch):
    """整改3：年级修正同步 analysis.grade（chip 更新），但不触发整组重锚（不清 anchored）。"""
    async def rewrite_stub(prompt_msgs, **kwargs):
        return '{"solvable": true, "solution": "新解析"}'

    async def check_stub(item, facts, idx, total):
        item["check"] = {"badge": "ok", "tier": "verified"}
        return item, None

    monkeypatch.setattr(variant_mod, "_ainvoke_text", rewrite_stub)
    monkeypatch.setattr(variant_mod, "_check_one_item", check_stub)
    monkeypatch.setattr(variant_mod, "_emit_artifact", lambda *a, **k: None)

    state = {
        "analysis": {
            "grade": {"value": "八年级", "confidence": 0.6},
            "kp": {"value": "一元一次方程", "anchored": {"id": "100", "code": "100"}},
        },
        "mother_dna": {"dna": {"main_kp": {"id": "100", "name": "一元一次方程"}}},
        "items": [{"stem": "q", "answer": "a", "solution": "s", "qtype": "解答"}],
        "pending": {
            "intent": INTENT_SOLUTION_ONLY,
            "method_constraint": "只能用一元一次方程",
            "grade_correction": "七年级",
        },
    }
    out = asyncio.run(variant_mod.exec_solution_only(state, {}))
    assert out["analysis"]["grade"]["value"] == "七年级"  # chip 更新
    assert out["analysis"]["kp"].get("anchored") is not None  # anchored 未清（不整组重锚）


# ===========================================================================
# 整改4：回炉瘦身 max_tokens
# ===========================================================================
def test_regen_max_tokens_uses_setting(monkeypatch):
    monkeypatch.setattr(settings, "VARIANT_REGEN_MAX_TOKENS", 1500)
    assert _regen_max_tokens() == 1500
    # ≤0 → 回退 VARIANT_MAX_TOKENS（关闸）
    monkeypatch.setattr(settings, "VARIANT_REGEN_MAX_TOKENS", 0)
    assert _regen_max_tokens() == settings.VARIANT_MAX_TOKENS


def test_ainvoke_text_accepts_max_tokens_override():
    """整改4：_ainvoke_text 接受 per-call max_tokens（回炉瘦身用）。仅校验签名兼容（不发真请求）。"""
    import inspect

    sig = inspect.signature(variant_mod._ainvoke_text)
    assert "max_tokens" in sig.parameters
