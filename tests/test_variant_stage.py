# -*- coding: utf-8 -*-
"""Unit tests for stage events (thought-process externalization, PRD-C-009 需求B).

Contract (BE/FE must match exactly):
  custom channel -> service stream_mode=custom -> SSE custom_data.stage =
  {"key": analyze|classify|knobs|generate|gene_gate|verify|persist,
   "title": ..., "status": running|done|warn, "detail": optional}

Iron rules under test:
- _emit_stage wraps the payload in ChatMessage(role="custom", content=[{...}])
  (the only shape langchain_to_chat_message accepts);
- outside a langgraph runtime (unit tests call nodes directly) it is a silent
  no-op -- never a crash;
- a writer that raises is swallowed -- stage is an enhancement, never a gate;
- key nodes actually emit at the contracted points (recorder monkeypatch).

Zero LLM / zero network: every LLM/IO touchpoint is monkeypatched.
"""

import asyncio
import json

from langchain_core.messages import ChatMessage, HumanMessage

import agents.variant as variant_mod
from agents import math_verify
from agents.variant import (
    analyze,
    classify,
    gene_gate,
    generate,
    persist_to_bank,
    solve_explain,
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


def _record_stages(monkeypatch):
    """Replace _emit_stage with a recorder; returns the call list."""
    calls: list[tuple] = []

    def rec(key, title, status, detail=None):
        calls.append((key, title, status, detail))

    monkeypatch.setattr(variant_mod, "_emit_stage", rec)
    return calls


# ---------------------------------------------------------------------------
# _emit_stage itself: payload shape + both swallow paths
# ---------------------------------------------------------------------------


def test_emit_stage_wraps_custom_chatmessage_with_contract_payload(monkeypatch):
    captured = []
    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: captured.append)

    variant_mod._emit_stage("verify", "程序验算", "running", "第 1/3 道")
    variant_mod._emit_stage("gene_gate", "平行度比对", "done")  # no detail

    assert len(captured) == 2
    msg = captured[0]
    assert isinstance(msg, ChatMessage) and msg.role == "custom"
    # 🔴 content must be a single-element list -> service utils takes content[0]
    assert msg.content == [
        {"stage": {"key": "verify", "title": "程序验算", "status": "running", "detail": "第 1/3 道"}}
    ]
    # detail omitted (not None-filled) when absent
    assert captured[1].content == [
        {"stage": {"key": "gene_gate", "title": "平行度比对", "status": "done"}}
    ]


def test_emit_stage_is_silent_noop_outside_runtime():
    # direct call with no langgraph runnable context: get_stream_writer raises
    # RuntimeError internally -> must be swallowed, never propagate
    assert variant_mod._emit_stage("analyze", "读图分析", "running") is None


def test_emit_stage_swallows_writer_exception(monkeypatch):
    def boom(_msg):
        raise RuntimeError("transport down")

    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: boom)
    # must not raise
    variant_mod._emit_stage("persist", "入库", "done", "成功 3 道")


def test_node_survives_raising_writer_end_to_end(monkeypatch):
    """Real _emit_stage + a writer that always raises: node still completes."""

    def boom(_msg):
        raise RuntimeError("transport down")

    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: boom)

    async def judge_pass(item, facts):
        return {
            "qtype_match": True,
            "difficulty_match": True,
            "structure_match": True,
            "surface_swapped": True,
        }

    monkeypatch.setattr(variant_mod, "_gene_judge_one", judge_pass)
    state = dict(_FACTS_STATE, items=[{"stem": "v1"}])
    out = asyncio.run(gene_gate(state, {}))
    assert out["messages"] == []
    assert out["items"][0]["gene"]["gate"] == "pass"


# ---------------------------------------------------------------------------
# node instrumentation points (recorder monkeypatch, contract details)
# ---------------------------------------------------------------------------


def test_analyze_emits_running_then_done(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def fake_llm(messages, retry=True):
        return json.dumps(
            {
                "is_question_image": True,
                "images_count": 1,
                "questions_in_image": 1,
                "grade": {"value": "七年级上学期", "confidence": 0.9},
                "kp": {"value": "一元一次方程", "confidence": 0.8},
                "qtype": {"value": "解答", "confidence": 0.9},
                "stem": "题干",
                "answer": "x=1",
                "difficulty": 3,
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = {"messages": [HumanMessage(content="https://oss.example.com/q.png")]}
    out = asyncio.run(analyze(state, {}))
    assert out["mother_dna"]["stem"] == "题干"
    assert calls == [
        ("analyze", "读图分析", "running", None),
        ("analyze", "读图分析", "done", None),
    ]


def test_analyze_emits_warn_on_non_question_image(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def fake_llm(messages, retry=True):
        return json.dumps({"is_question_image": False})

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = {"messages": [HumanMessage(content="https://oss.example.com/cat.png")]}
    asyncio.run(analyze(state, {}))
    assert calls[-1] == ("analyze", "读图分析", "warn", "未识别为题目图")


def test_classify_emits_done_with_kp_and_grade(monkeypatch):
    calls = _record_stages(monkeypatch)
    monkeypatch.setattr(variant_mod, "anchor_subject", lambda coarse: [])
    state = dict(_FACTS_STATE)
    out = asyncio.run(classify(state, {}))
    assert out["mother_confirmed"] is True
    assert calls == [
        ("classify", "锚定考点", "done", "考点「一元一次方程」·年级「七年级上学期」")
    ]


_ITEM_JSON = {
    "stem": "新题",
    "answer": "x=2",
    "solution": "略",
    "qtype": "解答",
    "difficulty": 3,
    "level": "normal",
    "injected_kp": None,
}


def test_generate_emits_knobs_summary_and_running_done_counts(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def fake_llm(messages, retry=True):
        prompt = messages[0].content
        if "出题配方" in prompt:  # KNOBS extraction round
            return json.dumps(
                {"count": 2, "difficulty_plan": None, "qtype_dist": None, "note": ""}
            )
        return json.dumps([_ITEM_JSON, _ITEM_JSON], ensure_ascii=False)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = dict(
        _FACTS_STATE,
        messages=[HumanMessage(content="https://oss.example.com/q.png 出2道题")],
    )
    out = asyncio.run(generate(state, {}))
    assert len(out["items"]) == 2
    assert calls == [
        ("knobs", "解析配方", "done", "2 道"),
        ("generate", "生成题目", "running", "2 道"),
        ("generate", "生成题目", "done", "2 道"),
    ]


def test_generate_without_teacher_text_reports_default_recipe(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def fake_llm(messages, retry=True):
        return json.dumps([_ITEM_JSON] * 3, ensure_ascii=False)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = dict(_FACTS_STATE, messages=[HumanMessage(content="https://o.ss/q.png")])
    asyncio.run(generate(state, {}))
    assert calls[0] == ("knobs", "解析配方", "done", "未指定，走默认配方（3 道 = 2 普通 + 1 难）")
    assert calls[1:] == [
        ("generate", "生成题目", "running", "3 道"),
        ("generate", "生成题目", "done", "3 道"),
    ]


def test_gene_gate_emits_per_item_running_warn_on_rework_then_done(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def judge_rework(item, facts):
        return {
            "qtype_match": True,
            "difficulty_match": True,
            "structure_match": False,
            "surface_swapped": True,
            "reason": "结构漂了",
        }

    async def regen_none(item, facts, feedback=None):
        return None  # rework fails -> keep original marked warn

    monkeypatch.setattr(variant_mod, "_gene_judge_one", judge_rework)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_none)
    state = dict(_FACTS_STATE, items=[{"stem": "v1"}, {"stem": "v2"}])
    out = asyncio.run(gene_gate(state, {}))
    assert all(it["gene"]["gate"] == "warn" for it in out["items"])
    assert calls == [
        ("gene_gate", "平行度比对", "running", "第 1/2 道"),
        ("gene_gate", "平行度比对", "warn", "第 1 道回炉重生中"),
        ("gene_gate", "平行度比对", "running", "第 2/2 道"),
        ("gene_gate", "平行度比对", "warn", "第 2 道回炉重生中"),
        ("gene_gate", "平行度比对", "done", None),
    ]


def test_solve_explain_emits_per_item_progress_then_done(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def solve_stub(stem):
        return {"solved_answer": "x=2", "solution": "解析"}

    async def verify_pass(item, solved_answer):
        return {"verdict": math_verify.PASS, "detail": "ok", "computed": "x=2"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_pass)
    items = [
        {"stem": "a", "answer": "x=2", "qtype": "解答"},
        {"stem": "b", "answer": "x=2", "qtype": "解答"},
    ]
    state = dict(_FACTS_STATE, items=items)
    out = asyncio.run(solve_explain(state, {}))
    assert all(it["check"]["badge"] == "ok" for it in out["items"])
    assert calls == [
        ("verify", "程序验算", "running", "第 1/2 道"),
        ("verify", "程序验算", "running", "第 2/2 道"),
        ("verify", "程序验算", "done", None),
    ]


def test_solve_explain_emits_warn_when_item_goes_back_to_furnace(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def solve_stub(stem):
        return {"solved_answer": "x=9", "solution": "解析"}

    async def verify_fail(item, solved_answer):
        return {"verdict": math_verify.FAIL, "detail": "mismatch", "computed": "x=9"}

    async def regen_none(item, facts, feedback=None):
        return None  # regen fails -> original kept with warn badge

    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fail)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_none)
    state = dict(_FACTS_STATE, items=[{"stem": "a", "answer": "x=2", "qtype": "解答"}])
    out = asyncio.run(solve_explain(state, {}))
    assert out["items"][0]["check"]["badge"] == "warn"
    assert ("verify", "程序验算", "warn", "第 1 道回炉重生中") in calls
    assert calls[-1] == ("verify", "程序验算", "done", None)


def test_persist_emits_running_then_done_with_counts(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def fake_persist(items, facts, token=None):
        return [
            {"role": "mother", "ok": True, "id": 100},
            {"ok": True, "id": 101},
            {"ok": True, "id": 102},
        ]

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    state = dict(
        _FACTS_STATE,
        items=[
            {"stem": "a", "answer": "1", "check": {"badge": "ok"}},
            {"stem": "b", "answer": "2", "check": {"badge": "ok"}},
        ],
    )
    out = asyncio.run(persist_to_bank(state, {}))
    assert "入库完成" in out["messages"][0].content
    assert calls == [
        ("persist", "入库", "running", "2 道"),
        ("persist", "入库", "done", "成功 2 道"),
    ]


def test_persist_emits_warn_when_bank_unreachable(monkeypatch):
    calls = _record_stages(monkeypatch)

    async def fake_persist(items, facts, token=None):
        raise RuntimeError("connect refused")

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    state = dict(_FACTS_STATE, items=[{"stem": "a", "answer": "1"}])
    out = asyncio.run(persist_to_bank(state, {}))
    assert "连不上题库服务" in out["messages"][0].content
    assert calls[-1] == ("persist", "入库", "warn", "连不上题库服务")


# ---------------------------------------------------------------------------
# artifact 快照帧（PRD-C-011 Bucket 3）：FE 题卡数据源
# 契约：ChatMessage(role="custom", content=[{"artifact": {"items":[...], "header":{...}}}])
# 发射点 = assemble 收尾 + persist_to_bank 成功后（persisted per-item 标）
# ---------------------------------------------------------------------------


def _capture_frames(monkeypatch):
    """Monkeypatch get_stream_writer with a capturer; returns the frame list."""
    captured: list = []
    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: captured.append)
    return captured


def _artifact_frames(captured):
    """Filter captured custom frames down to artifact payloads (skip stage frames)."""
    out = []
    for msg in captured:
        assert isinstance(msg, ChatMessage) and msg.role == "custom"
        assert isinstance(msg.content, list) and len(msg.content) == 1
        if "artifact" in msg.content[0]:
            out.append(msg.content[0]["artifact"])
    return out


_RICH_ITEM = {
    "stem": "解方程 2x+1=5",
    "answer": "x=2",
    "solution": "移项得 2x=4，x=2",
    "qtype": "解答",
    "difficulty": 3,
    "level": "normal",
    "check": {"badge": "ok", "verify": "sympy_pass"},
    "gene": {"gate": "pass"},
}


def test_assemble_emits_artifact_snapshot_with_contract_fields(monkeypatch):
    captured = _capture_frames(monkeypatch)
    state = dict(
        _FACTS_STATE,
        items=[dict(_RICH_ITEM), dict(_RICH_ITEM, level="hard", difficulty=4)],
        knobs={"count": 2},
    )
    out = asyncio.run(variant_mod.assemble(state, {}))
    # 节点返回值不变：单条 AIMessage，artifact 走 writer 自定义通道不进 messages
    assert len(out["messages"]) == 1
    arts = _artifact_frames(captured)
    assert len(arts) == 1
    art = arts[0]
    # items[0] 契约字段齐 + verify/gene 透传
    assert art["items"][0] == {
        "index": 1,
        "stem": "解方程 2x+1=5",
        "answer": "x=2",
        "solution": "移项得 2x=4，x=2",
        "qtype": "解答",
        "difficulty": 3,
        "level": "normal",
        "verify": "sympy_pass",
        "gene": "pass",
        "persisted": False,
    }
    assert art["items"][1]["index"] == 2
    assert art["items"][1]["level"] == "hard"
    # header：recipe 来自 knobs_desc，kp/grade 来自 analysis
    assert art["header"] == {"recipe": "2 道", "kp": "一元一次方程", "grade": "七年级上学期"}


def test_artifact_verify_falls_back_to_review_for_proof_items(monkeypatch):
    captured = _capture_frames(monkeypatch)
    # 证明类分支：check 只有 review（proof_needs_human），无 verify 键
    item = dict(_RICH_ITEM, check={"badge": "warn", "review": "proof_needs_human"})
    state = dict(_FACTS_STATE, items=[item])
    asyncio.run(variant_mod.assemble(state, {}))
    art = _artifact_frames(captured)[0]
    assert art["items"][0]["verify"] == "proof_needs_human"
    assert art["items"][0]["gene"] == "pass"


def test_artifact_nulls_and_defaults_when_fields_missing(monkeypatch):
    captured = _capture_frames(monkeypatch)
    # 裸 item（持久化旧线程存量题：无 check / 无 gene / 无 difficulty）+ 无 analysis/knobs
    state = {"items": [{"stem": "裸题"}]}
    asyncio.run(variant_mod.assemble(state, {}))
    art = _artifact_frames(captured)[0]
    assert art["items"][0] == {
        "index": 1,
        "stem": "裸题",
        "answer": "",
        "solution": "",
        "qtype": "",
        "difficulty": 0,
        "level": "normal",
        "verify": None,
        "gene": None,
        "persisted": False,
    }
    # 缺省哨兵值（未知考点/未知年级）→ None 化；空 knobs → recipe None
    assert art["header"] == {"recipe": None, "kp": None, "grade": None}


def test_persist_emits_artifact_with_per_item_persisted_flags(monkeypatch):
    captured = _capture_frames(monkeypatch)

    async def fake_persist(items, facts, token=None):
        return [
            {"role": "mother", "ok": True, "id": 100},
            {"ok": True, "id": 101},
            {"ok": False, "error": "boom"},
        ]

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    state = dict(_FACTS_STATE, items=[dict(_RICH_ITEM), dict(_RICH_ITEM)])
    out = asyncio.run(persist_to_bank(state, {}))
    assert "入库完成" in out["messages"][0].content
    arts = _artifact_frames(captured)
    assert len(arts) == 1
    # 母题回执已滤掉：var_receipts[i] 与 items[i] 同序 → persisted [True, False]
    assert [it["persisted"] for it in arts[0]["items"]] == [True, False]
    assert arts[0]["items"][0]["verify"] == "sympy_pass"


def test_persist_unreachable_path_emits_no_artifact(monkeypatch):
    captured = _capture_frames(monkeypatch)

    async def fake_persist(items, facts, token=None):
        raise RuntimeError("connect refused")

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    state = dict(_FACTS_STATE, items=[{"stem": "a", "answer": "1"}])
    asyncio.run(persist_to_bank(state, {}))
    assert _artifact_frames(captured) == []  # 异常早退：items 未变，不发更新快照


def test_emit_artifact_is_silent_noop_outside_runtime():
    # 直调无 langgraph runnable context：get_stream_writer 抛 → 必须静默吞
    state = dict(_FACTS_STATE, items=[dict(_RICH_ITEM)])
    assert variant_mod._emit_artifact(state) is None


def test_emit_artifact_swallows_writer_exception(monkeypatch):
    def boom(_msg):
        raise RuntimeError("transport down")

    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: boom)
    # must not raise
    variant_mod._emit_artifact(dict(_FACTS_STATE, items=[dict(_RICH_ITEM)]))


def test_assemble_survives_raising_writer(monkeypatch):
    """Real _emit_artifact + a writer that always raises: assemble still completes."""

    def boom(_msg):
        raise RuntimeError("transport down")

    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: boom)
    state = dict(_FACTS_STATE, items=[dict(_RICH_ITEM)])
    out = asyncio.run(variant_mod.assemble(state, {}))
    assert "举一反三" in out["messages"][0].content


# ---------------------------------------------------------------------------
# persisted 簿记（PRD-C-011 G5 修复）：persist_to_bank 把回执写回 state.items，
# 已收录的题在后续编辑轮快照不回退、二次入库被跳过、母题 id 回写 mother_dna。
# ---------------------------------------------------------------------------


def test_persist_writes_back_persisted_and_mother_id_to_state(monkeypatch):
    _capture_frames(monkeypatch)

    async def fake_persist(items, facts, token=None):
        return [
            {"role": "mother", "ok": True, "id": 100},
            {"ok": True, "id": 101},
            {"ok": False, "error": "boom"},
        ]

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    state = dict(_FACTS_STATE, items=[dict(_RICH_ITEM), dict(_RICH_ITEM)])
    out = asyncio.run(persist_to_bank(state, {}))
    # 簿记进 state：per-item persisted 按回执
    assert [it["persisted"] for it in out["items"]] == [True, False]
    # 母题雪花 id 回写 mother_dna（重试不再重复建母题）
    assert out["mother_dna"]["mother_question_id"] == 100
    # 原 DNA 字段保留
    assert out["mother_dna"]["stem"] == "母题题干"


def test_persist_skips_already_persisted_items(monkeypatch):
    captured = _capture_frames(monkeypatch)
    seen: list[list] = []

    async def fake_persist(items, facts, token=None):
        seen.append(list(items))
        return [{"ok": True, "id": 200}]

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    items = [
        dict(_RICH_ITEM, stem="已收录的题", persisted=True),
        dict(_RICH_ITEM, stem="还没入库的题"),
    ]
    state = dict(
        _FACTS_STATE,
        items=items,
        mother_dna=dict(_FACTS_STATE["mother_dna"], mother_question_id=100),
    )
    out = asyncio.run(persist_to_bank(state, {}))
    # 只把未收录的那道送去入库（绝不重复落行）
    assert len(seen) == 1 and len(seen[0]) == 1
    assert seen[0][0]["stem"] == "还没入库的题"
    # 回写后两道都 persisted=True；快照帧同步
    assert [it["persisted"] for it in out["items"]] == [True, True]
    art = _artifact_frames(captured)[-1]
    assert [it["persisted"] for it in art["items"]] == [True, True]
    assert "跳过" in out["messages"][0].content


def test_persist_all_already_persisted_is_noop(monkeypatch):
    captured = _capture_frames(monkeypatch)
    called = []

    async def fake_persist(items, facts, token=None):
        called.append(items)
        return []

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    state = dict(_FACTS_STATE, items=[dict(_RICH_ITEM, persisted=True)])
    out = asyncio.run(persist_to_bank(state, {}))
    assert called == []  # 一道都不重发
    assert "不会重复落库" in out["messages"][0].content
    assert "items" not in out  # 状态不动
    assert _artifact_frames(captured) == []


def test_assemble_snapshot_keeps_persisted_from_state(monkeypatch):
    """入库后编辑轮：assemble 重发快照读 item.persisted 簿记，徽章不回退。"""
    captured = _capture_frames(monkeypatch)
    state = dict(
        _FACTS_STATE,
        items=[dict(_RICH_ITEM, persisted=True), dict(_RICH_ITEM)],
    )
    asyncio.run(variant_mod.assemble(state, {}))
    art = _artifact_frames(captured)[0]
    assert [it["persisted"] for it in art["items"]] == [True, False]


def test_patch_hard_anchor_emits_empty_artifact_snapshot(monkeypatch):
    """patch 清 items 走 clarify/裸奔兜底不经 assemble → 必须先发空快照对齐右栏。"""
    captured = _capture_frames(monkeypatch)
    state = dict(
        _FACTS_STATE,
        items=[dict(_RICH_ITEM)],
        pending={"mother_correction": {"grade": "八年级"}},
    )
    out = asyncio.run(variant_mod.patch(state, {}))
    assert out["items"] == []
    arts = _artifact_frames(captured)
    assert len(arts) == 1
    assert arts[0]["items"] == []


def test_patch_no_field_does_not_emit_artifact(monkeypatch):
    """没拿到可 patch 字段（退化回问，items 仍在）→ 不发快照、不动右栏。"""
    captured = _capture_frames(monkeypatch)
    state = dict(_FACTS_STATE, items=[dict(_RICH_ITEM)], pending={"mother_correction": {}})
    out = asyncio.run(variant_mod.patch(state, {}))
    assert "items" not in out
    assert _artifact_frames(captured) == []
