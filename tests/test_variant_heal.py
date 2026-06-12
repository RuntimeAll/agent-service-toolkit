# -*- coding: utf-8 -*-
"""Unit tests for PRD-C-010 adversarial-review fixes inside agents.variant.

Coverage map (all LLM calls monkeypatched -> zero network):
- G5 anti-hang: _machine_verify wraps math_verify.verify in asyncio.wait_for;
  a stalled verify degrades instead of freezing solve_explain
- G5 anti-crash: _solve_one / _regen_once swallow transient LLM exceptions
  (gateway blip during a rework must never blow up gene_gate / solve_explain)
- gene mark survival: solve_explain heal paths (sympy-FAIL heal + degrade heal)
  carry the original item's gene mark onto the healed draft (auxTags.gene_gate
  stays queryable; later edit turns never re-judge / silently replace the item)
- 4d visibility matrix (PRD-C-012): _apply_visibility maps check x gene truth
  onto display tier/badge/note -- positive/neutral/silent only, warn requires
  BOTH gates low; sympy-proven-wrong items get dropped, never shown
- EXTRACT_PROMPT contract pin: the root-rejection (she-gen) guidance for
  equation_solve stays in the payload contract
"""

import asyncio
import time

import agents.variant as variant_mod
from agents import math_verify
from agents.variant import (
    EXTRACT_PROMPT,
    GENE_GATE_SKIPPED,
    NOTE_BOTH_GATES_LOW,
    NOTE_PROOF_REVIEW,
    NOTE_SELF_CHECK_OK,
    NOTE_VERIFIED_OK,
    REVIEW_PROOF,
    TIER_BOTH_LOW,
    TIER_PROOF,
    TIER_SELF_OK,
    TIER_SILENT,
    TIER_VERIFIED,
    VERIFY_FAIL_AFTER_REGEN,
    VERIFY_SYMPY_PASS,
    VERIFY_UNVERIFIED,
    _apply_visibility,
    _machine_verify,
    _regen_once,
    _solve_one,
    solve_explain,
)

_STATE_BASE = {
    "analysis": {
        "grade": {"value": "g7", "confidence": 0.9},
        "kp": {"value": "kp-x", "confidence": 0.9},
        "qtype": {"value": "qt", "confidence": 0.9},
    },
    "mother_dna": {"stem": "mother-stem", "answer": "1", "difficulty": 3},
}

_FACTS = {"kp_name": "kp-x", "grade": "g7", "qtype": "qt"}


# ---------------------------------------------------------------------------
# G5 anti-hang: verify timeout -> degrade, flow unlocked
# ---------------------------------------------------------------------------

def test_machine_verify_timeout_degrades(monkeypatch):
    async def fake_extract(stem, answer, solved_answer, qtype):
        return {"kind": "numeric", "expr": "1", "claimed": "1"}

    def stalled_verify(payload):
        time.sleep(1)  # simulates sympy stuck on a pathological payload
        return {"verdict": "pass", "detail": "too late", "computed": "1"}

    monkeypatch.setattr(variant_mod, "_extract_payload", fake_extract)
    monkeypatch.setattr(variant_mod.math_verify, "verify", stalled_verify)
    monkeypatch.setattr(variant_mod, "VERIFY_TIMEOUT_S", 0.2)

    async def main():
        # elapsed measured INSIDE the loop: asyncio.run's shutdown still joins the
        # stalled executor thread, but the agent flow itself must be unlocked fast
        t0 = time.monotonic()
        res = await _machine_verify({"stem": "s", "answer": "1", "qtype": "qt"}, "1")
        return res, time.monotonic() - t0

    res, elapsed = asyncio.run(main())
    assert res["verdict"] == math_verify.DEGRADE
    assert elapsed < 0.8  # unblocked at the 0.2s budget, well before the 1s stall


def test_machine_verify_normal_path_unaffected(monkeypatch):
    async def fake_extract(stem, answer, solved_answer, qtype):
        return {"kind": "numeric", "expr": "1+1", "claimed": "2"}

    monkeypatch.setattr(variant_mod, "_extract_payload", fake_extract)
    res = asyncio.run(_machine_verify({"stem": "s", "answer": "2", "qtype": "qt"}, "2"))
    assert res["verdict"] == math_verify.PASS


# ---------------------------------------------------------------------------
# G5 anti-crash: transient LLM exception swallowed by _solve_one / _regen_once
# ---------------------------------------------------------------------------

def test_solve_one_swallows_llm_exception(monkeypatch):
    async def boom(messages, retry=True):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", boom)
    assert asyncio.run(_solve_one("stem")) == {}  # no raise


def test_regen_once_swallows_llm_exception(monkeypatch):
    async def boom(messages, retry=True):
        raise RuntimeError("gateway down")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", boom)
    out = asyncio.run(_regen_once({"stem": "s"}, _FACTS, feedback="why"))
    assert out is None  # no raise -> caller takes the keep-original-warn path


# ---------------------------------------------------------------------------
# gene mark survives solve_explain heal (sympy-FAIL heal + degrade heal)
# ---------------------------------------------------------------------------

def _run_solve(items, monkeypatch, solve_fn, verify_fn, regen_fn):
    monkeypatch.setattr(variant_mod, "_solve_one", solve_fn)
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fn)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_fn)
    state = dict(_STATE_BASE, items=items)
    out = asyncio.run(solve_explain(state, {}))
    assert out["messages"] == []
    return out["items"]


def _solve_by_stem(answers):
    async def solve_fn(stem):
        return {
            "solved_answer": answers[stem],
            "solution": f"sol-{stem}",
            "kp_name": "kp-x",   # conserving
            "grade": "g7",
        }

    return solve_fn


def test_fail_heal_carries_original_gene(monkeypatch):
    # original: sympy FAIL -> regen draft passes -> healed draft must keep gene
    async def verify_fn(item, solved_answer):
        if item.get("stem") == "old":
            return {"verdict": "fail", "detail": "wrong", "computed": "3"}
        return {"verdict": "pass", "detail": "ok", "computed": "5"}

    async def regen_fn(item, facts, feedback=None):
        assert feedback  # sympy computed/detail injected into the rework prompt
        return {"stem": "new", "answer": "5", "solution": "s", "qtype": "qt",
                "difficulty": 3, "level": "normal", "injected_kp": None}

    items = _run_solve(
        [{"stem": "old", "answer": "1", "qtype": "qt", "gene": {"gate": "pass"}}],
        monkeypatch,
        _solve_by_stem({"old": "3", "new": "5"}),
        verify_fn,
        regen_fn,
    )
    assert items[0]["stem"] == "new"
    assert items[0]["check"]["verify"] == VERIFY_SYMPY_PASS
    assert items[0]["gene"] == {"gate": "pass"}  # carried, not lost


def test_degrade_heal_marks_gene_skipped_when_original_had_none(monkeypatch):
    # degrade branch heal: original had no gene (legacy thread) -> healed draft
    # gets the skipped mark (gene key always present after a heal)
    async def verify_fn(item, solved_answer):
        return {"verdict": "degrade", "detail": "cannot model", "computed": None}

    async def regen_fn(item, facts, feedback=None):
        return {"stem": "new", "answer": "5", "solution": "s", "qtype": "qt",
                "difficulty": 3, "level": "normal", "injected_kp": None}

    items = _run_solve(
        [{"stem": "old", "answer": "1", "qtype": "qt"}],
        monkeypatch,
        _solve_by_stem({"old": "2", "new": "5"}),  # old mismatch -> heal; new match
        verify_fn,
        regen_fn,
    )
    assert items[0]["stem"] == "new"
    assert items[0]["check"]["verify"] == VERIFY_UNVERIFIED
    assert items[0]["gene"] == {"gate": GENE_GATE_SKIPPED, "reason": "healed-in-solve"}


def test_degrade_heal_carries_original_gene(monkeypatch):
    async def verify_fn(item, solved_answer):
        return {"verdict": "degrade", "detail": "cannot model", "computed": None}

    async def regen_fn(item, facts, feedback=None):
        return {"stem": "new", "answer": "5", "solution": "s", "qtype": "qt",
                "difficulty": 3, "level": "normal", "injected_kp": None}

    items = _run_solve(
        [{"stem": "old", "answer": "1", "qtype": "qt", "gene": {"gate": "warn", "reason": "r"}}],
        monkeypatch,
        _solve_by_stem({"old": "2", "new": "5"}),
        verify_fn,
        regen_fn,
    )
    assert items[0]["gene"] == {"gate": "warn", "reason": "r"}


# ---------------------------------------------------------------------------
# 4d visibility matrix (PRD-C-012): only-say-good, warn needs BOTH gates low
# ---------------------------------------------------------------------------

def test_visibility_sympy_pass_is_verified_even_with_gene_warn():
    # strong positive wins; a single low gate (gene) stays silent
    item = {"solution": "sol", "check": {"badge": "ok", "verify": VERIFY_SYMPY_PASS},
            "gene": {"gate": "warn"}}
    _apply_visibility(item)
    assert item["check"]["tier"] == TIER_VERIFIED
    assert item["check"]["badge"] == "ok"
    assert NOTE_VERIFIED_OK in item["solution"]
    assert "⚠" not in item["solution"]


def test_visibility_self_check_match_is_light_positive():
    item = {"solution": "sol",
            "check": {"badge": "ok", "verify": VERIFY_UNVERIFIED, "self_check": "match"}}
    _apply_visibility(item)
    assert item["check"]["tier"] == TIER_SELF_OK
    assert NOTE_SELF_CHECK_OK in item["solution"]
    assert "未经程序验算" not in item["solution"]  # old scary banner is gone (G3)


def test_visibility_single_low_gate_stays_silent():
    # verify-side low (mismatch) but gene ok -> silent: no note, badge ok
    item = {"solution": "sol",
            "check": {"badge": "warn", "verify": VERIFY_UNVERIFIED, "self_check": "mismatch"},
            "gene": {"gate": "pass"}}
    _apply_visibility(item)
    assert item["check"]["tier"] == TIER_SILENT
    assert item["check"]["badge"] == "ok"
    assert item["solution"] == "sol"  # nothing appended


def test_visibility_both_gates_low_warns():
    item = {"solution": "sol",
            "check": {"badge": "warn", "verify": VERIFY_UNVERIFIED, "self_check": "mismatch"},
            "gene": {"gate": "warn"}}
    _apply_visibility(item)
    assert item["check"]["tier"] == TIER_BOTH_LOW
    assert item["check"]["badge"] == "warn"
    assert NOTE_BOTH_GATES_LOW in item["solution"]


def test_visibility_proof_is_neutral_and_idempotent():
    item = {"solution": "sol",
            "check": {"badge": "ok", "review": REVIEW_PROOF, "solved_answer": None}}
    _apply_visibility(item)
    _apply_visibility(item)  # idempotent: note appended once
    assert item["check"]["tier"] == TIER_PROOF
    assert item["solution"].count(NOTE_PROOF_REVIEW) == 1


def test_visibility_proof_struct_low_plus_gene_warn_is_both_low():
    item = {"solution": "sol",
            "check": {"badge": "warn", "review": REVIEW_PROOF, "solved_answer": None},
            "gene": {"gate": "warn"}}
    _apply_visibility(item)
    assert item["check"]["tier"] == TIER_BOTH_LOW


# ---------------------------------------------------------------------------
# 🔴 整改4（2026-06-12·闸B 回炉松绑）：sympy FAIL → 回炉 1 次仍 FAIL → **标 ⚠ 放行**
#   （verify=fail_after_regen），不再剔除、不再补题、不再二次回炉。
# ---------------------------------------------------------------------------

def test_fail_after_failed_heal_kept_with_warn(monkeypatch):
    async def verify_fn(item, solved_answer):
        return {"verdict": "fail", "detail": "wrong", "computed": "3"}

    async def regen_fn(item, facts, feedback=None):
        return None  # heal attempt fails -> 整改4：保留打 ⚠（不剔除、不补题）

    monkeypatch.setattr(variant_mod, "_solve_one", _solve_by_stem({"old": "3"}))
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fn)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_fn)
    state = dict(_STATE_BASE, items=[{"stem": "old", "answer": "1", "qtype": "解答"}])
    out = asyncio.run(solve_explain(state, {}))
    # 整改4：题不再被剔除，保留 + 标 ⚠（verify=fail_after_regen）
    assert len(out["items"]) == 1
    assert out["items"][0]["check"]["verify"] == VERIFY_FAIL_AFTER_REGEN
    assert out["dropped_notes"] == []  # 不再有剔除叙事


def test_fail_heal_success_keeps_count_and_no_drop(monkeypatch):
    async def verify_fn(item, solved_answer):
        if item.get("stem") == "old":
            return {"verdict": "fail", "detail": "wrong", "computed": "3"}
        return {"verdict": "pass", "detail": "ok", "computed": "5"}

    async def regen_fn(item, facts, feedback=None):
        return {"stem": "new", "answer": "5", "solution": "s", "qtype": "qt",
                "difficulty": 3, "level": "normal", "injected_kp": None}

    monkeypatch.setattr(variant_mod, "_solve_one", _solve_by_stem({"old": "3", "new": "5"}))
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fn)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_fn)
    state = dict(_STATE_BASE, items=[{"stem": "old", "answer": "1", "qtype": "qt"}])
    out = asyncio.run(solve_explain(state, {}))
    assert len(out["items"]) == 1
    assert out["dropped_notes"] == []
    assert out["items"][0]["check"]["tier"] == TIER_VERIFIED


# ---------------------------------------------------------------------------
# 🔴 整改4（2026-06-12·闸B 回炉松绑）：回炉只 1 次/题；仍 FAIL → 标 ⚠ 放行，
#   **不再补题、不再二次回炉**。补题（replenish）路径整体退役。
# ---------------------------------------------------------------------------

def test_fail_regen_once_then_kept_warn_no_replenish(monkeypatch):
    """回炉(heal) 1 次失败 → 不再补题、不二次回炉；保留原题打 ⚠（verify=fail_after_regen）。"""
    calls = {"n": 0}

    async def verify_fn(item, solved_answer):
        return {"verdict": "fail", "detail": "wrong", "computed": "3"}

    async def regen_fn(item, facts, feedback=None):
        calls["n"] += 1
        return None  # heal fails -> 整改4：直接标 ⚠ 放行，不再有第二次回炉/补题

    monkeypatch.setattr(variant_mod, "_solve_one", _solve_by_stem({"old": "3"}))
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fn)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_fn)
    state = dict(_STATE_BASE, items=[{"stem": "old", "answer": "1", "qtype": "qt",
                                      "gene": {"gate": "pass"}}])
    out = asyncio.run(solve_explain(state, {}))
    assert calls["n"] == 1  # 整改4：只一次回炉，不再补题/二次回炉
    assert len(out["items"]) == 1  # 题保留（不剔除）
    assert out["items"][0]["check"]["verify"] == VERIFY_FAIL_AFTER_REGEN
    assert out["dropped_notes"] == []  # 不再有剔除/少一道叙事


def test_fail_regen_succeeds_keeps_count(monkeypatch):
    """回炉 1 次成功（守恒+sympy 双过）→ 顶位，数量不变（与整改4 无冲突，回炉成功路径不变）。"""
    calls = {"n": 0}

    async def verify_fn(item, solved_answer):
        if item.get("stem") == "补":
            return {"verdict": "pass", "detail": "ok", "computed": "5"}
        return {"verdict": "fail", "detail": "wrong", "computed": "3"}

    async def regen_fn(item, facts, feedback=None):
        calls["n"] += 1
        return {"stem": "补", "answer": "5", "solution": "s", "qtype": "qt",
                "difficulty": 3, "level": "normal", "injected_kp": None}

    monkeypatch.setattr(variant_mod, "_solve_one", _solve_by_stem({"old": "3", "补": "5"}))
    monkeypatch.setattr(variant_mod, "_machine_verify", verify_fn)
    monkeypatch.setattr(variant_mod, "_regen_once", regen_fn)
    state = dict(_STATE_BASE, items=[{"stem": "old", "answer": "1", "qtype": "qt",
                                      "gene": {"gate": "pass"}}])
    out = asyncio.run(solve_explain(state, {}))
    assert calls["n"] == 1  # 回炉成功只调一次
    assert len(out["items"]) == 1
    assert out["items"][0]["stem"] == "补"
    assert out["items"][0]["check"]["verify"] == VERIFY_SYMPY_PASS
    assert out["items"][0]["gene"] == {"gate": "pass"}  # 槽位 gene 标记随愈合保留
    assert out["dropped_notes"] == []


# ---------------------------------------------------------------------------
# EXTRACT_PROMPT contract pin: root-rejection guidance present (F4)
# ---------------------------------------------------------------------------

def test_extract_prompt_contract_covers_root_rejection():
    assert "舍根" in EXTRACT_PROMPT
    assert "增根" in EXTRACT_PROMPT
