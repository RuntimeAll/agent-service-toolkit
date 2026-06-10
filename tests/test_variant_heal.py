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
- card dedupe: _fmt_item skips its warn line when solve_explain already
  appended the same NOTE_* into item.solution (no double banner on the card)
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
    NOTE_PROOF_REVIEW,
    NOTE_UNVERIFIED,
    NOTE_VERIFY_FAIL,
    REVIEW_PROOF,
    VERIFY_FAIL_AFTER_REGEN,
    VERIFY_SYMPY_PASS,
    VERIFY_UNVERIFIED,
    _append_card_note,
    _fmt_item,
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
# card dedupe: NOTE embedded in solution -> _fmt_item must not repeat the banner
# ---------------------------------------------------------------------------

def test_fmt_item_proof_note_rendered_once():
    item = {"stem": "s", "answer": "a", "solution": "sol",
            "check": {"badge": "warn", "review": REVIEW_PROOF, "solved_answer": None}}
    _append_card_note(item, NOTE_PROOF_REVIEW)  # what solve_explain does
    card = _fmt_item(1, item)
    assert card.count(NOTE_PROOF_REVIEW) == 1


def test_fmt_item_verify_fail_note_rendered_once():
    item = {"stem": "s", "answer": "a", "solution": "sol",
            "check": {"badge": "warn", "verify": VERIFY_FAIL_AFTER_REGEN, "computed": "3"}}
    _append_card_note(item, NOTE_VERIFY_FAIL)
    card = _fmt_item(1, item)
    assert card.count("程序验算未通过") == 1


def test_fmt_item_unverified_note_rendered_once():
    item = {"stem": "s", "answer": "a", "solution": "sol",
            "check": {"badge": "warn", "verify": VERIFY_UNVERIFIED, "solved_answer": "2"}}
    _append_card_note(item, NOTE_UNVERIFIED)
    card = _fmt_item(1, item)
    assert card.count(NOTE_UNVERIFIED) == 1
    assert "我没算准" not in card  # solution note wins; no second overlapping banner


def test_fmt_item_legacy_warn_item_still_gets_banner():
    # items without an embedded note (pre-PRD-C-010 state) keep the warn line
    card = _fmt_item(
        1,
        {"stem": "s", "answer": "a", "solution": "sol",
         "check": {"badge": "warn", "solved_answer": "2"}},
    )
    assert "我没算准" in card


# ---------------------------------------------------------------------------
# EXTRACT_PROMPT contract pin: root-rejection guidance present (F4)
# ---------------------------------------------------------------------------

def test_extract_prompt_contract_covers_root_rejection():
    assert "舍根" in EXTRACT_PROMPT
    assert "增根" in EXTRACT_PROMPT
