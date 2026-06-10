# -*- coding: utf-8 -*-
"""Unit tests for Gate-A (gene gate, PRD-C-010): parallel-question DNA check.

Orthogonal to Gate-B (sympy answer verification):
- skeleton genes (qtype / difficulty / structure) MUST match, otherwise rework;
- surface (numbers / scene) MUST be swapped, otherwise it is a replay -> rework;
- judge LLM failure / JSON failure -> item passes through marked "skipped"
  (Gate-A is an enhancement, never a blocker -> G5);
- after one regen the item still failing -> keep original, mark "warn",
  visible card note + auxTags.gene_gate transmission (warn does NOT block persist).

gene_gate_decision is a pure function (zero LLM / zero IO).
Node-level behavior is tested with monkeypatched judge/regen (no network).
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import (
    GENE_GATE_PASS,
    GENE_GATE_SKIPPED,
    GENE_GATE_WARN,
    NOTE_GENE_WARN,
    _fmt_item,
    gene_gate,
    gene_gate_decision,
    variant,
)
from agents.variant_support import build_create_bo


def _judge(**over):
    base = {
        "qtype_match": True,
        "difficulty_match": True,
        "structure_match": True,
        "surface_swapped": True,
        "reason": "parallel",
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# gene_gate_decision: pure-function verdict
# ---------------------------------------------------------------------------

def test_all_genes_match_and_surface_swapped_passes():
    assert gene_gate_decision(_judge()) == "pass"


def test_qtype_mismatch_reworks():
    assert gene_gate_decision(_judge(qtype_match=False)) == "rework"


def test_difficulty_mismatch_reworks():
    assert gene_gate_decision(_judge(difficulty_match=False)) == "rework"


def test_structure_mismatch_reworks():
    assert gene_gate_decision(_judge(structure_match=False)) == "rework"


def test_surface_not_swapped_is_replay_reworks():
    # all skeleton genes intact but the skin was not changed -> a replay, not a variant
    assert gene_gate_decision(_judge(surface_swapped=False)) == "rework"


def test_missing_keys_treated_as_false_reworks():
    assert gene_gate_decision({}) == "rework"
    assert gene_gate_decision({"reason": "no flags at all"}) == "rework"


def test_string_booleans_are_coerced():
    # LLMs occasionally emit "true"/"false" strings instead of JSON booleans
    assert (
        gene_gate_decision(
            _judge(
                qtype_match="true",
                difficulty_match="true",
                structure_match="true",
                surface_swapped="true",
            )
        )
        == "pass"
    )
    assert gene_gate_decision(_judge(structure_match="false")) == "rework"


def test_decision_does_not_mutate_input():
    judge = _judge(structure_match=False)
    snapshot = dict(judge)
    gene_gate_decision(judge)
    assert judge == snapshot


# ---------------------------------------------------------------------------
# gene_gate node behavior (judge / regen monkeypatched -> no LLM, no network)
# ---------------------------------------------------------------------------

_STATE_BASE = {
    "analysis": {
        "grade": {"value": "g7", "confidence": 0.9},
        "kp": {"value": "kp-x", "confidence": 0.9},
        "qtype": {"value": "qt", "confidence": 0.9},
    },
    "mother_dna": {"stem": "mother-stem", "answer": "1", "difficulty": 3},
}


def _run_gate(items, monkeypatch, judge_fn, regen_fn=None, solve_fn=None):
    monkeypatch.setattr(variant_mod, "_gene_judge_one", judge_fn)
    if regen_fn is not None:
        monkeypatch.setattr(variant_mod, "_regen_once", regen_fn)

    if solve_fn is None:
        # rework acceptance now runs a code-level conservation check via _solve_one;
        # default stub = conserving (kp/grade match the mother facts), no LLM/network
        async def solve_fn(stem):
            return {"kp_name": "kp-x", "grade": "g7"}

    monkeypatch.setattr(variant_mod, "_solve_one", solve_fn)
    state = dict(_STATE_BASE, items=items)
    out = asyncio.run(gene_gate(state, {}))
    assert out["messages"] == []  # node contract: always returns a messages key
    return out["items"]


def test_judge_failure_marks_skipped_and_lets_item_through(monkeypatch):
    async def judge_none(item, facts):
        return None

    items = _run_gate([{"stem": "v1"}], monkeypatch, judge_none)
    assert items[0]["gene"]["gate"] == GENE_GATE_SKIPPED
    assert items[0]["stem"] == "v1"  # item itself untouched


def test_judge_pass_marks_pass_without_regen(monkeypatch):
    regen_calls = []

    async def judge_ok(item, facts):
        return _judge()

    async def regen_spy(item, facts, feedback=None):
        regen_calls.append(item)
        return None

    items = _run_gate([{"stem": "v1"}], monkeypatch, judge_ok, regen_spy)
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS
    assert regen_calls == []  # no rework budget spent on a passing item


def test_rework_then_regen_passes_replaces_item_without_check(monkeypatch):
    async def judge_by_stem(item, facts):
        # original fails (replay), regenerated draft passes
        if item.get("stem") == "old":
            return _judge(surface_swapped=False)
        return _judge()

    async def regen_draft(item, facts, feedback=None):
        assert feedback  # gene feedback must be injected into the rework prompt
        return {"stem": "new", "answer": "2", "solution": "s", "level": "normal"}

    items = _run_gate([{"stem": "old"}], monkeypatch, judge_by_stem, regen_draft)
    assert items[0]["stem"] == "new"
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS
    # Gate-B orthogonality: regenerated draft carries no check -> solve_explain must verify it
    assert "check" not in items[0]


def test_rework_still_failing_keeps_original_with_warn(monkeypatch):
    async def judge_always_fail(item, facts):
        return _judge(structure_match=False, reason="skeleton broken")

    async def regen_draft(item, facts, feedback=None):
        return {"stem": "new", "answer": "2"}

    items = _run_gate([{"stem": "old"}], monkeypatch, judge_always_fail, regen_draft)
    assert items[0]["stem"] == "old"  # regen draft discarded, original kept
    assert items[0]["gene"]["gate"] == GENE_GATE_WARN
    assert items[0]["gene"]["reason"] == "skeleton broken"


def test_rework_regen_breaking_conservation_keeps_original_warn(monkeypatch):
    # regen draft satisfies the LLM re-judge, but the code-level conservation
    # check (main kp + grade hard-conserved) rejects it -> original kept, warn
    async def judge_by_stem(item, facts):
        if item.get("stem") == "old":
            return _judge(surface_swapped=False, reason="replay")
        return _judge()

    async def regen_draft(item, facts, feedback=None):
        return {"stem": "new", "answer": "2", "solution": "s", "level": "normal"}

    async def solve_breaks_conservation(stem):
        return {"kp_name": "totally-different-topic", "grade": "ninth"}

    items = _run_gate(
        [{"stem": "old"}], monkeypatch, judge_by_stem, regen_draft,
        solve_fn=solve_breaks_conservation,
    )
    assert items[0]["stem"] == "old"  # non-conserving draft discarded
    assert items[0]["gene"]["gate"] == GENE_GATE_WARN


def test_rework_regen_conserving_draft_accepted(monkeypatch):
    async def judge_by_stem(item, facts):
        if item.get("stem") == "old":
            return _judge(structure_match=False)
        return _judge()

    async def regen_draft(item, facts, feedback=None):
        return {"stem": "new", "answer": "2", "solution": "s", "level": "normal"}

    async def solve_conserving(stem):
        return {"kp_name": "kp-x", "grade": "g7"}

    items = _run_gate(
        [{"stem": "old"}], monkeypatch, judge_by_stem, regen_draft,
        solve_fn=solve_conserving,
    )
    assert items[0]["stem"] == "new"
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS
    assert "check" not in items[0]  # Gate-B still must verify the draft


def test_regen_failure_degrades_to_warn_not_exception(monkeypatch):
    async def judge_fail(item, facts):
        return _judge(qtype_match=False)

    async def regen_none(item, facts, feedback=None):
        return None  # regen JSON unparseable

    items = _run_gate([{"stem": "old"}], monkeypatch, judge_fail, regen_none)
    assert items[0]["gene"]["gate"] == GENE_GATE_WARN


def test_already_marked_items_are_not_rejudged(monkeypatch):
    calls = []

    async def judge_spy(item, facts):
        calls.append(item.get("stem"))
        return _judge()

    items = _run_gate(
        [
            {"stem": "old", "gene": {"gate": GENE_GATE_PASS}, "check": {"badge": "ok"}},
            {"stem": "fresh"},
        ],
        monkeypatch,
        judge_spy,
    )
    assert calls == ["fresh"]  # budget guard: old items skip the judge entirely
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS
    assert items[1]["gene"]["gate"] == GENE_GATE_PASS


# ---------------------------------------------------------------------------
# card text + auxTags transmission (warn must be visible, not blocking)
# ---------------------------------------------------------------------------

def test_fmt_item_renders_gene_warn_note():
    card = _fmt_item(
        1,
        {
            "stem": "s",
            "answer": "a",
            "solution": "sol",
            "check": {"badge": "ok"},
            "gene": {"gate": GENE_GATE_WARN, "reason": "replay"},
        },
    )
    assert NOTE_GENE_WARN in card


def test_fmt_item_no_gene_note_when_pass():
    card = _fmt_item(
        1,
        {
            "stem": "s",
            "answer": "a",
            "solution": "sol",
            "check": {"badge": "ok"},
            "gene": {"gate": GENE_GATE_PASS},
        },
    )
    assert NOTE_GENE_WARN not in card


def test_aux_tags_carry_gene_gate_mark():
    facts = {"qtype": "qt", "subject_id": "100200300"}
    for gate in (GENE_GATE_PASS, GENE_GATE_WARN, GENE_GATE_SKIPPED):
        bo = build_create_bo(
            {"stem": "s", "answer": "a", "gene": {"gate": gate}}, facts
        )
        assert bo["auxTags"]["gene_gate"] == gate
    # no gene mark -> key absent (legacy items keep working)
    bo = build_create_bo({"stem": "s", "answer": "a"}, facts)
    assert "gene_gate" not in bo["auxTags"]


# ---------------------------------------------------------------------------
# graph wiring: Gate-A sits between producers and Gate-B
# ---------------------------------------------------------------------------

def test_graph_wiring_gene_gate_between_producers_and_solve():
    g = variant.get_graph()
    assert "gene_gate" in g.nodes
    edges = {(e.source, e.target) for e in g.edges}
    assert ("gene_gate", "solve_explain") in edges
    assert ("exec_regenerate", "gene_gate") in edges
    assert ("exec_add", "gene_gate") in edges
    # remove produces no new items -> stays wired straight to Gate-B
    assert ("exec_remove", "solve_explain") in edges
    # conditional edge generate -> gene_gate exists in the drawable graph
    assert ("generate", "gene_gate") in edges
