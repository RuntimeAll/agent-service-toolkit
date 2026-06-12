# -*- coding: utf-8 -*-
"""Unit tests for Gate-A (gene gate, PRD-C-010): parallel-question DNA check.

Orthogonal to Gate-B (sympy answer verification):
- skeleton genes (qtype / difficulty / structure) MUST match, otherwise rework;
- surface (numbers / scene) MUST be swapped, otherwise it is a replay -> rework;
- judge LLM failure / JSON failure -> item passes through marked "skipped"
  (Gate-A is an enhancement, never a blocker -> G5);
- after one regen the item still failing -> keep original, mark "warn"
  (4d/PRD-C-012: warn is a single low gate -> card display stays SILENT;
  the truth still flows into auxTags.gene_gate and the both-low matrix).

gene_gate_decision is a pure function (zero LLM / zero IO).
Node-level behavior is tested with monkeypatched judge/regen (no network).
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import (
    GENE_GATE_PASS,
    GENE_GATE_SKIPPED,
    GENE_GATE_WARN,
    gene_gate,
    gene_gate_decision,
    variant,
)
from agents.variant_support import build_create_bo


def _judge(**over):
    # P12.1 (PRD-C-013): difficulty_match removed from Gate-A — judge no longer
    # second-guesses difficulty (subjective eyeballing vs generate's declared value =
    # two noise sources). Difficulty consistency is now a pure-function relative check
    # (difficulty_consistency_defects), warn-only, zero LLM.
    base = {
        "qtype_match": True,
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


def test_difficulty_match_is_no_longer_a_gate_a_judgment(monkeypatch):
    # P12.1: even if a (legacy) judge dict carries difficulty_match=False, Gate-A's
    # pure decision must NOT rework on it — difficulty is no longer a skeleton gene.
    assert gene_gate_decision(_judge(difficulty_match=False)) == "pass"


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
# RC2 (PRD-C-013): edit-round output is judged but NEVER reworked.
# The teacher already named the rewrite; Gate-A failing only marks warn, the
# teacher's intent wins over the gene gate. No regen budget spent on it.
# ---------------------------------------------------------------------------

def test_edit_round_item_failing_judge_is_warned_not_reworked(monkeypatch):
    regen_calls = []

    async def judge_fail(item, facts):
        return _judge(structure_match=False, reason="teacher reshaped it")

    async def regen_spy(item, facts, feedback=None):
        regen_calls.append(item)
        return {"stem": "new", "answer": "2"}

    items = _run_gate(
        [{"stem": "edited", "from_edit": True}], monkeypatch, judge_fail, regen_spy
    )
    assert regen_calls == []  # RC2: edit-round product never reworks
    assert items[0]["stem"] == "edited"  # original (edited) item kept verbatim
    assert items[0]["gene"]["gate"] == GENE_GATE_WARN
    assert items[0]["gene"]["reason"] == "teacher reshaped it"


def test_edit_round_item_passing_judge_still_passes(monkeypatch):
    # from_edit short-circuit only kicks in on a failing judge; a passing edited
    # item still passes normally.
    async def judge_ok(item, facts):
        return _judge()

    items = _run_gate([{"stem": "edited", "from_edit": True}], monkeypatch, judge_ok)
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS


# ---------------------------------------------------------------------------
# auxTags removed from create BO (PRD-C-014 B1: schema 收敛 DROP 了 biz_question.aux_tags
# 列；gene/verify 审计标记走 BE ai 表 conflict_flags，不再塞进 create BO)。
# 标记仍活在 item.gene.gate（FE 4d 展示/快照），但 BO 不再带 auxTags / gene_gate。
# ---------------------------------------------------------------------------

def test_create_bo_no_longer_carries_aux_tags():
    facts = {"qtype": "qt", "subject_id": "3071", "dim1_kp_id": "3071001"}
    for gate in (GENE_GATE_PASS, GENE_GATE_WARN, GENE_GATE_SKIPPED):
        bo = build_create_bo(
            {"stem": "s", "answer": "a", "gene": {"gate": gate}}, facts
        )
        # B1: auxTags 列已 DROP，BO 不再带该键（gene 标记仍活在 item.gene 供 FE 展示）
        assert "auxTags" not in bo
        assert "dim3Skill" not in bo and "freeTag" not in bo  # 三件套全删


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


# ---------------------------------------------------------------------------
# P12.1 difficulty consistency: pure-function (zero LLM) intra-group relative
# check that replaces the deleted Gate-A difficulty_match judgment.
# ---------------------------------------------------------------------------

def test_difficulty_consistency_no_defect_when_hard_ge_normal():
    items = [
        {"level": "normal", "difficulty": 2},
        {"level": "normal", "difficulty": 3},
        {"level": "hard", "difficulty": 4},
    ]
    assert variant_mod.difficulty_consistency_defects(items) == []


def test_difficulty_consistency_flags_hard_easier_than_normal():
    items = [
        {"level": "normal", "difficulty": 3},
        {"level": "hard", "difficulty": 2},  # hard easier than a normal -> defect
    ]
    defects = variant_mod.difficulty_consistency_defects(items)
    assert len(defects) == 1
    assert "第2道" in defects[0]


def test_difficulty_consistency_empty_when_no_hard_or_no_normal():
    assert variant_mod.difficulty_consistency_defects(
        [{"level": "normal", "difficulty": 3}]
    ) == []
    assert variant_mod.difficulty_consistency_defects(
        [{"level": "hard", "difficulty": 2}]
    ) == []


def test_difficulty_consistency_skips_unparseable_difficulty():
    # missing/unparseable difficulty -> excluded from comparison, no false alarm
    items = [
        {"level": "normal", "difficulty": None},
        {"level": "hard", "difficulty": "x"},
    ]
    assert variant_mod.difficulty_consistency_defects(items) == []


def test_difficulty_consistency_hard_equal_to_normal_is_ok():
    # hard == max normal is allowed (>=, not strictly >)
    items = [
        {"level": "normal", "difficulty": 3},
        {"level": "hard", "difficulty": 3},
    ]
    assert variant_mod.difficulty_consistency_defects(items) == []


# ---------------------------------------------------------------------------
# RC1 (PRD-C-013): qtype-conversion structure judgment. When the variant qtype
# differs from the mother qtype, structure_match must target the *target* qtype's
# canonical structure (not the mother's original skeleton).
# ---------------------------------------------------------------------------

def test_gene_target_qtype_detects_conversion():
    facts = {"qtype": "解答题"}
    assert variant_mod._gene_target_qtype({"qtype": "选择"}, facts) == "选择"


def test_gene_target_qtype_none_when_same_qtype_after_alias():
    facts = {"qtype": "解答题"}  # aliases to 解答
    assert variant_mod._gene_target_qtype({"qtype": "计算题"}, facts) is None  # also 解答


def test_gene_facts_for_injects_target_qtype_spec_on_conversion():
    facts = {"qtype": "解答", "kp_name": "kp", "grade": "g7", "stem": "m", "skeleton": "sk"}
    out = variant_mod._gene_facts_for({"qtype": "选择"}, facts, None)
    assert "target_qtype_spec" in out
    assert "选择" in out["target_qtype_spec"]
    # judge prompt carries the target-qtype structure rule
    prompt = variant_mod._gene_judge_prompt({"qtype": "选择", "stem": "v"}, out)
    assert "题型转换语境" in prompt


def test_gene_facts_for_no_target_spec_when_same_qtype():
    facts = {"qtype": "解答"}
    out = variant_mod._gene_facts_for({"qtype": "解答"}, facts, None)
    assert "target_qtype_spec" not in out


# ---------------------------------------------------------------------------
# P12.3: rework feedback carries the mother skeleton/stem (was missing -> the
# regen prompt asked "match the mother" with no mother in sight -> ~random).
# ---------------------------------------------------------------------------

def test_gene_feedback_embeds_mother_skeleton_and_stem():
    facts = {"skeleton": "set up equation then solve", "stem": "mother question text"}
    fb = variant_mod._gene_feedback(
        _judge(structure_match=False, reason="r"), facts
    )
    assert "set up equation then solve" in fb
    assert "mother question text" in fb
    assert "同源" in fb  # 解法核心步骤同源（考点级）


def test_gene_feedback_target_qtype_path_drops_mother_skeleton_requirement():
    facts = {"skeleton": "sk", "stem": "mst"}
    fb = variant_mod._gene_feedback(
        _judge(qtype_match=False), facts, target_qtype="选择"
    )
    assert "选择" in fb
    assert "规范结构" in fb  # target qtype canonical structure, not mother skeleton
