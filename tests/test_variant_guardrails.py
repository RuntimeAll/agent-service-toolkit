# -*- coding: utf-8 -*-
"""Unit tests for the parse_instruction physical guardrail (PRD-C-010 G4/FP4).

validate_instruction is a pure function (zero LLM / zero IO):
it clamps the LLM classifier output into the constrained payload that
downstream nodes (dispatch / exec_remove / exec_regenerate / exec_add /
patch / answer_question / persist_to_bank / ask_clarify) consume.

Coverage map:
- R0: JSON parse failure (None / non-dict) -> downgrade to clarify, never remove
- R1: intent outside whitelist -> clarify
- R3: remove/regenerate with missing / out-of-range / non-int index -> clarify
- R4: add count missing / <=0 / huge -> clamped into [1, ADD_COUNT_MAX]
- R5: intent=edit with empty/garbage ops -> clarify
- R6: qa/confirm/revise physically cannot carry edit ops (stripped)
- R7: mixed action classes in one utterance (e.g. remove+add) -> clarify
      (the executor runs exactly one action class per turn; partial execution
      and index shifting are both forbidden -> ask the teacher to split)
- pass-through: legal instructions survive unchanged (normalized types)
- routing: every guardrail output intent maps onto an existing graph branch
"""

from agents.variant import (
    ADD_COUNT_MAX,
    EDIT_ACTIONS,
    INTENT_CLARIFY,
    INTENT_CONFIRM,
    INTENT_EDIT,
    INTENT_QA,
    INTENT_REVISE,
    VALID_INTENTS,
    route_after_parse,
    validate_instruction,
)

N = 3  # default current_item_count for most cases


def _v(parsed, n=N):
    out = validate_instruction(parsed, n)
    # shape invariant: always a dict with the full pending skeleton
    for key in ("intent", "ops", "knobs", "comp", "extra_constraints", "mother_correction"):
        assert key in out
    assert out["intent"] in VALID_INTENTS
    assert isinstance(out["ops"], list)
    for op in out["ops"]:
        assert op["action"] in EDIT_ACTIONS
    return out


# ---------------------------------------------------------------------------
# R0: parse failure -> clarify (never defaults to remove)
# ---------------------------------------------------------------------------

def test_parse_failure_none_downgrades_to_clarify():
    out = _v(None)
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


def test_parse_failure_non_dict_downgrades_to_clarify():
    for garbage in ("just some text", ["remove", 1], 42):
        out = _v(garbage)
        assert out["intent"] == INTENT_CLARIFY
        assert out["ops"] == []


# ---------------------------------------------------------------------------
# R1: intent whitelist
# ---------------------------------------------------------------------------

def test_unknown_intent_downgrades_to_clarify():
    # LLM inventing English enum values must not slip through
    out = _v({"intent": "remove", "ops": [{"action": "remove", "index": 1}]})
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


def test_missing_intent_downgrades_to_clarify():
    out = _v({"ops": [{"action": "remove", "index": 1}]})
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


# ---------------------------------------------------------------------------
# R3: remove / regenerate index bounds
# ---------------------------------------------------------------------------

def test_remove_index_out_of_range_downgrades_to_clarify():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": 5}]})
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


def test_remove_index_zero_or_negative_downgrades_to_clarify():
    for idx in (0, -1):
        out = _v({"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": idx}]})
        assert out["intent"] == INTENT_CLARIFY


def test_remove_index_missing_or_non_numeric_downgrades_to_clarify():
    for idx in (None, "abc", True):
        out = _v({"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": idx}]})
        assert out["intent"] == INTENT_CLARIFY


def test_regenerate_index_out_of_range_downgrades_to_clarify():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "regenerate", "index": 4}]})
    assert out["intent"] == INTENT_CLARIFY


def test_remove_on_empty_item_list_downgrades_to_clarify():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": 1}]}, n=0)
    assert out["intent"] == INTENT_CLARIFY


def test_mixed_ops_one_invalid_index_downgrades_whole_thing():
    # valid add + out-of-range remove: never partially execute -> ask back
    out = _v(
        {
            "intent": INTENT_EDIT,
            "ops": [
                {"action": "add", "count": 2},
                {"action": "remove", "index": 99},
            ],
        }
    )
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


# ---------------------------------------------------------------------------
# R4: add count clamping
# ---------------------------------------------------------------------------

def test_add_count_missing_clamped_to_one():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "add"}]})
    assert out["intent"] == INTENT_EDIT
    assert out["ops"] == [{"action": "add", "count": 1}]


def test_add_count_zero_or_negative_clamped_to_one():
    for cnt in (0, -3):
        out = _v({"intent": INTENT_EDIT, "ops": [{"action": "add", "count": cnt}]})
        assert out["ops"][0]["count"] == 1


def test_add_count_huge_clamped_to_max():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "add", "count": 99}]})
    assert out["ops"][0]["count"] == ADD_COUNT_MAX


def test_add_count_string_coerced():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "add", "count": "3"}]})
    assert out["ops"][0]["count"] == 3


# ---------------------------------------------------------------------------
# R5: edit intent with empty / garbage ops
# ---------------------------------------------------------------------------

def test_edit_with_empty_ops_downgrades_to_clarify():
    out = _v({"intent": INTENT_EDIT, "ops": []})
    assert out["intent"] == INTENT_CLARIFY


def test_edit_with_unknown_actions_only_downgrades_to_clarify():
    out = _v(
        {
            "intent": INTENT_EDIT,
            "ops": [{"action": "delete_all"}, "remove", {"no_action": 1}],
        }
    )
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


# ---------------------------------------------------------------------------
# R6: qa / confirm / revise physically carry no edit ops
# ---------------------------------------------------------------------------

def test_qa_with_sneaky_ops_keeps_intent_but_strips_ops():
    out = _v({"intent": INTENT_QA, "ops": [{"action": "remove", "index": 1}]})
    assert out["intent"] == INTENT_QA
    assert out["ops"] == []  # answer branch can never carry an edit op


def test_confirm_strips_ops():
    out = _v({"intent": INTENT_CONFIRM, "ops": [{"action": "add", "count": 2}]})
    assert out["intent"] == INTENT_CONFIRM
    assert out["ops"] == []


def test_revise_keeps_mother_correction_and_strips_ops():
    out = _v(
        {
            "intent": INTENT_REVISE,
            "ops": [{"action": "remove", "index": 1}],
            "mother_correction": {"grade": "八年级", "kp": None},
        }
    )
    assert out["intent"] == INTENT_REVISE
    assert out["ops"] == []
    assert out["mother_correction"] == {"grade": "八年级", "kp": None}


# ---------------------------------------------------------------------------
# pass-through: legal instructions survive unchanged
# ---------------------------------------------------------------------------

def test_legal_remove_passes_through():
    out = _v({"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": 2}]})
    assert out["intent"] == INTENT_EDIT
    assert out["ops"] == [{"action": "remove", "index": 2}]


def test_legal_regenerate_with_note_and_string_index():
    out = _v(
        {
            "intent": INTENT_EDIT,
            "ops": [{"action": "regenerate", "index": "1", "note": "easier numbers"}],
        }
    )
    assert out["intent"] == INTENT_EDIT
    assert out["ops"] == [{"action": "regenerate", "index": 1, "note": "easier numbers"}]


def test_legal_multi_op_same_action_class_kept():
    # several ops of the SAME action class are executed in one pass by exec_remove
    out = _v(
        {
            "intent": INTENT_EDIT,
            "ops": [
                {"action": "remove", "index": 1},
                {"action": "remove", "index": 3},
            ],
            "comp": "soft pref",
            "extra_constraints": ["keep scene"],
            "confidence": 0.9,
        }
    )
    assert out["intent"] == INTENT_EDIT
    assert out["ops"] == [
        {"action": "remove", "index": 1},
        {"action": "remove", "index": 3},
    ]
    assert out["comp"] == "soft pref"
    assert out["extra_constraints"] == ["keep scene"]
    assert out["confidence"] == 0.9


# ---------------------------------------------------------------------------
# R7: mixed action classes -> clarify (never silently drop / partially execute)
# ---------------------------------------------------------------------------

def test_mixed_action_classes_downgrade_to_clarify():
    # the executor (dispatch -> exec_*) runs exactly one action class per turn:
    # accepting remove+add would silently drop the add -> ask the teacher to split
    out = _v(
        {
            "intent": INTENT_EDIT,
            "ops": [
                {"action": "remove", "index": 3},
                {"action": "add", "count": 2, "note": "harder"},
            ],
        }
    )
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


def test_mixed_regenerate_and_add_downgrade_to_clarify():
    out = _v(
        {
            "intent": INTENT_EDIT,
            "ops": [
                {"action": "regenerate", "index": 1},
                {"action": "add", "count": 1},
            ],
        }
    )
    assert out["intent"] == INTENT_CLARIFY
    assert out["ops"] == []


def test_input_not_mutated():
    parsed = {"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": "2"}]}
    snapshot = {"intent": INTENT_EDIT, "ops": [{"action": "remove", "index": "2"}]}
    validate_instruction(parsed, N)
    assert parsed == snapshot  # pure function: caller's dict untouched


# ---------------------------------------------------------------------------
# routing: guardrail output always lands on an existing graph branch
# ---------------------------------------------------------------------------

def test_every_guardrail_intent_routes_to_existing_branch():
    expected = {
        INTENT_REVISE: "patch",
        INTENT_EDIT: "dispatch",
        INTENT_CONFIRM: "save",
        INTENT_QA: "answer",
        INTENT_CLARIFY: "ask_clarify",
    }
    assert set(expected) == VALID_INTENTS  # enum and routing stay in lockstep
    for intent, branch in expected.items():
        state = {"pending": {"intent": intent, "ops": []}}
        assert route_after_parse(state) == branch
