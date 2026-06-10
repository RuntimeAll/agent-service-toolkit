# -*- coding: utf-8 -*-
"""Unit tests for agents.math_verify (pure function, zero LLM/network).

Coverage map (PRD-C-010):
- G1: each kind has correct->pass and wrong->fail cases
- G2: a distractor numerically hitting the ground truth -> fail
- G5: garbage input (natural language / empty / illegal kind / geometry text)
      -> degrade, never raises
- G5 anti-hang: explosive payloads (9**9**9, oversized expressions) -> degrade
      FAST instead of dragging sympy into unbounded computation
- choice robustness: claimed option judged first (dirty distractor cannot mask
      a provable fail); unparseable distractors treated as non-matching
- equivalence: claimed in equivalent forms (1/2 vs 0.5, 2*sqrt(2) vs sqrt(8)) -> pass
"""

import time

import pytest

from agents.math_verify import verify


def _v(payload):
    res = verify(payload)
    assert set(res.keys()) == {"verdict", "detail", "computed"}
    assert isinstance(res["detail"], str)
    assert res["computed"] is None or isinstance(res["computed"], str)
    return res


# ---------------------------------------------------------------------------
# G1: equation_solve
# ---------------------------------------------------------------------------

def test_equation_solve_pass():
    res = _v({
        "kind": "equation_solve",
        "equations": ["x**2-5*x+6=0"],
        "unknowns": ["x"],
        "claimed": ["2", "3"],
    })
    assert res["verdict"] == "pass"


def test_equation_solve_pass_order_insensitive():
    res = _v({
        "kind": "equation_solve",
        "equations": ["x**2-5*x+6=0"],
        "unknowns": ["x"],
        "claimed": ["3", "2"],
    })
    assert res["verdict"] == "pass"


def test_equation_solve_fail_wrong_root():
    res = _v({
        "kind": "equation_solve",
        "equations": ["x**2-5*x+6=0"],
        "unknowns": ["x"],
        "claimed": ["2", "4"],
    })
    assert res["verdict"] == "fail"


def test_equation_solve_fail_missing_root():
    res = _v({
        "kind": "equation_solve",
        "equations": ["x**2-5*x+6=0"],
        "unknowns": ["x"],
        "claimed": ["2"],
    })
    assert res["verdict"] == "fail"


def test_equation_solve_linear_pass():
    res = _v({
        "kind": "equation_solve",
        "equations": ["2*x - 6 = 0"],
        "unknowns": ["x"],
        "claimed": ["3"],
    })
    assert res["verdict"] == "pass"
    assert res["computed"] == "[3]"


def test_equation_solve_linear_fail():
    res = _v({
        "kind": "equation_solve",
        "equations": ["2*x - 6 = 0"],
        "unknowns": ["x"],
        "claimed": ["4"],
    })
    assert res["verdict"] == "fail"


# ---------------------------------------------------------------------------
# G1: expr_equiv
# ---------------------------------------------------------------------------

def test_expr_equiv_pass():
    res = _v({"kind": "expr_equiv", "expr_a": "(x+1)**2", "expr_b": "x**2+2*x+1"})
    assert res["verdict"] == "pass"


def test_expr_equiv_fail():
    res = _v({"kind": "expr_equiv", "expr_a": "(x+1)**2", "expr_b": "x**2+1"})
    assert res["verdict"] == "fail"


def test_expr_equiv_factored_pass():
    res = _v({"kind": "expr_equiv", "expr_a": "x**2-9", "expr_b": "(x-3)*(x+3)"})
    assert res["verdict"] == "pass"


def test_expr_equiv_sign_fail():
    res = _v({"kind": "expr_equiv", "expr_a": "x**2-9", "expr_b": "(x-3)*(x-3)"})
    assert res["verdict"] == "fail"


# ---------------------------------------------------------------------------
# G1: numeric
# ---------------------------------------------------------------------------

def test_numeric_pass():
    res = _v({"kind": "numeric", "expr": "3*7/2", "claimed": "10.5", "tol": 1e-6})
    assert res["verdict"] == "pass"


def test_numeric_fail():
    res = _v({"kind": "numeric", "expr": "3*7/2", "claimed": "10.6", "tol": 1e-6})
    assert res["verdict"] == "fail"


def test_numeric_default_tol_pass():
    res = _v({"kind": "numeric", "expr": "1/3", "claimed": "0.333333333"})
    assert res["verdict"] == "pass"


def test_numeric_fail_far_off():
    res = _v({"kind": "numeric", "expr": "2**10", "claimed": "1000"})
    assert res["verdict"] == "fail"


# ---------------------------------------------------------------------------
# G1: choice
# ---------------------------------------------------------------------------

def test_choice_pass_equation_ground():
    res = _v({
        "kind": "choice",
        "ground": {"kind": "equation_solve", "equations": ["2*x-6=0"], "unknowns": ["x"]},
        "options": {"A": "2", "B": "3", "C": "4", "D": "6"},
        "claimed_correct": "B",
    })
    assert res["verdict"] == "pass"


def test_choice_fail_wrong_claimed_correct():
    res = _v({
        "kind": "choice",
        "ground": {"kind": "equation_solve", "equations": ["2*x-6=0"], "unknowns": ["x"]},
        "options": {"A": "2", "B": "3", "C": "4", "D": "6"},
        "claimed_correct": "A",
    })
    assert res["verdict"] == "fail"


def test_choice_pass_numeric_ground():
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "3*4"},
        "options": {"A": "12", "B": "7", "C": "1"},
        "claimed_correct": "A",
    })
    assert res["verdict"] == "pass"


def test_choice_fail_numeric_ground_wrong_correct():
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "3*4"},
        "options": {"A": "12", "B": "7", "C": "1"},
        "claimed_correct": "B",
    })
    assert res["verdict"] == "fail"


# ---------------------------------------------------------------------------
# G2: distractor hits the ground truth -> fail
# ---------------------------------------------------------------------------

def test_choice_g2_distractor_equals_truth_numeric():
    # truth = 3; option D "sqrt(9)" is an equivalent form of the truth -> fail
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "6/2"},
        "options": {"A": "3", "B": "5", "C": "1", "D": "sqrt(9)"},
        "claimed_correct": "A",
    })
    assert res["verdict"] == "fail"
    assert "G2" in res["detail"]


def test_choice_g2_duplicate_distractor_equation_ground():
    # truth: x=3; distractor B literally repeats the correct value -> fail
    res = _v({
        "kind": "choice",
        "ground": {"kind": "equation_solve", "equations": ["2*x-6=0"], "unknowns": ["x"]},
        "options": {"A": "3", "B": "3", "C": "1"},
        "claimed_correct": "A",
    })
    assert res["verdict"] == "fail"
    assert "G2" in res["detail"]


def test_choice_g2_expr_equiv_ground():
    # ground reference (x+1)**2; distractor C is the expanded equivalent -> fail
    res = _v({
        "kind": "choice",
        "ground": {"kind": "expr_equiv", "expr_a": "(x+1)**2"},
        "options": {"A": "(x+1)**2", "B": "x**2+1", "C": "x**2+2*x+1"},
        "claimed_correct": "A",
    })
    assert res["verdict"] == "fail"
    assert "G2" in res["detail"]


# ---------------------------------------------------------------------------
# G5: garbage in -> degrade, never raise
# ---------------------------------------------------------------------------

GARBAGE_PAYLOADS = [
    {},                                       # empty payload
    {"kind": "banana"},                       # illegal kind
    {"kind": None},                           # kind missing/None
    "just a string, not a dict",              # non-dict payload
    None,                                     # None payload
    {"kind": "numeric", "expr": "画一个半径为2的圆", "claimed": "3"},      # NL text
    {"kind": "numeric", "expr": "x+1", "claimed": "3"},                    # symbolic, not a number
    {"kind": "expr_equiv", "expr_a": "证明: AB平行于CD", "expr_b": "x"},   # geometry proof text
    {"kind": "equation_solve", "equations": ["求三角形ABC的面积"],
     "unknowns": ["x"], "claimed": ["2"]},                                 # geometry NL
    {"kind": "equation_solve", "equations": ["y**2 = 4"],
     "unknowns": ["x"], "claimed": ["2"]},                                 # unknown absent
    {"kind": "equation_solve", "equations": [], "unknowns": ["x"], "claimed": []},
    {"kind": "choice", "ground": {"kind": "weird"},
     "options": {"A": "1"}, "claimed_correct": "A"},                       # bad ground kind
    {"kind": "choice", "ground": {"kind": "numeric", "expr": "1+1"},
     "options": {"A": "2"}, "claimed_correct": "Z"},                       # bad option key
    {"kind": "choice", "ground": "not a dict",
     "options": {"A": "2"}, "claimed_correct": "A"},                       # ground not dict
    {"kind": "numeric", "expr": "", "claimed": ""},                        # empty strings
    {"kind": "equation_solve", "equations": ["x < 5"],
     "unknowns": ["x"], "claimed": ["1"]},                                 # inequality
]


@pytest.mark.parametrize("payload", GARBAGE_PAYLOADS, ids=range(len(GARBAGE_PAYLOADS)))
def test_g5_garbage_degrades_without_raising(payload):
    res = _v(payload)  # must not raise
    assert res["verdict"] == "degrade"


# ---------------------------------------------------------------------------
# G5 anti-hang: explosive payloads must degrade FAST (no unbounded computation)
# ---------------------------------------------------------------------------

EXPLOSIVE_PAYLOADS = [
    {"kind": "numeric", "expr": "9**9**9", "claimed": "1"},        # ~370M-digit int
    {"kind": "numeric", "expr": "9^9^9", "claimed": "1"},          # caret form
    {"kind": "numeric", "expr": "2**2**2**2**2", "claimed": "1"},  # nested pow tower
    {"kind": "numeric", "expr": "1", "claimed": "9**9**9"},        # explosive claimed
    {"kind": "expr_equiv", "expr_a": "9**9**9", "expr_b": "1"},
    {"kind": "equation_solve", "equations": ["x**99999 = 1"],
     "unknowns": ["x"], "claimed": ["1"]},                          # absurd degree
    {"kind": "numeric", "expr": "(" * 60 + "1" + ")" * 60 + "+" + "1+" * 65 + "1",
     "claimed": "1"},                                               # oversized expr (>200 chars)
    {"kind": "numeric", "expr": "10000000000000000000", "claimed": "1"},  # huge int literal
]


@pytest.mark.parametrize("payload", EXPLOSIVE_PAYLOADS, ids=range(len(EXPLOSIVE_PAYLOADS)))
def test_g5_explosive_payload_degrades_fast(payload):
    t0 = time.monotonic()
    res = _v(payload)  # must not hang, must not raise
    assert res["verdict"] == "degrade"
    assert time.monotonic() - t0 < 5.0  # guard fires before sympy evaluates


def test_legit_powers_still_verified():
    # complexity guard must not eat normal K-12 powers
    res = _v({"kind": "numeric", "expr": "2**10", "claimed": "1024"})
    assert res["verdict"] == "pass"
    res = _v({"kind": "expr_equiv", "expr_a": "x**3*x**2", "expr_b": "x**5"})
    assert res["verdict"] == "pass"


# ---------------------------------------------------------------------------
# choice: dirty (unparseable) distractors must not mask a provable verdict
# ---------------------------------------------------------------------------

def test_choice_dirty_distractor_does_not_mask_wrong_claimed():
    # truth=3; claimed D has value 5 (provably wrong); option A is K-12 text
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "6/2"},
        "options": {"A": "无解", "B": "3", "C": "1", "D": "5"},
        "claimed_correct": "D",
    })
    assert res["verdict"] == "fail"


def test_choice_dirty_distractor_with_correct_claimed_passes():
    # text distractor cannot numerically equal the truth -> treated non-matching
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "6/2"},
        "options": {"A": "以上都不对", "B": "3", "C": "1"},
        "claimed_correct": "B",
    })
    assert res["verdict"] == "pass"


def test_choice_unparseable_claimed_option_still_degrades():
    # the claimed answer itself unverifiable -> genuinely cannot verify -> degrade
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "6/2"},
        "options": {"A": "无解", "B": "3"},
        "claimed_correct": "A",
    })
    assert res["verdict"] == "degrade"


def test_choice_clean_distractor_hitting_truth_still_fails_g2():
    # G2 contract intact after the dirty-distractor fix
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "6/2"},
        "options": {"A": "无解", "B": "3", "C": "sqrt(9)"},
        "claimed_correct": "B",
    })
    assert res["verdict"] == "fail"
    assert "G2" in res["detail"]


# ---------------------------------------------------------------------------
# Equivalent forms of claimed answers -> pass (no raw string comparison)
# ---------------------------------------------------------------------------

def test_equiv_form_half_vs_decimal():
    res = _v({"kind": "numeric", "expr": "1/2", "claimed": "0.5"})
    assert res["verdict"] == "pass"


def test_equiv_form_sqrt8_vs_2sqrt2_expr():
    res = _v({"kind": "expr_equiv", "expr_a": "2*sqrt(2)", "expr_b": "sqrt(8)"})
    assert res["verdict"] == "pass"


def test_equiv_form_equation_claimed_sqrt8():
    # solution is 2*sqrt(2); claimed in the equivalent form sqrt(8)
    res = _v({
        "kind": "equation_solve",
        "equations": ["x**2 = 8"],
        "unknowns": ["x"],
        "claimed": ["sqrt(8)", "-2*sqrt(2)"],
    })
    assert res["verdict"] == "pass"


def test_equiv_form_equation_claimed_decimal():
    res = _v({
        "kind": "equation_solve",
        "equations": ["2*x - 1 = 0"],
        "unknowns": ["x"],
        "claimed": ["0.5"],
    })
    assert res["verdict"] == "pass"


def test_equiv_form_caret_power():
    # '^' accepted as power via convert_xor
    res = _v({"kind": "expr_equiv", "expr_a": "x^2+2x+1", "expr_b": "(x+1)**2"})
    assert res["verdict"] == "pass"
