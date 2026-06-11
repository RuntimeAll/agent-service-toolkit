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


# ---------------------------------------------------------------------------
# 2026-06-10 widened parser: Min/Max/Abs/sqrt + comma allowed; python syntax
# and non-whitelisted functions -> degrade (real-machine gap: comparison /
# smallest-number payloads were all degraded by the comma-less charset)
# ---------------------------------------------------------------------------

def test_numeric_min_with_abs_sqrt_pass():
    # the exact real-machine case: smallest of -sqrt(25), -(-4), -|3|, -7/2
    res = _v({
        "kind": "numeric",
        "expr": "Min(-sqrt(25), -(-4), -Abs(3), -7/2)",
        "claimed": "-5",
    })
    assert res["verdict"] == "pass"


def test_numeric_min_wrong_claimed_fail():
    res = _v({
        "kind": "numeric",
        "expr": "Min(-sqrt(25), -(-4), -Abs(3), -7/2)",
        "claimed": "-4",
    })
    assert res["verdict"] == "fail"


def test_numeric_lowercase_min_abs_aliases_pass():
    res = _v({"kind": "numeric", "expr": "min(3, -4, abs(-2))", "claimed": "-4"})
    assert res["verdict"] == "pass"


def test_choice_min_ground_pass():
    res = _v({
        "kind": "choice",
        "ground": {"kind": "numeric", "expr": "Min(-Abs(-2), -sqrt(9), -2.5, 0)", "claimed": "-3"},
        "options": {"A": "-Abs(-2)", "B": "-sqrt(9)", "C": "-2.5", "D": "0"},
        "claimed_correct": "B",
    })
    assert res["verdict"] == "pass"


def test_python_comprehension_degrades():
    res = _v({
        "kind": "numeric",
        "expr": "len([n for n in range(-100,101)])",
        "claimed": "201",
    })
    assert res["verdict"] == "degrade"


def test_python_ternary_degrades():
    res = _v({"kind": "numeric", "expr": "(1 if x else 0)", "claimed": "1"})
    assert res["verdict"] == "degrade"


def test_non_whitelisted_function_degrades():
    # factorial parses in raw sympy but is outside our function whitelist
    res = _v({"kind": "numeric", "expr": "factorial(20)", "claimed": "1"})
    assert res["verdict"] == "degrade"


# ---------------------------------------------------------------------------
# PRD-C-013 4b (1): inequality_solve — claimed solution SET vs true solution SET
# Sets are compared symbolically (Interval/Set equality), never by string. So a
# strict/non-strict boundary mismatch fails, equivalent writings pass, and any
# non-relational / NL garbage degrades.
# ---------------------------------------------------------------------------

def test_inequality_solve_linear_pass():
    res = _v({
        "kind": "inequality_solve",
        "inequality": "2*x-3>5",
        "unknown": "x",
        "claimed": "x>4",
    })
    assert res["verdict"] == "pass"
    assert res["computed"] == "Interval.open(4, oo)"


def test_inequality_solve_wrong_set_fail():
    # true set is x>4; claiming x>3 is a different (larger) set -> fail
    res = _v({
        "kind": "inequality_solve",
        "inequality": "2*x-3>5",
        "unknown": "x",
        "claimed": "x>3",
    })
    assert res["verdict"] == "fail"


def test_inequality_solve_boundary_strictness_fail():
    # 2x>=6 -> x>=3 (closed at 3); claiming x>3 (open) misses the boundary -> fail
    res = _v({
        "kind": "inequality_solve",
        "inequality": "2*x>=6",
        "unknown": "x",
        "claimed": "x>3",
    })
    assert res["verdict"] == "fail"


def test_inequality_solve_compound_quadratic_pass():
    # x**2-4<=0 has solution set [-2, 2]; the compound form -2<=x<=2 is equal
    res = _v({
        "kind": "inequality_solve",
        "inequality": "x**2-4<=0",
        "unknown": "x",
        "claimed": "-2<=x<=2",
    })
    assert res["verdict"] == "pass"


def test_inequality_solve_equivalent_writing_pass():
    # 2x>=6 and the reversed/equivalent 3<=x denote the same set -> pass
    res = _v({
        "kind": "inequality_solve",
        "inequality": "2*x>=6",
        "unknown": "x",
        "claimed": "3<=x",
    })
    assert res["verdict"] == "pass"


def test_inequality_solve_closed_boundary_pass():
    # exact boundary match: 2x>=6 -> x>=3, claimed x>=3 -> pass
    res = _v({
        "kind": "inequality_solve",
        "inequality": "2*x>=6",
        "unknown": "x",
        "claimed": "x>=3",
    })
    assert res["verdict"] == "pass"


def test_inequality_solve_nl_garbage_degrades():
    # natural-language "inequality" has no relational operator -> degrade, no raise
    res = _v({
        "kind": "inequality_solve",
        "inequality": "画一个开口向上的抛物线",
        "unknown": "x",
        "claimed": "x>0",
    })
    assert res["verdict"] == "degrade"


def test_inequality_solve_garbage_claimed_degrades():
    # claimed side is prose -> cannot build a comparable set -> degrade
    res = _v({
        "kind": "inequality_solve",
        "inequality": "2*x-3>5",
        "unknown": "x",
        "claimed": "x 大于 四",
    })
    assert res["verdict"] == "degrade"


# ---------------------------------------------------------------------------
# PRD-C-013 4b (2): rational_roots — 分式方程舍根/增根子集模式
# Design choice: a DEDICATED kind (not an equation_solve overload) so the
# polynomial path stays untouched. PASS iff claimed EXACTLY equals the set of
# domain-valid roots (roots that do not zero any denominator) and that set is
# non-empty. 漏剔增根 / 申报使分母为 0 的根 / 漏根 都 -> fail; 解析不了 -> degrade.
# ---------------------------------------------------------------------------

def test_rational_roots_no_spurious_pass():
    # 1/(x-1)=2 -> x=3/2, no spurious root, denominator nonzero there
    res = _v({
        "kind": "rational_roots",
        "equation": "1/(x-1)=2",
        "unknowns": ["x"],
        "claimed": ["3/2"],
    })
    assert res["verdict"] == "pass"


def test_rational_roots_correctly_dropped_spurious_pass():
    # x**2/(x-2)=4/(x-2): candidates {2,-2}; x=2 zeroes denom (增根) -> only -2 valid.
    # claimed correctly剔除 x=2 and keeps -2 -> pass
    res = _v({
        "kind": "rational_roots",
        "equation": "x**2/(x-2)=4/(x-2)",
        "unknowns": ["x"],
        "claimed": ["-2"],
    })
    assert res["verdict"] == "pass"


def test_rational_roots_kept_spurious_fail():
    # same equation; claimed keeps the spurious root x=2 (漏剔增根) -> fail
    res = _v({
        "kind": "rational_roots",
        "equation": "x**2/(x-2)=4/(x-2)",
        "unknowns": ["x"],
        "claimed": ["-2", "2"],
    })
    assert res["verdict"] == "fail"


def test_rational_roots_denominator_zero_root_fail():
    # 5/(x-2)=(x+3)/(x-2): only candidate x=2 zeroes the denominator -> no valid
    # root at all; claiming x=2 (a分母为0的根) -> fail
    res = _v({
        "kind": "rational_roots",
        "equation": "5/(x-2)=(x+3)/(x-2)",
        "unknowns": ["x"],
        "claimed": ["2"],
    })
    assert res["verdict"] == "fail"


def test_rational_roots_missing_valid_root_fail():
    # valid root is 3/2 but claimed set is empty -> misses -> fail
    res = _v({
        "kind": "rational_roots",
        "equation": "1/(x-1)=2",
        "unknowns": ["x"],
        "claimed": [],
    })
    assert res["verdict"] == "fail"


def test_rational_roots_unknown_singular_key_pass():
    # accepts the singular 'unknown' alias as well as 'unknowns'
    res = _v({
        "kind": "rational_roots",
        "equation": "1/(x-1)=2",
        "unknown": "x",
        "claimed": ["3/2"],
    })
    assert res["verdict"] == "pass"


def test_rational_roots_nl_garbage_degrades():
    # natural-language equation -> degrade, never raise
    res = _v({
        "kind": "rational_roots",
        "equation": "求三角形ABC的面积",
        "unknowns": ["x"],
        "claimed": ["2"],
    })
    assert res["verdict"] == "degrade"


# ---------------------------------------------------------------------------
# PRD-C-013 4b (3): 应用题建模验算 — REUSES equation_solve, NO new function.
# These prove that a 4a application-problem payload (modelled equation +
# reported answer) verifies through the existing equation_solve path: the
# equation's root must equal the申报答案. correct answer -> pass, wrong -> fail.
# ---------------------------------------------------------------------------

def test_word_problem_travel_pass():
    # 行程: speed*time = distance, 60*t = 180 -> t = 3 hours
    res = _v({
        "kind": "equation_solve",
        "equations": ["60*t = 180"],
        "unknowns": ["t"],
        "claimed": ["3"],
    })
    assert res["verdict"] == "pass"
    assert res["computed"] == "[3]"


def test_word_problem_work_rate_pass():
    # 工程: 1/a + 1/15 = 1/6 (combined rate) -> a = 10 days
    res = _v({
        "kind": "equation_solve",
        "equations": ["1/a + 1/15 = 1/6"],
        "unknowns": ["a"],
        "claimed": ["10"],
    })
    assert res["verdict"] == "pass"


def test_word_problem_concentration_pass():
    # 浓度: 0.2*x + 0.5*(10-x) = 3 -> x = 20/3 (claimed as fraction, equiv to float)
    res = _v({
        "kind": "equation_solve",
        "equations": ["0.2*x + 0.5*(10-x) = 3"],
        "unknowns": ["x"],
        "claimed": ["20/3"],
    })
    assert res["verdict"] == "pass"


def test_word_problem_wrong_answer_fail():
    # 行程 model is right but the reported answer is wrong -> fail (catches a
    # correct equation paired with a miscomputed final number)
    res = _v({
        "kind": "equation_solve",
        "equations": ["60*t = 180"],
        "unknowns": ["t"],
        "claimed": ["4"],
    })
    assert res["verdict"] == "fail"


def test_word_problem_price_discount_pass():
    # 销售: original price x, 20% off then -5 = 75 -> 0.8*x - 5 = 75 -> x = 100
    res = _v({
        "kind": "equation_solve",
        "equations": ["0.8*x - 5 = 75"],
        "unknowns": ["x"],
        "claimed": ["100"],
    })
    assert res["verdict"] == "pass"
