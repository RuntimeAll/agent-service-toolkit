"""Pure deterministic math verification module for the variant agent.

Contract (see PRD-C-010):
- single public entry: ``verify(payload: dict) -> dict``
- payload kinds: ``equation_solve`` / ``expr_equiv`` / ``numeric`` / ``choice``
  / ``inequality_solve`` / ``rational_roots`` (PRD-C-013 4b 扩面)
- return: ``{"verdict": "pass"|"fail"|"degrade", "detail": str, "computed": str|None}``

Hard rules:
- pure function: zero LLM, zero network, zero global mutable state
- any sympy parse/solve failure or out-of-scope input degrades (never raises) -> G5
- ``degrade`` (cannot verify) is strictly distinct from ``fail`` (verified wrong)
- claimed answers are compared by symbolic equivalence (simplify(diff)==0 / tol),
  never by raw string comparison
- choice: the claimed correct option must truly match the ground truth AND every
  distractor must truly NOT match it (a distractor hitting the truth -> fail) -> G2;
  the claimed option is checked FIRST (a provably wrong claimed answer fails even if
  some distractor is unparseable text); an unparseable distractor (e.g. "无解") is
  treated as non-matching instead of degrading the whole verdict
- complexity guard: oversized expressions / huge integer powers (e.g. 9**9**9)
  degrade up-front instead of dragging sympy into unbounded computation (G5:
  the caller additionally wraps verify() in a wall-clock timeout)
"""

from __future__ import annotations

import re
from typing import Any, Callable

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

PASS = "pass"
FAIL = "fail"
DEGRADE = "degrade"

_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

# Charset whitelist for a single (no '=') math expression. Anything outside
# (Chinese prose, punctuation, '?', ':', ...) is treated as non-math -> degrade.
# 逗号放行 = 支持多参函数 Min(3,-4) / Max / Abs（2026-06-10 真机踩坑：比大小类载荷
# 全被逗号挡成 degrade）；配套下面的函数名白名单，不会放进任意 sympy 调用。
_EXPR_ALLOWED_RE = re.compile(r"^[A-Za-z0-9_+\-*/^(),. \t]+$")

# 函数/名字白名单：多字母标识符只认这些（单字母一律视作变量）。挡两类东西：
# ① extractor 偶发吐的 Python 语法（if/else 三元、列表推导、len/range）→ degrade；
# ② sympy 命名空间里的重型函数（factorial/integrate…）→ degrade（防 CPU 炸弹，
#    复杂度护栏管不到"小字面量大计算"的调用类载荷）。
_ALLOWED_FUNCS = {
    "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs,
    "min": sp.Min, "Min": sp.Min,
    "max": sp.Max, "Max": sp.Max,
}
_ALLOWED_NAMES = set(_ALLOWED_FUNCS) | {"pi"}
_PY_TOKEN_RE = re.compile(r"\b(if|else|elif|for|in|lambda|len|range|and|or|not|while|def|import|True|False|None)\b")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

_DEFAULT_TOL = 1e-6

# Relational operators for inequality_solve. Order in the split regex matters:
# multi-char ops ('>=','<=') must come before their single-char prefixes.
_REL_SPLIT_RE = re.compile(r"(>=|<=|>|<)")
_REL_FUNCS = {">": sp.Gt, "<": sp.Lt, ">=": sp.Ge, "<=": sp.Le}

# ---------------------------------------------------------------------------
# Complexity guard (G5, anti-hang): K-12 payloads are tiny. Anything beyond
# these budgets is treated as unverifiable -> degrade BEFORE sympy evaluates
# (parse_expr(evaluate=True) would instantly compute e.g. 9**9**9, a ~370M
# digit integer, hanging the worker thread). Limits are deliberately generous
# for real K-12 math and absurdly small for pathological payloads.
# ---------------------------------------------------------------------------
_MAX_EXPR_LEN = 200      # characters per expression string
_MAX_NODES = 200         # sympy tree nodes per expression
_MAX_POW_ABS = 512.0     # |numeric exponent| upper bound
_MAX_INT_ABS = 10**15    # |integer literal| upper bound


def _complexity_guard(expr: sp.Expr) -> None:
    """Reject oversized / explosive expressions before evaluation. -> _DegradeError."""
    count = 0
    for node in sp.preorder_traversal(expr):
        count += 1
        if count > _MAX_NODES:
            raise _DegradeError(f"expression too complex (> {_MAX_NODES} nodes)")
        if isinstance(node, sp.Integer) and abs(node) > _MAX_INT_ABS:
            raise _DegradeError(f"integer literal too large: {node}")
        if isinstance(node, sp.Pow):
            exponent = node.exp
            if exponent.free_symbols:
                continue  # symbolic exponent (x**n) is fine
            try:
                size = abs(float(exponent.evalf()))  # cheap even for nested pows
            except (OverflowError, TypeError, ValueError) as exc:
                raise _DegradeError(f"cannot bound exponent size: {exc}") from exc
            if size > _MAX_POW_ABS:
                raise _DegradeError(f"exponent too large: |{sp.sstr(exponent)}| > {_MAX_POW_ABS}")


class _DegradeError(Exception):
    """Internal signal: input cannot be verified by sympy -> verdict=degrade."""


def _result(verdict: str, detail: str, computed: Any = None) -> dict:
    return {
        "verdict": verdict,
        "detail": detail,
        "computed": None if computed is None else str(computed),
    }


# ---------------------------------------------------------------------------
# parsing helpers (all failures -> _DegradeError, never raise outward)
# ---------------------------------------------------------------------------

def _parse(value: Any) -> sp.Expr:
    """Parse one expression (no '='). Non-math input -> _DegradeError."""
    if isinstance(value, bool):
        raise _DegradeError(f"boolean is not a math expression: {value!r}")
    if isinstance(value, (int, float)):
        try:
            return sp.sympify(value)
        except Exception as exc:  # pragma: no cover - extremely unlikely
            raise _DegradeError(f"cannot sympify number {value!r}: {exc}") from exc
    if not isinstance(value, str) or not value.strip():
        raise _DegradeError(f"not a parseable expression: {value!r}")
    text = value.strip()
    if len(text) > _MAX_EXPR_LEN:
        raise _DegradeError(f"expression too long ({len(text)} > {_MAX_EXPR_LEN} chars)")
    if not _EXPR_ALLOWED_RE.fullmatch(text):
        raise _DegradeError(f"expression contains non-math characters: {text!r}")
    if _PY_TOKEN_RE.search(text):
        raise _DegradeError(f"python syntax is not a math expression: {text!r}")
    for ident in _IDENT_RE.findall(text):
        if len(ident) > 1 and ident not in _ALLOWED_NAMES:
            raise _DegradeError(f"function/name not in whitelist: {ident!r}")
    # Two-phase parse (anti-hang): evaluate=False keeps integer powers unevaluated
    # so the complexity guard can reject 9**9**9-style payloads BEFORE sympy is
    # asked to actually compute them with evaluate=True.
    try:
        unevaluated = parse_expr(
            text, transformations=_TRANSFORMATIONS, evaluate=False, local_dict=_ALLOWED_FUNCS
        )
    except Exception as exc:
        raise _DegradeError(f"sympy cannot parse {text!r}: {exc}") from exc
    _complexity_guard(unevaluated)
    try:
        return parse_expr(
            text, transformations=_TRANSFORMATIONS, evaluate=True, local_dict=_ALLOWED_FUNCS
        )
    except Exception as exc:
        raise _DegradeError(f"sympy cannot parse {text!r}: {exc}") from exc


def _parse_equation(value: Any) -> sp.Eq:
    """Parse 'lhs=rhs' (or a bare expression meaning expr=0) into sympy Eq."""
    if not isinstance(value, str) or not value.strip():
        raise _DegradeError(f"not a parseable equation: {value!r}")
    text = value.strip().replace("==", "=")
    if any(ch in text for ch in "<>"):
        raise _DegradeError(f"inequalities are out of scope: {text!r}")
    if "=" in text:
        sides = text.split("=")
        if len(sides) != 2:
            raise _DegradeError(f"equation has multiple '=': {text!r}")
        lhs, rhs = (_parse(sides[0]), _parse(sides[1]))
    else:
        lhs, rhs = _parse(text), sp.Integer(0)
    eq = sp.Eq(lhs, rhs)
    if eq is sp.true or eq is sp.false:
        # e.g. "1=1" / "0=1" collapse to booleans; nothing to solve
        return sp.Eq(lhs - rhs, 0, evaluate=False)
    return eq


def _equiv(a: sp.Expr, b: sp.Expr) -> bool:
    """Symbolic equivalence: simplify(a-b)==0, with .equals / numeric fallback."""
    try:
        diff = sp.simplify(a - b)
    except Exception as exc:
        raise _DegradeError(f"cannot simplify difference: {exc}") from exc
    if diff == 0:
        return True
    try:
        eq = diff.equals(0)
    except Exception:
        eq = None
    if eq is True:
        return True
    if eq is False:
        return False
    try:
        if diff.is_number:
            return abs(complex(diff.evalf())) < 1e-9
    except Exception:
        pass
    return False


def _as_list(value: Any) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise _DegradeError(f"expected a list, got: {value!r}")


# ---------------------------------------------------------------------------
# kind: equation_solve
# ---------------------------------------------------------------------------

def _solve_equations(equations: Any, unknowns: Any) -> tuple[sp.Symbol, list]:
    eq_list = _as_list(equations)
    unk_list = _as_list(unknowns)
    if not eq_list or not unk_list:
        raise _DegradeError("equations/unknowns must be non-empty")
    if len(unk_list) != 1:
        raise _DegradeError("only a single unknown is supported")
    name = str(unk_list[0]).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise _DegradeError(f"invalid unknown name: {name!r}")
    sym = sp.Symbol(name)
    parsed = [_parse_equation(e) for e in eq_list]
    if not any(sym in eq.free_symbols for eq in parsed):
        raise _DegradeError(f"unknown {name!r} does not appear in the equations")
    try:
        solutions = sp.solve(parsed, sym)
    except Exception as exc:
        raise _DegradeError(f"sympy cannot solve the equations: {exc}") from exc
    if isinstance(solutions, dict):
        solutions = [solutions[sym]] if sym in solutions else []
    norm: list = []
    for s in solutions:
        if isinstance(s, (tuple, list)):
            if len(s) != 1:
                raise _DegradeError("solution shape unsupported (multi-valued tuple)")
            s = s[0]
        norm.append(s)
    return sym, norm


def _match_solution_set(solutions: list, claimed_exprs: list) -> tuple[bool, str]:
    """Multiset comparison with symbolic equivalence. -> (matched, detail)."""
    remaining = list(solutions)
    for c in claimed_exprs:
        hit = None
        for s in remaining:
            if _equiv(c, s):
                hit = s
                break
        if hit is None:
            return False, f"claimed value {sp.sstr(c)} is not a solution"
        remaining.remove(hit)
    if remaining:
        return False, f"claimed misses solution(s): {[sp.sstr(s) for s in remaining]}"
    return True, "claimed set matches the solution set"


def _verify_equation_solve(payload: dict) -> dict:
    _, solutions = _solve_equations(payload.get("equations"), payload.get("unknowns"))
    claimed_exprs = [_parse(c) for c in _as_list(payload.get("claimed"))]
    ok, detail = _match_solution_set(solutions, claimed_exprs)
    computed = "[" + ", ".join(sp.sstr(s) for s in solutions) + "]"
    return _result(PASS if ok else FAIL, detail, computed)


# ---------------------------------------------------------------------------
# kind: inequality_solve  (PRD-C-013 4b: single-variable inequality solution set)
# ---------------------------------------------------------------------------
# Strategy: turn BOTH the inequality and the claimed answer into sympy real
# solution Sets, then compare the Sets for exact equality. We never trust a
# raw string match. A chained / compound claimed form ("-2<=x<=2") is the AND
# of two relationals -> we intersect the per-relational solution sets (solveset
# rejects a bare And, so intersection is the robust path). Equivalent writings
# ("x**2-4<=0" vs "-2<=x<=2", "2*x>=6" vs "x>=3") collapse to the same Interval
# and therefore compare equal; a strict/non-strict boundary mismatch ("x>3" vs
# "x>=3") yields a non-empty symmetric difference -> fail.


def _relational_solution_set(text: str, sym: sp.Symbol) -> Any:
    """Parse one (possibly chained) relational string into its real solution Set.

    'a<x<b' is split on the relational operators and each piece is solved over
    the reals; the answer is the intersection. Operands are routed through the
    module's ``_parse`` so the charset / function whitelist / complexity guard
    all still apply. Any failure -> _DegradeError.
    """
    if not isinstance(text, str) or not text.strip():
        raise _DegradeError(f"not a parseable inequality: {text!r}")
    raw = text.strip().replace("==", "=")
    parts = _REL_SPLIT_RE.split(raw)
    operands = parts[0::2]
    ops = parts[1::2]
    if not ops:
        raise _DegradeError(f"no relational operator in {text!r}")
    if any(not o.strip() for o in operands):
        raise _DegradeError(f"malformed relational (empty operand): {text!r}")
    result: Any = sp.S.Reals
    sym_name = sym.name
    for i, op in enumerate(ops):
        lhs = _parse(operands[i])
        rhs = _parse(operands[i + 1])
        free_names = {s.name for s in (lhs - rhs).free_symbols}
        if sym_name not in free_names:
            raise _DegradeError(f"unknown {sym_name} absent from relational piece {op!r}")
        # _parse builds a fresh Symbol with no real-assumption; rebind it to the
        # real-domain unknown so solveset(..., S.Reals) treats it as real.
        relation = _REL_FUNCS[op](lhs, rhs).subs(sp.Symbol(sym_name), sym)
        try:
            piece = sp.solveset(relation, sym, sp.S.Reals)
        except Exception as exc:
            raise _DegradeError(f"sympy cannot solve relational {raw!r}: {exc}") from exc
        if not isinstance(piece, sp.Set):
            raise _DegradeError(f"inequality solution is not a Set: {piece!r}")
        result = result.intersect(piece)
    return result


def _verify_inequality_solve(payload: dict) -> dict:
    unk_list = _as_list(payload.get("unknown"))
    if len(unk_list) != 1:
        raise _DegradeError("inequality_solve supports exactly one unknown")
    name = str(unk_list[0]).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise _DegradeError(f"invalid unknown name: {name!r}")
    sym = sp.Symbol(name, real=True)

    inequality = payload.get("inequality")
    if not isinstance(inequality, str) or not _REL_SPLIT_RE.search(inequality):
        raise _DegradeError(f"inequality must contain a relational operator: {inequality!r}")

    true_set = _relational_solution_set(inequality, sym)
    claimed_set = _relational_solution_set(payload.get("claimed"), sym)

    try:
        diff = true_set.symmetric_difference(claimed_set)
        same = (true_set == claimed_set) or (diff == sp.S.EmptySet)
    except Exception as exc:
        raise _DegradeError(f"cannot compare solution sets: {exc}") from exc

    detail = (
        "claimed solution set matches the true solution set"
        if same
        else "claimed solution set does NOT match the true solution set"
    )
    return _result(PASS if same else FAIL, detail, sp.sstr(true_set))


# ---------------------------------------------------------------------------
# kind: rational_roots  (PRD-C-013 4b: 分式方程舍根/增根子集模式)
# ---------------------------------------------------------------------------
# A separate kind (NOT an overload of equation_solve) so the existing
# polynomial path stays untouched and the rational-equation semantics are
# explicit. Procedure:
#   1. parse 'lhs=rhs', form diff = together(lhs - rhs) = num/den
#   2. candidate roots = roots of the numerator (the "cleared" polynomial)
#   3. a candidate is SPURIOUS (增根) if it zeroes any denominator of lhs/rhs/diff
#      -> domain says x must keep every denominator != 0
#   4. valid roots = candidates that survive the domain check
#   5. PASS iff claimed (as a set) EXACTLY equals the valid-root set AND the
#      valid set is non-empty; otherwise FAIL. So claimed that keeps a spurious
#      root (漏剔增根) fails, claimed that names a denominator-killing root fails,
#      and the no-valid-root case can never "pass" with a stray claimed root.


def _rational_denominators(lhs: sp.Expr, rhs: sp.Expr) -> list[sp.Expr]:
    """Collect non-constant denominators from lhs, rhs and their combined form."""
    dens: list[sp.Expr] = []
    seen: set[str] = set()
    for side in (lhs, rhs, sp.together(lhs - rhs)):
        try:
            _, den = sp.fraction(sp.together(side))
        except Exception as exc:  # pragma: no cover - defensive
            raise _DegradeError(f"cannot extract denominator: {exc}") from exc
        if den.free_symbols:
            key = sp.sstr(den)
            if key not in seen:
                seen.add(key)
                dens.append(den)
    return dens


def _verify_rational_roots(payload: dict) -> dict:
    unk_list = _as_list(payload.get("unknowns", payload.get("unknown")))
    if len(unk_list) != 1:
        raise _DegradeError("rational_roots supports exactly one unknown")
    name = str(unk_list[0]).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise _DegradeError(f"invalid unknown name: {name!r}")
    sym = sp.Symbol(name)

    eq = _parse_equation(payload.get("equation"))
    lhs, rhs = eq.lhs, eq.rhs
    if sym not in (lhs - rhs).free_symbols:
        raise _DegradeError(f"unknown {name!r} does not appear in the equation")

    diff = sp.together(lhs - rhs)
    num, _ = sp.fraction(diff)
    try:
        candidates = sp.solve(sp.Eq(num, 0), sym)
    except Exception as exc:
        raise _DegradeError(f"sympy cannot solve numerator: {exc}") from exc
    if isinstance(candidates, dict):
        candidates = [candidates[sym]] if sym in candidates else []

    dens = _rational_denominators(lhs, rhs)

    valid: list = []
    for c in candidates:
        if isinstance(c, (tuple, list)):
            raise _DegradeError("solution shape unsupported (multi-valued)")
        spurious = False
        for den in dens:
            try:
                if sp.simplify(den.subs(sym, c)) == 0:
                    spurious = True
                    break
            except Exception as exc:
                raise _DegradeError(f"cannot test denominator at root: {exc}") from exc
        if not spurious:
            valid.append(c)

    claimed_exprs = [_parse(c) for c in _as_list(payload.get("claimed"))]
    computed = "[" + ", ".join(sp.sstr(s) for s in valid) + "]"

    if not valid:
        # equation has NO valid root (all candidates are spurious / none exist):
        # the only correct answer is the empty set; any claimed value -> fail.
        if claimed_exprs:
            return _result(FAIL, "equation has no valid root (all roots are spurious)", computed)
        return _result(PASS, "equation has no valid root and claimed set is empty", computed)

    ok, detail = _match_solution_set(valid, claimed_exprs)
    return _result(PASS if ok else FAIL, detail, computed)


# ---------------------------------------------------------------------------
# kind: expr_equiv
# ---------------------------------------------------------------------------

def _verify_expr_equiv(payload: dict) -> dict:
    a = _parse(payload.get("expr_a"))
    b = _parse(payload.get("expr_b"))
    ok = _equiv(a, b)
    computed = sp.sstr(sp.simplify(a - b))
    detail = "expressions are equivalent" if ok else "expressions are NOT equivalent"
    return _result(PASS if ok else FAIL, detail, computed)


# ---------------------------------------------------------------------------
# kind: numeric
# ---------------------------------------------------------------------------

def _to_float(expr: sp.Expr, label: str) -> float:
    if expr.free_symbols:
        raise _DegradeError(f"{label} is not a pure number: {sp.sstr(expr)}")
    try:
        val = complex(expr.evalf())
    except Exception as exc:
        raise _DegradeError(f"cannot evaluate {label} numerically: {exc}") from exc
    if abs(val.imag) > 1e-12:
        raise _DegradeError(f"{label} is not a real number: {val}")
    return float(val.real)


def _get_tol(payload: dict) -> float:
    tol = payload.get("tol", _DEFAULT_TOL)
    try:
        tol = float(tol)
    except Exception as exc:
        raise _DegradeError(f"invalid tol: {tol!r}") from exc
    if tol < 0:
        raise _DegradeError(f"tol must be >= 0: {tol!r}")
    return tol


def _verify_numeric(payload: dict) -> dict:
    value = _to_float(_parse(payload.get("expr")), "expr")
    claimed = _to_float(_parse(payload.get("claimed")), "claimed")
    tol = _get_tol(payload)
    ok = abs(value - claimed) <= tol
    detail = (
        f"computed={value!r}, claimed={claimed!r}, tol={tol!r}: "
        + ("within tolerance" if ok else "outside tolerance")
    )
    return _result(PASS if ok else FAIL, detail, value)


# ---------------------------------------------------------------------------
# kind: choice  (G2: a distractor hitting the ground truth -> fail)
# ---------------------------------------------------------------------------

def _ground_matcher(ground: Any) -> tuple[Callable[[Any], bool], str]:
    """Build candidate->bool matcher from the ground sub-payload.

    Returns (matcher, computed_truth_str). Unsupported ground -> _DegradeError.
    """
    if not isinstance(ground, dict):
        raise _DegradeError("choice.ground must be a dict sub-payload")
    gkind = ground.get("kind")

    if gkind == "equation_solve":
        _, solutions = _solve_equations(ground.get("equations"), ground.get("unknowns"))

        def matcher(candidate: Any) -> bool:
            if isinstance(candidate, str) and "," in candidate:
                parts = [p for p in candidate.split(",") if p.strip()]
            else:
                parts = [candidate]
            exprs = [_parse(p) for p in parts]
            ok, _ = _match_solution_set(solutions, exprs)
            return ok

        computed = "[" + ", ".join(sp.sstr(s) for s in solutions) + "]"
        return matcher, computed

    if gkind == "expr_equiv":
        ref = _parse(ground.get("expr_a"))
        return (lambda candidate: _equiv(_parse(candidate), ref)), sp.sstr(ref)

    if gkind == "numeric":
        truth = _to_float(_parse(ground.get("expr")), "ground.expr")
        tol = _get_tol(ground)

        def matcher(candidate: Any) -> bool:
            expr = _parse(candidate)
            if expr.free_symbols:
                return False
            return abs(_to_float(expr, "option") - truth) <= tol

        return matcher, str(truth)

    raise _DegradeError(f"unsupported ground kind for choice: {gkind!r}")


def _verify_choice(payload: dict) -> dict:
    options = payload.get("options")
    if not isinstance(options, dict) or not options:
        raise _DegradeError("choice.options must be a non-empty dict")
    claimed_correct = payload.get("claimed_correct")
    if claimed_correct not in options:
        raise _DegradeError(f"claimed_correct {claimed_correct!r} is not an option key")

    matcher, computed = _ground_matcher(payload.get("ground"))

    # 1) The claimed option is judged FIRST: if it provably misses the ground
    #    truth, that is a hard FAIL — a dirty (unparseable) distractor must not
    #    mask it behind a degrade. (Unparseable claimed option still degrades:
    #    we genuinely cannot verify the answer then.)
    if not matcher(options[claimed_correct]):
        return _result(
            FAIL,
            f"claimed correct option {claimed_correct!r} does not match the ground truth",
            computed,
        )

    # 2) Distractors: an unparseable option (text like "无解"/"以上都不对") cannot
    #    numerically equal the truth -> treated as non-matching, never degrades
    #    the whole verdict.
    for key in sorted(options):
        if key == claimed_correct:
            continue
        try:
            matches = matcher(options[key])
        except _DegradeError:
            continue
        if matches:
            return _result(
                FAIL,
                f"distractor {key!r} also matches the ground truth (G2)",
                computed,
            )
    return _result(
        PASS,
        f"option {claimed_correct!r} uniquely matches the ground truth",
        computed,
    )


# ---------------------------------------------------------------------------
# ⑦ 反退化代码闸 (PRD-C-015 批3): endpoint-degeneracy detection — PURE ALGEBRA, ZERO LLM
# ---------------------------------------------------------------------------
# Detect the "临界 k 陷阱" (predicted by H1 preflight as a cross-model defect):
# a 最值/动点 construction whose optimal-solution stationary point happens to land
# on an ENDPOINT of the moving-point's interval (e.g. 胡不归 P=B / PB=0). The
# answer may be numerically correct but the construction mechanism has collapsed —
# a degenerate junk variant that must trigger REGEN.
#
# Contract: ``check_endpoint_degeneracy(payload) -> dict`` returns
#   {"verdict": "degenerate"|"ok"|"degrade", "detail": str, "computed": str|None}.
# Hard rules (与 verify() 同精神):
#   - pure function: zero LLM, zero network; any sympy failure / out-of-scope -> degrade
#     (NEVER raises, NEVER mis-flags as degenerate) — 闸必有降级路径, 算不了 -> ⚠继续不卡死。
#   - 判决只读代数计算结果, 永不采信 LLM 自评 (本函数压根不碰 LLM)。
#   - 容差 (eps): 驻点在端点 eps 内 (相对区间长度归一) 才算落端点 = 退化。
#
# payload schema (extractor 抽不成 -> {"kind":"none"} -> 上层不判, 即降级放行):
#   {"kind": "endpoint_extremum",
#    "objective": "<f(t) 表达式, 单变量 t>",   # 要被最小化/最大化的目标函数
#    "var": "t",                               # 动点参数名
#    "interval": ["<lo>", "<hi>"],             # 动点定义区间端点 (表达式或数)
#    "sense": "min"|"max"}                     # 求最小还是最大 (缺省 min)
DEGENERATE = "degenerate"
DEGEN_OK = "ok"
# 端点判定相对容差 (归一到区间长度): |t* - 端点| / |hi - lo| < eps -> 落端点。
_DEGEN_EPS = 1e-6


def check_endpoint_degeneracy(payload: Any) -> dict:
    """⑦ 反退化闸: 目标函数在动点区间上的最优点是否落在区间端点 (退化构型)。Never raises.

    Returns {"verdict": "degenerate"|"ok"|"degrade", "detail": str, "computed": str|None}.
      - degenerate: 最优点落在区间端点 (机制失效的退化废题) -> 上层 REGEN。
      - ok        : 最优点在区间内部 (合法构型) -> 放行。
      - degrade   : 抽不成 / 超范围 / sympy 算不了 -> 上层标 ⚠ 继续 (降级路径, 不卡死)。
    """
    try:
        if not isinstance(payload, dict):
            return _result(DEGRADE, f"payload must be a dict, got {type(payload).__name__}")
        if payload.get("kind") != "endpoint_extremum":
            return _result(DEGRADE, f"not an endpoint_extremum payload: {payload.get('kind')!r}")

        name = str(payload.get("var") or "t").strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            return _result(DEGRADE, f"invalid var name: {name!r}")
        var = sp.Symbol(name, real=True)

        # _parse builds plain Symbols (no real-assumption) -> rebind by NAME to the
        # real-domain var so diff/solve/subs all reference the same symbol (matches
        # the inequality_solve rebinding pattern). Compare by free-symbol NAMES.
        objective = _parse(payload.get("objective"))
        free_names = {s.name for s in objective.free_symbols}
        if name not in free_names:
            return _result(DEGRADE, f"objective does not contain var {name!r}")
        if free_names - {name}:
            return _result(DEGRADE, "objective has free symbols other than var (out of scope)")
        objective = objective.subs(sp.Symbol(name), var)

        interval = payload.get("interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            return _result(DEGRADE, f"interval must be a [lo, hi] pair: {interval!r}")
        lo = _to_float(_parse(interval[0]), "interval lo")
        hi = _to_float(_parse(interval[1]), "interval hi")
        if not (hi > lo):
            return _result(DEGRADE, f"degenerate interval (hi must exceed lo): [{lo}, {hi}]")
        span = hi - lo

        sense = str(payload.get("sense") or "min").strip().lower()
        if sense not in ("min", "max"):
            return _result(DEGRADE, f"sense must be 'min' or 'max': {sense!r}")

        # 候选最优点 = 内部驻点 (f'=0 在 (lo,hi) 内) + 两端点。逐个数值求目标值, 取最优。
        try:
            deriv = sp.diff(objective, var)
            crit = sp.solve(sp.Eq(deriv, 0), var)
        except Exception as exc:
            return _result(DEGRADE, f"sympy cannot find critical points: {exc}")
        if isinstance(crit, dict):
            crit = [crit[var]] if var in crit else []

        interior: list[float] = []
        for c in crit:
            if isinstance(c, (tuple, list)):
                continue
            try:
                cf = _to_float(c, "critical point")
            except _DegradeError:
                continue  # 复根/符号根 -> 跳过该驻点 (不降级整闸)
            if lo + _DEGEN_EPS * span < cf < hi - _DEGEN_EPS * span:
                interior.append(cf)

        def _val(t: float) -> float:
            return _to_float(objective.subs(var, sp.Float(t)), "objective value")

        candidates: list[tuple[float, float, str]] = []  # (objective_value, t, where)
        candidates.append((_val(lo), lo, "endpoint"))
        candidates.append((_val(hi), hi, "endpoint"))
        for cf in interior:
            candidates.append((_val(cf), cf, "interior"))

        best = min(candidates, key=lambda x: x[0]) if sense == "min" else max(
            candidates, key=lambda x: x[0]
        )
        best_val, best_t, where = best
        # 平局判定: 若内部驻点取到与端点相同的最优值 (容差内), 视为「内部也最优」= 不退化
        # (机制并未失效, 端点只是碰巧并列)。只有「最优值唯一在端点」才判退化。
        tol = max(abs(best_val), 1.0) * 1e-9 + 1e-12
        interior_best = None
        for v, t, w in candidates:
            if w == "interior" and abs(v - best_val) <= tol:
                interior_best = t
                break
        computed = f"{sense} at {name}={best_t!r} (value={best_val!r}); interval=[{lo}, {hi}]"
        if where == "interior" or interior_best is not None:
            return _result(DEGEN_OK, "optimum is attained at an interior stationary point", computed)
        return _result(
            DEGENERATE,
            f"optimum is attained only at interval endpoint {name}={best_t!r} "
            "(degenerate construction — mechanism collapsed)",
            computed,
        )
    except _DegradeError as exc:
        return _result(DEGRADE, str(exc))
    except Exception as exc:  # absolute last resort: never propagate (G5)
        return _result(DEGRADE, f"unexpected error: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# public entry
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Callable[[dict], dict]] = {
    "equation_solve": _verify_equation_solve,
    "expr_equiv": _verify_expr_equiv,
    "numeric": _verify_numeric,
    "choice": _verify_choice,
    "inequality_solve": _verify_inequality_solve,
    "rational_roots": _verify_rational_roots,
}


def verify(payload: dict) -> dict:
    """Verify one structured math payload. Never raises.

    Returns {"verdict": "pass"|"fail"|"degrade", "detail": str, "computed": str|None}.
    """
    try:
        if not isinstance(payload, dict):
            return _result(DEGRADE, f"payload must be a dict, got {type(payload).__name__}")
        kind = payload.get("kind")
        handler = _HANDLERS.get(kind)
        if handler is None:
            return _result(DEGRADE, f"unknown payload kind: {kind!r}")
        return handler(payload)
    except _DegradeError as exc:
        return _result(DEGRADE, str(exc))
    except Exception as exc:  # absolute last resort: never propagate (G5)
        return _result(DEGRADE, f"unexpected error: {type(exc).__name__}: {exc}")
