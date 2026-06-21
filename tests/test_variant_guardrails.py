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

from langchain_core.messages import AIMessage, HumanMessage

from agents.variant import (
    ADD_COUNT_MAX,
    EDIT_ACTIONS,
    INTENT_CLARIFY,
    INTENT_CONFIRM,
    INTENT_EDIT,
    INTENT_QA,
    INTENT_REVISE,
    INTENT_SOLUTION_ONLY,
    VALID_INTENTS,
    _is_multi_subquestion,
    _latest_ai_text,
    _looks_like_conservation_hit,
    _qtype_from_note,
    route_after_parse,
    structure_lint,
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
        INTENT_SOLUTION_ONLY: "solution_only",  # 整改3（2026-06-12）：解法修正分支
    }
    assert set(expected) == VALID_INTENTS  # enum and routing stay in lockstep
    for intent, branch in expected.items():
        # 🔴 PRD-A-021 R4·F7：答疑分支现有「空题组护栏」——QA 须有题组才进 answer。
        #   本 lockstep 测验证「每个 intent 落在存在的分支」，给一道占位题让 QA 走正常 answer。
        state = {"pending": {"intent": intent, "ops": []}, "items": [{"stem": "x"}]}
        assert route_after_parse(state) == branch


def test_qa_empty_items_routes_to_ask_clarify():
    """🔴 PRD-A-021 R4·F7：答疑前置空护栏 —— items 空时 INTENT_QA 不进 answer（白烧一次 LLM），
    改落 ask_clarify 让老师先贴图/出题。items 非空时仍正常进 answer。"""
    empty = {"pending": {"intent": INTENT_QA, "ops": []}, "items": []}
    assert route_after_parse(empty) == "ask_clarify"
    nonempty = {"pending": {"intent": INTENT_QA, "ops": []}, "items": [{"stem": "x"}]}
    assert route_after_parse(nonempty) == "answer"


# ---------------------------------------------------------------------------
# P11.3: structure_lint —— 单题题型结构纯函数（_QTYPE_CONTRACT 代码镜像）
# ---------------------------------------------------------------------------

def test_structure_lint_choice_multi_subquestion_caught():
    # 选择题混入 (1)(2) 多小问 = 嵌合体缺陷（必抓）
    item = {
        "qtype": "选择",
        "stem": "已知函数 f(x)=x+1。(1) 求 f(2)；(2) 求 f(3)。下列说法正确的是",
        "answer": "A",
    }
    defects = structure_lint(item)
    assert defects, "选择题嵌合体应被抓"
    assert any("多小问" in d for d in defects)


def test_structure_lint_choice_too_few_options_caught():
    # 只识别到 2 个选项 (<3) → 选项不足缺陷
    item = {
        "qtype": "选择题",  # 走 _QTYPE_ALIAS 归一为「选择」
        "stem": "下列哪个是质数？ A. 4  B. 6",
        "answer": "A",
    }
    defects = structure_lint(item)
    assert any("选项不足" in d for d in defects)


def test_structure_lint_choice_answer_not_letter_caught():
    item = {
        "qtype": "选择",
        "stem": "下列哪个是质数？ A. 2  B. 4  C. 6  D. 8",
        "answer": "2",  # 标答应是字母 A，不是数值
    }
    defects = structure_lint(item)
    assert any("字母" in d for d in defects)


def test_structure_lint_compliant_choice_passes():
    item = {
        "qtype": "选择",
        "stem": "下列哪个是质数？ A. 2  B. 4  C. 6  D. 8",
        "answer": "A",
    }
    assert structure_lint(item) == []


def test_structure_lint_fill_blank_missing_blank_caught():
    item = {"qtype": "填空", "stem": "计算 1+1 的结果是多少", "answer": "2"}
    defects = structure_lint(item)
    assert any("空位" in d for d in defects)


def test_structure_lint_fill_blank_with_blank_passes():
    item = {"qtype": "填空题", "stem": "计算 1+1 = ____", "answer": "2"}
    assert structure_lint(item) == []


def test_structure_lint_solve_allows_multi_subquestion():
    # 解答题允许 (1)(2) 多小问 → 不报缺陷
    item = {
        "qtype": "解答",
        "stem": "已知一元二次方程。(1) 求根；(2) 求判别式。",
        "answer": "见解析",
    }
    assert structure_lint(item) == []


def test_structure_lint_degrades_on_garbage_qtype():
    # 未知题型/异常输入 → 降级放行（绝不卡死）
    assert structure_lint({"qtype": None, "stem": None, "answer": None}) == []
    assert structure_lint({"qtype": "未知", "stem": "随便", "answer": "x"}) == []


# ---------------------------------------------------------------------------
# 对抗审④ — _MULTI_SUBQ 误伤函数记号 / 单(1)：收紧为「≥2 连号小问标记」才判嵌合体
# ---------------------------------------------------------------------------

def test_multi_subq_function_notation_not_flagged():
    # f(1)/g(2)/点(1)：左括号前是字母/字 → 函数记号，不算多小问（旧正则误伤的核心场景）
    assert not _is_multi_subquestion("已知 f(1)=2，求 f(2) 的值，下列正确的是")
    assert not _is_multi_subquestion("设 g(2) 与 h(3) 满足关系，问")


def test_multi_subq_single_marker_not_flagged():
    # 单个 (1) / 单个 ① → 不算嵌合体（需 ≥2 连号）
    assert not _is_multi_subquestion("根据题意 (1) 处应填什么")
    assert not _is_multi_subquestion("第 ① 步的结果是")


def test_multi_subq_consecutive_paren_flagged():
    # (1)(2) 连号小问 → 真嵌合体
    assert _is_multi_subquestion("(1) 求根；(2) 求判别式")
    # 中文括号 + 句中（左括号前是标点非 \w）
    assert _is_multi_subquestion("已知方程。（1）求 x；（2）求 y。")


def test_multi_subq_consecutive_circled_flagged():
    # ①② 连续圆圈编号 → 真嵌合体
    assert _is_multi_subquestion("步骤：① 移项 ② 合并同类项")


def test_multi_subq_non_consecutive_not_flagged():
    # 非连号（如只有 (1) 和 (3)，缺 (2)）→ 不判（保守，避免误伤散落编号）
    assert not _is_multi_subquestion("参考 (1) 和 (3) 两处")


def test_structure_lint_choice_function_notation_passes():
    # 选择题题干含函数记号 f(1)/f(2) → 不再误报多小问缺陷（白烧预算的根因）
    item = {
        "qtype": "选择",
        "stem": "已知 f(x)=2x，则 f(1)+f(2) 的值是？ A. 4  B. 5  C. 6  D. 7",
        "answer": "C",
    }
    assert structure_lint(item) == []


def test_structure_lint_choice_real_chimera_still_caught():
    # 真嵌合体 (1)(2) 连号 → 仍被抓（收紧没放过真缺陷）
    item = {
        "qtype": "选择",
        "stem": "解方程。(1) 求 x；(2) 验根。下列正确的是 A.1 B.2 C.3 D.4",
        "answer": "A",
    }
    defects = structure_lint(item)
    assert any("多小问" in d for d in defects)


# ---------------------------------------------------------------------------
# BUG-001 · _qtype_from_note：编辑 note → 目标题型抽取（改题型才真改）
# ---------------------------------------------------------------------------

def test_qtype_from_note_change_to_choice():
    assert _qtype_from_note("改成选择题") == "选择"
    assert _qtype_from_note("这题改填空") == "填空"
    assert _qtype_from_note("换成解答题") == "解答"


def test_qtype_from_note_aliases_map_to_canon():
    # 计算/应用/证明/大题 → 解答；单选 → 选择
    assert _qtype_from_note("改成计算题") == "解答"
    assert _qtype_from_note("换成应用题") == "解答"
    assert _qtype_from_note("改成单选") == "选择"


def test_qtype_from_note_takes_last_when_from_x_to_y():
    # 「从填空改成选择」→ 取最后出现的目标题型（选择）
    assert _qtype_from_note("从填空改成选择题") == "选择"


def test_qtype_from_note_no_change_verb_returns_none():
    # 只是陈述题型 / 无改动词 → 不判改题型（避免「这是选择题，数字简单点」误伤）
    assert _qtype_from_note("数字简单点") is None
    assert _qtype_from_note("这是一道选择题") is None  # 含"选择"但无改/换/变/成动词
    assert _qtype_from_note("") is None
    assert _qtype_from_note(None) is None


# ---------------------------------------------------------------------------
# BUG-003 · _looks_like_conservation_hit：clarify 文案归因二分
# ---------------------------------------------------------------------------

def test_conservation_hit_detects_kp_grade_change():
    # 显式动「考点/知识点/年级/学段」对象词 + 改动词 → 判疑似撞守恒（给守恒说明）
    assert _looks_like_conservation_hit("换个考点")
    assert _looks_like_conservation_hit("改成八年级的")
    assert _looks_like_conservation_hit("换知识点")
    assert _looks_like_conservation_hit("改下年级")


def test_conservation_hit_false_for_normal_edits():
    # 改题型/换场景/改数量 都不是撞守恒 → 不触发守恒文案
    assert not _looks_like_conservation_hit("第1题改成选择题")
    assert not _looks_like_conservation_hit("加入杭州场景")
    assert not _looks_like_conservation_hit("出两道")
    assert not _looks_like_conservation_hit("阿巴阿巴乱说一通")
    assert not _looks_like_conservation_hit("")


# ---------------------------------------------------------------------------
# BUG-006 · _latest_ai_text：注入上一轮 AI 消息供承接判别
# ---------------------------------------------------------------------------

def test_latest_ai_text_returns_most_recent_ai():
    msgs = [
        HumanMessage(content="贴图"),
        AIMessage(content="我可以把第2题完整讲一遍"),
        HumanMessage(content="给学生讲"),
    ]
    assert _latest_ai_text(msgs) == "我可以把第2题完整讲一遍"


def test_latest_ai_text_empty_when_no_ai():
    assert _latest_ai_text([HumanMessage(content="hi")]) == ""
    assert _latest_ai_text([]) == ""
