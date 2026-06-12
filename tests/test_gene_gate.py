# -*- coding: utf-8 -*-
"""Unit tests for Gate-A (gene gate) — PRD-C-014 B2·T2 退役换血.

🔴 B2·T2 起 Gate-A LLM judge 全链退役（GENE_JUDGE_PROMPT / _gene_judge_one /
gene_gate_decision / _gene_feedback / _gene_judge_prompt / gene_judge_knobs_spec /
_gene_target_qtype* / _gene_facts_for 全删）。内涵换为**纯代码三检**（gene_gate_check）：
  ① structure_lint：题型/结构闭集 lint（选择题别长成多小问嵌合体/选项不足/标答非字母；填空缺空位）。
  ② _surface_check：题干归一化相似度 >0.85 或 数字与母题全同 → 抄题。
  ③ 守恒透传：W2 注入的题型/考察类型守恒在 item 上的声明性校验。
任一检命中 = gene={gate:"warn", flags, reason}（**只警示不硬拦、不回炉、不剔题**，闸门降级路径）；
三检全过 = gene={gate:"pass", flags:[]}。判决仍只碰形态/表皮，不碰答案对错（对错归闸B sympy）。

旧断言 → 新断言映射（语义升级，非洗绿）：
  - gene_gate_decision(judge dict) == pass/rework      → gene_gate_check(item, facts) -> {gate}
    （从「LLM judge JSON 的纯函数裁决」升级为「item+母题 facts 的纯代码三检裁决」）。
  - qtype_match=False -> rework                         → 题型守恒破（变式 qtype≠母题）-> warn + flag qtype_conservation。
  - structure_match=False -> rework                     → structure_lint 命中（选择题嵌多小问等）-> warn + flag structure。
  - surface_swapped=False -> rework                     → _surface_check 命中（相似度高/数字全同）-> warn + flag surface。
  - judge 失败 -> skipped 放行                          → 纯代码不会失败；异常一律降级 pass（不再产生 skipped）。
  - rework -> 回炉重生 1 次再判 / 守恒校验 / from_edit 永不回炉  → 整段删除：Gate-A 不再回炉（只警示）。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import (
    GENE_GATE_PASS,
    GENE_GATE_WARN,
    gene_gate,
    gene_gate_check,
    variant,
)
from agents.variant_support import build_create_bo

# 母题 facts（_mother_facts 形态的最小子集 + dna.exam_type 供 ③ 守恒透传）
_FACTS = {
    "kp_name": "一元一次方程",
    "grade": "七年级上学期",
    "qtype": "解答",
    "stem": "解方程 2x + 3 = 11",
    "dna": {"exam_type": "直接计算"},
}


# ---------------------------------------------------------------------------
# gene_gate_check: pure-function three-check verdict
# ---------------------------------------------------------------------------


def test_clean_parallel_item_passes():
    # qtype 守恒（解答=解答）+ 表皮已换（题面/数字全不同）+ 结构合规 → 三检全过
    item = {"stem": "解方程 5y - 7 = 18", "qtype": "解答"}
    out = gene_gate_check(item, _FACTS)
    assert out["gate"] == GENE_GATE_PASS
    assert out["flags"] == []


def test_qtype_conservation_break_warns():
    # ③ 变式题型「选择」≠ 母题「解答」→ 守恒破 → warn + flag
    item = {"stem": "下列哪个是方程的解 5y-7=18", "qtype": "选择"}
    out = gene_gate_check(item, _FACTS)
    assert out["gate"] == GENE_GATE_WARN
    assert "qtype_conservation" in out["flags"]


def test_qtype_conservation_alias_normalized_is_ok():
    # 「计算题」别名归一为「解答」== 母题「解答」→ 不算守恒破
    item = {"stem": "解方程 5y - 7 = 18", "qtype": "计算题"}
    out = gene_gate_check(item, _FACTS)
    assert out["gate"] == GENE_GATE_PASS


def test_from_edit_exempts_qtype_conservation():
    # 转题型编辑（老师明确点名改造，from_edit）→ 题型守恒豁免，不因 qtype 不同打 warn
    item = {"stem": "下列哪个是方程的解 5y-7=18", "qtype": "选择", "from_edit": True}
    out = gene_gate_check(item, _FACTS)
    assert "qtype_conservation" not in out["flags"]


def test_exam_type_conservation_break_warns():
    # ③ 若 item 声明 exam_type 且与母题不同 → 守恒破 + flag
    item = {"stem": "求证 ...", "qtype": "解答", "exam_type": "证明推理"}
    out = gene_gate_check(item, _FACTS)
    assert out["gate"] == GENE_GATE_WARN
    assert "exam_type_conservation" in out["flags"]


def test_exam_type_undeclared_does_not_flag():
    # item 未声明 exam_type（generate 产物常态）→ 不做声明性校验，不误报
    item = {"stem": "解方程 5y - 7 = 18", "qtype": "解答"}
    out = gene_gate_check(item, _FACTS)
    assert "exam_type_conservation" not in out["flags"]


def test_surface_copy_high_similarity_warns():
    # ② 题干与母题几乎相同（只复读）→ 相似度 >0.85 → 抄题 flag
    item = {"stem": "解方程 2x + 3 = 11", "qtype": "解答"}
    out = gene_gate_check(item, _FACTS)
    assert out["gate"] == GENE_GATE_WARN
    assert "surface" in out["flags"]


def test_surface_same_numbers_warns():
    # ② 数字与母题完全相同（表皮没换）→ flag（即便文字场景改了）
    item = {"stem": "某商店进货 2 件，售出 3 件后剩 11 件，问……", "qtype": "解答"}
    facts = dict(_FACTS, stem="买苹果 2 个又 3 个共 11 元")  # 数字集合 [2,3,11] 相同
    out = gene_gate_check(item, facts)
    assert out["gate"] == GENE_GATE_WARN
    assert "surface" in out["flags"]


def test_structure_lint_choice_multi_subquestion_warns():
    # ① 选择题混入多小问 (1)(2) 嵌合体 → structure flag（与表皮/守恒正交）
    item = {
        "stem": "已知 x=2。(1) 求 y。(2) 求 z。\nA. 1 B. 2 C. 3 D. 4",
        "qtype": "选择",
    }
    # qtype 用母题同型避免 qtype 守恒 flag 干扰本检
    facts = dict(_FACTS, qtype="选择")
    out = gene_gate_check(item, facts)
    assert out["gate"] == GENE_GATE_WARN
    assert "structure" in out["flags"]


def test_check_does_not_mutate_input_item():
    item = {"stem": "解方程 5y - 7 = 18", "qtype": "选择"}
    snapshot = dict(item)
    gene_gate_check(item, _FACTS)
    assert item == snapshot  # gene_gate_check 不改入参（写 gene 在 _gene_one_item 做）


def test_check_degrades_to_pass_on_exception(monkeypatch):
    # 三检任何环节异常 → 降级 pass（铁律④：闸门必有降级路径，绝不卡死出题）
    def boom(_item):
        raise RuntimeError("lint blew up")

    monkeypatch.setattr(variant_mod, "structure_lint", boom)
    out = gene_gate_check({"stem": "x", "qtype": "解答"}, _FACTS)
    assert out["gate"] == GENE_GATE_PASS


# ---------------------------------------------------------------------------
# gene_gate node behavior (pure code, no LLM / no network monkeypatch needed)
# ---------------------------------------------------------------------------

_STATE_BASE = {
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.9},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {"stem": "解方程 2x + 3 = 11", "answer": "x=4", "difficulty": 3},
}


def _run_gate(items):
    state = dict(_STATE_BASE, items=items)
    out = asyncio.run(gene_gate(state, {}))
    assert out["messages"] == []  # node contract: always returns a messages key
    return out["items"]


def test_node_clean_item_passes():
    items = _run_gate([{"stem": "解方程 7m - 1 = 20", "qtype": "解答"}])
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS


def test_node_qtype_break_warns():
    items = _run_gate([{"stem": "全新选择题面 7m-1=20", "qtype": "选择"}])
    assert items[0]["gene"]["gate"] == GENE_GATE_WARN
    assert "qtype_conservation" in items[0]["gene"]["flags"]


def test_node_already_marked_items_not_rejudged():
    items = _run_gate(
        [
            {"stem": "old", "gene": {"gate": GENE_GATE_PASS}, "check": {"badge": "ok"}},
            {"stem": "全新题面 7m-1=20", "qtype": "解答"},
        ]
    )
    assert items[0]["gene"]["gate"] == GENE_GATE_PASS  # untouched
    assert items[1]["gene"]["gate"] == GENE_GATE_PASS  # freshly checked


def test_node_concurrent_results_keep_input_order():
    items = _run_gate(
        [{"stem": f"全新题面 {i}x + {i} = {i*2}", "qtype": "解答"} for i in (3, 5, 7)]
    )
    assert [it["stem"][:4] for it in items] == ["全新题面", "全新题面", "全新题面"]
    assert all(it["gene"]["gate"] == GENE_GATE_PASS for it in items)


def test_node_no_llm_judge_call_in_generate_round(monkeypatch):
    # G9 佐证：gene_gate 节点跑一轮，绝无任何 LLM 调用（_ainvoke_text 一次都不触达）。
    calls = {"n": 0}

    async def spy(*a, **k):
        calls["n"] += 1
        return "{}"

    monkeypatch.setattr(variant_mod, "_ainvoke_text", spy)
    _run_gate(
        [
            {"stem": "全新题面 4a-2=10", "qtype": "解答"},
            {"stem": "全新选择题面 4a-2=10", "qtype": "选择"},  # warn 路径同样零 LLM
        ]
    )
    assert calls["n"] == 0  # 闸A 纯代码：generate 一轮无 judge LLM 调用


# ---------------------------------------------------------------------------
# auxTags removed from create BO (PRD-C-014 B1: schema 收敛 DROP 了 biz_question.aux_tags
# 列；gene/verify 审计标记走 BE ai 表 conflict_flags，不再塞进 create BO)。
# 标记仍活在 item.gene.gate（FE 4d 展示/快照），但 BO 不再带 auxTags / gene_gate。
# ---------------------------------------------------------------------------


def test_create_bo_no_longer_carries_aux_tags():
    facts = {"qtype": "qt", "subject_id": "3071", "dim1_kp_id": "3071001"}
    for gate in (GENE_GATE_PASS, GENE_GATE_WARN):
        bo = build_create_bo(
            {"stem": "s", "answer": "a", "gene": {"gate": gate}}, facts
        )
        # B1: auxTags 列已 DROP，BO 不再带该键（gene 标记仍活在 item.gene 供 FE 展示）
        assert "auxTags" not in bo
        assert "dim3Skill" not in bo and "freeTag" not in bo  # 三件套全删


# ---------------------------------------------------------------------------
# graph wiring: Gate-A sits between producers and Gate-B (unchanged by T2)
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
# check (unchanged by T2 —难度评级归 LLM rubric / 组内一致性归本纯函数；判对错仍归 sympy).
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
    items = [
        {"level": "normal", "difficulty": None},
        {"level": "hard", "difficulty": "x"},
    ]
    assert variant_mod.difficulty_consistency_defects(items) == []


def test_difficulty_consistency_hard_equal_to_normal_is_ok():
    items = [
        {"level": "normal", "difficulty": 3},
        {"level": "hard", "difficulty": 3},
    ]
    assert variant_mod.difficulty_consistency_defects(items) == []
