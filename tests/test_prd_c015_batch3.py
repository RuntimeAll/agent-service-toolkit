# -*- coding: utf-8 -*-
"""PRD-C-015 批3·双轴注卡 + 守恒软警 + 反退化代码闸 单测（mock LLM/DB，零网络）。

覆盖 gate：
- G3（W2' 注卡条件矩阵：难度≥3+非M00 注卡 / 难度<3 或仅M00 不注；卡片文本逐字一致 + 反退化约束）
- G4（W3' 守恒软警：变式 models ⊄ 母题∪{M00}∪候选池 → ⚠+待命名池一条；不越界→无⚠；均不打回）
- G9（⑦ 反退化闸：最优点落区间端点→判退化触发 REGEN；内部驻点→放行；算不了→降级放行）

铁律对照：
- 注卡是 prompt 引导（生成侧），不混 pass/fail 判决。
- 守恒软警不打回（仅 ⚠ + 记录）；反退化闸可 REGEN（超限弃）——两者别搞反。
- 反退化闸纯代数·零 LLM（check_endpoint_degeneracy 全程不碰 LLM）。
- 一切闸门必有降级路径（取卡失败→不注卡；反退化算不了→放行）。
"""

import asyncio

import agents.math_verify as mv
import agents.model_anchor as ma
import agents.variant as V
from agents.model_anchor import (
    M00,
    M00_ID,
    ANTI_DEGEN_CLAUSE,
    build_model_cards_clause,
    model_conservation_warn,
)


# ===========================================================================
# G3·W2' 注卡（model_anchor.build_model_cards_clause + variant._should_note_card/_model_cards_clause）
# ===========================================================================
_CARD_M25 = {
    "id": "M25", "name": "定边对定角（隐圆主模型）",
    "trigger_feature": "定线段对定角的动顶点",
    "action_conclusion": "顶点轨迹=定弧（R=边/2sinθ）→ 转圆问题",
}
_CARD_M32 = {
    "id": "M32", "name": "隐圆最值",
    "trigger_feature": "动点在定圆/定弧上",
    "action_conclusion": "连心距 ± 半径 = 最值；圆外点先定圆心",
}


def test_build_cards_clause_verbatim_and_anti_degen():
    """卡片文本逐字取词库行 + 含反退化约束（G3）。"""
    clause = build_model_cards_clause([_CARD_M25, _CARD_M32], with_anti_degen=True)
    assert "定线段对定角的动顶点" in clause          # trigger 逐字
    assert "顶点轨迹=定弧（R=边/2sinθ）→ 转圆问题" in clause  # action 逐字
    assert "隐圆最值" in clause
    assert "反退化" in clause and "区间端点" in clause  # 反退化约束在
    assert ANTI_DEGEN_CLAUSE in clause


def test_build_cards_clause_empty_returns_blank():
    assert build_model_cards_clause([]) == ""
    assert build_model_cards_clause([{"id": "x"}]) == ""  # 无 name 视为空


def _facts(difficulty, models):
    return {
        "dna": {"difficulty": difficulty, "models": models},
        "mother_difficulty": difficulty,
        "qtype": "解答",
    }


def test_should_note_card_matrix():
    """注卡条件矩阵（G3）：难度≥3+非M00→注；难度<3→不注；仅M00→不注。"""
    nonm00 = [{"id": "M25", "name": "定边对定角"}]
    m00only = [dict(M00)]
    # 难度≥3 + 非M00 → 注
    assert V._should_note_card(_facts(3, nonm00)) is True
    assert V._should_note_card(_facts(4, nonm00)) is True
    # 难度<3 → 不注（即便命中非M00）
    assert V._should_note_card(_facts(2, nonm00)) is False
    assert V._should_note_card(_facts(1, nonm00)) is False
    # 难度≥3 但仅M00 → 不注（基础题兜底，无需模型卡）
    assert V._should_note_card(_facts(3, m00only)) is False
    # 难度缺失 → 不注（保守）
    assert V._should_note_card({"dna": {"models": nonm00}, "qtype": "解答"}) is False


def test_model_cards_clause_injects_when_eligible(monkeypatch):
    """难度≥3+非M00 → _model_cards_clause 反查取卡并注入（含卡片文本）。"""
    monkeypatch.setattr(
        ma, "fetch_model_cards", lambda ids: {"M25": _CARD_M25, "M32": _CARD_M32}
    )
    facts = _facts(4, [{"id": "M25", "name": "定边对定角"}, {"id": "M32", "name": "隐圆最值"}])
    clause = V._model_cards_clause(facts)
    assert "定线段对定角的动顶点" in clause
    assert "隐圆最值" in clause
    assert "反退化" in clause


def test_model_cards_clause_not_injected_when_low_difficulty(monkeypatch):
    """难度<3 → 不注卡（不调反查）。"""
    def boom(ids):
        raise AssertionError("难度<3 不该反查取卡")
    monkeypatch.setattr(ma, "fetch_model_cards", boom)
    assert V._model_cards_clause(_facts(2, [{"id": "M25", "name": "x"}])) == ""


def test_model_cards_clause_degrades_on_lookup_failure(monkeypatch):
    """取卡（反查库）故障 → 降级不注卡（不卡死出题，G5/C-010 闸门降级路径）。"""
    def boom(ids):
        raise RuntimeError("库未起")
    monkeypatch.setattr(ma, "fetch_model_cards", boom)
    assert V._model_cards_clause(_facts(4, [{"id": "M25", "name": "x"}])) == ""


def test_maybe_note_card_block_prefix(monkeypatch):
    monkeypatch.setattr(ma, "fetch_model_cards", lambda ids: {"M25": _CARD_M25})
    block = V._maybe_note_card_block(_facts(3, [{"id": "M25", "name": "x"}]))
    assert block.startswith("\n\n")  # 有卡 → 前缀换行接进 prompt
    # 不注卡 → 空串（不留空行）
    assert V._maybe_note_card_block(_facts(1, [{"id": "M25", "name": "x"}])) == ""


# ===========================================================================
# G4·W3' 守恒软警（model_anchor.model_conservation_warn + variant._model_conservation_check）
# ===========================================================================
def test_conservation_no_warn_when_subset():
    """变式 models ⊆ 母题∪{M00} → 无 ⚠。"""
    r = model_conservation_warn(["M25"], ["M25", "M32"])
    assert r["warn"] is False and r["out_of_set"] == []


def test_conservation_m00_always_allowed():
    """M00 永在守恒集合（兜底值）。"""
    r = model_conservation_warn([M00_ID], [])
    assert r["warn"] is False


def test_conservation_warn_when_out_of_set():
    """变式用母题外模型 → ⚠ + 列越界 id（不打回）。"""
    r = model_conservation_warn(["M25", "M99"], ["M25"])
    assert r["warn"] is True and r["out_of_set"] == ["M99"]


def test_conservation_candidate_pool_widens_set():
    """H2 放宽：守恒集合并入反查候选池（防欠选误报）。"""
    r = model_conservation_warn(["M32"], ["M25"], candidate_pool_ids=["M32"])
    assert r["warn"] is False  # M32 在候选池 → 不报


def test_conservation_no_warn_when_variant_has_no_models():
    """变式无独立 models（继承母题）→ 不报（不误杀）。"""
    r = model_conservation_warn([], ["M25"])
    assert r["warn"] is False


def test_variant_model_conservation_check_pure():
    """variant 层封装：item.models ⊄ 母题 → warn。"""
    item = {"models": [{"id": "M99"}]}
    facts = {"dna": {"models": [{"id": "M25"}]}}
    r = V._model_conservation_check(item, facts)
    assert r["warn"] is True and r["out_of_set"] == ["M99"]


def test_gene_gate_check_soft_warn_not_block():
    """gene_gate_check：模型守恒越界 → gene=warn + model_out_of_set 透传，**不打回**（gate 仍是 warn 非 fail）。"""
    item = {"stem": "完全不同的新题面，避免表皮抄题相似度触发其它 flag" * 2,
            "qtype": "解答", "models": [{"id": "M99"}]}
    facts = {"qtype": "解答", "stem": "母题题面", "dna": {"models": [{"id": "M25"}]}}
    g = V.gene_gate_check(item, facts)
    assert g["gate"] == V.GENE_GATE_WARN
    assert "model_conservation" in g["flags"]
    assert g.get("model_out_of_set") == ["M99"]
    # 🔴 软警不打回：gate 只到 warn，绝无 fail/剔题语义（warn = 透传不阻断）。
    assert g["gate"] != "fail"


def test_gene_gate_check_clean_when_models_inherited():
    """变式无独立 models（继承母题）→ 不报模型守恒。"""
    item = {"stem": "完全不同的新题面" * 5, "qtype": "解答"}
    facts = {"qtype": "解答", "stem": "母题题面", "dna": {"models": [{"id": "M25"}]}}
    g = V.gene_gate_check(item, facts)
    assert "model_conservation" not in g.get("flags", [])


def test_gene_one_item_records_overflow_candidate(monkeypatch, tmp_path):
    """越界 → 落待命名池一条（G4：写失败不静默，记录可审）。"""
    p = tmp_path / "model_candidates.jsonl"
    monkeypatch.setattr(ma, "_CANDIDATES_PATH", p)
    item = {"stem": "全新题面别触发表皮抄题" * 4, "qtype": "解答", "models": [{"id": "M99"}]}
    facts = {"qtype": "解答", "stem": "母题题面", "dna": {"models": [{"id": "M25"}]}}
    out = asyncio.run(V._gene_one_item(dict(item), facts, 0, 1))
    assert "model_conservation" in (out.get("gene") or {}).get("flags", [])
    import json
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert any(rec["name"] == "M99" for rec in lines)
    # 守恒软警不打回：item 仍在（不剔题），gene 仅 warn。
    assert out.get("stem")


# ===========================================================================
# G9·⑦ 反退化代码闸（math_verify.check_endpoint_degeneracy 纯代数·零 LLM）
# ===========================================================================
def test_degeneracy_endpoint_min_flagged():
    """目标 f(t)=t 在 [0,5] 求 min → 最优点 t=0（端点）→ 退化。"""
    r = mv.check_endpoint_degeneracy(
        {"kind": "endpoint_extremum", "objective": "t", "var": "t",
         "interval": ["0", "5"], "sense": "min"}
    )
    assert r["verdict"] == mv.DEGENERATE
    assert "endpoint" in r["detail"]


def test_degeneracy_endpoint_max_flagged():
    """f(t)=t 在 [0,5] 求 max → 端点 t=5 → 退化。"""
    r = mv.check_endpoint_degeneracy(
        {"kind": "endpoint_extremum", "objective": "t", "var": "t",
         "interval": ["0", "5"], "sense": "max"}
    )
    assert r["verdict"] == mv.DEGENERATE


def test_degeneracy_interior_minimum_ok():
    """f(t)=(t-2)**2 在 [0,5] 求 min → 内部驻点 t=2 → 不退化（放行）。"""
    r = mv.check_endpoint_degeneracy(
        {"kind": "endpoint_extremum", "objective": "(t-2)**2", "var": "t",
         "interval": ["0", "5"], "sense": "min"}
    )
    assert r["verdict"] == mv.DEGEN_OK


def test_degeneracy_no_payload_degrades():
    """无 endpoint_extremum 载荷 → degrade（不适用·放行，绝不误判退化）。"""
    assert mv.check_endpoint_degeneracy({"kind": "numeric"})["verdict"] == mv.DEGRADE
    assert mv.check_endpoint_degeneracy(None)["verdict"] == mv.DEGRADE


def test_degeneracy_bad_interval_degrades():
    """区间退化（lo≥hi）/缺界 → degrade（降级不卡死）。"""
    r = mv.check_endpoint_degeneracy(
        {"kind": "endpoint_extremum", "objective": "t", "var": "t",
         "interval": ["5", "5"], "sense": "min"}
    )
    assert r["verdict"] == mv.DEGRADE


def test_degeneracy_objective_without_var_degrades():
    """目标函数不含动点参数 → degrade（无所谓退化）。"""
    r = mv.check_endpoint_degeneracy(
        {"kind": "endpoint_extremum", "objective": "3", "var": "t",
         "interval": ["0", "5"], "sense": "min"}
    )
    assert r["verdict"] == mv.DEGRADE


def test_degeneracy_never_raises_on_garbage():
    """任何垃圾载荷都不抛（G5），只 degrade。"""
    for bad in [{"kind": "endpoint_extremum"}, {"kind": "endpoint_extremum", "objective": "import os"},
                {"kind": "endpoint_extremum", "objective": "t", "var": "1bad", "interval": ["0", "1"]}]:
        assert mv.check_endpoint_degeneracy(bad)["verdict"] == mv.DEGRADE


def test_degeneracy_is_zero_llm():
    """🔴 自证纯代数零 LLM：check_endpoint_degeneracy 模块无任何 LLM/网络依赖——
    monkeypatch 掉一切 LLM 入口仍能判（本测仅断言函数纯净：传入即算，不需 invoke）。"""
    # 该函数签名无 invoke/model 参数；只吃 payload dict、只用 sympy。直接断言其可独立运行。
    r = mv.check_endpoint_degeneracy(
        {"kind": "endpoint_extremum", "objective": "t", "var": "t",
         "interval": ["0", "1"], "sense": "min"}
    )
    assert r["verdict"] in (mv.DEGENERATE, mv.DEGEN_OK)
    # 🔴 自证零 LLM：① 函数签名只吃 payload（无 invoke/model/client 形参）；
    #   ② math_verify 整个模块不 import 任何 LLM/网络栈（openai/langchain/httpx）。
    import inspect
    import agents.math_verify as _mvmod
    sig = inspect.signature(_mvmod.check_endpoint_degeneracy)
    assert set(sig.parameters) == {"payload"}  # 唯一入参 = payload，无 invoke/model/client
    modsrc = inspect.getsource(_mvmod)
    for forbidden in ("import openai", "from openai", "langchain", "httpx", "AsyncOpenAI"):
        assert forbidden not in modsrc


# ===========================================================================
# ⑦ 反退化闸·pipeline 集成（variant._anti_degen_gate：退化→REGEN / 超限弃 / 降级放行）
# ===========================================================================
def _mk_facts():
    return {"kp_name": "隐圆", "grade": "九年级", "qtype": "解答",
            "dna": {"models": [{"id": "M32", "name": "隐圆最值"}], "difficulty": 4}}


def test_anti_degen_gate_passes_non_degenerate():
    """非退化（内部驻点）→ 原样放行（item, False）。"""
    item = {"stem": "x", "answer": "1", "degen_payload":
            {"kind": "endpoint_extremum", "objective": "(t-2)**2", "var": "t",
             "interval": ["0", "5"], "sense": "min"}}
    out, dropped = asyncio.run(V._anti_degen_gate(dict(item), _mk_facts(), 0, 1))
    assert dropped is False and out.get("stem") == "x"


def test_anti_degen_gate_passes_no_payload():
    """无 degen_payload → 不判退化（放行）。"""
    item = {"stem": "x", "answer": "1"}
    out, dropped = asyncio.run(V._anti_degen_gate(dict(item), _mk_facts(), 0, 1))
    assert dropped is False


def test_anti_degen_gate_regens_then_accepts(monkeypatch):
    """退化 → REGEN：重生稿非退化 + sympy PASS → 采纳（dropped=False）。"""
    degen_pl = {"kind": "endpoint_extremum", "objective": "t", "var": "t",
                "interval": ["0", "5"], "sense": "min"}  # 端点退化
    good_pl = {"kind": "endpoint_extremum", "objective": "(t-2)**2", "var": "t",
               "interval": ["0", "5"], "sense": "min"}  # 非退化

    async def fake_regen(item, facts, feedback=None):
        return {"stem": "重生非退化题", "answer": "9", "degen_payload": good_pl,
                "qtype": "解答", "difficulty": 4, "level": "hard"}

    async def fake_solve(stem):
        return {"solved_answer": "9", "solution": "解析"}

    async def fake_machine_verify(item, solved):
        return {"verdict": mv.PASS, "detail": "ok", "computed": "9"}

    monkeypatch.setattr(V, "_regen_once", fake_regen)
    monkeypatch.setattr(V, "_solve_one", fake_solve)
    monkeypatch.setattr(V, "_machine_verify", fake_machine_verify)
    monkeypatch.setattr(V, "_budget_exhausted", lambda: False)

    item = {"stem": "退化题", "answer": "1", "degen_payload": degen_pl,
            "gene": {"gate": "pass"}}
    out, dropped = asyncio.run(V._anti_degen_gate(dict(item), _mk_facts(), 0, 1))
    assert dropped is False
    assert out["stem"] == "重生非退化题"
    assert out["check"]["verify"] == V.VERIFY_SYMPY_PASS


def test_anti_degen_gate_drops_when_regen_exhausted(monkeypatch):
    """退化 → REGEN 超限仍退化 → 弃该变式（§1⑦「超限则弃」，dropped=True）。"""
    degen_pl = {"kind": "endpoint_extremum", "objective": "t", "var": "t",
                "interval": ["0", "5"], "sense": "min"}

    async def fake_regen(item, facts, feedback=None):
        return {"stem": "重生仍退化", "answer": "1", "degen_payload": degen_pl,
                "qtype": "解答", "difficulty": 4, "level": "hard"}

    monkeypatch.setattr(V, "_regen_once", fake_regen)
    monkeypatch.setattr(V, "_budget_exhausted", lambda: False)

    item = {"stem": "退化题", "answer": "1", "degen_payload": degen_pl}
    out, dropped = asyncio.run(V._anti_degen_gate(dict(item), _mk_facts(), 0, 1))
    assert dropped is True
    assert "_dropped" in out


def test_anti_degen_gate_keeps_from_edit(monkeypatch):
    """老师点名编辑题（from_edit）退化也不弃不换——老师意志优先，仅标注交人审。"""
    async def boom(*a, **k):
        raise AssertionError("from_edit 退化题不该回炉")
    monkeypatch.setattr(V, "_regen_once", boom)
    degen_pl = {"kind": "endpoint_extremum", "objective": "t", "var": "t",
                "interval": ["0", "5"], "sense": "min"}
    item = {"stem": "老师改的题", "answer": "1", "degen_payload": degen_pl, "from_edit": True}
    out, dropped = asyncio.run(V._anti_degen_gate(dict(item), _mk_facts(), 0, 1))
    assert dropped is False
    assert "反退化" in str(out.get("solution") or "")


def test_anti_degen_gate_degrades_when_uncomputable(monkeypatch):
    """反退化算不了（degrade）→ 放行不卡死（降级路径）。"""
    item = {"stem": "x", "answer": "1", "degen_payload":
            {"kind": "endpoint_extremum", "objective": "t", "var": "t",
             "interval": ["bad", "5"], "sense": "min"}}  # 区间界抽不成 → degrade
    out, dropped = asyncio.run(V._anti_degen_gate(dict(item), _mk_facts(), 0, 1))
    assert dropped is False  # degrade → 放行
