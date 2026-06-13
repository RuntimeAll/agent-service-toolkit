# -*- coding: utf-8 -*-
"""PRD-C-015 批3 真机冒烟（trace 取证）：注卡 / 软警 / 反退化闸 五项。

跑法（cwd = toolkit）：.venv\\Scripts\\python.exe tools\\batch3_smoke.py
依赖：MySQL :3307（fetch_model_cards 真反查词库表）。不依赖 :8090（五项都在落库之前）。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agents.variant as V  # noqa: E402
import agents.model_anchor as ma  # noqa: E402
import agents.math_verify as mv  # noqa: E402


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    # ① 难度≥3 母题 → generate prompt 含模型卡片 + 反退化约束（真反查词库表 M25/M32）
    banner("① 难度≥3 母题 → 注卡（模型卡片 + 反退化约束）·真反查词库表")
    facts_hard = {
        "dna": {"difficulty": 4, "models": [{"id": "M25", "name": "定边对定角"},
                                            {"id": "M32", "name": "隐圆最值"}]},
        "mother_difficulty": 4, "qtype": "解答",
    }
    block = V._maybe_note_card_block(facts_hard)
    print("注卡块长度:", len(block))
    print(block[:1200])
    assert block, "难度≥3+非M00 应注卡"
    assert "反退化" in block and "区间端点" in block, "应含反退化约束"
    assert "degen_payload" in block, "应含反退化载荷契约"
    print(">>> ① PASS：注卡含模型卡片（逐字反查）+ 反退化约束 + degen 载荷契约")

    # ② 难度<3 → 不注卡
    banner("② 难度<3 母题 → 不注卡")
    facts_easy = {"dna": {"difficulty": 2, "models": [{"id": "M25", "name": "定边对定角"}]},
                  "mother_difficulty": 2, "qtype": "解答"}
    block2 = V._maybe_note_card_block(facts_easy)
    print("注卡块:", repr(block2))
    assert block2 == "", "难度<3 不该注卡"
    # 仅 M00 也不注卡
    facts_m00 = {"dna": {"difficulty": 4, "models": [dict(ma.M00)]}, "mother_difficulty": 4, "qtype": "解答"}
    assert V._maybe_note_card_block(facts_m00) == "", "仅 M00 不该注卡"
    print(">>> ② PASS：难度<3 不注卡；仅 M00 也不注卡")

    # ③ 变式 models 出母题集 → ⚠ 软警不打回（题仍出）
    banner("③ 变式 models ⊄ 母题集 → ⚠ 软警·不打回（gene=warn，题保留）")
    item = {"stem": "一道全新的隐圆变式题面，数字场景全换，避免触发表皮抄题闸" * 2,
            "qtype": "解答", "models": [{"id": "M99池外技巧"}]}
    facts_m = {"qtype": "解答", "stem": "母题题面", "dna": {"models": [{"id": "M25"}]}}
    out = asyncio.run(V._gene_one_item(dict(item), facts_m, 0, 1))
    print("gene:", out.get("gene"))
    assert (out.get("gene") or {}).get("gate") == V.GENE_GATE_WARN, "应打 warn"
    assert "model_conservation" in (out.get("gene") or {}).get("flags", [])
    assert out.get("gene", {}).get("model_out_of_set") == ["M99池外技巧"]
    assert out.get("stem"), "软警不打回：题必须仍在（不剔除）"
    print(">>> ③ PASS：模型守恒越界 → ⚠ 软警 + 待命名池记录；题保留（不打回）")

    # ④ 构造退化构型变式 → 反退化闸判废 → REGEN（mock 重生稿非退化 + PASS → 采纳）
    banner("④ 退化构型变式 → 反退化闸判废 → REGEN")
    degen_pl = {"kind": "endpoint_extremum", "objective": "t", "var": "t",
                "interval": ["0", "5"], "sense": "min"}  # 最优点 t=0 落端点 = 退化
    direct = mv.check_endpoint_degeneracy(degen_pl)
    print("纯代数判退化:", direct)
    assert direct["verdict"] == mv.DEGENERATE, "端点退化应判 DEGENERATE"

    good_pl = {"kind": "endpoint_extremum", "objective": "(t-2)**2", "var": "t",
               "interval": ["0", "5"], "sense": "min"}

    async def fake_regen(it, fa, feedback=None):
        print("  [REGEN 触发] feedback 前 80 字:", str(feedback)[:80])
        return {"stem": "重生·最优点落内部驻点 t=2", "answer": "9", "degen_payload": good_pl,
                "qtype": "解答", "difficulty": 4, "level": "hard"}

    async def fake_solve(stem):
        return {"solved_answer": "9", "solution": "解析"}

    async def fake_mv(it, solved):
        return {"verdict": mv.PASS, "detail": "ok", "computed": "9"}

    V._regen_once = fake_regen
    V._solve_one = fake_solve
    V._machine_verify = fake_mv
    V._budget_exhausted = lambda: False
    facts_d = {"kp_name": "隐圆", "grade": "九年级", "qtype": "解答",
               "dna": {"models": [{"id": "M32"}], "difficulty": 4}}
    item_d = {"stem": "退化题", "answer": "1", "degen_payload": degen_pl, "gene": {"gate": "pass"}}
    res_item, dropped = asyncio.run(V._anti_degen_gate(dict(item_d), facts_d, 0, 1))
    print("反退化闸后题面:", res_item.get("stem"), "| dropped:", dropped)
    assert dropped is False and res_item["stem"] == "重生·最优点落内部驻点 t=2"
    print(">>> ④ PASS：退化→REGEN→采纳非退化重生稿（最优点落内部驻点）")

    # ④b 超限仍退化 → 弃该变式
    banner("④b 超限仍退化 → 弃该变式（§1⑦「超限则弃」）")

    async def regen_still_degen(it, fa, feedback=None):
        return {"stem": "重生仍退化", "answer": "1", "degen_payload": degen_pl,
                "qtype": "解答", "difficulty": 4, "level": "hard"}

    V._regen_once = regen_still_degen
    item_d2 = {"stem": "退化题2", "answer": "1", "degen_payload": degen_pl}
    res2, dropped2 = asyncio.run(V._anti_degen_gate(dict(item_d2), facts_d, 0, 1))
    print("dropped:", dropped2, "| _dropped:", res2.get("_dropped"))
    assert dropped2 is True
    print(">>> ④b PASS：REGEN 超限仍退化 → 弃该变式")

    # ⑤ 反退化算不了 → 降级 ⚠ 继续不卡死
    banner("⑤ 反退化算不了 → 降级放行（不卡死）")
    bad_pl = {"kind": "endpoint_extremum", "objective": "t", "var": "t",
              "interval": ["bad", "5"], "sense": "min"}  # 区间界抽不成
    direct_bad = mv.check_endpoint_degeneracy(bad_pl)
    print("不可判:", direct_bad)
    assert direct_bad["verdict"] == mv.DEGRADE
    item_bad = {"stem": "x", "answer": "1", "degen_payload": bad_pl}
    res_bad, dropped_bad = asyncio.run(V._anti_degen_gate(dict(item_bad), facts_d, 0, 1))
    assert dropped_bad is False, "算不了应降级放行不剔题"
    print(">>> ⑤ PASS：反退化算不了 → degrade → 放行（降级路径，不卡死）")

    banner("批3 五项冒烟全 PASS")


if __name__ == "__main__":
    main()
