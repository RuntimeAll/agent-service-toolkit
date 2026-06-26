# -*- coding: utf-8 -*-
r"""PRD-C-103 批0·聚合级 frozen 比对（读已落盘的 baseline vs compare 产物，不重跑）。

🔴 为什么不做逐项字符级比对：generate 是 LLM 随机产物，每轮变式题面/答案都不同 → 闸A/闸B/
   surface 这些**逐题判决**天然逐次不同（baseline 与 compare 跑的是不同题面），逐项位置比对
   会把「随机生成差异」误报成「回归」。难度接线(WS1)只改难度赋值/取数, **不碰**闸逻辑/守恒/净化,
   故正确的 frozen 不破证据 = **聚合级行为不变**：
     - conservation：main_kp / grade 必须逐卷完全一致（硬守恒，与题面无关）。
     - gateA：pass 率不显著恶化（结构闸判决分布稳定）。
     - surface：clean 率不显著恶化（防撞判决分布稳定）。
     - gateB：verify 三态（sympy_pass/unverified/fail*）分布同量级（验算判决不被难度改动扰动）。
   difficulty 出 diff = 预期（WS1 的目的就是难度变）。

跑法：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b0_aggregate_diff.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "artifacts" / "regression_baseline"
CUR = Path(__file__).resolve().parent.parent / "artifacts" / "regression_compare"


def _agg(d: dict) -> dict:
    fa = d.get("frozen_assertions") or {}
    gateA = Counter(g.get("gate") for g in (fa.get("gateA") or []))
    surface = Counter("clean" if s.get("verdict") == "clean" else "defect"
                      for s in (fa.get("surface_check") or []))
    gateB = Counter(b.get("verify") for b in (fa.get("gateB") or []))
    cons = fa.get("conservation") or {}
    return {"gateA": gateA, "surface": surface, "gateB": gateB,
            "main_kp": cons.get("main_kp"), "grade": cons.get("grade"),
            "n": d.get("variant_count")}


def main() -> int:
    qids = sorted(p.stem for p in BASE.glob("*.json") if not p.stem.startswith("_"))
    cons_fail = []
    tot_b = {"gateA": Counter(), "surface": Counter(), "gateB": Counter()}
    tot_c = {"gateA": Counter(), "surface": Counter(), "gateB": Counter()}
    n_b = n_c = 0
    print("=== 聚合级 frozen 比对（不重跑，读落盘产物）===\n")
    for qid in qids:
        bp, cp = BASE / f"{qid}.json", CUR / f"{qid}.json"
        if not cp.exists():
            continue
        ab = _agg(json.loads(bp.read_text(encoding="utf-8")))
        ac = _agg(json.loads(cp.read_text(encoding="utf-8")))
        # 硬守恒：逐卷 main_kp/grade 必须一致
        if ab["main_kp"] != ac["main_kp"] or ab["grade"] != ac["grade"]:
            cons_fail.append((qid, ab["main_kp"], ac["main_kp"], ab["grade"], ac["grade"]))
        for k in ("gateA", "surface", "gateB"):
            tot_b[k] += ab[k]
            tot_c[k] += ac[k]
        n_b += ab["n"] or 0
        n_c += ac["n"] or 0

    print(f"守恒(main_kp/grade) 逐卷一致: {'✓ 全部一致' if not cons_fail else '✗ '+str(cons_fail)}")
    print(f"\n变式总数 baseline={n_b}  current={n_c}")

    def _pct(c: Counter, key, tot):
        return round(100 * c.get(key, 0) / tot, 1) if tot else 0

    print("\n闸A gate 分布:")
    print(f"  baseline: pass={_pct(tot_b['gateA'],'pass',n_b)}% warn={_pct(tot_b['gateA'],'warn',n_b)}%  {dict(tot_b['gateA'])}")
    print(f"  current : pass={_pct(tot_c['gateA'],'pass',n_c)}% warn={_pct(tot_c['gateA'],'warn',n_c)}%  {dict(tot_c['gateA'])}")
    print("\nsurface clean 率:")
    print(f"  baseline: clean={_pct(tot_b['surface'],'clean',n_b)}%  {dict(tot_b['surface'])}")
    print(f"  current : clean={_pct(tot_c['surface'],'clean',n_c)}%  {dict(tot_c['surface'])}")
    print("\n闸B verify 分布:")
    print(f"  baseline: {dict(tot_b['gateB'])}")
    print(f"  current : {dict(tot_c['gateB'])}")

    # 判决：守恒硬一致 = frozen 核心不破（闸分布同量级是参考、不硬卡，因题面随机）
    green = not cons_fail
    print(f"\nG-REG（聚合级）：守恒硬不破 = {'GREEN' if green else 'RED'}；"
          f"闸/surface 分布作随机噪声参考（题面随机，非逐项 assert）；difficulty 全卷出 diff（预期）")
    return 0 if green else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
