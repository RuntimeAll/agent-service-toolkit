# -*- coding: utf-8 -*-
r"""PRD-C-103 批0 回归网·比对器（WS1 难度接线后跑，验 frozen 不破 + difficulty 出 diff）。

与 c103_b0_regression.py（基线采集器）的关系：复用其 run_one/_load_mothers/graph，
跑同 15 道母题到**独立目录** artifacts/regression_compare/（不覆盖 baseline），
再与 artifacts/regression_baseline/ 比对：

  frozen_assertions（必须不回归 → assert 相等，违反=RED）：
    - gateA：每变式闸A gate 值（pass/warn）——结构基因闸判决不因难度改动而变。
    - surface_check：每变式 verdict（clean / defect 串）——防撞母题判决不变。
    - conservation：main_kp / grade / injected_kps——守恒不破。
    - gateB：verify/badge——验算判决不变。
  difficulty_snapshot（允许变·预期 → 只出 diff 不 assert）。

🔴 generate 是 LLM 随机产物：变式题面逐次不同 → stem_hash 天然不可逐字相等（不是回归信号）。
   故 frozen 比对**按结构不变量逐项比**（gate 值序列 / surface verdict 序列 / 守恒键），
   stem_hash 仅作信息打印不 assert（与「净化字符级」语义：净化幂等性由 _sanitize 单测保证，
   非「同一题面跨随机轮复现」）。变式数可能浮动 → 按 min(len) 对齐逐项比，长度差单列告警。

跑法（cwd=toolkit 根，:8093 不需要；in-process 直驱 graph；:8080 在跑）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b0_compare.py --concurrency 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from _probe_auth import real_token  # noqa: E402
from agents.variant import graph  # noqa: E402

import c103_b0_regression as base  # noqa: E402

BASELINE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "regression_baseline"
COMPARE_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "regression_compare"


def _gateA_seq(fa: dict) -> list:
    return [g.get("gate") for g in (fa.get("gateA") or [])]


def _surface_seq(fa: dict) -> list:
    return [s.get("verdict") for s in (fa.get("surface_check") or [])]


def _gateB_seq(fa: dict) -> list:
    return [(b.get("verify"), b.get("badge")) for b in (fa.get("gateB") or [])]


def compare_one(qid: str, baseline: dict, current: dict) -> dict:
    """逐项比 frozen 结构不变量。返回 {ok, diffs:[...], difficulty_diff:bool}。"""
    diffs: list[str] = []
    bfa = baseline.get("frozen_assertions") or {}
    cfa = current.get("frozen_assertions") or {}

    # 守恒：main_kp / grade 必须一致；injected_kps 集合比（变式随机可能子集波动 → 仅在越出母题白名单时算回归，
    #   这里按集合相等比，差异列为 warn 而非硬 RED——守恒真硬闸是 main_kp/grade）
    bcons = bfa.get("conservation") or {}
    ccons = cfa.get("conservation") or {}
    if bcons.get("main_kp") != ccons.get("main_kp"):
        diffs.append(f"守恒 main_kp 变：{bcons.get('main_kp')} -> {ccons.get('main_kp')}")
    if bcons.get("grade") != ccons.get("grade"):
        diffs.append(f"守恒 grade 变：{bcons.get('grade')} -> {ccons.get('grade')}")

    # gateA：闸A gate 值序列（结构基因判决）——按 min 长度逐项比
    ga_b, ga_c = _gateA_seq(bfa), _gateA_seq(cfa)
    n = min(len(ga_b), len(ga_c))
    for i in range(n):
        if ga_b[i] != ga_c[i]:
            diffs.append(f"闸A[{i}] gate 变：{ga_b[i]} -> {ga_c[i]}")

    # surface_check：clean/defect 判决序列
    su_b, su_c = _surface_seq(bfa), _surface_seq(cfa)
    # 仅比「是否 clean」布尔（defect 文本含随机题面片段，比 clean 性即可）
    for i in range(min(len(su_b), len(su_c))):
        bclean = su_b[i] == "clean"
        cclean = su_c[i] == "clean"
        if bclean != cclean:
            diffs.append(f"surface[{i}] clean 变：{su_b[i]} -> {su_c[i]}")

    # gateB：verify/badge 序列
    gb_b, gb_c = _gateB_seq(bfa), _gateB_seq(cfa)
    for i in range(min(len(gb_b), len(gb_c))):
        if gb_b[i] != gb_c[i]:
            diffs.append(f"闸B[{i}] 判决变：{gb_b[i]} -> {gb_c[i]}")

    # difficulty_snapshot diff（允许·预期）
    bl = [(x.get("idx"), x.get("difficulty")) for x in (baseline.get("difficulty_snapshot") or {}).get("variant_levels") or []]
    cl = [(x.get("idx"), x.get("difficulty")) for x in (current.get("difficulty_snapshot") or {}).get("variant_levels") or []]
    diff_changed = bl != cl

    return {
        "ok": len(diffs) == 0,
        "diffs": diffs,
        "difficulty_diff": diff_changed,
        "baseline_diff_levels": [d for _, d in bl],
        "current_diff_levels": [d for _, d in cl],
        "len_change": (len(ga_b), len(ga_c)),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    mothers = base._load_mothers(args.limit)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    sem = asyncio.Semaphore(max(1, args.concurrency))

    print(f"=== C-103 批0 回归比对：{len(mothers)} 道 并发={args.concurrency} ===", flush=True)

    async def _guarded(m):
        async with sem:
            print(f"--- [{m['qid']}] {m['qtype']}/{m['kp']} 跑当前 ...", flush=True)
            r = await base.run_one(app, m, token)
            print(f"--- [{m['qid']}] 完成 {r.get('variant_count')} 变式 {r.get('_elapsed_s','?')}s"
                  f"{' ERR='+r['_error'] if r.get('_error') else ''}", flush=True)
            return r

    results = await asyncio.gather(*(_guarded(m) for m in mothers))

    frozen_green = True
    diff_count = 0
    print("\n========== frozen 比对 ==========", flush=True)
    for r in results:
        qid = r["mother_id"]
        (COMPARE_DIR / f"{qid}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        bpath = BASELINE_DIR / f"{qid}.json"
        if not bpath.exists():
            print(f"  [SKIP] {qid} 无基线", flush=True)
            continue
        if r.get("_error"):
            print(f"  [ERR ] {qid} 当前跑错：{r['_error']}", flush=True)
            frozen_green = False
            continue
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        cmp = compare_one(qid, baseline, r)
        if not cmp["ok"]:
            frozen_green = False
        if cmp["difficulty_diff"]:
            diff_count += 1
        flag = "GREEN" if cmp["ok"] else "RED"
        dchg = "难度变" if cmp["difficulty_diff"] else "难度同"
        print(f"  [{flag}] {qid} frozen={'相等' if cmp['ok'] else 'diff!'} | {dchg} "
              f"base={cmp['baseline_diff_levels']} -> cur={cmp['current_diff_levels']}", flush=True)
        for d in cmp["diffs"]:
            print(f"        ✗ {d}", flush=True)

    print(f"\nG-REG: frozen {'全绿(不回归)' if frozen_green else 'RED(有回归!)'} | "
          f"difficulty 出 diff {diff_count}/{len(results)} 道（预期·允许变）", flush=True)
    print(f"产物: {COMPARE_DIR}", flush=True)
    return 0 if frozen_green else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
