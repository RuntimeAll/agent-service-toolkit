# -*- coding: utf-8 -*-
r"""PRD-C-103 批4·WS5/AC11「全流程逐站确认」端到端真机（1 道真母题）。

跑 1 道系统库真母题穿过 graph 端到端（in-process 直驱，绕读图/确认 SSE——读图入口 D6 已稳
单独确认），逐站 dump 产出供维护者验收：
  ① generate（出变式题面）→ 变式数 + 题面预览
  ② 闸A·基因闸（gene.gate pass/warn + flags）—— 纯代码三检，结构基因
  ③ 闸B·solve_explain（sympy 硬门**默认关**·批4）—— 验算照算产证据，但 fail/退化不剔除（人工兜底）
  ④ assemble·难度 = grade_observed 确定账单（item.difficulty_bill.level + modelHits + K/R/D）——非 LLM 自评
  ⑤ 双旋钮：variant_similarity→算子带；difficulty_target→目标档（recipe md 移）
  ⑥ 变式血缘 trace_block（operator=真算子 / similarity=真系数 / actual_level=bill.level）
🔴 难度判档只读 grade_observed（AC1 铁律），不采信 LLM 自评。中转慢/断 → 单母题单跑、不重试整跑。

跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b4_e2e_stations.py \
       [--qid <qid>] [--sim 0.9] [--difficulty keep|1..4]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

import c103_b0_regression as b0  # noqa: E402
from _probe_auth import real_token  # noqa: E402
from agents import variant as V  # noqa: E402
from agents.variant import graph, variant_trace_block, operator_band_from_similarity  # noqa: E402


def _hr(t):
    print("\n" + "=" * 8 + f" {t} " + "=" * 8, flush=True)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None)
    ap.add_argument("--sim", type=float, default=0.9)
    ap.add_argument("--difficulty", default="keep")
    args = ap.parse_args()

    mothers = b0._load_mothers(0)
    if args.qid:
        mothers = [m for m in mothers if str(m["qid"]) == str(args.qid)] or mothers[:1]
    m = mothers[0]
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())

    # 批4 默认：sympy 硬门关（验明默认值；config 不注 sympy_gate → 走 settings False）
    gate_default = V._sympy_gate_on(None)
    print(f"=== 批4 全流程逐站确认：母题 {m['qid']} ({m['qtype']}/{m['kp']}) "
          f"sim={args.sim} difficulty={args.difficulty} | sympy硬门={'开' if gate_default else '关(批4默认)'} ===",
          flush=True)

    diff_target = args.difficulty
    if diff_target not in ("keep", "", None):
        try:
            diff_target = int(diff_target)
        except ValueError:
            diff_target = "keep"

    tid = f"c103-b4-e2e-{m['qid']}"
    cfg = {"configurable": {
        "thread_id": tid, "ruoyi_token": token,
        "variant_similarity": args.sim,
        "difficulty_target": diff_target,
        # auto_verify 缺省 → True（节点级自动验算，让闸B sympy 真跑出证据；硬门仍由 settings 关）
    }}
    init = b0._inject_state(m)
    init["messages"] = [HumanMessage(content="请基于已确认的母题，出一组举一反三变式题。")]

    try:
        await app.ainvoke(init, cfg)
    except Exception as e:  # noqa: BLE001
        print(f"[RED] 图执行报错（中转可能断，单跑不重试整跑）：{type(e).__name__}: {e}", flush=True)
        return 1

    st = app.get_state(cfg).values
    items = st.get("items") or []
    knobs = st.get("knobs") or {}
    ok = True

    # ⑤ 双旋钮（先报，证 config→knobs 接通）
    _hr("站⑤ 双旋钮")
    exp_band = operator_band_from_similarity(args.sim) or {}
    kop = (knobs.get("operator_band") or {}).get("operator")
    print(f"  变式系数 sim={args.sim} → 算子带={kop}（期望={exp_band.get('operator')}）", flush=True)
    print(f"  难度轴 difficulty_target={diff_target}", flush=True)
    if kop != exp_band.get("operator"):
        ok = False
        print("  [FAIL] 算子带未接通", flush=True)

    # ① generate
    _hr("站① generate 出变式")
    print(f"  变式数={len(items)}", flush=True)
    for i, it in enumerate(items):
        print(f"  [{i}] {str(it.get('stem') or '')[:80]}", flush=True)
    if not items:
        print("  [RED] 0 变式（generate 未出题）", flush=True)
        return 1

    # ② 闸A 基因闸
    _hr("站② 闸A·基因闸（结构基因·纯代码三检）")
    for i, it in enumerate(items):
        g = it.get("gene") or {}
        print(f"  [{i}] gate={g.get('gate')} flags={g.get('flags') or []}", flush=True)

    # ③ 闸B solve_explain（sympy 硬门关）
    _hr("站③ 闸B·solve_explain（sympy 硬门关·fail/退化不剔除）")
    for i, it in enumerate(items):
        c = it.get("check") or {}
        print(f"  [{i}] verify={c.get('verify')} badge={c.get('badge')} "
              f"computed={c.get('computed')} dropped={'_dropped' in it}", flush=True)
    # 批4 断言：sympy 硬门关 → 没有任何变式被剔除（items 数 = generate 数，无 _dropped）
    dropped_n = sum(1 for it in items if it.get("_dropped"))
    if dropped_n:
        ok = False
        print(f"  [FAIL] 有 {dropped_n} 道被剔除（sympy 硬门应关·人工兜底，不该剔除）", flush=True)
    else:
        print(f"  [OK] 无变式被 sympy 剔除（{len(items)} 道全留，人工审核兜底）", flush=True)

    # ④ assemble·难度 = grade_observed
    _hr("站④ 难度 = grade_observed 确定账单（非 LLM 自评·AC1）")
    all_bill = True
    for i, it in enumerate(items):
        bill = it.get("difficulty_bill") or {}
        if not bill:
            all_bill = False
            print(f"  [{i}] [WARN] 缺 difficulty_bill", flush=True)
            continue
        mh = [(h.get("name"), h.get("tier"), h.get("freqBand")) for h in (bill.get("modelHits") or [])]
        print(f"  [{i}] level={bill.get('level')}({bill.get('levelName')}) "
              f"K/R/D={bill.get('K')}/{bill.get('R')}/{bill.get('D')} rule={bill.get('rule')} "
              f"modelHits={mh}", flush=True)
    if not all_bill:
        ok = False
        print("  [FAIL] 有变式无 grade_observed 判档账单", flush=True)

    # ⑥ trace 血缘
    _hr("站⑥ 变式血缘 trace_block（operator/similarity/actual_level 真值）")
    for i, it in enumerate(items):
        tb = variant_trace_block(it, knobs)
        print(f"  [{i}] {json.dumps(tb, ensure_ascii=False)}", flush=True)
        if tb.get("operator") in (None, "forward-gen"):
            print(f"  [{i}] [warn] operator 非真算子（可能母题无 knobs）", flush=True)

    _hr("结论")
    print(f"全流程逐站：{'GREEN（六站产出齐 + 难度=grade_observed + sympy 未剔除）' if ok else 'RED'}", flush=True)
    print("  注：读图入口(D6 已稳·线上在用)单独确认；本 e2e 走系统库真母题穿 generate→闸A→闸B→assemble。",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
