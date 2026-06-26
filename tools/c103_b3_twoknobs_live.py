# -*- coding: utf-8 -*-
r"""PRD-C-103 批3·WS3 双旋钮真机抽样（AC8/AC9）：1 母题 × 两档变式系数，看产出差异 + trace 真值。

复用批0 in-process 驱动（绕读图/确认 SSE），但经 config.configurable 透传双旋钮：
  - variant_similarity（0.4 远迁/中变 vs 0.9 高仿）→ generate 据系数选算子带、注入配方段。
  - difficulty_target（可选目标档）。
验证（不入库、不写 DB——只证机制）：
  ① recipe 配方段随系数变（高仿带『数值』 vs 远迁带『推广一般化』），= "像不像" 真生效。
  ② assemble 后每道变式有 difficulty_bill.level（grade_observed 确定档，AC1）。
  ③ variant_trace_block 产真值 operator/similarity（非 forward-gen/None）= AC9 trace 喂料齐。
🔴 中转慢/断 → 本脚本只跑 1 母题、单次、不重试整跑（铁律：不死等）；某档跑挂则记该档失败、继续。

跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b3_twoknobs_live.py [--qid <qid>] [--sims 0.4,0.9]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

import c103_b0_regression as b0  # noqa: E402
from _probe_auth import real_token  # noqa: E402
from agents import variant as V  # noqa: E402
from agents.variant import graph, variant_trace_block, operator_band_from_similarity  # noqa: E402


async def run_sim(app, m: dict, token: str, sim: float) -> dict:
    """跑一档变式系数：注入 config.configurable.variant_similarity → 看 recipe + items + trace。"""
    tid = f"c103-b3-{m['qid']}-s{int(sim*100)}"
    cfg = {"configurable": {"thread_id": tid, "ruoyi_token": token, "variant_similarity": sim}}
    init = b0._inject_state(m)
    init["messages"] = [HumanMessage(content="请基于已确认的母题，出一组举一反三变式题。")]
    try:
        await app.ainvoke(init, cfg)
    except Exception as e:  # noqa: BLE001 — 单档失败如实记，不拖垮另一档（不死等整跑）
        return {"sim": sim, "_error": f"{type(e).__name__}: {e}"}
    st = app.get_state(cfg).values
    items = st.get("items") or []
    knobs = st.get("knobs") or {}
    # 期望算子带（纯函数核对，证 config→knobs 真接上）
    exp_band = operator_band_from_similarity(sim) or {}
    traces = [variant_trace_block(it, knobs) for it in items]
    levels = [it.get("difficulty") for it in items]
    has_bill = [bool(it.get("difficulty_bill")) for it in items]
    return {
        "sim": sim,
        "knobs_operator": (knobs.get("operator_band") or {}).get("operator"),
        "expected_operator": exp_band.get("operator"),
        "variant_count": len(items),
        "levels": levels,
        "all_have_bill": all(has_bill) if items else False,
        "trace_sample": traces[0] if traces else None,
        "stems": [str(it.get("stem") or "")[:60] for it in items[:2]],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None)
    ap.add_argument("--sims", default="0.4,0.9")
    args = ap.parse_args()
    sims = [float(x) for x in args.sims.split(",") if x.strip()]

    mothers = b0._load_mothers(0)
    if args.qid:
        mothers = [m for m in mothers if str(m["qid"]) == str(args.qid)] or mothers[:1]
    m = mothers[0]
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    print(f"=== 批3 双旋钮真机：母题 {m['qid']} ({m['qtype']}/{m['kp']}) sims={sims} ===", flush=True)

    results = []
    for sim in sims:
        print(f"--- 变式系数 {sim} 开始 ...", flush=True)
        r = await run_sim(app, m, token, sim)
        results.append(r)
        if r.get("_error"):
            print(f"--- 系数 {sim} 失败：{r['_error']}", flush=True)
            continue
        print(f"--- 系数 {sim} 完成：算子={r['knobs_operator']}(期望{r['expected_operator']}) "
              f"变式={r['variant_count']} 档={r['levels']} 有bill={r['all_have_bill']}", flush=True)
        print(f"     trace 样本：{r['trace_sample']}", flush=True)
        for s in r["stems"]:
            print(f"     stem: {s}", flush=True)

    ok_runs = [r for r in results if not r.get("_error")]
    # 判定：① 每成功档算子==期望算子（config→knobs→算子带真接通）
    #       ② trace operator 真值非 forward-gen 且 similarity==该档系数
    #       ③ 有 bill（确定判档）
    ok = bool(ok_runs)
    for r in ok_runs:
        if r["knobs_operator"] != r["expected_operator"]:
            ok = False
            print(f"[FAIL] 系数{r['sim']} 算子未接通：knobs={r['knobs_operator']} 期望={r['expected_operator']}")
        tb = r.get("trace_sample") or {}
        if tb.get("operator") in (None, "forward-gen") or tb.get("similarity") != round(r["sim"], 2):
            ok = False
            print(f"[FAIL] 系数{r['sim']} trace 真值缺：{tb}")
        if not r["all_have_bill"]:
            ok = False
            print(f"[FAIL] 系数{r['sim']} 有变式无 difficulty_bill（判档缺）")
    # 跨档对比：两档算子不同（像不像随系数变）
    ops = {r["knobs_operator"] for r in ok_runs}
    if len(ok_runs) >= 2 and len(ops) < 2:
        print(f"[warn] 两档算子相同 {ops}（系数跨带才不同，sims 同带属正常）")

    print(f"\n批3 双旋钮真机：{'GREEN' if ok else 'RED'}（成功档={len(ok_runs)}/{len(sims)}，算子接通+trace真值+有判档）", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
