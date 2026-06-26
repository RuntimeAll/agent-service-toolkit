# -*- coding: utf-8 -*-
r"""PRD-C-103 WS1·北极星验证（AC3 / G-WS1c）：改 biz_solution_model.difficulty_tier →
同母题举一反三 → 变式 grade_observed 确定档**均值双向单调随 tier 变**。

做法（真起 graph，in-process 直驱，复用 c103_b0_regression 的母题注入/运行）：
  1. 选一道母题（默认数轴类，锚定模型可控），跑一轮，读出该母题实际锚定的模型 id（mother_dna.dna.models）。
  2. 对该锚定模型，pymysql 把 difficulty_tier 设 **基础(1)** → 重跑 R 轮 → 记变式难度均值 μ_low。
  3. 同模型 tier 设 **高阶(2)** → 重跑 R 轮 → μ_high。
  4. 还原 tier 设回 **基础(1)** → 重跑 R 轮 → μ_low2（验回落·双向）。
  断言：μ_high > μ_low 且 μ_low2 ≈ μ_low（高→均难升、改回→落）。

🔴 跑前自动备份原 tier/freq，结束**恢复原值**（探针不留改动）。
跑法：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_polaris.py --mother <qid> --rounds 3
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymysql  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from _probe_auth import real_token  # noqa: E402
from agents.variant import graph  # noqa: E402

import c103_b0_regression as base  # noqa: E402

DB = dict(host="127.0.0.1", port=3307, user="root", password="123456",
          database="ai_lesson_prep", charset="utf8mb4")

# 默认母题：数轴类（kp 100001002002，锚 DZ01/02/03/08/09/10 之一），变式难度对模型 tier 敏感。
DEFAULT_MOTHER = "2069819178373632002"


def _set_tier(model_id: str, tier: int, freq: int | None = None):
    conn = pymysql.connect(**DB)
    try:
        cur = conn.cursor()
        if freq is None:
            cur.execute("UPDATE biz_solution_model SET difficulty_tier=%s WHERE id=%s", (tier, model_id))
        else:
            cur.execute("UPDATE biz_solution_model SET difficulty_tier=%s, freq_band=%s WHERE id=%s",
                        (tier, freq, model_id))
        conn.commit()
    finally:
        conn.close()


def _get_tier(model_id: str):
    conn = pymysql.connect(**DB)
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT difficulty_tier, freq_band FROM biz_solution_model WHERE id=%s", (model_id,))
        return cur.fetchone()
    finally:
        conn.close()


async def _run_levels(app, m, token, rounds: int) -> tuple[list[int], list[str]]:
    """跑 rounds 轮举一反三，收集所有变式 difficulty（确定档）+ 锚定模型 id。"""
    levels: list[int] = []
    model_ids: set[str] = set()
    for _ in range(rounds):
        tid_suffix = base.__dict__.get("_polaris_round", 0)
        base.__dict__["_polaris_round"] = tid_suffix + 1
        cfg_tid = f"c103-polaris-{m['qid']}-{tid_suffix}"
        cfg = {"configurable": {"thread_id": cfg_tid, "ruoyi_token": token}}
        from langchain_core.messages import HumanMessage
        init = base._inject_state(m)
        init["messages"] = [HumanMessage(content="请基于已确认的母题，出一组举一反三变式题。")]
        try:
            await app.ainvoke(init, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"    轮跑错: {type(e).__name__}: {e}")
            continue
        st = app.get_state(cfg).values
        items = st.get("items") or []
        for it in items:
            d = it.get("difficulty")
            if isinstance(d, int):
                levels.append(d)
        mdna = (st.get("mother_dna") or {}).get("dna") or {}
        for mm in mdna.get("models") or []:
            mid = str(mm.get("id") or "").strip()
            if mid and mid != "M00":
                model_ids.add(mid)
    return levels, sorted(model_ids)


def _mean(xs):
    return round(statistics.mean(xs), 3) if xs else None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mother", default=DEFAULT_MOTHER)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    mothers = {m["qid"]: m for m in base._load_mothers(0)}
    m = mothers.get(args.mother)
    if not m:
        print(f"母题 {args.mother} 不在 b0 母题集，可选: {list(mothers)[:5]}...")
        return 2

    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())

    print(f"=== 北极星 AC3：母题 {m['qid']} ({m['qtype']}/{m['kp']}) rounds={args.rounds} ===", flush=True)

    # 第 0 轮探锚定模型
    print("--- 探测锚定模型（跑 1 轮）...", flush=True)
    _, mids = await _run_levels(app, m, token, 1)
    if not mids:
        print("✗ 该母题没锚到具体模型（只 M00），换一道母题再验。", flush=True)
        return 3
    target = mids[0]
    orig = _get_tier(target)
    print(f"锚定模型 = {mids}；选 {target} 做翻转。原表值 tier={orig['difficulty_tier']} freq={orig['freq_band']}", flush=True)

    try:
        # tier=基础(1) freq=低频(1)
        _set_tier(target, 1, 1)
        lo, _ = await _run_levels(app, m, token, args.rounds)
        mu_lo = _mean(lo)
        print(f"[tier=1基础] 变式难度 n={len(lo)} 均值μ_low={mu_lo}  样本={lo}", flush=True)

        # tier=高阶(2) freq=高频(2)
        _set_tier(target, 2, 2)
        hi, _ = await _run_levels(app, m, token, args.rounds)
        mu_hi = _mean(hi)
        print(f"[tier=2高阶] 变式难度 n={len(hi)} 均值μ_high={mu_hi}  样本={hi}", flush=True)

        # 改回基础(1) 低频(1) → 验回落
        _set_tier(target, 1, 1)
        lo2, _ = await _run_levels(app, m, token, args.rounds)
        mu_lo2 = _mean(lo2)
        print(f"[tier=1改回] 变式难度 n={len(lo2)} 均值μ_low2={mu_lo2}  样本={lo2}", flush=True)

    finally:
        # 🔴 恢复原表值（探针不留改动）
        _set_tier(target, orig["difficulty_tier"], orig["freq_band"])
        print(f"已恢复 {target} 原值 tier={orig['difficulty_tier']} freq={orig['freq_band']}", flush=True)

    print("\n========== 北极星判决 ==========", flush=True)
    up = (mu_hi is not None and mu_lo is not None and mu_hi > mu_lo)
    back = (mu_lo2 is not None and mu_hi is not None and mu_lo2 < mu_hi)
    print(f"高→均难升: μ_high({mu_hi}) > μ_low({mu_lo}) → {'✓' if up else '✗'}", flush=True)
    print(f"改回→落:  μ_low2({mu_lo2}) < μ_high({mu_hi}) → {'✓' if back else '✗'}", flush=True)
    green = up and back
    print(f"\nG-WS1c 北极星: {'GREEN（表反控双向单调成立）' if green else 'RED'}", flush=True)
    return 0 if green else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
