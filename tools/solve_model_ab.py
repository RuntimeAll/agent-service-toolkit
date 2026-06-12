"""按环节分档模型路由·SOLVE 档数据定档对照（闸B 独立重解 + 载荷抽取降 nano 是否安全）。

同一批母题（c018 MOTHERS 子集，stem/answer/qtype 已知），分别用 nano-solve 与 5.4-solve 跑
闸B 链路（_solve_one → _machine_verify(=_extract_payload + sympy verify)），比对 sympy 判决一致率
+ ⚠warn(degrade) 率。判决永远只读 sympy verdict（铁律），本脚本零裁决，只统计两档差异。

裁决口径（任务）：一致率 ≥90% 且 warn 不明显升 → SOLVE 默认 nano；否则保持 5.4。

跑法（cwd = toolkit 根；RuoYi 不必在跑，本脚本不入库不查树）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/solve_model_ab.py
  --limit N   只跑前 N 道
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.settings import settings  # noqa: E402

from c018_benchmark_replay import MOTHERS  # noqa: E402

import agents.variant as V  # noqa: E402


async def solve_and_verify(item: dict, model: str) -> dict:
    """对单题强制 SOLVE 档 = model 跑闸B 重解 + 程序验算，返回 verdict。"""
    # 临时覆盖 SOLVE 档（VARIANT_MODEL_SOLVE 直接置 model，绕过缺省回退）
    old = settings.VARIANT_MODEL_SOLVE
    settings.VARIANT_MODEL_SOLVE = model
    try:
        t0 = time.monotonic()
        solved = await V._solve_one(item.get("stem", ""))
        res = await V._machine_verify(
            {"stem": item.get("stem"), "answer": item.get("answer"), "qtype": item.get("qtype")},
            solved.get("solved_answer"),
        )
        return {
            "verdict": res.get("verdict"),
            "solved_answer": (solved.get("solved_answer") or "")[:80],
            "detail": (res.get("detail") or "")[:80],
            "dur_s": round(time.monotonic() - t0, 1),
        }
    finally:
        settings.VARIANT_MODEL_SOLVE = old


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    # 取「可进 sympy」的母题（证明/作图类不进 sympy，对照无意义 → 跳过）
    mothers = [m for m in MOTHERS if m.get("qtype") not in ("证明", "作图")]
    if args.limit:
        mothers = mothers[: args.limit]

    rows = []
    agree = 0
    nano_warn = 0
    f54_warn = 0
    for m in mothers:
        nano = await solve_and_verify(m, "gpt-5-nano")
        f54 = await solve_and_verify(m, "gpt-5.4")
        same = nano["verdict"] == f54["verdict"]
        agree += int(same)
        nano_warn += int(nano["verdict"] == V.math_verify.DEGRADE)
        f54_warn += int(f54["verdict"] == V.math_verify.DEGRADE)
        rows.append(
            {
                "stem": (m.get("stem") or "")[:50],
                "nano": nano["verdict"],
                "5.4": f54["verdict"],
                "agree": same,
                "nano_s": nano["dur_s"],
                "54_s": f54["dur_s"],
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False))

    n = len(rows)
    print("=" * 60)
    print(f"题数: {n}")
    print(f"判决一致: {agree}/{n} = {agree / n * 100:.1f}%（阈值 ≥90%）")
    print(f"nano degrade(⚠): {nano_warn}/{n}  vs  5.4 degrade: {f54_warn}/{n}")
    verdict = "SOLVE → nano（一致率达标且 warn 未明显升）" if (
        agree / n >= 0.90 and nano_warn <= f54_warn + 1
    ) else "SOLVE 保持 5.4（一致率不足或 warn 升）"
    print(f"裁决: {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
