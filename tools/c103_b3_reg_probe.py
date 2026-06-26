# -*- coding: utf-8 -*-
r"""PRD-C-103 批3·回归抽样探针：1 道母题 in-process 跑 + 与基线比对 frozen 结构不变。

批0 回归网脚本(c103_b0_regression.py)**直接覆盖** artifacts/regression_baseline/——不能用它复跑
(会毁基线)。本探针复用它的 in-process 驱动(run_one)但**写临时目录**,跑后与基线比对：
  - conservation(main_kp/grade)：母题守恒,确定性,**必等**(注入态决定,与 LLM 无关)。
  - gateA/gateB 判决分布(pass/warn/fail 计数)：闸结构,允许 ±(LLM 重生变式不同),只看「无新 fail 类」。
  - variant_count > 0：仍出题(catch import/runtime 破坏)。
  - difficulty 允许变(WS1 预期)。
🔴 frozen 的 stem_hash 本就随 LLM 重生变(run-to-run 非确定),不作 assert——本探针只证「我的批3
   改动没破坏默认路径的图执行 + 守恒」,不追字符级复现(那要固定 seed,LLM 中转做不到)。
中转断 → 单题重试由调用方控,本脚本只跑 --qid 指定的一道。

跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b3_reg_probe.py [--qid <qid>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

import c103_b0_regression as b0  # noqa: E402
from _probe_auth import real_token  # noqa: E402
from agents.variant import graph  # noqa: E402

BASE = Path(__file__).resolve().parent.parent / "artifacts" / "regression_baseline"


def _gate_dist(rows: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for r in rows or []:
        v = str(r.get(key) or "")
        out[v] = out.get(v, 0) + 1
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default=None, help="指定母题 qid（默认取基线第一道）")
    args = ap.parse_args()

    mothers = b0._load_mothers(0)
    if args.qid:
        mothers = [m for m in mothers if str(m["qid"]) == str(args.qid)] or mothers[:1]
    else:
        mothers = mothers[:1]
    m = mothers[0]
    qid = str(m["qid"])

    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    print(f"=== 批3 回归抽样：母题 {qid} ({m['qtype']}/{m['kp']}) ===", flush=True)
    cap = await b0.run_one(app, m, token)

    if cap.get("_error"):
        print(f"[RED] 图执行报错：{cap['_error']}", flush=True)
        return 1
    vc = cap.get("variant_count", 0)
    fa = cap.get("frozen_assertions") or {}
    print(f"变式数={vc} frozen类={list(fa.keys())}", flush=True)

    base_f = BASE / f"{qid}.json"
    if not base_f.exists():
        print(f"[warn] 无基线文件 {base_f.name}，仅证图能跑（变式数>0={vc>0}）", flush=True)
        return 0 if vc > 0 else 1

    base = json.loads(base_f.read_text(encoding="utf-8"))
    bfa = base.get("frozen_assertions") or {}
    ok = True

    # 1) 守恒（确定性，必等）
    bc = bfa.get("conservation") or {}
    cc = fa.get("conservation") or {}
    cons_ok = (bc.get("main_kp") == cc.get("main_kp")) and (bc.get("grade") == cc.get("grade"))
    print(f"  守恒 main_kp/grade：基线={bc.get('main_kp')}/{bc.get('grade')} "
          f"本次={cc.get('main_kp')}/{cc.get('grade')} → {'等' if cons_ok else '变!!!'}", flush=True)
    ok = ok and cons_ok

    # 2) 闸 pass/fail 分布（允许变，看无新 fail 类涌现）
    for gk in ("gateA", "gateB"):
        bd = _gate_dist(bfa.get(gk), "gate" if gk == "gateA" else "verify")
        nd = _gate_dist(fa.get(gk), "gate" if gk == "gateA" else "verify")
        new_fail = {k for k in nd if ("fail" in k.lower()) and k not in bd}
        print(f"  {gk} 分布：基线={bd} 本次={nd}{' 新fail类='+str(new_fail) if new_fail else ''}", flush=True)
        ok = ok and not new_fail

    # 3) 出题
    print(f"  variant_count>0：{vc>0}", flush=True)
    ok = ok and vc > 0

    # 4) 难度（允许变，仅打印）
    bl = [x.get("level") for x in (base.get("difficulty_snapshot") or {}).get("variant_levels", [])]
    nl = [x.get("level") for x in (cap.get("difficulty_snapshot") or {}).get("variant_levels", [])]
    print(f"  难度 level（允许变/不 assert）：基线={bl} 本次={nl}", flush=True)

    print(f"\n批3 回归抽样：{'GREEN（守恒等 + 无新 fail 类 + 仍出题）' if ok else 'RED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
