# -*- coding: utf-8 -*-
"""今晚四项整改的 API/管线冒烟（2026-06-12）。走真 LLM（relay_pool），落 data/llm_trace.jsonl。

验证：
  ① generate prompt 含「确定上下文块」(整改1)。
  ② 难度随 generate 产出、无独立 difficulty 调用 (整改2)。
  ③ 模拟「只能用XX方法解」的编辑指令走 solution_only 路径（题面不变仅解析重写）(整改3)。
跑法：.venv\\Scripts\\python.exe tools\\smoke_integ_2026_06_12.py
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

import agents.variant as v  # noqa: E402
from agents.variant_support import RuoyiClient  # noqa: E402

TRACE = Path(__file__).resolve().parents[1] / "data" / "llm_trace.jsonl"


def _trace_since(start_seq: int):
    rows = []
    if not TRACE.exists():
        return rows
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("seq", 0) > start_seq:
            rows.append(r)
    return rows


def _last_seq():
    if not TRACE.exists():
        return 0
    last = 0
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        try:
            last = json.loads(line).get("seq", last)
        except Exception:
            pass
    return last


async def main():
    # 1) 真登录拿 token（teacher_id 硬闸用）
    client = RuoyiClient()
    token = await client.login()
    await client.aclose()
    teacher_id = v.conv_trace.teacher_id_from_token(token)
    print(f"[login] teacher_id={teacher_id} token={'ok' if token else 'MISSING'}")

    cfg = {"configurable": {"thread_id": f"smoke-{int(time.time())}", "ruoyi_token": token}}

    # 2) 预置库内母题（mother_confirmed + 真锚定 DNA）→ route_entry 走 generate（跳读图）
    #    用七年级·一元一次方程当母题（leaf id 100 仅示意；走 generate 不再校验池，守恒读 DNA）
    seed = {
        "messages": [HumanMessage(content="出3道这个考点的变式题")],
        "mother_confirmed": True,
        "analysis": {
            "grade": {"value": "七年级上学期", "confidence": 0.95, "code": "3071"},
            "subject": "数学",
            "kp": {"value": "一元一次方程", "confidence": 0.95,
                   "anchored": {"id": "100", "code": "100", "name": "一元一次方程"}},
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {
            "stem": "解方程：2x + 3 = 11。",
            "answer": "x = 4",
            "solution_skeleton": "移项；合并；系数化为1",
            "difficulty": 2,
            "dna": {
                "main_kp": {"id": "100", "name": "一元一次方程"},
                "secondary_kps": [{"id": "101", "name": "移项"}],
                "exam_type": "直接计算",
                "skeleton": ["移项", "合并同类项", "系数化为1"],
                "qtype": "解答",
                "difficulty": 2,
            },
        },
    }

    seq0 = _last_seq()
    print(f"[trace] baseline seq={seq0}")

    # 3) 跑出题轮（generate→gene_gate→solve_explain→assemble）
    t0 = time.monotonic()
    out = await v.variant.ainvoke(seed, config=cfg)
    gen_ms = int((time.monotonic() - t0) * 1000)
    items = out.get("items") or []
    print(f"\n[generate round] {gen_ms} ms, items={len(items)}, "
          f"difficulties={[it.get('difficulty') for it in items]}")

    rows = _trace_since(seq0)
    labels = [r.get("label") for r in rows]
    print(f"[trace] labels this round: {labels}")
    # ① 确定上下文块在 generate prompt
    gen_rows = [r for r in rows if r.get("label") == "generate"]
    ctx_in_gen = any(
        "确定上下文" in json.dumps(r.get("request"), ensure_ascii=False) for r in gen_rows
    )
    print(f"[CHK①] generate prompt 含「确定上下文」块: {ctx_in_gen}")
    # ② 无独立 difficulty 调用（旧 _GRADE_DIFFICULTY_PROMPT「难度评定器」label 不应出现）
    diff_rows = [
        r for r in rows
        if "难度评定器" in json.dumps(r.get("request"), ensure_ascii=False)
    ]
    print(f"[CHK②] 独立难度复评调用数（应为 0）: {len(diff_rows)}")
    # dna_extract 可观测性（Q1）：本轮走 generate 不经 classify，故此轮无 dna_extract label——
    #   (Q1 的 trace 证据在「读图轮」classify 时产生；这里只验出题轮)
    for r in rows:
        print(f"   seq={r.get('seq')} label={r.get('label')} dur={r.get('duration_ms')}ms")

    # 4) solution_only 冒烟：发「只能用一元一次方程解」→ 题面不变仅解析重写
    if items:
        seq1 = _last_seq()
        state2 = dict(out)
        state2["messages"] = [HumanMessage(content="这些题只能用一元一次方程来解，别用二元方程，把解析改过来")]
        stems_before = [it.get("stem") for it in items]
        t1 = time.monotonic()
        out2 = await v.variant.ainvoke(state2, config=cfg)
        so_ms = int((time.monotonic() - t1) * 1000)
        items2 = out2.get("items") or []
        stems_after = [it.get("stem") for it in items2]
        pending = out2.get("pending")
        rows2 = _trace_since(seq1)
        labels2 = [r.get("label") for r in rows2]
        same_stems = stems_before == stems_after
        print(f"\n[solution_only round] {so_ms} ms, items={len(items2)}")
        print(f"[trace] labels this round: {labels2}")
        print(f"[CHK③] 题面保持不变: {same_stems}")
        print(f"   parse intent (pending after): {pending}")
        print(f"   stems_before={stems_before}")
        print(f"   stems_after ={stems_after}")
    else:
        print("\n[solution_only] skipped (generate 出 0 题)")

    print("\n[done] 冒烟结束。完整 trace 见 data/llm_trace.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
