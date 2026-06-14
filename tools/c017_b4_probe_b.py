# -*- coding: utf-8 -*-
r"""B4 辅助探针：诊断 (b) happy-path —— 换一张「可锚定」纯文本母题图，看 classify 能否
定死(grade_code + main_kp 锚池内叶子)直通 generate，还是仍卡 clarify。

直连 graph（带 MemorySaver），先注入 analyze 在途态（含正确年级），再 resume 确认章 → classify。
对比「analyze 读出的 grade」vs「confirmed_chapter 推出的 grade」缺口（B4 报告 root cause 证据）。

跑法: $env:PYTHONUTF8='1'; .venv/Scripts/python.exe tools/c017_b4_probe_b.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _probe_auth import real_token  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import graph, _pin_status, _resolve_grade_code  # noqa: E402

# id 1184 纯文本：已知一元二次方程 x^2+3x-m=0 的一个根是 x=1，求 m。可锚 3082002 内叶子。
IMG_1184 = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-04-25/4c6dae77-47b0-4306-9f02-c75fde151565/list/1/question.png"
CHAPTER = "3082002"  # 浙教版 八年级下册·第二章一元二次方程
GRADE_BOOK = "3082"


async def main() -> None:
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    tid = f"c017-b4pb-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid, "ruoyi_token": token}}

    # 轮1：传图 → analyze → mother_precheck → needConfirm（停）
    out1 = {"customs": []}
    async for sm, ev in app.astream(
        {"messages": [__import__("langchain_core.messages", fromlist=["HumanMessage"]).HumanMessage(
            content=f"{IMG_1184} 帮我举一反三出3道")]},
        cfg, stream_mode=["custom"],
    ):
        if isinstance(ev, object):
            out1["customs"].append(getattr(ev, "content", None))
    st1 = app.get_state(cfg).values
    a1 = st1.get("analysis") or {}
    print("=== 轮1 后 analyze 读出 ===")
    print("grade.value =", (a1.get("grade") or {}).get("value"),
          " conf =", (a1.get("grade") or {}).get("confidence"))
    print("analyze 推 grade_code =", await _resolve_grade_code(a1))
    print("awaiting_mother_confirm =", st1.get("awaiting_mother_confirm"),
          " mother_rejected =", st1.get("mother_rejected"))

    # 轮2：resume 确认章 → classify
    cfg2 = {"configurable": {"thread_id": tid, "ruoyi_token": token,
                             "confirmed_chapter_id": CHAPTER,
                             "confirmed_grade_book_id": GRADE_BOOK}}
    from langchain_core.messages import HumanMessage
    await app.ainvoke({"messages": [HumanMessage(content="确认：八年级下册 第二章一元二次方程，继续")]}, cfg2)
    st2 = app.get_state(cfg).values
    pin = _pin_status(st2)
    mdna = st2.get("mother_dna") or {}
    dna = mdna.get("dna") or {}
    print("\n=== 轮2 后 classify 结果 ===")
    print("pin =", json.dumps(pin, ensure_ascii=False))
    print("mother_confirmed =", st2.get("mother_confirmed"))
    print("main_kp =", dna.get("main_kp"), " skeleton_lines =", len(dna.get("skeleton") or []))
    print("solved_answer =", (mdna.get("solved_answer") or "")[:60])
    print("items =", len(st2.get("items") or []))
    print("last_ai =", _last_ai(st2)[:200])


def _last_ai(state) -> str:
    from langchain_core.messages import AIMessage
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


if __name__ == "__main__":
    asyncio.run(main())
