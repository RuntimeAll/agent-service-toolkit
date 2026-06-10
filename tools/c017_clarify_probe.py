"""17 号修复任务 · 真机验收探针：clarify 在途 → 老师纯文字答年级 → 必须继续出题。

复现 17 号 §1 现象的 state（有 mother_dna/analysis、grade 低置信、未确认、items 空 =
停在 clarify 等老师回答），注入 checkpointer 后发「这个是9年级上的题目」（纯文字无 URL），
真 LLM 走 parse(修正) → patch(改年级) → classify(重锚) → generate → 双闸 → assemble。

断言：
  C1 不掉催图（修复前回「我还没看到题目图…」）
  C2 年级被改为九年级（patch 生效）
  C3 真出了题（items 非空 → 多轮断裂愈合）

跑法: $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c017_clarify_probe.py
前置: RuoYi :8090（classify 重锚要查知识点树）+ lk888 可达。
"""

import asyncio
import sys
import uuid

from _probe_auth import real_token

sys.path.insert(0, r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\src")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import graph  # noqa: E402

# 17 号 §2 的 state 特征：mother_dna + analysis 在、grade 低置信、未确认、无题组
CLARIFY_STATE = {
    "image_url": "https://example.com/mother.png",
    "analysis": {
        "grade": {"value": "七年级下学期", "confidence": 0.4},
        "kp": {"value": "一元二次方程求根", "confidence": 0.92},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "解方程：$x^2-3x+2=0$",
        "answer": "$x_1=1, x_2=2$",
        "solution_skeleton": "因式分解 $(x-1)(x-2)=0$，得 $x_1=1, x_2=2$",
        "difficulty": 3,
    },
    "mother_confirmed": False,
    "items": [],
}


def last_ai(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


async def main() -> int:
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    tid = f"c017-clarify-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid, "ruoyi_token": token}}

    # 注入「停在 clarify 等回答」的在途 state（连同 clarify 的提问消息，贴近真实历史）
    await app.aupdate_state(
        cfg,
        {
            **CLARIFY_STATE,
            "messages": [
                HumanMessage(content="https://example.com/mother.png 帮我举一反三"),
                AIMessage(content="我先确认母题 DNA：年级我拿不准（看着像七年级下学期），是几年级上/下？"),
            ],
        },
        as_node="analyze",
    )

    # 老师纯文字回答澄清（无 URL）—— 17 号修复前这里掉催图
    await app.ainvoke({"messages": [HumanMessage(content="这个是9年级上的题目")]}, cfg)
    st = app.get_state(cfg).values
    reply = last_ai(st)
    items = st.get("items") or []
    grade = ((st.get("analysis") or {}).get("grade") or {}).get("value") or ""

    ok = True
    c1 = "还没看到题目图" not in reply and "贴一张题目图" not in reply
    print(f"[{'PASS' if c1 else 'FAIL'}] C1 不掉催图: reply={reply[:80]}")
    c2 = "九年级" in grade
    print(f"[{'PASS' if c2 else 'FAIL'}] C2 年级已修正: grade={grade}")
    c3 = len(items) > 0
    print(f"[{'PASS' if c3 else 'FAIL'}] C3 继续出题: items={len(items)} 道")
    ok = c1 and c2 and c3
    print("OK" if ok else "C017 FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
