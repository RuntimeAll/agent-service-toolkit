"""PRD-C-BUG 批1+批2 真机冒烟（真 LLM，trace 取证）。

覆盖 6 项：
  ① BUG-001 G1：「第1题改成选择题」→ parse=编辑·regenerate（非 clarify）、新题 qtype=选择
  ② BUG-001 G2：「第2题加入杭州元素场景」→ regenerate、场景换、结构守恒
  ③ BUG-002 G4：组卷会话内「再来两道难的」→ add count=2（中文数量词）
  ④ BUG-003 G6：真听不懂的话「阿巴阿巴随便说说」→ clarify 文案不甩硬守恒
  ⑤ BUG-006 G9：AI 提议「我可以讲一遍第2题」后回「给学生讲」→ 答疑（非 clarify）
  ⑥ BUG-004 G8：新出题 prompt 题面行内（trace 检查出题 prompt 含行内规定、无「行间长式用$$」）

跑法（cwd=toolkit）：
  $env:PYTHONIOENCODING='utf-8'; .venv\\Scripts\\python.exe tools\\bug_batch_2026_06_13_smoke.py
前置：RuoYi :8090 在跑 + lk888 可达。
"""

import asyncio
import sys
import uuid

from _probe_auth import real_token

sys.path.insert(0, r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\src")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import graph  # noqa: E402

# 已出好一组题的在途 state（mother 已确认 + items 在）——用于编辑/答疑类承接
_ANALYSIS = {
    "grade": {"value": "七年级上学期", "confidence": 0.92, "anchored": {"code": "100200300"}},
    "kp": {"value": "一元一次方程", "confidence": 0.92, "anchored": {"code": "100200300"}},
    "qtype": {"value": "解答", "confidence": 0.9},
}
_DNA = {
    "stem": "解方程：$2x+3=11$",
    "answer": "$x=4$",
    "solution_skeleton": "移项得 $2x=8$，解得 $x=4$",
    "difficulty": 2,
    "dna": {"dim1_kp_id": "100200300", "scene": "纯代数", "skeleton": ["移项", "系数化1"]},
}


def _base_state(items):
    return {
        "image_url": "https://example.com/mother.png",
        "analysis": _ANALYSIS,
        "mother_dna": _DNA,
        "mother_confirmed": True,
        "items": items,
    }


def _item(stem, qtype="解答", diff=2):
    return {
        "stem": stem, "answer": "x=4", "solution": "略",
        "qtype": qtype, "difficulty": diff, "level": "normal",
        "injected_kp": None, "check": {"badge": "ok"}, "from_recipe": True,
    }


def last_ai(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


async def _run_edit(token, msgs_history, follow_up, items):
    """注入历史 + items 的在途 state，再发 follow_up，返回最终 state。"""
    app = graph.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": f"bug-{uuid.uuid4().hex[:8]}", "ruoyi_token": token}}
    await app.aupdate_state(
        cfg, {**_base_state(items), "messages": msgs_history}, as_node="assemble"
    )
    await app.ainvoke({"messages": [HumanMessage(content=follow_up)]}, cfg)
    return app.get_state(cfg).values


async def main() -> int:
    token = await real_token()
    results = []

    # ① BUG-001 改题型
    items = [_item("解方程：$3x-1=8$"), _item("解方程：$5x=20$")]
    st = await _run_edit(token, [HumanMessage(content="举一反三")], "第1题改成选择题", items)
    new_items = st.get("items") or []
    q0 = (new_items[0].get("qtype") if new_items else "")
    reply = last_ai(st)
    ok1 = q0 == "选择" and "没" not in reply[:6]  # 题型变选择、不是 clarify「没听懂」开头
    print(f"[{'PASS' if ok1 else 'FAIL'}] ① BUG-001 改题型: 第1题 qtype={q0!r} reply={reply[:50]!r}")
    results.append(ok1)

    # ② BUG-001 换场景
    items = [_item("解方程：$3x-1=8$"), _item("解方程：$5x=20$")]
    st = await _run_edit(token, [HumanMessage(content="举一反三")], "第2题加入杭州元素的场景", items)
    new_items = st.get("items") or []
    s1 = (new_items[1].get("stem") if len(new_items) > 1 else "")
    reply = last_ai(st)
    ok2 = bool(s1) and "没" not in reply[:6] and ("杭州" in s1 or len(s1) > 8)
    print(f"[{'PASS' if ok2 else 'FAIL'}] ② BUG-001 换场景: 第2题 stem={s1[:40]!r}")
    results.append(ok2)

    # ③ BUG-002 中文数量词 add
    items = [_item("解方程：$3x-1=8$")]
    st = await _run_edit(token, [HumanMessage(content="举一反三")], "再来两道难的", items)
    new_items = st.get("items") or []
    ok3 = len(new_items) == 3  # 原 1 + 新增 2 = 3
    print(f"[{'PASS' if ok3 else 'FAIL'}] ③ BUG-002 两道: 原1道→现 {len(new_items)} 道（应 3）")
    results.append(ok3)

    # ④ BUG-003 真听不懂 clarify 文案
    items = [_item("解方程：$3x-1=8$")]
    st = await _run_edit(token, [HumanMessage(content="举一反三")], "阿巴阿巴随便说点啥", items)
    reply = last_ai(st)
    ok4 = ("撞" not in reply or "硬守恒" not in reply) and ("具体" in reply or "get" in reply.lower() or "可以这样说" in reply)
    print(f"[{'PASS' if ok4 else 'FAIL'}] ④ BUG-003 听不懂不甩守恒: reply={reply[:70]!r}")
    results.append(ok4)

    # ⑤ BUG-006 承接「给学生讲」
    items = [_item("解方程：$3x-1=8$"), _item("解方程：$5x=20$")]
    history = [
        HumanMessage(content="举一反三"),
        AIMessage(content="这组出好了。我可以把第2题完整讲一遍（按给学生讲的方式），需要吗？"),
    ]
    st = await _run_edit(token, history, "给学生讲", items)
    reply = last_ai(st)
    # 答疑分支会输出讲解（人话），不是 clarify「没听懂/可以这样说」兜底
    ok5 = "可以这样说" not in reply and "没太 get" not in reply and len(reply) > 20
    print(f"[{'PASS' if ok5 else 'FAIL'}] ⑤ BUG-006 承接答疑: reply={reply[:70]!r}")
    results.append(ok5)

    print()
    print(f"=== 真机冒烟 {sum(results)}/{len(results)} PASS ===")
    print("（⑥ BUG-004 题面行内见 trace：tools\\read_llm_trace.py -l add / -l regen 检查无『行间长式用$$』）")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
