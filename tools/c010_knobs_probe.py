"""PRD-C-010 首轮配方旋钮（knobs）· 真机验收探针。

重放用户原话「把这个题目给我出5个，难度递增，题型分布是2个选择题，2个填空题，1个应用题」，
graph 直驱（真 LLM + 真图 + 不经 :8093），断言配方被严格执行：

  K1 数量 = 5
  K2 题型分布 = 2 选择 + 2 填空 + 1 解答(应用题归一为解答)
  K3 难度递增（单调不减 且 末题 > 首题）
  K4 每题带 闸B verify / 闸A gene 标记（双闸对新配方仍生效）
  K5 题组头部外显「按你的要求」而非「默认配方」

跑法（toolkit 根）: $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c010_knobs_probe.py
"""

import asyncio
import sys
import uuid

sys.path.insert(0, r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\src")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import graph  # noqa: E402

IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)
INSTRUCTION = "把这个题目给我出5个，难度递增，题型分布是2个选择题，2个填空题，1个应用题"

RESULTS: list[tuple[str, bool, str]] = []


def rec(probe: str, ok: bool, detail: str) -> None:
    RESULTS.append((probe, ok, detail[:220]))
    print(f"[{'PASS' if ok else 'FAIL'}] {probe}: {detail[:200]}")


def last_ai(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


async def main() -> int:
    app = graph.compile(checkpointer=MemorySaver())
    tid = f"c010-knobs-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid}}

    await app.ainvoke(
        {"messages": [HumanMessage(content=f"{IMG}\n{INSTRUCTION}")]}, cfg
    )
    st = app.get_state(cfg).values
    items = st.get("items") or []
    reply = last_ai(st)
    knobs = st.get("knobs")
    print(f"knobs = {knobs}")
    print(f"items = {[(it.get('qtype'), it.get('difficulty'), it.get('level')) for it in items]}")

    # K1 数量
    rec("K1数量", len(items) == 5, f"出题 {len(items)} 道 (期望 5)")

    # K2 题型分布（应用题归一为解答）
    counts: dict[str, int] = {}
    for it in items:
        q = str(it.get("qtype") or "")
        key = "选择" if "选择" in q else "填空" if "填空" in q else "解答"
        counts[key] = counts.get(key, 0) + 1
    k2 = counts.get("选择") == 2 and counts.get("填空") == 2 and counts.get("解答") == 1
    rec("K2配比", k2, f"分布 {counts} (期望 选择2/填空2/解答1)")

    # K3 难度递增（单调不减 + 真有上升）
    diffs = [it.get("difficulty") for it in items]
    nums = [d for d in diffs if isinstance(d, (int, float))]
    k3 = (
        len(nums) == len(items)
        and all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))
        and (len(nums) < 2 or nums[-1] > nums[0])
    )
    rec("K3递增", k3, f"难度序列 {diffs}")

    # K4 双闸标记仍生效
    VERIFY_OK = {"sympy_pass", "unverified", "fail_after_regen"}
    GENE_OK = {"pass", "warn", "skipped"}
    k4_parts = []
    k4 = True
    for i, it in enumerate(items, 1):
        ck = it.get("check") or {}
        ok = (ck.get("verify") in VERIFY_OK) or (ck.get("review") == "proof_needs_human")
        gk = (it.get("gene") or {}).get("gate") in GENE_OK
        k4 = k4 and ok and gk
        k4_parts.append(f"#{i}:v={ck.get('verify') or ck.get('review')},g={(it.get('gene') or {}).get('gate')}")
    rec("K4双闸", k4, " ".join(k4_parts))

    # K5 头部外显按要求（不再是"默认配方"）
    k5 = ("按你的要求" in reply) or ("默认" not in reply.split("\n")[0] and "5 道" in reply)
    rec("K5外显", k5, reply.split("\n")[0][:120])

    print("\n" + "=" * 60)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"SUMMARY: {npass}/{len(RESULTS)} PASS")
    for probe, ok, det in RESULTS:
        if not ok:
            print(f"  FAIL {probe}: {det}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
