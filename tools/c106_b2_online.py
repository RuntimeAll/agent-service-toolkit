# -*- coding: utf-8 -*-
"""PRD-C-106 B2·闸3：端到端基础流程跑通 + 参照正确传到阶段二（in-process，真 LLM）。

驱动方式：用内存 checkpointer 直驱 compiled graph（真 LLM、真 RuoYi token），
构造一个「STOP1 已通过」的 pinned 母题态（mother_dna + analysis 锚定 + awaiting_mother_review），
然后 resume（start_variants=True）→ route_entry → **compress → generate → 闸A/闸B → assemble**。

为什么不走 :8093 HTTP 真贴图：贴图后是否落到 await_review(pinned) 取决于该图能否被高置信锚定
（need_anchor_review=True 的图会停在 clarify、非 await_review），那是 stage1 锚定质量的事、与 B2
压缩闸接线无关。本闸要验的是「resume → 阶段二经 compress 吃固化参照出题」这条 B2 新链路，故直接
喂一个已 pinned 的母题态，确定性地打到 compress→generate（真 LLM 出真变式）。

断言：① compress 真跑（state.mother_core_ref 被固化、incomplete=False）；
     ② 阶段二出变式 items ≥1（参照传到 generate 出了题）；③ 全程 0 error。

🔴 需 :8090 RuoYi 在跑（compose 落库）+ 真 LLM key；SSE 经 relay 须 NO_PROXY=*。
跑法：NO_PROXY=* PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c106_b2_online.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402


def _pinned_mother_state() -> dict:
    """一个 STOP1 已通过的 pinned 母题态（高置信锚定，_pin_status.pinned=True）。"""
    dna = {
        "main_kp": {"id": "K100", "name": "一元二次方程根与系数的关系"},
        "secondary_kps": [{"id": "K101", "name": "一元二次方程的解法"}],
        "qtype": "解答",
        "models": [{"id": "M12", "name": "韦达定理整体代入", "tier_int": 3, "freq_int": 2}],
        "model_flag": None,
        "skeleton": ["设两根 x1 x2", "由韦达定理 x1+x2、x1·x2", "目标式整体代入求值"],
        "scene": "纯代数",
        "difficulty": 3,
        "hard_points": ["目标式向 x1+x2 / x1x2 的整体转化"],
        "tags": ["韦达定理", "整体代入"],
        "exam_type": "计算",
    }
    return {
        "messages": [HumanMessage(content="（占位：阶段一对话原文，阶段二不应继承）")],
        "image_url": "",
        "analysis": {
            # 🔴 pinned 三锚：年级 4 位 code（3082=八下）+ main_kp 锚到真叶子 code + 高置信
            "grade": {"value": "八年级下册", "code": "3082", "confidence": 0.95},
            "kp": {
                "value": "一元二次方程根与系数的关系",
                "anchored": {"code": "K100", "name": "一元二次方程根与系数的关系"},
                "confidence": 0.92,
            },
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {
            "stem": "已知关于 x 的方程 x^2-5x+3=0 的两个实数根为 x1, x2，求 x1^2+x2^2 的值。",
            "answer": "19",
            "analysis": "由韦达定理 x1+x2=5, x1x2=3；x1^2+x2^2=(x1+x2)^2-2x1x2=25-6=19。",
            "solution_skeleton": "x1+x2=5, x1x2=3 → (x1+x2)^2-2x1x2 = 25-6 = 19",
            "difficulty": 3,
            "dna": dna,
        },
        "mother_confirmed": True,
        "awaiting_mother_review": True,  # STOP1 停态（等老师点「开始举一反三」）
        "items": [],
        "confirmed_chapter_id": "3082",
        "confirmed_chapter_name": "第2章 一元二次方程",
    }


async def run() -> bool:
    from agents.variant.graph import graph as state_graph  # the StateGraph (uncompiled)
    from agents.variant_support import RuoyiClient

    # 真 token（generate compose 阶段落库需要；解题/出题 LLM 也透 teacher_id）
    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()

    # 用内存 checkpointer 重新 compile（service 的 sqlite checkpointer 不在本进程；
    # 节点/边/路由与线上一字不差，只换 saver）。
    saver = MemorySaver()
    g = state_graph.compile(checkpointer=saver)

    tid = "c106-b2-inproc"
    cfg = RunnableConfig(configurable={
        "thread_id": tid, "ruoyi_token": token,
        "auto_verify": False,  # 与产品默认一致（手动验算，秒就绪）
    })

    # 先把 pinned 母题态种进 checkpoint（模拟 STOP1 已通过、停在 await_review）。
    await g.aupdate_state(cfg, _pinned_mother_state())

    # resume：老师点「开始举一反三」→ start_variants=True → route_entry → compress → generate
    resume_cfg = RunnableConfig(configurable={
        "thread_id": tid, "ruoyi_token": token, "auto_verify": False,
        "start_variants": True,
    })

    error = None
    try:
        result = await g.ainvoke(
            {"messages": [HumanMessage(content="开始举一反三")]},
            config=resume_cfg,
        )
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        error = str(e)[:200]
        result = {}

    # 取最终 state（含 compress 固化的 mother_core_ref）
    snap = await g.aget_state(cfg)
    vals = snap.values or {}
    ref = vals.get("mother_core_ref") or {}
    items = vals.get("items") or result.get("items") or []

    print("== 闸3·in-process e2e（resume → compress → generate） ==")
    print(f"  compress 固化 mother_core_ref: 有={bool(ref)} incomplete={ref.get('incomplete')}")
    print(f"  ref.summary = {ref.get('summary')!r}")
    print(f"  ref.dna.models = {[m.get('name') for m in (ref.get('dna') or {}).get('models') or []]}")
    print(f"  阶段二出变式 items = {len(items)} 道")
    if items:
        print(f"  变式1 stem 前 50 字 = {str((items[0] or {}).get('stem'))[:50]!r}")
    print(f"  error = {error}")

    ok = bool(ref) and ref.get("incomplete") is False and len(items) >= 1 and not error
    print(f"\n闸3 -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("===== PRD-C-106 B2 e2e 闸3（in-process · 真 LLM）=====")
    try:
        ok = asyncio.run(run())
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"e2e 失败: {e}")
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
