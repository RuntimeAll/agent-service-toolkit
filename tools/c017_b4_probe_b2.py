# -*- coding: utf-8 -*-
r"""B4 辅助探针 b2：隔离「critical bug = classify 用 analyze 误读的 grade_code，不吃确认章前缀」。

做法：注入「analyze 在途态 + 把 grade.code 校正为确认章前缀 3082(八下)」后 resume 确认章 → classify。
若校正 grade_code 后 (b) happy-path 通（classify 定死 → generate → mother_card + items），
则证明下游（mother_card/变式/edit-dna/regen）健全，bug 单点 = grade_code 不吃确认章前缀。

再跑 (d) edit-dna(改母题 exam_type)→regen 验回流。

跑法: $env:PYTHONUTF8='1'; .venv/Scripts/python.exe tools/c017_b4_probe_b2.py
前置：:8090 RuoYi 在跑（classify 拉叶子池 + compose 不调，generate 走 relay）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _probe_auth import real_token  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import _build_mother_card, _pin_status, graph  # noqa: E402
from agents.variant import edit_dna_state, regen_dirty_items, _artifact_payload  # noqa: E402

# id 1184 纯文本（可锚 3082002 内叶子）。analyze 误读 grade=九上(人教框架)，校正为浙教 八下 3082。
IMG_1184 = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-04-25/4c6dae77-47b0-4306-9f02-c75fde151565/list/1/question.png"
CHAPTER = "3082002"
GRADE_BOOK = "3082"  # 八年级下册（确认章 4 位前缀）


async def main() -> None:
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    tid = f"c017-b4pb2-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid, "ruoyi_token": token}}

    # 轮1：传图 → needConfirm（停）
    await app.ainvoke({"messages": [HumanMessage(content=f"{IMG_1184} 帮我举一反三出3道")]}, cfg)
    st1 = app.get_state(cfg).values
    a1 = dict(st1.get("analysis") or {})
    print("轮1 analyze grade =", (a1.get("grade") or {}).get("value"),
          " awaiting =", st1.get("awaiting_mother_confirm"))

    # 🔧 模拟「fix 后行为」：把 grade.code 校正成确认章前缀（八下 3082），grade.value 也对齐。
    g = dict(a1.get("grade") or {})
    g["value"] = "八年级下学期"
    g["code"] = GRADE_BOOK
    g["confidence"] = 0.95
    a1["grade"] = g
    await app.aupdate_state(cfg, {"analysis": a1}, as_node="mother_precheck")

    # 轮2：resume 确认章 → classify（带校正后的 grade_code）
    cfg2 = {"configurable": {"thread_id": tid, "ruoyi_token": token,
                             "confirmed_chapter_id": CHAPTER,
                             "confirmed_grade_book_id": GRADE_BOOK}}
    await app.ainvoke({"messages": [HumanMessage(content="确认：八年级下册 第二章一元二次方程，继续")]}, cfg2)
    st2 = app.get_state(cfg).values
    pin = _pin_status(st2)
    print("\n轮2 classify pin =", json.dumps(pin, ensure_ascii=False))
    print("mother_confirmed =", st2.get("mother_confirmed"), " items =", len(st2.get("items") or []))
    card = _build_mother_card(st2)
    dna = (card or {}).get("dna") or {}
    print("mother_card.main_kp =", dna.get("main_kp"), " main_kp_id =", dna.get("main_kp_id"))
    print("mother_card.exam_type =", dna.get("exam_type"), " qtype =", dna.get("qtype"),
          " diff =", dna.get("difficulty"))
    print("solved_answer =", (card or {}).get("solved_answer"))
    items = st2.get("items") or []
    print("变式 items =", len(items))
    for i, it in enumerate(items, 1):
        print(f"  变式{i}: qtype={it.get('qtype')} diff={it.get('difficulty')} "
              f"check={(it.get('check') or {}).get('verdict')}")

    b_pass = bool(st2.get("mother_confirmed")) and bool(card) and len(items) > 0
    print(f"\n[(b) happy-path 校正 grade_code 后] = {'PASS' if b_pass else 'FAIL'}")

    if not b_pass:
        print("last_ai =", _last_ai(st2)[:200])
        return

    # ===== (d) edit-dna 改母题 exam_type → regen =====
    print("\n===== (d) 改母题 exam_type → regen =====")
    upd, edited, err = edit_dna_state(st2, 1, "exam_type", "证明推理")
    print("edit-dna err =", err)
    merged = {**st2, **upd}
    mdirty = bool((merged.get("mother_dna") or {}).get("dirty"))
    from agents.variant import dirty_item_indexes
    pend = dirty_item_indexes(merged.get("items") or [])
    print("after edit: mother_dirty =", mdirty, " regen_pending =", pend)
    await app.aupdate_state(cfg, upd, as_node="exec_regenerate")

    st3 = app.get_state(cfg).values
    rupd, result, rerr = await regen_dirty_items(st3, None)
    print("regen err =", rerr, " regenerated =", (result or {}).get("regenerated"),
          " failed =", (result or {}).get("failed"))
    merged3 = {**st3, **rupd}
    art = _artifact_payload(merged3)
    hdr = art.get("header") or {}
    print("after regen: mother_dirty =", hdr.get("mother_dirty"),
          " regen_pending =", hdr.get("regen_pending"), " n_items =", len(art.get("items") or []))
    d_pass = mdirty and bool((result or {}).get("regenerated"))
    print(f"\n[(d) edit-dna→regen 回流] = {'PASS' if d_pass else 'FAIL'}")


def _last_ai(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


if __name__ == "__main__":
    asyncio.run(main())
