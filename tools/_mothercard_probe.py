# -*- coding: utf-8 -*-
"""跑 variant 图到母题卡专帧，dump header.mother_card 的 keys + stem，定位 母题入库「题面尚未产出」根因。"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.variant import variant
from agents.variant_support import RuoyiClient

variant.checkpointer = MemorySaver()
IMG = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-05-17/e3008503-956a-4134-b6ab-4e4b7f405a64/list/6/question.png"


async def drive(cfg, text, extra=None):
    """跑一轮，返回该轮所有携 mother_card 的帧 [(has_stem, keys)]。"""
    c = dict(cfg["configurable"])
    if extra:
        c.update(extra)
    frames = []
    async for mode, chunk in variant.astream(
        {"messages": [HumanMessage(content=text)]}, config={"configurable": c},
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            p = chunk[1] if isinstance(chunk, tuple) and len(chunk) == 2 else chunk
            content = getattr(p, "content", None) or (p.get("content") if isinstance(p, dict) else None)
            if isinstance(content, list):
                for cc in content:
                    if isinstance(cc, dict) and isinstance(cc.get("artifact"), dict):
                        hdr = cc["artifact"].get("header") or {}
                        if "mother_card" in hdr:
                            mc = hdr["mother_card"]
                            st = (mc or {}).get("stem") if isinstance(mc, dict) else None
                            frames.append((bool(st),
                                           "null" if not isinstance(mc, dict) else ("has_stem" if st else "NO_STEM")))
    return frames


async def main():
    client = RuoyiClient(); token = await client.login(); await client.aclose()
    cfg = {"configurable": {"thread_id": "mc-probe2", "ruoyi_token": token}}
    f1 = await drive(cfg, f"{IMG} 出2道")
    print("turn1 (出2道) 携 mother_card 帧:", f1)
    snap = await variant.aget_state(cfg)
    print("turn1 后 awaiting_mother_review:", bool((snap.values or {}).get("awaiting_mother_review")))
    f2 = await drive(cfg, "开始举一反三", extra={"start_variants": True})
    print("turn2 (开始举一反三) 携 mother_card 帧:", f2)
    # 🔴 关键：turn2 最后一个 mother_card 帧有没有 stem（FE artifact.value 整帧替换，以最后帧为准）
    snap2 = await variant.aget_state(cfg)
    md = (snap2.values or {}).get("mother_dna") or {}
    print("turn2 后 state mother_dna.stem present:", bool(md.get("stem")))


asyncio.run(main())
