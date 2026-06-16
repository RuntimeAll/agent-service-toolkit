# -*- coding: utf-8 -*-
"""PRD-C-100 B1a 真机 e2e：塌缩入口 in-process 跑图，验 G1/G3。

验：① 新图入口走 mother_opus_entry（不走 analyze/mother_precheck）；② 母题卡帧先出（G3）；
    ③ 高置信 → await_review 硬停 / 低置信 → needConfirm（D1/D3）。
需 RuoYi :8090 + MySQL :3307 在跑（leaf_pool）。真调 opus（~35-180s/张）。

跑法：$env:PYTHONUTF8=1; .venv/Scripts/python.exe tools/c100_b1a_e2e.py [--url <图>]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import variant  # noqa: E402
from agents.variant_support import RuoyiClient  # noqa: E402

DEFAULT_URL = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    client = RuoyiClient()
    token = await client.login()
    await client.aclose()
    print(f"[token ok: {token[:12]}…]")

    state = {"messages": [HumanMessage(content=f"{args.url} 出3道")]}
    config = {"configurable": {"thread_id": "c100-b1a-e2e", "ruoyi_token": token}}

    nodes_ran: list[str] = []
    custom_keys: list[str] = []
    mother_card_seen = False
    need_confirm_seen = False
    first_card_idx = None
    idx = 0

    async for mode, chunk in variant.astream(
        state, config=config, stream_mode=["updates", "custom"]
    ):
        idx += 1
        if mode == "updates" and isinstance(chunk, dict):
            for node in chunk.keys():
                nodes_ran.append(node)
        elif mode == "custom":
            payload = chunk[1] if isinstance(chunk, tuple) and len(chunk) == 2 else chunk
            content = getattr(payload, "content", None) or (payload.get("content") if isinstance(payload, dict) else None)
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    for k in c.keys():
                        custom_keys.append(k)
                        if k == "artifact" and isinstance(c[k], dict):
                            hdr = (c[k].get("header") or {})
                            if "mother_card" in hdr:
                                mother_card_seen = True
                                if first_card_idx is None:
                                    first_card_idx = idx
                        if k == "needConfirm":
                            need_confirm_seen = True
                        if k in ("error", "stage"):
                            print(f"  [{k}] {c[k]}")

    print("\n=== 结果 ===")
    print("nodes_ran:", nodes_ran)
    print("custom_keys:", custom_keys)
    print("mother_card_seen:", mother_card_seen, "| need_confirm_seen:", need_confirm_seen)
    # 断言
    g1 = "mother_opus_entry" in nodes_ran and "analyze" not in nodes_ran and "mother_precheck" not in nodes_ran
    print(f"\nG1 入口走塌缩节点(不走analyze/precheck): {g1}")
    if need_confirm_seen:
        print("→ 低置信路径：发了 needConfirm（D1/D3 条件 confirm）✅")
    elif mother_card_seen:
        print(f"→ 高置信路径：母题卡先出（first_card_idx={first_card_idx}）+ await_review 硬停 ✅ G3")
        print("await_review in nodes:", "await_review" in nodes_ran)
    else:
        print("⚠ 既无 needConfirm 也无 mother_card —— 查 opus 失败/池不可用")


if __name__ == "__main__":
    asyncio.run(main())
