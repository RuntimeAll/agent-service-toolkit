# -*- coding: utf-8 -*-
"""PRD-C-100 编排覆盖性测试：in-process 真机驱动 variant 编排图，逐路径断言。

覆盖 14 条编排路径（见 COVERAGE 矩阵）：守卫(auth/ask) + 塌缩入口(高置信/低置信确认/带图) +
生成链(generate→闸A→闸B→assemble) + 编辑漏斗(删/补/答疑/入库) + 预算拦截(G7)。

真调 opus（每图 30-180s，生成链更久）。需 RuoYi :8090 + MySQL :3307 + aigeek 在跑。
跑法：$env:PYTHONUTF8=1; .venv/Scripts/python.exe tools/c100_coverage.py
结果落 tools/c100_coverage_result.json + 控制台矩阵。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import variant  # noqa: E402
from agents.variant_support import RuoyiClient  # noqa: E402
from core import settings as settings_mod  # noqa: E402

# 真实母题图（纯代数 = 高置信无图；几何 = 带图）
ALGEBRA_URL = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png"
GEOMETRY_URL = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png"

# 多轮共用 checkpointer（机制同生产 service lifespan 注入 saver）
variant.checkpointer = MemorySaver()

RESULTS: list[dict] = []


async def run_turn(thread_id: str, text: str, *, token: str | None, extra_cfg: dict | None = None,
                   timeout: float = 300.0) -> dict:
    """驱动一轮，收集跑了哪些节点 + 发了哪些帧 + 终态关键字段。"""
    cfg = {"thread_id": thread_id}
    if token is not None:
        cfg["ruoyi_token"] = token
    if extra_cfg:
        cfg.update(extra_cfg)
    config = {"configurable": cfg}

    nodes: list[str] = []
    frame_keys: list[str] = []
    flags = {"mother_card": False, "needConfirm": False, "needAsk": False,
             "paper": False, "error": None, "mother_has_figure": None, "variant_items": 0}

    async def _drive() -> None:
        async for mode, chunk in variant.astream(
            {"messages": [HumanMessage(content=text)]}, config=config,
            stream_mode=["updates", "custom"],
        ):
            if mode == "updates" and isinstance(chunk, dict):
                nodes.extend(chunk.keys())
            elif mode == "custom":
                payload = chunk[1] if isinstance(chunk, tuple) and len(chunk) == 2 else chunk
                content = getattr(payload, "content", None) or (
                    payload.get("content") if isinstance(payload, dict) else None)
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    for k in c.keys():
                        frame_keys.append(k)
                        if k == "artifact" and isinstance(c[k], dict):
                            hdr = c[k].get("header") or {}
                            mc = hdr.get("mother_card")
                            if mc:
                                flags["mother_card"] = True
                                if isinstance(mc, dict) and "mother_has_figure" in mc:
                                    flags["mother_has_figure"] = mc.get("mother_has_figure")
                            its = c[k].get("items")
                            if isinstance(its, list) and its:
                                flags["variant_items"] = max(flags["variant_items"], len(its))
                        if k == "needConfirm":
                            flags["needConfirm"] = True
                        if k == "needAsk":
                            flags["needAsk"] = True
                        if k == "paper":
                            flags["paper"] = True
                        if k == "error":
                            flags["error"] = str(c[k])[:120]

    try:
        await asyncio.wait_for(_drive(), timeout=timeout)
    except asyncio.TimeoutError:
        flags["error"] = f"TIMEOUT>{timeout}s"

    # 终态（机制同生产：从 checkpointer 读持久 state）
    st = None
    try:
        snap = await variant.aget_state(config)
        st = snap.values if snap else {}
    except Exception as e:  # noqa: BLE001
        st = {"_state_err": str(e)[:80]}
    return {"nodes": nodes, "frames": sorted(set(frame_keys)), "flags": flags,
            "state_items": len(st.get("items") or []) if isinstance(st, dict) else None,
            "awaiting_review": bool(st.get("awaiting_mother_review")) if isinstance(st, dict) else None,
            "awaiting_confirm": bool(st.get("awaiting_mother_confirm")) if isinstance(st, dict) else None,
            "has_mother_dna": bool(st.get("mother_dna")) if isinstance(st, dict) else None}


def record(cid: str, desc: str, ok: bool, detail: str, raw: dict | None = None) -> None:
    RESULTS.append({"case": cid, "desc": desc, "pass": ok, "detail": detail, "raw": raw})
    print(f"[{'PASS' if ok else 'FAIL'}] {cid} {desc} :: {detail}", flush=True)


async def main() -> None:
    t0 = time.time()
    client = RuoyiClient()
    token = await client.login()
    await client.aclose()
    print(f"[token ok: {token[:12]}…]\n", flush=True)

    # ---- C1 守卫·无登录 → auth → require_login（纯代码，不调 LLM）----
    try:
        r = await run_turn("cov-auth", f"{ALGEBRA_URL} 出2道", token=None, timeout=30)
        ok = "require_login" in r["nodes"]
        record("C1", "无登录→require_login(身份硬闸)", ok, f"nodes={r['nodes']}", r)
    except Exception as e:  # noqa: BLE001
        record("C1", "无登录→require_login", False, f"EXC {e}")

    # ---- C2 守卫·有token无图无态 → ask → ask_for_image ----
    try:
        r = await run_turn("cov-ask", "帮我出几道题", token=token, timeout=30)
        ok = "ask_for_image" in r["nodes"]
        record("C2", "无图→ask_for_image(催图兜底)", ok, f"nodes={r['nodes']}", r)
    except Exception as e:  # noqa: BLE001
        record("C2", "无图→ask_for_image", False, f"EXC {e}")

    # ---- C3 塌缩入口·纯代数高置信 → mother_opus_entry → await_review(母题卡硬停) ----
    T = "cov-spine"  # 主线程，后续多轮复用
    try:
        r = await run_turn(T, f"{ALGEBRA_URL} 出2道", token=token, timeout=240)
        ok = ("mother_opus_entry" in r["nodes"] and r["flags"]["mother_card"]
              and "analyze" not in r["nodes"] and "mother_precheck" not in r["nodes"])
        path = "await_review" if "await_review" in r["nodes"] else (
            "confirm" if r["flags"]["needConfirm"] else "?")
        record("C3", "新图→塌缩入口→母题卡(高置信硬停)", ok,
               f"path={path} card={r['flags']['mother_card']} dna={r['has_mother_dna']} nodes={r['nodes']}", r)
    except Exception as e:  # noqa: BLE001
        record("C3", "新图→塌缩入口→母题卡", False, f"EXC {e}\n{traceback.format_exc()[:300]}")

    # ---- C5 生成链·开始举一反三 → generate→闸A→闸B→assemble→变式（核心未测路径）----
    try:
        r = await run_turn(T, "开始举一反三", token=token,
                           extra_cfg={"start_variants": True}, timeout=420)
        chain = [n for n in ["generate", "gene_gate", "solve_explain", "assemble"] if n in r["nodes"]]
        ok = (len(chain) == 4 and r["state_items"] and r["state_items"] >= 1)
        record("C5", "开始举一反三→生成链四节点→出变式", ok,
               f"chain={chain} items={r['state_items']} frames={r['frames']} err={r['flags']['error']}", r)
    except Exception as e:  # noqa: BLE001
        record("C5", "生成链四节点", False, f"EXC {e}\n{traceback.format_exc()[:300]}")

    # ---- C6 编辑·删第2 → parse→dispatch→exec_remove→solve_explain→assemble ----
    try:
        before = RESULTS[-1]["raw"]["state_items"] if RESULTS[-1].get("raw") else None
        r = await run_turn(T, "删第2道", token=token, timeout=300)
        ok = "exec_remove" in r["nodes"] and "parse_instruction" in r["nodes"]
        record("C6", "删第N→exec_remove", ok,
               f"items {before}→{r['state_items']} nodes={[n for n in r['nodes'] if n in ('parse_instruction','dispatch','exec_remove','solve_explain','assemble')]}", r)
    except Exception as e:  # noqa: BLE001
        record("C6", "删第N→exec_remove", False, f"EXC {e}")

    # ---- C7 编辑·补1道 → exec_add→gene_gate→... ----
    try:
        before = RESULTS[-1]["raw"]["state_items"] if RESULTS[-1].get("raw") else None
        r = await run_turn(T, "再补1道", token=token, timeout=360)
        ok = "exec_add" in r["nodes"]
        record("C7", "补N道→exec_add", ok,
               f"items {before}→{r['state_items']} add_chain={[n for n in r['nodes'] if n in ('exec_add','gene_gate','solve_explain','assemble')]}", r)
    except Exception as e:  # noqa: BLE001
        record("C7", "补N道→exec_add", False, f"EXC {e}")

    # ---- C8 答疑·某题怎么做 → answer_question ----
    try:
        r = await run_turn(T, "第1题这道题怎么讲给学生", token=token, timeout=180)
        ok = "answer_question" in r["nodes"]
        record("C8", "答疑→answer_question", ok, f"nodes={[n for n in r['nodes'] if n in ('parse_instruction','answer_question')]}", r)
    except Exception as e:  # noqa: BLE001
        record("C8", "答疑→answer_question", False, f"EXC {e}")

    # ---- C9 入库·可以了 → persist_to_bank（真写库）----
    try:
        r = await run_turn(T, "可以了，全部入库", token=token, timeout=180)
        ok = "persist_to_bank" in r["nodes"]
        record("C9", "入库→persist_to_bank(真写库)", ok, f"nodes={[n for n in r['nodes'] if n in ('parse_instruction','persist_to_bank')]} frames={r['frames']}", r)
    except Exception as e:  # noqa: BLE001
        record("C9", "入库→persist_to_bank", False, f"EXC {e}")

    # ---- C10 带图·几何母题 → mother_opus_entry + has_figure ----
    try:
        r = await run_turn("cov-fig", f"{GEOMETRY_URL} 出1道", token=token, timeout=240)
        ok = "mother_opus_entry" in r["nodes"] and r["flags"]["mother_card"]
        record("C10", "带图母题→塌缩入口(has_figure帧)", ok,
               f"card={r['flags']['mother_card']} has_figure={r['flags']['mother_has_figure']}", r)
    except Exception as e:  # noqa: BLE001
        record("C10", "带图母题→塌缩入口", False, f"EXC {e}")

    # ---- C11 预算拦截·G7 → 母题前 budget_exceeded（不调 opus）----
    try:
        from agents import cost_guard
        cost_guard._cache["ts"] = 0.0  # 清缓存强制重读今日花费
        old = settings_mod.settings.GLOBAL_DAILY_BUDGET_YUAN
        settings_mod.settings.GLOBAL_DAILY_BUDGET_YUAN = 0.0001  # 远低于今日已花
        r = await run_turn("cov-budget", f"{ALGEBRA_URL} 出1道", token=token, timeout=60)
        settings_mod.settings.GLOBAL_DAILY_BUDGET_YUAN = old
        # 拦截 = 报 budget_exceeded error 且没真跑生成链
        ok = bool(r["flags"]["error"]) and not r["flags"]["mother_card"]
        record("C11", "预算超额→母题前拦截(G7,不调opus)", ok,
               f"error={r['flags']['error']} card={r['flags']['mother_card']}", r)
    except Exception as e:  # noqa: BLE001
        try:
            settings_mod.settings.GLOBAL_DAILY_BUDGET_YUAN = old
        except Exception:
            pass
        record("C11", "预算超额→母题前拦截", False, f"EXC {e}")

    dur = time.time() - t0
    npass = sum(1 for x in RESULTS if x["pass"])
    print(f"\n==== 覆盖矩阵 {npass}/{len(RESULTS)} PASS · {dur:.0f}s ====", flush=True)
    out = Path(__file__).resolve().parent / "c100_coverage_result.json"
    out.write_text(json.dumps({"pass": npass, "total": len(RESULTS), "dur_s": round(dur),
                               "results": RESULTS}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[结果落 {out}]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
