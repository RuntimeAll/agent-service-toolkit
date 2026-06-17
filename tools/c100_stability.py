# -*- coding: utf-8 -*-
"""PRD-C-100 渠道切换·全量稳定性+覆盖测试（sui-xiang 主 + aigeek 熔断备 + 全 opus + base64 + retry）。

Phase A 守卫(auth/ask) · Phase B 功能脊线(母题→生成链→编辑→入库 全路径) ·
Phase C 稳定性循环(N 轮母题，量 relay/failover/parse成功/延迟) · Phase D 预算拦截。
真调 opus(sui-xiang 慢)。需 RuoYi :8090 + MySQL :3307。结果落 c100_stability_result.json。
"""
from __future__ import annotations
import asyncio, json, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.variant import variant
from agents.variant_support import RuoyiClient
from agents import conv_trace as ct
from core import settings as settings_mod

variant.checkpointer = MemorySaver()
GEO = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png"
RESULTS, LOG = [], []


def log(m):
    LOG.append(m); print(m, flush=True)


async def alg_url():
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT request FROM conv_llm_trace WHERE id=1385")
    req = cur.fetchone()[0]; cur.close(); conn.close()
    import re
    urls = re.findall(r'https?://[^\s"\\]+', req)
    return next((u for u in urls if "cos" in u or ".png" in u), GEO)


async def turn(thread, text, token=None, extra=None, timeout=420):
    cfg = {"thread_id": thread}
    if token is not None:
        cfg["ruoyi_token"] = token
    if extra:
        cfg.update(extra)
    config = {"configurable": cfg}
    nodes, frames = [], []
    flags = {"mother_card": False, "needConfirm": False, "paper": False, "error": None, "items": 0}

    async def drive():
        async for mode, chunk in variant.astream(
            {"messages": [HumanMessage(content=text)]}, config=config, stream_mode=["updates", "custom"]):
            if mode == "updates" and isinstance(chunk, dict):
                nodes.extend(chunk.keys())
            elif mode == "custom":
                p = chunk[1] if isinstance(chunk, tuple) and len(chunk) == 2 else chunk
                content = getattr(p, "content", None) or (p.get("content") if isinstance(p, dict) else None)
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            for k in c.keys():
                                frames.append(k)
                                if k == "artifact" and isinstance(c[k], dict):
                                    if (c[k].get("header") or {}).get("mother_card"):
                                        flags["mother_card"] = True
                                    its = c[k].get("items")
                                    if isinstance(its, list) and its:
                                        flags["items"] = max(flags["items"], len(its))
                                if k == "needConfirm":
                                    flags["needConfirm"] = True
                                if k == "paper":
                                    flags["paper"] = True
                                if k == "error":
                                    flags["error"] = str(c[k])[:140]
    t = time.monotonic()
    try:
        await asyncio.wait_for(drive(), timeout=timeout)
    except asyncio.TimeoutError:
        flags["error"] = f"TIMEOUT>{timeout}s"
    dur = time.monotonic() - t
    try:
        snap = await variant.aget_state(config); st = snap.values if snap else {}
    except Exception:
        st = {}
    return {"nodes": nodes, "frames": sorted(set(frames)), "flags": flags, "dur": round(dur),
            "state_items": len(st.get("items") or []), "await_review": bool(st.get("awaiting_mother_review")),
            "await_confirm": bool(st.get("awaiting_mother_confirm")), "has_dna": bool(st.get("mother_dna"))}


def rec(cid, desc, ok, detail, raw=None):
    RESULTS.append({"case": cid, "desc": desc, "pass": ok, "detail": detail, "raw": raw})
    log(f"[{'PASS' if ok else 'FAIL'}] {cid} {desc} :: {detail}")


def trace_since(ts_min):
    """取测试窗口内 conv_trace 行，聚合 relay/failover/parse/token/cost。"""
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("""SELECT label,model,relay,fallback_count,prompt_tokens,completion_tokens,cost_yuan,duration_ms,error
                   FROM conv_llm_trace WHERE ts>=%s ORDER BY id""", (ts_min,))
    rows = cur.fetchall(); cur.close(); conn.close()
    return rows


async def main():
    t0 = time.time()
    start_ts = None
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT NOW(3)"); start_ts = cur.fetchone()[0]; cur.close(); conn.close()
    log(f"[窗口起点 {start_ts}]")
    ALG = await alg_url()
    client = RuoyiClient(); token = await client.login(); await client.aclose()
    log(f"[token ok] ALG={ALG[:70]}")

    # ===== Phase A 守卫 =====
    r = await turn("st-auth", f"{ALG} 出2道", token=None, timeout=30)
    rec("A1", "无登录→require_login", "require_login" in r["nodes"], f"nodes={r['nodes']}", r)
    r = await turn("st-ask", "帮我出几道题", token=token, timeout=30)
    rec("A2", "无图→ask_for_image", "ask_for_image" in r["nodes"], f"nodes={r['nodes']}", r)

    # ===== Phase B 功能脊线(全路径) =====
    T = "st-spine"
    r = await turn(T, f"{ALG} 出2道", token=token, timeout=300)
    spine_ok = "mother_opus_entry" in r["nodes"] and r["flags"]["mother_card"]
    rec("B1", "新图→塌缩入口→母题卡", spine_ok,
        f"review={r['await_review']} confirm={r['await_confirm']} card={r['flags']['mother_card']} {r['dur']}s", r)
    if r["await_review"]:
        r = await turn(T, "开始举一反三", token=token, extra={"start_variants": True}, timeout=480)
        chain = [n for n in ["generate", "gene_gate", "solve_explain", "assemble"] if n in r["nodes"]]
        rec("B2", "开始举一反三→生成链四节点→出变式", len(chain) == 4 and r["state_items"] >= 1,
            f"chain={chain} items={r['state_items']} {r['dur']}s err={r['flags']['error']}", r)
        if r["state_items"] >= 1:
            r2 = await turn(T, "删第1道", token=token, timeout=360)
            rec("B3", "删第N→exec_remove", "exec_remove" in r2["nodes"], f"nodes·rm={[n for n in r2['nodes'] if 'exec' in n or n=='parse_instruction']} items={r2['state_items']}", r2)
            r3 = await turn(T, "再补1道", token=token, timeout=420)
            rec("B4", "补N道→exec_add", "exec_add" in r3["nodes"], f"add={[n for n in r3['nodes'] if 'exec' in n]} items={r3['state_items']}", r3)
            r4 = await turn(T, "第1题怎么讲给学生", token=token, timeout=240)
            rec("B5", "答疑→answer_question", "answer_question" in r4["nodes"], f"nodes={[n for n in r4['nodes'] if n in ('parse_instruction','answer_question')]}", r4)
            r5 = await turn(T, "可以了，全部入库", token=token, timeout=240)
            rec("B6", "入库→persist_to_bank", "persist_to_bank" in r5["nodes"], f"nodes={[n for n in r5['nodes'] if n in ('parse_instruction','persist_to_bank')]} paper={r5['flags']['paper']}", r5)
    else:
        rec("B2", "生成链", False, f"母题未达 await_review(confirm={r['await_confirm']})——脊线没进生成链", r)

    # ===== Phase C 稳定性循环(母题 N 轮) =====
    log("\n===== Phase C 稳定性循环 =====")
    stab = []
    for i in range(8):  # 8× √18(快) + 2× 几何(硬)
        img = ALG if i < 6 else GEO
        kind = "代数" if i < 6 else "几何"
        rr = await turn(f"st-loop-{i}", f"{img} 出1道", token=token, timeout=300)
        ok = "mother_opus_entry" in rr["nodes"] and rr["flags"]["mother_card"] and not rr["flags"]["error"]
        stab.append({"i": i, "kind": kind, "ok": ok, "dur": rr["dur"], "card": rr["flags"]["mother_card"],
                     "err": rr["flags"]["error"], "review": rr["await_review"], "confirm": rr["await_confirm"]})
        log(f"  loop{i}({kind}): {'OK' if ok else 'FAIL'} {rr['dur']}s card={rr['flags']['mother_card']} err={rr['flags']['error']}")
    npass_stab = sum(1 for s in stab if s["ok"])
    rec("C-stab", f"母题稳定性循环 {npass_stab}/{len(stab)}", npass_stab >= len(stab) * 0.7,
        f"成功={npass_stab}/{len(stab)} 延迟={[s['dur'] for s in stab]}", {"stab": stab})

    # ===== Phase D 预算拦截 =====
    try:
        from agents import cost_guard
        cost_guard._cache["ts"] = 0.0
        old = settings_mod.GLOBAL_DAILY_BUDGET_YUAN
        settings_mod.GLOBAL_DAILY_BUDGET_YUAN = 0.0001
        rr = await turn("st-budget", f"{ALG} 出1道", token=token, timeout=60)
        settings_mod.GLOBAL_DAILY_BUDGET_YUAN = old
        rec("D1", "预算超额→母题前拦截", bool(rr["flags"]["error"]) and not rr["flags"]["mother_card"],
            f"error={rr['flags']['error']} card={rr['flags']['mother_card']}", rr)
    except Exception as e:
        rec("D1", "预算拦截", False, f"EXC {e}\n{traceback.format_exc()[:200]}")

    # ===== 聚合 conv_trace =====
    rows = trace_since(start_ts)
    by_relay, fallbacks, errs, total_cost, total_pt, total_ct = {}, 0, 0, 0.0, 0, 0
    for label, model, relay, fb, pt, cct, cost, dur, err in rows:
        by_relay[relay] = by_relay.get(relay, 0) + 1
        if fb and fb > 0:
            fallbacks += 1
        if err:
            errs += 1
        total_cost += float(cost or 0); total_pt += int(pt or 0); total_ct += int(cct or 0)
    agg = {"calls": len(rows), "by_relay": by_relay, "fallback_calls": fallbacks, "error_calls": errs,
           "total_cost_yuan": round(total_cost, 4), "total_pt": total_pt, "total_ct": total_ct,
           "all_model_opus": all((m == "claude-opus-4-8") for _, m, *_ in rows) if rows else None}
    log(f"\n===== conv_trace 聚合 =====\n{json.dumps(agg, ensure_ascii=False)}")

    dur = time.time() - t0
    npass = sum(1 for x in RESULTS if x["pass"])
    log(f"\n==== 总览 {npass}/{len(RESULTS)} PASS · {dur:.0f}s ====")
    out = Path(__file__).resolve().parent / "c100_stability_result.json"
    out.write_text(json.dumps({"pass": npass, "total": len(RESULTS), "dur_s": round(dur),
                               "agg": agg, "results": RESULTS, "log": LOG}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[结果落 {out}]")


asyncio.run(main())
