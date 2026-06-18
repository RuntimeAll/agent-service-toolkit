# -*- coding: utf-8 -*-
"""PRD-C-100 sui-xiang 生成链压测——N 轮完整举一反三(母题→开始举一反三),量稳定频率分布。
post 超时-120 fix：挂起会现形(120s+failover 行)。给「保持 sui-xiang 主」的真实账。
"""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from agents.variant import variant
from agents.variant_support import RuoyiClient
from agents import conv_trace as ct

variant.checkpointer = MemorySaver()
N = 10  # 轮数


async def alg_url():
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT request FROM conv_llm_trace WHERE id=1385")
    req = cur.fetchone()[0]; cur.close(); conn.close()
    import re
    return next((u for u in re.findall(r'https?://[^\s"\\]+', req) if "cos" in u or ".png" in u), None)


async def turn(thread, text, token, extra=None, timeout=600):
    cfg = {"thread_id": thread, "ruoyi_token": token}
    if extra:
        cfg.update(extra)
    config = {"configurable": cfg}
    flags = {"items": 0, "card": False, "error": None}

    async def drive():
        async for mode, chunk in variant.astream({"messages": [HumanMessage(content=text)]},
                                                  config=config, stream_mode=["updates", "custom"]):
            if mode == "custom":
                p = chunk[1] if isinstance(chunk, tuple) and len(chunk) == 2 else chunk
                content = getattr(p, "content", None) or (p.get("content") if isinstance(p, dict) else None)
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            blob = json.dumps(c, ensure_ascii=False)
                            if '"mother_card"' in blob:
                                flags["card"] = True
                            if c.get("error"):
                                flags["error"] = str(c.get("error"))[:80]
    t = time.monotonic()
    try:
        await asyncio.wait_for(drive(), timeout=timeout)
    except asyncio.TimeoutError:
        flags["error"] = f"CLIENT_TIMEOUT>{timeout}s"
    try:
        snap = await variant.aget_state(config)
        flags["items"] = len((snap.values or {}).get("items") or []) if snap else 0
        flags["await_review"] = bool((snap.values or {}).get("awaiting_mother_review")) if snap else False
    except Exception:
        pass
    flags["dur"] = round(time.monotonic() - t)
    return flags


async def main():
    t0 = time.time()
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT NOW(3)"); _ = cur.fetchone(); cur.close(); conn.close()
    ALG = await alg_url()
    client = RuoyiClient(); token = await client.login(); await client.aclose()
    # 窗口起点 id（隔离本压测的 conv_trace 行）
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT MAX(id) FROM conv_llm_trace"); start_id = cur.fetchone()[0] or 0; cur.close(); conn.close()
    print(f"[start] ALG={ALG[:60]} start_id={start_id}", flush=True)

    rounds = []
    for i in range(N):
        T = f"stress-{i}"
        m = await turn(T, f"{ALG} 出2道", token, timeout=300)
        gen = {"items": 0, "dur": 0, "error": "母题未达review"}
        if m.get("await_review"):
            gen = await turn(T, "开始举一反三", token, extra={"start_variants": True}, timeout=600)
        ok = gen.get("items", 0) >= 1
        rounds.append({"i": i, "ok": ok, "mother_dur": m["dur"], "gen_dur": gen["dur"],
                       "items": gen.get("items", 0), "m_err": m.get("error"), "g_err": gen.get("error")})
        print(f"  round{i}: {'OK' if ok else 'FAIL'} 母题{m['dur']}s 生成{gen['dur']}s items={gen.get('items',0)} g_err={gen.get('error')}", flush=True)

    # conv_trace 聚合(本压测窗口)
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT relay,fallback_count,duration_ms,error,label FROM conv_llm_trace WHERE id>%s", (start_id,))
    rows = cur.fetchall(); cur.close(); conn.close()
    from collections import Counter
    relayc = Counter(); fb = 0; errc = 0; durs = []; slow120 = 0
    for relay, fbc, dur, err, label in rows:
        relayc[relay] += 1
        if fbc and fbc > 0:
            fb += 1
        if err:
            errc += 1
        if dur:
            durs.append(dur)
            if dur >= 115000:
                slow120 += 1
    ok_n = sum(1 for r in rounds if r["ok"])
    agg = {
        "rounds": N, "完成率": f"{ok_n}/{N}", "总LLM调用": len(rows),
        "relay分布": dict(relayc), "failover次数(挂起切aigeek)": fb, "error行": errc,
        "近120s准挂起调用数": slow120,
        "生成链延迟s": [r["gen_dur"] for r in rounds],
        "母题延迟s": [r["mother_dur"] for r in rounds],
    }
    if durs:
        agg["调用延迟ms_中位/p90/max"] = [sorted(durs)[len(durs)//2], sorted(durs)[int(len(durs)*0.9)], max(durs)]
    print("\n===== 压测稳定频率分布 =====", flush=True)
    print(json.dumps(agg, ensure_ascii=False, indent=2), flush=True)
    out = Path(__file__).resolve().parent / "c100_stress_result.json"
    out.write_text(json.dumps({"agg": agg, "rounds": rounds, "dur_s": round(time.time()-t0)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[结果落 {out}] 总耗时 {round(time.time()-t0)}s", flush=True)


asyncio.run(main())
