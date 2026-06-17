# -*- coding: utf-8 -*-
"""真打运行中的 :8093 /variant/stream 服务层冒烟(非 in-process)——母题→开始举一反三,验 SSE 帧。"""
import asyncio, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import httpx
from agents.variant_support import RuoyiClient
from agents import conv_trace as ct

BASE = "http://localhost:8093"


async def alg_url():
    conn = ct._conn(); cur = conn.cursor()
    cur.execute("SELECT request FROM conv_llm_trace WHERE id=1385")
    req = cur.fetchone()[0]; cur.close(); conn.close()
    urls = re.findall(r'https?://[^\s"\\]+', req)
    return next((u for u in urls if "cos" in u or ".png" in u), None)


async def stream_call(client, message, thread_id, token, extra_cfg=None):
    cfg = {"ruoyi_token": token}
    if extra_cfg:
        cfg.update(extra_cfg)
    body = {"message": message, "stream_tokens": True, "thread_id": thread_id, "agent_config": cfg}
    frames = {"mother_card": False, "stage": 0, "error": None, "items": 0, "paper": False, "raw_events": 0}
    t = asyncio.get_event_loop().time()
    async with client.stream("POST", f"{BASE}/variant/stream", json=body,
                             headers={"Content-Type": "application/json"}, timeout=480) as resp:
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            txt = (await resp.aread())[:300]
            return {"err": f"非SSE: status={resp.status_code} ct={ctype} body={txt!r}"}
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            frames["raw_events"] += 1
            try:
                obj = json.loads(data)
            except Exception:
                continue
            # toolkit StreamInput SSE：{type, content}；content 可能是 message 对象
            content = obj.get("content") if isinstance(obj, dict) else None
            blob = json.dumps(obj, ensure_ascii=False)
            if '"mother_card"' in blob:
                frames["mother_card"] = True
            if '"stage"' in blob:
                frames["stage"] += 1
            if '"error"' in blob and '"reason"' in blob:
                m = re.search(r'"reason"\s*:\s*"([^"]+)"', blob)
                if m and m.group(1) not in ("",):
                    frames["error"] = m.group(1)
            if '"paper"' in blob:
                frames["paper"] = True
            mi = re.findall(r'"items"\s*:\s*\[', blob)
            if mi:
                frames["items"] = max(frames["items"], blob.count('"stem"'))
    frames["dur"] = round(asyncio.get_event_loop().time() - t)
    return frames


async def main():
    ALG = await alg_url()
    client0 = RuoyiClient(); token = await client0.login(); await client0.aclose()
    print(f"[token ok] ALG={ALG[:64]}")
    async with httpx.AsyncClient(trust_env=False) as client:
        tid = "http-smoke-1"
        print("\n--- ① 母题(打 :8093 服务) ---")
        r1 = await stream_call(client, f"{ALG} 出2道", tid, token)
        print(json.dumps(r1, ensure_ascii=False))
        print("\n--- ② 开始举一反三(同 thread) ---")
        r2 = await stream_call(client, "开始举一反三", tid, token, extra_cfg={"start_variants": True})
        print(json.dumps(r2, ensure_ascii=False))
        ok = r1.get("mother_card") and not r1.get("err") and not r1.get("error")
        ok2 = r2.get("items", 0) >= 1 and not r2.get("error")
        print(f"\n服务层冒烟: 母题={'PASS' if ok else 'FAIL'} 生成链={'PASS' if ok2 else 'FAIL'}")


asyncio.run(main())
