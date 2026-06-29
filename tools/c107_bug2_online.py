# -*- coding: utf-8 -*-
"""PRD-C-107 BUG-2·在线复现：确认章后母题降级锚到章级(待人审) → 点开始举一反三 → 0 变式。
轮1 贴图母题→needConfirm；轮2 确认章(BUG-03 单轮闸断)；轮3 再确认同章(坚持→await_review)；
轮4 开始举一反三(start_variants，不带 confirmed_chapter_id)→ **期望出变式**。
"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
BASE = "http://localhost:8093"
IMG = ("https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
       "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png")


async def _stream(client, message, tid, token, cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": tid,
            "agent_config": {"ruoyi_token": token, **(cfg or {})}}
    fr = {"stage": 0, "items": [], "error": None, "need_confirm": None, "msgs": "", "stages": []}
    async with client.stream("POST", f"{BASE}/variant/stream", json=body,
                             headers={"Content-Type": "application/json"}, timeout=400) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if not data or data == "[DONE]": continue
            try: obj = json.loads(data)
            except Exception: continue
            _walk(obj, fr)
    return fr


def _walk(obj, fr):
    if isinstance(obj, dict):
        st = obj.get("stage")
        if isinstance(st, dict):
            fr["stage"] += 1
            fr["stages"].append(f"{st.get('status')}={st.get('detail','')[:50]}")
        elif isinstance(st, str): fr["stage"] += 1
        for key in ("items", "variants"):
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict) and any("stem" in x for x in v if isinstance(x, dict)):
                fr["items"] = v
        if isinstance(obj.get("needConfirm"), dict): fr["need_confirm"] = obj["needConfirm"]
        if isinstance(obj.get("content"), str): fr["msgs"] += " " + obj["content"]
        if isinstance(obj.get("message"), str): fr["msgs"] += " " + obj["message"]
        if obj.get("error") and not fr["error"]: fr["error"] = str(obj.get("error"))[:200]
        for v in obj.values(): _walk(v, fr)
    elif isinstance(obj, list):
        for v in obj: _walk(v, fr)


async def main():
    import httpx
    from agents.variant_support import RuoyiClient
    rc = RuoyiClient(); token = await rc.login(); await rc.aclose()
    tid = "c107-bug2-final-clean"
    chap_id = "2000020"
    async with httpx.AsyncClient() as client:
        print("== 轮1·贴图母题 ==")
        f1 = await _stream(client, f"帮我对这道题举一反三 {IMG}", tid, token)
        print(f"  stage={f1['stage']} needConfirm={bool(f1['need_confirm'])} err={f1['error']}")
        cfg = {"confirmed_chapter_id": chap_id, "confirmed_grade_book_id": chap_id[:4]}
        print("== 轮2·确认章（不带 start_variants）==")
        f2 = await _stream(client, "确认", tid, token, cfg)
        print(f"  stage={f2['stage']} items={len(f2['items'])} needConfirm={bool(f2['need_confirm'])}")
        for s in f2["stages"]: print(f"    · {s}")
        print("== 轮3·再确认同章（坚持→await_review）==")
        f3 = await _stream(client, "确认", tid, token, cfg)
        print(f"  stage={f3['stage']} items={len(f3['items'])} needConfirm={bool(f3['need_confirm'])}")
        for s in f3["stages"]: print(f"    · {s}")
        print("== 轮4·开始举一反三（start_variants，不带 confirmed_chapter_id）==")
        f4 = await _stream(client, "开始举一反三", tid, token, {"start_variants": True})
        print(f"  stage={f4['stage']} items={len(f4['items'])} err={f4['error']}")
        for s in f4["stages"]: print(f"    · {s}")
        print(f"  msgs(tail)= ...{f4['msgs'][-160:]}")
        print(f"\n[{'PASS' if f4['items'] else 'FAIL'}] 轮4 流出变式 {len(f4['items'])} 道")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
