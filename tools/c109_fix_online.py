"""PRD-C-109 fix 在线复现：母题卡就绪态（无题组）打字「加易错标签 / 改题型」真生效。

复现 e2e 暴露的根因：贴题→确认一次→就绪→打字「加个易错标签：判别式符号」时，旧实现
把就绪卡当母题全量重解 → 反复弹确认 + 标签没加上 + 就绪卡被打回存疑。本脚本断言修后：
  ① 加标签：标签真加上（母题卡 tags 含「判别式符号」）+ 不重弹确认章（needConfirm 不再出）
     + 就绪卡不打回存疑（无 classify/solve 重解阶段灯）+ 本轮不出变式。
  ② 改题型：题型变（既有重锚链）+ endorsed（已确认过）不重弹确认章。

跑法：set NO_PROXY=*；.venv\\Scripts\\python.exe tools\\c109_fix_online.py
依赖：toolkit :8093 已加载新码 + RuoYi :8090 活。
"""
import asyncio
import json
import sys

BASE = "http://localhost:8093"

IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)


def _walk(obj, fr):
    if isinstance(obj, dict):
        st = obj.get("stage")
        if isinstance(st, dict):
            key = str(st.get("key") or st.get("name") or "")
            if key:
                fr["stages"].append(key)
        elif isinstance(st, str):
            fr["stages"].append(st)
        for holder in (obj.get("header"), obj):
            if isinstance(holder, dict) and isinstance(holder.get("mother_card"), dict):
                mc = holder["mother_card"]
                fr["mother_card"] = True
                dna = (mc.get("dna") or {}) if isinstance(mc.get("dna"), dict) else {}
                if dna.get("tags") is not None:
                    fr["card_tags"] = list(dna.get("tags") or [])
                if dna.get("qtype"):
                    fr["card_qtype"] = dna.get("qtype")
        for key in ("items", "variants"):
            v = obj.get(key)
            if isinstance(v, list) and v and any(isinstance(x, dict) and "stem" in x for x in v):
                fr["items"] = v
        if isinstance(obj.get("needConfirm"), dict):
            fr["need_confirm"] = obj["needConfirm"]
        c = obj.get("content")
        if isinstance(c, str) and c.strip():
            fr["texts"].append(c.strip()[:140])
        if obj.get("error") and not fr["error"]:
            fr["error"] = str(obj.get("error"))[:200]
        for v in obj.values():
            _walk(v, fr)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, fr)


async def _stream(client, message, tid, token, cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": tid,
            "agent_config": {"ruoyi_token": token, **(cfg or {})}}
    fr = {"stages": [], "items": [], "error": None, "need_confirm": None,
          "mother_card": False, "card_tags": None, "card_qtype": None, "texts": []}
    async with client.stream("POST", f"{BASE}/variant/stream", json=body,
                             headers={"Content-Type": "application/json"}, timeout=400) as resp:
        if "text/event-stream" not in resp.headers.get("content-type", ""):
            fr["error"] = f"非SSE status={resp.status_code}"
            return fr
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            _walk(obj, fr)
    return fr


async def main() -> bool:
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    tid = "c109-fix-rerun2"
    ok = True
    async with httpx.AsyncClient() as client:
        print("== 1) 贴图母题 → 母题卡 / needConfirm ==")
        f1 = await _stream(client, f"帮我对这道题举一反三 {IMG}", tid, token)
        print(f"  stages={f1['stages'][:12]} motherCard={f1['mother_card']} "
              f"needConfirm={bool(f1['need_confirm'])} err={f1['error']}")
        ok = ok and bool(f1["stages"]) and not f1["error"]

        # 确认一次（落 await_review 就绪态；带 confirmed_chapter_id = 隐式背书 endorsed）
        nc = f1.get("need_confirm") or {}
        if nc:
            ch = nc.get("chapter") or {}
            gb = nc.get("grade_book") or {}
            cfg = {"confirmed_chapter_id": ch.get("id") or ch.get("name") or "",
                   "confirmed_grade_book_id": gb.get("id") or "",
                   "confirmed_chapter_name": ch.get("name") or ""}
            print(f"  确认一次「{ch.get('name')}」→ 落就绪态（隐式背书）")
            fc = await _stream(client, "确认", tid, token, cfg=cfg)
            print(f"  确认后 stages={fc['stages'][:12]} motherCard={fc['mother_card']} "
                  f"cardTags={fc['card_tags']} err={fc['error']}")
            ok = ok and not fc["error"]

        print("\n== 2) 就绪态打字【加个易错标签：判别式符号】(C-109 fix 主复现) ==")
        fa = await _stream(client, "加个易错标签：判别式符号", tid, token)
        tags = fa.get("card_tags") or []
        tag_added = any("判别式" in str(t) or "符号" in str(t) for t in tags)
        no_reconfirm = not fa.get("need_confirm")
        no_resolve = not any(k in fa["stages"] for k in ("solve",)) and \
            fa["stages"].count("classify") == 0
        no_new_items = len(fa["items"]) == 0
        print(f"  stages={fa['stages'][:14]}")
        print(f"  cardTags={tags}  texts={fa['texts'][:2]}")
        print(f"  needConfirm={bool(fa.get('need_confirm'))}  items={len(fa['items'])}  err={fa['error']}")
        print(f"  [{'PASS' if tag_added else 'FAIL'}] 标签真加上（母题卡 tags 含判别式符号）")
        print(f"  [{'PASS' if no_reconfirm else 'FAIL'}] 不重弹确认章（needConfirm 不再出）")
        print(f"  [{'PASS' if no_resolve else 'FAIL'}] 就绪卡不打回存疑（无 classify/solve 重解阶段灯）")
        print(f"  [{'PASS' if no_new_items else 'FAIL'}] 本轮不出变式（meta 即时生效，不抢跑）")
        ok = ok and tag_added and no_reconfirm and no_resolve and no_new_items and not fa["error"]

        print("\n== 3) 就绪态打字【题型改成填空】(重出·无题组在位改、不重弹确认) ==")
        fb = await _stream(client, "题型改成填空", tid, token)
        no_reconfirm2 = not fb.get("need_confirm")
        no_resolve2 = "solve" not in fb["stages"] and fb["stages"].count("classify") == 0
        qtype_changed = str(fb.get("card_qtype") or "") == "填空"
        not_ignored = bool(fb["stages"]) or bool(fb["texts"])
        print(f"  stages={fb['stages'][:14]} cardQtype={fb.get('card_qtype')} "
              f"needConfirm={bool(fb.get('need_confirm'))} err={fb['error']}")
        print(f"  texts={fb['texts'][:2]}")
        print(f"  [{'PASS' if qtype_changed else 'FAIL'}] 题型真改成填空（母题卡 qtype=填空）")
        print(f"  [{'PASS' if no_reconfirm2 else 'FAIL'}] 不重弹确认章（needConfirm 不再出）")
        print(f"  [{'PASS' if no_resolve2 else 'FAIL'}] 就绪卡不打回存疑（无 classify/solve 重锚阶段灯）")
        print(f"  [{'PASS' if not_ignored else 'FAIL'}] 老师的话没被忽略（有阶段帧/AI 应答）")
        ok = ok and qtype_changed and no_reconfirm2 and no_resolve2 and not_ignored and not fb["error"]

    print(f"\n===== C-109 fix 在线复现: {'GREEN' if ok else 'RED'} =====")
    return ok


if __name__ == "__main__":
    sys.path.insert(0, "src")
    res = asyncio.run(main())
    sys.exit(0 if res else 1)
