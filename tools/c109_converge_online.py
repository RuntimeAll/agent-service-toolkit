"""PRD-C-109 收敛修在线复现（三样连跑·防顾此失彼）：
贴题 → 确认一次 → 就绪 → 加标签(在位 OK) → 难度难一点(旋钮·不重解) → 点开始(出 ≥3 道)。

断言：
  G1 守：加标签 → 母题卡 tags 真加上 + 不重弹确认 + 不打回存疑（无 solve/classify 重解灯）+ 本轮不出题。
  G3 治：难度难一点 → 不重弹确认 + 不重解（无 solve 灯 + ack 不含「重新解题」）+ 标签维不被冲（tags 仍在）。
  回归2 治：点开始 → 真出 ≥3 道变式（不返「不能出题」拒造消息）。

跑法：set NO_PROXY=*；.venv\\Scripts\\python.exe tools\\c109_converge_online.py
依赖：toolkit :8093 已加载新码（改完必重启）+ RuoYi :8090 活。
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
            fr["texts"].append(c.strip()[:160])
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
                             headers={"Content-Type": "application/json"}, timeout=600) as resp:
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


def _has_resolve(fr) -> bool:
    """是否触发了母题重解（solve 灯 / classify 重锚灯 / ack 含「重新解题」）。"""
    if "solve" in fr["stages"]:
        return True
    if fr["stages"].count("classify") > 0:
        return True
    return any("重新解题" in t for t in fr["texts"])


async def main() -> bool:
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    tid = "c109-converge-online-1"
    ok = True
    async with httpx.AsyncClient() as client:
        print("== 1) 贴图母题 ==")
        f1 = await _stream(client, f"帮我对这道题举一反三 {IMG}", tid, token)
        print(f"  stages={f1['stages'][:12]} motherCard={f1['mother_card']} "
              f"needConfirm={bool(f1['need_confirm'])} err={f1['error']}")
        ok = ok and bool(f1["stages"]) and not f1["error"]

        print("\n== 2) 确认一次（隐式背书 endorsed）→ 就绪 ==")
        nc = f1.get("need_confirm") or {}
        if nc:
            ch = nc.get("chapter") or {}
            gb = nc.get("grade_book") or {}
            cfg = {"confirmed_chapter_id": ch.get("id") or ch.get("name") or "",
                   "confirmed_grade_book_id": gb.get("id") or "",
                   "confirmed_chapter_name": ch.get("name") or ""}
            fc = await _stream(client, "确认", tid, token, cfg=cfg)
            print(f"  确认后 stages={fc['stages'][:12]} motherCard={fc['mother_card']} "
                  f"cardTags={fc['card_tags']} err={fc['error']}")
            ok = ok and not fc["error"]

        print("\n== 3) 加标签（在位 OK·守 G1） ==")
        fa = await _stream(client, "加个易错标签：判别式符号", tid, token)
        tags_a = fa.get("card_tags") or []
        tag_added = any("判别式" in str(t) or "符号" in str(t) for t in tags_a)
        g1_no_reconfirm = not fa.get("need_confirm")
        g1_no_resolve = not _has_resolve(fa)
        g1_no_items = len(fa["items"]) == 0
        print(f"  stages={fa['stages'][:14]} cardTags={tags_a} items={len(fa['items'])} err={fa['error']}")
        print(f"  [{'PASS' if tag_added else 'FAIL'}] 标签真加上")
        print(f"  [{'PASS' if g1_no_reconfirm else 'FAIL'}] 不重弹确认")
        print(f"  [{'PASS' if g1_no_resolve else 'FAIL'}] 不重解（无 solve/classify 灯）")
        print(f"  [{'PASS' if g1_no_items else 'FAIL'}] 本轮不出题")
        ok = ok and tag_added and g1_no_reconfirm and g1_no_resolve and g1_no_items and not fa["error"]

        print("\n== 4) 难度难一点（旋钮·不重解·守 G3 收敛修） ==")
        fd = await _stream(client, "难度难一点", tid, token)
        tags_d = fd.get("card_tags")
        g3_no_resolve = not _has_resolve(fd)
        g3_no_reconfirm = not fd.get("need_confirm")
        # 标签维不被冲：本轮若刷了卡，tags 仍应含判别式符号；卡没刷(card_tags=None)=没动母题，也算守住。
        g3_tags_kept = (tags_d is None) or any("判别式" in str(t) or "符号" in str(t) for t in (tags_d or []))
        print(f"  stages={fd['stages'][:14]} cardTags={tags_d} needConfirm={bool(fd.get('need_confirm'))} err={fd['error']}")
        print(f"  texts={fd['texts'][:2]}")
        print(f"  [{'PASS' if g3_no_resolve else 'FAIL'}] 难度不触发母题重解（无 solve 灯 + ack 不含「重新解题」）")
        print(f"  [{'PASS' if g3_no_reconfirm else 'FAIL'}] 难度不重弹确认")
        print(f"  [{'PASS' if g3_tags_kept else 'FAIL'}] 已编辑的标签维不被冲掉")
        ok = ok and g3_no_resolve and g3_no_reconfirm and g3_tags_kept and not fd["error"]

        print("\n== 5) 点开始（出 ≥3 道·治回归2） ==")
        fs = await _stream(client, "开始举一反三", tid, token,
                           cfg={"start_variants": True, "mother_endorsed": True})
        n = len(fs["items"])
        blocked = any(p in t for t in fs["texts"]
                      for p in ("不能出题", "先不造题", "已暂停生成", "请补充确认", "请确认年级"))
        print(f"  stages={fs['stages'][:16]} items={n} err={fs['error']}")
        print(f"  texts={fs['texts'][:2]}")
        print(f"  [{'PASS' if n >= 3 else 'FAIL'}] 真出 ≥3 道变式（items={n}）")
        print(f"  [{'PASS' if not blocked else 'FAIL'}] 无拒造消息")
        ok = ok and n >= 3 and not blocked and not fs["error"]

    print(f"\n===== C-109 收敛修在线复现: {'GREEN' if ok else 'RED'} =====")
    return ok


if __name__ == "__main__":
    sys.path.insert(0, "src")
    res = asyncio.run(main())
    sys.exit(0 if res else 1)
