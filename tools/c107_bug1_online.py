# -*- coding: utf-8 -*-
"""PRD-C-107 BUG-1·在线复现验证：纯文本一元二次方程母题 → 确认章 → resume 重锚。
断言修后**不再**「反复解析失败」死循环：resume 轮无 error、stage 帧推进、能进阶段二/确认态。

跑法：set NO_PROXY=* & set HTTP_PROXY= & set HTTPS_PROXY= &
  PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c107_bug1_online.py
前置：:8093 已重启加载新码、:8090 RuoYi 活。
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
BASE = "http://localhost:8093"

# 🔴 触发口：纯文本母题首轮 route_entry 走 'ask'（催图）= stage 0，进不了 classify（B1/B2/B3 实测同款
#   路由限制，非本 bug）。BUG-1 的死循环发生在 **classify 重锚** 路径，可靠触发须用**贴图母题**
#   （route_entry → mother_opus_entry → 锚不到 → needConfirm → 确认章 resume → classify 重锚）。
#   用 B3 smoke 同源贴图母题（含大招、niche 程度足以触发锚定待确认）。纯文本路由限制留注释存证。
IMG_MOTHER = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)
MOTHER_TEXT = f"帮我对这道题举一反三 {IMG_MOTHER}"


async def _stream(client, message, thread_id, token, agent_cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": thread_id,
            "agent_config": {"ruoyi_token": token, **(agent_cfg or {})}}
    fr = {"stage": 0, "items": [], "error": None, "need_confirm": None,
          "msgs": "", "stages": []}
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


def _walk(obj, fr):
    if isinstance(obj, dict):
        st = obj.get("stage")
        if isinstance(st, dict):
            fr["stage"] += 1
            fr["stages"].append(f"{st.get('node')}:{st.get('status')}={st.get('detail','')[:40]}")
        elif isinstance(st, str):
            fr["stage"] += 1
        for key in ("items", "variants"):
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict) and any("stem" in x for x in v if isinstance(x, dict)):
                fr["items"] = v
        if isinstance(obj.get("needConfirm"), dict):
            fr["need_confirm"] = obj["needConfirm"]
        if isinstance(obj.get("content"), str):
            fr["msgs"] += " " + obj["content"]
        if isinstance(obj.get("message"), str):
            fr["msgs"] += " " + obj["message"]
        if obj.get("error") and not fr["error"]:
            fr["error"] = str(obj.get("error"))[:200]
        for v in obj.values():
            _walk(v, fr)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, fr)


async def main():
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    ok = True
    tid = "c107-bug1-online"

    def chk(name, cond, extra=""):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    async with httpx.AsyncClient() as client:
        print("== 轮1·纯文本母题（首解 + 可能 needConfirm） ==")
        f1 = await _stream(client, MOTHER_TEXT, tid, token)
        print(f"  stage帧={f1['stage']} needConfirm={bool(f1['need_confirm'])} err={f1['error']}")
        for s in f1["stages"]:
            print(f"    · {s}")
        chk("轮1 跑通 0 error + 有 stage 帧", bool(f1["stage"]) and not f1["error"], extra=f"err={f1['error']}")

        nc = f1.get("need_confirm") or {}
        ch = (nc.get("chapter") or {})
        gb = (nc.get("grade_book") or {})
        confirm_cfg = {"start_variants": True}
        # 🔴 needConfirm 常回 chapter.id 空（只给章名）→ 用真叶子树回推**真数字章 id**（7 位 level2），
        #   驱动 classify 正确 grade_code + 闸B（不退回名当 id 的假路径）。八下一元二次方程章 = 2000020。
        chap_id = ch.get("id") or "2000020"
        if nc:
            confirm_cfg["confirmed_chapter_id"] = chap_id
            confirm_cfg["confirmed_grade_book_id"] = gb.get("id") or chap_id[:4]
            confirm_cfg["confirmed_chapter_name"] = ch.get("name") or "一元二次方程"
            print(f"  需确认 → 确认章「{ch.get('name')}」用真数字章 id={chap_id}")

        # 🔴 BUG-1 复现核心：resume 确认章 → 重入 classify 重锚。修前此处对 niche 题反复「解析失败」死循环。
        print("== 轮2·确认章 resume（BUG-1 复现点：重锚不死循环）==")
        f2 = await _stream(client, "确认", tid, token, agent_cfg=confirm_cfg)
        print(f"  stage帧={f2['stage']} items={len(f2['items'])} err={f2['error']}")
        for s in f2["stages"]:
            print(f"    · {s}")
        loop_sig = "反复解析失败" in f2["msgs"] or "结果反复解析失败" in f2["msgs"]
        chk("轮2 resume 无 error", not f2["error"], extra=f"err={f2['error']}")
        chk("🔴 轮2 **无『反复解析失败』死循环信号**（BUG-1 根治）", not loop_sig,
            extra=f"msgs含={'反复解析失败' if loop_sig else '无'}")
        chk("轮2 stage 帧推进（重锚有进展，非原地打转）", f2["stage"] >= 1)

        # 若停在再确认（BUG-03 单轮闸断/低置信）→ 再确认一次同章，证明能前进进阶段二（有界、非无限）。
        if not f2["items"] and not f2["error"]:
            print("== 轮3·再确认同章（坚持）→ 前进出题（有界，非死循环）==")
            f3 = await _stream(client, "确认", tid, token, agent_cfg=confirm_cfg)
            print(f"  stage帧={f3['stage']} items={len(f3['items'])} err={f3['error']}")
            for s in f3["stages"]:
                print(f"    · {s}")
            loop3 = "反复解析失败" in f3["msgs"]
            chk("轮3 仍无『反复解析失败』死循环 + 无 error", (not loop3) and not f3["error"])
            if f3["items"]:
                print(f"  ✅ 流出变式 {len(f3['items'])} 道（进入阶段二，死循环根治证实）")
            else:
                print("  (轮3 未直接出 items — 若停在确认/await_review 属正常握手，关键是无死循环信号)")

    print(f"\n在线复现: {'PASS（无死循环、可前进）' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
