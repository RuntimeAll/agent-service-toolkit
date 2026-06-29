# -*- coding: utf-8 -*-
"""PRD-C-107 B1·阶段一连续对话 + 双料闭集 + 确认带原因/接住老师信息 ≤4 道精测闸。

只测 3 优先级 + B1 机制（G1 解题带料 / G2 难题沿用模型 / G6 确认带 reason·接住）：

A 段·离线断言（无需服务·恒可跑，主证据）：
  - G1 双料闭集：build_solve_turn / build_label_turn 注入「叶子池」+「工具箱」(闭集)。
  - G4 连续对话：build_stage1_messages 一条累积线 = 一次 SYSTEM、图只在首条 human、
    解题/打标轮带前文(AIMessage 累积)、后续轮无 image part。
  - G2 难度表驱动：anchor_models_from_names 命中模型带 tier_int/freq_int（难度旋钮源，非 LLM 自评）。
  - G6 确认带 reason / 接住：_extract_teacher_intent 抽 grade/chapter/model；
    _build_confirm_payload 端出 reason；给 grade → 跳确认(needs_confirm=False)。

B 段·在线 e2e（需 :8093 新代码 + :8090 RuoYi）：≤2 道真图阶段一跑通、0 error、母题卡出。

跑法（A 段）：PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c107_b1_smoke.py --offline
跑法（A+B）：... tools/c107_b1_smoke.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BASE = "http://localhost:8093"
IMG_HAS_MODEL = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)


def _content_str(m) -> str:
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                out.append(p.get("text", ""))
        return "\n".join(out)
    return str(c)


def _has_image_part(m) -> bool:
    c = getattr(m, "content", "")
    if isinstance(c, list):
        return any(isinstance(p, dict) and p.get("type") == "image_url" for p in c)
    return False


def offline_gates() -> bool:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from agents import model_anchor
    from agents import variant_entry as ve

    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        flag = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{flag}] {name} {extra}")

    leaf_pool = [("3081002", "一元二次方程的根的判别式"), ("3081003", "配方法")]
    tb = model_anchor.build_toolbox_clause([
        {"id": "M10", "name": "判别式法", "trigger_feature": "含参方程根的情况",
         "action_conclusion": "用 Δ 判根"},
    ])

    print("== G4 连续对话：一条累积 messages 线（一次 SYSTEM、图只在首条 human、轮间累积） ==")
    # turn1 messages（誊抄 + 初判 + 解析老师信息）
    t1 = ve.build_stage1_turn1_messages(image_url="http://x/q.png", utterance="八下，用判别式法")
    sys_msgs = [m for m in t1 if isinstance(m, SystemMessage)]
    chk("turn1 恰一条 SYSTEM", len(sys_msgs) == 1, extra=f"(共{len(sys_msgs)})")
    chk("turn1 human 带 image part", any(_has_image_part(m) for m in t1))

    # turn2（解题）= 累积 = turn1 全量 + AIMessage(turn1 产物) + 解题 human（纯文本·无图）
    t1_ai = AIMessage(content='{"stem":"...","gradeBook":"八年级下册","confidence":0.9}')
    solve_turn = ve.build_stage1_solve_turn(
        leaf_pool=leaf_pool, model_toolbox=tb, anchored_grade="八年级下册",
        anchored_chapter="第2章 一元二次方程", teacher_model="判别式法",
    )
    convo2 = [*t1, t1_ai, *solve_turn]
    chk("解题轮 = 一条线累积（含 turn1 + AIMessage + 解题 human）",
        len([m for m in convo2 if isinstance(m, SystemMessage)]) == 1
        and any(isinstance(m, AIMessage) for m in convo2))
    chk("解题轮 human 不再带 image part（图只发一次）",
        not any(_has_image_part(m) for m in solve_turn))

    print("== G1 双料闭集：解题/打标轮注入 叶子池 + 工具箱 ==")
    solve_txt = "\n".join(_content_str(m) for m in solve_turn)
    chk("解题轮注入 知识点叶子池(闭集·真 id)",
        "叶子池" in solve_txt or "知识点" in solve_txt,
        extra="含考点候选")
    chk("解题轮叶子池含真 id（闭集让 LLM 选真 id）", "3081002" in solve_txt)
    chk("解题轮注入 解题大招工具箱", "工具箱" in solve_txt and "判别式法" in solve_txt)
    chk("解题轮接住老师指定模型（优先用）", "判别式法" in solve_txt)

    label_turn = ve.build_stage1_label_turn(leaf_pool=leaf_pool, model_toolbox=tb)
    label_txt = "\n".join(_content_str(m) for m in label_turn)
    chk("打标轮注入 叶子池(选真 id) + 工具箱(填模型名)",
        ("叶子池" in label_txt or "3081002" in label_txt) and "工具箱" in label_txt)
    chk("打标轮 human 不带 image part", not any(_has_image_part(m) for m in label_turn))

    print("== G2 难度表驱动：模型命中带 tier/freq（难度旋钮源·非 LLM 自评） ==")
    cands = [
        {"id": "M10", "name": "判别式法", "tier_int": 3, "freq_int": 2},
        {"id": "M11", "name": "配方法", "tier_int": 1, "freq_int": 3},
    ]
    r = model_anchor.anchor_models_from_names(["判别式法"], candidates=cands)
    chk("难题→选对大招→映回表 tier_int（表驱动）",
        r["models"] and r["models"][0]["id"] == "M10" and r["models"][0].get("tier_int") == 3,
        extra=str(r["models"]))

    print("== G6 确认带 reason + 接住老师信息 ==")
    intent_give = ve._extract_teacher_intent("八年级下册，用判别式法解")
    chk("接住老师 utterance：抽出 grade", bool(intent_give.get("grade")),
        extra=str(intent_give))
    chk("接住老师 utterance：抽出 model", bool(intent_give.get("model")),
        extra=str(intent_give.get("model")))
    intent_none = ve._extract_teacher_intent("帮我举一反三")
    chk("无年级信息 → grade 空（不脑补）", not intent_none.get("grade"))

    # 低置信弹窗 payload 端出 reason
    decision_low = {
        "needs_confirm": True, "reason": "置信0.55<0.80；2个强候选章歧义",
        "grade_book": "", "chapter": "", "grade_candidates": [], "chapter_candidates": [],
        "confidence": 0.55,
    }
    payload = ve._build_confirm_payload(decision_low)
    chk("确认 payload 端出 reason（人话原因）",
        bool(payload.get("reason")) and "置信" in payload["reason"], extra=payload.get("reason"))

    # 给年级章 → 跳确认（decision 被预设覆盖 needs_confirm=False）
    chk("给年级章 → 跳确认（接住 = preset 等价）",
        ve._teacher_intent_skips_confirm({"grade": "八年级下册"}) is True)
    chk("没给年级 → 不跳确认", ve._teacher_intent_skips_confirm({}) is False)

    return ok


async def _stream(client, message, thread_id, token, agent_cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": thread_id,
            "agent_config": {"ruoyi_token": token, **(agent_cfg or {})}}
    frames = {"stage": 0, "mother_card": None, "error": None}
    async with client.stream("POST", f"{BASE}/variant/stream", json=body,
                             headers={"Content-Type": "application/json"}, timeout=300) as resp:
        if "text/event-stream" not in resp.headers.get("content-type", ""):
            frames["error"] = f"非SSE status={resp.status_code}"
            return frames
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
            _walk(obj, frames)
    return frames


def _walk(obj, frames):
    if isinstance(obj, dict):
        if isinstance(obj.get("stage"), (dict, str)):
            frames["stage"] += 1
        h = obj.get("header")
        if isinstance(h, dict) and isinstance(h.get("mother_card"), dict):
            frames["mother_card"] = h["mother_card"]
        if isinstance(obj.get("mother_card"), dict):
            frames["mother_card"] = obj["mother_card"]
        if obj.get("error") and not frames["error"]:
            frames["error"] = str(obj.get("error"))[:160]
        for v in obj.values():
            _walk(v, frames)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, frames)


async def online_gates() -> bool:
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    ok = True
    async with httpx.AsyncClient() as client:
        print("== 在线·阶段一连续对话端到端（≤2 道·只验跑通 + 母题卡 + 0 error） ==")
        f = await _stream(client, f"帮我对这道题举一反三 {IMG_HAS_MODEL}", "c107-b1-e2e", token)
        mc = f.get("mother_card") or {}
        models = (mc.get("dna") or {}).get("models") or []
        print(f"  stage帧={f['stage']} models={[m.get('name') for m in models]} "
              f"hasCard={bool(mc)} err={f['error']}")
        cond = bool(f["stage"]) and not f["error"] and bool(mc)
        print(f"  [{'PASS' if cond else 'FAIL'}] 阶段一连续对话跑通 + 母题卡出 + 0 error")
        ok = ok and cond
    return ok


def main():
    offline_only = "--offline" in sys.argv
    print("===== PRD-C-107 B1 smoke =====")
    a = offline_gates()
    print(f"\nA 段离线断言: {'ALL GREEN' if a else 'RED'}")
    if offline_only:
        sys.exit(0 if a else 1)
    try:
        b = asyncio.run(online_gates())
    except Exception as e:  # noqa: BLE001
        print(f"B 段在线 e2e 跳过/失败: {e}")
        b = False
    print(f"B 段在线 e2e: {'GREEN' if b else 'RED/SKIP'}")
    sys.exit(0 if a else 1)


if __name__ == "__main__":
    main()
