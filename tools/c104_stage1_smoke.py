# -*- coding: utf-8 -*-
"""PRD-C-104 B4·stage1 e2e 冒烟（本批命脉，4 道精测网覆盖不到 stage1）。

真打运行中的 :8093 /variant/stream：发一张母题图 → 断言 stage1（solve+label+mother_card）真跑通：
  ① 有 stage 帧流（锚定/解析配方等阶段灯）→ 证 classify(label) 跑了；
  ② 出 mother_card 帧 → 证 _emit_mother_card / _build_mother_card 跑了；
  ③ 母题卡含「solved 答案（answer/solved_answer 任一非空）+ DNA 维（主考点 main_kp + 题型 qtype）」
     → 证 solve（阅卷解答经 classify 回填 mdna.answer/solved_answer）+ label（DNA 主考点/题型锚定）真跑通。
红=停报告别 commit③。B5 复用本工具。

跑法：cd <root> && PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c104_stage1_smoke.py
前置：:8093 在跑（health=200）、:8080 RuoYi 在跑（classify 拉叶子池）、:3307。
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import httpx  # noqa: E402
from agents.variant_support import RuoyiClient  # noqa: E402

BASE = "http://localhost:8093"
# 复用既有冒烟同款母题图（COS 真题图，stage1 走 opus 直读图解题+10维打标）。
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)


async def stream_call(client, message, thread_id, token):
    body = {
        "message": message,
        "stream_tokens": True,
        "thread_id": thread_id,
        "agent_config": {"ruoyi_token": token},
    }
    frames = {
        "raw_events": 0,
        "stage": 0,
        "stage_keys": set(),
        "mother_card_frames": 0,
        "mother_card": None,  # 最后一份母题卡 payload（含 dna/answer）
        "error": None,
    }
    async with client.stream(
        "POST", f"{BASE}/variant/stream", json=body,
        headers={"Content-Type": "application/json"}, timeout=300,
    ) as resp:
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" not in ctype:
            txt = (await resp.aread())[:300]
            frames["error"] = f"非SSE: status={resp.status_code} ct={ctype} body={txt!r}"
            return frames
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
            # 帧体可能层层嵌套（custom artifact / stage / token）→ 全文 walk 找 stage / mother_card。
            _walk(obj, frames)
    frames["stage_keys"] = sorted(frames["stage_keys"])
    return frames


def _walk(obj, frames):
    """递归扫帧体：累计 stage 帧 + 抓 header.mother_card payload + 抓 error。"""
    if isinstance(obj, dict):
        # stage 帧：custom_data.stage = {"key":..,"title":..,"status":..,"detail":..}（阶段灯）。
        #   服务层把 ChatMessage(role=custom, content=[{"stage": <dict>}]) → custom_data.stage（dict）。
        st = obj.get("stage")
        if st is not None and not isinstance(st, (dict, str)):
            st = None
        if isinstance(st, dict):
            frames["stage"] += 1
            k = st.get("key") or st.get("title")
            if k:
                frames["stage_keys"].add(str(k))
        elif isinstance(st, str) and st:
            frames["stage"] += 1
            frames["stage_keys"].add(st)
        # mother_card 帧：artifact.header.mother_card（_emit_mother_card / _artifact_payload）
        header = obj.get("header")
        if isinstance(header, dict) and isinstance(header.get("mother_card"), dict):
            frames["mother_card_frames"] += 1
            frames["mother_card"] = header["mother_card"]
        if "mother_card" in obj and isinstance(obj.get("mother_card"), dict):
            frames["mother_card_frames"] += 1
            frames["mother_card"] = obj["mother_card"]
        # error 帧
        if obj.get("error") and isinstance(obj.get("error"), (str, dict)):
            frames["error"] = frames["error"] or str(obj.get("error"))[:160]
        for v in obj.values():
            _walk(v, frames)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, frames)


def _assert_mother_card(mc):
    """断言母题卡含 solved 答案 + DNA 维（主考点 + 题型）。返回 (ok, detail)。"""
    if not isinstance(mc, dict):
        return False, "无 mother_card payload"
    answer = (mc.get("answer") or mc.get("solved_answer") or "").strip()
    dna = mc.get("dna") or {}
    main_kp = (dna.get("main_kp") or mc.get("main_kp") or "")
    main_kp = str(main_kp).strip()
    qtype = str(dna.get("qtype") or mc.get("qtype") or "").strip()
    ok = bool(answer) and bool(main_kp) and bool(qtype)
    detail = {
        "answer_present": bool(answer),
        "answer_preview": answer[:40],
        "main_kp": main_kp,
        "qtype": qtype,
        "stem_present": bool((mc.get("stem") or "").strip()),
    }
    return ok, detail


async def main():
    client0 = RuoyiClient()
    token = await client0.login()
    await client0.aclose()
    print(f"[token ok] IMG={IMG[:72]}")
    async with httpx.AsyncClient(trust_env=False) as client:
        tid = "c104-stage1-smoke-1"
        print("\n--- stage1 e2e：发母题图（solve+label+mother_card） ---")
        r = await stream_call(client, f"{IMG} 出2道", tid, token)

    mc = r.get("mother_card")
    card_ok, card_detail = _assert_mother_card(mc)
    summary = {
        "raw_events": r["raw_events"],
        "stage_frames": r["stage"],
        "stage_keys": r["stage_keys"],
        "mother_card_frames": r["mother_card_frames"],
        "error": r["error"],
        "mother_card_assert": card_detail,
    }
    print("\n帧型计数 + 母题卡关键字段：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    has_stage = r["stage"] > 0
    has_mc_frame = r["mother_card_frames"] > 0
    no_error = not r["error"]
    ok = has_stage and has_mc_frame and card_ok and no_error
    print(
        "\nstage1 e2e 冒烟："
        f"stage帧={'Y' if has_stage else 'N'} "
        f"mother_card帧={'Y' if has_mc_frame else 'N'} "
        f"卡含answer+主考点+题型={'Y' if card_ok else 'N'} "
        f"无error={'Y' if no_error else 'N'} "
        f"→ {'PASS' if ok else 'FAIL'}"
    )
    sys.exit(0 if ok else 1)


asyncio.run(main())
