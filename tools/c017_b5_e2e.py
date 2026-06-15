# -*- coding: utf-8 -*-
r"""PRD-C-017 B5-toolkit · API 级全图 e2e（母题卡硬停闸 + resume）。

直连 toolkit :8093 SSE /variant/stream 跑通 B5 新两段式：
  (a) 传纯文本母题图 → needConfirm（停，不出变式）。
  (b) 老师确认 chapter_id 续聊 → 期望 mother_card 专帧 + 阶段灯 await 中性态 +
      **awaiting_mother_review 停（无变式 items）**。
  (c) 老师点「开始举一反三」（agent_config.start_variants=True）续聊 → 期望出变式 items，
      且 **opus 不被二次调用**（复用 thread state 的 mother_dna，直奔 generate）。

落盘 tools/c017_b5_e2e_result.json。

跑法（cwd=toolkit；:8093 + :8090 在跑）：
  $env:PYTHONUTF8='1'; .venv/Scripts/python.exe tools/c017_b5_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from _probe_auth import real_token  # noqa: E402
from agents import conv_trace  # noqa: E402

BASE = "http://localhost:8093"
PURE_TEXT_IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png"
)
CONFIRMED_CHAPTER_ID = "3082002"


async def stream_variant(*, message, thread_id, token, confirmed_chapter_id=None,
                         start_variants=False, timeout=240.0):
    body = {
        "message": message, "thread_id": thread_id, "stream_tokens": True,
        "agent_config": {"ruoyi_token": token},
    }
    if confirmed_chapter_id:
        body["agent_config"]["confirmed_chapter_id"] = confirmed_chapter_id
    if start_variants:
        body["agent_config"]["start_variants"] = True
    out = {"customs": {}, "stages": [], "tokens_n": 0, "messages": [], "errors": []}
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as cli:
        async with cli.stream("POST", f"{BASE}/variant/stream", json=body,
                              headers={"Accept": "text/event-stream"}) as resp:
            if resp.status_code != 200:
                out["errors"].append(f"HTTP {resp.status_code}: {(await resp.aread()).decode('utf-8','ignore')[:300]}")
                return out
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                except Exception:  # noqa: BLE001
                    continue
                typ = evt.get("type")
                if typ == "token":
                    out["tokens_n"] += 1
                    continue
                if typ == "error":
                    out["errors"].append(evt.get("content"))
                    continue
                if typ != "message":
                    continue
                msg = evt.get("content") or {}
                if msg.get("type") == "custom":
                    for k, v in (msg.get("custom_data") or {}).items():
                        out["customs"].setdefault(k, []).append(v)
                        if k == "stage":
                            out["stages"].append(v)
                elif msg.get("type") == "ai":
                    c = msg.get("content")
                    if isinstance(c, str) and c.strip():
                        out["messages"].append(c[:200])
    return out


def _summary(s):
    customs = s.get("customs") or {}
    art = customs.get("artifact") or []
    mcards, items = [], []
    for a in art:
        hdr = (a or {}).get("header") or {}
        if hdr.get("mother_card"):
            mcards.append(hdr["mother_card"])
        if (a or {}).get("items"):
            items.append(len(a["items"]))
    return {
        "has_needConfirm": bool(customs.get("needConfirm")),
        "stages": [(st or {}).get("status") for st in (s.get("stages") or [])],
        "stage_statuses_set": sorted({(st or {}).get("status") for st in (s.get("stages") or []) if st}),
        "mother_card_count": len(mcards),
        "mother_card": mcards[0] if mcards else None,
        "items_counts": items,
        "ai_messages": s.get("messages"),
        "errors": s.get("errors"),
    }


def _opus_rows(thread_id):
    try:
        conn = conv_trace._conn()
    except Exception as e:  # noqa: BLE001
        return [{"_trace_error": str(e)[:160]}]
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT label, model, duration_ms FROM conv_llm_trace "
                "WHERE thread_id=%s ORDER BY id", (thread_id,))
            return [{"label": r[0], "model": r[1], "dur_ms": r[2]} for r in cur.fetchall()]
    finally:
        conn.close()


async def main():
    token = await real_token()
    result = {"base": BASE, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    tid = f"c017-b5-{uuid.uuid4().hex[:8]}"
    result["thread_id"] = tid

    print("=" * 70)
    print(f"[(a)] 纯文本图 → needConfirm（停）  thread={tid}")
    s_a = await stream_variant(message=f"{PURE_TEXT_IMG} 帮我举一反三出3道", thread_id=tid, token=token)
    sum_a = _summary(s_a)
    result["a_precheck"] = sum_a
    print(json.dumps({k: v for k, v in sum_a.items() if k != "mother_card"}, ensure_ascii=False, indent=2)[:1000])

    print("=" * 70)
    print(f"[(b)] 确认 chapter_id={CONFIRMED_CHAPTER_ID} → 期望 mother_card + await 停（无变式）")
    s_b = await stream_variant(message="确认：第二章一元二次方程",
                               thread_id=tid, token=token, confirmed_chapter_id=CONFIRMED_CHAPTER_ID)
    sum_b = _summary(s_b)
    result["b_mother_card_stop"] = sum_b
    print(json.dumps({k: v for k, v in sum_b.items() if k != "mother_card"}, ensure_ascii=False, indent=2)[:1400])
    rows_b = _opus_rows(tid)
    opus_b = [r for r in rows_b if r.get("model") == "claude-opus-4-8"]
    result["b_opus_calls"] = len(opus_b)
    card = sum_b.get("mother_card") or {}
    result["b_card_fields"] = {
        "answer": bool(card.get("answer")),
        "solved_answer": bool(card.get("solved_answer")),
        "analysis": bool(card.get("analysis")),
        "difficulty": card.get("difficulty"),
    }

    print("=" * 70)
    print("[(c)] 点开始（start_variants=True） → 期望出变式 items，opus 不二次调用")
    s_c = await stream_variant(message="开始举一反三", thread_id=tid, token=token, start_variants=True)
    sum_c = _summary(s_c)
    result["c_variants"] = sum_c
    print(json.dumps({k: v for k, v in sum_c.items() if k != "mother_card"}, ensure_ascii=False, indent=2)[:1400])
    rows_c = _opus_rows(tid)
    opus_total = len([r for r in rows_c if r.get("model") == "claude-opus-4-8"])
    result["c_opus_total_after"] = opus_total
    result["c_opus_no_second_call"] = (opus_total == len(opus_b))  # resume 不重调 opus

    outp = Path(__file__).resolve().parent / "c017_b5_e2e_result.json"
    outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 逐行为点速判 ==========")
    a_pass = sum_a.get("has_needConfirm") and not any(sum_a.get("items_counts") or [])
    print(f"(a) needConfirm 停、不出变式 = {'PASS' if a_pass else 'FAIL'}")
    b_pass = (sum_b.get("mother_card_count", 0) > 0 and not any(sum_b.get("items_counts") or [])
              and "warn" not in [(st or {}) for st in sum_b.get("stage_statuses_set", [])])
    print(f"(b) mother_card 出 + 无变式（硬停闸）= {'PASS' if b_pass else 'FAIL'}")
    print(f"    阶段灯 statuses = {sum_b.get('stage_statuses_set')}  (期望含 await，不含 warn)")
    print(f"    母题卡 answer/solved/analysis/difficulty = {result['b_card_fields']}")
    c_pass = any(sum_c.get("items_counts") or []) and result["c_opus_no_second_call"]
    print(f"(c) 点开始→出变式 + opus 不二次调用 = {'PASS' if c_pass else 'FAIL'}")
    print(f"    opus 调用数 b={len(opus_b)} → 总={opus_total}（应相等=不重调）")
    print(f"\n[落盘] {outp}")


if __name__ == "__main__":
    asyncio.run(main())
