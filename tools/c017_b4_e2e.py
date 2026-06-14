# -*- coding: utf-8 -*-
r"""PRD-C-017 B4-A · API 级全图 e2e（G5/G6 实测口径，行为点 2-5）。

直连 toolkit :8093 SSE 端点 POST /variant/stream（不走浏览器），逐行为点跑通母题两段式：
  (a) 传纯文本母题图 → 应出 needConfirm（停），不直接出变式。
  (b) 模拟老师确认（agent_config.confirmed_chapter_id=真章 id）续聊 → 应出 stage 阶段灯 +
      mother_card 专帧（在变式 items 之前）+ 变式 items。
  (c) 验 mother_card 帧字段齐（§10）；opus 真被调（查 conv_llm_trace model==claude-opus-4-8）。
  (d) 改母题一维 DNA(edit-dna index=1 改 exam_type) → regen → 验 motherDirty/regenPending/新 artifact。
  (e) 带图打回：传带图几何题 → 应 reject(with_figure) + 终止，无 needConfirm/opus/变式。

每步关键 SSE 帧落盘 tools/c017_b4_e2e_result.json，人读判定。

跑法（cwd = toolkit 根；三服务须在跑 :8093/:8090）：
  $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c017_b4_e2e.py

前置：toolkit :8093 在跑（已注 checkpointer）+ book-server :8090（classify 调底座）+ relay 可达。
凭据走 _probe_auth 真登录（不入 commit/报告）。
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

# 纯文本母题图（韦达 img3，B0 探针实测纯文本/正确/10维齐全）。章 = 浙教版第二章一元二次方程。
PURE_TEXT_IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png"
)
# 确认章 = 浙教版「第二章一元二次方程」level2 章 id（lazyTree 实查；韦达叶子在其 3082002008 下）。
CONFIRMED_CHAPTER_ID = "3082002"

# 带图几何题（B0 探针 img1 二次函数与几何综合，nano 判 has_figure=true）。
WITH_FIGURE_IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png"
)


# ---------------------------------------------------------------------------
# SSE 客户端：直连 /variant/stream，逐帧解析 data: {...} → 分类收集 custom 帧。
# ---------------------------------------------------------------------------
async def stream_variant(
    *, message: str, thread_id: str, token: str, confirmed_chapter_id: str | None = None,
    timeout: float = 240.0,
) -> dict:
    """打一轮 /variant/stream，收集所有 SSE 帧 → 归类返回。

    返回 {frames:[...原始], customs:{needConfirm/reject/artifact:[...]}, stages:[...],
          tokens_n, error:[...], raw_count}。
    """
    body: dict = {
        "message": message,
        "thread_id": thread_id,
        "stream_tokens": True,
        "agent_config": {"ruoyi_token": token},
    }
    if confirmed_chapter_id:
        body["agent_config"]["confirmed_chapter_id"] = confirmed_chapter_id

    out: dict = {
        "customs": {},          # key(needConfirm/reject/artifact/error...) -> [payload,...] (按到达序)
        "custom_order": [],     # [(key, idx_in_stream)] 保留到达顺序（验 mother_card 早于 items）
        "stages": [],           # stage 帧 [{stage,label,status,...}]
        "tokens_n": 0,
        "messages": [],         # 普通 ai message 文本
        "raw_count": 0,
        "errors": [],
    }
    seq = 0
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as cli:
        async with cli.stream(
            "POST", f"{BASE}/variant/stream", json=body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            if resp.status_code != 200:
                txt = (await resp.aread()).decode("utf-8", "ignore")[:300]
                out["errors"].append(f"HTTP {resp.status_code}: {txt}")
                return out
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                out["raw_count"] += 1
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
                mtype = msg.get("type")
                if mtype == "custom":
                    cd = msg.get("custom_data") or {}
                    for k, v in cd.items():
                        out["customs"].setdefault(k, []).append(v)
                        out["custom_order"].append((k, seq))
                        seq += 1
                        # stage 帧单列（custom_data 里 key='stage'）
                        if k == "stage":
                            out["stages"].append(v)
                elif mtype == "ai":
                    c = msg.get("content")
                    if isinstance(c, str) and c.strip():
                        out["messages"].append(c[:200])
    return out


def _summarize_frames(s: dict) -> dict:
    """从一轮 stream 结果抽人读摘要（去掉海量 token，只留关键帧）。"""
    customs = s.get("customs") or {}
    art = (customs.get("artifact") or [])
    mother_cards = []
    items_frames = []
    for a in art:
        hdr = (a or {}).get("header") or {}
        if hdr.get("mother_card"):
            mother_cards.append(hdr.get("mother_card"))
        if (a or {}).get("items"):
            items_frames.append(len((a or {}).get("items") or []))
    # mother_card 是否早于 items：看 custom_order 中首个 mother_card-bearing artifact vs 首个带 items 的
    return {
        "raw_count": s.get("raw_count"),
        "tokens_n": s.get("tokens_n"),
        "has_needConfirm": bool(customs.get("needConfirm")),
        "needConfirm": (customs.get("needConfirm") or [None])[0],
        "has_reject": bool(customs.get("reject")),
        "reject": (customs.get("reject") or [None])[0],
        "n_stage_frames": len(s.get("stages") or []),
        "stages_seen": sorted({(st or {}).get("stage") for st in (s.get("stages") or []) if st}),
        "n_artifact_frames": len(art),
        "mother_card_count": len(mother_cards),
        "mother_card": mother_cards[0] if mother_cards else None,
        "items_counts": items_frames,
        "errors": s.get("errors"),
        "ai_messages": s.get("messages"),
    }


def _check_mother_card_fields(card: dict | None) -> dict:
    """(c) 母题卡帧字段齐检（§10：stem/solution_skeleton/solved_answer/dna 10维/anchor）。"""
    if not isinstance(card, dict):
        return {"ok": False, "missing": ["__no_card__"]}
    miss: list[str] = []
    for k in ("stem", "solution_skeleton", "solved_answer"):
        if not card.get(k):
            miss.append(k)
    dna = card.get("dna") or {}
    # 10 维键对齐 _build_mother_card 的 dna 子对象（注意是 scenario 不是 scene）。
    dims = ("main_kp", "secondary_kps", "qtype", "exam_type", "difficulty",
            "scenario", "hard_point_count", "breakthrough_points", "models", "tags")
    for d in dims:
        if d not in dna:
            miss.append(f"dna.{d}")
    anchor = card.get("anchor") or {}
    for k in ("grade_book_id", "chapter_id"):
        if k not in anchor:
            miss.append(f"anchor.{k}")
    return {"ok": not miss, "missing": miss, "dna_keys": sorted(dna.keys())}


def _opus_trace_rows(thread_id: str) -> list[dict]:
    """查 conv_llm_trace 本会话所有调用 → 返回 [{label,model,relay,pt,ct,cost,dur_ms,error}]。

    G3/F1 验证 = 存在 model=='claude-opus-4-8' 的行；H4 = 取该行 token/墙钟。
    """
    rows: list[dict] = []
    try:
        conn = conv_trace._conn()
    except Exception as e:  # noqa: BLE001
        return [{"_trace_error": str(e)[:160]}]
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT label, model, relay, prompt_tokens, completion_tokens, "
                "cost_yuan, duration_ms, error, ts FROM conv_llm_trace "
                "WHERE thread_id=%s ORDER BY id",
                (thread_id,),
            )
            for r in cur.fetchall():
                rows.append({
                    "label": r[0], "model": r[1], "relay": r[2],
                    "prompt_tokens": r[3], "completion_tokens": r[4],
                    "cost_yuan": float(r[5]) if r[5] is not None else None,
                    "duration_ms": r[6], "error": r[7],
                    "ts": str(r[8]),
                })
    finally:
        conn.close()
    return rows


async def edit_dna_and_regen(thread_id: str, token: str) -> dict:
    """(d) 改母题守恒维(exam_type) → regen → 抓回流。HTTP 直连两端点（非 SSE）。"""
    out: dict = {}
    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as cli:
        # 改母题 exam_type（守恒维 → mother_dna.dirty=True + 下游变式标 dirty）
        r1 = await cli.post(
            f"{BASE}/variant/edit-dna",
            json={"thread_id": thread_id, "index": 1, "field": "exam_type", "value": "证明推理"},
        )
        out["edit_dna_status"] = r1.status_code
        d1 = r1.json() if r1.status_code == 200 else {"_err": r1.text[:200]}
        art1 = (d1 or {}).get("artifact") or {}
        hdr1 = art1.get("header") or {}
        out["after_edit"] = {
            "ok": d1.get("ok"),
            "mother_dirty": hdr1.get("mother_dirty"),
            "regen_pending": hdr1.get("regen_pending"),
        }
        # regen 待重生集合（None=全）
        r2 = await cli.post(
            f"{BASE}/variant/regen",
            json={"thread_id": thread_id, "indexes": None},
        )
        out["regen_status"] = r2.status_code
        d2 = r2.json() if r2.status_code == 200 else {"_err": r2.text[:200]}
        art2 = (d2 or {}).get("artifact") or {}
        hdr2 = art2.get("header") or {}
        out["after_regen"] = {
            "ok": d2.get("ok"),
            "regenerated": d2.get("regenerated"),
            "failed": d2.get("failed"),
            "mother_dirty_after": hdr2.get("mother_dirty"),
            "regen_pending_after": hdr2.get("regen_pending"),
            "n_items": len((art2.get("items") or [])),
        }
    return out


async def main() -> None:
    token = await real_token()
    result: dict = {"base": BASE, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ===== (a) + (b) + (c): 同一 thread —— 传纯文本图 → needConfirm → 确认续聊 → mother_card + items =====
    tid = f"c017-b4-{uuid.uuid4().hex[:8]}"
    result["thread_id"] = tid

    print("=" * 70)
    print(f"[(a)] 传纯文本母题图 → 期望 needConfirm（停，不出变式）  thread={tid}")
    s_a = await stream_variant(
        message=f"{PURE_TEXT_IMG} 帮我举一反三出3道", thread_id=tid, token=token,
    )
    sum_a = _summarize_frames(s_a)
    result["a_precheck"] = sum_a
    print(json.dumps(sum_a, ensure_ascii=False, indent=2)[:1500])

    print("=" * 70)
    print(f"[(b)] 老师确认 chapter_id={CONFIRMED_CHAPTER_ID} 续聊 → 期望 stage + mother_card + items")
    s_b = await stream_variant(
        message="确认：第二章一元二次方程，继续举一反三",
        thread_id=tid, token=token, confirmed_chapter_id=CONFIRMED_CHAPTER_ID,
    )
    sum_b = _summarize_frames(s_b)
    result["b_classify"] = sum_b
    # mother_card 早于 items？取 custom_order：首个 mother_card artifact 序 vs 首个含 items artifact 序
    mc_seq = None
    items_seq = None
    art_list = (s_b.get("customs") or {}).get("artifact") or []
    # 用 custom_order 重建 artifact 到达序
    art_i = 0
    for k, sq in s_b.get("custom_order") or []:
        if k != "artifact":
            continue
        a = art_list[art_i] if art_i < len(art_list) else {}
        art_i += 1
        hdr = (a or {}).get("header") or {}
        if hdr.get("mother_card") and mc_seq is None:
            mc_seq = sq
        if (a or {}).get("items") and items_seq is None:
            items_seq = sq
    result["b_classify"]["mother_card_seq"] = mc_seq
    result["b_classify"]["first_items_seq"] = items_seq
    result["b_classify"]["mother_card_before_items"] = (
        mc_seq is not None and (items_seq is None or mc_seq <= items_seq)
    )
    print(json.dumps({k: v for k, v in sum_b.items() if k != "mother_card"},
                     ensure_ascii=False, indent=2)[:1800])

    print("=" * 70)
    print("[(c)] 验 mother_card 字段齐 + opus 真被调（conv_llm_trace）")
    card = sum_b.get("mother_card")
    card_check = _check_mother_card_fields(card)
    result["c_mother_card_fields"] = card_check
    result["c_mother_card"] = card
    traces = _opus_trace_rows(tid)
    result["c_trace_rows"] = traces
    opus_rows = [r for r in traces if r.get("model") == "claude-opus-4-8"]
    result["c_opus_called"] = bool(opus_rows)
    result["c_opus_rows"] = opus_rows
    print(f"mother_card 字段齐 = {card_check['ok']}  缺={card_check['missing']}")
    print(f"opus 真被调 = {bool(opus_rows)}  ({len(opus_rows)} 行 model=claude-opus-4-8)")
    for r in opus_rows:
        print(f"  opus trace: label={r['label']} dur={r['duration_ms']}ms "
              f"pt={r['prompt_tokens']} ct={r['completion_tokens']} relay={r['relay']} cost={r['cost_yuan']}")

    # ===== (d) 改母题一维 DNA → regen → 回流 =====
    print("=" * 70)
    print("[(d)] 改母题 exam_type → regen → 验 motherDirty/regenPending/新 artifact")
    if (sum_b.get("items_counts") or [0])[-1:] != [0] and sum_b.get("items_counts"):
        d_out = await edit_dna_and_regen(tid, token)
    else:
        d_out = {"_skip": "(b) 未出变式 items，跳过 (d) regen"}
    result["d_edit_regen"] = d_out
    print(json.dumps(d_out, ensure_ascii=False, indent=2)[:1200])

    # ===== (e) 带图打回 =====
    tid_e = f"c017-b4e-{uuid.uuid4().hex[:8]}"
    result["thread_id_e"] = tid_e
    print("=" * 70)
    print(f"[(e)] 传带图几何题 → 期望 reject(with_figure) + 终止  thread={tid_e}")
    s_e = await stream_variant(
        message=f"{WITH_FIGURE_IMG} 帮我举一反三", thread_id=tid_e, token=token,
    )
    sum_e = _summarize_frames(s_e)
    result["e_reject"] = sum_e
    traces_e = _opus_trace_rows(tid_e)
    result["e_trace_rows"] = traces_e
    opus_e = [r for r in traces_e if r.get("model") == "claude-opus-4-8"]
    result["e_no_opus"] = not opus_e
    result["e_no_needConfirm"] = not sum_e.get("has_needConfirm")
    result["e_no_items"] = all(c == 0 for c in (sum_e.get("items_counts") or []))
    print(json.dumps({
        "has_reject": sum_e.get("has_reject"), "reject": sum_e.get("reject"),
        "has_needConfirm": sum_e.get("has_needConfirm"),
        "items_counts": sum_e.get("items_counts"),
        "opus_called": bool(opus_e), "n_traces": len(traces_e),
    }, ensure_ascii=False, indent=2))

    # ===== 落盘 =====
    outp = Path(__file__).resolve().parent / "c017_b4_e2e_result.json"
    outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"[落盘] {outp}")

    # 逐行为点 pass/fail 速判
    print("\n========== 逐行为点速判 ==========")
    a_pass = sum_a.get("has_needConfirm") and not any(sum_a.get("items_counts") or [])
    print(f"(a) needConfirm 停、不出变式 = {'PASS' if a_pass else 'FAIL'}")
    b_pass = (sum_b.get("n_stage_frames", 0) > 0 and sum_b.get("mother_card_count", 0) > 0
              and any(sum_b.get("items_counts") or []))
    print(f"(b) stage + mother_card + items = {'PASS' if b_pass else 'FAIL'}")
    c_pass = card_check["ok"] and bool(opus_rows)
    print(f"(c) 母题卡字段齐 + opus 真被调 = {'PASS' if c_pass else 'FAIL'}")
    mc_before = result['b_classify'].get('mother_card_before_items')
    print(f"    mother_card 早于 items = {mc_before}")
    d_pass = (isinstance(d_out, dict) and (d_out.get("after_edit") or {}).get("mother_dirty")
              and (d_out.get("after_regen") or {}).get("regenerated"))
    print(f"(d) edit-dna→regen 回流 = {'PASS' if d_pass else ('SKIP' if '_skip' in d_out else 'FAIL')}")
    e_pass = (sum_e.get("has_reject") and not opus_e and not sum_e.get("has_needConfirm")
              and result["e_no_items"])
    print(f"(e) 带图 reject + 零下游 = {'PASS' if e_pass else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
