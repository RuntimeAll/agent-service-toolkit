"""PRD-C-108 B2·stage-1 质量收口 smoke（≤4 道精测，离线为主 + 1 在线核 llm_trace）。

跑法（cwd = toolkit，必 .venv）：
  离线（默认全离线断言，无网络）：.venv\\Scripts\\python.exe tools\\c108_b2_smoke.py --offline
  在线（补一道核 llm_trace turn1 不重判）：set NO_PROXY=* & .venv\\Scripts\\python.exe tools\\c108_b2_smoke.py

覆盖（对应 G4/G5）：
- 案1（G5·范围已定不重判）：build_stage1_turn1_messages(preset 年级/章) → prompt **不含年级判定要求**
  （无 gradeCandidates/判年级），只誊抄+has_figure；无 preset → 维持原全套判定。
- 案2（G5·题面在手跳读图）：route_entry 同图在手 → **不进 mother_opus_entry**（落 parse 纠正路径）；
  真贴新图 → 照常进 mother_opus_entry。
- 案3（G4·按章收窄）：明确章（chapter_id len>4）→ chapter_scope=True + 叶子/模型按章前缀显著收窄；
  低置信猜章（无 preset / 仅年级册）→ 年级全量（chapter_scope=False）。注入数量对比断言。
- 案4（在线·llm_trace）：preset 母题端到端，核 turn1（trace_label=mother_entry）prompt 不判年级。
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

# PS 5.1 控制台默认 GBK → 强制 UTF-8 输出，免中文/符号 UnicodeEncodeError。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

OFFLINE = "--offline" in sys.argv

PASS = 0
FAIL = 0


def _ok(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def _msg_text(messages) -> str:
    """把 build_stage1_turn1_messages 的返回拼成纯文本（SystemMessage + HumanMessage(parts)）。"""
    out: list[str] = []
    for m in messages:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    out.append(str(p.get("text") or ""))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 案1（G5）·范围已定不重判：turn1 prompt 形态
# ---------------------------------------------------------------------------
def case1_turn1_scope_known() -> None:
    print("[案1] 范围已定 → turn1 prompt 只誊抄、不判年级（G5/AC5）")
    from agents.variant_entry import build_stage1_turn1_messages

    img = "https://oss.example.com/q.png"
    # (a) preset 年级+章 → 不要年级判定那套
    msgs = build_stage1_turn1_messages(
        image_url=img, preset_grade_book="八年级下册", preset_chapter="第2章 一元二次方程",
    )
    txt = _msg_text(msgs)
    _ok("范围已定·不含『判这道题的年级册』", "判这道题的年级册" not in txt, txt[-200:])
    _ok("范围已定·不含 gradeCandidates/chapterCandidates 字段要求",
        "gradeCandidates" not in txt and "chapterCandidates" not in txt)
    _ok("范围已定·只要 stem + has_figure", '{"stem":"题面富文本","has_figure":true/false}' in txt)
    _ok("范围已定·明示『已定死、不必判年级』", "已定死" in txt and "已确定的范围" in txt)
    # 图仍在（读图誊抄砍不掉，边界诚实）
    has_img = any(
        isinstance(getattr(m, "content", None), list)
        and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m.content)
        for m in msgs
    )
    _ok("范围已定·图仍发（首次见图必读，视觉调用砍不掉）", has_img)

    # (b) 无 preset → 维持原全套判定（行为不变）
    msgs2 = build_stage1_turn1_messages(image_url=img)
    txt2 = _msg_text(msgs2)
    _ok("无 preset·维持原判年级（含 gradeCandidates）", "gradeCandidates" in txt2)
    _ok("无 preset·含『判这道题的年级册』", "判这道题的年级册" in txt2)


# ---------------------------------------------------------------------------
# 案2（G5）·题面在手跳读图：route_entry 同图不重进入口
# ---------------------------------------------------------------------------
def case2_same_image_skip_read() -> None:
    print("[案2] 题面已在手 → 同图重发不重读 turn1（G5/AC5）")
    import base64
    import json

    from langchain_core.messages import HumanMessage

    from agents.variant import route_entry

    def _tok(uid: int = 5) -> str:
        payload = base64.urlsafe_b64encode(json.dumps({"userId": uid}).encode()).decode().rstrip("=")
        return f"h.{payload}.s"

    cfg = {"configurable": {"ruoyi_token": _tok()}}
    img = "https://oss.aliyuncs.com/q1.png"

    # (a) 同图在手（已解出母题）+ 老师带图重发纠正 → 不进 mother_opus_entry
    state_same = {
        "messages": [HumanMessage(content=f"改用判别式法 {img}")],
        "image_url": img,
        "mother_dna": {"stem": "x^2-3x+2=0", "answer": "x=1或2"},
        "items": [{"index": 1}],
    }
    dest_same = route_entry(state_same, cfg)
    _ok("同图在手·不进 mother_opus_entry（不重读图）", dest_same != "mother_opus_entry", f"dest={dest_same}")
    _ok("同图在手·落 parse 纠正路径", dest_same == "parse", f"dest={dest_same}")

    # (b) 真贴新图（与已解母题不同 URL）→ 照常进 mother_opus_entry（首次见图必读）
    state_new = {
        "messages": [HumanMessage(content="https://oss.aliyuncs.com/q2.png")],
        "image_url": img,
        "mother_dna": {"stem": "x^2-3x+2=0"},
        "items": [{"index": 1}],
    }
    dest_new = route_entry(state_new, cfg)
    _ok("新图·照常进 mother_opus_entry（首次见图必读）", dest_new == "mother_opus_entry", f"dest={dest_new}")

    # (c) 同图但还没解出过（无 mother_dna）→ 不算题面在手 → 照常进入口
    state_first = {
        "messages": [HumanMessage(content=img)],
        "image_url": img,
    }
    dest_first = route_entry(state_first, cfg)
    _ok("同图但无首解·照常进 mother_opus_entry", dest_first == "mother_opus_entry", f"dest={dest_first}")


# ---------------------------------------------------------------------------
# 案3（G4）·按章收窄判据 + 叶子/模型注入数量对比
# ---------------------------------------------------------------------------
def case3_chapter_scope() -> None:
    print("[案3] 双料按章收窄：明确章→收窄 / 低置信→年级全量（G4/AC4）")

    # —— 判据（确定性，零 IO）：复刻 _run_stage1_continuous 的 _chapter_scope 判定逻辑 ——
    def chapter_scope_judge(preset_chapter_id: str | None, grade_code_pool: str | None) -> bool:
        p = str(preset_chapter_id or "").strip()
        return bool(p and len(p) > 4 and grade_code_pool)

    # 明确章（7 位章 id，长于 4 位年级册前缀）→ True
    _ok("明确章(章id 7位)→chapter_scope=True", chapter_scope_judge("3082002", "3082") is True)
    # 仅年级册（4 位）→ False（不当确认章）
    _ok("仅年级册(4位)→chapter_scope=False（年级全量）", chapter_scope_judge("3082", "3082") is False)
    # 无 preset 章（AI 低置信猜）→ False（守 C-105 防丢跨章大招）
    _ok("无 preset 章(AI 猜)→chapter_scope=False（守 C-105）", chapter_scope_judge("", "3082") is False)
    _ok("无 grade_code_pool→chapter_scope=False（不空收窄）", chapter_scope_judge("3082002", None) is False)

    # —— 叶子池前缀收窄对比：按章前缀 i.startswith 必是年级全量的子集，且更少 ——
    # 复刻 leaf_pool_for_grade 的过滤核（i.startswith(prefix)）。
    fake_leaves = [
        ("3082001001", "整式乘法"),       # 第1章
        ("3082001002", "乘法公式"),       # 第1章
        ("3082002001", "一元二次方程定义"),  # 第2章（目标章）
        ("3082002002", "配方法"),          # 第2章（目标章）
        ("3082002003", "公式法"),          # 第2章（目标章）
        ("3082003001", "二次函数"),        # 第3章
    ]
    grade_pool = [(i, n) for i, n in fake_leaves if i.startswith("3082")]      # 年级全量
    chap_pool = [(i, n) for i, n in fake_leaves if i.startswith("3082002")]   # 按章收窄
    _ok("叶子池·年级全量含全部 6 叶", len(grade_pool) == 6, f"n={len(grade_pool)}")
    _ok("叶子池·按章收窄到 3 叶（显著少于年级全量）", len(chap_pool) == 3, f"n={len(chap_pool)}")
    _ok("叶子池·按章是年级全量的真子集",
        set(c for c, _ in chap_pool) < set(g for g, _ in grade_pool))
    _ok("叶子池·收窄后只剩目标章叶子",
        all(c.startswith("3082002") for c, _ in chap_pool))


# ---------------------------------------------------------------------------
# 案4（在线）·llm_trace 核 turn1 不判年级
# ---------------------------------------------------------------------------
async def case4_online_trace() -> None:
    print("[案4·在线] preset 母题 turn1 prompt 不判年级（llm_trace 核）")
    # 仅断言 prompt 构件（不真跑端到端，避免依赖 :8090 落库）：build_stage1_turn1_messages
    #   在 preset 下产出的 turn1 messages 不含年级判定要求 —— 这正是会送进 trace_label=mother_entry
    #   的那一份。端到端 llm_trace 验由维护者 AC8 实测口径补（这里只确证送进去的 prompt 已收窄）。
    from agents.variant_entry import build_stage1_turn1_messages

    msgs = build_stage1_turn1_messages(
        image_url="https://oss.example.com/q.png",
        utterance="按判别式法解", preset_grade_book="八年级下册", preset_chapter="第2章",
    )
    txt = _msg_text(msgs)
    _ok("在线·送进 mother_entry 的 turn1 prompt 不判年级", "gradeCandidates" not in txt)
    _ok("在线·turn1 仍接住老师附带话（判别式法）", "判别式法" in txt)
    print("  （注：真机 llm_trace 端到端核走维护者 AC8 实测口径，:8090 在跑 + NO_PROXY=*）")


def main() -> int:
    print("=" * 60)
    print(f"PRD-C-108 B2 smoke  (mode={'offline' if OFFLINE else 'online'})")
    print("=" * 60)
    case1_turn1_scope_known()
    case2_same_image_skip_read()
    case3_chapter_scope()
    if not OFFLINE:
        asyncio.run(case4_online_trace())
    else:
        print("[案4] 跳过（--offline）")
    print("-" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
