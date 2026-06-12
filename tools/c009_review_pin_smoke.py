"""批1~3 整改真机冒烟（2026-06-13）· 锚定池剔复习册 + 定死硬闸 + 事实源冻结。

直接驱动编译后的 variant graph（同 c010_e2e_probe 模式）。4 场景：
  S1 正常图（年级清楚）：锚定池只含教材叶（候选 ~200-330，非 1473）、定死直通、
     generate prompt 主考点/年级真值、solve 走 gpt-5.4。
  S2 无年级线索母题：停确认不出题；补年级后继续。
  S3 带「中考复习」意图：复习册进池可锚。
  S4 定死后模拟一次 LLM 输出试图改 grade → 被忽略 + warning（直接单测 _fact_edit 取证）。

取证走 data/llm_trace.jsonl（candidate pool 行数 / generate prompt 主考点·年级 / solve model）。
跑法（toolkit 根）：$env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c009_review_pin_smoke.py
前置：book-server :8090 在跑、MySQL :3307、lk888 可达。
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import _fact_edit, graph  # noqa: E402

IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)
TRACE_PATH = Path(__file__).resolve().parent.parent / "data" / "llm_trace.jsonl"
RESULTS: list[tuple[str, bool, str]] = []
_TOKEN: str | None = None


def rec(probe: str, ok: bool, detail: str) -> None:
    RESULTS.append((probe, ok, detail[:240]))
    print(f"[{'PASS' if ok else 'FAIL'}] {probe}: {detail[:220]}")


def last_ai(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


def _trace_since(mark: str) -> list[dict]:
    """读 llm_trace.jsonl 中 ts>mark 的行（按 ISO 时间戳；seq 每进程重置不可靠）。"""
    out: list[dict] = []
    if not TRACE_PATH.exists():
        return out
    for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("ts", "")) > mark:
            out.append(r)
    return out


def _max_seq() -> str:
    """当前时刻 ISO 戳作为分界（之后写入的 trace 才算本场景产出）。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _req_text(rec: dict) -> str:
    """拼一条 trace 的请求文本（多模态取 text part）。"""
    parts = []
    for m in rec.get("request") or []:
        if m.get("text"):
            parts.append(m["text"])
        for p in m.get("parts") or []:
            if p.get("text"):
                parts.append(p["text"])
    return "\n".join(parts)


async def run_turn(app, thread_id, text, image=False):
    msg = f"{text}：{IMG}" if image else text
    cfg = {"configurable": {"thread_id": thread_id, "ruoyi_token": _TOKEN}}
    await app.ainvoke({"messages": [HumanMessage(content=msg)]}, cfg)
    return app.get_state(cfg).values


async def main() -> int:
    global _TOKEN
    from _probe_auth import real_token

    _TOKEN = await real_token()
    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)

    # ===== S1 正常图（年级清楚）=====
    mark = _max_seq()
    tid = f"smoke-s1-{uuid.uuid4().hex[:6]}"
    st = await run_turn(app, tid, "帮我对这道题举一反三，出3道", image=True)
    items = st.get("items") or []
    tr = _trace_since(mark)
    dna_tr = [t for t in tr if t.get("label") == "dna_extract"]
    gen_tr = [t for t in tr if t.get("label") == "generate"]
    solve_tr = [t for t in tr if t.get("label") == "solve"]

    # 候选池行数：dna_extract prompt 里【知识点候选池】实际行（以 id 数字开头的行）。
    # 🔴 prompt 里「候选池」字样在硬约束段也出现，故用唯一池头 `【知识点候选池】（该年级` 定界。
    def _pool_rows(rec_) -> list[str]:
        txt = _req_text(rec_)
        if "【知识点候选池】（该年级" in txt:
            seg = txt.split("【知识点候选池】（该年级", 1)[-1]
        else:
            seg = txt.split("【知识点候选池】", 1)[-1]
        seg = seg.split("【标签复用池】", 1)[0]
        return [ln.strip() for ln in seg.splitlines() if ln.strip() and ln.strip()[0].isdigit()]

    pool_n = None
    review_in_pool = False
    if dna_tr:
        rows = _pool_rows(dna_tr[-1])
        pool_n = len(rows)
        review_in_pool = any(r.startswith(("3010", "3100", "3120")) for r in rows)
    rec("S1池剔复习册", bool(pool_n and pool_n < 1473 and not review_in_pool),
        f"候选池行数={pool_n}（<1473 且无复习册前缀={not review_in_pool}）")

    rec("S1出题直通", bool(items), f"定死直通出题 {len(items)} 道（无 clarify）")

    # generate prompt 主考点/年级真值（非「未解析/未知年级」）
    gen_ok = False
    gen_detail = "无 generate trace"
    if gen_tr:
        g = _req_text(gen_tr[-1])
        has_kp = ("未解析" not in g) and ("主考点(硬守恒，不可改):" in g)
        has_grade = "未知年级" not in g
        gen_ok = has_kp and has_grade
        # 抽出主考点行
        kp_line = next((ln.strip() for ln in g.splitlines() if "主考点(硬守恒" in ln), "")
        grade_line = next((ln.strip() for ln in g.splitlines() if ln.strip().startswith("- 年级(硬守恒)")), "")
        gen_detail = f"{kp_line[:60]} | {grade_line[:40]}"
    rec("S1出题prompt真值", gen_ok, gen_detail)

    # solve 走 gpt-5.4（非 nano）
    solve_models = {t.get("model") for t in solve_tr}
    solve_ok = bool(solve_tr) and all("nano" not in str(m) for m in solve_models)
    rec("S1solve走5.4", solve_ok, f"solve trace model={solve_models or '无'}")

    # ===== S2 无年级线索母题：停确认 → 补年级继续 =====
    # 直接构造 state：母题 DNA 在但年级未知 → 经 classify/gate 停 clarify
    tid2 = f"smoke-s2-{uuid.uuid4().hex[:6]}"
    cfg2 = {"configurable": {"thread_id": tid2, "ruoyi_token": _TOKEN}}
    # 用一道纯文字母题：先贴图分析，再人为问到「无年级」难复现；改走库内 confirmed=False 路径——
    # 这里用 S1 的 patch 反查：把年级清空再问。简化：直接断言 clarify 文案含「定死」。
    # 复用 S1 thread 走 patch 清年级（mother_correction 无 grade → 触发重锚但年级仍在），
    # 故 S2 用独立桩：构造 analysis 无 grade.code 的库内母题直进 generate 防御。
    from agents.variant import generate as _gen
    st_block = await _gen({
        "mother_confirmed": True,
        "analysis": {
            "grade": {"value": None, "confidence": 0.9},
            "kp": {"value": "x", "confidence": 0.9, "anchored": {"code": "3071001"}},
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {"stem": "s", "dna": {"main_kp": {"id": "3071001", "name": "x"}}},
    }, cfg2)
    blocked = (not st_block.get("items")) and ("定死" in last_ai(st_block))
    rec("S2无年级停确认", blocked, f"缺年级拒造：reply={last_ai(st_block)[:70]}")

    # 补年级后继续：补 grade.code → 通过防御
    st_pass = await _gen({
        "mother_confirmed": True,
        "analysis": {
            "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
            "kp": {"value": "一元一次方程", "confidence": 0.9, "anchored": {"code": "3071001001001"}},
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {"stem": "2x+3=7", "answer": "x=2",
                       "dna": {"main_kp": {"id": "3071001001001", "name": "一元一次方程"},
                               "secondary_kps": [], "exam_type": "直接计算", "skeleton": ["移项"]}},
    }, cfg2)
    rec("S2补年级后出题", bool(st_pass.get("items")), f"补年级后出题 {len(st_pass.get('items') or [])} 道")

    # ===== S3 带「中考复习」意图：复习册可进池可锚 =====
    # 🔴 注意：grade_code 一旦解析出具体年级，叶子池被 scoped 到该年级（复习册同前缀才入）；
    #   复习册前缀只在「无具体年级 → 兜底全池」路径才会现身。故 S3 真机分两层取证：
    #   ① 关键词路由（_wants_review_books）确实触发 include_review_books；
    #   ② 兜底路径（无 grade_code）复习册入池——直接调 leaf_pool_for_grade 取证（真 RuoYi 树）。
    from agents.variant import _wants_review_books
    from agents.variant_support import RuoyiClient, _is_review_book, leaf_pool_for_grade

    kw_fired = _wants_review_books("这是中考复习题，帮我举一反三出2道")
    cl = RuoyiClient(token=_TOKEN)
    try:
        pool_closed = await leaf_pool_for_grade(None, cl, include_review_books=False)
        pool_open = await leaf_pool_for_grade(None, cl, include_review_books=True)
    finally:
        await cl.aclose()
    review_closed = any(_is_review_book(i) for i, _ in pool_closed)
    review_open = any(_is_review_book(i) for i, _ in pool_open)
    s3 = kw_fired and (not review_closed) and review_open and len(pool_open) > len(pool_closed)
    rec("S3复习册进池", s3,
        f"关键词触发={kw_fired} 教材池复习册={review_closed}(应False) 开放池复习册={review_open}(应True) "
        f"池大小 {len(pool_closed)}→{len(pool_open)}")

    # ===== S4 定死后 LLM 改 grade 被忽略 + warning =====
    import logging
    captured = []

    class _H(logging.Handler):
        def emit(self, r):
            captured.append(r.getMessage())

    flog = logging.getLogger("variant.facts")
    flog.addHandler(_H())
    flog.setLevel(logging.WARNING)
    analysis = {"grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"}}
    ok = _fact_edit(analysis, "grade", "九年级下学期", source="llm", locked=True, audit=[])
    s4 = (ok is False) and analysis["grade"]["value"] == "七年级上学期" and any(
        "facts_locked" in m for m in captured
    )
    rec("S4锁后LLM回写被忽略", s4,
        f"ignored={not ok} 值未变={analysis['grade']['value']} warn={any('facts_locked' in m for m in captured)}")

    print("\n" + "=" * 60)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"SUMMARY: {npass}/{len(RESULTS)} PASS")
    for probe, ok, det in RESULTS:
        if not ok:
            print(f"  FAIL {probe}: {det}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
