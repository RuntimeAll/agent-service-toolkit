# -*- coding: utf-8 -*-
"""PRD-C-107 BUG-2·确认章后母题降级锚到章级(待人审)进不了阶段二·0 变式 修复 ≤4 道精测闸。

root cause（在线 + 离线双证）：BUG-1 修后 niche 母题首解锚不到叶子 → 老师确认章 → graceful 降级
锚到确认章节点（待人审）。但 _reanchor_reuse_first_solve 的降级分支**只在 dna.qtype 非空时**抬
analysis.qtype 置信（`if dna.get("qtype")`）。niche 首解 DNA 常**无 qtype** → 第三锚 qtype 缺 →
_conf_ok（要求 grade/kp/qtype 三锚齐）=False → mother_confirmed=False → gate_after_classify 走
clarify（而非 await_review）→ awaiting_mother_review **永不置位** → 老师点「开始举一反三」时
route_entry 落 parse 兜底 → 0 变式、「不知道你想改什么」+ FE「母题确认已暂停」。
（BUG-1 治了 main_kp 锚定；qtype 锚定是同一「首解锚叶 ⊥ 三锚齐」正交性的下一层。）

修法（最小面·节点内·不动图拓扑）：degraded 降级分支补齐第三锚——有真 qtype 用真的；degraded
且无 qtype → 安全默认「解答题」(待人审) 抬置信 + 回填 DNA（与 _bounded_degrade_to_chapter 同口径）。
非 degraded（锚真叶子）路径保持原行为。

A 段·离线断言（无需服务·主证据）：
跑法：PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c107_bug2_smoke.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CHAP = "3082002"
GRADE = "3082"
LEAF_POOL = [("3082002009001", "其它叶子A"), ("3082002009002", "其它叶子B")]


def _cap(L):
    import agents.variant as V
    noop_stage = lambda *a, **k: None
    noop = lambda *a, **k: None
    for mod in (L, V):
        mod._emit_stage = noop_stage
        mod._emit_need_confirm = noop
        mod._emit_mother_card = noop
        mod._emit_figure_stage = noop


async def _reanchor(prev_dna, analysis, insist=True):
    from agents.variant.stage1_anchor import label as L
    _cap(L)
    state = {"messages": [], "knobs": None, "mother_dna": {"dna": prev_dna}}
    if insist:
        state["_bug03_gated_chapter"] = CHAP
    return await L._reanchor_reuse_first_solve(
        state=state, analysis=dict(analysis), mother_dna={"dna": prev_dna},
        prev_dna=prev_dna, grade_code=GRADE, chapter_id=CHAP, leaf_pool=LEAF_POOL,
        confirmed_chapter_id=CHAP, include_review_books=False, knobs=None,
    )


def offline_gates() -> bool:
    import agents.variant as V
    from agents.variant.entry import route as RT
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        if not cond:
            ok = False
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    print("== ① degraded 无 qtype（BUG-2 真因）→ 补默认 qtype → confirmed → await_review → generate ==")
    # niche 首解 DNA：main_kp name 空 + **无 qtype**；analysis 也无 qtype 节点。
    prev = {"main_kp": {"id": "9999", "name": ""}, "secondary_kps": [],
            "skeleton": ["移项"], "model_candidates": []}
    analysis = {"kp": {}, "grade": {"value": "八年级下册", "code": GRADE}}
    out = asyncio.run(_reanchor(prev, analysis))
    full = {**{"messages": [], "knobs": None, "mother_dna": {"dna": prev}, "_bug03_gated_chapter": CHAP}, **out}
    an = full.get("analysis") or {}
    dna = (out.get("mother_dna") or {}).get("dna") or {}
    chk("第三锚 qtype 抬置信（补默认「解答题」·待人审）",
        (an.get("qtype") or {}).get("value") == "解答题"
        and float((an.get("qtype") or {}).get("confidence") or 0) >= V.CONF_GATE)
    chk("DNA 回填 qtype（FE/下游一致）", dna.get("qtype") == "解答题")
    chk("_conf_ok=True（三锚齐）", V._conf_ok(an) is True)
    chk("mother_confirmed=True", out.get("mother_confirmed") is True)
    chk("gate_after_classify=await_review（不再 clarify）",
        RT.gate_after_classify(full) == "await_review")
    chk("守恒白名单非空·不暂停生成", V._conservation_blocked(dna) is None,
        extra=f"wl={V._kp_whitelist(dna)}")

    # await_review 置位 + 老师点开始 → route_entry → generate（修前此处落 parse=0 变式）
    async def _route():
        rev = await V.await_mother_review(full, {"configurable": {}})
        f2 = {**full, **rev}
        ct = V.conv_trace; o = ct.teacher_id_from_token
        ct.teacher_id_from_token = lambda t: 5
        try:
            return RT.route_entry(f2, {"configurable": {"ruoyi_token": "x", "start_variants": True}}), f2
        finally:
            ct.teacher_id_from_token = o
    r, f2 = asyncio.run(_route())
    chk("await_review 置位 awaiting_mother_review", f2.get("awaiting_mother_review") is True)
    chk("🔴 route_entry(start_variants)=generate（修前=parse·0 变式）", r == "generate", extra=f"r={r}")

    print("== ② degraded **有**真 qtype → 用真 qtype，不被默认覆盖（无回归）==")
    prev2 = {"main_kp": {"id": "9999", "name": ""}, "secondary_kps": [],
             "skeleton": ["移项"], "model_candidates": [], "qtype": "选择题"}
    out2 = asyncio.run(_reanchor(prev2, {"kp": {}, "grade": {"value": "八年级下册", "code": GRADE}}))
    chk("真 qtype「选择题」保留", ((out2.get("analysis") or {}).get("qtype") or {}).get("value") == "选择题")
    chk("②也 confirmed=True", out2.get("mother_confirmed") is True)

    print("== ③ 非 degraded（锚到真叶子·无 qtype）→ 不强补默认（不污染正常路径）==")
    # 主考点名命中池内叶子 → 锚真叶子（main_kp 非 None）→ 不进 degraded → 无 qtype 时不强补
    leaf_with_kp = [("3082002009001", "一元二次方程的解法"), ("3082002009002", "别的")]
    prev3 = {"main_kp": {"id": "", "name": "一元二次方程的解法"}, "secondary_kps": [],
             "skeleton": ["移项"], "model_candidates": []}

    async def _reanchor_leaf():
        from agents.variant.stage1_anchor import label as L
        _cap(L)
        st = {"messages": [], "knobs": None, "mother_dna": {"dna": prev3}}
        return await L._reanchor_reuse_first_solve(
            state=st, analysis={"kp": {"value": "一元二次方程的解法"}, "grade": {"value": "八年级下册", "code": GRADE}},
            mother_dna={"dna": prev3}, prev_dna=prev3, grade_code=GRADE, chapter_id=CHAP,
            leaf_pool=leaf_with_kp, confirmed_chapter_id=CHAP, include_review_books=False, knobs=None)
    out3 = asyncio.run(_reanchor_leaf())
    dna3 = (out3.get("mother_dna") or {}).get("dna") or {}
    an3 = out3.get("analysis") or {}
    chk("非 degraded（锚真叶子 id 命中）", str((dna3.get("main_kp") or {}).get("id") or "").startswith("3082002009"),
        extra=f"id={(dna3.get('main_kp') or {}).get('id')}")
    chk("非 degraded·无 qtype → 不强补「解答题」（保持原 clarify 语义）",
        (an3.get("qtype") or {}).get("value") != "解答题",
        extra=f"qtype={(an3.get('qtype') or {}).get('value')}")

    return ok


def main():
    print("===== PRD-C-107 BUG-2 smoke (offline) =====")
    a = offline_gates()
    print(f"\nA 段离线断言: {'ALL GREEN' if a else 'RED'}")
    sys.exit(0 if a else 1)


if __name__ == "__main__":
    main()
