# -*- coding: utf-8 -*-
"""PRD-C-107 BUG-1·确认年级章后重锚解题打标死循环修复 ≤4 道精测闸。

root cause（复核已确认）：classify(_reuse_ok) 旧前置要求 dna.main_kp.name 非空才走
_reanchor_reuse_first_solve 的 graceful 降级；但首解「锚不到叶子」最常见场景里 opus primaryKp 常
「有 id 无 name」或为空 → anchor_to_chapter 收成 main_kp=None/name空 → _reuse_ok=False → 绕过降级 →
回退重 solve → niche 二次坏 JSON → _SolveLabelError → picker → 老师再确认 → 又重 solve = 死循环。

A 段·离线断言（无需服务·恒可跑，主证据）：
  ① _reuse_ok 放宽：首解成功（stem+opus+非空 dna）但 main_kp **name 空** → 仍判复用（不重 solve）。
  ② graceful 降级真触发：_reanchor_reuse_first_solve 在 name 空 + 锚不到叶子 → 锚到确认章节点 +
     confirmed=True + need_anchor_review，**不退 picker**（不 emit need_confirm）、能进阶段二。
  ③ 有界兜底：_bounded_degrade_to_chapter（无首解产物 + 同章已失败一次）→ confirmed=True + 清失败标记。

跑法：PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c107_bug1_smoke.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _install_emit_capture(L):
    """把 label 模块里的 emit 名换成捕获器，返回 captured 列表（('stage'/'need_confirm'/'mother_card'/'figure', payload)）。"""
    captured: list[tuple] = []

    def cap_stage(node, title, status, detail=""):
        captured.append(("stage", {"node": node, "status": status, "detail": detail}))

    def cap_need_confirm(payload):
        captured.append(("need_confirm", payload))

    def cap_mother_card(state):
        captured.append(("mother_card", None))

    def cap_figure(state):
        captured.append(("figure", None))

    L._emit_stage = cap_stage
    L._emit_need_confirm = cap_need_confirm
    L._emit_mother_card = cap_mother_card
    L._emit_figure_stage = cap_figure
    # classify/reanchor 体内还会 `from agents.variant import _emit_figure_stage, _emit_mother_card`
    # （延迟 import 覆盖本地名）→ 同步把 facade 上的也换掉，确保延迟 import 拿到捕获器。
    import agents.variant as V
    V._emit_stage = cap_stage
    V._emit_need_confirm = cap_need_confirm
    V._emit_mother_card = cap_mother_card
    V._emit_figure_stage = cap_figure
    return captured


def offline_gates() -> bool:
    import agents.variant as V
    from agents.variant.stage1_anchor import label as L

    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        flag = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{flag}] {name} {extra}")

    # 确认章 id（八下·一元二次方程章，7 位 level2）；其前 4 位 = 年级册 code。
    CHAP = "3082002"
    GRADE = "3082"
    # 确认章收窄后的叶子池：故意**不含**贴近主考点的叶子（模拟 niche 锚不到），
    #   但含一个章内别的叶子，证明「池非空但主 kp 锚不到」→ 走降级而非「池空」分支。
    LEAF_POOL = [("3082002009001", "其它叶子A"), ("3082002009002", "其它叶子B")]

    print("== ①_reuse_ok 放宽：首解成功(stem+opus+非空 dna) 但 main_kp **name 空** → 判复用 ==")
    # 模拟 anchor_to_chapter 把 niche 主考点收成「有 id 无 name」后被归一的态：
    #   场景 A：main_kp = {"id": "<越界id>", "name": ""}（name 空，旧 _reuse_ok 会 False）。
    prev_dna_nameless = {
        "main_kp": {"id": "9999999999999", "name": ""},  # 越界 id + 空 name
        "secondary_kps": [], "skeleton": ["移项", "因式分解"], "model_candidates": [],
        "qtype": "解答题",
    }
    md_nameless = {
        "stem": "解方程 x^2-7x+12=0",
        "mother_solve_source": "opus",
        "_solve_range_fp": GRADE,
        "dna": prev_dna_nameless,
    }
    _prev_dna = md_nameless["dna"]
    # 复刻 classify 里 _reuse_ok 的判据（修后版本：不再要求 main_kp 有 name）
    _reuse_ok = bool(
        CHAP
        and md_nameless.get("mother_solve_source") == "opus"
        and str(md_nameless.get("stem") or "").strip()
        and _prev_dna
    )
    chk("name 空时 _reuse_ok=True（修后：复用首解、不回退重 solve）", _reuse_ok is True)
    # 场景 B：main_kp 整个为 None（opus primaryKp 全空被归一成 None）→ _prev_dna 仍非空 → 仍复用
    prev_dna_nonekp = dict(prev_dna_nameless); prev_dna_nonekp["main_kp"] = None
    _reuse_ok_b = bool(CHAP and "opus" == "opus" and "stem" and prev_dna_nonekp)
    chk("main_kp=None 时 _reuse_ok=True（产物在即复用）", _reuse_ok_b is True)

    print("== ②graceful 降级真触发：name 空 + 锚不到叶子 → 锚确认章 + confirmed + 不退 picker ==")

    async def _run_reanchor(prev_dna):
        captured = _install_emit_capture(L)
        analysis = {
            "kp": {"value": "一元二次方程的解法"},  # 老师/analyze 读出的考点名（name 回退链②）
            "grade": {"value": "八年级下册", "code": GRADE},
        }
        state = {"messages": [], "knobs": None, "mother_dna": {"dna": prev_dna}}
        out = await L._reanchor_reuse_first_solve(
            state=state, analysis=dict(analysis), mother_dna={"dna": prev_dna},
            prev_dna=prev_dna, grade_code=GRADE, chapter_id=CHAP, leaf_pool=LEAF_POOL,
            confirmed_chapter_id=CHAP, include_review_books=False, knobs=None,
        )
        return out, captured

    # 场景 A：name 空 → 名回退到 analysis.kp.value「一元二次方程的解法」；池里仍锚不到 → 降级锚到 CHAP。
    out_a, cap_a = asyncio.run(_run_reanchor(prev_dna_nameless))
    main_kp_a = (out_a.get("mother_dna") or {}).get("dna", {}).get("main_kp") or {}
    chk("降级锚到确认章节点本身（main_kp.id == 确认章 id）", main_kp_a.get("id") == CHAP,
        extra=f"id={main_kp_a.get('id')}")
    chk("need_anchor_review=True（标待人审）",
        bool((out_a.get("mother_dna") or {}).get("dna", {}).get("need_anchor_review")))
    # 关键：第一次 degraded 会触发 BUG-03 单轮闸断（要求老师再确认同章）——这是**有界 1 轮**，非死循环；
    #   再确认同章即放行。本断言验：第一轮不是「重 solve / 反复解析失败」，且记了 _bug03_gated_chapter。
    chk("第一次降级 → BUG-03 闸断（awaiting_mother_confirm，非重 solve 失败）",
        out_a.get("awaiting_mother_confirm") is True
        and out_a.get("_bug03_gated_chapter") == CHAP,
        extra=f"await={out_a.get('awaiting_mother_confirm')}")
    # 老师**第二次确认同章**（state 带 _bug03_gated_chapter）→ 接受强锚放行进阶段二。
    async def _run_reanchor_insist(prev_dna):
        _install_emit_capture(L)
        analysis = {"kp": {"value": "一元二次方程的解法"}, "grade": {"value": "八年级下册", "code": GRADE}}
        state = {"messages": [], "knobs": None, "mother_dna": {"dna": prev_dna},
                 "_bug03_gated_chapter": CHAP}
        return await L._reanchor_reuse_first_solve(
            state=state, analysis=dict(analysis), mother_dna={"dna": prev_dna},
            prev_dna=prev_dna, grade_code=GRADE, chapter_id=CHAP, leaf_pool=LEAF_POOL,
            confirmed_chapter_id=CHAP, include_review_books=False, knobs=None,
        )
    out_a2 = asyncio.run(_run_reanchor_insist(prev_dna_nameless))
    chk("二次确认同章 → confirmed=True 进阶段二（不退 picker、未重 solve）",
        out_a2.get("mother_confirmed") is True and out_a2.get("awaiting_mother_confirm") is False,
        extra=f"confirmed={out_a2.get('mother_confirmed')}")

    print("== ③有界兜底 _bounded_degrade_to_chapter（无首解产物 + 同章已失败一次）==")

    async def _run_bounded():
        captured = _install_emit_capture(L)
        analysis = {"kp": {"value": "一元二次方程的解法"}, "grade": {"value": "八年级下册", "code": GRADE}}
        state = {"messages": [], "knobs": None}
        out = L._bounded_degrade_to_chapter(
            state=state, analysis=dict(analysis), mother_dna={},
            grade_code=GRADE, confirmed_chapter_id=CHAP,
            chapter_text="一元二次方程", knobs=None,
        )
        return out, captured

    out_c, cap_c = asyncio.run(_run_bounded())
    nc_emitted = any(k == "need_confirm" for k, _ in cap_c)
    chk("无产物有界降级 → confirmed=True（按确认章出题）", out_c.get("mother_confirmed") is True,
        extra=f"confirmed={out_c.get('mother_confirmed')}")
    chk("有界降级**不退 picker**（无 need_confirm 帧）", not nc_emitted)
    chk("清失败标记 _resolve_failed_chapter=None（不带 stale）",
        out_c.get("_resolve_failed_chapter") is None)
    mk_c = (out_c.get("mother_dna") or {}).get("dna", {}).get("main_kp") or {}
    chk("主考点锚到确认章节点 + no_model 诚实三态",
        mk_c.get("id") == CHAP
        and (out_c.get("mother_dna") or {}).get("dna", {}).get("model_flag") == "no_model")

    return ok


def main():
    print("===== PRD-C-107 BUG-1 smoke (offline) =====")
    a = offline_gates()
    print(f"\nA 段离线断言: {'ALL GREEN' if a else 'RED'}")
    sys.exit(0 if a else 1)


if __name__ == "__main__":
    main()
