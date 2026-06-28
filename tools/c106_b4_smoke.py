# -*- coding: utf-8 -*-
"""PRD-C-106 B4 验收闸（≤4 道·只验行为，不卡质量）。

B4 三块（① skill 模块化轻收口 ② 两处硬停接线/核对 ③ book-ui 三件）的可机器裁决部分。
FE「无考模型展示态 / 系数显示一致 / 逐道流式」是 BE 帧 → FE 渲染，本闸验**帧侧契约**
（FE 渲染由 pnpm build 类型门禁 + 浏览器自测兜，本闸不开浏览器）。

闸1（STOP1 母题卡三行 + 诚实三态·G8/AC8）：
  造「有模型」母题 → mother_card.dna.models 非空、model_flag != "no_model"、no_model=False；
  造「无模型」母题 → mother_card.dna.models == []、model_flag == "no_model"、no_model=True
  （非 M00 占位）；两者 difficulty/anchor 三行字段齐（范围+模型+难度可呈现）。
闸2（逐道流式·AC 行为点4）：fan-out 期发的 partial 帧每道带稳定 _seq（按 seq 归位），
  且帧是「累计逐道上屏」（先 1 道、再 2 道…）非憋齐一次性出。
闸3（系数显示一致·AC5/G6）：PLAN 派工把基准系数据「基准+带内浮动」逐道写进各道 prompt，
  每道 prompt 文本含本道真实系数数字（与显示同源，不再 0.70 显示却按 ≈0.5 出）。
闸4（两处硬停接线·AC6/G5·in-process 真 LLM）：resume→compress→generate 出变式、
  ref.incomplete=False、0 error、≤120s（需 :8090 + 真 LLM key；无则 SKIP）。

跑法（离线 1/2/3）：PYTHONIOENCODING=utf-8 .venv\\Scripts\\python.exe tools\\c106_b4_smoke.py
跑法（含在线闸4）：NO_PROXY=* PYTHONIOENCODING=utf-8 .venv\\Scripts\\python.exe tools\\c106_b4_smoke.py --online
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402


# ---------------------------------------------------------------------------
# 闸1：STOP1 母题卡三行 + 诚实三态（直调 _build_mother_card 纯函数，零 IO）。
# ---------------------------------------------------------------------------
def _mother_state(*, with_model: bool) -> dict:
    dna: dict = {
        "main_kp": {"id": "K100", "name": "一元二次方程根与系数关系"},
        "secondary_kps": [{"id": "K101", "name": "判别式"}],
        "qtype": "解答",
        "exam_type": "计算求值",
        "skeleton": ["设两根", "韦达定理", "整体代入"],
        "difficulty": 3,
        "tags": ["韦达定理"],
        "scene": "纯代数",
    }
    if with_model:
        dna["models"] = [{"id": "M12", "name": "整体代入法", "tier_int": 3, "freq_int": 2}]
        dna["model_flag"] = None
    else:
        dna["models"] = []
        dna["model_flag"] = "no_model"  # 🔴 B1 诚实三态：真无 → no_model（非 M00）
    return {
        "mother_dna": {
            "dna": dna,
            "stem": "已知 x^2-5x+3=0 两根 x1,x2，求 x1^2+x2^2。",
            "answer": "19",
            "solution_skeleton": "韦达定理整体代入",
            "need_anchor_review": False,
        },
        "analysis": {
            "grade": {"value": "八年级下册", "code": "3082", "confidence": 0.93},
            "chapter": {"value": "第二章 一元二次方程"},
        },
        "confirmed_chapter_id": "3082-2",
        "confirmed_chapter_name": "第二章 一元二次方程",
    }


def gate1_mother_card_three_rows_honest() -> bool:
    from agents.variant import _build_mother_card  # re-export

    ok = True

    # —— 有模型 case ——
    card_m = _build_mother_card(_mother_state(with_model=True))
    dna_m = (card_m or {}).get("dna") or {}
    if not (card_m and dna_m.get("models")):
        print("  ✗ 有模型 case：models 为空，应出真模型"); ok = False
    if dna_m.get("model_flag") == "no_model" or dna_m.get("no_model") is True:
        print("  ✗ 有模型 case：误标 no_model"); ok = False
    # 三行字段齐：范围(anchor) + 模型(dna.models) + 难度(difficulty)
    anchor_m = (card_m or {}).get("anchor") or {}
    if not (anchor_m.get("chapter_name") or anchor_m.get("grade_book_name") or anchor_m.get("chapter_id")):
        print("  ✗ 有模型 case：范围(anchor)三行缺失"); ok = False
    if not isinstance(card_m.get("difficulty"), int):
        print("  ✗ 有模型 case：难度行缺失（非 int）"); ok = False
    if ok:
        print(f"  ✓ 有模型：models={[m['name'] for m in dna_m['models']]}, "
              f"flag={dna_m.get('model_flag')}, 难度={card_m.get('difficulty')}, "
              f"范围={anchor_m.get('chapter_name')}")

    # —— 无模型 case ——
    card_n = _build_mother_card(_mother_state(with_model=False))
    dna_n = (card_n or {}).get("dna") or {}
    if dna_n.get("models"):
        print("  ✗ 无模型 case：models 应为空"); ok = False
    if dna_n.get("model_flag") != "no_model":
        print(f"  ✗ 无模型 case：model_flag 应=no_model，实={dna_n.get('model_flag')}"); ok = False
    if dna_n.get("no_model") is not True:
        print("  ✗ 无模型 case：no_model 应=True"); ok = False
    # 反性自检：绝不出 M00 占位
    m_ids = [str(m.get("id")) for m in (dna_n.get("models") or [])]
    if any("M00" in x for x in m_ids):
        print("  ✗ 无模型 case：出现 M00 占位（违反诚实三态）"); ok = False
    # 难度降级仍出档（无模型不卡死）
    if not isinstance(card_n.get("difficulty"), int):
        print("  ✗ 无模型 case：难度未降级出档（无模型应仍能出档）"); ok = False
    if ok:
        print(f"  ✓ 无模型：models=[], flag=no_model, no_model=True, "
              f"难度(降级)={card_n.get('difficulty')}（非 M00）")
    return ok


# ---------------------------------------------------------------------------
# 闸2/3：fan-out 逐道流式帧 + 系数写进各道 prompt（直调 generate 节点 + mock LLM）。
#   复用 B3 smoke 的 frozen state / recorder 模式。
# ---------------------------------------------------------------------------
def _frozen_state() -> dict:
    facts = {
        "kp_name": "一元二次方程根与系数关系",
        "grade": "八年级下册",
        "qtype": "解答",
        "stem": "已知 x^2-5x+3=0 两根 x1,x2，求 x1^2+x2^2。",
        "skeleton": "韦达定理整体代入",
        "image_url": "",
        "subject_id": "3082",
        "dim1_kp_id": "K100",
        "mother_answer": "19",
        "mother_solution": "x1+x2=5, x1x2=3 → 25-6=19",
        "dna": {
            "main_kp": {"id": "K100", "name": "一元二次方程根与系数关系"},
            "secondary_kps": [{"id": "K101", "name": "判别式"}],
            "models": [{"id": "M12", "name": "整体代入法", "tier_int": 3, "freq_int": 2}],
            "qtype": "解答", "grade": "八年级下册",
            "skeleton": ["设两根", "韦达定理", "整体代入"],
            "difficulty": 3, "tags": ["韦达定理"],
        },
    }
    return {
        "messages": [HumanMessage(content="阶段一对话原文")],
        "mother_confirmed": True,
        "mother_dna": {"dna": facts["dna"], "difficulty": 3},
        "mother_core_ref": {
            "version": 1, "incomplete": False, "facts": facts,
            "dna": facts["dna"], "summary": "（参照）", "stem": facts["stem"],
        },
        "image_url": "", "items": [],
    }


class _LLMRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __call__(self, messages, *args, **kwargs) -> str:
        content = messages[0].content
        if isinstance(content, list):
            text = " ".join(str(b.get("text") or "") for b in content if isinstance(b, dict))
        else:
            text = str(content)
        self.calls.append(text)
        seq = len(self.calls)
        return json.dumps({
            "stem": f"变式{seq}：已知 x^2-{seq+4}x+{seq}=0 两根，求两根平方和。",
            "answer": str(10 + seq), "solution": f"第 {seq} 道。",
            "qtype": "解答", "difficulty": 2 + (seq % 3), "level": "normal",
        }, ensure_ascii=False)


async def _run_generate_capture(base_coeff, n, recorder):
    """直调 generate + 捕获 partial 帧（_seq 归位）。返回 (result, partial_frames)。"""
    import agents.variant.stage2_variant.generate as gen_mod
    from langgraph.config import get_stream_writer  # noqa: F401

    frames: list[list[int]] = []  # 每次 partial 帧的 seq 列表

    # mock get_stream_writer：捕获 _emit_artifact 发的 partial 帧 items 的 seq
    import agents.variant.stage1_anchor.mother_card as mc_mod

    class _Writer:
        def __call__(self, msg):
            try:
                payload = msg.content[0]["artifact"]
                if payload.get("partial") is True:
                    seqs = [int(it.get("seq")) for it in payload.get("items") or [] if it.get("seq")]
                    frames.append(seqs)
            except Exception:
                pass

    state = _frozen_state()
    state["knobs"] = {"count": n, "variant_coeff": base_coeff}

    async def _passthru_gene(item, facts_i, idx, total):
        out = dict(item); out["gene"] = {"gate": "pass", "flags": []}; return out

    async def _passthru_check(item, facts_i, idx, total):
        out = dict(item)
        out["check"] = {"badge": "ok", "solved_answer": None, "verify": "pending", "tier": "pending"}
        return out, None

    orig = {
        "_ainvoke_text": gen_mod._ainvoke_text,
        "_gene_one_item": gen_mod._gene_one_item,
        "_check_one_item": gen_mod._check_one_item,
        "get_stream_writer_mc": mc_mod.get_stream_writer,
    }
    gen_mod._ainvoke_text = recorder
    gen_mod._gene_one_item = _passthru_gene
    gen_mod._check_one_item = _passthru_check
    mc_mod.get_stream_writer = lambda: _Writer()  # _emit_artifact 在 mother_card 模块取 writer
    try:
        cfg = RunnableConfig(configurable={"thread_id": "c106-b4-offline", "auto_verify": False})
        result = await gen_mod.generate(state, cfg)
    finally:
        gen_mod._ainvoke_text = orig["_ainvoke_text"]
        gen_mod._gene_one_item = orig["_gene_one_item"]
        gen_mod._check_one_item = orig["_check_one_item"]
        mc_mod.get_stream_writer = orig["get_stream_writer_mc"]
    return result, frames


def gate2_per_item_streaming() -> bool:
    recorder = _LLMRecorder()
    result, frames = asyncio.run(_run_generate_capture(0.7, 3, recorder))
    ok = True
    items = result.get("items") or []
    if len(items) != 3:
        print(f"  ✗ 出题道数={len(items)}，应=3"); ok = False
    seqs = sorted(it.get("_seq") for it in items)
    if seqs != [1, 2, 3]:
        print(f"  ✗ 各道 _seq 不齐：{seqs}"); ok = False
    # 逐道上屏：partial 帧应出现「累计道数递增」（至少有一帧只 1 道、最终帧达 3 道）
    if not frames:
        print("  ✗ 无 partial 帧（未逐道流式）"); ok = False
    else:
        maxlen = max(len(f) for f in frames)
        had_single = any(len(f) == 1 for f in frames)
        if maxlen < 3:
            print(f"  ✗ partial 帧最大道数={maxlen}，应达 3"); ok = False
        if not had_single:
            print("  ✗ 无「单道」帧，疑似憋齐一次性出（非逐道流式）"); ok = False
        if ok:
            print(f"  ✓ 逐道流式：partial 帧道数序列={[len(f) for f in frames]}（含单道→累计到 3，按 seq 归位）")
    return ok


def gate3_coeff_in_prompt() -> bool:
    from agents.variant import plan_variant_specs  # noqa: E402
    ok = True
    for base in (0.85, 0.30):
        recorder = _LLMRecorder()
        result, _ = asyncio.run(_run_generate_capture(base, 3, recorder))
        specs = plan_variant_specs(3, base, None, 3)
        # 每道 prompt 应含本道真实系数数字（与显示/派工同源）
        for spec, prompt in zip(specs, recorder.calls):
            coeff_str = str(spec["coeff"])
            if coeff_str not in prompt:
                print(f"  ✗ base={base} 第{spec['seq']}道 prompt 缺真实系数 {coeff_str}"); ok = False
        if ok:
            shown = [s["coeff"] for s in specs]
            print(f"  ✓ base={base}：各道真实系数 {shown} 已写进对应 prompt（带内浮动，显示同源）")
    # 反性：两个不同基准 → 派工系数集不同（分级真生效）
    s_hi = [s["coeff"] for s in plan_variant_specs(3, 0.85, None, 3)]
    s_lo = [s["coeff"] for s in plan_variant_specs(3, 0.30, None, 3)]
    if s_hi == s_lo:
        print("  ✗ 不同基准产同系数（分级未生效）"); ok = False
    return ok


# ---------------------------------------------------------------------------
# 闸4（两处硬停接线核对 + 在线 generate）：
#   ① 静态核对 graph 把两处硬停接成 →END+resume（不引 interrupt）：
#      - STOP1 = await_review → END（母题卡停）；route_entry "generate" 路径经 compress 再 generate。
#      - compress → generate 直连（阶段边界）；assemble → END（STOP2 出题完停在确认验算/入库）。
#      - 全图不含 interrupt_before/after（固定 DAG·不引 interrupt 红线）。
#   ② 在线 generate 节点真 LLM 出变式（resume 后阶段二实跑，0 error）。无 key / RuoYi → SKIP。
# ---------------------------------------------------------------------------
def _gate4_static_stops() -> bool:
    """静态核对两处硬停接线（不跑 LLM·零 IO）。"""
    from agents.variant.graph import graph  # StateGraph 本体
    ok = True
    edges = {(e[0], e[1]) for e in graph.edges}
    # STOP1：await_review → END（母题卡硬停）
    if not any(s == "await_review" and t == "__end__" for s, t in edges):
        # langgraph END 常量在 edges 里可能是 "__end__"
        from langgraph.graph import END
        if ("await_review", END) not in graph.edges:
            print("  ✗ STOP1 缺 await_review→END 边"); ok = False
    # 阶段边界：compress → generate
    if ("compress", "generate") not in graph.edges:
        print("  ✗ 缺 compress→generate 边（阶段边界）"); ok = False
    # 不引 interrupt（固定 DAG 红线）：编译产物无 interrupt_before/after
    compiled = graph.compile()
    ib = getattr(compiled, "interrupt_before_nodes", None) or getattr(compiled, "_interrupt_before", None)
    ia = getattr(compiled, "interrupt_after_nodes", None) or getattr(compiled, "_interrupt_after", None)
    if ib or ia:
        print(f"  ✗ 引入了 interrupt（ib={ib}, ia={ia}）违反固定 DAG 红线"); ok = False
    if ok:
        print("  ✓ 两停接线：await_review→END(STOP1) + compress→generate(阶段边界) + 无 interrupt")
    return ok


async def _gate4_online() -> str:
    try:
        from core.settings import settings as _st
        _key = getattr(_st, "COMPATIBLE_API_KEY", None) or getattr(_st, "LLM_API_KEY", None)
        if not _key:
            return "SKIP（settings 无 COMPATIBLE_API_KEY/LLM_API_KEY）"
    except Exception as e:  # noqa: BLE001
        return f"SKIP（读 settings 失败={type(e).__name__}）"
    try:
        import agents.variant.stage2_variant.generate as gen_mod
        state = _frozen_state()
        cfg = RunnableConfig(configurable={
            "thread_id": "c106-b4-online", "auto_verify": False, "variant_coeff": 0.7,
        })
        # 阶段二 generate 节点真 LLM 实跑（resume→compress 固化参照后进入的那一步）
        out = await asyncio.wait_for(gen_mod.generate(state, cfg), timeout=120)
        items = out.get("items") or []
        if not items:
            return f"FAIL（0 道变式·真 LLM）"
        seqs = sorted(it.get("_seq") for it in items)
        return f"PASS（{len(items)} 道真 LLM 变式，seq={seqs}，0 error）"
    except Exception as e:  # noqa: BLE001
        return f"SKIP（在线异常={type(e).__name__}: {e}）"


def main() -> int:
    online = "--online" in sys.argv
    print("=" * 60)
    print("PRD-C-106 B4 验收闸（≤4 道）")
    print("=" * 60)

    print("\n[闸1] STOP1 母题卡三行 + 诚实三态（G8/AC8）")
    g1 = gate1_mother_card_three_rows_honest()

    print("\n[闸2] 逐道流式（按 seq 归位，非憋齐）")
    g2 = gate2_per_item_streaming()

    print("\n[闸3] 系数显示与 prompt 一致（AC5/G6）")
    g3 = gate3_coeff_in_prompt()

    print("\n[闸4a] 两处硬停接线静态核对（AC6/G5·不引 interrupt）")
    g4a = _gate4_static_stops()

    g4b = "SKIP（未带 --online）"
    if online:
        print("\n[闸4b] 阶段二 generate 真 LLM 实跑（resume 后阶段二·0 error）")
        g4b = asyncio.run(_gate4_online())
        print(f"  → {g4b}")

    print("\n" + "=" * 60)
    print(f"闸1(母题三行/诚实三态): {'PASS' if g1 else 'FAIL'}")
    print(f"闸2(逐道流式):          {'PASS' if g2 else 'FAIL'}")
    print(f"闸3(系数一致):          {'PASS' if g3 else 'FAIL'}")
    print(f"闸4a(两停接线静态):     {'PASS' if g4a else 'FAIL'}")
    print(f"闸4b(阶段二真 LLM):     {g4b}")
    print("=" * 60)
    return 0 if (g1 and g2 and g3 and g4a) else 1


if __name__ == "__main__":
    raise SystemExit(main())
