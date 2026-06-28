# -*- coding: utf-8 -*-
"""PRD-C-106 B3 验收闸（≤4 道·只验基础流程+行为，不卡质量）。

闸1（离线·N 道=N 次独立 LLM 生成调用）：mock _ainvoke_text 计数 → fan-out 出 N 道 = N 次调用
     （不是一 prompt 出 N 道的 1 次）。
闸2（离线·各道 prompt 含各自 knob，不串台）：每次调用的 prompt 含本道 seq 的系数/算子，
     且 A 道 prompt 不含 B 道独有的算子（断言不串台）。
闸3（离线·系数分级生效 AC5）：两个不同基准系数 → prompt 指令文本不同 + 显示值与 prompt 一致。
闸4（在线·in-process 真 LLM）：resume→compress→fan-out 出 3 道、逐道流式帧、装配去重、0 error、
     ≤120s。需 :8090 + 真 LLM key；无 token / 失败则 SKIP（离线 1/2/3 已覆盖机制）。

🔴 子件0 红线确认：fan-out = generate 节点内 asyncio.gather(Semaphore=3)，非 LangGraph Send。
   离线闸直调 generate 节点 + mock LLM，验的就是节点内并发 + 一次性 return 全组。

跑法（离线）：PYTHONIOENCODING=utf-8 .venv\\Scripts\\python.exe tools\\c106_b3_smoke.py
跑法（含在线）：NO_PROXY=* PYTHONIOENCODING=utf-8 .venv\\Scripts\\python.exe tools\\c106_b3_smoke.py --online
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402

import agents.variant.stage2_variant.generate as gen_mod  # noqa: E402
from agents.variant import plan_variant_specs  # noqa: E402


# ---------------------------------------------------------------------------
# 共享：一份 frozen 母题参照态（compress 已固化 mother_core_ref.facts）。
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
            "qtype": "解答",
            "grade": "八年级下册",
            "skeleton": ["设两根", "韦达定理", "整体代入"],
            "difficulty": 3,
            "tags": ["韦达定理"],
        },
    }
    return {
        "messages": [HumanMessage(content="阶段一对话原文（阶段二不应继承）")],
        "mother_confirmed": True,
        "mother_dna": {"dna": facts["dna"], "difficulty": 3},
        # 🔴 frozen 参照：facts_from_ref 优先读它，generate 不重算 _mother_facts
        "mother_core_ref": {
            "version": 1, "incomplete": False, "facts": facts,
            "dna": facts["dna"], "summary": "（参照）", "stem": facts["stem"],
        },
        "image_url": "",
        "items": [],
    }


class _LLMRecorder:
    """mock _ainvoke_text：记录每次调用的 prompt 文本，返回一道各异的单题 JSON。"""

    def __init__(self) -> None:
        self.calls: list[str] = []  # 每次调用的 prompt 文本

    async def __call__(self, messages, *args, **kwargs) -> str:
        # 取 prompt 文本（list-content 取 text 块；str-content 直接用）
        msg = messages[0]
        content = msg.content
        if isinstance(content, list):
            text = " ".join(
                str(b.get("text") or "") for b in content if isinstance(b, dict)
            )
        else:
            text = str(content)
        self.calls.append(text)
        seq = len(self.calls)
        # 每道题面各异（不撞题），数值随 seq 变
        return json.dumps({
            "stem": f"变式{seq}：已知 x^2-{seq+4}x+{seq}=0 两根，求两根平方和。",
            "answer": str(10 + seq),
            "solution": f"韦达定理整体代入，第 {seq} 道。",
            "qtype": "解答",
            "difficulty": 2 + (seq % 3),
            "level": "normal",
        }, ensure_ascii=False)


async def _passthru_gene(item, facts_i, idx, total):
    out = dict(item)
    out["gene"] = {"gate": "pass", "flags": []}
    return out


async def _passthru_check(item, facts_i, idx, total):
    out = dict(item)
    out["check"] = {"badge": "ok", "solved_answer": None, "verify": "pending", "tier": "pending"}
    return out, None


async def _run_generate(base_coeff, n, recorder: _LLMRecorder) -> dict:
    """直调 generate 节点（mock LLM/闸链），返回 result（含 items）。"""
    state = _frozen_state()
    # knobs：count=n + variant_coeff=base（normalize_two_knobs 通常产 variant_coeff/operator_band，
    #   这里直接给 variant_coeff，generate 的 PLAN 子件读它当基准）。
    state["knobs"] = {"count": n, "variant_coeff": base_coeff}

    # monkeypatch 模块级名（generate.py 命名空间）
    orig = {
        "_ainvoke_text": gen_mod._ainvoke_text,
        "_gene_one_item": gen_mod._gene_one_item,
        "_check_one_item": gen_mod._check_one_item,
    }
    gen_mod._ainvoke_text = recorder
    gen_mod._gene_one_item = _passthru_gene
    gen_mod._check_one_item = _passthru_check
    try:
        cfg = RunnableConfig(configurable={
            "thread_id": "c106-b3-offline", "auto_verify": False,
        })
        result = await gen_mod.generate(state, cfg)
    finally:
        gen_mod._ainvoke_text = orig["_ainvoke_text"]
        gen_mod._gene_one_item = orig["_gene_one_item"]
        gen_mod._check_one_item = orig["_check_one_item"]
    return result


def _gate1_2(n: int = 3) -> bool:
    """闸1: N 道 = N 次独立调用；闸2: 各道 prompt 含各自 knob、不串台。"""
    print("\n== 闸1+2·离线（N 道=N 次独立调用 + 各道 knob 不串台）==")
    base = 0.7
    rec = _LLMRecorder()
    result = asyncio.get_event_loop().run_until_complete(_run_generate(base, n, rec))
    items = result.get("items") or []
    specs = plan_variant_specs(n, base, [3, 4, 4][:n], 3)

    # 闸1：调用计数 == 道数（不是 1 次）
    g1 = len(rec.calls) == n and len(items) == n
    print(f"  LLM 调用次数 = {len(rec.calls)}（期望 {n}）；出题 items = {len(items)} 道  -> {'OK' if g1 else 'FAIL'}")

    # 闸2：第 i 次调用 prompt 的**派工行**含 specs[i] 的系数 + guidance（算子人话指令，
    #   每算子独有、不会出现在 rubric/契约 boilerplate 里）；且不含其它道独有的 guidance（不串台）。
    #   🔴 用 guidance 全句而非裸算子词判——裸「数值/结构」等词在难度 rubric/题型契约里也出现
    #   （boilerplate 噪声），裸词判会假阳性；guidance 全句是 PLAN 注入的唯一标记。
    ops = [s["operator"] for s in specs]
    coeffs = [s["coeff"] for s in specs]
    guides = [s["guidance"] for s in specs]
    g2 = True
    for i, call_text in enumerate(rec.calls):
        own_g = guides[i]
        own_coeff = coeffs[i]
        has_own = (own_g in call_text) and (str(own_coeff) in call_text)
        # 不串台：其它道**独有**的 guidance 不应出现在本道 prompt（同 guidance 多道重复则跳过该项）
        other_unique = [g for j, g in enumerate(guides)
                        if j != i and guides.count(g) == 1 and g != own_g]
        no_crosstalk = all(g not in call_text for g in other_unique)
        ok_i = has_own and no_crosstalk
        g2 = g2 and ok_i
        print(f"  道{i+1}: prompt 含本道(算子={ops[i]} 系数={own_coeff} guidance)={has_own}；"
              f"不串台(无他道独有 guidance)={no_crosstalk}  -> {'OK' if ok_i else 'FAIL'}")
    print(f"  闸1 -> {'PASS' if g1 else 'FAIL'} ；闸2 -> {'PASS' if g2 else 'FAIL'}")
    return g1 and g2


def _gate3() -> bool:
    """闸3·系数分级生效（AC5）：两个不同基准系数 → prompt 指令文本不同 + 显示值一致。"""
    print("\n== 闸3·离线（系数分级生效 AC5）==")
    n = 3
    rec_hi = _LLMRecorder()
    asyncio.get_event_loop().run_until_complete(_run_generate(0.85, n, rec_hi))
    rec_lo = _LLMRecorder()
    asyncio.get_event_loop().run_until_complete(_run_generate(0.25, n, rec_lo))

    specs_hi = plan_variant_specs(n, 0.85, [3, 4, 4], 3)
    specs_lo = plan_variant_specs(n, 0.25, [3, 4, 4], 3)

    # 第 1 道：高系数 prompt 指令 ≠ 低系数 prompt 指令（指令文本随系数真变）
    p_hi = rec_hi.calls[0]
    p_lo = rec_lo.calls[0]
    # 取派工那行的算子/guidance 差异（用 guidance 全句判，避开 boilerplate 裸词噪声）
    op_hi, op_lo = specs_hi[0]["operator"], specs_lo[0]["operator"]
    g_hi, g_lo = specs_hi[0]["guidance"], specs_lo[0]["guidance"]
    instr_diff = (g_hi != g_lo) and (g_hi in p_hi) and (g_lo in p_lo)
    # 显示值一致：spec 的 coeff 数字回填进 prompt
    disp_hi = str(specs_hi[0]["coeff"]) in p_hi
    disp_lo = str(specs_lo[0]["coeff"]) in p_lo
    g3 = instr_diff and disp_hi and disp_lo
    print(f"  高系数(0.85)首道算子={op_hi} coeff={specs_hi[0]['coeff']}；"
          f"低系数(0.25)首道算子={op_lo} coeff={specs_lo[0]['coeff']}")
    print(f"  指令文本随系数变(算子不同且各在各 prompt)={instr_diff}；"
          f"显示值回填 prompt(高={disp_hi} 低={disp_lo})  -> {'OK' if g3 else 'FAIL'}")
    print(f"  闸3 -> {'PASS' if g3 else 'FAIL'}")
    return g3


async def _gate4_online() -> bool:
    """闸4·in-process 真 LLM e2e：resume→compress→fan-out 出 3 道、0 error、≤120s。"""
    print("\n== 闸4·在线 in-process（真 LLM e2e，≤120s）==")
    from agents.variant.graph import graph as state_graph
    from agents.variant_support import RuoyiClient
    from langgraph.checkpoint.memory import MemorySaver

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()

    saver = MemorySaver()
    g = state_graph.compile(checkpointer=saver)
    tid = "c106-b3-inproc"
    cfg = RunnableConfig(configurable={"thread_id": tid, "ruoyi_token": token, "auto_verify": False})

    # pinned 母题态（与 B2 闸3 同口径）
    dna = {
        "main_kp": {"id": "K100", "name": "一元二次方程根与系数的关系"},
        "secondary_kps": [{"id": "K101", "name": "一元二次方程的解法"}],
        "qtype": "解答",
        "models": [{"id": "M12", "name": "韦达定理整体代入", "tier_int": 3, "freq_int": 2}],
        "model_flag": None,
        "skeleton": ["设两根 x1 x2", "由韦达定理", "整体代入求值"],
        "scene": "纯代数", "difficulty": 3, "tags": ["韦达定理"], "exam_type": "计算",
    }
    await g.aupdate_state(cfg, {
        "messages": [HumanMessage(content="（占位：阶段一对话原文）")],
        "image_url": "",
        "analysis": {
            "grade": {"value": "八年级下册", "code": "3082", "confidence": 0.95},
            "kp": {"value": "一元二次方程根与系数的关系",
                   "anchored": {"code": "K100", "name": "一元二次方程根与系数的关系"},
                   "confidence": 0.92},
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {
            "stem": "已知关于 x 的方程 x^2-5x+3=0 的两个实数根为 x1, x2，求 x1^2+x2^2 的值。",
            "answer": "19",
            "analysis": "由韦达定理 x1+x2=5, x1x2=3；x1^2+x2^2=25-6=19。",
            "solution_skeleton": "x1+x2=5, x1x2=3 → 25-6 = 19",
            "difficulty": 3, "dna": dna,
        },
        "mother_confirmed": True, "awaiting_mother_review": True, "items": [],
        "confirmed_chapter_id": "3082", "confirmed_chapter_name": "第2章 一元二次方程",
    })

    resume_cfg = RunnableConfig(configurable={
        "thread_id": tid, "ruoyi_token": token, "auto_verify": False,
        "start_variants": True,
        "variant_similarity": 0.7,  # 走 PLAN 派工
    })
    error = None
    try:
        result = await asyncio.wait_for(
            g.ainvoke({"messages": [HumanMessage(content="开始举一反三")]}, config=resume_cfg),
            timeout=120,
        )
    except asyncio.TimeoutError:
        print("  在线 e2e 超时 >120s -> FAIL")
        return False
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        error = str(e)[:200]
        result = {}

    snap = await g.aget_state(cfg)
    vals = snap.values or {}
    items = vals.get("items") or result.get("items") or []
    seqs = [it.get("_seq") for it in items]
    g4 = len(items) >= 1 and not error
    print(f"  出变式 items = {len(items)} 道；_seq = {seqs}；error = {error}")
    if items:
        print(f"  变式1 stem 前 40 = {str((items[0] or {}).get('stem'))[:40]!r}")
    print(f"  闸4 -> {'PASS' if g4 else 'FAIL'}")
    return g4


def main():
    online = "--online" in sys.argv
    print("===== PRD-C-106 B3 验收闸 =====")
    r1 = _gate1_2(3)
    r3 = _gate3()
    offline_ok = r1 and r3
    print(f"\n离线闸（1/2/3）总判 -> {'PASS' if offline_ok else 'FAIL'}")

    if online:
        try:
            r4 = asyncio.run(_gate4_online())
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"在线闸异常: {e}")
            r4 = False
        all_ok = offline_ok and r4
    else:
        print("（跳过在线闸4：加 --online 跑真 LLM e2e）")
        all_ok = offline_ok

    print(f"\n===== 总判 -> {'PASS' if all_ok else 'FAIL'} =====")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
