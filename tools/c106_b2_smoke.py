"""PRD-C-106 B2 验收闸（≤3 道·只验流程通 + 母题核心参照契约正确，不卡质量）。

闸1：参照 schema 完整（dna/summary/stem/answer/solution 齐；无模型时 dna 带 no_model flag）。
闸2：阶段二入口不含阶段一对话原文（facts_from_ref 返回 frozen 参照、不读 messages；
     compress 自身不读 messages）。
闸3（在线·可选）：端到端基础流程跑通（贴图→母题卡→确认→出变式），0 error、参照传到阶段二。
     需 :8093 在跑；无 token / 服务未起则跳过（离线闸 1/2 已覆盖契约正确性）。

🔴 离线闸（1/2）零依赖服务，纯函数验契约；在线闸（3）走 SSE，须 NO_PROXY=* 绕本机代理。
跑法：.venv\\Scripts\\python.exe tools\\c106_b2_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agents.variant import (  # noqa: E402
    build_mother_core_ref,
    facts_from_ref,
)


def _make_state(*, with_model: bool, dialogue_secret: str) -> dict:
    """造一个阶段一已锚定的 state（含一段「阶段一对话原文」放进 messages，验隔离）。"""
    models = (
        [{"id": "M12", "name": "整体代入法", "tier_int": 3, "freq_int": 2}]
        if with_model
        else []
    )
    dna = {
        "main_kp": {"id": "K100", "name": "一元二次方程根与系数关系"},
        "secondary_kps": [{"id": "K101", "name": "判别式"}],
        "qtype": "解答",
        "models": models,
        "model_flag": (None if with_model else "no_model"),
        "skeleton": ["设根 x1 x2", "韦达定理", "代入求值"],
        "scene": "纯代数",
        "difficulty": 3,
        "tags": ["韦达定理"],
        "exam_type": "计算",
    }
    return {
        # 🔴 阶段一对话原文（含一个独有 secret 串）—— 阶段二绝不该看到它
        "messages": [
            HumanMessage(content=f"这道题怎么解 {dialogue_secret}"),
            AIMessage(content=f"阶段一解题过程草稿 {dialogue_secret}"),
        ],
        "image_url": "https://example.com/mother.png",
        "mother_has_figure": True,
        "analysis": {
            "grade": {"value": "八年级下册", "code": "3082", "confidence": 0.95},
            "kp": {"value": "一元二次方程", "anchored": {"code": "K100"}, "confidence": 0.9},
        },
        "mother_dna": {
            "stem": "已知方程 x^2-3x+1=0 的两根为 x1,x2，求 x1^2+x2^2。",
            "answer": "7",
            "solution_skeleton": "x1+x2=3, x1x2=1 → x1^2+x2^2=(x1+x2)^2-2x1x2=9-2=7",
            "difficulty": 3,
            "dna": dna,
        },
    }


def gate1_schema() -> bool:
    print("\n=== 闸1：MotherCoreRef schema 完整 ===")
    ok = True

    # 有模型
    ref_m = build_mother_core_ref(_make_state(with_model=True, dialogue_secret="SECRET_A"))
    required = ["dna", "summary", "stem", "answer", "solution", "figure", "facts", "version", "incomplete"]
    for k in required:
        present = k in ref_m and (ref_m[k] not in (None, "") or k in ("incomplete",))
        print(f"  [有模型] {k:12} = {str(ref_m.get(k))[:60]!r}  -> {'OK' if present else 'MISSING'}")
        ok = ok and present
    # dna 关键字段（G3 反性自检：摘要不能把 kp/models/grade 丢了 → 走结构化 dna）
    dna = ref_m["dna"]
    for k in ("main_kp", "models", "grade", "grade_code", "difficulty"):
        has = dna.get(k) not in (None, "", [])
        print(f"  [有模型] dna.{k:12} = {str(dna.get(k))[:50]!r}  -> {'OK' if has else 'EMPTY'}")
        ok = ok and has
    # 有模型 → no_model=False + flag 非 no_model + models 真名
    cond = (not dna["no_model"]) and dna["model_flag"] != "no_model" and dna["models"] and dna["models"][0]["name"]
    print(f"  [有模型] no_model=False + 真模型名 -> {'OK' if cond else 'FAIL'} (no_model={dna['no_model']}, name={dna['models'][0]['name'] if dna['models'] else None})")
    ok = ok and bool(cond)
    print(f"  [有模型] summary = {ref_m['summary']!r}")
    ok = ok and bool(ref_m["summary"].strip())
    ok = ok and not ref_m["incomplete"]

    # 无模型 → dna 带 no_model flag（诚实三态）
    ref_n = build_mother_core_ref(_make_state(with_model=False, dialogue_secret="SECRET_B"))
    dnan = ref_n["dna"]
    cond2 = dnan["no_model"] is True and dnan["model_flag"] == "no_model" and dnan["models"] == []
    print(f"  [无模型] no_model flag 正确 -> {'OK' if cond2 else 'FAIL'} (no_model={dnan['no_model']}, flag={dnan['model_flag']}, models={dnan['models']})")
    print(f"  [无模型] summary = {ref_n['summary']!r}")
    ok = ok and bool(cond2)
    ok = ok and ("无考模型" in ref_n["summary"])

    print(f"闸1 -> {'PASS' if ok else 'FAIL'}")
    return ok


def gate2_isolation() -> bool:
    print("\n=== 闸2：阶段二入口不含阶段一对话原文 ===")
    ok = True
    secret = "SECRET_ISO_XYZ"
    state = _make_state(with_model=True, dialogue_secret=secret)
    ref = build_mother_core_ref(state)

    # 参照对象（整体序列化）不得含对话 secret
    import json
    blob = json.dumps(ref, ensure_ascii=False, default=str)
    no_secret = secret not in blob
    print(f"  MotherCoreRef 序列化不含对话 secret({secret}) -> {'OK' if no_secret else 'LEAK'}")
    ok = ok and no_secret

    # facts_from_ref：把 frozen ref 塞进 state → 返回的应是 ref.facts，且不含 messages 原文
    state_with_ref = dict(state, mother_core_ref=ref)
    facts = facts_from_ref(state_with_ref)
    facts_blob = json.dumps(facts, ensure_ascii=False, default=str)
    facts_no_secret = secret not in facts_blob
    is_frozen = facts is ref["facts"]
    print(f"  facts_from_ref 返回 frozen ref.facts(同对象) -> {'OK' if is_frozen else 'FAIL'}")
    print(f"  facts 不含对话 secret -> {'OK' if facts_no_secret else 'LEAK'}")
    print(f"  facts.stem = {str(facts.get('stem'))[:40]!r}（母题原文，非对话）")
    ok = ok and is_frozen and facts_no_secret

    # 反性：缺 ref → 回退 _mother_facts（仍不含 messages 原文，但每次重算）
    facts_fallback = facts_from_ref(state)  # 无 mother_core_ref
    fb_no_secret = secret not in json.dumps(facts_fallback, ensure_ascii=False, default=str)
    print(f"  缺参照回退 _mother_facts 也不含对话原文 -> {'OK' if fb_no_secret else 'LEAK'}")
    ok = ok and fb_no_secret

    print(f"闸2 -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = {"闸1·schema": gate1_schema(), "闸2·隔离": gate2_isolation()}
    print("\n==== B2 离线闸汇总 ====")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("  闸3·在线端到端：走 tools/c106_b2_online.py（需 :8093 在跑）")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
