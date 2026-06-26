# -*- coding: utf-8 -*-
r"""PRD-C-103 批4·WS4·AC10「去 sympy 硬验算」离线单测。

验证（不调 LLM / 不起 graph）：
  ① sympy 硬门**默认关**（settings.VARIANT_SYMPY_GATE_ON=False）→ 退化构型变式
     **不被剔除**（dropped=False），降级标 ⚠ 放行到 assemble（人工兜底）。
  ② config.configurable.sympy_gate=True 临时开门 → 恢复旧硬门（退化构型超限剔除 dropped=True）。
  ③ _sympy_gate_on 优先级：config 覆盖 > settings 默认。

手法：monkeypatch _degeneracy_verdict 恒返 DEGENERATE（绕开 sympy 真算 + LLM），
   只验「检出退化后是否硬拦」这条 WS4 改的逻辑分支。判分铁律不破：不改 verdict 怎么判。

跑法（cwd=toolkit 根，.venv 解释器）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b4_sympy_gate_offline.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agents.variant as v  # noqa: E402

math_verify = v.math_verify

FACTS = {"qtype": "解答", "kp_id": "x", "kp": "x", "grade": "七年级上册", "dna": {}}


async def _fake_degenerate(_item):
    return {"verdict": math_verify.DEGENERATE, "detail": "最优点落区间端点（测试注入）", "computed": None}


def _mk_item():
    return {
        "stem": "测试退化构型变式题（动点最优落端点）",
        "answer": "0",
        "qtype": "解答",
        "gene": {"gate": "pass"},
        "_seq": 1,
    }


async def main() -> int:
    passed, failed = 0, 0

    def chk(name, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}")

    # monkeypatch 退化判定 + 避免预算/回炉真调 LLM（gate-on 路径会尝试 regen → 让 budget 耗尽走超限弃）
    v._degeneracy_verdict = _fake_degenerate  # type: ignore[assignment]
    v._budget_exhausted = lambda: True  # type: ignore[assignment]  # gate-on 时跳过 regen 直接走超限弃

    print("=== AC10 sympy 硬门去留离线单测 ===")

    # ① 默认（gate 关）：退化构型不剔除、放行
    orig = v.settings.VARIANT_SYMPY_GATE_ON
    v.settings.VARIANT_SYMPY_GATE_ON = False
    item, dropped = await v._anti_degen_gate(_mk_item(), FACTS, 0, 1)
    chk("gate关默认: 退化构型不剔除 (dropped=False)", dropped is False)
    chk("gate关默认: item 仍返回 (非 None)", item is not None)
    chk("gate关默认: 卡片标 ⚠ 放行注记 (solution 尾)",
        "硬门已关" in str(item.get("solution") or ""))

    # ③ helper 优先级
    chk("helper: settings False → _sympy_gate_on()=False", v._sympy_gate_on(None) is False)
    chk("helper: config sympy_gate=True 覆盖 → True",
        v._sympy_gate_on({"configurable": {"sympy_gate": True}}) is True)

    # ② 临时开门（config 覆盖 settings）：恢复旧硬门 → 退化构型超限剔除
    v.settings.VARIANT_SYMPY_GATE_ON = True
    item2, dropped2 = await v._anti_degen_gate(_mk_item(), FACTS, 0, 1)
    chk("gate开: 退化构型超限剔除 (dropped=True)", dropped2 is True)
    chk("gate开: 剔除题带 _dropped 叙事", bool((item2 or {}).get("_dropped")))

    v.settings.VARIANT_SYMPY_GATE_ON = orig

    print(f"\n=== {passed} PASS / {failed} FAIL ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
