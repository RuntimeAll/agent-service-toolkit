# -*- coding: utf-8 -*-
r"""PRD-C-017 B5-fix4 真机探针：老师把填空变式改成解答 → 重生后题面应是解答结构（无 ___、有"求…过程"）。

直驱真实 LLM（generate 档 gpt-5.4），不经 classify/RuoYi：
  1. 造一个填空母题 facts + 一道填空变式 item；
  2. edit_dna_state(index=1, field=qtype, value=解答) → 该题 dirty_dims 含 qtype；
  3. regen_dirty_items([1]) → _regen_once 注改题型重构强指令 → 重出；
  4. 打印重出题面，判它是否变成解答结构（无 ____ 空位、含"求/过程/解答"）。

跑法: $env:PYTHONUTF8='1'; .venv/Scripts/python.exe tools/c017_b5fix4_probe.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.variant import edit_dna_state, regen_dirty_items  # noqa: E402

_STATE = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {"value": "一元一次方程", "confidence": 0.9,
               "anchored": {"id": "30710101", "code": "30710101", "name": "一元一次方程"}},
        "qtype": {"value": "填空", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "若方程 $2x+1=5$ 的解是 $x=\\underline{\\quad}$。",
        "answer": "2", "difficulty": 2,
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [], "qtype": "填空", "exam_type": "直接计算",
            "skeleton": ["移项", "系数化为1"], "hard_points": [], "tags": ["解方程"],
            "scene": "纯代数", "difficulty": 2, "flags": [],
        },
    },
    "items": [
        {
            "stem": "方程 $3x-2=7$ 的解是 $x=\\underline{\\quad\\quad}$。",
            "answer": "3", "qtype": "填空", "difficulty": 2, "level": "normal",
            "solution": "移项得 $3x=9$，系数化为1得 $x=3$。",
        }
    ],
}


async def main() -> None:
    print("=== 改题型前·变式题面（填空）===")
    print(_STATE["items"][0]["stem"])

    # 步骤2：老师 edit-dna 把题型改成解答
    upd, edited, err = edit_dna_state(_STATE, 1, "qtype", "解答")
    assert err is None, f"edit_dna_state 失败：{err}"
    st = {**_STATE, **upd}
    it = st["items"][0]
    print("\n=== edit-dna 后 ===")
    print("item.qtype =", it.get("qtype"), " dna_dirty =", it.get("dna_dirty"),
          " dirty_dims =", it.get("dirty_dims"))
    assert "qtype" in (it.get("dirty_dims") or []), "qtype 未进 dirty_dims（检测前提断了）"

    # 步骤3：点重生这道
    upd2, result, err2 = await regen_dirty_items(st, [1])
    assert err2 is None, f"regen 失败：{err2}"
    print("\n=== regen 结果 ===")
    print("regenerated =", result.get("regenerated"), " failed =", result.get("failed"))
    new_it = upd2["items"][0]
    stem = new_it.get("stem") or ""
    qtype = new_it.get("qtype")
    print("\n=== 重出后·变式题面 ===")
    print("qtype =", qtype)
    print(stem)
    print("\n--- answer ---")
    print((new_it.get("answer") or "")[:200])

    # 步骤4：判定是否变成解答结构
    has_blank = ("____" in stem) or ("\\underline" in stem) or ("（  ）" in stem) or ("(  )" in stem)
    solve_words = any(w in (stem + (new_it.get("solution") or "")) for w in ("求", "过程", "解答", "写出", "解题", "解方程", "计算", "证明", "化简"))
    print("\n=== 判定 ===")
    print("题面含填空空位标记 =", has_blank, "（应为 False）")
    print("含解答型设问词(求/过程/写出) =", solve_words, "（应为 True）")
    print("qtype 字段 =", qtype, "（应为 解答）")
    verdict = (qtype == "解答") and (not has_blank) and solve_words
    print("\n>>> 修复验证：", "PASS（已重构为解答结构）" if verdict else "FAIL（仍是填空/未重构）")


if __name__ == "__main__":
    asyncio.run(main())
