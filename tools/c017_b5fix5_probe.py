# -*- coding: utf-8 -*-
r"""PRD-C-017 B5-fix5 真机探针：老师把变式场景改成「行程问题」→ 重生后题面应改写到该场景。

直驱真实 LLM（generate 档 gpt-5.4），不经 classify/RuoYi：
  1. 造一个纯代数母题 facts + 一道纯代数变式 item；
  2. edit_dna_state(index=1, field=scene, value=行程问题) → 落 mother_dna.dna.scene + 该题 dirty_dims 含 scene；
  3. regen_dirty_items([1]) → _regen_once 注场景改写强指令 → 重出；
  4. 打印重出题面，判它是否真改写到「行程问题」场景。

跑法: $env:PYTHONUTF8='1'; .venv/Scripts/python.exe tools/c017_b5fix5_probe.py
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
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "解方程：$2x+1=5$。",
        "answer": "x=2", "difficulty": 2,
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
            "skeleton": ["移项", "系数化为1"], "hard_points": [], "tags": ["解方程"],
            "scene": "纯代数", "difficulty": 2, "flags": [],
        },
    },
    "items": [
        {
            "stem": "解方程：$3x-2=7$。",
            "answer": "x=3", "qtype": "解答", "difficulty": 2, "level": "normal",
            "solution": "移项得 $3x=9$，系数化为1得 $x=3$。",
        }
    ],
}

_NEW_SCENE = "行程问题"


async def main() -> None:
    print("=== 改场景前·变式题面（纯代数）===")
    print(_STATE["items"][0]["stem"])

    # 步骤2：老师 edit-dna 把场景改成 行程问题（落 mother_dna.dna.scene = 组级共享）
    upd, edited, err = edit_dna_state(_STATE, 1, "scene", _NEW_SCENE)
    assert err is None, f"edit_dna_state 失败：{err}"
    st = {**_STATE, **upd}
    it = st["items"][0]
    print("\n=== edit-dna 后 ===")
    print("mother_dna.dna.scene =", (st["mother_dna"].get("dna") or {}).get("scene"))
    print("item.dna_dirty =", it.get("dna_dirty"), " dirty_dims =", it.get("dirty_dims"))
    assert "scene" in (it.get("dirty_dims") or []), "scene 未进 dirty_dims（检测前提断了）"

    # 步骤3：点重生这道
    upd2, result, err2 = await regen_dirty_items(st, [1])
    assert err2 is None, f"regen 失败：{err2}"
    print("\n=== regen 结果 ===")
    print("regenerated =", result.get("regenerated"), " failed =", result.get("failed"))
    new_it = upd2["items"][0]
    stem = new_it.get("stem") or ""
    print("\n=== 重出后·变式题面 ===")
    print(stem)
    print("\n--- answer ---")
    print((new_it.get("answer") or "")[:200])

    # 步骤4：判定题面是否改写到行程问题场景（行程问题典型词）
    trip_words = ("路程", "速度", "时间", "行驶", "相遇", "出发", "千米", "公里",
                  "小时", "甲乙", "汽车", "火车", "走", "步行", "骑", "追", "相向")
    hit = [w for w in trip_words if w in stem]
    print("\n=== 判定 ===")
    print("命中行程问题场景词 =", hit, "（应非空）")
    verdict = bool(hit)
    print("\n>>> 修复验证：",
          "PASS（已改写到行程问题场景）" if verdict else "FAIL（仍是纯代数/未改场景）")


if __name__ == "__main__":
    asyncio.run(main())
