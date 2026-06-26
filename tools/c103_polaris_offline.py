# -*- coding: utf-8 -*-
r"""PRD-C-103 WS1·北极星离线证明（AC3 / G-WS1c 的确定性内核）：直接走真 SQL 取数路径
（model_anchor.lookup_candidates 读 biz_solution_model 真表）→ 改 difficulty_tier → 看同一道
变式经 grade_observed 的确定档**双向单调随表变**。

与 c103_polaris.py（真起 graph 端到端）互补：本脚本剥掉 LLM 随机性，**只验控制电线**——
「改表 → lookup 取出新 tier → 锚定模型带新 tier_int → grade_variant_item 档随之变」。
LLM 选哪个模型/写什么题面的随机性在这里被固定（直接指定 target 模型 + 固定一道变式题面），
故结果**确定可复现**，是 AC3 北极星的可机器裁决内核。

跑法：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_polaris_offline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pymysql  # noqa: E402

from agents import model_anchor  # noqa: E402
from agents.variant import grade_variant_item  # noqa: E402

DB = dict(host="127.0.0.1", port=3307, user="root", password="123456",
          database="ai_lesson_prep", charset="utf8mb4")

# 选一个挂在数轴 kp 上的大招模型 + 它的 subject_id 叶子（lookup 反查能命中）。
TARGET_MODEL = "DZ02"          # 大招2 数轴循环规律
LEAF_CODE = "100001002001"     # DZ02 绑定的 subject_id（见 biz_solution_model_kp）

# 固定一道「继承该模型的变式」题面。🔴 刻意用**中等 K/R/D**（R∈{2,3}、单问、不触发 L4(a) 的 K≥3+R≥4），
#   让模型 tier 成为决定档的旋钮 —— 这样翻 tier 才能看到档随之单调移动（K/R/D 太强会饱和 L4、
#   太弱会饱和 L1，都看不出 tier 的作用）。R=3、K=2、D=1、无高阶策略词。
SAMPLE_ITEM = {
    "stem": "点 A 从原点出发，按规律移动，求第几次后落在数 6 处？",
    "solution": "由题意得每次位移；根据周期定位；可得答案。",
    "difficulty": 1,  # LLM 自评（会被 grade_observed 覆盖，仅作对照）
}


def _set_tier(model_id: str, tier: int, freq: int):
    conn = pymysql.connect(**DB)
    try:
        cur = conn.cursor()
        cur.execute("UPDATE biz_solution_model SET difficulty_tier=%s, freq_band=%s WHERE id=%s",
                    (tier, freq, model_id))
        conn.commit()
    finally:
        conn.close()


def _get_tier(model_id: str):
    conn = pymysql.connect(**DB)
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute("SELECT difficulty_tier, freq_band FROM biz_solution_model WHERE id=%s", (model_id,))
        return cur.fetchone()
    finally:
        conn.close()


def _anchored_model_with_table_tier() -> dict | None:
    """走真 SQL 反查取出 TARGET_MODEL，带表真值 tier_int/freq_int（模拟 anchor_models 输出的单个模型）。"""
    cands = model_anchor.lookup_candidates([LEAF_CODE])
    for c in cands:
        if c["id"] == TARGET_MODEL:
            # 与 confirm_models 透传同形：带 tier_int/freq_int
            return {"id": c["id"], "name": c["name"],
                    "tier_int": c.get("tier_int"), "freq_int": c.get("freq_int")}
    return None


def _grade_at(tier: int, freq: int) -> dict:
    _set_tier(TARGET_MODEL, tier, freq)
    model = _anchored_model_with_table_tier()
    assert model is not None, f"反查未命中 {TARGET_MODEL}（检查 LEAF_CODE 绑定）"
    mother_dna = {"dna": {"main_kp": {"id": LEAF_CODE}, "secondary_kps": [], "models": [model]}}
    bill = grade_variant_item(SAMPLE_ITEM, mother_dna)
    return {"tier_read": model["tier_int"], "freq_read": model["freq_int"],
            "level": bill["level"], "rule": bill["rule"],
            "modelHits_tier": bill["modelHits"][0]["tier"] if bill["modelHits"] else None,
            "K": bill["K"], "R": bill["R"], "D": bill["D"]}


def main() -> int:
    orig = _get_tier(TARGET_MODEL)
    print(f"=== 北极星离线证明：模型 {TARGET_MODEL} @ leaf {LEAF_CODE} ===")
    print(f"原表值 tier={orig['difficulty_tier']} freq={orig['freq_band']}\n")
    try:
        lo = _grade_at(1, 1)
        print(f"[表 tier=1基础/freq=1低频] lookup读出tier={lo['tier_read']} → 变式确定档 level={lo['level']} "
              f"({lo['rule']}) modelHits.tier={lo['modelHits_tier']} K/R/D={lo['K']}/{lo['R']}/{lo['D']}")
        hi = _grade_at(2, 2)
        print(f"[表 tier=2高阶/freq=2高频] lookup读出tier={hi['tier_read']} → 变式确定档 level={hi['level']} "
              f"({hi['rule']}) modelHits.tier={hi['modelHits_tier']} K/R/D={hi['K']}/{hi['R']}/{hi['D']}")
        lo2 = _grade_at(1, 1)
        print(f"[表 tier=1改回基础]       lookup读出tier={lo2['tier_read']} → 变式确定档 level={lo2['level']} "
              f"({lo2['rule']}) modelHits.tier={lo2['modelHits_tier']}")
    finally:
        _set_tier(TARGET_MODEL, orig["difficulty_tier"], orig["freq_band"])
        print(f"\n已恢复 {TARGET_MODEL} 原值 tier={orig['difficulty_tier']} freq={orig['freq_band']}")

    print("\n========== 判决 ==========")
    up = hi["level"] > lo["level"]
    back = lo2["level"] < hi["level"] and lo2["level"] == lo["level"]
    print(f"改表 tier_int 1→2 经 lookup 真读出: {lo['tier_read']}→{hi['tier_read']} （表真值已被读进锚定模型）")
    print(f"高→档升: level {lo['level']}→{hi['level']} → {'✓' if up else '✗'}")
    print(f"改回→落: level {hi['level']}→{lo2['level']}(=低档{lo['level']}) → {'✓' if back else '✗'}")
    green = up and back
    print(f"\nG-WS1c 内核: {'GREEN（改表→真读→确定档双向单调，控制电线通）' if green else 'RED'}")
    return 0 if green else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(main())
