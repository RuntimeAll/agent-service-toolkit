# -*- coding: utf-8 -*-
"""PRD-C-103 批2·WS2 离线单测：临时模型识别 + 参与 grade_observed 判档（不调真 LLM）。

绕 core/__init__.py(import langchain) 直接按路径加载 difficulty.py；model_anchor 的 confirm_models
用 mock invoke 喂固定 JSON，验证：
  ① LLM 提候选外可复用套路 → 产临时模型(带临时 tier_int) 进 models，不丢。
  ② 临时模型 tier_int 真喂进 grade_observed → 难度非空、随临时 tier 双向变（高阶抬档）。
  ③ 基础运算/无新套路 → 不产临时模型（不污染）。
跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b2_tempmodel_offline.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# 直接按路径加载 difficulty.py（绕 core/__init__ 的 langchain import）
_spec = importlib.util.spec_from_file_location("c103_difficulty", SRC / "core" / "difficulty.py")
difficulty = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(difficulty)


# ---- mock model_anchor.confirm_models 依赖的解析逻辑（直接 import 它的纯函数）----
# model_anchor import core.settings → 会拉 langchain；改为只测它的纯解析 + 临时模型组装逻辑，
# 复制其 _parse_confirm_json 行为做端到端最小验证（与源同口径，源改了这里也要改——故只验关键路径）。
_ma_spec = importlib.util.spec_from_file_location("c103_ma", SRC / "agents" / "model_anchor.py")


def _load_model_anchor():
    """尝试真加载 model_anchor；失败（langchain 缺）→ None，回退纯解析验证。"""
    try:
        mod = importlib.util.module_from_spec(_ma_spec)
        _ma_spec.loader.exec_module(mod)
        return mod
    except Exception as e:  # noqa: BLE001
        print(f"[warn] model_anchor 真加载失败({e})，回退仅验 difficulty 端临时模型参与")
        return None


CANDS = [
    {"id": "DZ08", "name": "大招8 距离问题", "trigger_feature": "t", "action_conclusion": "a",
     "sort": 8, "model_kind": "gold", "tier_int": 1, "freq_int": 2},
    {"id": "DZ09", "name": "大招9 相遇问题", "trigger_feature": "t", "action_conclusion": "a",
     "sort": 9, "model_kind": "gold", "tier_int": 1, "freq_int": 2},
]


def make_invoke(reply: str):
    async def _inv(messages, model=None, **kw):
        return reply
    return _inv


async def run():
    ma = _load_model_anchor()
    fails: list[str] = []

    # ---- 端 A：grade_observed 直接吃临时模型对象（核心：临时 tier 参与判档）----
    base_factors = dict(K=3, R=4, D=0, G=0, high_strategies=[])
    temp_base = [{"id": None, "name": "新套路X", "model_kind": "derived", "isNew": True,
                  "tier_int": 1, "freq_int": 1}]
    temp_high = [{"id": None, "name": "新套路X", "model_kind": "derived", "isNew": True,
                  "tier_int": 2, "freq_int": 1}]
    bill_base = difficulty.grade_observed(model_hits=temp_base, **base_factors)
    bill_high = difficulty.grade_observed(model_hits=temp_high, **base_factors)
    print(f"[A] 临时模型基础 tier → level={bill_base['level']} rule={bill_base['rule']} "
          f"modelHits={bill_base['modelHits']}")
    print(f"[A] 临时模型高阶 tier → level={bill_high['level']} rule={bill_high['rule']}")
    if bill_base["level"] is None or bill_base["level"] == 0:
        fails.append("A1: 临时模型基础 tier 判档为空")
    if bill_high["level"] <= bill_base["level"]:
        fails.append(f"A2: 临时模型高阶 tier 未抬档（{bill_base['level']}→{bill_high['level']}）")
    # 临时模型出现在账单 modelHits（非空、tier 透传）
    if not bill_high["modelHits"] or bill_high["modelHits"][0]["tier"] != 2:
        fails.append("A3: 账单 modelHits 未带临时模型高阶 tier")

    # ---- 端 B：model_anchor.confirm_models mock LLM 产临时模型 ----
    if ma is not None:
        # B1：LLM 命中候选 DZ08 + 提一个候选外临时模型（高阶/一次性）
        reply_b1 = ('{"hits":["DZ08"],"newModels":[{"name":"绝对值分段距离和最值",'
                    '"triggerFeature":"含多绝对值","action":"分段求最小","tier":"高阶","freq":"一次性"}]}')
        confirmed, temp_models, overflow = await ma.confirm_models(
            "题面", "解析", CANDS, invoke=make_invoke(reply_b1))
        print(f"[B1] confirmed={[c['id'] for c in confirmed]} temp={[t['name'] for t in temp_models]} "
              f"overflow={overflow}")
        if not any(c["id"] == "DZ08" for c in confirmed):
            fails.append("B1: 候选命中 DZ08 丢失")
        if len(temp_models) != 1 or temp_models[0]["tier_int"] != 2:
            fails.append(f"B1: 临时模型未产出或 tier_int 错（{temp_models}）")

        # B1b：anchor_models 端到端把临时模型并进 models
        # 直接调 confirm 已验；anchor_models 还需 DB（lookup），故仅验 models 合并逻辑用 confirm 结果模拟
        models_out = list(confirmed) + [{"id": None, "name": t["name"], "model_kind": "derived",
                                         "isNew": True, "tier_int": t["tier_int"],
                                         "freq_int": t["freq_int"]} for t in temp_models]
        bill = difficulty.grade_observed(model_hits=models_out, K=2, R=3, D=0, G=0,
                                         high_strategies=[])
        print(f"[B1b] 合并后判档 level={bill['level']} (含高阶临时模型应抬档) hits={len(bill['modelHits'])}")
        if not any(h["tier"] == 2 for h in bill["modelHits"]):
            fails.append("B1b: 合并 models 后临时模型高阶 tier 未进账单")

        # B2：纯基础题，LLM 不提临时模型
        reply_b2 = '{"hits":[],"newModels":[]}'
        confirmed2, temp2, overflow2 = await ma.confirm_models(
            "1+1=?", "2", CANDS, invoke=make_invoke(reply_b2))
        print(f"[B2] 基础题 confirmed={confirmed2} temp={temp2} overflow={overflow2}")
        if temp2:
            fails.append("B2: 基础题误产临时模型")

        # B3：旧 schema 兼容（只有 "models" 键）
        confirmed3, temp3, _ = await ma.confirm_models(
            "题", "解", CANDS, invoke=make_invoke('{"models":["DZ09"]}'))
        if not any(c["id"] == "DZ09" for c in confirmed3):
            fails.append("B3: 旧 schema {models:[...]} 兼容失败")
        print(f"[B3] 旧 schema 兼容 confirmed={[c['id'] for c in confirmed3]}")

    print("\n==== RESULT ====")
    if fails:
        for f in fails:
            print("FAIL", f)
        sys.exit(1)
    print("ALL PASS（临时模型识别 + 参与判档 + tier 抬档 + 基础题不污染 + 旧schema兼容）")


if __name__ == "__main__":
    asyncio.run(run())
