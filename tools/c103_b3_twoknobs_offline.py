# -*- coding: utf-8 -*-
"""PRD-C-103 批3·WS3 离线单测：双旋钮两轴纯函数 + 变式血缘 trace 真值（不调真 LLM/不连库）。

绕 core/__init__.py(import langchain) 直接按路径加载 variant.py 的纯函数子集——variant 顶层
import langchain 太重，改为只测 4 个纯函数的等价实现核对 + 直接 exec 加载受测函数。

验证：
  ① operator_band_from_similarity：系数→算子带映射（高仿/中变/远迁三带 + 边界 + 钳制 + None 兜底）。
  ② normalize_two_knobs：config.configurable 双键→knobs 增量段（keep/缺/非法各轴不设键）。
  ③ recipe_from_knobs：难度轴 difficulty_target 真移 md（md+i 起点变）；变式系数注入算子配方段。
  ④ variant_trace_block：算子/相似度真值 + actual_level 读 difficulty_bill + target/retries。
跑：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b3_twoknobs_offline.py
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# variant.py 顶层 import 一堆 langchain/core —— 直接 import 会拉一坨。改用「抠源码片段 exec」法：
# 只把要测的 4 个纯函数 + 它们的依赖常量/小函数 exec 进一个干净命名空间，零外部依赖。
_VARIANT_SRC = (SRC / "agents" / "variant.py").read_text(encoding="utf-8")

# 提取依赖：_to_int / DIFFICULTY_CAP / DEFAULT_SHAPE / 双旋钮段 / recipe_from_knobs / variant_trace_block。
# 用 ast 取这些顶层定义源码，避免 import 整模块。
import ast

tree = ast.parse(_VARIANT_SRC)
WANT_FUNC = {
    "_to_int", "operator_band_from_similarity", "normalize_two_knobs",
    "recipe_from_knobs", "variant_trace_block",
}
WANT_ASSIGN = {
    "DIFFICULTY_CAP", "DEFAULT_SHAPE", "VARIANT_COEFF_DEFAULT",
    "_OPERATOR_BANDS", "_OPERATOR_GUIDANCE", "PLAN_INCREASING", "_PLAN_INCREASING_WORDS",
}
src_chunks: list[str] = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT_FUNC:
        src_chunks.append(ast.get_source_segment(_VARIANT_SRC, node))
    elif isinstance(node, ast.Assign):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        # 处理多目标赋值（KNOBS_COUNT_MIN, KNOBS_COUNT_MAX = 1, 8 / 元组目标）
        for t in node.targets:
            if isinstance(t, ast.Tuple):
                names |= {e.id for e in t.elts if isinstance(e, ast.Name)}
        if names & WANT_ASSIGN:
            src_chunks.append(ast.get_source_segment(_VARIANT_SRC, node))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        # 带类型注解的赋值（如 _OPERATOR_BANDS: list[...] = [...]）
        if node.target.id in WANT_ASSIGN:
            src_chunks.append(ast.get_source_segment(_VARIANT_SRC, node))

ns: dict = {"Any": __import__("typing").Any}
# difficulty 模块给 grade_observed 用——variant_trace_block 不调它，但保险塞个占位
ns["difficulty"] = types.SimpleNamespace()
for chunk in src_chunks:
    exec(compile(chunk, "<variant-subset>", "exec"), ns)

operator_band_from_similarity = ns["operator_band_from_similarity"]
normalize_two_knobs = ns["normalize_two_knobs"]
recipe_from_knobs = ns["recipe_from_knobs"]
variant_trace_block = ns["variant_trace_block"]
VARIANT_COEFF_DEFAULT = ns["VARIANT_COEFF_DEFAULT"]

_pass = 0
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _pass, _fail
    if cond:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name} {detail}")


print("== ① operator_band_from_similarity ==")
b9 = operator_band_from_similarity(0.9)
check("0.9→数值(高仿)", b9 and b9["operator"] == "数值" and b9["band"] == "高", str(b9))
check("0.9 similarity 透传", b9 and b9["similarity"] == 0.9, str(b9))
b6 = operator_band_from_similarity(0.6)
check("0.6→结构(中变)", b6 and b6["operator"] == "结构" and b6["band"] == "中", str(b6))
b3 = operator_band_from_similarity(0.3)
check("0.3→推广一般化(远迁)", b3 and b3["operator"] == "推广一般化" and b3["band"] == "低", str(b3))
check("guidance 非空（注入配方段用）", b9 and b9.get("guidance"), str(b9))
# 边界 + 钳制 + None
check("0.75 边界归高仿带", operator_band_from_similarity(0.75)["operator"] == "数值")
check("0.45 边界归中变带", operator_band_from_similarity(0.45)["operator"] == "结构")
check("1.5 钳到 1.0", operator_band_from_similarity(1.5)["similarity"] == 1.0)
check("None→None（不设算子）", operator_band_from_similarity(None) is None)
check("非数→None", operator_band_from_similarity("abc") is None)

print("== ② normalize_two_knobs ==")
out = normalize_two_knobs({"variant_similarity": 0.3, "difficulty_target": 4})
check("两轴齐：operator_band 落远迁", out.get("operator_band", {}).get("operator") == "推广一般化", str(out))
check("两轴齐：variant_coeff=0.3", out.get("variant_coeff") == 0.3, str(out))
check("两轴齐：difficulty_target=4", out.get("difficulty_target") == 4, str(out))
out_keep = normalize_two_knobs({"variant_similarity": 0.7, "difficulty_target": "keep"})
check("difficulty=keep → 不设 difficulty_target", "difficulty_target" not in out_keep, str(out_keep))
check("similarity=0.7 仍设 operator_band", out_keep.get("operator_band", {}).get("operator") == "结构", str(out_keep))
out_empty = normalize_two_knobs({})
check("空 config → {}（回落默认）", out_empty == {}, str(out_empty))
out_cam = normalize_two_knobs({"variantSimilarity": 0.9, "difficultyTarget": 2})
check("camelCase 容错", out_cam.get("variant_coeff") == 0.9 and out_cam.get("difficulty_target") == 2, str(out_cam))
out_badtgt = normalize_two_knobs({"difficulty_target": 99})
check("difficulty_target=99 钳到 4", out_badtgt.get("difficulty_target") == 4, str(out_badtgt))

print("== ③ recipe_from_knobs 难度轴移 md + 变式系数注入配方段 ==")
# 母题档 md=2，无难度轴 → md+i 从 2 起
r_keep = recipe_from_knobs({"count": 3, "difficulty_plan": "increasing"}, mother_difficulty=2)
check("keep：expected 从母题档 2 起", r_keep["expected_difficulties"] == [2, 3, 4], str(r_keep["expected_difficulties"]))
# 难度轴目标档 4 → md 移到 4，md+i 从 4 起（封顶 4）
r_hi = recipe_from_knobs(
    {"count": 3, "difficulty_plan": "increasing", "difficulty_target": 4}, mother_difficulty=2
)
check("难度轴 target=4：expected 从 4 起（封顶）", r_hi["expected_difficulties"] == [4, 4, 4], str(r_hi["expected_difficulties"]))
# 难度轴目标档 1 → md 移到 1，md+i 从 1 起
r_lo = recipe_from_knobs(
    {"count": 3, "difficulty_plan": "increasing", "difficulty_target": 1}, mother_difficulty=4
)
check("难度轴 target=1：expected 从 1 起（双向）", r_lo["expected_difficulties"] == [1, 2, 3], str(r_lo["expected_difficulties"]))
# 变式系数轴注入算子配方段
band = operator_band_from_similarity(0.3)
r_op = recipe_from_knobs({"count": 2, "operator_band": band}, mother_difficulty=2)
check("变式系数注入算子段（spec 含『推广一般化』）", "推广一般化" in r_op["spec"], r_op["spec"][:120])
# 空 knobs 回归锚（行为不变）
r0 = recipe_from_knobs({}, mother_difficulty=2)
check("空 knobs → n=3/2普1难（回归锚）", r0["n"] == 3 and r0["n_normal"] == 2 and r0["n_hard"] == 1, str(r0))

print("== ④ variant_trace_block 真值 ==")
item = {"difficulty": 3, "difficulty_bill": {"level": 3, "modelHits": [{"tier": 2}]}}
knobs = {"operator_band": operator_band_from_similarity(0.3), "variant_coeff": 0.3, "difficulty_target": 4}
tb = variant_trace_block(item, knobs)
check("operator=推广一般化（真算子非 forward-gen）", tb["operator"] == "推广一般化", str(tb))
check("similarity=0.3（真系数非 None）", tb["similarity"] == 0.3, str(tb))
check("actual_level=3（读 difficulty_bill）", tb["actual_level"] == 3, str(tb))
check("target_level=4（难度轴目标档）", tb["target_level"] == 4, str(tb))
check("created_by=forward-gen", tb["created_by"] == "forward-gen", str(tb))
check("similarity_band=低", tb["similarity_band"] == "低", str(tb))
# 无双旋钮（旧路径）→ 仍给确定值（similarity 默认 0.7、operator forward-gen），非 None
tb_def = variant_trace_block({"difficulty": 2, "difficulty_bill": {"level": 2}}, {})
check("无旋钮：similarity 默认 0.7（非 None）", tb_def["similarity"] == VARIANT_COEFF_DEFAULT, str(tb_def))
check("无旋钮：operator=forward-gen 兜底", tb_def["operator"] == "forward-gen", str(tb_def))
check("无旋钮：target_level=None（keep）", tb_def["target_level"] is None, str(tb_def))

print(f"\n==== 汇总：PASS {_pass} / FAIL {_fail} ====")
sys.exit(1 if _fail else 0)
