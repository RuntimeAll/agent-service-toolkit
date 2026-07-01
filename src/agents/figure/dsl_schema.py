# -*- coding: utf-8 -*-
"""中性几何 JSON DSL 的 schema 闸（PRD-C-110 B1）。

事实源 = book-ui/public/geo-engine/geo-dsl-render.js 的 `switch(o.type)`（25 type 白名单 +
各 type 必填字段 + 引用 id 可解析 + functiongraph.expr 数学白名单）。AI 直出 DSL（路 A，A2 预飞行
8/8=100% 验过），本模块做**确定性校验**——不合 schema 由调用方降级（占位/文字/退 GeoGebra），绝不
让脏 DSL 进渲染端。

🔴 与 A2 spike（tools/c110_a2_spike.py 的 validate_dsl）同一套规则，固化为产品模块。
🔴 安全红线：functiongraph.expr 走数学记号白名单（承 PRD-A-100，渲染器内 new Function 限定单参 x，
   无任意 eval）；本闸在产 DSL 端先挡一道。
"""
from __future__ import annotations

import re
from typing import Any

# 各 type 的必填字段（按 geo-dsl-render.js buildObjects switch 实际读取的字段）。
# circle 特判：center+through 或 center+r。
TYPE_REQUIRED: dict[str, list[str]] = {
    "point": ["coords"],
    "segment": ["points"],
    "line": ["points"],
    "ray": ["points"],
    "vector": ["points"],
    "polygon": ["points"],
    "midpoint": ["points"],
    "circle": [],
    "circumcircle": ["points"],
    "perpendicular": ["line", "point"],
    "parallel": ["line", "point"],
    "anglebisector": ["points"],
    "intersection": ["of"],
    "angle": ["points"],
    "glider": ["on", "coords"],
    "tangent": ["at"],
    "functiongraph": ["expr"],
    "curve": ["xs", "ys"],
    "ellipse": ["cx", "cy", "rx", "ry"],
    "text": ["x", "y", "text"],
    "axisArrow": ["from", "to"],
    "bar": ["x0", "x1", "h"],
    "sector": ["cx", "cy", "r", "start", "end"],
    "numberline": ["xmin", "xmax"],
    "circleOutline": ["cx", "cy", "r"],
}
WHITELIST: set[str] = set(TYPE_REQUIRED.keys())

# 引用型字段（值是别的 object 的 id，必须能在本 DSL 里解析到）。
REF_FIELDS: dict[str, str] = {
    "points": "list", "center": "id", "through": "id", "line": "id",
    "point": "id", "of": "list", "on": "id", "at": "id",
}


# ---------------------------------------------------------------------------
# 高层「构件 DSL」(spec.build) 的浅校验（构件层 = book-ui/public/geo-engine/figure-builder.js）。
#
# 🔴 与低层 DSL 的分工：低层 objects 由本模块深校验（type/必填/引用/expr）；高层 build 的**几何深校验
#    在前端 figure-builder（代码精确解坐标、几何不变量由 75 项 node 测保证）**，toolkit 侧无 JS 运行时、
#    只做**结构浅校验**（每项恰有一个 shape/add/mark/transform 键 + 值在已知集合）——挡住"完全不对"的输出，
#    深层几何交给 figure-builder。事实源 = figure-builder.js 的 expand 分发键 + system-prompt-figures.txt 白名单。
# ---------------------------------------------------------------------------
CONSTRUCT_SHAPES: set[str] = {
    "triangle", "quad", "regular", "circle", "arc", "function", "solid", "chart",
    "numberline", "angle", "parallelCut", "coordinate", "clock", "fractionBar",
    "fractionCircle", "grid",
}
CONSTRUCT_ADDS: set[str] = {
    "midpoint", "median", "altitude", "diagonal", "circumcircle", "incircle",
    "intersection", "centroid",
}
CONSTRUCT_MARKS: set[str] = {"rightangle", "angle"}
CONSTRUCT_TRANSFORMS: set[str] = {"reflect", "translate", "rotate", "central"}


def validate_construct(spec: Any) -> tuple[bool, list[str]]:
    """高层构件 DSL（spec.build）浅校验：build 是非空数组，每项恰含一个
    shape/add/mark/transform 键且值在已知集合（几何深校验交前端 figure-builder）。

    返回 (ok, errors)。errors 非空即结构不合（调用方据此降级）。
    """
    errs: list[str] = []
    if not isinstance(spec, dict):
        return False, ["顶层非 dict"]
    build = spec.get("build")
    if not isinstance(build, list) or not build:
        return False, ["build 缺失或非数组"]
    for i, it in enumerate(build):
        if not isinstance(it, dict):
            errs.append(f"build[{i}] 非对象")
            continue
        kinds = [k for k in ("shape", "add", "mark", "transform") if k in it]
        if len(kinds) != 1:
            errs.append(f"build[{i}] 须恰含一个 shape/add/mark/transform 键（现有: {kinds or '无'}）")
            continue
        key = kinds[0]
        val = it.get(key)
        allowed = {
            "shape": CONSTRUCT_SHAPES, "add": CONSTRUCT_ADDS,
            "mark": CONSTRUCT_MARKS, "transform": CONSTRUCT_TRANSFORMS,
        }[key]
        if val not in allowed:
            errs.append(f"build[{i}] {key}='{val}' 不在白名单")
        # 复合/标注/变换须指向已建图形 id（of）；基础图形(shape)无需 of。
        if key in ("add", "mark", "transform") and not it.get("of"):
            errs.append(f"build[{i}] {key} 须给 of（指向已建图形 id）")
    return (len(errs) == 0), errs


def validate_dsl(spec: Any) -> tuple[bool, list[str]]:
    """过 geo-engine schema 校验：type 白名单 + 必填字段齐 + 引用 id 可解析 + functiongraph 白名单。

    🔴 双格式：spec 含 `build`（高层构件 DSL）→ 走 validate_construct（结构浅校验，几何交前端）；
       否则按低层 DSL（objects）深校验。二者由调用方据 ok/errs 降级，行为一致。

    返回 (ok, errors)。errors 非空即不合 schema（调用方据此降级）。
    """
    errs: list[str] = []
    if not isinstance(spec, dict):
        return False, ["顶层非 dict"]
    # 高层构件 DSL：有 build 且无 objects → 走构件浅校验。
    if spec.get("build") and not spec.get("objects"):
        return validate_construct(spec)
    objs = spec.get("objects")
    if spec.get("solid3d") and not objs:
        objs = []  # 纯 3D spec 允许无 objects
    if not isinstance(objs, list) or (not objs and not spec.get("solid3d")):
        return False, ["objects 缺失或非数组"]
    ids = {o.get("id") for o in objs if isinstance(o, dict) and o.get("id")}
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            errs.append(f"objects[{i}] 非对象")
            continue
        t = o.get("type")
        tag = f"objects[{i}](id={o.get('id')},type={t})"
        if t not in WHITELIST:
            errs.append(f"{tag} type 不在白名单")
            continue
        for fld in TYPE_REQUIRED[t]:
            if fld not in o or o[fld] in (None, "", []):
                errs.append(f"{tag} 缺必填字段 {fld}")
        if t == "circle":
            has_through = "center" in o and "through" in o
            has_r = "center" in o and "r" in o
            if not (has_through or has_r):
                errs.append(f"{tag} circle 需 center+through 或 center+r")
        for fld, _kind in REF_FIELDS.items():
            if fld not in o:
                continue
            val = o[fld]
            refs = val if isinstance(val, list) else [val]
            for r in refs:
                # of/which 第三项可能是数字；只校验字符串型引用是否定义。
                if isinstance(r, str) and r not in ids:
                    errs.append(f"{tag} 字段 {fld} 引用未定义 id '{r}'")
        if t == "functiongraph":
            expr = str(o.get("expr", ""))
            cleaned = re.sub(r"\b(sin|cos|tan|sqrt|abs|exp|log|pi)\b", "", expr, flags=re.I)
            if not re.fullmatch(r"[-+*/^().,0-9xeE \t]*", cleaned):
                errs.append(f"{tag} functiongraph.expr 含非白名单字符: {expr!r}")
    return (len(errs) == 0), errs
