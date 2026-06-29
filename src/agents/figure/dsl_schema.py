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


def validate_dsl(spec: Any) -> tuple[bool, list[str]]:
    """过 geo-engine schema 校验：type 白名单 + 必填字段齐 + 引用 id 可解析 + functiongraph 白名单。

    返回 (ok, errors)。errors 非空即不合 schema（调用方据此降级）。
    """
    errs: list[str] = []
    if not isinstance(spec, dict):
        return False, ["顶层非 dict"]
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
