# -*- coding: utf-8 -*-
"""mathfig 渲染薄包装（进程内 import render_geogebra + node 子进程隔离）。

🔴 PRD-C-100 B-mathfig/B3：opus 翻 GeoGebra 命令 → 一轮直出 PNG（**不调 verify_figure 回炉**，D12）。
🔴 护栏3（artifacts/python-langgraph-健康 §四）：渲染是 IO（node 子进程 + 浏览器），务必 try/except +
   降级——失败返回 {ok:False}，调用方标 needs_figure ⚠ 继续，**绝不让子进程崩溃冒泡掐 SSE 流**。
🔴 图全链 PNG 无损（render_geogebra 默认无损 PNG）。
🔴 单次 ≤120s（契约面板：造图后处理 单次≤120s）。

mathfig 运行时（recon 定）= 进程内 `from mathfig.geogebra import render_geogebra` 直 import；
内部 spawn `node render.js`，借 book-test 的 @playwright/test（MATHFIG_NODE_CWD env，默认下方兜底）。
"""
from __future__ import annotations

import os
from typing import Any

# 渲染单次墙钟上限（契约：造图后处理单次≤120s）。
RENDER_TIMEOUT_S = 120

# MATHFIG_NODE_CWD 默认兜底（部署经 env 覆盖）：借 codeplace-O/book-test 的 @playwright/test。
_DEFAULT_NODE_CWD = r"d:/workplace/book-ai/codeplace-O/book-test"


def _ensure_env() -> None:
    os.environ.setdefault("MATHFIG_NODE_CWD", _DEFAULT_NODE_CWD)


def render(
    commands: list[str],
    *,
    dashed: list[str] | None = None,
    hide: list[str] | None = None,
    vals: list[str] | None = None,
    styles: dict | None = None,
    axes: bool = False,
    grid: bool = False,
    stem: str = "variant_fig",
    fig_scale: float | None = None,
    point_size: float | None = None,
    relabel: dict | None = None,
) -> dict[str, Any]:
    """一轮直出渲染。返回归一 dict：
       {ok, png_path?, n_cmd, n_fail, commands:[{cmd,ok,err}], vals, warnings, view_final, error?}。
    🔴 任何异常（import 失败/node 缺/子进程崩/超时）→ {ok:False, error:...}，绝不抛。
    🔴 fig_scale（<1 缩画布让标签相对放大；无头渲染下 fontSize 无效，figScale 是唯一杠杆）、
       point_size（=0 隐圆点只留字母）—— 均透传本地 render_geogebra，None 走引擎默认（不传 = 字小、点全显）。
    🔴 relabel（{内部点名:"显示文字"}，如 {"Ap":"A′"}）：旋转/对称像标识符不能含撇号，relabel 让
       自动标签印 A′/B′ 而非内部名 Ap/Bp，opus 不必再手放 Text 撇号标签（双标打架根因）。
    """
    _ensure_env()
    try:
        from mathfig.geogebra import render_geogebra  # 进程内 import（mathfig pip install -e）
    except Exception as e:  # noqa: BLE001 — 引擎不可用 → 降级（不抛，调用方标 needs_figure）
        return {"ok": False, "error": f"mathfig import 失败: {e}", "png_path": None,
                "n_cmd": len(commands or []), "n_fail": len(commands or []), "commands": [],
                "vals": {}, "warnings": ["mathfig 引擎不可用"]}
    try:
        r = render_geogebra(
            list(commands or []), dashed=dashed, hide=hide, vals=vals, styles=styles,
            axes=axes, grid=grid, mono=True, stem=stem, timeout=RENDER_TIMEOUT_S,
            fig_scale=fig_scale, point_size=point_size, relabel=relabel,
        )
        # render_geogebra 失败（n_fail>0 / 无 png）→ ok 已为 False；原样透传 + 兜 png 存在性
        png = r.get("png_path")
        if r.get("ok") and png and os.path.exists(png):
            return {"ok": True, "png_path": png, "n_cmd": r.get("n_cmd"),
                    "n_fail": r.get("n_fail", 0), "commands": r.get("commands", []),
                    "vals": r.get("vals", {}), "warnings": r.get("warnings", []),
                    "view_final": r.get("view_final")}
        return {"ok": False, "error": "渲染未成功（n_fail>0 或无 png）",
                "png_path": png if (png and os.path.exists(png)) else None,
                "n_cmd": r.get("n_cmd"), "n_fail": r.get("n_fail"),
                "commands": r.get("commands", []), "vals": r.get("vals", {}),
                "warnings": r.get("warnings", [])}
    except Exception as e:  # noqa: BLE001 — 子进程崩/超时 → 降级，绝不冒泡掐 SSE
        return {"ok": False, "error": f"render_geogebra 异常: {e}", "png_path": None,
                "n_cmd": len(commands or []), "n_fail": len(commands or []), "commands": [],
                "vals": {}, "warnings": [str(e)[:80]]}
