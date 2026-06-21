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


def _load_render_geogebra():
    """取 mathfig 的 render_geogebra。
    🔴 2026-06-21（用户终审「配图生成失败」根因）：prod 容器服务跑 `python run_service.py`，
       sys.path[0]=/app；而 mathfig 源码 vendored 在 `/app/mathfig`（repo 根，无 __init__）→ 被 Python
       当**命名空间包 mathfig** 遮蔽 pip -e 装的真包 → `from mathfig.geogebra import render_geogebra`
       拿到的是命名空间目录 `/app/mathfig/geogebra/`（无该函数）→ 报 `cannot import name
       'render_geogebra' from 'mathfig.geogebra' (unknown location)` → 造图全失败（crop 不受影响，它走
       doclayout_yolo 非 mathfig）。先试正常 import；遮蔽时按**文件路径直载**真模块——geogebra.py 自包含
       （仅 os/json/subprocess，无相对导入），路径直载安全，其 `__file__` 推出的 _ROOT/render.js/node_cwd
       仍指向 /app/mathfig 的 node 资源，无需改 env/Dockerfile。"""
    try:
        from mathfig.geogebra import render_geogebra  # 正常路径（pip install -e）
        return render_geogebra
    except Exception:
        import importlib.util
        cands: list[str] = []
        try:
            import mathfig as _m  # 真包时 __path__=[.../mathfig/mathfig]，命名空间时=[/app/mathfig]
            for base in getattr(_m, "__path__", []):
                cands.append(os.path.join(base, "geogebra.py"))
        except Exception:
            pass
        cands += ["/app/mathfig/mathfig/geogebra.py", "/app/mathfig_src/mathfig/geogebra.py"]
        for p in cands:
            if os.path.isfile(p):
                spec = importlib.util.spec_from_file_location("mathfig_geogebra_real", p)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    fn = getattr(mod, "render_geogebra", None)
                    if callable(fn):
                        return fn
        raise ImportError("render_geogebra 未找到（namespace 遮蔽且文件兜底未命中）")


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
    angle_labels: dict | None = None,
) -> dict[str, Any]:
    """一轮直出渲染。返回归一 dict：
       {ok, png_path?, n_cmd, n_fail, commands:[{cmd,ok,err}], vals, warnings, view_final, error?}。
    🔴 任何异常（import 失败/node 缺/子进程崩/超时）→ {ok:False, error:...}，绝不抛。
    🔴 fig_scale（<1 缩画布让标签相对放大；无头渲染下 fontSize 无效，figScale 是唯一杠杆）、
       point_size（=0 隐圆点只留字母）—— 均透传本地 render_geogebra，None 走引擎默认（不传 = 字小、点全显）。
    🔴 relabel（{内部点名:"显示文字"}，如 {"Ap":"A′"}）：旋转/对称像标识符不能含撇号，relabel 让
       自动标签印 A′/B′ 而非内部名 Ap/Bp，opus 不必再手放 Text 撇号标签（双标打架根因）。
    🔴 angle_labels（{角对象名:"显示文字"}，如 {"a2":"30°"}）：默认不传——Angle() 对象由引擎按角平分线
       自动出实测度数（不手放 Text 角度数字，治「角度数字坐标手放必偏」）。仅示意图（画的角≠题面标的
       度数 / 标 α 等符号）时对该角显式覆盖。同顶点多角弧引擎自动按角大小递增半径错开（不传任何字段即生效）。
    """
    _ensure_env()
    try:
        render_geogebra = _load_render_geogebra()  # 进程内取（含 namespace 遮蔽兜底，见函数注释）
    except Exception as e:  # noqa: BLE001 — 引擎不可用 → 降级（不抛，调用方标 needs_figure）
        return {"ok": False, "error": f"mathfig import 失败: {e}", "png_path": None,
                "n_cmd": len(commands or []), "n_fail": len(commands or []), "commands": [],
                "vals": {}, "warnings": ["mathfig 引擎不可用"]}
    try:
        r = render_geogebra(
            list(commands or []), dashed=dashed, hide=hide, vals=vals, styles=styles,
            axes=axes, grid=grid, mono=True, stem=stem, timeout=RENDER_TIMEOUT_S,
            fig_scale=fig_scale, point_size=point_size, relabel=relabel,
            angle_labels=angle_labels,
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
