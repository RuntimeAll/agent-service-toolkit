# -*- coding: utf-8 -*-
"""PRD-C-100 B-mathfig：figure 子包纯函数 + 降级路径测（不打 torch/node，桩注入）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_geogebra_samples_cover_6_kinds() -> None:
    from agents.figure import geogebra_samples as gs
    kinds = {s["kind"] for s in gs.SAMPLES}
    # 6 变换骨架 + 1 标注角范式（治「首次造图漏标角」补的标注 few-shot）= 7
    assert len(gs.SAMPLES) == 7
    for k in ("旋转", "平移", "对称", "折叠"):
        assert k in kinds
    # 立体/三视图两条（名字带前缀）
    assert any("立体" in k for k in kinds)
    assert any("三视图" in k for k in kinds)
    # 标注角范式：含 Angle 标注命令的样例（名字带「标注角」前缀）
    annot = [s for s in gs.SAMPLES if "标注角" in s["kind"]]
    assert annot, "缺标注角 few-shot"
    assert any("Angle(" in c for c in annot[0]["commands"])


def test_samples_prompt_block_renders_commands_and_dashed() -> None:
    from agents.figure import geogebra_samples as gs
    block = gs.samples_prompt_block()
    assert "Rotate" in block and "Reflect" in block and "Translate" in block
    assert "dashed" in block  # 像/隐藏棱区分提示在
    assert "Point((" not in block  # 反例不该出现在样例命令里


def test_figure_crop_dedup_drops_overlap() -> None:
    from agents.figure.figure_crop import _dedup, _iou
    a = {"bbox": [0, 0, 10, 10], "conf": 0.9}
    b = {"bbox": [1, 1, 11, 11], "conf": 0.5}  # 与 a 高度重叠
    c = {"bbox": [100, 100, 110, 110], "conf": 0.8}  # 不重叠
    kept = _dedup([a, b, c])
    assert a in kept and c in kept and b not in kept
    assert _iou([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0


def test_mathfig_render_degrades_when_engine_missing(monkeypatch) -> None:
    """mathfig import 失败 → ok:False，绝不抛（护栏3：渲染失败标 needs_figure 继续）。"""
    import builtins

    from agents.figure import mathfig_render

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name.startswith("mathfig"):
            raise ImportError("mathfig not installed (simulated)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    r = mathfig_render.render(["A=(0,0)", "B=(1,1)"], stem="degrade_test")
    assert r["ok"] is False
    assert "png_path" in r and r["png_path"] is None
    assert r["n_fail"] == 2  # 全部命令计失败（降级口径）
