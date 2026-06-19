# -*- coding: utf-8 -*-
"""PRD-C-100 配图人在回路「主动引导」信号 —— compose_variant_figure 在合适触发点附加
need_user_desc / direction_review（service 后处理 D14，四节点字节不动）。

桩注入 invoke（伪 opus）+ parse_json，控制返回的 JSON 走不同分支：
  · 触发 1/2：opus needs_figure / 空 commands + 题面含图形关键词 → need_user_desc=True。
  · 纯代数题（无图形关键词）+ 空 commands → 不催（need_user_desc 不置）。
  · 触发 3：渲染失败（monkeypatch mathfig_render.render 抛 / 返 ok:False）→ need_user_desc=True。
  · 触发 4：造图成功 + 题面/命令含方向元素（旋转/箭头/镜像）→ direction_review=True。
不破坏 needs_figure 降级契约（ok=False/needs_figure=True 照旧）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _parse_json(text):
    import json
    return json.loads(text)


def _no_budget(monkeypatch):
    from agents import cost_guard

    async def _ok():
        return False

    monkeypatch.setattr(cost_guard, "is_budget_exceeded_async", _ok)


def _invoke_returning(payload_json: str):
    async def fake_invoke(messages, **kw):
        return payload_json

    return fake_invoke


def test_trigger12_needs_figure_with_geo_keyword_sets_need_user_desc(monkeypatch) -> None:
    """opus 自评 needs_figure + 题面含图形关键词（三角形）→ need_user_desc + 补描述文案。"""
    _no_budget(monkeypatch)
    from agents.figure import compose

    r = _run(
        compose.compose_variant_figure(
            stem="如图，三角形 ABC 中，∠A 的角平分线交 BC 于 D，求证…",
            answer="略",
            invoke=_invoke_returning('{"needs_figure": true, "commands": []}'),
            parse_json=_parse_json,
            item_id="1",
        )
    )
    assert r["ok"] is False and r["needs_figure"] is True  # 降级契约不破
    assert r.get("need_user_desc") is True
    assert "图形描述" in (r.get("reason") or "")


def test_trigger2_pure_algebra_empty_commands_does_not_nag(monkeypatch) -> None:
    """纯代数题（无图形关键词）+ 空 commands → 不置 need_user_desc（不误催）。"""
    _no_budget(monkeypatch)
    from agents.figure import compose

    r = _run(
        compose.compose_variant_figure(
            stem="解方程 2x + 3 = 11，求 x 的值。",
            invoke=_invoke_returning('{"needs_figure": true, "commands": []}'),
            parse_json=_parse_json,
            item_id="2",
        )
    )
    # 🔴 PRD-A-018 RED#1：纯代数题(无图形关键词)→ needs_figure=False（无需配图，非待补图），
    #   且不置 need_user_desc（不误催）。FE 据此把配图灯跳过(done)、不卡题组就绪。
    assert r["ok"] is False and r["needs_figure"] is False
    assert r.get("need_user_desc") is not True


def test_trigger3_render_failure_sets_need_user_desc(monkeypatch) -> None:
    """渲染失败（mathfig_render.render 返 ok:False）→ need_user_desc + 补描述/重试文案。"""
    _no_budget(monkeypatch)
    from agents.figure import compose, mathfig_render

    monkeypatch.setattr(
        mathfig_render, "render",
        lambda *a, **k: {"ok": False, "png_path": None, "error": "引擎缺失", "warnings": []},
    )
    r = _run(
        compose.compose_variant_figure(
            stem="画一个正方形 ABCD",
            invoke=_invoke_returning('{"commands": ["A=(0,0)", "B=(1,0)"]}'),
            parse_json=_parse_json,
            item_id="3",
        )
    )
    assert r["ok"] is False and r["needs_figure"] is True
    assert r.get("need_user_desc") is True
    assert "图形描述" in (r.get("reason") or "")


def test_trigger4_direction_element_sets_direction_review(monkeypatch) -> None:
    """造图成功 + 题面含方向元素（旋转）→ direction_review + 方向待确认文案（图照常交付）。"""
    _no_budget(monkeypatch)
    from agents.figure import compose, mathfig_render

    monkeypatch.setattr(
        mathfig_render, "render",
        lambda *a, **k: {"ok": True, "png_path": __file__, "vals": {}, "warnings": []},
    )
    # _png_to_b64 读 __file__ 能拿到 bytes（非 None），让成功路径走通。
    r = _run(
        compose.compose_variant_figure(
            stem="将三角形 ABC 绕点 O 顺时针旋转 90°，画出旋转后的图形。",
            answer="略",
            invoke=_invoke_returning('{"commands": ["A=(0,0)", "tri=Polygon(A,B,C)"]}'),
            parse_json=_parse_json,
            item_id="4",
        )
    )
    assert r["ok"] is True and r["needs_figure"] is False  # 成功路径不破
    assert r.get("direction_review") is True
    assert "方向" in (r.get("reason") or "")


def test_success_no_direction_keyword_no_review(monkeypatch) -> None:
    """造图成功 + 无方向元素 → 不置 direction_review（不误标）。"""
    _no_budget(monkeypatch)
    from agents.figure import compose, mathfig_render

    monkeypatch.setattr(
        mathfig_render, "render",
        lambda *a, **k: {"ok": True, "png_path": __file__, "vals": {}, "warnings": []},
    )
    r = _run(
        compose.compose_variant_figure(
            stem="画一个边长为 3 的正方形 ABCD。",
            invoke=_invoke_returning('{"commands": ["A=(0,0)", "sq=Polygon(A,B,C,D)"]}'),
            parse_json=_parse_json,
            item_id="5",
        )
    )
    assert r["ok"] is True
    assert r.get("direction_review") is not True


def test_direction_from_commands_only(monkeypatch) -> None:
    """题面无方向词但 commands 含 Rotate → 仍标 direction_review（命令也是信号源）。"""
    _no_budget(monkeypatch)
    from agents.figure import compose, mathfig_render

    monkeypatch.setattr(
        mathfig_render, "render",
        lambda *a, **k: {"ok": True, "png_path": __file__, "vals": {}, "warnings": []},
    )
    r = _run(
        compose.compose_variant_figure(
            stem="如图，已知点 A、B、C，作出对应图形。",
            invoke=_invoke_returning('{"commands": ["A=(0,0)", "Ap=Rotate(A,90deg,B)"]}'),
            parse_json=_parse_json,
            item_id="6",
        )
    )
    assert r["ok"] is True
    assert r.get("direction_review") is True
