# -*- coding: utf-8 -*-
"""PRD-A-021 R3b·章节×图型定型闸单测（章节命中→允许集 / 空集→逃生 / 两落点注入 / 表读不到降级）。

不打真 DB/opus：chapter_figure 缓存用 _seed_cache_for_test 注入桩表；compose 的 invoke/render 桩注入。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.figure import chapter_figure, compose  # noqa: E402


def _parse_json(text):
    import json
    try:
        return json.loads(text)
    except Exception:
        return None


@pytest.fixture(autouse=True)
def _reset_cache():
    chapter_figure.reset_cache()
    yield
    chapter_figure.reset_cache()


# ===========================================================================
# 1) allowed_figure_types：章节/考点命中 → 允许集正确（包含匹配 + 并集）
# ===========================================================================
class TestAllowedFigureTypes:
    def test_chapter_hit_union(self):
        chapter_figure._seed_cache_for_test([
            ("有理数", "number_line"),
            ("数轴", "number_line"),
            ("一次函数", "cartesian"),
            ("一次函数", "line_func"),
            ("圆", "circle"),
        ])
        # 章节名「有理数与数轴」+ 考点「数轴上的点」→ 命中 有理数/数轴 两行 → {number_line}
        assert chapter_figure.allowed_figure_types("有理数与数轴", "数轴上的点") == {"number_line"}

    def test_multi_row_keyword_union(self):
        chapter_figure._seed_cache_for_test([
            ("一次函数", "cartesian"),
            ("一次函数", "line_func"),
            ("函数", "cartesian"),
        ])
        # 「一次函数的图象」命中 一次函数(2行) + 函数(1行) → 并集 {cartesian, line_func}
        assert chapter_figure.allowed_figure_types("一次函数的图象", None) == {"cartesian", "line_func"}

    def test_kp_name_also_matched(self):
        chapter_figure._seed_cache_for_test([("旋转", "transform")])
        # 章节名不含、但考点名含 → 仍命中（章节+考点拼起来匹配）
        assert chapter_figure.allowed_figure_types("图形的变化", "图形的旋转") == {"transform"}

    def test_case_insensitive(self):
        chapter_figure._seed_cache_for_test([("Circle", "circle")])
        assert chapter_figure.allowed_figure_types("圆 Circle 专题", None) == {"circle"}


# ===========================================================================
# 2) 逃生窗口：空集（无章节/无匹配/表不可用）→ 不约束
# ===========================================================================
class TestEscapeWindow:
    def test_no_chapter_no_kp_escape(self):
        chapter_figure._seed_cache_for_test([("数轴", "number_line")])
        assert chapter_figure.allowed_figure_types(None, None) == set()
        assert chapter_figure.allowed_figure_types("", "") == set()

    def test_no_keyword_match_escape(self):
        chapter_figure._seed_cache_for_test([("数轴", "number_line")])
        # 章节名与任何 keyword 都不沾 → 空集（逃生）
        assert chapter_figure.allowed_figure_types("勾股定理", "直角三角形") == set()

    def test_table_unavailable_degrades_to_escape(self, monkeypatch):
        # 模拟表读不到（dev 未 apply 迁移 / 库故障）：_load_map 走异常分支 → 空集 + unavailable
        def _boom(**kw):
            raise RuntimeError("table doesn't exist")
        monkeypatch.setattr(chapter_figure.pymysql, "connect", _boom)
        chapter_figure.reset_cache()
        assert chapter_figure.allowed_figure_types("有理数与数轴", "数轴") == set()
        assert chapter_figure._MAP_UNAVAILABLE is True

    def test_empty_table_escape(self, monkeypatch):
        chapter_figure._seed_cache_for_test([])  # 表存在但空
        assert chapter_figure.allowed_figure_types("有理数", "数轴") == set()

    def test_constraint_clause_empty_on_empty_set(self):
        assert chapter_figure.constraint_clause(set()) == ""

    def test_constraint_clause_nonempty(self):
        clause = chapter_figure.constraint_clause({"number_line"})
        assert "number_line" in clause and "数轴" in clause
        assert "只能在这些图型范围内" in clause


# ===========================================================================
# 3) 落点②·compose fallback：定型约束拼进 compose prompt（spec 路径 + fallback 路径都拼）
# ===========================================================================
class TestComposeInjection:
    def _stub_render(self, monkeypatch):
        """让渲染恒成功（绕 mathfig/node），_png_to_b64 返桩 base64。"""
        from agents.figure import mathfig_render
        monkeypatch.setattr(
            mathfig_render, "render",
            lambda *a, **k: {"ok": True, "png_path": "/tmp/x.png", "vals": {}, "warnings": []},
        )
        monkeypatch.setattr(compose, "_png_to_b64", lambda p: "ZmFrZQ==")

    @pytest.mark.asyncio
    async def test_fallback_path_injects_constraint(self, monkeypatch):
        """figure_spec 空（= add/regen/库内母题主路径）+ 章节命中 → fallback prompt 含定型约束。"""
        self._stub_render(monkeypatch)
        chapter_figure._seed_cache_for_test([("数轴", "number_line")])
        captured = {}

        async def invoke(messages, **kw):
            captured["user"] = messages[1].content
            import json
            return json.dumps({"commands": ["A=(0,0)"], "needs_figure": False})

        r = await compose.compose_variant_figure(
            stem="在数轴上表示 -3", invoke=invoke, parse_json=_parse_json,
            item_id="1", figure_spec=None, chapter="有理数与数轴", kp="数轴",
        )
        assert r["ok"] is True
        assert r["figure_type_constrained"] is True
        assert r["allowed_figure_types"] == ["number_line"]
        # fallback 退化提示 + 定型约束都在 user 段
        assert "退化提示" in captured["user"]
        assert "本题章节定型约束" in captured["user"]
        assert "number_line" in captured["user"]

    @pytest.mark.asyncio
    async def test_spec_path_injects_constraint(self, monkeypatch):
        """figure_spec 非空（正常出题路径）+ 章节命中 → spec prompt 也含定型约束。"""
        self._stub_render(monkeypatch)
        chapter_figure._seed_cache_for_test([("圆", "circle")])
        captured = {}

        async def invoke(messages, **kw):
            captured["user"] = messages[1].content
            import json
            return json.dumps({"commands": ["c=Circle((0,0),3)"], "needs_figure": False})

        r = await compose.compose_variant_figure(
            stem="圆的题", invoke=invoke, parse_json=_parse_json, item_id="2",
            figure_spec={"layout": "圆 O 半径 3", "angle_labels": []},
            chapter="圆的基本性质", kp="圆心角",
        )
        assert r["ok"] is True and r["figure_type_constrained"] is True
        assert "配图决策" in captured["user"]  # spec 路径标志
        assert "本题章节定型约束" in captured["user"]
        assert "circle" in captured["user"]

    @pytest.mark.asyncio
    async def test_escape_no_constraint_in_prompt(self, monkeypatch):
        """无映射（逃生）→ prompt **不含**定型约束段，opus 自由翻命令。"""
        self._stub_render(monkeypatch)
        chapter_figure._seed_cache_for_test([("数轴", "number_line")])
        captured = {}

        async def invoke(messages, **kw):
            captured["user"] = messages[1].content
            import json
            return json.dumps({"commands": ["A=(0,0)"], "needs_figure": False})

        r = await compose.compose_variant_figure(
            stem="勾股定理题", invoke=invoke, parse_json=_parse_json, item_id="3",
            figure_spec=None, chapter="勾股定理", kp="直角三角形",  # 不命中任何 keyword
        )
        assert r["ok"] is True
        assert r["figure_type_constrained"] is False
        assert r["allowed_figure_types"] == []
        assert "本题章节定型约束" not in captured["user"]

    @pytest.mark.asyncio
    async def test_chapter_kp_default_none_no_crash(self, monkeypatch):
        """chapter/kp 缺省（老调用不传）→ 逃生，不崩。"""
        self._stub_render(monkeypatch)
        chapter_figure._seed_cache_for_test([("数轴", "number_line")])

        async def invoke(messages, **kw):
            import json
            return json.dumps({"commands": ["A=(0,0)"], "needs_figure": False})

        r = await compose.compose_variant_figure(
            stem="题", invoke=invoke, parse_json=_parse_json, item_id="4",
        )
        assert r["ok"] is True and r["figure_type_constrained"] is False


# ===========================================================================
# 4) 落点①·GENERATE gate block（variant._figure_type_gate_block）
# ===========================================================================
class TestGenerateGateBlock:
    def test_gate_block_hit(self):
        from agents import variant
        chapter_figure._seed_cache_for_test([("数轴", "number_line"), ("有理数", "number_line")])
        state = {"confirmed_chapter_name": "有理数与数轴"}
        facts = {"kp_name": "数轴上的点"}
        block = variant._figure_type_gate_block(state, facts)
        assert "本题章节定型约束" in block and "number_line" in block

    def test_gate_block_escape_empty(self):
        from agents import variant
        chapter_figure._seed_cache_for_test([("数轴", "number_line")])
        # 章节取不到（state 无章名）→ 空集 → 空串（逃生）
        block = variant._figure_type_gate_block({}, {"kp_name": "勾股定理"})
        assert block == ""

    def test_gate_block_table_unavailable_escape(self, monkeypatch):
        from agents import variant
        def _boom(**kw):
            raise RuntimeError("no table")
        monkeypatch.setattr(chapter_figure.pymysql, "connect", _boom)
        chapter_figure.reset_cache()
        block = variant._figure_type_gate_block(
            {"confirmed_chapter_name": "有理数与数轴"}, {"kp_name": "数轴"}
        )
        assert block == ""  # 表读不到 → 逃生
