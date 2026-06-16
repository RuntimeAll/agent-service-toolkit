# -*- coding: utf-8 -*-
"""PRD-C-100 B3 带图管线 · compose 降级路径单测（G4/G11：失败必 needs_figure，不抛不卡流程）。

不打真 opus/node：invoke/parse_json 桩注入；mathfig_render monkeypatch。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.figure import compose  # noqa: E402


def _parse_json(text):
    import json
    try:
        return json.loads(text)
    except Exception:
        return None


class TestComposeVariantFigure:
    @pytest.mark.asyncio
    async def test_opus_translate_failure_degrades(self):
        async def invoke(messages, **kw):
            raise RuntimeError("relay down")
        r = await compose.compose_variant_figure(
            stem="题", invoke=invoke, parse_json=_parse_json, item_id="x")
        assert r["ok"] is False and r["needs_figure"] is True
        assert "翻命令失败" in r["reason"]

    @pytest.mark.asyncio
    async def test_opus_says_needs_figure_degrades(self):
        async def invoke(messages, **kw):
            return '{"commands":[],"needs_figure":true}'
        r = await compose.compose_variant_figure(
            stem="纯代数题", invoke=invoke, parse_json=_parse_json)
        assert r["ok"] is False and r["needs_figure"] is True

    @pytest.mark.asyncio
    async def test_render_failure_degrades(self, monkeypatch):
        async def invoke(messages, **kw):
            return '{"commands":["A=(0,0)","B=(1,1)"],"dashed":[]}'
        monkeypatch.setattr(
            compose.mathfig_render, "render",
            lambda *a, **k: {"ok": False, "error": "node missing", "png_path": None,
                             "warnings": ["x"]},
        )
        r = await compose.compose_variant_figure(
            stem="题", invoke=invoke, parse_json=_parse_json)
        assert r["ok"] is False and r["needs_figure"] is True
        assert "node missing" in r["reason"]

    @pytest.mark.asyncio
    async def test_success_returns_base64(self, monkeypatch, tmp_path):
        png = tmp_path / "f.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
        async def invoke(messages, **kw):
            return '{"commands":["A=(0,0)","B=(4,0)","C=(1,3)","t=Polygon(A,B,C)"],"dashed":[]}'
        monkeypatch.setattr(
            compose.mathfig_render, "render",
            lambda *a, **k: {"ok": True, "png_path": str(png), "n_fail": 0,
                             "vals": {"C": "(1,3)"}, "warnings": []},
        )
        r = await compose.compose_variant_figure(
            stem="题", invoke=invoke, parse_json=_parse_json, item_id="t1")
        assert r["ok"] is True and r["needs_figure"] is False
        assert r["png_base64"] and len(r["png_base64"]) > 0
        assert r["item_id"] == "t1"

    @pytest.mark.asyncio
    async def test_correction_prompt_in_messages(self, monkeypatch, tmp_path):
        captured = {}
        async def invoke(messages, **kw):
            captured["text"] = "\n".join(
                str(m.content) for m in messages if hasattr(m, "content"))
            return '{"commands":["A=(0,0)"],"dashed":[]}'
        png = tmp_path / "f.png"; png.write_bytes(b"PNG")
        monkeypatch.setattr(compose.mathfig_render, "render",
                            lambda *a, **k: {"ok": True, "png_path": str(png), "n_fail": 0})
        await compose.compose_variant_figure(
            stem="题", invoke=invoke, parse_json=_parse_json,
            correction_prompt="把三角形改成钝角")
        assert "把三角形改成钝角" in captured["text"]  # 图片重生修正词进了 prompt


class TestCropMotherFigure:
    @pytest.mark.asyncio
    async def test_download_failure_degrades(self):
        # 非法 url → 下载失败 → needs_figure（降级不抛）
        r = await compose.crop_mother_figure("http://127.0.0.1:1/nonexistent.png")
        assert r["ok"] is False
        assert r["needs_figure"] is True
