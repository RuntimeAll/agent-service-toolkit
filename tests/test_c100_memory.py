# -*- coding: utf-8 -*-
"""PRD-C-100 B4 记忆层 toolkit 侧：format/fetch 注入 + 自动写（桩 client，不打真 HTTP）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import teacher_memory as TM  # noqa: E402


class _StubClient:
    def __init__(self, rows=None, fail=False):
        self._rows = rows or []
        self._fail = fail
        self.added: list[dict] = []

    async def list_ai_memory(self, *, enabled_only=False, mem_type=None):
        if self._fail:
            raise RuntimeError("ruoyi down")
        return self._rows

    async def add_ai_memory(self, **kw):
        self.added.append(kw)
        return {"id": 1, **kw}


class TestFormatMemoryBlock:
    def test_empty_returns_none(self):
        assert TM.format_memory_block([]) is None
        assert TM.format_memory_block([{"memValue": ""}]) is None

    def test_formats_enabled_rows(self):
        rows = [
            {"memType": "偏好", "memKey": "常教年级册", "memValue": "八下", "enabled": 1},
            {"memType": "纠正", "memKey": "教材版本", "memValue": "浙教版", "enabled": 1},
        ]
        blk = TM.format_memory_block(rows)
        assert "[偏好] 常教年级册：八下" in blk
        assert "[纠正] 教材版本：浙教版" in blk

    def test_disabled_excluded(self):
        rows = [{"memType": "习惯", "memKey": "k", "memValue": "v", "enabled": 0}]
        assert TM.format_memory_block(rows) is None  # 停用不注入 G14


class TestFetchMemoryBlock:
    @pytest.mark.asyncio
    async def test_fetch_enabled(self):
        c = _StubClient(rows=[{"memType": "偏好", "memKey": "x", "memValue": "y", "enabled": 1}])
        blk = await TM.fetch_memory_block(c)
        assert "y" in blk

    @pytest.mark.asyncio
    async def test_fetch_failure_degrades_to_none(self):
        c = _StubClient(fail=True)
        assert await TM.fetch_memory_block(c) is None


class TestAutoWrite:
    @pytest.mark.asyncio
    async def test_write_preference_on_persist(self):
        c = _StubClient()
        await TM.write_preference_on_persist(c, grade_book="八下", qtype="选择")
        keys = {(a["mem_type"], a["mem_key"], a["mem_value"]) for a in c.added}
        assert ("偏好", "常教年级册", "八下") in keys
        assert ("偏好", "常出题型", "选择") in keys
        assert all(a["source"] == "自动" for a in c.added)

    @pytest.mark.asyncio
    async def test_write_correction_grade_chapter(self):
        c = _StubClient()
        await TM.write_correction_grade_chapter(c, grade_book="九上", chapter="第2章")
        keys = {(a["mem_type"], a["mem_key"]) for a in c.added}
        assert ("纠正", "常教年级册") in keys and ("纠正", "常确认章") in keys

    @pytest.mark.asyncio
    async def test_write_skips_none(self):
        c = _StubClient()
        await TM.write_preference_on_persist(c, grade_book=None, qtype=None)
        assert c.added == []
