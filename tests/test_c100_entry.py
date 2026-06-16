# -*- coding: utf-8 -*-
"""PRD-C-100 B1a 塌缩入口（variant_entry）单测：D1 条件 confirm 门控 + 开集 kp 后锚 + prompt 结构。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import variant_entry as VE  # noqa: E402


class TestDecideConfirm:
    def test_high_conf_no_ambiguity_passes(self):
        d = VE.decide_confirm({"gradeBook": "八年级下册", "chapter": "第2章 一元二次方程", "confidence": 0.92})
        assert d["needs_confirm"] is False
        assert d["grade_book"] == "八年级下册"

    def test_low_confidence_triggers_confirm(self):
        d = VE.decide_confirm({"gradeBook": "八年级下册", "confidence": 0.6})
        assert d["needs_confirm"] is True
        assert "置信" in d["reason"]

    def test_threshold_boundary_080_passes(self):
        # 恰好 0.80 不触发（D1：<0.80 才确认）
        d = VE.decide_confirm({"gradeBook": "九年级上册", "chapter": "二次函数", "confidence": 0.80})
        assert d["needs_confirm"] is False

    def test_two_chapter_candidates_triggers_confirm(self):
        d = VE.decide_confirm({
            "gradeBook": "八年级下册", "confidence": 0.95,
            "chapterCandidates": ["第1章 二次根式", "第2章 一元二次方程"],
        })
        assert d["needs_confirm"] is True
        assert "歧义" in d["reason"]

    def test_empty_grade_book_triggers_confirm(self):
        d = VE.decide_confirm({"gradeBook": "", "confidence": 0.99})
        assert d["needs_confirm"] is True
        assert "年级册判不出" in d["reason"]

    def test_dedup_chapter_candidates(self):
        # 同名候选去重后只剩 1 → 不算歧义
        d = VE.decide_confirm({
            "gradeBook": "八年级下册", "confidence": 0.95,
            "chapterCandidates": ["第2章 一元二次方程", "第2章 一元二次方程"],
        })
        assert d["needs_confirm"] is False


class TestMatchKpInPool:
    POOL = [
        ("3082002001", "一元二次方程的定义"),
        ("3082002002", "配方法解一元二次方程"),
        ("3082002003", "根的判别式"),
    ]

    def test_exact_name_match(self):
        assert VE._match_kp_in_pool("根的判别式", self.POOL) == "3082002003"

    def test_substring_match_prefers_shortest(self):
        # opus 名「配方法」⊆ 池名「配方法解一元二次方程」→ 命中
        assert VE._match_kp_in_pool("配方法", self.POOL) == "3082002002"

    def test_no_match_returns_none(self):
        assert VE._match_kp_in_pool("勾股定理", self.POOL) is None

    def test_empty_name_returns_none(self):
        assert VE._match_kp_in_pool("", self.POOL) is None


class TestEntryPrompt:
    def test_prompt_has_judge_solve_label_and_open_set(self):
        p = VE.build_entry_prompt()
        assert "判年级册" in p and "解出来" in p and "10 维 DNA" in p
        assert "gradeBook" in p and "confidence" in p
        # 开集：id 留空由系统后锚
        assert "id 留空" in p or '"id":""' in p or '"id": ""' in p

    def test_prompt_carries_utterance_context(self):
        p = VE.build_entry_prompt(utterance="出5道压轴题")
        assert "出5道压轴题" in p

    def test_entry_schema_extends_mother_schema(self):
        props = VE.ENTRY_SCHEMA["properties"]
        for k in ("gradeBook", "chapter", "confidence", "has_figure", "richText", "dna"):
            assert k in props
        assert "gradeBook" in VE.ENTRY_SCHEMA["required"]
