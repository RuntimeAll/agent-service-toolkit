# -*- coding: utf-8 -*-
"""PRD-C-100 B2 缓存接缝（AC8）：稳定前缀 ‖ 变量后缀（teacher 记忆后置）+ conv_trace cached_tokens 捕获。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import conv_trace, variant_entry as VE  # noqa: E402


class TestCacheSeamStructure:
    """AC8：prompt 切成稳定前缀 ‖ 变量后缀；前缀字节级稳定；变量全在后缀。"""

    def test_prefix_byte_stable_no_volatile(self):
        p = VE.ENTRY_SYSTEM_PREFIX
        for bad in ("teacher_id", "{utterance}", "userId", "2026-", "now()"):
            assert bad not in p

    def test_prefix_idempotent(self):
        assert VE._build_entry_system_prefix() == VE.ENTRY_SYSTEM_PREFIX
        assert VE._build_entry_system_prefix() == VE._build_entry_system_prefix()

    def test_variables_only_in_suffix(self):
        msgs = VE.build_entry_messages(
            image_url="http://oss/q.png", utterance="出5道难题", teacher_memory="常教八年级下册",
        )
        assert [m.__class__.__name__ for m in msgs] == ["SystemMessage", "HumanMessage"]
        sys_txt = msgs[0].content
        assert sys_txt == VE.ENTRY_SYSTEM_PREFIX  # system = 稳定前缀
        usr_text = " ".join(p.get("text", "") for p in msgs[1].content if isinstance(p, dict))
        # 变量（query + teacher 记忆）在后缀
        assert "出5道难题" in usr_text and "常教八年级下册" in usr_text
        # 变量绝不进前缀
        assert "出5道难题" not in sys_txt and "常教八年级下册" not in sys_txt
        # 题图在后缀
        assert any(p.get("type") == "image_url" for p in msgs[1].content if isinstance(p, dict))

    def test_teacher_memory_after_query_in_suffix(self):
        # teacher 记忆放变量后缀（per-teacher，进前缀毁多用户共享）
        msgs = VE.build_entry_messages(image_url="http://oss/q.png", teacher_memory="记忆X")
        usr_text = " ".join(p.get("text", "") for p in msgs[1].content if isinstance(p, dict))
        assert "记忆X" in usr_text
        assert "记忆X" not in msgs[0].content

    def test_no_image_only_prefix_when_no_vars(self):
        msgs = VE.build_entry_messages(image_url="http://oss/q.png")
        # 无 utterance/memory → user 只有题图块
        assert all(p.get("type") == "image_url" for p in msgs[1].content if isinstance(p, dict))


class TestCachedTokensCapture:
    """B2：conv_trace 从 usage 捕获 cached_tokens（aigeek 自动缓存对账列）。"""

    def test_openai_style_cached_tokens(self):
        raw = {"response_metadata": {"token_usage": {"prompt_tokens_details": {"cached_tokens": 1280}}}}
        assert conv_trace.cached_tokens_of(raw) == 1280

    def test_langchain_input_token_details(self):
        raw = {"usage_metadata": {"input_token_details": {"cache_read": 512}}}
        assert conv_trace.cached_tokens_of(raw) == 512

    def test_native_claude_cache_read(self):
        raw = {"response_metadata": {"usage": {"cache_read_input_tokens": 999}}}
        assert conv_trace.cached_tokens_of(raw) == 999

    def test_no_cache_field_returns_none(self):
        assert conv_trace.cached_tokens_of({"response_metadata": {"token_usage": {"prompt_tokens": 100}}}) is None
        assert conv_trace.cached_tokens_of({}) is None
        assert conv_trace.cached_tokens_of(None) is None
