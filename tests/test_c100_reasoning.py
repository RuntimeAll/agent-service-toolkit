# -*- coding: utf-8 -*-
"""PRD-C-100 B6·D18 思考流式后端管线：relay 捞 reasoning_content + _emit_reasoning 帧形。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _Chunk:
    """伪 langchain chunk：content + additional_kwargs.reasoning_content。"""

    def __init__(self, content="", reasoning=None):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}
        self.response_metadata = {}
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}

    def __add__(self, other):
        return _Chunk(self.content + (other.content or ""))


class _StubLLM:
    def bind(self, **kw):
        return self

    async def astream(self, messages, config=None):
        # 思考型：先吐 reasoning，再吐 content
        yield _Chunk(reasoning="先分析题目…")
        yield _Chunk(reasoning="再算一步…")
        yield _Chunk(content='{"ok":true}')


@pytest.mark.asyncio
async def test_relay_captures_reasoning(monkeypatch):
    from core import relay_pool
    monkeypatch.setattr(relay_pool, "_relays",
                        lambda: [relay_pool.Relay("stub", "http://x", "k", "m")])
    monkeypatch.setattr(relay_pool, "_chat", lambda relay: _StubLLM())
    monkeypatch.setattr(relay_pool, "_chat_override", lambda *a, **k: _StubLLM())

    seen = []
    resp, name, model, fb, _detail = await relay_pool.ainvoke_failover(
        [], max_tokens=100, on_reasoning=lambda t: seen.append(t))
    # reasoning 累计：第一帧「先分析题目…」，第二帧累加
    assert seen and seen[-1] == "先分析题目…再算一步…"
    # content 仍在 resp
    assert "ok" in resp.content


def test_emit_reasoning_frame_shape(monkeypatch):
    from agents import variant
    sent = []

    class _W:
        def __call__(self, msg):
            sent.append(msg)

    monkeypatch.setattr(variant, "get_stream_writer", lambda: _W())
    variant._emit_reasoning("思考中的内容")
    assert sent, "应发一帧"
    content = sent[0].content
    assert isinstance(content, list) and len(content) == 1
    assert "reasoning" in content[0]
    assert content[0]["reasoning"]["text"] == "思考中的内容"


def test_emit_reasoning_empty_noop(monkeypatch):
    from agents import variant
    sent = []
    monkeypatch.setattr(variant, "get_stream_writer", lambda: (lambda m: sent.append(m)))
    variant._emit_reasoning("")  # 空 → no-op
    assert not sent
