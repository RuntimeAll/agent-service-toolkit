# -*- coding: utf-8 -*-
"""PRD-C-017 B5-fix4 · 老师改题型时重生按新题型重构题面（不再"改了个寂寞"）。

bug：老师把变式题型从「填空」改成「解答」→ 点重生 → 重出还是填空。
root cause：REGEN_PROMPT 是「等价变式·换数字保型」框架，传入的新 qtype 是弱信号被模型无视。
fix：当本题 dirty_dims 含 "qtype"（=老师 edit-dna 显式改了题型）→ _regen_once 往 prompt 注一段
     强指令（_qtype_change_clause），压过保型框架，要求按新题型标准结构重构题面。

本测纯文本断言（monkeypatch _ainvoke_text 捕获真实拼出的 prompt，零网络）：
- 改了题型（dirty_dims 含 qtype）→ prompt 含改题型重构关键词。
- 没改题型（dirty_dims 不含 qtype）→ prompt 不含该指令（保型路不回归）。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import _regen_once

_FACTS = {
    "kp_name": "一元一次方程",
    "grade": "七年级上学期",
    "qtype": "填空",
    "dna": {
        "main_kp": {"id": "30710101", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "填空", "exam_type": "直接计算",
        "skeleton": ["移项"], "hard_points": [], "tags": ["解方程"],
        "scene": "纯代数", "difficulty": 3, "flags": [],
    },
}


def _capture_prompt(monkeypatch):
    """monkeypatch _ainvoke_text 捕获 prompt 文本，返回 captured dict。"""
    captured = {}

    async def fake_llm(messages, *a, **k):
        captured["prompt"] = messages[0].content
        # 返回一个解答结构的草稿（解析成功，函数走完整路径）
        return (
            '{"stem":"求 x 的值，写出完整解题过程：2x+1=5","answer":"x=2",'
            '"solution":"移项得 2x=4，解得 x=2","qtype":"解答","difficulty":3,'
            '"level":"normal","injected_kp":null,"verify_payload":{"kind":"none","reason":"-"}}'
        )

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    return captured


def test_qtype_dirty_injects_rebuild_instruction(monkeypatch):
    """① dirty_dims 含 qtype + qtype=解答 → prompt 含「改题型 → 重构为【解答】」强指令。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "解答", "difficulty": 3, "level": "normal",
        "dirty_dims": ["qtype"],  # 老师 edit-dna field=qtype 写入的脏维
    }
    draft = asyncio.run(_regen_once(item, _FACTS, feedback=None))
    assert draft is not None
    prompt = captured["prompt"]
    # 含改题型重构关键词
    assert "老师显式要求改题型" in prompt
    assert "重构" in prompt
    assert "【解答】" in prompt
    # 强指令体现"不是换数字保型"
    assert "不是" in prompt and "保型" in prompt


def test_no_qtype_dirty_no_instruction(monkeypatch):
    """② dirty_dims 不含 qtype → prompt 不含改题型重构指令（保型路不回归）。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "填空", "difficulty": 3, "level": "normal",
        "dirty_dims": ["difficulty"],  # 只改了难度，没改题型
    }
    draft = asyncio.run(_regen_once(item, _FACTS, feedback=None))
    assert draft is not None
    prompt = captured["prompt"]
    assert "老师显式要求改题型" not in prompt


def test_no_dirty_dims_at_all_no_instruction(monkeypatch):
    """②补 dirty_dims 缺失（普通重生，非编辑）→ 同样不注（保型路不回归）。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "填空", "difficulty": 3, "level": "normal",
    }
    draft = asyncio.run(_regen_once(item, _FACTS, feedback=None))
    assert draft is not None
    assert "老师显式要求改题型" not in captured["prompt"]
