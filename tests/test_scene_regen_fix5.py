# -*- coding: utf-8 -*-
"""PRD-C-017 B5-fix5 · 老师改场景时重生按指定场景重写题面（不再"改了个寂寞"）。

bug：老师把变式场景改成「行程问题」→ 点重生 → 题面还是原场景（与 qtype 同构坑）。
root cause：新场景写进 mother_dna.dna.scene（组级共享维）→ facts["dna"]["scene"]，但没有任何
     重生 prompt 消费这个键（REGEN_PROMPT 保型框架 + 守恒段④只说"场景可换同类"= 让模型随机换/沿用）。
fix：当本题 dirty_dims 含 "scene" 且 facts.dna.scene 非空 → _regen_once 往 prompt 注一段强指令
     （_scene_change_clause），压过保型/随机换场景，要求把题面改写到老师指定的场景下。

本测纯文本断言（monkeypatch _ainvoke_text 捕获真实拼出的 prompt，零网络）：
- 改了场景（dirty_dims 含 scene + dna.scene=新值）→ prompt 含场景重构指令（断言含新场景文本）。
- 没改场景（dirty_dims 不含 scene）→ prompt 不含该指令（保型/随机换场景路不回归）。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import _regen_once


def _facts_with_scene(scene: str) -> dict:
    return {
        "kp_name": "一元一次方程",
        "grade": "七年级上学期",
        "qtype": "填空",
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [], "qtype": "填空", "exam_type": "直接计算",
            "skeleton": ["移项"], "hard_points": [], "tags": ["解方程"],
            "scene": scene, "difficulty": 3, "flags": [],
        },
    }


def _capture_prompt(monkeypatch):
    """monkeypatch _ainvoke_text 捕获 prompt 文本，返回 captured dict。"""
    captured = {}

    async def fake_llm(messages, *a, **k):
        captured["prompt"] = messages[0].content
        return (
            '{"stem":"小明从家到学校走了若干米：方程 2x+1=5 的解是 x=____。",'
            '"answer":"x=2","solution":"移项得 2x=4，解得 x=2","qtype":"填空",'
            '"difficulty":3,"level":"normal","injected_kp":null,'
            '"verify_payload":{"kind":"none","reason":"-"}}'
        )

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    return captured


def test_scene_dirty_injects_rewrite_instruction(monkeypatch):
    """① dirty_dims 含 scene + dna.scene=行程问题 → prompt 含「指定了新场景【行程问题】」强指令。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "填空", "difficulty": 3, "level": "normal",
        "dirty_dims": ["scene"],  # 老师 edit-dna field=scene 写入的脏维
    }
    draft = asyncio.run(_regen_once(item, _facts_with_scene("行程问题"), feedback=None))
    assert draft is not None
    prompt = captured["prompt"]
    # 含场景改写关键词 + 老师指定的新场景文本
    assert "老师显式指定了新场景" in prompt
    assert "行程问题" in prompt
    assert "改写到这个场景" in prompt
    # 强指令体现"不是随机换、不是保留原场景"
    assert "不是" in prompt


def test_no_scene_dirty_no_instruction(monkeypatch):
    """② dirty_dims 不含 scene → prompt 不含场景改写指令（保型/随机换场景路不回归）。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "填空", "difficulty": 3, "level": "normal",
        "dirty_dims": ["difficulty"],  # 只改了难度，没改场景
    }
    draft = asyncio.run(_regen_once(item, _facts_with_scene("行程问题"), feedback=None))
    assert draft is not None
    assert "老师显式指定了新场景" not in captured["prompt"]


def test_scene_dirty_but_empty_scene_no_instruction(monkeypatch):
    """②补 dirty_dims 含 scene 但 dna.scene 为空 → 不注（无新场景可传，保型路不回归）。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "填空", "difficulty": 3, "level": "normal",
        "dirty_dims": ["scene"],
    }
    draft = asyncio.run(_regen_once(item, _facts_with_scene(""), feedback=None))
    assert draft is not None
    assert "老师显式指定了新场景" not in captured["prompt"]


def test_no_dirty_dims_at_all_no_instruction(monkeypatch):
    """②补 dirty_dims 缺失（普通重生，非编辑）→ 同样不注（保型路不回归）。"""
    captured = _capture_prompt(monkeypatch)
    item = {
        "stem": "填空：方程 2x+1=5 的解是 x=____。",
        "answer": "2", "qtype": "填空", "difficulty": 3, "level": "normal",
    }
    draft = asyncio.run(_regen_once(item, _facts_with_scene("行程问题"), feedback=None))
    assert draft is not None
    assert "老师显式指定了新场景" not in captured["prompt"]
