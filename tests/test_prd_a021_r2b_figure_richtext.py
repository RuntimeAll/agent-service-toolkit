# -*- coding: utf-8 -*-
"""PRD-A-021 R2b·U1+U8 单测：配图持久化（生成态 base64 入 state 取回）+ 母题富文本免转义哨兵框。

U1：set_item_figure_state 写生成态 figure_base64 → 走 items merge reducer → 从 state 取回
    （刷新即取回的 checkpoint 持久化骨架）；撤图清两态；reducer 同 stem 续/异 stem 不嫁接。
U8：extract_sentinel_richtext 从「JSON + 尾随哨兵框」原文抠出 stem/answer/analysis 三段，
    绕开 JSON 字符串转义坑（段内含未转义引号/$LaTeX$/换行/<table> 照样不坏）；一段缺不连累另两段。

全部纯函数 / 零网络。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.variant import merge_items, set_item_figure_state  # noqa: E402
from agents.variant_entry import extract_sentinel_richtext  # noqa: E402


# ---------------------------------------------------------------------------
# U1 · 生成态配图 base64 入 state checkpoint（刷新取回）
# ---------------------------------------------------------------------------
def test_u1_set_figure_base64_writes_state_and_roundtrips():
    """生成态 base64 写进 state.items[i].figure_base64 → 取回（= 刷新从 checkpoint 取回的骨架）。"""
    state = {"items": [{"stem": "题1", "_seq": 1}, {"stem": "题2", "_seq": 2}]}
    update, item, err = set_item_figure_state(
        state, 1, figure_url=None, figure_base64="iVBORw0KGgoAAAA=="
    )
    assert err is None
    assert item["figure_base64"] == "iVBORw0KGgoAAAA=="
    # 取回：update.items 即新 state（端点 aupdate_state 回写 checkpointer）
    assert update["items"][0]["figure_base64"] == "iVBORw0KGgoAAAA=="
    assert "figure_base64" not in update["items"][1]  # 只动第 1 题


def test_u1_set_figure_url_and_base64_together():
    """入库态 url + 生成态 base64 可一并写。"""
    state = {"items": [{"stem": "题1", "_seq": 1}]}
    update, item, err = set_item_figure_state(
        state, 1, figure_url="https://oss.example/x.png", figure_base64="AAA"
    )
    assert err is None
    assert item["figure_url"] == "https://oss.example/x.png"
    assert item["figure_base64"] == "AAA"


def test_u1_revoke_clears_both_figure_states():
    """撤图：两态都传空 → figure_url + figure_base64 都清。"""
    state = {"items": [{"stem": "题1", "_seq": 1, "figure_url": "https://oss/x.png",
                        "figure_base64": "AAA"}]}
    update, item, err = set_item_figure_state(state, 1, figure_url=None, figure_base64="")
    assert err is None
    assert "figure_url" not in item
    assert "figure_base64" not in item


def test_u1_legacy_call_without_base64_does_not_touch_base64():
    """旧调用（只传 figure_url，base64 缺省 None）→ 不碰已有 figure_base64（向后兼容）。"""
    state = {"items": [{"stem": "题1", "_seq": 1, "figure_base64": "KEEP"}]}
    update, item, err = set_item_figure_state(state, 1, figure_url="https://oss/x.png")
    assert err is None
    assert item["figure_base64"] == "KEEP"  # 未传 base64 形参 → 不动


def test_u1_invalid_url_rejected():
    update, item, err = set_item_figure_state(
        {"items": [{"stem": "x", "_seq": 1}]}, 1, figure_url="http://not-https"
    )
    assert err and item is None


def test_u1_reducer_preserves_base64_same_stem_drops_on_stem_change():
    """reducer：figure_base64 同 _seq 且题面未变才续（题面变=重生=旧图失效不嫁接）。"""
    old = [{"stem": "解方程 x=2", "_seq": 1, "figure_base64": "PNGA"}]
    # 新写整组但漏带 base64、题面未变 → reducer 续上
    same = merge_items(old, [{"stem": "解方程 x=2", "_seq": 1}])
    assert same[0]["figure_base64"] == "PNGA"
    # 题面变了（重生）→ 不续（旧图对新内容失效）
    changed = merge_items(old, [{"stem": "解方程 x=5", "_seq": 1}])
    assert "figure_base64" not in changed[0]


# ---------------------------------------------------------------------------
# U8 · 免转义哨兵框（richText 三段破损隔离 / 段内元字符不坏）
# ---------------------------------------------------------------------------
def test_u8_sentinel_extracts_segments_with_unescaped_metachars():
    """三段含未转义引号 + $LaTeX$ + 换行 + <table> → 哨兵框原文抠出、不因 JSON 转义坏。"""
    text = (
        '{"confidence":0.92,"gradeBook":"八年级下册","chapter":"第2章","has_figure":true,'
        '"richText":{"stem":"","answer":"","analysis":""},"solvedAnswer":"x=2",'
        '"dna":{"primaryKp":{"id":"","name":"一元二次方程"}}}\n'
        '⟦STEM⟧解方程 $x^2-4=0$，求其"实根"\n<table><tr><td>a</td></tr></table>⟦/STEM⟧\n'
        '⟦ANSWER⟧$x=\\pm 2$⟦/ANSWER⟧\n'
        '⟦ANALYSIS⟧因式分解：$x^2-4=(x-2)(x+2)$\n故 $x=2$ 或 $x=-2$⟦/ANALYSIS⟧'
    )
    r = extract_sentinel_richtext(text)
    assert isinstance(r, dict)
    # 结构化字段从瘦 JSON 解出
    assert r["confidence"] == 0.92
    assert r["gradeBook"] == "八年级下册"
    assert r["has_figure"] is True
    assert r["solvedAnswer"] == "x=2"
    assert r["dna"]["primaryKp"]["name"] == "一元二次方程"
    # 三段原文（含未转义引号/$LaTeX$/换行/<table>）完整抠出
    rich = r["richText"]
    assert '求其"实根"' in rich["stem"]
    assert "<table>" in rich["stem"]
    assert rich["answer"] == "$x=\\pm 2$"
    assert "因式分解" in rich["analysis"] and "\n" in rich["analysis"]


def test_u8_one_broken_segment_does_not_lose_others():
    """一段哨兵缺（ANALYSIS 没框）→ 另两段照常抠出、题不丢（宁存残段不丢题）。"""
    text = (
        '{"confidence":0.8,"gradeBook":"八下","has_figure":false,'
        '"richText":{"stem":"","answer":"","analysis":""}}\n'
        '⟦STEM⟧题面在⟦/STEM⟧\n⟦ANSWER⟧答案在⟦/ANSWER⟧'
        # 故意不给 ⟦ANALYSIS⟧ 框
    )
    r = extract_sentinel_richtext(text)
    assert isinstance(r, dict)
    assert r["richText"]["stem"] == "题面在"
    assert r["richText"]["answer"] == "答案在"
    # analysis 段缺 → 留瘦 JSON 的空占位（不报错、不丢另两段）
    assert r["richText"].get("analysis", "") == ""


def test_u8_thin_json_broken_still_delivers_richtext():
    """瘦 JSON 结构段坏（缺闭合）→ 仍交付三段富文本（U8 宁丢结构维不丢题面）。"""
    text = (
        '{"confidence":0.8,"gradeBook":"八下"'  # 故意不闭合
        '\n⟦STEM⟧题面原文⟦/STEM⟧\n⟦ANSWER⟧答案⟦/ANSWER⟧\n⟦ANALYSIS⟧解析⟦/ANALYSIS⟧'
    )
    r = extract_sentinel_richtext(text)
    assert isinstance(r, dict)
    assert r["richText"]["stem"] == "题面原文"
    assert r["richText"]["answer"] == "答案"
    assert r["richText"]["analysis"] == "解析"


def test_u8_no_sentinel_returns_none_for_fallback():
    """无哨兵（旧式整 JSON / mock）→ 返回 None，调用方回退既有 _parse_json（向后兼容）。"""
    assert extract_sentinel_richtext('{"richText":{"stem":"x"}}') is None
    assert extract_sentinel_richtext("") is None
