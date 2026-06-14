# -*- coding: utf-8 -*-
"""PRD-C-017 B3.5 · classify 末尾发母题卡「先出」专帧（header.mother_card）解锁 AC4/G12。

覆盖（零网络，纯函数 + monkeypatch LLM/writer）：
- payload 字段齐全（§10 契约逐项）：stem / solution_skeleton / solved_answer / dna 10 维 / anchor。
- 帧机制 = custom 帧 header.mother_card（FE pickMotherCard 路①），items 留空。
- 时序「先出」：classify 跑完发了 mother_card 帧，且帧 items 为空（早于变式 item 帧）。
- 无 mother_dna（库内母题直进 generate）→ 不发帧（FE 走 items[0] 兜底）。
"""

import asyncio
import json

import agents.variant as variant_mod
from agents import mother_opus
from agents.variant import _build_mother_card, _emit_mother_card, classify


# ===========================================================================
# 公共桩（镜像 B2 test）
# ===========================================================================
class _FakeClient:
    def __init__(self, token=None):
        pass

    async def aclose(self):
        pass


_POOL = [
    ("3071001001001", "一元一次方程"),
    ("3071001001002", "合并同类项"),
]

_OPUS_GOOD = {
    "has_figure": False,
    "richText": {
        "stem": "解方程 $2x+3=7$",
        "answer": "$x=2$",
        "analysis": "移项得 $2x=4$，故 $x=2$",
    },
    "solvedAnswer": "$x=2$",
    "dna": {
        "primaryKp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondaryKps": [{"id": "3071001001002", "name": "合并同类项"}],
        "qtype": "解答",
        "assessmentType": "直接计算",
        "solutionSkeleton": ["移项", "【解一元一次方程】"],
        "hardPointCount": 1,
        "breakthroughPoints": ["移项变号易错"],
        "scenario": "纯代数",
        "difficulty": 2,
        "tags": ["解方程", "一元一次"],
        "modelCandidates": ["方程通解套路"],
    },
}

_BASE = {
    "image_url": "https://x/q.png",
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {"value": "一元一次方程", "confidence": 0.4},
        "qtype": {"value": "解答", "confidence": 0.5},
    },
    "mother_dna": {"stem": "2x+3=7", "answer": "x=2", "solution_skeleton": "抄图骨架(旧)"},
}


def _patch_classify(monkeypatch, *, pool, opus_text, chapter_name=None):
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_error", lambda *a, **k: None)

    async def fake_leaf_pool(grade_code, client, **kw):
        return pool

    async def fake_solve(**kw):
        return opus_text

    async def fake_anchor_models(dna, **kw):
        return {"models": [dict(variant_mod.model_anchor.M00)], "model_overflow": [],
                "model_warn": False, "model_flag": "m00_fallback"}

    async def fake_chapter_name(chapter_id, client):
        return chapter_name

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.mother_opus, "solve_and_label", fake_solve)
    monkeypatch.setattr(variant_mod.model_anchor, "anchor_models", fake_anchor_models)
    monkeypatch.setattr(variant_mod, "chapter_name_for_id", fake_chapter_name)


def _capture_frames(monkeypatch):
    """打桩 get_stream_writer → 捕获 custom 帧列表（保留发射顺序，验时序）。"""
    frames: list = []

    def fake_writer_factory():
        def _w(msg):
            frames.append(msg)
        return _w

    monkeypatch.setattr(variant_mod, "get_stream_writer", fake_writer_factory)
    return frames


def _mother_card_frames(frames):
    """从捕获帧抠出带 header.mother_card 的 artifact 帧。"""
    out = []
    for f in frames:
        content = getattr(f, "content", None)
        if not isinstance(content, list) or not content:
            continue
        art = content[0].get("artifact") if isinstance(content[0], dict) else None
        if isinstance(art, dict) and (art.get("header") or {}).get("mother_card"):
            out.append(art)
    return out


# ===========================================================================
# _build_mother_card 纯函数 · §10 payload 字段齐全
# ===========================================================================
def _full_state():
    """模拟 classify 跑完后的 state（mother_dna 含 opus dna v1 + 富文本 + 解答）。"""
    dna = mother_opus.opus_to_dna(_OPUS_GOOD)
    dna = mother_opus.anchor_to_chapter(dna, chapter_id="3071001", leaf_pool=_POOL)
    dna["models"] = [{"id": "M01", "name": "方程通解"}]
    mdna = {
        "stem": "解方程 $2x+3=7$",
        "answer": "$x=2$",
        "analysis": "移项得 $2x=4$",
        "solved_answer": "$x=2$",
        "solution_skeleton": "移项\n【解一元一次方程】",
        "difficulty": 2,
        "dna": dna,
    }
    return {**_BASE, "mother_dna": mdna, "confirmed_chapter_id": "3071001"}


def test_build_mother_card_all_fields_present():
    card = _build_mother_card(_full_state())
    assert card is not None
    # —— 顶层 §10 ——
    assert card["stem"] == "解方程 $2x+3=7$"            # 🔴 stem 必带（入库靠它）
    assert "解一元一次方程" in card["solution_skeleton"]  # 解法骨架（【】标最难步）
    assert card["solved_answer"] == "$x=2$"             # opus 解答
    # —— dna 10 维 ——
    d = card["dna"]
    assert d["main_kp"] == "一元一次方程"          # ① 主考点名
    assert d["main_kp_id"] == "3071001001001"      #    主考点 id
    assert d["secondary_kps"] == [{"id": "3071001001002", "name": "合并同类项"}]  # ② 副考点 {id,name}
    assert d["qtype"] == "解答"                     # ③ 题型
    assert d["exam_type"] == "直接计算"             # ④ 考察类型
    assert d["difficulty"] == 2                     # ⑤ 难度
    assert d["scenario"] == "纯代数"               # ⑥ 场景
    assert d["hard_point_count"] == 1              # ⑦ 难点数（= breakthrough 长度）
    assert d["breakthrough_points"] == ["移项变号易错"]  # ⑧ 突破口
    assert "解一元一次方程" in d["skeleton"]        # ⑨ 骨架
    assert d["models"] == [{"id": "M01", "name": "方程通解"}]  # ⑩ 模型
    assert d["tags"] == ["解方程", "一元一次"]      # 标签
    # —— anchor ——
    a = card["anchor"]
    assert a["chapter_id"] == "3071001"            # 确认章 id
    assert a["grade_book_id"] == "3071"            # 年级册
    assert "need_anchor_review" in a


def test_build_mother_card_hard_point_count_equals_len():
    """难点克制：hard_point_count == breakthrough_points 数组长度（不信自报）。"""
    card = _build_mother_card(_full_state())
    assert card["dna"]["hard_point_count"] == len(card["dna"]["breakthrough_points"])


def test_build_mother_card_none_when_no_mother_dna():
    """无 mother_dna（库内母题直进 generate）→ None（不发帧，FE 走 items[0] 兜底）。"""
    assert _build_mother_card({**_BASE, "mother_dna": {}}) is None
    assert _build_mother_card({"analysis": {}}) is None


def test_build_mother_card_need_anchor_review_propagates():
    """闸B 留空标记 need_anchor_review → 母题卡顶层 + anchor 双带（FE 标「锚定待人审」）。"""
    st = _full_state()
    st["mother_dna"]["need_anchor_review"] = True
    card = _build_mother_card(st)
    assert card["need_anchor_review"] is True
    assert card["anchor"]["need_anchor_review"] is True


# ===========================================================================
# _emit_mother_card · 帧机制 = header.mother_card + items 空
# ===========================================================================
def test_emit_mother_card_frame_shape(monkeypatch):
    frames = _capture_frames(monkeypatch)
    _emit_mother_card(_full_state())
    cards = _mother_card_frames(frames)
    assert len(cards) == 1
    art = cards[0]
    assert art["items"] == []                       # 🔴 items 空 → 早于变式 item 帧
    assert art["header"]["mother_card"]["stem"]     # 帧确实携母题卡


def test_emit_mother_card_noop_when_no_card(monkeypatch):
    """无 mother_dna → 不发帧（_build_mother_card None → no-op）。"""
    frames = _capture_frames(monkeypatch)
    _emit_mother_card({**_BASE, "mother_dna": {}})
    assert _mother_card_frames(frames) == []


# ===========================================================================
# classify 端到端 · 时序「先出」：发了 mother_card 帧，且在任何变式 item 帧之前
# ===========================================================================
def test_classify_emits_mother_card_before_items(monkeypatch):
    _patch_classify(monkeypatch, pool=_POOL,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False))
    frames = _capture_frames(monkeypatch)
    cfg = {"configurable": {"confirmed_chapter_id": "3071001"}}
    out = asyncio.run(classify(dict(_BASE), cfg))
    assert out["mother_confirmed"] is True

    # 发了 mother_card 帧
    cards = _mother_card_frames(frames)
    assert len(cards) >= 1, "classify 末尾必发母题卡专帧"
    mc = cards[0]["header"]["mother_card"]
    # payload 关键字段齐（§10）
    assert mc["stem"]
    assert mc["solution_skeleton"]
    assert mc["solved_answer"]
    assert mc["dna"]["main_kp_id"] == "3071001001001"
    assert mc["anchor"]["chapter_id"] == "3071001"

    # 🔴 时序：classify 阶段任何含非空 items 的 artifact 帧都不得早于 mother_card 帧
    #   （classify 自身不出 items；generate 是后续节点。本断言守住 classify 内不抢跑）。
    first_card_idx = None
    for i, f in enumerate(frames):
        content = getattr(f, "content", None)
        if not isinstance(content, list) or not content or not isinstance(content[0], dict):
            continue
        art = content[0].get("artifact")
        if not isinstance(art, dict):
            continue
        has_card = bool((art.get("header") or {}).get("mother_card"))
        has_items = bool(art.get("items"))
        if has_card and first_card_idx is None:
            first_card_idx = i
        if has_items:
            assert first_card_idx is not None and i > first_card_idx, \
                "变式 item 帧不得早于母题卡帧（母题卡必先出）"
