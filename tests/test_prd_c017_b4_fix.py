# -*- coding: utf-8 -*-
"""PRD-C-017 B4-fix · classify 确认章驱动 grade_code + leaf_pool（修 AC2 零作用 critical bug）。

Root cause（B4 全图 e2e 抓出）：classify 的 grade_code 旧实现只读 analyze 图读的 grade，不吃
老师确认章前缀。纯文本母题题面常无年级标记 → analyze 瞎猜 grade（且人教/浙教版错位）→
leaf_pool 用错册圈池 → opus 找不到叶子 → 闸B 锚空 + grade_code 错 → pin 三锚全缺 →
classify 卡 clarify、不出变式 → 老师确认章零作用（违背 AC2）。

本测专测「grade_code/leaf_pool 圈池来源 = 确认章册前缀，而非 analyze 瞎猜」——
故 leaf_pool_for_grade 的桩**捕获并按 grade_code 圈池**（不像 B2 共享桩忽略 grade_code），
据此断言：analyze 猜错 grade 时，锚定仍走确认章册（堵此 bug 回归）。

biz_subject id 编码：根=4 位（年级册 level1，如 3082=浙教八下），章=7 位（level2，如
3082002），叶子=完整 id（如 3082002001004）。确认章前 4 位 = 年级册 code。
"""

import asyncio
import json

import pytest

import agents.variant as variant_mod
from agents import mother_opus
from agents.variant import CONF_GATE, classify


class _FakeClient:
    def __init__(self, token=None):
        pass

    async def aclose(self):
        pass


# 浙教八下（3082）一元二次方程叶子 + 七上（3071）一元一次方程叶子（模拟错册）。
# 关键：两册叶子并存，圈对册才能锚到对的叶子。
_LEAVES_BY_BOOK = {
    "3082": [
        ("3082002001004", "利用一元二次方程的根求字母的值或代数式的值"),
        ("3082002001001", "一元二次方程的定义"),
    ],
    "3071": [
        ("3071001001001", "一元一次方程"),
    ],
}
# 全教材册叶子（grade_code 缺时的兜底池）
_ALL_LEAVES = [leaf for leaves in _LEAVES_BY_BOOK.values() for leaf in leaves]

# opus 解出的母题主考点 = 浙教八下一元二次方程真叶子（3082002001004）。
_OPUS_GOOD = {
    "has_figure": False,
    "richText": {
        "stem": "已知关于 x 的方程 $x^2-3x+m=0$ 的一个根为 1，求 m。",
        "answer": "$m=2$",
        "analysis": "代入 $x=1$：$1-3+m=0$，得 $m=2$。",
    },
    "solvedAnswer": "m=2",
    "dna": {
        "primaryKp": {"id": "3082002001004", "name": "利用一元二次方程的根求字母的值或代数式的值"},
        "secondaryKps": [],
        "qtype": "选择", "assessmentType": "直接计算",
        "solutionSkeleton": ["代入根", "【一元二次方程求参】"],
        "hardPointCount": 0, "breakthroughPoints": [],
        "scenario": "纯代数", "difficulty": 2,
        "tags": ["一元二次方程"], "modelCandidates": [],
    },
}

# 🔴 analyze 瞎猜的 grade —— 故意错（判成人教九上 3091 / 高置信），与确认章 3082 不符。
#   旧实现会信这个 grade（_resolve_grade_code → 圈错册 3091）→ 圈不到 3082 叶子 → 锚空 → clarify。
_ANALYZE_WRONG_GRADE = {
    "image_url": "https://x/q.png",
    "analysis": {
        "grade": {"value": "九年级上学期", "code": "3091", "confidence": 0.86},  # ← 瞎猜错
        "kp": {"value": "一元二次方程", "confidence": 0.4},
        "qtype": {"value": "选择", "confidence": 0.5},
    },
    "mother_dna": {"stem": "x^2-3x+m=0", "answer": "m=2", "solution_skeleton": "抄图骨架(旧)"},
}


def _patch_classify(monkeypatch, *, opus_text, grade_code_seen, chapter_id_seen,
                    chapter_name=None):
    """打桩 classify 的 IO/LLM。
    🔴 与 B2 共享桩不同：leaf_pool_for_grade 桩**按传入 grade_code 真实圈池**（捕获到
       grade_code_seen），证明圈池来源 = 确认章册前缀而非 analyze 瞎猜。
    """
    monkeypatch.setattr(variant_mod, "RuoyiClient", _FakeClient)
    monkeypatch.setattr(variant_mod, "_emit_stage", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_error", lambda *a, **k: None)
    monkeypatch.setattr(variant_mod, "_emit_mother_card", lambda *a, **k: None)

    async def fake_leaf_pool(grade_code, client, **kw):
        grade_code_seen["gc"] = grade_code  # 捕获 classify 实际用来圈池的 grade_code
        gc = str(grade_code or "").strip()
        if gc and gc in _LEAVES_BY_BOOK:
            return list(_LEAVES_BY_BOOK[gc])
        if gc:  # 圈到一个不存在的册 → 空（模拟错册圈不到目标叶子）
            return []
        return list(_ALL_LEAVES)  # grade_code 缺 → 全册兜底

    async def fake_solve(**kw):
        if isinstance(opus_text, Exception):
            raise opus_text
        return opus_text

    async def fake_anchor_models(dna, **kw):
        return {"models": [dict(variant_mod.model_anchor.M00)], "model_overflow": [],
                "model_warn": False, "model_flag": "m00_fallback"}

    async def fake_chapter_name(chapter_id, client):
        return chapter_name

    real_anchor = mother_opus.anchor_to_chapter

    def spy_anchor(dna, **kw):
        chapter_id_seen["cid"] = kw.get("chapter_id")
        return real_anchor(dna, **kw)

    monkeypatch.setattr(variant_mod, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(variant_mod.mother_opus, "solve_and_label", fake_solve)
    monkeypatch.setattr(variant_mod.mother_opus, "anchor_to_chapter", spy_anchor)
    monkeypatch.setattr(variant_mod.model_anchor, "anchor_models", fake_anchor_models)
    monkeypatch.setattr(variant_mod, "chapter_name_for_id", fake_chapter_name)


# ===========================================================================
# 核心回归：确认章存在 → grade_code/leaf_pool 圈池来自确认章册（不是 analyze 瞎猜）
# ===========================================================================
def test_b4fix_confirmed_chapter_drives_grade_code_for_leaf_pool(monkeypatch):
    """确认章 3082002（浙教八下一元二次方程章）→ classify 圈池 grade_code = 3082（前 4 位），
    **不是** analyze 瞎猜的 3091。圈对册 → opus 主 kp 3082002001004 命中 → 锚上 → 出题。"""
    gc_seen, cid_seen = {}, {}
    _patch_classify(monkeypatch,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    grade_code_seen=gc_seen, chapter_id_seen=cid_seen,
                    chapter_name="第2章 一元二次方程")
    cfg = {"configurable": {"confirmed_chapter_id": "3082002"}}
    out = asyncio.run(classify(dict(_ANALYZE_WRONG_GRADE), cfg))

    # ① 圈 leaf_pool 用的 grade_code = 确认章前 4 位 3082（堵 bug：不再是 analyze 瞎猜的 3091）
    assert gc_seen["gc"] == "3082"
    # ② 闸B 收到的 chapter_id = 确认章 id 3082002（收窄到 level2 章，不是只到册）
    assert cid_seen["cid"] == "3082002"
    # ③ 圈对册 → opus 主 kp 3082002001004 以 3082002 为前缀且在池内 → 锚上 → 放行出题
    assert out["mother_confirmed"] is True
    assert out["analysis"]["kp"]["anchored"]["code"] == "3082002001004"
    # ④ grade.code 被同步置成确认章册（pin 闸 _pin_status 读这里）+ 抬置信过闸
    assert out["analysis"]["grade"]["code"] == "3082"
    assert out["analysis"]["grade"]["confidence"] >= CONF_GATE
    # ⑤ 留痕确认章 id（接下游/防重 route）
    assert out["confirmed_chapter_id"] == "3082002"
    assert out["awaiting_mother_confirm"] is False


def test_b4fix_analyze_wrong_grade_alone_would_miss_leaf(monkeypatch):
    """对照组：没有确认章（旧路径）+ analyze 瞎猜成 3091 → 圈错册 3091（空池）→ 锚不到 →
    clarify、不出题。证明 bug 的链式失败真存在（确认章是唯一解药）。"""
    gc_seen, cid_seen = {}, {}
    _patch_classify(monkeypatch,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    grade_code_seen=gc_seen, chapter_id_seen=cid_seen)
    # 无 confirmed_chapter_id → 走 _resolve_grade_code（analyze grade=九上 → 3091）
    out = asyncio.run(classify(dict(_ANALYZE_WRONG_GRADE), {}))
    assert gc_seen["gc"] == "3091"          # 信了 analyze 瞎猜
    # 3091 册无该叶子（fake 池只在 3071/3082 有）→ 空池 → early-return clarify 态
    assert out["mother_confirmed"] is False


def test_b4fix_no_confirmed_id_falls_back_to_analyze_grade(monkeypatch):
    """防御性回退（不回归 B1）：无确认章 + analyze grade 对（七上 3071，叶子在池）→
    回退原 analyze 行为，圈对册照常锚上。"""
    gc_seen, cid_seen = {}, {}
    # analyze 这次猜对了七上一元一次方程
    state = {
        "image_url": "https://x/q.png",
        "analysis": {
            "grade": {"value": "七年级上学期", "code": "3071", "confidence": 0.9},
            "kp": {"value": "一元一次方程", "confidence": 0.4},
            "qtype": {"value": "选择", "confidence": 0.5},
        },
        "mother_dna": {"stem": "2x+3=7", "answer": "x=2", "solution_skeleton": "x"},
    }
    opus = json.loads(json.dumps(_OPUS_GOOD))
    opus["dna"]["primaryKp"] = {"id": "3071001001001", "name": "一元一次方程"}
    _patch_classify(monkeypatch, opus_text=json.dumps(opus, ensure_ascii=False),
                    grade_code_seen=gc_seen, chapter_id_seen=cid_seen)
    out = asyncio.run(classify(state, {}))
    assert gc_seen["gc"] == "3071"          # 回退 analyze grade（B1 行为）
    assert cid_seen["cid"] == "3071"        # 无确认章 → 闸B 用年级册前缀兜底
    assert out["mother_confirmed"] is True


def test_b4fix_grade_code_prefix_is_four_digits(monkeypatch):
    """编码层级断言：确认章前缀取**前 4 位**作 grade_code（biz_subject 根=4 位年级册）。
    用 13 位叶子作 confirmed_chapter_id（极端）也只取前 4 位圈册。"""
    gc_seen, cid_seen = {}, {}
    _patch_classify(monkeypatch,
                    opus_text=json.dumps(_OPUS_GOOD, ensure_ascii=False),
                    grade_code_seen=gc_seen, chapter_id_seen=cid_seen,
                    chapter_name="第2章 一元二次方程")
    cfg = {"configurable": {"confirmed_chapter_id": "3082002001004"}}
    out = asyncio.run(classify(dict(_ANALYZE_WRONG_GRADE), cfg))
    assert gc_seen["gc"] == "3082"          # 前 4 位 = 年级册
    assert cid_seen["cid"] == "3082002001004"  # 闸B 用完整确认章 id 前缀
