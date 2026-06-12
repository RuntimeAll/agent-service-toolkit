# -*- coding: utf-8 -*-
"""批2（2026-06-13 整改）· 「没定死」硬闸：缺锚必停确认，从机制上绝迹缺锚出题。

「定死」三件同时满足：年级 4 位教材册 code + main_kp 锚真叶子（非复习册） + 置信达 CONF_GATE。
覆盖：
- _pin_status 判定（齐全=pinned / 缺年级 / 缺 kp / 复习册前缀不算定死 / 低置信）；
- gate_after_classify：定死直通 generate；没定死一律 clarify；
- generate 入口防御：facts 缺年级或主考点 → 拒造回确认态；
- 编辑轮（exec_* 经 gene_gate，不过 generate 入口）不被定死闸拦。
"""

import asyncio

from agents.variant import (
    CONF_GATE,
    _pin_status,
    clarify,
    gate_after_classify,
    generate,
)


def _pinned_state():
    return {
        "mother_confirmed": True,
        "analysis": {
            "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
            "kp": {
                "value": "一元一次方程", "confidence": 0.9,
                "anchored": {"id": "3071001001001", "code": "3071001001001", "name": "一元一次方程"},
            },
            "qtype": {"value": "解答", "confidence": 0.8},
        },
        "mother_dna": {"stem": "2x+3=7", "dna": {"main_kp": {"id": "3071001001001", "name": "一元一次方程"}}},
    }


# ---------------------------------------------------------------------------
# _pin_status 判定
# ---------------------------------------------------------------------------
def test_pin_status_all_satisfied():
    pin = _pin_status(_pinned_state())
    assert pin["pinned"] is True
    assert pin["reasons"] == []
    assert pin["grade_code"] == "3071"
    assert pin["kp_code"] == "3071001001001"
    assert pin["kp_book"] == "七年级上册"


def test_pin_status_missing_grade_code():
    st = _pinned_state()
    st["analysis"]["grade"].pop("code")  # 无年级册 code → 没定死
    pin = _pin_status(st)
    assert pin["pinned"] is False
    assert "grade" in pin["reasons"]


def test_pin_status_missing_kp_anchor():
    st = _pinned_state()
    st["analysis"]["kp"].pop("anchored")  # 没锚到叶子 → 没定死
    pin = _pin_status(st)
    assert pin["pinned"] is False
    assert "kp" in pin["reasons"]


def test_pin_status_review_book_not_pinned():
    # 锚到复习册（3120 新题抢先）→ 不算定死的教材锚
    st = _pinned_state()
    st["analysis"]["grade"]["code"] = "3120"
    st["analysis"]["kp"]["anchored"] = {"id": "3120004", "code": "3120004", "name": "未解析"}
    pin = _pin_status(st)
    assert pin["pinned"] is False
    assert "grade" in pin["reasons"] and "kp" in pin["reasons"]


def test_pin_status_low_confidence():
    st = _pinned_state()
    st["analysis"]["kp"]["confidence"] = CONF_GATE - 0.1
    pin = _pin_status(st)
    assert pin["pinned"] is False
    assert "confidence" in pin["reasons"]


# ---------------------------------------------------------------------------
# gate_after_classify
# ---------------------------------------------------------------------------
def test_gate_pinned_goes_generate():
    assert gate_after_classify(_pinned_state()) == "generate"


def test_gate_unpinned_goes_clarify():
    st = _pinned_state()
    st["analysis"]["kp"].pop("anchored")
    assert gate_after_classify(st) == "clarify"


def test_gate_confirmed_but_no_grade_code_still_clarify():
    # mother_confirmed=True 但年级 code 缺（边角路径）→ 定死闸仍拦
    st = _pinned_state()
    st["analysis"]["grade"].pop("code")
    assert gate_after_classify(st) == "clarify"


# ---------------------------------------------------------------------------
# clarify 确认态如实回报年级 + 主考点
# ---------------------------------------------------------------------------
def test_clarify_reports_grade_and_kp_status():
    st = _pinned_state()
    st["analysis"]["kp"].pop("anchored")  # 主考点未锚定
    st["analysis"]["kp"]["confidence"] = 0.3
    out = asyncio.run(clarify(st, {}))
    body = out["messages"][0].content
    assert "定死" in body
    assert "未锚定" in body  # 主考点未锚定如实回报
    assert "七年级上学期" in body  # 年级已识别照样回报


# ---------------------------------------------------------------------------
# generate 入口防御断言
# ---------------------------------------------------------------------------
def test_generate_refuses_when_grade_unknown(monkeypatch):
    import agents.variant as v
    monkeypatch.setattr(v, "_emit_stage", lambda *a, **k: None)
    st = {
        "mother_confirmed": True,
        "analysis": {
            "grade": {"value": None, "confidence": 0.9},  # 年级未知 → facts.grade=未知年级
            "kp": {"value": "x", "confidence": 0.9, "anchored": {"code": "3071001"}},
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {"stem": "s", "dna": {"main_kp": {"id": "3071001", "name": "x"}}},
    }
    out = asyncio.run(generate(st, {}))
    body = out["messages"][0].content
    assert "定死" in body and "年级" in body
    assert not out.get("items")  # 没出题


def test_generate_refuses_when_kp_unanchored(monkeypatch):
    import agents.variant as v
    monkeypatch.setattr(v, "_emit_stage", lambda *a, **k: None)
    st = {
        "mother_confirmed": True,
        "analysis": {
            "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
            "kp": {"value": None, "confidence": 0.9},  # 无 anchored → dim1_kp_id None
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {"stem": "s", "dna": {}},
    }
    out = asyncio.run(generate(st, {}))
    body = out["messages"][0].content
    assert "定死" in body and "主考点" in body
    assert not out.get("items")
