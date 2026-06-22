# -*- coding: utf-8 -*-
"""单题入库 persist_one_to_bank 单测（PRD-C-014 B2·T5·B3 前置）。

复用 persist_items 的单 item 路径；item 级 persisted 防重（已收录的重复调直接回已有 id，
不二次落行）。母题血缘回写。index 越界 → error。单题落库失败 → ok=False（不抛）。
persist_items 整体 monkeypatch（零网络）。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import persist_one_to_bank

_STATE = {
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.9, "anchored": {"code": "100"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干",
        "answer": "x=1",
        "difficulty": 3,
        "mother_question_id": 555,  # 母题已在库 → 不重复建母题
    },
    "items": [
        {"stem": "v1", "answer": "a1", "qtype": "解答"},
        {"stem": "v2", "answer": "a2", "qtype": "解答"},
    ],
}


def _patch_persist(monkeypatch, *, variant_id=999, mother=None, fail=False):
    calls = {"n": 0, "items": []}

    async def fake_persist(items, facts, token=None, publish=False):
        calls["n"] += 1
        calls["items"].extend(items)
        receipts = []
        if mother is not None:
            receipts.append(mother)
        if fail:
            receipts.append({"ok": False, "error": "boom", "role": "variant"})
        else:
            receipts.append({"ok": True, "id": variant_id, "role": "variant"})
        return receipts

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    return calls


def test_persist_one_success_marks_persisted_and_returns_id(monkeypatch):
    calls = _patch_persist(monkeypatch, variant_id=999)
    update, result, error = asyncio.run(
        persist_one_to_bank(dict(_STATE), 1, token="tok")
    )
    assert error is None
    assert result == {"ok": True, "id": 999, "role": "variant"}
    assert calls["n"] == 1 and len(calls["items"]) == 1  # 只落这一道
    assert update["items"][0]["persisted"] is True
    assert update["items"][0]["_persist_id"] == 999
    assert update["items"][1].get("persisted") is None  # 另一道不动


def test_persist_one_dedup_skips_already_persisted(monkeypatch):
    calls = _patch_persist(monkeypatch)
    state = dict(_STATE)
    state["items"] = [dict(state["items"][0], persisted=True, _persist_id=42), state["items"][1]]
    update, result, error = asyncio.run(persist_one_to_bank(state, 1, token="tok"))
    assert error is None
    assert result == {"ok": True, "id": 42, "role": "variant", "skipped": True}
    assert calls["n"] == 0  # 防重：根本没调 persist_items（不二次落行）
    assert update == {}  # 无 state 变更


def test_persist_one_index_out_of_range(monkeypatch):
    _patch_persist(monkeypatch)
    _, _, error = asyncio.run(persist_one_to_bank(dict(_STATE), 9, token="tok"))
    assert error is not None and "越界" in error


def test_persist_one_failure_returns_ok_false_no_persist_mark(monkeypatch):
    _patch_persist(monkeypatch, fail=True)
    update, result, error = asyncio.run(persist_one_to_bank(dict(_STATE), 1, token="tok"))
    assert error is None  # 单题落库失败不当 error/500
    assert result["ok"] is False and result.get("error") == "boom"
    assert update == {}  # 失败不打 persisted 标记


def test_persist_one_writes_back_mother_lineage(monkeypatch):
    # 图母题不在库：persist_items 先落母题回填 id → persist_one 把它写回 state.mother_dna
    _patch_persist(
        monkeypatch, variant_id=999, mother={"ok": True, "id": 777, "role": "mother"}
    )
    state = dict(_STATE)
    state["mother_dna"] = {"stem": "母题题干", "answer": "x=1", "difficulty": 3}  # 无 mother_question_id
    update, result, error = asyncio.run(persist_one_to_bank(state, 2, token="tok"))
    assert result["ok"] is True
    assert update["mother_dna"]["mother_question_id"] == 777
    assert update["items"][1]["persisted"] is True


def test_persist_one_does_not_mutate_input_items(monkeypatch):
    _patch_persist(monkeypatch, variant_id=999)
    state = dict(_STATE)
    snapshot = [dict(it) for it in state["items"]]
    asyncio.run(persist_one_to_bank(state, 1, token="tok"))
    assert state["items"] == snapshot  # update 是新 list，不原地改入参
