# -*- coding: utf-8 -*-
"""PRD-A-018 批1·簇1 入库幂等/数据完整（M1 / F16 / F12）单测。

验收口径（AC1/G1）：入库→改题→重生→再入库 库里单行+内容已更新；双击全部入库不重复。
- M1①：persist_to_bank 成功回写 _persist_id（对齐 persist_one_to_bank）+ 清 _content_dirty。
- M1③/F16：已入库题内容改了(标 _content_dirty) → 再入库不跳过、走 _persist_id 覆盖原行（不新落）。
- M1④/F12：build_create_bo 带稳定 sourceHash 幂等键（内容派生，库端去重兜底）。
- helper：mark_content_dirty_if_persisted 仅对已入库题打标。

persist_items 整体 monkeypatch（零网络）。
"""

import asyncio

import agents.variant as variant_mod
from agents.variant import (
    mark_content_dirty_if_persisted,
    persist_one_to_bank,
    persist_to_bank,
)
from agents.variant_support import build_create_bo, compute_source_hash


def _state(items):
    return {
        "analysis": {
            "grade": {"value": "七年级上学期", "confidence": 0.9},
            "kp": {"value": "一元一次方程", "confidence": 0.9, "anchored": {"code": "100"}},
            "qtype": {"value": "解答", "confidence": 0.9},
        },
        "mother_dna": {"stem": "母题", "answer": "x=1", "difficulty": 3, "mother_question_id": 555},
        "items": items,
    }


def _patch_persist(monkeypatch):
    """记录每次 persist_items 收到哪些 item（含是否带 _persist_id → 走 update 而非 create）。"""
    calls = {"create": [], "update": []}

    async def fake_persist(items, facts, token=None, publish=False):
        receipts = []
        for it in items:
            pid = it.get("_persist_id")
            if pid:
                calls["update"].append((it.get("stem"), pid))
                receipts.append({"ok": True, "id": pid, "role": "variant", "updated": True})
            else:
                new_id = 1000 + len(calls["create"])
                calls["create"].append(it.get("stem"))
                receipts.append({"ok": True, "id": new_id, "role": "variant"})
        return receipts

    monkeypatch.setattr(variant_mod, "persist_items", fake_persist)
    return calls


# --- helper 语义 ----------------------------------------------------------
def test_mark_content_dirty_only_for_persisted():
    persisted = {"persisted": True}
    mark_content_dirty_if_persisted(persisted)
    assert persisted.get("_content_dirty") is True

    fresh = {}
    mark_content_dirty_if_persisted(fresh)
    assert "_content_dirty" not in fresh  # 未入库题不打标（首次入库走 create）


# --- M1①：persist_to_bank 回写 _persist_id + 清 _content_dirty -------------
def test_persist_to_bank_writes_back_persist_id_and_clears_content_dirty(monkeypatch):
    calls = _patch_persist(monkeypatch)
    state = _state([{"stem": "v1", "answer": "a1", "qtype": "解答"}])
    update = asyncio.run(persist_to_bank(state, None))
    new = update["items"][0]
    assert new["persisted"] is True
    assert new["_persist_id"] == 1000  # 🔴 M1①：回写了 id（旧 bug 不回写）
    assert "_content_dirty" not in new
    assert calls["create"] == ["v1"] and calls["update"] == []


# --- M1③/F16：已入库 + 内容已改 → 覆盖入库（update by _persist_id，不新落） -----
def test_persist_to_bank_repersists_content_dirty_via_update(monkeypatch):
    calls = _patch_persist(monkeypatch)
    # 第二道是已入库题，内容被编辑/重生过 → 标 _content_dirty，期望走 update 覆盖原行
    state = _state(
        [
            {"stem": "v1", "answer": "a1", "qtype": "解答"},  # 未入库 → create
            {"stem": "v2-edited", "answer": "a2", "qtype": "解答",
             "persisted": True, "_persist_id": 42, "_content_dirty": True},
        ]
    )
    update = asyncio.run(persist_to_bank(state, None))
    # v2 不被「已入库就跳过」漏掉，且走 update（带 _persist_id）= 覆盖原行不新落
    assert ("v2-edited", 42) in calls["update"]
    assert "v1" in calls["create"]
    new2 = update["items"][1]
    assert new2["persisted"] is True and new2["_persist_id"] == 42
    assert "_content_dirty" not in new2  # 覆盖入库后清标记


# --- 双击/已入库且干净 → 跳过（不重复落库）-------------------------------
def test_persist_to_bank_skips_clean_persisted(monkeypatch):
    calls = _patch_persist(monkeypatch)
    state = _state(
        [{"stem": "v1", "answer": "a1", "qtype": "解答", "persisted": True, "_persist_id": 7}]
    )
    update = asyncio.run(persist_to_bank(state, None))
    # 全已入库且干净 → pending 空 → 根本不调 persist_items（无重复落库）
    assert calls["create"] == [] and calls["update"] == []
    assert "之前都已入库过" in update["messages"][0].content


# --- persist_one_to_bank：_content_dirty 题不被跳过 ------------------------
def test_persist_one_repersists_content_dirty(monkeypatch):
    calls = _patch_persist(monkeypatch)
    state = _state(
        [{"stem": "v1-edited", "answer": "a1", "qtype": "解答",
          "persisted": True, "_persist_id": 9, "_content_dirty": True}]
    )
    update, result, error = asyncio.run(persist_one_to_bank(state, 1, token="tok"))
    assert error is None and result["ok"] is True
    assert ("v1-edited", 9) in calls["update"]  # 走覆盖、不跳过
    assert update["items"][0]["_persist_id"] == 9
    assert "_content_dirty" not in update["items"][0]


def test_persist_one_skips_clean_persisted(monkeypatch):
    calls = _patch_persist(monkeypatch)
    state = _state(
        [{"stem": "v1", "answer": "a1", "qtype": "解答", "persisted": True, "_persist_id": 9}]
    )
    update, result, error = asyncio.run(persist_one_to_bank(state, 1, token="tok"))
    assert result == {"ok": True, "id": 9, "role": "variant", "skipped": True}
    assert calls["create"] == [] and calls["update"] == []  # 不二次落行
    assert update == {}


# --- M1④/F12：sourceHash 幂等键 ------------------------------------------
def test_source_hash_stable_and_content_sensitive():
    facts = {"mother_question_id": 555}
    h1 = compute_source_hash({"stem": "a", "answer": "b"}, facts)
    h2 = compute_source_hash({"stem": "a", "answer": "b"}, facts)
    h3 = compute_source_hash({"stem": "a2", "answer": "b"}, facts)
    assert h1 == h2 and h1 != h3 and len(h1) == 64


def test_build_create_bo_carries_source_hash_no_internal_leak():
    bo = build_create_bo(
        {"stem": "s", "answer": "a", "qtype": "解答", "difficulty": 2,
         "_content_dirty": True, "_persist_id": 1},
        {"qtype": "解答", "subject_id": "3071"},
    )
    assert "sourceHash" in bo and len(bo["sourceHash"]) == 64
    # 内部键绝不入库（白名单挡）
    assert "_content_dirty" not in bo and "_persist_id" not in bo
