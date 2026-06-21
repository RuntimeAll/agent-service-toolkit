# -*- coding: utf-8 -*-
"""PRD-A-021 R3a · B3 变式多样性重组 + B7 改主考点可回退 的单元测试。

B3 红线（守恒）：
- per-variant 注入不同副考点/标签子集（多样性轴），但**骨架四维仍锁**（题型/主考点/模型/难点）；
- scene 维**保持组级**（不给每道各带场景，与 B5-fix5 契约一致）；
- 稀疏数据优雅降级（锚不足/无锚 不报错、不空转）；
- 只 GENERATE 注入：REGEN/单题(n<2) 不注。

B7 回归：改主考点写 main_kp_prev（可回退）+ 标全组 dirty（传播）+ 不自动重出。

零 LLM / 零网络：纯函数直测 + edit_dna_state 纯函数路径。
"""

import agents.variant as variant_mod
from agents.variant import (
    _diversity_anchors,
    _diversity_clause,
    _maybe_diversity_block,
    edit_dna_state,
)


def _dna(secondary=None, tags=None, scene=None):
    return {
        "main_kp": {"id": "30710101", "name": "一元一次方程"},
        "secondary_kps": secondary or [],
        "tags": tags or [],
        "scene": scene or "",
        "exam_type": "直接计算",
        "skeleton": ["移项【最难步】", "合并同类项"],
        "models": [{"id": "M25", "name": "去分母模型"}],
    }


# ── _diversity_anchors：副考点 + 标签去重去空保序 ─────────────────────────
def test_anchors_merge_secondary_and_tags_dedup():
    dna = _dna(
        secondary=[{"id": "1", "name": "整式运算"}, {"id": "2", "name": "因式分解"}],
        tags=["配方法", "十字相乘", "配方法", "  ", "整式运算"],
    )
    anchors = _diversity_anchors(dna)
    # 副考点先、标签后；"配方法"重复只留一次；空白丢弃；"整式运算"既是副考点又是标签 → 只一次
    assert anchors == ["整式运算", "因式分解", "配方法", "十字相乘"]


def test_anchors_empty_when_no_dna():
    assert _diversity_anchors(None) == []
    assert _diversity_anchors({}) == []
    assert _diversity_anchors({"secondary_kps": [], "tags": []}) == []


# ── per-variant 子集分派：n 道拿到不同切入角度 ──────────────────────────
def test_per_variant_assignment_round_robin():
    dna = _dna(
        secondary=[{"id": "1", "name": "A考点"}, {"id": "2", "name": "B考点"}],
        tags=["C标签"],
    )
    clause = _diversity_clause(dna, 3)
    # 三道分别点名 A考点 / B考点 / C标签（round-robin 覆盖锚池）
    assert "第 1 道" in clause and "A考点" in clause
    assert "第 2 道" in clause and "B考点" in clause
    assert "第 3 道" in clause and "C标签" in clause


def test_per_variant_assignment_wraps_when_more_variants_than_anchors():
    dna = _dna(secondary=[{"id": "1", "name": "A考点"}, {"id": "2", "name": "B考点"}])
    clause = _diversity_clause(dna, 4)
    # 锚池 2 个、4 道 → round-robin 回绕：第4道复用第1锚，但仍要求情境/设问差异
    assert "第 4 道" in clause
    assert "情境" in clause or "设问" in clause  # ② 即便同锚也要差异


# ── 🔴 守恒：骨架四维锁死申明必在（多样性只在副考点/标签/场景细节） ──────
def test_locks_four_skeleton_dims_always():
    for dna in (_dna(secondary=[{"id": "1", "name": "A"}, {"id": "2", "name": "B"}]), _dna()):
        clause = _diversity_clause(dna, 3)
        assert "题型" in clause and "主考点" in clause
        assert "模型" in clause and "难点" in clause
        # 申明「不许换赛道/超纲/简化最难步」
        assert "超纲" in clause or "换赛道" in clause


# ── 🔴 scene 保持组级：有场景 → 申明整组共享、不要每道各换 ──────────────
def test_scene_stays_group_level():
    dna = _dna(secondary=[{"id": "1", "name": "A"}, {"id": "2", "name": "B"}], scene="购物")
    clause = _diversity_clause(dna, 3)
    assert "整组共享" in clause
    assert "购物" in clause
    # 不给每道各带不同大场景
    assert "不要" in clause


def test_no_scene_line_when_scene_absent():
    dna = _dna(secondary=[{"id": "1", "name": "A"}, {"id": "2", "name": "B"}])
    clause = _diversity_clause(dna, 3)
    assert "整组共享" not in clause  # 无场景 → 不注③


# ── 🔴 稀疏降级：锚不足/无锚 不报错、不空转、不做硬分派 ─────────────────
def test_sparse_one_anchor_degrades_softly():
    dna = _dna(secondary=[{"id": "1", "name": "唯一副考点"}])
    clause = _diversity_clause(dna, 3)
    # 单锚 → 不做 per-variant「第 N 道：突出从…」硬分派，走软指令
    assert "第 1 道：突出从" not in clause
    assert "唯一副考点" in clause  # 软指令里仍点出可用锚


def test_sparse_no_anchor_degrades_softly():
    dna = _dna()  # 无副考点无标签
    clause = _diversity_clause(dna, 3)
    assert "第 1 道：突出从" not in clause
    # 仍鼓励情境/设问/数据组织差异，且不崩
    assert "情境" in clause or "设问" in clause
    assert "克隆题" in clause or "只换数字" in clause


# ── 只 GENERATE / n>=2 注入：n<2 不注 ──────────────────────────────────
def test_maybe_block_empty_for_single_variant():
    dna = _dna(secondary=[{"id": "1", "name": "A"}, {"id": "2", "name": "B"}])
    assert _maybe_diversity_block(dna, 1) == ""
    assert _maybe_diversity_block(dna, 0) == ""


def test_maybe_block_nonempty_for_multi_variant():
    dna = _dna(secondary=[{"id": "1", "name": "A"}, {"id": "2", "name": "B"}])
    block = _maybe_diversity_block(dna, 3)
    assert block.startswith("\n\n")
    assert "变式多样性" in block


# ── REGEN_PROMPT 不含多样性段（注入点语义：REGEN 是单题等价保型，不注） ──
def test_regen_prompt_has_no_diversity_clause():
    assert "变式多样性" not in variant_mod.REGEN_PROMPT


# =====================================================================
# B7 回归：改主考点 = 可回退 + 标全组 dirty 传播 + 不自动重出
# =====================================================================
def _state_two_items():
    return {
        "items": [
            {"stem": "题1", "answer": "a1", "qtype": "解答"},
            {"stem": "题2", "answer": "a2", "qtype": "解答"},
        ],
        "mother_dna": {
            "dna": {
                "main_kp": {"id": "30710101", "name": "一元一次方程"},
                "secondary_kps": [],
            }
        },
        "analysis": {
            "kp": {"value": "一元一次方程", "anchored": {"id": "30710101", "code": "30710101", "name": "一元一次方程"}, "confidence": 0.8},
            "grade": {"value": "七年级上学期"},
        },
    }


def test_b7_main_kp_change_writes_prev_snapshot_for_rollback():
    state = _state_two_items()
    update, edited, err = edit_dna_state(
        state, 1, "main_kp", {"id": "30710202", "name": "二元一次方程组"}
    )
    assert err is None
    # ④ 可回退：旧考点快照外发，撤销 = 把 main_kp 改回旧值
    assert update["main_kp_prev"]["main_kp"] == {"id": "30710101", "name": "一元一次方程"}
    assert update["main_kp_prev"]["kp"]["value"] == "一元一次方程"
    # 新考点已写入
    assert update["mother_dna"]["dna"]["main_kp"] == {"id": "30710202", "name": "二元一次方程组"}


def test_b7_main_kp_change_marks_whole_group_dirty_and_no_auto_regen():
    state = _state_two_items()
    update, edited, err = edit_dna_state(
        state, 1, "main_kp", {"id": "30710202", "name": "二元一次方程组"}
    )
    assert err is None
    # 母题脏（致命① 拦入库）
    assert update["mother_dna"]["dirty"] is True
    # 下游全组变式标 dirty + 记 main_kp 波及（传播机制）；items 仍保留（不清、不自动重出）
    items = update["items"]
    assert len(items) == 2
    for it in items:
        assert it["dna_dirty"] is True
        assert "main_kp" in (it.get("mother_dirty_dims") or [])
    # 不自动重出：题面还在
    assert items[0]["stem"] == "题1" and items[1]["stem"] == "题2"
