# -*- coding: utf-8 -*-
"""PRD-C-015 批4·DNA 改→重生状态机单测（四分流 / 手动重生 / 母题回流 / 撤销 / 入库防脏 / 入库覆盖 / 冻结4维）。

覆盖 gate：G12（软重生维标 dirty + 硬锚立即重锚 + 点重生真重出）、G13（元数据维只标注 + 骨架/models 重写解析）、
G14（母题维改→变式标 dirty 不自动重出 + 保留手改）、G17（致命① dirty 拒入库）、
G19（缺口10 入库覆盖 update by _persist_id）、G20（缺口12 撤销重生）、G18 补（skeleton/hard_points 冻结 4/4 维）。

铁律对照（PRD §3.2c / §3.4 / §3.5 / D-merge6/7/8/9 + 致命① + 缺口7/10/12）：
- 四分流：硬锚【主考点/年级】改=解冻重锚立即（不进 dirty）；软重生维【题型/难度/考察类型/场景】+
  重写解析维【骨架/models】改=标 dirty 攒批；元数据维【标签/副考点/难点】改=只标注不进 dirty。
- 手动重生：点「重生」对待重生集合一次性重出/重写解析 + 闸B sympy 重验，保留手改 manual 维，清 dirty。
- 母题守恒维改→下游变式标 dirty 不自动重出，并入待重生集合。
- dna_dirty 题不许入库（致命①）；重生后入库=覆盖原行 update by _persist_id（缺口10）。
- 重生前存快照→撤销重生回上版（缺口12）。
- 判决只读 sympy（重生稿走 _check_one_item 闸B，批3 反退化闸自动复用）。
"""

import asyncio
import copy

import agents.variant as variant_mod
from agents import dna_extract
from agents.variant import (
    REGEN_CLASS,
    clear_item_dirty,
    dirty_item_indexes,
    edit_dna_state,
    has_dirty,
    mark_item_dirty,
    mark_mother_dirty,
    persist_dirty_guard,
    regen_class_of,
    regen_dirty_items,
    snapshot_item,
    undo_regen_item,
)

# ---------------------------------------------------------------------------
# 基线 state：定死 + 守恒维无异常 + 一道已就绪变式（not dirty）。
# ---------------------------------------------------------------------------
_BASE_STATE = {
    "mother_confirmed": True,
    "facts_locked": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {
            "value": "一元一次方程",
            "confidence": 0.9,
            "anchored": {"id": "30710101", "code": "30710101", "name": "一元一次方程"},
        },
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干",
        "answer": "x=1",
        "difficulty": 3,
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [{"id": "30710102", "name": "等式性质"}],
            "qtype": "解答",
            "exam_type": "直接计算",
            "skeleton": ["移项", "【合并同类项】"],
            "hard_points": ["符号处理"],
            "tags": ["解方程"],
            "scene": "纯代数",
            "models": [{"id": "M00", "name": "概念直用"}],
            "difficulty": 3,
            "flags": [],
        },
    },
    "facts_audit": [],
}


def _state(items):
    s = copy.deepcopy(_BASE_STATE)
    s["items"] = items
    return s


def _no_llm(monkeypatch):
    async def _boom(*a, **k):
        raise AssertionError("edit-dna / 分流 不该调用 LLM")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", _boom)
    monkeypatch.setattr(variant_mod, "_regen_once", _boom)
    monkeypatch.setattr(dna_extract, "extract_dna", _boom)


# ===========================================================================
# 0) 四分流映射 + 纯函数辅助
# ===========================================================================


def test_regen_class_mapping_covers_all_four_streams():
    # 🔴 A-2/契约C4（PRD-A-018）：main_kp 由 hard_anchor → soft_regen（单一真相，与 FE 逐字一致）。
    assert regen_class_of("main_kp") == "soft_regen"
    assert regen_class_of("grade") == "hard_anchor"
    assert regen_class_of("qtype") == "soft_regen"
    assert regen_class_of("difficulty") == "soft_regen"
    assert regen_class_of("exam_type") == "soft_regen"
    assert regen_class_of("scene") == "soft_regen"
    assert regen_class_of("skeleton") == "rewrite_solve"
    assert regen_class_of("models") == "rewrite_solve"
    assert regen_class_of("tags") == "meta"
    assert regen_class_of("secondary_kps") == "meta"
    assert regen_class_of("不存在") is None


def test_mark_and_clear_item_dirty():
    it = {"stem": "x"}
    mark_item_dirty(it, "qtype")
    mark_item_dirty(it, "difficulty")
    mark_item_dirty(it, "qtype")  # 去重
    assert it["dna_dirty"] is True
    assert it["dirty_dims"] == ["qtype", "difficulty"]
    clear_item_dirty(it)
    assert it["dna_dirty"] is False
    assert "dirty_dims" not in it


def test_dirty_item_indexes_and_has_dirty():
    items = [{"dna_dirty": False}, {"dna_dirty": True}, {"dna_dirty": True}]
    assert dirty_item_indexes(items) == [2, 3]
    assert has_dirty({"items": items}) is True
    assert has_dirty({"items": [{"dna_dirty": False}]}) is False
    # 母题脏也算
    assert has_dirty({"items": [{}], "mother_dna": {"dirty": True}}) is True


def test_snapshot_item_strips_old_snapshot_and_deepcopies():
    it = {"stem": "a", "nested": {"k": 1}, "regen_snapshot": {"old": True}}
    snap = snapshot_item(it)
    assert "regen_snapshot" not in snap
    snap["nested"]["k"] = 99  # 深拷：改快照不影响原 item
    assert it["nested"]["k"] == 1


# ===========================================================================
# 1) 软重生维改 → 标 dirty 不重出（G12）
# ===========================================================================


def test_soft_regen_dim_marks_dirty_not_rewrite(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "qtype": "解答", "difficulty": 3}])
    update, item, err = edit_dna_state(state, 1, "qtype", "填空")
    assert err is None
    # 改了软重生维 → 标 dirty + 角标，但不立即重出（题面不变，qtype 即时写但 check 置 manual）
    assert update["items"][0]["dna_dirty"] is True
    assert "qtype" in update["items"][0]["dirty_dims"]
    assert update["items"][0]["qtype"] == "填空"  # 维即时写（FE 面板反映）
    # difficulty 同理
    state2 = _state([{"stem": "q1", "difficulty": 2}])
    u2, _i, _e = edit_dna_state(state2, 1, "difficulty", 4)
    assert u2["items"][0]["dna_dirty"] is True
    assert "difficulty" in u2["items"][0]["dirty_dims"]


def test_scene_soft_regen_marks_dirty(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, _i, err = edit_dna_state(state, 1, "scene", "杭州西湖")
    assert err is None
    assert update["items"][0]["dna_dirty"] is True
    assert update["mother_dna"]["dna"]["scene"] == "杭州西湖"


# ===========================================================================
# 2) 硬锚维改 → 立即解冻重锚（不进 dirty 攒批，G12 缺口7）
# ===========================================================================


def test_main_kp_soft_regen_marks_dirty_not_clears_items(monkeypatch):
    # 🔴 A-2/契约C4（PRD-A-018）：main_kp 现为 soft_regen 母题级守恒维（不再 hard_anchor 特例分支）。
    #   承接 BUG-01（2026-06-19）·改主考点「不强制重出、可回退」：
    #   ① 不清 items（变式都还在）；② 不解冻；③ 下游变式标 dirty + 母题脏（点「重生」才据新考点重出）；
    #   ④ 旧考点快照外发可回退。行为不变，只是路由由常量直驱、无 hard_anchor 特判。
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}, {"stem": "q2"}])
    update, _i, err = edit_dna_state(state, 1, "main_kp", {"code": "30710202", "name": "二元一次方程"})
    assert err is None
    assert len(update["items"]) == 2  # 不清
    assert all(it.get("dna_dirty") for it in update["items"])  # 下游变式标脏待重生
    assert update["mother_dna"]["dirty"] is True
    assert "mother_confirmed" not in update  # 不解冻 → 不触发自动重锚重出
    assert "facts_locked" not in update
    assert update["main_kp_prev"]["main_kp"] == {"id": "30710101", "name": "一元一次方程"}  # 可回退


def test_hard_anchor_grade_unfreezes(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, _i, err = edit_dna_state(state, 1, "grade", "八年级上学期")
    assert err is None
    assert update["items"] == []
    assert update["mother_confirmed"] is False
    assert update["facts_locked"] is False


# ===========================================================================
# 3) 重写解析维（骨架/models）改 → 标 dirty（重写解析，D-merge9）
# ===========================================================================


def test_skeleton_edit_marks_dirty_rewrite_solve(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, _i, err = edit_dna_state(state, 1, "skeleton", ["新步骤一", "【新最难步】"])
    assert err is None
    # 骨架 = 守恒维（冻结 setter 留痕）+ rewrite_solve（置 dirty）
    assert update["mother_dna"]["dna"]["skeleton"] == ["新步骤一", "【新最难步】"]
    assert update["items"][0]["dna_dirty"] is True
    assert "skeleton" in update["items"][0]["dirty_dims"]
    # facts_audit 留痕（守恒维改）
    assert any(a["field"] == "skeleton" for a in update["facts_audit"])


def test_models_edit_marks_dirty_rewrite_solve(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, _i, err = edit_dna_state(
        state, 1, "models", [{"id": "M25", "name": "定边对定角（隐圆主模型）"}]
    )
    assert err is None
    # models 题级覆盖（_item_dna 读 item.models）；置 dirty（重写解析）
    assert update["items"][0]["models"] == [{"id": "M25", "name": "定边对定角（隐圆主模型）"}]
    assert update["items"][0]["dna_dirty"] is True
    assert "models" in update["items"][0]["dirty_dims"]


# ===========================================================================
# 4) 元数据维（标签/难点）改 → 只标注不进 dirty（G13）
# ===========================================================================


def test_meta_tags_edit_not_dirty(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, _i, err = edit_dna_state(state, 1, "tags", ["新标签"])
    assert err is None
    assert update["mother_dna"]["dna"]["tags"] == ["新标签"]
    # 元数据维改不进 dirty
    assert not update["items"][0].get("dna_dirty")


def test_meta_hard_points_edit_is_pure_annotation_no_regen(monkeypatch):
    """🔴 PRD-C-017 B5-fix6·hard_points 改为「纯标注」语义（对齐 UI「只标注」维）：
    只更新值 + 留痕，**不置 mother_dirty、不标下游变式 dirty、不触发任何重生波及**
    （消除全组空转：重生 prompt 全程不读 hard_points，旧逻辑波及全组 = 结果等价的空转）。"""
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}, {"stem": "q2"}])
    update, _i, err = edit_dna_state(state, 1, "hard_points", ["新难点"])
    assert err is None
    # 值即时更新（母题 DNA + item 级覆盖，母题卡/变式卡显示新难点）
    assert update["mother_dna"]["dna"]["hard_points"] == ["新难点"]
    assert update["items"][0]["hard_points"] == ["新难点"]
    # 仍走冻结 setter → facts_audit 留痕（缺口6，标注也审计）
    assert any(a["field"] == "hard_points" for a in update["facts_audit"])
    # 🔴 核心断言：不触发任何重生波及——母题不脏、所有下游变式不脏、无 mother_dirty_dims
    assert not update["mother_dna"].get("dirty")
    for it in update["items"]:
        assert not it.get("dna_dirty")
        assert not it.get("mother_dirty_dims")
    # 待重生集合为空 = 不会空转一次 LLM
    assert dirty_item_indexes(update["items"]) == []


def test_other_conserve_dims_still_propagate_after_hard_points_fix(monkeypatch):
    """🔴 B5-fix6 回归护栏：摘除 hard_points 波及不得误伤其它母题守恒维——
    scene（soft_regen 题级）+ exam_type / skeleton（母题守恒）改仍按现状触发 dirty 波及。"""
    _no_llm(monkeypatch)
    # exam_type（母题守恒维）→ 母题脏 + 下游全标 dirty
    s1 = _state([{"stem": "v1"}, {"stem": "v2"}])
    u1, _i, e1 = edit_dna_state(s1, 1, "exam_type", "证明推理")
    assert e1 is None
    assert u1["mother_dna"]["dirty"] is True
    assert all(it["dna_dirty"] for it in u1["items"])
    assert all("exam_type" in (it.get("mother_dirty_dims") or []) for it in u1["items"])
    # skeleton（母题守恒/重写解析维）→ 母题脏 + 下游全标 dirty
    s2 = _state([{"stem": "v1"}, {"stem": "v2"}])
    u2, _i, e2 = edit_dna_state(s2, 1, "skeleton", ["新骨架步"])
    assert e2 is None
    assert u2["mother_dna"]["dirty"] is True
    assert all(it["dna_dirty"] for it in u2["items"])
    # scene（soft_regen，题级）→ 本题标 dirty（非母题波及，但仍是「会重生」维，未被误摘）
    s3 = _state([{"stem": "v1"}])
    u3, _i, e3 = edit_dna_state(s3, 1, "scene", "杭州西湖")
    assert e3 is None
    assert u3["items"][0]["dna_dirty"] is True


# ===========================================================================
# 5) 母题守恒维改 → 下游变式标 dirty 不自动重出 + 并入待重生集合（G14·D-merge8）
# ===========================================================================


def test_mother_conserve_dim_change_propagates_dirty_to_variants(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "v1"}, {"stem": "v2"}, {"stem": "v3"}])
    update, _i, err = edit_dna_state(state, 1, "exam_type", "证明推理")
    assert err is None
    # 母题脏 + 全下游变式标 dirty（不自动重出——本函数不重出题面）
    assert update["mother_dna"]["dirty"] is True
    for it in update["items"]:
        assert it["dna_dirty"] is True
        assert "exam_type" in (it.get("mother_dirty_dims") or [])
    # 待重生集合 = 全部 3 道
    assert dirty_item_indexes(update["items"]) == [1, 2, 3]


def test_mark_mother_dirty_helper():
    md = {"dna": {"exam_type": "x"}}
    items = [{"stem": "a"}, {"stem": "b"}]
    mark_mother_dirty(md, items, "skeleton")
    assert md["dirty"] is True
    for it in items:
        assert it["dna_dirty"] is True
        assert it["mother_dirty_dims"] == ["skeleton"]


# ===========================================================================
# 6) 手动重生：对待重生集合一次性重出 + 闸B + 清 dirty + 保留手改（G12/G14）
# ===========================================================================


def test_regen_dirty_items_regenerates_and_clears_dirty(monkeypatch):
    # 桩 _regen_once 返回新题面；_check_one_item 直接回传（闸B 通过）
    async def fake_regen(item, facts, feedback=None):
        return {"stem": "重出后题面", "answer": item.get("answer", "x=2"), "solution": "新解析",
                "qtype": item.get("qtype"), "difficulty": item.get("difficulty")}

    async def fake_check(item, facts, idx, total):
        out = dict(item)
        out["check"] = {"tier": "verified", "badge": "ok"}
        return out, None

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)

    items = [
        {"stem": "原1", "qtype": "解答", "difficulty": 3, "dna_dirty": True, "dirty_dims": ["qtype"]},
        {"stem": "原2", "dna_dirty": False},
    ]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    assert result["regenerated"] == [1]
    assert result["failed"] == []
    # 第1道：题面真重出 + dirty 清 + 存快照（撤销用）
    new1 = update["items"][0]
    assert new1["stem"] == "重出后题面"
    assert new1["dna_dirty"] is False
    assert isinstance(new1["regen_snapshot"], dict)
    assert new1["regen_snapshot"]["stem"] == "原1"  # 快照 = 重生前
    # 第2道（not dirty）：不动
    assert update["items"][1]["stem"] == "原2"


def test_regen_only_rewrite_solve_dim_rewrites_solution_not_stem(monkeypatch):
    # 仅 models/骨架脏（无软重生维）→ 走重写解析（题面/答案不动，只改 solution）
    async def boom_regen(*a, **k):
        raise AssertionError("仅重写解析维脏不该走 _regen_once 整题重出")

    async def fake_rewrite(item, facts):
        return "按新模型重写的解析"

    async def fake_check(item, facts, idx, total):
        return dict(item), None

    monkeypatch.setattr(variant_mod, "_regen_once", boom_regen)
    monkeypatch.setattr(variant_mod, "_rewrite_solve_once", fake_rewrite)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)

    items = [{
        "stem": "题面不动", "answer": "x=2", "solution": "旧解析",
        "dna_dirty": True, "dirty_dims": ["models"],
        "models": [{"id": "M25", "name": "隐圆"}],
    }]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    assert result["regenerated"] == [1]
    new = update["items"][0]
    assert new["stem"] == "题面不动"  # 题面/答案不动
    assert new["answer"] == "x=2"
    assert new["solution"] == "按新模型重写的解析"  # 只重写解析
    assert new["dna_dirty"] is False


def test_regen_preserves_manual_edits_not_overwritten_by_mother(monkeypatch):
    # D-merge8 核心：母题脏波及 + 老师手改过本题（manual_edited）→ 重生保留手改题级维。
    async def fake_regen(item, facts, feedback=None):
        # 重生稿把 qtype 改回母题默认（解答），模拟"按新母题基准重出"
        return {"stem": "新题面", "answer": "x=2", "solution": "新解析",
                "qtype": "解答", "difficulty": 2}

    async def fake_check(item, facts, idx, total):
        return dict(item), None

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)

    # 老师手改过本题 qtype=填空、difficulty=4（manual_edited）；母题脏波及 exam_type（非 qtype/difficulty）
    items = [{
        "stem": "原", "qtype": "填空", "difficulty": 4, "level": "hard",
        "manual_edited": True, "dna_dirty": True,
        "dirty_dims": ["scene"], "mother_dirty_dims": ["exam_type"],
    }]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    new = update["items"][0]
    # 手改的 qtype/difficulty 保留（不被重生稿的"解答/2"覆盖）——母题脏维不含它们
    assert new["qtype"] == "填空"
    assert new["difficulty"] == 4
    # 题面/解析按新基准重出
    assert new["stem"] == "新题面"


def test_regen_mother_dirty_dim_does_overwrite(monkeypatch):
    # 反面：母题脏波及维若恰是手改维（qtype），则重生吃新基准（不保留旧手改）。
    async def fake_regen(item, facts, feedback=None):
        return {"stem": "新", "answer": "x=2", "solution": "s", "qtype": "解答", "difficulty": 2}

    async def fake_check(item, facts, idx, total):
        return dict(item), None

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)
    items = [{
        "stem": "原", "qtype": "填空", "manual_edited": True, "dna_dirty": True,
        "dirty_dims": ["scene"], "mother_dirty_dims": ["qtype"],  # 母题脏维 = qtype
    }]
    state = _state(items)
    update, _r, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    # qtype 是母题脏波及维 → 吃新基准（解答），不保留旧手改（填空）
    assert update["items"][0]["qtype"] == "解答"


def test_regen_clears_mother_dirty_when_all_done(monkeypatch):
    async def fake_regen(item, facts, feedback=None):
        return {"stem": "新", "answer": "1", "solution": "s", "qtype": "解答", "difficulty": 2}

    async def fake_check(item, facts, idx, total):
        return dict(item), None

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)
    items = [{"stem": "a", "dna_dirty": True, "dirty_dims": ["qtype"], "mother_dirty_dims": ["exam_type"]}]
    state = _state(items)
    state["mother_dna"]["dirty"] = True
    update, _r, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    # 全待重生集合重生完 → 清母题脏
    assert update["mother_dna"]["dirty"] is False


def test_regen_noop_when_no_dirty(monkeypatch):
    _no_llm(monkeypatch)  # 无 dirty → 不该调 LLM
    state = _state([{"stem": "a", "dna_dirty": False}])
    update, result, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    assert result["regenerated"] == [] and result["failed"] == []


def test_regen_failure_keeps_original(monkeypatch):
    async def fake_regen(item, facts, feedback=None):
        return None  # 重出失败

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    items = [{"stem": "原题保留", "dna_dirty": True, "dirty_dims": ["qtype"]}]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state))
    assert err is None
    assert result["regenerated"] == []
    assert result["failed"] and result["failed"][0]["index"] == 1
    # 重生失败 → 原题保留（G5），仍 dirty
    assert update["items"][0]["stem"] == "原题保留"


# ===========================================================================
# 6b) B5-fix·「重生这道」语义：显式 indexes = 强制整题重出（不管脏不脏）；
#     None 路 = 仍只重生 dirty 集合（原行为不变）。
# ===========================================================================


def test_regen_explicit_index_non_dirty_force_full_regen(monkeypatch):
    # ① 显式 index 指向「没改过」的非 dirty 题 → 强制整题重出（走 _regen_once，给一道全新变式）。
    regen_called = {"n": 0}

    async def fake_regen(item, facts, feedback=None):
        regen_called["n"] += 1
        return {"stem": "强制重出题面", "answer": "x=2", "solution": "新解析",
                "qtype": item.get("qtype"), "difficulty": item.get("difficulty")}

    async def boom_rewrite(*a, **k):
        raise AssertionError("force 整题重出不该走 _rewrite_solve_once（只重写解析）")

    async def fake_check(item, facts, idx, total):
        return dict(item), None

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    monkeypatch.setattr(variant_mod, "_rewrite_solve_once", boom_rewrite)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)

    # 第2道非 dirty、dirty_dims 空 —— 若按脏维分流会错走 rewrite_solve；force 必须整题重出。
    items = [
        {"stem": "原1", "dna_dirty": False},
        {"stem": "原2", "qtype": "解答", "difficulty": 3, "dna_dirty": False},
    ]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state, [2]))
    assert err is None
    assert result["regenerated"] == [2]
    assert regen_called["n"] == 1  # 真走了 _regen_once
    new2 = update["items"][1]
    assert new2["stem"] == "强制重出题面"  # 整题真重出
    assert new2["dna_dirty"] is False
    assert isinstance(new2["regen_snapshot"], dict)
    assert new2["regen_snapshot"]["stem"] == "原2"  # 快照 = 重生前（撤销用）
    # 第1道（未指定）不动
    assert update["items"][0]["stem"] == "原1"


def test_regen_indexes_none_non_dirty_stays_noop(monkeypatch):
    # ② indexes=None（自动重生待重生集合那条路）+ 非 dirty → 仍空（原行为不变）。
    _no_llm(monkeypatch)  # 不该调任何 LLM
    items = [{"stem": "a", "dna_dirty": False}, {"stem": "b", "dna_dirty": False}]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state, None))
    assert err is None
    assert result["regenerated"] == [] and result["failed"] == []


def test_regen_explicit_index_dirty_soft_dim_same_as_before(monkeypatch):
    # ③ 显式 index 指向改过软重生维（qtype）的 dirty 题 → 行为同前（整题重出 + 保留快照）。
    async def fake_regen(item, facts, feedback=None):
        return {"stem": "重出后题面", "answer": "x=2", "solution": "新解析",
                "qtype": item.get("qtype"), "difficulty": item.get("difficulty")}

    async def fake_check(item, facts, idx, total):
        return dict(item), None

    monkeypatch.setattr(variant_mod, "_regen_once", fake_regen)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)

    items = [
        {"stem": "原1", "qtype": "解答", "difficulty": 3, "dna_dirty": True, "dirty_dims": ["qtype"]},
    ]
    state = _state(items)
    update, result, err = asyncio.run(regen_dirty_items(state, [1]))
    assert err is None
    assert result["regenerated"] == [1]
    new1 = update["items"][0]
    assert new1["stem"] == "重出后题面"
    assert new1["dna_dirty"] is False
    assert new1["regen_snapshot"]["stem"] == "原1"


# ===========================================================================
# 7) 撤销重生（缺口12·G20）
# ===========================================================================


def test_undo_regen_restores_snapshot():
    snap = {"stem": "上一版题面", "qtype": "解答", "dna_dirty": True}
    items = [{"stem": "重生后题面", "qtype": "填空", "regen_snapshot": snap, "dna_dirty": False}]
    state = _state(items)
    update, restored, err = undo_regen_item(state, 1)
    assert err is None
    assert restored["stem"] == "上一版题面"
    assert update["items"][0]["stem"] == "上一版题面"
    assert update["items"][0]["qtype"] == "解答"


def test_undo_regen_restores_mother_dna_dirty_when_snapshot_was_mother_dirty():
    """🔴 PRD-A-021 R4·F4：撤销重生须复位 mother_dna.dirty，修不变量撕裂。

    场景：母题守恒维改 → 该题 dna_dirty + mother_dirty_dims（母题维波及）；重生完最后一道
    dirty 时 regen_dirty_items 把 mother_dna.dirty 清成 False。此刻老师撤销重生 → 题回到
    dna_dirty=True（重生前快照），mother_dna.dirty 必须随之回 True，否则出现
    「item.dna_dirty=True 而 mother_dna.dirty=False」撕裂态（母题守恒维同步分支漏触发）。
    """
    snap = {"stem": "上一版", "qtype": "解答", "dna_dirty": True,
            "mother_dirty_dims": ["secondary_kps"]}
    items = [{"stem": "重生后", "qtype": "解答", "regen_snapshot": snap, "dna_dirty": False}]
    state = _state(items)
    state["mother_dna"] = {**state["mother_dna"], "dirty": False}  # 重生完已清母题脏
    update, restored, err = undo_regen_item(state, 1)
    assert err is None
    assert restored["dna_dirty"] is True
    # 🔴 F4：母题脏随之复位（不变量自洽）
    assert update["mother_dna"]["dirty"] is True
    # 不变量自洽：item dna_dirty 与 mother_dna.dirty 同为 True
    assert update["items"][0]["dna_dirty"] is True


def test_undo_regen_no_mother_dirty_leaves_mother_dna_untouched():
    """🔴 F4 对照组：被撤销项不是因母题维脏（无 mother_dirty_dims）→ 不动 mother_dna.dirty
    （避免误置；单题自身重生撤销与母题守恒维无关）。"""
    snap = {"stem": "上一版", "qtype": "解答", "dna_dirty": True}  # 无 mother_dirty_dims
    items = [{"stem": "重生后", "qtype": "解答", "regen_snapshot": snap, "dna_dirty": False}]
    state = _state(items)
    state["mother_dna"] = {**state["mother_dna"], "dirty": False}
    update, restored, err = undo_regen_item(state, 1)
    assert err is None
    # 没碰 mother_dna（update 不含该键）
    assert "mother_dna" not in update


def test_undo_regen_no_snapshot_errors():
    state = _state([{"stem": "没重生过"}])
    update, restored, err = undo_regen_item(state, 1)
    assert update == {} and restored is None
    assert "没有可撤销" in err


def test_undo_regen_index_out_of_range():
    state = _state([{"stem": "a"}])
    _u, _r, err = undo_regen_item(state, 5)
    assert "越界" in err


# ===========================================================================
# 8) 入库防脏（致命①·G17）
# ===========================================================================


def test_persist_dirty_guard_blocks_dirty():
    state = _state([{"stem": "a", "dna_dirty": False}, {"stem": "b", "dna_dirty": True}])
    msg = persist_dirty_guard(state)
    assert msg is not None
    assert "第 2 题" in msg

    # 全 not dirty → None（放行）
    clean = _state([{"stem": "a", "dna_dirty": False}])
    assert persist_dirty_guard(clean) is None


def test_persist_dirty_guard_mother_dirty():
    state = _state([{"stem": "a"}])
    state["mother_dna"]["dirty"] = True
    msg = persist_dirty_guard(state)
    assert msg is not None and "母题" in msg


def test_persist_to_bank_rejects_dirty(monkeypatch):
    async def boom_persist(*a, **k):
        raise AssertionError("dirty 题不该走到 persist_items")

    monkeypatch.setattr("agents.variant.persist_items", boom_persist)
    state = _state([{"stem": "脏题", "dna_dirty": True}])
    out = asyncio.run(variant_mod.persist_to_bank(state, {}))
    msg = str(out["messages"][-1].content)
    assert "暂不能入库" in msg and "第 1 题" in msg


def test_persist_to_bank_skip_persisted_only_if_not_dirty(monkeypatch):
    # persisted 且 not dirty → 跳过；persisted 但 dirty 已被防脏闸拦在前（不会到防重逻辑）。
    captured = {}

    async def fake_persist(items, facts, token=None):
        captured["count"] = len(items)
        return [{"role": "variant", "ok": True, "id": 999} for _ in items]

    monkeypatch.setattr("agents.variant.persist_items", fake_persist)
    monkeypatch.setattr(variant_mod, "_emit_artifact", lambda *a, **k: None)
    # 两道：一道 persisted not dirty（跳过）、一道未入库 not dirty（入）
    state = _state([
        {"stem": "已收录", "persisted": True, "dna_dirty": False, "_persist_id": 100},
        {"stem": "新题", "dna_dirty": False},
    ])
    asyncio.run(variant_mod.persist_to_bank(state, {}))
    assert captured["count"] == 1  # 只入未收录那道（persisted not dirty 跳过）


# ===========================================================================
# 9) 入库覆盖（缺口10·G19）：item 带 _persist_id → update by id（非新写）
# ===========================================================================


def test_persist_overwrites_by_persist_id():
    from agents import variant_support

    calls = {"create": [], "update": []}

    class FakeClient:
        def __init__(self, token=None):
            pass

        async def create_question(self, body):
            calls["create"].append(body)
            return {"id": 12345}

        async def update_question(self, body):
            calls["update"].append(body)
            return {"id": body["id"]}

        async def aclose(self):
            pass

    orig = variant_support.RuoyiClient
    variant_support.RuoyiClient = FakeClient
    try:
        facts = {"mother_question_id": 999, "stem": "母题", "kp_name": "x", "grade": "七上",
                 "qtype": "解答", "dna": {}}
        # 一道带 _persist_id（重生后再入库 → update）、一道无（新写 → create）
        items = [
            {"stem": "重生过", "_persist_id": 555, "qtype": "解答"},
            {"stem": "全新", "qtype": "解答"},
        ]
        receipts = asyncio.run(variant_support.persist_items(items, facts))
    finally:
        variant_support.RuoyiClient = orig
    # 带 _persist_id 的走 update（覆盖原行 555），全新的走 create
    assert len(calls["update"]) == 1 and calls["update"][0]["id"] == 555
    assert len(calls["create"]) == 1 and "id" not in calls["create"][0]
    # 回执：updated 标记
    upd = [r for r in receipts if r.get("updated")]
    assert upd and upd[0]["id"] == 555


def test_build_update_bo_carries_id():
    from agents.variant_support import build_update_bo

    bo = build_update_bo({"stem": "s", "qtype": "解答"}, {"qtype": "解答", "dna": {}}, 777)
    assert bo["id"] == 777
    assert bo["stem"] == "s"


def test_persist_syncs_mother_row_when_mother_dirty():
    from agents import variant_support

    calls = {"update": []}

    class FakeClient:
        def __init__(self, token=None):
            pass

        async def create_question(self, body):
            return {"id": 1}

        async def update_question(self, body):
            calls["update"].append(body)
            return {"id": body.get("id")}

        async def aclose(self):
            pass

    orig = variant_support.RuoyiClient
    variant_support.RuoyiClient = FakeClient
    try:
        # 母题已入库（有 id）+ 母题脏 → update role=mother 行同步
        facts = {"mother_question_id": 888, "mother_dirty": True, "stem": "母题",
                 "kp_name": "x", "grade": "七上", "qtype": "解答", "dna": {}}
        receipts = asyncio.run(variant_support.persist_items([{"stem": "v", "qtype": "解答"}], facts))
    finally:
        variant_support.RuoyiClient = orig
    # role=mother 行被 update 同步（id=888）
    assert any(c.get("id") == 888 for c in calls["update"])
    assert any(r.get("role") == "mother" and r.get("updated") for r in receipts)


# ===========================================================================
# 10) skeleton/hard_points 冻结 4/4 维（批4 依赖债·G18 补）
# ===========================================================================


def test_skeleton_hardpoints_in_edit_dna_fields():
    from agents.variant import _EDIT_DNA_FIELDS

    assert "skeleton" in _EDIT_DNA_FIELDS
    assert "hard_points" in _EDIT_DNA_FIELDS
    assert "models" in _EDIT_DNA_FIELDS


def test_skeleton_frozen_setter_blocks_llm_after_lock():
    from agents.variant import _dna_fact_edit

    md = {"dna": {"skeleton": ["旧"]}}
    audit = []
    # LLM 来源 + locked → 忽略 + audit(ignored)
    wrote = _dna_fact_edit(md, "skeleton", ["新"], source="llm", locked=True, audit=audit)
    assert wrote is False
    assert md["dna"]["skeleton"] == ["旧"]  # 未改
    assert audit and audit[-1]["ignored"] is True
    # 老师来源 → 放行 + 留痕
    wrote2 = _dna_fact_edit(md, "skeleton", ["师改"], source="teacher", locked=True, audit=audit)
    assert wrote2 is True
    assert md["dna"]["skeleton"] == ["师改"]
    assert audit[-1].get("source") == "teacher" and not audit[-1].get("ignored")


def test_hard_points_frozen_setter_blocks_llm_after_lock():
    from agents.variant import _dna_fact_edit

    md = {"dna": {"hard_points": ["旧难点"]}}
    audit = []
    wrote = _dna_fact_edit(md, "hard_points", ["新难点"], source="llm", locked=True, audit=audit)
    assert wrote is False
    assert md["dna"]["hard_points"] == ["旧难点"]
    assert audit[-1]["ignored"] is True


def test_skeleton_edit_blocked_for_llm_via_edit_path_is_teacher(monkeypatch):
    # edit_dna_state 是老师手动路径（source=teacher）→ locked 也放行 + 留痕
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])  # facts_locked=True
    update, _i, err = edit_dna_state(state, 1, "skeleton", ["老师改的骨架"])
    assert err is None
    assert update["mother_dna"]["dna"]["skeleton"] == ["老师改的骨架"]
    # 留痕（teacher 源）
    sk_audits = [a for a in update["facts_audit"] if a["field"] == "skeleton"]
    assert sk_audits and not sk_audits[-1].get("ignored")
