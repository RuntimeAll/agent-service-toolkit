# -*- coding: utf-8 -*-
"""PRD-C-014 B4·T1 · DNA 结构化回写单测（edit_dna_state，🔴 零 LLM·G11）。

铁律对照（22-SSOT §1 / CLAUDE.md §4/§5）：
- edit-dna = 零 LLM：合法值校验（题型/考察类型闭集、难度 1-4、副 kp≤3）→ 结构化回写 →
  标 manual_edited（item 级）。全程不调任何 LLM（断言 _ainvoke_text 不被调）。
- 老师改动最高优先：main_kp/grade 手动锚定 confidence 置 1.0、不质疑。
- 回写映射：main_kp/grade 同步 header(analysis)+BO 源；其余 DNA 维改 mother_dna.dna；
  qtype/difficulty 改 item。
"""

import agents.variant as variant_mod
from agents import dna_extract
from agents.variant import (
    TIER_MANUAL,
    _artifact_payload,
    _mother_facts,
    edit_dna_state,
)
from agents.variant_support import build_create_bo

_BASE_STATE = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {
            "value": "一元一次方程",
            "confidence": 0.8,
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
            "secondary_kps": [],
            "qtype": "解答",
            "exam_type": "直接计算",
            "skeleton": ["移项", "【合并同类项】"],
            "hard_points": [],
            "tags": ["解方程", "移项"],
            "scene": "纯代数",
            "difficulty": 3,
            "flags": [],
        },
    },
}


def _state(items):
    import copy

    s = copy.deepcopy(_BASE_STATE)
    s["items"] = items
    return s


def _no_llm(monkeypatch):
    """把所有 LLM 入口替换成炸弹：被调到 = 测试失败（G11 零 LLM 断言）。"""

    async def _boom(*a, **k):
        raise AssertionError("edit-dna 不该调用 LLM（G11 零 LLM）")

    monkeypatch.setattr(variant_mod, "_ainvoke_text", _boom)
    monkeypatch.setattr(variant_mod, "_solve_one", _boom)
    monkeypatch.setattr(variant_mod, "_regen_once", _boom)
    monkeypatch.setattr(dna_extract, "extract_dna", _boom)


# ===========================================================================
# 合法回写各 field（映射表逐条）
# ===========================================================================


def test_edit_dna_main_kp_updates_dna_and_analysis(monkeypatch):
    _no_llm(monkeypatch)
    items = [{"stem": "q1", "check": {"tier": "verified"}}]
    state = _state(items)
    update, item, err = edit_dna_state(
        state, 1, "main_kp", {"code": "30710202", "name": "二元一次方程"}
    )
    assert err is None
    # mother_dna.dna.main_kp 改
    assert update["mother_dna"]["dna"]["main_kp"] == {"id": "30710202", "name": "二元一次方程"}
    # analysis.kp 同步（header kp + BO dim1 源）：value/anchored/confidence
    kp = update["analysis"]["kp"]
    assert kp["value"] == "二元一次方程"
    assert kp["anchored"] == {"id": "30710202", "code": "30710202", "name": "二元一次方程"}
    assert kp["confidence"] == 1.0  # 老师手动锚定最高优先
    # 🔴 PRD-C-015 批4·缺口7·硬锚【主考点】改 = 立即解冻重锚（走既有 patch 路径）：
    #   清 items + mother_confirmed=False + facts_locked=False（不进 dirty 攒批）。
    assert update["items"] == []
    assert update["mother_confirmed"] is False
    assert update["facts_locked"] is False
    # BO dim1 落到新 kp（重锚后下一轮 classify→generate 据新 analysis.kp 出题）
    facts = _mother_facts({**state, **update})
    assert facts["dim1_kp_id"] == "30710202"


def test_edit_dna_main_kp_accepts_bare_code(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "main_kp", "30710303")
    assert err is None
    assert update["mother_dna"]["dna"]["main_kp"]["id"] == "30710303"
    assert update["analysis"]["kp"]["anchored"]["code"] == "30710303"


def test_edit_dna_secondary_kps(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(
        state, 1, "secondary_kps",
        [{"code": "30710102", "name": "等式性质"}, "30710103"],
    )
    assert err is None
    sec = update["mother_dna"]["dna"]["secondary_kps"]
    assert [s["id"] for s in sec] == ["30710102", "30710103"]
    # BO secondaryKpIds 源
    facts = _mother_facts({**state, **update})
    bo = build_create_bo(update["items"][0], facts)
    assert bo["secondaryKpIds"] == [30710102, 30710103]


def test_edit_dna_secondary_kps_dedup(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(
        state, 1, "secondary_kps", ["30710102", "30710102"]
    )
    assert err is None
    assert len(update["mother_dna"]["dna"]["secondary_kps"]) == 1


def test_edit_dna_qtype_updates_item_and_dna(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "qtype": "解答"}])
    update, item, err = edit_dna_state(state, 1, "qtype", "填空")
    assert err is None
    assert update["items"][0]["qtype"] == "填空"  # 题级
    assert update["mother_dna"]["dna"]["qtype"] == "填空"  # 守恒维同步


def test_edit_dna_qtype_alias_normalized(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "qtype": "解答"}])
    update, item, err = edit_dna_state(state, 1, "qtype", "单选题")
    assert err is None
    assert update["items"][0]["qtype"] == "选择"


def test_edit_dna_exam_type(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "exam_type", "证明推理")
    assert err is None
    assert update["mother_dna"]["dna"]["exam_type"] == "证明推理"
    facts = _mother_facts({**state, **update})
    bo = build_create_bo(update["items"][0], facts)
    assert bo["examType"] == "证明推理"


def test_edit_dna_difficulty_updates_item_and_level(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "difficulty": 2, "level": "normal"}])
    update, item, err = edit_dna_state(state, 1, "difficulty", 4)
    assert err is None
    assert update["items"][0]["difficulty"] == 4
    assert update["items"][0]["level"] == "hard"  # >=4 → hard 星级表达
    # 难度 BO dim4 取 item
    facts = _mother_facts({**state, **update})
    bo = build_create_bo(update["items"][0], facts)
    assert bo["dim4Difficulty"] == 4
    assert bo["difficult"] == 4


def test_edit_dna_difficulty_low_sets_normal(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "difficulty": 4, "level": "hard"}])
    update, item, err = edit_dna_state(state, 1, "difficulty", 1)
    assert err is None
    assert update["items"][0]["level"] == "normal"


def test_edit_dna_tags(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "tags", ["新标签A", "新标签B", "  "])
    assert err is None
    assert update["mother_dna"]["dna"]["tags"] == ["新标签A", "新标签B"]  # 空白项剔除
    facts = _mother_facts({**state, **update})
    bo = build_create_bo(update["items"][0], facts)
    assert bo["tags"] == ["新标签A", "新标签B"]


def test_edit_dna_scene(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "scene", "购物找零场景")
    assert err is None
    assert update["mother_dna"]["dna"]["scene"] == "购物找零场景"
    facts = _mother_facts({**state, **update})
    bo = build_create_bo(update["items"][0], facts)
    assert bo["scene"] == "购物找零场景"


def test_edit_dna_grade_updates_header_and_subject_id(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "grade", "八年级上学期")
    assert err is None
    g = update["analysis"]["grade"]
    assert g["value"] == "八年级上学期"
    assert g["confidence"] == 1.0  # 老师手动最高优先
    assert g["code"] == "3081"  # _grade_to_code 同步 → BO subjectId 源
    # 🔴 PRD-C-015 批4·缺口7·硬锚【年级】改 = 立即解冻重锚（同 main_kp）：清 items + 解冻。
    assert update["items"] == []
    assert update["mother_confirmed"] is False
    assert update["facts_locked"] is False
    facts = _mother_facts({**state, **update})
    assert facts["subject_id"] == "3081"  # 重锚后据新年级 code → BO subjectId 源


# ===========================================================================
# 非法值拒收（→ 400 的错误串）
# ===========================================================================


def test_edit_dna_illegal_field_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "nonsense", "x")
    assert err is not None and "非法 field" in err
    assert update == {} and item is None


def test_edit_dna_illegal_qtype_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "qtype", "多选")
    assert err is not None and "非法题型" in err
    assert update == {}


def test_edit_dna_illegal_exam_type_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "exam_type", "瞎编类型")
    assert err is not None and "非法考察类型" in err
    assert update == {}


def test_edit_dna_difficulty_out_of_range_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    for bad in (0, 5, -1, "abc", 3.5):
        update, item, err = edit_dna_state(state, 1, "difficulty", bad)
        assert err is not None and "难度" in err
        assert update == {}


def test_edit_dna_secondary_kps_over_max_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    too_many = [str(30710100 + i) for i in range(dna_extract.SECONDARY_KP_MAX + 1)]
    update, item, err = edit_dna_state(state, 1, "secondary_kps", too_many)
    assert err is not None and "副知识点最多" in err
    assert update == {}


def test_edit_dna_main_kp_empty_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, item, err = edit_dna_state(state, 1, "main_kp", "")
    assert err is not None
    assert update == {}


def test_edit_dna_index_out_of_range_rejected(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}, {"stem": "q2"}])
    for bad in (0, 3, -1):
        update, item, err = edit_dna_state(state, bad, "scene", "x")
        assert err is not None and "越界" in err
        assert update == {} and item is None


# ===========================================================================
# manual 标记 + check 中性 + 不变 input state
# ===========================================================================


def test_edit_dna_marks_item_manual(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "check": {"tier": "verified", "verify": "sympy_pass"}}])
    update, item, err = edit_dna_state(state, 1, "scene", "新场景")
    assert err is None
    assert item["manual_edited"] is True
    assert item["from_edit"] is True
    assert item["check"] == {"tier": TIER_MANUAL}  # 洗掉旧 verify 徽章
    # artifact 透传 tier=manual
    art = _artifact_payload({**state, **update})
    assert art["items"][0]["tier"] == TIER_MANUAL
    assert art["items"][0]["verify"] is None


def test_edit_dna_only_edited_item_marked_manual(monkeypatch):
    # 母题级维度改动对整组生效（守恒共享），但只有老师点的那道被标 manual
    _no_llm(monkeypatch)
    state = _state([
        {"stem": "q1", "check": {"tier": "verified"}},
        {"stem": "q2", "check": {"tier": "verified"}},
    ])
    update, item, err = edit_dna_state(state, 1, "exam_type", "纠错")
    assert err is None
    assert update["items"][0]["check"] == {"tier": TIER_MANUAL}
    assert update["items"][1]["check"] == {"tier": "verified"}  # 第 2 题不动
    assert "manual_edited" not in update["items"][1]
    # 但守恒维（exam_type）对第 2 题入库也生效（母题级共享）
    facts = _mother_facts({**state, **update})
    bo2 = build_create_bo(update["items"][1], facts)
    assert bo2["examType"] == "纠错"


def test_edit_dna_does_not_mutate_input_state(monkeypatch):
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1", "check": {"tier": "verified"}}])
    edit_dna_state(state, 1, "main_kp", "99999999")
    # 原 state 不被改（端点回写走 update）
    assert state["items"][0].get("manual_edited") is None
    assert state["mother_dna"]["dna"]["main_kp"]["id"] == "30710101"
    assert state["analysis"]["kp"]["anchored"]["code"] == "30710101"


# ===========================================================================
# artifact item.dna 嵌套对象（PRD-C-014 B4·FE DNA 面板数据源，键名钉死）
# ===========================================================================


def test_artifact_item_carries_dna_block(monkeypatch):
    # 每个 artifact item 带 dna 嵌套对象（组级维度共享 + item 级 hard_points/manual_edited）
    _no_llm(monkeypatch)
    state = _state([
        {"stem": "q1", "check": {"tier": "verified"}},
        {"stem": "q2", "hard_points": ["这道题独有的难点"]},
    ])
    art = _artifact_payload(state)
    d0 = art["items"][0]["dna"]
    # 组级（来自 mother_dna.dna）：main_kp 名称 + id，副 kp 归一 {id,name}，exam_type/tags/scene
    assert d0["main_kp"] == "一元一次方程"
    assert d0["main_kp_id"] == "30710101"
    assert d0["secondary_kps"] == []
    assert d0["exam_type"] == "直接计算"
    assert d0["tags"] == ["解方程", "移项"]
    assert d0["scene"] == "纯代数"
    # skeleton：DNA 里是 list → FE 拿 str（换行拼）
    assert d0["skeleton"] == "移项\n【合并同类项】"
    # item1 无 hard_points → 回退母题 DNA（此处母题 hard_points 空）
    assert d0["hard_points"] == []
    assert d0["manual_edited"] is False
    # item2 有 item 级 hard_points → 用 item 的
    assert art["items"][1]["dna"]["hard_points"] == ["这道题独有的难点"]


def test_artifact_item_dna_secondary_kps_normalized():
    # 副 kp 历史形态（裸 code / {code,name}）一律归一成 {id,name}，FE 弹层回填要 id
    import copy

    s = copy.deepcopy(_BASE_STATE)
    s["mother_dna"]["dna"]["secondary_kps"] = ["30710102", {"code": "30710103", "name": "等式性质"}]
    s["items"] = [{"stem": "q1"}]
    art = _artifact_payload(s)
    sec = art["items"][0]["dna"]["secondary_kps"]
    assert sec == [
        {"id": "30710102", "name": ""},
        {"id": "30710103", "name": "等式性质"},
    ]


def test_artifact_item_dna_empty_when_dna_absent():
    # 库内母题路径：mother_dna.dna 缺失 → dna 各键给空值/空数组，绝不崩
    art = _artifact_payload({
        "analysis": {"kp": {"value": "x"}, "grade": {"value": "七年级上"}},
        "mother_dna": {"stem": "母题"},  # 无 .dna
        "items": [{"stem": "q1"}],
    })
    d = art["items"][0]["dna"]
    assert d["main_kp"] is None
    assert d["main_kp_id"] is None
    assert d["secondary_kps"] == []
    assert d["exam_type"] is None
    assert d["tags"] == []
    assert d["scene"] is None
    assert d["skeleton"] is None
    assert d["hard_points"] == []
    assert d["manual_edited"] is False


def test_artifact_item_dna_manual_edited_after_edit(monkeypatch):
    # edit_dna 标 manual 的题 → artifact dna.manual_edited=True
    _no_llm(monkeypatch)
    state = _state([{"stem": "q1"}])
    update, _item, err = edit_dna_state(state, 1, "scene", "新场景")
    assert err is None
    art = _artifact_payload({**state, **update})
    assert art["items"][0]["dna"]["manual_edited"] is True
    assert art["items"][0]["dna"]["scene"] == "新场景"


def test_edit_dna_manual_keys_not_leaked_to_bo(monkeypatch):
    # manual_edited/from_edit 内部键不入库（build_create_bo 白名单挡）
    _no_llm(monkeypatch)
    state = _state([{"stem": "题干", "answer": "x=2", "solution": "解析"}])
    update, item, err = edit_dna_state(state, 1, "scene", "场景")
    facts = _mother_facts({**state, **update})
    bo = build_create_bo(item, facts)
    assert "manual_edited" not in bo
    assert "from_edit" not in bo
