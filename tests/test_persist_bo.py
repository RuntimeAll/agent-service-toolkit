"""build_create_bo / build_mother_bo 入库 BO 回归（PRD-C-013 Phase0 收口）。

🔴 真因回归：biz_question.difficult 是 NOT NULL 且无 DB 默认值（book-server 实测
   SQLException "Field 'difficult' doesn't have a default value" → create 500）。
   二期 P8 _grade_difficulty 降级路径可能给 item 留下缺失/非数难度，旧代码
   `if difficult is not None` 会漏掉 difficult 列 → c012 P5b item#0 入库 500。
   断言：无论 item/母题难度缺失或非数，BO 必带合法 difficult(1~4)。
"""

from agents.variant_support import build_create_bo, build_mother_bo


def _variant(difficulty):
    return {"qtype": "选择", "stem": "x", "answer": "A", "solution": "因为", "difficulty": difficulty}


def test_variant_bo_always_has_difficult_when_missing():
    bo = build_create_bo(_variant(None), {"subject_id": "1"})
    assert "difficult" in bo and 1 <= bo["difficult"] <= 4


def test_variant_bo_always_has_difficult_when_nonnumeric():
    bo = build_create_bo(_variant("难"), {"subject_id": "1"})
    assert "difficult" in bo and 1 <= bo["difficult"] <= 4


def test_variant_bo_falls_back_to_mother_difficulty():
    bo = build_create_bo(_variant(None), {"subject_id": "1", "mother_difficulty": 3})
    assert bo["difficult"] == 3


def test_variant_bo_keeps_and_clamps_item_difficulty():
    assert build_create_bo(_variant(2), {})["difficult"] == 2
    assert build_create_bo(_variant(9), {})["difficult"] == 4  # 越界钳 1~4


def test_mother_bo_always_has_difficult_when_missing():
    bo = build_mother_bo({"qtype": "选择", "stem": "x"})  # 无 mother_difficulty
    assert "difficult" in bo and 1 <= bo["difficult"] <= 4


# ── subject_id NOT NULL 真因回归（2026-06-12 实测：锚定失败 subject_id=None →
#    INSERT subject_id=NULL → "Column 'subject_id' cannot be null" → 母题+全部变式 500）──


def test_variant_bo_subject_id_falls_back_to_unclassified_when_none():
    # 锚定失败（facts 无 subject_id）→ 必带 subjectId="0"（未分类），绝不漏列
    bo = build_create_bo(_variant(2), {})
    assert bo.get("subjectId") == "0"


def test_variant_bo_keeps_real_subject_id():
    bo = build_create_bo(_variant(2), {"subject_id": "3001002"})
    assert bo["subjectId"] == "3001002"


def test_mother_bo_subject_id_falls_back_to_unclassified_when_none():
    bo = build_mother_bo({"qtype": "选择", "stem": "x"})  # 无 subject_id
    assert bo.get("subjectId") == "0"


def test_mother_bo_keeps_real_subject_id():
    bo = build_mother_bo({"qtype": "选择", "stem": "x", "subject_id": "3071"})
    assert bo["subjectId"] == "3071"


# ── PRD-A-023 B11 守恒维兜底回归（裸变式根治）：dim1_kp_id(analysis.anchored.code) 空但
#    DNA 锚到组级守恒主考点(dna.main_kp.id) → 每道变式一律绑守恒 kp，绝不落「无考点」裸变式。
#    根因：add/regenerate 等增量轮次 analysis 无 anchored → facts.dim1_kp_id 为空，而
#    母题/同组其余变式走确认面定死轮 anchored 齐 → 同批部分裸、部分正常（DB 实测 16:13 批）。


def _facts_dna(main_kp_id, *, dim1=None, secondary=None):
    dna = {"main_kp": ({"id": main_kp_id, "name": "守恒考点"} if main_kp_id else None),
           "secondary_kps": secondary or [], "tags": ["t"], "flags": []}
    return {"subject_id": "3071", "dim1_kp_id": dim1, "dna": dna,
            "mother_question_id": 123456789, "kp_confidence": 0.9}


def test_variant_dim1_falls_back_to_conserved_main_kp_when_anchored_missing():
    # dim1_kp_id 空 + dna.main_kp 有 → dim1KpId 兜底取守恒主考点（不再裸变式）
    bo = build_create_bo(_variant(2), _facts_dna("3071005"))
    assert bo["dim1KpId"] == "3071005"
    assert bo["anchorId"] == "3071005"
    assert bo["needAnchorReview"] is False  # 守恒主考点在 → 不转人审


def test_variant_primary_anchored_wins_over_conserved():
    # dim1_kp_id（确认面定死）非空 → 优先用它，不被守恒兜底覆盖
    bo = build_create_bo(_variant(2), _facts_dna("3071005", dim1="3071999"))
    assert bo["dim1KpId"] == "3071999"


def test_variant_secondary_kps_bind_when_conserved_main_present():
    # 守恒主考点在 → main_kp_unclassified=False → 副考点照落（不被守恒一致性闸抑制）
    bo = build_create_bo(_variant(2), _facts_dna("3071005", secondary=[{"id": "3071006"}]))
    assert bo.get("secondaryKpIds") == [3071006]


def test_variant_truly_unclassified_when_both_sources_empty():
    # 两源皆空（anchored 缺 + dna.main_kp 缺）→ 真未分类：无 dim1KpId、转人审、副考点抑制
    bo = build_create_bo(_variant(2), _facts_dna(None, secondary=[{"id": "3071006"}]))
    assert "dim1KpId" not in bo
    assert bo["needAnchorReview"] is True
    assert "secondaryKpIds" not in bo


def test_variant_inherits_mother_lineage():
    bo = build_create_bo(_variant(2), _facts_dna("3071005"))
    assert bo["motherQuestionId"] == 123456789


def test_mother_bo_dim1_falls_back_to_conserved_main_kp():
    facts = {"qtype": "选择", "stem": "x", "dim1_kp_id": None,
             "dna": {"main_kp": {"id": "3071005", "name": "守恒考点"}, "flags": []}}
    bo = build_mother_bo(facts)
    assert bo["dim1KpId"] == "3071005"
