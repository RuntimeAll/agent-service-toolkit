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
