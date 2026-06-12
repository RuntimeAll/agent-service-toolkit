# -*- coding: utf-8 -*-
"""表皮闸纯函数单测（PRD-C-014 B2·T4）：_surface_check 防抄母题。

闸A 三检之②。规则（纯函数，零 LLM）：
  - 题干归一化（去空白/$/反斜杠/花括号、小写）后 difflib 相似度 > 0.85 → 抄题（疑似复读母题）。
  - 题干数字集合与母题完全相同（有序相等）→ 表皮没换。
  - 母题题干为空 → 无从比对 → None（不误报）。
正样本（抄题该抓） + 负样本（正常变式该放）各覆盖。
"""

from agents.variant import (
    _surface_check,
    _surface_norm_stem,
    _surface_nums,
    _surface_threshold_for_qtype,
)

MOTHER = "解方程 2x + 3 = 11"


# ---------------------------------------------------------------------------
# 正样本：抄题该抓（返回非 None 缺陷描述）
# ---------------------------------------------------------------------------


def test_identical_stem_is_flagged():
    assert _surface_check("解方程 2x + 3 = 11", MOTHER) is not None


def test_near_identical_stem_high_similarity_flagged():
    # 只动一个标点/空格 → 相似度 >0.85
    out = _surface_check("解方程：2x+3=11", MOTHER)
    assert out is not None
    assert "相似度" in out


def test_same_numbers_different_words_flagged():
    # 数字集合 [2,3,11] 与母题完全相同（表皮没换数）→ flag，即便场景文字改了
    out = _surface_check("某商店进货 2 件、退 3 件后剩 11 件，求……", "买 2 个又 3 个共 11 元")
    assert out is not None
    assert "数字" in out


# ---------------------------------------------------------------------------
# 负样本：正常变式该放（返回 None）
# ---------------------------------------------------------------------------


def test_fully_swapped_stem_passes():
    # 数字全换、文字不同 → 正常变式
    assert _surface_check("解方程 5y - 7 = 18", MOTHER) is None


def test_different_scene_and_numbers_passes():
    assert _surface_check("一条绳子长 9 米，剪去 4 米，还剩多少米？", MOTHER) is None


def test_empty_mother_stem_never_flags():
    # 母题题干空 → 无从比对 → None（不误报）
    assert _surface_check("任意题面 5y-7=18", "") is None
    assert _surface_check("任意题面", None) is None


def test_empty_variant_stem_against_nonempty_mother_passes():
    # 变式题干空（极端兜底）：相似度低、无数字 → None（不误报，留给 structure_lint 抓空）
    assert _surface_check("", MOTHER) is None


# ---------------------------------------------------------------------------
# helpers 纯函数
# ---------------------------------------------------------------------------


def test_norm_stem_strips_whitespace_and_latex_delims():
    assert _surface_norm_stem("解方程 $2x+3=11$") == _surface_norm_stem("解方程2x+3=11")


def test_surface_nums_extracts_integers_and_decimals():
    assert _surface_nums("a 2 b 3.5 c 11") == ["2", "3.5", "11"]
    assert _surface_nums("无数字") == []


# ---------------------------------------------------------------------------
# 🔴 PRD-C-014 T1：表皮阈值按题型定死（AC4 扩样集 c018_result_v2.json §G4_surface_by_qtype 落锤）
#   选择/填空 = 0.85（扩样 max 0.577，裕量大）；解答/证明 = 0.92（长多小问共享脚手架系统性误警，
#   扩样 max 0.889 → 放宽到 0.92 不再误警，仍抓 >0.92 真复读）。
# ---------------------------------------------------------------------------

# 共享脚手架长题（解答型）：变式仅换尾部数值。归一化相似度落 (0.85, 0.92) 之间——
# 旧单值 0.85 阈值会误警，按题型 0.92 阈值不再误警（对齐扩样 max=0.889 的解答题）。
_ANS_MOTHER = (
    "解方程组并写出完整解题步骤要求分两小问作答详细说明每一步推理依据和计算过程"
    "铺垫文字数值第一个是111第二个是222"
)
_ANS_VARIANT_0_88 = (  # 实测相似度 ~0.8947，落 (0.85, 0.92) 之间
    "解方程组并写出完整解题步骤要求分两小问作答详细说明每一步推理依据和计算过程"
    "铺垫文字数值第一个是444第二个是555"
)
# 近乎复读（仅个位差异）：相似度 ~0.94 > 0.92，解答题也该警。
_ANS_MOTHER_LONG = (
    "解方程组并写出完整解题步骤要求分两小问作答详细说明每一步推理依据和计算过程"
    "数值第一个是111第二个是222第三个是333第四个是444"
)
_ANS_VARIANT_0_94 = (
    "解方程组并写出完整解题步骤要求分两小问作答详细说明每一步推理依据和计算过程"
    "数值第一个是115第二个是226第三个是337第四个是448"
)
# 选择题母题 + ~0.86 相似度变式：>0.85 → 选择题该警（同一相似度若按解答 0.92 阈值则会放过）。
_CHOICE_MOTHER = "已知方程x2+7x+10=0的两根为x1x2求乘积x1x2的值是哪个选项填A或B或C或D"
_CHOICE_VARIANT_0_86 = "已知方程x2+7x+12=0的两根为y1y2求乘积y1y2的值是哪个选项填A或B或C或D"


def test_threshold_lookup_by_qtype():
    # 选择/填空 → 0.85；解答/证明（经 _QTYPE_ALIAS 归一）→ 0.92；未知/None → 默认 0.85。
    assert _surface_threshold_for_qtype("选择") == 0.85
    assert _surface_threshold_for_qtype("填空") == 0.85
    assert _surface_threshold_for_qtype("解答") == 0.92
    assert _surface_threshold_for_qtype("证明") == 0.92  # 证明经别名归一进解答
    assert _surface_threshold_for_qtype("证明题") == 0.92
    assert _surface_threshold_for_qtype(None) == 0.85
    assert _surface_threshold_for_qtype("莫名题型") == 0.85


def test_answer_qtype_089_no_longer_false_warns():
    # 0.889 量级的解答长题（共享脚手架）：按题型 0.92 阈值 → 不再误警。
    out = _surface_check(_ANS_VARIANT_0_88, _ANS_MOTHER, qtype="解答")
    assert out is None
    # 同一对若不传 qtype（默认 0.85）则会被误警 —— 证明 T1 确实改变了判决。
    assert _surface_check(_ANS_VARIANT_0_88, _ANS_MOTHER) is not None


def test_answer_qtype_093_still_warns():
    # >0.92 的解答题（近乎复读）：解答阈值下仍判抄题。
    out = _surface_check(_ANS_VARIANT_0_94, _ANS_MOTHER_LONG, qtype="解答")
    assert out is not None
    assert "相似度" in out


def test_choice_qtype_086_still_warns():
    # 选择题 ~0.86 相似度：>0.85 → 仍警（选择/填空裕量足，阈值不放宽）。
    out = _surface_check(_CHOICE_VARIANT_0_86, _CHOICE_MOTHER, qtype="选择")
    assert out is not None
    assert "相似度" in out
    # 反证：同一 0.86 相似度若按解答 0.92 阈值则放过 —— 凸显按题型分阈值的必要。
    assert _surface_check(_CHOICE_VARIANT_0_86, _CHOICE_MOTHER, qtype="解答") is None
