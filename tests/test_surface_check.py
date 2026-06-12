# -*- coding: utf-8 -*-
"""表皮闸纯函数单测（PRD-C-014 B2·T4）：_surface_check 防抄母题。

闸A 三检之②。规则（纯函数，零 LLM）：
  - 题干归一化（去空白/$/反斜杠/花括号、小写）后 difflib 相似度 > 0.85 → 抄题（疑似复读母题）。
  - 题干数字集合与母题完全相同（有序相等）→ 表皮没换。
  - 母题题干为空 → 无从比对 → None（不误报）。
正样本（抄题该抓） + 负样本（正常变式该放）各覆盖。
"""

from agents.variant import _surface_check, _surface_norm_stem, _surface_nums

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
