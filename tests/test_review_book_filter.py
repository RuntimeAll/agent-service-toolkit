# -*- coding: utf-8 -*-
"""批1（2026-06-13 整改）· 锚定池只圈教材册、剔复习册（防锚到「新题抢先」等同名节点照样出题）。

覆盖：
- leaf_pool_for_grade 兜底（grade_code 缺/过滤空）剔复习册，绝不裸 return 全量含复习册；
- 复习册前缀（3010/3100/3120）正常 grade 路径不入池；include_review_books=True 才并入；
- _resolve_grade_code 反查年级排除复习册候选（exclude_review_books 透传）；
- dna_extract._validate 复习册 main_kp 未开放 → 按越界处置（FLAG_MAIN_KP_OOB+REVIEW_OOB）；
- _wants_review_books 关键词开关（中考|复习|专题|模考|一模|二模）。
"""

import asyncio

import agents.dna_extract as dna_mod
import agents.variant as variant_mod
from agents.dna_extract import (
    FLAG_MAIN_KP_OOB,
    FLAG_MAIN_KP_REVIEW_OOB,
    _validate,
)
from agents.variant import _wants_review_books
from agents.variant_support import (
    REVIEW_BOOK_PREFIXES,
    _is_review_book,
    _non_review_leaves,
    leaf_pool_for_grade,
)

# 含教材册 + 复习册的混合全量叶子夹具（模拟 lazyTree 全树抽叶子结果）
_MIXED_LEAVES = [
    ("3071001001001", "一元一次方程"),       # 七上教材册
    ("3081002003004", "二次根式有意义的条件"),  # 八上教材册
    ("3010005", "中考综合卷题"),               # 中考一轮复习
    ("3100009", "解题技巧专题"),               # 数学解题技巧与专题
    ("3120004", "未解析"),                     # 新题抢先（实锚事故节点）
]


class _FakeTreeClient:
    """leaf_pool_for_grade 用的 client 桩：lazy_tree 返回扁平叶子列表（无 children = 叶子）。"""

    def __init__(self, leaves):
        self._leaves = leaves

    async def lazy_tree(self, body=None):
        return [{"id": i, "name": n} for i, n in self._leaves]


# ---------------------------------------------------------------------------
# 复习册前缀识别
# ---------------------------------------------------------------------------
def test_review_prefixes_pinned():
    assert REVIEW_BOOK_PREFIXES == {"3010", "3100", "3120"}
    assert _is_review_book("3120004") is True
    assert _is_review_book("3010005") is True
    assert _is_review_book("3071001001001") is False


def test_non_review_leaves_strips_review_books():
    out = _non_review_leaves(_MIXED_LEAVES)
    ids = {i for i, _ in out}
    assert ids == {"3071001001001", "3081002003004"}  # 只剩两道教材册叶子


# ---------------------------------------------------------------------------
# leaf_pool_for_grade 兜底剔复习册
# ---------------------------------------------------------------------------
def test_pool_fallback_excludes_review_books():
    # grade_code=None → 走兜底全量；不该裸 return 含复习册的 1473 等价池
    client = _FakeTreeClient(_MIXED_LEAVES)
    pool = asyncio.run(leaf_pool_for_grade(None, client))
    ids = {i for i, _ in pool}
    assert ids == {"3071001001001", "3081002003004"}
    assert not any(_is_review_book(i) for i in ids)


def test_pool_scoped_grade_path_excludes_review_books():
    # grade_code=3081 命中八上教材册叶子；复习册同前缀不存在 → 池纯教材册
    client = _FakeTreeClient(_MIXED_LEAVES)
    pool = asyncio.run(leaf_pool_for_grade("3081", client))
    assert {i for i, _ in pool} == {"3081002003004"}


def test_pool_include_review_books_admits_them():
    # 显式开放 → 复习册并入兜底池
    client = _FakeTreeClient(_MIXED_LEAVES)
    pool = asyncio.run(leaf_pool_for_grade(None, client, include_review_books=True))
    ids = {i for i, _ in pool}
    assert "3120004" in ids and "3010005" in ids and "3100009" in ids


def test_pool_include_review_books_scoped_to_review_prefix():
    # 开放复习册 + grade_code=3120（新题抢先）→ 能圈到复习册叶子
    client = _FakeTreeClient(_MIXED_LEAVES)
    pool = asyncio.run(
        leaf_pool_for_grade("3120", client, include_review_books=True)
    )
    assert {i for i, _ in pool} == {"3120004"}


# ---------------------------------------------------------------------------
# _resolve_grade_code 反查排除复习册
# ---------------------------------------------------------------------------
def test_resolve_grade_code_reverse_excludes_review(monkeypatch):
    captured = {}

    def fake_anchor(coarse, *a, **kw):
        captured["exclude"] = kw.get("exclude_review_books")
        # 模拟：剔复习册后只剩教材册候选（grade_code=3081 八上）
        return [{"id": "3081002003004", "grade_code": "3081", "name": "二次根式有意义的条件"}]

    monkeypatch.setattr(variant_mod, "anchor_subject", fake_anchor)
    analysis = {
        "grade": {"value": None, "confidence": 0},  # LLM 没给年级 → 反查
        "kp": {"value": "二次根式有意义的条件", "confidence": 0.4},
    }
    gc = asyncio.run(variant_mod._resolve_grade_code(analysis))
    assert gc == "3081"
    assert captured["exclude"] is True  # 反查时排除复习册


# ---------------------------------------------------------------------------
# dna_extract._validate 复习册闸（二道保险）
# ---------------------------------------------------------------------------
def test_validate_review_main_kp_treated_oob_when_closed():
    pool_ids = {"3120004", "3071001001001"}
    raw = {"main_kp": {"id": "3120004", "name": "未解析"}, "secondary_kps": [],
           "qtype": "解答", "exam_type": "直接计算", "skeleton": [], "hard_points": [],
           "tags": [], "scene": "", "difficulty": 2}
    out = _validate(raw, pool_ids, [], include_review_books=False)
    assert out["main_kp"] is None  # 复习册未开放 → 锚定失败
    assert FLAG_MAIN_KP_OOB in out["flags"]
    assert FLAG_MAIN_KP_REVIEW_OOB in out["flags"]


def test_validate_review_main_kp_allowed_when_open():
    pool_ids = {"3120004"}
    raw = {"main_kp": {"id": "3120004", "name": "新题考点"}, "secondary_kps": [],
           "qtype": "解答", "exam_type": "直接计算", "skeleton": [], "hard_points": [],
           "tags": [], "scene": "", "difficulty": 2}
    out = _validate(raw, pool_ids, [], include_review_books=True)
    assert out["main_kp"] == {"id": "3120004", "name": "新题考点"}  # 开放 → 锚定成功
    assert FLAG_MAIN_KP_OOB not in out["flags"]


def test_validate_textbook_main_kp_unaffected():
    pool_ids = {"3071001001001"}
    raw = {"main_kp": {"id": "3071001001001", "name": "一元一次方程"}, "secondary_kps": [],
           "qtype": "解答", "exam_type": "直接计算", "skeleton": [], "hard_points": [],
           "tags": [], "scene": "", "difficulty": 2}
    out = _validate(raw, pool_ids, [], include_review_books=False)
    assert out["main_kp"]["id"] == "3071001001001"
    assert FLAG_MAIN_KP_OOB not in out["flags"]


# ---------------------------------------------------------------------------
# 关键词开关
# ---------------------------------------------------------------------------
def test_wants_review_books_keywords():
    assert _wants_review_books("出几道中考压轴题") is True
    assert _wants_review_books("这是复习用的") is True
    assert _wants_review_books("二次根式专题") is True
    assert _wants_review_books("来套模考题") is True
    assert _wants_review_books("一模真题") is True
    assert _wants_review_books("出5道一元二次方程") is False
    assert _wants_review_books("") is False
    assert _wants_review_books(None) is False


# ---------------------------------------------------------------------------
# dna_extract 候选行带册子归属
# ---------------------------------------------------------------------------
def test_build_prompt_shows_book_attribution():
    prompt = dna_mod._build_prompt(
        stem="s", answer="a", analyze="z", grade="八年级上学期",
        leaf_pool=[("3081002003004", "二次根式有意义的条件")], tag_pool=[],
    )
    assert "（册：八年级上册）" in prompt
    # prompt 含禁选复习册的硬约束
    assert "禁止选复习类册子" in prompt
