# -*- coding: utf-8 -*-
"""PRD-C-014 B1 · dna_extract 单题 DNA 抽取器单测（mock LLM，零网络）。

覆盖（22-SSOT 维度铁律）：
- 池内校验：main_kp 池内 id → anchored；
- 禁造词：main_kp 池外 id → main_kp=None + FLAG_MAIN_KP_OOB（锚定失败，上层走 clarify）；
- 副 kp 越界丢弃 +flag（不报错）+ ≤3 截断 + 不重复主 kp；
- 难点克制：基础题空输入 → 空输出不凑（个数代码重算）；
- 标签优先复用：复用统计正确，空池降级 +FLAG_TAG_POOL_EMPTY；
- 考察类型/题型闭集校验；难度 LLM rubric 断言 + 缺/非法兜底 2；
- LLM 解析失败 / 调用异常 → empty_dna + FLAG_MAIN_KP_OOB（触发上层 clarify），绝不抛。
"""

import asyncio
import json

from langchain_core.messages import AIMessage

import agents.dna_extract as dna_mod
from agents.dna_extract import (
    FLAG_DIFFICULTY_FALLBACK,
    FLAG_EXAM_TYPE_OOB,
    FLAG_LLM_ERROR,
    FLAG_LLM_PARSE_FAIL,
    FLAG_MAIN_KP_OOB,
    FLAG_SECONDARY_KP_OOB,
    FLAG_TAG_POOL_EMPTY,
    extract_dna,
)

# 年级叶子池夹具（[(id, name)]）—— 单测只作内存夹具，不连库/读 tsv
_POOL = [
    ("3071001001001", "一元一次方程"),
    ("3071001001002", "合并同类项"),
    ("3071001001003", "移项"),
    ("3071001001004", "去括号"),
]
_TAG_POOL = ["解方程", "移项变号", "等式性质"]


def _patch_llm(monkeypatch, payload):
    """把 relay_pool.ainvoke_failover 打桩成返回固定 JSON 文本（mock LLM）。"""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)

    async def fake_failover(messages, *, max_tokens, tags=None, on_delta=None, model=None):
        return AIMessage(content=text), "fake-relay", model or "fake-model", 0, None

    monkeypatch.setattr(dna_mod.relay_pool, "ainvoke_failover", fake_failover)


def _run(**over):
    base = dict(
        stem="解方程 2x+3=7", answer="x=2", analyze="移项得 2x=4，x=2",
        grade="七年级上学期", leaf_pool=_POOL, tag_pool=_TAG_POOL,
    )
    base.update(over)
    return asyncio.run(extract_dna(**base))


# ---------------------------------------------------------------------------
# 池内校验 / 禁造词
# ---------------------------------------------------------------------------
def test_main_kp_in_pool_anchors(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["移项", "求解"], "hard_points": [], "tags": ["解方程"],
        "scene": "纯代数", "difficulty": 2,
    })
    dna = _run()
    assert dna["main_kp"] == {"id": "3071001001001", "name": "一元一次方程"}
    assert FLAG_MAIN_KP_OOB not in dna["flags"]


def test_main_kp_out_of_pool_rejected_as_anchor_fail(monkeypatch):
    # 禁造词：LLM 编了个池外 id → 锚定失败（main_kp=None），上层据此走 clarify
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "9999999999999", "name": "瞎编的考点"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [], "tags": ["a"], "scene": "x", "difficulty": 2,
    })
    dna = _run()
    assert dna["main_kp"] is None
    assert FLAG_MAIN_KP_OOB in dna["flags"]


def test_main_kp_missing_rejected(monkeypatch):
    _patch_llm(monkeypatch, {
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": [], "hard_points": [], "tags": [], "scene": "", "difficulty": 1,
    })
    dna = _run()
    assert dna["main_kp"] is None and FLAG_MAIN_KP_OOB in dna["flags"]


# ---------------------------------------------------------------------------
# 副 kp 越界丢弃 +flag / ≤3 / 不重复主 kp
# ---------------------------------------------------------------------------
def test_secondary_kp_out_of_pool_dropped_with_flag(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [
            {"id": "3071001001002", "name": "合并同类项"},  # 池内，保留
            {"id": "8888888888888", "name": "池外副点"},       # 越界，丢弃+flag
        ],
        "qtype": "解答", "exam_type": "直接计算", "skeleton": ["x"],
        "hard_points": [], "tags": ["a"], "scene": "x", "difficulty": 2,
    })
    dna = _run()
    assert [s["id"] for s in dna["secondary_kps"]] == ["3071001001002"]
    assert FLAG_SECONDARY_KP_OOB in dna["flags"]
    # 越界副 kp 被丢弃但 main_kp 仍成功锚定（越界不报错、不拖垮主锚）
    assert dna["main_kp"]["id"] == "3071001001001"


def test_secondary_kp_capped_at_three_and_dedupe_main(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [
            {"id": "3071001001001", "name": "一元一次方程"},  # = 主 kp，剔除
            {"id": "3071001001002", "name": "合并同类项"},
            {"id": "3071001001003", "name": "移项"},
            {"id": "3071001001004", "name": "去括号"},
        ],
        "qtype": "解答", "exam_type": "直接计算", "skeleton": ["x"],
        "hard_points": [], "tags": ["a"], "scene": "x", "difficulty": 2,
    })
    dna = _run()
    sec_ids = [s["id"] for s in dna["secondary_kps"]]
    assert "3071001001001" not in sec_ids  # 不重复主 kp
    assert len(sec_ids) <= 3


# ---------------------------------------------------------------------------
# 难点克制：基础题空输入 → 空输出不凑（个数代码重算）
# ---------------------------------------------------------------------------
def test_hard_points_restraint_empty_stays_empty(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["移项", "求解"], "hard_points": [],  # 基础题 → 空
        "tags": ["解方程"], "scene": "纯代数", "difficulty": 2,
    })
    dna = _run()
    assert dna["hard_points"] == [] and dna["hard_point_count"] == 0


def test_hard_point_count_recomputed_not_trusting_llm(monkeypatch):
    # LLM 给了 2 个难点 → 个数由代码重算 = 2（不读 LLM 自报字段）
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "应用建模",
        "skeleton": ["建模", "【求解约束】"], "hard_points": ["分类讨论", "边界取舍"],
        "tags": ["应用题"], "scene": "行程问题", "difficulty": 4,
    })
    dna = _run()
    assert dna["hard_point_count"] == 2


# ---------------------------------------------------------------------------
# 标签优先复用 / 空池降级
# ---------------------------------------------------------------------------
def test_tags_reuse_count(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [],
        "tags": ["解方程", "移项变号", "全新词"],  # 前两个在复用池
        "scene": "x", "difficulty": 2,
    })
    dna = _run()
    assert dna["tag_reused_count"] == 2
    assert FLAG_TAG_POOL_EMPTY not in dna["flags"]


def test_empty_tag_pool_degrades_with_flag(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [], "tags": ["自拟标签"],
        "scene": "x", "difficulty": 2,
    })
    dna = _run(tag_pool=[])  # 复用池拉空（T4 端点未上线）→ 降级继续
    assert FLAG_TAG_POOL_EMPTY in dna["flags"]
    assert dna["tags"] == ["自拟标签"]  # 不卡死，仍出标签


def test_tags_capped_at_max(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [],
        "tags": [f"t{i}" for i in range(10)],  # 超 6 个
        "scene": "x", "difficulty": 2,
    })
    dna = _run()
    assert len(dna["tags"]) == dna_mod.TAGS_MAX


# ---------------------------------------------------------------------------
# 闭集校验：考察类型 / 题型
# ---------------------------------------------------------------------------
def test_exam_type_out_of_closed_set_nulled_with_flag(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "瞎编类型",
        "skeleton": ["x"], "hard_points": [], "tags": ["a"], "scene": "x", "difficulty": 2,
    })
    dna = _run()
    assert dna["exam_type"] is None and FLAG_EXAM_TYPE_OOB in dna["flags"]


def test_qtype_alias_normalized(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "应用题", "exam_type": "应用建模",
        "skeleton": ["x"], "hard_points": [], "tags": ["a"], "scene": "x", "difficulty": 2,
    })
    dna = _run()
    assert dna["qtype"] == "解答"  # 应用题 → 解答（闭集归一）


# ---------------------------------------------------------------------------
# 难度：LLM rubric 断言 + 缺/非法兜底 2
# ---------------------------------------------------------------------------
def test_difficulty_from_llm_rubric(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "证明推理",
        "skeleton": ["x"], "hard_points": ["难点"], "tags": ["a"], "scene": "x",
        "difficulty": 3,
    })
    dna = _run()
    assert dna["difficulty"] == 3  # 直接采信 LLM rubric 断言（非代码从难点派生）


def test_difficulty_missing_falls_back(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [], "tags": ["a"], "scene": "x",
        # difficulty 缺失
    })
    dna = _run()
    assert dna["difficulty"] == dna_mod.DIFFICULTY_FALLBACK
    assert FLAG_DIFFICULTY_FALLBACK in dna["flags"]


def test_difficulty_clamped(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [], "tags": ["a"], "scene": "x",
        "difficulty": 9,  # 越界 → 钳到 4
    })
    dna = _run()
    assert dna["difficulty"] == dna_mod.DIFFICULTY_MAX


def test_scene_truncated(monkeypatch):
    _patch_llm(monkeypatch, {
        "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["x"], "hard_points": [], "tags": ["a"],
        "scene": "场" * 100, "difficulty": 2,
    })
    dna = _run()
    assert len(dna["scene"]) == dna_mod.SCENE_MAX_LEN


# ---------------------------------------------------------------------------
# LLM 失败兜底（绝不抛 + 触发上层 clarify）
# ---------------------------------------------------------------------------
def test_llm_non_json_returns_empty_dna(monkeypatch):
    _patch_llm(monkeypatch, "这不是 JSON，模型抽风了")
    dna = _run()
    assert dna["main_kp"] is None
    assert FLAG_LLM_PARSE_FAIL in dna["flags"] and FLAG_MAIN_KP_OOB in dna["flags"]


def test_llm_raises_returns_empty_dna(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("all relays down")

    monkeypatch.setattr(dna_mod.relay_pool, "ainvoke_failover", boom)
    dna = _run()
    assert dna["main_kp"] is None
    assert FLAG_LLM_ERROR in dna["flags"] and FLAG_MAIN_KP_OOB in dna["flags"]


def test_extract_uses_light_model_by_default(monkeypatch):
    seen = {}

    async def fake_failover(messages, *, max_tokens, tags=None, on_delta=None, model=None):
        seen["model"] = model
        seen["tags"] = tags
        return AIMessage(content="{}"), "r", model, 0, None

    monkeypatch.setattr(dna_mod.relay_pool, "ainvoke_failover", fake_failover)
    _run()
    # per-call 覆盖走 nano 档（settings.LLM_MODEL_LIGHT）；JSON 中间产物 → skip_stream
    assert seen["model"] == dna_mod.settings.LLM_MODEL_LIGHT
    assert seen["tags"] == ["skip_stream"]


# ---------------------------------------------------------------------------
# 🔴 T2：refine_tags_with_pool —— 锚定后按 kp 拉标签池重选 tags（G5b 复用率 76.7%→≥80%）
# ---------------------------------------------------------------------------
from agents.dna_extract import refine_tags_with_pool  # noqa: E402

# 模拟 extract_dna 首锚产物：tag_pool 空时标签全靠 LLM 自拟、复用 0、带空池 flag
_DNA_BEFORE = {
    "main_kp": {"id": "3071001001001", "name": "一元一次方程"},
    "secondary_kps": [{"id": "3071001001002", "name": "合并同类项"}],
    "qtype": "解答", "exam_type": "直接计算",
    "skeleton": ["移项", "求解"], "hard_points": [], "hard_point_count": 0,
    "tags": ["自拟甲", "自拟乙", "自拟丙"], "tag_reused_count": 0,
    "scene": "纯代数", "difficulty": 2,
    "flags": [FLAG_TAG_POOL_EMPTY],
}
_KP_TAG_POOL = ["移项变号", "等式两边同除", "去分母", "一元一次方程标准型"]


def _refine(monkeypatch, llm_payload, *, tag_pool=None, dna=None):
    _patch_llm(monkeypatch, llm_payload)
    return asyncio.run(
        refine_tags_with_pool(
            dict(dna or _DNA_BEFORE),
            stem="解方程 2x+3=7",
            tag_pool=_KP_TAG_POOL if tag_pool is None else tag_pool,
        )
    )


def test_refine_prompt_contains_pool_words_and_prefers_reuse(monkeypatch):
    # 池非空 → prompt 含池词；LLM 从池里复用 → 复用数重算、去空池 flag
    seen = {}

    async def fake_failover(messages, *, max_tokens, tags=None, on_delta=None, model=None):
        seen["prompt"] = messages[0].content
        return AIMessage(
            content=json.dumps({"tags": ["移项变号", "去分母", "自拟新词"]}, ensure_ascii=False)
        ), "r", model, 0, None

    monkeypatch.setattr(dna_mod.relay_pool, "ainvoke_failover", fake_failover)
    out = asyncio.run(
        refine_tags_with_pool(dict(_DNA_BEFORE), stem="解方程 2x+3=7", tag_pool=_KP_TAG_POOL)
    )
    # ① prompt 注入了池词
    assert "移项变号" in seen["prompt"] and "去分母" in seen["prompt"]
    # ② 复用优先：重选后 tags 来自池 2 个 + 新词 1 个 → 复用数 = 2
    assert out["tags"] == ["移项变号", "去分母", "自拟新词"]
    assert out["tag_reused_count"] == 2
    # ③ 池非空且已重选 → 去掉空池 flag（76.7% 复用率根因 flag 消除）
    assert FLAG_TAG_POOL_EMPTY not in out["flags"]


def test_refine_empty_pool_returns_dna_unchanged(monkeypatch):
    # 池空（拉池失败/未上线）→ 原样返回，保留首锚 tags + 空池 flag（降级不卡死）
    out = _refine(monkeypatch, {"tags": ["不该被采用"]}, tag_pool=[])
    assert out["tags"] == _DNA_BEFORE["tags"]
    assert FLAG_TAG_POOL_EMPTY in out["flags"]


def test_refine_llm_failure_keeps_original_tags(monkeypatch):
    # 窄调用抛 → 保留原 tags（不退化成空，不卡死）
    async def boom(*a, **k):
        raise RuntimeError("relay down")

    monkeypatch.setattr(dna_mod.relay_pool, "ainvoke_failover", boom)
    out = asyncio.run(
        refine_tags_with_pool(dict(_DNA_BEFORE), stem="解方程", tag_pool=_KP_TAG_POOL)
    )
    assert out["tags"] == _DNA_BEFORE["tags"]


def test_refine_non_json_keeps_original_tags(monkeypatch):
    out = _refine(monkeypatch, "模型抽风非 JSON")
    assert out["tags"] == _DNA_BEFORE["tags"]


def test_refine_empty_tags_from_llm_keeps_original(monkeypatch):
    # LLM 返回空 tags → 不退化成空，保留首锚 tags
    out = _refine(monkeypatch, {"tags": []})
    assert out["tags"] == _DNA_BEFORE["tags"]


def test_refine_caps_tags_at_max(monkeypatch):
    out = _refine(monkeypatch, {"tags": [f"t{i}" for i in range(10)]})
    assert len(out["tags"]) == dna_mod.TAGS_MAX
