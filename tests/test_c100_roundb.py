# -*- coding: utf-8 -*-
"""PRD-C-100 B-converge Round-B 单测：修 B1（锚不到叶子→needs_confirm 而非 clarify 死胡同）
+ B2（classify 重锚 opus 坏 JSON 自愈网复用入口，不死循环）。

全部 LLM/HTTP 调用 monkeypatch → 零网络。覆盖：
- B2: parse_or_repair_entry 对坏 JSON（fence/未转义引号/数组包裹）自愈成 dict；
      solve_and_label_resilient 坏 JSON 重读一次后成功；自愈网耗尽 → _SolveLabelError（不无限回环）。
- B1: _finalize_high_conf 主考点锚不到叶子 → awaiting_mother_confirm=True + 发 needConfirm
      （导向 picker）、清 awaiting_mother_review；锚到叶子的高置信 happy path 仍 confirmed/正常。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage  # noqa: E402

from agents import mother_opus, model_anchor  # noqa: E402
from agents import variant_entry as VE  # noqa: E402


# ===========================================================================
# B2 · 解析自愈口径（parse_or_repair_entry）
# ===========================================================================
class _FakeVParse:
    """最小 V：parse_or_repair_entry 只需 _parse_json + (LLM 兜底用) _ainvoke_text。"""

    @staticmethod
    def _parse_json(text):
        import json
        import re
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        body = (m.group(1) if m else text).strip()
        try:
            return json.loads(body)
        except Exception:
            return None

    @staticmethod
    async def _ainvoke_text(messages, **kw):  # LLM 兜底（本组用例不触发到这里）
        raise RuntimeError("llm-repair-not-expected-here")


class TestParseOrRepairEntry:
    def test_clean_json_parses(self):
        out = asyncio.run(VE.parse_or_repair_entry('{"a": 1}', _FakeVParse))
        assert out == {"a": 1}

    def test_markdown_fence_stripped(self):
        out = asyncio.run(VE.parse_or_repair_entry('```json\n{"a": 2}\n```', _FakeVParse))
        assert out == {"a": 2}

    def test_array_unwrapped_to_first_dict(self):
        out = asyncio.run(VE.parse_or_repair_entry('[{"a": 3}, {"a": 4}]', _FakeVParse))
        assert out == {"a": 3}

    def test_unescaped_quote_repaired_deterministically(self):
        # 字符串值内未转义 ASCII 引号（id=1399 失败模式）→ 确定性引号修复救活，不触 LLM
        broken = '{"name": "关于"x"的方程", "v": 5}'
        out = asyncio.run(VE.parse_or_repair_entry(broken, _FakeVParse))
        assert isinstance(out, dict)
        assert out.get("v") == 5

    def test_unrepairable_returns_none(self):
        # 彻底坏（截断半截）+ LLM 兜底也 raise → None（调用方据此重读/降级，不卡死）
        out = asyncio.run(VE.parse_or_repair_entry('{"a": ', _FakeVParse))
        assert out is None


# ===========================================================================
# B2 · 重锚自愈网（solve_and_label_resilient）—— 复用入口同口径（重试≤2 + 解析修复）
# ===========================================================================
class _VResilient(_FakeVParse):
    pass


def _patch_solve(monkeypatch, returns):
    """让 mother_opus.solve_and_label 按 returns 序列逐次返回（str）或抛（Exception 实例）。"""
    seq = list(returns)
    calls = {"n": 0}

    async def fake(**kw):
        i = calls["n"]
        calls["n"] += 1
        item = seq[i] if i < len(seq) else seq[-1]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(mother_opus, "solve_and_label", fake)
    return calls


def test_resilient_bad_json_then_good_on_retry(monkeypatch):
    # 第 1 次坏 JSON（解析+修复都救不活）→ 自愈网重读母题一次 → 第 2 次好 → 成功（不早退）
    calls = _patch_solve(monkeypatch, ['{"oops": ', '{"dna": {"x": 1}}'])
    progress = []
    out = asyncio.run(VE.solve_and_label_resilient(
        image_url="u", prompt="p", V=_VResilient, model="m",
        on_progress=progress.append,
    ))
    assert out == {"dna": {"x": 1}}
    assert calls["n"] == 2  # 重读了一次（入口同口径 ≤2）
    assert any("重读" in p for p in progress)


def test_resilient_first_call_raises_then_succeeds(monkeypatch):
    # 第 1 次调用异常（超时）→ 重试 → 第 2 次好
    calls = _patch_solve(monkeypatch, [TimeoutError("blip"), '{"ok": 1}'])
    out = asyncio.run(VE.solve_and_label_resilient(
        image_url="u", prompt="p", V=_VResilient, model="m",
    ))
    assert out == {"ok": 1}
    assert calls["n"] == 2


def test_resilient_parse_exhausted_raises_controlled(monkeypatch):
    # 两次都坏 JSON → 自愈耗尽 → 抛 _SolveLabelError(parse_only=True)，不无限循环
    calls = _patch_solve(monkeypatch, ['{"bad": ', '{"still": '])
    try:
        asyncio.run(VE.solve_and_label_resilient(
            image_url="u", prompt="p", V=_VResilient, model="m",
        ))
        assert False, "should raise"
    except VE._SolveLabelError as e:
        assert e.parse_only is True
        assert calls["n"] == 2  # 有界：恰好 2 次，绝不无限回环


def test_resilient_call_exhausted_raises_controlled(monkeypatch):
    # 两次都调用异常 → 抛 _SolveLabelError(parse_only=False, last_exc 携带)
    _patch_solve(monkeypatch, [TimeoutError("a"), TimeoutError("b")])
    try:
        asyncio.run(VE.solve_and_label_resilient(
            image_url="u", prompt="p", V=_VResilient, model="m",
        ))
        assert False, "should raise"
    except VE._SolveLabelError as e:
        assert e.parse_only is False
        assert isinstance(e.last_exc, TimeoutError)


# ===========================================================================
# B1 · _finalize_high_conf 锚不到叶子 → needs_confirm（picker），非 clarify 死胡同
# ===========================================================================
class _FakeClient:
    async def aclose(self):
        pass


def _make_fake_V(monkeypatch, *, kp_anchors: bool):
    """构造 _finalize_high_conf 所需的 V 桩。kp_anchors=False → 池里没有 opus 给的考点名 → 锚不到叶子。"""
    emitted = {"need_confirm": [], "mother_card": 0, "stages": []}

    async def leaf_pool_for_grade(grade_code, client, include_review_books=False):
        # 池里只有一个无关叶子；opus 主考点名「根的判别式」永远锚不到（kp_anchors=False 时）
        if kp_anchors:
            return [("3082002003", "根的判别式")]
        return [("3082009999", "无关叶子")]

    async def anchor_models(dna, **kw):  # 不实际调；让上层 try 成功走正常（M00 也行）
        return {"models": [dict(model_anchor.M00)], "model_overflow": [], "model_warn": False}

    fakeV = SimpleNamespace(
        CONF_GATE=0.75,
        CONF_CONFIRM_THRESHOLD=0.80,
        RuoyiClient=lambda token=None: _FakeClient(),
        leaf_pool_for_grade=leaf_pool_for_grade,
        _grade_to_code=lambda v: "3082",
        _resolve_grade_code=None,
        _wants_review_books=lambda t: False,
        _latest_human_text=lambda msgs: "出3道",
        _sanitize_rich_text=lambda s: s,
        _conf_ok=lambda a: bool((a.get("kp") or {}).get("anchored")),
        knobs_desc=lambda k: "默认配方",
        settings=SimpleNamespace(variant_model=lambda key: "m"),
        build_mother_confirm=lambda st: {"needs_confirm": not st.get("mother_confirmed")},
        _emit_need_confirm=lambda p: emitted["need_confirm"].append(p),
        _emit_mother_card=lambda st: emitted.__setitem__("mother_card", emitted["mother_card"] + 1),
        _emit_stage=lambda *a, **k: emitted["stages"].append(a),
        _ainvoke_text=None,
    )
    # model_anchor 走真模块，但 invoke=fakeV._ainvoke_text=None → anchor_models 内部会异常 →
    # _finalize_high_conf 的 except 兜 M00；为稳定改成桩。
    monkeypatch.setattr(model_anchor, "anchor_models", anchor_models)
    return fakeV, emitted


_ENTRY_ANCHORABLE = {
    "gradeBook": "八年级下册", "chapter": "第2章 一元二次方程", "confidence": 0.92,
    "has_figure": False,
    "richText": {"stem": "解方程 $x^2-5x+6=0$", "answer": "$x=2$ 或 $x=3$", "analysis": "因式分解"},
    "solvedAnswer": "x=2 或 x=3",
    "dna": {
        "primaryKp": {"id": "", "name": "根的判别式"},
        "secondaryKps": [], "qtype": "解答", "assessmentType": "公式套用",
        "solutionSkeleton": ["移项", "【因式分解】"], "hardPointCount": 0,
        "breakthroughPoints": [], "scenario": "纯代数", "difficulty": 2,
        "tags": ["一元二次方程", "解方程"], "modelCandidates": [],
    },
}


def _run_finalize(monkeypatch, *, kp_anchors):
    fakeV, emitted = _make_fake_V(monkeypatch, kp_anchors=kp_anchors)
    entry = dict(_ENTRY_ANCHORABLE)
    decision = VE.decide_confirm(entry)  # 高置信无歧义 → needs_confirm=False
    assert decision["needs_confirm"] is False
    base_out = {
        "image_url": "u", "images_count": 1, "questions_in_image": 1, "knobs": {},
        "shape_defects": [], "mother_precheck": None, "awaiting_mother_confirm": False,
        "mother_rejected": False, "confirmed_chapter_id": None, "confirmed_grade_book_id": None,
        "mother_has_figure": False, "entry_decision": decision, "_entry_finalized": True,
    }
    out = asyncio.run(VE._finalize_high_conf({"messages": []}, {}, entry, decision, base_out, fakeV))
    return out, emitted


def test_b1_unanchored_leaf_routes_to_needs_confirm(monkeypatch):
    """锚不到章内叶子 → awaiting_mother_confirm=True + 发 needConfirm（picker），不是 clarify 死胡同。"""
    out, emitted = _run_finalize(monkeypatch, kp_anchors=False)
    assert out["mother_confirmed"] is False
    assert out["awaiting_mother_confirm"] is True        # 关键：进确认态而非 clarify 死胡同
    assert out["awaiting_mother_review"] is False         # 清 stale review，防「开始」误路由
    assert out["facts_locked"] is False
    assert len(emitted["need_confirm"]) == 1             # 真发了 picker
    assert emitted["mother_card"] == 1                   # 母题卡仍先出（卡先出不变）
    # 有一句明确「请确认年级与章」的引导话，不是静默
    body = out["messages"][0].content if out["messages"] else ""
    assert "确认" in body and "章" in body


def test_b1_anchored_leaf_high_conf_happy_path_unchanged(monkeypatch):
    """对照：锚到真叶子 → confirmed=True、不发 needConfirm、awaiting_mother_confirm=False（happy path 不破）。"""
    out, emitted = _run_finalize(monkeypatch, kp_anchors=True)
    assert out["mother_confirmed"] is True
    assert out["awaiting_mother_confirm"] is False
    assert out["facts_locked"] is True
    assert len(emitted["need_confirm"]) == 0             # 高置信锚上 → 不弹 picker
    assert emitted["mother_card"] == 1
    # happy path 不带「开始举一反三」回 parse 死胡同 → 由 gate_after_classify 走 await_review
