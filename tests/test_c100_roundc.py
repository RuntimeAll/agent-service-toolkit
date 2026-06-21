# -*- coding: utf-8 -*-
"""PRD-C-100 B-converge Round-C 单测：修 B2（重锚复用母题首解·不重 solve）+ B3（造图失败→
needs_figure 不抛 / 变式 stream 异常兜住·绝不裸 500）。

全部 LLM/HTTP 调用 monkeypatch → 零网络。覆盖：
- B2: classify 在「有确认章 + 首解来源 opus + 首解 DNA 在 state」时走 _reanchor_reuse_first_solve，
      **绝不重调 opus**（solve_and_label_resilient 被 spy 断言零调用），确认章收窄池能锚到叶子 →
      mother_confirmed=True（端到端可出变式）；极端 niche 锚不到 → graceful 降级锚到确认章节点 +
      待人审，仍 mother_confirmed=True（仍出变式，不退回 picker 死循环）。
- B2 对照: 无确认章（首图首解）仍走原 opus solve 路径（复用闸不误触）。
- B3: compose_variant_figure 渲染层抛异常（非 ok:False）→ 被兜成 needs_figure（不冒泡、不卡流程）；
      _stream_error_reason 把底层异常翻成可读有界文案（墙钟超时/全站失败/通用），绝不外泄堆栈。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import model_anchor  # noqa: E402
from agents import variant as V  # noqa: E402
from agents import variant_entry as VE  # noqa: E402


# ===========================================================================
# 共用桩：让 classify 的网络/锚定不打真实 HTTP/LLM
# ===========================================================================
class _FakeClient:
    async def aclose(self):
        pass


def _patch_classify_net(monkeypatch, *, leaf_pool, opus_spy):
    """打掉 classify 内一切 IO：RuoyiClient / leaf_pool_for_grade / chapter_name_for_id /
    model_anchor.anchor_models / solve_and_label_resilient(spy 记录是否被调) / 阶段灯/帧。
    返回 emitted 收集器。"""
    emitted = {"stages": [], "mother_card": 0, "need_confirm": []}

    async def fake_leaf_pool(grade_code, client, include_review_books=False):
        return list(leaf_pool)

    async def fake_chapter_name(cid, client):
        return None  # 不当聚合章处理（mother_precheck.is_aggregation_chapter_name(None)=False）

    async def fake_anchor_models(dna, **kw):
        return {"models": [dict(model_anchor.M00)], "model_overflow": [], "model_warn": False}

    async def spy_solve_and_label(*a, **k):
        opus_spy["calls"] += 1
        return {"dna": {}, "richText": {}, "solvedAnswer": ""}

    monkeypatch.setattr(V, "RuoyiClient", lambda token=None: _FakeClient())
    monkeypatch.setattr(V, "leaf_pool_for_grade", fake_leaf_pool)
    monkeypatch.setattr(V, "chapter_name_for_id", fake_chapter_name)
    monkeypatch.setattr(model_anchor, "anchor_models", fake_anchor_models)
    monkeypatch.setattr(VE, "solve_and_label_resilient", spy_solve_and_label)
    monkeypatch.setattr(V, "_emit_stage",
                        lambda *a, **k: emitted["stages"].append(a))
    monkeypatch.setattr(V, "_emit_mother_card",
                        lambda st: emitted.__setitem__("mother_card", emitted["mother_card"] + 1))
    monkeypatch.setattr(V, "_emit_need_confirm",
                        lambda p: emitted["need_confirm"].append(p))
    return emitted


def _first_solve_state(*, kp_name, image_url="https://x/q.png"):
    """模拟母题首解（mother_opus_entry 高置信但锚不到叶子）后停在 awaiting_mother_confirm 的 state：
    mother_dna 已含 opus 首解的 stem/answer/analysis/solved_answer/skeleton + dna（main_kp 仅有名）。"""
    return {
        "messages": [],
        "image_url": image_url,
        "knobs": {},
        "mother_solve_source": None,
        "awaiting_mother_confirm": True,
        "analysis": {
            "grade": {"value": "八年级下册", "code": "3082", "confidence": 0.9},
            "subject": "数学",
            "kp": {"value": kp_name, "confidence": 0.4},
            "qtype": {"value": "解答", "confidence": 0.6},
        },
        "mother_dna": {
            "stem": "已知方程 $x^2-5x+6=0$ 的两根为 $x_1,x_2$，求 $x_1+x_2$。",
            "answer": "$5$",
            "analysis": "由根与系数关系 $x_1+x_2=5$。",
            "solved_answer": "5",
            "solution_skeleton": "【韦达定理】\n两根之和 = 5",
            "mother_solve_source": "opus",  # 🔴 首解来源 opus（复用闸前置②）
            "dna": {
                "main_kp": {"id": "", "name": kp_name},  # id 空（首解锚不到）
                "secondary_kps": [],
                "qtype": "解答",
                "exam_type": "公式套用",
                "skeleton": ["【韦达定理】", "两根之和 = 5"],
                "hard_points": [],
                "hard_point_count": 0,
                "tags": ["一元二次方程", "韦达定理"],
                "scene": "纯代数",
                "difficulty": 3,
                "model_candidates": [],
                "flags": [],
            },
        },
    }


def _cfg(confirmed_chapter_id):
    return {"configurable": {"ruoyi_token": "t-teacher5",
                             "confirmed_chapter_id": confirmed_chapter_id}}


# ===========================================================================
# B2 · 重锚复用母题首解（不重 solve）
# ===========================================================================
class TestReanchorReuseFirstSolve:
    def test_reanchor_reuses_first_solve_no_opus_recall(self, monkeypatch):
        """确认章 3082002，池里有韦达定理叶子 → 重锚按名锚上 → confirmed=True + 绝不重调 opus。"""
        kp = "一元二次方程根与系数的关系（韦达定理）"
        # 确认章 3082002 收窄池里有贴切叶子（id 以确认章为前缀）
        leaf_pool = [("3082002005", kp), ("3082002001", "一元二次方程的解法")]
        opus_spy = {"calls": 0}
        _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)

        st = _first_solve_state(kp_name=kp)
        out = asyncio.run(V.classify(st, _cfg("3082002")))

        # 🔴 核心断言：重锚没有重调 opus（复用首解）
        assert opus_spy["calls"] == 0
        # 锚到了确认章内真叶子 → 定死
        assert out["mother_confirmed"] is True
        assert out["facts_locked"] is True
        anchored = (out["analysis"].get("kp") or {}).get("anchored") or {}
        assert anchored.get("id") == "3082002005"
        # 复用了首解富文本（stem 未被清空/重写）
        assert "x^2-5x+6" in (out["mother_dna"].get("stem") or "")
        # 母题卡仍先出
        # （_emit_mother_card 在 _reanchor_reuse_first_solve 末尾调一次）
        assert out["confirmed_chapter_id"] == "3082002"

    def test_reanchor_graceful_degrade_first_gates_bug03(self, monkeypatch):
        """🔴 R2a·闸3（BUG-03）：极端 niche·确认章里无贴切叶子（手选章↔主考点冲突）→ **首次闸断/
        强确认**（mother_confirmed=False + 发 needConfirm + awaiting_mother_confirm=True），不再静默
        强锚放行出锚错章的变式。仍**不重调 opus**（复用首解，B2 不破）。记下闸断章 _bug03_gated_chapter。"""
        kp = "某个题库里根本没有的怪考点"
        leaf_pool = [("3082002001", "一元二次方程的解法")]  # 没有 kp 对应叶子
        opus_spy = {"calls": 0}
        emitted = _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)

        st = _first_solve_state(kp_name=kp)
        out = asyncio.run(V.classify(st, _cfg("3082002")))

        assert opus_spy["calls"] == 0  # 仍不重调 opus（B2 复用不破）
        assert out["mother_confirmed"] is False  # 🔴 首次冲突 → 闸断（不静默放行）
        assert out["awaiting_mother_confirm"] is True
        assert out["_bug03_gated_chapter"] == "3082002"
        assert len(emitted["need_confirm"]) >= 1  # 发了再确认/换章的 picker

    def test_reanchor_graceful_degrade_insist_confirms(self, monkeypatch):
        """🔴 R2a·闸3（BUG-03）防死循环：老师**再确认同一章**（state._bug03_gated_chapter == 确认章）
        → 接受强锚放行：graceful 锚到确认章节点本身 + 待人审，mother_confirmed=True（仍能出变式）。"""
        kp = "某个题库里根本没有的怪考点"
        leaf_pool = [("3082002001", "一元二次方程的解法")]
        opus_spy = {"calls": 0}
        _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)

        st = _first_solve_state(kp_name=kp)
        st["_bug03_gated_chapter"] = "3082002"  # 老师已就该章闸断过一次，本轮再确认同章 = 坚持
        out = asyncio.run(V.classify(st, _cfg("3082002")))

        assert opus_spy["calls"] == 0  # 仍不重调 opus
        assert out["mother_confirmed"] is True  # 坚持 → 接受强锚 → 仍定死 → 仍出变式
        anchored = (out["analysis"].get("kp") or {}).get("anchored") or {}
        assert anchored.get("id") == "3082002"  # 降级锚到确认章节点本身
        assert out["mother_dna"].get("need_anchor_review") is True
        assert out.get("_bug03_gated_chapter") is None  # 放行后清闸断标记

    def test_no_confirmed_chapter_does_not_trigger_reuse(self, monkeypatch):
        """对照：无确认章（首图首解语境）→ 复用闸不触发 → 走原 opus solve 路径（spy 被调）。"""
        kp = "一元二次方程根与系数的关系（韦达定理）"
        leaf_pool = [("3082002005", kp)]
        opus_spy = {"calls": 0}
        _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)

        st = _first_solve_state(kp_name=kp)
        st["awaiting_mother_confirm"] = False
        # 无 confirmed_chapter_id → 不该走复用，落原 opus 路径
        out = asyncio.run(V.classify(st, {"configurable": {"ruoyi_token": "t"}}))
        assert opus_spy["calls"] == 1  # 走了原 opus solve（复用闸正确地没误触）
        assert isinstance(out, dict)


# ===========================================================================
# 🔴 PRD-A-021 R2a·闸2（B5b）·range-fingerprint：册变才重解 / 空 fp·同册不重解（护 B2）
# ===========================================================================
class TestRangeFingerprintGate:
    def test_same_book_reuses_no_resolve(self, monkeypatch):
        """首解 fp=3082，确认章 3082002（同册 3082）→ 不触发重解，走复用（opus 零调用，B2 字节级不变）。"""
        kp = "一元二次方程根与系数的关系（韦达定理）"
        leaf_pool = [("3082002005", kp)]
        opus_spy = {"calls": 0}
        _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)
        st = _first_solve_state(kp_name=kp)
        st["mother_dna"]["_solve_range_fp"] = "3082"  # 首解落在 3082 册
        out = asyncio.run(V.classify(st, _cfg("3082002")))
        assert opus_spy["calls"] == 0  # 同册 → 复用，不重解

    def test_empty_fp_reuses_no_resolve(self, monkeypatch):
        """🔴 空 fp（纯文字母题首解判不出册）→ **不判册变**（防 B2 死循环）→ 走复用，opus 零调用。"""
        kp = "一元二次方程根与系数的关系（韦达定理）"
        leaf_pool = [("3082002005", kp)]
        opus_spy = {"calls": 0}
        _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)
        st = _first_solve_state(kp_name=kp)
        st["mother_dna"]["_solve_range_fp"] = ""  # 空 fp
        out = asyncio.run(V.classify(st, _cfg("3082002")))
        assert opus_spy["calls"] == 0  # 空 fp → 不触发册变重解（B2 不破）

    def test_book_changed_triggers_resolve(self, monkeypatch):
        """首解 fp=3082，确认章 3091001（换到 3091 册）→ 册变 → 放弃复用、全量重 solve（opus 被调）。"""
        kp = "一元二次方程根与系数的关系（韦达定理）"
        # 新册池（确认章 3091001 前缀）；solve_and_label spy 返回最小可解析体即可
        leaf_pool = [("3091001005", kp)]
        opus_spy = {"calls": 0}
        _patch_classify_net(monkeypatch, leaf_pool=leaf_pool, opus_spy=opus_spy)
        st = _first_solve_state(kp_name=kp)
        st["mother_dna"]["_solve_range_fp"] = "3082"  # 首解落在 3082
        out = asyncio.run(V.classify(st, _cfg("3091001")))  # 老师把范围换到 3091 册
        assert opus_spy["calls"] == 1  # 册变 → 重解（窄子集触发，不撞 B2）


# ===========================================================================
# B3 · 造图渲染抛异常 → needs_figure（不冒泡）
# ===========================================================================
class TestComposeFigureRaiseContained:
    @pytest.mark.asyncio
    async def test_render_raises_is_contained_as_needs_figure(self, monkeypatch):
        """mathfig_render.render 直接抛（非 ok:False）→ compose 仍兜成 needs_figure，绝不冒泡。"""
        from agents.figure import compose

        async def invoke(messages, **kw):
            return '{"commands":["A=(0,0)","B=(1,1)"],"dashed":[]}'

        def _boom(*a, **k):
            raise RuntimeError("render subprocess crashed")

        monkeypatch.setattr(compose.mathfig_render, "render", _boom)

        def _pj(t):
            import json
            try:
                return json.loads(t)
            except Exception:
                return None

        r = await compose.compose_variant_figure(
            stem="几何题", invoke=invoke, parse_json=_pj, item_id="g1")
        assert r["ok"] is False and r["needs_figure"] is True
        assert "造图异常" in r["reason"] or "render" in r["reason"]


# ===========================================================================
# B3 · 变式 stream 异常兜住（绝不裸 500）：_stream_error_reason 文案有界可读
# ===========================================================================
class TestStreamErrorReason:
    def _reason(self, exc):
        from service.service import _stream_error_reason
        return _stream_error_reason(exc)

    def test_wallclock_timeout_reason(self):
        r = self._reason(RuntimeError("sui-xiang:WallClockTimeout; aigeek:WallClockTimeout"))
        assert "超时" in r and "重试" in r

    def test_no_relay_reason(self):
        r = self._reason(RuntimeError("no relay available"))
        assert "重试" in r and "稳定" in r

    def test_generic_reason_no_stack_leak(self):
        r = self._reason(ValueError("KeyError at /opt/secret/path traceback foo"))
        assert "重试" in r
        # 绝不外泄底层细节（路径/堆栈）
        assert "/opt/secret/path" not in r and "traceback" not in r
