# -*- coding: utf-8 -*-
"""PRD-C-100 B2 死循环根治补丁单测：低置信暂存 DNA 与高置信路径对齐。

现象（修复前）：母题低置信 → 弹确认框 → 老师确认章 → 没出变式、又弹确认框 = 死循环。
根因：低置信分支构造的暂存 DNA（prov_dna，variant_entry 低置信分支）只存 stem/answer/analysis，
      漏存 mother_solve_source="opus" + 完整 dna（含 main_kp/skeleton），导致 confirm 后
      classify 的 _reuse_ok（variant.py）判 False → 重锚不复用首解 → 重调 opus → niche 题坏 JSON
      → 退回 needs_confirm → 死循环。

本组用例：
- 驱动 mother_opus_entry 走低置信分支，断言返回的 mother_dna（=prov_dna）现在含
  mother_solve_source=="opus" 且 dna 非空含 main_kp（有 name）。
- 用 variant._reuse_ok 同款断言（同一逻辑，不 import private）复算：补字段后复用闸成立。

全部 LLM/HTTP monkeypatch → 零网络。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents import variant_entry as VE  # noqa: E402


# opus 一把输出（低置信：confidence=0.0 < CONF_CONFIRM_THRESHOLD → decide_confirm needs_confirm=True），
# 但解题/DNA 数据齐全（与高置信路径同 entry 结构：richText + dna.primaryKp 有 name）。
_LOWCONF_ENTRY = {
    "gradeBook": "",  # 年级册判不出 → 必弹确认
    "chapter": "第2章 一元二次方程",
    "confidence": 0.0,
    "has_figure": True,
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


def _make_fake_V():
    """mother_opus_entry 低置信分支所需的最小 V 桩（运行期 from agents import variant as V）。"""
    emitted: dict = {"need_confirm": [], "stages": [], "errors": []}

    async def _ainvoke_text(messages, **kw):
        import json
        return json.dumps(_LOWCONF_ENTRY, ensure_ascii=False)

    async def _extract_knobs(state):
        return {}

    fakeV = SimpleNamespace(
        STAGE_AWAIT="await",
        _extract_image_url=lambda t: "https://oss.example/q.png",
        _latest_human_text=lambda msgs: "出3道",
        _strip_urls=lambda t: t,
        _emit_stage=lambda *a, **k: emitted["stages"].append(a),
        _emit_need_confirm=lambda p: emitted["need_confirm"].append(p),
        _emit_error=lambda *a, **k: emitted["errors"].append(a),
        _emit_reasoning=lambda *a, **k: None,
        _sanitize_rich_text=lambda s: s,
        _parse_json=lambda t: __import__("json").loads(t),
        _extract_knobs=_extract_knobs,
        _ainvoke_text=_ainvoke_text,
        RuoyiClient=lambda token=None: SimpleNamespace(aclose=_noop_aclose),
        settings=SimpleNamespace(
            variant_model=lambda key: "m",
            MOTHER_OPUS_MAX_TOKENS=4096,
        ),
    )
    return fakeV, emitted


async def _noop_aclose():
    return None


def _run_entry(monkeypatch):
    fakeV, emitted = _make_fake_V()
    # 注入运行期 import 的 variant 模块。
    # 🔴 mother_opus_entry 内 `from agents import variant as V` = 读 agents 包的 variant 属性
    #   （agents.variant 早已被加载），setitem(sys.modules) 不生效 → 必须 patch 包属性。
    import agents
    monkeypatch.setattr(agents, "variant", fakeV)
    monkeypatch.setitem(sys.modules, "agents.variant", fakeV)
    # 旁路重型外围：预算护栏、base64 下载、记忆拉取
    from agents import cost_guard
    monkeypatch.setattr(cost_guard, "is_budget_exceeded", lambda: False)

    async def _b64(url):
        return url
    monkeypatch.setattr(VE, "_to_b64_data_url", _b64)

    # teacher_memory 模块注入（from agents import teacher_memory as TM；同 variant 走包属性 patch）
    async def _fetch_memory_block(client):
        return None
    _fake_tm = SimpleNamespace(fetch_memory_block=_fetch_memory_block)
    monkeypatch.setattr(agents, "teacher_memory", _fake_tm, raising=False)
    monkeypatch.setitem(sys.modules, "agents.teacher_memory", _fake_tm)

    state = {"messages": [HumanMessage(content="出3道 https://oss.example/q.png")]}
    out = asyncio.run(VE.mother_opus_entry(state, {"configurable": {}}))
    return out, emitted


def _reuse_ok_check(mother_dna: dict, *, confirmed_chapter_id: str) -> bool:
    """variant.py:1879-1886 _reuse_ok 同款判决（同一逻辑复刻，不 import private）。"""
    prev_dna = mother_dna.get("dna") if isinstance(mother_dna.get("dna"), dict) else None
    prev_dna = prev_dna or {}
    prev_main_kp = prev_dna.get("main_kp") if isinstance(prev_dna.get("main_kp"), dict) else None
    return bool(
        confirmed_chapter_id
        and mother_dna.get("mother_solve_source") == "opus"
        and str(mother_dna.get("stem") or "").strip()
        and prev_dna
        and prev_main_kp
        and str(prev_main_kp.get("name") or "").strip()
    )


def test_lowconf_branch_routes_to_confirm(monkeypatch):
    """低置信母题 → 确实走弹确认分支（awaiting_mother_confirm=True + 发 needConfirm）。"""
    out, emitted = _run_entry(monkeypatch)
    assert out["awaiting_mother_confirm"] is True
    assert out["mother_confirmed"] is False
    assert len(emitted["need_confirm"]) == 1


def test_lowconf_prov_dna_has_opus_source_and_dna(monkeypatch):
    """🔴 核心断言：暂存 prov_dna 现在含 mother_solve_source=="opus" 且 dna 非空含 main_kp(name)。"""
    out, _ = _run_entry(monkeypatch)
    prov_dna = out["mother_dna"]
    # 修复前漏的两项
    assert prov_dna.get("mother_solve_source") == "opus", "漏存 mother_solve_source → _reuse_ok 永 False"
    assert isinstance(prov_dna.get("dna"), dict) and prov_dna["dna"], "dna 为空 → _reuse_ok 永 False"
    main_kp = prov_dna["dna"].get("main_kp")
    assert isinstance(main_kp, dict) and (main_kp.get("name") or "").strip(), "main_kp 无 name → _reuse_ok False"
    # 原有 richText 暂存仍在（不回归）
    assert (prov_dna.get("stem") or "").strip()
    assert prov_dna.get("answer")
    assert prov_dna.get("analysis")
    # 解题骨架/答案也补了（与高置信路径对齐）
    assert prov_dna.get("solution_skeleton")
    assert prov_dna.get("solved_answer")


def test_lowconf_prov_dna_makes_reuse_gate_pass(monkeypatch):
    """端到端意义：补字段后，老师确认章（confirmed_chapter_id）→ _reuse_ok 成立 → 不再重调 opus。"""
    out, _ = _run_entry(monkeypatch)
    prov_dna = out["mother_dna"]
    # 老师确认章后（route_entry 把暂存 mother_dna 带进 classify state），复用闸成立
    assert _reuse_ok_check(prov_dna, confirmed_chapter_id="3082002") is True
    # 反证：修复前（缺 mother_solve_source / dna）必为 False
    stale = {k: v for k, v in prov_dna.items() if k not in ("mother_solve_source", "dna")}
    assert _reuse_ok_check(stale, confirmed_chapter_id="3082002") is False
