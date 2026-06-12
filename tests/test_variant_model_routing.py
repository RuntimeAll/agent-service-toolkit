# -*- coding: utf-8 -*-
"""按环节分档模型路由单测（PRD-C-009 变式·2026-06-12，mock LLM 零网络）。

覆盖：
- settings.variant_model() 回退链（VARIANT_MODEL_* 缺省 → 现行为默认值）；
- VARIANT_MODEL_* 显式 .env 覆盖时优先生效；
- 各调用点吃到对应环节的 model：
  · analyze（读图）→ analyze 档（默认 None=深度档，红线不降）；
  · _solve_one / _extract_payload（闸B）→ solve 档（默认 None=深度档，对照一致率不达标不降）；
  · generate / _regen_once（出题/回炉）→ generate 档（默认 None=深度档，红线不降）；
  · DNA 抽取 → dna 档（默认 nano）。
"""

import asyncio
import json

import agents.dna_extract as dna_mod
import agents.variant as variant_mod
from agents.dna_extract import extract_dna
from agents.variant import _extract_payload, _regen_once, _solve_one
from core.settings import settings


# ---------------------------------------------------------------------------
# settings.variant_model() 回退链 + 覆盖
# ---------------------------------------------------------------------------
def test_variant_model_defaults_match_current_behavior():
    """缺省（VARIANT_MODEL_* 均 None）= 现行为：analyze/solve/generate 走深度档(None→relay model)，
    dna 走 LLM_MODEL_LIGHT(nano)。solve 默认深度档：nano-vs-5.4 对照一致率 66.7%<90% 不达标，不降。"""
    assert settings.VARIANT_MODEL_ANALYZE is None
    assert settings.VARIANT_MODEL_DNA is None
    assert settings.VARIANT_MODEL_SOLVE is None
    assert settings.VARIANT_MODEL_GENERATE is None
    assert settings.variant_model("analyze") is None
    assert settings.variant_model("generate") is None
    assert settings.variant_model("solve") is None
    assert settings.variant_model("dna") == settings.LLM_MODEL_LIGHT


def test_variant_model_env_override_wins(monkeypatch):
    """显式配置项一旦给值 → 覆盖缺省回退（以后换 deepseek 只改 .env 的证明）。"""
    monkeypatch.setattr(settings, "VARIANT_MODEL_ANALYZE", "deepseek-vl")
    monkeypatch.setattr(settings, "VARIANT_MODEL_DNA", "deepseek-chat")
    monkeypatch.setattr(settings, "VARIANT_MODEL_SOLVE", "deepseek-reasoner")
    monkeypatch.setattr(settings, "VARIANT_MODEL_GENERATE", "deepseek-r1")
    assert settings.variant_model("analyze") == "deepseek-vl"
    assert settings.variant_model("dna") == "deepseek-chat"
    assert settings.variant_model("solve") == "deepseek-reasoner"
    assert settings.variant_model("generate") == "deepseek-r1"


def test_variant_model_unknown_env_returns_none():
    assert settings.variant_model("nonsense") is None


# ---------------------------------------------------------------------------
# 各调用点吃到对应环节 model
# ---------------------------------------------------------------------------
def _capture_ainvoke(monkeypatch, ret="{}"):
    """打桩 variant._ainvoke_text，记下每次调用的 model kwarg。返回 seen list。"""
    seen: list = []

    async def fake(messages, retry=True, **kw):
        seen.append(kw.get("model"))
        return ret

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
    return seen


def test_solve_one_uses_solve_model(monkeypatch):
    monkeypatch.setattr(settings, "VARIANT_MODEL_SOLVE", "solve-x")
    seen = _capture_ainvoke(monkeypatch)
    asyncio.run(_solve_one("解方程 x+1=2"))
    assert seen == ["solve-x"]


def test_extract_payload_uses_solve_model(monkeypatch):
    monkeypatch.setattr(settings, "VARIANT_MODEL_SOLVE", "solve-x")
    seen = _capture_ainvoke(monkeypatch, ret=json.dumps({"kind": "none"}))
    asyncio.run(_extract_payload("stem", "ans", "ans", "解答"))
    assert seen and all(m == "solve-x" for m in seen)


def test_regen_uses_generate_model_not_downgraded(monkeypatch):
    """红线：回炉走 generate 档（默认 None=深度档，绝不降 nano）。"""
    monkeypatch.setattr(settings, "VARIANT_MODEL_GENERATE", None)  # 默认深度档
    seen = _capture_ainvoke(monkeypatch, ret=json.dumps({"stem": "n", "answer": "9"}))
    asyncio.run(_regen_once({"stem": "old", "qtype": "解答"}, {
        "kp_name": "x", "grade": "七年级", "qtype": "解答",
    }))
    assert seen == [None]  # None = relay 配置 model（COMPATIBLE_MODEL 深度档），未降档


def test_regen_respects_generate_override(monkeypatch):
    monkeypatch.setattr(settings, "VARIANT_MODEL_GENERATE", "gen-x")
    seen = _capture_ainvoke(monkeypatch, ret=json.dumps({"stem": "n", "answer": "9"}))
    asyncio.run(_regen_once({"stem": "old", "qtype": "解答"}, {
        "kp_name": "x", "grade": "七年级", "qtype": "解答",
    }))
    assert seen == ["gen-x"]


def test_dna_extract_uses_dna_model_via_injected_invoke(monkeypatch):
    """DNA 抽取经注入的 invoke（生产 = variant._ainvoke_text）时，吃到 dna 档 model。"""
    monkeypatch.setattr(settings, "VARIANT_MODEL_DNA", "dna-x")
    seen: list = []

    async def fake_invoke(messages, *, model=None, **kw):
        seen.append(model)
        return json.dumps({"main_kp": None, "tags": []})

    asyncio.run(extract_dna(
        stem="s", grade="七年级",
        leaf_pool=[("1", "一元一次方程")], tag_pool=[],
        model=settings.variant_model("dna"), invoke=fake_invoke,
    ))
    assert seen == ["dna-x"]


def test_dna_extract_default_model_is_light(monkeypatch):
    """dna 档缺省 = LLM_MODEL_LIGHT（dna_extract 自身回退也是它，双保险一致）。"""
    seen: list = []

    async def fake_invoke(messages, *, model=None, **kw):
        seen.append(model)
        return json.dumps({"main_kp": None, "tags": []})

    asyncio.run(extract_dna(
        stem="s", grade="七年级",
        leaf_pool=[("1", "一元一次方程")], tag_pool=[],
        model=settings.variant_model("dna"), invoke=fake_invoke,
    ))
    assert seen == [settings.LLM_MODEL_LIGHT]
