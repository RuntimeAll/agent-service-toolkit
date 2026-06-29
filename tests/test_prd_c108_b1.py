"""PRD-C-108 B1·薄意图层单测（G1/G2/G3 + 安全网，离线·mock 分诊，无网络）。

覆盖（in-process，确定性）：
- mother_in_doubt：锚未定死 / 无解法骨架 / 无 mother_dna → 存疑。
- route_entry_v2：对话型纯文本歧义态 → intent_triage；结构化信号（按钮/确认章/编辑 op/新图/无 token）
  → 原 route_entry 分诊（不进意图层）。
- route_after_triage：高置信意图 → 现有节点映射（G1）；母题存疑+开始→停确认（G2·AC2）；
  生成后调整母题→parse 重锚链（G3·AC3）；低置信/无意图 → 回退原 route_entry（安全网）。
- _rule_intent：易混区（改解法/重解→调整母题）规则兜底高置信命中。
"""

import base64
import json

from langchain_core.messages import HumanMessage

from agents.variant import (
    mother_in_doubt,
    route_after_triage,
    route_entry,
    route_entry_v2,
)
from agents.variant.entry.intent import (
    INTENT_CONF_THRESHOLD,
    _rule_intent,
)


def _tok(uid: int = 5) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"userId": uid}).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


def _cfg(**extra) -> dict:
    conf = {"ruoyi_token": _tok()}
    conf.update(extra)
    return {"configurable": conf}


# 立住的母题（锚定死：grade.code + kp.anchored.code，且有 solution_skeleton）。
_MOTHER_OK = dict(
    analysis={
        "grade": {"value": "八年级下学期", "confidence": 0.9, "code": "3082"},
        "kp": {"value": "一元二次方程", "confidence": 0.9,
               "anchored": {"code": "3082002001", "name": "一元二次方程"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    mother_dna={"stem": "x^2-3x+2=0", "answer": "x=1或2",
                "solution_skeleton": ["因式分解", "求根"]},
    mother_confirmed=True,
)


def _state(text: str, **kw) -> dict:
    base: dict = {"messages": [HumanMessage(content=text)]}
    base.update(kw)
    return base


def _mother_card(text: str) -> dict:
    return _state(text, awaiting_mother_review=True, items=[], **_MOTHER_OK)


def _variants(text: str, n: int = 3) -> dict:
    return _state(text, items=[{"stem": f"q{i}"} for i in range(n)], **_MOTHER_OK)


def _with_intent(state: dict, intent: str, conf: float = 0.9, **kw) -> dict:
    dec = {"intent": intent, "confidence": conf,
           "correction": {"field": None, "value": None},
           "edit": {"target_seq": None, "action": None}, "count": None}
    dec.update(kw)
    return {**state, "intent_decision": dec}


class TestMotherInDoubt:
    def test_no_mother_dna_is_doubt(self):
        assert mother_in_doubt(_state("x")) is True

    def test_pinned_with_skeleton_not_doubt(self):
        assert mother_in_doubt(_mother_card("x")) is False

    def test_unpinned_is_doubt(self):
        # 无 anchored code / 低置信 → _pin_status 不 pinned → 存疑
        st = _state("x", mother_dna={"stem": "s", "answer": "1"},
                    analysis={"grade": {"value": "?", "confidence": 0.2}})
        assert mother_in_doubt(st) is True

    def test_no_skeleton_is_doubt(self):
        st = _state("x", mother_confirmed=True,
                    analysis=_MOTHER_OK["analysis"],
                    mother_dna={"stem": "s"})  # 无 skeleton/answer
        assert mother_in_doubt(st) is True


class TestRouteEntryV2Wrap:
    """route_entry_v2 = 对话型歧义态进意图层 / 结构化信号走原 route_entry。"""

    def test_conversational_text_goes_intent_triage(self):
        assert route_entry_v2(_mother_card("嗯嗯改一下"), _cfg()) == "intent_triage"

    def test_variants_text_goes_intent_triage(self):
        assert route_entry_v2(_variants("第2题删掉"), _cfg()) == "intent_triage"

    def test_button_start_not_intent_layer(self):
        # 结构化按钮 resume → 原 route_entry 直奔 generate（不进意图层）
        assert route_entry_v2(_mother_card("开始"), _cfg(start_variants=True)) == "generate"

    def test_confirmed_chapter_not_intent_layer(self):
        st = _state("确认", awaiting_mother_confirm=True, mother_dna={"stem": "s"})
        assert route_entry_v2(st, _cfg(confirmed_chapter_id="3082002")) == "classify"

    def test_no_token_not_intent_layer(self):
        assert route_entry_v2(_mother_card("x"), {"configurable": {}}) == "auth"

    def test_image_url_not_intent_layer(self):
        st = _variants("https://oss.example.com/q.png 再出几道")
        assert route_entry_v2(st, _cfg()) == "mother_opus_entry"

    def test_empty_state_not_intent_layer(self):
        # 无母题无题组 → 不进意图层（没上下文可听懂），原 route_entry 催图
        assert route_entry_v2(_state("出题"), _cfg()) == "ask"


class TestRouteAfterTriage:
    """意图 → 现有节点映射（G1/G2/G3 + 安全网）。"""

    # --- G2·AC2·不自动开始 + 母题存疑停等 ---
    def test_doubt_start_stays_await_review(self):
        doubt = _state("开始出3道", awaiting_mother_review=True, items=[],
                       analysis={"grade": {"value": "?", "confidence": 0.2}},
                       mother_dna={"stem": "s"})
        assert route_after_triage(_with_intent(doubt, "开始出题"), _cfg()) == "await_review"

    def test_pinned_start_no_items_goes_generate(self):
        st = _mother_card("开始")
        assert route_after_triage(_with_intent(st, "开始出题"), _cfg()) == "generate"

    def test_start_with_existing_items_falls_back(self):
        # 已出题组 + 又说"开始" → 不重造，回退 route_entry（落 parse 编辑/答疑）
        st = _variants("开始")
        assert route_after_triage(_with_intent(st, "开始出题"), _cfg()) == \
            route_entry(st, _cfg())

    # --- G3·AC3·母题纠正全状态重锚（生成后也能纠，进 parse 重锚链不进编辑器）---
    def test_adjust_mother_after_generate_goes_parse(self):
        st = _variants("重新解一下")
        assert route_after_triage(_with_intent(st, "调整母题"), _cfg()) == "parse"

    def test_adjust_mother_on_card_goes_parse(self):
        st = _mother_card("按判别式法解")
        assert route_after_triage(_with_intent(st, "调整母题"), _cfg()) == "parse"

    # --- G1·编辑/答疑/确认范围映射 ---
    def test_edit_variant_with_items_goes_parse(self):
        st = _variants("第2题删掉")
        assert route_after_triage(_with_intent(st, "编辑变式"), _cfg()) == "parse"

    def test_edit_variant_no_items_falls_back(self):
        st = _mother_card("第2题删掉")  # 无 items
        assert route_after_triage(_with_intent(st, "编辑变式"), _cfg()) == \
            route_entry(st, _cfg())

    def test_qa_goes_parse(self):
        st = _variants("第3题怎么解")
        assert route_after_triage(_with_intent(st, "答疑"), _cfg()) == "parse"

    def test_confirm_scope_stays_await_review(self):
        st = _mother_card("对就是八下")
        assert route_after_triage(_with_intent(st, "确认范围"), _cfg()) == "await_review"

    # --- 安全网·低置信 / 无意图 → 回退原 route_entry ---
    def test_low_conf_falls_back_to_route_entry(self):
        st = _variants("重新解一下")
        low = route_after_triage(
            _with_intent(st, "调整母题", conf=INTENT_CONF_THRESHOLD - 0.1), _cfg())
        assert low == route_entry(st, _cfg())

    def test_no_decision_falls_back_to_route_entry(self):
        st = _variants("随便说点啥")
        assert route_after_triage(st, _cfg()) == route_entry(st, _cfg())

    def test_unknown_intent_falls_back(self):
        st = _variants("x")
        assert route_after_triage(_with_intent(st, "乱七八糟"), _cfg()) == \
            route_entry(st, _cfg())


class TestRuleFallback:
    """易混区规则兜底（高置信，离线即生效，不依赖 LLM）。"""

    def test_resolve_hits_adjust_mother(self):
        d = _rule_intent("重新解一遍", _mother_card("重新解一遍"))
        assert d is not None and d["intent"] == "调整母题"
        assert d["correction"]["field"] == "resolve" and d["confidence"] >= 0.9

    def test_solve_method_hits_adjust_mother(self):
        d = _rule_intent("解析别用因式分解，改用配方法", _variants("x"))
        assert d is not None and d["intent"] == "调整母题"
        assert d["correction"]["field"] == "solve"

    def test_plain_text_returns_none(self):
        # 非易混区 → 规则不命中，交 LLM
        assert _rule_intent("第2题删掉", _variants("第2题删掉")) is None
        assert _rule_intent("", _variants("")) is None
