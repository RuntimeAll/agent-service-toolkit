"""route_entry 入口路由回归（17 号修复任务 + 身份硬闸 2026-06-11）。

覆盖：
- 🔴 17 号漏洞本体：在途母题（停在 clarify、items 空、mother_confirmed=False）+ 老师
  纯文字答年级 → 必须进 parse 分诊，不许掉催图（修复前掉 "ask"）。
- 身份硬闸：ruoyi_token 缺失/解不出 userId → "auth"（一步不进 LLM 节点）。
- 既有分支不回归：有图→analyze / 已确认库内母题→generate / 已出题组→parse / 全空→ask。
- 续聊链路（确定性段）：parse 判「修正」后 patch 改年级 → after_patch 回 classify 重锚。
"""

import asyncio
import base64
import json

from langchain_core.messages import HumanMessage

from agents.variant import (
    after_patch,
    entry_lowconf_block,
    patch,
    route_entry,
    validate_instruction,
)
from agents.variant_entry import read_preset


def _tok(uid: int = 1) -> str:
    """构造可被 teacher_id_from_token 解出 userId 的最小 JWT（不验签，只读 payload）。"""
    payload = base64.urlsafe_b64encode(json.dumps({"userId": uid}).encode()).decode().rstrip("=")
    return f"head.{payload}.sig"


def _cfg(token: str | None = "default") -> dict:
    if token == "default":
        token = _tok()
    conf: dict = {}
    if token is not None:
        conf["ruoyi_token"] = token
    return {"configurable": conf}


def _state(text: str = "这个是9年级上的题目", **kw) -> dict:
    base: dict = {"messages": [HumanMessage(content=text)]}
    base.update(kw)
    return base


class TestAuthGate:
    def test_no_token_routes_auth(self):
        assert route_entry(_state(), _cfg(token=None)) == "auth"

    def test_garbage_token_routes_auth(self):
        assert route_entry(_state(), _cfg(token="not-a-jwt")) == "auth"

    def test_token_without_userid_routes_auth(self):
        payload = base64.urlsafe_b64encode(json.dumps({"foo": 1}).encode()).decode().rstrip("=")
        assert route_entry(_state(), _cfg(token=f"h.{payload}.s")) == "auth"

    def test_valid_token_passes_gate(self):
        # 全空 state → 过闸后落催图分支（不是 auth）
        assert route_entry(_state(), _cfg()) == "ask"


class TestRouteEntry17Fix:
    def test_inflight_mother_clarify_answer_goes_parse(self):
        """🔴 17 号漏洞本体：有 mother_dna + analysis、items 空、未确认 → parse（修复前掉 ask 催图）。"""
        state = _state(
            mother_dna={"stem": "x^2-3x+2=0"},
            analysis={"grade": {"value": "七年级下学期", "confidence": 0.4}},
            mother_confirmed=False,
            items=[],
        )
        assert route_entry(state, _cfg()) == "parse"

    def test_items_present_goes_parse(self):
        state = _state(items=[{"stem": "q1"}])
        assert route_entry(state, _cfg()) == "parse"

    def test_confirmed_mother_no_items_goes_generate(self):
        state = _state(mother_confirmed=True, mother_dna={"stem": "s"}, items=[])
        assert route_entry(state, _cfg()) == "generate"

    def test_confirmed_mother_with_items_goes_parse(self):
        # 已出过题组的编辑轮：即便 mother_confirmed 仍应进 parse 分诊，不能再直造
        state = _state(mother_confirmed=True, mother_dna={"stem": "s"}, items=[{"stem": "q"}])
        assert route_entry(state, _cfg()) == "parse"

    def test_url_always_wins(self):
        # 🔴 PRD-C-100 B1a：新图入口由 analyze 改走塌缩节点 mother_opus_entry（opus 一把判章+解题+打标）。
        state = _state(
            text="https://oss.example.com/q.png 出5道",
            mother_dna={"stem": "old"},
            items=[{"stem": "q"}],
        )
        assert route_entry(state, _cfg()) == "mother_opus_entry"

    def test_empty_state_asks_for_image(self):
        assert route_entry(_state(), _cfg()) == "ask"


class TestClarifyAnswerContinuation:
    """parse 判「修正」后的确定性续链：patch 改年级 → after_patch 回 classify 重锚。"""

    def test_validate_instruction_correction_with_zero_items(self):
        parsed = {"intent": "修正", "mother_correction": {"grade": "九年级上学期", "kp": None}}
        out = validate_instruction(parsed, current_item_count=0)
        assert out["intent"] == "修正"
        assert out["ops"] == []
        assert out["mother_correction"]["grade"] == "九年级上学期"

    def test_validate_instruction_edit_with_zero_items_degrades_clarify(self):
        # 无题组时编辑类 index 必越界 → R3 整体降级 clarify（永不乱删）
        parsed = {"intent": "编辑", "ops": [{"action": "remove", "index": 1}]}
        out = validate_instruction(parsed, current_item_count=0)
        assert out["intent"] == "clarify"

    def test_patch_grade_then_reclassify(self):
        state = _state(
            analysis={
                "grade": {"value": "七年级下学期", "confidence": 0.4},
                "kp": {"value": "一元二次方程", "confidence": 0.9},
                "qtype": {"value": "解答", "confidence": 0.9},
            },
            mother_dna={"stem": "x^2-3x+2=0"},
            mother_confirmed=False,
            items=[],
            pending={"intent": "修正", "mother_correction": {"grade": "九年级上学期", "kp": None}},
        )
        update = asyncio.run(patch(state, _cfg()))
        new_analysis = update.get("analysis") or {}
        assert new_analysis["grade"]["value"] == "九年级上学期"
        assert new_analysis["grade"]["confidence"] >= 0.9
        merged = {**state, **update}
        # 改了年级 → items 清空 + 未确认 → after_patch 必回 classify 重锚（续出题，不掉催图）
        assert after_patch(merged) == "classify"


class TestF2PatchClearsConfirmedChapter:
    """🔴 R2a·F2：patch 改年级时必清 confirmed_chapter_id（防「先确认章再纠正年级」锚错册）。"""

    def test_patch_grade_clears_confirmed_chapter_id(self):
        state = _state(
            analysis={
                "grade": {"value": "八年级下学期", "confidence": 0.9, "code": "3082"},
                "kp": {"value": "一元二次方程", "confidence": 0.9},
            },
            mother_dna={"stem": "x^2-3x+2=0"},
            mother_confirmed=False,
            items=[],
            confirmed_chapter_id="3082002",  # 老师此前确认过章（旧册 3082）
            pending={"intent": "修正", "mother_correction": {"grade": "九年级上学期", "kp": None}},
        )
        update = asyncio.run(patch(state, _cfg()))
        # 🔴 改年级 → 旧确认章作废，必清（否则 classify 仍用旧章前 4 位 3082 当年级册 = 锚错册）
        assert update.get("confirmed_chapter_id") is None
        assert update.get("_bug03_gated_chapter") is None

    def test_patch_kp_only_keeps_confirmed_chapter_id(self):
        # 只改考点（不改年级）→ 不清 confirmed_chapter_id（章语境仍有效，只换 kp 锚定）
        state = _state(
            analysis={
                "grade": {"value": "八年级下学期", "confidence": 0.9, "code": "3082"},
                "kp": {"value": "一元二次方程", "confidence": 0.9},
            },
            mother_dna={"stem": "x^2-3x+2=0"},
            mother_confirmed=False,
            items=[],
            confirmed_chapter_id="3082002",
            pending={"intent": "修正", "mother_correction": {"grade": None, "kp": "二次函数"}},
        )
        update = asyncio.run(patch(state, _cfg()))
        assert "confirmed_chapter_id" not in update  # 没改年级 → 不动 confirmed_chapter_id


class TestGate4LowconfBlock:
    """🔴 R2a·闸4（BUG-04）：读图极低置信 resume → route_entry 前置拦截（不进 classify）。"""

    def _resume_state(self, confidence, chapter="第2章 一元二次方程", **kw):
        base = _state(
            text="确认",
            awaiting_mother_confirm=True,
            entry_decision={"confidence": confidence, "chapter": chapter},
            mother_dna={"stem": "s"},
        )
        base.update(kw)
        return base

    def _cfg_confirm(self, chapter_id="3082002"):
        conf = {"ruoyi_token": _tok(), "confirmed_chapter_id": chapter_id}
        return {"configurable": conf}

    def test_very_low_conf_blocks_before_classify(self):
        st = self._resume_state(confidence=0.3)
        assert route_entry(st, self._cfg_confirm()) == "entry_lowconf_block"

    def test_chapter_unjudged_blocks(self):
        st = self._resume_state(confidence=0.9, chapter="")  # 置信高但章未判出 → 也拦
        assert route_entry(st, self._cfg_confirm()) == "entry_lowconf_block"

    def test_normal_conf_goes_classify(self):
        st = self._resume_state(confidence=0.7)
        assert route_entry(st, self._cfg_confirm()) == "classify"

    def test_threshold_040_not_blocked(self):
        # 恰好 0.40 不拦（< 0.40 才拦）
        st = self._resume_state(confidence=0.40)
        assert route_entry(st, self._cfg_confirm()) == "classify"

    def test_insist_after_block_goes_classify(self):
        # 已拦过一次（_lowconf_blocked=True）→ 老师坚持再确认 → 放行进 classify（防永久卡死）
        st = self._resume_state(confidence=0.3, _lowconf_blocked=True)
        assert route_entry(st, self._cfg_confirm()) == "classify"

    def test_block_node_sets_flag_and_keeps_awaiting(self):
        st = self._resume_state(confidence=0.2)
        out = asyncio.run(entry_lowconf_block(st, self._cfg_confirm()))
        assert out["_lowconf_blocked"] is True
        assert out["awaiting_mother_confirm"] is True  # 续接老师下一句
        assert any("换" in str(m.content) for m in out["messages"])  # 建议换图

    def test_no_entry_decision_does_not_block(self):
        # 旧线程无 entry_decision → 不拦（向后兼容）
        st = _state(
            text="确认", awaiting_mother_confirm=True, mother_dna={"stem": "s"},
        )
        assert route_entry(st, self._cfg_confirm()) == "classify"


class TestReadPreset:
    """🔴 R2a·闸1（B5）·预设输入契约：config.configurable.preset_grade_book / preset_chapter_id。"""

    def test_no_preset_returns_none(self):
        assert read_preset({"configurable": {}}) is None
        assert read_preset(None) is None

    def test_grade_only(self):
        p = read_preset({"configurable": {"preset_grade_book": "八年级下册"}})
        assert p == {"grade_book": "八年级下册", "chapter_id": "", "chapter_name": ""}

    def test_chapter_only(self):
        p = read_preset({"configurable": {"preset_chapter_id": "3082002"}})
        assert p == {"grade_book": "", "chapter_id": "3082002", "chapter_name": ""}

    def test_both(self):
        p = read_preset({"configurable": {
            "preset_grade_book": "八年级下册", "preset_chapter_id": "3082002"}})
        assert p["grade_book"] == "八年级下册" and p["chapter_id"] == "3082002"
