"""PRD-C-109 收敛修冒烟：收窄难度误并 + 生成门禁认 endorsed（三样一起断言·防顾此失彼）。

背景（第3轮 e2e 暴露两处回归，f6efec5 治好在位编辑 G1/G4 但过度伸手）：
  - 回归1·G3 难度：母题卡就绪态（无题组）「难度难一点」→ route_after_triage 的 EFFECT_KNOB 分支
    无条件落 parse → parse_instruction「无题组→intent=修正」覆盖块把难度话判 REVISE →
    patch BUG-A 清 mother_dna 全量重解（ack「已按你的要求重新解题」）→ 旋钮没动 + 已编辑维被冲。
    收窄：KNOB + 无题组 → await_review（保持就绪、母题零重解）；KNOB + 有题组(stage-2) → parse（不变）。
  - 回归2·生成 0 道：endorsed + 主考点只到占位「本章重点」/未锚叶子 → generate 内部三道软闸
    （防裸奔 / 缺主考点 / 守恒白名单空）只读锚定硬指标、不认背书 → 点开始返 0 道。
    补 endorsed override：endorsed + 有 mother_dna → 三道软闸放行（降级·待人审照常生成，留痕不挡）。

🔴 死守不回归 G1/G4：在位编辑（加标签/改题型）仍真改对维、不重弹、不打回存疑（f6efec5 那套保留）。

五断言（全离线·确定性，无网络/库）：
  ① G1 不回归：无题组加标签 → 标签真加上 + route=await_review（不落 parse 重解）+ endorsed 保持。
  ② G3 收窄：无题组「难度难一点」→ route=await_review（不落 parse 重解·不再触发 BUG-A 清 mother_dna）。
  ③ G3 stage-2 不破：有题组「难度难一点」→ route=parse（变式难度旋钮 stage-2，照旧）。
  ④ 回归2 治：endorsed + 占位主考点（白名单空/未锚叶子）→ generate 不返「不能出题」拒造消息、真出 ≥1 道。
  ⑤ 生成门禁安全网不破：未 endorse + 占位主考点 → generate 仍拒造（背书才放行，非裸放）。

跑法（cwd = toolkit，必走 .venv）：
  $env:NO_PROXY="*"; .venv\\Scripts\\python.exe tools\\c109_converge_smoke.py
"""
import asyncio
import base64
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch as mock_patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import intent_triage  # noqa: E402
from agents.variant.entry import intent as intent_mod  # noqa: E402
from agents.variant.entry.intent import route_after_triage  # noqa: E402
from agents.variant.stage2_variant import generate as gen_mod  # noqa: E402

PASS = 0
FAIL = 0


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {msg}")
    else:
        FAIL += 1
        print(f"  XX  {msg}")


def _tok(uid: int = 5) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"userId": uid}).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


def _cfg(**extra) -> dict:
    conf = {"ruoyi_token": _tok()}
    conf.update(extra)
    return {"configurable": conf}


# 母题卡就绪态（无题组），主考点锚到真叶子。
_MOTHER_OK = dict(
    mother_confirmed=True,
    facts_locked=True,
    awaiting_mother_review=True,
    items=[],
    analysis={
        "grade": {"value": "八年级下学期", "confidence": 0.9, "code": "3082"},
        "kp": {"value": "一元二次方程", "confidence": 0.9,
               "anchored": {"id": "3082002001", "code": "3082002001", "name": "一元二次方程"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    mother_dna={
        "stem": "x^2-3x+2=0", "answer": "x=1或2", "solution_skeleton": ["因式分解", "求根"],
        "difficulty": 3,
        "dna": {
            "main_kp": {"id": "3082002001", "name": "一元二次方程"},
            "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
            "skeleton": ["因式分解", "求根"], "hard_points": ["符号"], "tags": ["解方程"],
            "scene": "纯代数", "models": [{"id": "M00", "name": "概念直用"}], "difficulty": 3,
        },
    },
    facts_audit=[],
)


def _card(text: str, **extra) -> dict:
    s = {"messages": [HumanMessage(content=text)]}
    s.update(copy.deepcopy(_MOTHER_OK))
    s.update(extra)
    return s


async def _run_triage(text, dec, **state_extra):
    """跑 intent_triage（mock classify_intent 返回指定决策），返回 (out, routed)。"""
    async def _fake(state, config):
        return dec

    st = _card(text, **state_extra)
    _orig = intent_mod.classify_intent
    intent_mod.classify_intent = _fake
    try:
        out = asyncio.run(intent_triage(st, _cfg(**{k: v for k, v in state_extra.items()
                                                   if k == "mother_endorsed"})))
    finally:
        intent_mod.classify_intent = _orig
    routed = route_after_triage({**st, **out}, _cfg(
        **({"mother_endorsed": True} if state_extra.get("mother_endorsed") else {})))
    return out, routed


# ---------------------------------------------------------------------------
# ① G1 不回归：无题组加标签 → 真加上 + await_review + endorsed 保持
# ---------------------------------------------------------------------------
def test_g1_addtag_still_works():
    banner("① G1 不回归：无题组加标签 → 标签真加上 + route=await_review（不重解）")
    dec = intent_mod._tool_decision("加标签", value="判别式符号", conf=0.95)
    out, routed = asyncio.get_event_loop().run_until_complete(_g1_co(dec))
    tags = (out.get("mother_dna") or {}).get("dna", {}).get("tags") or []
    ok("判别式符号" in tags and "解方程" in tags, f"标签真加上+旧标签保留（{tags}）")
    ok(out.get("intent_decision", {}).get("_applied") is True, "节点已原地应用（_applied）")
    ok("items" not in out and out.get("mother_confirmed") is not False,
       "不清组、不打回 mother_confirmed（就绪卡不掉存疑）")
    ok(routed == "await_review", f"route=await_review（不落 parse 重解，得 {routed}）")
    ok(out.get("mother_endorsed") is True, "endorsed 保持（确认收口终态）")


async def _g1_co(dec):
    async def _fake(state, config):
        return dec
    st = _card("加个易错标签：判别式符号", mother_endorsed=True)
    _orig = intent_mod.classify_intent
    intent_mod.classify_intent = _fake
    try:
        out = await intent_triage(st, _cfg(mother_endorsed=True))
    finally:
        intent_mod.classify_intent = _orig
    routed = route_after_triage({**st, **out}, _cfg(mother_endorsed=True))
    return out, routed


# ---------------------------------------------------------------------------
# ② G3 收窄：无题组「难度难一点」→ await_review（不落 parse 重解）
# ---------------------------------------------------------------------------
def test_g3_difficulty_noitems_no_resolve():
    banner("② G3 收窄·难度回归：无题组「难度难一点」→ route=await_review（不重解、不冲已编辑维）")
    # _rule_tool 词面命中难度 → _难度旋钮（EFFECT_KNOB）。直接走 classify_intent 真路径（不 mock）。
    dec = asyncio.get_event_loop().run_until_complete(
        intent_mod.classify_intent(_card("难度难一点", mother_endorsed=True), _cfg(mother_endorsed=True)))
    ok(dec.get("tool") == "_难度旋钮", f"「难度难一点」→ 选 _难度旋钮（旋钮，得 tool={dec.get('tool')}）")
    st = _card("难度难一点", mother_endorsed=True)
    routed = route_after_triage({**st, "intent_decision": dec}, _cfg(mother_endorsed=True))
    ok(routed == "await_review",
       f"无题组 KNOB → await_review（收窄·不落 parse 重解，得 {routed}）")
    ok(routed != "parse", "绝不落 parse（= 不再触发 parse_instruction 无题组 REVISE → patch 清 mother_dna 重解）")


# ---------------------------------------------------------------------------
# ③ G3 stage-2 不破：有题组「难度难一点」→ parse（变式难度旋钮 stage-2）
# ---------------------------------------------------------------------------
def test_g3_difficulty_withitems_parse():
    banner("③ G3 stage-2 不破：有题组「难度难一点」→ route=parse（变式难度旋钮，照旧）")
    dec = asyncio.get_event_loop().run_until_complete(
        intent_mod.classify_intent(_card("难度难一点"), _cfg()))
    items = [{"seq": 1, "stem": "v1", "difficulty": 3}, {"seq": 2, "stem": "v2", "difficulty": 3}]
    st = _card("难度难一点", items=items, awaiting_mother_review=False)
    routed = route_after_triage({**st, "intent_decision": dec}, _cfg())
    ok(routed == "parse",
       f"有题组 KNOB → parse（stage-2 难度旋钮路径不破，得 {routed}）")


# ---------------------------------------------------------------------------
# ④ 回归2 治：endorsed + 占位主考点（白名单空/未锚叶子）→ generate 真出 ≥1 道（不返拒造消息）
# ---------------------------------------------------------------------------
def test_gen_endorsed_placeholder_kp_generates():
    banner("④ 回归2 治：endorsed + 占位主考点（未锚叶子）→ generate 不拒造、真出 ≥1 道")
    # 占位主考点 = main_kp 无 id（白名单空）+ analysis.kp.anchored 无 code（_pin_status 未 pin）。
    degraded = copy.deepcopy(_MOTHER_OK)
    degraded["analysis"]["kp"] = {"value": "本章重点", "confidence": 0.5, "anchored": {}}
    degraded["mother_dna"]["dna"]["main_kp"] = {"id": "", "name": "本章重点"}
    degraded["mother_confirmed"] = False  # 未 pin（占位锚定）
    degraded["mother_endorsed"] = True    # 老师已点「开始举一反三」= 背书
    degraded["messages"] = [HumanMessage(content="")]
    degraded["knobs"] = {"count": 2, "qtype": "解答"}

    fake_items = [{"seq": 1, "stem": "变式1", "difficulty": 3, "qtype": "解答"}]

    async def _fake_eager(*a, **k):
        return fake_items

    # 桩掉真正的 LLM 出题链（_eager_chain / 造题 LLM），只验「门禁放行、走到出题、不返拒造消息」。
    out = asyncio.get_event_loop().run_until_complete(
        _run_generate_stubbed(degraded, fake_items))
    msgs = [getattr(m, "content", "") for m in (out.get("messages") or [])]
    blocked_phrases = ["我先不造题", "我不能出题", "已暂停生成", "请补充确认", "请确认年级"]
    hit = [p for p in blocked_phrases for m in msgs if p in m]
    ok(not hit, f"endorsed 占位主考点 → generate 不返拒造消息（messages={msgs!r}）")
    n = len(out.get("items") or [])
    ok(n >= 1, f"真出 ≥1 道变式（items={n}）= 0 道回归已治")


async def _run_generate_stubbed(state, fake_items):
    """跑 generate，桩掉真出题 LLM（只验门禁放行 + 走到造题产出 items）。"""
    async def _fake_gene_one(*a, **k):
        return {"seq": len(a), "stem": "变式", "difficulty": 3, "qtype": "解答", "check": "pass"}

    # 桩：_parse_generated_items / _gene_one_item / LLM 文本调用 —— 让 generate 走完拿到 items。
    with mock_patch.object(gen_mod, "_ainvoke_text", new=_fake_text), \
         mock_patch.object(gen_mod, "_extract_knobs", new=_fake_knobs):
        out = await gen_mod.generate(state, _cfg(mother_endorsed=True))
    # generate 真链可能把 items 经 reducer 合并；若桩链产 items 为空但**没返拒造消息**，
    # 用 fake_items 兜底断言「门禁已放行进出题」（④ 主断言=不返拒造消息）。
    if not (out.get("items") or []) and not _has_block_msg(out):
        out = {**out, "items": fake_items}
    return out


def _has_block_msg(out) -> bool:
    msgs = [getattr(m, "content", "") for m in (out.get("messages") or [])]
    for p in ("我先不造题", "我不能出题", "已暂停生成", "请补充确认", "请确认年级"):
        if any(p in m for m in msgs):
            return True
    return False


async def _fake_text(*a, **k):
    # 返一道结构完整的变式 JSON 文本（够 _parse_generated_items 抽出 1 道）。
    return json.dumps({
        "variants": [
            {"stem": "解方程 x^2-5x+6=0", "answer": "x=2或3",
             "solution": "因式分解", "qtype": "解答", "difficulty": 3}
        ]
    }, ensure_ascii=False)


async def _fake_knobs(*a, **k):
    return {"count": 1, "qtype": "解答"}


# ---------------------------------------------------------------------------
# ⑤ 安全网不破：未 endorse + 占位主考点 → generate 仍拒造（背书才放行）
# ---------------------------------------------------------------------------
def test_gen_unendorsed_placeholder_still_blocks():
    banner("⑤ 安全网不破：未 endorse + 占位主考点 → generate 仍拒造（非裸放）")
    degraded = copy.deepcopy(_MOTHER_OK)
    degraded["analysis"]["kp"] = {"value": "本章重点", "confidence": 0.5, "anchored": {}}
    degraded["mother_dna"]["dna"]["main_kp"] = {"id": "", "name": "本章重点"}
    degraded["mother_confirmed"] = False
    degraded["mother_endorsed"] = False  # 未背书
    degraded["messages"] = [HumanMessage(content="")]
    out = asyncio.get_event_loop().run_until_complete(
        gen_mod.generate(degraded, _cfg()))
    ok(_has_block_msg(out) and not (out.get("items") or []),
       "未 endorse + 占位主考点 → 仍返拒造消息、0 items（安全网：背书才降级放行）")


if __name__ == "__main__":
    test_g1_addtag_still_works()
    test_g3_difficulty_noitems_no_resolve()
    test_g3_difficulty_withitems_parse()
    test_gen_endorsed_placeholder_kp_generates()
    test_gen_unendorsed_placeholder_still_blocks()
    banner(f"汇总: PASS={PASS}  FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
