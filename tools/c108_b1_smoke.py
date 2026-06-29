"""PRD-C-108 B1·薄意图层 smoke（A1 spike + ≤4 道精测）。

跑法（cwd = toolkit，必 .venv）：
  在线（含 A1 LLM 分类准确率）：set NO_PROXY=* & .venv\\Scripts\\python.exe tools\\c108_b1_smoke.py
  离线（只跑路由断言 + 规则兜底，不调 LLM）：... tools\\c108_b1_smoke.py --offline

覆盖：
- A1 spike：~12 条多状态话术 → classify_intent 准确率（在线）；规则兜底命中（离线/在线都跑）。
- G1：意图分类多状态（确认/调整/开始/编辑/答疑/新任务）判对。
- G2：母题卡态不点开始/不明说 → 不进 generate；母题存疑 → await_review（停确认）。
- G3：生成后说改解法/重解 → route_after_triage 回 parse（重锚链入口），不进变式编辑器。
- 安全网：低置信意图 → route_after_triage 回退原 route_entry 分诊。
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys

sys.path.insert(0, "src")

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import (  # noqa: E402
    classify_intent,
    mother_in_doubt,
    route_after_triage,
    route_entry,
    route_entry_v2,
)
from agents.variant.entry.intent import _rule_intent  # noqa: E402

OFFLINE = "--offline" in sys.argv


def _tok(uid: int = 5) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"userId": uid}).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


def _cfg(**extra) -> dict:
    conf = {"ruoyi_token": _tok()}
    conf.update(extra)
    return {"configurable": conf}


# 一个"立住的母题"（锚定死 + 有解法骨架）。注意 analysis 要带 code + anchored 才 pinned。
_MOTHER_OK = dict(
    analysis={
        "grade": {"value": "八年级下学期", "confidence": 0.9, "code": "3082"},
        "kp": {"value": "一元二次方程", "confidence": 0.9, "anchored": {"code": "3082002001", "name": "一元二次方程"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    mother_dna={"stem": "x^2-3x+2=0", "answer": "x=1 或 x=2", "solution_skeleton": ["因式分解", "求根"]},
    mother_confirmed=True,
)


def _state(text: str, **kw) -> dict:
    base: dict = {"messages": [HumanMessage(content=text)]}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# A1 spike·~12 条多状态话术（state 上下文 + 期望意图）
# ---------------------------------------------------------------------------
def _mother_card_state(text: str) -> dict:
    # 母题卡态（已立住，等老师确认/开始）
    s = _state(text, awaiting_mother_review=True, items=[], **_MOTHER_OK)
    return s


def _variants_state(text: str, n: int = 3) -> dict:
    s = _state(text, items=[{"stem": f"q{i}"} for i in range(n)], **_MOTHER_OK)
    return s


# (label, state, utterance, expected_intent)
A1_CASES = [
    ("母题卡·确认章", _mother_card_state, "对，就是八年级下的", "确认范围"),
    ("母题卡·改解法", _mother_card_state, "按判别式法解一下", "调整母题"),
    ("母题卡·重解", _mother_card_state, "重新解一遍这道题", "调整母题"),
    ("母题卡·开始", _mother_card_state, "可以了，开始出3道", "开始出题"),
    ("母题卡·改考点", _mother_card_state, "其实主考点应该是二次函数", "调整母题"),
    ("出题后·改解法", _variants_state, "解析别用因式分解，改用配方法", "调整母题"),
    ("出题后·重解", _variants_state, "母题重新解一下", "调整母题"),
    ("出题后·编辑删", _variants_state, "第2题删掉", "编辑变式"),
    ("出题后·编辑难", _variants_state, "第1题难一点", "编辑变式"),
    ("出题后·再来N道", _variants_state, "再来2道难的", "编辑变式"),
    ("出题后·答疑", _variants_state, "第3题怎么解？给学生讲讲", "答疑"),
    ("出题后·开始(易混)", _variants_state, "出吧", "开始出题"),
]


async def run_a1_spike() -> tuple[int, int, list[str]]:
    print("\n=== A1 spike·意图分类准确率（%s）===" % ("离线·仅规则" if OFFLINE else "在线·LLM"))
    ok = 0
    misses: list[str] = []
    for label, mk, utt, expect in A1_CASES:
        st = mk(utt)
        if OFFLINE:
            ruled = _rule_intent(utt, st)
            if ruled is None:
                # 离线无 LLM：规则没命中的留作 SKIP（不计入分母失败，只看规则覆盖的易混区）
                print(f"  SKIP(规则未命中,留LLM判) [{label}] {utt!r}")
                continue
            got = ruled["intent"]
            total_label = "规则"
        else:
            dec = await classify_intent(st, _cfg())
            got = dec.get("intent")
            total_label = f"conf={dec.get('confidence')}"
        hit = got == expect
        ok += 1 if hit else 0
        mark = "OK " if hit else "MISS"
        if not hit:
            misses.append(f"[{label}] {utt!r} 期望 {expect} 得 {got}")
        print(f"  {mark} [{label}] {utt!r} → {got} (期望 {expect}; {total_label})")
    denom = ok + len(misses)
    print(f"  >>> 命中 {ok}/{denom}")
    return ok, denom, misses


# ---------------------------------------------------------------------------
# G1/G2/G3 + 安全网（纯路由断言，离线可跑，mock intent_decision）
# ---------------------------------------------------------------------------
def _with_intent(state: dict, intent: str, conf: float = 0.9, **kw) -> dict:
    dec = {"intent": intent, "confidence": conf,
           "correction": {"field": None, "value": None},
           "edit": {"target_seq": None, "action": None}, "count": None}
    dec.update(kw)
    return {**state, "intent_decision": dec}


def run_routing_asserts() -> list[str]:
    print("\n=== G1/G2/G3 + 安全网·路由断言（离线 mock 分诊）===")
    fails: list[str] = []

    def check(name: str, got, expect):
        ok = got == expect
        print(f"  {'OK ' if ok else 'FAIL'} {name}: {got} (期望 {expect})")
        if not ok:
            fails.append(f"{name}: got {got} expect {expect}")

    # --- G2·不自动开始：母题卡态、未点按钮、未明说 → route_entry_v2 进 intent_triage（交 LLM）
    st_card = _mother_card_state("嗯嗯")
    check("G2·母题卡纯文字→意图层", route_entry_v2(st_card, _cfg()), "intent_triage")

    # --- G2·按钮路径不进意图层（结构化 start_variants → 原 route_entry 直奔 generate）
    check("G2·按钮 start_variants 不进意图层",
          route_entry_v2(st_card, _cfg(start_variants=True)), "generate")

    # --- G2·母题存疑 + 明说开始 → await_review（停确认，不冲 generate）
    doubt = _state("开始出3道", awaiting_mother_review=True, items=[],
                   analysis={"grade": {"value": "?", "confidence": 0.2}},
                   mother_dna={"stem": "s"})  # 无 anchored/无 skeleton → 存疑
    check("G2·母题存疑判定", mother_in_doubt(doubt), True)
    check("G2·存疑+开始→停确认",
          route_after_triage(_with_intent(doubt, "开始出题"), _cfg()), "await_review")

    # --- G1/G2·母题立住 + 开始 + 未出题 → generate
    start_ok = _mother_card_state("开始")
    check("G1·立住+开始→generate",
          route_after_triage(_with_intent(start_ok, "开始出题"), _cfg()), "generate")

    # --- G3·生成后改解法/重解 → parse（重锚链入口），不进变式编辑器(dispatch)
    after_gen = _variants_state("重新解一下")
    check("G3·生成后调整母题→parse",
          route_after_triage(_with_intent(after_gen, "调整母题"), _cfg()), "parse")

    # --- G1·编辑变式（有题组）→ parse
    check("G1·编辑变式→parse",
          route_after_triage(_with_intent(after_gen, "编辑变式"), _cfg()), "parse")

    # --- 安全网·低置信 → 回退原 route_entry（此 state 下 route_entry 给 parse）
    low = route_after_triage(_with_intent(after_gen, "调整母题", conf=0.2), _cfg())
    base = route_entry(after_gen, _cfg())
    check("安全网·低置信回退=route_entry", low, base)

    # --- 安全网·无 intent_decision → 回退 route_entry
    check("安全网·无分诊回退=route_entry",
          route_after_triage(after_gen, _cfg()), route_entry(after_gen, _cfg()))

    return fails


async def main():
    fails = run_routing_asserts()
    a_ok, a_denom, a_miss = await run_a1_spike()
    print("\n=== 汇总 ===")
    print(f"路由断言失败: {len(fails)}")
    for f in fails:
        print("  -", f)
    print(f"A1 spike: {a_ok}/{a_denom}" + (" (离线·部分SKIP)" if OFFLINE else ""))
    for m in a_miss:
        print("  MISS", m)
    bad = len(fails) + (0 if (a_denom and a_ok / a_denom >= 0.83) else 1)
    print("\nRESULT:", "GREEN" if bad == 0 else f"RED (issues={bad})")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
