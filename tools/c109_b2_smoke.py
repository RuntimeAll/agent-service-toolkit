"""PRD-C-109 B2 真机冒烟：intent_triage 工具选择器 + route_after_triage effect 三分支（精测）。

验四件事（离线为主·可断言；补一道在线真分类）：
  ① classify_intent 对多状态话术产对的 {tool, value}（工具准确率）——规则路径离线 + 在线一道。
  ② route_after_triage 按 resolve_tool(tool).effect 走对的分支（即时→await_review / 重写解析·重出→parse /
     开始→generate / 重新解题·改题面→parse / 旋钮→parse / 未知回退 route_entry）。
  ③ 「难度难一点」→ _难度旋钮（旋钮 effect），不进任何母题工具、不重解（AC3）。
  ④ 低置信 → 回退 route_entry（安全网）。
  ⑤ 即时生效(meta)工具经 intent_triage 节点**原地改母题对象一字段**（apply_tool 薄包）+ G5 边界断言：
     改即时维后变式 figure_url 仍在 + len(messages) 不暴涨。

跑法（cwd = toolkit，必走 .venv）：
  $env:NO_PROXY="*"; .venv\\Scripts\\python.exe tools\\c109_b2_smoke.py
  在线段需 :8090 不必、需 .env 的 LLM key（与 A1 spike 同 key/同站）。--offline 跳在线段。
"""
import asyncio
import base64
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import (  # noqa: E402
    classify_intent,
    intent_triage,
    resolve_tool,
    route_after_triage,
    route_entry,
)
from agents.variant.entry import intent as intent_mod  # noqa: E402

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


# 立住的母题（锚定死 + 1 道变式，含 figure_url 验 G5）。
_MOTHER_OK = dict(
    mother_confirmed=True,
    facts_locked=True,
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
            "secondary_kps": [{"id": "3082002002", "name": "韦达定理"}],
            "qtype": "解答", "exam_type": "直接计算", "skeleton": ["因式分解", "求根"],
            "hard_points": ["符号"], "tags": ["解方程"], "scene": "纯代数",
            "models": [{"id": "M00", "name": "概念直用"}], "difficulty": 3, "flags": [],
        },
    },
    facts_audit=[],
)


def _variants(text: str, n: int = 2) -> dict:
    s = {"messages": [HumanMessage(content=text)]}
    s.update(copy.deepcopy(_MOTHER_OK))
    s["items"] = [
        {"stem": f"变式{i}题面", "qtype": "解答", "difficulty": 3, "_seq": i,
         "figure_url": f"https://oss/fig{i}.png", "models": [{"id": "M00", "name": "概念直用"}]}
        for i in range(1, n + 1)
    ]
    return s


def _card(text: str) -> dict:
    s = {"messages": [HumanMessage(content=text)], "awaiting_mother_review": True, "items": []}
    s.update(copy.deepcopy(_MOTHER_OK))
    return s


def _with_tool(state: dict, tool: str, conf: float = 0.9, value=None) -> dict:
    """把 intent_triage 已写好的 tool 决策塞进 state（绕过 LLM，单测 route_after_triage）。"""
    dec = intent_mod._tool_decision(tool, value=value, conf=conf)
    return {**state, "intent_decision": dec}


# ---------------------------------------------------------------------------
# ① classify_intent 规则路径（离线·确定性，验工具准确率不靠网络）
# ---------------------------------------------------------------------------
def test_classify_rule_path():
    banner("① classify_intent 规则路径（_rule_tool 高置信词面，离线确定性）")
    # 🔴 rule 层只硬钉「高危易混」（解法四件套 + 难度 + 开始）；标签/副考点/难点交 LLM 抽干净 value。
    cases = [
        ("重新解一遍这道题", "重新解题"),
        ("换成判别式法模型", "换模型"),
        ("再加个韦达定理模型", "加模型"),
        ("把因式分解那个模型去掉", "删模型"),
        ("解法骨架第二步改成移项", "改解法骨架"),
        ("开始出3道", "开始出变式"),
        ("难度难一点", "_难度旋钮"),
        ("来个简单点的，一颗星", "_难度旋钮"),
        ("难度调高到压轴", "_难度旋钮"),
    ]
    for utt, exp in cases:
        st = _variants(utt)
        dec = asyncio.run(classify_intent(st, _cfg()))
        ok(dec.get("tool") == exp, f"「{utt}」→ tool={dec.get('tool')} 期望 {exp}")
        # effect 由代码查表（LLM/规则不产 effect）
        ok("effect" not in dec, f"「{utt}」: 决策不含 effect（effect 代码查表）")


# ---------------------------------------------------------------------------
# ② route_after_triage：按 resolve_tool(tool).effect 走对的分支
# ---------------------------------------------------------------------------
def test_route_effect_branches():
    banner("② route_after_triage effect 三分支 + 执行/旋钮/未知回退")
    # 即时生效(meta) + 有题组 → await_review（mutator 已在节点改完，刷母题卡）
    st = _variants("加个易错标签")
    ok(route_after_triage(_with_tool(st, "加标签"), _cfg()) == "await_review",
       "加标签(即时生效·有题组) → await_review")
    # 🔴 PRD-C-109 fix·即时生效 + 无题组（母题卡就绪态=举一反三主场景）→ await_review
    #   （节点已用 edit_mother_dna_meta 去 item 化原地改完），**绝不落 parse 母题全量重解**。
    #   旧实现「无题组→parse」是根因 bug：标签没加上 + parse 把就绪卡当母题重锚打回存疑 + 反复弹确认。
    cst = _card("加个易错标签")
    ok(route_after_triage(_with_tool(cst, "加标签"), _cfg()) == "await_review",
       "加标签(即时生效·无题组) → await_review（去 item 化原地改、不落 parse 重解·C-109 fix）")
    # 重写解析（换模型）→ parse
    ok(route_after_triage(_with_tool(_variants("换成判别式法模型"), "换模型"), _cfg()) == "parse",
       "换模型(重写解析) → parse")
    # 重出本题（set_主考点）→ parse
    ok(route_after_triage(_with_tool(_variants("主考点改成根的判别式"), "set_主考点"), _cfg()) == "parse",
       "set_主考点(重出本题) → parse")
    # 重出本题·hard_anchor（set_年级章）→ parse
    ok(route_after_triage(_with_tool(_variants("年级改成八下"), "set_年级章"), _cfg()) == "parse",
       "set_年级章(重出·hard_anchor) → parse")
    # 执行·开始出变式 + 母题立住 + 无题组 → generate
    ok(route_after_triage(_with_tool(_card("开始"), "开始出变式"), _cfg()) == "generate",
       "开始出变式(母题立住·无题组) → generate")
    # 执行·开始 + 母题存疑 → await_review（AC2 不抢跑）
    doubt = {"messages": [HumanMessage(content="开始")], "awaiting_mother_review": True,
             "items": [], "analysis": {"grade": {"value": "?", "confidence": 0.2}},
             "mother_dna": {"stem": "s"}}
    ok(route_after_triage(_with_tool(doubt, "开始出变式"), _cfg()) == "await_review",
       "开始出变式(母题存疑) → await_review（AC2）")
    # 执行·重新解题 → parse
    ok(route_after_triage(_with_tool(_variants("重新解一遍"), "重新解题"), _cfg()) == "parse",
       "重新解题(执行) → parse")
    # 执行·改题面 → parse
    ok(route_after_triage(_with_tool(_variants("题面排版乱了重排"), "改题面"), _cfg()) == "parse",
       "改题面(执行·重排版) → parse")
    # 未知工具 → 回退 route_entry（决策置 conf 高但 tool 不识别：模拟 route 层兜底）
    st_unknown = {**_variants("乱说"), "intent_decision":
                  {"intent": "答疑", "tool": "不存在", "tool_value": None, "confidence": 0.9,
                   "correction": {"field": None, "value": None},
                   "edit": {"target_seq": None, "action": None}, "count": None}}
    ok(route_after_triage(st_unknown, _cfg()) == route_entry(st_unknown, _cfg()),
       "未知工具 → 回退 route_entry（安全网）")


# ---------------------------------------------------------------------------
# ③ 难度 → 旋钮，不进母题工具、不重解（AC3）
# ---------------------------------------------------------------------------
def test_difficulty_to_knob():
    banner("③ 「难度」→ _难度旋钮（旋钮 effect）→ parse 变式旋钮，不碰母题（AC3）")
    st = _variants("难度难一点")
    dec = asyncio.run(classify_intent(st, _cfg()))
    ok(dec.get("tool") == "_难度旋钮", f"「难度难一点」→ tool={dec.get('tool')}（旋钮，非母题工具）")
    spec = resolve_tool("_难度旋钮")
    ok(spec is not None and spec["effect"] == "旋钮", "_难度旋钮 effect=旋钮")
    ok(spec["fn"] is None and spec["dna_field"] is None, "_难度旋钮 不改任何母题维（fn/field=None·只读）")
    ok(route_after_triage(_with_tool(st, "_难度旋钮"), _cfg()) == "parse",
       "_难度旋钮 → parse（变式难度旋钮 stage-2，不重解母题）")


# ---------------------------------------------------------------------------
# ④ 低置信 → 回退 route_entry
# ---------------------------------------------------------------------------
def test_low_conf_fallback():
    banner("④ 低置信 → 回退 route_entry（安全网）")
    st = _variants("换成判别式法模型")
    low = route_after_triage(_with_tool(st, "换模型", conf=0.3), _cfg())
    ok(low == route_entry(st, _cfg()), f"低置信(conf=0.3) → 回退 route_entry（得 {low}）")


# ---------------------------------------------------------------------------
# ⑤ intent_triage 节点：即时生效(meta)工具原地改母题对象一字段 + G5 边界断言
# ---------------------------------------------------------------------------
def test_node_meta_inplace_and_g5():
    banner("⑤ intent_triage 节点：meta 工具原地改一字段 + G5（figure_url 仍在 / messages 不暴涨）")
    st = _variants("加个易错标签")
    fig_before = [it.get("figure_url") for it in st["items"]]

    # mock 工具选择器：LLM 产干净 value="易错"（meta 工具在节点直改字段，需干净值；离线确定性）。
    async def _fake_classify(state, config):
        return intent_mod._tool_decision("加标签", value="易错", conf=0.95)

    _orig = intent_mod.classify_intent
    intent_mod.classify_intent = _fake_classify
    try:
        out = asyncio.run(intent_triage(st, _cfg()))
    finally:
        intent_mod.classify_intent = _orig
    dec = out.get("intent_decision") or {}
    ok(dec.get("tool") == "加标签", f"节点选中 tool={dec.get('tool')}（加标签）")
    # 母题对象原地改：tags 含新标签 + 旧标签保留
    new_tags = (out.get("mother_dna", {}).get("dna", {}).get("tags") or [])
    ok("易错" in new_tags, f"加标签: 母题 tags 原地加（{new_tags}）")
    ok("解方程" in new_tags, "加标签: 旧标签保留（原地改非整组清）")
    ok(dec.get("_applied") is True, "加标签: intent_decision 标 _applied（节点已应用）")
    # G5① 即时维不波及变式 stem → figure_url 仍在
    fig_after = [it.get("figure_url") for it in (out.get("items") or [])]
    ok(fig_after == fig_before and all(fig_after),
       f"G5①: 改即时维后变式 figure_url 仍在（{fig_after}）")
    # G5② messages 不因 meta 编辑暴涨（节点返回 messages:[] 增量；不重解不重出）
    ok(out.get("messages") == [], "G5②: intent_triage 返回 messages 增量为空（不累积、不重解）")
    # scene 别维不变
    ok((out.get("mother_dna", {}).get("dna", {}).get("scene")) == "纯代数",
       "加标签: scene 别维不变（只改一字段）")


# ---------------------------------------------------------------------------
# ⑥ 在线一道真分类（验 prompt 落生产 + opus 真选工具）。--offline 跳过。
# ---------------------------------------------------------------------------
def test_online_one():
    if "--offline" in sys.argv:
        print("\n(跳过在线段：--offline)")
        return
    banner("⑥ 在线真分类一道（opus@sui-xiang，验生产 prompt 通）")
    import os
    os.environ["NO_PROXY"] = "*"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    st = _variants("把主考点改成根的判别式")
    try:
        dec = asyncio.run(classify_intent(st, _cfg()))
        ok(dec.get("tool") == "set_主考点",
           f"在线:「把主考点改成根的判别式」→ tool={dec.get('tool')} conf={dec.get('confidence')}")
    except Exception as e:  # noqa: BLE001
        print(f"  !!  在线段异常（网络/key）：{str(e)[:120]}")


if __name__ == "__main__":
    test_classify_rule_path()
    test_route_effect_branches()
    test_difficulty_to_knob()
    test_low_conf_fallback()
    test_node_meta_inplace_and_g5()
    test_online_one()
    banner(f"汇总: PASS={PASS}  FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
