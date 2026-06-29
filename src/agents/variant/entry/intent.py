"""variant 引擎 · entry/intent.py 意图层（PRD-C-108 B1 薄意图分诊 → PRD-C-109 B2 工具选择器）。

把「绑死状态位的握手」换成「LLM 选工具 + 确定性 effect 路由」——罩在 route_entry 之上，**不推倒重写**：
- C-108：薄 LLM 意图分诊（6 类）→ 代码映射现有节点；低置信 → 回退 route_entry（安全网）。
- 🔴 C-109 B2：意图层升级成「母题编辑·工具选择器」（15 工具 + 难度旋钮特例）。
  LLM 只产 `{tool, value}`，**绝不产 effect**；effect 由代码 `resolve_tool(tool).effect` 查表
  （§11 铁律）。出口从 if 链塌成「effect 三分支」（即时生效 / 重写解析 / 重出本题）+ 执行/旋钮特例。
  - 即时生效（meta）：mutator 在 intent_triage **节点内**原地改母题对象一字段 → 刷母题卡（await_review）。
  - 重写解析 / 重出本题：→ parse（既有 patch→重锚/重解链，复用现成重生机器，不在本层重写）。
  - 执行类：开始出变式 → generate；重新解题/改题面 → parse。
  - 旋钮（难度只读·表驱动）：→ parse（变式难度旋钮=stage-2，不碰母题、不重解）。
  - 未知工具 / 低置信 → 回退原 route_entry 代码分诊（安全网，历史补丁一条不丢）。

🔴 单步、非自决循环（§11 铁律）：一个老师 turn = LLM 选**一个**工具 → 执行 → →END 等下一 turn。
   **绝不用 ToolNode / agent loop**、绝不让 LLM 决定「再调一个工具 / 循环继续」。延续 →END+resume。
🔴 LLM 只选工具，effect 代码查表，控制流仍确定性 DAG（守 CLAUDE.md §4）；re-wire 不 rebuild
   （route_entry 函数体零改，意图层只在前面加；图拓扑不变，intent_triage 是既有节点，扩 prompt+出口）。
🔴 难度只读·表驱动，绝不 LLM 自评/开放编辑（_难度旋钮 桩条目仅作路由口径）。
🔴 strangler：顶部 from agents.variant import 取依赖（运行期解析，本模块在 __init__ 末尾导入）。

🟢 向后兼容（C-108 26 单测 + 既有 parse 链）：intent_decision 仍带 legacy `intent`（由工具派生），
   route_after_triage 既能吃新 `tool`（按 effect 路由），也能吃纯 legacy `intent`（旧 if 链兜底）。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、route + tool_registry re-export 之后导入）
    EFFECT_EXEC,
    EFFECT_IMMEDIATE,
    EFFECT_KNOB,
    EFFECT_REGEN,
    EFFECT_REWRITE,
    VariantState,
    _ainvoke_text,
    _editor_op,
    _entry_read_lowconf,
    _extract_image_url,
    _latest_ai_text,
    _latest_human_text,
    _parse_json,
    _pin_status,
    apply_tool,
    conv_trace,
    resolve_tool,
    route_entry,
    settings,
)
from agents.variant.prompts import TOOL_SELECT_PROMPT  # noqa: E402

# 意图枚举（C-108·§10 intent spec；仍是 route_after_triage legacy 路径 + parse 链口径）。
INTENT_CONFIRM_SCOPE = "确认范围"
INTENT_ADJUST_MOTHER = "调整母题"
INTENT_START = "开始出题"
INTENT_EDIT_VARIANT = "编辑变式"
INTENT_QA_TRIAGE = "答疑"
INTENT_NEW_TASK = "新任务"
_INTENT_ENUM = frozenset(
    {
        INTENT_CONFIRM_SCOPE,
        INTENT_ADJUST_MOTHER,
        INTENT_START,
        INTENT_EDIT_VARIANT,
        INTENT_QA_TRIAGE,
        INTENT_NEW_TASK,
    }
)

# 🔴 意图层置信闸（A1 spike 调过）：< 此值 → 回退原 route_entry 代码分诊（安全网）。
INTENT_CONF_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# 🔴 PRD-C-109 B2·工具 → legacy intent 映射（双向桥）：
#   ① 给 route_after_triage 的 legacy 路径 + parse 链一个口径（工具选不中也能落对的下游）；
#   ② 让 C-108 26 单测（纯 legacy intent 决策）继续绿（新 tool 字段缺省时走旧 if 链）。
#   effect **不**从这里推（effect 永远查 resolve_tool）；本表只管「派到现有节点」的 legacy 语义。
# ---------------------------------------------------------------------------
_TOOL_TO_INTENT: dict[str, str] = {
    # 即时生效（meta）：节点内原地改一字段后停母题卡（确认范围语义=停 await_review）。
    "选副考点": INTENT_CONFIRM_SCOPE,
    "set_难点": INTENT_CONFIRM_SCOPE,
    "加标签": INTENT_CONFIRM_SCOPE,
    "删标签": INTENT_CONFIRM_SCOPE,
    # 重写解析 / 重出本题：回 stage-1 重锚/重解链（既有 parse「调整母题」分诊）。
    "改解法骨架": INTENT_ADJUST_MOTHER,
    "加模型": INTENT_ADJUST_MOTHER,
    "删模型": INTENT_ADJUST_MOTHER,
    "换模型": INTENT_ADJUST_MOTHER,
    "set_主考点": INTENT_ADJUST_MOTHER,
    "set_题型": INTENT_ADJUST_MOTHER,
    "set_考察类型": INTENT_ADJUST_MOTHER,
    "set_场景": INTENT_ADJUST_MOTHER,
    "set_年级章": INTENT_ADJUST_MOTHER,
    "改题面": INTENT_ADJUST_MOTHER,
    # 执行类
    "重新解题": INTENT_ADJUST_MOTHER,  # resolve 语义（既有 parse 重解链）
    "开始出变式": INTENT_START,
    # 旋钮（难度只读）：落 parse（变式难度旋钮=stage-2 编辑流水线）。
    "_难度旋钮": INTENT_EDIT_VARIANT,
}


# ---------------------------------------------------------------------------
# 母题存疑（确定性读 state，无 LLM）：解法骨架未出 / 题型 AI 自己存疑 / 锚定未定死（待人审）
# → 即便老师点了「开始」也先停在母题卡解决疑点，绝不抢跑 generate（AC2）。
# 🔴 PRD-C-109 A3：mother_endorsed（老师明确背书）override —— B3 接确认收口（本卡先留闸）。
# ---------------------------------------------------------------------------
def mother_in_doubt(state: VariantState) -> bool:
    """母题是否「存疑」（停等确认的判据，确定性、可单测）。任一为真即存疑：
    ① 锚未定死（_pin_status.pinned=False：年级/主考点/置信缺任一）；
    ② 解法骨架未出（mother_dna.solution_skeleton / answer 皆空）；
    ③ 读图极低置信 / 章未判出（_entry_read_lowconf）。
    无 mother_dna（还没解出母题）→ 视为存疑（更不该冲生成）。
    🔴 mother_endorsed（老师背书）= 终态 override（用户确认 > 代码硬锚，PRD-C-109 §11）。"""
    # 🔴 用户确认(背书) > 代码硬锚：老师明确确认即终，不再被锚定/骨架细节拦（B3 置位，本卡先生效）。
    if state.get("mother_endorsed"):
        return False
    mdna = state.get("mother_dna")
    if not isinstance(mdna, dict) or not mdna:
        return True
    # ① 锚定未定死
    if not _pin_status(state).get("pinned"):
        return True
    # ② 解法骨架未出（解题没成型 = 母题没立住）
    skeleton = mdna.get("solution_skeleton") or mdna.get("answer") or ""
    if not str(skeleton).strip():
        return True
    # ③ 读图极低置信 / 章未判出
    if _entry_read_lowconf(state):
        return True
    return False


# ---------------------------------------------------------------------------
# 规则兜底（LLM 失败 / 想加硬规则压易混区时的确定性护栏）。
#   🔴 PRD-C-109 B2：从「6 意图」扩到「15 工具」——只在「极有把握」的高危易混词面命中时返回 tool 决策，
#   否则 None（交给 LLM / 回退）。把 A1 spike 暴露的高危易混（换/加/删模型↔改解法↔重解；主↔副；难度↔题型）
#   词面硬钉死，LLM 失败也不退化。
# ---------------------------------------------------------------------------
# —— 解法四件套（A1 易混核心：四个都关解法，工具不同，硬区分）——
_RESOLVE_HINTS = ("重新解", "再解一遍", "再解一次", "重解", "重新做", "重新算", "再解一下")
_MODEL_REPLACE_HINTS = ("换模型", "换成", "换个模型", "模型换", "替换成", "替换模型")
_MODEL_ADD_HINTS = ("加模型", "加个模型", "再加个", "补一个模型", "补个模型", "增加模型", "多加")
_MODEL_DEL_HINTS = ("删模型", "去掉", "删掉", "删除模型", "移除模型", "把那个模型")
_SKELETON_HINTS = ("解法骨架", "解题步骤", "解题骨架", "步骤改", "这一步", "第二步", "第几步", "骨架")
# —— 难度（只读旋钮，绝不进母题工具；与题型/考察类型硬区分）——
_DIFFICULTY_HINTS = ("难度", "难一点", "简单点", "简单些", "难一些", "一颗星", "两颗星", "三颗星",
                     "调高难度", "调低难度", "压轴", "送分", "难点的", "简单的")
# —— 开始（执行）——
_START_HINTS = ("开始举一反三", "开始出", "开始生成", "出吧", "可以了开始", "生成变式", "开始变式")


def _rule_tool(utterance: str, state: VariantState) -> dict[str, Any] | None:
    """🔴 高置信词面工具规则（只在十拿九稳时命中）。返回 tool 决策（含 legacy intent）或 None。
    覆盖 A1 暴露的高危易混工具；命中即不退化（LLM 失败也守得住）。顺序：先难度旋钮（防被题型吞）→
    解法四件套（重解>换模型>加模型>删模型>骨架，长词面优先）→ 标签 → 开始。"""
    u = (utterance or "").strip()
    if not u:
        return None

    # ① 难度（只读旋钮）最先判 —— 防「难度调高」被后面的题型/考察类型规则误吃。
    #    排除「难点 set_难点」（"难点"≠"难度"）+「标签」（"把压轴标签删了" 是标签操作非难度，
    #    "压轴/送分" 在难度词面里但语义是标签值）→ 让 LLM 判。
    if "难点" not in u and "标签" not in u and any(h in u for h in _DIFFICULTY_HINTS):
        return _tool_decision("_难度旋钮", value=u, conf=0.9)

    # ② 解法四件套（最易混，硬钉）。重解 > 换 > 加 > 删 > 骨架（语义特异度递减）。
    #    🔴 解法四件套是 A1 暴露的高危易混核心，rule 层硬钉死（LLM 失败也守得住）；
    #    value=u（粗值）只作 parse 链口径——这四类工具 route 到 parse，由 parse 再解析细节，
    #    不在 intent_triage 节点直改字段（meta 才在节点改），故粗 value 不污染对象。
    if any(h in u for h in _RESOLVE_HINTS):
        return _tool_decision("重新解题", value="按老师要求重新解题", conf=0.92)
    if any(h in u for h in _MODEL_REPLACE_HINTS):
        return _tool_decision("换模型", value=u, conf=0.88)
    if any(h in u for h in _MODEL_ADD_HINTS):
        return _tool_decision("加模型", value=u, conf=0.85)
    if any(h in u for h in _MODEL_DEL_HINTS) and "模型" in u:
        return _tool_decision("删模型", value=u, conf=0.85)
    if any(h in u for h in _SKELETON_HINTS):
        return _tool_decision("改解法骨架", value=u, conf=0.82)

    # ③ 开始出变式（执行·高频显式触发，rule 硬钉）。
    if any(h in u for h in _START_HINTS):
        return _tool_decision("开始出变式", value=None, conf=0.9)

    # 🔴 标签/副考点/难点（即时生效·meta）**不在 rule 层命中** —— 它们要在 intent_triage 节点
    #    原地改字段，value 必须是干净的标签/难点值（"易错"），而 rule 层只有粗 utterance、抽不准。
    #    A1 spike 证 meta 工具 LLM 100% 命中且能抽干净 value，故交 LLM（产 {tool, value}）。
    return None


# —— C-108 legacy 词面规则（指定/纠正解法 → 调整母题/solve；C-108 单测口径，向后兼容保留）——
_SOLVE_METHOD_HINTS = ("判别式", "配方法", "因式分解", "换个解法", "换种解法", "用韦达", "别用", "不用")


def _rule_intent(utterance: str, state: VariantState) -> dict[str, Any] | None:
    """🟢 C-108 legacy 词面规则（向后兼容 shim，保留给 C-108 26 单测 + legacy 调用方）。
    返回**legacy 6-意图**决策（无 tool 字段，走 route_after_triage 旧 if 链）：
      重解类 → 调整母题/resolve（conf 0.92）；指定解法类 → 调整母题/solve（conf 0.85）；否则 None。
    🔴 B2 生产路径用 _rule_tool（产 tool 决策）；本 shim 不参与 classify_intent，仅供旧测/旧引用。"""
    u = (utterance or "").strip()
    if not u:
        return None
    if any(h in u for h in _RESOLVE_HINTS):
        return _decision(INTENT_ADJUST_MOTHER, conf=0.92,
                         correction={"field": "resolve", "value": "按老师要求重新解题"})
    if any(h in u for h in _SOLVE_METHOD_HINTS):
        return _decision(INTENT_ADJUST_MOTHER, conf=0.85,
                         correction={"field": "solve", "value": u})
    return None


def _tool_decision(
    tool: str,
    value: Any = None,
    conf: float = 0.0,
) -> dict[str, Any]:
    """🔴 工具决策 = LLM 只产 {tool, value, confidence}；effect 由代码查表（不在此带 effect）。
    同时派生 legacy `intent` + correction/edit（向后兼容 route_after_triage legacy 路径 + parse 链 + C-108 单测）。
    """
    intent = _TOOL_TO_INTENT.get(tool, INTENT_QA_TRIAGE)
    # legacy correction/edit：只为 parse 链/旧测提供口径；细节由 parse 再解析（intent_decision 只是路由信号）。
    correction: dict[str, Any] = {"field": None, "value": None}
    edit: dict[str, Any] = {"target_seq": None, "action": None}
    if tool == "重新解题":
        correction = {"field": "resolve", "value": value or "按老师要求重新解题"}
    elif intent == INTENT_ADJUST_MOTHER:
        correction = {"field": "solve" if tool in ("换模型", "加模型", "删模型", "改解法骨架")
                      else "kp", "value": value}
    return {
        "intent": intent,
        "tool": tool,
        "tool_value": value,
        "correction": correction,
        "edit": edit,
        "count": None,
        "confidence": float(conf),
    }


def _decision(
    intent: str,
    conf: float,
    correction: dict[str, Any] | None = None,
    edit: dict[str, Any] | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    """legacy 6-意图决策（C-108 形态，无 tool 字段；route_after_triage 走旧 if 链兜底）。"""
    return {
        "intent": intent,
        "tool": None,
        "tool_value": None,
        "correction": correction or {"field": None, "value": None},
        "edit": edit or {"target_seq": None, "action": None},
        "count": count,
        "confidence": float(conf),
    }


def _state_summary(state: VariantState) -> str:
    """给意图 prompt 的当前状态摘要（人话短句，让 LLM 知道老师处在哪一步）。"""
    items = state.get("items") or []
    mdna = state.get("mother_dna") if isinstance(state.get("mother_dna"), dict) else {}
    bits: list[str] = []
    if state.get("awaiting_mother_confirm"):
        bits.append("正在等老师确认母题的年级/章（母题卡前，锚定待定）")
    elif state.get("awaiting_mother_review"):
        bits.append("母题卡已出、还没生成变式，等老师确认母题或点「开始举一反三」")
    elif items:
        bits.append(f"已经出了 {len(items)} 道变式题，老师在看这组题")
    elif mdna:
        bits.append("母题已在手、还没出变式")
    else:
        bits.append("刚开始，还没有母题")
    if mother_in_doubt(state) and (mdna or state.get("awaiting_mother_review")):
        bits.append("（母题仍有存疑：解法/题型/锚定未完全定死）")
    return "；".join(bits)


async def classify_intent(state: VariantState, config: RunnableConfig) -> dict[str, Any]:
    """🔴 PRD-C-109 B2·母题编辑「工具选择器」（节点内有界 LLM·低温；规则兜底优先命中）。
    LLM 只产 `{tool, value, confidence}`（照 A1 spike prompt），代码派生 legacy intent；
    任何异常 / 工具不识别 → 低置信回退（intent=答疑·conf=0），上层 route_after_triage 回退 route_entry。
    🔴 LLM 不产 effect（effect 由 route_after_triage 查 resolve_tool 得出，§11 铁律）。"""
    utterance = _latest_human_text(state.get("messages", []))
    # 高置信词面规则优先（压住 A1 易混区：解法四件套/难度/标签）。
    ruled = _rule_tool(utterance, state)
    if ruled is not None:
        return ruled
    prev_ai = _latest_ai_text(state.get("messages", []))
    if len(prev_ai) > 600:
        prev_ai = prev_ai[:600] + "…"
    prompt = TOOL_SELECT_PROMPT.format(
        state_summary=_state_summary(state),
        prev_ai=prev_ai or "(无)",
        utterance=utterance or "(空)",
    )
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)],
            model=settings.LLM_MODEL_LIGHT,  # opus（A1 spike 100%/0%）/ 受约束分类器
            temperature=0.1,
            trace_label="tool_select",
        )
        parsed = _parse_json(text)
    except Exception:  # noqa: BLE001  分类失败绝不卡死 → 低置信回退
        return _decision(INTENT_QA_TRIAGE, conf=0.0)
    if not isinstance(parsed, dict):
        return _decision(INTENT_QA_TRIAGE, conf=0.0)
    tool = parsed.get("tool")
    # 🔴 工具不在注册表（resolve_tool=None）→ 低置信回退（B2 不强行映射未知工具）。
    if not isinstance(tool, str) or resolve_tool(tool) is None:
        return _decision(INTENT_QA_TRIAGE, conf=0.0)
    try:
        conf = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    dec = _tool_decision(tool, value=parsed.get("value"), conf=conf)
    return dec


# ---------------------------------------------------------------------------
# intent_triage（节点）：跑工具选择器 + 即时生效（meta）类工具**节点内原地改母题对象一字段**。
#   🔴 单步非自决：本节点只选一个工具 + 至多改一字段，路由在 route_after_triage（确定性边）。
#   meta 工具（标签/副考点/难点）= 纯元数据、不重跑 → 在此应用 mutator 即时改对象（apply_tool 薄包
#   edit_dna_state），然后 route_after_triage 路由到 await_review 刷母题卡。
#   非 meta 工具（重写解析/重出本题/执行/旋钮）**不在此改字段**——交 parse/generate 既有重生机器
#   （复用现成 edit_dna_state/regen_dirty_items，本层零重写）。
# ---------------------------------------------------------------------------
def _meta_apply_index(state: VariantState) -> int | None:
    """meta 工具原地改母题对象时 edit_dna_state 需要 1-based item index（它写守恒维进 mother_dna）。
    有题组 → 用第 1 道（守恒维改动对整组生效，index 只决定哪道被标 manual）；
    无题组（纯母题卡阶段）→ None（edit_dna_state 要求 index≥1；此时不在节点改，交 parse 链）。"""
    items = state.get("items") or []
    return 1 if items else None


async def intent_triage(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 PRD-C-109 B2·意图层节点：跑工具选择器写 state['intent_decision']；
    即时生效（meta）类工具 + 有题组 → 节点内原地改母题对象一字段（apply_tool 薄包 edit_dna_state）。
    其余工具不在此改业务态（路由到 parse/generate 既有重生机器）。本节点是图里唯一「意图层」节点（拓扑不变）。
    """
    decision = await classify_intent(state, config)
    update: VariantState = {"intent_decision": decision, "messages": []}

    tool = decision.get("tool")
    spec = resolve_tool(tool) if isinstance(tool, str) else None
    try:
        conf = float(decision.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    # 🔴 即时生效（meta）类 + 高置信 + 有题组 → 节点内原地改一字段（不重跑、不清组、不波及变式 stem）。
    if (
        spec is not None
        and spec.get("effect") == EFFECT_IMMEDIATE
        and conf >= INTENT_CONF_THRESHOLD
    ):
        idx = _meta_apply_index(state)
        if idx is not None:
            mut_update, _edited, err = apply_tool(
                tool, state, idx, decision.get("tool_value")
            )
            if err is None and mut_update:
                # 原地合并母题对象字段改动（edit_dna_state 返回的 partial state：
                #   mother_dna/items/analysis/facts_audit 等），绝不整组清空（§11 状态原地改）。
                update.update(mut_update)
                update["intent_decision"] = {**decision, "_applied": True}
    return update


# ---------------------------------------------------------------------------
# route_after_triage：🔴 PRD-C-109 B2·出口从「6 意图 if 链」塌成「effect 三分支」（确定性）。
#   有 tool（resolve_tool 命中）→ 按 spec.effect 走 3 条确定性边 + 执行/旋钮特例：
#     即时生效(meta) → await_review（mutator 已在节点改完，刷母题卡）
#     重写解析 / 重出本题 → parse（既有 patch→重锚/重解链，复用现成重生机器）
#     执行：开始出变式 → generate（母题存疑则 await_review·AC2）；重新解题/改题面 → parse
#     旋钮（难度只读）→ parse（变式难度旋钮 stage-2，不碰母题）
#   无 tool（纯 legacy intent，如 C-108 单测）→ 旧 6-意图 if 链（向后兼容）。
#   低置信 / 无意图 / 未知工具 → 回退原 route_entry（安全网，历史补丁一条不丢）。
# ---------------------------------------------------------------------------
_ROUTE_DEST = Literal[
    "mother_opus_entry", "parse", "generate", "ask", "auth", "classify",
    "editor_entry", "entry_lowconf_block", "await_review",
]


def _route_by_legacy_intent(
    state: VariantState, config: RunnableConfig, intent: str
) -> _ROUTE_DEST:
    """C-108 legacy 6-意图 if 链（向后兼容：无 tool 决策时走它；26 单测口径不变）。"""
    items = state.get("items") or []

    if intent == INTENT_START:
        if mother_in_doubt(state):
            return "await_review"
        if items:
            return route_entry(state, config)
        if state.get("mother_dna"):
            return "generate"
        return route_entry(state, config)

    if intent == INTENT_ADJUST_MOTHER:
        return "parse"

    if intent == INTENT_EDIT_VARIANT:
        if items:
            return "parse"
        return route_entry(state, config)

    if intent == INTENT_CONFIRM_SCOPE:
        base = route_entry(state, config)
        if base in ("classify", "entry_lowconf_block"):
            return base
        return "await_review"

    if intent == INTENT_QA_TRIAGE:
        return "parse"

    if intent == INTENT_NEW_TASK:
        return "parse"

    return route_entry(state, config)


def route_after_triage(state: VariantState, config: RunnableConfig) -> _ROUTE_DEST:
    decision = state.get("intent_decision") or {}
    intent = decision.get("intent")
    tool = decision.get("tool")
    try:
        conf = float(decision.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    # 🔴 低置信 / 无意图 → 回退原 route_entry 代码分诊（安全网）。
    if intent not in _INTENT_ENUM or conf < INTENT_CONF_THRESHOLD:
        return route_entry(state, config)

    spec = resolve_tool(tool) if isinstance(tool, str) else None

    # ====== 🔴 PRD-C-109 B2·effect 三分支（有 tool 决策时按 resolve_tool(tool).effect 确定性派）======
    if spec is not None:
        effect = spec.get("effect")
        items = state.get("items") or []

        # ① 即时生效（meta）：mutator 已在 intent_triage 节点改完母题对象 → 停母题卡刷新（不重跑）。
        if effect == EFFECT_IMMEDIATE:
            # 无题组（节点未能原地改）→ 落 parse 让既有 patch 链改母题级 meta（不丢动作）。
            if not items:
                return "parse"
            return "await_review"

        # ② 重写解析 / 重出本题 → parse（既有 patch→重锚/重解链，复用现成 edit_dna_state/regen 机器）。
        #    set_年级章(hard_anchor) 同走 parse（patch 重锚链清 items 重出，与 B1 mutator 语义一致）。
        if effect in (EFFECT_REWRITE, EFFECT_REGEN):
            return "parse"

        # ③ 执行类：开始出变式 → generate（母题存疑则停母题卡·AC2）；重新解题/改题面 → parse。
        if effect == EFFECT_EXEC:
            if tool == "开始出变式":
                if mother_in_doubt(state):
                    return "await_review"
                if items:
                    return route_entry(state, config)  # 已出题组 → 不重造（守 route_entry 语义）
                if state.get("mother_dna"):
                    return "generate"
                return route_entry(state, config)
            # 重新解题 / 改题面 → parse（既有重解/重排版链；intent_decision 带 correction）。
            return "parse"

        # ④ 旋钮（难度只读·表驱动）→ parse（变式难度旋钮 stage-2，不碰母题、不重解·AC3）。
        if effect == EFFECT_KNOB:
            return "parse"

        # 兜底（理论不可达 effect）→ 回退原 route_entry。
        return route_entry(state, config)

    # ====== 🟢 无 tool（纯 legacy intent，如 C-108 单测）→ 旧 6-意图 if 链（向后兼容）======
    return _route_by_legacy_intent(state, config, intent)


# ---------------------------------------------------------------------------
# route_entry_v2：新 conditional entry point（罩在 route_entry 之上）。
#   先跑 route_entry 的「确定性高优先级硬闸」——这些是无需 LLM 的结构信号，行为与原图一致：
#     auth（登录硬闸）/ editor_op（结构化编辑 op）/ 新图（mother_opus_entry）/
#     按钮 start_variants resume / 结构化确认章 resume / 低置信闸4 / 库内母题直造。
#   只有「对话型纯文本歧义态」（老师打字、需要听懂意图）才进 intent_triage 走 LLM。
#   → 既不破历史结构化 resume（route_entry 那些 config 信号分支原样吃），又把"听懂老师"加在前面。
# ---------------------------------------------------------------------------
def _is_conversational_turn(state: VariantState, config: RunnableConfig) -> bool:
    """本轮是否「对话型纯文本歧义态」= 该交给意图层 LLM 听懂的轮。
    判据：route_entry 会把它分到 parse / generate(非按钮) / ask 这类"靠 state 惯性"的分支，
    且不是结构化 resume（无 start_variants / confirmed_chapter_id / editor_op / 新图 / 无 token）。"""
    cfg = (config or {}).get("configurable") or {}
    # 无 token → auth 硬闸，不进意图层（route_entry_v2 已先吃）。
    if conv_trace.teacher_id_from_token(cfg.get("ruoyi_token")) is None:
        return False
    # 结构化信号在 → 让 route_entry 原样吃（按钮/确认章/编辑 op/新图）。
    if cfg.get("start_variants") or cfg.get("confirmed_chapter_id"):
        return False
    if _editor_op(config) is not None and (state.get("items") or state.get("mother_dna")):
        return False
    if _extract_image_url(_latest_human_text(state.get("messages", []))):
        return False
    # 老师有打字 + 有母题/题组在手（在途母题或已出题组）→ 对话型歧义态，进意图层。
    has_text = bool(_latest_human_text(state.get("messages", [])).strip())
    has_context = bool(state.get("items") or state.get("mother_dna"))
    return has_text and has_context


def route_entry_v2(
    state: VariantState, config: RunnableConfig
) -> Literal[
    "intent_triage", "mother_opus_entry", "parse", "generate", "ask", "auth",
    "classify", "editor_entry", "entry_lowconf_block",
]:
    """🔴 PRD-C-108 B1·意图层入口（wrap-not-rewrite）。
    对话型歧义态 → intent_triage（LLM 听懂意图）；其余一切 → 原 route_entry 确定性分诊（零改）。"""
    if _is_conversational_turn(state, config):
        return "intent_triage"
    return route_entry(state, config)
