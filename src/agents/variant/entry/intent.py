"""variant 引擎 · entry/intent.py 薄意图层（PRD-C-108 B1）。

把「绑死状态位的握手」换成「薄 LLM 意图分诊」——罩在 route_entry 之上，**不推倒重写**：
- 高置信意图 → 代码映射到现有 route_entry 的目标节点（节点零改）。
- 低置信 / 分类失败 / 无法处置 → **回退原 route_entry 代码分诊**（历史 BUG 补丁一条不丢 = 安全网）。

结构件（re-wire 不 rebuild，图拓扑只 +1 节点 +1 条件路由）：
- INTENT_TRIAGE_PROMPT（prompts.py）：短稳定·低温 prompt，节点内有界 LLM 只判意图。
- classify_intent / _rule_intent：意图分类器（LLM + 规则兜底，§10 intent spec 升级）。
- mother_in_doubt：确定性读 state 判「母题存疑」（锚未定死 / 无解法骨架 / 题型未判 / 章待人审）。
- intent_triage（节点）：跑 classify_intent，写 state['intent_decision']（不路由）。
- route_after_triage（条件边）：意图 → 现有节点的映射 + 母题存疑停等 + 低置信回退 route_entry。
- route_entry_v2（新 conditional entry point）：先跑 route_entry 的确定性高优先级硬闸
  （auth/editor_op/新图——这些是结构信号，无需 LLM），对话型歧义态才进 intent_triage 走 LLM。

🔴 铁律：LLM 只判意图，控制流仍确定性 DAG（守 CLAUDE.md §4「节点内有界 LLM」）；
   re-wire 不 rebuild（route_entry 函数体零改，意图层只在前面加）；难度表驱动不在本层碰。
🔴 strangler：顶部 from agents.variant import 取依赖（运行期解析，本模块在 __init__ 末尾导入）。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、route re-export 之后导入）
    VariantState,
    _ainvoke_text,
    _editor_op,
    _entry_read_lowconf,
    _extract_image_url,
    _latest_ai_text,
    _latest_human_text,
    _parse_json,
    _pin_status,
    conv_trace,
    route_entry,
    settings,
)
from agents.variant.prompts import INTENT_TRIAGE_PROMPT

# 意图枚举（§10 intent spec 升级；与 prompt 内枚举一一对应）
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
# 母题存疑（确定性读 state，无 LLM）：解法骨架未出 / 题型 AI 自己存疑 / 锚定未定死（待人审）
# → 即便老师点了「开始」也先停在母题卡解决疑点，绝不抢跑 generate（AC2）。
# ---------------------------------------------------------------------------
def mother_in_doubt(state: VariantState) -> bool:
    """母题是否「存疑」（停等确认的判据，确定性、可单测）。任一为真即存疑：
    ① 锚未定死（_pin_status.pinned=False：年级/主考点/置信缺任一）；
    ② 解法骨架未出（mother_dna.solution_skeleton / answer 皆空）；
    ③ 读图极低置信 / 章未判出（_entry_read_lowconf）。
    无 mother_dna（还没解出母题）→ 视为存疑（更不该冲生成）。"""
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
#   只在「极有把握」的词面命中时返回意图（高置信），否则 None（交给 LLM / 回退）。
#   防止 A1 spike 暴露的三类误判：调整↔开始↔答疑。
# ---------------------------------------------------------------------------
_RESOLVE_HINTS = ("重新解", "再解一遍", "再解一次", "重解", "重新做", "重新算")
_SOLVE_HINTS = ("判别式", "配方法", "因式分解", "换个解法", "换种解法", "用韦达", "别用", "不用")
_START_HINTS = ("开始举一反三", "开始出", "开始生成", "出吧", "可以了开始", "生成变式")
_QA_HINTS = ("怎么解", "怎么做", "讲一下", "讲讲", "讲一遍", "为什么", "给学生讲", "解释一下")


def _rule_intent(utterance: str, state: VariantState) -> dict[str, Any] | None:
    """高置信词面规则（只在十拿九稳时命中）。返回 intent_decision 或 None（让 LLM 判）。"""
    u = (utterance or "").strip()
    if not u:
        return None
    # 「重新解 / 再解一遍」= 调整母题(resolve)——最易被误判成开始/编辑，硬钉死。
    if any(h in u for h in _RESOLVE_HINTS):
        return _decision(INTENT_ADJUST_MOTHER, conf=0.92,
                         correction={"field": "resolve", "value": "按老师要求重新解题"})
    # 指定/纠正解法 = 调整母题(solve)
    if any(h in u for h in _SOLVE_HINTS):
        return _decision(INTENT_ADJUST_MOTHER, conf=0.85,
                         correction={"field": "solve", "value": u})
    return None


def _decision(
    intent: str,
    conf: float,
    correction: dict[str, Any] | None = None,
    edit: dict[str, Any] | None = None,
    count: int | None = None,
) -> dict[str, Any]:
    return {
        "intent": intent,
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
    """薄意图分诊（节点内有界 LLM·低温）：规则兜底优先命中 → 直接返回；否则跑 LLM。
    返回 §10 intent_decision；任何异常 → 低置信 clarify-ish（intent=答疑·conf=0），上层回退。"""
    utterance = _latest_human_text(state.get("messages", []))
    # 规则兜底优先（高置信词面，压住 A1 易混区）
    ruled = _rule_intent(utterance, state)
    if ruled is not None:
        return ruled
    prev_ai = _latest_ai_text(state.get("messages", []))
    if len(prev_ai) > 600:
        prev_ai = prev_ai[:600] + "…"
    prompt = INTENT_TRIAGE_PROMPT.format(
        state_summary=_state_summary(state),
        prev_ai=prev_ai or "(无)",
        utterance=utterance or "(空)",
    )
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)],
            model=settings.LLM_MODEL_LIGHT,  # nano 轻活降本（受约束分类器）
            temperature=0.1,
            trace_label="intent",
        )
        parsed = _parse_json(text)
    except Exception:  # noqa: BLE001  分类失败绝不卡死 → 低置信回退
        return _decision(INTENT_QA_TRIAGE, conf=0.0)
    if not isinstance(parsed, dict):
        return _decision(INTENT_QA_TRIAGE, conf=0.0)
    intent = parsed.get("intent")
    if intent not in _INTENT_ENUM:
        return _decision(INTENT_QA_TRIAGE, conf=0.0)
    try:
        conf = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "intent": intent,
        "correction": parsed.get("correction") or {"field": None, "value": None},
        "edit": parsed.get("edit") or {"target_seq": None, "action": None},
        "count": parsed.get("count"),
        "confidence": conf,
    }


async def intent_triage(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 意图层节点：跑分诊 LLM，把结果写进 state['intent_decision']（**不路由、不改业务态**）。
    路由在 route_after_triage（确定性）。本节点是图里唯一的「意图层」新节点（拓扑 +1）。"""
    decision = await classify_intent(state, config)
    return {"intent_decision": decision, "messages": []}


# ---------------------------------------------------------------------------
# route_after_triage：意图 → 现有 route_entry 目标的映射（确定性）。
#   高置信 → 按意图派现有节点；母题存疑 → 停母题卡（即便"开始"也先解决疑点）；
#   低置信/无法处置 → 回退原 route_entry 代码分诊（安全网，历史补丁一条不丢）。
# ---------------------------------------------------------------------------
_ROUTE_DEST = Literal[
    "mother_opus_entry", "parse", "generate", "ask", "auth", "classify",
    "editor_entry", "entry_lowconf_block", "await_review",
]


def route_after_triage(state: VariantState, config: RunnableConfig) -> _ROUTE_DEST:
    decision = state.get("intent_decision") or {}
    intent = decision.get("intent")
    try:
        conf = float(decision.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    # 🔴 低置信 / 无意图 → 回退原 route_entry 代码分诊（安全网）。
    if intent not in _INTENT_ENUM or conf < INTENT_CONF_THRESHOLD:
        return route_entry(state, config)

    items = state.get("items") or []

    # 「开始出题」=显式开始触发（AC2 二触发之一；按钮触发由 route_entry_v2 前置硬闸先吃）。
    if intent == INTENT_START:
        # 🔴 母题存疑 → 即便老师明说开始，也先停在母题卡等确认（绝不抢跑 generate）。
        if mother_in_doubt(state):
            return "await_review"
        # 已出过题组 → 不重造（落 parse 既有编辑/答疑路径，守 route_entry 语义）。
        if items:
            return route_entry(state, config)
        # 母题已立住、未出题 → 直奔 generate（复用 state.mother_dna，等价按钮路径）。
        if state.get("mother_dna"):
            return "generate"
        return route_entry(state, config)

    # 「调整母题」=全状态重锚（AC3）：改年级/章/考点/解法/重解，哪怕已生成变式 → 回 stage-1。
    #   走 parse（既有「修正」意图 → patch → after_patch → classify 重锚链；patch 清 items）。
    #   不进变式编辑器、不被吞、不冲 generate。intent_decision 已带 correction，parse 会再解析细节。
    if intent == INTENT_ADJUST_MOTHER:
        return "parse"

    # 「编辑变式」：必须已有题组才有意义（无题组 → 回退，落 route_entry 兜底）。
    if intent == INTENT_EDIT_VARIANT:
        if items:
            return "parse"
        return route_entry(state, config)

    # 「确认范围」：把母题锚牢。停在母题卡（await_review）等老师进一步确认/开始，不冲生成。
    #   若 route_entry 已有结构化确认章 resume（confirmed_chapter_id）→ 让它走 classify 重锚。
    if intent == INTENT_CONFIRM_SCOPE:
        base = route_entry(state, config)
        if base in ("classify", "entry_lowconf_block"):
            return base
        return "await_review"

    # 「答疑」：走 parse（route_after_parse 的 answer 分支；空题组它自带 ask_clarify 护栏）。
    if intent == INTENT_QA_TRIAGE:
        return "parse"

    # 「新任务」：纯文本新需求 → 落 parse（既有母题纠正/答疑分诊），图片 URL 由 route_entry_v2 前置吃。
    if intent == INTENT_NEW_TASK:
        return "parse"

    # 兜底（理论不可达）→ 回退原 route_entry。
    return route_entry(state, config)


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
