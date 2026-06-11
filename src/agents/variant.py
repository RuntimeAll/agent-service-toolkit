"""PRD-C-009 · 图片举一反三 agent（块二·单题→N 道变式）。

事实源 = codeplace-C/claude-code-sign/14-举一反三-设计.md（vibe-coding：先改文档再改代码）。

主干 4 步流水线（§2）+ solve 自愈 + assemble 外显：
  analyze(读图) → classify(锚图谱+DNA置信闸) → [clarify] → generate(3=2普通+1难)
                → solve_explain(真解+自愈1次+守恒校验) → assemble(题组快照)

🔴 不变量（§3）：
  - 主考点 + 年级 = 硬守恒（贯穿 generate / 重生 / 补题 / patch）；
  - 凡进 items 的题一律过 solve_explain（无 check 不许进 assemble）；
  - generate 入口前置：mother_confirmed==true 或 三锚高置信（DNA 闸收口，多入口都过）。

LLM 层（§9）：toolkit 原生 get_model（COMPATIBLE=LangChain ChatOpenAI）→ model.ainvoke；
  多模态走 HumanMessage(content=[{type:text},{type:image_url,image_url:{url:OSS_URL}}])；
  思考型只取 content（先 reasoning_content 后 content）；max_tokens≥4096；
  LLM 外呼 lk888 走默认（别套 trust_env=False，那是治本地 localhost 的）。

checkpointer 不在此 compile（service lifespan 注入 saver；多轮 state 按 thread_id 持久）。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langgraph.config import get_stream_writer
from langgraph.graph import END, MessagesState, StateGraph

from agents import conv_trace, math_verify
from agents.variant_support import anchor_subject, persist_items
from core import get_model, relay_pool, settings

# DNA 三锚高置信门槛
CONF_GATE = 0.75
# 自愈上限（设计 §5：1 次防死循环）
MAX_HEAL = 1
# 默认配方：3 道 = 2 普通 + 1 难（设计 §5）
DEFAULT_SHAPE = {"normal": 2, "hard": 1}

# ---------------------------------------------------------------------------
# 闸B（PRD-C-010）：sympy 程序验算 + 题型分流的标记值
# —— 落 item.check.verify / item.check.review，入库时 _apply_labels 透传进 auxTags。
# 🔴 判决只读 math_verify.verify() 的 verdict（pass/fail/degrade），永不采信 LLM 自评。
# ---------------------------------------------------------------------------
VERIFY_SYMPY_PASS = "sympy_pass"  # 程序验算通过（入库可查的抓手）
# 🔴 验算墙钟预算（G5 反挂死）：sympy 对病态载荷（如 9**9**9 / 超高次方程）可能无界计算，
# _machine_verify 用 asyncio.wait_for 包 to_thread —— 超时按 degrade 降级（线程不可杀但流程解锁）。
VERIFY_TIMEOUT_S = 10.0
# 🔴 4d 方案A 后 fail_after_regen 不再外发（真 fail 题剔除）；常量保留：旧库存量 aux_tags
# 仍有该值（FE 兜底/审计查询用），勿删。
VERIFY_FAIL_AFTER_REGEN = "fail_after_regen"
VERIFY_UNVERIFIED = "unverified"  # degrade：sympy 吃不下 → 退回 LLM 自检 fallback
REVIEW_PROOF = "proof_needs_human"  # 证明/开放/作图类：不进 sympy，转人审

# 题卡可见文本（追加在 item.solution 尾部 → 入库 analyze 字段同步可见）。
# 🔴 4d 可见性矩阵（PRD-C-012，用户拍板 2026-06-11「只说好、不说坏，除非双闸都不高」）：
# 外显只有 正面/中性/沉默 三类，⚠ 仅双闸（闸B verify × 闸A gene）皆存疑；
# 真值 check.verify / gene.gate 原样入 aux_tags 审计，不洗白。
NOTE_VERIFIED_OK = "✓ 程序验算通过。"
NOTE_SELF_CHECK_OK = "✓ 已独立复算一致。"
NOTE_BOTH_GATES_LOW = "⚠ 程序验算与平行度双重存疑，请老师重点核对。"
NOTE_PROOF_REVIEW = "ℹ 证明/开放类题不做程序验算，已转人工复核。"

# 题卡外显层级（check.tier → artifact.tier → FE 徽章；展示层与真值解耦）
TIER_VERIFIED = "verified"  # 强正面：sympy 验算通过
TIER_SELF_OK = "self_ok"  # 轻正面：程序不可验，LLM 独立复算一致
TIER_PROOF = "proof"  # 中性：证明/开放类转人审
TIER_SILENT = "silent"  # 沉默：单闸存疑（不说坏）
TIER_BOTH_LOW = "both_low"  # ⚠：双闸皆存疑

# ---------------------------------------------------------------------------
# 闸A·基因闸（验"是不是平行题"，与闸B"答案对不对"正交。依据 12-题目DNA方法论 §2/§5）：
# 骨架基因(题型/难度/结构=解法骨架)必须 match —— 一变就不是平行题；
# 皮肤变量(数字/场景)必须换 —— 没换 = 复读母题，也不算平行题。
# 判决 = 一次轻量 LLM 比对(受约束 JSON) + 纯函数 gene_gate_decision；
# rework → 既有 REGEN 回炉 1 次再判；仍不过 → gene_gate:"warn" 只警示不硬拦(v1)；
# judge 调用/解析失败 → 按 pass 放行标 "skipped"（闸A是增强不是关卡，绝不卡死流程，G5）。
# 标记落 item.gene.gate → 入库透传 auxTags.gene_gate（_apply_labels）；4d 后 gene=warn
# 单闸不再外显负面（沉默），仅参与 _apply_visibility 双闸裁决。
# ---------------------------------------------------------------------------
GENE_GATE_PASS = "pass"  # 基因比对通过（平行题）
GENE_GATE_WARN = "warn"  # 回炉 1 次后仍不过 → 真值留档；外显层按 4d 矩阵裁决
GENE_GATE_SKIPPED = "skipped"  # judge 调用失败/JSON 解析失败 → 放行留痕


# ---------------------------------------------------------------------------
# State（设计 prompt 指定结构）
# ---------------------------------------------------------------------------
class VariantState(MessagesState, total=False):
    image_url: str | None
    images_count: int
    questions_in_image: int
    # analysis：年级/考点(kp)/题型(qtype) 各带置信
    analysis: dict[str, Any]
    mother_dna: dict[str, Any]
    mother_confirmed: bool
    # items[{stem, answer, solution, qtype, difficulty, level, injected_kp?,
    #        check:{badge:ok|warn, solved_answer}  ← 闸B(solve_explain)填,
    #        gene:{gate:pass|warn|skipped, reason?} ← 闸A(gene_gate)填}]
    items: list[dict[str, Any]]
    history: list[dict[str, Any]]
    # 交互层：parse_instruction 的解析结果（intent/ops/knobs/...），路由后各分支消费并清空
    pending: dict[str, Any] | None
    # 🔴 首轮配方旋钮（设计 §5 五旋钮的"首轮接线"）：None=还没抽过；{}=抽过但老师没提(走默认)。
    # {"count": int, "difficulty_plan": "increasing"|str, "qtype_dist": {"选择":2,...}, "note": str}
    # analyze（新母题轮）负责抽取/重置：新图新要求 → 重抽覆盖；同图重贴无新要求 → 保留；
    # 新图无要求 → 重置 {}（旧母题配方绝不泄漏到新母题）。generate 仅对库内母题路径兜底抽。
    knobs: dict[str, Any] | None
    # generate 的代码级配方校验缺陷清单（整组 retry 1 次后仍不符 → assemble 头部外显 ⚠）
    shape_defects: list[str]
    # 4d 方案A（PRD-C-012）：本轮被剔除题的叙事（sympy 证实标答错且重生未果 → 不外发），
    # solve_explain 每轮重写（非累计），assemble 摘要外显「本组少 N 道」
    dropped_notes: list[str]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _model():
    # COMPATIBLE provider → gemini-3-flash-preview（思考型+多模态）
    return get_model(settings.DEFAULT_MODEL)


def _content_text(resp: BaseMessage) -> str:
    """思考型只取 content（reasoning_content 不外放）。content 可能是 str 或 parts list。"""
    c = resp.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


# ---------------------------------------------------------------------------
# LLM 往返持久化（PRD-C-009：给维护者过 prompt/排查用）
# —— 每次 LLM 调用落一行 JSONL：填充后的完整 prompt + 模型原始返回 + 哪个 prompt + 耗时。
# 关掉设 env VARIANT_LLM_TRACE=0；落盘失败绝不影响主流程。
# ---------------------------------------------------------------------------
_LLM_TRACE_ENABLED = os.getenv("VARIANT_LLM_TRACE", "1").lower() not in ("0", "false", "no")
_LLM_TRACE_PATH = Path(__file__).resolve().parents[2] / "data" / "llm_trace.jsonl"
_LLM_TRACE_SEQ = 0  # 进程内自增序号（同一进程内调用顺序）

# prompt 内容前缀 → 标签（哪个 prompt）。新增/造同含「基于母题 DNA」，先判 add 再判 generate。
_TRACE_MARKERS: list[tuple[str, str]] = [
    ("看这张题目图", "analyze"),
    ("出题配方", "knobs"),
    ("数学验算载荷抽取器", "extract"),
    ("平行题基因比对器", "gene_judge"),
    ("独立解出的答案与题面标答不一致", "regen"),
    ("你是严谨的数学阅卷老师", "solve"),
    ("举一反三 agent 的指令解析器", "parse"),
    ("老师对下面这组变式题的某道有疑问", "answer"),
    ("**新增**", "add"),
    ("举一反三变式", "generate"),
]


def _msg_text(m: BaseMessage) -> str:
    """取一条消息的文本（多模态 list 取其中 text part）。"""
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                return p.get("text", "")
    return ""


def _trace_label(messages: list[BaseMessage]) -> str:
    head = "".join(_msg_text(m) for m in messages)[:120]
    for marker, label in _TRACE_MARKERS:
        if marker in head:
            return label
    return "unknown"


def _serialize_request(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.__class__.__name__
        c = m.content
        if isinstance(c, str):
            out.append({"role": role, "text": c})
        elif isinstance(c, list):
            parts: list[dict] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append({"text": p.get("text", "")})
                elif isinstance(p, dict) and p.get("type") == "image_url":
                    parts.append({"image_url": (p.get("image_url") or {}).get("url")})
                else:
                    parts.append({"raw": str(p)})
            out.append({"role": role, "parts": parts})
        else:
            out.append({"role": role, "text": str(c)})
    return out


def _trace_llm(
    label: str,
    messages: list[BaseMessage],
    response_text: str,
    response_raw: Any,
    duration_ms: int,
    error: str | None = None,
    retried: bool = False,
    model: str | None = None,
) -> None:
    if not _LLM_TRACE_ENABLED:
        return
    global _LLM_TRACE_SEQ
    _LLM_TRACE_SEQ += 1
    try:
        rec = {
            "seq": _LLM_TRACE_SEQ,
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            # 🔴 与 conv_trace 同源：实际成交中转站的 model（不再写 DEFAULT_MODEL 枚举名）
            "model": model or settings.COMPATIBLE_MODEL,
            "duration_ms": duration_ms,
            "retried": retried,
            "request": _serialize_request(messages),
            "response": response_text,
            "response_raw": response_raw,
        }
        if error:
            rec["error"] = error
        _LLM_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LLM_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass  # 持久化绝不拖垮主流程


async def _ainvoke_text(
    messages: list[BaseMessage],
    retry: bool = True,
    *,
    public_stream: bool = False,
    on_delta: Any = None,
) -> str:
    """ainvoke + 取 content；偶发空返回重试一次。max_tokens≥4096 给思考型留头。

    🔴 每次调用落 JSONL 往返记录（_trace_llm）：发送的完整 prompt + 原始返回。
    🔴 思维外放（用户反馈 2026-06-11）：默认打 skip_stream 标签 —— JSON 类中间产物的
    token 流对用户是乱码，service 按标签丢弃；只有人话型调用（答疑等）传
    public_stream=True，token 才透到前端打字机。on_delta=流内进度回调（拿累计文本），
    generate 用它数「已写到第几题」。
    """
    label = _trace_label(messages)
    tags = None if public_stream else ["skip_stream"]
    # 用户级/会话级归属：从 graph config 取 thread_id + ruoyi_token(→teacher_id)
    conf = (ensure_config() or {}).get("configurable", {}) or {}
    thread_id = conf.get("thread_id")
    teacher_id = conv_trace.teacher_id_from_token(conf.get("ruoyi_token"))
    max_tokens = settings.VARIANT_MAX_TOKENS
    t0 = time.monotonic()
    relay = settings.RELAY_NAME
    model_used = settings.COMPATIBLE_MODEL
    fallback = 0
    try:
        # 🔴 走中转站熔断转移池（Block B）：返回实际成交中转站 + 该站 model + 转移次数
        #   （RELAY_POOL 各站可配不同模型，trace/计费必须按成交站归因）
        resp, relay, model_used, fallback = await relay_pool.ainvoke_failover(
            messages, max_tokens=max_tokens, tags=tags, on_delta=on_delta
        )
        text = _content_text(resp).strip()
        retried = False
        if not text and retry:
            retried = True
            resp, relay, model_used, fb2 = await relay_pool.ainvoke_failover(
                messages, max_tokens=max_tokens, tags=tags, on_delta=on_delta
            )
            fallback += fb2
            text = _content_text(resp).strip()
    except Exception as e:  # noqa: BLE001 — 记下失败往返后照常抛
        dur = int((time.monotonic() - t0) * 1000)
        _trace_llm(label, messages, "", None, dur, error=str(e), model=model_used)
        conv_trace.write(
            teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
            model=model_used, relay=relay, fallback_count=fallback,
            request=_serialize_request(messages),
            response="", response_raw=None, duration_ms=dur, error=str(e),
        )
        raise
    dur = int((time.monotonic() - t0) * 1000)
    pt, ct = relay_pool.usage_tokens(resp)
    cost = relay_pool.cost_yuan(model_used, pt, ct)
    # 原始返回：content（思考型可能是 parts list）+ reasoning/usage 等附加信息（best-effort）
    raw: dict[str, Any] = {}
    try:
        raw["content"] = resp.content
        if getattr(resp, "additional_kwargs", None):
            raw["additional_kwargs"] = resp.additional_kwargs
        if getattr(resp, "response_metadata", None):
            raw["response_metadata"] = resp.response_metadata
        if getattr(resp, "usage_metadata", None):
            raw["usage_metadata"] = resp.usage_metadata  # 🔴 token 取数主源
    except Exception:
        raw = {"content": str(getattr(resp, "content", ""))}
    _trace_llm(label, messages, text, raw, dur, retried=retried, model=model_used)
    # 🔴 用户级对话持久化（优化基础数据源）→ 独立解耦库 conv_trace
    conv_trace.write(
        teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
        model=model_used, relay=relay, fallback_count=fallback,
        request=_serialize_request(messages),
        response=text, response_raw=raw, duration_ms=dur, retried=retried,
        prompt_tokens=pt, completion_tokens=ct, cost_yuan=cost,
    )
    return text


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> Any:
    """剥 markdown fence + JSON 容错（抄 ai-orchestrator llm/client 范式）。"""
    text = (text or "").strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        # 退而求其次：截第一个 { 到最后一个 }
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                pass
    return None


# --- 富文本净化（用户反馈 2026-06-11：解析裸字符不渲染的根因） -----------------
# LLM（gpt-5.4）产 JSON 时两类脏输出：① LaTeX 用 \( \) / \[ \] 定界（前端
# markdown-it-katex 只认 $/$$，且 markdown 会把 \( 的反斜杠当转义吃掉）；② 把换行
# 写成双反斜杠 → 解出字面 \n 两个字符。统一在解析边界净化，入库/快照/气泡三处共净。
_PAREN_MATH_RE = re.compile(r"\\\(\s*(.+?)\s*\\\)", re.DOTALL)
_BRACKET_MATH_RE = re.compile(r"\\\[\s*(.+?)\s*\\\]", re.DOTALL)
# 字面 \n 后跟小写字母 = 可能是 LaTeX 命令（\neq \nabla \newline \nu …），不动；其余视为换行
_LITERAL_NL_RE = re.compile(r"\\n(?![a-z])")


def _sanitize_rich_text(s: Any) -> Any:
    """LLM 产出的 stem/answer/solution 净化：\\(..\\)→$..$、\\[..\\]→$$..$$、字面 \\n→换行。"""
    if not isinstance(s, str) or not s:
        return s
    s = _BRACKET_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", s)
    s = _PAREN_MATH_RE.sub(lambda m: f"${m.group(1)}$", s)
    return _LITERAL_NL_RE.sub("\n", s)


def _sanitize_item(it: dict[str, Any]) -> dict[str, Any]:
    """就地净化一道题的富文本字段，返回原 dict（链式用）。"""
    for k in ("stem", "answer", "solution"):
        it[k] = _sanitize_rich_text(it.get(k))
    return it


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else _content_text(msg)
    return ""


_URL_RE = re.compile(r"https?://[^\s)>'\"]+", re.IGNORECASE)


def _extract_image_url(text: str) -> str | None:
    """从用户消息抽 OSS 题图 URL（MVP 贴 URL，file_uploader future）。"""
    m = _URL_RE.search(text or "")
    return m.group(0) if m else None


def _strip_urls(text: str) -> str:
    """去掉文本里所有 URL（首轮"图 URL + 人话要求"里把人话剥出来给 knobs 抽取）。"""
    return _URL_RE.sub("", text or "").strip()


def _conf_ok(analysis: dict[str, Any]) -> bool:
    """三锚（年级/考点/题型）任一低置信 → 闸不过。"""
    for k in ("grade", "kp", "qtype"):
        node = analysis.get(k) or {}
        if float(node.get("confidence", 0) or 0) < CONF_GATE:
            return False
    return True


# ---------------------------------------------------------------------------
# 思维外放（stage 思路条事件）：langgraph custom 通道 → service stream_mode=custom
# → SSE 帧 {"type":"message","content":{"type":"custom","custom_data":{"stage":{...}}}}
# → FE（book-ui variant 页）按 key 更新/追加紫色思路条。
# 🔴 stage 是增强不是关卡（G5）：runtime 外调用（单测直调节点无 runnable context →
#    get_stream_writer 抛 RuntimeError）/ writer 发送失败，一律静默吞，绝不影响主流程。
# 🔴 必须包成 role="custom" 的 ChatMessage 且 content 是单元素 list ——
#    service utils.langchain_to_chat_message 只认这个形状，裸 dict 会变成 error 帧。
# ---------------------------------------------------------------------------
def _emit_stage(key: str, title: str, status: str, detail: str | None = None) -> None:
    """发思路条 stage 事件（key/title/status/detail 契约与 FE 严格一致）。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    stage: dict[str, Any] = {"key": key, "title": title, "status": status}
    if detail:
        stage["detail"] = detail
    try:
        writer(ChatMessage(content=[{"stage": stage}], role="custom"))
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点
        pass


# ---------------------------------------------------------------------------
# artifact 快照帧（PRD-C-011 Bucket 3）：FE 题卡数据源 = 本帧，不 parse markdown。
# 契约（BE/FE 严格一致）：ChatMessage(role="custom", content=[{"artifact": {
#   "items": [{index/stem/answer/solution/qtype/difficulty/level/verify/gene/persisted}],
#   "header": {recipe/kp/grade}}}])
# 发射点：assemble 收尾（每轮题组变化必过）+ persist_to_bank 成功后（persisted=true 更新）。
# 🔴 独立函数、不复用 _emit_stage（test_variant_stage 对 _emit_stage 调用序列精确断言）；
#    同 _emit_stage 双层静默吞：无 runtime context / writer 抛 → 绝不影响主流程。
# ---------------------------------------------------------------------------
def _artifact_payload(
    state: VariantState, persisted_flags: list[bool] | None = None
) -> dict[str, Any]:
    """纯组帧（零 IO 可单测）：state.items/check/gene/knobs/analysis → artifact 契约 dict。"""
    items = state.get("items") or []
    facts = _mother_facts(state)
    out_items: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        chk = it.get("check") or {}
        out_items.append(
            {
                "index": i + 1,
                "stem": str(it.get("stem") or ""),
                "answer": str(it.get("answer") or ""),
                "solution": str(it.get("solution") or ""),
                "qtype": str(it.get("qtype") or ""),
                "difficulty": _to_int(it.get("difficulty")) or 0,
                "level": str(it.get("level") or "normal"),
                # 🔴 verify 与 review 互斥不同键：证明类只有 review（proof_needs_human）
                "verify": chk.get("verify") or chk.get("review") or None,
                # 4d 外显层级（FE 徽章唯一依据；旧线程恢复无 tier → FE 按「只说好」兜底）
                "tier": chk.get("tier") or None,
                "gene": (it.get("gene") or {}).get("gate") or None,
                # persisted：flags 优先（persist 节点按回执现算）；否则读 item 簿记
                # （persist_to_bank 成功后回写 state.items[i].persisted → 后续编辑轮
                #  assemble 重发快照时「已收录」徽章不回退，G5 二次入库不重复落行）
                "persisted": (
                    bool(persisted_flags[i])
                    if persisted_flags is not None and i < len(persisted_flags)
                    else bool(it.get("persisted"))
                ),
            }
        )
    return {
        "items": out_items,
        "header": {
            "recipe": knobs_desc(state.get("knobs")) or None,
            "kp": facts["kp_name"] if facts["kp_name"] != "未知考点" else None,
            "grade": facts["grade"] if facts["grade"] != "未知年级" else None,
        },
    }


def _emit_artifact(
    state: VariantState, persisted_flags: list[bool] | None = None
) -> None:
    """发 artifact 快照帧（FE 题卡数据源）。任何异常静默吞，artifact 是增强不是关卡。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    try:
        writer(ChatMessage(content=[{"artifact": _artifact_payload(state, persisted_flags)}], role="custom"))
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点
        pass


# ---------------------------------------------------------------------------
# Router（入口分诊：登录? 有图? 在途母题? 库内母题跳 analyze/classify）
# ---------------------------------------------------------------------------
def route_entry(
    state: VariantState, config: RunnableConfig
) -> Literal["analyze", "parse", "generate", "ask", "auth"]:
    # 🔴 身份硬闸（用户拍板 2026-06-11）：每次对话绑死登录老师。token 缺失/解不出 userId
    # → 一步不走（不进任何 LLM 节点，conv_trace 也不会产生无主行；表级 NOT NULL 双保险）。
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    if conv_trace.teacher_id_from_token(token) is None:
        return "auth"
    url = _extract_image_url(_latest_human_text(state.get("messages", [])))
    # 跨轮新图 = 视作新母题（设计 §6：重走 analyze，覆盖在途状态）
    if url:
        return "analyze"
    # 库内母题（已确认 DNA）、还没出题 → 直接造（跳 analyze/classify）
    if state.get("mother_confirmed") and state.get("mother_dna") and not state.get("items"):
        return "generate"
    # 老会话·纯文字：已出题组 或 🔴 在途母题（已分析停在 clarify 等老师答年级/考点）
    # → parse 分诊。修 17 号多轮路由漏洞：旧版要求有 items 才进 parse，把「clarify 的
    # 回答」漏成催图（root cause 见 claude-code-sign/17-route_entry-多轮路由漏洞-修复任务.md §2）。
    if state.get("items") or state.get("mother_dna"):
        return "parse"
    # 没图、无在途母题、无题组 → 催图（设计 §6 输入边界兜底）
    return "ask"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = """你是浙教版初中数学命题专家。看这张题目图，**流式**输出母题分析。

只输出一个 JSON（不要解释），结构：
{
  "is_question_image": true/false,   // 非题目图(风景/截图/空白)填 false
  "images_count": 1,                  // 检测到几张图
  "questions_in_image": 1,            // 这张图里有几道题
  "grade": {"value": "七年级上学期", "confidence": 0.0~1.0},
  "subject": "数学",
  "kp": {"value": "核心考点粗描述(如:一元二次方程求根)", "confidence": 0.0~1.0},
  "qtype": {"value": "选择/填空/解答", "confidence": 0.0~1.0},
  "stem": "题干(Markdown+LaTeX)",
  "answer": "标准答案(图里没有就先解母题得出)",
  "difficulty": 1~5,
  "structure": "结构指纹/解法骨架简述",
  "solution_skeleton": "解法骨架(步骤)"
}
🔴 无答案先解母题得答案/解法骨架(作 verify 基准)。各锚(年级/考点/题型)如实给 confidence。"""


async def analyze(state: VariantState, config: RunnableConfig) -> VariantState:
    """① 分析：multimodal 读图 → 年级/学科/粗考点/题型 + 题干/答案/难度/结构 + 几图几题 + 各锚置信。"""
    # 🔴 本轮消息里的 URL 优先（跨轮新图 = 新母题，必须分析新图）；无则沿用在途母题图。
    #   旧序（state 优先）会让同 thread 第二张图被静默忽略、永远重分析第一张。
    url = _extract_image_url(_latest_human_text(state.get("messages", []))) or state.get(
        "image_url"
    )
    if not url:
        return {
            "messages": [AIMessage(content="请先贴一张题目图的 OSS URL，我才能开始举一反三。")]
        }

    _emit_stage("analyze", "读图分析", "running")
    # 🔴 多模态走 LangChain HumanMessage(content=[text, image_url]) → model.ainvoke
    msg = HumanMessage(
        content=[
            {"type": "text", "text": ANALYZE_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )
    text = await _ainvoke_text([msg])
    data = _parse_json(text) or {}

    if data.get("is_question_image") is False:
        _emit_stage("analyze", "读图分析", "warn", "未识别为题目图")
        return {
            "image_url": url,
            "messages": [
                AIMessage(content="这张图我没认出是题目（可能是风景/截图/空白）。请换一张清晰的题目图。")
            ],
        }

    analysis = {
        "grade": data.get("grade") or {"value": None, "confidence": 0},
        "subject": data.get("subject") or "数学",
        "kp": data.get("kp") or {"value": None, "confidence": 0},
        "qtype": data.get("qtype") or {"value": None, "confidence": 0},
    }
    mother_dna = {
        "stem": data.get("stem"),
        "answer": data.get("answer"),
        "difficulty": data.get("difficulty"),
        "structure": data.get("structure"),
        "solution_skeleton": data.get("solution_skeleton"),
    }
    _emit_stage("analyze", "读图分析", "done")
    # 🔴 旋钮跨母题防泄漏 + clarify 迂回防丢：
    #   - 本轮人话抽出新配方 → 覆盖旧配方（新母题新要求，跨轮第二张图的文字不再被丢）；
    #   - 没抽出新配方：同图重贴（clarify 迂回/澄清应答）→ 保留首轮已抽配方；
    #     新图 → 重置 {}（旧母题的「5道递增」绝不错套到新母题上）。
    # shape_defects 一并清零（上一母题的缺陷外显不带进新母题轮）。
    user_text = _strip_urls(_latest_human_text(state.get("messages", [])))
    new_knobs = await _extract_knobs(state) if user_text else {}
    if not new_knobs and url == state.get("image_url") and state.get("knobs") is not None:
        knobs = state.get("knobs")  # 同图重贴且本轮无新配方 → 保留
    else:
        knobs = new_knobs
    return {
        "image_url": url,
        "images_count": int(data.get("images_count") or 1),
        "questions_in_image": int(data.get("questions_in_image") or 1),
        "analysis": analysis,
        "mother_dna": mother_dna,
        "knobs": knobs,
        "shape_defects": [],
        "messages": [],
    }


async def classify(state: VariantState, config: RunnableConfig) -> VariantState:
    """② 分类：粗考点 → 锚 biz_subject 真实节点 + 年级(编码前4位)。DNA 置信闸。"""
    analysis = dict(state.get("analysis") or {})
    kp_node = dict(analysis.get("kp") or {})
    coarse = kp_node.get("value")

    # 只读 SQL 锚定（同步，丢线程池）
    candidates: list[dict] = []
    if coarse:
        try:
            candidates = await asyncio.to_thread(anchor_subject, coarse)
        except Exception as e:  # 库未起/连接失败 → 降级，置信不抬，转 clarify
            candidates = []
            analysis["_anchor_error"] = str(e)

    if candidates:
        best = candidates[0]
        kp_node["anchored"] = {
            "id": best["id"],
            "code": best["code"],
            "name": best["name"],
        }
        kp_node["value"] = best["name"]  # 用标准考点名
        # 命中真实节点 → 抬考点置信
        kp_node["confidence"] = max(float(kp_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["kp"] = kp_node
        # 年级 = 编码前4位反推（比 LLM 裸猜准）→ 抬年级置信
        if best.get("grade_code"):
            grade_node = dict(analysis.get("grade") or {})
            grade_node["code"] = best["grade_code"]
            grade_node["confidence"] = max(
                float(grade_node.get("confidence", 0) or 0), CONF_GATE
            )
            analysis["grade"] = grade_node

    confirmed = _conf_ok(analysis)
    kp_name = (analysis.get("kp") or {}).get("value") or "?"
    grade_name = (analysis.get("grade") or {}).get("value") or (
        analysis.get("grade") or {}
    ).get("code") or "?"
    _emit_stage(
        "classify",
        "锚定考点",
        "done" if confirmed else "warn",
        f"考点「{kp_name}」·年级「{grade_name}」",
    )
    return {
        "analysis": analysis,
        "mother_confirmed": bool(confirmed),
        "messages": [],
    }


def gate_after_classify(state: VariantState) -> Literal["generate", "clarify"]:
    """🔴 DNA 闸：三锚高置信 或 已确认 → 造题；否则先 clarify。"""
    if state.get("mother_confirmed"):
        return "generate"
    return "clarify"


async def clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """DNA 置信不足 → 回问老师确认（只问不造，进 WAIT 等下一句）。"""
    analysis = state.get("analysis") or {}
    asks = []
    g = analysis.get("grade") or {}
    k = analysis.get("kp") or {}
    q = analysis.get("qtype") or {}
    if float(g.get("confidence", 0) or 0) < CONF_GATE:
        asks.append(f"年级我拿不准（看着像「{g.get('value') or '?'}」），是几年级上/下学期？")
    if float(k.get("confidence", 0) or 0) < CONF_GATE:
        asks.append(f"核心考点我没锚准（粗看是「{k.get('value') or '?'}」），对吗？或请指正。")
    if float(q.get("confidence", 0) or 0) < CONF_GATE:
        asks.append(f"题型我读着像「{q.get('value') or '?'}」，对吗？")
    if not asks:
        asks.append("我对母题 DNA 还不够确定，请确认下年级/考点/题型再继续。")
    body = "我先确认母题 DNA，确认后再造变式：\n\n" + "\n".join(f"- {a}" for a in asks)
    return {"messages": [AIMessage(content=body)]}


GENERATE_PROMPT = """你是浙教版初中数学命题专家。基于母题 DNA，造 {n} 道举一反三变式。

母题 DNA：
- 主考点(硬守恒，不可改): {kp_name}
- 年级(硬守恒): {grade}
- 题型: {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}

配方(默认)：共 {n} 道 = {n_normal} 道普通(守难度) + {n_hard} 道难题(升一档)。
铁律：
- **主考点 + 年级 硬守恒**：每道题都必须仍考「{kp_name}」、仍在该年级范围内。
- 守{{解题结构, 难度(普通题)}}；只换{{数字, 场景}}。
- 难题 1 道升一档；可综合 1 个相邻知识点(主考点仍守，注入为副点)，填到 injected_kp。
- **答案可程序验算**(PRD-C-012 4c)：answer 优先给可计算的数值/表达式（如 $x_1=2, x_2=3$、
  $3\\sqrt{{2}}$、选项字母），能出数值答案就不要出纯文字表述答案（证明/作图类除外）；
  数字设计成解恰好整洁可验（避免无理数逼近、区间叙述、带单位混排）。

只输出 JSON 数组(不要解释)，每个元素：
{{"stem":"题干(Markdown+LaTeX)","answer":"标准答案","solution":"完整解析(过程+答案)",
  "qtype":"选择/填空/解答","difficulty":1~5,"level":"normal/hard","injected_kp":"相邻kp名或null"}}

格式硬规定（stem/answer/solution 三个字段都遵守）：
- 数学式一律用 $...$ 包裹（行间长式用 $$...$$），如 $\\sqrt{{2}}$、$x^2-3x+2=0$；
  **禁止**裸 LaTeX 命令、禁止 \\( \\) / \\[ \\] 定界符。
- 换行用 JSON 标准转义 \\n（一个反斜杠），不要写成 \\\\n。"""


def _mother_facts(state: VariantState) -> dict:
    analysis = state.get("analysis") or {}
    dna = state.get("mother_dna") or {}
    kp = analysis.get("kp") or {}
    anchored = kp.get("anchored") or {}
    return {
        "kp_name": kp.get("value") or "未知考点",
        "grade": (analysis.get("grade") or {}).get("value") or "未知年级",
        "qtype": (analysis.get("qtype") or {}).get("value") or "解答",
        "stem": dna.get("stem") or "",
        "skeleton": dna.get("solution_skeleton") or dna.get("answer") or "",
        # 入库用：锚定到的真实节点编码 + 母题 id（图母题 MVP 无 id）
        "subject_id": anchored.get("code"),
        "mother_question_id": dna.get("mother_question_id"),
        # 🔴 图母题不在库 → 入库时先把母题(原题)也落库挂血缘，下面这几项给 build_mother_bo 用
        "mother_answer": dna.get("answer"),
        "mother_solution": dna.get("solution_skeleton") or dna.get("answer"),
        "mother_difficulty": dna.get("difficulty"),
        # 🔴 PRD-C-009 入库存 DNA/打标：结构指纹(dim5) + 锚定置信(labelConfidence)
        "mother_structure": dna.get("structure"),
        "kp_confidence": (kp.get("confidence") if isinstance(kp, dict) else None),
        "image_url": state.get("image_url"),
    }


# ---------------------------------------------------------------------------
# 首轮配方旋钮（修 bug：首轮带图附带的文字要求被整条丢弃 → 旋钮成一等公民）
# 链路：generate 入口（knobs 为 None 时）一次受约束 LLM 抽取（KNOBS_PROMPT）
#   → 纯函数 normalize_knobs 钳制 → 存回 state（跨轮保留）
#   → recipe_from_knobs 驱动 GENERATE_PROMPT 配方段
#   → 代码级 shape_check（数量/题型分布/递增单调）不符整组 retry 1 次，仍不符 → 头部 ⚠ 外显
#   → 闸A 对齐：gene_judge_knobs_spec 注入 judge prompt（递增难度不被基因闸误判回炉）。
# 🔴 抽取失败/解析失败 → knobs={} 回落默认，绝不卡死出题（G5）。
# ---------------------------------------------------------------------------
KNOBS_PROMPT = """你是举一反三 agent 的出题配方抽取器（受约束抽取：只抽老师明说的，绝不脑补）。老师贴题目图时附带了下面这句话，请从中抽出出题配方旋钮。

老师的话：
{utterance}

只输出一个 JSON（不要解释、不要 markdown fence）：
{{
  "count": null,            // 要出几道题（正整数，1~8）；没说填 null
  "difficulty_plan": null,  // 难度安排："increasing"=难度递增/越来越难/一道比一道难；没提难度安排填 null；其它难度要求把老师原话填进来（如"都出难题"）
  "qtype_dist": null,       // 题型配比，如 {{"选择":2,"填空":2,"解答":1}}；题型只能用 选择/填空/解答 三类（应用题/计算题/证明题等都归"解答"）；没说填 null
  "note": ""                // 其余装不进上面旋钮的自由要求原话（如"贴近生活场景"、"数字简单点"）；没有填 ""
}}

硬约束：
- 只抽老师明确说了的；没说的旋钮一律 null/""，绝不脑补默认值。
- qtype_dist 的值必须是正整数；如与 count 看似矛盾也如实抽取，程序会做最终校验。"""

# 题型归一表（normalize_knobs 用）：只认 选择/填空/解答 三类
_QTYPE_ALIAS: dict[str, str] = {
    "选择": "选择",
    "选择题": "选择",
    "单选": "选择",
    "单选题": "选择",
    "填空": "填空",
    "填空题": "填空",
    "解答": "解答",
    "解答题": "解答",
    "计算": "解答",
    "计算题": "解答",
    "应用": "解答",
    "应用题": "解答",
    "证明": "解答",
    "证明题": "解答",
    "大题": "解答",
}
# 归一后要在 note 里保留语义的原始题型词（应用题 → 解答 + note "应用场景"）
_QTYPE_NOTE_HINTS: dict[str, str] = {"应用": "应用场景", "应用题": "应用场景"}
KNOBS_COUNT_MIN, KNOBS_COUNT_MAX = 1, 8
PLAN_INCREASING = "increasing"
_PLAN_INCREASING_WORDS = ("increasing", "递增", "越来越难", "逐题变难", "一道比一道难")
DIFFICULTY_CAP = 5  # 难度封顶（递增计划逐题 +1 的上限）


def normalize_knobs(parsed: Any) -> dict[str, Any]:
    """🔴 纯函数钳制（零 LLM/零 IO，可单测）：LLM 抽取产物 → 受约束 knobs dict。

    规则：
    - 非 dict/解析失败 → {}（回落默认配方）。
    - count：宽容转 int，钳到 [1, 8]；非法 → 丢弃。
    - qtype_dist：键过 _QTYPE_ALIAS 归一（应用题/计算题/证明题→解答，且"应用"在 note 保留
      「应用场景」语义）；不认识的题型键丢弃；值须为正整数；同义键合并求和。
    - dist 总和同样钳到 KNOBS_COUNT_MAX（按点名顺序累计到上限封口，截断写进 note 外显）——
      防失控护栏对 dist 路径同等生效，不被「逐项点名」绕过。
    - dist 总和与 count 不一致 → 以 dist 总和为准，且把「数量从 X 调整为 Y」写进 note 外显
      （assemble 头部可见，不静默吞老师的数）。
    - difficulty_plan：命中递增词 → "increasing"；"default"/空 → 丢弃；其余原话保留。
    - note：strip 后非空才保留。
    - 全空 → {}。
    """
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, Any] = {}

    cnt = _to_int(parsed.get("count"))
    if cnt is not None:
        out["count"] = max(KNOBS_COUNT_MIN, min(cnt, KNOBS_COUNT_MAX))

    note_bits: list[str] = []
    raw_dist = parsed.get("qtype_dist")
    if isinstance(raw_dist, dict):
        dist: dict[str, int] = {}
        for k, v in raw_dist.items():
            qt = _QTYPE_ALIAS.get(str(k or "").strip())
            n = _to_int(v)
            if not qt or n is None or n <= 0:
                continue
            dist[qt] = dist.get(qt, 0) + n
            hint = _QTYPE_NOTE_HINTS.get(str(k or "").strip())
            if hint and hint not in note_bits:
                note_bits.append(hint)
        if dist:
            total = sum(dist.values())
            if total > KNOBS_COUNT_MAX:
                # 🔴 防失控护栏（与 count 钳制/ADD_COUNT_MAX 同哲学）：按点名顺序累计到上限封口
                clipped: dict[str, int] = {}
                budget = KNOBS_COUNT_MAX
                for qt, n in dist.items():
                    if budget <= 0:
                        break
                    take = min(n, budget)
                    clipped[qt] = take
                    budget -= take
                note_bits.append(
                    f"题型配比共 {total} 道超单轮上限，已截到 {KNOBS_COUNT_MAX} 道"
                )
                dist = clipped
                total = KNOBS_COUNT_MAX
            out["qtype_dist"] = dist
            if out.get("count") != total:
                if out.get("count") is not None:
                    # count 与配比总和冲突 → dist 为准，但调整必须外显（不静默吞老师的数）
                    note_bits.append(f"按题型配比把数量从 {out['count']} 调整为 {total}")
                out["count"] = total  # dist 总和为准（含 count 缺失时补齐）

    plan = str(parsed.get("difficulty_plan") or "").strip()
    if plan and plan.lower() != "default":
        if any(w in plan.lower() for w in _PLAN_INCREASING_WORDS):
            out["difficulty_plan"] = PLAN_INCREASING
        else:
            out["difficulty_plan"] = plan  # 自由难度要求原话保留（generate/闸A 原样注入）

    note = str(parsed.get("note") or "").strip()
    if note:
        note_bits.append(note)
    if note_bits:
        out["note"] = "；".join(note_bits)

    return out


def recipe_from_knobs(knobs: dict[str, Any] | None, mother_difficulty: Any = None) -> dict[str, Any]:
    """🔴 纯函数：knobs → generate 配方（n/n_normal/n_hard + 老师配方段 spec + 递增预期档位）。

    knobs 空 → 与旧默认完全等价：n=3、2 普 1 难、spec=""（行为不变的回归锚点）。
    递增计划：从母题难度起逐题 +1、封顶 DIFFICULTY_CAP。expected_difficulties 只用于
    prompt 文案 + n_hard 推算；闸A/代码闸的同尺判据 = generate 落在 item 级的
    expected_difficulty 印记 + shape_check(mother_difficulty) 现算（二者公式一致）。
    """
    knobs = knobs or {}
    if not knobs:
        n_normal, n_hard = DEFAULT_SHAPE["normal"], DEFAULT_SHAPE["hard"]
        return {
            "n": n_normal + n_hard,
            "n_normal": n_normal,
            "n_hard": n_hard,
            "spec": "",
            "expected_difficulties": None,
        }

    dist = knobs.get("qtype_dist") or {}
    n = knobs.get("count") or (sum(dist.values()) if dist else 0) or (
        DEFAULT_SHAPE["normal"] + DEFAULT_SHAPE["hard"]
    )
    md = _to_int(mother_difficulty) or 3
    plan = knobs.get("difficulty_plan")
    expected: list[int] | None = None

    lines = [f"- 共 {n} 道（必须恰好 {n} 道，不多不少）。"]
    if dist:
        dist_s = "、".join(f"{k}×{v}" for k, v in dist.items())
        lines.append(f"- 题型配比：{dist_s}（每道题的 qtype 严格按此配比给）。")
    if plan == PLAN_INCREASING:
        expected = [min(md + i, DIFFICULTY_CAP) for i in range(n)]
        d_s = ",".join(str(d) for d in expected)
        lines.append(
            f"- 难度计划：递增 —— 从母题难度({md})起逐题升一档、封顶 {DIFFICULTY_CAP}；"
            f"各题 difficulty 依次为 {d_s}；难度高于母题的填 level=\"hard\"，否则 \"normal\"。"
        )
    elif plan:
        lines.append(f"- 难度要求（老师原话，best-effort 满足）：{plan}")
    if knobs.get("note"):
        lines.append(f"- 其他要求（best-effort 吸收，撞主考点/年级硬守恒的忽略）：{knobs['note']}")

    spec = "\n\n老师指定配方（🔴 优先于上面的默认配方，必须严格满足）：\n" + "\n".join(lines)
    if expected:
        n_hard = sum(1 for d in expected if d > md)
    else:
        n_hard = 1 if n >= 2 else 0
    return {
        "n": n,
        "n_normal": n - n_hard,
        "n_hard": n_hard,
        "spec": spec,
        "expected_difficulties": expected,
    }


def shape_check(
    items: list[dict[str, Any]],
    knobs: dict[str, Any] | None,
    mother_difficulty: Any = None,
) -> list[str]:
    """🔴 纯函数·代码级配方校验：返回缺陷清单（空 = 合格）。knobs 空 → 永远 []（旧行为）。

    规则表：
    - S1 数量：knobs 给了 count（或 dist 推得）→ len(items) 必须相等。
    - S2 题型分布：knobs 给了 qtype_dist → 各题 qtype（过 _QTYPE_ALIAS 归一）计数须逐项相等。
    - S3 递增：difficulty_plan=increasing 时与闸A 同一把尺 —— 给了 mother_difficulty →
      逐项比对预期档 min(md+i, DIFFICULTY_CAP)（让整组 retry 有机会一次修对，而不是代码闸
      放行后闸A 必 warn 且回炉结构性修不动）；母题难度未知 → 退化为单调不减（缺难度按违规算）。
    """
    knobs = knobs or {}
    if not knobs:
        return []
    defects: list[str] = []

    want_n = knobs.get("count")
    if want_n and len(items) != want_n:
        defects.append(f"数量不符：要求 {want_n} 道，实出 {len(items)} 道")

    dist = knobs.get("qtype_dist") or {}
    if dist:
        got: dict[str, int] = {}
        for it in items:
            qt = _QTYPE_ALIAS.get(str(it.get("qtype") or "").strip(), str(it.get("qtype") or "").strip())
            got[qt] = got.get(qt, 0) + 1
        if any(got.get(k, 0) != v for k, v in dist.items()):
            want_s = "、".join(f"{k}×{v}" for k, v in dist.items())
            got_s = "、".join(f"{k}×{v}" for k, v in got.items()) or "(空)"
            defects.append(f"题型分布不符：要求 {want_s}，实出 {got_s}")

    if knobs.get("difficulty_plan") == PLAN_INCREASING and len(items) >= 2:
        diffs = [_to_int(it.get("difficulty")) for it in items]
        md = _to_int(mother_difficulty)
        if md is not None:
            # 与闸A/GENERATE_PROMPT 同一把尺：逐项比对预期档
            expected = [min(md + i, DIFFICULTY_CAP) for i in range(len(items))]
            if diffs != expected:
                defects.append(
                    "要求难度递增但实出难度档与计划不符："
                    f"预期 {','.join(str(d) for d in expected)}，"
                    f"实出 {','.join(str(d) for d in diffs)}"
                )
        else:
            mono = all(
                a is not None and b is not None and b >= a for a, b in zip(diffs, diffs[1:])
            )
            if not mono:
                defects.append(
                    "要求难度递增但实出难度非单调不减：" + ",".join(str(d) for d in diffs)
                )
    return defects


def knobs_desc(knobs: dict[str, Any] | None) -> str:
    """纯函数：knobs → 题组头部人话描述（如「5 道·难度递增·2选择+2填空+1解答」）。空 → ""。"""
    knobs = knobs or {}
    bits: list[str] = []
    if knobs.get("count"):
        bits.append(f"{knobs['count']} 道")
    plan = knobs.get("difficulty_plan")
    if plan == PLAN_INCREASING:
        bits.append("难度递增")
    elif plan:
        bits.append(f"难度「{plan}」")
    dist = knobs.get("qtype_dist") or {}
    if dist:
        bits.append("+".join(f"{v}{k}" for k, v in dist.items()))
    if knobs.get("note"):
        bits.append(str(knobs["note"]))
    return "·".join(bits)


def gene_judge_knobs_spec(
    knobs: dict[str, Any] | None, expected_difficulty: Any = None
) -> str | None:
    """🔴 纯函数·闸A 配方对齐段：knobs 非空时注入 GENE_JUDGE_PROMPT 尾部，改判标准跟老师配方走。

    🔴 只对**产生该配方那一轮**生成的题注入（item 带 from_recipe 印记，gene_gate 把关）——
    编辑轮（再来2道简单的）新增的题不受旧配方改判，老师点名的简单补题不会被旧递增计划误警。

    - qtype_match：按老师指定题型集合判（属于配比内任一题型即 match），不再要求与母题题型一致；
    - difficulty_match：递增计划按该题 item 级 expected_difficulty 印记判（generate 落印，
      跟题走 —— remove 位移/add 追加都不会错档），不再按列表下标现算；
    - knobs 没碰题型/难度（如只给 count/note）→ 返回 None（判别标准保持现状）。
    """
    knobs = knobs or {}
    if not knobs:
        return None
    lines: list[str] = []

    dist = knobs.get("qtype_dist") or {}
    if dist:
        allowed = "/".join(dist.keys())
        lines.append(
            f"- qtype_match 改判：老师指定了题型配比（{'、'.join(f'{k}×{v}' for k, v in dist.items())}），"
            f"变式题型属于 {{{allowed}}} 之一即算 match（不再要求与母题题型一致；配比总量由程序另行校验）。"
        )

    plan = knobs.get("difficulty_plan")
    if plan == PLAN_INCREASING:
        exp = _to_int(expected_difficulty)
        if exp is not None:
            lines.append(
                f"- difficulty_match 改判：老师要求难度递增，该题预期难度档 = {exp}"
                "（该题难度为预期档即算 match，不再按 level=normal/hard 对母题判）。"
            )
    elif plan:
        lines.append(f"- difficulty_match 改判：按老师难度要求「{plan}」判，符合该要求即算 match。")

    if not lines:
        return None
    return "老师指定配方（🔴 优先于上面的判别标准）：\n" + "\n".join(lines)


def _parse_generated_items(text: str, facts: dict) -> list[dict[str, Any]]:
    """generate/重试共用：LLM 返回文本 → 规整 items（check 待 solve_explain 填）。"""
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    items: list[dict[str, Any]] = []
    for it in data or []:
        if not isinstance(it, dict):
            continue
        items.append(
            _sanitize_item(
                {
                    "stem": it.get("stem"),
                    "answer": it.get("answer"),
                    "solution": it.get("solution"),
                    "qtype": it.get("qtype") or facts["qtype"],
                    "difficulty": it.get("difficulty"),
                    "level": it.get("level") or "normal",
                    "injected_kp": it.get("injected_kp") or None,
                    # check 待 solve_explain 填（无 check 不许进 assemble）
                }
            )
        )
    return items


async def _extract_knobs(state: VariantState) -> dict[str, Any]:
    """首轮配方旋钮抽取：去掉图 URL 后的人话非空(len>3) → 一次受约束 LLM 抽取 + normalize 钳制。

    🔴 任何失败（LLM 异常/解析失败）→ {} 回落默认配方，绝不卡死出题（G5）。
    """
    user_text = _strip_urls(_latest_human_text(state.get("messages", [])))
    # 🔴 只过滤真空串：「出5道」「来5道」恰好 3 字也是完整数量指令，阈值高了会静默吞掉；
    #   抽取失败本身有 {} 兜底，不靠长度预筛。
    if not user_text:
        return {}
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=KNOBS_PROMPT.format(utterance=user_text))]
        )
    except Exception:  # noqa: BLE001 — 旋钮抽取是增强不是关卡
        return {}
    return normalize_knobs(_parse_json(text))


async def generate(state: VariantState, config: RunnableConfig) -> VariantState:
    """③ 造题：配方由首轮旋钮(knobs)驱动，无旋钮走旧默认 3 = 2 普通 + 1 难。

    🔴 入口断言 mother_confirmed 或三锚高置信（防裸奔）。
    🔴 代码级配方校验 shape_check：不符带缺陷反馈整组 retry 1 次，仍不符 → 接受 +
       shape_defects 外显到题组头部（不卡死，G5）。
    """
    if not (state.get("mother_confirmed") or _conf_ok(state.get("analysis") or {})):
        # DNA 闸未过却走到 generate（多入口兜底）→ 拒造，回 clarify 语义
        return {
            "messages": [
                AIMessage(content="母题 DNA 还没确认，我先不造题。请确认年级/考点/题型。")
            ]
        }

    # 🔴 旋钮：新母题轮 analyze 已抽好随 state 来；库内母题直进 generate（不经 analyze）→ 此处兜底抽
    knobs = state.get("knobs")
    if knobs is None:
        knobs = await _extract_knobs(state)
    _emit_stage(
        "knobs", "解析配方", "done", knobs_desc(knobs) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）"
    )

    facts = _mother_facts(state)
    mother_d = (state.get("mother_dna") or {}).get("difficulty")
    recipe = recipe_from_knobs(knobs, mother_d)
    _emit_stage("generate", "生成题目", "running", f"{recipe['n']} 道")
    prompt = (
        GENERATE_PROMPT.format(
            n=recipe["n"], n_normal=recipe["n_normal"], n_hard=recipe["n_hard"], **facts
        )
        + recipe["spec"]
    )

    # 🔴 思维外放（用户反馈 2026-06-11）：JSON token 对用户是乱码不外放，但流内数
    # "stem" 出现次数 → 思路条实时跳「正在写第 n/N 道」+ 当前题干前几个字，等待不再是黑盒。
    total_n = int(recipe["n"])
    _seen = {"n": 0}

    def _gen_progress(acc: str) -> None:
        n = min(acc.count('"stem"'), total_n)
        if n > _seen["n"]:
            _seen["n"] = n
            m = re.findall(r'"stem"\s*:\s*"([^"]{0,24})', acc)
            peek = (m[-1].replace("\\n", " ").strip() + "…") if m and m[-1] else ""
            _emit_stage("generate", "生成题目", "running", f"正在写第 {n}/{total_n} 道 {peek}")

    text = await _ainvoke_text([HumanMessage(content=prompt)], on_delta=_gen_progress)
    items = _parse_generated_items(text, facts)

    # 🔴 代码级配方校验（数量/题型分布/递增档位）：不符 → 带缺陷反馈整组 retry 1 次。
    #   含首稿解析为空（items=[] 时「要求N道实出0道」也是明确缺陷，值得一次重试）。
    defects = shape_check(items, knobs, mother_d)
    if defects:
        feedback = (
            "\n\n[配方校验反馈] 你上一稿不满足老师指定配方："
            + "；".join(defects)
            + "。请整组重出，严格满足配方（数量/题型配比/难度计划逐项核对后再输出）。"
        )
        try:
            retry_text = await _ainvoke_text([HumanMessage(content=prompt + feedback)])
            retry_items = _parse_generated_items(retry_text, facts)
        except Exception:  # noqa: BLE001 — 重试失败保留首稿（绝不卡死）
            retry_items = []
        if retry_items:
            items = retry_items
            defects = shape_check(items, knobs, mother_d)

    if not items:
        # 两稿皆空/解析失败 → 友好失败收尾（after_generate 走 done→END），绝不静默空轮
        _emit_stage("generate", "生成题目", "warn", "0 道（解析失败）")
        return {
            "items": [],
            "knobs": knobs,
            "shape_defects": defects,
            "messages": [
                AIMessage(
                    content="这一轮我没能产出可用的变式题（模型输出解析失败）。"
                    "请再发一次指令（可换种说法），或重贴题目图重试。"
                )
            ],
        }

    # 🔴 配方印记落 item 级（闸A 改判段只作用于产生该配方的这一轮生成的题）：
    #   from_recipe = 本轮按老师配方生成；expected_difficulty = 递增计划该题预期档（跟题走，
    #   remove 位移/add 追加不会错档；编辑轮新增题无印记 → 不被旧计划误改判）。
    if knobs:
        md_i = _to_int(mother_d) or 3
        increasing = knobs.get("difficulty_plan") == PLAN_INCREASING
        for i, it in enumerate(items):
            it["from_recipe"] = True
            if increasing:
                it["expected_difficulty"] = min(md_i + i, DIFFICULTY_CAP)

    _emit_stage("generate", "生成题目", "done", f"{len(items)} 道")
    return {"items": items, "knobs": knobs, "shape_defects": defects, "messages": []}


SOLVE_PROMPT = """你是严谨的数学阅卷老师。真解下面这道题（不看给定答案，独立算一遍）。

题干：{stem}

只输出 JSON：
{{"solved_answer":"你独立算出的答案","solution":"完整解题过程(含答案)",
  "kp_name":"这道题实际考的主考点","grade":"这道题适配的年级"}}

格式硬规定：solution 里数学式一律 $...$ 包裹（行间 $$...$$），禁止裸 LaTeX / \\( \\) 定界；换行用标准 \\n。"""

REGEN_PROMPT = """下面这道变式题，独立解出的答案与题面标答不一致，请**重新出一道**等价变式重做。

主考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
原题干: {stem}
要求：仍考「{kp_name}」、仍在「{grade}」、{level} 难度；换数字/场景使题面与答案自洽；
answer 优先给可计算的数值/表达式（可程序验算），数字设计成解恰好整洁。

只输出 JSON：
{{"stem":"新题干","answer":"标准答案","solution":"完整解析","qtype":"{qtype}","difficulty":{difficulty},"level":"{level}","injected_kp":{injected_kp}}}

格式硬规定：数学式一律 $...$ 包裹（行间 $$...$$），禁止裸 LaTeX / \\( \\) 定界；换行用标准 \\n。"""


# ---------------------------------------------------------------------------
# 闸B·程序验算（PRD-C-010）：LLM 只负责"人话题 → 结构化载荷"的有界抽取，
# pass/fail 判决只读 math_verify.verify()（纯 sympy，零 LLM）的 verdict。
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """你是数学验算载荷抽取器。把下面这道题的「题干 + 题面标准答案」抽成可被 sympy 程序验算的结构化载荷 JSON（验算对象 claimed = 题面标准答案）。

题型: {qtype}
题干: {stem}
题面标准答案(待验算的 claimed): {answer}
参考·另一次独立解答(仅帮助你理解答案格式，不是验算对象): {solved_answer}

载荷契约（kind 四选一；所有表达式必须是 sympy 可解析的纯 ASCII 数学串：乘号写 *、乘方写 ** 或 ^、分数写 /、根号写 sqrt()；禁止 LaTeX、中文、单位、等号外的标点）。
🔴 表达式硬边界（违反 = 程序直接拒收变 degrade，浪费一次验算机会）：
- 只允许这些函数：sqrt / Abs / Min / Max（比大小、最值题用 Min(...)/Max(...)，绝对值用 Abs()）。
- **禁止任何 Python 语法**：不许 if/else 三元、列表推导、len()/range()、布尔 True/False、比较式（< > ==）。
- 选项/claimed 必须是**可算出的数值或表达式**（如 "-sqrt(25)"、"7/2"），不是真假陈述；某选项本身不是数值（如文字判断、区间）→ 整题输出 kind:none。
- 计数类问题（"有几个是…"）若每项判定都是数值比较可写成 Min/Max/Abs 组合才抽，否则 kind:none——不要发明 Piecewise/Eq/逻辑与。
1. 方程求解: {{"kind":"equation_solve","equations":["x**2-5*x+6=0"],"unknowns":["x"],"claimed":["2","3"]}}
   （claimed = 标准答案申报的全部解；只支持单未知数。🔴 程序按"解集完全相等"判：若题目含舍根/取值范围
   约束——如分式方程验增根后舍去、几何/应用题边长必须为正——标答只保留部分解时，**不要用本 kind**
   （claimed 会少于裸方程全部解被误判 fail）：能数值验保留解就改用 kind 3 numeric，否则输出 kind:none）
2. 表达式等价(化简/展开/因式分解): {{"kind":"expr_equiv","expr_a":"题面原式","expr_b":"标准答案给的结果式"}}
3. 数值计算: {{"kind":"numeric","expr":"3*7/2","claimed":"10.5","tol":1e-6}}
4. 选择题: {{"kind":"choice","ground":{{上述1/2/3任一子载荷}},"options":{{"A":"2","B":"3"}},"claimed_correct":"B"}}
   （options 的值 = 各选项的数学值；ground = 由题干建立的真值载荷；claimed_correct = 标准答案的选项字母）

抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ 只输出 {{"kind":"none","reason":"原因"}}。
只输出 JSON（不要解释）。"""

_PAYLOAD_KINDS = {"equation_solve", "expr_equiv", "numeric", "choice"}

# 题型分流（分流键=题型；题干骨架词兜底）：证明/开放/作图类不进 sympy
_PROOF_QTYPE_RE = re.compile(r"(证明|求证|开放|作图|画图)")
_PROOF_STEM_RE = re.compile(r"(求证|请证明|证明[:：]|尺规作图)")
# 软校验骨架（代码正则级）：证明类题面应含"已知/求证/证明/作图"类结构
_PROOF_SKELETON_RE = re.compile(r"(已知|求证|证明|作图)")


def _is_proof_like(qtype: Any, stem: Any) -> bool:
    """闸B 分流键：题型含 证明/开放/作图 或 题干带求证骨架 → 不进 sympy，走软校验+人审。"""
    return bool(
        _PROOF_QTYPE_RE.search(str(qtype or "")) or _PROOF_STEM_RE.search(str(stem or ""))
    )


def _proof_struct_ok(stem: Any) -> bool:
    """轻量结构软校验：题面是否含「已知/求证/证明」类骨架（不验数学，只验结构）。"""
    s = str(stem or "")
    return len(s) >= 10 and bool(_PROOF_SKELETON_RE.search(s))


def _append_card_note(item: dict, note: str) -> None:
    """把验算标记行追加到题卡可见文本字段(solution)尾部 → UI 渲染 + 入库 analyze 同步可见。"""
    sol = str(item.get("solution") or "").rstrip()
    item["solution"] = f"{sol}\n\n> {note}" if sol else f"> {note}"


_TIER_NOTES = {
    TIER_VERIFIED: NOTE_VERIFIED_OK,
    TIER_SELF_OK: NOTE_SELF_CHECK_OK,
    TIER_PROOF: NOTE_PROOF_REVIEW,
    TIER_BOTH_LOW: NOTE_BOTH_GATES_LOW,
    # TIER_SILENT 无 note（沉默 = 不说话）
}


def _apply_visibility(item: dict) -> None:
    """4d 可见性矩阵（PRD-C-012）：check×gene 真值 → 外显 tier + badge + 题卡 note。

    🔴 只定展示层，真值不动（verify/gene 原样进 aux_tags 审计）。幂等：note 已在
    solution 里不重复追加。⚠ 仅 TIER_BOTH_LOW（双闸皆存疑）；单闸存疑 = 沉默。
    """
    chk = item.get("check") or {}
    gene_low = (item.get("gene") or {}).get("gate") == GENE_GATE_WARN
    if chk.get("review") == REVIEW_PROOF:
        # 证明类：badge=warn 表示结构软校验缺骨架（verify 侧低）
        struct_low = chk.get("badge") == "warn"
        tier = TIER_BOTH_LOW if (struct_low and gene_low) else TIER_PROOF
    elif chk.get("verify") == VERIFY_SYMPY_PASS:
        tier = TIER_VERIFIED  # 强正面（gene 即便 warn 也不外显负面——单闸沉默）
    elif chk.get("self_check") == "match":
        tier = TIER_SELF_OK  # 程序不可验但独立复算一致 → 轻正面
    else:
        # verify 侧低（unverified 且自检不一致）：gene 也低才 ⚠，否则沉默
        tier = TIER_BOTH_LOW if gene_low else TIER_SILENT
    chk["tier"] = tier
    chk["badge"] = "warn" if tier == TIER_BOTH_LOW else "ok"
    item["check"] = chk
    note = _TIER_NOTES.get(tier)
    if note and note not in str(item.get("solution") or ""):
        _append_card_note(item, note)


async def _extract_payload(
    stem: Any, answer: Any, solved_answer: Any, qtype: Any
) -> dict | None:
    """LLM 有界抽取：题干+标答 → 验算载荷 JSON。解析失败带错误反馈 retry，总共最多 2 次。

    返回 None = 抽不成/LLM 异常 → 调用方按 degrade 处理（绝不外抛，G5）。
    """
    base = EXTRACT_PROMPT.format(
        qtype=str(qtype or "解答"),
        stem=str(stem or ""),
        answer=str(answer or ""),
        solved_answer=str(solved_answer or "(无)"),
    )
    feedback = ""
    for _ in range(2):
        try:
            text = await _ainvoke_text([HumanMessage(content=base + feedback)])
        except Exception:  # noqa: BLE001 — 抽取层异常不许逃逸炸 solve_explain → degrade
            return None
        data = _parse_json(text)
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind in _PAYLOAD_KINDS:
                return data
            if kind == "none":
                return None  # LLM 明确说抽不成 → degrade，不浪费重试
        feedback = (
            "\n\n[错误反馈] 上次输出不是合法载荷 JSON（kind 必须是 "
            "equation_solve/expr_equiv/numeric/choice/none 之一，且为合法 JSON）。"
            "请严格按契约重新只输出 JSON。\n上次输出(截断)：" + (text or "")[:300]
        )
    return None


async def _machine_verify(item: dict, solved_answer: Any) -> dict:
    """程序验算一道题：抽载荷 → math_verify.verify（纯 sympy）。永不抛异常。

    返回 {"verdict": "pass"|"fail"|"degrade", "detail": str, "computed": str|None}。
    """
    payload = await _extract_payload(
        item.get("stem"), item.get("answer"), solved_answer, item.get("qtype")
    )
    if payload is None:
        return {"verdict": math_verify.DEGRADE, "detail": "载荷抽取失败/抽不成", "computed": None}
    try:
        # sympy solve/simplify 偶有耗时 → 丢线程池，不卡事件循环；
        # 🔴 wait_for 墙钟预算（G5 反挂死）：病态载荷把 sympy 拖入无界计算时按 degrade 解锁流程
        return await asyncio.wait_for(
            asyncio.to_thread(math_verify.verify, payload), timeout=VERIFY_TIMEOUT_S
        )
    except TimeoutError:  # py3.11: asyncio.TimeoutError == TimeoutError
        return {
            "verdict": math_verify.DEGRADE,
            "detail": f"验算超时（>{VERIFY_TIMEOUT_S}s），按未验算降级",
            "computed": None,
        }
    except Exception as e:  # noqa: BLE001 — verify 自身永不抛，此处纯保险
        return {"verdict": math_verify.DEGRADE, "detail": f"验算执行异常: {e}", "computed": None}


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


def _conservation_ok(solved_kp: str, solved_grade: str, facts: dict) -> bool:
    """主考点 + 年级 守恒校验（重生版仍须过；破则丢弃重生保留原版打⚠）。

    宽松包含匹配：标准考点/年级名 与 solve 回报的实际考点/年级 互含即视为守恒。
    """
    mk, mg = _norm(facts["kp_name"]), _norm(facts["grade"])
    sk, sg = _norm(solved_kp), _norm(solved_grade)
    kp_ok = (not sk) or (mk in sk) or (sk in mk) or (mk[:4] and mk[:4] in sk)
    # 年级守恒：比对编码前2字(如"七年")或互含
    grade_ok = (not sg) or (mg[:2] and mg[:2] in sg) or (mg in sg) or (sg in mg)
    return bool(kp_ok and grade_ok)


async def _solve_one(stem: str) -> dict:
    """真解一道题。🔴 LLM 调用异常吞掉返 {}（与 _extract_payload/_gene_judge_one 契约对齐）：
    瞬时网关抖动绝不外抛炸掉 solve_explain/gene_gate 节点（G5），调用方按"没解出来"降级。"""
    try:
        text = await _ainvoke_text([HumanMessage(content=SOLVE_PROMPT.format(stem=stem or ""))])
    except Exception:  # noqa: BLE001
        return {}
    solved = _parse_json(text) or {}
    # 🔴 阅卷解析会回写 item["solution"]（solve_explain 三处）—— 出口统一净化，
    # 否则 \( \) / 字面 \n 绕过 _parse_generated_items 的净化直达卡片/入库
    if isinstance(solved, dict) and solved.get("solution"):
        solved["solution"] = _sanitize_rich_text(solved["solution"])
    return solved


async def _regen_once(item: dict, facts: dict, feedback: str | None = None) -> dict | None:
    """REGEN 回炉一次：返回不带 check 的重生草稿（解析失败/LLM 异常返 None）。

    feedback（如 sympy 的 computed/detail）注回 prompt 的题干段，告诉 LLM 错在哪。
    🔴 LLM 调用异常吞掉返 None（G5）：回炉是增强不是关卡，网关抖动时调用方按
    "重生失败 → 保留原版打 ⚠/warn" 的既有降级路径走，绝不炸掉整轮出题。
    """
    stem = str(item.get("stem") or "")
    if feedback:
        stem = f"{stem}\n\n[程序验算反馈] {feedback}"
    try:
        regen_text = await _ainvoke_text(
            [
                HumanMessage(
                    content=REGEN_PROMPT.format(
                        kp_name=facts["kp_name"],
                        grade=facts["grade"],
                        stem=stem,
                        level=item.get("level") or "normal",
                        qtype=item.get("qtype") or facts["qtype"],
                        difficulty=item.get("difficulty") or 3,
                        injected_kp=json.dumps(item.get("injected_kp"), ensure_ascii=False),
                    )
                )
            ]
        )
    except Exception:  # noqa: BLE001 — 回炉 LLM 异常 → 视同重生失败（G5）
        return None
    regen = _parse_json(regen_text)
    if isinstance(regen, dict) and regen.get("stem"):
        return _sanitize_item(
            {
                "stem": regen.get("stem"),
                "answer": regen.get("answer"),
                "solution": regen.get("solution"),
                "qtype": regen.get("qtype") or item.get("qtype"),
                "difficulty": regen.get("difficulty") or item.get("difficulty"),
                "level": regen.get("level") or item.get("level"),
                "injected_kp": regen.get("injected_kp"),
            }
        )
    return None


async def solve_explain(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ solve + 闸B 程序验算（PRD-C-010）。每题真解产解析 → 题型分流 → sympy 判决：

    - 证明/开放/作图类（分流键=题型，题干骨架兜底）→ 不进 sympy，正则级结构软校验，
      标 check.review=proof_needs_human 转人审（G3/FP2，不误杀）。
    - 其余（计算/解答/填空/选择）→ LLM 抽验算载荷 → math_verify.verify（纯 sympy）：
      pass → check.verify=sympy_pass 照常；
      fail → 既有 REGEN 回炉 1 次（computed/detail 注回 prompt）→ 重生版再验（须 pass+守恒），
             🔴 仍不过 → **剔除不外发**（4d 方案A·用户拍板 2026-06-11：程序证实标答错的题
             老师永远看不到；剔除过程思路条透明叙事 + dropped_notes 进摘要）；
      degrade → 保留原有「LLM 独立解 + _norm 比对」自检作 fallback（check.self_check 记
             match/mismatch），外显层交 _apply_visibility 按 4d 矩阵定 tier/note。
    🔴 判决只读 verify() 的 verdict，永不采信 LLM 自评；任何验算环节失败均降级继续，绝不抛（G5）。
    🔴 凡进 items 的题一律过本节点，无 check 不许进 assemble（remove 后旧题带 check 原样通过）。
    """
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    out: list[dict] = []
    dropped: list[str] = []  # 本轮剔除叙事（assemble 摘要 + 思路条同步外显）

    for i, it in enumerate(items):
        item = dict(it)
        if item.get("check"):  # 已定状态（如自愈过的补题再次流经）→ 不重复
            out.append(item)
            continue

        _emit_stage("verify", "程序验算", "running", f"第 {i + 1}/{len(items)} 道")

        # ── 闸B·题型分流：证明/开放/作图 → 不进 sympy，软校验 + 人审标记 ──
        if _is_proof_like(item.get("qtype") or facts["qtype"], item.get("stem")):
            struct_ok = _proof_struct_ok(item.get("stem"))
            item["check"] = {
                "badge": "ok" if struct_ok else "warn",
                "solved_answer": None,
                "review": REVIEW_PROOF,
            }
            _apply_visibility(item)
            out.append(item)
            continue

        solved = await _solve_one(item.get("stem", ""))
        solved_answer = solved.get("solved_answer")
        # solve 产出解析（给老师当判题依据；优先用阅卷解析）
        if solved.get("solution"):
            item["solution"] = solved.get("solution")

        # ── 闸B·程序验算：判决只读 verdict（G1/G2），不再用字符串比对自判 ──
        res = await _machine_verify(item, solved_answer)
        verdict = res.get("verdict")

        if verdict == math_verify.PASS:
            item["check"] = {
                "badge": "ok",
                "solved_answer": solved_answer,
                "verify": VERIFY_SYMPY_PASS,
                "verify_detail": res.get("detail"),
                "computed": res.get("computed"),
            }
            _apply_visibility(item)
            out.append(item)
            continue

        if verdict == math_verify.FAIL:
            # sympy 判定标答真错 → 既有回炉机制重生 1 次，computed/detail 注回 prompt
            healed = None
            if MAX_HEAL >= 1:
                _emit_stage("verify", "程序验算", "warn", f"第 {i + 1} 道回炉重生中")
                feedback = (
                    f"程序(sympy)验算判定该题题面标答错误：程序算得 computed={res.get('computed')}；"
                    f"详情：{res.get('detail')}。请重新出一道题面与标答自洽、经得起程序验算的等价变式。"
                )
                draft = await _regen_once(item, facts, feedback=feedback)
                if draft:
                    resolved = await _solve_one(draft.get("stem", ""))
                    if resolved.get("solution"):
                        draft["solution"] = resolved.get("solution")
                    # 🔴 重生版仍须过守恒校验 + 程序验算双闸
                    cons = _conservation_ok(
                        resolved.get("kp_name", ""), resolved.get("grade", ""), facts
                    )
                    r_res = await _machine_verify(draft, resolved.get("solved_answer"))
                    if cons and r_res.get("verdict") == math_verify.PASS:
                        draft["check"] = {
                            "badge": "ok",
                            "solved_answer": resolved.get("solved_answer"),
                            "verify": VERIFY_SYMPY_PASS,
                            "verify_detail": r_res.get("detail"),
                            "computed": r_res.get("computed"),
                        }
                        # 🔴 闸A 标记随愈合保留（healed 整体替换不丢 gene → 入库 auxTags.gene_gate
                        # 不断档 + 后续编辑轮 gene_gate 不重判/不静默换题）；原版无 gene（如持久化
                        # 旧线程存量题）→ 按既有 skipped 语义留痕（闸A 没判过，REGEN 锁同骨架）。
                        draft["gene"] = item.get("gene") or {
                            "gate": GENE_GATE_SKIPPED,
                            "reason": "healed-in-solve",
                        }
                        healed = draft
            if healed:
                _apply_visibility(healed)
                out.append(healed)
            else:
                # 🔴 4d 方案A（PRD-C-012 用户拍板）：sympy 证实标答错、重生仍不过 → 剔除不外发。
                # 老师永远看不到错题（题卡层无负面文案需求）；剔除过程思路条透明叙事，
                # 摘要经 dropped_notes 说明本组少一道。真值（fail 详情）只进叙事/日志，不进题卡。
                _emit_stage(
                    "verify",
                    "程序验算",
                    "warn",
                    f"第 {i + 1} 道程序验出标答错误（重生一次仍未过），已剔除",
                )
                dropped.append(
                    f"1 道{item.get('qtype') or ''}题程序验出标答错误（程序算得"
                    f"「{res.get('computed')}」与标答不符，重生一次仍未过），已剔除"
                )
            continue

        # ── degrade：sympy 吃不下（载荷抽不成/超范围）→ 保留既有 LLM 自检 fallback ──
        match = _norm(solved_answer) == _norm(item.get("answer"))
        if match:
            item["check"] = {
                "badge": "ok",
                "solved_answer": solved_answer,
                "verify": VERIFY_UNVERIFIED,
                "verify_detail": res.get("detail"),
                "self_check": "match",  # 4d：独立复算一致 → 轻正面（不再打 ⚠ 未经程序验算）
            }
            _apply_visibility(item)
            out.append(item)
            continue

        # 独立解 ≠ 标答（LLM 自检）→ 既有自愈：重生 1 次 → 重解 + 守恒
        healed = None
        if MAX_HEAL >= 1:
            _emit_stage("verify", "程序验算", "warn", f"第 {i + 1} 道回炉重生中")
            draft = await _regen_once(item, facts)
            if draft:
                resolved = await _solve_one(draft.get("stem", ""))
                r_answer = resolved.get("solved_answer")
                r_match = _norm(r_answer) == _norm(draft.get("answer"))
                # 🔴 重生版仍须过守恒校验
                cons = _conservation_ok(
                    resolved.get("kp_name", ""), resolved.get("grade", ""), facts
                )
                if r_match and cons:
                    draft["solution"] = resolved.get("solution") or draft.get("solution")
                    draft["check"] = {
                        "badge": "ok",
                        "solved_answer": r_answer,
                        "verify": VERIFY_UNVERIFIED,
                        "self_check": "match",
                    }
                    # 🔴 闸A 标记随愈合保留（同 FAIL 自愈路径：不丢 gene、不被编辑轮重判）
                    draft["gene"] = item.get("gene") or {
                        "gate": GENE_GATE_SKIPPED,
                        "reason": "healed-in-solve",
                    }
                    _apply_visibility(draft)
                    healed = draft

        if healed:
            out.append(healed)
        else:
            # 守恒破 或 重生仍不过 → 保留（程序没证明它错，只是没把握）。
            # 4d：verify 侧低 → 单闸沉默 / gene 也低 → ⚠（_apply_visibility 矩阵裁决）
            item["check"] = {
                "badge": "warn",
                "solved_answer": solved_answer,
                "verify": VERIFY_UNVERIFIED,
                "verify_detail": res.get("detail"),
                "self_check": "mismatch",
            }
            _apply_visibility(item)
            out.append(item)

    if dropped:
        _emit_stage("verify", "程序验算", "done", f"剔除 {len(dropped)} 道，保留 {len(out)} 道")
    else:
        _emit_stage("verify", "程序验算", "done")
    return {"items": out, "dropped_notes": dropped, "messages": []}


# ---------------------------------------------------------------------------
# 闸A·基因闸（验"是不是平行题"）：generate / exec_regenerate / exec_add 产出新变式后、
# solve_explain 之前过本闸。与闸B（答案对不对）正交：本闸只比 DNA 基因，不验数学。
# 🔴 宏观控制流仍是确定性 DAG —— 闸A是图上固定节点，LLM 只产 judge JSON，
#    pass/rework 判决 = 纯函数 gene_gate_decision（可单测），不采信 LLM 自评流程走向。
# ---------------------------------------------------------------------------
GENE_JUDGE_PROMPT = """你是平行题基因比对器。对照母题，判断下面这道变式是否是母题的「平行题」：骨架基因(题型/难度/解法结构)必须一致，皮肤(数字/场景)必须已换。

母题摘要：
- 主考点: {kp_name} / 年级: {grade}
- 题型: {qtype} / 难度: {difficulty}
- 解法骨架: {skeleton}
- 题干: {mother_stem}

变式题（申报 level={level} / 题型 {v_qtype} / 难度 {v_difficulty}）：
{variant_stem}

判别标准：
- qtype_match: 变式题型与母题一致。
- difficulty_match: level=normal 应与母题难度相同；level=hard 允许且应当比母题高一档（恰高一档算 match；高两档以上或反而变简单不算）。
- structure_match: 解法主结构/骨架与母题一致（难题在主骨架外综合一个相邻考点不算破坏结构）。
- surface_swapped: 数字/场景至少一类已换；与母题题面几乎相同(只是复读) = false。

只输出一个 JSON（不要解释）：
{{"qtype_match": true/false, "difficulty_match": true/false, "structure_match": true/false, "surface_swapped": true/false, "reason": "一句话依据"}}"""

# 骨架基因键（必须全 match）；皮肤键（必须 swapped）
_GENE_SKELETON_KEYS = ("qtype_match", "difficulty_match", "structure_match")


def _clip(s: Any, n: int = 300) -> str:
    """截断长文本（控制 judge prompt 预算：只给摘要+题干，不塞全解析）。"""
    t = str(s or "")
    return t if len(t) <= n else t[:n] + "…"


def _gene_bool(v: Any) -> bool:
    """宽容布尔：LLM 偶发吐字符串 "true"/"false" 也能吃；其余非真值一律 False（偏保守→rework）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "是")
    if isinstance(v, (int, float)):
        return bool(v)
    return False


def gene_gate_decision(judge: dict) -> Literal["pass", "rework"]:
    """🔴 闸A 纯函数判决（零 LLM / 零 IO，可单测）。

    - 骨架基因（题型/难度/结构）任一不 match → rework（一变就不是平行题）；
    - surface_swapped=False → rework（皮肤没换 = 复读母题）；
    - 键缺失/类型怪 → 按 False 处理（保守偏 rework，绝不静默放行）。
    """
    skeleton_ok = all(_gene_bool(judge.get(k)) for k in _GENE_SKELETON_KEYS)
    surface_ok = _gene_bool(judge.get("surface_swapped"))
    return "pass" if (skeleton_ok and surface_ok) else "rework"


def _gene_feedback(judge: dict) -> str:
    """rework 时给 REGEN 回炉的人话反馈：点名哪条基因不过 + LLM 的 reason。"""
    fails = []
    if not _gene_bool(judge.get("qtype_match")):
        fails.append("题型变了")
    if not _gene_bool(judge.get("difficulty_match")):
        fails.append("难度档不对")
    if not _gene_bool(judge.get("structure_match")):
        fails.append("解法结构变了")
    if not _gene_bool(judge.get("surface_swapped")):
        fails.append("皮肤没换(数字/场景照抄母题)")
    reason = str(judge.get("reason") or "").strip()
    return (
        "基因闸判定与母题平行度不足：" + ("、".join(fails) or "未知")
        + (f"（{reason}）" if reason else "")
        + "。请重出一道：题型/难度档/解法结构与母题保持一致，只换数字与场景。"
    )


def _gene_judge_prompt(item: dict, facts: dict) -> str:
    """组装基因比对 prompt（纯函数，可单测格式化不炸）。

    🔴 闸A·配方对齐：facts 带 knobs_spec（gene_gate 按 state.knobs + 题序注入）时追加
    「老师指定配方」改判段 —— qtype 按老师指定题型集合判、difficulty 按递增计划预期档位判，
    否则递增难度的合法变式会被按母题基准误判 rework。knobs 为空 → 无此段，判别标准保持现状。
    """
    prompt = GENE_JUDGE_PROMPT.format(
        kp_name=facts["kp_name"],
        grade=facts["grade"],
        qtype=facts["qtype"],
        difficulty=facts.get("mother_difficulty") or "?",
        skeleton=_clip(facts.get("skeleton"), 200),
        mother_stem=_clip(facts.get("stem"), 300),
        level=item.get("level") or "normal",
        v_qtype=item.get("qtype") or "?",
        v_difficulty=item.get("difficulty") or "?",
        variant_stem=_clip(item.get("stem"), 300),
    )
    spec = facts.get("knobs_spec")
    if spec:
        prompt += "\n\n" + str(spec)
    return prompt


async def _gene_judge_one(item: dict, facts: dict) -> dict | None:
    """一次轻量 LLM 基因比对（受约束 JSON）。调用失败/解析失败返 None（调用方按 skipped 放行）。"""
    try:
        text = await _ainvoke_text([HumanMessage(content=_gene_judge_prompt(item, facts))])
    except Exception:  # noqa: BLE001 — 闸A是增强不是关卡：judge 异常绝不外抛卡死出题（G5）
        return None
    data = _parse_json(text)
    return data if isinstance(data, dict) else None


async def gene_gate(state: VariantState, config: RunnableConfig) -> VariantState:
    """闸A 节点：对每道**未判过基因**的变式做母题基因比对。

    - pass → item.gene={gate:"pass"}；
    - rework → 带 reason 走既有 _regen_once 回炉 1 次 → 重生版再判：
        re_judge 过 **且 代码级 _conservation_ok 守恒校验过**（主考点+年级硬守恒贯穿重生，
          不只采信 LLM）→ 换用重生版（gene=pass，🔴 不带 check → 下游 solve_explain 闸B 必判，正交不破）；
        仍不过/守恒破/重生失败 → 保留原版 gene={gate:"warn", reason}（v1 只警示不硬拦）；
    - judge 失败 → gene={gate:"skipped"} 放行。
    🔴 已带 gene 的题（exec_add 追加时的旧题等）原样通过，不重判不重复花预算。
    """
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    # 🔴 闸A·配方对齐：只对带 from_recipe 印记的题（产生该配方那一轮生成的）注入改判段，
    #   递增预期档读 item 级 expected_difficulty（generate 落印，跟题走不随下标位移）。
    #   编辑轮 add 的新题无印记 → 按默认标准判，绝不被旧配方（如「5道递增」）误改判。
    knobs = state.get("knobs") or {}
    out: list[dict] = []

    for i, it in enumerate(items):
        item = dict(it)
        if item.get("gene"):  # 已判过 → 不重判（旧题预算保护）
            out.append(item)
            continue

        facts_i = facts
        if knobs and item.get("from_recipe"):
            spec = gene_judge_knobs_spec(knobs, item.get("expected_difficulty"))
            if spec:
                facts_i = dict(facts, knobs_spec=spec)

        _emit_stage("gene_gate", "平行度比对", "running", f"第 {i + 1}/{len(items)} 道")
        judge = await _gene_judge_one(item, facts_i)
        if judge is None:
            item["gene"] = {"gate": GENE_GATE_SKIPPED}
            out.append(item)
            continue

        if gene_gate_decision(judge) == "pass":
            item["gene"] = {"gate": GENE_GATE_PASS}
            out.append(item)
            continue

        # rework：既有回炉重生 1 次（基因反馈注回 prompt）→ 重生版再判一次
        _emit_stage("gene_gate", "平行度比对", "warn", f"第 {i + 1} 道回炉重生中")
        draft = await _regen_once(item, facts_i, feedback=_gene_feedback(judge))
        if draft:
            re_judge = await _gene_judge_one(draft, facts_i)
            if re_judge is not None and gene_gate_decision(re_judge) == "pass":
                # 🔴 重生稿接受前仍须过**代码级**守恒闸（主考点+年级硬守恒贯穿"重生"，
                # 不能只采信 LLM re_judge）：破守恒 → 丢弃重生稿，落下方"保留原版打 warn"。
                resolved = await _solve_one(draft.get("stem", ""))
                if _conservation_ok(
                    resolved.get("kp_name", ""), resolved.get("grade", ""), facts
                ):
                    draft["gene"] = {"gate": GENE_GATE_PASS}
                    out.append(draft)  # 不带 check → solve_explain（闸B）必判
                    continue

        # 回炉仍不过 → 保留原版打 warn（不拦截入库，aux_tags + 题卡可见警示）
        item["gene"] = {
            "gate": GENE_GATE_WARN,
            "reason": str(judge.get("reason") or "").strip() or None,
        }
        out.append(item)

    _emit_stage("gene_gate", "平行度比对", "done")
    return {"items": out, "messages": []}


def _status_summary(items: list[dict]) -> str:
    """按 4d tier 汇总状态短语（只说好：正面/中性计数，⚠ 仅双闸低）。旧数据无 tier 不计入。"""
    tiers = [(it.get("check") or {}).get("tier") for it in items]
    parts = []
    if n := tiers.count(TIER_VERIFIED):
        parts.append(f"{n} 道程序验算通过")
    if n := tiers.count(TIER_SELF_OK):
        parts.append(f"{n} 道已独立复算一致")
    if n := tiers.count(TIER_PROOF):
        parts.append(f"{n} 道证明类已转人工复核")
    if n := tiers.count(TIER_BOTH_LOW):
        parts.append(f"{n} 道需重点核对（题卡已标注）")
    return "、".join(parts) or f"{len(items)} 道已生成"


async def assemble(state: VariantState, config: RunnableConfig) -> VariantState:
    """题组摘要（P1 聊天瘦身·PRD-C-012）：左栏只发摘要头——配方/守恒DNA/状态计数/旋钮提示；
    题干/答案/解析全文**只走右栏 artifact 题卡**，聊天流不再复读（用户拍板 2026-06-11）。"""
    items = state.get("items") or []
    facts = _mother_facts(state)

    # 配方外显：有 knobs → "按你的要求: ..."；无 → 旧默认文案（行为不变）
    desc = knobs_desc(state.get("knobs"))
    recipe_s = f"按你的要求：{desc}" if desc else "配方：默认 3 = 2 普通 + 1 难"
    # 代码级配方校验缺陷（generate 整组 retry 1 次后仍不符）→ 头部外显 ⚠，不拦截
    defects = state.get("shape_defects") or []
    defect_s = ("\n\n⚠ 配方未完全满足：" + "；".join(defects)) if defects else ""
    # 4d 方案A：被剔除题的摘要说明（过程已在思路条叙事，这里收口"本组为何少了"）
    dropped = state.get("dropped_notes") or []
    dropped_s = ("\n\n" + "；".join(dropped)) if dropped else ""

    head = (
        f"## 举一反三 · {len(items)} 道变式（{recipe_s}）{defect_s}{dropped_s}\n\n"
        f"**母题 DNA**：考点「{facts['kp_name']}」· 年级「{facts['grade']}」· 题型「{facts['qtype']}」（硬守恒）\n\n"
        f"**状态**：{_status_summary(items)}\n\n"
        "题目详情见右侧题卡。旋钮可拨：数量 / 数字 / 场景 / 难度 / 题型(可配比) / 解法。"
        "说「这组可以了」即入库。"
    )
    # artifact 快照帧（PRD-C-011）：每轮题组变化都过 assemble → FE 题卡每轮拿最新快照
    _emit_artifact(state)
    return {"messages": [AIMessage(content=head)]}


# ===========================================================================
# 交互层（设计 §6）：多轮 WAIT → parse 判 5 意图 → 三层漏斗分诊
#   修正 / 编辑(remove·regenerate·add) / 确认 / 答疑 / clarify
# ===========================================================================
# --- 受约束分类器（PRD-C-010 G4/FP4）：intent 闭集枚举 + 物理护栏 validate_instruction ---
INTENT_REVISE = "修正"  # 任务口径里的 revise
INTENT_EDIT = "编辑"  # remove/regenerate/add 细分在 ops.action（任务口径 remove|regenerate|add）
INTENT_CONFIRM = "确认"  # 任务口径里的 confirm
INTENT_QA = "答疑"  # 任务口径里的 qa
INTENT_CLARIFY = "clarify"
VALID_INTENTS = {INTENT_REVISE, INTENT_EDIT, INTENT_CONFIRM, INTENT_QA, INTENT_CLARIFY}
EDIT_ACTIONS = {"remove", "regenerate", "add"}
ADD_COUNT_MAX = 5  # 与 exec_add 单轮上限同口径（min(n,5)），护栏在源头就钳掉

PARSE_PROMPT = """你是举一反三 agent 的指令解析器（受约束分类器：intent 只能从 5 个枚举值里选 1 个，禁止发明新值）。老师正在看一组已出的变式题（共 {n} 道），下面是他的最新一句话。

母题 DNA（硬守恒，老师不能改这两项，撞它即 clarify 驳回）：
- 主考点: {kp_name}
- 年级: {grade}

老师最新一句话：
{utterance}

【分类标准】逐条对照，命中哪条选哪条；都不命中选 "clarify"（其中"编辑"细分 3 种 action）：
1. "答疑" —— 判别：只是在问某题怎么解/为什么，不要求改动任何题。例：「为什么第3题选B？」
2. "编辑"+remove —— 判别：点名删掉某道题，且能给出 1~{n} 内的题号。例：「第2题删掉」→ ops=[{{"action":"remove","index":2}}]
3. "编辑"+regenerate —— 判别：点名重出/换掉/改造某道题，且能给出 1~{n} 内的题号。例：「第1题换一道，数字简单点」→ ops=[{{"action":"regenerate","index":1,"note":"数字简单点"}}]
4. "编辑"+add —— 判别：要求再加 N 道题（N 为正整数，单轮最多 5）。例：「再来2道难的」→ ops=[{{"action":"add","count":2,"note":"难的"}}]
5. "修正" —— 判别：老师纠正的是母题的年级或考点本身（不是改某道变式）。例：「这其实是八年级的题」→ mother_correction={{"grade":"八年级","kp":null}}
6. "确认" —— 判别：老师对这组题满意，要入库/保存/结束。例：「这组可以了，入库吧」
7. "clarify" —— 判别：撞硬守恒（要换主考点/改年级）、题号给不出或超出 1~{n}、或意图真说不清。例：「改成考函数的题」（撞守恒）

只输出一个 JSON（不要解释、不要 markdown fence）：
{{
  "intent": "修正|编辑|确认|答疑|clarify",   // 5 选 1
  "ops": [                                  // intent=编辑 时的操作列表（其余为空数组）
    {{"action":"remove|regenerate|add", "index": 1, "count": 1, "note":"自由约束/旋钮说明"}}
  ],
  "knobs": {{"count":null, "number":null, "scene":null, "difficulty":null, "qtype":null, "method":null}},
  "comp": "可被旋钮吸收的软约束(超旋钮但 best-effort 能顺的)，没有填 null",
  "extra_constraints": ["其余自由约束句"],
  "mother_correction": {{"grade":null, "kp":null}},  // intent=修正 时老师纠正的年级/考点，否则全 null
  "confidence": 0.0~1.0
}}

硬约束（违反任何一条，程序护栏会把你的输出整体降级为 clarify）：
- intent 只能是上述 5 个枚举值之一；ops.action 只能是 remove/regenerate/add。
- remove/regenerate 的 index 从 1 起、必须 ≤ {n}；拿不准题号时 ops 留空、intent 取 "clarify"。
- add 的 count 必须是正整数；intent=答疑/确认/修正/clarify 时 ops 必须为空数组。
- 解析不出来 = "clarify"，绝不猜成删题。"""


def _items_brief(items: list[dict]) -> str:
    """给 parse / answer 当上下文：每题 index + 题干前 80 字。"""
    lines = []
    for i, it in enumerate(items):
        stem = (it.get("stem") or "").replace("\n", " ")[:80]
        lines.append(f"第{i + 1}题: {stem}")
    return "\n".join(lines)


def _to_int(v: Any) -> int | None:
    """宽容转 int（str/float 可转则转），失败返 None。bool 不算数字。"""
    if isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def validate_instruction(parsed: Any, current_item_count: int) -> dict[str, Any]:
    """🔴 物理护栏（PRD-C-010 G4/FP4）：把 LLM 分类器输出钳成受约束载荷。

    纯函数（零 LLM、零 IO，可单测）。输入 = _parse_json 的产物（可能为 None/非 dict），
    输出 = 规整后的 pending 雏形（不含 utterance，由 parse_instruction 补）。

    规则表：
    - R0 解析失败/非 dict → 整体降级 clarify（永不默认成 remove）。
    - R1 intent 不在 VALID_INTENTS 白名单 → clarify。
    - R2 ops 白名单清洗：仅保留 action∈EDIT_ACTIONS 的 dict op；index/count 强转 int。
    - R3 remove/regenerate：index 必须给出且 ∈ [1, current_item_count]，
         任何一个越界/缺失 → 整体降级 clarify（让 agent 反问而不是乱删）。
    - R4 add：count 缺失/非法/≤0 → 钳为 1；> ADD_COUNT_MAX → 钳为 ADD_COUNT_MAX。
    - R5 intent=编辑 但 ops 清洗后为空 → clarify。
    - R6 答疑/确认/修正/clarify → ops 强制清空（物理保证答疑/确认带不动编辑 op；
         answer_question 本身 return 不含 items，双保险不破坏）。
    - R7 同句多**类**操作（如 remove+add）→ 整体降级 clarify 请老师分句说：
         执行层（dispatch→exec_*）单轮只走一类分支，混类会被静默丢弃半截，且
         remove 后题号位移会使同句其它 index 失效 —— 与 R3「绝不部分执行」同哲学。
         同类多 op（删第2、3题 / 重出第1、2题）合法保留，exec_* 一次吃完。
    """

    def _normalized(p: dict) -> dict[str, Any]:
        return {
            "intent": p.get("intent"),
            "ops": [],
            "knobs": p.get("knobs") if isinstance(p.get("knobs"), dict) else {},
            "comp": p.get("comp"),
            "extra_constraints": (
                list(p.get("extra_constraints"))
                if isinstance(p.get("extra_constraints"), list)
                else []
            ),
            "mother_correction": (
                p.get("mother_correction") if isinstance(p.get("mother_correction"), dict) else {}
            ),
            "confidence": p.get("confidence"),
        }

    def _clarify(base: dict[str, Any] | None = None) -> dict[str, Any]:
        out = base if base is not None else _normalized({})
        out["intent"] = INTENT_CLARIFY
        out["ops"] = []
        return out

    if not isinstance(parsed, dict):  # R0
        return _clarify()

    base = _normalized(parsed)
    intent = base["intent"]
    if intent not in VALID_INTENTS:  # R1
        return _clarify(base)

    if intent != INTENT_EDIT:  # R6：非编辑意图物理上带不动 ops
        return base

    # intent = 编辑：清洗 ops（R2~R5）
    raw_ops = parsed.get("ops") if isinstance(parsed.get("ops"), list) else []
    ops: list[dict[str, Any]] = []
    for op in raw_ops:
        if not isinstance(op, dict):
            continue
        action = op.get("action")
        if action not in EDIT_ACTIONS:  # R2
            continue
        clean: dict[str, Any] = {"action": action}
        if op.get("note"):
            clean["note"] = str(op.get("note"))
        if action in ("remove", "regenerate"):
            idx = _to_int(op.get("index"))
            if idx is None or not (1 <= idx <= current_item_count):
                return _clarify(base)  # R3：越界/缺号 → 反问，绝不乱删
            clean["index"] = idx
        else:  # add
            cnt = _to_int(op.get("count"))
            if cnt is None or cnt <= 0:
                cnt = 1  # R4 下限
            clean["count"] = min(cnt, ADD_COUNT_MAX)  # R4 上限
        ops.append(clean)
    if not ops:
        return _clarify(base)  # R5
    if len({op["action"] for op in ops}) > 1:
        return _clarify(base)  # R7：混类操作绝不部分执行 → 反问分句
    base["ops"] = ops
    return base


async def parse_instruction(state: VariantState, config: RunnableConfig) -> VariantState:
    """WAIT 下一句 → 判 5 意图 + 三层漏斗参数。本节点只解析、不改 items（路由后各分支执行）。

    解析结果塞 state['pending']（含 intent/ops/knobs/comp/extra_constraints/mother_correction）。
    """
    utterance = _latest_human_text(state.get("messages", []))
    facts = _mother_facts(state)
    items = state.get("items") or []
    prompt = PARSE_PROMPT.format(
        n=len(items),
        kp_name=facts["kp_name"],
        grade=facts["grade"],
        utterance=utterance or "(空)",
    )
    # 🔴 17 号修复 §5：在途母题（停在 clarify、还没出题）的回答语境 —— 此时没有可编辑的
    # 题号，老师这句大概率是在答我们的澄清提问（年级/考点/题型）。显式钉死语境，防止
    # 分类器按「已有题组」惯性乱派编辑 op（编辑类 index 也会被 R3 护栏拦，双保险）。
    # ⚠ 覆盖必须盖过头部「硬守恒」段（真机踩过：答「9年级上」被判成撞守恒 clarify 驳回）：
    # 无题组语境下年级/考点不是守恒项，正是我们在求老师确认的待定项。
    if not items:
        prompt += (
            "\n\n【当前语境·最高优先级，覆盖上面所有规则】这一轮还没有出任何题（题组为空）——"
            "我们刚就母题的年级/考点/题型向老师提了澄清问题，老师这句话是**回答澄清**。"
            "此语境下上面『母题 DNA 硬守恒，撞它即 clarify 驳回』的规则**不适用**："
            "年级/考点正是待老师确认/纠正的项，不存在『改守恒』一说。\n"
            "判定规则（按此覆盖执行）：\n"
            "- 给出年级（如「这个是9年级上的题目」「八下的」）→ intent=修正，"
            "mother_correction.grade=规范化年级（如「九年级上学期」）。\n"
            "- 给出考点（如「考的是二次函数」）→ intent=修正，mother_correction.kp=该考点。\n"
            "- 同时给年级和考点 → 修正，两项都填。\n"
            "- 真说不清（与年级/考点/题型无关的闲聊）→ clarify。\n"
            "- 禁止输出任何编辑类 ops（无题可编），禁止判「确认/答疑」。"
        )
    text = await _ainvoke_text([HumanMessage(content=prompt)])
    parsed = _parse_json(text)
    # 🔴 物理护栏（G4/FP4）：白名单 + 越界钳制 + 解析失败整体降级 clarify（永不默认成 remove）
    pending = validate_instruction(parsed, len(items))
    pending["utterance"] = utterance
    return {"pending": pending, "messages": []}


def route_after_parse(
    state: VariantState,
) -> Literal["patch", "dispatch", "answer", "save", "ask_clarify"]:
    """parse 后分诊（设计 §3 mermaid）：修正→patch / 编辑→dispatch / 确认→save / 答疑→answer / clarify。"""
    intent = (state.get("pending") or {}).get("intent")
    if intent == "修正":
        return "patch"
    if intent == "编辑":
        return "dispatch"
    if intent == "确认":
        return "save"
    if intent == "答疑":
        return "answer"
    return "ask_clarify"


def dispatch(state: VariantState) -> Literal["remove", "regenerate", "add", "ask_clarify"]:
    """三层漏斗收口：硬旋钮命中 → remove/regenerate/add；ops 空/说不清 → clarify。"""
    ops = (state.get("pending") or {}).get("ops") or []
    actions = {str(op.get("action")) for op in ops if isinstance(op, dict)}
    # 🔴 护栏 R7（validate_instruction）保证 ops 只含单一 action 类（混类已降级 clarify），
    # 此处仅作收口映射；优先级序留作防御（万一上游漏钳也不会静默丢弃后执行半截）。
    if "remove" in actions:
        return "remove"
    if "regenerate" in actions:
        return "regenerate"
    if "add" in actions:
        return "add"
    return "ask_clarify"


# --- 答疑：只问不改（🔴 物理上 return 不含 items） -----------------------------
ANSWER_PROMPT = """你是数学老师，老师对下面这组变式题的某道有疑问，请耐心解惑（讲思路/为什么这么解）。

题组：
{brief}

各题答案/解析摘要：
{detail}

老师的问题：{question}

直接用人话回答（数学式一律 $...$ 包裹）。只解惑，不要改题、不要重出题。"""


async def answer_question(state: VariantState, config: RunnableConfig) -> VariantState:
    """答疑侧循环（设计 §6）：解老师对解析的疑问 → 回等待。

    🔴 代码层物理不写 state.items：return 的 update 不含 'items' 键，即便 parse 误判进此分支也改不了题。
    """
    items = state.get("items") or []
    question = (state.get("pending") or {}).get("utterance") or _latest_human_text(
        state.get("messages", [])
    )
    detail = "\n".join(
        f"第{i + 1}题 答案:{(it.get('answer') or '')[:60]} 解析:{(it.get('solution') or '')[:120]}"
        for i, it in enumerate(items)
    )
    # 🔴 public_stream：答疑是纯人话输出，token 不打 skip_stream → 前端打字机逐字外放
    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ANSWER_PROMPT.format(
                    brief=_items_brief(items), detail=detail, question=question or "(空)"
                )
            )
        ],
        public_stream=True,
    )
    # 🔴 只返回 messages（绝不含 items），并清空 pending
    return {"messages": [AIMessage(content=text or "我没太理解你的疑问，可以再说具体点吗？")], "pending": None}


# --- 编辑·remove：删 + 重编号（删完的题不再过 solve，直接 ASM） -----------------
async def exec_remove(state: VariantState, config: RunnableConfig) -> VariantState:
    items = list(state.get("items") or [])
    ops = (state.get("pending") or {}).get("ops") or []
    drop = set()
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "remove":
            idx = op.get("index")
            try:
                drop.add(int(idx) - 1)  # 1-based → 0-based
            except (TypeError, ValueError):
                pass
    kept = [it for i, it in enumerate(items) if i not in drop]
    # 🔴 shape_defects 只属于 generate 当轮：老师显式编辑 = 对配方的人工接管，旧缺陷清单
    #   不再陈旧外显（且不在 assemble 重算 —— 那会把老师主动删/换题误报为缺陷）。
    return {"items": kept, "pending": None, "shape_defects": [], "messages": []}


# --- 编辑·regenerate：改造指定题 → 过 solve_explain（清 check 触发重判） ----------
async def exec_regenerate(state: VariantState, config: RunnableConfig) -> VariantState:
    """改造某道（按 note 软约束）→ 该题清 check 重入 solve_explain（每题状态须重新定）。"""
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    ops = (state.get("pending") or {}).get("ops") or []
    targets = []
    notes: dict[int, str] = {}
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "regenerate":
            try:
                t = int(op.get("index")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= t < len(items):
                targets.append(t)
                if op.get("note"):
                    notes[t] = str(op.get("note"))

    for t in targets:
        old = items[t]
        regen_text = await _ainvoke_text(
            [
                HumanMessage(
                    content=REGEN_PROMPT.format(
                        kp_name=facts["kp_name"],
                        grade=facts["grade"],
                        stem=(old.get("stem") or "")
                        + (f"\n额外要求：{notes[t]}" if t in notes else ""),
                        level=old.get("level") or "normal",
                        qtype=old.get("qtype") or facts["qtype"],
                        difficulty=old.get("difficulty") or 3,
                        injected_kp=json.dumps(old.get("injected_kp"), ensure_ascii=False),
                    )
                )
            ]
        )
        regen = _parse_json(regen_text)
        if isinstance(regen, dict) and regen.get("stem"):
            # 🔴 新题清 check → 必过 solve_explain 才能进 assemble（不变量）
            new_item: dict[str, Any] = _sanitize_item(
                {
                    "stem": regen.get("stem"),
                    "answer": regen.get("answer"),
                    "solution": regen.get("solution"),
                    "qtype": regen.get("qtype") or old.get("qtype") or facts["qtype"],
                    "difficulty": regen.get("difficulty") or old.get("difficulty"),
                    "level": regen.get("level") or old.get("level") or "normal",
                    "injected_kp": regen.get("injected_kp"),
                }
            )
            # 配方印记跟题走（与 difficulty 同理：重出仍占原计划槽位，闸A 改判段不丢）
            for k in ("from_recipe", "expected_difficulty"):
                if old.get(k) is not None:
                    new_item[k] = old[k]
            items[t] = new_item
    # 🔴 编辑轮清陈旧缺陷外显（同 exec_remove 注释）
    return {"items": items, "pending": None, "shape_defects": [], "messages": []}


# --- 编辑·add：生成 N 道新题（吸收软约束/旋钮）→ 过 solve_explain ----------------
ADD_PROMPT = """你是浙教版初中数学命题专家。基于母题 DNA，**新增** {n} 道举一反三变式。

母题 DNA：
- 主考点(硬守恒): {kp_name}
- 年级(硬守恒): {grade}
- 题型(默认): {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}

老师的补充要求（best-effort 吸收，撞守恒的忽略）：{extra}

铁律：每道仍考「{kp_name}」、仍在「{grade}」；只换数字/场景（除非老师明确要改难度/题型）。

只输出 JSON 数组(不要解释)，每元素：
{{"stem":"题干","answer":"标准答案","solution":"完整解析","qtype":"选择/填空/解答","difficulty":1~5,"level":"normal/hard","injected_kp":"相邻kp名或null"}}

格式硬规定：数学式一律 $...$ 包裹（行间 $$...$$），禁止裸 LaTeX / \\( \\) 定界；换行用标准 \\n。"""


async def exec_add(state: VariantState, config: RunnableConfig) -> VariantState:
    """补题（设计 §5 三层漏斗②：超旋钮软约束 best-effort 吸收 + 外显）→ 追加 → 过 solve_explain。"""
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    pending = state.get("pending") or {}
    ops = pending.get("ops") or []
    n = 0
    notes = []
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "add":
            try:
                n += int(op.get("count") or 1)
            except (TypeError, ValueError):
                n += 1
            if op.get("note"):
                notes.append(str(op.get("note")))
    if n <= 0:
        n = 1
    n = min(n, 5)  # 单轮补题上限，防失控

    extra_bits = notes + list(pending.get("extra_constraints") or [])
    if pending.get("comp"):
        extra_bits.append(str(pending["comp"]))
    extra = "；".join(extra_bits) or "无"

    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ADD_PROMPT.format(
                    n=n,
                    kp_name=facts["kp_name"],
                    grade=facts["grade"],
                    qtype=facts["qtype"],
                    stem=facts["stem"],
                    skeleton=facts["skeleton"],
                    extra=extra,
                )
            )
        ]
    )
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    for it in data or []:
        if not isinstance(it, dict):
            continue
        # 🔴 新题不带 check → 下游 solve_explain 必判
        items.append(
            {
                "stem": it.get("stem"),
                "answer": it.get("answer"),
                "solution": it.get("solution"),
                "qtype": it.get("qtype") or facts["qtype"],
                "difficulty": it.get("difficulty"),
                "level": it.get("level") or "normal",
                "injected_kp": it.get("injected_kp") or None,
                # 不落 from_recipe 印记：补题轮的题不受首轮配方（递增/题型配比）改判
            }
        )
    # 🔴 编辑轮清陈旧缺陷外显（同 exec_remove 注释）
    return {"items": items, "pending": None, "shape_defects": [], "messages": []}


# --- 修正：patch 母题字段 → 只重算受影响下游（设计 §6 中途修正） ----------------
async def patch(state: VariantState, config: RunnableConfig) -> VariantState:
    """老师纠正年级/考点 → patch analysis；改年级/考点 → 清 items 触发重锚+重造（route_after_patch）。

    粒度 = 步边界：改年级/考点 = 重走 classify→generate（清 items + mother_confirmed）；
    其余（如只是补充场景偏好）= 当软约束，留待下次编辑指令，不在此重造。
    """
    analysis = dict(state.get("analysis") or {})
    corr = (state.get("pending") or {}).get("mother_correction") or {}
    changed = False

    if corr.get("grade"):
        g = dict(analysis.get("grade") or {})
        g["value"] = corr["grade"]
        g["confidence"] = 0.9  # 老师明示 → 高置信
        g.pop("code", None)  # 清旧编码，重锚时再填
        analysis["grade"] = g
        changed = True
    if corr.get("kp"):
        k = dict(analysis.get("kp") or {})
        k["value"] = corr["kp"]
        k["confidence"] = 0.9
        k.pop("anchored", None)  # 清旧锚定，classify 会重锚
        analysis["kp"] = k
        changed = True

    if not changed:
        # 没拿到可 patch 的字段 → 退化为 clarify 回问（不空转）
        return {
            "messages": [
                AIMessage(content="我没听准你要修正什么（年级还是考点？），可以再说一次吗？")
            ],
            "pending": None,
        }

    # 🔴 改了硬锚 → 清 items + mother_confirmed，触发重锚(classify)+重造(generate)
    # artifact 同步快照（PRD-C-011）：items 已清，但后续若走 clarify（重锚置信不足）或
    # generate 裸奔兜底（无 items）会绕开 assemble 不再发帧 → 先发空快照让 FE 右栏回
    # 空态，避免老师对着 agent 端已不存在的旧题组卡片点「第N题重出」（UI/状态错位）。
    _emit_artifact({**state, "items": [], "analysis": analysis})
    return {
        "analysis": analysis,
        "items": [],
        "mother_confirmed": False,
        "pending": None,
        "messages": [AIMessage(content="收到修正，我按新的年级/考点重锚并重出这组变式。")],
    }


async def ask_clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """三层漏斗第③层 / 答非所问兜底：撞守恒 / 说不清 → 回问（设计 §6，不改 items）。"""
    pending = state.get("pending") or {}
    utterance = pending.get("utterance") or ""
    facts = _mother_facts(state)
    body = (
        "我没完全 get 到你的意思（也可能撞到了不能改的硬守恒）。\n\n"
        f"这组变式的硬守恒是：考点「{facts['kp_name']}」+ 年级「{facts['grade']}」——这两项不能改"
        "（要换考点/年级等于换一道母题，请重新贴图）。\n\n"
        "你可以这样说：\n"
        "- 删/重出/再加题（如「第 2 题重出」「再来 2 道难的」）\n"
        "- 拨旋钮（数字 / 场景 / 难度 / 题型配比）\n"
        "- 问解析（如「第 1 题为什么这么解」）\n"
        "- 「这组可以了」入库"
    )
    if utterance:
        body += f"\n\n（你刚说的是：「{utterance}」）"
    return {"messages": [AIMessage(content=body)], "pending": None}


# --- 确认入库（设计 §7）：变式+解析经 RuoYi 写老师个人题库，只写不判 -----------
async def persist_to_bank(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ 入库：老师"这组可以了" → 逐题 POST /teacher/question/create（teacher token 定 owner）。

    🔴 只写不再判（质量门已在 solve_explain 闭合）。入库回执（入库 N 道 + 失败标注）。
    🔴 owner 由后端 LoginHelper 定，body 绝不传 createBy。
    """
    items = state.get("items") or []
    facts = _mother_facts(state)
    if not items:
        return {"messages": [AIMessage(content="当前没有可入库的变式题。先贴图举一反三吧。")]}

    # 🔴 入库簿记（PRD-C-011 G5）：persisted=true 的题跳过——入库后继续编辑再说「入库」、
    # 或部分失败后重试「全部入库」时，只补未收录项，绝不把已落库的题重复 POST 落行。
    pending_idx = [i for i, it in enumerate(items) if not it.get("persisted")]
    if not pending_idx:
        return {
            "messages": [
                AIMessage(content="这组变式之前都已入库过了，没有需要补录的题（不会重复落库）。可以继续编辑或换一批。")
            ]
        }
    pending_items = [items[i] for i in pending_idx]
    n_skipped = len(items) - len(pending_items)

    # 🔴 身份透传：book-ui 经 agent_config 透传登录老师 access_token（config.configurable.ruoyi_token）
    # → 入库 owner = 该老师本人（后端 LoginHelper 取 token 身份），而非 .env 服务账号。
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    _emit_stage("persist", "入库", "running", f"{len(pending_items)} 道")
    try:
        receipts = await persist_items(pending_items, facts, token=token)
    except Exception as e:  # noqa: BLE001 — 登录/网络整体失败 → 友好兜底，不崩
        _emit_stage("persist", "入库", "warn", "连不上题库服务")
        return {
            "messages": [
                AIMessage(content=f"入库时连不上题库服务（book-server :8090 是否在跑？）：{e}")
            ]
        }

    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var_receipts = [r for r in receipts if r.get("role") != "mother"]
    ok = [r for r in var_receipts if r.get("ok")]
    fail = [r for r in var_receipts if not r.get("ok")]
    _emit_stage(
        "persist",
        "入库",
        "done" if not fail else "warn",
        f"成功 {len(ok)} 道" + (f"，失败 {len(fail)} 道" if fail else ""),
    )
    # 🔴 簿记回写 state（不是只发快照帧）：persisted 标进 items、母题雪花 id 进 mother_dna。
    # 不回写的话，下一编辑轮 assemble 重发快照 persisted 全 false → 「已收录」徽章整组回退、
    # 「全部入库」重新可点 → 二次入库整组重复落行（G5 破）；图母题也会再落一份（双份血缘）。
    new_items = [dict(it) for it in items]
    for j, r in zip(pending_idx, var_receipts):
        new_items[j]["persisted"] = bool(r.get("ok"))
    update: VariantState = {"items": new_items}
    if mother and mother.get("ok") and mother.get("id") is not None:
        # persist_items 的 mother_question_id 回填发生在局部 facts 副本上 → 这里落回 state，
        # 重试/后续入库走「母题已在库」分支，不再重复建母题
        update["mother_dna"] = dict(state.get("mother_dna") or {}, mother_question_id=mother.get("id"))

    # artifact 更新快照（PRD-C-011）：按回写后的 items 组帧（_artifact_payload 读 item.persisted）
    _emit_artifact({**state, "items": new_items})

    lines = [f"## 入库完成 · 变式 {len(pending_items)} 道，成功 {len(ok)} 道"]
    if n_skipped:
        lines.append(f"（另有 {n_skipped} 道此前已收录，本次跳过、未重复入库。）")
    # 母题(原题)入库回执：图母题不在库 → 先落原题挂血缘
    if mother and mother.get("ok"):
        lines.append(f"📌 原题(母题)已一并入库，ID：{mother.get('id')}，变式都挂在它名下（血缘可追）。")
    elif mother and not mother.get("ok"):
        lines.append(f"⚠ 原题入库失败（变式仍已落，血缘暂缺）：{mother.get('error')}")
    if ok:
        ids = [str(r.get("id")) for r in ok if r.get("id") is not None]
        lines.append("已落入你的个人题库（来源标记「举一反三」）。" + (f"变式 ID：{', '.join(ids)}" if ids else ""))
    if fail:
        lines.append(f"\n⚠ {len(fail)} 道变式入库失败：")
        for i, r in enumerate(fail, 1):
            lines.append(f"  {i}. {r.get('error')}")
        lines.append("失败的题再说一次「入库」即可只补这几道（已成功的不会重复落库）。")
    lines.append("\n可回平台「我的题库」找题、组卷、导出 PDF。")
    update["messages"] = [AIMessage(content="\n".join(lines))]
    return update


# --- 输入边界兜底（设计 §6）：没图/无在途母题/无题组 → 催图 ------------------
async def require_login(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'auth' 分支落点：登录态缺失 → 拒入图（teacher_id 绑死硬闸的提示面）。"""
    return {
        "messages": [
            AIMessage(
                content=(
                    "🔒 登录态缺失或已过期，举一反三需要绑定到你的账号才能使用"
                    "（对话记录与入库的题都归属到你本人）。请重新登录平台后再试。"
                )
            )
        ]
    }


async def ask_for_image(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'ask' 分支落点：首轮无图无母题无题组，催老师贴题图。

    🔴 必须是真节点（不能直连 END）—— 否则首轮没有任何节点产消息，回复为空，
    '没图催' 提示从未触发（PRD-C-009 G15 红的 root cause）。
    """
    return {
        "messages": [
            AIMessage(
                content=(
                    "我还没看到题目图。请先贴一张题目图的 OSS URL，我才能开始举一反三。\n\n"
                    "（贴图后我会读图、锚定年级/考点/题型，再按你要的数量出变式题。）"
                )
            )
        ]
    }


# ---------------------------------------------------------------------------
# 图（StateGraph）
# ---------------------------------------------------------------------------
graph = StateGraph(VariantState)
graph.add_node("analyze", analyze)
graph.add_node("classify", classify)
graph.add_node("clarify", clarify)
graph.add_node("generate", generate)
graph.add_node("gene_gate", gene_gate)  # 闸A·基因闸（新变式 → 平行度比对 → 闸B）
graph.add_node("solve_explain", solve_explain)
graph.add_node("assemble", assemble)
# 交互层节点（多轮 WAIT 后的下一句）
graph.add_node("parse_instruction", parse_instruction)
graph.add_node("answer_question", answer_question)
graph.add_node("exec_remove", exec_remove)
graph.add_node("exec_regenerate", exec_regenerate)
graph.add_node("exec_add", exec_add)
graph.add_node("patch", patch)
graph.add_node("ask_clarify", ask_clarify)
graph.add_node("persist_to_bank", persist_to_bank)
graph.add_node("ask_for_image", ask_for_image)
graph.add_node("require_login", require_login)

graph.set_conditional_entry_point(
    route_entry,
    {
        "analyze": "analyze",
        "generate": "generate",
        "parse": "parse_instruction",
        # 🔴 'ask' 必落真节点（ask_for_image），不能直连 END —— 否则首轮无节点产消息，回复为空
        "ask": "ask_for_image",
        # 🔴 身份硬闸：无登录态 → 提示重登（同上，必落真节点）
        "auth": "require_login",
    },
)
graph.add_edge("ask_for_image", END)
graph.add_edge("require_login", END)

# analyze：非题目图/读图失败 → 直接 END（已吐友好报错）；成功 → classify
def after_analyze(state: VariantState) -> Literal["classify", "done"]:
    # analyze 友好报错时会塞 messages（且未产 analysis）→ 结束本轮等待
    if not state.get("analysis"):
        return "done"
    return "classify"


graph.add_conditional_edges("analyze", after_analyze, {"classify": "classify", "done": END})
graph.add_conditional_edges(
    "classify", gate_after_classify, {"generate": "generate", "clarify": "clarify"}
)
graph.add_edge("clarify", END)


# generate：裸奔兜底时只吐消息、无 items → 结束；正常 → 闸A 基因闸 → 闸B solve_explain
def after_generate(state: VariantState) -> Literal["gene_gate", "done"]:
    if not state.get("items"):
        return "done"
    return "gene_gate"


graph.add_conditional_edges(
    "generate", after_generate, {"gene_gate": "gene_gate", "done": END}
)
graph.add_edge("gene_gate", "solve_explain")
graph.add_edge("solve_explain", "assemble")
graph.add_edge("assemble", END)

# --- 交互层路由（设计 §3 mermaid：WAIT → parse → 5 意图分诊） ----------------
graph.add_conditional_edges(
    "parse_instruction",
    route_after_parse,
    {
        "patch": "patch",
        "dispatch": "dispatch",  # 编辑意图 → 三层漏斗节点收口（remove/regenerate/add）
        "answer": "answer_question",
        "save": "persist_to_bank",
        "ask_clarify": "ask_clarify",
    },
)


# 三层漏斗：编辑意图 → 选 remove/regenerate/add（route_after_parse 的 "dispatch" 实由本函数收口）
def route_dispatch(
    state: VariantState,
) -> Literal["exec_remove", "exec_regenerate", "exec_add", "ask_clarify"]:
    target = dispatch(state)
    return {
        "remove": "exec_remove",
        "regenerate": "exec_regenerate",
        "add": "exec_add",
        "ask_clarify": "ask_clarify",
    }[target]


# 编辑意图先经 dispatch 漏斗：把 route_after_parse 的 "dispatch" 桥到三原语。
# 用一个轻量调度节点统一收口（避免 route_after_parse 直连 exec_remove 误派）。
graph.add_node("dispatch", lambda state: {"messages": []})
graph.add_conditional_edges(
    "dispatch",
    route_dispatch,
    {
        "exec_remove": "exec_remove",
        "exec_regenerate": "exec_regenerate",
        "exec_add": "exec_add",
        "ask_clarify": "ask_clarify",
    },
)

# 三原语收口：regenerate/add 产**新变式** → 先过闸A基因闸再到闸B；
# remove 只删不产新题 → 直连 solve_explain（旧题带 check+gene 双标，两闸都原样通过）。
graph.add_edge("exec_remove", "solve_explain")
graph.add_edge("exec_regenerate", "gene_gate")
graph.add_edge("exec_add", "gene_gate")

# 答疑/clarify → END（不改 items，回等待下一句）
graph.add_edge("answer_question", END)
graph.add_edge("ask_clarify", END)
graph.add_edge("persist_to_bank", END)


# patch：改了硬锚（清 items + mother_confirmed=False）→ 重锚重造走 classify；
#        没改（仅回问消息，items 仍在）→ END 等下一句。
def after_patch(state: VariantState) -> Literal["classify", "done"]:
    if state.get("mother_confirmed") is False and not state.get("items"):
        return "classify"
    return "done"


graph.add_conditional_edges("patch", after_patch, {"classify": "classify", "done": END})

# 🔴 不在此 compile checkpointer：service lifespan 注入 saver（按 thread_id 持久 state）
variant = graph.compile()
variant.name = "variant"
