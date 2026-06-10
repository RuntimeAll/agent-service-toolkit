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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langgraph.graph import END, MessagesState, StateGraph

from agents import conv_trace, math_verify
from agents.variant_support import anchor_subject, persist_items
from core import get_model, settings

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
VERIFY_FAIL_AFTER_REGEN = "fail_after_regen"  # 验算 fail 且回炉 1 次后仍不过
VERIFY_UNVERIFIED = "unverified"  # degrade：sympy 吃不下 → 退回 LLM 自检 fallback
REVIEW_PROOF = "proof_needs_human"  # 证明/开放/作图类：不进 sympy，转人审

# 题卡可见文本（追加在 item.solution 尾部 → assemble/_fmt_item 渲染 + 入库 analyze 字段同步可见）
NOTE_VERIFY_FAIL = "⚠ 程序验算未通过，请老师核对。"
NOTE_UNVERIFIED = "⚠ 未经程序验算。"
NOTE_PROOF_REVIEW = "⚠ 证明/开放类题不做程序验算，请老师人工审核。"

# ---------------------------------------------------------------------------
# 闸A·基因闸（验"是不是平行题"，与闸B"答案对不对"正交。依据 12-题目DNA方法论 §2/§5）：
# 骨架基因(题型/难度/结构=解法骨架)必须 match —— 一变就不是平行题；
# 皮肤变量(数字/场景)必须换 —— 没换 = 复读母题，也不算平行题。
# 判决 = 一次轻量 LLM 比对(受约束 JSON) + 纯函数 gene_gate_decision；
# rework → 既有 REGEN 回炉 1 次再判；仍不过 → gene_gate:"warn" 只警示不硬拦(v1)；
# judge 调用/解析失败 → 按 pass 放行标 "skipped"（闸A是增强不是关卡，绝不卡死流程，G5）。
# 标记落 item.gene.gate → 入库透传 auxTags.gene_gate（_apply_labels）+ 题卡 _fmt_item 可见。
# ---------------------------------------------------------------------------
GENE_GATE_PASS = "pass"  # 基因比对通过（平行题）
GENE_GATE_WARN = "warn"  # 回炉 1 次后仍不过 → 警示不拦截
GENE_GATE_SKIPPED = "skipped"  # judge 调用失败/JSON 解析失败 → 放行留痕
NOTE_GENE_WARN = "⚠ 与母题平行度存疑"


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
            "model": settings.DEFAULT_MODEL,
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


async def _ainvoke_text(messages: list[BaseMessage], retry: bool = True) -> str:
    """ainvoke + 取 content；偶发空返回重试一次。max_tokens≥4096 给思考型留头。

    🔴 每次调用落 JSONL 往返记录（_trace_llm）：发送的完整 prompt + 原始返回。
    """
    model = _model().bind(max_tokens=settings.VARIANT_MAX_TOKENS)
    label = _trace_label(messages)
    # 用户级/会话级归属：从 graph config 取 thread_id + ruoyi_token(→teacher_id)
    conf = (ensure_config() or {}).get("configurable", {}) or {}
    thread_id = conf.get("thread_id")
    teacher_id = conv_trace.teacher_id_from_token(conf.get("ruoyi_token"))
    t0 = time.monotonic()
    try:
        resp = await model.ainvoke(messages)
        text = _content_text(resp).strip()
        retried = False
        if not text and retry:
            retried = True
            resp = await model.ainvoke(messages)
            text = _content_text(resp).strip()
    except Exception as e:  # noqa: BLE001 — 记下失败往返后照常抛
        dur = int((time.monotonic() - t0) * 1000)
        _trace_llm(label, messages, "", None, dur, error=str(e))
        conv_trace.write(
            teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
            model=settings.DEFAULT_MODEL, request=_serialize_request(messages),
            response="", response_raw=None, duration_ms=dur, error=str(e),
        )
        raise
    dur = int((time.monotonic() - t0) * 1000)
    # 原始返回：content（思考型可能是 parts list）+ reasoning/usage 等附加信息（best-effort）
    raw: dict[str, Any] = {}
    try:
        raw["content"] = resp.content
        if getattr(resp, "additional_kwargs", None):
            raw["additional_kwargs"] = resp.additional_kwargs
        if getattr(resp, "response_metadata", None):
            raw["response_metadata"] = resp.response_metadata
    except Exception:
        raw = {"content": str(getattr(resp, "content", ""))}
    _trace_llm(label, messages, text, raw, dur, retried=retried)
    # 🔴 用户级对话持久化（优化基础数据源）→ 独立解耦库 conv_trace
    conv_trace.write(
        teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
        model=settings.DEFAULT_MODEL, request=_serialize_request(messages),
        response=text, response_raw=raw, duration_ms=dur, retried=retried,
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


def _conf_ok(analysis: dict[str, Any]) -> bool:
    """三锚（年级/考点/题型）任一低置信 → 闸不过。"""
    for k in ("grade", "kp", "qtype"):
        node = analysis.get(k) or {}
        if float(node.get("confidence", 0) or 0) < CONF_GATE:
            return False
    return True


# ---------------------------------------------------------------------------
# Router（入口分诊：有图? 在途母题? 库内母题跳 analyze/classify）
# ---------------------------------------------------------------------------
def route_entry(state: VariantState) -> Literal["analyze", "parse", "generate", "ask"]:
    url = _extract_image_url(_latest_human_text(state.get("messages", [])))
    # 跨轮新图 = 视作新母题（设计 §6：重走 analyze，覆盖在途状态）
    if url:
        return "analyze"
    # 老会话·纯文字（已出过题组）→ parse 分诊 5 意图（设计 §3 mermaid G0）
    if state.get("items"):
        return "parse"
    # 库内母题（已确认 DNA）→ 直接造（跳 analyze/classify）
    if state.get("mother_confirmed") and state.get("mother_dna"):
        return "generate"
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
    url = state.get("image_url") or _extract_image_url(
        _latest_human_text(state.get("messages", []))
    )
    if not url:
        return {
            "messages": [AIMessage(content="请先贴一张题目图的 OSS URL，我才能开始举一反三。")]
        }

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
    return {
        "image_url": url,
        "images_count": int(data.get("images_count") or 1),
        "questions_in_image": int(data.get("questions_in_image") or 1),
        "analysis": analysis,
        "mother_dna": mother_dna,
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

只输出 JSON 数组(不要解释)，每个元素：
{{"stem":"题干(Markdown+LaTeX)","answer":"标准答案","solution":"完整解析(过程+答案)",
  "qtype":"选择/填空/解答","difficulty":1~5,"level":"normal/hard","injected_kp":"相邻kp名或null"}}"""


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


async def generate(state: VariantState, config: RunnableConfig) -> VariantState:
    """③ 造题：默认 3 = 2 普通 + 1 难。🔴 入口断言 mother_confirmed 或三锚高置信（防裸奔）。"""
    if not (state.get("mother_confirmed") or _conf_ok(state.get("analysis") or {})):
        # DNA 闸未过却走到 generate（多入口兜底）→ 拒造，回 clarify 语义
        return {
            "messages": [
                AIMessage(content="母题 DNA 还没确认，我先不造题。请确认年级/考点/题型。")
            ]
        }

    facts = _mother_facts(state)
    n_normal, n_hard = DEFAULT_SHAPE["normal"], DEFAULT_SHAPE["hard"]
    n = n_normal + n_hard
    prompt = GENERATE_PROMPT.format(
        n=n, n_normal=n_normal, n_hard=n_hard, **facts
    )
    text = await _ainvoke_text([HumanMessage(content=prompt)])
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    items = []
    for it in data or []:
        if not isinstance(it, dict):
            continue
        items.append(
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
    return {"items": items, "messages": []}


SOLVE_PROMPT = """你是严谨的数学阅卷老师。真解下面这道题（不看给定答案，独立算一遍）。

题干：{stem}

只输出 JSON：
{{"solved_answer":"你独立算出的答案","solution":"完整解题过程(含答案)",
  "kp_name":"这道题实际考的主考点","grade":"这道题适配的年级"}}"""

REGEN_PROMPT = """下面这道变式题，独立解出的答案与题面标答不一致，请**重新出一道**等价变式重做。

主考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
原题干: {stem}
要求：仍考「{kp_name}」、仍在「{grade}」、{level} 难度；换数字/场景使题面与答案自洽。

只输出 JSON：
{{"stem":"新题干","answer":"标准答案","solution":"完整解析","qtype":"{qtype}","difficulty":{difficulty},"level":"{level}","injected_kp":{injected_kp}}}"""


# ---------------------------------------------------------------------------
# 闸B·程序验算（PRD-C-010）：LLM 只负责"人话题 → 结构化载荷"的有界抽取，
# pass/fail 判决只读 math_verify.verify()（纯 sympy，零 LLM）的 verdict。
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """你是数学验算载荷抽取器。把下面这道题的「题干 + 题面标准答案」抽成可被 sympy 程序验算的结构化载荷 JSON（验算对象 claimed = 题面标准答案）。

题型: {qtype}
题干: {stem}
题面标准答案(待验算的 claimed): {answer}
参考·另一次独立解答(仅帮助你理解答案格式，不是验算对象): {solved_answer}

载荷契约（kind 四选一；所有表达式必须是 sympy 可解析的纯 ASCII 数学串：乘号写 *、乘方写 ** 或 ^、分数写 /、根号写 sqrt()；禁止 LaTeX、中文、单位、等号外的标点）：
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
    return _parse_json(text) or {}


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
        return {
            "stem": regen.get("stem"),
            "answer": regen.get("answer"),
            "solution": regen.get("solution"),
            "qtype": regen.get("qtype") or item.get("qtype"),
            "difficulty": regen.get("difficulty") or item.get("difficulty"),
            "level": regen.get("level") or item.get("level"),
            "injected_kp": regen.get("injected_kp"),
        }
    return None


async def solve_explain(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ solve + 闸B 程序验算（PRD-C-010）。每题真解产解析 → 题型分流 → sympy 判决：

    - 证明/开放/作图类（分流键=题型，题干骨架兜底）→ 不进 sympy，正则级结构软校验，
      标 check.review=proof_needs_human 转人审（G3/FP2，不误杀）。
    - 其余（计算/解答/填空/选择）→ LLM 抽验算载荷 → math_verify.verify（纯 sympy）：
      pass → check.verify=sympy_pass 照常；
      fail → 既有 REGEN 回炉 1 次（computed/detail 注回 prompt）→ 重生版再验（须 pass+守恒），
             仍不过 → 保留原版 badge=warn + verify=fail_after_regen + 题卡尾追加可见提示；
      degrade → 保留原有「LLM 独立解 + _norm 比对」自检作 fallback，verify=unverified + 可见提示。
    🔴 判决只读 verify() 的 verdict，永不采信 LLM 自评；任何验算环节失败均降级继续，绝不抛（G5）。
    🔴 凡进 items 的题一律过本节点，无 check 不许进 assemble（remove 后旧题带 check 原样通过）。
    """
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    out: list[dict] = []

    for it in items:
        item = dict(it)
        if item.get("check"):  # 已定状态（如自愈过的补题再次流经）→ 不重复
            out.append(item)
            continue

        # ── 闸B·题型分流：证明/开放/作图 → 不进 sympy，软校验 + 人审标记 ──
        if _is_proof_like(item.get("qtype") or facts["qtype"], item.get("stem")):
            struct_ok = _proof_struct_ok(item.get("stem"))
            item["check"] = {
                "badge": "ok" if struct_ok else "warn",
                "solved_answer": None,
                "review": REVIEW_PROOF,
            }
            note = NOTE_PROOF_REVIEW
            if not struct_ok:
                note += "（题面缺「已知/求证/证明」类结构，请重点看）"
            _append_card_note(item, note)
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
            out.append(item)
            continue

        if verdict == math_verify.FAIL:
            # sympy 判定标答真错 → 既有回炉机制重生 1 次，computed/detail 注回 prompt
            healed = None
            if MAX_HEAL >= 1:
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
                out.append(healed)
            else:
                # 回炉后仍不过 → 保留原版标 ⚠，不拦截流程
                item["check"] = {
                    "badge": "warn",
                    "solved_answer": solved_answer,
                    "verify": VERIFY_FAIL_AFTER_REGEN,
                    "verify_detail": res.get("detail"),
                    "computed": res.get("computed"),
                }
                _append_card_note(item, NOTE_VERIFY_FAIL)
                out.append(item)
            continue

        # ── degrade：sympy 吃不下（载荷抽不成/超范围）→ 保留既有 LLM 自检 fallback ──
        match = _norm(solved_answer) == _norm(item.get("answer"))
        if match:
            item["check"] = {
                "badge": "ok",
                "solved_answer": solved_answer,
                "verify": VERIFY_UNVERIFIED,
                "verify_detail": res.get("detail"),
            }
            _append_card_note(item, NOTE_UNVERIFIED)
            out.append(item)
            continue

        # 独立解 ≠ 标答（LLM 自检）→ 既有自愈：重生 1 次 → 重解 + 守恒
        healed = None
        if MAX_HEAL >= 1:
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
                    }
                    _append_card_note(draft, NOTE_UNVERIFIED)
                    # 🔴 闸A 标记随愈合保留（同 FAIL 自愈路径：不丢 gene、不被编辑轮重判）
                    draft["gene"] = item.get("gene") or {
                        "gate": GENE_GATE_SKIPPED,
                        "reason": "healed-in-solve",
                    }
                    healed = draft

        if healed:
            out.append(healed)
        else:
            # 守恒破 或 重生仍不过 → 保留原版打 ⚠（仍属未经程序验算）
            item["check"] = {
                "badge": "warn",
                "solved_answer": solved_answer,
                "verify": VERIFY_UNVERIFIED,
                "verify_detail": res.get("detail"),
            }
            _append_card_note(item, NOTE_UNVERIFIED)
            out.append(item)

    return {"items": out, "messages": []}


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


async def _gene_judge_one(item: dict, facts: dict) -> dict | None:
    """一次轻量 LLM 基因比对（受约束 JSON）。调用失败/解析失败返 None（调用方按 skipped 放行）。"""
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
    try:
        text = await _ainvoke_text([HumanMessage(content=prompt)])
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
    out: list[dict] = []

    for it in items:
        item = dict(it)
        if item.get("gene"):  # 已判过 → 不重判（旧题预算保护）
            out.append(item)
            continue

        judge = await _gene_judge_one(item, facts)
        if judge is None:
            item["gene"] = {"gate": GENE_GATE_SKIPPED}
            out.append(item)
            continue

        if gene_gate_decision(judge) == "pass":
            item["gene"] = {"gate": GENE_GATE_PASS}
            out.append(item)
            continue

        # rework：既有回炉重生 1 次（基因反馈注回 prompt）→ 重生版再判一次
        draft = await _regen_once(item, facts, feedback=_gene_feedback(judge))
        if draft:
            re_judge = await _gene_judge_one(draft, facts)
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

    return {"items": out, "messages": []}


def _fmt_item(idx: int, it: dict) -> str:
    badge = (it.get("check") or {}).get("badge", "warn")
    mark = "✓" if badge == "ok" else "⚠"
    lvl = "难题" if it.get("level") == "hard" else "普通"
    inj = it.get("injected_kp")
    inj_s = f"（综合相邻考点：{inj}）" if inj else ""
    warn_s = ""
    if badge != "ok":
        chk = it.get("check") or {}
        # 🔴 防双写：solve_explain 已把对应 NOTE_* 追加进 solution（_append_card_note，
        # 随入库 analyze 持久），题卡渲染时同义警示只出一遍 —— solution 已含则跳过 warn_s。
        sol = str(it.get("solution") or "")
        if chk.get("review") == REVIEW_PROOF:
            if NOTE_PROOF_REVIEW not in sol:
                warn_s = f"\n> {NOTE_PROOF_REVIEW}"
        elif chk.get("verify") == VERIFY_FAIL_AFTER_REGEN:
            if NOTE_VERIFY_FAIL not in sol:
                computed = chk.get("computed")
                warn_s = (
                    f"\n> ⚠ 程序验算未通过（sympy 算得「{computed}」与标答不符，回炉一次仍未过），请老师核对。"
                )
        else:
            if NOTE_UNVERIFIED not in sol:
                sa = chk.get("solved_answer")
                warn_s = f"\n> ⚠ 我没算准（独立解得「{sa}」与标答不一致），老师重点看。"
    # 闸A·基因闸警示（与闸B 正交：badge=ok 的题也可能平行度存疑）
    gene_s = ""
    gene = it.get("gene") or {}
    if gene.get("gate") == GENE_GATE_WARN:
        g_reason = gene.get("reason")
        gene_s = (
            f"\n> {NOTE_GENE_WARN}"
            + (f"（{g_reason}）" if g_reason else "（题型/难度/解法结构或换皮程度与母题不平行）")
            + "，请老师确认。"
        )
    return (
        f"### 第 {idx} 题 {mark}（{lvl}·{it.get('qtype', '')}·难度{it.get('difficulty', '?')}）{inj_s}\n\n"
        f"{it.get('stem', '')}\n\n"
        f"**答案**：{it.get('answer', '')}\n\n"
        f"**解析**：{it.get('solution', '')}{warn_s}{gene_s}\n"
    )


async def assemble(state: VariantState, config: RunnableConfig) -> VariantState:
    """题组快照：每题带 solution 解析 + check ✓/⚠（状态已定）+ 外显默认配方 + 变更摘要。"""
    items = state.get("items") or []
    facts = _mother_facts(state)
    n_ok = sum(1 for it in items if (it.get("check") or {}).get("badge") == "ok")
    n_warn = len(items) - n_ok

    head = (
        f"## 举一反三 · {len(items)} 道变式（配方：默认 3 = 2 普通 + 1 难）\n\n"
        f"**母题 DNA**：考点「{facts['kp_name']}」· 年级「{facts['grade']}」· 题型「{facts['qtype']}」（硬守恒）\n\n"
        f"**状态**：{n_ok} 道 ✓ 通过自检"
        + (f"，{n_warn} 道 ⚠ 需老师重点看" if n_warn else "")
        + "\n\n旋钮可拨：数量 / 数字 / 场景 / 难度 / 题型(可配比) / 解法。说「这组可以了」即入库。\n\n---\n"
    )
    body = "\n".join(_fmt_item(i + 1, it) for i, it in enumerate(items))
    return {"messages": [AIMessage(content=head + body)]}


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

直接用人话回答（可含 LaTeX）。只解惑，不要改题、不要重出题。"""


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
    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ANSWER_PROMPT.format(
                    brief=_items_brief(items), detail=detail, question=question or "(空)"
                )
            )
        ]
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
    return {"items": kept, "pending": None, "messages": []}


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
            items[t] = {
                "stem": regen.get("stem"),
                "answer": regen.get("answer"),
                "solution": regen.get("solution"),
                "qtype": regen.get("qtype") or old.get("qtype") or facts["qtype"],
                "difficulty": regen.get("difficulty") or old.get("difficulty"),
                "level": regen.get("level") or old.get("level") or "normal",
                "injected_kp": regen.get("injected_kp"),
            }
    return {"items": items, "pending": None, "messages": []}


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
{{"stem":"题干","answer":"标准答案","solution":"完整解析","qtype":"选择/填空/解答","difficulty":1~5,"level":"normal/hard","injected_kp":"相邻kp名或null"}}"""


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
            }
        )
    return {"items": items, "pending": None, "messages": []}


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

    # 🔴 身份透传：book-ui 经 agent_config 透传登录老师 access_token（config.configurable.ruoyi_token）
    # → 入库 owner = 该老师本人（后端 LoginHelper 取 token 身份），而非 .env 服务账号。
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    try:
        receipts = await persist_items(items, facts, token=token)
    except Exception as e:  # noqa: BLE001 — 登录/网络整体失败 → 友好兜底，不崩
        return {
            "messages": [
                AIMessage(content=f"入库时连不上题库服务（book-server :8090 是否在跑？）：{e}")
            ]
        }

    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var_receipts = [r for r in receipts if r.get("role") != "mother"]
    ok = [r for r in var_receipts if r.get("ok")]
    fail = [r for r in var_receipts if not r.get("ok")]

    lines = [f"## 入库完成 · 变式 {len(items)} 道，成功 {len(ok)} 道"]
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
    lines.append("\n可回平台「我的题库」找题、组卷、导出 PDF。")
    return {"messages": [AIMessage(content="\n".join(lines))]}


# --- 输入边界兜底（设计 §6）：没图/无在途母题/无题组 → 催图 ------------------
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

graph.set_conditional_entry_point(
    route_entry,
    {
        "analyze": "analyze",
        "generate": "generate",
        "parse": "parse_instruction",
        # 🔴 'ask' 必落真节点（ask_for_image），不能直连 END —— 否则首轮无节点产消息，回复为空
        "ask": "ask_for_image",
    },
)
graph.add_edge("ask_for_image", END)

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
