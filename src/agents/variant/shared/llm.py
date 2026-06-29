"""variant 引擎 · LLM 出口（PRD-C-104 B2b 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出 LLM 出口簇（约 615-911 行）：
  - _content_text（思考型只取 content）
  - LLM 往返持久化：_LLM_TRACE_ENABLED / _LLM_TRACE_PATH / _LLM_TRACE_SEQ / _TRACE_MARKERS
    / _msg_text / _trace_label / _serialize_request / _trace_llm
  - _ainvoke_text（LLM 主出口，靠 relay_pool failover + conv_trace + _budget_tick 记账）
  - _JSON_FENCE / _parse_json

🔴 行为零改，仅两处必要适配（保持行为字节级不变）：
  ① import：_emit_reasoning ← shared.emit；_budget_tick ← shared.budget；其余 stdlib/外部原样。
  ② _LLM_TRACE_PATH 的 __file__ 锚点：原 __init__.py（src/agents/variant/__init__.py）用
     parents[2]=src；本模块在 src/agents/variant/shared/llm.py，须用 parents[3] 才解析出
     **同一个** src/data/llm_trace.jsonl（逐字节一致）。这是「保持路径不变」的必要适配，非改逻辑。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import ensure_config

from agents import conv_trace
from core import relay_pool, settings

from agents.variant.shared.budget import _budget_tick
from agents.variant.shared.emit import _emit_reasoning


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
# 🔴 PRD-C-104 B2b：本模块在 .../variant/shared/llm.py，须 parents[3] 才得 src/（原 __init__ 用 parents[2]），
#   解析结果逐字节等同原 src/data/llm_trace.jsonl。
_LLM_TRACE_PATH = Path(__file__).resolve().parents[3] / "data" / "llm_trace.jsonl"
_LLM_TRACE_SEQ = 0  # 进程内自增序号（同一进程内调用顺序）

# prompt 内容前缀 → 标签（哪个 prompt）。新增/造同含「基于母题 DNA」，先判 add 再判 generate。
_TRACE_MARKERS: list[tuple[str, str]] = [
    # 🔴 PRD-C-100 B1b：塌缩入口 opus 一把（按顺序判章+解题+打标）+ 母题池注入重锚（classify），
    #   须排在 analyze/generate 泛 marker 前，独立标 label（G1：该轮无 analyze 行；G6：opus 调用点可查）。
    ("按顺序做三件事", "mother_entry"),
    ("先真正把题解出来", "mother_solve"),
    # 🔴 R6 母题三步编排（富文本化∥解题→打标）+ B9 数轴抽取——prompt 头重写后旧 marker 全失配 →
    #   label 落 unknown（2026-06-23 排查）。下面 4 个对齐 R6/B9 新 prompt 首句，须在泛 marker 前。
    ("题面誊抄员", "richtext"),       # R6 富文本化(sui-xiang·异步)
    ("数学解题专家", "mother_solve"),  # R6 母题解题(aigeek)
    ("题库打标师", "mother_label"),    # R6 母题打标/深度解析(sui-xiang)
    ("数学题信息抽取器", "numline"),   # B9 数轴点抽取(图配)
    # 🔴 PRD-C-100 B3：变式造图 opus 翻 GeoGebra 命令（G6：造图翻命令调用不漏计，含计费）。
    ("数学配图助手", "figure_geogebra"),
    ("看这张题目图", "analyze"),
    ("出题配方", "knobs"),
    ("数学验算载荷抽取器", "extract"),
    # 🔴 Q1 可观测性修复（2026-06-12）：锚定/DNA 抽取这一步从此经 _ainvoke_text → 落 trace。
    #   两个 marker（DNA 抽取主调 + 标签复用窄调）须排在 generate 的「命题专家」泛 marker 之前。
    ("打标式 DNA 抽取", "dna_extract"),
    ("检索标签师", "dna_tags"),
    # 🔴 B2·T2：闸A LLM judge 已退役（gene_judge 全链删），不再有「平行题基因比对器」调用。
    ("独立解出的答案与题面标答不一致", "regen"),
    ("你是严谨的数学阅卷老师", "solve"),
    ("举一反三 agent 的指令解析器", "parse"),
    ("老师对下面这组变式题的某道有疑问", "answer"),
    # 整改3（2026-06-12）：解法修正逐题解析重写器（题面留只改解析）。
    ("解题方法提了新约束", "solution_rewrite"),
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
    on_reasoning: Any = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    timeout: float | None = None,
    prefer_relay: str | None = None,
    trace_label: str | None = None,
) -> str:
    """ainvoke + 取 content；偶发空返回重试一次。max_tokens≥4096 给思考型留头。

    🔴 每次调用落 JSONL 往返记录（_trace_llm）：发送的完整 prompt + 原始返回。
    🔴 思维外放（用户反馈 2026-06-11）：默认打 skip_stream 标签 —— JSON 类中间产物的
    token 流对用户是乱码，service 按标签丢弃；只有人话型调用（答疑等）传
    public_stream=True，token 才透到前端打字机。on_delta=流内进度回调（拿累计文本），
    generate 用它数「已写到第几题」。
    model：per-call 模型覆盖（S1.1）。给了就用它换该次请求的 model（站点不变），轻活
    调用点（nano 降本）传 settings.LLM_MODEL_LIGHT；None = 沿用 relay 配置 model（旧行为不变）。
    max_tokens：per-call max_tokens 覆盖（整改4·回炉瘦身）。None/≤0 → 默认 VARIANT_MAX_TOKENS。
    temperature：per-call 温度覆盖（PRD-C-017 M9）。仅 model 覆盖时生效；None=默认 0.5（旧行为）。
      母题 opus 解题+打标档传低温（0.1~0.2）稳 JSON/解题，B1 母题节点用。
    response_format：per-call 结构化输出（PRD-C-017 B1·F3）。母题 opus 合并调用用它硬锁
      10 维 json_schema（中转实测支持）；None=不约束（旧行为）。
    timeout：per-call 超时上限（秒，PRD-C-017 B1·H4）。仅 model 覆盖时生效；None=不设。
      母题 opus 读图慢，B1 传 ≤180s 防挂死（超时抛 → 上层 SSE error，绝不静默退 gpt-5.4）。
    """
    # 🔴 PRD-C-107 可观测性修复：调用点可显式传 trace_label（连续对话三轮的 system 头一样，
    #   _trace_label 按头 120 字猜会把三轮全归 unknown / 同一桶 → 消费记录分不清誊抄/解题/打标）。
    #   显式 label 优先；没传才回退「按头猜」（旧行为不变，所有老调用点零感）。
    label = trace_label or _trace_label(messages)
    tags = None if public_stream else ["skip_stream"]
    # 用户级/会话级归属：从 graph config 取 thread_id + ruoyi_token(→teacher_id)
    conf = (ensure_config() or {}).get("configurable", {}) or {}
    thread_id = conf.get("thread_id")
    teacher_id = conv_trace.teacher_id_from_token(conf.get("ruoyi_token"))
    # 🔴 思考链开关（老师手动开/关 extended-thinking）：FE 经 agent_config.thinking_stream 回传 → 进
    #   config.configurable.thinking_stream。开 → 本轮调用优先路由到 THINKING_RELAY(aigeek) +
    #   bind reasoning_effort(THINKING_EFFORT)；该站吐 reasoning_content → on_reasoning 外显。
    #   关/省略（默认）→ prefer_relay=None、reasoning_effort=None，一切维持现状（主站、不带 thinking）。
    #   🔴 graceful：开了但优先站不可用 → relay_pool 正常 failover 回 kiro，参数被忽略、无 reasoning 帧，
    #      思考块不显示、绝不报错/不中断（prefer 只改尝试顺序，不改 failover 语义）。
    _thinking = bool(conf.get("thinking_stream"))
    # 🔴 调用点显式 prefer_relay（如母题解题轮指定 aigeek）优先；否则思考链开时用 THINKING_RELAY；都没有 → None。
    _prefer_relay = prefer_relay or (settings.THINKING_RELAY if _thinking else None)
    _reasoning_effort = settings.THINKING_EFFORT if _thinking else None
    # 🔴 思考链开 + 调用点没显式挂 on_reasoning → 默认挂 _emit_reasoning，让本轮所有 LLM 调用的 reasoning
    #   都外显（老师开了就想看思考；reasoning 走独立 custom 通道，不混 intent/outline 正文，判决永不采信）。
    #   显式传了 on_reasoning（如 mother_opus_entry）则尊重原值。关时不动（on_reasoning 行为完全不变）。
    if _thinking and on_reasoning is None:
        on_reasoning = _emit_reasoning
    # 🔴 per-call max_tokens 覆盖（整改4·回炉瘦身用）：None → 默认 VARIANT_MAX_TOKENS（旧行为）。
    max_tokens = max_tokens if max_tokens and max_tokens > 0 else settings.VARIANT_MAX_TOKENS
    t0 = time.monotonic()
    relay = settings.RELAY_NAME
    model_used = settings.COMPATIBLE_MODEL
    fallback = 0
    fb_detail: str | None = None  # 🔴 熔断回溯：每站失败原因（成功且 0 转移=None）
    try:
        # 🔴 PRD-C-100 B-converge roundE·节点级单一墙钟（修 perf 长尾根因）：
        #   relay_pool.ainvoke_failover 内的 asyncio.timeout 是【每站】预算（包在 for relay 循环内），
        #   叠上本函数空返重试（再调一次 failover）→ 节点级有效上限 ≈ timeout × 站数 × (1+重试)，
        #   无单一墙钟 → generate 的降级分支（_runtime_generate 的 except TimeoutError）触发不到，
        #   压轴几何变式 generate 拖到 9-11min 不收尾。
        #   这里用【一个】asyncio.timeout 把「全部站点轮询 + 全部空返重试」整体包成节点级总预算，
        #   到点抛 TimeoutError → 上层 generate 既有降级分支接住（eager 已出题先出 / 零题给可读文案）。
        #   inner failover 仍收到 timeout（其【每站】慢吐墙钟语义不变，原测全绿）；外层这层才是
        #   节点级总墙钟，二者同值 → 节点 wall-clock ≤ timeout（不再 ×站数×重试）。
        #   timeout=None（绝大多数调用）→ 不包，保留旧无界行为（contextlib.nullcontext）。
        _node_cap: Any = (
            asyncio.timeout(timeout)
            if (timeout is not None and timeout > 0)
            else contextlib.nullcontext()
        )
        async with _node_cap:
            # 🔴 走中转站熔断转移池（Block B）：返回实际成交中转站 + 该站 model + 转移次数 + 失败原因串
            #   （RELAY_POOL 各站可配不同模型，trace/计费必须按成交站归因）
            resp, relay, model_used, fallback, fb_detail = await relay_pool.ainvoke_failover(
                messages, max_tokens=max_tokens, tags=tags, on_delta=on_delta,
                on_reasoning=on_reasoning, model=model,
                temperature=temperature, response_format=response_format, timeout=timeout,
                prefer_relay=_prefer_relay, reasoning_effort=_reasoning_effort,
            )
            text = _content_text(resp).strip()
            retried = False
            if not text and retry:
                retried = True
                resp, relay, model_used, fb2, fbd2 = await relay_pool.ainvoke_failover(
                    messages, max_tokens=max_tokens, tags=tags, on_delta=on_delta, model=model,
                    temperature=temperature, response_format=response_format, timeout=timeout,
                    prefer_relay=_prefer_relay, reasoning_effort=_reasoning_effort,
                )
                fallback += fb2
                fb_detail = "; ".join(x for x in (fb_detail, fbd2) if x) or None
                text = _content_text(resp).strip()
    except Exception as e:  # noqa: BLE001 — 记下失败往返后照常抛
        dur = int((time.monotonic() - t0) * 1000)
        _trace_llm(label, messages, "", None, dur, error=str(e), model=model_used)
        # 🔴 P5：conv_trace 是同步 pymysql，丢线程池跑（_conn 已配死超时），绝不阻塞 loop。
        #   best-effort 不变：write() 内部自吞错；to_thread 包一层防慢库卡住事件循环。
        await asyncio.to_thread(
            conv_trace.write,
            teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
            model=model_used, relay=relay, fallback_count=fallback, fallback_detail=fb_detail,
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
    # 🔴 P5：同步 pymysql 丢线程池（_conn 已配死超时），慢库不阻塞 asyncio loop；best-effort 不变。
    await asyncio.to_thread(
        conv_trace.write,
        teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
        model=model_used, relay=relay, fallback_count=fallback, fallback_detail=fb_detail,
        request=_serialize_request(messages),
        response=text, response_raw=raw, duration_ms=dur, retried=retried,
        prompt_tokens=pt, completion_tokens=ct, cost_yuan=cost,
    )
    # 🔴 P13 预算闸记账：一次成功 LLM 往返 = 一票（per-round 计数，超限后增强类调用跳过）。
    # 记在「成功返回」处（失败/空返抛异常的早退路径不记——只数真正花掉的调用）。
    _budget_tick()
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
