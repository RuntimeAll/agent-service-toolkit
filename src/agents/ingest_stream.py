# -*- coding: utf-8 -*-
r"""PRD-A-002 路A · 录题「解题/批改」流式（SSE）—— 实时输出解题/批改过程，不黑盒等待。

🔴 维护者拍板（2026-06-26）：录题的解题/批改要**像举一反三第一阶段一样实时流式**——按钮直调
   封装好的预设 prompt，但 LLM 输出**逐字流式**（老师实时看到「怎么解」「怎么改」），不是黑盒。
   打标（DNA）是**异步额外**的事，跟在解题后面（FE 拿到解题结果再调 /label），不同步阻塞。

设计：
  - 解题流（/solve/stream）= **识别 + 解题一起做**（多模态读框区图）：流式输出 识别题干 + 解题过程
    + 答案（人类可读 markdown），末尾另起 ===STRUCT=== + 一段 JSON（供系统入库，不进流式显示）。
  - 批改流（/grade/stream）= 识别 + 先解题 + 判学生作答对错：流式输出批改过程，末尾 ===STRUCT===。
  - 复用 core.relay_pool.ainvoke_failover(on_delta=)（astream 聚合，拿累计文本），不依赖 LangGraph
    运行上下文 → 可独立 StreamingResponse。SSE 协议对齐 variant：{type:token|result|error} + [DONE]。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage

from agents import dna_extract
from agents.recognize import _normalize_image_ref, parse_json_lax

STREAM_MODEL = "claude-opus-4-8"
STREAM_TEMPERATURE = 0.1
STREAM_TIMEOUT_S = 180.0
SOLVE_MAX_TOKENS = 4096
GRADE_MAX_TOKENS = 4096
SENTINEL = "===STRUCT==="


def build_solve_stream_prompt() -> str:
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    return f"""你是数学老师。下面是老师框选的**一道题的照片**。请按顺序做两件事：

【第一部分·给老师看（用 Markdown + 行内 $LaTeX$，像给学生讲解一样自然流畅）】
1) 识别出图中的**印刷体原题**（去掉手写/批改痕迹），先把题干清晰写出来。
2) **一步步把它解出来**（展示完整解题过程，不跳步、不臆造），最后明确写出**答案**。

【第二部分·给系统入库（老师看不到）】
另起一行输出一行 `{SENTINEL}`，紧接一个 JSON（不要 markdown fence）：
{{"stem":"识别出的题干(markdown+$latex$)","qtype":"选择/填空/解答","options":["选项正文"],"answer":"最终答案","analysis":"干净的解题过程","need_grading":true/false}}
（need_grading：图中含学生手写作答/笔迹则 true，否则 false；options 非选择题给空数组。）

🔴 务必先输出第一部分完整人类可读内容，再输出 `{SENTINEL}` 和 JSON。"""


def build_grade_stream_prompt(*, knowledge: str | None, chapter: str | None) -> str:
    ctx = []
    if (chapter or "").strip():
        ctx.append(f"所属章节：{chapter.strip()}")
    if (knowledge or "").strip():
        ctx.append(f"相关知识点：{knowledge.strip()}")
    ctx_block = ("\n".join(ctx) + "\n") if ctx else ""
    return f"""你是数学老师，正在**批改学生作答**。下面是一道题区照片（含印刷体原题 + 学生手写作答）。
{ctx_block}请按顺序做两件事：

【第一部分·给老师看（Markdown + $LaTeX$，像老师当面批改讲解一样）】
1) 识别原题（去手写）。
2) **先自己严谨解出标准答案**（展示过程；题目没给答案也要自己解）。
3) 读学生手写作答，**对照标准答案判对错**，指出错在哪、扣分点、给出建议。
   - 图中无任何手写作答 → 明说「学生未作答」。

【第二部分·给系统（老师看不到）】
另起一行 `{SENTINEL}` + JSON（不要 fence）：
{{"stem":"原题","standard_answer":"标准答案","student_answer":"学生作答(无则空)","has_handwriting":true/false,"verdict":"correct/wrong/partial/blank/uncertain","feedback":"一句话点评"}}

🔴 先输出第一部分完整人类可读批改，再输出 `{SENTINEL}` 和 JSON。"""


async def _stream_messages(
    messages: list[Any], *, max_tokens: int
) -> AsyncGenerator[tuple[str, Any], None]:
    """跑一次 LLM 流式调用，yield ('token', delta)…最后 ('full', 完整文本)。

    复用 relay_pool.ainvoke_failover(on_delta=)：on_delta 每 chunk 拿【累计文本】。
    用队列把回调桥接到本异步生成器；SENTINEL 之后的内容（JSON）不再 yield token。
    """
    from core import relay_pool

    queue: asyncio.Queue = asyncio.Queue()

    def on_delta(acc: str) -> None:
        try:
            queue.put_nowait(acc)
        except Exception:  # noqa: BLE001
            pass

    async def run() -> str:
        resp, *_ = await relay_pool.ainvoke_failover(
            messages, max_tokens=max_tokens, on_delta=on_delta,
            model=STREAM_MODEL, temperature=STREAM_TEMPERATURE, timeout=STREAM_TIMEOUT_S,
        )
        c = getattr(resp, "content", resp)
        if isinstance(c, list):
            return "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
        return str(c)

    task = asyncio.create_task(run())
    visible_sent = 0
    full_acc = ""
    while True:
        try:
            acc = await asyncio.wait_for(queue.get(), timeout=0.3)
            full_acc = acc
            visible = acc.split(SENTINEL)[0]
            if len(visible) > visible_sent:
                yield ("token", visible[visible_sent:])
                visible_sent = len(visible)
        except asyncio.TimeoutError:
            if task.done():
                break
    # 排空残留
    while not queue.empty():
        acc = queue.get_nowait()
        full_acc = acc
        visible = acc.split(SENTINEL)[0]
        if len(visible) > visible_sent:
            yield ("token", visible[visible_sent:])
            visible_sent = len(visible)
    try:
        text = await task
    except Exception:  # noqa: BLE001
        text = full_acc
    yield ("full", text or full_acc)


def _parse_struct(full_text: str) -> dict[str, Any] | None:
    """从完整文本里 SENTINEL 之后抠出 JSON。"""
    if SENTINEL in full_text:
        tail = full_text.split(SENTINEL, 1)[1]
    else:
        tail = full_text
    try:
        return parse_json_lax(tail)
    except Exception:  # noqa: BLE001
        return None


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


async def solve_stream_sse(
    *, image_url: str | None = None, image_base64: str | None = None
) -> AsyncGenerator[str, None]:
    """解题流（识别+解题）SSE。yield 'data: {...}\\n\\n'。"""
    try:
        img = _normalize_image_ref(image_url, image_base64)
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "content": f"图无效: {str(e)[:80]}"})
        yield "data: [DONE]\n\n"
        return
    msg = HumanMessage(content=[
        {"type": "text", "text": build_solve_stream_prompt()},
        {"type": "image_url", "image_url": {"url": img}},
    ])
    full = ""
    try:
        async for kind, payload in _stream_messages([msg], max_tokens=SOLVE_MAX_TOKENS):
            if kind == "token":
                yield _sse({"type": "token", "content": payload})
            elif kind == "full":
                full = payload
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "content": f"解题异常: {str(e)[:100]}"})
        yield "data: [DONE]\n\n"
        return
    struct = _parse_struct(full) or {}
    result = {
        "stem": str(struct.get("stem") or "").strip(),
        "qtype": dna_extract._norm_qtype(struct.get("qtype")),
        "options": [str(o).strip() for o in (struct.get("options") or []) if str(o).strip()],
        "answer": str(struct.get("answer") or "").strip(),
        "analysis": str(struct.get("analysis") or "").strip(),
        "need_grading": bool(struct.get("need_grading")),
    }
    yield _sse({"type": "result", "content": result})
    yield "data: [DONE]\n\n"


async def grade_stream_sse(
    *, image_url: str | None = None, image_base64: str | None = None,
    knowledge: str | None = None, chapter: str | None = None,
) -> AsyncGenerator[str, None]:
    """批改流 SSE。yield 'data: {...}\\n\\n'。"""
    try:
        img = _normalize_image_ref(image_url, image_base64)
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "content": f"图无效: {str(e)[:80]}"})
        yield "data: [DONE]\n\n"
        return
    msg = HumanMessage(content=[
        {"type": "text", "text": build_grade_stream_prompt(knowledge=knowledge, chapter=chapter)},
        {"type": "image_url", "image_url": {"url": img}},
    ])
    full = ""
    try:
        async for kind, payload in _stream_messages([msg], max_tokens=GRADE_MAX_TOKENS):
            if kind == "token":
                yield _sse({"type": "token", "content": payload})
            elif kind == "full":
                full = payload
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "content": f"批改异常: {str(e)[:100]}"})
        yield "data: [DONE]\n\n"
        return
    struct = _parse_struct(full) or {}
    verdict = str(struct.get("verdict") or "").strip().lower()
    if verdict not in ("correct", "wrong", "partial", "blank", "uncertain"):
        verdict = "uncertain"
    result = {
        "stem": str(struct.get("stem") or "").strip(),
        "standard_answer": str(struct.get("standard_answer") or "").strip(),
        "student_answer": str(struct.get("student_answer") or "").strip(),
        "has_handwriting": bool(struct.get("has_handwriting")),
        "verdict": verdict,
        "feedback": str(struct.get("feedback") or "").strip(),
    }
    yield _sse({"type": "result", "content": result})
    yield "data: [DONE]\n\n"
