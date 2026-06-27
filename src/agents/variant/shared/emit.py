"""variant 引擎 · SSE 帧发射器（纯）（PRD-C-104 B2b 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出的**纯** SSE custom 帧发射器（仅依赖 get_stream_writer +
ChatMessage，零 variant 内部簇依赖 → 抽出不产生循环）：
  - _emit_stage / _emit_error / _emit_need_confirm / _emit_reject / _emit_reasoning
  - _emit_richtext_stem

🔴 B2b 决策（解循环）：`_artifact_payload` / `_emit_artifact` / `_emit_mother_card`
   依赖 stage1/mother_card 簇 helper（_norm_secondary_kps / _item_dna / _mother_facts，
   本批不搬）→ 抽到 shared/emit 会与 __init__ 互相 import 形成循环。按 PRD-C-104 §2「纯搬零改
   优先 > 抽全」，那三件**留在 __init__ 不抽**，本模块只收无外簇依赖的纯发射器。
   shared/llm.py 用到的 _emit_reasoning 即从本模块取 → 单向依赖，无循环。

🔴 contextvar 命脉：get_stream_writer() 是运行期调用、无 import 序问题；双层 try/except
   静默吞原样保留（无 runtime context / writer 抛 → no-op，绝不炸节点）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.config import get_stream_writer


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


def _emit_error(reason: str, message: str) -> None:
    """发 SSE error 事件（PRD-C-017 §10 / G3 / H4）。母题 opus 失败/超时必走这里，
    绝不静默退 gpt-5.4。契约 = ChatMessage(role="custom", content=[{"error": {reason, message}}])，
    与 _emit_stage 同双层静默吞（无 runtime context / writer 抛 → no-op，不炸节点）。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    try:
        writer(ChatMessage(content=[{"error": {"reason": reason, "message": message}}], role="custom"))
    except Exception:  # noqa: BLE001
        pass


def _emit_need_confirm(payload: dict[str, Any]) -> None:
    """发 SSE needConfirm 事件（PRD-C-017 B2·决策表「母题每次必停弹窗」）。

    payload 契约（FE pickNeedConfirm 解析弹窗）：
      {grade_book:{id,name}, chapter:{id,name}, grade_candidates:[{id?,name}], chapter_candidates:[...]}。
    🔴 无条件停（不是低置信才停）；候选为空也发（让老师手选）。
    与 _emit_stage 同双层静默吞（无 runtime context / writer 抛 → no-op，不炸节点）。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    try:
        writer(ChatMessage(content=[{"needConfirm": payload}], role="custom"))
    except Exception:  # noqa: BLE001
        pass


def _emit_reject(reason: str, message: str) -> None:
    """发 SSE reject 事件（PRD-C-017 B2·G13 带图打回）。含图题终止流程，不调 opus、不出变式。
    契约 = ChatMessage(role="custom", content=[{"reject": {reason, message}}])，同双层静默吞。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    try:
        writer(ChatMessage(content=[{"reject": {"reason": reason, "message": message}}], role="custom"))
    except Exception:  # noqa: BLE001
        pass


def _emit_reasoning(text: str) -> None:
    """🔴 PRD-C-100 D18 思考流式：opus reasoning 流式吐前端可折叠「思考中」块（透明、零额外成本）。
    契约 = ChatMessage(role="custom", content=[{"reasoning": {"text": <累计 reasoning>}}])，同双层静默吞。
    🔴 旧前端无该事件 → 静默丢弃（向后兼容，AC13）；reasoning 纯展示，不混入 intent/outline 正文，
       pass/fail 判决永不采信 reasoning（铁律不破）。注：opus 经 aigeek 当前未吐 reasoning（B0 实测），
       本管线 ready-but-dormant——上游若开 extended-thinking 则自动流式（无需再改码）。"""
    if not text:
        return
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    try:
        writer(ChatMessage(content=[{"reasoning": {"text": text}}], role="custom"))
    except Exception:  # noqa: BLE001
        pass


def _emit_richtext_stem(stem: str) -> None:
    """🔴 R6 富文本化早帧（2026-06-22 三步编排第①步）：富文本化任务跑完即发，让 FE 占位区
    尽早把母题原图替换成富文本题面（KaTeX 渲染），老师在解题/打标还在跑时就能读到干净题面。

    契约 = ChatMessage(role="custom", content=[{"richtextStem": {"stem": <富文本题面>}}])
      → 服务层 custom_data.richtextStem，FE pickRichtextStem 解析。
    同 _emit_stage 双层静默吞（无 runtime context / writer 抛 → no-op，不炸节点）；空串不发。"""
    if not stem or not stem.strip():
        return
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001
        return
    try:
        writer(ChatMessage(content=[{"richtextStem": {"stem": stem}}], role="custom"))
    except Exception:  # noqa: BLE001
        pass
