# -*- coding: utf-8 -*-
"""PRD-C-100 B5 单一全局日预算护栏（D6/§10/G7）。

当日（本地日界）conv_trace 累计 cost_yuan ≥ GLOBAL_DAILY_BUDGET_YUAN → 触发护栏：
  - 母题 opus 一把：**拦截**（不调，SSE 提示「今日额度用尽，稍后/明日再试」）。
  - 造图翻命令：**降级**（needs_figure，不调，题照常交付）。
绝非静默烧钱（G7）：拦截/降级都外显事件，非闷头继续。

🔴 读 conv_trace（独立观测库，同 :3307）SUM(cost_yuan) WHERE ts>=今日0点；缓存 ~60s 防每调一查。
🔴 阈值 None/≤0 = 关（不限，默认）。best-effort：查不到（库故障）→ 视为未超（不误拦真流量）。
🔴 单一全局日（本轮）；三级（会话/老师/日）推多用户期。
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from agents import conv_trace
from core.settings import settings

_CACHE_TTL_S = 60.0
_cache: dict[str, Any] = {"ts": 0.0, "spend": 0.0}


def _today_start_utc() -> str:
    """当日 0 点（UTC，与 conv_trace.ts 同口径——ts 落 UTC）。"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d 00:00:00")


def today_spend_yuan(*, force: bool = False) -> float:
    """当日累计花费（¥）。缓存 ~60s。库故障 → 返回上次缓存值或 0（不误拦）。"""
    now = time.monotonic()
    if not force and (now - _cache["ts"]) < _CACHE_TTL_S:
        return _cache["spend"]
    spend = 0.0
    try:
        conn = conv_trace._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COALESCE(SUM(cost_yuan),0) FROM conv_llm_trace WHERE ts >= %s",
                (_today_start_utc(),),
            )
            row = cur.fetchone()
            spend = float(row[0]) if row and row[0] is not None else 0.0
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — 库故障 → 用旧缓存（或 0），不误拦真流量
        return _cache["spend"]
    _cache["ts"] = now
    _cache["spend"] = spend
    return spend


def budget_status() -> dict[str, Any]:
    """{limit, spend, exceeded, remaining}。limit None/≤0 = 关（exceeded 恒 False）。"""
    limit = settings.GLOBAL_DAILY_BUDGET_YUAN
    if not limit or limit <= 0:
        return {"limit": None, "spend": today_spend_yuan(), "exceeded": False, "remaining": None}
    spend = today_spend_yuan()
    return {
        "limit": float(limit), "spend": round(spend, 6),
        "exceeded": spend >= float(limit),
        "remaining": round(float(limit) - spend, 6),
    }


def is_budget_exceeded() -> bool:
    """护栏闸：当日花费 ≥ 阈值 → True（母题拦截/造图降级）。阈值关 → 永 False。"""
    return budget_status()["exceeded"]


# ---------------------------------------------------------------------------
# 🔴 P5 并发炸弹兜底：护栏读库（today_spend_yuan → conv_trace._conn → 同步 pymysql SUM）
#   在 async 上下文（母题入口 / 造图）被调用。慢库（:3307 丢包/分区）下同步阻塞会卡死整个
#   asyncio 事件循环 → 所有并发 SSE 流一起卡。async 调用点必须走下面的 *_async 版本，
#   把同步读丢线程池（_conn 已配 connect/read/write 死超时，慢库快速抛 → best-effort 不误拦）。
# ---------------------------------------------------------------------------
async def budget_status_async() -> dict[str, Any]:
    """budget_status 的非阻塞版：同步读库丢线程池跑，不卡 asyncio loop。"""
    return await asyncio.to_thread(budget_status)


async def is_budget_exceeded_async() -> bool:
    """is_budget_exceeded 的非阻塞版（async 上下文必走它）。读库丢线程池，慢库不卡 loop。"""
    return await asyncio.to_thread(is_budget_exceeded)
