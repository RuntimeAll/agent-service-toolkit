"""PRD-C-011 Block B：LLM 出口多中转站主备 + 熔断转移。

一个有序中转站列表（主→备）。每个中转站一个熔断器：连续失败 N 次 → trip（开闸），
冷却 T 秒后 half-open 探活；任一次成功立即复位。一次请求按序尝试，跳过开闸的中转站，
失败则切下一个并累加 fallback_count。

同时为 Block A（持久化）产出可观测信号：
- relay：实际成交的中转站名（落 conv_trace.relay）。
- fallback_count：本次请求转移了几次（0 = 主站一次成功）。
- usage：langchain `usage_metadata`（开 stream_usage 后才有，修 token NULL 的关键）。
- cost_yuan：按 RELAY_PRICES 价表算的实际消费（无价表则 None，不瞎猜）。

🔴 不碰 core.get_model（那是别的 agent 的共享入口）——本模块只服务举一反三 _ainvoke_text。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from core.settings import settings


@dataclass
class Relay:
    name: str
    base_url: str
    api_key: str
    model: str


@dataclass
class _Breaker:
    fails: int = 0
    open_until: float = 0.0  # time.monotonic() 时间戳；> now 表示开闸（暂时不可用）

    def is_open(self) -> bool:
        return self.open_until > time.monotonic()


_relays_cache: list[Relay] | None = None
_breakers: dict[str, _Breaker] = {}
_chat_cache: dict[str, ChatOpenAI] = {}
_prices_cache: dict[str, dict[str, float]] | None = None


def _relays() -> list[Relay]:
    """解析中转站列表。RELAY_POOL(JSON 有序) 优先；否则从 COMPATIBLE_* 派生单站。"""
    global _relays_cache
    if _relays_cache is not None:
        return _relays_cache
    out: list[Relay] = []
    raw = (settings.RELAY_POOL or "").strip()
    if raw:
        try:
            for item in json.loads(raw):
                out.append(
                    Relay(
                        name=str(item["name"]),
                        base_url=str(item["base_url"]),
                        api_key=str(item.get("api_key") or ""),
                        model=str(item.get("model") or settings.COMPATIBLE_MODEL or ""),
                    )
                )
        except Exception:
            out = []  # 配置坏 → 退回单站派生，绝不因配置炸主流程
    if not out:
        key = settings.COMPATIBLE_API_KEY
        out = [
            Relay(
                name=settings.RELAY_NAME,
                base_url=settings.COMPATIBLE_BASE_URL or "",
                api_key=key.get_secret_value() if key else "",
                model=settings.COMPATIBLE_MODEL or "",
            )
        ]
    _relays_cache = out
    return out


def _chat(relay: Relay) -> ChatOpenAI:
    """每中转站一个 ChatOpenAI（缓存）。🔴 stream_usage=True 是 token 不再 NULL 的关键。"""
    c = _chat_cache.get(relay.name)
    if c is None:
        c = ChatOpenAI(
            model=relay.model,
            temperature=0.5,
            streaming=True,
            stream_usage=True,  # 🔴 streaming 下必须开，否则 usage_metadata 为空（token NULL 根因）
            openai_api_base=relay.base_url,
            openai_api_key=relay.api_key,
        )
        _chat_cache[relay.name] = c
    return c


def _breaker(name: str) -> _Breaker:
    b = _breakers.get(name)
    if b is None:
        b = _Breaker()
        _breakers[name] = b
    return b


async def ainvoke_failover(
    messages: list[BaseMessage], *, max_tokens: int
) -> tuple[Any, str, str, int]:
    """按主→备顺序调用，熔断转移。

    返回 (resp, relay_name, relay_model, fallback_count)。
    🔴 relay_model = 实际成交中转站的 per-relay model（RELAY_POOL 各站可配不同模型），
    供上层 conv_trace/_trace_llm/cost_yuan 正确归因（不再恒写 COMPATIBLE_MODEL）。
    全部中转站不可用 → 抛最后一个异常（由上层记 error 后照常抛）。
    """
    relays = _relays()
    # 🔴 单站无备援时禁用熔断跳过：开闸 fail-fast 只降可用性（30s 内全灭且 last_exc=None
    #   只能抛笼统 RuntimeError），单站永远做真实尝试（旧版语义）。
    single = len(relays) == 1
    last_exc: Exception | None = None
    fallback = 0
    for relay in relays:
        br = _breaker(relay.name)
        if br.is_open() and not single:
            fallback += 1  # 跳过开闸的主站 = 一次转移
            continue
        try:
            model = _chat(relay).bind(max_tokens=max_tokens)
            resp = await model.ainvoke(messages)
            br.fails = 0
            br.open_until = 0.0  # 成功即复位
            return resp, relay.name, relay.model, fallback
        except Exception as e:  # noqa: BLE001 — 失败 → trip 计数 + 切下一个
            last_exc = e
            br.fails += 1
            if br.fails >= settings.RELAY_FAIL_THRESHOLD:
                br.open_until = time.monotonic() + settings.RELAY_COOLDOWN_S
            fallback += 1
    raise last_exc or RuntimeError("no relay available")


def usage_tokens(resp: Any) -> tuple[int | None, int | None]:
    """从 langchain resp 取 (prompt_tokens, completion_tokens)。

    usage_metadata（标准字段，stream_usage 开后有）优先，response_metadata.token_usage 兜底。
    """
    try:
        um = getattr(resp, "usage_metadata", None) or {}
        pt = um.get("input_tokens")
        ct = um.get("output_tokens")
        if pt is not None or ct is not None:
            return (int(pt) if pt is not None else None, int(ct) if ct is not None else None)
    except Exception:
        pass
    try:
        meta = getattr(resp, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        return (int(pt) if pt is not None else None, int(ct) if ct is not None else None)
    except Exception:
        return (None, None)


def _prices() -> dict[str, dict[str, float]]:
    global _prices_cache
    if _prices_cache is not None:
        return _prices_cache
    out: dict[str, dict[str, float]] = {}
    raw = (settings.RELAY_PRICES or "").strip()
    if raw:
        try:
            for model, p in json.loads(raw).items():
                out[str(model)] = {"in": float(p.get("in", 0)), "out": float(p.get("out", 0))}
        except Exception:
            out = {}
    _prices_cache = out
    return out


def cost_yuan(model: str | None, pt: int | None, ct: int | None) -> float | None:
    """按 RELAY_PRICES（¥/1k token，分 in/out）算实际消费。无价表/无 token → None。"""
    if not model or pt is None or ct is None:
        return None
    p = _prices().get(model)
    if not p:
        return None
    return round((pt * p["in"] + ct * p["out"]) / 1000.0, 6)
