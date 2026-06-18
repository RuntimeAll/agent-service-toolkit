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

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
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

# 🔴 PRD-C-100 B1b·内建默认价表（¥/1k token，in=prompt, out=completion）：opus 母题主链/造图翻命令
#   必须有价才能算 cost_yuan（口径对，G6）。RELAY_PRICES(.env) **覆盖** 本默认（D5 单价配置化，
#   促销价可能涨 → 改 .env 不改码）。默认只是兜底防 cost=None，不是事实源。
#   claude-opus-4-8：限时价 输入¥7/M=0.007/1k、输出¥35/M=0.035/1k（D5）。
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"in": 0.007, "out": 0.035},
}


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


# 🔴 2026-06-17 v2：双层超时防 sui-xiang 逆向站两类挂起（压测 round0 实证：单 read 超时不够）。
#   ① read-gap（httpx read）= 相邻 chunk 间隔上限。流式下每来一个 token 重置计时器 → 健康调用免疫，
#      「连上但 0 字节/死寂」挂起在 read 秒内抛 ReadTimeout。防【死寂挂起】。
#   ② total（asyncio.timeout）= 整次调用墙钟硬上限。逆向站「慢吐 reasoning 一直不收尾」会持续重置
#      read 计时器、read 永不触发 → 必须再加墙钟闸（round0 跑满 300s 的真因）。防【慢吐挂起】。
#   任一触发 → 抛异常 → ainvoke_failover 切下一站。母题读图慢，total 走入口传的 180s；其余默认 150s。
_READ_GAP_S = 60.0   # httpx read：相邻 chunk 间隔上限（覆盖最慢首 token，留余量）
_DEFAULT_TOTAL_S = 150.0  # asyncio 墙钟：单次调用总时长硬上限（实测合法最慢 114s + 余量）


def _httpx_timeout() -> httpx.Timeout:
    """🔴 四元组超时：read=相邻chunk间隔闸（防死寂），connect/write/pool 给小常量快失败。
    总时长闸不在这里，由 ainvoke_failover 的 asyncio.timeout 兜（防慢吐）。"""
    return httpx.Timeout(connect=10.0, read=_READ_GAP_S, write=10.0, pool=5.0)


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
            timeout=_httpx_timeout(),  # 🔴 read-gap 闸防死寂挂起（总时长闸在 ainvoke_failover）
        )
        _chat_cache[relay.name] = c
    return c


def _chat_override(
    relay: Relay, model: str, temperature: float = 0.5, timeout: float | None = None
) -> ChatOpenAI:
    """per-call 模型覆盖（S1.1）：同站点 base_url/api_key、只换 model 字段，按 (站名|model|温度)
    缓存独立实例（不污染整站缓存 _chat_cache）。轻活模型（nano）走这条不动主链路。

    🔴 PRD-C-017 M9：temperature 加 per-call 覆盖（默认 0.5 = 旧行为不变）。母题 opus 精确
    解题/结构化打标须低温（0.1~0.2），降 JSON 不稳 + 解题采样波动。缓存键带温度，避免
    同 (站|model) 不同温度互相覆盖实例。
    🔴 PRD-C-017 B1·H4 → 2026-06-17 v2：timeout（秒）现仅作【总时长墙钟闸】的入参，由
    ainvoke_failover 的 asyncio.timeout 读取（母题读图慢传 180s）；httpx 层一律用 read-gap 闸
    （_httpx_timeout），不再把 timeout 当 httpx 超时。缓存键仍带 timeout（无副作用，保持隔离）。"""
    ck = f"{relay.name}|{model}|t={temperature}|to={timeout}"
    c = _chat_cache.get(ck)
    if c is None:
        c = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=True,
            stream_usage=True,
            openai_api_base=relay.base_url,
            openai_api_key=relay.api_key,
            timeout=_httpx_timeout(),  # 🔴 read-gap 闸（总时长闸由 asyncio.timeout 管）
        )
        _chat_cache[ck] = c
    return c


def _breaker(name: str) -> _Breaker:
    b = _breakers.get(name)
    if b is None:
        b = _Breaker()
        _breakers[name] = b
    return b


async def ainvoke_failover(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    tags: list[str] | None = None,
    on_delta: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    timeout: float | None = None,
    prefer_relay: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[Any, str, str, int, str | None]:
    """按主→备顺序调用，熔断转移。

    返回 (resp, relay_name, relay_model, fallback_count, fallback_detail)。
    🔴 fallback_detail = 每站失败原因串（"sui-xiang:ReadTimeout; ..."），成功且 0 转移时为 None。
    供 conv_trace 回溯「中途切了/为什么切」（用户可查的熔断证据），不影响主流程。
    🔴 relay_model = 实际成交中转站的 per-relay model（RELAY_POOL 各站可配不同模型），
    供上层 conv_trace/_trace_llm/cost_yuan 正确归因（不再恒写 COMPATIBLE_MODEL）。
    全部中转站不可用 → 抛最后一个异常（由上层记 error 后照常抛）。

    tags：透传 langchain config.tags。带 "skip_stream" 的调用其 token 不会被 service
    转发给前端（service.py 按 metadata.tags 过滤）——JSON 类中间产物调用必须带它。
    on_delta：流内回调（每个 chunk 后拿到【累计文本】），用于 generate 出题进度计数。
    给了 on_delta 走 astream 手动聚合（stream_usage 开着，聚合块仍有 usage_metadata）；
    回调异常静默吞，绝不影响主流程。首 chunk 后失败不再 failover（半截流不可重放）。
    model：per-call 模型覆盖（S1.1，nano 降本前置）。给了就只换该次请求的 model 字段
    （站点 base_url/api_key 不变，绕过 _chat 缓存临时 bind 模型），返回的 relay_model
    仍归因到实际成交站名 + 这个覆盖模型；None = 完全沿用各站配置 model（旧行为不变）。
    temperature：per-call 温度覆盖（PRD-C-017 M9）。仅在 model 覆盖时生效（走 _chat_override）；
    None = 默认 0.5（旧行为不变）。母题 opus 档传低温（0.1~0.2）稳 JSON/解题。
    response_format：per-call 结构化输出（PRD-C-017 B1·F3）。给了就 bind 进请求（中转
    OpenAI-compatible json_schema 实测支持）；母题 opus 合并调用用它硬锁 10 维 schema。
    timeout：per-call 超时上限（秒，PRD-C-017 B1·H4）。仅在 model 覆盖时生效（重建 chat）；
    None = 不设上限（旧行为）。母题 opus 读图慢，须设 ≤180s 防挂死。
    prefer_relay：本次调用优先中转站名（思考链开关用）。给了就把同名站提到候选首位（其余顺序不变），
      熔断/failover 语义完全不变 —— 优先站不可用仍按序切下一站（graceful，绝不报错）。
      None（默认）= 沿用 .env RELAY_POOL 原序（旧行为，默认路径不受影响）。
    reasoning_effort：本次调用 extended-thinking 强度（"low"/"medium"/"high"，思考链开关用）。
      给了就与 response_format 同位 bind 进请求；支持的站(aigeek)据此吐 reasoning_content → on_reasoning
      外显；不支持的站(sui-xiang/kiro)静默忽略（OpenAI-compatible 丢未知字段，不报错）。None=不带（旧行为）。
    """
    relays = _relays()
    # 🔴 思考链开关：把 prefer_relay 同名站提到候选首位（其余原序不变）。优先站不可用仍按序 failover
    #   到下一站 —— prefer 只改尝试顺序，不改熔断/failover/默认行为；找不到该名则原序不动（graceful）。
    if prefer_relay:
        _pref = [r for r in relays if r.name == prefer_relay]
        if _pref:
            relays = _pref + [r for r in relays if r.name != prefer_relay]
    # 🔴 单站无备援时禁用熔断跳过：开闸 fail-fast 只降可用性（30s 内全灭且 last_exc=None
    #   只能抛笼统 RuntimeError），单站永远做真实尝试（旧版语义）。
    single = len(relays) == 1
    last_exc: Exception | None = None
    fallback = 0
    fail_reasons: list[str] = []  # 🔴 每站失败原因（回溯用），成功合并成 fallback_detail
    # 🔴 总时长墙钟闸（防慢吐挂起）：母题入口传 180s，其余默认 150s。read-gap 闸（防死寂）在 httpx 层。
    total_cap = timeout if (timeout is not None and timeout > 0) else _DEFAULT_TOTAL_S
    cfg: dict[str, Any] | None = {"tags": tags} if tags else None
    for relay in relays:
        br = _breaker(relay.name)
        if br.is_open() and not single:
            fallback += 1  # 跳过开闸的主站 = 一次转移
            fail_reasons.append(f"{relay.name}:breaker_open")
            continue
        try:
            # per-call 模型覆盖（S1.1）：换 model 字段须重建 ChatOpenAI（model 是构造期字段，
            # 非 per-call kwarg），缓存的整站实例不动；relay_model 归因到这个覆盖模型。
            chat = (
                _chat(relay)
                if model is None
                else _chat_override(
                    relay, model,
                    0.5 if temperature is None else temperature,
                    timeout=timeout,
                )
            )
            relay_model = relay.model if model is None else model
            bind_kw: dict[str, Any] = {"max_tokens": max_tokens}
            if response_format is not None:
                bind_kw["response_format"] = response_format
            # 🔴 思考链开关：reasoning_effort 与 response_format 同位 bind（OpenAI-compatible 顶层参数）。
            #   支持站(aigeek)开 extended-thinking → 吐 reasoning_content；不支持站静默忽略未知字段，不报错。
            if reasoning_effort is not None:
                bind_kw["reasoning_effort"] = reasoning_effort
            llm = chat.bind(**bind_kw)
            # 🔴 2026-06-17 v2：墙钟硬闸包整次调用（防逆向站慢吐 reasoning 永不收尾 = round0 真因）。
            #   read-gap 闸（httpx）管死寂、total_cap（asyncio）管慢吐，两层互补缺一不可。
            async with asyncio.timeout(total_cap):
                if on_delta is None and on_reasoning is None:
                    # cfg 为空不传（兼容测试桩的窄签名 ainvoke(messages)）
                    resp = await (llm.ainvoke(messages, config=cfg) if cfg else llm.ainvoke(messages))
                else:
                    # 🔴 PRD-C-100 D18 思考流式：on_reasoning 给了走 astream，逐 chunk 捞 content +
                    #   reasoning_content（additional_kwargs）；reasoning 累计回调（可折叠「思考中」块）。
                    #   思考型模型先吐 reasoning 再吐 content；只把 content 当正文，reasoning 单独转发。
                    resp = None
                    acc = ""
                    racc = ""
                    async for chunk in llm.astream(messages, config=cfg):
                        resp = chunk if resp is None else resp + chunk
                        try:
                            c = chunk.content
                            if isinstance(c, str) and c and on_delta is not None:
                                acc += c
                                on_delta(acc)
                        except Exception:  # noqa: BLE001 — 进度回调绝不炸主流程
                            pass
                        if on_reasoning is not None:
                            try:
                                ak = getattr(chunk, "additional_kwargs", None) or {}
                                rc = ak.get("reasoning_content")
                                if isinstance(rc, str) and rc:
                                    racc += rc
                                    on_reasoning(racc)
                            except Exception:  # noqa: BLE001
                                pass
                    if resp is None:
                        raise RuntimeError("empty stream")
            # 🔴 2026-06-17：sui-xiang 逆向站偶发返 200 + 内容空/空白（非硬错误，常规 failover 不触发）
            #   → 显式当失败、切下一站（aigeek 兜底）。只挡真空/空白（合法返回都远超），不误伤短返回。
            _c = getattr(resp, "content", None)
            if not (_c if isinstance(_c, str) else "").strip():
                raise RuntimeError("blank relay response (suspected truncation)")
            # 🔴 H3（2026-06-17 架构审计补）：截断哨兵。finish_reason=="length"（max_tokens 截断或
            #   逆向站 ~6400 幻影 token 提前截断）会让「非空但残缺的 JSON」过上面的 blank 检测被当成功
            #   → 下游 _parse_json 拿坏 JSON（母题有重试+修复接住，generate/solve 静默丢质量）。
            #   显式当失败 → 切下一站（aigeek 大概率完整收尾）；两站都截断才报错（响应确实坏）。
            #   逆向站不报 finish_reason 时本检查为 no-op（None != "length"），不误伤。
            try:
                _meta = getattr(resp, "response_metadata", None) or {}
                _fr = str(_meta.get("finish_reason") or _meta.get("stop_reason") or "")
            except Exception:  # noqa: BLE001
                _fr = ""
            if _fr == "length":
                raise RuntimeError("truncated relay response (finish_reason=length)")
            br.fails = 0
            br.open_until = 0.0  # 成功即复位
            return resp, relay.name, relay_model, fallback, ("; ".join(fail_reasons) or None)
        except Exception as e:  # noqa: BLE001 — 失败 → trip 计数 + 切下一个
            last_exc = e
            # 🔴 回溯标注（落 fallback_detail）：墙钟闸/截断/空返各给清晰短名，便于区分挂起类型。
            _m = str(e)
            if isinstance(e, asyncio.TimeoutError):
                ename = "WallClockTimeout"  # asyncio 墙钟闸（慢吐挂起），3.11 起=内置 TimeoutError
            elif "truncated" in _m:
                ename = "Truncated"  # H3 截断哨兵（finish_reason=length）
            elif "blank" in _m:
                ename = "BlankResp"  # 200 空/空白返回
            else:
                ename = type(e).__name__  # ReadTimeout（httpx 死寂闸）/ 硬错误走 type 名
            fail_reasons.append(f"{relay.name}:{ename}")
            br.fails += 1
            if br.fails >= settings.RELAY_FAIL_THRESHOLD:
                br.open_until = time.monotonic() + settings.RELAY_COOLDOWN_S
            fallback += 1
    # 全站失败：异常 message 带上所有站失败原因（落 conv_trace.error 供回溯）；链上 last_exc 保留类型。
    detail = "; ".join(fail_reasons) or "no relay available"
    raise RuntimeError(detail) from last_exc


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
    # 内建默认价（opus 等）打底，RELAY_PRICES(.env) 覆盖（D5 配置化，促销价改 .env 不改码）。
    out: dict[str, dict[str, float]] = {k: dict(v) for k, v in _DEFAULT_PRICES.items()}
    raw = (settings.RELAY_PRICES or "").strip()
    if raw:
        try:
            for model, p in json.loads(raw).items():
                out[str(model)] = {"in": float(p.get("in", 0)), "out": float(p.get("out", 0))}
        except Exception:
            pass  # 配置坏 → 保留默认（不清空，opus 仍有价兜底）
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
