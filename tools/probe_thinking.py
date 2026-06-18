"""一次性探针：实测中转(sui-xiang / aigeek)对 opus extended-thinking 的支持。

目的=证实「中转是否透传 thinking 参数 + 上游是否真吐 reasoning_content」，
供用户拍开/不开 extended-thinking。**只探，不改业务代码。**

逐个试三种业界传法 × 流式/非流式 × 两站点，记录：
  报错? / 正常返回? / 返回里有没有 reasoning(_content)/thinking 字段 / content 是否正常。

用法（cwd = agent-service-toolkit）：
  .venv\Scripts\python.exe tools\probe_thinking.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

# 🔴 PS5.1 控制台 GBK，强制 stdout UTF-8 防中文/数学符号 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

from openai import AsyncOpenAI

# ── 从 .env 直接读 RELAY_POOL（不经 settings，探针自包含）────────────────
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


def _load_relays() -> list[dict[str, str]]:
    raw = ""
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("RELAY_POOL="):
                raw = line[len("RELAY_POOL="):]
                break
    return json.loads(raw)


PROMPT = "一元二次方程 x^2-5x+6=0 的两根之和是多少？简要说明你的推理。"

# 三种 extended-thinking 传法（逐个试）。每个给一个调用参数构造器。
VARIANTS: list[dict[str, Any]] = [
    {
        "id": "anthropic-style-thinking",
        "desc": 'extra_body={"thinking":{"type":"enabled","budget_tokens":1024}}',
        "extra": {"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 1024}}},
    },
    {
        "id": "openai-reasoning_effort",
        "desc": 'reasoning_effort="low"',
        "extra": {"reasoning_effort": "low"},
    },
    {
        "id": "extra_body-reasoning-effort",
        "desc": 'extra_body={"reasoning":{"effort":"low"}}',
        "extra": {"extra_body": {"reasoning": {"effort": "low"}}},
    },
    {
        "id": "baseline-no-thinking",
        "desc": "对照组：完全不带 thinking 参数",
        "extra": {},
    },
]

REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def _scan_reasoning(obj: Any) -> dict[str, Any]:
    """在任意对象上找 reasoning 类字段（含 model_extra / dict）。返回命中映射。"""
    hits: dict[str, Any] = {}
    candidates: list[Any] = []
    # pydantic model_extra（OpenAI SDK 把未知字段塞这）
    me = getattr(obj, "model_extra", None)
    if isinstance(me, dict):
        candidates.append(me)
    # 直接属性
    for k in REASONING_KEYS:
        v = getattr(obj, k, None)
        if v:
            hits[f"attr.{k}"] = v
    # dict 形态
    if isinstance(obj, dict):
        candidates.append(obj)
    for d in candidates:
        for k in REASONING_KEYS:
            if d.get(k):
                hits[f"extra.{k}"] = d[k]
    return hits


def _shorten(s: Any, n: int = 200) -> str:
    t = str(s).replace("\n", " ")
    return t if len(t) <= n else t[:n] + f"...<{len(t)}字>"


async def run_nonstream(client: AsyncOpenAI, model: str, variant: dict) -> dict:
    res: dict[str, Any] = {"mode": "non-stream", "variant": variant["id"], "desc": variant["desc"]}
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=2048,
            **variant["extra"],
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning_hits: dict[str, Any] = {}
        reasoning_hits.update(_scan_reasoning(msg))
        # 有些站把 reasoning 放 choice 上
        reasoning_hits.update({f"choice.{k}": v for k, v in _scan_reasoning(resp.choices[0]).items()})
        res.update(
            ok=True,
            error=None,
            has_reasoning=bool(reasoning_hits),
            reasoning_keys=list(reasoning_hits.keys()),
            reasoning_sample={k: _shorten(v) for k, v in reasoning_hits.items()},
            content_len=len(content),
            content_sample=_shorten(content),
            finish_reason=resp.choices[0].finish_reason,
        )
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {_shorten(e, 400)}")
    return res


async def run_stream(client: AsyncOpenAI, model: str, variant: dict) -> dict:
    res: dict[str, Any] = {"mode": "stream", "variant": variant["id"], "desc": variant["desc"]}
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=2048,
            stream=True,
            **variant["extra"],
        )
        content = ""
        reasoning_keys: set[str] = set()
        reasoning_sample: dict[str, str] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            if getattr(delta, "content", None):
                content += delta.content
            hits = _scan_reasoning(delta)
            for k, v in hits.items():
                reasoning_keys.add(k)
                reasoning_sample[k] = reasoning_sample.get(k, "") + str(v)
        res.update(
            ok=True,
            error=None,
            has_reasoning=bool(reasoning_keys),
            reasoning_keys=sorted(reasoning_keys),
            reasoning_sample={k: _shorten(v) for k, v in reasoning_sample.items()},
            content_len=len(content),
            content_sample=_shorten(content),
        )
    except Exception as e:  # noqa: BLE001
        res.update(ok=False, error=f"{type(e).__name__}: {_shorten(e, 400)}")
    return res


async def probe_station(relay: dict) -> list[dict]:
    name = relay["name"]
    client = AsyncOpenAI(base_url=relay["base_url"], api_key=relay["api_key"], timeout=120.0)
    model = relay["model"]
    results: list[dict] = []
    for variant in VARIANTS:
        for runner in (run_nonstream, run_stream):
            print(f"  [{name}] {runner.__name__} / {variant['id']} ...", flush=True)
            r = await runner(client, model, variant)
            r["station"] = name
            results.append(r)
    return results


async def main() -> None:
    relays = _load_relays()
    all_results: list[dict] = []
    for relay in relays:
        print(f"\n=== 探针站点: {relay['name']} ({relay['base_url']}, model={relay['model']}) ===", flush=True)
        all_results.extend(await probe_station(relay))

    # 汇总打印
    print("\n\n############ 汇总 ############")
    for r in all_results:
        flag = "ERROR" if not r.get("ok") else ("REASONING✓" if r.get("has_reasoning") else "no-reasoning")
        print(f"\n[{r['station']}] {r['mode']} | {r['variant']}  => {flag}")
        print(f"    传法: {r['desc']}")
        if not r.get("ok"):
            print(f"    报错: {r['error']}")
        else:
            print(f"    reasoning字段: {r.get('reasoning_keys')}")
            if r.get("reasoning_sample"):
                print(f"    reasoning样例: {r['reasoning_sample']}")
            print(f"    content({r.get('content_len')}字): {r.get('content_sample')}")

    out_path = os.path.join(os.path.dirname(__file__), "probe_thinking_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
