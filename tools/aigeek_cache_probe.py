"""aigeek 缓存命中真机探针（PRD-C-012 中转切换验收 G7 前置）。

验证两件事：
1. RELAY_POOL 主站（aigeek）/chat/completions 可用；
2. 长固定前缀连发两次，第二次 usage.prompt_tokens_details.cached_tokens > 0（缓存命中坐实，
   命中价 = 输入价 1/10，见 中转站迁移预研-aigeek.md §3/§7）。

跑法: .venv/Scripts/python.exe tools/aigeek_cache_probe.py
零依赖服务（直连中转，不经 :8093）。
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# 长固定前缀：稳定 system prompt，凑够 OpenAI 前缀缓存门槛（≥1024 token）
_RULES = (
    "你是 K12 数学变式题质检员。判定规则："
    "①主考点必须与母题一致，不得漂移到相邻章节；"
    "②年级学段硬守恒，七年级题不得使用八年级及以上知识点；"
    "③题型骨架（选择/填空/解答/证明）必须与母题相同；"
    "④难度档位 normal 与母题持平，hard 恰高一档；"
    "⑤数字与场景皮肤必须更换，禁止复读母题题面；"
    "⑥答案必须可程序验算，优先数值或表达式形式；"
    "⑦解析需含规范步骤与最终答案，数学式用 $...$ 包裹。"
)
SYSTEM_PREFIX = "".join(f"[规则组{i:02d}] {_RULES}\n" for i in range(24))  # ~4000 汉字


async def _call(client: httpx.AsyncClient, base: str, key: str, model: str, q: str) -> dict:
    r = await client.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user", "content": q},
            ],
        },
    )
    r.raise_for_status()
    return r.json()


async def main() -> int:
    pool = json.loads(os.environ["RELAY_POOL"])
    relay = pool[0]
    print(f"主站: {relay['name']} {relay['base_url']} model={relay['model']}")
    hits = []
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as c:
        for i in range(5):
            u = (await _call(c, relay["base_url"], relay["api_key"], relay["model"], f"回复『{i}』即可"))["usage"]
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            hits.append(cached)
            print(f"第{i + 1}发: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} cached={cached}")
            await asyncio.sleep(2)
    if any(h > 0 for h in hits[1:]):
        print(f"OK 缓存命中坐实（5 发中 {sum(1 for h in hits[1:] if h > 0)} 次命中，best-effort 路由）")
        return 0
    print("FAIL 连发 5 次 cached_tokens 均 0（聚合路由不黏/中转未回报，需查）")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
