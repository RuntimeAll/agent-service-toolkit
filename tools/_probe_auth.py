"""探针/冒烟共用：拿**真** RuoYi token（.env 服务账号登录）。

2026-06-11 身份硬闸后，variant graph 入口要求 config.configurable.ruoyi_token 能解出
userId，否则一步不走（route_entry → require_login）。探针不伪造 token（伪 token 过得了
入口闸但 persist 调 RuoYi 会 401），统一走服务账号真登录 —— 与生产链路同构，
conv_llm_trace 归属 = 服务账号 userId（与真实老师流量可区分）。

前置：book-server :8090 在跑（探针们本来就要求）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents.variant_support import RuoyiClient  # noqa: E402


async def real_token() -> str:
    """服务账号登录拿 access_token（调用方自己缓存，别每轮登一次）。"""
    client = RuoyiClient()
    try:
        return await client.login()
    finally:
        await client.aclose()
