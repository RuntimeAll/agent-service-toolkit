"""母题图上传链路真机探针：服务账号登录 → multipart 传 1x1 PNG → 验 envelope + 公网可读。

跑法: .venv/Scripts/python.exe tools/upload_probe.py
前置: book-server :8090 在跑。
"""

import asyncio
import base64
import sys

import httpx

from _probe_auth import real_token

sys.path.insert(0, r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\src")

from core import settings  # noqa: E402

# 1x1 红色 PNG
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


async def main() -> int:
    token = await real_token()
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as c:
        r = await c.post(
            f"{settings.RUOYI_BASE_URL}/teacher/variant/upload-image",
            headers={"Authorization": f"Bearer {token}", "clientid": settings.RUOYI_CLIENT_ID},
            files={"file": ("probe.png", PNG, "image/png")},
        )
        print("HTTP", r.status_code, r.text[:200])
        data = r.json()
        if data.get("code") != 1:
            print("FAIL: envelope code != 1")
            return 1
        url = (data.get("response") or {}).get("url")
        print("oss url:", url)
        # 公网匿名可读（LLM 中转要抓）
        r2 = await c.get(url)
        print("匿名 GET:", r2.status_code, f"{len(r2.content)} bytes")
        ok = r2.status_code == 200 and len(r2.content) == len(PNG)
        print("OK" if ok else "FAIL: 公网不可读")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
