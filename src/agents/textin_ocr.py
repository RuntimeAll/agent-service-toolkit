# -*- coding: utf-8 -*-
r"""PRD-A-002 路B · 整页 → 富文本 外部 OCR（TextIn pdf_to_markdown）。

🔴 维护者拍板（2026-06-26）：路B 批量拆题的「整页抽取」改走 TextIn 外部 OCR
   —— 放宽 PRD-C-101 R3「零外部 OCR」，**仅限路B 整页**（路A 框选单题仍 opus 多模态）。
   产物 = 干净 Markdown 富文本（版式/表格/公式还原远胜 opus 裸读图），喂给 split_doc 切单题。

无状态：纯 HTTP 调用 + 文本归一，不读会话/DB。trust_env=False 绕本机代理（同 RuoyiClient）。
TextIn 接受 pdf / 常见图片字节（png/jpg/webp），按 endpoint 文档走 raw body。
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from core import settings

TEXTIN_TIMEOUT_S = 120.0


class TextinError(Exception):
    """TextIn 调用失败（凭据缺失 / 网关错误 / 非 200 code）。上层据此降级到 opus 兜底。"""


def _creds() -> tuple[str, str, str]:
    app_id = settings.TEXTIN_APP_ID
    secret = settings.TEXTIN_SECRET_CODE
    endpoint = settings.TEXTIN_ENDPOINT
    if not app_id or not secret:
        raise TextinError("TextIn 凭据未配置（.env TEXTIN_APP_ID/TEXTIN_SECRET_CODE）")
    secret_val = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
    return str(app_id), secret_val, str(endpoint)


def _to_bytes(file_base64: str | None, file_bytes: bytes | None) -> bytes:
    if file_bytes:
        return file_bytes
    b64 = (file_base64 or "").strip()
    if not b64:
        raise TextinError("file_base64 / file_bytes 至少给一个")
    # 容忍 data uri 前缀
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[-1]
    try:
        return base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        raise TextinError(f"base64 解码失败: {str(e)[:80]}") from e


def available() -> bool:
    """凭据是否就位（worker 据此决定走 TextIn 还是 opus 兜底）。"""
    return bool(settings.TEXTIN_APP_ID and settings.TEXTIN_SECRET_CODE)


async def page_to_markdown(
    *,
    file_base64: str | None = None,
    file_bytes: bytes | None = None,
) -> dict[str, Any]:
    """整页文件（pdf / 图片字节）→ Markdown 富文本。

    返回 {ok, markdown, pages, error}。永不抛到端点外由调用方收口；本函数内异常统一转
    TextinError 由 endpoint try 兜。无凭据/网关错误 → ok=False（上层可降级 opus 读图）。
    """
    app_id, secret, endpoint = _creds()
    data = _to_bytes(file_base64, file_bytes)

    # TextIn pdf_to_markdown：raw body + 头鉴权；参数走 query。
    #   markdown_details=0 只要纯 markdown；apply_document_tree=1 还原标题层级；
    #   parse_mode=auto 自动判扫描/文字层；get_image=none 不回传切图（我们只要文本+公式）。
    params = {
        "markdown_details": "1",
        "apply_document_tree": "1",
        "parse_mode": "auto",
        "get_image": "none",
        "formula_level": "1",
    }
    headers = {
        "x-ti-app-id": app_id,
        "x-ti-secret-code": secret,
        "Content-Type": "application/octet-stream",
    }
    async with httpx.AsyncClient(timeout=TEXTIN_TIMEOUT_S, trust_env=False) as client:
        try:
            resp = await client.post(endpoint, params=params, headers=headers, content=data)
        except Exception as e:  # noqa: BLE001
            raise TextinError(f"TextIn 网络异常: {str(e)[:120]}") from e

    if resp.status_code != 200:
        raise TextinError(f"TextIn HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        body = resp.json()
    except Exception as e:  # noqa: BLE001
        raise TextinError(f"TextIn 返回非 JSON: {str(e)[:120]}") from e

    code = body.get("code")
    if code != 200:
        raise TextinError(f"TextIn code={code} msg={body.get('message')}")

    result = body.get("result") or {}
    markdown = str(result.get("markdown") or "").strip()
    pages = result.get("pages")
    page_count = len(pages) if isinstance(pages, list) else None
    if not markdown:
        return {"ok": False, "markdown": "", "pages": page_count,
                "error": "TextIn 未抽出 Markdown（空文档或纯图无文字层）"}
    return {"ok": True, "markdown": markdown, "pages": page_count, "error": None}
