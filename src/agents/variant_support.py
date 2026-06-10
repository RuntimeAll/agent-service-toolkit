"""PRD-C-009 举一反三 agent 的支撑层：只读图谱 SQL 锚定 + RuoYi 入库客户端。

🔴 架构铁律（14-举一反三-设计.md §9）：
- 数据归 RuoYi、计算归 agent；只读图谱 SQL 是架构允许的"纯只读"用途，不写业务数据。
- 入库走 RuoYi HTTP（双头鉴权 + envelope 解包），不直写 DB。
- 连接/口令从 toolkit settings(.env) 读，绝不内联明文（复用 ai-orchestrator/app/review.py 写法）。
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
import pymysql

from core import settings


# ---------------------------------------------------------------------------
# 只读图谱：锚 biz_subject 真实节点（SQL 精确/LIKE 优先，bge-m3 向量召回作降级增强）
# ---------------------------------------------------------------------------
def _db_kwargs() -> dict:
    pwd = settings.VARIANT_DB_PASSWORD
    return dict(
        host=settings.VARIANT_DB_HOST,
        port=settings.VARIANT_DB_PORT,
        user=settings.VARIANT_DB_USER,
        password=pwd.get_secret_value() if pwd else "",
        database=settings.VARIANT_DB_NAME,
        charset="utf8mb4",
    )


def anchor_subject(coarse_kp: str, limit: int = 8) -> list[dict[str, Any]]:
    """粗考点名 → biz_subject 真实节点候选（标准考点名 + 编码 + 年级=编码前4位）。

    SQL 精确匹配优先，命中不足时退 LIKE 模糊（MVP 用名匹配，向量召回 future）。
    返回 [{id, code, name, grade_code(编码前4位)}...]，按相关度粗排（精确在前）。
    🔴 纯只读 SELECT，pymysql 同步（节点已在 asyncio.to_thread 包裹外或 def 节点里调）。
    """
    coarse_kp = (coarse_kp or "").strip()
    if not coarse_kp:
        return []
    conn = pymysql.connect(**_db_kwargs())
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        rows: list[dict] = []
        seen: set = set()

        def _take(sql: str, params: tuple) -> None:
            cur.execute(sql, params)
            for r in cur.fetchall():
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                code = str(r.get("code") or "")
                rows.append(
                    {
                        "id": r["id"],
                        "code": code,
                        "name": r.get("name"),
                        "grade_code": code[:4] if len(code) >= 4 else None,
                    }
                )

        # 1) 精确名匹配
        _take(
            "SELECT id, code, name FROM biz_subject WHERE name=%s LIMIT %s",
            (coarse_kp, limit),
        )
        # 2) LIKE 模糊补足
        if len(rows) < limit:
            _take(
                "SELECT id, code, name FROM biz_subject WHERE name LIKE %s LIMIT %s",
                (f"%{coarse_kp}%", limit - len(rows)),
            )
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# RuoYi 入库客户端（双头鉴权 + envelope 解包；范式移植自 ai-orchestrator/app/clients/ruoyi.py）
# ---------------------------------------------------------------------------
class RuoyiError(Exception):
    pass


class RuoyiClient:
    """C 线 book-server :8090 客户端。

    🔴 双头铁律：/teacher/** 必带 Authorization Bearer + clientid，缺 clientid 必 401。
    🔴 envelope：/teacher/** 响应被 advice 重写成 {code:1, message, response}，按 code==1 取 response。
    🔴 trust_env=False：调本地/内网 :8090 时禁读系统代理，否则被 Clash 吞超时。
    """

    def __init__(self) -> None:
        self._token: Optional[str] = settings.RUOYI_TOKEN or None
        self._client = httpx.AsyncClient(
            base_url=settings.RUOYI_BASE_URL, timeout=30.0, trust_env=False
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def login(self) -> str:
        pwd = settings.RUOYI_PASSWORD
        body = {
            "clientId": settings.RUOYI_CLIENT_ID,
            "grantType": "password",
            "tenantId": settings.RUOYI_TENANT_ID,
            "username": settings.RUOYI_USERNAME,
            "password": pwd.get_secret_value() if pwd else "",
        }
        resp = await self._client.post(
            "/auth/login", json=body, headers={"clientid": settings.RUOYI_CLIENT_ID}
        )
        try:
            data = resp.json()
        except Exception:
            raise RuoyiError(f"登录响应非 JSON: status={resp.status_code} body={resp.text[:200]}")
        if data.get("code") != 200:
            raise RuoyiError(f"登录失败 code={data.get('code')} msg={data.get('msg')}")
        token = (data.get("data") or {}).get("access_token")
        if not token:
            raise RuoyiError(f"登录返回无 access_token: {data}")
        self._token = token
        return token

    async def _ensure_token(self) -> str:
        if not self._token:
            await self.login()
        return self._token  # type: ignore[return-value]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "clientid": settings.RUOYI_CLIENT_ID,
            "Content-Type": "application/json",
        }

    async def teacher_post(
        self, path: str, body: Optional[dict] = None, _retry: bool = True
    ) -> Any:
        """调 /teacher/** 接口，解 envelope（code==1 取 response）。401 自动重登一次。"""
        await self._ensure_token()
        resp = await self._client.post(path, json=body or {}, headers=self._headers())
        if resp.status_code == 401 and _retry:
            self._token = None
            await self.login()
            return await self.teacher_post(path, body, _retry=False)
        try:
            data = resp.json()
        except Exception:
            raise RuoyiError(f"{path} 响应非 JSON: status={resp.status_code} body={resp.text[:200]}")
        if data.get("code") != 1:
            msg = data.get("message") or data.get("msg")
            raise RuoyiError(f"{path} 非 code==1: code={data.get('code')} msg={msg}")
        return data.get("response")

    async def lazy_tree(self, body: Optional[dict] = None) -> Any:
        return await self.teacher_post("/teacher/question/lazyTree", body or {})

    async def create_question(self, body: dict) -> Any:
        """入库 = 新写 teacher 侧接口（落老师个人题库，身份由后端 LoginHelper 取，不信前端 createBy）。

        🔴 设计 §7：接口 /teacher/question/create（ruoyi-book），与 admin 解耦。
        """
        return await self.teacher_post("/teacher/question/create", body)
