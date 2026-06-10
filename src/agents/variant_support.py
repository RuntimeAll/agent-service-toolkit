"""PRD-C-009 举一反三 agent 的支撑层：只读图谱 SQL 锚定 + RuoYi 入库客户端。

🔴 架构铁律（14-举一反三-设计.md §9）：
- 数据归 RuoYi、计算归 agent；只读图谱 SQL 是架构允许的"纯只读"用途，不写业务数据。
- 入库走 RuoYi HTTP（双头鉴权 + envelope 解包），不直写 DB。
- 连接/口令从 toolkit settings(.env) 读，绝不内联明文（复用 ai-orchestrator/app/review.py 写法）。
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import pymysql

from core import settings


def _head_keyword(coarse_kp: str) -> str | None:
    """从 LLM 粗考点描述抽「头部考点词」，给全词 LIKE 落空时的降级匹配用。

    LLM 常给「二次根式的定义与识别」这类释义式描述，库内规范名是「二次根式有意义的条件」
    等叶子节点 —— 整句 LIKE 必落空。截到首个结构性虚词（的/与/及/和/或）前的实词头
    （如「二次根式」），按它再 LIKE 一次（仍是纯 SQL 名匹配降级，bge-m3 向量召回 future）。
    头部词须 ≥2 字且短于原串才有降级意义，否则返回 None（避免误命中无关节点）。
    """
    head = re.split(r"[的与及和或，,、（(]", coarse_kp, maxsplit=1)[0].strip()
    if len(head) >= 2 and len(head) < len(coarse_kp):
        return head
    return None


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

    🔴 编码 = 主键 id 本身（DDL：id="层级数字编码，每3位一层；根=学段+学科"），
       biz_subject 无独立 code 列（实际列 = id/parent_id/name/level/...）。
       原 SQL 选不存在的 code 列 → execute 抛 unknown column → classify 吞错 → 锚定恒空。
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
                # id 即层级数字编码（无独立 code 列）→ code = str(id)
                code = str(r.get("id") or "")
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
            "SELECT id, name FROM biz_subject WHERE name=%s LIMIT %s",
            (coarse_kp, limit),
        )
        # 2) LIKE 模糊补足（整句）
        if len(rows) < limit:
            _take(
                "SELECT id, name FROM biz_subject WHERE name LIKE %s LIMIT %s",
                (f"%{coarse_kp}%", limit - len(rows)),
            )
        # 3) 头部考点词降级 LIKE：整句释义落空时，截实词头再匹配（限叶子/节 level>=3，避顶层学科误命中）
        if not rows:
            head = _head_keyword(coarse_kp)
            if head:
                _take(
                    "SELECT id, name FROM biz_subject WHERE name LIKE %s AND level >= 3 LIMIT %s",
                    (f"%{head}%", limit),
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
        self._token: str | None = settings.RUOYI_TOKEN or None
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
        self, path: str, body: dict | None = None, _retry: bool = True
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

    async def lazy_tree(self, body: dict | None = None) -> Any:
        return await self.teacher_post("/teacher/question/lazyTree", body or {})

    async def create_question(self, body: dict) -> Any:
        """入库 = 新写 teacher 侧接口（落老师个人题库，身份由后端 LoginHelper 取，不信前端 createBy）。

        🔴 设计 §7：接口 /teacher/question/create（ruoyi-book），与 admin 解耦。
        """
        return await self.teacher_post("/teacher/question/create", body)


# ---------------------------------------------------------------------------
# 入库 BO 构造 + 逐题落库（设计 §7：变式+解析经 RuoYi HTTP 写老师个人题库，只写不判）
# ---------------------------------------------------------------------------
# 题型中文 → CreateQuestionBo.questionType（1=选择 / 4=填空 / 5=简答；misikt 真实 3 种）
QTYPE_MAP: dict[str, int] = {
    "选择": 1,
    "选择题": 1,
    "填空": 4,
    "填空题": 4,
    "解答": 5,
    "解答题": 5,
    "简答": 5,
    "简答题": 5,
    "计算": 5,
    "计算题": 5,
    "证明": 5,
    "证明题": 5,
}
DEFAULT_QTYPE = 5  # 拿不准 → 简答（最宽容）
DEFAULT_IMPORT_SOURCE = "AI-Orchestrator"  # 设计 §7 默认来源标记


def _map_qtype(qtype: Any) -> int:
    s = str(qtype or "").strip()
    if s.isdigit():
        return int(s)
    return QTYPE_MAP.get(s, DEFAULT_QTYPE)


def _clamp_difficult(difficulty: Any) -> int | None:
    """item.difficulty(1~5) → biz_question.difficult(1~4 星)；越界夹紧，缺则不传。"""
    try:
        d = int(difficulty)
    except (TypeError, ValueError):
        return None
    return max(1, min(4, d))


def build_create_bo(item: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """单道变式 item + 母题 facts → CreateQuestionBo camelCase body。

    🔴 只放契约允许的字段；createBy/createUser/status/id 绝不放（后端强制，传了也忽略）。
    题型映射成整数；难度夹 1~4；带 AI 血缘三件套（母题/变式关系/来源）。
    """
    bo: dict[str, Any] = {
        "questionType": _map_qtype(item.get("qtype") or facts.get("qtype")),
        "stem": item.get("stem") or "",
        "importSource": DEFAULT_IMPORT_SOURCE,
        "variantRelation": item.get("variant_relation") or "AI-数值变式",
    }
    answer = item.get("answer")
    if answer:
        bo["answer"] = answer
    analyze = item.get("solution") or item.get("analyze")
    if analyze:
        bo["analyze"] = analyze
    difficult = _clamp_difficult(item.get("difficulty"))
    if difficult is not None:
        bo["difficult"] = difficult

    # 知识点编码：classify 锚定到的真实节点 code（落 subjectId）
    subject_id = facts.get("subject_id")
    if subject_id:
        bo["subjectId"] = str(subject_id)

    # 母题血缘（库内母题时才有 id；图母题 MVP 无 id → 不传）
    mother_id = facts.get("mother_question_id")
    if mother_id:
        try:
            bo["motherQuestionId"] = int(mother_id)
        except (TypeError, ValueError):
            pass
    return bo


async def persist_items(
    items: list[dict[str, Any]], facts: dict[str, Any]
) -> list[dict[str, Any]]:
    """逐题落库（设计 §7：只写不判，质量门已在 solve_explain 闭合）。

    一个 RuoyiClient（teacher token 登录定身份）；逐题 POST /teacher/question/create。
    返回回执 [{ok, id?, error?}...]（部分失败不中断后续，回执如实标）。
    """
    client = RuoyiClient()
    receipts: list[dict[str, Any]] = []
    try:
        for item in items:
            bo = build_create_bo(item, facts)
            try:
                resp = await client.create_question(bo)
                # 后端回 question.getId()（雪花，回填后随 envelope.response 返回；
                # 形态可能是裸 id / {id:...} / {questionId:...}，宽容取）
                new_id = None
                if isinstance(resp, dict):
                    new_id = resp.get("id") or resp.get("questionId")
                elif resp is not None:
                    new_id = resp
                receipts.append({"ok": True, "id": new_id})
            except Exception as e:  # noqa: BLE001 — 单题失败如实记，不拖垮整组
                receipts.append({"ok": False, "error": str(e)})
    finally:
        await client.aclose()
    return receipts
