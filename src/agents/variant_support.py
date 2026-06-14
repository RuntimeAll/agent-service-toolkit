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

from agents import dna_extract
from core import settings


# ---------------------------------------------------------------------------
# 🔴 复习册前缀（PRD-C-009 整改·2026-06-13）：知识点树 level1 共 9 册——教材册 6
#   （3071/3072/3081/3082/3091/3092=七~九年级上下册）+ 复习册 3（下列三册）。数据层无
#   类型字段，按 level1 前缀（叶子 id 前 4 位）识别。
# 复习册同名考点与教材册**双挂**（如「二次根式有意义的条件」既挂八上教材册也挂复习册）——
# 锚定按考点名反查会命中复习册同名节点，generate 据此能锚到「3120004 未解析」「未知年级」
# 照样出题（实锚事故）。故锚定池**默认只圈教材册叶子，剔复习册**；老师明确要中考/复习/专题
# 时才并入（include_review_books=True）。三册名注释如下，号定不改。
# ---------------------------------------------------------------------------
REVIEW_BOOK_PREFIXES: set[str] = {
    "3010",  # 中考一轮复习
    "3100",  # 数学解题技巧与专题
    "3120",  # 新题抢先
}


def _is_review_book(node_id: Any) -> bool:
    """叶子/节点 id 是否属复习册（按 level1 前 4 位前缀判）。"""
    s = str(node_id or "").strip()
    return any(s.startswith(p) for p in REVIEW_BOOK_PREFIXES)


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


def anchor_subject(
    coarse_kp: str, limit: int = 8, *, exclude_review_books: bool = False
) -> list[dict[str, Any]]:
    """粗考点名 → biz_subject 真实节点候选（标准考点名 + 编码 + 年级=编码前4位）。

    SQL 精确匹配优先，命中不足时退 LIKE 模糊（MVP 用名匹配，向量召回 future）。
    返回 [{id, code, name, grade_code(编码前4位)}...]，按相关度粗排（精确在前）。
    🔴 纯只读 SELECT，pymysql 同步（节点已在 asyncio.to_thread 包裹外或 def 节点里调）。

    🔴 编码 = 主键 id 本身（DDL：id="层级数字编码，每3位一层；根=学段+学科"），
       biz_subject 无独立 code 列（实际列 = id/parent_id/name/level/...）。
       原 SQL 选不存在的 code 列 → execute 抛 unknown column → classify 吞错 → 锚定恒空。

    🔴 exclude_review_books（2026-06-13 整改）：按考点名反查时，同名考点双挂复习册（如
       「二次根式有意义的条件」既挂教材册又挂复习册）会让 _resolve_grade_code 反推出复习册
       的 grade_code（3010/3100/3120）→ 年级=未知/池跑偏。反查年级时传 True，从候选里
       剔掉复习册前缀节点（出题路径恒教材册年级），保锚定不串到复习册。
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
                # 反查年级时剔复习册同名节点（防 grade_code 反推到复习册前缀）
                if exclude_review_books and _is_review_book(r.get("id")):
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

    def __init__(self, token: str | None = None) -> None:
        # 🔴 身份透传：token 由 book-ui 透传的登录老师 access_token 时，直接用它定 owner
        # （后端 LoginHelper.getUserId() = 该老师），不再用 .env 服务账号登录。
        # forwarded=True 时禁止自动 login（会切回服务账号，归属就错了）。
        self._forwarded = bool(token)
        self._token: str | None = token or (settings.RUOYI_TOKEN or None)
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
            if self._forwarded:
                raise RuoyiError("缺登录老师 token，无法入库")
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
            # 透传 token 失效 → 不能重登（会切服务账号致归属错），让老师重登
            if self._forwarded:
                raise RuoyiError(f"{path} 401：登录老师 token 失效，请在平台重新登录后再试")
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

    async def tags_by_kp(self, kp_id: Any, limit: int = 300) -> list[dict[str, Any]]:
        """拉某知识点的高频标签复用池（T4·标签复用池客户端）。

        🔴 BE 派生端点 GET /teacher/question/tagsByKp?kpId={id}&limit={n}，misikt envelope
           {code:1, response:[{id,name,count}]}。**端点正在并行开发，现在调不通是预期**——
           本方法实现成「拉不到→返回空列表」由上层降级（绝不卡死）。
        """
        if kp_id in (None, "", "0"):
            return []
        await self._ensure_token()
        try:
            resp = await self._client.get(
                "/teacher/question/tagsByKp",
                params={"kpId": str(kp_id), "limit": int(limit)},
                headers=self._headers(),
            )
            data = resp.json()
        except Exception:  # noqa: BLE001 — 端点未上线/网络故障 → 降级空池，不抛
            return []
        if data.get("code") != 1:
            return []
        rows = data.get("response") or []
        return [r for r in rows if isinstance(r, dict)]

    async def create_question(self, body: dict) -> Any:
        """入库 = 新写 teacher 侧接口（落老师个人题库，身份由后端 LoginHelper 取，不信前端 createBy）。

        🔴 设计 §7：接口 /teacher/question/create（ruoyi-book），与 admin 解耦。
        """
        return await self.teacher_post("/teacher/question/create", body)

    async def update_question(self, body: dict) -> Any:
        """🔴 PRD-C-015 批4·缺口10·覆盖原行：重生后再入库 = update by id（不是新写一行）。

        接口 /teacher/question/update（ruoyi-book）：按 body.id 覆盖原行的题面/答案/解析 +
        重写 knowledges/free_tags/ai（先清后写，幂等）。owner 仍由后端 LoginHelper 校验
        （只许改自己的题）。body.id 必填。
        """
        return await self.teacher_post("/teacher/question/update", body)


# ---------------------------------------------------------------------------
# 两步锚定·叶子池 + 标签复用池（PRD-C-014 B1·dna_extract 的生产数据源）
# 🔴 池子走既有 RuoYi lazyTree/tagsByKp HTTP 封装，不直读 tsv、不直连库（kp_leaves.tsv
#    只作单测夹具）。任一拉取故障 → 返回空池由上层降级（绝不卡死）。
# ---------------------------------------------------------------------------
def _collect_leaves(tree: Any) -> list[tuple[str, str]]:
    """递归抽叶子（无 children 节点）→ [(id, name)...]（移植自 ai-orchestrator wf3）。"""
    leaves: list[tuple[str, str]] = []

    def walk(nodes: Any) -> None:
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            children = n.get("children") or []
            if not children:
                nid = str(n.get("id", "")).strip()
                nm = str(n.get("name", "")).strip()
                if nid:
                    leaves.append((nid, nm))
            else:
                walk(children)

    walk(tree if isinstance(tree, list) else [])
    return leaves


def _non_review_leaves(leaves: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """从全量叶子剔掉复习册（3010/3100/3120）—— 兜底圈池的安全集 = 全教材册叶子。"""
    return [(i, n) for i, n in leaves if not _is_review_book(i)]


def _collect_nodes_by_id(tree: Any) -> dict[str, str]:
    """递归把整棵树压成 {id: name}（含非叶子内部节点，给 M7 按 chapter_id 反查章名用）。纯函数。"""
    out: dict[str, str] = {}

    def walk(nodes: Any) -> None:
        for n in nodes or []:
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id", "")).strip()
            if nid:
                out[nid] = str(n.get("name", "")).strip()
            walk(n.get("children") or [])

    walk(tree if isinstance(tree, list) else [])
    return out


async def chapter_name_for_id(chapter_id: Any, client: RuoyiClient) -> str | None:
    """按 chapter_id（biz_subject 节点 id，通常是 level2 章 id）反查章名。

    🔴 M7：聚合/复习章识别需要章**名**（_is_review_book 只判 4 位册前缀，拦不住册内聚合章）。
    走既有 lazyTree HTTP 取树（不新写 pymysql），整树压平按 id 命中。拉不到/无命中 → None
    （上层据此降级：拿不到章名就不当聚合章处置，宁可不排除也不误排）。
    """
    cid = str(chapter_id or "").strip()
    if not cid:
        return None
    try:
        tree = await client.lazy_tree({})
    except Exception:  # noqa: BLE001 — 树拉不到 → None（上层降级）
        return None
    by_id = _collect_nodes_by_id(tree)
    return by_id.get(cid)


async def leaf_pool_for_grade(
    grade_code: str | None, client: RuoyiClient, *, include_review_books: bool = False
) -> list[tuple[str, str]]:
    """该年级叶子知识点池 [(id, name)...]（两步锚定第二步：年级 → 叶子池）。

    lazyTree 拉全树 → 抽叶子 → 按 grade_code（叶子 id 前缀）过滤本年级；grade_code 缺/
    过滤后空 → 兜底退回**全教材册叶子（剔复习册）**——而非裸 return 全量（1473 叶把复习册
    264 叶全放进池 → LLM 能锚到「3120004 未解析」照样出题，实锚事故的根因之一）。
    🔴 include_review_books=True（老师明确要中考/复习/专题时）→ 不剔复习册，全量参与圈池。
    拉取故障 → 返回 []（上层 → 锚定失败 → clarify）。
    """
    try:
        tree = await client.lazy_tree({})
    except Exception:  # noqa: BLE001 — 树拉不到 → 空池（上层降级走 clarify）
        return []
    leaves = _collect_leaves(tree)
    # 复习册除非显式开放，否则全程不入池（年级过滤段 + 兜底段都剔）
    if not include_review_books:
        leaves = _non_review_leaves(leaves)
    gc = str(grade_code or "").strip()
    if gc:
        scoped = [(i, n) for i, n in leaves if i.startswith(gc)]
        if scoped:
            return scoped
    # 兜底：grade_code 缺/过滤空 → 全教材册叶子（已剔复习册，绝不裸 return 全量含复习册）
    return leaves


async def tag_pool_for_kp(kp_id: Any, client: RuoyiClient, limit: int = 300) -> list[str]:
    """该知识点高频标签复用池（标签名列表，按 count 降序由 BE 给定）。

    🔴 BE 端点并行开发中，调不通 → 返回 [] 由 dna_extract 降级空池（标 flag 继续）。
    """
    rows = await client.tags_by_kp(kp_id, limit=limit)
    return [str(r.get("name")).strip() for r in rows if str(r.get("name") or "").strip()]


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
# 🔴 来源标记 = "举一反三"（与组卷服务 "AI-Orchestrator" 区分；之前照搬错标成组卷来源）
IMPORT_SOURCE = "举一反三"
REL_MOTHER = "原题(图)"  # 母题(从上传图抽出的原题)的 variant_relation
REL_VARIANT = "AI-数值变式"  # 变式题默认 variant_relation

# 🔴 PRD-C-009 轻量打标：举一反三入库即 AI 已标（label_status=1），打标人 = agent/模型标识。
LABEL_STATUS_AI = 1

# 🔴 PRD-C-015 批2·W5'：模型维落库走标签三轨的前缀（`模型:<name>`，零 DDL；26号 §0 注入协议④）。
MODEL_TAG_PREFIX = "模型:"

# 🔴 biz_question.subject_id 是 varchar(20) NOT NULL 无默认值（V1 建表）。锚定失败时
#   facts.subject_id = None（classify 没命中 biz_subject 节点，但 DNA 闸可能凭 LLM 置信放行）→
#   旧代码 `if subject_id:` 漏列 → 入库 INSERT subject_id=NULL → SQLException "Column 'subject_id'
#   cannot be null" → 母题+全部变式 500「发生未知异常」，成功 0 道（2026-06-12 实测真因，与
#   difficult NOT NULL 同类）。兜底 "0" = 未分类 sentinel（题库分页查询「subjectId 空或 '0' 不过滤」，
#   库里 0 行占用，无 DB 外键），落「未分类」可入库、可后续补打标，绝不再 NULL 炸库。
UNCLASSIFIED_SUBJECT_ID = "0"


def _labeled_by() -> str:
    """打标人标识 = 举一反三/<当前配置模型>。动态读 settings（换模型不再标错来源；
    多站 failover 下记的是配置默认站，逐次成交站已在 conv_trace/llm_trace 留痕）。"""
    try:
        from core import settings

        model = settings.COMPATIBLE_MODEL or str(settings.DEFAULT_MODEL or "")
    except Exception:
        model = ""
    return f"举一反三/{model or 'unknown'}"


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


def _clamp_conf(conf: Any) -> float | None:
    """锚定置信 → labelConfidence(0~1)；越界夹紧，缺/非数则不传（后端 @DecimalMin/Max 校验 0~1）。"""
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, c))


# DNA flags that signal the anchoring is unsafe → mark need_anchor_review
# （主 kp 越界 / LLM 解析失败 / LLM 调用异常 = 锚定不可信，转人审）
dna_extract_oob_flags: set[str] = {
    dna_extract.FLAG_MAIN_KP_OOB,
    dna_extract.FLAG_LLM_PARSE_FAIL,
    dna_extract.FLAG_LLM_ERROR,
}


def _to_int_or_none(v: Any) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _apply_labels(
    bo: dict[str, Any], facts: dict[str, Any], item: dict[str, Any] | None, role: str
) -> None:
    """把全维 DNA + 轻量打标塞进 CreateQuestionBo（PRD-C-014 B1·新 8 表契约）。

    🔴 维度来源（schema 收敛后，banner 覆盖正文）：
    - dim1KpId = facts.dim1_kp_id（主 kp 叶子 code；不再 = subjectId，subjectId 改科目锚 level1）；
    - dim2Qtype = 题型整数（与 questionType 同 _map_qtype 口径）；
    - dim4Difficulty = 难度 1~4（变式取 item，母题取 facts.mother_difficulty）；
    - dim5Structure = 母题结构简述（兼容存量；DNA 骨架另走 skeleton）。
    🔴 B1 新增键（与 BE 并行契约，键名钉死）：secondaryKpIds(int[]) / tags(str[]) /
       skeleton / scene / examType / hardPoints(str[]) / anchorId / needAnchorReview(bool) /
       reasoning。数据源 = facts.dna（classify 抽的 DNA 契约 v1；变式与母题共享守恒维 DNA）。
    🔴 已删除键（BE 实体已 DROP 这些列）：dim3Skill / auxTags / freeTag。
    label_status=1(AI已标) / labeled_by=agent标识 / labelConfidence=锚定置信。
    """
    dna = facts.get("dna") or {}

    # 主 kp 叶子 code（DNA 锚到的真知识点，知识点必绑）
    dim1 = facts.get("dim1_kp_id")
    if dim1:
        bo["dim1KpId"] = str(dim1)
    bo["dim2Qtype"] = bo.get("questionType")
    dim4 = _clamp_difficult(
        (item or {}).get("difficulty") if item is not None else facts.get("mother_difficulty")
    )
    if dim4 is not None:
        bo["dim4Difficulty"] = dim4
    if facts.get("mother_structure"):
        bo["dim5Structure"] = facts["mother_structure"]

    # 🔴 PRD-C-015 批2·缺口11 守恒维落库一致性（AC21/G21）：主 kp「0」未分类（dim1 缺）时，
    #   副考点/考察类型这两个**依赖知识图谱锚定的守恒维一并不落**，绝不产「主考点未分类却有副考点」
    #   的矛盾行。dim1 在库 → 正常落。（标签/骨架/难点不依赖主 kp 锚定，照旧落。）
    main_kp_unclassified = not dim1

    # 🔴 B1 全维 DNA（变式守恒 main kp/副 kp/考察类型/骨架，与母题共享 facts.dna） ---
    sec = dna.get("secondary_kps") or []
    sec_ids = [i for i in (_to_int_or_none(s.get("id")) for s in sec if isinstance(s, dict)) if i is not None]
    if sec_ids and not main_kp_unclassified:
        bo["secondaryKpIds"] = sec_ids
    tags = [str(t).strip() for t in (dna.get("tags") or []) if str(t).strip()]
    # 🔴 PRD-C-015 批2·W5' 模型维落库：模型名走标签三轨（前缀 `模型:`），零 DDL（不加列、不动 biz_question_ai）。
    #   透传给 book-server create（RuoYi 写库），toolkit 不直写库（架构铁律）。M00=概念直用 也照落（模型维非空）。
    #   去重：同名不重复挂；与普通 tag 各自独立（`模型:` 前缀避免与普通标签撞）。
    for m in dna.get("models") or []:
        if not isinstance(m, dict):
            continue
        mname = str(m.get("name") or "").strip()
        if not mname:
            continue
        mtag = f"{MODEL_TAG_PREFIX}{mname}"
        if mtag not in tags:
            tags.append(mtag)
    if tags:
        bo["tags"] = tags
    skeleton = dna.get("skeleton")
    if skeleton:
        # 骨架步骤序列 → 落 ai.solution_skeleton（全文）；list 拼成换行串
        bo["skeleton"] = "\n".join(str(s) for s in skeleton) if isinstance(skeleton, list) else str(skeleton)
    if dna.get("scene"):
        bo["scene"] = str(dna["scene"])
    # 考察类型守恒维（缺口11）：主 kp 未分类时不落（不产矛盾行）。
    if dna.get("exam_type") and not main_kp_unclassified:
        bo["examType"] = str(dna["exam_type"])
    hard = [str(h).strip() for h in (dna.get("hard_points") or []) if str(h).strip()]
    if hard:
        bo["hardPoints"] = hard  # BE 重算个数 → hard_point_count（不信 LLM 自报）

    # 锚定审计（→ ai 表 anchor_id / need_anchor_review / reasoning）
    if facts.get("dim1_kp_id"):
        bo["anchorId"] = str(facts["dim1_kp_id"])
    # 锚定存疑：DNA flags 含主 kp 越界 / 解析失败 → 需人审
    flags = dna.get("flags") or []
    bo["needAnchorReview"] = bool(
        not facts.get("dim1_kp_id")
        or dna_extract_oob_flags & set(flags)
    )
    if dna.get("reasoning"):
        bo["reasoning"] = str(dna["reasoning"])

    # 轻量打标
    bo["labelStatus"] = LABEL_STATUS_AI
    bo["labeledBy"] = _labeled_by()
    conf = _clamp_conf(facts.get("kp_confidence"))
    if conf is not None:
        bo["labelConfidence"] = conf


def build_mother_bo(facts: dict[str, Any]) -> dict[str, Any]:
    """从上传图抽出的「母题(原题)」→ CreateQuestionBo。

    🔴 设计 §7 原为「图母题不入库」；维护者 2026-06-10 拍板改为：图母题不在库时**先把原题入库**，
       变式再 motherQuestionId 指向它 → 血缘完整可追。母题图 URL 落 stemImg。
    🔴 PRD-C-014 B1：全维 DNA + 轻量打标由 _apply_labels 统一塞（role="mother"）；
       subjectId=科目锚 level1、dim1KpId=主 kp 叶子；新 8 表键见 _apply_labels（已删 auxTags 等）。
    """
    bo: dict[str, Any] = {
        "questionType": _map_qtype(facts.get("qtype")),
        "stem": facts.get("stem") or "",
        "importSource": IMPORT_SOURCE,
        "variantRelation": REL_MOTHER,
    }
    if facts.get("mother_answer"):
        bo["answer"] = facts["mother_answer"]
    if facts.get("mother_solution"):
        bo["analyze"] = facts["mother_solution"]
    # 🔴 biz_question.difficult 是 NOT NULL 且无 DB 默认值（book-server 实测：缺列 → SQLException
    #   "Field 'difficult' doesn't have a default value" → create 500）。故难度缺/非数时必兜底，绝不省列。
    difficult = _clamp_difficult(facts.get("mother_difficulty"))
    if difficult is None:
        difficult = 2  # 常规档兜底
    bo["difficult"] = difficult
    # 🔴 NOT NULL 列：锚定失败(None)必兜底 "0"（未分类），绝不漏列致 INSERT NULL → 500（同 build_create_bo）。
    msid = facts.get("subject_id")
    bo["subjectId"] = str(msid) if msid else UNCLASSIFIED_SUBJECT_ID
    if facts.get("image_url"):
        bo["stemImg"] = facts["image_url"]  # 母题图落题干图字段
    _apply_labels(bo, facts, item=None, role="mother")
    return bo


def build_create_bo(item: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """单道变式 item + 母题 facts → CreateQuestionBo camelCase body。

    🔴 只放契约允许的字段；createBy/createUser/status/id 绝不放（后端强制，传了也忽略）。
    题型映射成整数；难度夹 1~4；带 AI 血缘（母题 id / 变式关系 / 来源）。
    🔴 PRD-C-014 B1：全维 DNA（副 kp/标签/骨架/场景/考察类型/难点 + 锚定审计）+ 轻量打标
       由 _apply_labels 统一塞（role="variant"）；变式与母题共享 facts.dna 守恒维。
    """
    bo: dict[str, Any] = {
        "questionType": _map_qtype(item.get("qtype") or facts.get("qtype")),
        "stem": item.get("stem") or "",
        "importSource": IMPORT_SOURCE,
        "variantRelation": item.get("variant_relation") or REL_VARIANT,
    }
    answer = item.get("answer")
    if answer:
        bo["answer"] = answer
    analyze = item.get("solution") or item.get("analyze")
    if analyze:
        bo["analyze"] = analyze
    # 🔴 biz_question.difficult 是 NOT NULL 且无 DB 默认值（缺列 → create 500）。难度缺/非数时
    #   按 item → 母题 → 2(常规) 兜底链，绝不省列。二期 P8 _grade_difficulty 降级可能留下非数难度，
    #   旧代码 `if difficult is not None` 会漏列 → c012 P5b item#0 入库 500 的真因。
    difficult = _clamp_difficult(item.get("difficulty"))
    if difficult is None:
        difficult = _clamp_difficult(facts.get("mother_difficulty"))
    if difficult is None:
        difficult = 2  # 常规档兜底
    bo["difficult"] = difficult

    # subjectId = 科目锚 level1（学段学科册，= grade code，如 3071=七上）；知识点叶子 code
    #   不落这里——那是 dim1KpId（主 kp）的活（V905 后语义，见 _mother_facts §B1）。
    # 🔴 NOT NULL 列：锚定失败(None)必兜底 "0"（未分类），绝不漏列致 INSERT NULL → 500。
    subject_id = facts.get("subject_id")
    bo["subjectId"] = str(subject_id) if subject_id else UNCLASSIFIED_SUBJECT_ID

    # 母题血缘：图母题入库后回填的 mother_question_id（雪花大整数）
    mother_id = facts.get("mother_question_id")
    if mother_id:
        try:
            bo["motherQuestionId"] = int(mother_id)
        except (TypeError, ValueError):
            pass
    _apply_labels(bo, facts, item=item, role="variant")
    return bo


def build_update_bo(item: dict[str, Any], facts: dict[str, Any], qid: Any) -> dict[str, Any]:
    """🔴 PRD-C-015 批4·缺口10·覆盖原行 BO = build_create_bo + id（雪花大整数）。

    BE /teacher/question/update 据 id 覆盖题面/答案/解析 + 重写 knowledges/free_tags/ai
    （先清后写，幂等）。owner 由后端校验（只许改自己的题）。
    """
    bo = build_create_bo(item, facts)
    try:
        bo["id"] = int(qid)
    except (TypeError, ValueError):
        bo["id"] = qid
    return bo


def _extract_new_id(resp: Any) -> Any:
    """从 /teacher/question/create 回执宽容取雪花 id（裸 id / {id} / {questionId}）。"""
    if isinstance(resp, dict):
        return resp.get("id") or resp.get("questionId")
    return resp if resp is not None else None


async def persist_items(
    items: list[dict[str, Any]], facts: dict[str, Any], token: str | None = None
) -> list[dict[str, Any]]:
    """落库（设计 §7：只写不判，质量门已在 solve_explain 闭合）。

    🔴 母题优先：图母题不在库（facts 无 mother_question_id）且有原题题干时，**先入库母题**，拿到
       雪花 id 回填 facts.mother_question_id，变式再挂血缘指向它。组键 = 母题 id。
    🔴 身份：token 非空 = book-ui 透传的登录老师 access_token → owner=该老师；为空则退回 .env
       服务账号（regression/直连脚本用）。
    逐题 POST /teacher/question/create。返回回执 [{ok, id?, error?, role}...]（role=mother/variant）。
    """
    facts = dict(facts)
    client = RuoyiClient(token=token)
    receipts: list[dict[str, Any]] = []
    try:
        # 0) 母题(原题)入库 → 回填 mother_question_id（仅图母题不在库且有题干时）
        if not facts.get("mother_question_id") and (facts.get("stem") or "").strip():
            try:
                mid = _extract_new_id(await client.create_question(build_mother_bo(facts)))
                if mid is not None:
                    facts["mother_question_id"] = mid
                receipts.append({"ok": True, "id": mid, "role": "mother"})
            except Exception as e:  # noqa: BLE001 — 母题入库失败：变式仍照常落（血缘缺而已）
                receipts.append({"ok": False, "error": f"母题入库失败：{e}", "role": "mother"})
        # 🔴 PRD-C-015 批4·缺口10·母题已入库 + 母题 DNA 改了（mother_dirty）→ update role=mother 行
        #   同步守恒维（变式血缘基准一致）。仅当母题已有 id（在库）且本次标了脏才同步。
        elif facts.get("mother_question_id") and facts.get("mother_dirty") and (facts.get("stem") or "").strip():
            try:
                mbo = build_mother_bo(facts)
                mbo["id"] = int(facts["mother_question_id"]) if str(facts["mother_question_id"]).isdigit() \
                    else facts["mother_question_id"]
                await client.update_question(mbo)
                receipts.append({"ok": True, "id": facts["mother_question_id"], "role": "mother", "updated": True})
            except Exception as e:  # noqa: BLE001 — 母题同步失败：变式仍照常落
                receipts.append({"ok": False, "error": f"母题同步失败：{e}", "role": "mother"})

        # 1) 逐题入库变式（此时 facts.mother_question_id 已回填）。
        #    🔴 缺口10·覆盖原行：item 带 _persist_id（已入库、重生后再入库）→ update by id；否则 create。
        for item in items:
            persist_id = item.get("_persist_id")
            try:
                if persist_id:
                    new_id = _extract_new_id(await client.update_question(build_update_bo(item, facts, persist_id)))
                    receipts.append({"ok": True, "id": new_id or persist_id, "role": "variant", "updated": True})
                else:
                    new_id = _extract_new_id(await client.create_question(build_create_bo(item, facts)))
                    receipts.append({"ok": True, "id": new_id, "role": "variant"})
            except Exception as e:  # noqa: BLE001 — 单题失败如实记，不拖垮整组
                receipts.append({"ok": False, "error": str(e), "role": "variant"})
    finally:
        await client.aclose()
    return receipts
