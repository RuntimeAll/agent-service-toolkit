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
# 🔴 来源标记 = "举一反三"（与组卷服务 "AI-Orchestrator" 区分；之前照搬错标成组卷来源）
IMPORT_SOURCE = "举一反三"
REL_MOTHER = "原题(图)"  # 母题(从上传图抽出的原题)的 variant_relation
REL_VARIANT = "AI-数值变式"  # 变式题默认 variant_relation

# 🔴 PRD-C-009 轻量打标：举一反三入库即 AI 已标（label_status=1），打标人 = agent/模型标识。
LABEL_STATUS_AI = 1


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


def _apply_labels(
    bo: dict[str, Any], facts: dict[str, Any], item: dict[str, Any] | None, role: str
) -> None:
    """把 5 维度 DNA + 轻量打标 + auxTags 塞进 CreateQuestionBo（PRD-C-009 入库存 DNA/打标）。

    🔴 维度来源：
    - dim1KpId = subjectId（锚定到的真实节点 code，知识点必绑，与 subjectId 同源）；
    - dim2Qtype = 题型整数（与 questionType 同 _map_qtype 口径）；
    - dim4Difficulty = 难度 1~4（变式取 item，母题取 facts.mother_difficulty）；
    - dim5Structure = 母题结构指纹（mother_dna.structure；变式与母题共享结构 DNA）；
    - dim3Skill = 思维方法数组（analyze 暂未抽 → 缺则不带）。
    label_status=1(AI已标) / labeled_by=agent标识 / labelConfidence=锚定置信。
    auxTags = {agent, role, sourceImage}（V16 aux_tags 列已补，可恢复存）。
    """
    src = item or facts
    subject_id = facts.get("subject_id")
    if subject_id:
        bo["dim1KpId"] = str(subject_id)  # 知识点必绑
    bo["dim2Qtype"] = bo.get("questionType")
    dim4 = _clamp_difficult(
        src.get("difficulty") if item is not None else facts.get("mother_difficulty")
    )
    if dim4 is not None:
        bo["dim4Difficulty"] = dim4
    if facts.get("mother_structure"):
        bo["dim5Structure"] = facts["mother_structure"]
    skills = (item or {}).get("skills") or facts.get("skills")
    if isinstance(skills, list) and skills:
        bo["dim3Skill"] = [str(s) for s in skills]

    # 轻量打标
    bo["labelStatus"] = LABEL_STATUS_AI
    bo["labeledBy"] = _labeled_by()
    conf = _clamp_conf(facts.get("kp_confidence"))
    if conf is not None:
        bo["labelConfidence"] = conf

    # 血缘溯源标签（aux_tags 列 V16 已补，恢复存；不放业务数据）
    aux: dict[str, Any] = {"agent": IMPORT_SOURCE, "role": role}
    if facts.get("image_url"):
        aux["sourceImage"] = facts["image_url"]
    # 🔴 PRD-C-010 闸B 验算标记透传（item.check 由 solve_explain 填）：
    # verify = sympy_pass / fail_after_regen / unverified（入库可查的真机验证抓手）；
    # review = proof_needs_human（证明/开放类人审兜底——labelStatus 维持 LABEL_STATUS_AI=1，
    # 待审语义挂 auxTags.review，不发明新 label_status 值）。
    check = (item or {}).get("check") or {}
    if check.get("verify"):
        aux["verify"] = str(check["verify"])
    if check.get("review"):
        aux["review"] = str(check["review"])
    # 🔴 PRD-C-010 闸A·基因闸标记透传（item.gene 由 gene_gate 填）：
    # gene_gate = pass / warn / skipped（平行度入库可查抓手；v1 只警示不硬拦）。
    gene = (item or {}).get("gene") or {}
    if gene.get("gate"):
        aux["gene_gate"] = str(gene["gate"])
    bo["auxTags"] = aux


def build_mother_bo(facts: dict[str, Any]) -> dict[str, Any]:
    """从上传图抽出的「母题(原题)」→ CreateQuestionBo。

    🔴 设计 §7 原为「图母题不入库」；维护者 2026-06-10 拍板改为：图母题不在库时**先把原题入库**，
       变式再 motherQuestionId 指向它 → 血缘完整可追。母题图 URL 落 stemImg。
    🔴 PRD-C-009：V16 维度列已补到 dev 库 → 入库存 DNA/打标（dim1~5 + label_* + auxTags），
       由 _apply_labels 统一塞（role="mother"）。aux_tags 列已存在，恢复溯源标签。
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
    if facts.get("subject_id"):
        bo["subjectId"] = str(facts["subject_id"])
    if facts.get("image_url"):
        bo["stemImg"] = facts["image_url"]  # 母题图落题干图字段
    _apply_labels(bo, facts, item=None, role="mother")
    return bo


def build_create_bo(item: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """单道变式 item + 母题 facts → CreateQuestionBo camelCase body。

    🔴 只放契约允许的字段；createBy/createUser/status/id 绝不放（后端强制，传了也忽略）。
    题型映射成整数；难度夹 1~4；带 AI 血缘（母题 id / 变式关系 / 来源）。
    🔴 PRD-C-009：5 维度 DNA + 轻量打标 + auxTags 由 _apply_labels 统一塞（role="variant"）。
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

    # 知识点编码：classify 锚定到的真实节点 code（落 subjectId）
    subject_id = facts.get("subject_id")
    if subject_id:
        bo["subjectId"] = str(subject_id)

    # 母题血缘：图母题入库后回填的 mother_question_id（雪花大整数）
    mother_id = facts.get("mother_question_id")
    if mother_id:
        try:
            bo["motherQuestionId"] = int(mother_id)
        except (TypeError, ValueError):
            pass
    _apply_labels(bo, facts, item=item, role="variant")
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

        # 1) 逐题入库变式（此时 facts.mother_question_id 已回填）
        for item in items:
            bo = build_create_bo(item, facts)
            try:
                new_id = _extract_new_id(await client.create_question(bo))
                receipts.append({"ok": True, "id": new_id, "role": "variant"})
            except Exception as e:  # noqa: BLE001 — 单题失败如实记，不拖垮整组
                receipts.append({"ok": False, "error": str(e), "role": "variant"})
    finally:
        await client.aclose()
    return receipts
