"""PRD-C-009 延续：用户级对话持久化（优化基础数据源）。

每次 LLM 往返写**独立解耦库** `conv_trace.conv_llm_trace`（同一台 :3307 server 上的另一个
database，**不碰业务库 miskt_data2、不走 RuoYi HTTP**——观测/优化数据非业务数据，独立库直写
是正解，不违"数据归 RuoYi"铁律）。

- teacher_id：从 book-ui 透传的 `ruoyi_token`(JWT) 解出 `userId`（用户级归属）。
- thread_id：graph config.configurable.thread_id（会话级）。
- 表归 toolkit 自管：首次写时 `CREATE TABLE IF NOT EXISTS`，**不进 flyway**（flyway 只管 miskt_data2）。
- best-effort：写失败绝不拖垮主流程。env `CONV_TRACE_ENABLED=0` 可关。

连接复用 `VARIANT_DB_*`（同 server），database 名走 env `CONV_TRACE_DB`（默认 conv_trace）。
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Any

import pymysql

from core import settings

_ENABLED = os.getenv("CONV_TRACE_ENABLED", "1").lower() not in ("0", "false", "no")
_DB_NAME = os.getenv("CONV_TRACE_DB", "conv_trace")
_table_ready = False

_DDL = """
CREATE TABLE IF NOT EXISTS conv_llm_trace (
  id                BIGINT AUTO_INCREMENT PRIMARY KEY,
  ts                DATETIME(3) NOT NULL,
  teacher_id        BIGINT NOT NULL      COMMENT '登录老师 user_id(从 ruoyi_token JWT 解);0=历史无主存量。🔴 NOT NULL=表级绑死用户(2026-06-11 用户拍板),图入口 route_entry 有同源硬闸',
  thread_id         VARCHAR(64) NULL     COMMENT '会话 id',
  source            VARCHAR(32) NULL     COMMENT 'variant/chat/...哪个服务/agent',
  label             VARCHAR(32) NULL     COMMENT 'analyze/generate/solve/...哪个 prompt',
  model             VARCHAR(64) NULL     COMMENT '实际模型名(如 gemini-3-flash-preview)',
  relay             VARCHAR(64) NULL     COMMENT '实际成交中转站名(PRD-C-011)',
  request           MEDIUMTEXT NULL      COMMENT '填充后完整 prompt(JSON messages, 多模态含图URL)',
  response          MEDIUMTEXT NULL      COMMENT '模型返回 content',
  reasoning         MEDIUMTEXT NULL      COMMENT '思考型 reasoning_content(可空)',
  prompt_tokens     INT NULL,
  completion_tokens INT NULL,
  cached_tokens     INT NULL             COMMENT 'PRD-C-100 B2 缓存命中 token(aigeek 自动缓存 prompt_tokens_details.cached_tokens);NULL=未回报',
  cost_yuan         DECIMAL(12,6) NULL   COMMENT '实际消费¥=token×价表(PRD-C-011),无价表则NULL',
  fallback_count    INT NOT NULL DEFAULT 0 COMMENT '中转站转移次数,0=主站一次成功(PRD-C-011)',
  duration_ms       INT NULL,
  retried           TINYINT(1) NOT NULL DEFAULT 0,
  error             VARCHAR(512) NULL,
  KEY idx_teacher_thread_ts (teacher_id, thread_id, ts),
  KEY idx_thread_ts (thread_id, ts),
  KEY idx_source_label (source, label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LLM对话往返记录·优化基础数据源(PRD-C-009/011)'
"""

# PRD-C-011 加列：老表（PRD-C-009 建的）补 relay/cost_yuan/fallback_count。
# MySQL 不支持 ADD COLUMN IF NOT EXISTS，逐条 ALTER + 吞重复列错误（best-effort 幂等）。
_MIGRATE = [
    "ALTER TABLE conv_llm_trace ADD COLUMN relay VARCHAR(64) NULL AFTER model",
    "ALTER TABLE conv_llm_trace ADD COLUMN cost_yuan DECIMAL(12,6) NULL AFTER completion_tokens",
    "ALTER TABLE conv_llm_trace ADD COLUMN fallback_count INT NOT NULL DEFAULT 0 AFTER cost_yuan",
    # PRD-C-100 B2：缓存命中对账列（aigeek 自动缓存 cached_tokens）。
    "ALTER TABLE conv_llm_trace ADD COLUMN cached_tokens INT NULL AFTER completion_tokens",
    # 2026-06-17 v2 熔断回溯：每站失败原因串（"sui-xiang:ReadTimeout; aigeek:ok"），供查「中途切了/为什么」。
    "ALTER TABLE conv_llm_trace ADD COLUMN fallback_detail VARCHAR(255) NULL AFTER fallback_count",
    # 2026-06-11 用户拍板「对话绑死用户·表级限制」：历史无主行回填 0，列改 NOT NULL。
    # 此后 teacher_id 为 NULL 的 INSERT 会被数据库拒绝（write() 静默吞 = 无主调用不留痕，
    # 真实流量由图入口 route_entry 硬闸保证必有 token，二者同源双保险。MODIFY 幂等可重跑。）
    "UPDATE conv_llm_trace SET teacher_id = 0 WHERE teacher_id IS NULL",
    (
        "ALTER TABLE conv_llm_trace MODIFY teacher_id BIGINT NOT NULL "
        "COMMENT '登录老师 user_id(从 ruoyi_token JWT 解);0=历史无主存量;NOT NULL=表级绑死用户'"
    ),
]


def _conn() -> pymysql.connections.Connection:
    pwd = settings.VARIANT_DB_PASSWORD
    # 🔴 P5 并发炸弹兜底：trace 库(:3307)慢/丢包时，同步 pymysql 不设超时会阻塞整个
    #   asyncio 事件循环 ~默认TCP超时(可达分钟级)→ 拖垮所有并发 SSE 流。这里给死超时
    #   (connect 2s / read 3s / write 3s)，库慢时快速抛 → 上层 best-effort 吞错降级，
    #   绝不让一条慢连接卡住全局 loop（调用点已配 asyncio.to_thread，超时也只占线程池一格）。
    return pymysql.connect(
        host=settings.VARIANT_DB_HOST,
        port=settings.VARIANT_DB_PORT,
        user=settings.VARIANT_DB_USER,
        password=pwd.get_secret_value() if pwd else "",
        database=_DB_NAME,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=2,
        read_timeout=3,
        write_timeout=3,
    )


def teacher_id_from_token(token: str | None) -> int | None:
    """从 RuoYi JWT(ruoyi_token) 解 userId（不验签，只读 payload）。失败返回 None。"""
    if not token:
        return None
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)  # base64url 补 padding
        data = json.loads(base64.urlsafe_b64decode(payload))
        uid = data.get("userId")
        return int(uid) if uid is not None else None
    except Exception:
        return None


def _usage_tokens(raw: Any) -> tuple[int | None, int | None]:
    """从 response_raw 里宽容取 prompt/completion tokens（OpenAI 兼容 usage）。"""
    try:
        meta = (raw or {}).get("response_metadata") or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        return (int(pt) if pt is not None else None, int(ct) if ct is not None else None)
    except Exception:
        return (None, None)


def cached_tokens_of(raw: Any) -> int | None:
    """PRD-C-100 B2：从 response_raw 取缓存命中 token（aigeek 自动缓存）。
    优先 usage_metadata.input_token_details.cache_read（langchain 标准），退 response_metadata
    的 prompt_tokens_details.cached_tokens（OpenAI 兼容）/ cache_read_input_tokens（原生 Claude）。
    取不到 → None（未回报，非 0）。画图链/图片重生不挂缓存，由调用方不传。"""
    try:
        um = (raw or {}).get("usage_metadata") or {}
        itd = um.get("input_token_details") or {}
        cr = itd.get("cache_read")
        if cr is not None:
            return int(cr)
    except Exception:
        pass
    try:
        meta = (raw or {}).get("response_metadata") or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        ptd = usage.get("prompt_tokens_details") or {}
        ct = ptd.get("cached_tokens")
        if ct is not None:
            return int(ct)
        cr = usage.get("cache_read_input_tokens")
        if cr is not None:
            return int(cr)
    except Exception:
        pass
    return None


def _reasoning(raw: Any) -> str | None:
    """思考型 reasoning_content（如有）。"""
    try:
        ak = (raw or {}).get("additional_kwargs") or {}
        r = ak.get("reasoning_content") or ak.get("reasoning")
        return str(r) if r else None
    except Exception:
        return None


def write(
    *,
    teacher_id: int | None,
    thread_id: str | None,
    source: str,
    label: str,
    model: str | None,
    request: Any,
    response: str,
    response_raw: Any,
    duration_ms: int,
    retried: bool = False,
    error: str | None = None,
    relay: str | None = None,
    fallback_count: int = 0,
    fallback_detail: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_yuan: float | None = None,
    cached_tokens: int | None = None,
) -> None:
    """落一条 LLM 往返到 conv_trace.conv_llm_trace。best-effort，永不抛。

    PRD-C-011：relay/fallback_count/cost_yuan 由调用方（relay_pool 路径）传入；
    prompt_tokens/completion_tokens 优先用传入值，缺省再从 response_raw 兜底解析。
    """
    if not _ENABLED:
        return
    global _table_ready
    try:
        if prompt_tokens is None and completion_tokens is None:
            prompt_tokens, completion_tokens = _usage_tokens(response_raw)
        if cached_tokens is None:
            cached_tokens = cached_tokens_of(response_raw)
        req_text = (
            request if isinstance(request, str) else json.dumps(request, ensure_ascii=False, default=str)
        )
        conn = _conn()
        try:
            cur = conn.cursor()
            if not _table_ready:
                cur.execute(_DDL)
                for stmt in _MIGRATE:
                    try:
                        cur.execute(stmt)
                    except Exception:
                        pass  # 列已存在 → 忽略（幂等）
                _table_ready = True
            cur.execute(
                """INSERT INTO conv_llm_trace
                   (ts, teacher_id, thread_id, source, label, model, relay, request, response,
                    reasoning, prompt_tokens, completion_tokens, cached_tokens, cost_yuan,
                    fallback_count, fallback_detail, duration_ms, retried, error)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    teacher_id,
                    thread_id,
                    source,
                    label,
                    model,
                    relay,
                    req_text,
                    response,
                    _reasoning(response_raw),
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    cost_yuan,
                    int(fallback_count or 0),
                    (fallback_detail or None) and str(fallback_detail)[:255],
                    duration_ms,
                    1 if retried else 0,
                    (error or None) and str(error)[:512],
                ),
            )
        finally:
            conn.close()
    except Exception:
        pass  # 持久化绝不拖垮主流程
