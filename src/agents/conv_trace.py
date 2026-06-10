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
  teacher_id        BIGINT NULL          COMMENT '登录老师 user_id(从 ruoyi_token JWT 解)',
  thread_id         VARCHAR(64) NULL     COMMENT '会话 id',
  source            VARCHAR(32) NULL     COMMENT 'variant/chat/...哪个服务/agent',
  label             VARCHAR(32) NULL     COMMENT 'analyze/generate/solve/...哪个 prompt',
  model             VARCHAR(64) NULL,
  request           MEDIUMTEXT NULL      COMMENT '填充后完整 prompt(JSON messages, 多模态含图URL)',
  response          MEDIUMTEXT NULL      COMMENT '模型返回 content',
  reasoning         MEDIUMTEXT NULL      COMMENT '思考型 reasoning_content(可空)',
  prompt_tokens     INT NULL,
  completion_tokens INT NULL,
  duration_ms       INT NULL,
  retried           TINYINT(1) NOT NULL DEFAULT 0,
  error             VARCHAR(512) NULL,
  KEY idx_teacher_thread_ts (teacher_id, thread_id, ts),
  KEY idx_thread_ts (thread_id, ts),
  KEY idx_source_label (source, label)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LLM对话往返记录·优化基础数据源(PRD-C-009)'
"""


def _conn() -> pymysql.connections.Connection:
    pwd = settings.VARIANT_DB_PASSWORD
    return pymysql.connect(
        host=settings.VARIANT_DB_HOST,
        port=settings.VARIANT_DB_PORT,
        user=settings.VARIANT_DB_USER,
        password=pwd.get_secret_value() if pwd else "",
        database=_DB_NAME,
        charset="utf8mb4",
        autocommit=True,
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
) -> None:
    """落一条 LLM 往返到 conv_trace.conv_llm_trace。best-effort，永不抛。"""
    if not _ENABLED:
        return
    global _table_ready
    try:
        pt, ct = _usage_tokens(response_raw)
        req_text = (
            request if isinstance(request, str) else json.dumps(request, ensure_ascii=False, default=str)
        )
        conn = _conn()
        try:
            cur = conn.cursor()
            if not _table_ready:
                cur.execute(_DDL)
                _table_ready = True
            cur.execute(
                """INSERT INTO conv_llm_trace
                   (ts, teacher_id, thread_id, source, label, model, request, response,
                    reasoning, prompt_tokens, completion_tokens, duration_ms, retried, error)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    teacher_id,
                    thread_id,
                    source,
                    label,
                    model,
                    req_text,
                    response,
                    _reasoning(response_raw),
                    pt,
                    ct,
                    duration_ms,
                    1 if retried else 0,
                    (error or None) and str(error)[:512],
                ),
            )
        finally:
            conn.close()
    except Exception:
        pass  # 持久化绝不拖垮主流程
