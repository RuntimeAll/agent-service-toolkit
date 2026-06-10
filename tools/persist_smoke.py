"""PRD-C-009 举一反三 · 入库(persist_to_bank) smoke。

直接驱动 persist 支撑层（persist_items → RuoyiClient → POST /teacher/question/create），
然后 pymysql 回查 biz_question + biz_text_content，确认「确认后 DB 查到新行」（设计 §7）。

用法（toolkit 根，venv 内）:
    .venv/Scripts/python.exe tools/persist_smoke.py

前置：
  - C 线 book-server :8090 在跑（/teacher/question/create 落库 + LoginHelper 定 owner）。
    冒烟: curl.exe -s --noproxy "*" -o NUL -w "%{http_code}" http://localhost:8090/actuator/health  (401=活着)
  - MySQL miskt_data2:3307 在跑（回查新行）。
  - .env 配好 RUOYI_USERNAME/PASSWORD（teacher 账号，token 定 owner）+ VARIANT_DB_*。

🔴 只验入库链路：造一道带唯一指纹的题 → 入库 → 回查 stem 外置文本(content_type='S')
   + biz_question.import_source/variant_relation/mother 列。不跑 LLM、不连 lk888。
"""

import asyncio
import sys
import uuid

sys.path.insert(0, "src")

import pymysql  # noqa: E402

from agents.variant_support import _db_kwargs, persist_items  # noqa: E402

# 唯一指纹：回查时凭它定位刚入库的行，避免误判老数据
MARK = f"[PERSIST-SMOKE-{uuid.uuid4().hex[:8]}]"


def _query_new_row() -> dict | None:
    """回查：凭外置题干(content_type='S') 命中指纹 → join 回 biz_question 取 AI 来源列。"""
    conn = pymysql.connect(**_db_kwargs())
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(
            """
            SELECT q.id, q.question_type, q.difficult, q.import_source,
                   q.variant_relation, q.status, q.create_user, q.create_by,
                   tc.content AS stem_content, tc.content_type
            FROM biz_text_content tc
            JOIN biz_question q ON q.id = tc.question_id
            WHERE tc.content_type = 'S' AND tc.content LIKE %s
            ORDER BY q.id DESC LIMIT 1
            """,
            (f"%{MARK}%",),
        )
        return cur.fetchone()
    finally:
        conn.close()


def _query_text(question_id: int, content_type: str) -> str | None:
    conn = pymysql.connect(**_db_kwargs())
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(
            "SELECT content FROM biz_text_content WHERE question_id=%s AND content_type=%s",
            (question_id, content_type),
        )
        row = cur.fetchone()
        return row["content"] if row else None
    finally:
        conn.close()


async def main() -> int:
    item = {
        "stem": f"{MARK} 解一元二次方程 x^2 - 5x + 6 = 0，求 x 的值。",
        "answer": "x = 2 或 x = 3",
        "solution": "因式分解 (x-2)(x-3)=0，故 x=2 或 x=3。",
        "qtype": "解答",
        "difficulty": 2,
        "level": "normal",
        "injected_kp": None,
        "variant_relation": "AI-数值变式",
    }
    facts = {
        "kp_name": "一元二次方程求根",
        "grade": "九年级上学期",
        "qtype": "解答",
        "subject_id": None,  # 图母题 MVP 无锚定编码 → 不带 subjectId（可选列）
        "mother_question_id": None,
    }

    print("=== 入库 (persist_items → POST /teacher/question/create) ===")
    try:
        receipts = await persist_items([item], facts)
    except Exception as e:  # noqa: BLE001
        print(f"!!! FAIL: 入库整体异常（book-server :8090 是否在跑？）: {e}")
        return 2
    print("回执:", receipts)

    ok = [r for r in receipts if r.get("ok")]
    if not ok:
        print("!!! FAIL: 无成功入库回执")
        return 1

    print("\n=== 回查 DB（确认后查到新行）===")
    row = _query_new_row()
    if not row:
        print("!!! FAIL: biz_question/biz_text_content 未查到带指纹的新行")
        return 1
    qid = row["id"]
    print(f"命中新行 id={qid}")
    print(f"  question_type={row['question_type']} (期望 5=解答)")
    print(f"  difficult={row['difficult']} (期望 2)")
    print(f"  import_source={row['import_source']!r} (期望 'AI-Orchestrator')")
    print(f"  variant_relation={row['variant_relation']!r} (期望 'AI-数值变式')")
    print(f"  status={row['status']!r} (期望 '1' 已发布)")
    print(f"  create_user={row['create_user']} create_by={row['create_by']} (owner=登录老师)")

    ans = _query_text(qid, "A")
    exp = _query_text(qid, "E")
    print(f"  外置答案(A)={ans!r}")
    print(f"  外置解析(E)={(exp or '')[:40]!r}...")

    checks = {
        "题型=5": row["question_type"] == 5,
        "难度=2": row["difficult"] == 2,
        "来源=AI-Orchestrator": row["import_source"] == "AI-Orchestrator",
        "变式关系=AI-数值变式": row["variant_relation"] == "AI-数值变式",
        "status=已发布": str(row["status"]) == "1",
        "外置答案非空": bool(ans),
        "外置解析非空": bool(exp),
    }
    print("\n=== 校验 ===")
    all_ok = True
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
        all_ok = all_ok and v

    if all_ok:
        print(f"\nOK: 入库链路通（新行 id={qid}，题干/答案/解析外置 + AI 来源列全对）")
        return 0
    print("\n!!! FAIL: 入库行存在但部分字段不符，请人工核对上文")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
