"""PRD-C-010 举一反三可靠性加固 · 真机 e2e 探针。

直接驱动编译后的 variant graph（不经 :8093 服务，模式同 c009_regression.py），
真 LLM + 真 OSS 题图 + 真 RuoYi 入库 + 真 DB 回查，断言四件事：

  P1 闸B: 每道题带程序验算标记 check.verify ∈ {sympy_pass, unverified, fail_after_regen}
          （或证明题分流 check.review=proof_needs_human）
  P2 闸A: 每道题带基因闸标记 gene.gate ∈ {pass, warn, skipped}
  P3 分类器护栏: "删第9题"(只有3题) → 整体降级 clarify, 题组不被乱删
  P4 答疑物理护栏: 答疑不改题组
  P5 入库: biz_question.aux_tags 含 verify / gene_gate 键（V901 列, JSON）

跑法（toolkit 根）: $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c010_e2e_probe.py
前置: book-server :8090 在跑（入库）、MySQL :3307 在跑、lk888 可达。
"""

import asyncio
import json
import re
import sys
import uuid

sys.path.insert(0, r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\src")

import pymysql  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import graph  # noqa: E402

ENV = dotenv_values(r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\.env")
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)

RESULTS: list[tuple[str, bool, str]] = []


def rec(probe: str, ok: bool, detail: str) -> None:
    RESULTS.append((probe, ok, detail[:200]))
    print(f"[{'PASS' if ok else 'FAIL'}] {probe}: {detail[:180]}")


def db():
    return pymysql.connect(
        host=ENV["VARIANT_DB_HOST"], port=int(ENV["VARIANT_DB_PORT"]),
        user=ENV["VARIANT_DB_USER"], password=ENV["VARIANT_DB_PASSWORD"],
        database=ENV["VARIANT_DB_NAME"], charset="utf8mb4",
    )


def last_ai(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


def norm(s):
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


_TOKEN: str | None = None  # 身份硬闸（2026-06-11）：服务账号真 token，main 里登一次


async def run_turn(app, thread_id, text):
    cfg = {"configurable": {"thread_id": thread_id, "ruoyi_token": _TOKEN}}
    await app.ainvoke({"messages": [HumanMessage(content=text)]}, cfg)
    return app.get_state(cfg).values


async def main() -> int:
    global _TOKEN
    from _probe_auth import real_token

    _TOKEN = await real_token()
    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)
    tid = f"c010-probe-{uuid.uuid4().hex[:8]}"

    # ===== Turn 1: 真图出 3 道 =====
    st = await run_turn(app, tid, f"帮我对这道题举一反三，出3道：{IMG}")
    items = st.get("items") or []
    if not items:
        rec("P0", False, f"turn1 未出题组: reply={last_ai(st)[:100]}")
        return 1
    rec("P0", True, f"turn1 出题 {len(items)} 道")

    # P1 闸B: 每题有 verify 标记或 proof 分流标记
    VERIFY_OK = {"sympy_pass", "unverified", "fail_after_regen"}
    p1_details = []
    p1 = True
    for i, it in enumerate(items, 1):
        ck = it.get("check") or {}
        v, r = ck.get("verify"), ck.get("review")
        ok = (v in VERIFY_OK) or (r == "proof_needs_human")
        p1 = p1 and ok
        p1_details.append(f"#{i}:verify={v},review={r},badge={ck.get('badge')}")
    rec("P1闸B", p1, " ".join(p1_details))

    # P2 闸A: 每题有 gene.gate 标记
    GENE_OK = {"pass", "warn", "skipped"}
    p2_details = []
    p2 = True
    for i, it in enumerate(items, 1):
        g = (it.get("gene") or {}).get("gate")
        p2 = p2 and (g in GENE_OK)
        p2_details.append(f"#{i}:gene={g}")
    rec("P2闸A", p2, " ".join(p2_details))

    items_t1 = [dict(it) for it in items]

    # P3 分类器护栏: 删越界题号 → clarify 反问, 题组原封不动
    st_del = await run_turn(app, tid, "删第9题")
    items_del = st_del.get("items") or []
    reply_del = last_ai(st_del)
    same = (
        len(items_del) == len(items_t1)
        and all(norm(a.get("stem")) == norm(b.get("stem")) for a, b in zip(items_del, items_t1))
    )
    p3 = same and len(reply_del) > 5
    rec("P3护栏", p3, f"删第9题(只有{len(items_t1)}题): 题组不变={same} reply={reply_del[:60]}")

    # P4 答疑物理护栏: 不改题组
    st_qa = await run_turn(app, tid, "为什么第1题这么出？")
    items_qa = st_qa.get("items") or []
    same_qa = (
        len(items_qa) == len(items_t1)
        and all(norm(a.get("stem")) == norm(b.get("stem")) for a, b in zip(items_qa, items_t1))
    )
    rec("P4答疑", same_qa and len(last_ai(st_qa)) > 20, f"答疑后题组不变={same_qa}")

    # P5 入库: aux_tags 含 verify / gene_gate
    conn = db()
    cur = conn.cursor()
    # 🔴 变式 agent 的 import_source='举一反三'（非 c009 老值 'AI-Orchestrator'，2026-06-10 实测踩过）
    cur.execute("SELECT COALESCE(MAX(id),0) FROM biz_question WHERE import_source='举一反三'")
    max_id_before = cur.fetchone()[0]
    conn.close()

    final_items = st_qa.get("items") or []
    st_save = await run_turn(app, tid, "这组可以了，入库吧")
    save_reply = last_ai(st_save)

    conn = db()
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(
        "SELECT id, aux_tags, label_status, labeled_by, dim2_qtype, dim4_difficulty "
        "FROM biz_question WHERE import_source='举一反三' AND id > %s ORDER BY id",
        (max_id_before,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        rec("P5入库", False, f"入库后无新行 (reply={save_reply[:60]})")
    else:
        # 变式行应带 verify + gene_gate；母题行(无 check/gene)允许没有
        with_verify = [r for r in rows if r["aux_tags"] and "verify" in str(r["aux_tags"])]
        with_gene = [r for r in rows if r["aux_tags"] and "gene_gate" in str(r["aux_tags"])]
        n_variant_expected = len(final_items)
        p5 = len(with_verify) >= n_variant_expected and len(with_gene) >= n_variant_expected
        sample = rows[-1]
        rec(
            "P5入库", p5,
            f"新行{len(rows)}条 带verify={len(with_verify)} 带gene_gate={len(with_gene)} "
            f"(期望>= {n_variant_expected}) sample_aux={str(sample['aux_tags'])[:120]}",
        )
        print("  入库新行明细:")
        for r in rows:
            print(f"    id={r['id']} label_status={r['label_status']} aux_tags={str(r['aux_tags'])[:160]}")

    print("\n" + "=" * 60)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"SUMMARY: {npass}/{len(RESULTS)} PASS")
    for probe, ok, det in RESULTS:
        if not ok:
            print(f"  FAIL {probe}: {det}")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
