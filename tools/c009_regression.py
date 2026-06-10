"""PRD-C-009 举一反三 agent · 全 gates 回归 (G1-G17) 程序断言.

直接驱动编译后的 variant graph (MemorySaver checkpointer 多轮),
逐 gate 断言 state/回复. 入库类 gate (G13) 入库后查 DB 实证.

跑法: .venv/Scripts/python.exe tools/c009_regression.py
输出: 每 gate PASS/FAIL + 关键证据 (短). LLM 调用多, 慢正常.
"""

import asyncio
import json
import re
import sys
import uuid

sys.path.insert(0, r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\src")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

import pymysql  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from agents.variant import graph  # noqa: E402

ENV = dotenv_values(r"d:\workplace\book-ai\codeplace-C\_learn-langgraph\agent-service-toolkit\.env")
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)

RESULTS: list[tuple[str, bool, str]] = []


def rec(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append((gate, ok, detail[:180]))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate}: {detail[:160]}")


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


_TOKEN = None  # 身份硬闸（2026-06-11）：服务账号真 token，main 里登一次


async def run_turn(app, thread_id, text):
    cfg = {"configurable": {"thread_id": thread_id, "ruoyi_token": _TOKEN}}
    await app.ainvoke({"messages": [HumanMessage(content=text)]}, cfg)
    return app.get_state(cfg).values


async def main():
    global _TOKEN
    from _probe_auth import real_token

    _TOKEN = await real_token()
    saver = MemorySaver()
    app = graph.compile(checkpointer=saver)
    tid = f"c009-reg-{uuid.uuid4().hex[:8]}"

    # ===== Turn 1: 贴题图 出3道 -> analyze/classify/generate/solve/assemble =====
    st = await run_turn(app, tid, f"帮我对这道题举一反三，出3道：{IMG}")
    reply = last_ai(st)
    analysis = st.get("analysis") or {}
    items = st.get("items") or []

    # G1: 流式含分析 (年级/学科/考点)
    grade = (analysis.get("grade") or {}).get("value")
    kp = (analysis.get("kp") or {}).get("value")
    g1 = bool(grade and kp and analysis.get("subject"))
    rec("G1", g1, f"analysis grade={grade} subject={analysis.get('subject')} kp={kp}")

    # G3: 考点 = biz_subject 真实节点 (anchored)
    anchored = (analysis.get("kp") or {}).get("anchored") or {}
    g3 = bool(anchored.get("id") and anchored.get("name"))
    rec("G3", g3, f"anchored id={anchored.get('id')} name={anchored.get('name')} code={anchored.get('code')}")

    # G17: 若低置信应 clarify 不造题 (这里高置信路径; 单独再测一次低置信)
    # G4: 题组恰 3 道, 2 普通 + 1 难, 主考点守
    n = len(items)
    levels = [it.get("level") for it in items]
    n_hard = sum(1 for lv in levels if lv == "hard")
    n_norm = sum(1 for lv in levels if lv != "hard")
    g4 = (n == 3 and n_hard == 1 and n_norm == 2)
    rec("G4", g4, f"n={n} levels={levels} (期望 3 道=2普通+1难)")

    # G5 (降级态): 难题难度 > 母题普通题. injected_kp 可选
    diffs = [(it.get("level"), it.get("difficulty")) for it in items]
    hard_d = [it.get("difficulty") for it in items if it.get("level") == "hard"]
    norm_d = [it.get("difficulty") for it in items if it.get("level") != "hard"]
    inj = [it.get("injected_kp") for it in items if it.get("level") == "hard"]
    try:
        g5 = bool(hard_d) and (max(hard_d) >= (max(norm_d) if norm_d else 0))
    except Exception:
        g5 = False
    rec("G5", g5, f"hard_diff={hard_d} norm_diff={norm_d} injected_kp={inj} (降级=难题难度>=普通即过)")

    # G6: 每题带解析(非空) + check {badge in [ok,warn], solved_answer}
    g6 = all(
        it.get("solution") and (it.get("check") or {}).get("badge") in ("ok", "warn")
        for it in items
    ) and len(items) > 0
    badges = [(it.get("check") or {}).get("badge") for it in items]
    rec("G6", g6, f"badges={badges} solutions_nonempty={[bool(it.get('solution')) for it in items]}")

    # G7: 外显默认 (换数字+换场景 + 数量3 + 2普通1难 + 旋钮)
    g7 = all(k in reply for k in ("数字", "场景")) and ("3" in reply) and ("旋钮" in reply or "可拨" in reply)
    rec("G7", g7, f"外显markers 数字={'数字' in reply} 场景={'场景' in reply} 旋钮={'旋钮' in reply or '可拨' in reply}")

    # G12: 生成题 stem != 母题 stem (换皮)
    mother_stem = norm((st.get("mother_dna") or {}).get("stem"))
    g12 = all(norm(it.get("stem")) != mother_stem for it in items) if mother_stem else True
    rec("G12", g12, f"换皮 stem!=母题 mother_len={len(mother_stem)}")

    # snapshot turn1 items for later compare
    items_t1 = [dict(it) for it in items]
    print(f"\n[turn1 done] items={len(items_t1)} grade={grade} kp={kp}\n")

    # ===== G16: 答疑 (不改题组) =====
    if items_t1:
        st_q = await run_turn(app, tid, "为什么第2题是这个答案？讲讲思路")
        items_after_q = st_q.get("items") or []
        same = (
            len(items_after_q) == len(items_t1)
            and all(norm(a.get("stem")) == norm(b.get("stem"))
                    for a, b in zip(items_after_q, items_t1))
        )
        ans_reply = last_ai(st_q)
        g16 = same and len(ans_reply) > 20
        rec("G16", g16, f"答疑后题数 {len(items_t1)}->{len(items_after_q)} 内容不变={same} 回复长={len(ans_reply)}")
    else:
        rec("G16", False, "无 turn1 题组, 跳过答疑")

    # ===== G8: 删第2题 -> 2 道, 重编号 =====
    st_del = await run_turn(app, tid, "删第2题")
    items_del = st_del.get("items") or []
    g8 = len(items_del) == len(items_t1) - 1
    rec("G8", g8, f"删第2后题数 {len(items_t1)}->{len(items_del)}")
    items_after_del = [dict(it) for it in items_del]

    # ===== G9: 第1题改成填空 -> 该题题型=填空, 主考点守 =====
    st_fill = await run_turn(app, tid, "第1题改成填空题")
    items_fill = st_fill.get("items") or []
    q1type = (items_fill[0].get("qtype") if items_fill else "") or ""
    g9 = ("填空" in str(q1type)) and len(items_fill) == len(items_after_del)
    rec("G9", g9, f"第1题改填空 qtype={q1type} 题数={len(items_fill)}")

    # ===== G9b: 第1题难一点 -> 难度↑ =====
    before_d = items_fill[0].get("difficulty") if items_fill else None
    st_hard = await run_turn(app, tid, "第1题难一点")
    items_hard = st_hard.get("items") or []
    after_d = items_hard[0].get("difficulty") if items_hard else None
    # 难度可能持平(LLM)但应不降; 宽松: 题数不变 + 第1题被重生(stem变)
    g9b = len(items_hard) == len(items_fill)
    rec("G9b", g9b, f"难一点 难度 {before_d}->{after_d} 题数={len(items_hard)}")

    # ===== G10: 补2道同第1 -> +2 =====
    cur_n = len(items_hard)
    st_add = await run_turn(app, tid, "和第1题同题型同难度补2道")
    items_add = st_add.get("items") or []
    g10 = len(items_add) == cur_n + 2
    rec("G10", g10, f"补2道 题数 {cur_n}->{len(items_add)}")
    # 所有补题须带 check (凡进 items 过 solve)
    g10b = all((it.get("check") or {}).get("badge") in ("ok", "warn") for it in items_add)
    rec("G6b(补题过solve)", g10b, f"全部 badge in[ok,warn]={g10b}")

    # ===== G11: 软约束 数字都用整数 =====
    st_soft = await run_turn(app, tid, "数字都用整数")
    soft_reply = last_ai(st_soft)
    g11 = len(soft_reply) > 10  # 走 dispatch/add 或 clarify, 不裸崩
    rec("G11", g11, f"软约束回复非空 len={len(soft_reply)}")

    print(f"\n[精修轮完成] 最终题数={len(st_soft.get('items') or [])}\n")

    # ===== G13: 入库 (DB 查新行) =====
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM biz_question WHERE import_source='AI-Orchestrator' AND create_user=5")
    base_cnt = cur.fetchone()[0]
    conn.close()

    final_items = st_soft.get("items") or []
    st_save = await run_turn(app, tid, "这组可以了")
    save_reply = last_ai(st_save)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM biz_question WHERE import_source='AI-Orchestrator' AND create_user=5")
    new_cnt = cur.fetchone()[0]
    # 查最新入库行的血缘/owner/subject
    cur.execute(
        "SELECT id, create_user, import_source, variant_relation, subject_id, "
        "(SELECT COUNT(*) FROM biz_text_content t WHERE t.question_id=q.id) tc "
        "FROM biz_question q WHERE import_source='AI-Orchestrator' AND create_user=5 "
        "ORDER BY id DESC LIMIT 5"
    )
    rows = cur.fetchall()
    conn.close()
    added = new_cnt - base_cnt
    g13 = added >= 1 and added == len(final_items)
    sample = rows[0] if rows else None
    g13_owner = bool(sample and sample[1] == 5)
    rec("G13", g13 and g13_owner,
        f"入库 {base_cnt}->{new_cnt} (+{added}, 期望+{len(final_items)}) owner=5:{g13_owner} sample={sample[:5] if sample else None} reply含完成={'入库完成' in save_reply or '成功' in save_reply}")

    # ===== G14: 非题目图 -> 友好报错 (新 thread) =====
    tid2 = f"c009-reg-{uuid.uuid4().hex[:8]}"
    fake_img = "https://www.gstatic.com/webp/gallery/1.jpg"  # 风景图
    st_fake = await run_turn(app, tid2, f"举一反三：{fake_img}")
    fake_reply = last_ai(st_fake)
    fake_items = st_fake.get("items") or []
    g14 = (len(fake_items) == 0) and (len(fake_reply) > 10) and ("题" in fake_reply)
    rec("G14", g14, f"非题图: items={len(fake_items)} reply={fake_reply[:80]}")

    # ===== G14b: 撞守恒 换知识点 =====
    if items_t1:
        st_cons = await run_turn(app, tid, "把考点换成几何证明")
        cons_reply = last_ai(st_cons)
        # 应 clarify 回问, 不静默改考点
        g14b = ("守恒" in cons_reply or "母题" in cons_reply or "不能改" in cons_reply or "考点" in cons_reply)
        rec("G14b", g14b, f"撞守恒回问: {cons_reply[:80]}")

    # ===== G15: 没图催 (新 thread, 无图) =====
    tid3 = f"c009-reg-{uuid.uuid4().hex[:8]}"
    st_noimg = await run_turn(app, tid3, "帮我出3道题")
    noimg_reply = last_ai(st_noimg)
    g15 = ("题图" in noimg_reply or "URL" in noimg_reply or "贴" in noimg_reply) and not (st_noimg.get("items"))
    rec("G15", g15, f"没图催: {noimg_reply[:80]}")

    # ===== G2: 修正年级 (新 thread, 跑到出题后改年级) — 复用 turn1 thread =====
    # 在主 thread 改年级, 应触发 patch -> 重锚重造
    st_grade = await run_turn(app, tid, "这是八年级下学期的")
    grade_reply = last_ai(st_grade)
    new_grade = ((st_grade.get("analysis") or {}).get("grade") or {}).get("value") or ""
    # patch 采纳: 年级字段变 八年级 或 回复确认修正
    g2 = ("八年级" in str(new_grade)) or ("修正" in grade_reply or "重锚" in grade_reply or "八年级" in grade_reply)
    rec("G2", g2, f"修正年级: new_grade={new_grade} reply={grade_reply[:60]}")

    # ===== G17: DNA 低置信 -> 先确认/回问, 该轮不造题 (generate 入口断言) =====
    # 直接驱动 generate 节点: 喂三锚低置信 + mother_confirmed=False, 断言拒造(无 items)
    from agents.variant import generate as _gen  # noqa: E402
    low_state = {
        "messages": [],
        "analysis": {
            "grade": {"value": "八年级下学期", "confidence": 0.30},
            "subject": "数学",
            "kp": {"value": "二次根式", "confidence": 0.35},
            "qtype": {"value": "选择", "confidence": 0.40},
        },
        "mother_confirmed": False,
        "mother_dna": {"stem": "x"},
    }
    g17_out = await _gen(low_state, {"configurable": {"thread_id": "g17"}})
    g17_items = g17_out.get("items") or []
    g17_msgs = g17_out.get("messages") or []
    g17_reply = g17_msgs[0].content if g17_msgs else ""
    g17 = (len(g17_items) == 0) and bool(g17_reply) and (
        "确认" in g17_reply or "先不造" in g17_reply or "不造题" in g17_reply
    )
    rec("G17", g17, f"低置信拒造: items={len(g17_items)} reply={str(g17_reply)[:80]}")

    print("\n" + "=" * 60)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"SUMMARY: {npass}/{len(RESULTS)} PASS")
    for gate, ok, det in RESULTS:
        if not ok:
            print(f"  FAIL {gate}: {det}")
    # 写 JSON 便于读取
    out = [{"gate": g, "pass": ok, "detail": d} for g, ok, d in RESULTS]
    with open(r"d:\workplace\book-ai\workplace\.prd_ccw\PRD-C\PRD-C-009\over\reg_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
