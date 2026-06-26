# -*- coding: utf-8 -*-
"""复查 ai_lesson_prep 真实数据：控制面板/KG树/一道真题端到端/DNA/算子。"""
import json, sys, io, pymysql
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
mcp = json.load(open(r"D:\workplace\book-ai\workplace\.mcp.json", encoding="utf-8"))["mcpServers"]["mysql"]["env"]
cx = pymysql.connect(host=mcp["MYSQL_HOST"], port=int(mcp["MYSQL_PORT"]), user=mcp["MYSQL_USER"],
                     password=mcp["MYSQL_PASS"], database="ai_lesson_prep", charset="utf8mb4",
                     cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()

def short(s, n=70):
    s = str(s or "").replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")

print("=" * 60)
print("① 控制面板 biz_solution_model（27 个模型 · tier/freq）")
print("=" * 60)
cur.execute("SELECT id,name,model_kind,difficulty_tier,freq_band FROM biz_solution_model ORDER BY id")
for r in cur.fetchall():
    t = "高阶" if r["difficulty_tier"] == 2 else "基础"
    f = "高频通法" if r["freq_band"] == 2 else "一次性"
    print(f"  {r['id']:6} {r['name'][:24]:24} [{r['model_kind']:7}] {t}/{f}")

print("\n" + "=" * 60)
print("② 知识图谱 biz_subject（树结构 · 按 level 统计 + 样本分支）")
print("=" * 60)
cur.execute("SELECT level, COUNT(*) n FROM biz_subject GROUP BY level ORDER BY level")
print("  各层节点数:", {r["level"]: r["n"] for r in cur.fetchall()})
cur.execute("SELECT id,name,parent_id,level FROM biz_subject ORDER BY level, id LIMIT 14")
print("  样本节点(前14):")
for r in cur.fetchall():
    print(f"    L{r['level']} {r['id']:11} {short(r['name'],30):30} ←parent {r['parent_id']}")

print("\n" + "=" * 60)
print("③ 一道真题端到端（看实际存了什么）")
print("=" * 60)
cur.execute("SELECT id,stem_text,dim1_kp_id,dim2_qtype,question_type,difficult,source_raw,mother_question_id,is_anchor FROM biz_question WHERE status=1 AND stem_text<>'' ORDER BY id LIMIT 1")
q = cur.fetchone()
qid = q["id"]
print(f"  qid={qid}")
print(f"  题干: {short(q['stem_text'],90)}")
print(f"  dim1_kp_id(锚知识点)={q['dim1_kp_id']}  dim2_qtype(题型)={q['dim2_qtype']}  难度difficult={q['difficult']}")
print(f"  来源={q['source_raw']}  母题id={q['mother_question_id']}  压轴={q['is_anchor']}")
cur.execute("SELECT content_type,LENGTH(content) L FROM biz_text_content WHERE question_id=%s", (qid,))
print(f"  文本块(biz_text_content): {[(r['content_type'],str(r['L'])+'字') for r in cur.fetchall()]}")
cur.execute("SELECT k.knowledge_id,s.name FROM biz_question_knowledge k LEFT JOIN biz_subject s ON s.id=k.knowledge_id WHERE k.question_id=%s", (qid,))
print(f"  锚知识点(biz_question_knowledge): {[(r['knowledge_id'],r['name']) for r in cur.fetchall()]}")
cur.execute("SELECT p.name FROM biz_question_pattern_rel r LEFT JOIN biz_question_pattern p ON p.id=r.pattern_id WHERE r.question_id=%s", (qid,))
print(f"  题型(biz_question_pattern_rel): {[r['name'] for r in cur.fetchall()]}")
cur.execute("SELECT COUNT(*) n FROM biz_question_pitfall WHERE question_id=%s", (qid,))
print(f"  易错关联数(biz_question_pitfall): {cur.fetchone()['n']}")
cur.execute("SELECT COUNT(*) n FROM biz_question_model WHERE question_id=%s", (qid,))
nm = cur.fetchone()['n']
print(f"  命中模型(biz_question_model): {nm if nm else '空(打标未写库)'}")

print("\n" + "=" * 60)
print("④ DNA 派生维 biz_question_ai（现有几条 · 看打标写了啥）")
print("=" * 60)
cur.execute("SELECT COUNT(*) n FROM biz_question_ai")
print(f"  共 {cur.fetchone()['n']} 条")
cur.execute("SELECT * FROM biz_question_ai LIMIT 1")
row = cur.fetchone()
if row:
    for k, v in row.items():
        if v not in (None, "", "[]", "{}"):
            print(f"    {k} = {short(v,80)}")

print("\n" + "=" * 60)
print("⑤ 9 算子 biz_variation_method")
print("=" * 60)
cur.execute("SELECT * FROM biz_variation_method ORDER BY id LIMIT 9")
rows = cur.fetchall()
cols = [c for c in rows[0].keys() if c in ("id", "name", "code", "method_name", "similarity", "sort")] if rows else []
for r in rows:
    nm = r.get("name") or r.get("method_name") or list(r.values())[1]
    print(f"  {short(nm,40)}")
cx.close()
