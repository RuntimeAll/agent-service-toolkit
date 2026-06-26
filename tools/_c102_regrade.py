# -*- coding: utf-8 -*-
"""免费重算：吃两批已存 factors，用修后 grade_observed 重判档，出 L4(d) 修复前后分布对比。"""
import json, sys, io, pymysql, importlib
sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from core import difficulty as d
importlib.reload(d)

ENV = json.load(open(r"D:\workplace\book-ai\workplace\.mcp.json", encoding="utf-8"))["mcpServers"]["mysql"]["env"]
cx = pymysql.connect(host=ENV["MYSQL_HOST"], port=int(ENV["MYSQL_PORT"]), user=ENV["MYSQL_USER"],
                     password=ENV["MYSQL_PASS"], database="ai_lesson_prep", charset="utf8mb4",
                     cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()

files = ["tools/_c102_sample100_out.json", "tools/_c102_sample2_out.json"]
recs = []
for fp in files:
    for r in json.load(open(fp, encoding="utf-8"))["results"]:
        if r.get("ok"): recs.append(r)
qids = list({r["qid"] for r in recs})
cur.execute(f"SELECT id,is_anchor FROM biz_question WHERE id IN ({','.join(['%s']*len(qids))})", qids)
anc = {r["id"]: r["is_anchor"] for r in cur.fetchall()}; cx.close()

from collections import Counter
LV = {1:"L1基础",2:"L2中等",3:"L3较难",4:"L4压轴"}

def regrade(rec):
    lab = rec["label"]; f = lab.get("factors", {})
    b = d.grade_observed(model_hits=lab.get("models") or [], K=f.get("K",0), R=f.get("R",0),
                         D=f.get("D",0), G=f.get("G",0), high_strategies=f.get("highStrategies") or [])
    return b["level"], b["rule"]

# 去重（两批有重叠qid，按最后一次）
by_qid = {}
for r in recs: by_qid[r["qid"]] = r
recs = list(by_qid.values())

before = Counter(); after = Counter(); rules = Counter()
ahb=ahn=0; aN=0; ahb2=ahn2=0
moved = []
for r in recs:
    old = r["label"].get("difficulty",{}).get("level")
    new, rule = regrade(r)
    before[old]+=1; after[new]+=1; rules[rule]+=1
    isanc = anc.get(r["qid"])
    if isanc:
        aN+=1
        if (old or 0)>=3: ahb+=1
        if new>=3: ahb2+=1
    if old != new: moved.append((r["qid"], old, new, rule, isanc))

n=len(recs)
print(f"=== 合并 {n} 道（两批去重）L4(d) 修复前后分布 ===")
print(f"{'档':9}{'修前':>7}{'修后':>7}")
for lv in [1,2,3,4]:
    print(f"{LV[lv]:9}{before[lv]:>7}{after[lv]:>7}  ({before[lv]/n*100:.0f}%→{after[lv]/n*100:.0f}%)")
print(f"\n压轴金标→L3+L4: 修前 {ahb}/{aN}={ahb/max(aN,1)*100:.0f}%  修后 {ahb2}/{aN}={ahb2/max(aN,1)*100:.0f}%")
print(f"\n=== 修后 rule 分布 ===")
for ru,c in rules.most_common(): print(f"   {ru:20}{c}")
print(f"\n=== 变档 {len(moved)} 道（多为 L4→L3 降）===")
dn = Counter((m[1],m[2]) for m in moved)
for (o,nw),c in dn.most_common(): print(f"   {LV.get(o)}→{LV.get(nw)}: {c}")
