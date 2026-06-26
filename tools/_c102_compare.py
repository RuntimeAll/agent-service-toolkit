# -*- coding: utf-8 -*-
"""合并 31 道重跑结果回原 100，出修复前后难度交叉表对比 + 抬档清单。"""
import json, sys, io, pymysql
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

orig = json.load(open("tools/_c102_sample100_out.json", encoding="utf-8"))
rerun = json.load(open("tools/_c102_rerun_out.json", encoding="utf-8"))
rr = {r["qid"]: r for r in rerun["results"]}

ENV = json.load(open(r"D:\workplace\book-ai\workplace\.mcp.json", encoding="utf-8"))["mcpServers"]["mysql"]["env"]
cx = pymysql.connect(host=ENV["MYSQL_HOST"], port=int(ENV["MYSQL_PORT"]), user=ENV["MYSQL_USER"],
                     password=ENV["MYSQL_PASS"], database="ai_lesson_prep", charset="utf8mb4",
                     cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()
qids = [r["qid"] for r in orig["results"]]
cur.execute(f"SELECT id, is_anchor FROM biz_question WHERE id IN ({','.join(['%s']*len(qids))})", qids)
anc = {r["id"]: r["is_anchor"] for r in cur.fetchall()}
cx.close()

LV = {1: "L1基础", 2: "L2中等", 3: "L3较难", 4: "L4压轴", 0: "?", None: "?"}

def lvl(rec):
    return (rec.get("label", {}) or {}).get("difficulty", {}).get("level") if rec and rec.get("ok") else None

def xtab(getrec):
    t = defaultdict(lambda: Counter()); tot = Counter(); ok = 0
    for o in orig["results"]:
        rec = getrec(o); lv = lvl(rec)
        if lv is None: continue
        ok += 1
        t[lv]["压轴" if anc.get(o["qid"]) else "普通"] += 1; tot[lv] += 1
    return t, tot, ok

def merged(o):
    return rr[o["qid"]] if o["qid"] in rr else o

before = xtab(lambda o: o)
after = xtab(lambda o: merged(o))

def render(name, x, use_merged):
    t, tot, ok = x
    print(f"\n--- {name} (ok={ok}) ---")
    print(f"{'档':9}{'压轴':>6}{'普通':>6}{'合计':>6}")
    for lv in [1,2,3,4]:
        print(f"{LV[lv]:9}{t[lv]['压轴']:>6}{t[lv]['普通']:>6}{tot[lv]:>6}")
    aok = [o for o in orig['results'] if anc.get(o['qid'])]
    ahi = 0; acnt = 0
    for o in aok:
        rec = merged(o) if use_merged else o
        lv = lvl(rec)
        if lv is None: continue
        acnt += 1
        if lv >= 3: ahi += 1
    print(f"压轴→L3+L4: {ahi}/{acnt} = {ahi/max(acnt,1)*100:.0f}%")

render("修前 100", before, False)
render("修后 100(含31重跑)", after, True)

# 抬档清单
print("\n=== 31 道重跑的抬档变化 ===")
lifts = []
for o in orig["results"]:
    if o["qid"] not in rr: continue
    b = lvl(o); a = lvl(rr[o["qid"]])
    arec = rr[o["qid"]]
    tag = "↑" if (a or 0) > (b or 0) else ("↓" if (a or 0) < (b or 0) else "=")
    lifts.append((tag, o["qid"], b, a, arec))
for tag in ["↑","=","↓"]:
    grp = [l for l in lifts if l[0]==tag]
    print(f"\n  {tag} {len(grp)} 道:")
    for _, qid, b, a, arec in grp[:30]:
        lab = arec.get("label",{}); f=lab.get("factors",{})
        ms=[(m['name'],m['tier']) for m in lab.get('models',[])]
        print(f"    {qid} {LV.get(b)}→{LV.get(a)} {lab.get('difficulty',{}).get('rule'):16} models={ms} strat={f.get('highStrategies')}")

cost = rerun["summary"]["cost_estimate"]
print(f"\n重跑成本: {cost}")
