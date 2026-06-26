# -*- coding: utf-8 -*-
"""C-102 样本打标结果 → 难度 4 档 × is_anchor 金标交叉表 + 题型 dim2 冲突审计。"""
import json, sys, io, pymysql
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = sys.argv[1] if len(sys.argv) > 1 else "tools/_c102_sample100_out.json"
data = json.load(open(OUT, encoding="utf-8"))
results = data["results"]
summ = data["summary"]

ENV = json.load(open(r"D:\workplace\book-ai\workplace\.mcp.json", encoding="utf-8"))["mcpServers"]["mysql"]["env"]
cx = pymysql.connect(host=ENV["MYSQL_HOST"], port=int(ENV["MYSQL_PORT"]), user=ENV["MYSQL_USER"],
                     password=ENV["MYSQL_PASS"], database="ai_lesson_prep", charset="utf8mb4",
                     cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()
qids = [r["qid"] for r in results]
cur.execute(f"SELECT id, is_anchor, dim2_qtype FROM biz_question WHERE id IN ({','.join(['%s']*len(qids))})", qids)
meta = {r["id"]: r for r in cur.fetchall()}
cx.close()

DIM2 = {1:"选择",2:"判断",3:"应用",4:"填空",5:"解答",6:"作图",7:"计算",8:"证明"}
LV = {1:"L1基础",2:"L2中等",3:"L3较难",4:"L4压轴"}

ok = [r for r in results if r.get("ok")]
print(f"=== 跑批汇总 ===  题数={summ['questions']} 解析成功={len(ok)} "
      f"wall={summ['wall_s']}s cost={summ['cost_estimate']}")

# 难度 × is_anchor
xtab = defaultdict(lambda: Counter())
lv_total = Counter()
warn_n = 0
conflict = []
for r in ok:
    lab = r.get("label", {})
    diff = lab.get("difficulty", {})
    lv = diff.get("level")
    if diff.get("warn"): warn_n += 1
    anchor = "压轴金标" if meta.get(r["qid"], {}).get("is_anchor") else "普通"
    xtab[lv][anchor] += 1
    lv_total[lv] += 1
    if lab.get("questionTypeConflict"):
        conflict.append((r["qid"], lab.get("questionType"), lab.get("questionTypeLLM")))

print("\n=== 难度 4 档 × is_anchor 金标交叉表 ===")
print(f"{'档':10} {'压轴金标':>8} {'普通':>6} {'合计':>6} {'占比':>6}")
for lv in [1,2,3,4]:
    a = xtab[lv]["压轴金标"]; n = xtab[lv]["普通"]; t = lv_total[lv]
    print(f"{LV[lv]:10} {a:>8} {n:>6} {t:>6} {t/max(len(ok),1)*100:>5.0f}%")
# 压轴命中率（金标=压轴的题落 L3+L4 的比例）
anchor_ok = [r for r in ok if meta.get(r['qid'],{}).get('is_anchor')]
anchor_hi = sum(1 for r in anchor_ok if (r['label'].get('difficulty',{}).get('level') or 0) >= 3)
norm_ok = [r for r in ok if not meta.get(r['qid'],{}).get('is_anchor')]
norm_lo = sum(1 for r in norm_ok if (r['label'].get('difficulty',{}).get('level') or 9) <= 2)
print(f"\n压轴金标→L3+L4 命中率: {anchor_hi}/{len(anchor_ok)} = {anchor_hi/max(len(anchor_ok),1)*100:.0f}%  (期望高)")
print(f"普通题  →L1+L2 命中率: {norm_lo}/{len(norm_ok)} = {norm_lo/max(len(norm_ok),1)*100:.0f}%  (期望高)")
print(f"warn 兜底数: {warn_n}")

# rule 分布
rules = Counter(r['label'].get('difficulty',{}).get('rule') for r in ok)
print("\n=== rule 命中分布 ===")
for ru, n in rules.most_common(): print(f"   {ru:20} {n}")

# 题型 dim2 权威 vs labeler 冲突
print(f"\n=== 题型冲突（dim2 权威覆盖 labeler）: {len(conflict)} 道 ===")
for qid, dim2lab, llmlab in conflict[:15]:
    print(f"   {qid}  dim2={dim2lab}  ← labeler判={llmlab}")

# 模型/变式/难点 抽检（看克制是否守住）
print("\n=== 抽检 5 道（看打标内容）===")
for r in ok[:5]:
    lab = r["label"]; dna = lab.get("dna", {}); f = lab.get("factors", {})
    vp = {k:v for k,v in (lab.get("variationProfile") or {}).items() if v.get("usable")}
    print(f"\n  qid={r['qid']} {lab.get('questionType')} {lab.get('difficulty',{}).get('levelDictLabel')} "
          f"({lab.get('difficulty',{}).get('rule')})")
    print(f"    models={[ (m['name'],m['tier'],m['freqHint']) for m in lab.get('models',[]) ]}")
    print(f"    K/R/D/G={f.get('K')}/{f.get('R')}/{f.get('D')}/{f.get('G')} highStrat={f.get('highStrategies')}")
    print(f"    难点={dna.get('hardPoints')} 突破={dna.get('breakthroughPoints')}")
    print(f"    可用变式算子={list(vp.keys())}")

# isNew 提议模型
prop = summ.get("proposed_new_models", [])
print(f"\n=== isNew 提议新模型: {len(prop)} 条（批后归并转正用，本步不入库）===")
for p in prop[:15]: print(f"   qid={p['qid']} {p['name']}")
