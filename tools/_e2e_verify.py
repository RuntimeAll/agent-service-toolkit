import sys
sys.path.insert(0, 'src')
from agents import conv_trace as ct
conn = ct._conn(); cur = conn.cursor()
# 证明 fallback_detail 列存在(迁移已应用)+ 看最近 variant 行(浏览器 e2e 这轮)
cur.execute("""SELECT id, label, relay, fallback_count, fallback_detail,
                      ROUND(duration_ms/1000) AS dur_s, error
               FROM conv_llm_trace
               WHERE source='variant' ORDER BY id DESC LIMIT 14""")
rows = cur.fetchall()
print("fallback_detail 列可查 = 迁移已应用 ✓")
print("最近 14 条 variant LLM 调用:")
for r in rows:
    rid, label, relay, fb, fbd, dur, err = r
    print(f"  id={rid} {label:14s} relay={relay} fb={fb} fbd={fbd or '-'} {dur}s err={err or '-'}")
# 有无 failover/error
cur.execute("SELECT COUNT(*),SUM(fallback_count>0),SUM(error IS NOT NULL) FROM conv_llm_trace WHERE source='variant' AND id> (SELECT MAX(id)-40 FROM conv_llm_trace)")
tot, fb, errc = cur.fetchone()
print(f"\n近40窗口: 共{tot} failover行={fb or 0} error行={errc or 0}")
cur.close(); conn.close()
