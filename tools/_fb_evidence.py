import sys
sys.path.insert(0, 'src')
from agents import conv_trace as ct
conn = ct._conn(); cur = conn.cursor()
# stress2 窗口(start_id=1506)里的 failover 行 = 真实挂起证据
cur.execute("""SELECT id, label, relay, fallback_count, fallback_detail, ROUND(duration_ms/1000) dur_s, error
               FROM conv_llm_trace
               WHERE id>1506 AND (fallback_count>0 OR fallback_detail IS NOT NULL OR error IS NOT NULL)
               ORDER BY id""")
rows = cur.fetchall()
print(f"stress2 窗口熔断/失败行数 = {len(rows)}")
for r in rows:
    print(f"  id={r[0]} {r[1]} 成交={r[2]} fb={r[3]} detail={r[4]} {r[5]}s err={r[6]}")
cur.close(); conn.close()
