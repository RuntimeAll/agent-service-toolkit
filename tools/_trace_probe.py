# -*- coding: utf-8 -*-
"""查压测窗口(id>1443) conv_trace 真实落行：挂起是否留痕 / 最慢完成调用 / 时间空洞(挂起的影子) / failover / cached_tokens。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agents import conv_trace as ct

conn = ct._conn(); cur = conn.cursor()
# 1. 窗口总览
cur.execute("""SELECT id, ts, label, relay, fallback_count, duration_ms, retried, error,
                      prompt_tokens, completion_tokens, cached_tokens, cost_yuan
               FROM conv_llm_trace WHERE id>1443 ORDER BY id""")
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
print(f"[窗口 id>1443] 落行数={len(rows)}")
err = [r for r in rows if r[7]]
fb  = [r for r in rows if r[4] and r[4] > 0]
print(f"  error行={len(err)}  failover行(fb>0)={len(fb)}")
# cached_tokens 是否有值
cached_nonnull = [r for r in rows if r[10] is not None]
print(f"  cached_tokens 非空行={len(cached_nonnull)} / {len(rows)}")
# 最慢完成调用 top5
slow = sorted(rows, key=lambda r: (r[5] or 0), reverse=True)[:5]
print("  最慢完成调用 top5 (id/label/relay/dur_s/fb):")
for r in slow:
    print(f"    id={r[0]} {r[2]} {r[3]} dur={round((r[5] or 0)/1000)}s fb={r[4]}")
# 2. 时间空洞 = 挂起的影子(相邻行 ts 差 > 150s)
print("  时间空洞(相邻行 ts 间隔>150s = 疑似挂起没留痕):")
gaps = 0
for i in range(1, len(rows)):
    dt = (rows[i][1] - rows[i-1][1]).total_seconds()
    if dt > 150:
        gaps += 1
        print(f"    id{rows[i-1][0]}->id{rows[i][0]} 间隔 {round(dt)}s (前一行label={rows[i-1][2]})")
if gaps == 0:
    print("    (无)")
# 3. label 分布(看母题/generate/solve各几次)
from collections import Counter
lc = Counter(r[2] for r in rows)
print("  label分布:", dict(lc))
cur.close(); conn.close()
