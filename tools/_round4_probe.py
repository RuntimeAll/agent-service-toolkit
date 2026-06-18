# -*- coding: utf-8 -*-
"""查上一轮压测窗口(id 1443..1542)的 mother_entry 行：找 ~45s 返回但没解析成卡的那条(round4)，看 root cause。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from agents import conv_trace as ct

conn = ct._conn(); cur = conn.cursor()
cur.execute("""SELECT id, duration_ms, error, fallback_count, fallback_detail,
                      CHAR_LENGTH(response) AS rlen, LEFT(response,400) AS head, RIGHT(response,200) AS tail
               FROM conv_llm_trace
               WHERE id BETWEEN 1444 AND 1542 AND label='mother_entry' ORDER BY id""")
for r in cur.fetchall():
    rid, dur, err, fb, fbd, rlen, head, tail = r
    print(f"=== id={rid} dur={round((dur or 0)/1000)}s err={err} fb={fb} fbd={fbd} resp_len={rlen}")
    print("  HEAD:", (head or "").replace("\n", " ")[:380])
    print("  TAIL:", (tail or "").replace("\n", " ")[-180:])
    print()
cur.close(); conn.close()
