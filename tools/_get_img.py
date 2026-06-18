import sys
sys.path.insert(0, 'src')
from agents import conv_trace as ct
conn = ct._conn(); cur = conn.cursor()
cur.execute("SELECT request FROM conv_llm_trace WHERE id=1385")
row = cur.fetchone(); cur.close(); conn.close()
req = row[0] if row else ""
toks = req.replace('"', ' ').replace('\\', ' ').replace(',', ' ').split()
u = next((t for t in toks if t.startswith('http') and ('myqcloud' in t or '.png' in t or 'cos' in t)), None)
print("IMG_URL=" + (u or "NONE"))
