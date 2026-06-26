# -*- coding: utf-8 -*-
"""盘点 ai_lesson_prep 的 biz_* 表 + 行数 + 注释，给三要素核对 + 删表决策用。"""
import json, sys, io, pymysql
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ENV = json.load(open(r"D:\workplace\book-ai\workplace\.mcp.json", encoding="utf-8"))["mcpServers"]["mysql"]["env"]
cx = pymysql.connect(host=ENV["MYSQL_HOST"], port=int(ENV["MYSQL_PORT"]), user=ENV["MYSQL_USER"],
                     password=ENV["MYSQL_PASS"], database="ai_lesson_prep", charset="utf8mb4",
                     cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()
cur.execute("SELECT TABLE_NAME t, TABLE_COMMENT c FROM information_schema.tables "
            "WHERE TABLE_SCHEMA='ai_lesson_prep' AND TABLE_NAME LIKE 'biz_%' ORDER BY TABLE_NAME")
rows = cur.fetchall()
print(f"=== ai_lesson_prep biz_* 表 ({len(rows)} 张) ===")
for r in rows:
    t = r["t"]
    try:
        cur.execute("SELECT COUNT(*) n FROM `%s`" % t)
        n = cur.fetchone()["n"]
    except Exception:
        n = "ERR"
    print(f"  {t:36} {str(n):>7}   {(r['c'] or '')[:34]}")
cx.close()
