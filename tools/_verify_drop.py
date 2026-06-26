# -*- coding: utf-8 -*-
"""只读：验证 组A/组B 表可安全删（查外部FK + 验空 + 备份DDL），不执行 DROP。"""
import json, sys, io, pymysql
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
GROUP_A = ["biz_paper", "biz_paper_analysis", "biz_paper_basket", "biz_paper_category",
           "biz_paper_favorite", "biz_paper_question", "biz_paper_section"]
GROUP_B = ["biz_class", "biz_class_student", "biz_task", "biz_task_question_answer",
           "biz_task_submission", "biz_question_wrong", "biz_question_favorite",
           "biz_question_folder", "biz_question_note", "biz_material",
           "biz_material_section", "biz_question_basket"]
TARGETS = GROUP_A + GROUP_B
DROP_SET = set(TARGETS)
ENV = json.load(open(r"D:\workplace\book-ai\workplace\.mcp.json", encoding="utf-8"))["mcpServers"]["mysql"]["env"]
cx = pymysql.connect(host=ENV["MYSQL_HOST"], port=int(ENV["MYSQL_PORT"]), user=ENV["MYSQL_USER"],
                     password=ENV["MYSQL_PASS"], database="ai_lesson_prep", charset="utf8mb4",
                     cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()
cur.execute("""SELECT TABLE_NAME child, REFERENCED_TABLE_NAME parent FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA='ai_lesson_prep' AND REFERENCED_TABLE_NAME IN (%s)""" % ",".join(["%s"]*len(TARGETS)), TARGETS)
ext_fk = [r for r in cur.fetchall() if r["child"] not in DROP_SET]
print("外部 FK 依赖（阻止删除的）:", ext_fk if ext_fk else "无 ✓ 可安全删")
backup = io.StringIO()
print("\n行数核验:")
for t in TARGETS:
    cur.execute("SELECT COUNT(*) n FROM `%s`" % t); n = cur.fetchone()["n"]
    cur.execute("SHOW CREATE TABLE `%s`" % t); ddl = cur.fetchone()["Create Table"]
    backup.write(f"-- {t} (rows={n})\n{ddl};\n\n")
    print(f"  [{'A' if t in GROUP_A else 'B'}] {t:30} rows={n}")
open(r"D:\workplace\book-ai\codeplace-C\_dropped_tables_backup_20260626.sql", "w", encoding="utf-8").write(backup.getvalue())
print("\nDDL 已备份 → codeplace-C/_dropped_tables_backup_20260626.sql（可恢复）")
print(f"待删 {len(TARGETS)} 张（组A {len(GROUP_A)} + 组B {len(GROUP_B)}），删后 biz_* 从 68 → {68-len(TARGETS)}")
cx.close()
