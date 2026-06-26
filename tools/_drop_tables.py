# -*- coding: utf-8 -*-
"""安全删除 组A(旧试卷legacy) + 组B(无用) 表：备份DDL → 查FK → 验空 → DROP。"""
import json, sys, io, pymysql, datetime
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
                     autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = cx.cursor()

# 1. 查【外部】FK：有没有 DROP_SET 之外的表 FK 指向这些表（指向才阻止 DROP）
cur.execute("""SELECT TABLE_NAME child, REFERENCED_TABLE_NAME parent, CONSTRAINT_NAME
  FROM information_schema.KEY_COLUMN_USAGE
  WHERE TABLE_SCHEMA='ai_lesson_prep' AND REFERENCED_TABLE_NAME IN (%s)""" % ",".join(["%s"]*len(TARGETS)), TARGETS)
ext_fk = [r for r in cur.fetchall() if r["child"] not in DROP_SET]
print("=== 外部 FK 指向待删表（阻止删除的）===")
print("  ", ext_fk if ext_fk else "无 — 没有外部表 FK 依赖这些表，可安全删 ✓")

# 2. 验空 + 备份 DDL
ts = "20260626"
backup = io.StringIO()
backup.write(f"-- 删表前 DDL 备份 {ts}（ai_lesson_prep 组A旧试卷 + 组B无用）\n")
print("\n=== 行数核验 + DDL 备份 ===")
for t in TARGETS:
    cur.execute("SELECT COUNT(*) n FROM `%s`" % t)
    n = cur.fetchone()["n"]
    cur.execute("SHOW CREATE TABLE `%s`" % t)
    ddl = cur.fetchone()["Create Table"]
    backup.write(f"\n-- {t} (rows={n})\n{ddl};\n")
    grp = "A" if t in GROUP_A else "B"
    print(f"  [{grp}] {t:30} rows={n}")

open(r"D:\workplace\book-ai\codeplace-C\_dropped_tables_backup_20260626.sql", "w", encoding="utf-8").write(backup.getvalue())
print("\n  DDL 已备份 → codeplace-C/_dropped_tables_backup_20260626.sql")

# 3. DROP（无外部FK阻止才执行）
if ext_fk:
    print("\n🔴 有外部 FK 依赖，停止删除，先解依赖")
else:
    print("\n=== 执行 DROP ===")
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    for t in TARGETS:
        cur.execute("DROP TABLE IF EXISTS `%s`" % t)
        print(f"  dropped {t}")
    cur.execute("SET FOREIGN_KEY_CHECKS=1")
    # 复核
    cur.execute("""SELECT TABLE_NAME t FROM information_schema.tables
      WHERE TABLE_SCHEMA='ai_lesson_prep' AND TABLE_NAME IN (%s)""" % ",".join(["%s"]*len(TARGETS)), TARGETS)
    left = [r["t"] for r in cur.fetchall()]
    print(f"\n复核：删后还剩 {left if left else '0 张 ✓ 全删干净'}")
    cur.execute("SELECT COUNT(*) n FROM information_schema.tables WHERE TABLE_SCHEMA='ai_lesson_prep' AND TABLE_NAME LIKE 'biz_%'")
    print(f"现 biz_* 表总数: {cur.fetchone()['n']}（删前 68）")
cx.close()
