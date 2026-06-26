# -*- coding: utf-8 -*-
"""PRD-C-102 批1·离线验证：对 ai_lesson_prep 全 868 题真跑确定性难度评级。

只读库（root/123456@127.0.0.1:3307/ai_lesson_prep，来自 .mcp.json mysql.env），**不写库不重启服务**。
覆盖 AC2/AC3/AC4：判档分布 + column_type 金标交叉表 + 20 题账单 + 确定性自检 + 无 LLM 自检。

跑法（cwd=agent-service-toolkit）：
  .venv\\Scripts\\python.exe tools\\c102_b1_difficulty_offline.py
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pymysql

# 让本脚本能 import src/core/difficulty（与 toolkit 同进程结构）。
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from core import difficulty  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DB = dict(host="127.0.0.1", port=3307, user="root", password="123456",
          database="ai_lesson_prep", charset="utf8mb4")


def _analysis_text(block_json: str | None) -> str:
    """从 analyze_block_json 抽纯文本。"""
    if not block_json:
        return ""
    try:
        blk = json.loads(block_json)
    except Exception:
        return ""
    out = []
    for row in blk.get("rows", []):
        for cell in row.get("cells", []):
            if cell.get("md"):
                out.append(cell["md"])
            for c in cell.get("content", []) or []:
                if c.get("md"):
                    out.append(c["md"])
    return " ".join(out)


def _stem_text(block_json: str | None, fallback: str | None) -> str:
    if block_json:
        try:
            blk = json.loads(block_json)
            out = []
            for row in blk.get("rows", []):
                for cell in row.get("cells", []):
                    if cell.get("type") == "option":
                        continue
                    if cell.get("md"):
                        out.append(cell["md"])
            if out:
                return " ".join(out)
        except Exception:
            pass
    return fallback or ""


def load_questions(cur) -> list[dict]:
    """组装每题的确定性 DNA 原料（只读 join）。返回 868 题列表。"""
    # 题→模型链接 + 每模型 link_count（考频带代理）。
    cur.execute("""
        SELECT qm.question_id, qm.model_id, sm.name, sm.model_kind,
               (SELECT COUNT(*) FROM biz_question_model x WHERE x.model_id=qm.model_id) AS link_count
        FROM biz_question_model qm
        LEFT JOIN biz_solution_model sm ON sm.id = qm.model_id
    """)
    models_by_q: dict[int, list[dict]] = defaultdict(list)
    for r in cur.fetchall():
        models_by_q[r["question_id"]].append({
            "modelId": r["model_id"], "name": r["name"],
            "model_kind": r["model_kind"], "link_count": r["link_count"],
        })

    # 知识点锚定数（K 用，可空）。
    cur.execute("SELECT question_id, COUNT(DISTINCT knowledge_id) c FROM biz_question_knowledge GROUP BY question_id")
    kp_by_q: dict[int, int] = {r["question_id"]: r["c"] for r in cur.fetchall()}

    # 解析/题面 block。
    cur.execute("SELECT question_id, block_json, analyze_block_json FROM biz_question_block")
    blocks: dict[int, dict] = {r["question_id"]: r for r in cur.fetchall()}

    # ai 维（skeleton + verify_kind/dna_type）。
    cur.execute("SELECT question_id, solution_skeleton, verify_kind, dna_type FROM biz_question_ai")
    ai_by_q: dict[int, dict] = {r["question_id"]: r for r in cur.fetchall()}

    # 868 题 = 有 biz_book_question 行的；带上 column_type 金标。
    cur.execute("""
        SELECT bq.question_id, bq.column_type, q.stem_text
        FROM biz_book_question bq
        LEFT JOIN biz_question q ON q.id = bq.question_id
    """)
    rows = cur.fetchall()

    out = []
    for r in rows:
        qid = r["question_id"]
        blk = blocks.get(qid, {})
        ai = ai_by_q.get(qid, {})
        skeleton = None
        sk = ai.get("solution_skeleton")
        if sk:
            try:
                skeleton = json.loads(sk)
            except Exception:
                skeleton = None
        dna = {
            "model_kind_hits": models_by_q.get(qid, []),
            "skeleton": skeleton,
            "analysis_text": _analysis_text(blk.get("analyze_block_json")),
            "stem_text": _stem_text(blk.get("block_json"), r.get("stem_text")),
            "kp_count": kp_by_q.get(qid),
            "verify_kind": ai.get("verify_kind"),
            "dna_type": ai.get("dna_type"),
        }
        out.append({"qid": qid, "column_type": r["column_type"], "dna": dna})
    return out


def main() -> None:
    conn = pymysql.connect(**DB)
    cur = conn.cursor(pymysql.cursors.DictCursor)
    qs = load_questions(cur)
    conn.close()
    print(f"加载 {len(qs)} 题\n")

    # ---- 判档全跑 ----
    bills = []
    for item in qs:
        bill = difficulty.grade(item["dna"])
        bills.append({"qid": item["qid"], "column_type": item["column_type"], **bill})

    # ---- AC4·1 判档分布表 ----
    dist = Counter(b["level"] for b in bills)
    print("=" * 56)
    print("判档分布表（AC4·1 / G4 每档 >0）")
    print("=" * 56)
    total = len(bills)
    for lv in (1, 2, 3, 4):
        c = dist.get(lv, 0)
        print(f"  L{lv} {difficulty._LEVEL_NAMES[lv]:<4} : {c:>4}  ({c/total*100:5.1f}%)")
    proxy_n = sum(1 for b in bills if b["proxy"])
    warn_n = sum(1 for b in bills if b["warn"])
    print(f"  proxy(无链模型用关键词代理): {proxy_n}  warn(哨兵兜底): {warn_n}")
    assert all(dist.get(lv, 0) > 0 for lv in (1, 2, 3, 4)), "G4 反性自检失败：有档为 0！"
    print("  ✅ G4 反性自检：每档 >0，非全挤一档")

    # ---- AC4·2 与 column_type 金标交叉表 ----
    print("\n" + "=" * 56)
    print("难度档 × column_type 金标交叉表（AC4·2 / G4）")
    print("=" * 56)
    cols = sorted({b["column_type"] for b in bills}, key=lambda c: -sum(1 for b in bills if b["column_type"] == c))
    header = "column_type".ljust(22) + "".join(f"L{lv}".rjust(7) for lv in (1, 2, 3, 4)) + "   tot"
    print(header)
    for col in cols:
        sub = [b for b in bills if b["column_type"] == col]
        line = (col or "—").ljust(22)
        for lv in (1, 2, 3, 4):
            line += str(sum(1 for b in sub if b["level"] == lv)).rjust(7)
        line += str(len(sub)).rjust(6)
        print(line)

    # 金标趋势断言（G4）：刷基础 L1+L2 占比  vs 刷难关；刷难关/素养 L4 占比 vs 刷基础。
    def share(col, levels):
        sub = [b for b in bills if b["column_type"] == col]
        if not sub:
            return None
        return sum(1 for b in sub if b["level"] in levels) / len(sub)

    print("\n  金标趋势核对（G4）：")
    base_easy = share("刷基础", {1, 2})
    hard_easy = share("刷难关", {1, 2})
    base_l4 = share("刷基础", {4})
    hard_l4 = share("刷难关", {4})
    suzhi_l4 = share("刷素养", {4})
    print(f"    刷基础 L1+L2 占比={fmt(base_easy)} vs 刷难关 L1+L2 占比={fmt(hard_easy)}  期望 基础>难关 → {gt(base_easy, hard_easy)}")
    print(f"    刷难关 L4 占比={fmt(hard_l4)} vs 刷基础 L4 占比={fmt(base_l4)}  期望 难关>基础 → {gt(hard_l4, base_l4)}")
    print(f"    刷素养 L4 占比={fmt(suzhi_l4)} vs 刷基础 L4 占比={fmt(base_l4)}  期望 素养>基础 → {gt(suzhi_l4, base_l4)}")

    # ---- AC3 20 题账单 ----
    print("\n" + "=" * 56)
    print("20 题理由账单抽样（AC3 / G3，含命中模型 + 因子 + 规则号）")
    print("=" * 56)
    # 抽样：覆盖各档 + proxy + 已链模型 + warn，各取若干。
    sample = []
    linked = [b for b in bills if b["modelHits"]]
    warned = [b for b in bills if b["warn"]]
    sample.extend(linked[:6])
    sample.extend(warned[:3])
    seen = {id(x) for x in sample}
    for lv in (1, 2, 3, 4):
        for b in bills:
            if b["level"] == lv and id(b) not in seen:
                sample.append(b); seen.add(id(b)); break
    for b in bills:
        if len(sample) >= 20:
            break
        if id(b) not in seen:
            sample.append(b); seen.add(id(b))
    for b in sample[:20]:
        mh = ",".join(f"{m['modelId']}/{m['name']}(t{m['tier']},f{m['freqBand']})" for m in b["modelHits"]) or ("[proxy]" if b["proxy"] else "[无模型]")
        print(f"qid={b['qid']} → L{b['level']}{b['levelName']} rule={b['rule']} "
              f"模型=[{mh}] K={b['K']} R={b['R']} D={b['D']} T={b['T']} G={b['G']} L={b['L']}"
              + (f" ⚠{b['notes']}" if b["warn"] else (f"  {b['notes']}" if b["notes"] else "")))

    # ---- AC2 确定性自检：同题判 3 次全等 ----
    print("\n" + "=" * 56)
    print("确定性自检（AC2 / G2）")
    print("=" * 56)
    det_ok = True
    for item in qs[:50]:
        r1 = difficulty.grade(item["dna"])
        r2 = difficulty.grade(item["dna"])
        r3 = difficulty.grade(item["dna"])
        if not (r1 == r2 == r3):
            det_ok = False
            print(f"  ❌ qid={item['qid']} 三次不一致")
    print(f"  同题判 3 次全等（抽 50 题）: {'✅ 通过' if det_ok else '❌ 失败'}")

    # 无 LLM 自检：扫**代码行**（剥注释/docstring）里的 import 与调用模式，禁词出现在
    # 警示性 docstring（"禁 import openai"）里不算违规——只认真正的 import 语句 / LLM 调用。
    import re as _re
    src_lines = (SRC / "core" / "difficulty.py").read_text(encoding="utf-8").splitlines()
    code = []
    in_doc = False
    for ln in src_lines:
        s = ln
        if '"""' in s:
            # 处理单行/起止 docstring。
            cnt = s.count('"""')
            if not in_doc and cnt >= 2:
                s = _re.sub(r'""".*?"""', "", s)
            elif not in_doc:
                in_doc = True
                s = s.split('"""', 1)[0]
            else:
                in_doc = False
                s = s.split('"""', 1)[1] if '"""' in s else ""
        elif in_doc:
            continue
        s = s.split("#", 1)[0]  # 剥行内注释
        if s.strip():
            code.append(s)
    code_text = "\n".join(code)
    banned_pat = [
        r"\bimport\s+openai", r"\bimport\s+anthropic", r"\bfrom\s+openai",
        r"\bimport\s+langchain", r"\bfrom\s+langchain", r"\bfrom\s+core\s+import.*\bget_model",
        r"\bget_model\s*\(", r"\bainvoke\s*\(", r"\bAsyncOpenAI\b", r"\bhttpx\.", r"requests\.",
    ]
    found = [p for p in banned_pat if _re.search(p, code_text)]
    print(f"  无 LLM/网络调用路径（扫代码行 import/调用模式，剥 docstring/注释）: "
          f"{'✅ 干净' if not found else '❌ 命中 '+str(found)}")
    print("\n完成。")


def fmt(x):
    return "n/a" if x is None else f"{x*100:.1f}%"


def gt(a, b):
    if a is None or b is None:
        return "n/a"
    return "✅" if a > b else "❌ 不符（见校准线索）"


if __name__ == "__main__":
    main()
