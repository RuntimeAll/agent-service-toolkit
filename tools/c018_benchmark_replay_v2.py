"""PRD-C-014 B5 量化验收 · 基准回放 v2：c018 母题集 + W6 压轴扩样 → graph 直驱出题 →
逐题收集闸B/闸A/难度/表皮，落 c018_result_v2.json，逐项给 G1/G2/G3/G4/G8/G9。

相对 c018_benchmark_replay.py 的增量：
  1) 母题集 = 原 10 道文字母题（从 c018_benchmark_replay.MOTHERS 复用）+ 3 道 W6 压轴真题
     （带题干图 URL + DNA 守恒白名单 main_kp/secondary_kps，让 W2 守恒真实生效）。
  2) 每道变式额外收集 difficulty/level/gene.flags/stem，供 G2 守恒、G4 表皮、G8 难度量化。
  3) G1 锚定：每母题 main_kp + secondary_kps 名 → anchor_subject → 年级叶子池越界计数。
  4) G2 守恒：变式 injected_kp(副 kp) ⊄ 母题白名单 = 违例；跨学段（年级 code 不一致）单列。
  5) G4 表皮：对全部 (变式, 母题) 对算 _surface_check 归一化相似度分布（分位数 + 直方）。
  6) G9 终值：跑后查 conv_trace.conv_llm_trace 按 thread_id 统计每母题轮 LLM 调用数。
  7) 支持并发（--concurrency N），墙钟预算友好。

判决真值只读 state 里闸B/闸A 落的标记；难度只认 _grade_difficulty 的 rubric 输出；本脚本零裁决。
不入库（不调 persist）。DB 只 SELECT（conv_trace + anchor 叶子池）。

跑法（cwd = toolkit 根，RuoYi :8090 在跑）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c018_benchmark_replay_v2.py --json
  --limit N      只跑前 N 道母题
  --finale-only  只跑 3 道压轴（快速验 G8 rubric=4）
  --concurrency  并发母题数（默认 3）
"""

import argparse
import asyncio
import difflib
import json
import re
import sys
import time
from pathlib import Path

from _probe_auth import real_token

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import (  # noqa: E402
    _SURFACE_SIM_THRESHOLD,
    _grade_to_code,
    _surface_norm_stem,
    graph,
)
from agents.variant_support import RuoyiClient, anchor_subject, leaf_pool_for_grade  # noqa: E402

# c018 原 10 道文字母题（复用单一事实源，不复制粘贴）
from c018_benchmark_replay import MOTHERS as BASE_MOTHERS  # noqa: E402

PER_MOTHER_TIMEOUT_S = 1200.0
RESULT_JSON = Path(__file__).resolve().parent / "c018_result_v2.json"

# ---------------------------------------------------------------------------
# W6 压轴扩样：3 道 dev 库真题（difficult=4，初中数学，多知识点综合/分类讨论/探究归纳）。
# 选题口径见脚本回报。每道带：
#   - stem_img_url：题干图 URL（真题原图，证明是真压轴）。
#   - dna.main_kp + dna.secondary_kps：W2 守恒白名单（主 kp + 副 kp），让守恒真实生效。
#   - dna.exam_type / hard_points / skeleton：rubric 判 4 的依据（≥2 难点 / 多突破口综合）。
#   - answer/skeleton：供闸B（绝大多数压轴变式落 self_ok/proof，不强求 sympy_pass）。
# ---------------------------------------------------------------------------
FINALE_MOTHERS: list[dict] = [
    {
        "name": "压轴·二次函数与几何综合(将军饮马)",
        "qid": 12599,
        "grade": "九年级上学期",
        "kp": "二次函数与几何综合",
        "qtype": "解答",
        "difficulty": 4,
        "stem_img_url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png",
        "stem": "如图，二次函数 $y=-x^2+bx+c$ 的图象与 $x$ 轴交于 $A$、$B$ 两点，与 $y$ 轴交于点 $C$，"
        "点 $B(3,0)$，点 $C(0,3)$，直线 $l$ 经过 $B$、$C$。"
        "(1) 求该二次函数的表达式及顶点坐标；"
        "(2) 点 $P$ 为直线 $l$ 上一动点，过 $P$ 作 $y$ 轴平行线交抛物线于 $M$，过 $M$ 作 $x$ 轴平行线交抛物线于另一点 $N$，"
        "当 $PM=\\dfrac12 MN$ 时求点 $P$ 的横坐标；"
        "(3) 点 $C$ 关于 $x$ 轴的对称点为 $D$，$Q$ 为线段 $AP$ 上一点且 $AQ=3PQ$，当 $3AP+4DQ$ 最小时求 $DQ$。",
        "answer": "(1) $y=-x^2+2x+3$，顶点 $(1,4)$；(2) $P$ 横坐标为 $0$ 或 $4$（分类）；(3) $DQ$ 取最小时的值见解析。",
        "skeleton": "待定系数定抛物线 → 直线与抛物线联立设动点 → PM/MN 比例分类讨论列方程 → "
        "对称点+定比分点构造『将军饮马』把 3AP+4DQ 转化为折线最短（胡不归型系数转化）求最小值",
        "dna": {
            "main_kp": {"id": "3010003003", "name": "二次函数的图象与性质"},
            "secondary_kps": [
                {"id": "3010003003005", "name": "二次函数的解析式"},
                {"id": "3100021001", "name": "将军饮马模型"},
            ],
            "exam_type": "探究归纳",
            "tags": ["二次函数综合", "动点", "最值", "分类讨论", "将军饮马"],
            "scene": "平面直角坐标系",
            "hard_points": ["动点比例分类讨论列方程", "带系数折线最短的转化构造(胡不归/将军饮马)"],
            "skeleton": ["待定系数求抛物线", "联立设动点坐标", "比例条件分类讨论", "对称构造折线最短求最值"],
        },
    },
    {
        "name": "压轴·圆内接四边形综合(对角线垂直)",
        "qid": 17294,
        "grade": "九年级上学期",
        "kp": "圆内接四边形对角互补",
        "qtype": "解答",
        "difficulty": 4,
        "stem_img_url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-26/06a3e548-e3d6-43ce-900f-c0d0e9cdd601/list/13/question.png",
        "stem": "四边形 $ABCD$ 内接于 $\\odot O$，$AC\\perp BD$，$AC$、$BD$ 交于点 $E$。"
        "(1) 若 $AC=BD=4$，求 $\\angle BDC$ 与四边形 $ABCD$ 的面积；"
        "(2) 在 $AD$ 上取点 $M$，连 $BM$、$OM$，使 $BM\\perp OM$，求证 $BM^2=AM\\cdot DM$；"
        "(3) 已知 $BD=4$ 且 $CD=\\sqrt3\\,AB$，当 $AB=2\\sqrt3$ 时求 $AC$；并当 $AB$ 取最小时直接写 $\\tan\\angle ABP$。",
        "answer": "(1) $\\angle BDC=45^\\circ$，面积 $8$；(2) 证明见解析（射影/相似）；(3) $AC$ 由勾股+相似求得，$\\tan$ 值见解析。",
        "skeleton": "对角线垂直的圆内接四边形性质 → 等腰直角求角与面积 → 构造相似(母子相似/射影定理)证比例式 → "
        "结合 CD=√3·AB 与圆周角定理分类求线段长 → 极值位置求三角比",
        "dna": {
            "main_kp": {"id": "3091003010001", "name": "圆内接四边形对角互补"},
            "secondary_kps": [
                {"id": "3091003010002", "name": "圆内接四边形外角等于内对角"},
                {"id": "3010006001", "name": "6.1圆的相关性质"},
            ],
            "exam_type": "证明推理",
            "tags": ["圆综合", "相似三角形", "射影定理", "最值", "分类讨论"],
            "scene": "圆内接四边形",
            "hard_points": ["射影定理/相似证比例式", "结合圆周角与边比的极值分类讨论"],
            "skeleton": ["对角线垂直性质求角与面积", "构造相似证 BM²=AM·DM", "圆周角+边比求线段", "极值位置求三角比"],
        },
    },
    {
        "name": "压轴·韦达定理拓展到三次方程(探究)",
        "qid": 35427,
        "grade": "八年级下学期",
        "kp": "一元二次方程根与系数的关系",
        "qtype": "填空",
        "difficulty": 4,
        "stem_img_url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
        "stem": "由韦达定理：一元二次方程 $ax^2+bx+c=0$ 可化为 $a(x-x_1)(x-x_2)=0$ 展开比对系数得根与系数关系。"
        "类比地，若一元三次方程 $ax^3+bx^2+cx+d=0$ 有三根 $x_1,x_2,x_3$，则三根之和、三根之积与系数也有类似关系。"
        "已知方程 $2x^3+x^2-7x-6=0$ 的三个实数根为 $\\alpha,\\beta,\\gamma$，"
        "求 $\\dfrac1{\\alpha\\beta}+\\dfrac1{\\beta\\gamma}+\\dfrac1{\\alpha\\gamma}$ 的值。",
        "answer": "$-\\dfrac16$",
        "skeleton": "类比二次韦达定理展开三次 a(x-x₁)(x-x₂)(x-x₃) 比对系数得："
        "Σx=-b/a、Σxᵢxⱼ=c/a、x₁x₂x₃=-d/a；目标式 = (x₁+x₂+x₃)/(x₁x₂x₃) = (-b/a)/(-d/a) = b/d，代入 b=1,d=-6 得 -1/6",
        "dna": {
            "main_kp": {"id": "3082002008", "name": "一元二次方程根与系数的关系"},
            "secondary_kps": [
                {"id": "3082002008005", "name": "利用韦达定理来求代数式的值"},
                {"id": "3082002008006", "name": "已知等式关系，借助韦达定理来求值"},
            ],
            "exam_type": "探究归纳",
            "tags": ["韦达定理", "类比推广", "代数式求值", "高次方程"],
            "scene": "代数探究",
            "hard_points": ["二次韦达定理类比推广到三次(系数比对)", "目标式恒等变形成 Σx/Πx 结构"],
            "skeleton": ["展开 a(x-x₁)(x-x₂)(x-x₃) 比对系数", "得三次韦达三式", "目标式拆成对称式", "代入系数求值"],
        },
    },
]


def _inject_state(m: dict) -> dict:
    """母题 → 注入 state：三锚高置信 + mother_dna（含可选 dna 守恒白名单）+ 已确认 + 空 knobs。"""
    mdna: dict = {
        "stem": m["stem"],
        "answer": m["answer"],
        "solution_skeleton": m["skeleton"],
        "difficulty": m["difficulty"],
    }
    if m.get("dna"):
        mdna["dna"] = m["dna"]  # W2 守恒白名单（_mother_facts 读 mother_dna.dna）
    return {
        "analysis": {
            "grade": {"value": m["grade"], "confidence": 0.95},
            "kp": {"value": m["kp"], "confidence": 0.95},
            "qtype": {"value": m["qtype"], "confidence": 0.95},
        },
        "mother_dna": mdna,
        "mother_confirmed": True,
        "knobs": {},
        "items": [],
    }


def _item_row(it: dict) -> dict:
    """每道变式的量化捕获（比 v1 多 difficulty/level/gene_flags/stem，供 G2/G4/G8）。"""
    chk = it.get("check") or {}
    gene = it.get("gene") or {}
    return {
        "verify": chk.get("verify") or (chk.get("review") and "proof_review") or "missing",
        "tier": chk.get("tier") or "missing",
        "gene_gate": gene.get("gate") or "missing",
        "gene_flags": gene.get("flags") or [],
        "difficulty": it.get("difficulty"),
        "level": it.get("level") or "normal",
        "injected_kp": it.get("injected_kp"),
        "qtype": it.get("qtype"),
        "stem": it.get("stem") or "",
    }


async def run_one(app, idx: int, m: dict, token: str) -> dict:
    cfg = {"configurable": {"thread_id": f"c018v2-{idx}", "ruoyi_token": token}}
    await app.aupdate_state(cfg, _inject_state(m), as_node="analyze")
    t0 = time.monotonic()
    await app.ainvoke(
        {"messages": [HumanMessage(content="请基于已确认的母题，出一组举一反三变式题。")]},
        cfg,
    )
    st = app.get_state(cfg).values
    items = st.get("items") or []
    dropped = st.get("dropped_notes") or []
    return {
        "idx": idx,
        "name": m["name"],
        "thread_id": f"c018v2-{idx}",
        "is_finale": bool(m.get("difficulty") == 4 and m.get("qid")),
        "qid": m.get("qid"),
        "mother_grade": m["grade"],
        "mother_kp": m["kp"],
        "ok": True,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "items": [_item_row(it) for it in items],
        "dropped": len(dropped),
        "dropped_notes": dropped,
    }


# ---------------------------------------------------------------------------
# G1 锚定（越界口径 = 课程图谱越界，目标 0）：母题 main_kp + secondary_kps 名 → anchor_subject。
#   真·越界 = 锚不到任何叶子候选（= 凭空生造、不在浙教版图谱里）。这是 G1 的硬指标（=0）。
#   register_span（信息项，非越界）= kp 锚到的册 code ≠ 母题年级册 code。压轴综合/专题题的考点
#   合法地落在「中考一轮复习(3010)」「解题技巧专题(3100)」册而非单一学期册——这是浙教版图谱
#   的真实组织方式，不是越界。单独报，不计入 out_of_pool_total。
# ---------------------------------------------------------------------------
async def g1_anchoring(mothers: list[dict], token: str) -> dict:
    client = RuoyiClient(token=token)
    out = {"per_mother": [], "out_of_pool_total": 0, "register_span_total": 0}
    try:
        full_pool = await leaf_pool_for_grade(None, client)  # 全量叶子（图谱真值）
        full_ids = {pid for pid, _ in full_pool}
        for m in mothers:
            grade_code = _grade_to_code(m["grade"])
            kp_names = [m["kp"]]
            dna = m.get("dna") or {}
            mk = dna.get("main_kp") or {}
            if mk.get("name"):
                kp_names.append(mk["name"])
            for s in dna.get("secondary_kps") or []:
                if s.get("name"):
                    kp_names.append(s["name"])
            kp_names = list(dict.fromkeys(kp_names))
            checks = []
            for name in kp_names:
                cands = await asyncio.to_thread(anchor_subject, name)
                anchored = bool(cands)
                # 真·越界 = 锚不到任一叶子（不在图谱里）
                in_full = any(str(c.get("id")) in full_ids for c in cands)
                out_of_pool = not (anchored and in_full)
                # register_span（信息项）= 命中册 code 都不等于母题年级册
                top_codes = sorted({str(c.get("grade_code")) for c in cands[:3]})
                reg_span = bool(grade_code) and anchored and (grade_code not in top_codes)
                checks.append({
                    "kp": name, "anchored": anchored, "in_full_graph": in_full,
                    "top_register_codes": top_codes,
                    "out_of_pool": out_of_pool, "register_span": reg_span,
                })
            n_oop = sum(1 for c in checks if c["out_of_pool"])
            n_span = sum(1 for c in checks if c["register_span"])
            out["out_of_pool_total"] += n_oop
            out["register_span_total"] += n_span
            out["per_mother"].append({
                "name": m["name"], "grade": m["grade"], "grade_code": grade_code,
                "checks": checks, "out_of_pool": n_oop, "register_span": n_span,
            })
    finally:
        await client.aclose()
    return out


# ---------------------------------------------------------------------------
# G2 守恒：变式 injected_kp(副 kp 名) 必须 ⊆ 母题 {主 kp + 副 kp} 名集合（null/None = 不注入副 kp = 守恒）。
# 跨学段串题：变式 injected_kp 若锚到的 grade_code ≠ 母题 grade_code（且都已知）= 跨学段。
# 注：流水线不产出每道变式的『主解题 kp』，故 G2 以声明性 injected_kp + 母题白名单为口径
# （这也正是 W2 守恒在 generate 侧能落到 item 上的真值；主考点硬守恒由 prompt 钉死、组级共享）。
# ---------------------------------------------------------------------------
def g2_conservation(results: list[dict], anchor_cache: dict) -> dict:
    per = []
    viol_total = 0
    cross_grade_total = 0
    for r in results:
        if not r.get("ok"):
            continue
        dna_names = set()
        # 从对应母题白名单（results 里没存 dna，用 mother_kp + 我们另存的 whitelist）
        wl = r.get("_whitelist_names") or {r["mother_kp"]}
        m_grade_code = r.get("_grade_code")
        viols = []
        crosses = []
        for row in r["items"]:
            ik = row.get("injected_kp")
            if not ik or str(ik).strip().lower() in ("null", "none", ""):
                continue  # 没注入副 kp = 守恒（默认就是主考点）
            ik = str(ik).strip()
            if ik not in dna_names and ik not in wl:
                viols.append(ik)
            # 跨学段：injected_kp 锚到的年级 code 与母题不一致
            gc = anchor_cache.get(ik)
            if gc and m_grade_code and gc != m_grade_code:
                crosses.append({"kp": ik, "grade_code": gc, "mother": m_grade_code})
        viol_total += len(viols)
        cross_grade_total += len(crosses)
        per.append({"name": r["name"], "violations": viols, "cross_grade": crosses})
    return {"per_mother": per, "violation_total": viol_total, "cross_grade_total": cross_grade_total}


# ---------------------------------------------------------------------------
# G4 表皮：对每道变式 vs 其母题题干算归一化相似度（与 _surface_check 同口径），给分布。
# ---------------------------------------------------------------------------
def g4_surface(results: list[dict], mother_stems: dict) -> dict:
    ratios = []
    pairs = []
    for r in results:
        if not r.get("ok"):
            continue
        mstem = _surface_norm_stem(mother_stems.get(r["idx"], ""))
        if not mstem:
            continue
        for row in r["items"]:
            v = _surface_norm_stem(row.get("stem", ""))
            ratio = difflib.SequenceMatcher(None, v, mstem).ratio()
            ratios.append(ratio)
            pairs.append({"mother": r["name"], "ratio": round(ratio, 3)})
    ratios.sort()
    n = len(ratios)

    def q(p):
        if not n:
            return None
        return round(ratios[min(n - 1, int(p * n))], 3)

    # 直方（0.0-1.0 分 10 桶）
    hist = [0] * 10
    for x in ratios:
        hist[min(9, int(x * 10))] += 1
    return {
        "n_pairs": n,
        "min": round(min(ratios), 3) if n else None,
        "max": round(max(ratios), 3) if n else None,
        "p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99),
        "threshold": _SURFACE_SIM_THRESHOLD,
        "n_above_threshold": sum(1 for x in ratios if x > _SURFACE_SIM_THRESHOLD),
        "histogram_0to1_by0.1": hist,
        "margin_max_to_threshold": round(_SURFACE_SIM_THRESHOLD - max(ratios), 3) if n else None,
    }


# ---------------------------------------------------------------------------
# G9 终值：查 conv_trace.conv_llm_trace 按 thread_id 统计每母题轮 LLM 调用数。
# ---------------------------------------------------------------------------
def g9_llm_calls(thread_ids: list[str]) -> dict:
    try:
        import pymysql

        from core import settings
        pwd = settings.VARIANT_DB_PASSWORD
        conn = pymysql.connect(
            host=settings.VARIANT_DB_HOST, port=settings.VARIANT_DB_PORT,
            user=settings.VARIANT_DB_USER,
            password=pwd.get_secret_value() if pwd else "",
            database="conv_trace", charset="utf8mb4", autocommit=True,
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"conv_trace 连接失败: {e}"}
    try:
        cur = conn.cursor()
        per = {}
        for tid in thread_ids:
            cur.execute("SELECT COUNT(*) FROM conv_llm_trace WHERE thread_id=%s", (tid,))
            per[tid] = int(cur.fetchone()[0])
        vals = [v for v in per.values() if v > 0]
        return {
            "per_thread": per,
            "mean_calls_per_mother": round(sum(vals) / len(vals), 2) if vals else 0,
            "baseline_old_pipeline": "~13.6 judge + 7.8 回炉 (约 21+/母题)",
        }
    finally:
        conn.close()


async def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-C-014 B5 量化验收基准回放 v2")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 道母题（0=全量）")
    ap.add_argument("--finale-only", action="store_true", help="只跑 3 道压轴")
    ap.add_argument("--concurrency", type=int, default=3, help="并发母题数")
    ap.add_argument("--json", action="store_true", help=f"结果落 {RESULT_JSON.name}")
    ap.add_argument("--reuse-finale1", action="store_true",
                    help="复用已落 c018_result_v2.json 的压轴#1(将军饮马)结果，不重跑(省 ~400s + LLM)")
    args = ap.parse_args()

    if args.finale_only:
        mothers = list(FINALE_MOTHERS)
    else:
        mothers = list(BASE_MOTHERS) + list(FINALE_MOTHERS)
    if args.limit:
        mothers = mothers[: args.limit]

    # 复用压轴#1：从已落 JSON 取 idx==该母题在本次 mothers 中的位置的 items（按 qid 12599 匹配）
    reuse_idx = None
    reuse_result = None
    if args.reuse_finale1 and RESULT_JSON.exists():
        try:
            prev = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            for pr in prev.get("results", []):
                if pr.get("qid") == 12599 and pr.get("ok") and pr.get("items"):
                    reuse_result = pr
                    break
        except Exception as e:  # noqa: BLE001
            print(f"[warn] reuse-finale1 读旧 JSON 失败，将重跑: {e}", flush=True)
        if reuse_result is not None:
            for i, m in enumerate(mothers):
                if m.get("qid") == 12599:
                    reuse_idx = i + 1
                    break
            print(f"[reuse] 压轴#1(qid 12599)复用旧结果({len(reuse_result['items'])}道)，跳过重跑 idx={reuse_idx}",
                  flush=True)

    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())

    print(f"=== C018 v2 量化回放：{len(mothers)} 道母题（含 {sum(1 for m in mothers if m.get('qid'))} 压轴）"
          f" 并发={args.concurrency} ===", flush=True)

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _guarded(i, m):
        # 复用压轴#1：直接套用旧结果（不占并发槽、不调 LLM）
        if reuse_idx is not None and i == reuse_idx and reuse_result is not None:
            r = dict(reuse_result)
            r["idx"] = i
            r["thread_id"] = reuse_result.get("thread_id", f"c018v2-{i}")
            r["_reused"] = True
            dna = m.get("dna") or {}
            wl = {m["kp"]}
            if (dna.get("main_kp") or {}).get("name"):
                wl.add(dna["main_kp"]["name"])
            for s in dna.get("secondary_kps") or []:
                if s.get("name"):
                    wl.add(s["name"])
            r["_whitelist_names"] = sorted(wl)
            r["_grade_code"] = _grade_to_code(m["grade"])
            print(f"--- [{i}] {m['name']} 复用旧结果：{len(r.get('items', []))} 道（未重跑）", flush=True)
            return r
        async with sem:
            print(f"--- [{i}] {m['name']} 开始 ...", flush=True)
            try:
                r = await asyncio.wait_for(run_one(app, i, m, token), timeout=PER_MOTHER_TIMEOUT_S)
            except Exception as e:  # noqa: BLE001
                r = {"idx": i, "name": m["name"], "ok": False, "error": f"{type(e).__name__}: {e}",
                     "thread_id": f"c018v2-{i}", "is_finale": bool(m.get("qid")),
                     "mother_grade": m["grade"], "mother_kp": m["kp"], "items": [], "dropped": 0}
            # 附母题白名单 + grade_code（供 G2）
            dna = m.get("dna") or {}
            wl = {m["kp"]}
            if (dna.get("main_kp") or {}).get("name"):
                wl.add(dna["main_kp"]["name"])
            for s in dna.get("secondary_kps") or []:
                if s.get("name"):
                    wl.add(s["name"])
            r["_whitelist_names"] = sorted(wl)
            r["_grade_code"] = _grade_to_code(m["grade"])
            n_items = len(r.get("items", []))
            print(f"--- [{i}] {m['name']} 完成：出 {n_items} 道，剔 {r.get('dropped', 0)}，"
                  f"{r.get('elapsed_s', '?')}s {'(ERROR)' if not r.get('ok') else ''}", flush=True)
            return r

    results = await asyncio.gather(*(_guarded(i + 1, m) for i, m in enumerate(mothers)))
    results = sorted(results, key=lambda r: r["idx"])

    mother_stems = {i + 1: m["stem"] for i, m in enumerate(mothers)}

    # ── G3 sympy 一次过 ──
    all_rows = [row for r in results if r.get("ok") for row in r["items"]]
    total = len(all_rows)
    n_sympy = sum(1 for x in all_rows if x["verify"] == "sympy_pass")
    g3_rate = (n_sympy / total) if total else 0.0
    # 分式方程单列（H3 盯防）
    frac = [r for r in results if "分式" in r["name"]]
    frac_rows = [row for r in frac for row in r["items"]]
    frac_pass = sum(1 for x in frac_rows if x["verify"] == "sympy_pass")

    # ── G8 难度 ──
    diff_hist = {1: 0, 2: 0, 3: 0, 4: 0, "none": 0}
    for x in all_rows:
        d = x["difficulty"]
        diff_hist[d if d in (1, 2, 3, 4) else "none"] += 1
    finale_diffs = {}
    for r in results:
        if r.get("is_finale") and r.get("ok"):
            ds = [row["difficulty"] for row in r["items"]]
            finale_diffs[r["name"]] = {
                "difficulties": ds,
                "max": max([d for d in ds if d], default=None),
                "rubric_judges_4": any(d == 4 for d in ds),
            }

    # ── G1 锚定 ──
    g1 = await g1_anchoring(mothers, token)

    # ── G2 守恒（anchor_cache：injected_kp 名 → grade_code，用于跨学段判定）──
    anchor_cache = {}
    inj_kps = {str(row.get("injected_kp")).strip() for r in results if r.get("ok")
               for row in r["items"] if row.get("injected_kp")}
    for ik in inj_kps:
        if ik.lower() in ("null", "none", ""):
            continue
        cands = await asyncio.to_thread(anchor_subject, ik)
        anchor_cache[ik] = (cands[0].get("grade_code") if cands else None)
    g2 = g2_conservation(results, anchor_cache)

    # ── G4 表皮 ──
    g4 = g4_surface(results, mother_stems)

    # ── G9 终值 ──
    g9 = g9_llm_calls([r["thread_id"] for r in results])

    # ── 汇总打印 ──
    n_err = sum(1 for r in results if not r.get("ok"))
    print("\n=== 汇总 ===")
    print(f"母题数: {len(results)}（失败 {n_err}）  成题总数: {total}")
    print(f"[G3] sympy 一次过: {n_sympy}/{total} = {g3_rate:.1%}  目标≥80%  → "
          f"{'PASS' if g3_rate >= 0.80 else 'FAIL'}")
    print(f"     分式方程单列: {frac_pass}/{len(frac_rows)} (H3 盯防，救不回标已知局限不阻塞)")
    print(f"[G8] 难度分布 1-4: {diff_hist}")
    for nm, fd in finale_diffs.items():
        print(f"     压轴「{nm}」难度={fd['difficulties']} 判4={fd['rubric_judges_4']}")
    print(f"[G1] 锚定越界: {g1['out_of_pool_total']} 处  目标=0  → "
          f"{'PASS' if g1['out_of_pool_total'] == 0 else 'CHECK'}")
    print(f"[G2] 守恒违例: {g2['violation_total']}  跨学段串题: {g2['cross_grade_total']}  目标=0")
    print(f"[G4] 表皮相似度: max={g4['max']} p95={g4['p95']} 阈值={g4['threshold']} "
          f"超阈={g4['n_above_threshold']} 裕量(阈-max)={g4['margin_max_to_threshold']}")
    print(f"[G9] LLM 调用/母题均值: {g9.get('mean_calls_per_mother')} (旧管线 {g9.get('baseline_old_pipeline')})")

    if args.json or True:
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mothers_run": len(results), "mothers_failed": n_err, "total_items": total,
            "rubric_prompt_snapshot": _rubric_snapshot(),
            "G3_sympy": {"pass": n_sympy, "total": total, "rate": round(g3_rate, 4),
                         "target": 0.80, "fraction_eq": {"pass": frac_pass, "total": len(frac_rows)}},
            "G8_difficulty": {"histogram": {str(k): v for k, v in diff_hist.items()},
                              "finale": finale_diffs},
            "G1_anchoring": g1,
            "G2_conservation": g2,
            "G4_surface": g4,
            "G9_llm_calls": g9,
            "results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        }
        RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 已落: {RESULT_JSON}")
    return 0


def _rubric_snapshot() -> str:
    """当前难度 rubric prompt 快照（从 variant 模块直接读，避免漂移）。"""
    try:
        from agents.variant import _GRADE_DIFFICULTY_PROMPT
        return _GRADE_DIFFICULTY_PROMPT
    except Exception as e:  # noqa: BLE001
        return f"(读取失败: {e})"


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
