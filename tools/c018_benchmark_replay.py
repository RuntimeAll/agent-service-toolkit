"""PRD-C-012 G8 · 基准回放：10 道固化文字母题 → graph 直驱出题 → 闸B 结果分布统计。

固化 10 道文字母题（一元一次/二次方程、分式方程含舍根、二次根式化简、数值计算、
选择题单选数值型、不等式、几何证明 1 道、文字应用题 2 道，各配 answer），逐题注入
「三锚高置信 + mother_confirmed」state 直接触发 generate（绕过 analyze 读图），跑完
从最终 state.items 收集 (check.verify, check.tier, gene.gate) + dropped_notes，汇总打印
sympy 可验率（= verify==sympy_pass 占比）对照基线 42%（AC5 目标 ≥80%）。

不调 persist（不入库）；判决真值只读 state 里闸B/闸A 落的标记，本脚本零裁决逻辑。

跑法（cwd = toolkit 根）:
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c018_benchmark_replay.py --limit 2
  全量: .venv/Scripts/python.exe tools/c018_benchmark_replay.py --json
前置: RuoYi :8090 在跑（_probe_auth.real_token 服务账号登录；curl 探活 401=活）；
      LLM 出口可达（graph 直驱，:8093 服务不需要在跑）。
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from _probe_auth import real_token

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from agents.variant import graph  # noqa: E402

# 单道母题全链路（generate + 闸A judge×N + 闸B solve/验算/可能回炉×N）墙钟上限。
# 超时按 G5 降级语义记 error 行继续下一道，绝不让一道病题挂死整场回放。
PER_MOTHER_TIMEOUT_S = 1200.0

RESULT_JSON = Path(__file__).resolve().parent / "c018_result.json"

# ---------------------------------------------------------------------------
# 10 道固化母题（覆盖面 = PRD-C-012 G8 要求；answer 全配齐供闸B 验算）
# analysis 三锚均给高置信（≥0.9，过 CONF_GATE=0.75）+ mother_confirmed=True
# → route_entry 直接走 generate（不经 analyze/classify，不需要图）。
# ---------------------------------------------------------------------------
MOTHERS: list[dict] = [
    {
        "name": "一元一次方程",
        "grade": "七年级上学期",
        "kp": "一元一次方程的解法",
        "qtype": "解答",
        "difficulty": 2,
        "stem": "解方程：$3x + 5 = 2x - 1$",
        "answer": "$x = -6$",
        "skeleton": "移项合并同类项：$3x - 2x = -1 - 5$，得 $x = -6$",
    },
    {
        "name": "一元二次方程",
        "grade": "八年级下学期",
        "kp": "一元二次方程求根（因式分解法）",
        "qtype": "解答",
        "difficulty": 3,
        "stem": "解方程：$x^2 - 5x + 6 = 0$",
        "answer": "$x_1 = 2, x_2 = 3$",
        "skeleton": "因式分解 $(x-2)(x-3)=0$，得 $x_1=2, x_2=3$",
    },
    {
        "name": "分式方程（含舍根）",
        "grade": "七年级下学期",
        "kp": "分式方程的解法（验根舍增根）",
        "qtype": "解答",
        "difficulty": 3,
        "stem": "解方程：$\\dfrac{x^2}{x-2} = \\dfrac{4}{x-2} + 2$",
        "answer": "$x = 0$（$x = 2$ 是增根，舍去）",
        "skeleton": "两边同乘 $(x-2)$ 化整式方程 $x^2 = 4 + 2(x-2)$，即 $x^2-2x=0$，"
        "解得 $x=0$ 或 $x=2$；检验 $x=2$ 使分母为零是增根舍去，故 $x=0$",
    },
    {
        "name": "二次根式化简",
        "grade": "八年级下学期",
        "kp": "二次根式的化简与加减",
        "qtype": "解答",
        "difficulty": 2,
        "stem": "计算：$\\sqrt{18} + \\sqrt{8} - \\sqrt{2}$",
        "answer": "$4\\sqrt{2}$",
        "skeleton": "化最简二次根式 $3\\sqrt{2} + 2\\sqrt{2} - \\sqrt{2}$，合并同类二次根式得 $4\\sqrt{2}$",
    },
    {
        "name": "数值计算",
        "grade": "七年级上学期",
        "kp": "有理数的混合运算",
        "qtype": "解答",
        "difficulty": 2,
        "stem": "计算：$(-2)^2 \\times 3 - 12 \\div (-4)$",
        "answer": "$15$",
        "skeleton": "先乘方 $(-2)^2=4$，再乘除 $4\\times3=12$、$12\\div(-4)=-3$，最后加减 $12-(-3)=15$",
    },
    {
        "name": "选择题（单选数值型）",
        "grade": "八年级下学期",
        "kp": "一元二次方程根与系数的关系",
        "qtype": "选择",
        "difficulty": 3,
        "stem": "方程 $x^2 - 4x + 3 = 0$ 的两根之和是（  ）\nA. $3$  B. $4$  C. $-4$  D. $-3$",
        "answer": "B（两根之和为 $4$）",
        "skeleton": "由根与系数关系 $x_1+x_2=-\\dfrac{b}{a}=4$（或因式分解得根 1、3 相加），选 B",
    },
    {
        "name": "不等式",
        "grade": "八年级上学期",
        "kp": "一元一次不等式的解法",
        "qtype": "解答",
        "difficulty": 2,
        "stem": "解不等式：$2x - 3 > 5x + 6$",
        "answer": "$x < -3$",
        "skeleton": "移项合并 $-3x > 9$，两边同除以 $-3$ 不等号反向，得 $x < -3$",
    },
    {
        "name": "几何证明",
        "grade": "八年级上学期",
        "kp": "等腰三角形的性质与三角形全等",
        "qtype": "证明",
        "difficulty": 3,
        "stem": "已知：在 $\\triangle ABC$ 中，$AB = AC$，$D$ 是 $BC$ 的中点。求证：$AD \\perp BC$。",
        "answer": "证明：由 $AB=AC$、$BD=CD$、$AD=AD$（SSS）得 $\\triangle ABD \\cong \\triangle ACD$，"
        "故 $\\angle ADB = \\angle ADC$；又两角互为邻补角，和为 $180^\\circ$，"
        "所以 $\\angle ADB = 90^\\circ$，即 $AD \\perp BC$。",
        "skeleton": "SSS 证全等 → 对应角相等 → 邻补角相等即各 90° → 垂直",
    },
    {
        "name": "文字应用题·一元一次",
        "grade": "七年级上学期",
        "kp": "一元一次方程的应用（分配问题）",
        "qtype": "解答",
        "difficulty": 3,
        "stem": "某班学生去公园划船，若每条船坐 4 人，则多出 8 人没有座位；"
        "若每条船坐 5 人，则恰好空出 2 个座位。问共有几条船、几名学生？",
        "answer": "10 条船，48 名学生",
        "skeleton": "设 $x$ 条船，人数两种表示相等：$4x+8 = 5x-2$，解得 $x=10$，学生 $4\\times10+8=48$ 人",
    },
    {
        "name": "文字应用题·一元二次",
        "grade": "八年级下学期",
        "kp": "一元二次方程的应用（平均变化率）",
        "qtype": "解答",
        "difficulty": 3,
        "stem": "某商品原价 100 元，经过连续两次降价后售价为 81 元，"
        "且两次降价的百分率相同。求平均每次降价的百分率。",
        "answer": "$10\\%$",
        "skeleton": "设每次降价 $x$，列 $100(1-x)^2 = 81$，解得 $1-x=\\pm0.9$，取 $x=0.1$，即 10%",
    },
]


def _inject_state(m: dict) -> dict:
    """母题 → 注入 state：三锚高置信 analysis + mother_dna + 已确认 + 空 knobs（默认配方 3 道）。"""
    return {
        "analysis": {
            "grade": {"value": m["grade"], "confidence": 0.95},
            "kp": {"value": m["kp"], "confidence": 0.95},
            "qtype": {"value": m["qtype"], "confidence": 0.95},
        },
        "mother_dna": {
            "stem": m["stem"],
            "answer": m["answer"],
            "solution_skeleton": m["skeleton"],
            "difficulty": m["difficulty"],
        },
        "mother_confirmed": True,
        "knobs": {},  # 抽过但老师没提 → 默认配方（3 = 2 普通 + 1 难），绕过 knobs 抽取 LLM
        "items": [],
    }


def _item_row(it: dict) -> dict:
    chk = it.get("check") or {}
    gene = it.get("gene") or {}
    return {
        "verify": chk.get("verify") or (chk.get("review") and "proof_review") or "missing",
        "tier": chk.get("tier") or "missing",
        "gene_gate": gene.get("gate") or "missing",
        "stem_head": (it.get("stem") or "")[:24],
    }


async def run_one(app, idx: int, m: dict, token: str) -> dict:
    """单道母题：注入 state → 一句无 URL 人话触发 route_entry→generate → 收集闸结果。"""
    cfg = {"configurable": {"thread_id": f"c018-{idx}", "ruoyi_token": token}}
    # 同 c017「graph 直驱 + 注入 state」模式：as_node=analyze 落 checkpoint，
    # 下一次 ainvoke 从入口路由（mother_confirmed+mother_dna+无 items → generate）。
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
        "ok": True,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "items": [_item_row(it) for it in items],
        "dropped": len(dropped),
        "dropped_notes": dropped,
    }


def _fmt_mother_line(r: dict) -> str:
    if not r.get("ok"):
        return f"  [{r['idx']:>2}] {r['name']}：ERROR（{r.get('error', '?')}）"
    cells = " | ".join(
        f"verify={row['verify']},tier={row['tier']},gene={row['gene_gate']}" for row in r["items"]
    )
    return (
        f"  [{r['idx']:>2}] {r['name']}：出 {len(r['items'])} 道，剔除 {r['dropped']} 道，"
        f"{r['elapsed_s']}s\n        {cells or '（无成题）'}"
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-C-012 G8 基准回放（sympy 可验率 vs 基线 42%）")
    ap.add_argument("--limit", type=int, default=len(MOTHERS), help="只跑前 N 道母题（默认全 10 道）")
    ap.add_argument("--json", action="store_true", help=f"结果落 {RESULT_JSON.name}")
    args = ap.parse_args()
    mothers = MOTHERS[: max(1, args.limit)]

    token = await real_token()  # 身份硬闸：graph 入口要能解出 userId（缓存一次全场复用）
    app = graph.compile(checkpointer=MemorySaver())

    print(f"=== C018 基准回放：{len(mothers)}/{len(MOTHERS)} 道母题（基线可验率 42%，目标 ≥80%）===")
    results: list[dict] = []
    for i, m in enumerate(mothers, start=1):
        print(f"--- [{i}/{len(mothers)}] {m['name']} ...", flush=True)
        try:
            r = await asyncio.wait_for(run_one(app, i, m, token), timeout=PER_MOTHER_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — 单道失败降级记错继续（G5 语义），不挂全场
            r = {"idx": i, "name": m["name"], "ok": False, "error": f"{type(e).__name__}: {e}",
                 "items": [], "dropped": 0, "dropped_notes": []}
        results.append(r)
        print(_fmt_mother_line(r), flush=True)

    # ── 汇总（真值只来自闸B/闸A 落的标记，本脚本不做任何裁决）──
    all_rows = [row for r in results for row in r["items"]]
    total = len(all_rows)
    n_sympy = sum(1 for x in all_rows if x["verify"] == "sympy_pass")
    tiers = {}
    for x in all_rows:
        tiers[x["tier"]] = tiers.get(x["tier"], 0) + 1
    n_dropped = sum(r["dropped"] for r in results)
    n_err = sum(1 for r in results if not r.get("ok"))
    rate = (n_sympy / total) if total else 0.0

    print("\n=== 汇总 ===")
    print(f"母题数: {len(results)}（失败 {n_err}）  最终成题总数: {total}  剔除: {n_dropped}")
    print(f"sympy_pass: {n_sympy}/{total} = {rate:.1%}  （基线 42% / AC5 目标 ≥80%）")
    print(
        "tier 分布: "
        + "  ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
        + (f"\n剔除叙事: {'；'.join(n for r in results for n in r['dropped_notes'])}" if n_dropped else "")
    )
    verdict = "PASS(≥80%)" if rate >= 0.80 else ("ABOVE-BASELINE(>42%)" if rate > 0.42 else "BELOW-BASELINE")
    print(f"判定: {verdict}")

    if args.json:
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mothers_run": len(results),
            "mothers_failed": n_err,
            "total_items": total,
            "sympy_pass": n_sympy,
            "sympy_pass_rate": round(rate, 4),
            "baseline": 0.42,
            "target": 0.80,
            "tier_counts": tiers,
            "dropped": n_dropped,
            "verdict": verdict,
            "results": results,
        }
        RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已落: {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # PowerShell 下双保险（配合 PYTHONIOENCODING）
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
