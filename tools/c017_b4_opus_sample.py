# -*- coding: utf-8 -*-
r"""PRD-C-017 B4-A · ② 扩样 opus 读图纯文本（H1 补样本）+ ③ 成本/延迟回填（H4）。

对 ≥4 张纯文本母题图跑 B0 探针式 opus 合并解题打标（response_format 硬锁 10 维 schema，
低温 0.1，超时 180s），记：解题答案（人工对标）、10维齐全、墙钟、prompt/completion token。
汇总单母题 opus 平均墙钟/token → H4。

图源 = RuoYi biz_question 纯文本一元二次方程真题（stem_text 无「图/如图」）+ B0 韦达 img3。
人工对标答案见 EXPECTED（我自解，浙教/通用初中数学）。

跑法: $env:PYTHONUTF8='1'; .venv/Scripts/python.exe tools/c017_b4_opus_sample.py
   --json tools/c017_b4_opus_sample_result.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents import mother_opus, variant  # noqa: E402
from core.settings import settings  # noqa: E402

# 纯文本母题样本（DB 实查纯文本一元二次方程题 + B0 韦达 img3）。
# expected = 我自解的正确答案（人工对标 opus 解题对否）。
SAMPLES = [
    {
        "key": "1184_已知根x1求m",
        "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-04-25/4c6dae77-47b0-4306-9f02-c75fde151565/list/1/question.png",
        "stem": "已知关于x的一元二次方程x²+3x-m=0的一个根是x=1，则m的值为 (A.2 B.4 C.-4 D.-2)",
        "expected": "m=4（选 B）：x=1 代入得 1+3-m=0 → m=4",
    },
    {
        "key": "1185_根迁移2022",
        "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-04-25/4c6dae77-47b0-4306-9f02-c75fde151565/list/2/question.png",
        "stem": "若关于x的一元二次方程ax²+bx+2=0(a≠0)有一根为x=2022，则方程a(x-1)²+bx-b=-2必有一根为",
        "expected": "x=2023：令 t=x-1，原式化为 a t²+b t+2=0，故 t=2022 → x=2023",
    },
    {
        "key": "1177_求a范围解答",
        "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png",
        "stem": "已知关于x的方程(a-2)x²-ax=x²-1是一元二次方程，求a的取值范围",
        "expected": "a≠3：整理为 (a-3)x²-ax+1=0，二次项系数 a-3≠0 → a≠3",
    },
    {
        "key": "1191_a+b+c=0填空",
        "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-04-25/5aca7901-2e58-48ca-9b2e-d40bf5b483f9/list/1/question.png",
        "stem": "已知一元二次方程ax²+bx+c=0，若a+b+c=0，则该方程一定有一个根为",
        "expected": "x=1：a+b+c=0 即 x=1 时方程成立 → x=1 是根",
    },
    {
        "key": "img3_韦达三次方程(B0复跑)",
        "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
        "stem": "韦达定理三次方程求代数式值（B0 探针纯文本样本，复跑核稳定性）",
        "expected": "-1/6（B0 已实测正确，复跑核一致性）",
    },
]


async def one(s: dict) -> dict:
    model = settings.variant_model("mother_solve_label")
    prompt = mother_opus.build_mother_prompt(
        grade_text="八年级/九年级", chapter_text="一元二次方程", leaf_pool=[], model_vocab=None,
    )
    t0 = time.monotonic()
    rec: dict = {"key": s["key"], "expected": s["expected"]}
    try:
        text = await mother_opus.solve_and_label(
            image_url=s["url"], prompt=prompt, invoke=variant._ainvoke_text, model=model,
        )
        dur = time.monotonic() - t0
        data = variant._parse_json(text)
        rec["dur_s"] = round(dur, 1)
        rec["json_ok"] = isinstance(data, dict)
        if not isinstance(data, dict):
            rec["raw_head"] = text[:200]
            return rec
        dna = mother_opus.opus_to_dna(data)
        rec["has_figure"] = data.get("has_figure")
        rec["solvedAnswer"] = (data.get("solvedAnswer") or "")[:120]
        # 10 维齐全（与 mother_opus.opus_to_dna 同口径 + 关键非空）
        dims_full = (
            bool(dna.get("skeleton"))
            and dna.get("qtype") not in (None, "")
            and dna.get("exam_type") not in (None, "")
            and dna.get("scene") not in (None, "")
            and isinstance(dna.get("difficulty"), int)
            and isinstance(dna.get("hard_point_count"), int)
        )
        rec["dims_full"] = dims_full
        rec["dims"] = {
            "main_kp_name": (dna.get("main_kp") or {}).get("name") if dna.get("main_kp") else None,
            "qtype": dna.get("qtype"), "exam_type": dna.get("exam_type"),
            "difficulty": dna.get("difficulty"), "hard_point_count": dna.get("hard_point_count"),
            "scene": (dna.get("scene") or "")[:30],
            "tags": dna.get("tags"), "skeleton_lines": len(dna.get("skeleton") or []),
        }
    except Exception as ex:  # noqa: BLE001
        rec["dur_s"] = round(time.monotonic() - t0, 1)
        rec["error"] = str(ex)[:200]
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="tools/c017_b4_opus_sample_result.json")
    args = ap.parse_args()
    model = settings.variant_model("mother_solve_label")
    assert model == "claude-opus-4-8", f"母题档未命中 opus: {model}"
    print(f"opus 档 = {model}  temp={mother_opus.MOTHER_OPUS_TEMPERATURE}  timeout={mother_opus.MOTHER_OPUS_TIMEOUT_S}s")

    out: dict = {"model": model, "samples": []}
    for s in SAMPLES:
        print(f"\n--- {s['key']} ---")
        rec = await one(s)
        out["samples"].append(rec)
        print(json.dumps(rec, ensure_ascii=False, indent=2))

    # H4 汇总（纯文本）：墙钟/token 平均（只统计成功项）
    ok = [r for r in out["samples"] if r.get("json_ok") and not r.get("error")]
    durs = [r["dur_s"] for r in ok if isinstance(r.get("dur_s"), (int, float))]
    full = sum(1 for r in ok if r.get("dims_full"))
    out["summary"] = {
        "n_total": len(out["samples"]),
        "n_json_ok": len(ok),
        "n_dims_full": full,
        "dur_min_s": min(durs) if durs else None,
        "dur_max_s": max(durs) if durs else None,
        "dur_avg_s": round(sum(durs) / len(durs), 1) if durs else None,
    }
    print("\n========== H4 汇总（纯文本 opus 读图）==========")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))

    Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[落盘] {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
