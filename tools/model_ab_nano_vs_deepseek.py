"""gpt-5.4-nano vs deepseek-v4-flash 三项对比评测（只评测，零改默认值/零动 .env/零重启 :8093）。

背景：昨晚误用旧名 `gpt-5-nano`（settings.py:183 写死）→ 视觉空返/站点404/solve 慢且一致率 66.7%，
结论全作废（错牌子被路由到兜底）。本脚本用 GET /models 核出的**真名** `gpt-5.4-nano` 重测。

三项（同题同 prompt，双模型并排，判决只读 sympy）：
  vision  3 张真母题图视觉读图：题干/年级/考点抽得出且不胡编 + 单次耗时。
  solve   闸B 重解：3 母题 vs gpt-5.4 基准的 sympy 判决一致率 + 耗时（判决永远只读 sympy verdict）。
  light   轻活结构化抽取：母题文本各跑一次 DNA 式抽取，看 JSON 守 schema 程度 + 耗时。

跑法（cwd = toolkit 根；solve/light 不入库不查树，RuoYi 不必在跑）：
  $env:PYTHONIOENCODING='utf-8'; .venv\\Scripts\\python.exe tools\\model_ab_nano_vs_deepseek.py
  --model gpt-5.4-nano   被测型号（也支持 deepseek-v4-flash 等）
  --step vision|solve|light|all
  --site aigeek|lk888    指定站点（缺省主站 aigeek）
  --limit N              solve 只跑前 N 道
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from core import relay_pool  # noqa: E402
from core.settings import settings  # noqa: E402

import agents.variant as V  # noqa: E402
from agents.variant import ANALYZE_PROMPT  # noqa: E402

from c018_benchmark_replay import MOTHERS  # noqa: E402

BASELINE = "gpt-5.4"

# 三张真实母题图（与 nano_vision_probe / deepseek_probe 同源 = c018 压轴真题）
IMAGES = [
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-26/06a3e548-e3d6-43ce-900f-c0d0e9cdd601/list/13/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
]

# 轻活·DNA 式结构化抽取（锚定+配方），看 JSON 守不守 schema
LIGHT_PROMPT = """你是浙教版初中数学命题专家。对下面这道题做**结构化 DNA 抽取**，只输出一个 JSON（不要任何解释、不要 markdown fence）：
{{
  "main_kp": "主知识点名（<=16字）",
  "qtype": "选择/填空/解答 之一",
  "exam_type": "直接计算/公式套用/性质判定/证明推理/应用建模/探究归纳 之一",
  "skeleton": ["解法步骤1", "解法步骤2"],
  "hard_points": [],
  "tags": ["3~6个检索标签"],
  "difficulty": 1
}}
年级：{grade}
题干：{stem}
标准答案：{answer}"""

LIGHT_REQUIRED = {"main_kp", "qtype", "exam_type", "skeleton", "tags", "difficulty"}


def _relay_at(site: str):
    for r in relay_pool._relays():
        if r.name == site:
            return r
    return relay_pool._relays()[0]


def _chat(site: str, model: str) -> ChatOpenAI:
    r = _relay_at(site)
    return ChatOpenAI(
        model=model,
        temperature=0.5,
        streaming=False,
        openai_api_base=r.base_url,
        openai_api_key=r.api_key,
        timeout=300,
    )


def _strip_json(text: str):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t
        t = t.replace("json", "", 1).strip() if t.lstrip().startswith("json") else t
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except Exception:
            return None
    return None


# --- vision ---------------------------------------------------------------
async def step_vision(model: str, site: str) -> None:
    print(f"\n=== STEP vision · model={model} @ {site} · 3 张真母题图 ===")
    chat = _chat(site, model)
    for url in IMAGES:
        msg = HumanMessage(content=[
            {"type": "text", "text": ANALYZE_PROMPT.format(utterance="（无）")},
            {"type": "image_url", "image_url": {"url": url}},
        ])
        t0 = time.monotonic()
        try:
            resp = await chat.ainvoke([msg], max_tokens=settings.VARIANT_MAX_TOKENS)
            dur = time.monotonic() - t0
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _strip_json(text)
            out = {
                "url": url[-46:], "dur_s": round(dur, 1), "parsed_ok": data is not None,
                "is_question_image": (data or {}).get("is_question_image"),
                "grade": (data or {}).get("grade"), "subject": (data or {}).get("subject"),
                "kp": (data or {}).get("kp"), "qtype": (data or {}).get("qtype"),
                "stem": ((data or {}).get("stem") or "")[:160],
                "raw_head": None if data else (text or "")[:160],
            }
        except Exception as ex:  # noqa: BLE001
            out = {"url": url[-46:], "error": str(ex)[:220], "dur_s": round(time.monotonic() - t0, 1)}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print("-" * 56)


# --- solve (A/B sympy) ----------------------------------------------------
async def _solve_verify(item: dict, model: str) -> dict:
    """强制 SOLVE 档 = model 跑闸B 重解 + sympy 验算，返回 verdict + 耗时。判决只读 sympy。"""
    old = settings.VARIANT_MODEL_SOLVE
    settings.VARIANT_MODEL_SOLVE = model
    try:
        t0 = time.monotonic()
        solved = await V._solve_one(item.get("stem", ""))
        res = await V._machine_verify(
            {"stem": item.get("stem"), "answer": item.get("answer"), "qtype": item.get("qtype")},
            solved.get("solved_answer"),
        )
        return {
            "verdict": res.get("verdict"),
            "solved": (solved.get("solved_answer") or "")[:60],
            "dur_s": round(time.monotonic() - t0, 1),
        }
    finally:
        settings.VARIANT_MODEL_SOLVE = old


async def step_solve(model: str, limit: int) -> None:
    print(f"\n=== STEP solve A/B · {model} vs {BASELINE} · sympy 判决（只读 sympy）===")
    mothers = [m for m in MOTHERS if m.get("qtype") not in ("证明", "作图")]
    if limit:
        mothers = mothers[:limit]
    agree = m_warn = b_warn = 0
    m_dur = b_dur = 0.0
    for m in mothers:
        cand = await _solve_verify(m, model)
        base = await _solve_verify(m, BASELINE)
        same = cand["verdict"] == base["verdict"]
        agree += int(same)
        m_warn += int(cand["verdict"] == V.math_verify.DEGRADE)
        b_warn += int(base["verdict"] == V.math_verify.DEGRADE)
        m_dur += cand["dur_s"]
        b_dur += base["dur_s"]
        print(json.dumps({
            "stem": (m.get("stem") or "")[:46], "cand": cand["verdict"], "base": base["verdict"],
            "agree": same, "cand_s": cand["dur_s"], "base_s": base["dur_s"],
        }, ensure_ascii=False))
    n = len(mothers)
    print("=" * 56)
    print(f"题数 {n} · 判决一致 {agree}/{n} = {agree / n * 100:.1f}% （阈值 >=90%）")
    print(f"{model} degrade {m_warn}/{n}  vs  {BASELINE} degrade {b_warn}/{n}")
    print(f"耗时均: {model}={m_dur / n:.1f}s/题  {BASELINE}={b_dur / n:.1f}s/题")


# --- light (DNA extraction) ------------------------------------------------
async def step_light(model: str, site: str) -> None:
    print(f"\n=== STEP light · {model} @ {site} · DNA 式结构化抽取 ===")
    chat = _chat(site, model)
    samples = [MOTHERS[1], MOTHERS[7]]  # 一计算一（不等式/含分支）
    for m in samples:
        prompt = LIGHT_PROMPT.format(grade=m.get("grade", ""), stem=m.get("stem", ""), answer=m.get("answer", ""))
        t0 = time.monotonic()
        try:
            resp = await chat.ainvoke([HumanMessage(content=prompt)], max_tokens=1024)
            dur = time.monotonic() - t0
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _strip_json(text)
            missing = sorted(LIGHT_REQUIRED - set(data.keys())) if isinstance(data, dict) else list(LIGHT_REQUIRED)
            out = {
                "name": m.get("name"), "dur_s": round(dur, 1), "parsed_ok": data is not None,
                "schema_ok": data is not None and not missing, "missing_keys": missing,
                "json": data if data else None, "raw_head": None if data else (text or "")[:160],
            }
        except Exception as ex:  # noqa: BLE001
            out = {"name": m.get("name"), "error": str(ex)[:220], "dur_s": round(time.monotonic() - t0, 1)}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print("-" * 56)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-nano")
    ap.add_argument("--step", default="all", choices=["all", "vision", "solve", "light"])
    ap.add_argument("--site", default="aigeek")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    print(f"[被测 = {args.model} · 站点 = {args.site} · 基准 = {BASELINE}]")
    if args.step in ("all", "vision"):
        await step_vision(args.model, args.site)
    if args.step in ("all", "solve"):
        await step_solve(args.model, args.limit)
    if args.step in ("all", "light"):
        await step_light(args.model, args.site)


if __name__ == "__main__":
    asyncio.run(main())
