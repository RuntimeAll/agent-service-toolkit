# -*- coding: utf-8 -*-
r"""PRD-C-017 B6 · 隔离实验：同一条接地 prompt，三模型读图判「年级册 + 章」命中率对照。

核心 = **只换模型、prompt 不变**，把「模型能力」从「prompt 接地」里隔离出来：
  - prompt 复用刚改好的接地版 `mother_precheck.build_precheck_prompt()`（不传 grade_hint，三模型共用一份）。
  - 三模型：
      nano  = settings.LLM_MODEL_LIGHT            （被测对象，应是 gpt-5.4-nano）
      mid   = settings.COMPATIBLE_MODEL           （深度思考档，应是 gpt-5.4；ANALYZE 在 .env 被降到 nano，
                                                   用它会塌成 nano vs nano，故中档取 COMPATIBLE_MODEL）
      ceil  = settings.VARIANT_MODEL_MOTHER_SOLVE_LABEL  （天花板，应是 claude-opus-4-8）
  - 调用通路复用生产路径 relay_pool.ainvoke_failover（= _ainvoke_text 内部调用的同一条多中转池 + per-call
    model 覆盖 + 熔断转移 + usage），传多模态 HumanMessage([{text},{image_url}])。
  - 解析复用 mother_precheck.normalize_precheck。

判命中：
  - 年级命中（主指标）：模型判 grade_book 归一 == 真值 year_name（册/上下学期口径容差）。
  - 章命中（次指标）：仅对卷名带精确章节号的题算（true_chapter 非空），模型判 chapter 归一命中真章名。
    七下复习卷 true_chapter=None → 只算年级、章 N/A 不计入章命中率。

🔴 跑前必设 $env:PYTHONUTF8=1（防 GBK 中文/数学符号崩）。凭据全部从 .env 读，脚本零硬编码 key。
🔴 渲染图（COS URL）非老师拍照 → 命中率是最好情况上限；真实拍照只会更差（报告须带此限定）。

跑法（cwd = toolkit 根）：
  $env:PYTHONUTF8=1; .venv/Scripts/python.exe tools/c017_model_probe_grade_chapter.py \
    --json artifacts/...json --concurrency 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.mother_precheck import build_precheck_prompt, normalize_precheck  # noqa: E402
from core import relay_pool  # noqa: E402
from core.settings import settings  # noqa: E402
from agents.variant import _parse_json  # noqa: E402

# --- 三模型真名（去 settings 读，不硬编码）---------------------------------
NANO = settings.LLM_MODEL_LIGHT                       # gpt-5.4-nano
MID = settings.COMPATIBLE_MODEL                       # gpt-5.4（深度思考档；ANALYZE 在 .env 降到 nano，不用它）
CEIL = settings.VARIANT_MODEL_MOTHER_SOLVE_LABEL      # claude-opus-4-8
MODELS = {"nano": NANO, "mid": MID, "opus": CEIL}

TIMEOUT_FAST = 60.0    # nano / mid
TIMEOUT_OPUS = 180.0
TEMPERATURE = 0.2
MAX_TOKENS = settings.VARIANT_MAX_TOKENS

# --- 30 张测试样本（5 年级各 6 张，已选好，照单接收）------------------------
# true_chapter = 真章名（核对自 biz_subject level2）；None = 七下复习卷，章不定，只算年级。
SAMPLES: list[dict] = [
    {"qid": 17701, "year_code": "3071", "year_name": "七年级上册", "paper": "七年级上册 1.2 数轴", "true_chapter": "第一章 有理数", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-30/7164e5fc-ba08-4bbf-8a77-7e8c06023a40/list/36/question.png"},
    {"qid": 11790, "year_code": "3071", "year_name": "七年级上册", "paper": "七年级上册 1.2 数轴", "true_chapter": "第一章 有理数", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-09-25/dce8881d-51bf-47b0-ba77-dd80843a12fb/list/5/question.png"},
    {"qid": 9652, "year_code": "3071", "year_name": "七年级上册", "paper": "七年级上册 1.2 数轴", "true_chapter": "第一章 有理数", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-07-31/e811f226-f40c-4fe9-8be5-a50ecb9c233c/list/3/question.png"},
    {"qid": 24704, "year_code": "3071", "year_name": "七年级上册", "paper": "七年级上册 3.1 平方根", "true_chapter": "第三章 实数", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-07-21/aa8cc258-ef5a-4a2d-8fcf-60b422f2e914/list/5/question.png"},
    {"qid": 24266, "year_code": "3071", "year_name": "七年级上册", "paper": "七年级上册 3.1 平方根", "true_chapter": "第三章 实数", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-07-17/587b7e6a-1519-4d1b-89a2-883db7a254cd/list/4/question.png"},
    {"qid": 15934, "year_code": "3071", "year_name": "七年级上册", "paper": "七年级上册 3.1 平方根", "true_chapter": "第三章 实数", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-11-28/ca8a6d23-05b7-4e98-bc7e-fb012986d848/list/3/question.png"},
    {"qid": 20383, "year_code": "3072", "year_name": "七年级下册", "paper": "七下期末复习之几何类", "true_chapter": None, "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-03-06/f4b8fb89-8245-4420-8660-ff4093f9ff51/list/10/question.png"},
    {"qid": 20869, "year_code": "3072", "year_name": "七年级下册", "paper": "七下期末复习之几何类", "true_chapter": None, "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-03-25/484937f0-50c8-4a8c-b820-3eb37f14b9b0/list/9/question.png"},
    {"qid": 21550, "year_code": "3072", "year_name": "七年级下册", "paper": "七下期末复习之几何类", "true_chapter": None, "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-04-17/eba3c6a5-e337-4337-a8b4-5ee9979e3d09/list/10/question.png"},
    {"qid": 20863, "year_code": "3072", "year_name": "七年级下册", "paper": "七下期末复习整数解", "true_chapter": None, "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-03-25/484937f0-50c8-4a8c-b820-3eb37f14b9b0/list/3/question.png"},
    {"qid": 21555, "year_code": "3072", "year_name": "七年级下册", "paper": "七下期末复习整数解", "true_chapter": None, "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-04-17/eba3c6a5-e337-4337-a8b4-5ee9979e3d09/list/15/question.png"},
    {"qid": 7877, "year_code": "3072", "year_name": "七年级下册", "paper": "七下期末复习整数解", "true_chapter": None, "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-06-22/331fc45d-c121-48b0-b64c-aa274b90394b/list/22/question.png"},
    {"qid": 18105, "year_code": "3081", "year_name": "八年级上册", "paper": "八年级上册 1.1 认识三角形", "true_chapter": "第一章 三角形的初步知识", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-01-12/215c5fcf-e708-4263-b248-818289ddb5d0/list/3/question.png"},
    {"qid": 17424, "year_code": "3081", "year_name": "八年级上册", "paper": "八年级上册 1.1 认识三角形", "true_chapter": "第一章 三角形的初步知识", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-30/74a2a88f-0455-48a3-90b4-23cbf46d5520/list/7/question.png"},
    {"qid": 15732, "year_code": "3081", "year_name": "八年级上册", "paper": "八年级上册 1.1 认识三角形", "true_chapter": "第一章 三角形的初步知识", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-11-28/8887c4a6-5ce7-4ff2-8bb2-72f2e753935d/list/3/question.png"},
    {"qid": 14226, "year_code": "3081", "year_name": "八年级上册", "paper": "八年级上册 2.7 勾股定理", "true_chapter": "第二章 特殊三角形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-11-08/936b49c0-6707-4b6c-a986-0eae798703d2/list/1/question.png"},
    {"qid": 13920, "year_code": "3081", "year_name": "八年级上册", "paper": "八年级上册 2.7 勾股定理", "true_chapter": "第二章 特殊三角形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-11-04/e0f2fda5-b489-42ec-a37c-6f78a72984b5/list/4/question.png"},
    {"qid": 13921, "year_code": "3081", "year_name": "八年级上册", "paper": "八年级上册 2.7 勾股定理", "true_chapter": "第二章 特殊三角形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-11-04/e0f2fda5-b489-42ec-a37c-6f78a72984b5/list/5/question.png"},
    {"qid": 23018, "year_code": "3082", "year_name": "八年级下册", "paper": "2025年八年级下册期末复习之一元二次方程", "true_chapter": "第二章一元二次方程", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-05-22/49d919f7-a360-4b18-897a-f4ba4d33e79b/list/1/question.png"},
    {"qid": 23019, "year_code": "3082", "year_name": "八年级下册", "paper": "2025年八年级下册期末复习之一元二次方程", "true_chapter": "第二章一元二次方程", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-05-22/49d919f7-a360-4b18-897a-f4ba4d33e79b/list/2/question.png"},
    {"qid": 23021, "year_code": "3082", "year_name": "八年级下册", "paper": "2025年八年级下册期末复习之一元二次方程", "true_chapter": "第二章一元二次方程", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-05-22/49d919f7-a360-4b18-897a-f4ba4d33e79b/list/4/question.png"},
    {"qid": 23161, "year_code": "3082", "year_name": "八年级下册", "paper": "2025年八年级下册期末复习之特殊平行四边形", "true_chapter": "第五章特殊平行四边形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-05-22/75ebe996-55e8-401c-bba7-6c2b66fc997d/list/1/question.png"},
    {"qid": 23162, "year_code": "3082", "year_name": "八年级下册", "paper": "2025年八年级下册期末复习之特殊平行四边形", "true_chapter": "第五章特殊平行四边形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-05-22/75ebe996-55e8-401c-bba7-6c2b66fc997d/list/2/question.png"},
    {"qid": 23164, "year_code": "3082", "year_name": "八年级下册", "paper": "2025年八年级下册期末复习之特殊平行四边形", "true_chapter": "第五章特殊平行四边形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-05-22/75ebe996-55e8-401c-bba7-6c2b66fc997d/list/4/question.png"},
    {"qid": 16871, "year_code": "3091", "year_name": "九年级上册", "paper": "九年级上册 3.5 圆周角（1）", "true_chapter": "第三章 圆的基本性质", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-24/8bbe57bc-dc8f-4ef2-b137-61cdb328f667/list/4/question.png"},
    {"qid": 16265, "year_code": "3091", "year_name": "九年级上册", "paper": "九年级上册 3.5 圆周角（1）", "true_chapter": "第三章 圆的基本性质", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-05/ea03d29c-4f72-4878-849a-c727dac92c6d/list/5/question.png"},
    {"qid": 14433, "year_code": "3091", "year_name": "九年级上册", "paper": "九年级上册 3.5 圆周角（1）", "true_chapter": "第三章 圆的基本性质", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-11-15/c92499d7-9efc-47ab-a78d-c147ce5d50bd/list/4/question.png"},
    {"qid": 21003, "year_code": "3091", "year_name": "九年级上册", "paper": "九年级上册 4.1 比例线段", "true_chapter": "第四章 相似三角形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-03-28/3d5f6c62-3b08-4b6c-ad03-e1b235c53d36/list/3/question.png"},
    {"qid": 17818, "year_code": "3091", "year_name": "九年级上册", "paper": "九年级上册 4.1 比例线段", "true_chapter": "第四章 相似三角形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2025-01-04/74ae4073-162f-4d75-b605-b6751f94dba4/list/6/question.png"},
    {"qid": 16004, "year_code": "3091", "year_name": "九年级上册", "paper": "九年级上册 4.1 比例线段", "true_chapter": "第四章 相似三角形", "url": "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-01/2f8f3cbc-ed3b-4053-8fe0-c2a6d2a8bdf2/list/1/question.png"},
]


# --- 归一 + 命中判定 -------------------------------------------------------
# 年级归一：识别 七/八/九/初一/初二/初三/阿拉伯年级 + 上下（册/学期口径折叠），输出紧凑键 '8下'。
_GRADE_ALIASES = {
    "7": ["七年级", "七", "初一", "7年级"],
    "8": ["八年级", "八", "初二", "8年级"],
    "9": ["九年级", "九", "初三", "9年级"],
}


def normalize_grade(s: str) -> str:
    """年级册归一 → 形如 '8下' 的紧凑键。空/识别不出 → 空串。"""
    s = re.sub(r"\s+", "", str(s or ""))
    if not s:
        return ""
    grade = ""
    for g, aliases in _GRADE_ALIASES.items():
        if any(a in s for a in aliases):
            grade = g
            break
    if not grade:
        return ""
    half = "上" if "上" in s else ("下" if "下" in s else "")
    return f"{grade}{half}"


def grade_hit(pred: str, truth: str) -> bool:
    np_, nt = normalize_grade(pred), normalize_grade(truth)
    return bool(np_) and np_ == nt


# 章号归一：把中文数字章序号 + 阿拉伯数字统一，再比对核心章名关键词。
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def chapter_index(s: str) -> int | None:
    """抽「第X章」的序号（X 支持中文/阿拉伯）。无则 None。"""
    s = str(s or "")
    m = re.search(r"第\s*([一二三四五六七八九十\d])\s*章", s)
    if not m:
        return None
    t = m.group(1)
    if t.isdigit():
        return int(t)
    return _CN_NUM.get(t)


def chapter_core(s: str) -> str:
    """去「第X章」前缀 + 去空格 → 纯章名核心，用于名匹配。"""
    s = re.sub(r"第\s*[一二三四五六七八九十\d]+\s*章", "", str(s or ""))
    return re.sub(r"\s+", "", s)


def chapter_hit(pred: str, truth: str) -> bool:
    """章命中：章序号一致 且（章名核心互含）即算命中。两条任一稳健通道。

    - 序号通道：第X章 序号一致（容中文/阿拉伯差异）；
    - 名称通道：核心章名互为子串（容空格 + 多/少字）。
    两条都满足才算命中（防「第二章」配错名也算对）。无序号时退回纯名称互含。
    """
    if not pred or not truth:
        return False
    pi, ti = chapter_index(pred), chapter_index(truth)
    pc, tc = chapter_core(pred), chapter_core(truth)
    name_ok = bool(tc) and (tc in pc or pc in tc) if pc else False
    if pi is not None and ti is not None:
        return pi == ti and name_ok
    # 模型没写章号 → 仅靠名称互含
    return name_ok


# --- 单次调用：生产路径 relay_pool.ainvoke_failover（= _ainvoke_text 内核）----
async def call_model(label: str, model: str, url: str, prompt: str) -> dict:
    """一次多模态调用 + 归一。失败重试 1 次；仍失败 → error。"""
    timeout = TIMEOUT_OPUS if label == "opus" else TIMEOUT_FAST
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    last_err = None
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            resp, relay, model_used, fb = await relay_pool.ainvoke_failover(
                [msg],
                max_tokens=MAX_TOKENS,
                tags=["skip_stream"],
                model=model,
                temperature=TEMPERATURE,
                timeout=timeout,
            )
            dur = round(time.monotonic() - t0, 1)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _parse_json(text)
            norm = normalize_precheck(data)
            pt, ct = relay_pool.usage_tokens(resp)
            return {
                "ok": True,
                "attempt": attempt,
                "relay": relay,
                "model_used": model_used,
                "fallback": fb,
                "dur_s": dur,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "grade_book": norm["grade_book"],
                "chapter": norm["chapter"],
                "grade_candidates": norm["grade_candidates"],
                "chapter_candidates": norm["chapter_candidates"],
                "has_figure": norm["has_figure"],
                "confidence": norm["confidence"],
                "parsed_ok": data is not None,
                "raw_head": None if data is not None else text[:200],
            }
        except Exception as ex:  # noqa: BLE001
            last_err = str(ex)[:240]
    return {"ok": False, "error": last_err, "attempt": 2}


async def run_sample(sem: asyncio.Semaphore, sample: dict, prompt: str) -> dict:
    """对一个样本，串行跑三模型（同图三调用并发会更易触发限流；样本间才并发）。"""
    async with sem:
        rec = {k: sample[k] for k in ("qid", "year_code", "year_name", "paper", "true_chapter", "url")}
        rec["models"] = {}
        for label, model in MODELS.items():
            res = await call_model(label, model, sample["url"], prompt)
            if res.get("ok"):
                res["grade_hit"] = grade_hit(res["grade_book"], sample["year_name"])
                if sample["true_chapter"]:
                    res["chapter_hit"] = chapter_hit(res["chapter"], sample["true_chapter"])
                else:
                    res["chapter_hit"] = None  # N/A（复习卷）
            rec["models"][label] = res
            print(
                f"  [{sample['qid']}/{label:<4}] "
                + (
                    f"grade={res.get('grade_book','')!r}({'O' if res.get('grade_hit') else 'X'}) "
                    f"chap={res.get('chapter','')!r}"
                    f"({'O' if res.get('chapter_hit') else ('NA' if res.get('chapter_hit') is None else 'X')}) "
                    f"{res.get('dur_s')}s relay={res.get('relay')}"
                    if res.get("ok")
                    else f"ERROR {res.get('error')}"
                ),
                flush=True,
            )
        return rec


def tally(results: list[dict]) -> dict:
    """汇总命中率（总 + 分年级）+ error 计数。"""
    grades_order = ["七年级上册", "七年级下册", "八年级上册", "八年级下册", "九年级上册"]
    summary: dict = {"models": {}, "by_grade": {}}
    for label in MODELS:
        g_hit = g_tot = c_hit = c_tot = err = 0
        per_grade: dict[str, dict] = {g: {"g_hit": 0, "g_tot": 0, "c_hit": 0, "c_tot": 0, "err": 0} for g in grades_order}
        for r in results:
            m = r["models"].get(label, {})
            gname = r["year_name"]
            pg = per_grade[gname]
            if not m.get("ok"):
                err += 1
                pg["err"] += 1
                continue
            g_tot += 1
            pg["g_tot"] += 1
            if m.get("grade_hit"):
                g_hit += 1
                pg["g_hit"] += 1
            if m.get("chapter_hit") is not None:  # 非 N/A 才计章
                c_tot += 1
                pg["c_tot"] += 1
                if m.get("chapter_hit"):
                    c_hit += 1
                    pg["c_hit"] += 1
        summary["models"][label] = {
            "model": MODELS[label],
            "grade_hit": f"{g_hit}/{g_tot}",
            "grade_rate": round(g_hit / g_tot, 3) if g_tot else None,
            "chapter_hit": f"{c_hit}/{c_tot}",
            "chapter_rate": round(c_hit / c_tot, 3) if c_tot else None,
            "errors": err,
        }
        summary["by_grade"][label] = per_grade
    return summary


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="原始日志落盘路径")
    ap.add_argument("--concurrency", type=int, default=3, help="样本间并发（≤3~4 防限流）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个样本（调试用，0=全部）")
    args = ap.parse_args()

    prompt = build_precheck_prompt()  # 不传 grade_hint，三模型公平共用同一份接地 prompt

    print("=== C-017 B6 · 三模型读图判章隔离实验 ===")
    print(f"模型真名：nano={NANO!r}  mid={MID!r}  opus={CEIL!r}")
    print(f"调用：relay_pool.ainvoke_failover  temp={TEMPERATURE}  max_tokens={MAX_TOKENS}  "
          f"timeout(fast/opus)={TIMEOUT_FAST}/{TIMEOUT_OPUS}  并发={args.concurrency}")
    print(f"中转池：{[r.name for r in relay_pool._relays()]}")
    print(f"prompt 字符数={len(prompt)}（接地版 build_precheck_prompt，无 grade_hint）")
    print("-" * 70)

    samples = SAMPLES[: args.limit] if args.limit else SAMPLES
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()
    results = await asyncio.gather(*[run_sample(sem, s, prompt) for s in samples])
    # 按 qid 在原样本顺序排回（gather 顺序与传入一致，已稳定）
    elapsed = round(time.monotonic() - t0, 1)

    summary = tally(results)
    summary["models_meta"] = {"nano": NANO, "mid": MID, "opus": CEIL}
    summary["params"] = {
        "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
        "timeout_fast": TIMEOUT_FAST, "timeout_opus": TIMEOUT_OPUS,
        "concurrency": args.concurrency, "n_samples": len(samples),
        "prompt_len": len(prompt), "relays": [r.name for r in relay_pool._relays()],
        "elapsed_s": elapsed,
    }

    print("\n" + "=" * 70)
    print("命中率总表：")
    for label in MODELS:
        m = summary["models"][label]
        print(f"  {label:<5} {m['model']:<20} 年级 {m['grade_hit']} ({m['grade_rate']})  "
              f"章 {m['chapter_hit']} ({m['chapter_rate']})  errors={m['errors']}")
    print(f"\n总耗时 {elapsed}s")

    out = {"prompt": prompt, "summary": summary, "results": results}
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[原始日志已落盘 {args.json}]")


if __name__ == "__main__":
    asyncio.run(main())
