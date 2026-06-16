# -*- coding: utf-8 -*-
"""PRD-C-100 B0 预飞行 LLM 探针（go/no-go 主体）。

覆盖：
  h5   opus 一把（读图+解题+10维打标，真 build_mother_prompt + response_format）在**高 max_tokens
       (默认 16000，远超现 4096)** 下跑，量 completion 真分布(含 reasoning)，看是否被截断、
       定护栏值 = p99 + 余量。【go/no-go：定不下来 B1 不开工】
  h1   缓存：同一稳定前缀（真母题 prefix，≥4096 token）连发 N 次，看
       usage.prompt_tokens_details.cached_tokens > 0（aigeek 自动缓存坐实）。
  h3   opus 把一道几何题翻成 GeoGebra 命令 → render_geogebra 一轮直出，看渲染 ok。

跑法（cwd=toolkit 根）：
  $env:PYTHONUTF8=1; .venv/Scripts/python.exe tools/c100_b0_probe.py --step all --json tools/c100_b0_result.json
  --max-tokens 16000   h5 的护栏探测上限（设高=不截断才能量真分布）
  --reps 2             每图重复次数（看方差）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from agents import mother_opus  # noqa: E402
from core import relay_pool  # noqa: E402
from core.settings import settings  # noqa: E402

OPUS = settings.variant_model("mother_solve_label")  # claude-opus-4-8

# 三张真实母题图（含图几何/压轴，H5 量最大题分布）。
IMAGES = [
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-26/06a3e548-e3d6-43ce-900f-c0d0e9cdd601/list/13/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
]

# 合成一个 ~40 叶子的知识点池，让 build_mother_prompt 的稳定前缀够长（opus 缓存门槛 4096 token）。
_LEAF_POOL = [
    (f"110201{i:02d}", name)
    for i, name in enumerate([
        "一元二次方程的定义", "配方法解一元二次方程", "公式法解一元二次方程", "因式分解法解方程",
        "根的判别式", "根与系数的关系", "一元二次方程的应用", "二次函数的图象与性质",
        "二次函数与一元二次方程", "二次函数的最值", "相似三角形的判定", "相似三角形的性质",
        "圆的基本性质", "垂径定理", "圆周角定理", "切线的判定与性质", "正多边形与圆",
        "弧长与扇形面积", "锐角三角函数", "解直角三角形", "全等三角形的判定", "等腰三角形性质",
        "勾股定理", "平行四边形判定", "矩形菱形正方形", "中位线定理", "反比例函数",
        "一次函数图象", "图形的旋转", "中心对称", "轴对称", "图形的平移",
        "概率初步", "频率与概率", "统计图表", "数据的集中趋势", "数据的离散程度",
        "整式的运算", "分式方程", "二次根式",
    ], 1)
]


def _build_prefix() -> str:
    return mother_opus.build_mother_prompt(
        grade_text="九年级上册",
        chapter_text="第1章 二次根式 / 第2章 一元二次方程",
        leaf_pool=_LEAF_POOL,
        model_vocab=["十字相乘配方", "判别式分类讨论", "韦达定理整体代入"],
    )


def _main_relay():
    return relay_pool._relays()[0]


def _chat(model: str, max_tokens: int, *, temperature: float = 0.1, response_format=None) -> ChatOpenAI:
    r = _main_relay()
    kw = dict(
        model=model, temperature=temperature, streaming=True, stream_usage=True,
        openai_api_base=r.base_url, openai_api_key=r.api_key, timeout=600,
        max_tokens=max_tokens,
    )
    c = ChatOpenAI(**kw)
    if response_format is not None:
        c = c.bind(response_format=response_format)
    return c


def _strip_json(text: str):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t[3:]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            return None
    return None


async def _stream_opus(prefix: str, image_url: str, max_tokens: int):
    """astream 捞 content + reasoning_content + usage + finish_reason。"""
    chat = _chat(OPUS, max_tokens, response_format=mother_opus.RESPONSE_FORMAT)
    msg = HumanMessage(content=[
        {"type": "text", "text": prefix},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    content = ""
    reasoning = ""
    resp = None
    finish = None
    async for chunk in chat.astream([msg]):
        resp = chunk if resp is None else resp + chunk
        c = chunk.content
        if isinstance(c, str):
            content += c
        ak = getattr(chunk, "additional_kwargs", None) or {}
        rc = ak.get("reasoning_content")
        if isinstance(rc, str):
            reasoning += rc
        rm = getattr(chunk, "response_metadata", None) or {}
        if rm.get("finish_reason"):
            finish = rm.get("finish_reason")
    um = getattr(resp, "usage_metadata", None) or {}
    return {
        "content": content, "reasoning": reasoning, "finish_reason": finish,
        "input_tokens": um.get("input_tokens"), "output_tokens": um.get("output_tokens"),
        "output_token_details": um.get("output_token_details"),
    }


async def step_h5(max_tokens: int, reps: int) -> dict:
    print(f"\n=== H5 · opus 高 max_tokens={max_tokens} 量 completion 真分布(含 reasoning) ===")
    prefix = _build_prefix()
    out = {"max_tokens_probe": max_tokens, "opus": OPUS, "samples": []}
    cts = []
    for url in IMAGES:
        for rep in range(reps):
            t0 = time.monotonic()
            rec = {"url": url[-46:], "rep": rep}
            try:
                r = await _stream_opus(prefix, url, max_tokens)
                data = _strip_json(r["content"])
                miss = _missing(data)
                rec.update({
                    "dur_s": round(time.monotonic() - t0, 1),
                    "input_tokens": r["input_tokens"],
                    "output_tokens": r["output_tokens"],
                    "reasoning_chars": len(r["reasoning"]),
                    "content_chars": len(r["content"]),
                    "finish_reason": r["finish_reason"],
                    "truncated": r["finish_reason"] == "length",
                    "json_complete": data is not None,
                    "dims_full": data is not None and not miss,
                    "dims_missing": miss,
                    "output_token_details": r["output_token_details"],
                })
                if r["output_tokens"]:
                    cts.append(r["output_tokens"])
            except Exception as ex:  # noqa: BLE001
                rec.update({"error": str(ex)[:200], "dur_s": round(time.monotonic() - t0, 1)})
            out["samples"].append(rec)
            print(json.dumps(rec, ensure_ascii=False, indent=2))
            print("-" * 60)
    if cts:
        cts.sort()
        out["completion_stats"] = {
            "n": len(cts), "min": cts[0], "max": cts[-1],
            "p50": cts[len(cts) // 2], "p99_approx": cts[-1],
            "any_truncated": any(s.get("truncated") for s in out["samples"]),
        }
        # 护栏建议 = max * 1.5 向上取整到 1024（防失控但不截断；母题节点宽护栏不省钱）
        rec_guard = ((int(cts[-1] * 1.5) // 1024) + 1) * 1024
        out["recommended_guardrail"] = rec_guard
        print(f">>> completion: min={cts[0]} p50={cts[len(cts)//2]} max={cts[-1]} | 建议护栏={rec_guard} | 截断={out['completion_stats']['any_truncated']}")
    return out


def _missing(data) -> list:
    if not isinstance(data, dict):
        return ["__not_dict__"]
    miss = []
    for top in ("has_figure", "solvedAnswer"):
        if data.get(top) in (None, ""):
            miss.append(top)
    rt = data.get("richText") or {}
    for k in ("stem", "answer", "analysis"):
        if not (isinstance(rt, dict) and rt.get(k)):
            miss.append(f"richText.{k}")
    dna = data.get("dna") or {}
    for k in ("primaryKp", "secondaryKps", "qtype", "assessmentType", "solutionSkeleton",
              "hardPointCount", "scenario", "difficulty", "tags", "modelCandidates"):
        if not isinstance(dna, dict) or k not in dna:
            miss.append(f"dna.{k}")
    return miss


async def step_h1_cache(reps: int = 4) -> dict:
    print(f"\n=== H1 · opus 缓存：同稳定前缀连发 {reps} 次看 cached_tokens ===")
    prefix = _build_prefix()
    r = _main_relay()
    out = {"opus": OPUS, "prefix_chars": len(prefix), "hits": []}
    async with httpx.AsyncClient(timeout=600, trust_env=False) as client:
        for i in range(reps):
            payload = {
                "model": OPUS, "max_tokens": 256, "temperature": 0.1,
                "stream": True, "stream_options": {"include_usage": True},
                "messages": [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": f"只回复『确认第{i}发』四字，不要解释。"},
                ],
            }
            usage = None
            try:
                async with client.stream(
                    "POST", f"{r.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {r.api_key}"}, json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                            try:
                                obj = json.loads(line[6:])
                                if obj.get("usage"):
                                    usage = obj["usage"]
                            except Exception:
                                pass
                cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                cc5 = (usage or {}).get("claude_cache_creation_5_m_tokens")
                rec = {"rep": i, "prompt_tokens": (usage or {}).get("prompt_tokens"),
                       "cached_tokens": cached, "cache_creation_5m": cc5}
            except Exception as ex:  # noqa: BLE001
                rec = {"rep": i, "error": str(ex)[:160]}
            out["hits"].append(rec)
            print(json.dumps(rec, ensure_ascii=False))
            await asyncio.sleep(2)
    cached_vals = [h.get("cached_tokens", 0) for h in out["hits"][1:]]
    out["cache_hit"] = any((v or 0) > 0 for v in cached_vals)
    print(f">>> 缓存命中 = {out['cache_hit']}（后续发中 {sum(1 for v in cached_vals if (v or 0)>0)} 次 cached>0）")
    return out


GEO_PROMPT = """你是数学配图助手。把下面这道几何题翻译成一组 GeoGebra evalCommand 命令（每行一条，能被 GeoGebra Math Apps 执行渲染）。
要求：①用 Point/Segment/Polygon/Circle 等构造；②变换题的像用 Rotate/Reflect/Translate，像的边放进 dashed；③只输出 JSON：{"commands":["...","..."],"dashed":["对象名"],"vals":{"关键点":null}}，不要解释。
题目：在平面直角坐标系中，△ABC 的顶点 A(0,0)、B(4,0)、C(1,3)。将 △ABC 绕点 A 逆时针旋转 90°，得到 △AB'C'。画出原三角形与旋转后的三角形。"""


async def step_h3_geogebra() -> dict:
    print("\n=== H3 · opus 翻 GeoGebra 命令 → render 一轮直出 ===")
    out = {"opus": OPUS}
    chat = _chat(OPUS, 4000, temperature=0.2)
    t0 = time.monotonic()
    try:
        resp = await chat.ainvoke([HumanMessage(content=GEO_PROMPT)])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _strip_json(text)
        out["llm_dur_s"] = round(time.monotonic() - t0, 1)
        out["commands"] = (data or {}).get("commands")
        out["dashed"] = (data or {}).get("dashed")
        if not data or not data.get("commands"):
            out["error"] = "opus 未给出 commands：" + text[:160]
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return out
        # 渲染
        sys.path.insert(0, r"d:/workplace/book-ai/codeplace-O/math-figure-builder")
        os.environ.setdefault("MATHFIG_NODE_CWD", r"d:/workplace/book-ai/codeplace-O/book-test")
        from mathfig.geogebra import render_geogebra
        t1 = time.monotonic()
        r = render_geogebra(data["commands"], dashed=data.get("dashed"),
                            vals=data.get("vals"), axes=True, grid=False, mono=True)
        out["render_dur_s"] = round(time.monotonic() - t1, 1)
        out["render_ok"] = r.get("ok")
        out["n_cmd"] = r.get("n_cmd")
        out["n_fail"] = r.get("n_fail")
        out["png_path"] = r.get("png_path")
        out["png_size"] = os.path.getsize(r["png_path"]) if r.get("png_path") and os.path.exists(r["png_path"]) else 0
        out["warnings"] = r.get("warnings")
        out["failed_cmds"] = [c for c in (r.get("commands") or []) if not c.get("ok")]
    except Exception as ex:  # noqa: BLE001
        out["error"] = str(ex)[:240]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "h5", "h1", "h3"])
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    result = {"opus": OPUS, "relay": _main_relay().name, "relay_base": _main_relay().base_url}
    if args.step in ("all", "h5"):
        result["h5"] = await step_h5(args.max_tokens, args.reps)
    if args.step in ("all", "h1"):
        result["h1"] = await step_h1_cache()
    if args.step in ("all", "h3"):
        result["h3"] = await step_h3_geogebra()
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[结果落盘 {args.json}]")


if __name__ == "__main__":
    asyncio.run(main())
