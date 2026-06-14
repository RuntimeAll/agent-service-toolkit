"""PRD-C-017 B0 预飞行探针：opus 读图 + 合并解题打标 + response_format + nano 判章/判带图。

只评测、不改默认值、不动 .env、不必起 :8093。直调中转池（aigeek 主 / lk888 备）。
凭据全部从 .env 读（core.settings），脚本零硬编码 key。

四步（--step 选，缺省全跑）：
  models   两站 GET /models，确认 claude-opus-4-8 在列（核 opus 真名，主备都核）。
  h1       opus 读图（image_url 多模态）一次合并输出「题面富文本 + 解答 + 10 维 DNA」(H1+H2)。
           同步测中转 response_format=json_schema 是否支持 (F3)：先试 schema 锁，失败退 prompt。
  m6       nano(gpt-5.4-nano) 判「年级册+章」准确率 + 判「题面是否含图形」准确率 (M6/G13 基础)。

跑法（cwd = toolkit 根）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/c017_b0_probe.py
  --step models|h1|m6   只跑某步
  --temp 0.1            opus 低温（M9 母题档低温）；缺省 0.1
  --json out.json       把结构化结果落盘（B1 复用 / 写报告）

判读：人读 JSON——h1 看 解答是否对、10 维齐全率、JSON 完整率、墙钟/token；
      m6 看 nano 判章命中、判带图命中。本脚本不下硬阈值（LLM 不确定）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from core import relay_pool  # noqa: E402
from core.settings import settings  # noqa: E402

OPUS = "claude-opus-4-8"
NANO = settings.LLM_MODEL_LIGHT  # gpt-5.4-nano（真名，.env 配）

# 三张真实母题图（复用 nano_vision_probe / c018_benchmark_replay_v2 的压轴真题 URL）。
# 注：本批未带「确定为纯文本」的母题样本——这些是题库压轴真题，含不含图由 opus/nano
# 在 has_figure 字段如实报告（H1 主测读图能力本身；纯文本图缺口由 B1 真机补，已在报告标注）。
IMAGES = [
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-26/06a3e548-e3d6-43ce-900f-c0d0e9cdd601/list/13/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
]

# 母题 opus 合并「解题 + 10 维 DNA 打标」prompt 雏形（按 artifacts/母题opus打标方法 §一）。
# 10 维 = primaryKp / secondaryKps / qtype / assessmentType / solutionSkeleton /
#        hardPointCount(+breakthroughPoints) / scenario / difficulty / tags / modelCandidates
# + has_figure（G13 带图判定钩子）+ richText（题面/答案/解析富文本）+ solvedAnswer（解答）。
MOTHER_SCHEMA = {
    "type": "object",
    "properties": {
        "has_figure": {"type": "boolean", "description": "题面是否含图形/图表/几何图（拍照纯文本题=false）"},
        "richText": {
            "type": "object",
            "properties": {
                "stem": {"type": "string"},
                "answer": {"type": "string"},
                "analysis": {"type": "string"},
            },
            "required": ["stem", "answer", "analysis"],
        },
        "solvedAnswer": {"type": "string", "description": "opus 真解出的最终答案"},
        "dna": {
            "type": "object",
            "properties": {
                "primaryKp": {"type": "string"},
                "secondaryKps": {"type": "array", "items": {"type": "string"}},
                "qtype": {"type": "string"},
                "assessmentType": {"type": "string"},
                "solutionSkeleton": {"type": "array", "items": {"type": "string"}},
                "hardPointCount": {"type": "integer"},
                "breakthroughPoints": {"type": "array", "items": {"type": "string"}},
                "scenario": {"type": "string"},
                "difficulty": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "modelCandidates": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "primaryKp", "secondaryKps", "qtype", "assessmentType",
                "solutionSkeleton", "hardPointCount", "breakthroughPoints",
                "scenario", "difficulty", "tags", "modelCandidates",
            ],
        },
    },
    "required": ["has_figure", "richText", "solvedAnswer", "dna"],
}

# 10 维齐全率考核：dna 下 10 个字段（has_figure/richText/solvedAnswer 单列）
DNA_10_DIMS = [
    "primaryKp", "secondaryKps", "qtype", "assessmentType", "solutionSkeleton",
    "hardPointCount", "scenario", "difficulty", "tags", "modelCandidates",
]

MOTHER_PROMPT = """你是浙教版初中数学命题专家。看这张题目图，**真正把题解出来**，然后做 10 维 DNA 打标。
🔴 先解题（一步步算到最终答案，不许抄图、不许跳步），再据你的解答打标。
🔴 难点克制：基础/纯套公式/直接计算/概念辨析题 hardPointCount 必为 0；hardPointCount 必须等于 breakthroughPoints 数组长度（不许自报数）。
🔴 难度四档：★压轴(≥2难点/多突破口综合)；★★★(1难点 或 考察∈{证明推理·应用建模·探究归纳} 或 骨架含最难步)；★★(常规无难点+考察∈{直接计算·公式套用·性质判定}+多步)；★(送分·无难点·单步或概念辨析)。
🔴 考察类型闭集10选1：概念辨析/直接计算/公式套用/性质判定/证明推理/应用建模/作图/探究归纳/阅读理解迁移/纠错。
🔴 has_figure：题面真含图形/图表/几何图填 true；只是拍照的纯文本题填 false。

只输出一个 JSON（不要解释、不要 markdown fence），结构：
{
  "has_figure": true/false,
  "richText": {"stem": "题干(Markdown+LaTeX)", "answer": "标准答案", "analysis": "解析"},
  "solvedAnswer": "你解出的最终答案",
  "dna": {
    "primaryKp": "主考点(≤16字)",
    "secondaryKps": ["副考点0~3个"],
    "qtype": "选择/填空/解答 之一",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["解法步骤1", "步骤2(最难一步用【】整步包住)"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["3~6个检索标签,禁近义增生"],
    "modelCandidates": ["真有可复用套路才给候选名,简单题空数组"]
  }
}"""

NANO_CHAPTER_PROMPT = """你是浙教版初中数学题库管理员。看这张题目图，判断它属于哪个**年级册 + 章**，以及题面**是否含图形**。
只输出一个 JSON（不要解释）：
{
  "grade_book": "年级册(如:八年级下册)",
  "chapter": "章名(如:第2章 一元二次方程)",
  "has_figure": true/false,
  "confidence": 0.0~1.0
}
🔴 has_figure：题面真含几何图/函数图象/统计图/图表填 true；纯文字拍照题填 false。"""


def _relays() -> list:
    return relay_pool._relays()


# --- step models ----------------------------------------------------------
def step_models() -> dict:
    print("=== STEP models · 两站 GET /models 核 opus 真名 ===")
    out: dict = {"sites": {}, "opus_available_on": []}
    for r in _relays():
        url = r.base_url.rstrip("/") + "/models"
        info: dict = {"base_url": r.base_url}
        try:
            resp = httpx.get(
                url, headers={"Authorization": f"Bearer {r.api_key}"},
                timeout=30, trust_env=False,
            )
            if resp.status_code != 200:
                info["error"] = f"HTTP {resp.status_code}: {resp.text[:120]}"
            else:
                data = resp.json()
                ids = [m.get("id", "") for m in data.get("data", data if isinstance(data, list) else [])]
                info["model_count"] = len(ids)
                info["opus_present"] = OPUS in ids
                info["opus_like"] = sorted([m for m in ids if "opus" in m.lower()])
                if OPUS in ids:
                    out["opus_available_on"].append(r.name)
        except Exception as ex:  # noqa: BLE001
            info["error"] = str(ex)[:160]
        out["sites"][r.name] = info
        print(json.dumps({r.name: info}, ensure_ascii=False, indent=2))
    print(f"\n>>> claude-opus-4-8 可用站点：{out['opus_available_on'] or '【无! go/no-go 红】'}")
    return out


def _chat_at(site: str, model: str, temperature: float) -> ChatOpenAI:
    for r in _relays():
        if r.name == site:
            return ChatOpenAI(
                model=model, temperature=temperature, streaming=False,
                openai_api_base=r.base_url, openai_api_key=r.api_key, timeout=300,
            )
    # 站名没匹配 → 用主站
    r = _relays()[0]
    return ChatOpenAI(
        model=model, temperature=temperature, streaming=False,
        openai_api_base=r.base_url, openai_api_key=r.api_key, timeout=300,
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


def _missing_dims(data) -> list[str]:
    """10 维 + 关键字段的齐全检查；返回缺失/空的维度名。"""
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
    for k in DNA_10_DIMS:
        v = dna.get(k) if isinstance(dna, dict) else None
        # 数组/对象类允许空（克制语义：secondaryKps/breakthrough/modelCandidates 可空），
        # 但字段必须存在；hardPointCount/difficulty 必须是数字。
        if k not in (dna if isinstance(dna, dict) else {}):
            miss.append(f"dna.{k}")
        elif k in ("primaryKp", "qtype", "assessmentType", "scenario") and not v:
            miss.append(f"dna.{k}(空)")
        elif k in ("hardPointCount", "difficulty") and not isinstance(v, int):
            miss.append(f"dna.{k}(非数字)")
    return miss


# --- step h1 (opus 读图 + 合并 + response_format) -------------------------
async def _opus_call(chat: ChatOpenAI, url: str, use_schema: bool) -> tuple[str, dict | None]:
    """一次 opus 多模态合并调用；use_schema=True 时带 response_format json_schema。"""
    msg = HumanMessage(content=[
        {"type": "text", "text": MOTHER_PROMPT},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    kwargs: dict = {"max_tokens": settings.VARIANT_MAX_TOKENS}
    if use_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "mother_label", "schema": MOTHER_SCHEMA, "strict": False},
        }
    resp = await chat.ainvoke([msg], **kwargs)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    usage = getattr(resp, "usage_metadata", None) or {}
    return text, usage


async def step_h1(site: str, temperature: float) -> dict:
    print(f"\n=== STEP h1 · opus={OPUS} @ {site} · temp={temperature} · 合并解题+10维打标 ===")
    out: dict = {"model": OPUS, "site": site, "temperature": temperature, "response_format": {}, "images": []}

    # 先一次性测中转支不支持 response_format json_schema（F3，go/no-go 关键）
    chat = _chat_at(site, OPUS, temperature)
    rf_supported = None
    rf_err = None
    print("--- F3：探测中转 response_format=json_schema 支持性 ---")
    try:
        text, _u = await _opus_call(chat, IMAGES[0], use_schema=True)
        # 能返回且能解析 = 支持
        data = _strip_json(text)
        rf_supported = data is not None
        if not rf_supported:
            rf_err = "返回非 JSON：" + text[:120]
    except Exception as ex:  # noqa: BLE001
        rf_supported = False
        rf_err = str(ex)[:200]
    out["response_format"]["supported"] = rf_supported
    out["response_format"]["error"] = rf_err
    print(f"response_format json_schema 支持 = {rf_supported}" + (f"  ({rf_err})" if rf_err else ""))

    # H1 主体：3 张图各跑一次合并调用（用 schema 若支持，否则纯 prompt 兜底）
    use_schema = bool(rf_supported)
    print(f"--- H1：3 张图合并调用（use_schema={use_schema}）---")
    full_ct = 0
    full_ok = 0
    for url in IMAGES:
        t0 = time.monotonic()
        rec: dict = {"url": url[-46:]}
        try:
            text, usage = await _opus_call(chat, url, use_schema=use_schema)
            dur = time.monotonic() - t0
            data = _strip_json(text)
            miss = _missing_dims(data)
            rec.update({
                "dur_s": round(dur, 1),
                "prompt_tokens": usage.get("input_tokens"),
                "completion_tokens": usage.get("output_tokens"),
                "json_complete": data is not None,
                "dims_missing": miss,
                "dims_full": data is not None and not miss,
                "has_figure": (data or {}).get("has_figure"),
                "solvedAnswer": ((data or {}).get("solvedAnswer") or "")[:120],
                "primaryKp": ((data or {}).get("dna") or {}).get("primaryKp"),
                "qtype": ((data or {}).get("dna") or {}).get("qtype"),
                "difficulty": ((data or {}).get("dna") or {}).get("difficulty"),
                "stem_head": (((data or {}).get("richText") or {}).get("stem") or "")[:160],
                "raw_head": None if data else text[:200],
            })
            full_ct += 1
            full_ok += int(rec["dims_full"])
        except Exception as ex:  # noqa: BLE001
            rec.update({"error": str(ex)[:200], "dur_s": round(time.monotonic() - t0, 1)})
            full_ct += 1
        out["images"].append(rec)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print("-" * 60)
    out["dims_full_rate"] = f"{full_ok}/{full_ct}"
    print(f">>> H1 10 维齐全率 = {full_ok}/{full_ct}")
    return out


# --- step m6 (nano 判章 + 判带图) -----------------------------------------
async def step_m6(site: str) -> dict:
    print(f"\n=== STEP m6 · nano={NANO} @ {site} · 判年级册+章 + 判带图 ===")
    out: dict = {"model": NANO, "site": site, "images": []}
    chat = _chat_at(site, NANO, 0.5)
    for url in IMAGES:
        msg = HumanMessage(content=[
            {"type": "text", "text": NANO_CHAPTER_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ])
        t0 = time.monotonic()
        rec: dict = {"url": url[-46:]}
        try:
            resp = await chat.ainvoke([msg], max_tokens=settings.VARIANT_MAX_TOKENS)
            dur = time.monotonic() - t0
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _strip_json(text)
            rec.update({
                "dur_s": round(dur, 1),
                "parsed_ok": data is not None,
                "grade_book": (data or {}).get("grade_book"),
                "chapter": (data or {}).get("chapter"),
                "has_figure": (data or {}).get("has_figure"),
                "confidence": (data or {}).get("confidence"),
                "raw_head": None if data else text[:160],
            })
        except Exception as ex:  # noqa: BLE001
            rec.update({"error": str(ex)[:200], "dur_s": round(time.monotonic() - t0, 1)})
        out["images"].append(rec)
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        print("-" * 60)
    print(">>> M6 判章/判带图为人工对照（脚本不裁决）——人读上面 grade_book/chapter/has_figure 与图核对")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["all", "models", "h1", "m6"])
    ap.add_argument("--temp", type=float, default=0.1, help="opus 低温（M9 母题档）")
    ap.add_argument("--json", default="", help="结果落盘路径")
    args = ap.parse_args()

    result: dict = {"opus_name": OPUS, "nano_name": NANO}
    site = _relays()[0].name  # 主站（aigeek）

    if args.step in ("all", "models"):
        result["models"] = step_models()
        # 若 opus 不在主站、在备站，h1/m6 走可用站
        avail = result["models"].get("opus_available_on") or []
        if avail and site not in avail:
            site = avail[0]
            print(f"\n[opus 不在主站 → h1 改走 {site}]")

    if args.step in ("all", "h1"):
        result["h1"] = await step_h1(site, args.temp)
    if args.step in ("all", "m6"):
        result["m6"] = await step_m6(site)

    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[结果已落盘 {args.json}]")


if __name__ == "__main__":
    asyncio.run(main())
