"""deepseek-v4-flash 评测探针（只评测，不改任何默认值 / 不动 .env / 不重启 :8093）。

任务：评估 deepseek 系模型在中转池（aigeek 主 / lk888 备）上能否用、值不值得替换管线某些环节。
红线：判决只读 sympy；评测脚本直调中转、不经服务；调用量控制在 ~15 次量级。

四步（用 --step 选，缺省全跑）：
  models   两站 GET /models，列出实际挂的 deepseek 系型号（名字 + 哪站可用）。
  vision   对可用 deepseek 跑 3 张真母题图视觉探针（复用 nano_vision_probe 的图 + 判定）。
  solve    闸B solve A/B：deepseek vs gpt-5.4 同母题，sympy 判决一致率 + 耗时。
  light    轻活抽取：一段母题文本跑 DNA 式结构化抽取，看 JSON 守不守 schema + 速度。

跑法（cwd = toolkit 根）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/deepseek_probe.py
  --model deepseek-v4-flash   指定要测的 deepseek 型号（默认按 models 步探到的第一个可用名）
  --step models|vision|solve|light   只跑某一步
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

from c018_benchmark_replay import MOTHERS  # noqa: E402
import agents.variant as V  # noqa: E402
from agents.variant import ANALYZE_PROMPT  # noqa: E402

# 候选 deepseek 系名（任务给定全试）
CANDIDATES = [
    "deepseek-v4-flash",
    "deepseek-v4",
    "deepseek-chat",
    "deepseek-v3.2",
    "deepseek-v3.1",
    "deepseek-v3",
    "deepseek-reasoner",
]

# 三张真实母题图（与 nano_vision_probe 同源）
IMAGES = [
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-26/06a3e548-e3d6-43ce-900f-c0d0e9cdd601/list/13/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
]

LIGHT_PROMPT = """你是浙教版初中数学命题专家。对下面这道题做**结构化 DNA 抽取**，只输出一个 JSON（不要任何解释、不要 markdown fence）：
{{
  "main_kp": "主知识点名（≤16字）",
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


def _relays() -> list:
    return relay_pool._relays()


# --- step: models ----------------------------------------------------------
def step_models() -> list[str]:
    print("=== STEP models · 两站 GET /models 列 deepseek 系 ===")
    available: dict[str, list[str]] = {}  # model -> [site,...]
    for r in _relays():
        url = r.base_url.rstrip("/") + "/models"
        ds_here: list[str] = []
        try:
            resp = httpx.get(
                url,
                headers={"Authorization": f"Bearer {r.api_key}"},
                timeout=30,
                trust_env=False,  # 绕本机代理
            )
            if resp.status_code != 200:
                print(f"[{r.name}] GET /models -> HTTP {resp.status_code}: {resp.text[:150]}")
            else:
                data = resp.json()
                ids = [m.get("id", "") for m in data.get("data", data if isinstance(data, list) else [])]
                ds_here = sorted([m for m in ids if "deepseek" in m.lower()])
                print(f"[{r.name}] 共 {len(ids)} 模型，deepseek 系 {len(ds_here)} 个：{ds_here}")
        except Exception as ex:  # noqa: BLE001
            print(f"[{r.name}] GET /models 失败：{ex}")
        for m in ds_here:
            available.setdefault(m, []).append(r.name)

    # 兜底：/models 拿不到时，对候选名发最小 chat 探测（每站每候选 1 次，控量）
    if not available:
        print("\n/models 未列出 deepseek，转最小 chat 探测候选名...")
        for r in _relays():
            for cand in CANDIDATES:
                ok = _probe_chat(r, cand)
                print(f"[{r.name}] chat probe {cand}: {'OK' if ok else 'X'}")
                if ok:
                    available.setdefault(cand, []).append(r.name)

    print("\n--- deepseek 可用型号 × 站点 ---")
    for m, sites in sorted(available.items()):
        print(f"  {m}: {', '.join(sites)}")
    return sorted(available.keys())


def _probe_chat(relay, model: str) -> bool:
    """对某站某型号发一次极小 chat 请求，看是否 200（控量：仅 models 兜底时用）。"""
    try:
        resp = httpx.post(
            relay.base_url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {relay.api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4},
            timeout=30,
            trust_env=False,
        )
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _chat_for(model: str) -> ChatOpenAI:
    """挑第一个挂该模型的站点构 ChatOpenAI（评测直调，不走熔断池）。"""
    for r in _relays():
        return ChatOpenAI(
            model=model,
            temperature=0.5,
            streaming=False,
            openai_api_base=r.base_url,
            openai_api_key=r.api_key,
            timeout=240,
        )
    raise RuntimeError("no relay")


def _chat_at(site: str, model: str) -> ChatOpenAI:
    for r in _relays():
        if r.name == site:
            return ChatOpenAI(
                model=model, temperature=0.5, streaming=False,
                openai_api_base=r.base_url, openai_api_key=r.api_key, timeout=240,
            )
    return _chat_for(model)


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


# --- step: vision ----------------------------------------------------------
async def step_vision(model: str, site: str) -> None:
    print(f"\n=== STEP vision · model={model} @ {site} · 3 张真母题图 ===")
    chat = _chat_at(site, model)
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
                "url": url[-40:], "dur_s": round(dur, 1), "parsed_ok": data is not None,
                "is_question_image": (data or {}).get("is_question_image"),
                "grade": (data or {}).get("grade"), "kp": (data or {}).get("kp"),
                "stem": ((data or {}).get("stem") or "")[:120],
                "raw_head": None if data else text[:150],
            }
        except Exception as ex:  # noqa: BLE001
            out = {"url": url[-40:], "error": str(ex)[:200], "dur_s": round(time.monotonic() - t0, 1)}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print("-" * 50)


# --- step: solve (A/B sympy) ----------------------------------------------
async def _solve_verify(item: dict, model: str) -> dict:
    old = settings.VARIANT_MODEL_SOLVE
    settings.VARIANT_MODEL_SOLVE = model
    try:
        t0 = time.monotonic()
        solved = await V._solve_one(item.get("stem", ""))
        res = await V._machine_verify(
            {"stem": item.get("stem"), "answer": item.get("answer"), "qtype": item.get("qtype")},
            solved.get("solved_answer"),
        )
        return {"verdict": res.get("verdict"),
                "solved": (solved.get("solved_answer") or "")[:60],
                "dur_s": round(time.monotonic() - t0, 1)}
    finally:
        settings.VARIANT_MODEL_SOLVE = old


async def step_solve(model: str, limit: int) -> None:
    print(f"\n=== STEP solve A/B · {model} vs gpt-5.4 · sympy 判决 ===")
    mothers = [m for m in MOTHERS if m.get("qtype") not in ("证明", "作图")]
    if limit:
        mothers = mothers[:limit]
    agree = ds_warn = f54_warn = 0
    for m in mothers:
        ds = await _solve_verify(m, model)
        f54 = await _solve_verify(m, "gpt-5.4")
        same = ds["verdict"] == f54["verdict"]
        agree += int(same)
        ds_warn += int(ds["verdict"] == V.math_verify.DEGRADE)
        f54_warn += int(f54["verdict"] == V.math_verify.DEGRADE)
        print(json.dumps({
            "stem": (m.get("stem") or "")[:46], "ds": ds["verdict"], "5.4": f54["verdict"],
            "agree": same, "ds_s": ds["dur_s"], "54_s": f54["dur_s"],
        }, ensure_ascii=False))
    n = len(mothers)
    print("=" * 50)
    print(f"题数 {n} · 一致 {agree}/{n} = {agree / n * 100:.1f}% （阈值≥90%）")
    print(f"deepseek degrade {ds_warn}/{n}  vs  5.4 degrade {f54_warn}/{n}")


# --- step: light (DNA extraction) -----------------------------------------
async def step_light(model: str, site: str) -> None:
    print(f"\n=== STEP light · {model} @ {site} · DNA 式结构化抽取 ===")
    chat = _chat_at(site, model)
    # 取 2 道母题文本（一计算一证明），看 JSON schema 守不守 + 速度
    samples = [MOTHERS[1], MOTHERS[7]]
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
                "json": data if data else None, "raw_head": None if data else text[:150],
            }
        except Exception as ex:  # noqa: BLE001
            out = {"name": m.get("name"), "error": str(ex)[:200], "dur_s": round(time.monotonic() - t0, 1)}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print("-" * 50)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--step", default="all", choices=["all", "models", "vision", "solve", "light"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    found: list[str] = []
    site_of: dict[str, str] = {}
    if args.step in ("all", "models"):
        # step_models 还顺手记录每型号首个可用站
        found = step_models()
        for m in found:
            for r in _relays():
                # 用一次极轻 chat 不必再发——models 步已知挂载，site 取列表第一个
                pass
    model = args.model or (found[0] if found else "deepseek-v4-flash")

    # 推导测试用站点：优先 found 里该 model 的站；否则主站
    def site_for(mdl: str) -> str:
        # 复用 models 步的可用映射不易跨函数传，这里直接主站（aigeek）；
        # 若主站不挂该 model，调用时 ainvoke 会报错、out 里能看出来。
        return _relays()[0].name

    site = site_for(model)
    if args.step != "models":
        print(f"\n[测试型号 = {model} · 站点 = {site}]")

    if args.step in ("all", "vision"):
        await step_vision(model, site)
    if args.step in ("all", "solve"):
        await step_solve(model, args.limit)
    if args.step in ("all", "light"):
        await step_light(model, site)


if __name__ == "__main__":
    asyncio.run(main())
