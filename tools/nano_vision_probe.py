"""按环节分档模型路由·先验脚本：直调 gpt-5-nano 多模态读真实母题图，确认 nano 接图且抽取质量
可用（题干/年级/考点抽得出来、不胡编）。过 → ANALYZE 默认降 nano；不过 → 保持 gpt-5.4。

判读：人读结果 JSON，看 is_question_image / grade / kp / stem 是否合理且非空。本脚本零裁决。

跑法（cwd = toolkit 根）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/nano_vision_probe.py
  --model gpt-5.4   对照 5.4 同图（看差距）
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import ANALYZE_PROMPT  # noqa: E402
from core import relay_pool  # noqa: E402
from core.settings import settings  # noqa: E402

# 三张真实母题图（复用 c018_benchmark_replay_v2 的压轴真题 URL）
IMAGES = [
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-10-17/d247e21b-388d-41d0-a46a-de19e17a5f70/list/16/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2024-12-26/06a3e548-e3d6-43ce-900f-c0d0e9cdd601/list/13/question.png",
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png",
]


async def probe_one(url: str, model: str) -> dict:
    msg = HumanMessage(
        content=[
            {"type": "text", "text": ANALYZE_PROMPT.format(utterance="（无）")},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )
    t0 = time.monotonic()
    try:
        resp, relay, model_used, fb = await relay_pool.ainvoke_failover(
            [msg], max_tokens=settings.VARIANT_MAX_TOKENS, tags=["skip_stream"], model=model
        )
        dur = time.monotonic() - t0
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # 剥 fence + JSON 容错
        t = text.strip()
        if t.startswith("```"):
            t = t.split("```")[1] if "```" in t[3:] else t
            t = t.replace("json", "", 1).strip() if t.lstrip().startswith("json") else t
        s, e = t.find("{"), t.rfind("}")
        data = None
        if s >= 0 and e > s:
            try:
                data = json.loads(t[s : e + 1])
            except Exception:
                data = None
        return {
            "url": url,
            "model_used": model_used,
            "relay": relay,
            "dur_s": round(dur, 1),
            "parsed_ok": data is not None,
            "is_question_image": (data or {}).get("is_question_image"),
            "grade": (data or {}).get("grade"),
            "subject": (data or {}).get("subject"),
            "kp": (data or {}).get("kp"),
            "qtype": (data or {}).get("qtype"),
            "stem": ((data or {}).get("stem") or "")[:300],
            "answer": ((data or {}).get("answer") or "")[:120],
            "raw_head": text[:200] if data is None else None,
        }
    except Exception as ex:  # noqa: BLE001
        return {"url": url, "model_used": model, "error": str(ex), "dur_s": round(time.monotonic() - t0, 1)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-nano")
    args = ap.parse_args()
    print(f"=== nano 视觉先验 · model={args.model} ===")
    for url in IMAGES:
        r = await probe_one(url, args.model)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
