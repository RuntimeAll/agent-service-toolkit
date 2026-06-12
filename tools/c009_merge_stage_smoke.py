"""PRD-C-009 整改 · 配方并入读图 + 两灯定时 冒烟（一次性）。

经 :8093 /variant/stream 真跑图片首轮，记录：
  - 每条 stage 帧的相对时间戳（验「锚定考点」首灯 ~10s 翻绿、「解析配方」次灯在锚定后翻绿）；
  - 首轮总耗时；
  - 之后由调用方查 data/llm_trace.jsonl 本轮 label 分布（独立 knobs 调用应=0 或仅兜底）。

跑法: .venv/Scripts/python.exe tools/c009_merge_stage_smoke.py
"""

import asyncio
import json
import sys
import time

import httpx

from _probe_auth import real_token

URL = "http://localhost:8093/variant/stream"
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)


async def main() -> int:
    body = {
        "message": f"帮我对这道题举一反三，出3道，难度递增：{IMG}",
        "stream_tokens": True,
        "thread_id": f"c009-merge-smoke-{int(time.time())}",
        "agent_config": {"ruoyi_token": await real_token()},
    }
    stages: list[tuple[float, str, str, str]] = []
    final = ""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as c:
        async with c.stream("POST", URL, json=body) as r:
            print("HTTP", r.status_code)
            if r.status_code != 200:
                print((await r.aread())[:300])
                return 2
            buf = b""
            async for chunk in r.aiter_raw():
                buf += chunk
                while b"\n\n" in buf:
                    frameb, buf = buf.split(b"\n\n", 1)
                    frame = frameb.decode("utf-8", "replace")
                    data = "\n".join(
                        ln[5:].lstrip() for ln in frame.splitlines() if ln.startswith("data:")
                    )
                    if not data or data == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get("type") != "message":
                        continue
                    m = ev.get("content") or {}
                    if m.get("type") == "custom":
                        st = (m.get("custom_data") or {}).get("stage") or {}
                        if st.get("key"):
                            stages.append(
                                (time.monotonic() - t0, st["key"], st.get("status"),
                                 st.get("detail") or "")
                            )
                    elif m.get("type") == "ai" and m.get("content"):
                        final = m["content"]
    total = time.monotonic() - t0
    print(f"\n首轮总耗时 = {total:.1f}s")
    print(f"stage 帧 {len(stages)} 条（含相对时间戳）:")
    for ts, key, status, detail in stages:
        print(f"  [{ts:6.1f}s] {key:10s} {status:8s} {detail}")
    # 关键序：锚定考点 done 的时间 vs 解析配方 done 的时间
    def done_t(key):
        return next((ts for ts, k, s, _ in stages if k == key and s == "done"), None)
    anchor_done = done_t("classify")
    knobs_done = done_t("knobs")
    print(f"\n锚定考点 done @ {anchor_done}  |  解析配方 done @ {knobs_done}")
    if anchor_done is not None and knobs_done is not None:
        print("时序断言: 锚定考点 done <= 解析配方 done ->",
              "OK" if anchor_done <= knobs_done + 0.05 else "FAIL")
    print(f"\n最终消息（前 400）:\n{final[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
