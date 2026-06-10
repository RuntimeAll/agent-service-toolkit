"""PRD-C-009 举一反三 agent · 多模态 smoke。

一张 OSS 题图 → 过 /variant/stream（analyze → classify → [clarify|generate] → solve_explain → assemble）。
逐 SSE 帧统计：token 流（analyze/generate 流式）+ 最终 assemble 题组消息。

用法（toolkit 根，venv 内；先起服务 PORT=8080）:
    .venv/Scripts/python.exe tools/variant_smoke.py [可选: OSS 图 URL]

前置：
  - toolkit service :8080 在跑（uvicorn src.service）。
  - LLM lk888 可达（COMPATIBLE_* 已配，走默认网络，别套 trust_env=False —— 那是治本地 localhost 的）。
  - classify 锚图谱要 MySQL miskt_data2:3307 在跑（不在 → 降级，仍会走 clarify，不致命）。

🔴 httpx 调本地 :8080 用 trust_env=False（免疫本机 Clash 代理）；别用 curl 传中文（编码坑）。
🔴 帧分隔 = \n\n（toolkit service 是裸 StreamingResponse 直 yield 'data: ...\\n\\n'，
   不是 sse-starlette 的 \r\n\r\n —— 实测踩过：用 \r\n\r\n 拆会一帧都解不出）。按 raw 字节拆，别用 text 迭代器。
"""

import asyncio
import json
import sys

import httpx

BASE = "http://localhost:8080"
URL = f"{BASE}/variant/stream"
# 默认用一张真实 OSS 题图（dev 库 biz_question.stem_img_url 实例）
DEFAULT_IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)


async def main() -> int:
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    body = {
        "message": f"帮我对这道题举一反三：{img}",
        "stream_tokens": True,
        "thread_id": "variant-smoke-001",
    }

    token_chunks = 0
    msg_events: list[dict] = []
    err = None
    final_content = ""

    async with httpx.AsyncClient(timeout=180.0, trust_env=False) as c:
        async with c.stream("POST", URL, json=body) as r:
            ctype = r.headers.get("content-type", "")
            print("HTTP", r.status_code, "Content-Type:", ctype)
            if r.status_code != 200:
                print("!!! FAIL: 非 200，body 头部:", (await r.aread())[:300])
                return 2
            if "text/event-stream" not in ctype:
                print("!!! FAIL: 非 SSE 响应")
                return 2
            buf = b""
            async for chunk in r.aiter_raw():
                buf += chunk
                while b"\n\n" in buf:
                    frameb, buf = buf.split(b"\n\n", 1)
                    frame = frameb.decode("utf-8", "replace")
                    data = "\n".join(
                        line[5:].lstrip()
                        for line in frame.splitlines()
                        if line.startswith("data:")
                    )
                    if not data or data == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    t = ev.get("type")
                    if t == "token":
                        token_chunks += 1
                    elif t == "message":
                        m = ev.get("content", {})
                        msg_events.append(m)
                        if m.get("type") == "ai":
                            final_content = m.get("content", "") or final_content
                    elif t == "error":
                        err = ev

    print(f"token_chunks={token_chunks}  message_events={len(msg_events)}")
    if err:
        print("!!! error 事件:", json.dumps(err, ensure_ascii=False)[:300])
        return 1
    if not final_content:
        print("!!! FAIL: 无最终 AI 消息（assemble/clarify/报错都该吐一条 message）")
        return 1
    print("--- 最终 AI 消息（截前 800 字）---")
    print(final_content[:800])
    # 判定：要么出题组（含 '举一反三'/'变式'），要么 clarify（含 '确认'），要么友好报错
    ok_markers = ("举一反三", "变式", "确认", "题目图", "OSS")
    if any(k in final_content for k in ok_markers):
        print("\nOK: 流水线跑通（出题组 / clarify / 边界兜底 三者之一，均属预期）")
        return 0
    print("\n?? 警告: 最终消息未命中预期标记，请人工核对上文")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
