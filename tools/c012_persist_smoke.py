"""会话持久化 + 思维流式 + 富文本净化 · 服务级冒烟（2026-06-11 四项反馈批次）。

流程：
  P1  POST /variant/stream 真跑一轮（thread_id 固定）——顺带统计：
        - stage 帧（含新「正在写第 n/N 道」generate 进度细化）
        - artifact 帧（题目文本应无字面 \\n、无 \\( \\) 定界 = 净化生效）
        - token 帧（JSON 中间产物已打 skip_stream → token 数应≈0；答疑轮才有）
  P2  POST /history {thread_id} → 应回放出 human+ai 消息（会话持久化 BE 半边）
  P3  POST /variant/artifact {thread_id} → 应重建出与 P1 相同题数的快照
  P4  对该 thread 追加一条「答疑」轮 → 应收到 token 帧（公开流式 = 打字机）

跑法: .venv/Scripts/python.exe tools/c012_persist_smoke.py
前置: toolkit :8093 + RuoYi :8090 在跑。
"""

import asyncio
import json
import sys

import httpx

BASE = "http://localhost:8093"
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)
THREAD = "c012-persist-smoke-001"


async def stream_round(c: httpx.AsyncClient, message: str):
    stages, artifacts, tokens, finals = [], [], 0, []
    body = {"message": message, "stream_tokens": True, "thread_id": THREAD}
    async with c.stream("POST", f"{BASE}/variant/stream", json=body) as r:
        assert r.status_code == 200, f"stream HTTP {r.status_code}"
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
                if ev.get("type") == "token":
                    tokens += 1
                    continue
                if ev.get("type") != "message":
                    continue
                m = ev.get("content") or {}
                if m.get("type") == "custom":
                    cd = m.get("custom_data") or {}
                    st = cd.get("stage") or {}
                    if st.get("key"):
                        stages.append(f"{st['key']}:{st.get('status')}({st.get('detail') or ''})")
                    art = cd.get("artifact") or {}
                    if art.get("items"):
                        artifacts.append(art)
                elif m.get("type") == "ai" and m.get("content"):
                    finals.append(m["content"])
    return stages, artifacts, tokens, finals


async def main() -> int:
    ok = True
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as c:
        # P1 出题轮
        stages, artifacts, tokens, finals = await stream_round(
            c, f"帮我对这道题举一反三：{IMG}"
        )
        prog = [s for s in stages if "正在写第" in s]
        print(f"P1 stage帧={len(stages)} 其中出题进度细化帧={len(prog)}")
        for s in prog:
            print("   ", s)
        print(f"P1 artifact帧={len(artifacts)} token帧={tokens}（JSON 中间产物应已静默）")
        if not artifacts:
            print("P1 FAIL: 无 artifact 帧")
            ok = False
        else:
            dirty = []
            for it in artifacts[-1].get("items", []):
                for k in ("stem", "answer", "solution"):
                    v = str(it.get(k) or "")
                    if "\\n" in v.replace("\\neq", "").replace("\\nabla", ""):
                        dirty.append(f"#{it.get('index')}.{k} 含字面\\n")
                    if "\\(" in v or "\\[" in v:
                        dirty.append(f"#{it.get('index')}.{k} 含 \\( \\[ 定界")
            print("P1 净化检查:", "PASS" if not dirty else f"FAIL {dirty}")
            ok = ok and not dirty
        if not prog:
            print("P1 WARN: 未见出题进度细化帧（chunk 较大可能一次跳满，非硬失败）")

        # P2 /history 回放
        r = await c.post(f"{BASE}/history", json={"thread_id": THREAD})
        msgs = (r.json() or {}).get("messages", []) if r.status_code == 200 else []
        kinds = [m.get("type") for m in msgs]
        print(f"P2 /history HTTP {r.status_code} 消息数={len(msgs)} 类型={kinds}")
        if r.status_code != 200 or "ai" not in kinds:
            print("P2 FAIL")
            ok = False

        # P3 /variant/artifact 重建
        r = await c.post(f"{BASE}/variant/artifact", json={"thread_id": THREAD})
        items = (r.json() or {}).get("items", []) if r.status_code == 200 else []
        print(f"P3 /variant/artifact HTTP {r.status_code} 重建题数={len(items)}")
        if r.status_code != 200 or len(items) == 0:
            print("P3 FAIL")
            ok = False
        # 对照流内最后一帧题数
        if artifacts and len(items) != len(artifacts[-1].get("items", [])):
            print("P3 FAIL: 重建题数与流内快照不一致")
            ok = False

        # P4 答疑轮 → 公开 token 流（打字机）
        stages2, _arts2, tokens2, finals2 = await stream_round(c, "第1题为什么这么解？")
        print(f"P4 答疑轮 token帧={tokens2}（应>0=打字机生效） 终稿气泡={len(finals2)}")
        if tokens2 <= 0:
            print("P4 FAIL: 答疑没有 token 流")
            ok = False

    print("OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
