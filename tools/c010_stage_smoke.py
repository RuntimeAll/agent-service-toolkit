"""PRD-C-010 思维外放 · 服务级 SSE 冒烟：数 stage 帧。

经 :8093 /variant/stream 真跑一轮，统计 type=custom 且 custom_data.stage 的帧——
证明思路条事件真的从 BE 节点流到了线缆上（FE 渲染逻辑由 vue-tsc+review 把关）。

跑法: .venv/Scripts/python.exe tools/c010_stage_smoke.py
"""

import asyncio
import json
import sys

import httpx

URL = "http://localhost:8093/variant/stream"
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)


async def main() -> int:
    body = {"message": f"帮我对这道题举一反三：{IMG}", "stream_tokens": True,
            "thread_id": "stage-smoke-001"}
    stages: list[str] = []
    artifacts: list[dict] = []
    final = ""
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as c:
        async with c.stream("POST", URL, json=body) as r:
            print("HTTP", r.status_code)
            if r.status_code != 200:
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
                        cd = m.get("custom_data") or {}
                        st = cd.get("stage") or {}
                        if st.get("key"):
                            stages.append(f"{st['key']}:{st.get('status')}({st.get('detail') or ''})")
                        art = cd.get("artifact") or {}
                        if art.get("items"):
                            artifacts.append(art)
                    elif m.get("type") == "ai" and m.get("content"):
                        final = m["content"]

    print(f"stage 帧 {len(stages)} 条:")
    for s in stages:
        print("  ", s)
    keys = {s.split(":")[0] for s in stages}
    need = {"analyze", "classify", "generate", "gene_gate", "verify"}
    print(f"覆盖关键节点 {sorted(keys)} (需含 {sorted(need)})")
    # PRD-C-011: artifact 快照帧（右栏卡片栅数据源）
    print(f"artifact 帧 {len(artifacts)} 条")
    art_ok = bool(artifacts)
    if artifacts:
        last = artifacts[-1]
        for it in last.get("items", []):
            print(f"   #{it.get('index')} {it.get('qtype')} 难度{it.get('difficulty')} "
                  f"verify={it.get('verify')} gene={it.get('gene')} persisted={it.get('persisted')}")
        print(f"   header={last.get('header')}")
    ok = need.issubset(keys) and bool(final) and art_ok
    print("OK" if ok else "MISSING STAGES/ARTIFACT OR NO FINAL MESSAGE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
