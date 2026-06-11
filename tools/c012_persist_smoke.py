"""会话持久化 + 思维流式 + 富文本净化 + 身份硬闸 + 入库直连 · 服务级冒烟。

流程：
  A0  无 token 调 /variant/stream → 应被身份硬闸拦下（🔒 提示，零 LLM 调用）
  P1  POST /variant/stream 真跑一轮（带服务账号真 token）——顺带统计：
        - stage 帧（含「正在写第 n/N 道」generate 进度细化）
        - artifact 帧（题目文本应无字面 \\n、无 \\( \\) 定界 = 净化生效）
        - token 帧（JSON 中间产物已打 skip_stream → token 数应≈0；答疑轮才有）
  P2  POST /history {thread_id} → 应回放出 human+ai 消息（会话持久化 BE 半边）
  P3  POST /variant/artifact {thread_id} → 应重建出与 P1 相同题数的快照
  P4  对该 thread 追加一条「答疑」轮 → 应收到 token 帧（公开流式 = 打字机）
  P5  POST /variant/persist 直连入库（不过 LLM 分类器）→ ok + 回执 + 全题 persisted；
      无 token 调它 → 401

跑法: .venv/Scripts/python.exe tools/c012_persist_smoke.py
前置: toolkit :8093 + RuoYi :8090 在跑。⚠ P5 会真往服务账号题库落一组题（import_source=举一反三）。
"""

import asyncio
import json
import re
import sys
import uuid

import httpx

from _probe_auth import real_token

BASE = "http://localhost:8093"
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-23/cd2f5750-692b-411d-a335-895ccdf848b0/list/1/question.png"
)
THREAD = f"c012-smoke-{uuid.uuid4().hex[:8]}"
TOKEN: str = ""


async def stream_round(c: httpx.AsyncClient, message: str, *, with_token: bool = True):
    stages, artifacts, tokens, finals = [], [], 0, []
    body = {"message": message, "stream_tokens": True, "thread_id": THREAD}
    if with_token:
        body["agent_config"] = {"ruoyi_token": TOKEN}
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
    global TOKEN
    TOKEN = await real_token()
    ok = True
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as c:
        # A0 身份硬闸：无 token → 🔒 提示（不进任何 LLM 节点）
        _s0, _a0, _t0, finals0 = await stream_round(
            c, f"帮我对这道题举一反三：{IMG}", with_token=False
        )
        gate_hit = any("登录" in f for f in finals0)
        print(f"A0 无token硬闸: {'PASS' if gate_hit else 'FAIL'} "
              f"(stage={len(_s0)} token={_t0} reply={finals0[-1][:40] if finals0 else '∅'})")
        ok = ok and gate_hit and not _s0  # 不应有任何 stage 帧（零节点执行）

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
            # 🔴 与 variant._LITERAL_NL_RE 同口径：\n 后跟小写字母 = LaTeX 命令（\ne/\neq/\nabla/\not…）
            #   合法保留；只有 \n 后非小写字母才算字面换行残留（2026-06-11 真机踩出 $m\ne 0$ 误报）
            literal_nl = re.compile(r"\\n(?![a-z])")
            for it in artifacts[-1].get("items", []):
                for k in ("stem", "answer", "solution"):
                    v = str(it.get(k) or "")
                    if literal_nl.search(v):
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

        # P5 入库直连（不过 LLM 分类器）：无 token → 401；带 token → ok + 全题 persisted
        r = await c.post(f"{BASE}/variant/persist", json={"thread_id": THREAD, "ruoyi_token": "bad"})
        print(f"P5a 伪token直连入库 HTTP {r.status_code}（应 401）")
        if r.status_code != 401:
            ok = False
        r = await c.post(
            f"{BASE}/variant/persist", json={"thread_id": THREAD, "ruoyi_token": TOKEN}
        )
        if r.status_code != 200:
            print(f"P5b FAIL: HTTP {r.status_code} {r.text[:200]}")
            ok = False
        else:
            data = r.json()
            arts = data.get("artifact") or {}
            persisted = [it.get("persisted") for it in arts.get("items", [])]
            print(f"P5b 直连入库 ok={data.get('ok')} persisted={persisted} "
                  f"回执首行={str(data.get('reply') or '').splitlines()[0][:60]}")
            if not (data.get("ok") and persisted and all(persisted)):
                ok = False
            # 幂等：再点一次「全部入库」→ 应走防重分支（不重复落库）
            r2 = await c.post(
                f"{BASE}/variant/persist", json={"thread_id": THREAD, "ruoyi_token": TOKEN}
            )
            rep2 = (r2.json().get("reply") or "") if r2.status_code == 200 else ""
            dedup = "不会重复" in rep2 or "已入库" in rep2
            print(f"P5c 重复入库防重: {'PASS' if dedup else 'FAIL'} ({rep2[:50]})")
            ok = ok and dedup

    print("OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
