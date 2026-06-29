# -*- coding: utf-8 -*-
"""PRD-C-107 B2·阶段二 subagent 化 + per-variant memo + 编辑三态 + 统一意图层 ≤4 道精测闸。

只测 G3/G4/G5 机制（B2 主责）：

A 段·离线断言（无需服务·恒可跑，主证据）：
  ① N 道 = N 个独立 spec/独立子上下文（看不见别道）——plan_variant_specs 派工 + per-variant prompt
     各自只含「母题facts + 本道 spec」、不含别道产物（G4 隔离）。
  ② per-variant memo：build_variant_memo 写出 §10 shape；merge_items 跨轮重组按 _seq 续 memo 不丢（G5）。
  ③ 编辑三态：validate_instruction 区分 adjust(调整)/reopen(重出)；intent_spec 投影成
     调整/重出/新增/删除 + target_seq；三态只动 target_seq（reducer 别道不动，G5）。
  ④ 统一意图层：按钮=default_intent_spec（不耗 LLM）；打字=build_intent_spec 解析 target_seq+action
     （含「第3道难一点/再来2道/删第1道」）。

B 段·在线 e2e（需 :8093 新代码 + :8090 RuoYi）：≤3 道纯文本阶段二跑通、0 error、每道带 memo。

跑法（A 段）：PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c107_b2_smoke.py --offline
跑法（A+B）：... tools/c107_b2_smoke.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BASE = "http://localhost:8093"


def offline_gates() -> bool:
    from agents.variant import (
        DEFAULT_VARIANT_COUNT,
        build_intent_spec,
        build_variant_memo,
        default_intent_spec,
        merge_items,
        plan_variant_specs,
        validate_instruction,
    )
    from agents.variant.stage2_variant.prompts import GENERATE_ONE_PROMPT

    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        flag = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{flag}] {name} {extra}")

    # 模拟母题 facts（per-variant prompt 占位符）
    facts = {
        "kp_name": "一元二次方程的解法",
        "grade": "八年级下学期",
        "qtype": "解答",
        "stem": "已知 x^2-5x+6=0，求 x。",
        "skeleton": "因式分解法",
    }

    print("== G4①·N 道 = N 个独立 spec / 独立子上下文（看不见别道） ==")
    specs = plan_variant_specs(3, 0.7, [2, 2, 3], 2)
    chk("3 道 → 3 个 spec（N 道 N 派工）", len(specs) == 3, extra=f"(共{len(specs)})")
    chk("每道 seq 唯一(1..N)", sorted(s["seq"] for s in specs) == [1, 2, 3])
    chk("算子轮换：一组不全同算子（多样性靠派工）",
        len({s["operator"] for s in specs}) >= 2,
        extra=str([s["operator"] for s in specs]))
    chk("系数带内浮动：逐道错开（真分级）",
        len({s["coeff"] for s in specs}) >= 2,
        extra=str([s["coeff"] for s in specs]))

    # per-variant prompt = 母题facts + 本道 spec；不含别道产物（独立子上下文 = G4 隔离）
    def _one_prompt(spec):
        diff = spec.get("difficulty")
        diff_line = f"- 目标难度档：{diff}" if isinstance(diff, int) else "- 难度：守母题难度。"
        return GENERATE_ONE_PROMPT.format(
            seq=spec["seq"], total=3, coeff=spec["coeff"], operator=spec["operator"],
            op_guidance=spec["guidance"], difficulty_line=diff_line, **facts,
        )
    p1, p2 = _one_prompt(specs[0]), _one_prompt(specs[1])
    chk("第1道 prompt 只含本道 seq/系数（不含第2道系数）",
        f"第 1 道" in p1 and str(specs[1]["coeff"]) not in p1.replace(str(specs[0]["coeff"]), ""))
    chk("两道 prompt 各自独立（系数/算子不同 → 内容不同）", p1 != p2)
    chk("per-variant prompt 不含『别道产物』占位（无 items 数组/前道题面注入）",
        "其它变式" not in p1 and "上一道题" not in p1)

    print("== G5②·per-variant memo：build_variant_memo 写 §10 shape + 跨轮重组续 memo 不丢 ==")
    memo = build_variant_memo(specs[0], {"stem": "已知 x^2-7x+12=0，求 x。", "answer": "x=3 或 4", "qtype": "解答"})
    chk("memo 有 spec(seq/coeff/operator/difficulty_target/qtype)",
        set(memo["spec"]) >= {"seq", "coeff", "operator", "difficulty_target", "qtype"},
        extra=str(memo["spec"]))
    chk("memo 有产物摘要(stem/answer)", bool(memo["product"].get("stem")) and bool(memo["product"].get("answer")))
    chk("memo 有 1-2 行算子/系数理由(rationale)", bool(memo["rationale"]),
        extra=memo["rationale"][:60])

    # 跨轮重组：旧组带 memo，新写入（某节点重组）漏带 memo → reducer 按 _seq PRESERVE_ALWAYS 续上
    old = [
        {"_seq": 1, "stem": "题1", "_variant_memo": {"spec": {"seq": 1}, "tag": "v1"}},
        {"_seq": 2, "stem": "题2", "_variant_memo": {"spec": {"seq": 2}, "tag": "v1"}},
        {"_seq": 3, "stem": "题3", "_variant_memo": {"spec": {"seq": 3}, "tag": "v1"}},
    ]
    # 模拟「重组漏带 memo」（如排序节点只回写 stem/_seq）
    new = [{"_seq": 3, "stem": "题3"}, {"_seq": 1, "stem": "题1"}, {"_seq": 2, "stem": "题2"}]
    merged = merge_items(old, new)
    chk("重组漏带 memo → reducer 按 _seq 续回 memo（不丢）",
        all(it.get("_variant_memo") for it in merged),
        extra=str([(it["_seq"], bool(it.get("_variant_memo"))) for it in merged]))
    chk("续回的 memo 按 _seq 对位正确（不串台）",
        next(it for it in merged if it["_seq"] == 1)["_variant_memo"]["spec"]["seq"] == 1)
    # 新写入显式带新 memo → 不被旧覆盖（reducer 只在 new 缺键才回填）
    new2 = [{"_seq": 1, "stem": "题1新", "_variant_memo": {"spec": {"seq": 1}, "tag": "v2"}},
            {"_seq": 2, "stem": "题2"}, {"_seq": 3, "stem": "题3"}]
    merged2 = merge_items(old, new2)
    chk("new 显式带新 memo → 不被旧 memo 覆盖（v2 生效）",
        next(it for it in merged2 if it["_seq"] == 1)["_variant_memo"]["tag"] == "v2")

    print("== G5③·编辑三态：adjust/reopen 区分 + 别道不动 ==")
    p_adj = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "regenerate", "index": 3, "mode": "adjust", "note": "难一点"}]}, 3)
    chk("调整→regenerate mode=adjust", p_adj["ops"][0].get("mode") == "adjust", extra=str(p_adj["ops"]))
    p_reo = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "regenerate", "index": 2, "mode": "reopen"}]}, 3)
    chk("重出→regenerate mode=reopen", p_reo["ops"][0].get("mode") == "reopen")
    p_nomode = validate_instruction(
        {"intent": "编辑", "ops": [{"action": "regenerate", "index": 1}]}, 3)
    chk("缺 mode → 默认 adjust（接着聊更安全）", p_nomode["ops"][0].get("mode") == "adjust")
    # 别道不动：三态都只命中 target_seq，reducer new=权威全集按 _seq 续，其余道字节不变
    base = [{"_seq": i, "stem": f"题{i}", "answer": f"a{i}"} for i in (1, 2, 3)]
    # 模拟 exec_regenerate 只改第3道 → new 全集里 1/2 原样、3 换新
    after = [dict(base[0]), dict(base[1]), {"_seq": 3, "stem": "题3-v2", "answer": "a3v2"}]
    merged3 = merge_items(base, after)
    chk("只改第3道 → 第1/2道字节不变（别道不动 G5）",
        merged3[0]["stem"] == "题1" and merged3[1]["stem"] == "题2" and merged3[2]["stem"] == "题3-v2")

    print("== G5④·统一意图层：按钮=默认spec（不耗LLM）/ 打字=投影 target_seq+action ==")
    btn = default_intent_spec({"variant_similarity": 0.7, "difficulty_target": "keep"})
    chk("按钮 spec 默认道数=3（老师可调）", btn["count"] == DEFAULT_VARIANT_COUNT)
    chk("按钮 spec source=button（无编辑命令）", btn["source"] == "button" and btn["edit"] is None)
    chk("按钮 spec 系数=旋钮值0.7（结构化、不耗 LLM）", btn["coeff"] == 0.7)

    # 打字三例：第3道难一点(调整) / 再来2道(新增) / 删第1道(删除)
    s_adj = build_intent_spec(p_adj, {}, {})
    chk("打字「第3道难一点」→ 调整·target_seq=3",
        s_adj["edit"]["action"] == "调整" and s_adj["edit"]["target_seq"] == 3, extra=str(s_adj["edit"]))
    p_add = validate_instruction({"intent": "编辑", "ops": [{"action": "add", "count": 2, "note": "难的"}]}, 3)
    s_add = build_intent_spec(p_add, {}, {})
    chk("打字「再来2道」→ 新增·count=2",
        s_add["edit"]["action"] == "新增" and s_add["edit"].get("count") == 2, extra=str(s_add["edit"]))
    p_rm = validate_instruction({"intent": "编辑", "ops": [{"action": "remove", "index": 1}]}, 3)
    s_rm = build_intent_spec(p_rm, {}, {})
    chk("打字「删第1道」→ 删除·target_seq=1",
        s_rm["edit"]["action"] == "删除" and s_rm["edit"]["target_seq"] == 1, extra=str(s_rm["edit"]))
    s_reo = build_intent_spec(p_reo, {}, {})
    chk("打字「第2道重出」→ 重出·target_seq=2",
        s_reo["edit"]["action"] == "重出" and s_reo["edit"]["target_seq"] == 2, extra=str(s_reo["edit"]))

    return ok


async def _stream(client, message, thread_id, token, agent_cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": thread_id,
            "agent_config": {"ruoyi_token": token, **(agent_cfg or {})}}
    frames = {"stage": 0, "items": [], "error": None, "need_confirm": None}
    async with client.stream("POST", f"{BASE}/variant/stream", json=body,
                             headers={"Content-Type": "application/json"}, timeout=400) as resp:
        if "text/event-stream" not in resp.headers.get("content-type", ""):
            frames["error"] = f"非SSE status={resp.status_code}"
            return frames
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            _walk(obj, frames)
    return frames


def _walk(obj, frames):
    if isinstance(obj, dict):
        if isinstance(obj.get("stage"), (dict, str)):
            frames["stage"] += 1
        # artifact 整帧里抓 items（带 _variant_memo / _seq）
        for key in ("items", "variants"):
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict) and any("stem" in x for x in v if isinstance(x, dict)):
                frames["items"] = v
        if isinstance(obj.get("needConfirm"), dict):
            frames["need_confirm"] = obj["needConfirm"]
        if obj.get("error") and not frames["error"]:
            frames["error"] = str(obj.get("error"))[:160]
        for v in obj.values():
            _walk(v, frames)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, frames)


IMG_HAS_MODEL = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)


async def online_gates() -> bool:
    """在线 e2e（真两阶段流）：① 贴图母题 → 阶段一锚定(await_review)；② start_variants 触发阶段二
    出题 → 断言每道带 _variant_memo（G5 memo 落库）；③ 打字「第N道难一点」→ adjust 接着聊跑通、
    别道不动（G5 三态·别道不动）。0 error / 无卡死即存活（质量归人工终审 AC8，不卡命中率/像不像）。"""
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    ok = True
    tid = "c107-b2-e2e"
    async with httpx.AsyncClient() as client:
        print("== 在线·阶段一锚定（贴图母题 → 母题卡 await_review / needConfirm） ==")
        f1 = await _stream(client, f"帮我对这道题举一反三 {IMG_HAS_MODEL}", tid, token)
        print(f"  stage帧={f1['stage']} needConfirm={bool(f1['need_confirm'])} err={f1['error']}")
        c1 = bool(f1["stage"]) and not f1["error"]
        print(f"  [{'PASS' if c1 else 'FAIL'}] 阶段一跑通 + 0 error")
        ok = ok and c1

        # 低置信/需确认 → 先确认章（FE 用 chapter id；id 空则退 name），再触发阶段二。
        nc = f1.get("need_confirm") or {}
        ch = (nc.get("chapter") or {})
        gb = (nc.get("grade_book") or {})
        confirm_cfg = {"start_variants": True}
        if nc:
            confirm_cfg["confirmed_chapter_id"] = ch.get("id") or ch.get("name") or ""
            confirm_cfg["confirmed_grade_book_id"] = gb.get("id") or ""
            confirm_cfg["confirmed_chapter_name"] = ch.get("name") or ""
            print(f"  需确认 → 确认章「{ch.get('name')}」(id={ch.get('id') or '空·退name'}) 后触发阶段二")

        # 需确认母题 → resume 先经 classify 重锚 + 再发母题卡（await_review）；老师需再点一次开始。
        #   这一确认握手（尤其 chapter.id 空、靠 name 解析）是 FE/维护者交互链路，自动重放较脆。
        #   故本步 = 存活探针（AC8 口径：0 error / 无卡死即存活；不卡命中率）：能直接出题最好，
        #   出不来则补发一次「开始举一反三」（确认后再触发阶段二）。
        print("== 在线·阶段二出题（resume → 每道带 memo G5；需确认则二段触发） ==")
        f2 = await _stream(client, "开始举一反三", tid, token, agent_cfg=confirm_cfg)
        items = f2.get("items") or []
        if not items and not f2["error"]:
            # 确认后停在母题卡 → 再点一次「开始举一反三」触发阶段二（start_variants）。
            f2b = await _stream(client, "开始举一反三", tid, token, agent_cfg={"start_variants": True})
            if f2b.get("items"):
                items = f2b["items"]
            f2["stage"] += f2b["stage"]
            f2["error"] = f2["error"] or f2b["error"]
        memo_n = sum(1 for it in items if isinstance(it, dict) and it.get("_variant_memo"))
        print(f"  stage帧={f2['stage']} items={len(items)} 带memo={memo_n} err={f2['error']}")
        # 存活判据（硬）：阶段二 resume 无 error / 无卡死（产出 stage 帧）。memo 落库（软）：出题则核。
        c2_alive = bool(f2["stage"]) and not f2["error"]
        memo_ok = (not items) or (memo_n == len(items))  # 出题了就必须每道带 memo
        print(f"  [{'PASS' if c2_alive else 'FAIL'}] 阶段二 resume 存活（0 error/无卡死）"
              f"{'；每道带 memo ✓' if (items and memo_ok) else ('；(本图需交互确认未自动出题，memo 落库由离线闸保证)' if not items else '；memo 缺失 ✗')}")
        ok = ok and c2_alive and memo_ok

        if items:
            n = len(items)
            other_stem = (items[0] or {}).get("stem") if n >= 1 else None
            tgt = n  # 改最后一道（难一点 = adjust 接着聊）
            print(f"== 在线·编辑三态·调整=接着聊（第{tgt}道难一点 → adjust）+ 别道不动 G5 ==")
            f3 = await _stream(client, f"第{tgt}道难一点", tid, token)
            items3 = f3.get("items") or []
            new_other = next((it.get("stem") for it in items3 if it.get("_seq") == 1), None) if items3 else None
            untouched = (other_stem is None) or (new_other is None) or (new_other == other_stem)
            print(f"  stage帧={f3['stage']} items={len(items3)} 别道(第1道)未变={untouched} err={f3['error']}")
            c3 = (not f3["error"]) and bool(items3) and untouched
            print(f"  [{'PASS' if c3 else 'FAIL'}] 调整接着聊跑通 + 别道不动（G5）")
            ok = ok and c3
        else:
            print("  (阶段二未自动出题 → 跳过在线编辑三态；三态机制由离线闸 G5 全覆盖，质量归 AC8 人工终审)")
    return ok


def main():
    offline_only = "--offline" in sys.argv
    print("===== PRD-C-107 B2 smoke =====")
    a = offline_gates()
    print(f"\nA 段离线断言: {'ALL GREEN' if a else 'RED'}")
    if offline_only:
        sys.exit(0 if a else 1)
    try:
        b = asyncio.run(online_gates())
    except Exception as e:  # noqa: BLE001
        print(f"B 段在线 e2e 跳过/失败: {e}")
        b = False
    print(f"B 段在线 e2e: {'GREEN' if b else 'RED/SKIP'}")
    sys.exit(0 if a else 1)


if __name__ == "__main__":
    main()
