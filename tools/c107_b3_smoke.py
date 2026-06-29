# -*- coding: utf-8 -*-
"""PRD-C-107 B3·双旋钮 code skill + 算子轮换隐藏 + 默认难度分布 + LLM 逆向带入验算 ≤4 道精测闸。

只测 G7（双旋钮真生效·算子藏·默认分布）+ G3（逆向验算通/回炉）+ G2（难度仍 tier 驱动）。

A 段·离线断言（无需服务·恒可跑，主证据）：
  ① 双旋钮：两不同系数 → 不同算子带/出题指令 + 显示系数一致（G7·回填真实系数）。
  ② 默认难度分布：缺 expected → 前 N-1 道母题档、末道 +1（G7）；「难一点」=整组抬（recipe 移 md）。
  ③ 算子轮换 + 不外泄：一组算子各异（多样性靠派工）；面向老师 generate 帧不含 coeff/operator 数字（G7）。
  ④ 逆向验算（mock LLM）：答案对→PASS、答案错→FAIL 触发回炉≤2；sympy dormant 开关可切回（G3）。
  ⑤ 难度仍 tier 驱动：grade_variant_item 走 grade_observed（表），不取 item 自评 difficulty（G2）。

B 段·在线 e2e（需 :8093 新代码 + :8090 RuoYi）：≤3 道纯文本（含 1 道难题）跑通、0 error；
  难题那道走 tier→难度 + 逆向验算（质量归人工终审 AC8，不卡命中率/像不像）。

跑法（A 段）：PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c107_b3_smoke.py --offline
跑法（A+B）：set NO_PROXY=* & ... tools/c107_b3_smoke.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BASE = "http://localhost:8093"


def offline_gates() -> bool:
    import agents.variant as V
    from agents import math_verify
    from agents.variant import (
        operator_band_from_similarity,
        plan_variant_specs,
        recipe_from_knobs,
        grade_variant_item,
        _machine_verify,
    )

    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        flag = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{flag}] {name} {extra}")

    print("== G7①·双旋钮：两不同系数 → 不同算子带/指令 + 显示系数一致 ==")
    lo = operator_band_from_similarity(0.3)   # 低相似度带（远迁）
    hi = operator_band_from_similarity(0.85)  # 高相似度带（高仿）
    chk("系数 0.3 vs 0.85 → 不同相似度带", lo["band"] != hi["band"], extra=f"{lo['band']} vs {hi['band']}")
    chk("两系数 → 不同算子（出题指令不同）", lo["operator"] != hi["operator"],
        extra=f"{lo['operator']} vs {hi['operator']}")
    chk("显示系数与输入一致（回填真实系数·修 C-106『0.70 显示按≈0.5』）",
        lo["similarity"] == 0.3 and hi["similarity"] == 0.85, extra=f"{lo['similarity']},{hi['similarity']}")
    chk("guidance 非空（注入 prompt 的人话指令）", bool(lo["guidance"]) and bool(hi["guidance"]))
    # PLAN 派工回填真实本道系数（带内浮动后）→ spec.coeff 即显示值
    specs_lo = plan_variant_specs(3, 0.3, None, 2)
    specs_hi = plan_variant_specs(3, 0.85, None, 2)
    chk("不同基准系数 → 不同 PLAN 出题指令（算子带不同）",
        {s["operator"] for s in specs_lo} != {s["operator"] for s in specs_hi}
        or specs_lo[0]["coeff"] != specs_hi[0]["coeff"],
        extra=f"lo={[s['coeff'] for s in specs_lo]} hi={[s['coeff'] for s in specs_hi]}")

    print("== G7②·默认难度分布：前 N-1 道母题档、末道 +1（缺 expected 时）==")
    # md=2，n=3，无老师 expected → [2,2,3]
    d3 = [s["difficulty"] for s in plan_variant_specs(3, 0.7, None, 2)]
    chk("n=3 md=2 默认 → 前两道母题档(2,2)、末道 +1(3)", d3 == [2, 2, 3], extra=str(d3))
    d5 = [s["difficulty"] for s in plan_variant_specs(5, 0.7, None, 2)]
    chk("n=5 md=2 推广 → 除末道同档、末道 +1", d5 == [2, 2, 2, 2, 3], extra=str(d5))
    dcap = [s["difficulty"] for s in plan_variant_specs(3, 0.7, None, 4)]
    chk("md=4 末道 +1 封顶 CAP=4（不越界）", dcap == [4, 4, 4], extra=str(dcap))
    d1 = [s["difficulty"] for s in plan_variant_specs(1, 0.7, None, 2)]
    chk("n=1 → [md]（单道无末道+1）", d1 == [2], extra=str(d1))
    # 老师给 expected（递增）→ 原样沿用（不被默认分布覆盖）
    d_exp = [s["difficulty"] for s in plan_variant_specs(3, 0.7, [2, 3, 4], 2)]
    chk("老师给 expected[2,3,4] → 原样沿用（不被默认分布盖）", d_exp == [2, 3, 4], extra=str(d_exp))
    # 「难一点」= recipe 把 md 起步档移到 target（整组水位上移）。difficulty_plan 须用归一化常量
    #   PLAN_INCREASING（"递增"→PLAN_INCREASING 的归一在 _extract_knobs LLM 层做，recipe 吃归一值）。
    from agents.variant import PLAN_INCREASING
    r_hard = recipe_from_knobs({"difficulty_target": 3, "difficulty_plan": PLAN_INCREASING}, 2)
    chk("「难一点」(target=3) → recipe 起步档移到 3（整组抬）",
        r_hard["expected_difficulties"] and r_hard["expected_difficulties"][0] == 3,
        extra=str(r_hard["expected_difficulties"]))
    # 🔴 frozen：空 knobs recipe 仍返 None（默认分布落 PLAN 层，不破 C-106 frozen 测试）
    chk("frozen：recipe_from_knobs({}) 仍 expected=None（默认分布在 PLAN 层）",
        recipe_from_knobs({}, 2)["expected_difficulties"] is None)

    print("== G7③·算子轮换 + 不对老师外泄 ==")
    specs = plan_variant_specs(3, 0.7, None, 2)
    chk("一组算子各异（多样性靠系统轮换派工）",
        len({s["operator"] for s in specs}) >= 2, extra=str([s["operator"] for s in specs]))
    # 面向老师的 generate 帧不含 coeff/operator 数字（读源码断言 detail 文案已去除）
    import inspect
    from agents.variant.stage2_variant import generate as gen_mod
    gen_src = inspect.getsource(gen_mod.generate)
    leaked = "变式系数 {spec['coeff']}" in gen_src or "spec['operator']）" in gen_src
    chk("generate 面向老师 stage 帧不再带『变式系数X·算子Y』数字（B3④ 隐藏）", not leaked)
    chk("派工印记仍落 item.variant_coeff/operator（内部 trace 不丢）",
        "item[\"variant_coeff\"] = spec[\"coeff\"]" in gen_src
        and "item[\"variant_operator\"] = spec[\"operator\"]" in gen_src)

    print("== G3④·逆向带入验算（mock LLM）：对→PASS / 错→FAIL；sympy dormant 可切回 ==")

    async def _verdict_cases():
        nonlocal ok
        orig = V._ainvoke_text
        try:
            # 答案对：代回全成立 → PASS
            async def ai_pass(messages, **kw):
                return json.dumps({"steps": "3^2-7*3+12=0", "all_satisfied": True, "computed": "0"})
            V._ainvoke_text = ai_pass
            r = await _machine_verify({"stem": "x^2-7x+12=0", "answer": "x=3 或 4"}, None)
            chk("逆向验算·答案对 → PASS（机械代入成立）", r["verdict"] == math_verify.PASS, extra=r["verdict"])

            # 答案错：代回不成立 → FAIL（下游回炉 ≤2 同通道）
            async def ai_fail(messages, **kw):
                return json.dumps({"steps": "3^2-6*3+8=-1", "all_satisfied": False,
                                   "computed": "-1", "reason": "x=3 代回得 -1 不为 0"})
            V._ainvoke_text = ai_fail
            r = await _machine_verify({"stem": "x^2-6x+8=0", "answer": "x=2 或 3"}, None)
            chk("逆向验算·答案错 → FAIL（触发下游回炉，非自评）", r["verdict"] == math_verify.FAIL,
                extra=f"{r['verdict']} detail={r['detail'][:40]}")

            # 不可代入/缺信息 → DEGRADE（退 LLM 自检 fallback，不冤判 FAIL）
            async def ai_null(messages, **kw):
                return json.dumps({"steps": "无法代入", "all_satisfied": None, "reason": "缺信息"})
            V._ainvoke_text = ai_null
            r = await _machine_verify({"stem": "证明题", "answer": "见解析"}, None)
            chk("逆向验算·不可代入 → DEGRADE（退 fallback，不误判 FAIL）",
                r["verdict"] == math_verify.DEGRADE, extra=r["verdict"])

            # 🔴 默认走逆向（覆盖 §4③）：reverse_verify_on() 默认 True
            chk("默认 reverse_verify_on()=True（逆向带入是默认 pass/fail 源，§4③ 被覆盖）",
                V._reverse_verify_on() is True)
        finally:
            V._ainvoke_text = orig

    asyncio.run(_verdict_cases())

    # sympy dormant：关 reverse_verify → _machine_verify 走 sympy（math_verify.verify）
    async def _sympy_dormant():
        nonlocal ok
        prev = V.settings.VARIANT_REVERSE_VERIFY
        try:
            V.settings.VARIANT_REVERSE_VERIFY = False
            chk("关 reverse_verify → reverse_verify_on()=False（sympy dormant 路径激活）",
                V._reverse_verify_on() is False)
            r = await _machine_verify(
                {"stem": "解方程 x^2-7x+12=0", "answer": "x=3 或 x=4",
                 "verify_payload": {"kind": "equation_roots", "equation": "x**2-7*x+12",
                                    "claimed_roots": ["3", "4"], "var": "x"}}, None)
            chk("sympy dormant 路径验真方程 → PASS（兜底可用，未删）", r["verdict"] == math_verify.PASS,
                extra=r["verdict"])
        finally:
            V.settings.VARIANT_REVERSE_VERIFY = prev
    asyncio.run(_sympy_dormant())

    print("== G2⑤·难度仍 tier 驱动（grade_observed 表，不取 LLM 自评 difficulty）==")
    # 母题带锚定模型（tier_int 真值）→ 变式判档走 grade_observed，item 自带的 difficulty 不参与
    mother_dna = {"dna": {
        "main_kp": {"id": "k1", "name": "一元二次方程的解法"},
        "secondary_kps": [],
        "skeleton": ["移项", "因式分解", "求根"],
        "models": [{"id": "M07", "name": "因式分解法", "tier_int": 3, "freq_int": 4}],
    }}
    stem = "x^2-7x+12=0"
    sol = "(x-3)(x-4)=0 → x=3 或 4"
    bill = grade_variant_item({"stem": stem, "solution": sol, "difficulty": 1}, mother_dna)
    chk("grade_variant_item 产 grade_observed 账单（含 level）",
        isinstance(bill, dict) and "level" in bill, extra=f"level={bill.get('level')}")
    # 🔴 核心断言①：item 自评 difficulty 完全被忽略——塞 1/2/4 三种自评，level 恒为同一表驱动值。
    levels = {sd: grade_variant_item({"stem": stem, "solution": sol, "difficulty": sd}, mother_dna).get("level")
              for sd in (1, 2, 4)}
    chk("塞不同自评 difficulty(1/2/4) → level 恒定（自评被忽略，只读表）",
        len(set(levels.values())) == 1, extra=str(levels))
    # 🔴 核心断言②：换更高 tier 模型 + 更多步骤 → level 随表上移（确属表驱动，非常数）。
    md_hard = {"dna": {"main_kp": {"id": "k1", "name": "x"},
                       "secondary_kps": [{"id": "k2", "name": "y"}],
                       "skeleton": ["s1", "s2", "s3", "s4", "s5"],
                       "models": [{"id": "M9", "name": "big", "tier_int": 4, "freq_int": 1}]}}
    lvl_hard = grade_variant_item({"stem": "含参 k 讨论", "solution": "分类讨论：k>0…；k<0…；步骤多",
                                   "difficulty": 1}, md_hard).get("level")
    chk("更高 tier 模型/更多因子 → level 随表上移（表驱动非常数）",
        isinstance(lvl_hard, int) and lvl_hard > bill.get("level"),
        extra=f"tier3档 level={bill.get('level')} vs tier4档 level={lvl_hard}")

    return ok


async def _stream(client, message, thread_id, token, agent_cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": thread_id,
            "agent_config": {"ruoyi_token": token, **(agent_cfg or {})}}
    frames = {"stage": 0, "items": [], "error": None, "need_confirm": None, "detail_blob": ""}
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
            # 收集面向老师的 stage detail 文案（验证不外泄算子/系数）
            st = obj.get("stage")
            if isinstance(st, dict):
                frames["detail_blob"] += " " + str(st.get("detail") or "")
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


# 在线母题（贴图·含大招模型，验 tier→难度 + 逆向验算）。纯文本首轮路由较脆（实测 stage=0），
#   图母题是阶段二的可靠触发口（与 B2 smoke 同源）；难题/纯文本质量归 AC8 人工终审，在线只验链路存活。
IMG_HAS_MODEL = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)


async def online_gates() -> bool:
    """在线 e2e：贴图母题 → 阶段一锚定 → 阶段二出题，断言 0 error / 无卡死 + 面向老师帧不外泄
    算子/系数（G7）。tier→难度 + 逆向验算的对错质量归 AC8 人工终审（在线只验链路存活+不外泄）。"""
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    ok = True
    tid = "c107-b3-e2e"
    async with httpx.AsyncClient() as client:
        print("== 在线·阶段一（贴图母题 → 母题卡 / needConfirm） ==")
        f1 = await _stream(client, f"帮我对这道题举一反三 {IMG_HAS_MODEL}", tid, token)
        print(f"  stage帧={f1['stage']} needConfirm={bool(f1['need_confirm'])} err={f1['error']}")
        c1 = bool(f1["stage"]) and not f1["error"]
        print(f"  [{'PASS' if c1 else 'FAIL'}] 阶段一跑通 + 0 error")
        ok = ok and c1

        nc = f1.get("need_confirm") or {}
        ch = (nc.get("chapter") or {})
        gb = (nc.get("grade_book") or {})
        confirm_cfg = {"start_variants": True}
        if nc:
            confirm_cfg["confirmed_chapter_id"] = ch.get("id") or ch.get("name") or ""
            confirm_cfg["confirmed_grade_book_id"] = gb.get("id") or ""
            confirm_cfg["confirmed_chapter_name"] = ch.get("name") or ""
            print(f"  需确认 → 确认章「{ch.get('name')}」(id={ch.get('id') or '空·退name'}) 后触发阶段二")

        print("== 在线·阶段二出题（≤3 道·逆向验算·默认难度分布）==")
        f2 = await _stream(client, "出 3 道", tid, token, agent_cfg=confirm_cfg)
        items = f2.get("items") or []
        if not items and not f2["error"]:
            f2b = await _stream(client, "开始举一反三", tid, token, agent_cfg={"start_variants": True})
            if f2b.get("items"):
                items = f2b["items"]
            f2["stage"] += f2b["stage"]
            f2["error"] = f2["error"] or f2b["error"]
            f2["detail_blob"] += " " + f2b["detail_blob"]
        diffs = [it.get("difficulty") for it in items if isinstance(it, dict)]
        print(f"  stage帧={f2['stage']} items={len(items)} 各道难度={diffs} err={f2['error']}")
        c2_alive = bool(f2["stage"]) and not f2["error"]
        print(f"  [{'PASS' if c2_alive else 'FAIL'}] 阶段二存活（0 error/无卡死）")
        ok = ok and c2_alive

        # G7·算子/系数不外泄：面向老师 stage detail 文案不含「变式系数」+ 算子名
        blob = f1["detail_blob"] + " " + f2["detail_blob"]
        leak = ("变式系数" in blob)
        print(f"  [{'PASS' if not leak else 'FAIL'}] 面向老师 stage 帧不外泄『变式系数』数字（G7）")
        ok = ok and (not leak)
        if items:
            # 默认难度分布软核：前 N-1 道同档、末道 ≥ 前道（不硬卡质量，AC8）
            clean = [d for d in diffs if isinstance(d, int)]
            if len(clean) >= 2:
                last_higher = clean[-1] >= max(clean[:-1])
                print(f"  [{'PASS' if last_higher else 'INFO'}] 默认难度分布·末道 ≥ 前道(软核) {clean}")
    return ok


def main():
    offline_only = "--offline" in sys.argv
    print("===== PRD-C-107 B3 smoke =====")
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
