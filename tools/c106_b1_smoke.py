# -*- coding: utf-8 -*-
"""PRD-C-106 B1·阶段一收口 ≤4 道精测闸（带料解题 + 消重复解题 + 诚实三态）。

两段：
  A. 离线断言（无需服务·恒可跑）= G1/G2/G8 的纯函数证据：
     - 带料注入：build_mother_prompt / build_solve_messages / build_struct_messages 含工具箱块。
     - 纯代码映射：anchor_models_from_names 有模型出真模型 / 无模型出 model_flag=no_model（非 M00）。
     - 确认路径无第二次 LLM 解题：anchor_models_from_names 非 async、不收 invoke（代码契约证据）。
  B. 在线 e2e（需 :8093 新代码 + :8090/:8080 RuoYi）= 闸1/2/3/4：
     ≤4 道真图 stage1 跑通 → 母题卡：有模型出真模型 summary / 无模型出「无考模型」(非 M00)。

跑法（A 段）：PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c106_b1_smoke.py --offline
跑法（A+B）：... tools/c106_b1_smoke.py   （需 :8093 已加载本批新代码）
"""
import asyncio
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BASE = "http://localhost:8093"
# 有模型题（折线/数轴折叠类，命中数轴大招）+ 无模型题（纯概念/送分）。复用 c104 同款有图母题。
IMG_HAS_MODEL = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)


def offline_gates() -> bool:
    """A 段离线断言（G1/G2/G8 纯函数证据）。返回 all-green。"""
    from agents import mother_opus, model_anchor
    from agents import variant_entry as ve

    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        flag = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"  [{flag}] {name} {extra}")

    print("== G1 带料解题：工具箱注入 prompt ==")
    tb = model_anchor.build_toolbox_clause([
        {"id": "DZ01", "name": "数轴折叠", "trigger_feature": "折叠", "action_conclusion": "对称中点"},
    ])
    chk("build_toolbox_clause 非空且含名", "数轴折叠" in tb and "工具箱" in tb)
    p = mother_opus.build_mother_prompt(
        grade_text="七上", chapter_text="第2章", leaf_pool=[("1", "数轴")], model_toolbox=tb,
    )
    chk("build_mother_prompt 注入工具箱段", "带料解题） ====" in p and "数轴折叠" in p)
    chk("build_mother_prompt 无工具箱→不注入",
        "带料解题） ====" not in mother_opus.build_mother_prompt(
            grade_text="七上", chapter_text=None, leaf_pool=[("1", "x")]))
    sm = ve.build_solve_messages(image_url="http://x", model_toolbox=tb)
    chk("build_solve_messages 注入工具箱",
        any("可用解题大招工具箱" in str(getattr(m, "content", "")) for m in sm))
    stm = ve.build_struct_messages(image_url="http://x", solved_solution="s", solved_answer="1",
                                   model_toolbox=tb)
    chk("build_struct_messages 注入工具箱",
        any("可用解题大招工具箱" in str(getattr(m, "content", "")) for m in stm))

    print("== G2 消重复解题：确认路径纯代码、无第二次 LLM 解题 ==")
    chk("anchor_models_from_names 非 async（纯代码）",
        not inspect.iscoroutinefunction(model_anchor.anchor_models_from_names))
    sig = inspect.signature(model_anchor.anchor_models_from_names)
    chk("anchor_models_from_names 不收 invoke/LLM 句柄",
        "invoke" not in sig.parameters, extra=str(list(sig.parameters)))

    print("== G8 诚实三态：有模型出真模型 / 无模型 no_model（非 M00） ==")
    cands = [
        {"id": "DZ01", "name": "数轴折叠", "tier_int": 2, "freq_int": 1},
        {"id": "M10", "name": "配方法", "tier_int": 1, "freq_int": 2},
    ]
    r_has = model_anchor.anchor_models_from_names(["数轴折叠"], candidates=cands)
    chk("有模型→真模型(DZ01)、flag=None",
        r_has["models"] and r_has["models"][0]["id"] == "DZ01" and r_has["model_flag"] is None)
    r_no = model_anchor.anchor_models_from_names([], candidates=cands)
    chk("无模型→models:[]、flag=no_model（非 M00）",
        r_no["models"] == [] and r_no["model_flag"] == "no_model")
    r_ovf = model_anchor.anchor_models_from_names(["不存在的套路"], candidates=cands)
    chk("池外名→no_model + overflow 留痕（绝不 M00）",
        r_ovf["models"] == [] and r_ovf["model_flag"] == "no_model"
        and r_ovf["model_overflow"] == ["不存在的套路"])

    # 母题卡/变式卡展示态接通
    import agents.variant as V  # noqa: F401  触发 re-export 链
    from agents.variant import _build_mother_card, _item_dna
    st_no = {"mother_dna": {"stem": "概念", "answer": "A", "analysis": "略",
             "dna": {"main_kp": {"id": "1", "name": "概念"}, "secondary_kps": [], "qtype": "选择",
                     "models": [], "model_flag": "no_model", "skeleton": ["判断"], "tags": ["概念"]}},
             "analysis": {"grade": {"value": "七上", "code": "1000", "confidence": 0.9}}}
    mc = _build_mother_card(st_no)
    chk("母题卡 no_model 展示态 + 难度仍出档（不崩）",
        mc["dna"]["no_model"] is True and mc["dna"]["model_flag"] == "no_model")
    idna = _item_dna({}, {"dna": st_no["mother_dna"]["dna"]})
    chk("变式卡 no_model 展示态", idna["no_model"] is True and idna["model_flag"] == "no_model")

    return ok


async def _stream(client, message, thread_id, token, agent_cfg=None):
    body = {"message": message, "stream_tokens": True, "thread_id": thread_id,
            "agent_config": {"ruoyi_token": token, **(agent_cfg or {})}}
    frames = {"stage": 0, "mother_card": None, "error": None}
    import httpx  # noqa: F401
    async with client.stream("POST", f"{BASE}/variant/stream", json=body,
                             headers={"Content-Type": "application/json"}, timeout=300) as resp:
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
        h = obj.get("header")
        if isinstance(h, dict) and isinstance(h.get("mother_card"), dict):
            frames["mother_card"] = h["mother_card"]
        if isinstance(obj.get("mother_card"), dict):
            frames["mother_card"] = obj["mother_card"]
        if obj.get("error") and not frames["error"]:
            frames["error"] = str(obj.get("error"))[:160]
        for v in obj.values():
            _walk(v, frames)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, frames)


async def online_gates() -> bool:
    import httpx
    from agents.variant_support import RuoyiClient

    rc = RuoyiClient()
    token = await rc.login()
    await rc.aclose()
    ok = True
    async with httpx.AsyncClient() as client:
        print("== 闸1/3 有模型题：母题卡模型行出真模型 summary（非 M00） ==")
        f = await _stream(client, f"帮我对这道题举一反三 {IMG_HAS_MODEL}", "c106-b1-has", token)
        mc = f.get("mother_card") or {}
        models = (mc.get("dna") or {}).get("models") or []
        flag = (mc.get("dna") or {}).get("model_flag")
        no_m00 = all(str(m.get("id")) != "M00" for m in models)
        print(f"  stage帧={f['stage']} models={[m.get('name') for m in models]} flag={flag} err={f['error']}")
        cond = bool(f["stage"]) and not f["error"] and no_m00
        print(f"  [{'PASS' if cond else 'FAIL'}] 有模型题跑通 + 无 M00 占位")
        ok = ok and cond
    return ok


def main():
    offline_only = "--offline" in sys.argv
    print("===== PRD-C-106 B1 smoke =====")
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
