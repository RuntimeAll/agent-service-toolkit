# -*- coding: utf-8 -*-
"""PRD-C-104 B5·resume 冒烟（route_entry 多轮 resume 分支覆盖）。

🔴 背景：4 道精测网（c104_gate）是 in-process 直驱 graph、注入 mother_confirmed=True 直奔
   generate→闸A→闸B→assemble，**刻意绕开** route_entry 的 7-8 个 awaiting_* resume 分支。
   B5 要搬 route_entry（frozen 行为最密、满是 BUG-编号历史补丁），搬前必须先有覆盖这些分支的
   安全网，否则纯搬漏一个 config 信号判断、精测照样绿而线上多轮崩。

🔴 设计 = **纯函数·确定性·零网络零 LLM**：route_entry 是同步纯路由函数（读 state + config →
   返回下一节点字符串），本冒烟直接 import 它、对每条 resume 分支构造 (state, config) → 断言
   路由决策 == 期望节点。无 interrupt（C 线 resume = →END + checkpointer 持久 + 下轮 config 回传，
   不引 LangGraph interrupt），故 route_entry 的路由决策即「resume 后走哪一步」的全部真相。

   token 用伪造 unsigned JWT（teacher_id_from_token 只读 payload 不验签）→ 无需真登录、无需 :8093。

覆盖分支（route_entry 体内逐条）：
  1. auth        —— token 缺/解不出 userId → 'auth'（身份硬闸）
  2. editor_op   —— config 带 editor_op 且在手有 items/mother_dna → 'editor_entry'
  3. new image   —— 本轮人话含图 URL → 'mother_opus_entry'（新母题）
  4. confirm resume —— awaiting_mother_confirm + config.confirmed_chapter_id → 'classify'（重锚）
  5. lowconf block  —— 同上但 entry_decision 极低置信/章空 且未拦过 → 'entry_lowconf_block'
  6. start_variants —— awaiting_mother_review + 有 mother_dna + config.start_variants → 'generate'
  7. await改章 resume —— awaiting_mother_review + config.confirmed_chapter_id → 'classify'（高置信改章）
  8. await其他      —— awaiting_mother_review 但没点开始/没改章 → 'parse'（落分诊，不架空硬停闸）
  9. lib mother     —— mother_confirmed + mother_dna + 无 items → 'generate'
 10. old text       —— 有 items 或 mother_dna（在途母题）→ 'parse'
 11. ask            —— 无图无在途母题无题组 → 'ask'（催图）

🔴 确定性断言（task 指定的命脉）：分支 4/7（confirm/改章 resume）→ route_entry 返回 'classify'，
   即 resume 后必走重锚节点；配合 c104_stage1_smoke / c017_b4_e2e（HTTP 真机）验 classify 重锚
   后 anchored.code==确认章 code。本冒烟在 route 层钉死「resume 信号 → 重锚路由」这一frozen决策。

跑法（cwd=toolkit 根，无需任何服务）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c104_resume_smoke.py
返回码 0=GREEN（全分支路由符合）/ 非 0=RED。
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import route_entry  # noqa: E402  ← B5 搬动目标，直驱其路由

# 一张真 COS 图 URL（route_entry 的 _extract_image_url 只认 http(s) 图片 URL）。
IMG = (
    "https://question-1256278081.cos.ap-shanghai.myqcloud.com/"
    "2024-04-25/3869ff78-9925-4ec2-ba3a-96f61d8fc677/list/5/question.png"
)
# 确认章 = 浙教版「第二章一元二次方程」level2 章 id（前 4 位 3082 = 年级册 code）。
CONFIRMED_CHAPTER_ID = "3082002"


def _fake_token(user_id: int = 5) -> str:
    """伪造 unsigned RuoYi JWT：header.payload.sig，payload 含 userId。
    teacher_id_from_token 只 base64url 解 payload[1] 读 userId、不验签 → 这里够用。"""
    def _b64(d: dict) -> str:
        raw = json.dumps(d).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")
    return f"{_b64({'alg': 'HS512'})}.{_b64({'userId': user_id})}.sig"


TOKEN = _fake_token()


def _cfg(**configurable) -> dict:
    return {"configurable": dict(configurable)}


def _state(messages_text: str = "继续", **kw) -> dict:
    """构造最小 VariantState dict（route_entry 只读其中几位）。"""
    st: dict = {"messages": [HumanMessage(content=messages_text)]}
    st.update(kw)
    return st


# (描述, state, config, 期望路由) —— 逐条对应 route_entry 体内分支。
CASES: list[tuple[str, dict, dict, str]] = [
    # 1. auth：token 缺 → 一步不走（身份硬闸，最前置）
    ("auth·无token", _state("贴图举一反三"), _cfg(), "auth"),
    ("auth·烂token", _state("继续"), _cfg(ruoyi_token="not.a.jwt"), "auth"),
    # 2. editor_op：config 带结构化编辑 op 且在手有题组 → editor_entry（优先于自然语言分诊）
    (
        "editor_op·有题组",
        _state("随便说", items=[{"stem": "x"}]),
        _cfg(ruoyi_token=TOKEN, editor_op={"kind": "revise", "index": 0}),
        "editor_entry",
    ),
    # editor_op 但无题组（op 无对象）→ 不拦，落后续（无图无母题 → ask）
    (
        "editor_op·无题组不拦",
        _state("随便说"),
        _cfg(ruoyi_token=TOKEN, editor_op={"kind": "revise", "index": 0}),
        "ask",
    ),
    # 3. 跨轮新图 = 新母题 → 塌缩入口
    (
        "new_image·新母题",
        _state(f"{IMG} 出3道"),
        _cfg(ruoyi_token=TOKEN),
        "mother_opus_entry",
    ),
    # 4. 低置信确认 resume：awaiting_mother_confirm + 回传确认章 → classify（重锚命脉）
    (
        "confirm_resume·重锚",
        _state("确认：第二章一元二次方程", awaiting_mother_confirm=True),
        _cfg(ruoyi_token=TOKEN, confirmed_chapter_id=CONFIRMED_CHAPTER_ID),
        "classify",
    ),
    # 5. 闸4·读图极低置信前置拦截：同上但 entry_decision 置信<0.40 且未拦过 → entry_lowconf_block
    (
        "lowconf_block·拦一次",
        _state(
            "确认",
            awaiting_mother_confirm=True,
            entry_decision={"confidence": 0.2, "chapter": ""},
        ),
        _cfg(ruoyi_token=TOKEN, confirmed_chapter_id=CONFIRMED_CHAPTER_ID),
        "entry_lowconf_block",
    ),
    # 5b. 已拦过一次（_lowconf_blocked=True）→ 老师坚持 → 放行 classify（防永久卡死）
    (
        "lowconf_block·拦过放行",
        _state(
            "坚持确认",
            awaiting_mother_confirm=True,
            entry_decision={"confidence": 0.2, "chapter": ""},
            _lowconf_blocked=True,
        ),
        _cfg(ruoyi_token=TOKEN, confirmed_chapter_id=CONFIRMED_CHAPTER_ID),
        "classify",
    ),
    # 6. 母题卡硬停闸 resume·点开始 → generate（复用 state.mother_dna，不重 classify）
    (
        "start_variants·直奔generate",
        _state(
            "开始举一反三",
            awaiting_mother_review=True,
            mother_dna={"stem": "母题", "dna": {"main_kp": {"id": "x", "name": "韦达"}}},
        ),
        _cfg(ruoyi_token=TOKEN, start_variants=True),
        "generate",
    ),
    # 7. 高置信 await_review 态改章 resume（BUG-01 R1·#4）→ classify 重锚
    (
        "await改章·重锚",
        _state(
            "改成第二章",
            awaiting_mother_review=True,
            mother_dna={"stem": "母题", "dna": {"main_kp": {"id": "x", "name": "韦达"}}},
        ),
        _cfg(ruoyi_token=TOKEN, confirmed_chapter_id=CONFIRMED_CHAPTER_ID),
        "classify",
    ),
    # 8. await_review 但没点开始/没改章（发别的话）→ parse 分诊（不架空硬停闸）
    (
        "await其他·落parse",
        _state(
            "这题难度太高了",
            awaiting_mother_review=True,
            mother_dna={"stem": "母题", "dna": {"main_kp": {"id": "x", "name": "韦达"}}},
        ),
        _cfg(ruoyi_token=TOKEN),
        "parse",
    ),
    # 9. 库内母题（已确认 DNA）、无 items → 直接造
    (
        "lib_mother·直造",
        _state(
            "出题",
            mother_confirmed=True,
            mother_dna={"stem": "母题", "dna": {"main_kp": {"id": "x", "name": "韦达"}}},
        ),
        _cfg(ruoyi_token=TOKEN),
        "generate",
    ),
    # 10. 老会话·纯文字：已出题组 → parse 分诊
    (
        "old_text·有题组→parse",
        _state("把第2题删了", items=[{"stem": "v1"}]),
        _cfg(ruoyi_token=TOKEN),
        "parse",
    ),
    # 10b. 在途母题（停 clarify 等答年级/考点）→ parse（修 17 号漏洞：clarify 回答别漏成催图）
    (
        "old_text·在途母题→parse",
        _state("八年级", mother_dna={"stem": "母题"}),
        _cfg(ruoyi_token=TOKEN),
        "parse",
    ),
    # 11. 无图无在途母题无题组 → 催图
    ("ask·催图", _state("帮我出题"), _cfg(ruoyi_token=TOKEN), "ask"),
]


def main() -> int:
    print("=== PRD-C-104 B5·route_entry resume 冒烟（纯函数·零网络零LLM）===", flush=True)
    fails: list[str] = []
    for desc, state, config, expect in CASES:
        try:
            got = route_entry(state, config)
        except Exception as e:  # noqa: BLE001
            got = f"<EXC {type(e).__name__}: {e}>"
        ok = got == expect
        flag = "OK " if ok else "RED"
        print(f"  [{flag}] {desc:<26} 期望={expect:<20} 实得={got}", flush=True)
        if not ok:
            fails.append(f"{desc}: 期望 {expect} 实得 {got}")

    print("-" * 64, flush=True)
    # 命脉断言：confirm/改章 resume → classify（resume 后必走重锚）
    mainline = [c for c in CASES if c[3] == "classify" and "resume" in c[0] or "改章" in c[0]]
    print(
        f"resume→重锚(classify) 命脉用例 {len(mainline)} 条；"
        f"全分支 {len(CASES)} 条，失败 {len(fails)} 条 → "
        f"{'GREEN' if not fails else 'RED'}",
        flush=True,
    )
    if fails:
        for f in fails:
            print(f"    RED: {f}", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
