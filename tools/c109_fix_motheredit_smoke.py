"""PRD-C-109 fix 冒烟：母题卡就绪态（无题组）下「在位改单维」真生效（根因修验收）。

根因（e2e 实测）：举一反三**主场景** = 母题卡就绪、还没点开始（无题组 / no items /
`awaiting_mother_review`）。这个态下老师随口「加个易错标签」「改题型」时，旧实现：
  - 即时生效（meta）工具：intent_triage 节点 `_meta_apply_index` 无题组返回 None → **跳过 apply**
    （标签没加上）→ route_after_triage 把它落 `parse` → parse_instruction 把就绪卡当母题**全量重解**
    → ① 反复弹「确认年级章」② AI 假宣称「已更新」③ 就绪卡被打回存疑（解法骨架/标签丢）。

修法（re-wire 不 rebuild）：
  - persist.py 新增 `edit_mother_dna_meta`（edit_dna_state 的 meta 子集「去 item 化」镜像，
    只改 mother_dna.dna、不需 item、不重解/重锚/清组）。
  - tool_registry：meta mutator(tags/副考点/难点) 在 index=None（无题组）时走 edit_mother_dna_meta。
  - intent.py：无题组 meta 也在 intent_triage 节点**原地改**；route_after_triage 即时生效一律 await_review
    （绝不再落 parse 重解）。
  - 重出/重写维（改题型等）：母题已 mother_endorsed（B3）→ 重锚走 endorsed 旁路、不重弹确认章。

四道断言（全离线·确定性，无网络/库）：
  ① 无题组 await_review 态加标签 → 标签真加上 + 不落 parse（route=await_review）+ scene 别维不变
     + endorsed 保持 + 无 classify/重解触发（mother_dna.stem 不变、items 仍空、mother_confirmed 不被清）。
  ② 无题组改题型（重出）= 路由 parse（既有重锚链）；母题已 endorsed → _reanchor 不重弹确认章
     （awaiting_mother_confirm 不置），就绪保持。
  ③ meta 改后别维不动 + len(messages) 不暴涨（守 G5·不因解题/重出累积）。
  ④ apply_tool(index=None) 直测：tags/副考点/难点三 meta 维去 item 化原地写真生效；
     非 meta 维（改题型）index=None → 被 edit_dna_state index 闸拒（重生须题组，语义正确）。

跑法（cwd = toolkit，必走 .venv）：
  $env:NO_PROXY="*"; .venv\\Scripts\\python.exe tools\\c109_fix_motheredit_smoke.py
"""
import asyncio
import base64
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents.variant import (  # noqa: E402
    apply_tool,
    edit_mother_dna_meta,
    edit_mother_dna_regen_noitems,
    gate_after_classify,
    intent_triage,
)
from agents.variant.entry import intent as intent_mod  # noqa: E402
from agents.variant.entry.intent import route_after_triage  # noqa: E402
from agents.variant.stage1_anchor import label as label_mod  # noqa: E402

PASS = 0
FAIL = 0


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {msg}")
    else:
        FAIL += 1
        print(f"  XX  {msg}")


def _tok(uid: int = 5) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"userId": uid}).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


def _cfg(**extra) -> dict:
    conf = {"ruoyi_token": _tok()}
    conf.update(extra)
    return {"configurable": conf}


# 母题卡就绪态（无题组）：mother_dna 立住（有解法骨架/答案/tags），awaiting_mother_review、items 空。
_MOTHER_OK = dict(
    mother_confirmed=True,
    facts_locked=True,
    awaiting_mother_review=True,
    items=[],
    analysis={
        "grade": {"value": "八年级下学期", "confidence": 0.9, "code": "3082"},
        "kp": {"value": "一元二次方程", "confidence": 0.9,
               "anchored": {"id": "3082002001", "code": "3082002001", "name": "一元二次方程"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    mother_dna={
        "stem": "x^2-3x+2=0", "answer": "x=1或2", "solution_skeleton": ["因式分解", "求根"],
        "difficulty": 3,
        "dna": {
            "main_kp": {"id": "3082002001", "name": "一元二次方程"},
            "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
            "skeleton": ["因式分解", "求根"], "hard_points": ["符号"], "tags": ["解方程"],
            "scene": "纯代数", "models": [{"id": "M00", "name": "概念直用"}], "difficulty": 3,
        },
    },
    facts_audit=[],
)


def _card(text: str, **extra) -> dict:
    s = {"messages": [HumanMessage(content=text)]}
    s.update(copy.deepcopy(_MOTHER_OK))
    s.update(extra)
    return s


# ---------------------------------------------------------------------------
# ① 无题组 await_review 态加标签 → 真生效 + 不重解 + 不弹确认 + endorsed 保持
# ---------------------------------------------------------------------------
def test_noitems_meta_addtag_applies():
    banner("① 无题组（母题卡就绪）加标签 → 标签真加上、不落 parse 重解、就绪不打回")

    async def _fake_classify(state, config):
        return intent_mod._tool_decision("加标签", value="判别式符号", conf=0.95)

    st = _card("加个易错标签：判别式符号", mother_endorsed=True)
    stem_before = st["mother_dna"]["stem"]
    scene_before = st["mother_dna"]["dna"]["scene"]

    _orig = intent_mod.classify_intent
    intent_mod.classify_intent = _fake_classify
    try:
        out = asyncio.run(intent_triage(st, _cfg(mother_endorsed=True)))
    finally:
        intent_mod.classify_intent = _orig

    new_tags = (out.get("mother_dna") or {}).get("dna", {}).get("tags") or []
    ok("判别式符号" in new_tags, f"标签真加上（去 item 化原地写，tags={new_tags}）")
    ok("解方程" in new_tags, "旧标签保留（原地改非整组清）")
    ok(out.get("intent_decision", {}).get("_applied") is True,
       "intent_decision 标 _applied（节点已原地应用、非掉 parse）")
    # 别维不变（只改 tags 一字段）
    out_scene = ((out.get("mother_dna") or {}).get("dna") or {}).get("scene")
    ok(out_scene == scene_before, f"scene 别维不变（{out_scene}）")
    # 不重解：mother_dna.stem 不变、items 没被清造（update 不含 items 清空）、mother_confirmed 不被置 False
    out_stem = (out.get("mother_dna") or {}).get("stem")
    ok(out_stem == stem_before, "mother_dna.stem 不变（无母题全量重解）")
    ok("items" not in out, "update 不写 items（meta 不清组、不重造）")
    ok(out.get("mother_confirmed") is not False, "mother_confirmed 不被打回 False（就绪卡不掉存疑）")
    ok(not out.get("awaiting_mother_confirm"), "不弹「确认年级章」（awaiting_mother_confirm 未置）")
    ok(out.get("mother_endorsed") is True, "endorsed 保持（确认收口终态，不撤）")

    # route：即时生效·无题组 → await_review（绝不落 parse 重解）
    routed = route_after_triage({**st, **out}, _cfg(mother_endorsed=True))
    ok(routed == "await_review",
       f"route_after_triage → await_review（不落 parse·根因修，得 {routed}）")


# ---------------------------------------------------------------------------
# ② 无题组改题型（重出）= 路由 parse；endorsed → _reanchor 不重弹确认章
# ---------------------------------------------------------------------------
def test_noitems_changeqtype_endorsed_no_reconfirm():
    banner("② 无题组改题型（重出）→ 原地写保持就绪（不落 parse 重锚）；endorsed → 不重弹确认章")

    async def _fake_classify(state, config):
        return intent_mod._tool_decision("set_题型", value="填空", conf=0.9)

    # 🔴 C-109 fix：无题组就绪态改题型 = 母题级守恒维原地写 mother_dna.dna，保持就绪、不落 parse 重锚。
    st = _card("题型改成填空", mother_endorsed=True)
    _orig = intent_mod.classify_intent
    intent_mod.classify_intent = _fake_classify
    try:
        out = asyncio.run(intent_triage(st, _cfg(mother_endorsed=True)))
    finally:
        intent_mod.classify_intent = _orig
    new_qtype = ((out.get("mother_dna") or {}).get("dna") or {}).get("qtype")
    ok(new_qtype == "填空", f"改题型原地写 mother_dna.dna.qtype（{new_qtype}）= 重出维就绪态在位改")
    ok(out.get("intent_decision", {}).get("_applied") is True, "改题型标 _applied（节点已原地应用）")
    ok("items" not in out, "改题型 update 不清 items（保持就绪、不重锚·§2.2②）")
    ok(out.get("mother_confirmed") is not False, "改题型不打回 mother_confirmed（就绪卡不掉存疑）")
    # route：已 _applied → await_review（不落 parse 重锚链）
    routed = route_after_triage({**st, **out}, _cfg(mother_endorsed=True))
    ok(routed == "await_review",
       f"改题型(重出·无题组已在位改) → await_review（不落 parse 重锚，得 {routed}）")

    # 🔴 grade（hard_anchor）仍走 parse 重锚链（学段变=整组失效，legitimately 重锚，不在无题组旁路）。
    st2 = _card("这是九年级上的", mother_endorsed=True)
    dec2 = intent_mod._tool_decision("set_年级章", value="九年级上学期", conf=0.9)
    routed2 = route_after_triage({**st2, "intent_decision": dec2}, _cfg(mother_endorsed=True))
    ok(routed2 == "parse", f"改年级(hard_anchor) → parse（学段变=legitimately 重锚，得 {routed2}）")

    # endorsed → _reanchor 锚不到叶子也不弹确认章（B3 旁路；本 fix 保证 endorsed 经 parse 仍在 state）。
    state_base = copy.deepcopy(_MOTHER_OK)
    state_base["messages"] = []
    prev_dna = {"main_kp": {"id": "", "name": "韦达定理"}, "secondary_kps": [],
                "qtype": "填空", "exam_type": "直接计算", "skeleton": ["设两根", "韦达"],
                "tags": ["韦达定理"], "models": []}
    leaf_pool = [("3082002009001", "不相关叶子A"), ("3082002009002", "不相关叶子B")]
    mother_dna = {"stem": "韦达母题", "answer": "x1+x2=...", "solution_skeleton": ["韦达"],
                  "mother_solve_source": "opus", "dna": dict(prev_dna)}
    out_e = asyncio.run(label_mod._reanchor_reuse_first_solve(
        state={**state_base, "mother_endorsed": True},
        analysis=dict(state_base["analysis"]), mother_dna=dict(mother_dna), prev_dna=dict(prev_dna),
        grade_code="3082", chapter_id="3082002", leaf_pool=leaf_pool,
        confirmed_chapter_id="3082002", include_review_books=False, knobs=None,
    ))
    ok(not out_e.get("awaiting_mother_confirm"),
       "endorsed 改题型重锚 + 锚不到叶子 → 不重弹确认章（B3 旁路、确认收口）")
    ok(out_e.get("mother_confirmed") is True, "endorsed → confirmed=True 往下（永有出边、不回 picker）")


# ---------------------------------------------------------------------------
# ③ meta 改后别维不动 + messages 不暴涨（守 G5）
# ---------------------------------------------------------------------------
def test_g5_noitems_meta_boundary():
    banner("③ 无题组 meta 改：别维不动 + messages 不暴涨（G5 边界）")
    st = _card("加个易错标签", mother_endorsed=True)
    # 直测去 item 化写：加标签
    update, edited, err = apply_tool("加标签", st, None, "易错")
    ok(err is None, f"无题组加标签 apply_tool(index=None) 无错（err={err}）")
    ok(edited is None, "无题组 meta 写 edited_item=None（无 item 可标，符合契约）")
    dna = (update.get("mother_dna") or {}).get("dna") or {}
    ok("易错" in (dna.get("tags") or []) and "解方程" in (dna.get("tags") or []),
       f"tags 原地加且旧标签保留（{dna.get('tags')}）")
    # 别维不动：scene/main_kp/skeleton 与原值一致
    ok(dna.get("scene") == "纯代数", "scene 别维不变")
    ok(dna.get("skeleton") == ["因式分解", "求根"], "skeleton 别维不变（解法骨架不丢）")
    # messages 不暴涨（meta 写不发解题/重出气泡）
    msgs = update.get("messages") or []
    ok(len(msgs) == 0, f"meta 写 messages 不暴涨（len={len(msgs)}，不重解/不重出累积·G5）")
    # 不清组、不打回存疑
    ok("items" not in update, "update 不写 items（不清组）")
    ok(update.get("mother_confirmed") is not False, "mother_confirmed 不被打回")


# ---------------------------------------------------------------------------
# ④ apply_tool(index=None) 直测：三 meta 维都生效；非 meta 维被 index 闸拒（语义正确）
# ---------------------------------------------------------------------------
def test_apply_tool_noitems_dispatch():
    banner("④ apply_tool(index=None) 去 item 化分流：meta 维生效 / 非 meta 维拒")
    st = _card("x")

    # 选副考点（list meta）
    up1, _e1, err1 = apply_tool("选副考点", st, None,
                                {"id": "3082002002", "name": "根的判别式"})
    sec = (up1.get("mother_dna") or {}).get("dna", {}).get("secondary_kps") or []
    ok(err1 is None and any(s.get("id") == "3082002002" for s in sec),
       f"无题组选副考点 → 原地加进 secondary_kps（{[s.get('name') for s in sec]}）")

    # set_难点（标量 meta）
    up2, _e2, err2 = apply_tool("set_难点", st, None, "判别式符号易错")
    hp = (up2.get("mother_dna") or {}).get("dna", {}).get("hard_points") or []
    ok(err2 is None and "判别式符号易错" in hp, f"无题组 set_难点 → 原地写 hard_points（{hp}）")

    # 删标签（list meta）
    up3, _e3, err3 = apply_tool("删标签", st, None, "解方程")
    tags3 = (up3.get("mother_dna") or {}).get("dna", {}).get("tags")
    ok(err3 is None and tags3 == [], f"无题组删标签 → 原地删（{tags3}）")

    # 非 meta 维（改题型=重出）index=None → edit_dna_state index 闸拒（重生须题组，语义正确、不静默乱改）
    up4, _e4, err4 = apply_tool("set_题型", st, None, "填空")
    ok(err4 is not None and not up4,
       f"非 meta 维(改题型) index=None → 被 index 闸拒（{err4!r}）= 重生须题组，不去 item 化")

    # edit_mother_dna_meta 防误用：非 meta field 直拒
    _u, _e, err5 = edit_mother_dna_meta(st, "qtype", "填空")
    ok(err5 is not None, "edit_mother_dna_meta 拒非 meta 维（qtype）— 防误用旁路重生")

    # edit_mother_dna_regen_noitems：重出/重写维生效 + 非白名单（grade）拒
    u6, _e6, err6 = edit_mother_dna_regen_noitems(_card("x"), "qtype", "填空")
    ok(err6 is None and ((u6.get("mother_dna") or {}).get("dna") or {}).get("qtype") == "填空",
       "edit_mother_dna_regen_noitems 改题型原地写 dna.qtype 生效")
    _u7, _e7, err7 = edit_mother_dna_regen_noitems(_card("x"), "grade", "九年级上学期")
    ok(err7 is not None, "edit_mother_dna_regen_noitems 拒 grade（hard_anchor 走重锚链，不在无题组旁路）")


# ---------------------------------------------------------------------------
# ⑤ gate_after_classify 确认收口（§2.2①）：endorsed + 有 mother_dna + 未 pin → await_review（不 clarify）
# ---------------------------------------------------------------------------
def test_gate_endorsed_no_clarify():
    banner("⑤ gate_after_classify·endorsed override：重锚未 pin 也保持就绪、不 clarify 重弹")
    # 未 pin（mother_confirmed=False）+ 未 endorse → clarify（既有行为不破）
    st_unpinned = {"mother_confirmed": False, "mother_dna": {"stem": "x", "dna": {}}}
    ok(gate_after_classify(st_unpinned) == "clarify",
       "未 pin + 未 endorse → clarify（既有行为不破）")
    # 未 pin + endorsed + 有 mother_dna → await_review（确认收口·§2.2①，不重弹年级章）
    ok(gate_after_classify({**st_unpinned, "mother_endorsed": True}) == "await_review",
       "未 pin + endorsed + 有 mother_dna → await_review（确认收口，不 clarify 重弹）")
    # endorsed 但无 mother_dna（母题还没立住）→ 仍 clarify（不滥用 endorse 跳过必要确认）
    ok(gate_after_classify({"mother_confirmed": False, "mother_endorsed": True}) == "clarify",
       "endorsed 但无 mother_dna → 仍 clarify（母题没立住不滥跳）")


if __name__ == "__main__":
    test_noitems_meta_addtag_applies()
    test_noitems_changeqtype_endorsed_no_reconfirm()
    test_g5_noitems_meta_boundary()
    test_apply_tool_noitems_dispatch()
    test_gate_endorsed_no_clarify()
    banner(f"汇总: PASS={PASS}  FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
