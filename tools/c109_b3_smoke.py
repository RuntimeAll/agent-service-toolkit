"""PRD-C-109 B3 真机冒烟：确认收口 mother_endorsed + 锚降级防死循环 + 降级态打标(标签/骨架) + G5 边界。

验四件事（全离线·确定性·可单测，不依赖网络/库）：
  ① 确认收口（AC4）：mother_endorsed=True → mother_in_doubt 恒 False；_endorsed_from_config 识别
     mother_endorsed / start_variants / confirmed_chapter_id 三信号；intent_triage 节点把 config endorse
     lift 进 state（终态）。多确认闸收口成一个（endorsed 后存疑判据一律不挡）。
  ② 锚降级防死循环（AC4）：_reanchor_reuse_first_solve 在「锚不到叶子(degraded)」时，endorsed →
     不再弹「再确认一次」(awaiting_mother_confirm)、confirmed=True 往下出题（永有出边、不回 picker）；
     非 endorsed 首次冲突仍走「拦一次」闸（既有 bounded 行为不破）。
  ③ 降级态打标（AC6）：_bounded_degrade_to_chapter（无首解·minimal DNA）产出的母题 dna.tags +
     skeleton 非空；_build_mother_card 把 tags 映射进母题卡 dna.tags（映射没丢）。
  ④ G5 守边界断言（评审硬加）：① meta 编辑（加标签）后已生成变式 figure_url 仍在（防 merge_items
     PRESERVE_IF_SAME_STEM 静默丢图）；② 编辑工具执行后 len(messages) 不因解题/重出暴涨。

跑法（cwd = toolkit，必走 .venv）：
  $env:NO_PROXY="*"; .venv\\Scripts\\python.exe tools\\c109_b3_smoke.py
全离线、无在线段。
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
    _build_mother_card,
    _endorsed_from_config,
    apply_tool,
    intent_triage,
    mother_in_doubt,
)
from agents.variant.entry import intent as intent_mod  # noqa: E402
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


# 存疑母题（解法骨架/答案皆空 + 锚未定死 → mother_in_doubt 三条都触发；用来证 endorse 一键收口）。
def _doubt_state(text: str = "开始") -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "awaiting_mother_review": True,
        "items": [],
        # 锚未定死（pinned=False：grade 低置信）
        "analysis": {"grade": {"value": "?", "confidence": 0.2}},
        "mother_dna": {"stem": "某母题题面"},  # 解法骨架/答案皆空 → ② 存疑
    }


# 立住母题 + 2 道变式（带 figure_url，验 G5）。
_MOTHER_OK = dict(
    mother_confirmed=True,
    facts_locked=True,
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


def _variants(text: str, n: int = 2) -> dict:
    s = {"messages": [HumanMessage(content=text)]}
    s.update(copy.deepcopy(_MOTHER_OK))
    s["items"] = [
        {"stem": f"变式{i}题面", "qtype": "解答", "difficulty": 3, "_seq": i,
         "figure_url": f"https://oss/fig{i}.png", "models": [{"id": "M00", "name": "概念直用"}]}
        for i in range(1, n + 1)
    ]
    return s


# ---------------------------------------------------------------------------
# ① 确认收口：mother_endorsed override mother_in_doubt + config 信号识别 + 节点 lift
# ---------------------------------------------------------------------------
def test_endorse_closure():
    banner("① 确认收口（AC4）：mother_endorsed override + config 信号 + intent_triage lift")
    # 存疑母题：未 endorse → in_doubt=True（三条都触发，老师点开始也先停）
    st = _doubt_state()
    ok(mother_in_doubt(st) is True, "未 endorse 的存疑母题 → mother_in_doubt=True（停母题卡，AC2 不抢跑）")
    # endorse 后 → in_doubt=False（用户确认 > 代码硬锚，多确认闸一键收口）
    st_e = {**st, "mother_endorsed": True}
    ok(mother_in_doubt(st_e) is False, "endorse 后 → mother_in_doubt=False（确认即终，不再被锚定/骨架拦）")

    # _endorsed_from_config 识别三信号（任一即背书），无信号 → False
    ok(_endorsed_from_config(_cfg(mother_endorsed=True)), "config.mother_endorsed=true → 背书")
    ok(_endorsed_from_config(_cfg(start_variants=True)), "config.start_variants=true → 背书（点开始=隐式背书）")
    ok(_endorsed_from_config(_cfg(confirmed_chapter_id="3082002")), "config.confirmed_chapter_id → 背书（确认章=背书）")
    ok(not _endorsed_from_config(_cfg()), "无背书信号 → 不背书（普通轮不误置）")

    # intent_triage 节点：config 带 start_variants → lift mother_endorsed 进 state（终态、持久）
    async def _fake_classify(state, config):
        # 选个 meta 工具（节点能跑通），但本测只验 endorse lift
        return intent_mod._tool_decision("加标签", value="易错", conf=0.95)

    _orig = intent_mod.classify_intent
    intent_mod.classify_intent = _fake_classify
    try:
        out = asyncio.run(intent_triage(_variants("加个标签"), _cfg(start_variants=True)))
    finally:
        intent_mod.classify_intent = _orig
    ok(out.get("mother_endorsed") is True, "intent_triage: config 带背书 → lift mother_endorsed=True 进 state（终态）")

    # 普通轮（无 config 背书、state 也无）→ 不置 endorse（不污染）
    intent_mod.classify_intent = _fake_classify
    try:
        out2 = asyncio.run(intent_triage(_variants("加个标签"), _cfg()))
    finally:
        intent_mod.classify_intent = _orig
    ok("mother_endorsed" not in out2, "intent_triage: 无背书信号 → 不置 mother_endorsed（不误污染）")


# ---------------------------------------------------------------------------
# ② 锚降级防死循环：_reanchor degraded + endorsed → 往下出题，不回 picker
# ---------------------------------------------------------------------------
def test_anchor_degrade_no_loop():
    banner("② 锚降级防死循环（AC4）：degraded + endorsed → 永有出边、不弹再确认")
    # 构造「锚不到叶子」场景：leaf_pool 给一个不含母题考点的池 → _match_kp_in_pool 必失败 → degraded。
    # endorsed=True → 闸3 放行（不弹 awaiting_mother_confirm），confirmed=True 往下。
    state_base = copy.deepcopy(_MOTHER_OK)
    state_base["messages"] = []
    prev_dna = {
        "main_kp": {"id": "", "name": "韦达定理"},  # 有名无 id（niche，锚不到叶子）
        "secondary_kps": [], "qtype": "解答", "exam_type": "直接计算",
        "skeleton": ["设两根", "韦达"], "tags": ["韦达定理"], "models": [],
    }
    leaf_pool = [("3082002009001", "完全不相关的叶子A"), ("3082002009002", "完全不相关的叶子B")]
    mother_dna = {"stem": "韦达母题", "answer": "x1+x2=...", "solution_skeleton": ["韦达"],
                  "mother_solve_source": "opus", "dna": dict(prev_dna)}

    # endorsed → 走放行（无 awaiting_mother_confirm，confirmed=True）
    out_e = asyncio.run(label_mod._reanchor_reuse_first_solve(
        state={**state_base, "mother_endorsed": True},
        analysis=dict(state_base["analysis"]), mother_dna=dict(mother_dna), prev_dna=dict(prev_dna),
        grade_code="3082", chapter_id="3082002", leaf_pool=leaf_pool,
        confirmed_chapter_id="3082002", include_review_books=False, knobs=None,
    ))
    ok(not out_e.get("awaiting_mother_confirm"),
       "endorsed + 锚不到叶子 → 不弹 awaiting_mother_confirm（不回 picker、不死循环）")
    ok(out_e.get("mother_confirmed") is True,
       "endorsed + 降级 → confirmed=True（按所选章范围出题·待人审，永有出边）")
    ok(out_e.get("mother_endorsed") is True, "endorse 终态随降级态落 state（下轮不再反复弹）")

    # 非 endorsed 首次冲突 → 既有「拦一次」闸（弹 awaiting_mother_confirm）行为不破。
    out_n = asyncio.run(label_mod._reanchor_reuse_first_solve(
        state={**state_base},  # 无 endorse、无 _bug03_gated_chapter
        analysis=dict(state_base["analysis"]), mother_dna=dict(mother_dna), prev_dna=dict(prev_dna),
        grade_code="3082", chapter_id="3082002", leaf_pool=leaf_pool,
        confirmed_chapter_id="3082002", include_review_books=False, knobs=None,
    ))
    ok(out_n.get("awaiting_mother_confirm") is True,
       "非 endorsed 首次冲突 → 仍走「拦一次」闸（既有 bounded 行为不破·防回归）")


# ---------------------------------------------------------------------------
# ③ 降级态打标：_bounded_degrade_to_chapter 产 tags + skeleton 非空 + 映射进卡
# ---------------------------------------------------------------------------
def test_degraded_label_tags():
    banner("③ 降级态打标（AC6）：minimal DNA 也产 tags+解法骨架非空 + tags 映射进母题卡")
    state_base = copy.deepcopy(_MOTHER_OK)
    state_base["messages"] = []
    # 无首解残片：mother_dna 几乎空（minimal DNA 路径），但有点解答文本供 skeleton 兜底。
    mother_dna = {"stem": "难图母题", "answer": "答案约为 46", "dna": {}}
    out = label_mod._bounded_degrade_to_chapter(
        state={**state_base, "mother_endorsed": True},
        analysis={"grade": {"value": "八年级下学期", "code": "3082"},
                  "kp": {"value": "二次函数"}},
        mother_dna=dict(mother_dna),
        grade_code="3082", confirmed_chapter_id="3082003", chapter_text="第3章 一元二次方程", knobs=None,
    )
    dna = (out.get("mother_dna") or {}).get("dna") or {}
    tags = dna.get("tags") or []
    ok(len(tags) > 0, f"降级态 dna.tags 非空（{tags}）= 降级也打完标（治标签空·AC6）")
    sk = dna.get("skeleton")
    sk_nonempty = bool(sk) if isinstance(sk, list) else bool(str(sk or "").strip())
    ok(sk_nonempty, f"降级态 dna.skeleton 非空（{sk}）= 解法骨架不空（AC6）")
    ok(out.get("mother_endorsed") is True, "降级态 endorse 终态落 state")

    # tags 映射没丢：_build_mother_card 把 dna.tags 透到母题卡 dna.tags（FE pickDna 读它）。
    card = _build_mother_card({**state_base, **out})
    card_tags = ((card or {}).get("dna") or {}).get("tags") or []
    ok(len(card_tags) > 0 and card_tags == [str(t) for t in tags if str(t).strip()],
       f"tags 映射进母题卡 dna.tags（{card_tags}）= 打标 tags→卡 映射没丢（AC6）")


# ---------------------------------------------------------------------------
# ④ G5 守边界断言：meta 编辑后 figure_url 仍在 + messages 不暴涨
# ---------------------------------------------------------------------------
def test_g5_boundary():
    banner("④ G5 守边界（评审硬加）：meta 编辑后变式 figure_url 仍在 + messages 不暴涨")
    st = _variants("加个易错标签")
    fig_before = [it.get("figure_url") for it in st["items"]]
    # 直接调 apply_tool（加标签 meta，1-based index=1）——薄包 edit_dna_state，不重解不重出。
    update, _edited, err = apply_tool("加标签", st, 1, "易错")
    ok(err is None, f"加标签 apply_tool 无错（err={err}）")
    new_items = update.get("items") or []
    fig_after = [it.get("figure_url") for it in new_items]
    ok(fig_after == fig_before and all(fig_after),
       f"G5①: 改即时维(加标签)后变式 figure_url 全在（{fig_after}）= 不波及变式 stem·不丢图")
    # tags 真加上 + 旧标签留
    new_tags = (update.get("mother_dna") or {}).get("dna", {}).get("tags") or []
    ok("易错" in new_tags and "解方程" in new_tags, f"加标签: tags 原地加且旧标签保留（{new_tags}）")
    # G5②：meta 编辑产生的 messages 不暴涨（edit_dna_state 不发解题/重出气泡，至多 0~1 条）。
    msgs = update.get("messages") or []
    ok(len(msgs) <= 1, f"G5②: meta 编辑 messages 不暴涨（len={len(msgs)}，不重解/不重出累积）")


if __name__ == "__main__":
    test_endorse_closure()
    test_anchor_degrade_no_loop()
    test_degraded_label_tags()
    test_g5_boundary()
    banner(f"汇总: PASS={PASS}  FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
