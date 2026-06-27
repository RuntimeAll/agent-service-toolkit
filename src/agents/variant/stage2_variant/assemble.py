"""variant 引擎 · stage2_variant 题组装配层（PRD-C-104 B3b 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出收尾节点 assemble
（题组快照 + 难度排序 + 一致性自检 + 外显）。

🔴 行为零改：grade_variant_item 从同包 difficulty 取（B3a 已抽）；
其余依赖从 facade agents.variant 运行期取。__init__.py 末尾 re-export assemble。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.variant.stage2_variant.difficulty import grade_variant_item  # noqa: E402
from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾导入）
    DIFFICULTY_CAP,
    VariantState,
    _autodraft_items,
    _budget_bind,
    _emit_artifact,
    _emit_stage,
    _facts_log,
    _format_item_stem,
    _mother_facts,
    _sort_by_difficulty,
    _status_summary,
    _to_int,
    difficulty_consistency_defects,
    knobs_desc,
)


async def assemble(state: VariantState, config: RunnableConfig) -> VariantState:
    """题组摘要（P1 聊天瘦身·PRD-C-012）：左栏只发摘要头——配方/守恒DNA/状态计数/旋钮提示；
    题干/答案/解析全文**只走右栏 artifact 题卡**，聊天流不再复读（用户拍板 2026-06-11）。

    🔴 整改2（2026-06-12，推翻此前 P8「独立 rubric 复评」设计）：难度判定并入生题——
    GENERATE/REGEN/ADD 出题 prompt 已嵌入四档 rubric，每道题的 difficulty 由出题调用同步产出，
    assemble **不再独立走一轮 _grade_difficulty 复评**（省掉首轮 + 整组重做各一次 nano 调用）。
    缺/非法 difficulty 沿用入库口径兜底（_clamp 到 1~4，缺→2）。P9 排序在下面做。"""
    budget = _budget_bind(state)  # P13：轮内下游节点携带预算（assemble 已无独立难度调用）
    # 🔴 PRD-C-103 WS1·AC1（2026-06-26，推翻整改2「难度随生题 LLM rubric 产出」）：变式难度档来源
    #   改为 **grade_observed 确定性判档**（继承母题锚定模型表 tier/freq + 该变式自抽 K/R/D），
    #   砍掉 generate 内嵌 rubric 的 LLM 自评数字（item 原 difficulty 不再被采信）。难度永不取 LLM 自评
    #   （铁律：pass/fail 只读 grade_observed）。降级（铁律④）：判档异常 → 回退 item 原 difficulty 钳
    #   1~4、缺→2，绝不卡死。账单挂 item['difficulty_bill']（AC9 trace 的 actual_level 读它）。
    mother_dna = state.get("mother_dna") or {}
    items = [dict(it) for it in (state.get("items") or [])]
    for it in items:
        try:
            bill = grade_variant_item(it, mother_dna)
            lvl = bill.get("level")
            if isinstance(lvl, int) and 1 <= lvl <= DIFFICULTY_CAP:
                it["difficulty"] = lvl
                it["difficulty_bill"] = bill  # 确定账单留痕（modelHits/K/R/D/rule），供 trace/外显
            else:
                d = _to_int(it.get("difficulty"))
                it["difficulty"] = max(1, min(DIFFICULTY_CAP, d)) if d is not None else 2
        except Exception:  # noqa: BLE001 — 判档是增强不是关卡，异常回退原值（铁律④/G5）
            d = _to_int(it.get("difficulty"))
            it["difficulty"] = max(1, min(DIFFICULTY_CAP, d)) if d is not None else 2
    # 🔴 P9 默认序（PRD-C-013）：assemble 前按总评难度**升序稳定排序**（同难度保持生成序）。
    #   seq 由 _artifact_payload/入库按当前 list 序现编（index=i+1），persisted 簿记跟 item 走
    #   （persisted 是 item 字段，排序不丢、不错位）。指令排序走 exec_reorder（纯代码重排）。
    # 🔴 跨轮 sticky（对抗审③修复）：老师手排过（state.manual_order）→ 跳过默认排序，否则
    #   手排后再走 remove/add/regenerate 路径过 assemble 会被静默重排，手排（及入库序）丢失。
    #   exec_add/exec_remove 改了题集会清掉该标记（次序失效回默认）；regenerate 原位改单题不清。
    if not state.get("manual_order"):
        items = _sort_by_difficulty(items)
    # 🔴 题型模版自动规范（PRD-C-009·BE）：题组定稿收口处统一规范 stem（在 solve_explain
    #   sympy/structure_lint 判决**之后**，绝不影响判决）。assemble 是 generate/add/regenerate/
    #   remove 四路的唯一收口（exec_* → gene_gate/solve_explain → assemble），在此规范一次即覆盖
    #   全部产新题/删题路径；幂等 → 多轮经过 assemble 不漂移。规范文本落进 state.items，
    #   随快照上屏（_emit_artifact）+ 入库（persist_to_bank/build_create_bo 读 item.stem）+ 会话恢复。
    for it in items:
        _format_item_stem(it)
    state = {**state, "items": items}  # 覆盖后的 difficulty + 排序后的序随 state 流给快照/入库
    facts = _mother_facts(state)

    # 🔴 PRD-A-022 批1·自动落草稿：题组定稿即逐题落「草稿」(status=0)，让「换一批/全部入库」走
    #   草稿生命周期（入库=promote 0→1 不重落；换一批=discard 0→2 软删）。
    #   - 幂等：item 已有 _draft_id 或 persisted → 跳过；母题 mother_question_id 已有 → 跳过。
    #   - 🔴 best-effort 铁律：落草稿**失败绝不阻塞题组展示** —— try/except 全包，失败只 emit warn
    #     stage + log，items/artifact 照常返回（题组必须照常显示）。回写 _draft_id 经 merge_items
    #     reducer（_ITEMS_PRESERVE_ALWAYS 已含 _draft_id）跨轮安全续上。
    #   🔴 仅在有登录老师 token 时落草稿（草稿须归属老师；无 token 的 regression/直连脚本不落草稿，
    #     入库时走「兜底 create status=1 直接发布」，行为对齐旧链路、且测试零网络）。
    mother_qid_update: dict[str, Any] | None = None
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    if token:
        try:
            drafted_items, new_mother_qid, _ok = await _autodraft_items(items, facts, token)
            items = drafted_items
            if new_mother_qid is not None and not (state.get("mother_dna") or {}).get("mother_question_id"):
                mother_qid_update = dict(state.get("mother_dna") or {}, mother_question_id=new_mother_qid)
                state = {**state, "mother_dna": mother_qid_update}
            state = {**state, "items": items}  # 回写带 _draft_id 的 items → 快照/返回都带草稿态
        except Exception as e:  # noqa: BLE001 — 落草稿失败绝不阻塞题组展示（best-effort 铁律）
            _emit_stage("persist", "存草稿", "warn", "草稿暂存失败，不影响出题")
            _facts_log.warning(f"assemble autodraft failed (best-effort, group still shown): {e}")

    # 配方外显：有 knobs → "按你的要求: ..."；无 → 旧默认文案（行为不变）
    desc = knobs_desc(state.get("knobs"))
    recipe_s = f"按你的要求：{desc}" if desc else "配方：默认 3 = 2 普通 + 1 难"
    # 代码级配方校验缺陷（generate 整组 retry 1 次后仍不符）→ 头部外显 ⚠，不拦截
    # 🔴 P12.1：难度一致性（纯函数组内相对关系，零 LLM）并入配方缺陷外显，只 warn 不回炉。
    defects = list(state.get("shape_defects") or [])
    defects += difficulty_consistency_defects(items)
    defect_s = ("\n\n⚠ 配方未完全满足：" + "；".join(defects)) if defects else ""
    # 4d 方案A：被剔除题的摘要说明（过程已在思路条叙事，这里收口"本组为何少了"）
    dropped = state.get("dropped_notes") or []
    dropped_s = ("\n\n" + "；".join(dropped)) if dropped else ""

    head = (
        f"## 举一反三 · {len(items)} 道变式（{recipe_s}）{defect_s}{dropped_s}\n\n"
        f"**母题 DNA**：考点「{facts['kp_name']}」· 年级「{facts['grade']}」· 题型「{facts['qtype']}」（硬守恒）\n\n"
        f"**状态**：{_status_summary(items)}\n\n"
        "题目详情见右侧题卡。旋钮可拨：数量 / 数字 / 场景 / 难度 / 题型(可配比) / 解法。"
        "说「这组可以了」即入库。"
    )
    # artifact 快照帧（PRD-C-011）：每轮题组变化都过 assemble → FE 题卡每轮拿最新快照
    _emit_artifact(state)
    # 🔴 回写 items：P8 难度总评覆盖的 difficulty + P9 排序后的序必须落进 graph state，入库/快照才跟随
    #   ＋ PRD-A-022：回写 _draft_id（在 items 里）+ 母题草稿 id（mother_dna，仅本轮新落母题草稿时）。
    ret: VariantState = {"items": items, "llm_call_budget": budget, "messages": [AIMessage(content=head)]}
    if mother_qid_update is not None:
        ret["mother_dna"] = mother_qid_update
    return ret
