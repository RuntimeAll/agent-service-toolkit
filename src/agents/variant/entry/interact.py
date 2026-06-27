"""variant 引擎 · entry/interact.py 多轮交互节点（PRD-C-104 B5 抽出，纯搬零改）。

WAIT 后老师下一句的交互编排（解析意图 → 分诊 → 执行原语 → 收口）：
- parse_instruction：解析老师指令（五意图分诊：confirm/edit/qa/revise/solution-only）。
- route_after_parse / dispatch / route_dispatch：意图漏斗 + 三原语路由。
- exec_remove / exec_regenerate / exec_add / exec_reorder / exec_solution_only：编辑原语执行。
- _rewrite_solution_one：单题解析重写器（exec_solution_only 专属 helper）。
- editor_entry / after_editor_entry：结构化编辑/重生入口（PRD-A-021 S1）。
- patch / after_patch：母题字段就地改（只改不波及变式）。

🔴 行为零改：五意图分诊/三原语/编辑路由/守恒注记逐字保持，已知行为照搬不修。
🔴 图 wiring 节点名（parse_instruction/exec_*/editor_entry/patch + 条件边 route_after_parse/
   dispatch/route_dispatch/after_patch/after_editor_entry）不变 → 拓扑零改。
🔴 strangler：顶部 from agents.variant import 取依赖（prompt 常量 / shared helper / persist+stage
   re-export 的 helper）。interact re-export 置于 __init__.py **所有 re-export 之后、图 wiring 之前**
   → 调用/装载期依赖全就绪、无循环；_is_full_permutation/_reorder_items 等共享 helper 仍留 facade。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、所有 re-export 之后导入）
    ADD_PROMPT,
    GATE_CONCURRENCY,
    INTENT_CONFIRM,
    INTENT_EDIT,
    INTENT_QA,
    INTENT_REVISE,
    INTENT_SOLUTION_ONLY,
    PARSE_PROMPT,
    REGEN_PROMPT,
    SOLUTION_ONLY_PROMPT,
    VariantState,
    _ainvoke_text,
    _budget_bind,
    _budget_exhausted,
    _check_one_item,
    _conservation_clause,
    _context_block,
    _discard_drafts_best_effort,
    _draft_id_to_discard,
    _editor_op,
    _emit_artifact,
    _emit_stage,
    _fact_edit,
    _format_item_stem,
    _is_full_permutation,
    _latest_ai_text,
    _latest_human_text,
    _mother_facts,
    _normalize_generated_item,
    _parse_json,
    _qtype_from_note,
    _regen_max_tokens,
    _regen_once,
    _reorder_items,
    _sanitize_item,
    _sanitize_rich_text,
    edit_item_state,
    mark_content_dirty_if_persisted,
    regen_dirty_items,
    reverify_item_state,
    revise_item,
    settings,
    validate_instruction,
)


async def parse_instruction(state: VariantState, config: RunnableConfig) -> VariantState:
    """WAIT 下一句 → 判 5 意图 + 三层漏斗参数。本节点只解析、不改 items（路由后各分支执行）。

    解析结果塞 state['pending']（含 intent/ops/knobs/comp/extra_constraints/mother_correction）。
    """
    # 🔴 P13：编辑轮入口重置预算（parse→dispatch/patch→exec_*→gene_gate→solve_explain→assemble
    #   同轮共享）。parse 本身是核心分类调用（受预算约束但不被跳过）。
    budget = _budget_bind(state, reset_limit=settings.VARIANT_BUDGET_EDIT)
    utterance = _latest_human_text(state.get("messages", []))
    # BUG-006：注入上一轮 AI 消息（尤其主动提议句），让分类器能判承接语（「给学生讲」承接
    #   AI「我可以讲一遍第N题」→ 答疑，而非无状态单句分类掉 clarify）。截断防 prompt 膨胀。
    prev_ai = _latest_ai_text(state.get("messages", []))
    if len(prev_ai) > 600:
        prev_ai = prev_ai[:600] + "…"
    facts = _mother_facts(state)
    items = state.get("items") or []
    prompt = PARSE_PROMPT.format(
        n=len(items),
        kp_name=facts["kp_name"],
        grade=facts["grade"],
        prev_ai=prev_ai or "(无，这是本组第一轮交互)",
        utterance=utterance or "(空)",
    )
    # 🔴 17 号修复 §5：在途母题（停在 clarify、还没出题）的回答语境 —— 此时没有可编辑的
    # 题号，老师这句大概率是在答我们的澄清提问（年级/考点/题型）。显式钉死语境，防止
    # 分类器按「已有题组」惯性乱派编辑 op（编辑类 index 也会被 R3 护栏拦，双保险）。
    # ⚠ 覆盖必须盖过头部「硬守恒」段（真机踩过：答「9年级上」被判成撞守恒 clarify 驳回）：
    # 无题组语境下年级/考点不是守恒项，正是我们在求老师确认的待定项。
    if not items:
        prompt += (
            "\n\n【当前语境·最高优先级，覆盖上面所有规则】这一轮还没有出任何题（题组为空）——"
            "我们刚就母题的年级/考点/题型向老师提了澄清问题，老师这句话是**回答澄清**。"
            "此语境下上面『母题 DNA 硬守恒，撞它即 clarify 驳回』的规则**不适用**："
            "年级/考点正是待老师确认/纠正的项，不存在『改守恒』一说。\n"
            "判定规则（按此覆盖执行）：\n"
            "- 给出年级（如「这个是9年级上的题目」「八下的」）→ intent=修正，"
            "mother_correction.grade=规范化年级（如「九年级上学期」）。\n"
            "- 给出考点（如「考的是二次函数」）→ intent=修正，mother_correction.kp=该考点。\n"
            "- 同时给年级和考点 → 修正，两项都填。\n"
            "- 真说不清（与年级/考点/题型无关的闲聊）→ clarify。\n"
            "- 禁止输出任何编辑类 ops（无题可编），禁止判「确认/答疑」。"
        )
    # 🔴 P12.4（PRD-C-013）：parse 是受约束分类器（5 意图闭集 + 物理护栏兜底），nano 足够 →
    # 走 LLM_MODEL_LIGHT(gpt-5.4-nano) 降本；语义不动，c017 探针把关。闸A judge 不在本阶段切 nano。
    text = await _ainvoke_text(
        [HumanMessage(content=prompt)], model=settings.LLM_MODEL_LIGHT
    )
    parsed = _parse_json(text)
    # 🔴 物理护栏（G4/FP4）：白名单 + 越界钳制 + 解析失败整体降级 clarify（永不默认成 remove）
    pending = validate_instruction(parsed, len(items))
    pending["utterance"] = utterance
    return {"pending": pending, "llm_call_budget": budget, "messages": []}


def route_after_parse(
    state: VariantState,
) -> Literal["patch", "dispatch", "answer", "save", "solution_only", "ask_clarify"]:
    """parse 后分诊（设计 §3 mermaid）：修正→patch / 编辑→dispatch / 确认→save / 答疑→answer /
    解法修正→solution_only（整改3：题面不动只改解析）/ clarify。"""
    intent = (state.get("pending") or {}).get("intent")
    if intent == INTENT_SOLUTION_ONLY:
        return "solution_only"
    if intent == INTENT_REVISE:
        return "patch"
    if intent == INTENT_EDIT:
        return "dispatch"
    if intent == INTENT_CONFIRM:
        return "save"
    if intent == INTENT_QA:
        # 🔴 PRD-A-021 R4·F7：答疑前置空护栏 —— 无题组（items 空）时进 answer 会拿空题组白烧一次
        #   LLM（ANSWER_PROMPT 的 brief/detail 全空，答了个寂寞）。空题组 → 落 ask_clarify 让老师先
        #   贴图/出题，不进 answer。（save 侧 persist_to_bank 已自带空护栏:6664；此处补 answer 侧。）
        if not (state.get("items") or []):
            return "ask_clarify"
        return "answer"
    return "ask_clarify"


def dispatch(state: VariantState) -> Literal["remove", "regenerate", "add", "reorder", "ask_clarify"]:
    """三层漏斗收口：硬旋钮命中 → remove/regenerate/add/reorder；ops 空/说不清 → clarify。"""
    ops = (state.get("pending") or {}).get("ops") or []
    actions = {str(op.get("action")) for op in ops if isinstance(op, dict)}
    # 🔴 护栏 R7（validate_instruction）保证 ops 只含单一 action 类（混类已降级 clarify），
    # 此处仅作收口映射；优先级序留作防御（万一上游漏钳也不会静默丢弃后执行半截）。
    if "remove" in actions:
        return "remove"
    if "regenerate" in actions:
        return "regenerate"
    if "add" in actions:
        return "add"
    if "reorder" in actions:
        return "reorder"
    return "ask_clarify"


# --- 编辑·remove：删 + 重编号（删完的题不再过 solve，直接 ASM） -----------------
async def exec_remove(state: VariantState, config: RunnableConfig) -> VariantState:
    budget = _budget_bind(state)  # P13：携带编辑轮预算（remove 不调 LLM，仅透传给下游）
    items = list(state.get("items") or [])
    ops = (state.get("pending") or {}).get("ops") or []
    drop = set()
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "remove":
            idx = op.get("index")
            try:
                drop.add(int(idx) - 1)  # 1-based → 0-based
            except (TypeError, ValueError):
                pass
    kept = [it for i, it in enumerate(items) if i not in drop]
    # 🔴 shape_defects 只属于 generate 当轮：老师显式编辑 = 对配方的人工接管，旧缺陷清单
    #   不再陈旧外显（且不在 assemble 重算 —— 那会把老师主动删/换题误报为缺陷）。
    return {
        "items": kept,
        "pending": None,
        "shape_defects": [],
        # 🔴 改变题集 → 清手排标记（对抗审③）：删题后旧手排次序已失效，assemble 回默认难度序。
        "manual_order": False,
        "llm_call_budget": budget,
        "messages": [],
    }


# --- 编辑·regenerate：改造指定题 → 过 solve_explain（清 check 触发重判） ----------
async def exec_regenerate(state: VariantState, config: RunnableConfig) -> VariantState:
    """改造某道（按 note 软约束）→ 该题清 check 重入 solve_explain（每题状态须重新定）。"""
    budget = _budget_bind(state)  # P13：携带编辑轮预算（REGEN 调 LLM，记票）
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    pending = state.get("pending") or {}
    ops = pending.get("ops") or []
    # 🔴 PRD-A-021 R4·F5：改造约束别吞半截 —— pending 级 extra_constraints / comp（老师本轮
    #   说的「其余自由约束 + 对比/补充句」）原 exec_regenerate 完全丢弃（只用 per-target op.note）。
    #   比照 exec_add:6153 口径，折进每道重出题的额外要求（per-target note 仍叠加在前）。
    _pending_extra_bits = list(pending.get("extra_constraints") or [])
    if pending.get("comp"):
        _pending_extra_bits.append(str(pending["comp"]))
    pending_extra = "；".join(_pending_extra_bits)
    targets = []
    notes: dict[int, str] = {}
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "regenerate":
            try:
                t = int(op.get("index")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= t < len(items):
                targets.append(t)
                if op.get("note"):
                    notes[t] = str(op.get("note"))

    drafts_to_discard: list[Any] = []  # 🔴 PRD-A-022：被替换掉的旧草稿 id（未发布）→ 末尾软删
    for t in targets:
        old = items[t]
        # 🔴 F5：本题额外要求 = per-target note + pending 级 extra_constraints/comp（任一为空跳过）。
        extra_req = "；".join(s for s in (notes.get(t), pending_extra) if s)
        # BUG-001 AC1：note 里含「改成X题」→ 抽出目标题型覆盖 REGEN 的 qtype（REGEN_PROMPT 把
        #   qtype 钉死成入参，不覆盖则改题型形同没改）。抽不出 → 沿用原题型（行为不变）。
        target_qtype = _qtype_from_note(notes.get(t)) or old.get("qtype") or facts["qtype"]
        regen_text = await _ainvoke_text(
            [
                HumanMessage(
                    content=REGEN_PROMPT.format(
                        kp_name=facts["kp_name"],
                        grade=facts["grade"],
                        stem=(old.get("stem") or "")
                        + (f"\n额外要求：{extra_req}" if extra_req else ""),
                        level=old.get("level") or "normal",
                        qtype=target_qtype,
                        difficulty=old.get("difficulty") or 3,
                        injected_kp=json.dumps(old.get("injected_kp"), ensure_ascii=False),
                    )
                    # 🔴 整改1：确定上下文硬约束（重出某道也压解题不越界）。
                    + "\n\n"
                    + _context_block(facts)
                    # 🔴 W2 守恒硬约束注入（T1）：重出仍守白名单/考察类型/最难步基因。
                    + "\n\n"
                    + _conservation_clause(facts.get("dna"))
                )
            ],
            model=settings.variant_model("generate"),
        )
        regen = _parse_json(regen_text)
        if isinstance(regen, dict) and regen.get("stem"):
            # 🔴 新题清 check → 必过 solve_explain 才能进 assemble（不变量）
            new_item: dict[str, Any] = _sanitize_item(
                {
                    "stem": regen.get("stem"),
                    "answer": regen.get("answer"),
                    "solution": regen.get("solution"),
                    "qtype": regen.get("qtype") or target_qtype,
                    "difficulty": regen.get("difficulty") or old.get("difficulty"),
                    "level": regen.get("level") or old.get("level") or "normal",
                    "injected_kp": regen.get("injected_kp"),
                }
            )
            # 4a：重出稿自带的验算载荷随新题走（REGEN_PROMPT 契约产出）
            if isinstance(regen.get("verify_payload"), dict):
                new_item["verify_payload"] = regen["verify_payload"]
            # 配方印记 + 入库簿记跟题走（PRD-A-018 M1②：carry _seq/persisted/_persist_id；不 carry
            #   → 重出后该题 persisted/_persist_id 丢失 → 再入库走 create 重复落行）。
            # 🔴 PRD-A-022：**绝不 carry _draft_id**（不在白名单）——旧草稿被这道新题替换，新题
            #   无 _draft_id，由 assemble 重落新草稿；旧草稿（未发布）下面收集去 discard 软删。
            for k in ("_seq", "persisted", "_persist_id", "level"):  # R4·F17: 去死键 from_recipe/expected_difficulty
                if old.get(k) is not None:
                    new_item[k] = old[k]
            # 已入库题被改造重出 → 内容已变 → 标「内容已编辑待覆盖」，入库走 _persist_id 覆盖原行。
            mark_content_dirty_if_persisted(new_item)
            # 🔴 PRD-A-022：旧草稿（有 _draft_id 且未发布）被替换 → 收集软删（已发布题不 discard）。
            _did = _draft_id_to_discard(old)
            if _did is not None:
                drafts_to_discard.append(_did)
            # 🔴 RC2（PRD-C-013）：编辑轮产物打 from_edit 印记 → 重入 gene_gate 时只判不回炉
            # （老师已点名改造，基因闸判不过只标 warn，不重出覆盖老师意志）；老师 note 存 edit_note，
            # 任何下游 REGEN（闸B 验算回炉）都注回，不被失败原因冲掉。
            new_item["from_edit"] = True
            if t in notes:
                new_item["edit_note"] = notes[t]
            items[t] = new_item
    # 🔴 PRD-A-022·best-effort 软删被替换掉的旧草稿（绝不阻塞，token 缺/失败仅 log）。
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    await _discard_drafts_best_effort(drafts_to_discard, token)
    # 🔴 编辑轮清陈旧缺陷外显（同 exec_remove 注释）
    return {
        "items": items,
        "pending": None,
        "shape_defects": [],
        "llm_call_budget": budget,
        "messages": [],
    }


async def exec_add(state: VariantState, config: RunnableConfig) -> VariantState:
    """补题（设计 §5 三层漏斗②：超旋钮软约束 best-effort 吸收 + 外显）→ 追加 → 过 solve_explain。"""
    budget = _budget_bind(state)  # P13：携带编辑轮预算（ADD 调 LLM，记票）
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    pending = state.get("pending") or {}
    ops = pending.get("ops") or []
    n = 0
    notes = []
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "add":
            try:
                n += int(op.get("count") or 1)
            except (TypeError, ValueError):
                n += 1
            if op.get("note"):
                notes.append(str(op.get("note")))
    if n <= 0:
        n = 1
    n = min(n, 5)  # 单轮补题上限，防失控

    extra_bits = notes + list(pending.get("extra_constraints") or [])
    if pending.get("comp"):
        extra_bits.append(str(pending["comp"]))
    extra = "；".join(extra_bits) or "无"

    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ADD_PROMPT.format(
                    n=n,
                    kp_name=facts["kp_name"],
                    grade=facts["grade"],
                    qtype=facts["qtype"],
                    stem=facts["stem"],
                    skeleton=facts["skeleton"],
                    extra=extra,
                )
                # 🔴 整改1：确定上下文硬约束（补题同样压解题不越界）。
                + "\n\n"
                + _context_block(facts)
                # 🔴 W2 守恒硬约束注入（T1）：补题同样守白名单/考察类型/最难步基因。
                + "\n\n"
                + _conservation_clause(facts.get("dna"))
            )
        ],
        model=settings.variant_model("generate"),
    )
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    for it in data or []:
        if not isinstance(it, dict):
            continue
        # 🔴 新题不带 check → 下游 solve_explain 必判。规整/净化走 generate 同一事实源
        # （_normalize_generated_item：富文本净化 + verify_payload 仅 dict 才携带，
        # 对抗审修复：编辑轮加题此前漏过 _sanitize_item 且永远拿不到 4a 载荷红利）。
        # 不落 from_recipe 印记：补题轮的题不受首轮配方（递增/题型配比）改判。
        items.append(_normalize_generated_item(it, facts))
    # 🔴 编辑轮清陈旧缺陷外显（同 exec_remove 注释）
    return {
        "items": items,
        "pending": None,
        "shape_defects": [],
        # 🔴 改变题集 → 清手排标记（对抗审③）：加题后旧手排次序不覆盖全部题，回默认难度序。
        "manual_order": False,
        "llm_call_budget": budget,
        "messages": [],
    }


async def exec_reorder(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 指令排序（P9）：按 order（1-based 全排列，护栏已保证合法）纯代码重排 items。

    零 LLM、零改题——只挪槽位（check/gene/persisted 等簿记字段跟题走、不错位）；seq 由
    _artifact_payload 按新 list 序现编（index=i+1）。**不过 assemble**（避免默认难度升序排序
    覆盖老师手排）→ 整帧重发 artifact + 简短确认，入库顺序 = 当前显示序。
    护栏失守（order 缺失/长度不符等罕见兜底）→ 原序不动、友好提示，绝不抛、绝不丢题（G5）。
    """
    budget = _budget_bind(state)  # P13：reorder 不调 LLM，仅透传预算
    items = list(state.get("items") or [])
    ops = (state.get("pending") or {}).get("ops") or []
    order: list[int] | None = None
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "reorder" and isinstance(op.get("order"), list):
            order = op["order"]
            break
    # 护栏兜底：order 非全排列（validate_instruction 应已拦，此处纯防御）→ 原序不动
    if not order or not _is_full_permutation(order, len(items)):
        _emit_artifact({**state, "items": items})
        return {
            "items": items,
            "pending": None,
            "shape_defects": [],
            "llm_call_budget": budget,
            "messages": [AIMessage(content="我没拿准你要的新次序（请给出覆盖全部题的完整顺序），这次先没调整。")],
        }
    reordered = _reorder_items(items, order)  # 1-based → 0-based 取位（公共纯函数）
    _emit_artifact({**state, "items": reordered})  # 整帧重发（seq 按新序现编）
    return {
        "items": reordered,
        "pending": None,
        "shape_defects": [],
        # 🔴 跨轮 sticky（对抗审③修复）：置位后即便后续走 remove/regenerate 过 assemble，
        #   也不被默认难度升序排序覆盖手排（assemble 读 manual_order 跳过 _sort_by_difficulty）。
        "manual_order": True,
        "llm_call_budget": budget,
        "messages": [AIMessage(content=f"已按你给的次序重排这组题（共 {len(reordered)} 道），入库会按当前顺序。")],
    }


# ===========================================================================
# 🔴 整改3（2026-06-12）：解法修正 scope —— 题面保留，只按新约束重写每道题的解析。
#   实测痛点：「这里是7年级的题目，没学二元方程，只能用一元一次去解题」原被误判 修正→整组
#   重做（302s）。现走本节点：题面不动、逐题重写解析（带确定上下文块 + 新方法约束）+ 重跑闸B；
#   某题在新约束下根本无法求解（如必须二元才能解）才单题重出题面（_regen_once），其余题不动。
#   年级修正同步更新锚定事实（analysis.grade + 头部 chip），但**不触发整组重出**。
# ===========================================================================
# 单题解析重写器：题面/答案保留不动，只按新方法约束重写解析。判断本题在新约束下是否可解：
#   solvable=true → 给出新解析；solvable=false → 说明为何（如必须二元才能解）→ 调用方单题重出。


async def _rewrite_solution_one(
    item: dict[str, Any], facts: dict, method_constraint: str
) -> tuple[dict[str, Any] | None, str | None]:
    """单题解析重写（整改3）。返回 (solution_text, None)=可解已重写；(None, reason)=新约束下不可解。
    LLM 异常/解析失败 → (None, 降级 reason)，调用方按「保留原解析打 ⚠」或单题重出处理（G5 不抛）。"""
    prompt = (
        SOLUTION_ONLY_PROMPT.format(
            kp_name=facts["kp_name"],
            grade=facts["grade"],
            qtype=str(item.get("qtype") or facts["qtype"]),
            stem=str(item.get("stem") or ""),
            answer=str(item.get("answer") or ""),
            solution=str(item.get("solution") or ""),
            method_constraint=method_constraint,
        )
        # 🔴 整改1：确定上下文硬约束（进度/考点/教材版本），压解析不越界。
        + "\n\n"
        + _context_block(facts)
    )
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)],
            max_tokens=_regen_max_tokens(),
            model=settings.variant_model("generate"),
        )
    except Exception:  # noqa: BLE001 — 重写是增强，LLM 异常 → 降级（G5）
        return None, "解析重写调用失败"
    parsed = _parse_json(text)
    if not isinstance(parsed, dict):
        return None, "解析重写结果无法解析"
    if parsed.get("solvable") is False:
        return None, str(parsed.get("reason") or "新方法约束下无法求解")
    sol = parsed.get("solution")
    if not sol or not str(sol).strip():
        return None, "解析重写未给出有效解析"
    return {"solution": _sanitize_rich_text(str(sol))}, None


async def exec_solution_only(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 整改3·解法修正节点：题面保留，逐题按新方法约束重写解析 + 重跑闸B；某题新约束下不可解 →
    单题重出题面（_regen_once 注 edit_note）。年级修正同步 analysis.grade（chip 更新）+ 重锚 facts，
    **不触发整组重出**。降级铁律：任何环节失败 → 保留原题打 ⚠，绝不卡死、绝不空轮。"""
    budget = _budget_bind(state)  # 携带编辑轮预算（每题解析重写 / 单题重出各记票）
    pending = state.get("pending") or {}
    method_constraint = str(pending.get("method_constraint") or "").strip()
    grade_correction = str(pending.get("grade_correction") or "").strip()
    items = [dict(it) for it in (state.get("items") or [])]
    if not items or not method_constraint:
        # 无题 / 无约束（护栏应已拦）→ 退化回问，不空转
        return {
            "pending": None,
            "messages": [
                AIMessage(content="我没拿准你要怎么改解法（能再说一次只能用什么方法吗？），这次先没改。")
            ],
        }

    update: VariantState = {"pending": None, "llm_call_budget": budget}
    analysis = dict(state.get("analysis") or {})
    # ① 年级修正：更新锚定事实（chip 跟头部 facts 走）。⚠ 只改年级文案/置信，不清 anchored（不重锚
    #    整组）——解法修正不换考点，年级是为了让确定上下文块的「进度边界」对，不触发 classify。
    # 🔴 批3·老师指令来源：经 setter 写（locked 也放行 + 记 audit）；这里不重锚故不解冻。
    if grade_correction:
        audit = list(state.get("facts_audit") or [])
        _fact_edit(
            analysis, "grade", grade_correction, source="teacher",
            locked=bool(state.get("facts_locked")), audit=audit,
            instruction=pending.get("utterance") or "", confidence=0.9,
            clear_keys=("code",),
        )
        update["analysis"] = analysis
        update["facts_audit"] = audit

    facts = _mother_facts({**state, "analysis": analysis})

    _emit_stage("solution", "按新方法重写解析", "running", f"共 {len(items)} 道")

    async def _one(i: int, it: dict[str, Any]) -> dict[str, Any]:
        sol, reason = await _rewrite_solution_one(it, facts, method_constraint)
        if sol is not None:
            # 可解 → 题面不动，仅换解析；清 check 重跑闸B（解析变了，验算结论须刷新）
            new_it = dict(it)
            new_it["solution"] = sol["solution"]
            new_it["from_edit"] = True  # 闸B 见 from_edit：FAIL 不回炉换题、保留打 ⚠ 交人审
            new_it["edit_note"] = f"解题方法约束：{method_constraint}"
            new_it.pop("check", None)
            # 🔴 PRD-A-022：解析已变 → 旧草稿内容失效，strip _draft_id（dict(it) 复制带过来的）→
            #   assemble 重落新草稿（带新解析）；旧草稿（未发布）由末尾 discard 软删。已发布题不动 _draft_id 本就无。
            new_it.pop("_draft_id", None)
            # 🔴 PRD-A-018 F16：已入库题重写解析后内容已变 → 标「内容已编辑待覆盖」，
            #   否则 persist_to_bank「已入库就跳过」→ 新解析永不落库（F16 根）。
            mark_content_dirty_if_persisted(new_it)
            rechecked, _dropped = await _check_one_item(new_it, facts, i, len(items))
            out_it = rechecked if rechecked is not None else new_it
            # _check_one_item 可能返回新 dict（复制 new_it）→ 重申标记不被冲掉。
            mark_content_dirty_if_persisted(out_it)
            return out_it
        # 不可解 → 仅此单题重出题面（_regen_once 注 edit_note 守新约束）→ 重跑闸B
        if _budget_exhausted():
            # 预算耗尽 → 不单题重出，保留原题打 ⚠ 注记（降级，不卡死）
            warn_it = dict(it)
            warn_it["structure_lint"] = {"badge": "warn", "defects": [f"新方法约束下原题难解：{reason}"]}
            return warn_it
        _emit_stage(
            "solution", "按新方法重写解析", "warn",
            f"第 {i + 1} 题原题面在新方法下难解，单题重出中",
        )
        seed = dict(it)
        seed["edit_note"] = f"必须可用以下方法求解：{method_constraint}"
        seed["from_edit"] = True
        draft = await _regen_once(seed, facts, feedback=f"原题在新方法约束下难解：{reason}")
        if not draft:
            warn_it = dict(it)
            warn_it["structure_lint"] = {"badge": "warn", "defects": [f"新方法约束下原题难解且重出失败：{reason}"]}
            return warn_it
        draft["from_edit"] = True
        draft["edit_note"] = seed["edit_note"]
        for k in ("_seq", "persisted", "_persist_id", "level"):  # R4·F17: 去死键 from_recipe/expected_difficulty
            if it.get(k) is not None:
                draft[k] = it[k]
        draft.pop("check", None)
        # 🔴 F16：已入库题在新方法约束下重出题面 → 内容已变 → 标待覆盖（入库走 _persist_id 覆盖）。
        mark_content_dirty_if_persisted(draft)
        rechecked, _dropped = await _check_one_item(draft, facts, i, len(items))
        out_it = rechecked if rechecked is not None else draft
        mark_content_dirty_if_persisted(out_it)
        return out_it

    sem = asyncio.Semaphore(GATE_CONCURRENCY)

    async def _guarded(i: int, it: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _one(i, it)

    new_items = await asyncio.gather(*[_guarded(i, it) for i, it in enumerate(items)])
    for it in new_items:
        _format_item_stem(it)
    # 🔴 PRD-A-022·best-effort 软删被替换掉的旧草稿：某题旧有未发布 _draft_id 但新题已无 _draft_id
    #   （= 解析重写/单题重出，草稿内容失效）→ 软删旧草稿。warn 兜底分支保留原题原 _draft_id →
    #   新题仍带 _draft_id → 不入收集，不误删。assemble 会对无 _draft_id 的新题重落新草稿。
    drafts_to_discard = [
        _draft_id_to_discard(old)
        for old, new in zip(items, new_items)
        if _draft_id_to_discard(old) is not None and not new.get("_draft_id")
    ]
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    await _discard_drafts_best_effort(drafts_to_discard, token)
    _emit_stage("solution", "按新方法重写解析", "done", f"{len(new_items)} 道")
    # 🔴 解法修正改了题集内容（解析/个别题面）→ 清陈旧缺陷外显，回 assemble 收口快照
    update["items"] = list(new_items)
    update["shape_defects"] = []
    # 🔴 PRD-A-021 R4·F8：解法修正只改解析/原位重出、不剔题，且本节点直连 assemble（绕过
    #   solve_explain 那处的 dropped_notes 复位）。若不在此复位，上一轮 generate/验算遗留的
    #   dropped_notes 会被 assemble 当本轮「剔除 N 道」误渲染进头部。本轮无剔除 → 显式清空。
    update["dropped_notes"] = []
    return update


# ===========================================================================
# 🔴 PRD-A-021 S1·editor_entry：把「结构化编辑/重生」收进 graph，让真状态帧（程序验算等）经
#   stream runtime 真发出（治 F1：通道B aget_state→aupdate_state 在 graph 外 → _emit_stage 静默
#   no-op → 状态条/思路条丢帧）。
#
# 设计要点（evidence 订正后）：
#   - 本节点**自身只应用 op、清 check**（不在此发完成帧）；真「程序验算」帧来自下游 solve_explain
#     （editor_entry → solve_explain 边）。故 op 应用后凡内容变动的题必须 pop("check")，让 solve_explain
#     重判并发 verify 帧；reorder/纯元数据改这类不需重验的，本节点直接收口（不串验算）。
#   - 复用通道B 既有纯逻辑（revise_item / regen_dirty_items / edit_item_state / reverify_item_state），
#     零业务分叉（单一事实源）。
#   - 🔴 verify-one（无状态纯验算，不依赖 thread state）**不**并入 graph——它在 route 层就被 _editor_op
#     的 kind 白名单排除（白名单不含 'verify-one'）。
#   - 降级：op 缺字段/越界/helper 报错 → 不抛、不空轮，返回 messages 友好提示 + 不动 items（G5）。
async def editor_entry(state: VariantState, config: RunnableConfig) -> VariantState:
    """结构化编辑/重生入口节点：按 config.editor_op 应用对应纯逻辑 → 清 check（内容变动题）→
    交下游 solve_explain 重验发真帧。kind: revise / regen / edit-item / reverify。"""
    op = _editor_op(config) or {}
    kind = op.get("kind")
    try:
        if kind == "revise":
            # 单题有界 LLM 锚定重做（skeleton/scene/whole）；whole 走 REGEN，内部已清 check。
            update, _result, error = await revise_item(
                state, int(op.get("index")), str(op.get("target") or "whole"),
                str(op.get("instruction") or ""), config,
            )
        elif kind == "regen":
            # 手动重生待重生集合（None/空=全 dirty 集合；显式 indexes=force 整题重出，helper 内判）。
            # helper 内部已跑闸B + 清 dirty + 清 check（_check_one_item）。
            idxs = op.get("indexes")
            # 🔴 PRD-A-022：透传 token → regen_dirty_items 软删被替换掉的旧草稿（best-effort）。
            _regen_token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
            update, _result, error = await regen_dirty_items(
                state, idxs if isinstance(idxs, list) and idxs else None, token=_regen_token,
            )
        elif kind == "edit-item":
            # 单题手动编辑（零 LLM）：edit_item_state 已把 check 置 manual 中性（待 reverify）。
            update, _item, error = edit_item_state(
                state, int(op.get("index")), stem=op.get("stem"),
                answer=op.get("answer"), solution=op.get("solution"),
            )
        elif kind == "reverify":
            # 单题重跑闸B（清 check → _check_one_item）；这条本身就是验算，下游 solve_explain
            # 见已带 check 的题会跳过（不重复判），但本节点在 graph 内 → _check_one_item 的 stage 帧真发。
            update, _item, error = await reverify_item_state(state, int(op.get("index")))
        else:  # 不该到（route 白名单已挡）→ 不动 items，回问
            return {"messages": [AIMessage(content="我没拿准这次要怎么编辑，这次先没改。")]}
    except (TypeError, ValueError) as e:  # index 非 int 等参数问题 → 降级不抛（G5）
        return {"messages": [AIMessage(content=f"这次编辑参数不对（{e}），没有改动。")]}

    if error:
        # 越界/非法 target 等 → 友好提示，不动 items（与端点 400 同语义，graph 内不抛）
        return {"messages": [AIMessage(content=f"这次编辑没能执行：{error}")]}
    return update or {"messages": []}


def after_editor_entry(state: VariantState) -> Literal["solve_explain", "done"]:
    """editor_entry 出口：有题且存在未判（清了 check）题 → solve_explain 重验发真帧；否则收口 END。

    🔴 凡进 items 的题一律须带 check（不变量）。本路由保证编辑后「清了 check」的题必经 solve_explain
    重新定状态（同时在 graph 内发「程序验算」真帧）；纯 reverify（已重判带 check）或无题 → 直接 done。"""
    items = state.get("items") or []
    if not items:
        return "done"
    if any(isinstance(it, dict) and not it.get("check") for it in items):
        return "solve_explain"
    return "done"


# --- 修正：patch 母题字段 → 只重算受影响下游（设计 §6 中途修正） ----------------
async def patch(state: VariantState, config: RunnableConfig) -> VariantState:
    """老师纠正年级/考点 → patch analysis；改年级/考点 → 清 items 触发重锚+重造（route_after_patch）。

    粒度 = 步边界：改年级/考点 = 重走 classify→generate（清 items + mother_confirmed）；
    其余（如只是补充场景偏好）= 当软约束，留待下次编辑指令，不在此重造。
    """
    analysis = dict(state.get("analysis") or {})
    pending = state.get("pending") or {}
    corr = pending.get("mother_correction") or {}
    instruction = pending.get("utterance") or ""
    # 🔴 批3·老师指令来源：经 setter 写（source="teacher" → locked 也放行 + 记 audit）。
    audit = list(state.get("facts_audit") or [])
    locked = bool(state.get("facts_locked"))
    changed = False

    if corr.get("grade"):
        # 清旧编码，重锚时再填
        changed |= _fact_edit(
            analysis, "grade", corr["grade"], source="teacher", locked=locked,
            audit=audit, instruction=instruction, confidence=0.9, clear_keys=("code",),
        )
    if corr.get("kp"):
        # 清旧锚定，classify 会重锚
        changed |= _fact_edit(
            analysis, "kp", corr["kp"], source="teacher", locked=locked,
            audit=audit, instruction=instruction, confidence=0.9, clear_keys=("anchored",),
        )

    if not changed:
        # 没拿到可 patch 的字段 → 退化为 clarify 回问（不空转）
        return {
            "messages": [
                AIMessage(content="我没听准你要修正什么（年级还是考点？），可以再说一次吗？")
            ],
            "pending": None,
        }

    # 🔴 改了硬锚 → 清 items + mother_confirmed，触发重锚(classify)+重造(generate)
    # artifact 同步快照（PRD-C-011）：items 已清，但后续若走 clarify（重锚置信不足）或
    # generate 裸奔兜底（无 items）会绕开 assemble 不再发帧 → 先发空快照让 FE 右栏回
    # 空态，避免老师对着 agent 端已不存在的旧题组卡片点「第N题重出」（UI/状态错位）。
    _emit_artifact({**state, "items": [], "analysis": analysis})
    # 🔴 批3：改硬锚 → 解冻（facts_locked=False），让 classify 重锚（重锚是合法锚定路径）；
    #   老师本次修正的 audit 留痕随 state 带走。
    # 🔴 PRD-A-021 R2a·F2（与 B5b 同 PR）：patch 改**年级**时必清 confirmed_chapter_id。否则
    #   「先确认章（state.confirmed_chapter_id 落了旧章）→ 再文字纠正年级」序列下，classify 仍读到旧
    #   state.confirmed_chapter_id（variant.py 确认章驱动 grade_code）→ 用**旧章前 4 位**当年级册前缀 →
    #   锚错册（老师纠正的新年级被旧章覆盖吞掉）。改年级 = 旧确认章作废，清掉它让 classify 退回按
    #   patch 后的 analysis.grade 归一 grade_code（_resolve_grade_code），新年级真正生效。
    #   🔴 与 B5b 先后顺序：本清在 patch 出口、先于下游 classify 读 confirmed_chapter_id；B5b 的 fp
    #   比较读的是 mother_dna._solve_range_fp（首解写的），与 confirmed_chapter_id 是两个独立字段，
    #   清这个不动那个，互不吞改。改年级后 confirmed_chapter_id=None → classify _reuse_ok 因
    #   confirmed_chapter_id 为空而 False → 走全量重 solve（新册重解，正确）。
    _patch_clear: dict[str, Any] = {}
    if corr.get("grade"):
        _patch_clear["confirmed_chapter_id"] = None
        _patch_clear["_bug03_gated_chapter"] = None  # 章语境作废，连带清闸3 标记
    return {
        "analysis": analysis,
        "items": [],
        "mother_confirmed": False,
        "facts_locked": False,
        "facts_audit": audit,
        "pending": None,
        **_patch_clear,
        # 🔴 M5/PRD-A-018：patch 改 grade/kp 清 items 后，流程实际硬停在母题卡 review（B5 硬停闸，
        #   after_patch→classify→gate_after_classify pinned→await_review），并不自动重出。旧文案承诺
        #   「重出这组变式」与硬停语义打架（两气泡矛盾）。改为与 await_review 对齐：只说重锚完、母题卡已更新，
        #   重出须老师确认无误后显式点「开始举一反三」。
        "messages": [
            AIMessage(
                content="已按新的年级/考点重锚，母题卡已更新；确认无误后点「开始举一反三」重新生成变式。"
            )
        ],
    }


# 三层漏斗：编辑意图 → 选 remove/regenerate/add（route_after_parse 的 "dispatch" 实由本函数收口）
def route_dispatch(
    state: VariantState,
) -> Literal["exec_remove", "exec_regenerate", "exec_add", "exec_reorder", "ask_clarify"]:
    target = dispatch(state)
    return {
        "remove": "exec_remove",
        "regenerate": "exec_regenerate",
        "add": "exec_add",
        "reorder": "exec_reorder",
        "ask_clarify": "ask_clarify",
    }[target]


# patch：改了硬锚（清 items + mother_confirmed=False）→ 重锚重造走 classify；
#        没改（仅回问消息，items 仍在）→ END 等下一句。
def after_patch(state: VariantState) -> Literal["classify", "done"]:
    if state.get("mother_confirmed") is False and not state.get("items"):
        return "classify"
    return "done"
