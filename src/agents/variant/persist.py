"""variant 引擎 · persist 入库 + state 操作 API 面（PRD-C-104 B5 抽出，纯搬零改）。

service.py 直调的入库 + 状态操作端点全在此（17 import 点的 persist/state-op 半边）：
- persist_to_bank / persist_one_to_bank：变式 + 解析经 RuoYi 写老师个人题库（只写不判）。
- _autodraft_items / _discard_drafts_best_effort / _draft_id_to_discard：草稿落/软删 helper。
- reorder_items_state / edit_item_state / set_item_figure_state / set_mother_figure_state
  / mark_item_manual_block_state / reverify_item_state / verify_one_stem / edit_dna_state
  / revise_item / regen_dirty_items / undo_regen_item：state 操作 API 面（service 逐个直 import）。

🔴 行为零改：簿记/血缘/owner/防脏闸/dirty 复位逐字保持，已知行为照搬不修。
🔴 strangler：本模块顶部 `from agents.variant import (...)` 运行期取依赖（LLM 出口 _ainvoke_text /
   SSE 帧 _emit_stage / RuoyiClient / persist_items / 难度桥 _grade_to_code 等）。
🔴 persist re-export 须置于 __init__.py 末尾、**stage1/stage2 re-export 之前**——因 stage2.assemble
   顶部 import _autodraft_items（本模块符号）。但本模块又用到 4 个 stage 子模块符号
   （_solve_one/_check_one_item 在 stage1_anchor.solve、_regen_once 在 stage2_variant.gates、
   _emit_artifact 在 stage1_anchor.mother_card），故这 4 个走**体内延迟 import**（调用期解析、破循环），
   不放顶部 from agents.variant import。
"""

from __future__ import annotations

import copy
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、所有依赖之后导入）
    DIFFICULTY_CAP,
    REVISE_FIELD_PROMPT,
    RuoyiClient,
    TIER_MANUAL,
    VariantState,
    _MOTHER_DIRTY_PROP_FIELDS,
    _REWRITE_SOLVE_PROMPT,
    _SOFT_REGEN_FIELDS,
    _ainvoke_text,
    _context_block,
    _dna_fact_edit,
    _emit_stage,
    _facts_log,
    _format_item_stem,
    _grade_to_code,
    _is_full_permutation,
    _is_proof_like,
    _machine_verify,
    _mother_facts,
    _parse_json,
    _reorder_items,
    _sanitize_rich_text,
    _to_int,
    clear_item_dirty,
    dirty_item_indexes,
    dna_extract,
    mark_content_dirty_if_persisted,
    mark_item_dirty,
    mark_mother_dirty,
    math_verify,
    persist_dirty_guard,
    persist_items,
    record_link_manifest,
    regen_class_of,
    settings,
    snapshot_item,
    variant_trace_block,
)


# --- 换一批/重生·软删旧草稿（PRD-A-022 批1）：被替换掉的旧草稿(status=0) → discard 0→2 -----
def _draft_id_to_discard(old_item: dict[str, Any]) -> Any:
    """🔴 重生/换题时取「应软删的旧草稿 id」：仅当 old 是**未发布草稿**（有 _draft_id 且 not persisted）。

    已发布题（persisted=True，走 _persist_id 覆盖原行语义）绝不 discard——返回 None。
    纯函数（可单测）：返回 old._draft_id 或 None。
    """
    if old_item.get("persisted"):
        return None
    return old_item.get("_draft_id") or None


async def _discard_drafts_best_effort(ids: list[Any], token: str | None) -> None:
    """🔴 best-effort 软删旧草稿（换一批/重生）：调 discard_drafts（owner+仅草稿双约束，传整组旧 id 安全）。

    绝不抛、绝不阻塞重生主链——失败只 log。token 缺 / ids 空 → no-op。
    BE discard 只改 status='0' 行，已发布(1)行天然不受影响（即便误传也无副作用）。
    """
    clean = [i for i in (ids or []) if i not in (None, "")]
    if not clean:
        return
    try:
        client = RuoyiClient(token=token)
        try:
            await client.discard_drafts(clean)
        finally:
            await client.aclose()
    except Exception as e:  # noqa: BLE001 — 软删失败绝不卡重生（旧草稿留存无害，从此无 item 指向它）
        _facts_log.warning(f"discard_drafts best-effort failed (regen continues): {e}")


# --- 自动落草稿（PRD-A-022 批1）：assemble 收尾即把题组逐题落「草稿」(status=0) ----------
async def _autodraft_items(
    items: list[dict[str, Any]], facts: dict[str, Any], token: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
    """🔴 PRD-A-022 批1·assemble 自动落草稿：对**还没草稿/没发布**的题逐题 create（status=0=草稿）。

    幂等：item 已有 _draft_id 或 persisted=True → 跳过（不重复建）；母题 mother_question_id 已有 → 跳过。
    回写：variant 回执 id → item._draft_id；mother 回执 id → mother_question_id（母题落的也是草稿）。

    🔴 best-effort：复用 persist_items（逐题 POST create，默认 status=0），但**绝不抛**——本函数
       内部已 try 兜底由调用方再 try（双层防御），返回 (回写后 items, mother_question_id 或 None, 是否真落过)。
       token 缺/落库失败 → items 原样返回、ok=False，调用方据此 emit warn（题组照常展示）。

    返回 (new_items, new_mother_qid, ok)：
      - new_items：回写了 _draft_id 的新 list（已跳过的题不变）。
      - new_mother_qid：母题草稿 id（本次新落才有；母题已在库/未落 → None）。
      - ok：是否成功走完（True=调过 persist_items 且无整体异常；False=未落/异常，但 items 仍可用）。
    """
    # 幂等过滤：只对「无草稿 且 未发布」的题落草稿。
    pending_idx = [
        i for i, it in enumerate(items)
        if not (it.get("_draft_id") or it.get("persisted"))
    ]
    if not pending_idx:
        return items, None, True  # 全有草稿/已发布 → 无需重落（幂等空操作，视为成功）
    pending_items = [items[i] for i in pending_idx]
    receipts = await persist_items(pending_items, facts, token=token)
    var_receipts = [r for r in receipts if r.get("role") != "mother"]
    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    new_items = [dict(it) for it in items]
    for j, r in zip(pending_idx, var_receipts):
        if r.get("ok") and r.get("id") is not None:
            new_items[j]["_draft_id"] = r.get("id")  # 草稿行雪花 id（status=0），入库时 promote 用
    new_mother_qid = mother.get("id") if (mother and mother.get("ok") and mother.get("id") is not None) else None
    return new_items, new_mother_qid, True


# --- 确认入库（设计 §7）：变式+解析经 RuoYi 写老师个人题库，只写不判 -----------
async def persist_to_bank(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ 入库：老师"这组可以了" → 逐题 POST /teacher/question/create（teacher token 定 owner）。

    🔴 只写不再判（质量门已在 solve_explain 闭合）。入库回执（入库 N 道 + 失败标注）。
    🔴 owner 由后端 LoginHelper 定，body 绝不传 createBy。
    """
    from agents.variant.stage1_anchor.mother_card import _emit_artifact  # strangler·破循环（调用期解析）

    items = state.get("items") or []
    facts = _mother_facts(state)
    if not items:
        return {"messages": [AIMessage(content="当前没有可入库的变式题。先贴图举一反三吧。")]}

    # 🔴 PRD-C-015 批4·致命① 入库防脏硬闸（新不变量）：存在 dna_dirty 题 / 母题脏 → 拒绝入库。
    #   与「凡进 items 必过 solve_explain」并列。提示精确到第 N 题（先点重生或撤销）。
    dirty_msg = persist_dirty_guard(state)
    if dirty_msg:
        return {"messages": [AIMessage(content="⚠ 暂不能入库：" + dirty_msg)]}

    # 🔴 入库簿记（PRD-C-011 G5 + 批4 缺口10 + PRD-A-018 M1③）：跳过判据 =
    #   「已入库(persisted) 且 非 DNA脏(dna_dirty) 且 非内容已编辑待覆盖(_content_dirty)」。
    #   - dna_dirty（改维待重生）已被上面硬闸拦掉，本就不入库；
    #   - _content_dirty（已入库题内容被编辑/重生/重写解析过）= 放行入 pending，走「覆盖原行
    #     update by _persist_id」（persist_items 据 _persist_id 走 update_question，幂等覆盖、不重复落行）。
    #   两者区分：DNA脏→拒入库（先重生）；内容已编辑→允许覆盖入库（F16/M1③）。
    pending_idx = [
        i for i, it in enumerate(items)
        if not (
            it.get("persisted")
            and not it.get("dna_dirty")
            and not it.get("_content_dirty")
        )
    ]
    if not pending_idx:
        return {
            "messages": [
                AIMessage(content="这组变式之前都已入库过了，没有需要补录的题（不会重复落库）。可以继续编辑或换一批。")
            ]
        }
    pending_items = [items[i] for i in pending_idx]
    n_skipped = len(items) - len(pending_items)

    # 🔴 身份透传：book-ui 经 agent_config 透传登录老师 access_token（config.configurable.ruoyi_token）
    # → 入库 owner = 该老师本人（后端 LoginHelper 取 token 身份），而非 .env 服务账号。
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    _emit_stage("persist", "入库", "running", f"{len(pending_items)} 道")
    try:
        # 🔴 PRD-A-022：publish=True → 草稿 promote 0→1（不重 create）；已发布编辑过 → update 覆盖；
        #   无草稿无发布 id 兜底 → create status='1' 直接发布。
        receipts = await persist_items(pending_items, facts, token=token, publish=True)
    except Exception as e:  # noqa: BLE001 — 登录/网络整体失败 → 友好兜底，不崩
        _emit_stage("persist", "入库", "warn", "连不上题库服务")
        return {
            "messages": [
                AIMessage(content=f"入库时连不上题库服务（book-server :8090 是否在跑？）：{e}")
            ]
        }

    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var_receipts = [r for r in receipts if r.get("role") != "mother"]
    # 🔴 PRD-C-103 WS2·AC6：落「题↔模型」清单（转正脚本 c103_promote_models.py 消费 → 落
    #   biz_question_model + 临时模型转正）。母题 id 用回执回填（persist_items 在局部 facts 回填，
    #   这里据 mother 回执补到 facts 供变式血缘指针）。best-effort，落盘失败不拦入库。
    try:
        manifest_facts = dict(facts)
        if mother and mother.get("ok") and mother.get("id") is not None:
            manifest_facts["mother_question_id"] = mother.get("id")
        # 🔴 PRD-C-103 WS3·AC9：把双旋钮真值（算子/相似度/目标档/实际档/回炉数）喂进变式回执 →
        #   record_link_manifest 据此落 trace 块（method=算子、variation_degree=相似度真值，补批2
        #   留下的 forward-gen/None 兜底）。var_receipts 与 pending_items 同序（persist_items 逐题回执）。
        knobs_now = state.get("knobs") or {}
        for r, it in zip(var_receipts, pending_items):
            if r.get("ok") and r.get("id") is not None:
                tb = variant_trace_block(it, knobs_now)
                r["operator"] = tb["operator"]
                r["similarity"] = tb["similarity"]
                r["trace_block"] = tb  # record_link_manifest 优先读它（含 target/actual/retries）
        record_link_manifest(receipts, manifest_facts)
    except Exception:  # noqa: BLE001 — 清单落盘是增强，绝不拖垮入库主流程
        pass
    ok = [r for r in var_receipts if r.get("ok")]
    fail = [r for r in var_receipts if not r.get("ok")]
    _emit_stage(
        "persist",
        "入库",
        "done" if not fail else "warn",
        f"成功 {len(ok)} 道" + (f"，失败 {len(fail)} 道" if fail else ""),
    )
    # 🔴 簿记回写 state（不是只发快照帧）：persisted 标进 items、母题雪花 id 进 mother_dna。
    # 不回写的话，下一编辑轮 assemble 重发快照 persisted 全 false → 「已收录」徽章整组回退、
    # 「全部入库」重新可点 → 二次入库整组重复落行（G5 破）；图母题也会再落一份（双份血缘）。
    new_items = [dict(it) for it in items]
    for j, r in zip(pending_idx, var_receipts):
        if r.get("ok"):
            new_items[j]["persisted"] = True
            # 🔴 M1①（PRD-A-018 簇1）：回写 _persist_id（对齐 persist_one_to_bank）。promote 回执
            #   id = 草稿行 id（原行就地改 status，id 不变）→ _persist_id 即原 _draft_id。
            #   不回写 → 重生/编辑后再入库时 item 无 _persist_id → 整组重复落行（M1 根）。
            if r.get("id") is not None:
                new_items[j]["_persist_id"] = r.get("id")
            # 🔴 PRD-A-022：草稿已发布 → 清 _draft_id（从此走「已发布」语义：编辑走 _persist_id 覆盖，
            #   不再被 autodraft/换一批 当草稿处理）。
            new_items[j].pop("_draft_id", None)
            # 落库即清「内容已编辑待覆盖」标记（已覆盖入库，本轮内容已同步到库行）。
            new_items[j].pop("_content_dirty", None)
        else:
            new_items[j]["persisted"] = bool(r.get("ok"))
    update: VariantState = {"items": new_items}
    if mother and mother.get("ok") and mother.get("id") is not None:
        # persist_items 的 mother_question_id 回填发生在局部 facts 副本上 → 这里落回 state，
        # 重试/后续入库走「母题已在库」分支，不再重复建母题。
        # 🔴 PRD-A-022：母题本次已发布（create status=1 / promote）→ 标 mother_published=True，
        #   后续 publish 不再重 promote（幂等）。
        update["mother_dna"] = dict(
            state.get("mother_dna") or {},
            mother_question_id=mother.get("id"),
            mother_published=True,
        )

    # artifact 更新快照（PRD-C-011）：按回写后的 items 组帧（_artifact_payload 读 item.persisted）
    _emit_artifact({**state, "items": new_items})

    lines = [f"## 入库完成 · 变式 {len(pending_items)} 道，成功 {len(ok)} 道"]
    if n_skipped:
        lines.append(f"（另有 {n_skipped} 道此前已发布，本次跳过、未重复入库。）")
    # 母题(原题)发布回执：图母题草稿一并发布 → 挂血缘
    if mother and mother.get("ok"):
        lines.append(f"📌 原题(母题)已一并发布到题库，ID：{mother.get('id')}，变式都挂在它名下（血缘可追）。")
    elif mother and not mother.get("ok"):
        lines.append(f"⚠ 原题发布失败（变式仍已落，血缘暂缺）：{mother.get('error')}")
    if ok:
        ids = [str(r.get("id")) for r in ok if r.get("id") is not None]
        lines.append("已发布到你的个人题库（来源标记「举一反三」）。" + (f"变式 ID：{', '.join(ids)}" if ids else ""))
    if fail:
        lines.append(f"\n⚠ {len(fail)} 道变式入库失败：")
        for i, r in enumerate(fail, 1):
            lines.append(f"  {i}. {r.get('error')}")
        lines.append("失败的题再说一次「入库」即可只补这几道（已成功的不会重复落库）。")
    lines.append("\n可回平台「我的题库」找题、组卷、导出 PDF。")
    update["messages"] = [AIMessage(content="\n".join(lines))]
    return update


async def persist_one_to_bank(
    state: VariantState, index: int, token: str | None = None
) -> tuple[VariantState, dict[str, Any], str | None]:
    """🔴 单题入库（PRD-C-014 B2·T5·B3 前置）：把第 index 道（1-based）单独落库。

    复用 persist_items 的单 item 路径（同一段簿记/血缘/owner 逻辑，零分叉）：
    - 🔴 item 级 persisted 防重：该题已 persisted → 直接回已有 id（item._persist_id），
      **不二次落行**（已收录的重复调幂等）。
    - 母题血缘：persist_items 在 facts 无 mother_question_id 且有母题题干时先落母题、回填 id；
      回写 state.mother_dna.mother_question_id（与「全部入库」同源，后续单题/全量不重复建母题）。

    返回 (update, result, error)：
      - index 越界 → ({}, {}, 错误串)（端点回 400）。
      - 成功/已收录 → (含 items[+mother_dna] 的 update, {ok, id, role, skipped?}, None)。
      - 单题落库失败 → (空 update, {ok:False, error}, None)（端点回 200 带 ok=False，不当 500）。
    """
    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, {}, f"index 越界（须 1..{len(items)}），收到 {index}"

    target = items[index - 1]
    # 🔴 批4·致命① 入库防脏：本题脏（或母题脏波及本题）→ 拒绝单题入库。
    if target.get("dna_dirty") or (state.get("mother_dna") or {}).get("dirty"):
        return (
            {},
            {"ok": False, "error": f"第 {index} 题改了还没重生，先点「重生」或「撤销重生」再入库。"},
            None,
        )
    # item 级防重（缺口10 + PRD-A-018 M1③）：persisted 且 not dna_dirty 且 not _content_dirty
    #   → 跳过（已收录、未改过）。_content_dirty（已入库题内容已编辑/重生待覆盖）→ 不跳过，
    #   往下走 persist_items 据 _persist_id 覆盖原行（与「全部入库」同口径）。
    if (
        target.get("persisted")
        and not target.get("dna_dirty")
        and not target.get("_content_dirty")
    ):
        return (
            {},
            {"ok": True, "id": target.get("_persist_id"), "role": "variant", "skipped": True},
            None,
        )

    facts = _mother_facts(state)
    # 🔴 PRD-A-022：单题「收录」= publish（草稿 promote 0→1 / 已发布编辑 update / 兜底 create status=1）。
    receipts = await persist_items([target], facts, token=token, publish=True)
    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var = next((r for r in receipts if r.get("role") != "mother"), None) or {"ok": False, "error": "无入库回执"}

    update: VariantState = {}
    if var.get("ok"):
        new_items = [dict(it) for it in items]
        new_items[index - 1]["persisted"] = True
        if var.get("id") is not None:
            new_items[index - 1]["_persist_id"] = var.get("id")  # 防重回查用（promote 回执 id=原草稿 id）
        # 🔴 PRD-A-022：草稿已发布 → 清 _draft_id（从此走「已发布」语义）。
        new_items[index - 1].pop("_draft_id", None)
        # 落库即清「内容已编辑待覆盖」标记（M1③：覆盖入库后库行已与本地内容一致）。
        new_items[index - 1].pop("_content_dirty", None)
        update["items"] = new_items
        if mother and mother.get("ok") and mother.get("id") is not None:
            update["mother_dna"] = dict(
                state.get("mother_dna") or {},
                mother_question_id=mother.get("id"),
                mother_published=True,
            )

    result = {"ok": bool(var.get("ok")), "id": var.get("id"), "role": "variant"}
    if not var.get("ok"):
        result["error"] = var.get("error")
    return update, result, None


# ---------------------------------------------------------------------------
# 题组编辑器直连动作（PRD-C-009 二期）：reorder / edit-item / reverify。
# 与 /variant/persist 同范式——确定性/单题动作不过 LLM 意图分类器，service 端点按
# thread_id 取 state → 调下面纯逻辑函数 → aupdate_state 回写 → 返回 _artifact_payload。
# 业务逻辑收在 variant.py（单一事实源），service.py 只做取/写/组帧的薄壳。
# ---------------------------------------------------------------------------
def reorder_items_state(state: VariantState, order: list[int]) -> tuple[VariantState, str | None]:
    """题组重排（零 LLM）：order=1-based 全排列 → _reorder_items 纯重排 + manual_order sticky。

    返回 (update, error)：order 非全排列 → (空 update, 错误串) 让端点回 400；合法 →
    (含 items/manual_order 的 update, None)。簿记字段（check/gene/persisted/_seq）随题搬位。
    与 exec_reorder 共用 _reorder_items / _is_full_permutation，重排语义零分叉。
    """
    items = list(state.get("items") or [])
    if not _is_full_permutation(order, len(items)):
        return {}, f"order 必须是 1..{len(items)} 的全排列（每号恰一次，长度={len(items)}）"
    reordered = _reorder_items(items, order)
    return {"items": reordered, "manual_order": True}, None


def edit_item_state(
    state: VariantState,
    index: int,
    stem: str | None = None,
    answer: str | None = None,
    solution: str | None = None,
) -> tuple[VariantState, dict[str, Any] | None, str | None]:
    """单题手动编辑（零 LLM）：只 patch 传入字段（其余不动）→ 净化富文本 → 标手动编辑。

    返回 (update, edited_item, error)：index 越界 → (空, None, 错误串) 让端点回 400。
    - 标记：manual_edited=True + from_edit=True（复用"老师意志优先"语义，下游 reverify/入库
      闸B 见 from_edit 不回炉换题）；这俩内部键不入库（build_create_bo/_artifact_payload 白名单挡）。
    - check 置中性 {'tier':'manual'}：清掉旧 verify/badge 误导，artifact 透传 tier='manual' 给 FE。
    """
    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
    new_items = [dict(it) for it in items]
    it = new_items[index - 1]
    content_changed = False
    if stem is not None:
        it["stem"] = _sanitize_rich_text(stem)
        content_changed = True
    if answer is not None:
        it["answer"] = _sanitize_rich_text(answer)
        content_changed = True
    if solution is not None:
        it["solution"] = _sanitize_rich_text(solution)
        content_changed = True
    it["manual_edited"] = True
    it["from_edit"] = True
    # 🔴 PRD-A-018 M1③：已入库题手动改了题面/答案/解析 → 标「内容已编辑待覆盖」，
    #   再入库走 _persist_id 覆盖原行（否则「已入库就跳过」→ 改后内容静默不写回库）。
    if content_changed:
        mark_content_dirty_if_persisted(it)
    # 🔴 题型模版自动规范（PRD-C-009·BE）：手动编辑回写后顺手规范 stem（净化之后）。
    #   edit-item 本身不跑 sympy 判决（check 置 manual 中性），此处规范无判决可影响——让老师
    #   手改的题立刻是 canonical 上屏，即便未点 reverify 也吃规范文本。cosmetic-only、幂等。
    _format_item_stem(it)
    # check 置中性：手动编辑、验算待重跑（清旧 verify/badge/tier，避免徽章误导）
    it["check"] = {"tier": TIER_MANUAL}
    return {"items": new_items}, it, None


def set_mother_figure_state(
    state: VariantState, figure_url: str | None,
) -> tuple[dict[str, Any], str | None]:
    """PRD-A-022 批2·D8：把母题「切图」OSS https url 回写进顶层 state.mother_figure_url（零 LLM）。

    FE 在母题切图就绪（autoCropMotherFigure/cropMotherFigure 拿到 base64）后上 OSS 拿 https url，
    调本端点回写 → 落 checkpoint。下游 build_mother_bo 据 facts.mother_figure_url 入库切图（D8：
    缺则不带图、绝不退原图兜底）。撤图 = 传 None/空 → 清空。仅收 https（与变式 figure_url 同口径）。

    返回 (update, error)：figure_url 非 https → ({}, 错误串) 让端点回 400。
    """
    url = str(figure_url or "").strip()
    if url and not url.startswith("https://"):
        return {}, "figure_url 必须是 https OSS 地址"
    # 非空设、空清（顶层 scalar，单次写定，无 items reducer 参与）
    return {"mother_figure_url": (url or None)}, None


def set_item_figure_state(
    state: VariantState, index: int, figure_url: str | None,
    figure_base64: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """PRD-C-100 BC2 + PRD-A-021 R2b·U1：把变式配图回写进 state.items[index-1]（零 LLM）。

    两类配图态（可单独或一并回写）：
      · figure_url（入库态 OSS url）：FE 入库时 uploadMotherImage 传 OSS 拿 https url 回写 →
        入库时 build_create_bo 据它产 A-015 image 块。仅收 https。
      · figure_base64（生成态 PNG base64）：🔴 R2b·U1 —— compose_variant_figure 产的 base64
        老师认账后**立即**回写 state（不等入库），落 checkpoint → 刷新/切 tab 从 state 取回，
        治「生成态配图只在 FE 内存、刷新即丢」根因。入库时若仅有 base64 无 url，persist 侧
        会 server-side 上 OSS 转 url（见 persist_items），故 base64 也是入库兜底来源。

    撤图语义（老师撤图）：figure_url 与 figure_base64 都传 None/空 → 两态都清。
    仅传其一为 None 而另一非空 → 只更/只清传入的那一态（区分「撤 OSS url 但留 base64」等场景）。
    🔴 reducer：figure_url/figure_base64 均挂 PRESERVE_IF_SAME_STEM（题面变=旧图失效不嫁接）。

    返回 (update, edited_item, error)：index 越界 → ({}, None, 错误串) 让端点回 400。
    这俩是展示/入库增强键，不入 biz_question 旧字段（仅 blockJson 用），不触碰 check/验算态。

    🔴 兼容：figure_base64 缺省（旧调用方只传 figure_url）→ 不动 base64（按 figure_url 单态语义，
       与 BC2 旧行为字节级一致）。
    """
    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
    url = str(figure_url or "").strip()
    if url and not url.startswith("https://"):
        return {}, None, "figure_url 必须是 https OSS 地址"
    b64 = str(figure_base64 or "").strip()
    new_items = [dict(it) for it in items]
    it = new_items[index - 1]
    # figure_url 态：非空设、空清（与 BC2 旧语义一致；旧调用方不传 base64 时仅此分支生效）
    if url:
        it["figure_url"] = url
    else:
        it.pop("figure_url", None)
    # figure_base64 态（R2b·U1）：仅当调用方**显式传了** figure_base64 形参时才动它
    #   （None=未传=不碰，保旧调用兼容；空串=显式撤=清；非空=写生成态 base64）。
    if figure_base64 is not None:
        if b64:
            it["figure_base64"] = b64
        else:
            it.pop("figure_base64", None)
    return {"items": new_items}, it, None


def mark_item_manual_block_state(
    state: VariantState, index: int, edited: bool = True
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """PRD-C-100 BC3：标/清「老师手动排版过」印记（零 LLM）。

    老师对已入库变式点「手动排版」→ FE 跳 A-015 网格编辑器存 blockJson 后回本会话调本端点，
    把该题标 manual_block=True + manual_edited=True（复用「老师意志优先」印记：下游 reverify/入库
    闸B 见 from_edit 不回炉换题；重生前 FE 据 manual_edited 弹二次确认）。这俩内部键不入库
    （build_create_bo/_artifact_payload 白名单挡，question_id/manual_block 仅透传给 FE 渲染）。

    edited=False = 清印记（确认重生时调，重生后该题不再算「手动排版态」）。重生本身仍走既有
    regen 通路（_regen_once），本函数只管印记；重生产物的脏 blockJson 由 FE 另调 BE delete-block
    清掉（避免详情/卷库按旧布局渲染新题面）。

    返回 (update, edited_item, error)：index 越界 → ({}, None, 错误串) 让端点回 400。
    """
    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
    new_items = [dict(it) for it in items]
    it = new_items[index - 1]
    if edited:
        it["manual_block"] = True
        it["manual_edited"] = True
        it["from_edit"] = True
    else:
        it.pop("manual_block", None)
    return {"items": new_items}, it, None


async def reverify_item_state(
    state: VariantState, index: int
) -> tuple[VariantState, dict[str, Any] | None, str | None]:
    """单题重跑闸B（on-demand，单题 LLM+sympy）：清旧 check → _check_one_item 同款判决路径。

    返回 (update, rechecked_item, error)：index 越界 → (空, None, 错误串) 让端点回 400。
    复用 solve_explain 节点对单题的同一函数 _check_one_item（_solve_one + _machine_verify →
    check{badge,solved_answer,verify,tier}），判决仍只读 sympy verdict（铁律不破）。跑完
    tier 变成真实验算结果（verified/self_ok/both_low/silent 按 4d 矩阵），洗掉 manual 的"待验算"。
    剔除（sympy 证实标答错且重生失败）情形：from_edit 题不剔除（_check_one_item 保留打 ⚠），
    故此处 rechecked 必非 None；万一仍 None 兜底保留原题不丢（G5）。
    """
    from agents.variant.stage1_anchor.solve import _check_one_item  # strangler·破循环（调用期解析）

    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
    facts = _mother_facts(state)
    new_items = [dict(it) for it in items]
    target = dict(new_items[index - 1])
    target.pop("check", None)  # 清旧 check → _check_one_item 重判（否则带 check 会原样跳过）
    target["from_edit"] = True  # 编辑后重验：保留老师意志（FAIL 不回炉换题，打 ⚠ 交人审）
    rechecked, _dropped = await _check_one_item(target, facts, index - 1, len(items))
    # from_edit 短路保证 rechecked 非 None；兜底（极端降级）保留原题不丢
    final = rechecked if rechecked is not None else new_items[index - 1]
    # 🔴 题型模版自动规范（PRD-C-009·BE）：reverify 是单题 sympy/structure_lint 判决路径
    #   （_check_one_item），规范在判决**之后**就地跑（绝不影响判决）。规范文本落进 item →
    #   随 _artifact_payload 上屏 + 后续入库 + 会话恢复都吃规范文本；幂等不漂移。
    _format_item_stem(final)
    new_items[index - 1] = final
    return {"items": new_items}, new_items[index - 1], None


async def verify_one_stem(
    stem: str, answer: str, *, qtype: str | None = None, options: Any = None
) -> dict[str, Any]:
    """🔴 BUG-09（2026-06-19）：无状态单题程序验算（/variant/verify-one 后端核心）。

    入参 = 题面 stem + 题面标答 answer（+ 可选 qtype/options）。复用闸B 同一判决路径
    （_solve_one 重解 → _machine_verify 纯 sympy），判决只读 verdict（铁律不破，零 LLM 自评）。
    返回 {"verdict": "pass"|"fail"|"degrade", "detail": str, "computed": str|None}：
      - pass    = sympy 证实题面标答自洽；
      - fail    = sympy 证实题面标答错（computed = 程序真算值）；
      - degrade = sympy 吃不下载荷（抽不成/超时/证明开放类）→ 老师人工判（非判错）。
    永不抛（任何异常按 degrade 收口），与 reverify 同源、与 solve_explain 单题逐字一致语义。
    """
    from agents.variant.stage1_anchor.solve import _solve_one  # strangler·破循环（调用期解析）

    s = str(stem or "").strip()
    a = str(answer or "").strip()
    if not s:
        return {"verdict": math_verify.DEGRADE, "detail": "题面为空，无从验算", "computed": None}
    item: dict[str, Any] = {"stem": s, "answer": a}
    if qtype:
        item["qtype"] = str(qtype)
    if options is not None:
        item["options"] = options
    # 证明/开放/作图类 → 不进 sympy（与 _check_one_item 同分流），按 degrade 转人工。
    if _is_proof_like(item.get("qtype"), s):
        return {
            "verdict": math_verify.DEGRADE,
            "detail": "证明/作图/开放类不做程序验算，转人工复核",
            "computed": None,
        }
    try:
        solved = await _solve_one(s)
        res = await _machine_verify(item, solved.get("solved_answer"))
    except Exception as e:  # noqa: BLE001 — 反挂死/反外抛（G5）：任何异常按 degrade 收口
        return {"verdict": math_verify.DEGRADE, "detail": f"验算异常: {str(e)[:80]}", "computed": None}
    verdict = res.get("verdict") or math_verify.DEGRADE
    computed = res.get("computed")
    return {
        "verdict": verdict,
        "detail": str(res.get("detail") or ""),
        "computed": str(computed) if computed is not None else None,
    }


# ===========================================================================
# DNA 双模态编辑（PRD-C-014 B4·T1/T2，事实源 PRD §10.5 / §3.5）
# —— 老师在题组编辑器里直接改某道题的 DNA（维度），两条路：
#   ① edit_dna_state（零 LLM）：结构化回写一个维度（点击/下拉选）→ 校验合法值 → 改 state
#      对应键（main_kp/grade 同步 header+BO，其余 DNA 维改 mother_dna.dna，qtype/difficulty
#      改 item）→ 标 manual_edited（item 级）→ 回 artifact。全程零 LLM（G11 断言点）。
#   ② revise_item（有界 LLM）：锚定重做某维（骨架/场景文本改写）或整题重出（whole 走 REGEN
#      闸B sympy 重验）。diff 锁 target：纯骨架/场景文本维改写不动其余维（G12 断言点）。
# 🔴 铁律：判决只读 sympy（whole 走既有 _check_one_item / from_edit 路径）；老师改动最高优先
#   （回写、标 manual、不质疑）。LLM 失败/解析不出 → 降级回 {ok:false,error}，绝不崩。
# 🔴 DNA 维度分两层落点：main_kp/secondary_kps/exam_type/tags/scene/skeleton 是**母题级**
#   （住 state.mother_dna.dna，由 _mother_facts 穿进每道题的 BO，整组共享守恒维）；
#   qtype/difficulty 是**题级**（住 item，build_create_bo 优先取 item 覆盖）。grade 走 analysis。
# ===========================================================================

# edit-dna 合法 field 白名单（与 FE 契约严格一致）。main_kp/secondary_kps 走母题级 +
# 同步 analysis.kp（header kp + BO dim1）；grade 走 analysis.grade（header grade + BO subject_id）。
# 🔴 PRD-C-015 批4·补齐 skeleton/hard_points/models（批1 依赖债：冻结 setter 早就位、
#   但 edit-dna 入口此前不含此三维）。四分流由 REGEN_CLASS 驱动（hard_anchor/soft_regen/
#   rewrite_solve/meta），edit_dna_state 据 regen_class_of(field) 置 dirty / 解冻重锚。
_EDIT_DNA_FIELDS = {
    "main_kp", "secondary_kps", "qtype", "exam_type", "difficulty", "tags", "scene", "grade",
    "skeleton", "hard_points", "models",
}


def _coerce_kp(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """把 edit-dna 传入的知识点 value（code 字符串 或 {code,name} / {id,name}）归一成
    DNA 契约用的 {id, name} dict。返回 (kp_dict, error)。空/非法 → (None, 错误串)。"""
    if isinstance(value, dict):
        kid = str(value.get("code") or value.get("id") or "").strip()
        name = value.get("name")
        if not kid:
            return None, "知识点缺少 code/id"
        return {"id": kid, "name": (str(name).strip() if name else None)}, None
    kid = str(value or "").strip()
    if not kid:
        return None, "知识点 code 为空"
    return {"id": kid, "name": None}, None


def edit_dna_state(
    state: VariantState, index: int, field: str, value: Any
) -> tuple[VariantState, dict[str, Any] | None, str | None]:
    """🔴 T1·结构化 DNA 回写（零 LLM·G11 断言点）：老师点击/下拉改第 index 道（1-based）的某个
    维度 field → 校验合法值（非法 → 400 拒收，返 error 串）→ 回写 state 对应键 → 标该题
    manual_edited=True → 返回 (update, edited_item, error)。

    field → 改了 state 哪些键（回写映射表）：
      main_kp        → mother_dna.dna.main_kp{id,name}（守恒白名单·BO dim1/secondaryKp 源）
                       + analysis.kp{value,anchored{id,code,name},confidence=1.0}（header kp + BO dim1）
      secondary_kps  → mother_dna.dna.secondary_kps（≤3，BO secondaryKpIds 源）
      qtype          → item.qtype（题级；BO questionType/dim2 源）+ mother_dna.dna.qtype（守恒）
      exam_type      → mother_dna.dna.exam_type（守恒·BO examType 源）
      difficulty     → item.difficulty(1..4) + item.level(>=4→hard 否则 normal)（题级；BO dim4/difficult 源）
      tags           → mother_dna.dna.tags（BO tags 源）
      scene          → mother_dna.dna.scene（BO scene 源）
      grade          → analysis.grade{value,confidence=1.0,code=_grade_to_code(value)}（header grade + BO subjectId 源）

    🔴 manual_edited 只标本题（item 级）；母题级维度的改动对整组生效（守恒维共享），但只有
       老师点的那道被标 manual（check 置 manual 中性，洗掉旧 verify 徽章避免误导）。
    🔴 main_kp 改动 = 老师手动锚定：confidence 置 1.0（老师意志最高优先，不再质疑），
       analysis.kp.anchored.code = 知识点 id（叶子 id 即层级编码，与 classify 同口径 → BO dim1）。
    🔴 难度改动同步 level（星级/difficult 的题级表达）。
    返回 (update, edited_item, error)：index 越界 / field 非法 / value 非法 → (空, None, 错误串)。
    """
    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
    if field not in _EDIT_DNA_FIELDS:
        return {}, None, f"非法 field「{field}」（合法：{'/'.join(sorted(_EDIT_DNA_FIELDS))}）"

    new_items = [dict(it) for it in items]
    it = new_items[index - 1]
    update: VariantState = {}
    # 母题级 DNA / analysis 的副本（只有改到才放进 update，避免无谓覆盖）
    mother_dna = dict(state.get("mother_dna") or {})
    dna = dict(mother_dna.get("dna") or {})
    analysis = dict(state.get("analysis") or {})
    # 🔴 PRD-C-015 批1·守恒维改走 facts_audit 留痕（缺口6）：edit_dna_state = 老师手动路径
    #   （source=teacher，永远放行 + 留痕；冻结只挡 LLM 来源）。审计追加进既有 facts_audit。
    audit = list(state.get("facts_audit") or [])
    _audit_n0 = len(audit)
    locked = bool(state.get("facts_locked"))

    if field == "main_kp":
        kp, err = _coerce_kp(value)
        if err:
            return {}, None, f"main_kp 非法：{err}"
        # 🔴 BUG-01（2026-06-19）：改主考点「可回退」—— 改前留一份旧考点快照（mother_dna.dna.main_kp
        #   + analysis.kp），FE 据它给「撤销改考点」入口；不破坏 items（旧改主考点=清 items 整组重出
        #   不可回退，已废，见下方 hard_anchor 分流注释）。
        _prev_main_kp = copy.deepcopy(dna.get("main_kp")) if dna.get("main_kp") is not None else None
        _prev_kp_node = copy.deepcopy(analysis.get("kp")) if analysis.get("kp") is not None else None
        dna["main_kp"] = kp
        mother_dna["dna"] = dna
        kp_node = dict(analysis.get("kp") or {})
        kp_node["anchored"] = {"id": kp["id"], "code": str(kp["id"]), "name": kp.get("name")}
        if kp.get("name"):
            kp_node["value"] = kp["name"]
        kp_node["confidence"] = 1.0  # 老师手动锚定 = 最高优先
        analysis["kp"] = kp_node
        update["mother_dna"] = mother_dna
        update["analysis"] = analysis

    elif field == "secondary_kps":
        raw = value if isinstance(value, list) else [value]
        sec: list[dict[str, Any]] = []
        for v in raw:
            kp, err = _coerce_kp(v)
            if err:
                return {}, None, f"secondary_kps 含非法项：{err}"
            if any(s["id"] == kp["id"] for s in sec):
                continue  # 去重
            sec.append(kp)
        if len(sec) > dna_extract.SECONDARY_KP_MAX:
            return {}, None, f"副知识点最多 {dna_extract.SECONDARY_KP_MAX} 个（收到 {len(sec)}）"
        # 守恒维 → 走冻结 setter（老师来源放行 + 留痕；缺口6）
        _dna_fact_edit(
            mother_dna, "secondary_kps", sec,
            source="teacher", locked=locked, audit=audit, instruction="edit-dna",
        )
        update["mother_dna"] = mother_dna

    elif field == "qtype":
        qt = dna_extract._norm_qtype(value)
        if qt is None:
            return {}, None, f"非法题型「{value}」（合法：{'/'.join(dna_extract.QTYPES)}）"
        it["qtype"] = qt  # 题级
        dna["qtype"] = qt  # 守恒维同步
        mother_dna["dna"] = dna
        update["mother_dna"] = mother_dna

    elif field == "exam_type":
        et = str(value or "").strip()
        if et not in dna_extract.EXAM_TYPES:
            return {}, None, f"非法考察类型「{value}」（合法：{'/'.join(dna_extract.EXAM_TYPES)}）"
        # 守恒维 → 走冻结 setter（老师来源放行 + 留痕；缺口6）
        _dna_fact_edit(
            mother_dna, "exam_type", et,
            source="teacher", locked=locked, audit=audit, instruction="edit-dna",
        )
        update["mother_dna"] = mother_dna

    elif field == "difficulty":
        # 🔴 必须是整数（含可无损转整的 "3" 字符串）；3.5 这种小数不静默截断 → 拒收（任务契约「1..4 整数」）。
        d = None
        if isinstance(value, bool):
            d = None
        elif isinstance(value, int):
            d = value
        elif isinstance(value, float) and value.is_integer():
            d = int(value)
        elif isinstance(value, str):
            try:
                d = int(value.strip())
            except (TypeError, ValueError):
                d = None
        if d is None or d < dna_extract.DIFFICULTY_MIN or d > dna_extract.DIFFICULTY_MAX:
            return {}, None, (
                f"难度须是 {dna_extract.DIFFICULTY_MIN}..{dna_extract.DIFFICULTY_MAX} 整数（收到 {value!r}）"
            )
        it["difficulty"] = d  # 题级
        it["level"] = "hard" if d >= dna_extract.DIFFICULTY_MAX else "normal"  # 星级/difficult 题级表达

    elif field == "tags":
        if not isinstance(value, list):
            return {}, None, "tags 必须是字符串数组"
        tags = [str(t).strip() for t in value if str(t).strip()]
        dna["tags"] = tags
        mother_dna["dna"] = dna
        update["mother_dna"] = mother_dna

    elif field == "scene":
        dna["scene"] = str(value or "").strip()
        mother_dna["dna"] = dna
        update["mother_dna"] = mother_dna

    elif field == "grade":
        gv = str(value or "").strip()
        if not gv:
            return {}, None, "年级 value 为空"
        g = dict(analysis.get("grade") or {})
        g["value"] = gv
        g["confidence"] = 1.0  # 老师手动 = 最高优先
        code = _grade_to_code(gv)
        if code:
            g["code"] = code  # 同步 BO subjectId 源（科目锚 level1）
        else:
            g.pop("code", None)  # 归一不出 → 清旧 code，build 走 _grade_to_code(value) 兜底
        analysis["grade"] = g
        update["analysis"] = analysis

    elif field == "skeleton":
        # 🔴 批4·重写解析维（rewrite_solve，D-merge9）：解法骨架 = 母题级守恒基因。
        #   走冻结 setter（守恒维 4/4 之一，缺口6）；落 mother_dna.dna.skeleton（list 或 str 均存）。
        #   置 dirty 后点「重生」→ regen_dirty_items 对该题走重写解析（_rewrite_solve_once 过闸B）。
        sk = value
        if isinstance(sk, str):
            sk = [s for s in sk.split("\n") if s.strip()] or [sk.strip()] if sk.strip() else []
        elif isinstance(sk, list):
            sk = [str(s).strip() for s in sk if str(s).strip()]
        else:
            return {}, None, "skeleton 必须是字符串或字符串数组"
        _dna_fact_edit(
            mother_dna, "skeleton", sk,
            source="teacher", locked=locked, audit=audit, instruction="edit-dna",
        )
        # item 级骨架覆盖同步（_item_dna 优先读 item.skeleton，FE 面板即时反映）
        it["skeleton"] = "\n".join(sk)
        update["mother_dna"] = mother_dna

    elif field == "hard_points":
        # 🔴 批4·元数据维（meta）：难点是标注属性，改不必重出题面、不进 dirty。
        # 🔴 PRD-C-017 B5-fix6·「纯标注」语义：只更新值（走冻结 setter 留痕，缺口6，落
        #   mother_dna.dna.hard_points + item 级覆盖），**不波及下游、不置 mother_dirty、不触发重出**
        #   （见下方四分流路由：hard_points 不在 _MOTHER_DIRTY_PROP_FIELDS，与 tags 同档）。
        hp = value
        if isinstance(hp, str):
            hp = [hp.strip()] if hp.strip() else []
        elif isinstance(hp, list):
            hp = [str(h).strip() for h in hp if str(h).strip()]
        else:
            return {}, None, "hard_points 必须是字符串或字符串数组"
        _dna_fact_edit(
            mother_dna, "hard_points", hp,
            source="teacher", locked=locked, audit=audit, instruction="edit-dna",
        )
        it["hard_points"] = hp  # item 级覆盖（_item_dna 优先读 item）
        update["mother_dna"] = mother_dna

    elif field == "models":
        # 🔴 批4·重写解析维（rewrite_solve，D-merge9：撤销「models=元数据维只标注」）：
        #   改 models = 换解法 → 置 dirty，点「重生」→ 重写解析过闸B；批3 W3' 软警据新 models 自动重判。
        #   value = [{id,name}] 或 ["M25",...]；归一成 [{id,name}]，落 item.models（题级覆盖，_item_dna 读）。
        raw = value if isinstance(value, list) else [value]
        models: list[dict[str, str]] = []
        for m in raw:
            if isinstance(m, dict):
                mid = str(m.get("id") or "").strip()
                nm = str(m.get("name") or "").strip()
            else:
                mid = str(m or "").strip()
                nm = ""
            if not mid and not nm:
                continue
            if any(x["id"] == mid and x["name"] == nm for x in models):
                continue
            models.append({"id": mid, "name": nm})
        it["models"] = models  # 题级覆盖（_item_dna 优先读 item.models；母题守恒维不动）

    # ===================================================================
    # 🔴 PRD-C-015 批4·四分流路由（块③ R1 + D-merge7/8/9 + 缺口7/致命①）
    #   据 regen_class_of(field) 决定：硬锚→解冻重锚立即 / 软重生维·重写解析维→标 dirty /
    #   元数据维→只标注。本段在所有维写入之后统一处置 dirty / 解冻。
    # ===================================================================
    rclass = regen_class_of(field)

    # 标本题手动编辑 + check 置中性 manual（洗掉旧 verify 徽章，与 edit_item_state 同语义）
    it["manual_edited"] = True
    it["from_edit"] = True
    it["check"] = {"tier": TIER_MANUAL}

    if field == "main_kp":
        # 🔴 A-2/契约C4（PRD-A-018）：main_kp 已是 soft_regen 维（REGEN_CLASS·380），不再有
        #   hard_anchor 特例分支。改主考点 = 母题级守恒维改 → 标全组 dirty + 可回退 + await 显式重生
        #   （行为同其余 soft_regen 维，由常量直驱、无 BE 特判）。main_kp 是**母题级**维（全组变式共享
        #   同一主考点），故标「整组 dirty」而非仅本道 mark_item_dirty——这是 main_kp 与普通题级
        #   soft_regen 维（qtype/difficulty 仅本道）的唯一差别，属维度归属层级，非路由特判。
        #   承接 BUG-01（2026-06-19）「改主考点不强制重出、可回退」语义：
        #   ① 只更新主考点本身（main_kp/analysis.kp 已在上面写好）+ 守恒维留痕；
        #   ② 不清 items、不解冻（mother_confirmed/facts_locked 不动）→ 不触发任何自动重出；
        #   ③ 把下游变式标 mother_dirty（母题脏，进待重生集合）——据新考点重生须老师**显式点「重生」**；
        #   ④ 可回退：旧考点快照（main_kp_prev）随 update 外发，FE 给「撤销改考点」入口。
        if len(audit) > _audit_n0:
            update["facts_audit"] = audit
        # 母题脏（致命① 拦入库）+ 下游变式标 dirty 不自动重出（点「重生」才据新考点重出）
        mother_dna["dirty"] = True
        mark_mother_dirty(mother_dna, new_items, "main_kp")
        update["mother_dna"] = mother_dna
        update["items"] = new_items
        # 旧考点快照（FE「撤销改考点」用；不破坏 items → 撤销=把主考点改回旧值即可）
        update["main_kp_prev"] = {"main_kp": _prev_main_kp, "kp": _prev_kp_node}
        update["messages"] = [
            AIMessage(
                content=(
                    f"已把主考点改为「{kp.get('name') or kp.get('id')}」。"
                    "我**没有自动重出**已有变式（不擅自烧 token）——如需据新考点重生这些变式，"
                    "请点「重生」；若是改错了，点「撤销改考点」或把主考点改回原值即可（变式都还在）。"
                )
            )
        ]
        return update, it, None

    if rclass == "hard_anchor":
        # 🔴 缺口7·硬锚【年级】改 = 解冻 + 重锚（立即，走既有 patch/classify 路径）：
        #   清 items + mother_confirmed=False + facts_locked=False → after_patch/after_classify
        #   触发重锚重造。不进 dirty 攒批（与软重生维分路）。
        #   ⚠ 注意：硬锚立即重锚是「整组重出」语义，本道的 manual 改动已写进 analysis/dna 留痕，
        #   classify 会按新锚重抽 → 此处不保留旧 items。
        #   ⚠ BUG-01 只拆「改主考点」的级联（见上分支）；年级改通常意味着学段/进度变 = 整组失效，
        #     语义上确实该重锚，故年级保持立即解冻重锚（用户本次只要求主考点不级联）。
        if len(audit) > _audit_n0:
            update["facts_audit"] = audit
        update["items"] = []
        update["mother_confirmed"] = False
        update["facts_locked"] = False
        return update, it, None

    if rclass in ("soft_regen", "rewrite_solve"):
        # 软重生维【题型/难度/考察类型/场景】+ 重写解析维【骨架/models】改 → 标 dna_dirty（不立即重出）。
        # 点「重生」→ regen_dirty_items 据 dirty_dims 分别走 _regen_once（soft）/ 重写解析（rewrite）。
        mark_item_dirty(it, field)
    # meta（tags/secondary_kps/hard_points）→ 不进 dirty（只标注即时生效，§3.2c）。

    # 🔴 D-merge8·母题守恒维改（secondary_kps/exam_type/skeleton 母题级）→ 母题脏 +
    #   下游所有变式标 dirty 不自动重出（并入待重生集合）；secondary_kps 同理（meta 但母题级守恒）。
    # 🔴 PRD-C-017 B5-fix6·hard_points 改为「纯标注」语义（对齐 UI「只标注」维）：虽走冻结 setter
    #   留痕（_dna_fact_edit，缺口6），但**不波及下游、不置 mother_dirty、不触发重出**——重生 prompt
    #   全程不读 hard_points，旧逻辑把它当母题守恒基准维波及全组 = 空转一次 LLM（结果等价）。故从
    #   波及集合（_MOTHER_DIRTY_PROP_FIELDS）剔除；与 tags 这种纯 meta 标注同档，只更新值即时生效。
    if field in _MOTHER_DIRTY_PROP_FIELDS:
        mother_dna["dirty"] = True
        mark_mother_dirty(mother_dna, new_items, field)
        update["mother_dna"] = mother_dna

    update["items"] = new_items
    # 🔴 批1·守恒维改追加了 facts_audit → 落回 update（缺口6 留痕）。
    if len(audit) > _audit_n0:
        update["facts_audit"] = audit
    return update, it, None


# ---------------------------------------------------------------------------
# T2·有界 LLM 锚定重做（revise_item）：骨架/场景文本维改写（diff 锁 target，不漂移其余维）
# 或 whole 整题重出（走 REGEN + 闸B sympy 重验）。
# ---------------------------------------------------------------------------

# revise 文本维 → (人话名, JSON 键, item 字段)。skeleton 落 item.solution（解法骨架=解析载体）；
# scene 落 mother_dna.dna.scene（场景是母题级表皮维）。
_REVISE_TEXT_TARGETS = {
    "skeleton": ("解法骨架", "skeleton", "solution"),
    "scene": ("场景", "scene", None),
}


async def revise_item(
    state: VariantState,
    index: int,
    target: str,
    instruction: str,
    config: RunnableConfig | None = None,
) -> tuple[VariantState, dict[str, Any], str | None]:
    """🔴 T2·有界 LLM 锚定重做（供端点 + router 后续打字指令复用的单一事实源）。

    target ∈ {skeleton, scene, whole}：
      - skeleton/scene（纯文本维）：有界 LLM 带母题上下文 + 本题现状 + 老师 instruction，**只重写
        该维**（diff 锁 target·G12：不动 qtype/answer/difficulty 等其余维）→ 标 manual。
        skeleton 落 item.solution（解法骨架=解析载体）；scene 落 mother_dna.dna.scene（母题级表皮）。
      - whole：复用 REGEN 路径（instruction 注入，如"难一点"→难度档语义）重出该题 → 闸B sympy
        重验（既有 _check_one_item / from_edit 模式，判决只读 verdict）→ tier 更新 → 标 manual。
    🔴 LLM 走 core.relay_pool（_ainvoke_text 内）；调用失败/解析不出 → (空 update, {ok:False}, None)
       降级不崩（G5）。index 越界 / target 非法 → (空, {}, 错误串) 让端点回 400。
    返回 (update, result{ok, [error]}, error)。
    """
    from agents.variant.stage1_anchor.solve import _check_one_item  # strangler·破循环（调用期解析）
    from agents.variant.stage2_variant.gates import _regen_once  # strangler·破循环（调用期解析）

    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, {}, f"index 越界（须 1..{len(items)}），收到 {index}"
    if target not in ("skeleton", "scene", "whole"):
        return {}, {}, f"非法 target「{target}」（合法：skeleton/scene/whole）"

    facts = _mother_facts(state)
    new_items = [dict(it) for it in items]
    it = new_items[index - 1]

    # --- whole：REGEN 整题重出 + 闸B sympy 重验 ---
    if target == "whole":
        # instruction 作老师软约束注回（edit_note）→ _regen_once 永远把它注进 prompt（老师意志优先）
        it["edit_note"] = str(instruction or "").strip()
        it["from_edit"] = True
        draft = await _regen_once(it, facts, feedback=None)
        if not draft:
            return {}, {"ok": False, "error": "重出失败（模型未返回有效题目），已保留原题"}, None
        draft["from_edit"] = True  # 闸B 见 from_edit：FAIL 不回炉换题、保留打 ⚠ 交人审
        if it.get("edit_note"):
            draft["edit_note"] = it["edit_note"]
        # 配方印记跟题走（重出仍占原计划槽位）。🔴 PRD-A-022：**不 carry _draft_id**——旧草稿被
        #   这道新题替换，draft 无 _draft_id（assemble 重落新草稿）；旧草稿（未发布）下面 discard 软删。
        _old_draft_id = _draft_id_to_discard(it)  # whole 重出前取旧草稿 id（已发布题=None 不删）
        for k in ("_seq", "persisted", "_persist_id"):  # R4·F17: 去死键 from_recipe/expected_difficulty
            if it.get(k) is not None:
                draft[k] = it[k]
        draft.pop("check", None)  # 清旧 check → _check_one_item 重判（闸B sympy）
        rechecked, _dropped = await _check_one_item(draft, facts, index - 1, len(items))
        final = rechecked if rechecked is not None else draft
        final["manual_edited"] = True
        _format_item_stem(final)
        # 🔴 整改2（2026-06-12，推翻 G12b「独立 _grade_difficulty 重判」）：whole 重做的难度由
        #   REGEN 出题调用按嵌入的四档 rubric 同步断言（不再额外打一轮 nano）。守铁律不破：
        #   难度=LLM rubric 评级，不做关键词 hack（「难一点」不直接 +1，由 rubric 对新题断言）。
        #   缺/非法 difficulty 钳到 1~4（缺→2 兜底）。
        old_difficulty = _to_int(it.get("difficulty"))  # 重做前原档（用于 level 升档判定）
        nd = _to_int(final.get("difficulty"))
        final["difficulty"] = max(1, min(DIFFICULTY_CAP, nd)) if nd is not None else 2
        # level 同步：新难度 ≥4（压轴）或较原值升档 → 标 hard（star/星级跟 difficulty 走，level 一致）。
        new_difficulty = _to_int(final.get("difficulty"))
        if new_difficulty is not None and (
            new_difficulty >= DIFFICULTY_CAP
            or (old_difficulty is not None and new_difficulty > old_difficulty)
        ):
            final["level"] = "hard"
        final["manual_edited"] = True  # 重申 manual 印记（rechecked 可能复制过 dict）
        # 🔴 PRD-A-018 M1③：已入库题 whole 重做后内容已变 → 标待覆盖（入库走 _persist_id 覆盖原行）。
        mark_content_dirty_if_persisted(final)
        new_items[index - 1] = final
        # 🔴 PRD-A-022·best-effort 软删被替换掉的旧草稿（token 从 config 取，缺/失败仅 log，绝不阻塞）。
        token = ((config or {}).get("configurable") or {}).get("ruoyi_token") if config else None
        await _discard_drafts_best_effort([_old_draft_id], token)
        return {"items": new_items}, {"ok": True}, None

    # --- skeleton/scene：纯文本维改写（diff 锁 target，不动其余维·G12） ---
    target_cn, target_key, item_field = _REVISE_TEXT_TARGETS[target]
    if target == "skeleton":
        current = str(it.get("solution") or "")
    else:  # scene
        current = str((facts.get("dna") or {}).get("scene") or "")
    prompt = REVISE_FIELD_PROMPT.format(
        target_cn=target_cn,
        target_key=target_key,
        kp_name=facts["kp_name"],
        grade=facts["grade"],
        stem=str(it.get("stem") or ""),
        qtype=str(it.get("qtype") or facts["qtype"]),
        current=current,
        instruction=str(instruction or "").strip(),
    )
    # 🔴 整改1：改写解法骨架/场景同样压「解题方法不越学生进度」。
    prompt = prompt + "\n\n" + _context_block(facts)
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)], model=settings.variant_model("generate")
        )
    except Exception as e:  # noqa: BLE001 — 改写是增强，LLM 异常 → 降级不崩（G5）
        return {}, {"ok": False, "error": f"改写调用失败，已保留原值：{e}"}, None
    parsed = _parse_json(text)
    revised = None
    if isinstance(parsed, dict):
        revised = parsed.get(target_key)
    if not revised or not str(revised).strip():
        return {}, {"ok": False, "error": "未能解析出改写结果，已保留原值"}, None
    revised_text = _sanitize_rich_text(str(revised))

    update: VariantState = {}
    if target == "skeleton":
        # 🔴 G12a：解法骨架同时落两处——① item.solution（题级解析载体，照旧）；② item.skeleton
        #   （item 级骨架覆盖，给 _item_dna 读，FE DNA 面板 skeleton 字段才反映本次改动）。
        #   母题组级 mother_dna.dna.skeleton 是守恒基因，绝不动；item 级优先、缺则回退母题。
        it["solution"] = revised_text
        it["skeleton"] = revised_text
    else:  # scene：母题级表皮维 → mother_dna.dna.scene
        mother_dna = dict(state.get("mother_dna") or {})
        dna = dict(mother_dna.get("dna") or {})
        dna["scene"] = revised_text
        mother_dna["dna"] = dna
        update["mother_dna"] = mother_dna
    # 纯文本维改写不影响判分 → 标 manual 中性（与 edit_dna_state 同语义），不重跑 sympy
    it["manual_edited"] = True
    it["from_edit"] = True
    it["check"] = {"tier": TIER_MANUAL}
    # 🔴 PRD-A-018 M1③：已入库题 skeleton/scene 改写后内容(解析/场景)已变 → 标待覆盖。
    mark_content_dirty_if_persisted(it)
    update["items"] = new_items
    return update, {"ok": True}, None


# ===========================================================================
# 🔴 PRD-C-015 批4·手动重生 / 撤销重生（块③ R1/R2/R2b + D-merge6/8 + 缺口12）
# ---------------------------------------------------------------------------
# 老师改完软重生维 / 重写解析维（攒批标 dirty）→ 点「重生」按钮 → regen_dirty_items 对
# **待重生集合**（所有 dna_dirty 题）一次性重出，保留手改 manual 维（D-merge8）→ 清 dirty。
# 重生前每道题存 regen_snapshot（缺口12），「撤销重生」回上一版。
# 🔴 判决只读 sympy：重生稿走 _check_one_item（闸B + 批3 反退化闸自动复用）。
# ===========================================================================

# 重写解析 prompt（rewrite_solve 维：骨架/models 改 → 按新解法重写 solution，不动题面/答案）。


async def _rewrite_solve_once(item: dict, facts: dict) -> str | None:
    """重写解析维（skeleton/models 改）：按新基准重写 solution（题面/答案不动）。返回新 solution 文本。

    LLM 异常 / 解析不出 → None（上层降级保留原解析，G5）。判分不受影响（题面+答案没变，
    闸B 重验仍验原答案对不对）。
    """
    dna = facts.get("dna") or {}
    sk_raw = item.get("skeleton")
    if sk_raw is None:
        sk_raw = dna.get("skeleton")
    if isinstance(sk_raw, list):
        skeleton = "\n".join(str(s) for s in sk_raw if str(s).strip())
    else:
        skeleton = str(sk_raw or "")
    models_raw = item.get("models") if item.get("models") is not None else dna.get("models")
    models_txt = "、".join(
        str(m.get("name") or m.get("id") or "")
        for m in (models_raw or [])
        if isinstance(m, dict)
    ) or "（无指定模型）"
    prompt = _REWRITE_SOLVE_PROMPT.format(
        kp_name=facts["kp_name"],
        grade=facts["grade"],
        stem=str(item.get("stem") or ""),
        answer=str(item.get("answer") or ""),
        skeleton=skeleton or "（老师未填具体骨架步骤）",
        models=models_txt,
    )
    prompt = prompt + "\n\n" + _context_block(facts)
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)], model=settings.variant_model("generate")
        )
    except Exception:  # noqa: BLE001 — 重写解析是增强，LLM 异常 → 降级保留原解析（G5）
        return None
    parsed = _parse_json(text)
    if isinstance(parsed, dict) and str(parsed.get("solution") or "").strip():
        return _sanitize_rich_text(str(parsed["solution"]))
    return None


def _merge_preserve_manual(
    regen_item: dict[str, Any], old_item: dict[str, Any], mother_dirty_dims: list[str]
) -> dict[str, Any]:
    """🔴 D-merge8 核心·重生时保留手改、不被母题基准覆盖。

    思路：重生稿（regen_item）= 按新母题基准 + 软重生维重出的题面/解析；老师此前对**本题**手改过
    （manual_edited 的题级维：题型/难度/场景/题面/解析手输）要保住，重生只补「母题基准变化部分」。
    实现 = 题级 manual 维以 old_item 为准覆盖回 regen_item（仅当老师确实手改过该题，manual_edited=True）：
      - 老师没手改过本题（纯母题脏波及）→ 全量吃重生稿（按新基准重出，无手改要保）。
      - 老师手改过本题 → 题级手改维（qtype/difficulty/level + 老师手输的 stem/answer/solution 若 manual）
        保留 old，其余（受母题基准影响的解析/骨架口径）吃重生稿。
    🔴 母题脏波及维（mother_dirty_dims，如 skeleton/exam_type）= 必须更新的母题基准 → 不在保留之列。
    """
    if not old_item.get("manual_edited"):
        return regen_item  # 没手改 → 全量吃新基准重生稿
    merged = dict(regen_item)
    # 题级手改维：老师调过的题型/难度 = 本题专属意志，重生不冲（除非该维本身是母题脏波及维）。
    for k in ("qtype", "difficulty", "level"):
        if k not in mother_dirty_dims and old_item.get(k) is not None:
            merged[k] = old_item[k]
    # 保留手改印记（重生后仍是「手改过 + 已按新基准补」的题）。
    merged["manual_edited"] = True
    return merged


async def regen_dirty_items(
    state: VariantState, indexes: list[int] | None = None, token: str | None = None
) -> tuple[VariantState, dict[str, Any], str | None]:
    """🔴 手动「重生」入口（D-merge6/8 + 缺口12）：对待重生集合（dna_dirty 题）一次性重出。

    - indexes=None（母题改→自动重生待重生集合那条路）→ 全待重生集合（dirty_item_indexes），
      只重生 dirty 题，按脏维分流（保持原行为）。
    - 显式给定 indexes（来自 FE「重生这道」按钮，B5-fix）→ **一键重生这些题的语义**：
      不管改没改一律强制整题重出（force full regen），即使非 dirty（非 dirty 题 dirty_dims 空、
      不能错走「只重写解析」，必须走 _regen_once 给一道全新变式）。dirty 题仍按脏维分流不变。
    - 每道目标题：① 存 regen_snapshot（缺口12 撤销用）② 重出：
        · 含软重生维（题型/难度/考察类型/场景）→ _regen_once 重出整题 + _check_one_item 闸B（批3
          反退化闸自动复用）；
        · 仅重写解析维（骨架/models）脏 → _rewrite_solve_once 只重写解析 + _check_one_item 闸B 重验。
      ③ 保留手改不被母题基准覆盖（_merge_preserve_manual，D-merge8）④ 清 dirty。
    - 🔴 重生后该题 persisted 保留但因内容已变 → 入库走「覆盖原行 update by _persist_id」（缺口10，
      persist_items 据 _persist_id 走 update_question；见 persist_to_bank/persist_items）。
    返回 (update, result{regenerated:[idx], failed:[{idx,error}]}, error)。无 dirty → (空, {...}, None)。
    """
    from agents.variant.stage1_anchor.solve import _check_one_item  # strangler·破循环（调用期解析）
    from agents.variant.stage2_variant.gates import _regen_once  # strangler·破循环（调用期解析）

    items = list(state.get("items") or [])
    facts = _mother_facts(state)
    if not items:
        return {}, {"regenerated": [], "failed": []}, None
    # B5-fix：区分两种入口 —— 显式 indexes = force（一键重生这道，不管脏不脏）；
    # None = 自动重生 dirty 集合（母题改那条路），仍只重生确实 dirty 的（防误触发非脏题重出）。
    force = bool(indexes)
    if force:
        targets = [n for n in indexes if 1 <= n <= len(items)]
    else:
        targets = [n for n in dirty_item_indexes(items) if items[n - 1].get("dna_dirty")]
    if not targets:
        return {}, {"regenerated": [], "failed": []}, None

    new_items = [dict(it) for it in items]
    regenerated: list[int] = []
    failed: list[dict[str, Any]] = []
    drafts_to_discard: list[Any] = []  # 🔴 PRD-A-022：被替换掉的旧草稿 id（未发布）→ 末尾软删

    for n in targets:
        old = new_items[n - 1]
        old_draft_to_discard = _draft_id_to_discard(old)  # 重生前取旧草稿 id（已发布=None）
        snapshot = snapshot_item(old)
        dirty_dims = list(old.get("dirty_dims") or [])
        mother_dims = list(old.get("mother_dirty_dims") or [])
        # B5-fix：force（显式「重生这道」）→ 无条件整题重出（非 dirty 题 dirty_dims 空，
        #   不能错走「只重写解析」，老师要的是一道全新变式）。
        # None 路（自动重生 dirty 集合）→ 本题脏维涉及软重生维 → 整题重出；否则（仅 rewrite_solve
        #   维脏）→ 只重写解析（保持原行为）。
        all_dims = set(dirty_dims) | set(mother_dims)
        need_full_regen = force or any(d in _SOFT_REGEN_FIELDS for d in all_dims)

        try:
            if need_full_regen:
                seed = dict(old)
                seed["from_edit"] = True  # 闸B 见 from_edit：FAIL 不回炉换题，保留打 ⚠
                draft = await _regen_once(seed, facts, feedback=None)
                if not draft:
                    failed.append({"index": n, "error": "重出失败（模型未返回有效题目），已保留原题"})
                    continue
                draft["from_edit"] = True
                # 配方印记 + 入库簿记跟题走（重出仍占原槽位；_persist_id 留着 → 入库走覆盖）。
                # 🔴 PRD-A-022：**不 carry _draft_id**——draft 无 _draft_id（assemble 重落新草稿）；
                #   旧草稿（未发布）成功后 discard 软删。
                for k in ("_seq", "persisted", "_persist_id", "level"):  # R4·F17: 去死键 from_recipe/expected_difficulty
                    if old.get(k) is not None:
                        draft[k] = old[k]
                draft.pop("check", None)
                rechecked, _dropped = await _check_one_item(draft, facts, n - 1, len(new_items))
                final = rechecked if rechecked is not None else draft
                final = _merge_preserve_manual(final, old, mother_dims)
                _format_item_stem(final)
            else:
                # 仅重写解析维（骨架/models）脏 → 题面/答案不动，只重写 solution + 闸B 重验。
                new_solution = await _rewrite_solve_once(old, facts)
                final = dict(old)
                if new_solution:
                    final["solution"] = new_solution
                final["from_edit"] = True
                final.pop("check", None)
                # 🔴 PRD-A-022：解析重写 → 旧草稿内容失效，strip _draft_id（dict(old) 带过来的）→
                #   assemble 重落新草稿（带新解析）；旧草稿（未发布）成功后 discard 软删。
                final.pop("_draft_id", None)
                rechecked, _dropped = await _check_one_item(final, facts, n - 1, len(new_items))
                final = rechecked if rechecked is not None else final
                _format_item_stem(final)
        except Exception as e:  # noqa: BLE001 — 单题重生异常 → 记 failed，保留原题不丢（G5）
            failed.append({"index": n, "error": f"重生异常，已保留原题：{e}"})
            continue

        # 重生成功：存快照（撤销用）+ 清 dirty + 标 manual（重生过的题）
        final["regen_snapshot"] = snapshot
        final["manual_edited"] = True
        clear_item_dirty(final)
        # 🔴 PRD-A-018 M1③：已入库题重生后内容已变 → 标「内容已编辑待覆盖」（dna_dirty 已被
        #   clear_item_dirty 清掉，仅靠它 persist 会「已入库就跳过」漏掉新内容）→ 入库走 _persist_id 覆盖。
        mark_content_dirty_if_persisted(final)
        new_items[n - 1] = final
        regenerated.append(n)
        # 🔴 PRD-A-022：本题重生成功 → 旧草稿（未发布）被新内容替换 → 收集软删。
        if old_draft_to_discard is not None:
            drafts_to_discard.append(old_draft_to_discard)

    # 🔴 PRD-A-022·best-effort 软删被替换掉的旧草稿（token 来自调用方；缺/失败仅 log，绝不阻塞重生）。
    await _discard_drafts_best_effort(drafts_to_discard, token)

    update: VariantState = {"items": new_items}
    # 🔴 D-merge8·母题脏：全待重生集合都重生完（无指定 indexes 或 indexes 已覆盖所有 dirty）→ 清母题脏。
    remaining_dirty = dirty_item_indexes(new_items)
    if not remaining_dirty and (state.get("mother_dna") or {}).get("dirty"):
        md = dict(state.get("mother_dna") or {})
        md["dirty"] = False
        update["mother_dna"] = md
    return update, {"regenerated": regenerated, "failed": failed}, None


def undo_regen_item(
    state: VariantState, index: int
) -> tuple[VariantState, dict[str, Any] | None, str | None]:
    """🔴 撤销重生（缺口12）：第 index 道（1-based）回上一版重生前快照（regen_snapshot）。

    无快照（没重生过）→ (空, None, 错误串)。回上版后清掉该快照（一次撤销一版，不串版本）。
    返回 (update, restored_item, error)。
    """
    items = list(state.get("items") or [])
    if not isinstance(index, int) or index < 1 or index > len(items):
        return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
    cur = items[index - 1]
    snap = cur.get("regen_snapshot")
    if not isinstance(snap, dict):
        return {}, None, f"第 {index} 题没有可撤销的重生记录（未重生过）。"
    new_items = [dict(it) for it in items]
    restored = copy.deepcopy(snap)
    new_items[index - 1] = restored  # 整体回上版（含 dirty 状态、manual 印记，与重生前一致）
    update: VariantState = {"items": new_items}
    # 🔴 PRD-A-021 R4·F4：撤销重生须同步复位 mother_dna.dirty，否则不变量自相矛盾。
    #   场景：母题守恒维改 → mark_mother_dirty 标该题 dna_dirty + mother_dirty_dims，且 mother_dna.dirty=True；
    #   regen_dirty_items 重生完最后一道 dirty → 把 mother_dna.dirty 清成 False（7734）。此时老师撤销重生 →
    #   restored 又带回 dna_dirty=True（这是「重生前」快照），但旧实现只回 items、不回 mother_dna.dirty →
    #   出现「item.dna_dirty=True 而 mother_dna.dirty=False」的撕裂态：母题守恒维同步分支（keyed on
    #   mother_dna.dirty）漏触发。入库侧有 persist_dirty_guard 的 dna_dirty 闸双保险兜底（不会脏入库），
    #   但不变量须自洽 → 这里按「被撤销项若是因母题维脏（带 mother_dirty_dims）而 dna_dirty」回置 mother_dna.dirty。
    #   🔴 与 B7 不打架：B7（改主考点/维）在各自 turn 把 dirty 置 True（前向），本处 undo 在另一 turn 因
    #   un-regen 回 True（后向），方向一致、均朝「有脏待重生」收敛，互不吞（B7 不在 undo 路径上跑）。
    if restored.get("dna_dirty") and (restored.get("mother_dirty_dims") or []):
        md = dict(state.get("mother_dna") or {})
        if md and not md.get("dirty"):
            md["dirty"] = True
            update["mother_dna"] = md
    return update, restored, None
