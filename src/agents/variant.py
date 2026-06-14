"""PRD-C-009 · 图片举一反三 agent（块二·单题→N 道变式）。

事实源 = codeplace-C/claude-code-sign/14-举一反三-设计.md（vibe-coding：先改文档再改代码）。

主干 4 步流水线（§2）+ solve 自愈 + assemble 外显：
  analyze(读图) → classify(锚图谱+DNA置信闸) → [clarify] → generate(3=2普通+1难)
                → solve_explain(真解+自愈1次+守恒校验) → assemble(题组快照)

🔴 不变量（§3）：
  - 主考点 + 年级 = 硬守恒（贯穿 generate / 重生 / 补题 / patch）；
  - 凡进 items 的题一律过 solve_explain（无 check 不许进 assemble）；
  - generate 入口前置：mother_confirmed==true 或 三锚高置信（DNA 闸收口，多入口都过）。

LLM 层（§9）：toolkit 原生 get_model（COMPATIBLE=LangChain ChatOpenAI）→ model.ainvoke；
  多模态走 HumanMessage(content=[{type:text},{type:image_url,image_url:{url:OSS_URL}}])；
  思考型只取 content（先 reasoning_content 后 content）；max_tokens≥4096；
  LLM 外呼 lk888 走默认（别套 trust_env=False，那是治本地 localhost 的）。

checkpointer 不在此 compile（service lifespan 注入 saver；多轮 state 按 thread_id 持久）。
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import difflib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langgraph.config import get_stream_writer
from langgraph.graph import END, MessagesState, StateGraph

from agents import conv_trace, dna_extract, math_verify, model_anchor
from agents.qtype_format import format_by_qtype
from agents.variant_support import (
    RuoyiClient,
    _is_review_book,
    anchor_subject,
    leaf_pool_for_grade,
    persist_items,
    tag_pool_for_kp,
)
from core import get_model, relay_pool, settings

# DNA 三锚高置信门槛
CONF_GATE = 0.75
# 自愈上限（设计 §5：1 次防死循环）
MAX_HEAL = 1
# 🔴 PRD-C-015 批3·⑦ 反退化闸重生成上限（§1⑦：≤重生成上限，超限则弃该变式）。1 次防死循环。
MAX_DEGEN_REGEN = 1


def _regen_max_tokens() -> int:
    """🔴 整改4：闸B 回炉（REGEN）瘦身 max_tokens。回炉只带单题题面+错因+确定上下文块，
    不灌整组上下文 → 单独压一个上限防输出失控。≤0 回退 VARIANT_MAX_TOKENS（关闸）。"""
    v = int(getattr(settings, "VARIANT_REGEN_MAX_TOKENS", 0) or 0)
    return v if v > 0 else settings.VARIANT_MAX_TOKENS
# 默认配方：3 道 = 2 普通 + 1 难（设计 §5）
DEFAULT_SHAPE = {"normal": 2, "hard": 1}
# P2 逐题过闸并发上限（PRD-C-012：generate 流内 eager + gene_gate/solve_explain 节点共用口径）
GATE_CONCURRENCY = 3

# ---------------------------------------------------------------------------
# P13 预算闸（PRD-C-013）：state 级 LLM 调用计数器（per-round 重置）。
# 🔴 铁律：是「超限跳过增强类调用」不是「LLM 决定流程」——宏观 DAG 一字不破。
#   核心链（parse/generate 首稿/grade 难度总评/solve 真解）永不跳；只有**增强类**调用
#   （闸A rework 回炉 / 闸B heal 回炉 / replenish 补题 / extract 兜底抽载荷）在超限后
#   跳过，落既有 G5 降级路径（标 ⚠ / 保留原题，绝不卡死）。
# 实现：contextvar 持一个 {"used":int,"limit":int} 计数器（per graph round 由出题/编辑节点
#   入口 _budget_begin 重置）。_ainvoke_text 每次成功调用 _budget_tick()+1；增强类调用点
#   先问 _budget_exhausted() 再决定跳不跳。contextvar 天然随 asyncio task 复制传播 →
#   eager 并发子 task / gather 并发都共享同一计数器（同一轮预算），单测直调节点（无
#   begin）时 _budget 为 None → 永不超限（行为回退到老逻辑，零侵入）。
# ---------------------------------------------------------------------------
_budget_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "variant_llm_budget", default=None
)


def _budget_begin(limit: int) -> None:
    """出题/编辑轮入口重置预算（per-round）。limit≤0 视为不设限（关闸）。"""
    _budget_ctx.set({"used": 0, "limit": int(limit)} if limit and limit > 0 else None)


def _budget_tick() -> None:
    """记一次成功 LLM 调用（_ainvoke_text 内部唯一调用点）。无预算上下文 → no-op。"""
    b = _budget_ctx.get()
    if b is not None:
        b["used"] += 1


def _budget_exhausted() -> bool:
    """增强类调用点的闸：True=预算已耗尽，本次增强调用应跳过走降级。无预算 → 永 False。"""
    b = _budget_ctx.get()
    return b is not None and b["used"] >= b["limit"]


def _budget_bind(state: VariantState, *, reset_limit: int | None = None) -> dict | None:
    """节点入口绑定预算到 contextvar，返回 live 计数器 dict（节点须把它放回返回值 state，
    used 才能跨节点累计——LangGraph 每个 superstep 用新 copy_context，contextvar 不跨节点存活，
    预算的**事实源是 state.llm_call_budget**，contextvar 只是给无 state 视野的 _ainvoke_text 记账）。

    - reset_limit 非 None（出题/编辑轮**入口**节点）→ 本轮重置 {"used":0,"limit":reset_limit}；
      limit≤0 视为关闸（返回 None，永不超限）。
    - reset_limit=None（轮内下游节点 gene_gate/solve_explain/assemble/exec_*）→ 从 state 携带；
      state 无簿记（单测直调 / 旧线程恢复）→ None（关闸，行为回退老逻辑）。
    """
    if reset_limit is not None:
        b = {"used": 0, "limit": int(reset_limit)} if reset_limit > 0 else None
    else:
        carried = state.get("llm_call_budget")
        b = dict(carried) if isinstance(carried, dict) and "limit" in carried else None
    _budget_ctx.set(b)
    return b

# ---------------------------------------------------------------------------
# 闸B（PRD-C-010）：sympy 程序验算 + 题型分流的标记值
# —— 落 item.check.verify / item.check.review（FE 4d 徽章/快照展示）。B1 后 auxTags 列已
#    DROP，验算审计走 BE ai 表 conflict_flags，不再随 create BO 透传。
# 🔴 判决只读 math_verify.verify() 的 verdict（pass/fail/degrade），永不采信 LLM 自评。
# ---------------------------------------------------------------------------
VERIFY_SYMPY_PASS = "sympy_pass"  # 程序验算通过（入库可查的抓手）
# 🔴 验算墙钟预算（G5 反挂死）：sympy 对病态载荷（如 9**9**9 / 超高次方程）可能无界计算，
# _machine_verify 用 asyncio.wait_for 包 to_thread —— 超时按 degrade 降级（线程不可杀但流程解锁）。
VERIFY_TIMEOUT_S = 10.0
# 🔴 4d 方案A 后 fail_after_regen 不再外发（真 fail 题剔除）；常量保留：旧库存量 aux_tags
# 仍有该值（FE 兜底/审计查询用），勿删。
VERIFY_FAIL_AFTER_REGEN = "fail_after_regen"
VERIFY_UNVERIFIED = "unverified"  # degrade：sympy 吃不下 → 退回 LLM 自检 fallback
REVIEW_PROOF = "proof_needs_human"  # 证明/开放/作图类：不进 sympy，转人审

# 题卡可见文本（追加在 item.solution 尾部 → 入库 analyze 字段同步可见）。
# 🔴 4d 可见性矩阵（PRD-C-012，用户拍板 2026-06-11「只说好、不说坏，除非双闸都不高」）：
# 外显只有 正面/中性/沉默 三类，⚠ 仅双闸（闸B verify × 闸A gene）皆存疑；
# 真值 check.verify / gene.gate 原样入 aux_tags 审计，不洗白。
NOTE_VERIFIED_OK = "✓ 程序验算通过。"
NOTE_SELF_CHECK_OK = "✓ 已独立复算一致。"
NOTE_BOTH_GATES_LOW = "⚠ 程序验算与平行度双重存疑，请老师重点核对。"
NOTE_PROOF_REVIEW = "ℹ 证明/开放类题不做程序验算，已转人工复核。"

# 题卡外显层级（check.tier → artifact.tier → FE 徽章；展示层与真值解耦）
TIER_VERIFIED = "verified"  # 强正面：sympy 验算通过
TIER_SELF_OK = "self_ok"  # 轻正面：程序不可验，LLM 独立复算一致
TIER_PROOF = "proof"  # 中性：证明/开放类转人审
TIER_SILENT = "silent"  # 沉默：单闸存疑（不说坏）
TIER_BOTH_LOW = "both_low"  # ⚠：双闸皆存疑
TIER_MANUAL = "manual"  # 中性：老师手动编辑、验算待重跑（题组编辑器 /variant/edit-item）

# ---------------------------------------------------------------------------
# 闸A·基因闸（验"是不是平行题"，与闸B"答案对不对"正交。依据 12-题目DNA方法论 §2/§5）：
# 🔴 B2·T2 退役换血（PRD-C-014）：闸A LLM judge 全链已删，内涵改为**纯代码三检**
#   （gene_gate_check）：① structure_lint 题型结构 lint；② _surface_check 表皮距离防抄题；
#   ③ 守恒透传（题型/考察类型守恒的声明性校验）。任一检命中 = gene=warn + flags（只警示
#   不硬拦、不回炉、不剔题，闸门必有降级路径）；三检全过 = gene=pass。
# 标记落 item.gene.gate（+ flags/reason）→ FE 4d 展示 + 快照透传；BO 不带 gene（B1 收敛）。
# 单闸不再外显负面（沉默），仅参与 _apply_visibility 双闸裁决。
# ---------------------------------------------------------------------------
GENE_GATE_PASS = "pass"  # 纯代码三检全过（合格平行题）
GENE_GATE_WARN = "warn"  # 三检任一命中 → 真值 + flags 留档；外显层按 4d 矩阵裁决
GENE_GATE_SKIPPED = "skipped"  # 历史值（judge 退役后纯代码不再产生 skipped；保留供旧线程兼容）


# ---------------------------------------------------------------------------
# State（设计 prompt 指定结构）
# ---------------------------------------------------------------------------
class VariantState(MessagesState, total=False):
    image_url: str | None
    images_count: int
    questions_in_image: int
    # analysis：年级/考点(kp)/题型(qtype) 各带置信
    analysis: dict[str, Any]
    mother_dna: dict[str, Any]
    mother_confirmed: bool
    # items[{stem, answer, solution, qtype, difficulty, level, injected_kp?,
    #        check:{badge:ok|warn, solved_answer}  ← 闸B(solve_explain)填,
    #        gene:{gate:pass|warn|skipped, reason?} ← 闸A(gene_gate)填}]
    items: list[dict[str, Any]]
    history: list[dict[str, Any]]
    # 交互层：parse_instruction 的解析结果（intent/ops/knobs/...），路由后各分支消费并清空
    pending: dict[str, Any] | None
    # 🔴 首轮配方旋钮（设计 §5 五旋钮的"首轮接线"）：None=还没抽过；{}=抽过但老师没提(走默认)。
    # {"count": int, "difficulty_plan": "increasing"|str, "qtype_dist": {"选择":2,...}, "note": str}
    # analyze（新母题轮）负责抽取/重置：新图新要求 → 重抽覆盖；同图重贴无新要求 → 保留；
    # 新图无要求 → 重置 {}（旧母题配方绝不泄漏到新母题）。generate 仅对库内母题路径兜底抽。
    knobs: dict[str, Any] | None
    # generate 的代码级配方校验缺陷清单（整组 retry 1 次后仍不符 → assemble 头部外显 ⚠）
    shape_defects: list[str]
    # 4d 方案A（PRD-C-012）：本轮被剔除题的叙事（sympy 证实标答错且重生未果 → 不外发），
    # solve_explain 每轮重写（非累计），assemble 摘要外显「本组少 N 道」
    dropped_notes: list[str]
    # 🔴 P9 手排 sticky（对抗审③·PRD-C-013）：exec_reorder 置 True，标记老师已手动排过序。
    # assemble 见 True 时**跳过** _sort_by_difficulty（默认难度升序排序），不静默重排覆盖手排；
    # 改变题集的编辑（exec_add/exec_remove）清掉该标记（题集变了，手排次序失效，回默认序）。
    # exec_regenerate 是原位改单题不变序 → 不清（手排保留）。
    manual_order: bool
    # 🔴 P13 预算闸（PRD-C-013）：state 级 LLM 调用计数器 {"used":int,"limit":int}。
    # 出题/编辑轮**入口**节点（generate / parse_instruction）按 settings 重置；轮内下游节点
    # （gene_gate/solve_explain/assemble/exec_*）从 state 携带、累计 used，并把它放回返回值
    # state（跨 superstep 保活）。超限后增强类调用跳过走 G5 降级（_budget_exhausted）。
    llm_call_budget: dict[str, int] | None
    # 🔴 批3（2026-06-13）·事实源冻结：每批次一份「老师锚准的事实源」（年级学期/主考点/DNA
    #   基础元素 = analysis.grade/kp + mother_dna.dna）。定死/确认时 facts_locked 置位 → 此后
    #   **只许老师指令改、LLM 输出不许反向覆盖**。每次老师修正记一条 audit（字段/旧值/新值/
    #   指令原文）。不搞重型版本系统，setter 统一收口写入。
    facts_locked: bool
    facts_audit: list[dict[str, Any]]
    # ===================================================================
    # 🔴 PRD-C-015 批1·DNA 契约 v2 增量（在 C-014 v1 上加，不动既有字段；一次升 v2 无 v1.5）
    # ===================================================================
    # (a) 模型维（原双轴；批1 仅建字段占位，批2 才真填）：
    #   mother_dna.dna.models = [{id,name}]（1~3 项，非空）由批2 model_anchor 写；
    #   mother_dna.dna.model_overflow = [str]（池外名，⚠/待命名池用）。批1 不动 dna_extract 产物，
    #   字段缺省即「未抽」（None/缺），下游容缺。
    # (b) 母题守恒确认状态（块①·D-merge7 确定性异常门控·非置信非硬锁 + 缺口5 合并闸）：
    #   mother_confirm = {
    #     flags: [str],            # 确定性异常标记（FLAG_SECONDARY_KP_OOB/EXAM_TYPE_OOB/SKELETON_EMPTY），无逐维置信分
    #     needs_confirm: bool,     # (flags 非空) ∨ (年级+主考点三锚没定死) → 弹合并确认面；否则直接放行
    #     confirmed_dims: [str],   # 老师已过/改的守恒维（留痕用）
    #     audit_ref: int | None,   # 指向 facts_audit 的索引（不另起审计表）
    #   }
    mother_confirm: dict[str, Any]
    # (c) DNA 改→重生四分流·待重生态·防脏·快照（块③·批1 仅建字段，批4 才接真重生逻辑）：
    #   - regen_class 是约定/枚举映射（哪维属哪类）= 模块级常量 REGEN_CLASS（前后端共用，不逐题落库）。
    #   - items[i].dna_dirty: bool（重生维/骨架/models 改置位；纯元数据维改不置位）→ 致命①入库硬闸。
    #   - items[i].regen_snapshot: 重生前快照（缺口12 撤销重生，批4 真用）。
    #   - mother_dna.dirty: bool（母题守恒维改置位）。
    #   - regen_dirty: 会话态，已改未重生的重生维列表（打角标；待重生集合=变式自身脏∪母题脏波及）。
    regen_dirty: list[str]


# ---------------------------------------------------------------------------
# 🔴 批3·事实源单向写 setter（年级/主考点 = analysis.grade/kp）
# 写来源两类：
#   - source="teacher"：老师指令路径（patch 的修正 / parse 路由的 mother_correction /
#     exec_solution_only 的 grade_correction）—— 永远放行，且记 audit 留痕。
#   - source="llm"：LLM 输出回写（classify 锚定抬置信等）—— facts_locked 后**忽略 + warn**，
#     定死前正常写（锚定就是靠它达成定死的）。
# ---------------------------------------------------------------------------
import logging as _logging  # noqa: E402

_facts_log = _logging.getLogger("variant.facts")


def _fact_edit(
    analysis: dict[str, Any],
    field: Literal["grade", "kp"],
    new_value: Any,
    *,
    source: Literal["teacher", "llm"],
    locked: bool,
    audit: list[dict[str, Any]],
    instruction: str | None = None,
    confidence: float | None = None,
    clear_keys: tuple[str, ...] = (),
) -> bool:
    """统一改 analysis.grade/kp 的 value（+可选 confidence）。返回是否实际写入。

    🔴 locked 后非老师指令来源（source!="teacher"）的写 → 直接忽略 + log warning（事实源冻结）。
    老师指令来源 → 始终放行，且记一条 audit（字段/旧值/新值/指令原文）。
    """
    if locked and source != "teacher":
        _facts_log.warning(
            "facts_locked: 忽略 LLM 来源对 %s 的回写（new=%r）——事实源已冻结，只许老师指令改",
            field, new_value,
        )
        return False
    node = dict(analysis.get(field) or {})
    old_value = node.get("value")
    node["value"] = new_value
    if confidence is not None:
        node["confidence"] = confidence
    for ck in clear_keys:
        node.pop(ck, None)
    analysis[field] = node
    if source == "teacher":
        audit.append({
            "field": field,
            "old": old_value,
            "new": new_value,
            "instruction": instruction or "",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    return True


# ===========================================================================
# 🔴 PRD-C-015 批1·DNA 契约 v2 地基：四分流枚举映射 + 守恒维确定性异常 + 合并确认闸
#    + 守恒维事实源冻结 setter（_dna_fact_edit）
# ===========================================================================
# DNA 改→重生四分流（regen_class）：前后端共用常量、不逐题落库（§10.1(c)）。
#   hard_anchor  【主考点/年级】→ 改 = 立即解冻重锚（不进 dirty，缺口7）
#   soft_regen   【题型/难度/考察类型/场景】→ 改 = 标 dirty，点「重生」_regen_once 重出
#   rewrite_solve【解法骨架/models】→ 改 = 重写解析过闸B，置 dirty 直到重写完（D-merge9）
#   meta         【标签/副考点】→ 改 = 只标注即时生效，不进 dirty
# 🔴 维名用 edit-dna 契约维 key（main_kp/grade/qtype/difficulty/exam_type/scene/skeleton/
#    models/tags/secondary_kps）。批1 仅建映射；批4 据此驱动 dirty/重生。
REGEN_CLASS: dict[str, str] = {
    "main_kp": "hard_anchor",
    "grade": "hard_anchor",
    "qtype": "soft_regen",
    "difficulty": "soft_regen",
    "exam_type": "soft_regen",
    "scene": "soft_regen",
    "skeleton": "rewrite_solve",
    "models": "rewrite_solve",
    "tags": "meta",
    "secondary_kps": "meta",
}

# 守恒基准维确定性异常 flag（D-merge7；与 dna_extract 同源常量，避免循环 import 各持一份引用）。
FLAG_SECONDARY_KP_OOB = dna_extract.FLAG_SECONDARY_KP_OOB
FLAG_EXAM_TYPE_OOB = dna_extract.FLAG_EXAM_TYPE_OOB
FLAG_SKELETON_EMPTY = dna_extract.FLAG_SKELETON_EMPTY
# 三个确定性异常 = 母题守恒维门控只看这三类（绝不引 LLM 自报置信，铁律「判决不信 LLM 自评」）。
_MOTHER_CONFIRM_FLAGS: tuple[str, ...] = (
    FLAG_SECONDARY_KP_OOB,
    FLAG_EXAM_TYPE_OOB,
    FLAG_SKELETON_EMPTY,
)


def mother_confirm_flags(dna: dict[str, Any] | None) -> list[str]:
    """🔴 从母题 DNA 算守恒基准维的**确定性异常**（D-merge7，纯函数·零 LLM·可单测）。

    只产三类，全部由代码确定性判出，**不读任何 LLM 自报置信分**：
      - FLAG_SECONDARY_KP_OOB：副考点越界（dna_extract._validate 已在抽取时把越界副 kp 丢弃并打此 flag，
        本函数据 DNA.flags 透出——副 kp 越界的事实只有抽取期才知道池，故采信抽取期 flag）。
      - FLAG_EXAM_TYPE_OOB：考察类型出闭集（兼采信抽取期 flag + 当前值兜底重算：exam_type 非空且不在闭集）。
      - FLAG_SKELETON_EMPTY：解法骨架为空（当前 DNA.skeleton 空即异常，确定性重算）。
    返回去重列表（保持检测顺序）。dna 缺失/空 → 视同骨架为空（[FLAG_SKELETON_EMPTY]）。
    """
    dna = dna or {}
    src_flags = set(dna.get("flags") or [])
    out: list[str] = []

    # 副考点越界：抽取期事实（_validate 丢弃越界副 kp 时打 flag），本函数透出。
    if FLAG_SECONDARY_KP_OOB in src_flags:
        out.append(FLAG_SECONDARY_KP_OOB)

    # 考察类型出闭集：抽取期 flag 或 当前值确定性重算（非空且不在闭集 = 异常）。
    exam_type = dna.get("exam_type")
    if FLAG_EXAM_TYPE_OOB in src_flags or (
        exam_type is not None and exam_type not in dna_extract.EXAM_TYPES
    ):
        out.append(FLAG_EXAM_TYPE_OOB)

    # 解法骨架为空：当前 DNA 确定性重算（不信抽取期，因老师可能已补/清骨架）。
    skeleton = [s for s in (dna.get("skeleton") or []) if str(s).strip()]
    if not skeleton:
        out.append(FLAG_SKELETON_EMPTY)

    return out


def build_mother_confirm(state: VariantState) -> dict[str, Any]:
    """🔴 合并确认闸（缺口5 + D-merge7）：把【年级+主考点三锚置信门控（C-014 既有，不动）】
    与【母题守恒维确定性异常】合并算成一份 mother_confirm 状态。

    needs_confirm = (守恒维有确定性异常) ∨ (年级+主考点三锚没定死) —— 任一未达标 → 弹合并确认面
    （一次外显三锚 + 守恒四维，不分两段问）；全达标无异常 → 直接放行出变式（非硬锁按钮，M2）。
    🔴 年级+主考点的三锚置信门控沿用 C-014 既有 _pin_status（不并入 mother_confirm，§10.1(b)）。
    """
    dna = (state.get("mother_dna") or {}).get("dna") or {}
    flags = mother_confirm_flags(dna)
    pin = _pin_status(state)
    needs_confirm = bool(flags) or not pin.get("pinned")
    return {
        "flags": flags,
        "needs_confirm": needs_confirm,
        # 现有确认维（批1 建字段，老师过/改时由 setter/编辑路径回填，批4/5 接 UI）
        "confirmed_dims": list((state.get("mother_confirm") or {}).get("confirmed_dims") or []),
        "audit_ref": (state.get("mother_confirm") or {}).get("audit_ref"),
    }


# 母题守恒基准维（4 维）→ 在 mother_dna.dna 里的键 + 人话名（facts_locked 扩维用，缺口6）。
#   副考点/考察类型/难点 是标注/基准维（改不必重出母题题面）；解法骨架最难步是基因维（改=重写解析）。
_DNA_CONSERVE_FIELDS: dict[str, str] = {
    "secondary_kps": "副考点",
    "exam_type": "考察类型",
    "skeleton": "解法骨架",
    "hard_points": "难点",
}


def _dna_fact_edit(
    mother_dna: dict[str, Any],
    field: Literal["secondary_kps", "exam_type", "skeleton", "hard_points"],
    new_value: Any,
    *,
    source: Literal["teacher", "llm"],
    locked: bool,
    audit: list[dict[str, Any]],
    instruction: str | None = None,
) -> bool:
    """🔴 缺口6·守恒维事实源冻结 setter：统一改 mother_dna.dna 的 4 个守恒维。返回是否实际写入。

    与 _fact_edit（grade/kp）同语义、把冻结物理边界扩到 4 守恒维（§3.4「事实源冻结」的物理前提）：
      - locked 后 source!="teacher"（LLM 来源）的回写 → **忽略 + warn + audit 留痕**（不静默）；
      - 老师来源（source=="teacher"）→ 始终放行，记一条 audit（字段/旧值/新值/指令原文）。
    🔴 只负责单向写守恒维 + 冻结拦截；不在此处置 dirty / 触发重生（那是批4 的活）。
    """
    if field not in _DNA_CONSERVE_FIELDS:
        return False
    dna = dict(mother_dna.get("dna") or {})
    old_value = dna.get(field)
    if locked and source != "teacher":
        _facts_log.warning(
            "facts_locked: 忽略 LLM 来源对守恒维 %s 的回写（new=%r）——事实源已冻结，只许老师指令改",
            field, new_value,
        )
        # 冻结拦截也留 audit（区别于放行：ignored=True），便于审计「谁试图改而被挡」。
        audit.append({
            "field": field,
            "old": old_value,
            "new": new_value,
            "source": "llm",
            "ignored": True,
            "instruction": instruction or "",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return False
    dna[field] = new_value
    mother_dna["dna"] = dna
    if source == "teacher":
        audit.append({
            "field": field,
            "old": old_value,
            "new": new_value,
            "source": "teacher",
            "instruction": instruction or "",
            "ts": datetime.now(timezone.utc).isoformat(),
        })
    return True


# ===========================================================================
# 🔴 PRD-C-015 批4·DNA 改→重生四分流状态机辅助（块③ R1/R2/R2b/R2c + 致命① + 缺口7/10/12）
# ---------------------------------------------------------------------------
# 四分流由 REGEN_CLASS 驱动（批1 已建映射）。本段是「改一个维 → 该做什么」的纯逻辑：
#   - regen_class_of(field)：维 → 分流类。
#   - mark_item_dirty / clear_item_dirty：item 级 dirty 角标增删（regen_dirty 是会话态待重生集合）。
#   - mark_mother_dirty：母题守恒维改 → mother_dna.dirty=True + 下游变式标 dirty（D-merge8 回流）。
#   - 重生入口 = regen_dirty_items（service.py /variant/regen 调）：对待重生集合一次性重出，
#     保留手改 manual 维（D-merge8）；重生前存 regen_snapshot（缺口12 撤销）。
#   - undo_regen_item（/variant/undo-regen）：item 回上一版快照。
#   - has_dirty / persist_dirty_guard：致命① 入库防脏硬闸。
# 🔴 判决只读 sympy（重生稿走 _check_one_item / _anti_degen_gate，批3 已接，自动复用）。
# ===========================================================================

# 软重生维（点「重生」按 _regen_once 重出整题）vs 重写解析维（只重写 solution，不重出题面）。
_SOFT_REGEN_FIELDS: tuple[str, ...] = tuple(
    k for k, v in REGEN_CLASS.items() if v == "soft_regen"
)
_REWRITE_SOLVE_FIELDS: tuple[str, ...] = tuple(
    k for k, v in REGEN_CLASS.items() if v == "rewrite_solve"
)


def regen_class_of(field: str) -> str | None:
    """维 → 四分流类（hard_anchor/soft_regen/rewrite_solve/meta）；未知维 → None。"""
    return REGEN_CLASS.get(field)


def _dirty_tag(field: str) -> str:
    """item.dna_dirty 是 bool；regen_dirty（会话态待重生集合）按「idx:field」打标，
    既记是哪道题脏、又记哪一维脏，FE 据此点角标。母题脏波及用「idx:mother:field」。
    """
    return field


def mark_item_dirty(item: dict[str, Any], field: str) -> None:
    """改了软重生维 / 重写解析维 → 置 item.dna_dirty=True 并记下脏维（dirty_dims，去重）。

    🔴 元数据维（meta：tags/secondary_kps/hard_points）改不调本函数（不进 dirty，§3.2c）；
       硬锚维（hard_anchor）改走解冻重锚路径，也不进 dirty（缺口7）。
    """
    item["dna_dirty"] = True
    dims = list(item.get("dirty_dims") or [])
    if field not in dims:
        dims.append(field)
    item["dirty_dims"] = dims


def clear_item_dirty(item: dict[str, Any]) -> None:
    """重生完成 → 清 item.dna_dirty + dirty_dims + 母题波及标记（mother_dirty_dims）。"""
    item["dna_dirty"] = False
    item.pop("dirty_dims", None)
    item.pop("mother_dirty_dims", None)
    item.pop("mother_baseline", None)


def has_dirty(state: VariantState) -> bool:
    """致命①：会话内是否存在 dna_dirty 的变式 或 mother_dna.dirty。"""
    if (state.get("mother_dna") or {}).get("dirty"):
        return True
    return any(bool(it.get("dna_dirty")) for it in (state.get("items") or []))


def dirty_item_indexes(items: list[dict[str, Any]]) -> list[int]:
    """待重生集合（1-based 题号）= 所有 dna_dirty 的变式。"""
    return [i + 1 for i, it in enumerate(items) if it.get("dna_dirty")]


def persist_dirty_guard(state: VariantState) -> str | None:
    """致命① 入库防脏硬闸：存在 dna_dirty 题 → 返回拒绝提示串；全 not dirty → None。

    提示精确到「第 N 题」（N = 1-based 待重生题号；母题脏单列）。
    """
    items = state.get("items") or []
    dirty_idx = dirty_item_indexes(items)
    mother_dirty = bool((state.get("mother_dna") or {}).get("dirty"))
    if not dirty_idx and not mother_dirty:
        return None
    parts: list[str] = []
    if dirty_idx:
        nums = "、".join(f"第 {n} 题" for n in dirty_idx)
        parts.append(nums)
    if mother_dirty:
        parts.append("母题")
    subjects = "、".join(parts)
    return f"{subjects}改了还没重生，先点「重生」或「撤销重生」再入库。"


# 守恒维改 → 下游变式要随基准变化的「母题基准维」集合（D-merge8 回流，重生时只补这些维）。
# secondary_kps / exam_type 是守恒白名单/考察类型基准；skeleton 是基因基准；hard_points 是难点基准。
_MOTHER_BASELINE_DIMS: tuple[str, ...] = ("secondary_kps", "exam_type", "skeleton", "hard_points")


def mark_mother_dirty(state_mother_dna: dict[str, Any], items: list[dict[str, Any]], field: str) -> None:
    """🔴 D-merge8·母题守恒维改 → 母题脏 + 下游变式标 dirty 不自动重出（并入待重生集合）。

    - mother_dna.dirty=True（致命① 也拦母题入库 / 已入库 role=mother 行待同步）。
    - 每道下游变式：置 dna_dirty + 记 mother_dirty_dims（哪些母题基准维波及本题）+ 存一份
      mother_baseline 快照（重生时据它「只补母题基准变化部分、不冲掉 manual 精修」，见 regen_dirty_items）。
    🔴 骨架（skeleton）改 = 基因变 → 走 rewrite_solve 语义（重生时变式 solution 跟新基因重写）；
       但回流统一标 dirty，点「重生」时按维分别处置，本函数只标脏不动题面。
    """
    state_mother_dna["dirty"] = True
    for it in items:
        it["dna_dirty"] = True
        mdims = list(it.get("mother_dirty_dims") or [])
        if field not in mdims:
            mdims.append(field)
        it["mother_dirty_dims"] = mdims


def snapshot_item(item: dict[str, Any]) -> dict[str, Any]:
    """重生前快照（缺口12）：深拷一份 item（去掉旧 regen_snapshot 防快照套娃膨胀）。"""
    snap = {k: v for k, v in item.items() if k != "regen_snapshot"}
    return copy.deepcopy(snap)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _model():
    # COMPATIBLE provider → gemini-3-flash-preview（思考型+多模态）
    return get_model(settings.DEFAULT_MODEL)


def _content_text(resp: BaseMessage) -> str:
    """思考型只取 content（reasoning_content 不外放）。content 可能是 str 或 parts list。"""
    c = resp.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


# ---------------------------------------------------------------------------
# LLM 往返持久化（PRD-C-009：给维护者过 prompt/排查用）
# —— 每次 LLM 调用落一行 JSONL：填充后的完整 prompt + 模型原始返回 + 哪个 prompt + 耗时。
# 关掉设 env VARIANT_LLM_TRACE=0；落盘失败绝不影响主流程。
# ---------------------------------------------------------------------------
_LLM_TRACE_ENABLED = os.getenv("VARIANT_LLM_TRACE", "1").lower() not in ("0", "false", "no")
_LLM_TRACE_PATH = Path(__file__).resolve().parents[2] / "data" / "llm_trace.jsonl"
_LLM_TRACE_SEQ = 0  # 进程内自增序号（同一进程内调用顺序）

# prompt 内容前缀 → 标签（哪个 prompt）。新增/造同含「基于母题 DNA」，先判 add 再判 generate。
_TRACE_MARKERS: list[tuple[str, str]] = [
    ("看这张题目图", "analyze"),
    ("出题配方", "knobs"),
    ("数学验算载荷抽取器", "extract"),
    # 🔴 Q1 可观测性修复（2026-06-12）：锚定/DNA 抽取这一步从此经 _ainvoke_text → 落 trace。
    #   两个 marker（DNA 抽取主调 + 标签复用窄调）须排在 generate 的「命题专家」泛 marker 之前。
    ("打标式 DNA 抽取", "dna_extract"),
    ("检索标签师", "dna_tags"),
    # 🔴 B2·T2：闸A LLM judge 已退役（gene_judge 全链删），不再有「平行题基因比对器」调用。
    ("独立解出的答案与题面标答不一致", "regen"),
    ("你是严谨的数学阅卷老师", "solve"),
    ("举一反三 agent 的指令解析器", "parse"),
    ("老师对下面这组变式题的某道有疑问", "answer"),
    # 整改3（2026-06-12）：解法修正逐题解析重写器（题面留只改解析）。
    ("解题方法提了新约束", "solution_rewrite"),
    ("**新增**", "add"),
    ("举一反三变式", "generate"),
]


def _msg_text(m: BaseMessage) -> str:
    """取一条消息的文本（多模态 list 取其中 text part）。"""
    c = m.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                return p.get("text", "")
    return ""


def _trace_label(messages: list[BaseMessage]) -> str:
    head = "".join(_msg_text(m) for m in messages)[:120]
    for marker, label in _TRACE_MARKERS:
        if marker in head:
            return label
    return "unknown"


def _serialize_request(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = m.__class__.__name__
        c = m.content
        if isinstance(c, str):
            out.append({"role": role, "text": c})
        elif isinstance(c, list):
            parts: list[dict] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append({"text": p.get("text", "")})
                elif isinstance(p, dict) and p.get("type") == "image_url":
                    parts.append({"image_url": (p.get("image_url") or {}).get("url")})
                else:
                    parts.append({"raw": str(p)})
            out.append({"role": role, "parts": parts})
        else:
            out.append({"role": role, "text": str(c)})
    return out


def _trace_llm(
    label: str,
    messages: list[BaseMessage],
    response_text: str,
    response_raw: Any,
    duration_ms: int,
    error: str | None = None,
    retried: bool = False,
    model: str | None = None,
) -> None:
    if not _LLM_TRACE_ENABLED:
        return
    global _LLM_TRACE_SEQ
    _LLM_TRACE_SEQ += 1
    try:
        rec = {
            "seq": _LLM_TRACE_SEQ,
            "ts": datetime.now(timezone.utc).isoformat(),
            "label": label,
            # 🔴 与 conv_trace 同源：实际成交中转站的 model（不再写 DEFAULT_MODEL 枚举名）
            "model": model or settings.COMPATIBLE_MODEL,
            "duration_ms": duration_ms,
            "retried": retried,
            "request": _serialize_request(messages),
            "response": response_text,
            "response_raw": response_raw,
        }
        if error:
            rec["error"] = error
        _LLM_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LLM_TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass  # 持久化绝不拖垮主流程


async def _ainvoke_text(
    messages: list[BaseMessage],
    retry: bool = True,
    *,
    public_stream: bool = False,
    on_delta: Any = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """ainvoke + 取 content；偶发空返回重试一次。max_tokens≥4096 给思考型留头。

    🔴 每次调用落 JSONL 往返记录（_trace_llm）：发送的完整 prompt + 原始返回。
    🔴 思维外放（用户反馈 2026-06-11）：默认打 skip_stream 标签 —— JSON 类中间产物的
    token 流对用户是乱码，service 按标签丢弃；只有人话型调用（答疑等）传
    public_stream=True，token 才透到前端打字机。on_delta=流内进度回调（拿累计文本），
    generate 用它数「已写到第几题」。
    model：per-call 模型覆盖（S1.1）。给了就用它换该次请求的 model（站点不变），轻活
    调用点（nano 降本）传 settings.LLM_MODEL_LIGHT；None = 沿用 relay 配置 model（旧行为不变）。
    max_tokens：per-call max_tokens 覆盖（整改4·回炉瘦身）。None/≤0 → 默认 VARIANT_MAX_TOKENS。
    temperature：per-call 温度覆盖（PRD-C-017 M9）。仅 model 覆盖时生效；None=默认 0.5（旧行为）。
      母题 opus 解题+打标档传低温（0.1~0.2）稳 JSON/解题，B1 母题节点用。
    """
    label = _trace_label(messages)
    tags = None if public_stream else ["skip_stream"]
    # 用户级/会话级归属：从 graph config 取 thread_id + ruoyi_token(→teacher_id)
    conf = (ensure_config() or {}).get("configurable", {}) or {}
    thread_id = conf.get("thread_id")
    teacher_id = conv_trace.teacher_id_from_token(conf.get("ruoyi_token"))
    # 🔴 per-call max_tokens 覆盖（整改4·回炉瘦身用）：None → 默认 VARIANT_MAX_TOKENS（旧行为）。
    max_tokens = max_tokens if max_tokens and max_tokens > 0 else settings.VARIANT_MAX_TOKENS
    t0 = time.monotonic()
    relay = settings.RELAY_NAME
    model_used = settings.COMPATIBLE_MODEL
    fallback = 0
    try:
        # 🔴 走中转站熔断转移池（Block B）：返回实际成交中转站 + 该站 model + 转移次数
        #   （RELAY_POOL 各站可配不同模型，trace/计费必须按成交站归因）
        resp, relay, model_used, fallback = await relay_pool.ainvoke_failover(
            messages, max_tokens=max_tokens, tags=tags, on_delta=on_delta, model=model,
            temperature=temperature,
        )
        text = _content_text(resp).strip()
        retried = False
        if not text and retry:
            retried = True
            resp, relay, model_used, fb2 = await relay_pool.ainvoke_failover(
                messages, max_tokens=max_tokens, tags=tags, on_delta=on_delta, model=model,
                temperature=temperature,
            )
            fallback += fb2
            text = _content_text(resp).strip()
    except Exception as e:  # noqa: BLE001 — 记下失败往返后照常抛
        dur = int((time.monotonic() - t0) * 1000)
        _trace_llm(label, messages, "", None, dur, error=str(e), model=model_used)
        conv_trace.write(
            teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
            model=model_used, relay=relay, fallback_count=fallback,
            request=_serialize_request(messages),
            response="", response_raw=None, duration_ms=dur, error=str(e),
        )
        raise
    dur = int((time.monotonic() - t0) * 1000)
    pt, ct = relay_pool.usage_tokens(resp)
    cost = relay_pool.cost_yuan(model_used, pt, ct)
    # 原始返回：content（思考型可能是 parts list）+ reasoning/usage 等附加信息（best-effort）
    raw: dict[str, Any] = {}
    try:
        raw["content"] = resp.content
        if getattr(resp, "additional_kwargs", None):
            raw["additional_kwargs"] = resp.additional_kwargs
        if getattr(resp, "response_metadata", None):
            raw["response_metadata"] = resp.response_metadata
        if getattr(resp, "usage_metadata", None):
            raw["usage_metadata"] = resp.usage_metadata  # 🔴 token 取数主源
    except Exception:
        raw = {"content": str(getattr(resp, "content", ""))}
    _trace_llm(label, messages, text, raw, dur, retried=retried, model=model_used)
    # 🔴 用户级对话持久化（优化基础数据源）→ 独立解耦库 conv_trace
    conv_trace.write(
        teacher_id=teacher_id, thread_id=thread_id, source="variant", label=label,
        model=model_used, relay=relay, fallback_count=fallback,
        request=_serialize_request(messages),
        response=text, response_raw=raw, duration_ms=dur, retried=retried,
        prompt_tokens=pt, completion_tokens=ct, cost_yuan=cost,
    )
    # 🔴 P13 预算闸记账：一次成功 LLM 往返 = 一票（per-round 计数，超限后增强类调用跳过）。
    # 记在「成功返回」处（失败/空返抛异常的早退路径不记——只数真正花掉的调用）。
    _budget_tick()
    return text


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> Any:
    """剥 markdown fence + JSON 容错（抄 ai-orchestrator llm/client 范式）。"""
    text = (text or "").strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        # 退而求其次：截第一个 { 到最后一个 }
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                pass
    return None


# --- 富文本净化（用户反馈 2026-06-11：解析裸字符不渲染的根因） -----------------
# LLM（gpt-5.4）产 JSON 时两类脏输出：① LaTeX 用 \( \) / \[ \] 定界（前端
# markdown-it-katex 只认 $/$$，且 markdown 会把 \( 的反斜杠当转义吃掉）；② 把换行
# 写成双反斜杠 → 解出字面 \n 两个字符。统一在解析边界净化，入库/快照/气泡三处共净。
_PAREN_MATH_RE = re.compile(r"\\\(\s*(.+?)\s*\\\)", re.DOTALL)
_BRACKET_MATH_RE = re.compile(r"\\\[\s*(.+?)\s*\\\]", re.DOTALL)
# 字面 \n 后跟小写字母 = 可能是 LaTeX 命令（\neq \nabla \newline \nu …），不动；其余视为换行
_LITERAL_NL_RE = re.compile(r"\\n(?![a-z])")


def _sanitize_rich_text(s: Any) -> Any:
    """LLM 产出的 stem/answer/solution 净化：\\(..\\)→$..$、\\[..\\]→$$..$$、字面 \\n→换行。"""
    if not isinstance(s, str) or not s:
        return s
    s = _BRACKET_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", s)
    s = _PAREN_MATH_RE.sub(lambda m: f"${m.group(1)}$", s)
    return _LITERAL_NL_RE.sub("\n", s)


def _sanitize_item(it: dict[str, Any]) -> dict[str, Any]:
    """就地净化一道题的富文本字段，返回原 dict（链式用）。"""
    for k in ("stem", "answer", "solution"):
        it[k] = _sanitize_rich_text(it.get(k))
    return it


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else _content_text(msg)
    return ""


def _latest_ai_text(messages: list[BaseMessage]) -> str:
    """BUG-006：取最近一条 AI 消息文本（注入 parse，让分类器能判老师对 AI 提议的承接）。

    只读最近一条 AI 消息这一项（不建状态机/不建 pending_offer，最小止血）。无 AI 消息 → ""。
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            c = msg.content
            return c if isinstance(c, str) else _content_text(msg)
    return ""


_URL_RE = re.compile(r"https?://[^\s)>'\"]+", re.IGNORECASE)


def _extract_image_url(text: str) -> str | None:
    """从用户消息抽 OSS 题图 URL（MVP 贴 URL，file_uploader future）。"""
    m = _URL_RE.search(text or "")
    return m.group(0) if m else None


def _strip_urls(text: str) -> str:
    """去掉文本里所有 URL（首轮"图 URL + 人话要求"里把人话剥出来给 knobs 抽取）。"""
    return _URL_RE.sub("", text or "").strip()


def _conf_ok(analysis: dict[str, Any]) -> bool:
    """三锚（年级/考点/题型）任一低置信 → 闸不过。"""
    for k in ("grade", "kp", "qtype"):
        node = analysis.get(k) or {}
        if float(node.get("confidence", 0) or 0) < CONF_GATE:
            return False
    return True


# ---------------------------------------------------------------------------
# 两步锚定·年级 → 年级 4 位 code 前缀（PRD-C-014 B1，对齐 biz_subject 叶子 id 前缀）。
# 浙教版叶子 id 前 4 位 = 学段学科册（如 3071=七上、3081=八上）。LLM 给的年级文案
# （"七年级上学期" / "七年级上册" / "7年级上"）归一到此 code，给 leaf_pool_for_grade 圈池。
# 拿不准 → None（圈全量叶子，仍受池内校验，不放空锚定）。
# ---------------------------------------------------------------------------
_GRADE_CN = {"七": "307", "八": "308", "九": "309"}
_TERM_CN = {"上": "1", "下": "2"}

# 🔴 复习册开关关键词（2026-06-13 整改·批1 step5）：老师文本含下列词 = 明确要中考/复习/专题/
#   模考题 → include_review_books=True（复习册进锚定池可锚）。简单关键词判，不另起 LLM。
_REVIEW_INTENT_RE = re.compile(r"中考|复习|专题|模考|一模|二模")


def _wants_review_books(text: str | None) -> bool:
    """老师文本是否明确要复习/模考类题（决定复习册是否并入锚定池）。"""
    return bool(_REVIEW_INTENT_RE.search(str(text or "")))


def _book_name_of(node_id: Any) -> str:
    """叶子/年级 code → 所属册名（前 4 位映射；复用 dna_extract 单一映射，未知前缀 → 空串）。"""
    return dna_extract._book_of(node_id)


def _grade_to_code(grade_value: Any) -> str | None:
    """年级文案 → 4 位 code 前缀（如 七年级上 → 3071）；拿不准返回 None。"""
    s = str(grade_value or "").strip()
    if not s:
        return None
    # 阿拉伯数字归一到中文（7→七）
    s = s.replace("7", "七").replace("8", "八").replace("9", "九")
    g = next((v for k, v in _GRADE_CN.items() if k in s), None)
    if not g:
        return None
    t = next((v for k, v in _TERM_CN.items() if k in s), None)
    return f"{g}{t}" if t else None


# ---------------------------------------------------------------------------
# 思维外放（stage 思路条事件）：langgraph custom 通道 → service stream_mode=custom
# → SSE 帧 {"type":"message","content":{"type":"custom","custom_data":{"stage":{...}}}}
# → FE（book-ui variant 页）按 key 更新/追加紫色思路条。
# 🔴 stage 是增强不是关卡（G5）：runtime 外调用（单测直调节点无 runnable context →
#    get_stream_writer 抛 RuntimeError）/ writer 发送失败，一律静默吞，绝不影响主流程。
# 🔴 必须包成 role="custom" 的 ChatMessage 且 content 是单元素 list ——
#    service utils.langchain_to_chat_message 只认这个形状，裸 dict 会变成 error 帧。
# ---------------------------------------------------------------------------
def _emit_stage(key: str, title: str, status: str, detail: str | None = None) -> None:
    """发思路条 stage 事件（key/title/status/detail 契约与 FE 严格一致）。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    stage: dict[str, Any] = {"key": key, "title": title, "status": status}
    if detail:
        stage["detail"] = detail
    try:
        writer(ChatMessage(content=[{"stage": stage}], role="custom"))
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点
        pass


# ---------------------------------------------------------------------------
# artifact 快照帧（PRD-C-011 Bucket 3）：FE 题卡数据源 = 本帧，不 parse markdown。
# 契约（BE/FE 严格一致）：ChatMessage(role="custom", content=[{"artifact": {
#   "items": [{index/stem/answer/solution/qtype/difficulty/level/verify/gene/persisted}],
#   "header": {recipe/kp/grade}}}])
# 发射点：assemble 收尾（每轮题组变化必过）+ persist_to_bank 成功后（persisted=true 更新）。
# 🔴 独立函数、不复用 _emit_stage（test_variant_stage 对 _emit_stage 调用序列精确断言）；
#    同 _emit_stage 双层静默吞：无 runtime context / writer 抛 → 绝不影响主流程。
# ---------------------------------------------------------------------------
def _norm_secondary_kps(raw: Any) -> list[dict[str, str]]:
    """副 kp 归一成 FE 弹层回填需要的 [{id,name}]（兼容历史形态：裸 code 串 / {code,name} /
    {id,name}）。id/name 缺则给空串、绝不抛。纯函数、零 IO。"""
    out: list[dict[str, str]] = []
    for s in raw or []:
        if isinstance(s, dict):
            sid = str(s.get("id") or s.get("code") or "").strip()
            name = str(s.get("name") or "").strip()
        else:
            sid = str(s or "").strip()
            name = ""
        out.append({"id": sid, "name": name})
    return out


def _item_dna(it: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """🔴 PRD-C-014 B4·FE DNA 面板数据源：每个 artifact item 的嵌套 `dna` 对象（键名钉死，
    FE pickDna 按此解析）。组级维度（main_kp/secondary_kps/exam_type/tags/scene/skeleton）来自
    母题 DNA（facts.dna，整组共享）；item 级（hard_points/manual_edited）优先取 item。

    DNA 未抽取（库内母题直进 generate 路径，facts.dna 为空）→ 各键给空值/空数组，绝不崩。
    """
    dna = facts.get("dna") or {}
    main_kp = dna.get("main_kp") or {}
    if not isinstance(main_kp, dict):
        main_kp = {}
    # skeleton：item 级覆盖（revise target=skeleton 落 it["skeleton"]）优先，缺则回退母题
    # dna.skeleton（与 hard_points 同模式·G12a）。母题骨架是守恒基因，老师改单题骨架只覆盖本题。
    sk_item = it.get("skeleton")
    if sk_item is not None and str(sk_item).strip():
        skeleton = str(sk_item)
    else:
        # 母题 DNA 里 skeleton 是 list[str]（dna_extract 抽的步骤序列）→ FE 要 str，按换行拼。
        skeleton_raw = dna.get("skeleton")
        if isinstance(skeleton_raw, list):
            skeleton = "\n".join(str(s) for s in skeleton_raw if str(s).strip())
        else:
            skeleton = str(skeleton_raw or "")
    # hard_points：item 级有则用 item 的，否则母题 DNA 的。
    hp_raw = it.get("hard_points")
    if hp_raw is None:
        hp_raw = dna.get("hard_points")
    hard_points = [str(h) for h in (hp_raw or []) if str(h).strip()]
    # 🔴 PRD-C-015 批2·models 维（双轴「怎么解」轴）：母题级，整组共享 facts.dna.models。
    #   item 级可被 edit-dna 覆盖（it["models"]，批4 接重写解析）；缺则回退母题 DNA。非空（M00 兜底）。
    models_raw = it.get("models") if it.get("models") is not None else dna.get("models")
    models = [
        {"id": str(m.get("id") or ""), "name": str(m.get("name") or "")}
        for m in (models_raw or [])
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    ]
    return {
        "main_kp": str(main_kp.get("name") or "") or None,
        "main_kp_id": str(main_kp.get("id") or "") or None,
        "secondary_kps": _norm_secondary_kps(dna.get("secondary_kps")),
        "exam_type": str(dna.get("exam_type") or "") or None,
        "tags": [str(t) for t in (dna.get("tags") or []) if str(t).strip()],
        "scene": str(dna.get("scene") or "") or None,
        "skeleton": skeleton or None,
        "hard_points": hard_points,
        # 双轴模型维（批2）：models 非空（M00 兜底）；model_overflow/model_warn 给 FE 标 ⚠（批5 渲染）。
        "models": models,
        "model_overflow": [str(x) for x in (dna.get("model_overflow") or []) if str(x).strip()],
        "model_warn": bool(dna.get("model_warn")),
        "manual_edited": bool(it.get("manual_edited")),
    }


def _artifact_payload(
    state: VariantState, persisted_flags: list[bool] | None = None
) -> dict[str, Any]:
    """纯组帧（零 IO 可单测）：state.items/check/gene/knobs/analysis → artifact 契约 dict。"""
    items = state.get("items") or []
    facts = _mother_facts(state)
    out_items: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        chk = it.get("check") or {}
        # 🔴 P2b 稳定 seq（PRD-C-013）：item 自带 `_seq`（generate 流内 eager 落的「题原始
        #   生成序」，整生命周期不变、剔除题不重压缩）优先；缺省（assemble/入库/会话恢复等
        #   定稿全量帧）回退 index=i+1（一期等价，单键不分叉）。FE pickArtifact 按 seq 原位 merge。
        seq = _to_int(it.get("_seq")) or (i + 1)
        cell: dict[str, Any] = {
            "index": i + 1,
            "seq": seq,
            "stem": str(it.get("stem") or ""),
            "answer": str(it.get("answer") or ""),
            "solution": str(it.get("solution") or ""),
            "qtype": str(it.get("qtype") or ""),
            "difficulty": _to_int(it.get("difficulty")) or 0,
            "level": str(it.get("level") or "normal"),
            # 🔴 verify 与 review 互斥不同键：证明类只有 review（proof_needs_human）
            "verify": chk.get("verify") or chk.get("review") or None,
            # 4d 外显层级（FE 徽章唯一依据；旧线程恢复无 tier → FE 按「只说好」兜底）
            "tier": chk.get("tier") or None,
            "gene": (it.get("gene") or {}).get("gate") or None,
            # persisted：flags 优先（persist 节点按回执现算）；否则读 item 簿记
            # （persist_to_bank 成功后回写 state.items[i].persisted → 后续编辑轮
            #  assemble 重发快照时「已收录」徽章不回退，G5 二次入库不重复落行）
            "persisted": (
                bool(persisted_flags[i])
                if persisted_flags is not None and i < len(persisted_flags)
                else bool(it.get("persisted"))
            ),
            # 🔴 PRD-C-014 B4·FE DNA 面板数据源（键名钉死，FE pickDna 解析）：组级维度共享 +
            #   item 级（hard_points/manual_edited）覆盖。DNA 未抽取 → 空值/空数组兜底，不崩。
            "dna": _item_dna(it, facts),
        }
        # 🔴 P2b 退场哨兵（PRD-C-013）：剔除题显式带 `_dropped: true`（字段白名单透传），
        #   驱动 FE upsertIncremental 走退场过渡（is-dropping/scheduleDropRemoval）后从
        #   mergedItems 移除——不再靠「压缩 index 隐式挤掉」，避免后题 index 前移嫁接错卡。
        if it.get("_dropped"):
            cell["_dropped"] = True
        # 🔴 PRD-C-015 批4·DNA 改→重生态透传给 FE（批5 渲染角标/重生·撤销按钮/dirty 入库拦截）：
        #   dna_dirty=本题待重生；dirty_dims=哪些维脏（驱动逐维角标）；can_undo=有重生快照可撤销。
        cell["dna_dirty"] = bool(it.get("dna_dirty"))
        cell["dirty_dims"] = list(it.get("dirty_dims") or [])
        cell["mother_dirty_dims"] = list(it.get("mother_dirty_dims") or [])
        cell["can_undo_regen"] = isinstance(it.get("regen_snapshot"), dict)
        out_items.append(cell)
    # 🔴 批4·组级重生态：mother_dirty（母题守恒维改）+ regen_pending（待重生集合 1-based 题号）。
    #   FE 据 regen_pending 非空 → 「重生」按钮可点 + 入库按钮禁用（致命① dirty 拒入库视觉）。
    mother_dirty = bool((state.get("mother_dna") or {}).get("dirty"))
    regen_pending = dirty_item_indexes(items)
    # 🔴 批5·合并确认面（G10/G11）数据源：透传 mother_confirm（flags/needs_confirm/
    #   confirmed_dims/audit_ref），FE pickMotherConfirm 解析弹合并确认面。缺省 → None
    #   （旧 FE 不读不坏，向后兼容）。
    mother_confirm = state.get("mother_confirm") or None
    return {
        "items": out_items,
        "header": {
            "recipe": knobs_desc(state.get("knobs")) or None,
            "kp": facts["kp_name"] if facts["kp_name"] != "未知考点" else None,
            "grade": facts["grade"] if facts["grade"] != "未知年级" else None,
            # 批4·母题脏 + 待重生集合（FE 批5 用；旧 FE 不读 header 这俩键也不坏）
            "mother_dirty": mother_dirty,
            "regen_pending": regen_pending,
            # 批5·合并确认面（G10/G11）：母题确认状态（needs_confirm 时 FE 弹面）
            "mother_confirm": mother_confirm,
        },
    }


def _emit_artifact(
    state: VariantState,
    persisted_flags: list[bool] | None = None,
    *,
    partial: bool = False,
    expected_total: int | None = None,
) -> None:
    """发 artifact 快照帧（FE 题卡数据源）。任何异常静默吞，artifact 是增强不是关卡。

    🔴 P2 增量帧（PRD-C-012）：partial=True 时帧带 {"partial": true, "expected_total": N}
    （每题闸链完成发一帧，items=按生成序已完成的题）；assemble 定稿帧不带 partial 键
    （FE 老逻辑只认最后一帧也不坏，向后兼容）。
    """
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    try:
        payload = _artifact_payload(state, persisted_flags)
        if partial:
            payload["partial"] = True
            if expected_total is not None:
                payload["expected_total"] = int(expected_total)
        writer(ChatMessage(content=[{"artifact": payload}], role="custom"))
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点
        pass


# ---------------------------------------------------------------------------
# Router（入口分诊：登录? 有图? 在途母题? 库内母题跳 analyze/classify）
# ---------------------------------------------------------------------------
def route_entry(
    state: VariantState, config: RunnableConfig
) -> Literal["analyze", "parse", "generate", "ask", "auth"]:
    # 🔴 身份硬闸（用户拍板 2026-06-11）：每次对话绑死登录老师。token 缺失/解不出 userId
    # → 一步不走（不进任何 LLM 节点，conv_trace 也不会产生无主行；表级 NOT NULL 双保险）。
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    if conv_trace.teacher_id_from_token(token) is None:
        return "auth"
    url = _extract_image_url(_latest_human_text(state.get("messages", [])))
    # 跨轮新图 = 视作新母题（设计 §6：重走 analyze，覆盖在途状态）
    if url:
        return "analyze"
    # 库内母题（已确认 DNA）、还没出题 → 直接造（跳 analyze/classify）
    if state.get("mother_confirmed") and state.get("mother_dna") and not state.get("items"):
        return "generate"
    # 老会话·纯文字：已出题组 或 🔴 在途母题（已分析停在 clarify 等老师答年级/考点）
    # → parse 分诊。修 17 号多轮路由漏洞：旧版要求有 items 才进 parse，把「clarify 的
    # 回答」漏成催图（root cause 见 claude-code-sign/17-route_entry-多轮路由漏洞-修复任务.md §2）。
    if state.get("items") or state.get("mother_dna"):
        return "parse"
    # 没图、无在途母题、无题组 → 催图（设计 §6 输入边界兜底）
    return "ask"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = """你是浙教版初中数学命题专家。看这张题目图，**流式**输出母题分析。
🔴 老师贴图时可能附带了出题要求（出几道、难度安排、题型配比等），如下：
{utterance}
（仅作语境参考——出题配方由独立纯文本抽取器另行解析，这里**不要**抽 knobs。）

只输出一个 JSON（不要解释），结构：
{{
  "is_question_image": true/false,   // 非题目图(风景/截图/空白)填 false
  "images_count": 1,                  // 检测到几张图
  "questions_in_image": 1,            // 这张图里有几道题
  "grade": {{"value": "七年级上学期", "confidence": 0.0~1.0}},
  "subject": "数学",
  "kp": {{"value": "核心考点粗描述(如:一元二次方程求根)", "confidence": 0.0~1.0}},
  "qtype": {{"value": "选择/填空/解答", "confidence": 0.0~1.0}},
  "stem": "题干(Markdown+LaTeX)",
  "answer": "标准答案(图里没有就先解母题得出)",
  "difficulty": 1~5,
  "structure": "结构指纹/解法骨架简述",
  "solution_skeleton": "解法骨架(步骤)"
}}
🔴 无答案先解母题得答案/解法骨架(作 verify 基准)。各锚(年级/考点/题型)如实给 confidence。
🔴 BUG-002 D1：出题配方（数量/难度/题型配比）由独立 _extract_knobs 纯文本抽取，本调用**不输出 knobs**。"""


# 🔴 阶段灯·锚定考点首灯定时翻绿（PRD-C-009 整改·改动2）：合并读图+配方+锚定后，
#   「锚定考点」这盏首灯背后是一次大调用（multimodal 读图 ~63s），没法真实分段。维护者拍板
#   折中——首灯在 min(STAGE1_TIMER_S, 大调用真实完成) 时翻绿（cosmetic，用户已认可）。
#   实现 = 一个 asyncio 定时任务与大调用 await 竞速：谁先到谁发 done，另一方被取消/跳过。
#   定时任务在节点 active context 内 spawn（get_stream_writer contextvar 随 create_task 复制
#   传播），且节点返回前必 join/cancel —— 不留游离 task。
STAGE1_TIMER_S = 10.0


async def _stage1_timer_done(emitted: dict) -> None:
    """首灯（锚定考点）定时翻绿：睡 STAGE1_TIMER_S 后若大调用还没发过 done → 由定时发 done。
    emitted 是与主协程共享的哨兵 {"done": bool}，保证「定时 / 真实完成」只翻绿一次（去重）。"""
    try:
        await asyncio.sleep(STAGE1_TIMER_S)
    except asyncio.CancelledError:  # 大调用先完成 → 取消定时（真实完成路径）
        return
    if not emitted.get("done"):
        emitted["done"] = True
        _emit_stage("classify", "锚定考点", "running", "读图中…")
        _emit_stage("classify", "锚定考点", "done")


async def analyze(state: VariantState, config: RunnableConfig) -> VariantState:
    """① 分析：multimodal 读图 → 年级/学科/粗考点/题型 + 题干/答案/难度/结构 + 几图几题 + 各锚置信
    + 🔴 出题配方旋钮（改动1：knobs 并入读图同一次调用，省掉独立 _extract_knobs 串行往返）。

    🔴 阶段灯（改动2）：首灯「锚定考点」在 min(10s 定时, 大调用真实完成) 翻绿（cosmetic）；
       次灯「解析配方」由 classify 在锚定真实完成后发 done（带道数）。读图/配方/锚定共一灯首段。"""
    # 🔴 本轮消息里的 URL 优先（跨轮新图 = 新母题，必须分析新图）；无则沿用在途母题图。
    #   旧序（state 优先）会让同 thread 第二张图被静默忽略、永远重分析第一张。
    url = _extract_image_url(_latest_human_text(state.get("messages", []))) or state.get(
        "image_url"
    )
    if not url:
        return {
            "messages": [AIMessage(content="请先贴一张题目图的 OSS URL，我才能开始举一反三。")]
        }

    # 🔴 首灯（锚定考点）running + 次灯（解析配方）running：两灯先点亮，done 各自在后。
    _emit_stage("classify", "锚定考点", "running", "读图中…")
    _emit_stage("knobs", "解析配方", "running")
    # 🔴 配方旋钮并入读图：把老师附带的人话填进 ANALYZE_PROMPT 的 {utterance} 段，一次调用同出
    #   analysis + knobs（省一次 ~55s 纯 reasoning 的独立 _extract_knobs 往返）。
    user_text = _strip_urls(_latest_human_text(state.get("messages", [])))
    msg = HumanMessage(
        content=[
            {"type": "text", "text": ANALYZE_PROMPT.format(utterance=user_text or "（无）")},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )
    # 首灯定时翻绿 vs 大调用真实完成竞速（共享 emitted 哨兵去重，只翻绿一次）
    emitted = {"done": False}
    timer = asyncio.create_task(_stage1_timer_done(emitted))
    try:
        text = await _ainvoke_text([msg], model=settings.variant_model("analyze"))
    finally:
        timer.cancel()  # 大调用结束 → 取消定时（无论成功/异常都不留游离 task）
        try:
            await timer  # join 已取消的 task，吞 CancelledError，免「Task was destroyed」告警
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if not emitted.get("done"):  # 大调用先于 10s 完成 → 首灯由真实完成翻绿（min 语义）
        emitted["done"] = True
        _emit_stage("classify", "锚定考点", "done")
    # 🔴 防御（2026-06-13 冒烟实测）：思考型/nano 偶发返回 JSON 数组而非对象 → _parse_json
    #   出 list，旧 `or {}` 不挡（非空 list 为真）→ data.get 抛 AttributeError 炸整轮。非 dict
    #   一律降级空 dict 走「未识别」兜底（G5 不卡死），不让一次模型抽风炸掉 analyze 节点。
    data = _parse_json(text)
    if not isinstance(data, dict):
        data = {}

    if data.get("is_question_image") is False:
        _emit_stage("knobs", "解析配方", "warn", "未识别为题目图")
        return {
            "image_url": url,
            "messages": [
                AIMessage(content="这张图我没认出是题目（可能是风景/截图/空白）。请换一张清晰的题目图。")
            ],
        }

    analysis = {
        "grade": data.get("grade") or {"value": None, "confidence": 0},
        "subject": data.get("subject") or "数学",
        "kp": data.get("kp") or {"value": None, "confidence": 0},
        "qtype": data.get("qtype") or {"value": None, "confidence": 0},
    }
    mother_dna = {
        "stem": data.get("stem"),
        "answer": data.get("answer"),
        "difficulty": data.get("difficulty"),
        "structure": data.get("structure"),
        "solution_skeleton": data.get("solution_skeleton"),
    }
    # 🔴 旋钮跨母题防泄漏 + clarify 迂回防丢：
    #   - 本轮人话抽出新配方 → 覆盖旧配方（新母题新要求，跨轮第二张图的文字不再被丢）；
    #   - 没抽出新配方：同图重贴（clarify 迂回/澄清应答）→ 保留首轮已抽配方；
    #     新图 → 重置 {}（旧母题的「5道递增」绝不错套到新母题上）。
    # shape_defects 一并清零（上一母题的缺陷外显不带进新母题轮）。
    # 🔴 BUG-002 D1（2026-06-13）：utterance 非空 → **强制走独立纯文本 _extract_knobs 抽取**，
    #   不再采信读图 multimodal 兼任产出的 knobs 段。根因：中文数量词（两/俩/一对）在读图主任务
    #   下召回不稳（"两道仍出 3 道"），纯文本 light 单跑对数量词稳。代价 = utterance 非空多一次
    #   nano 往返（D4 已接受）；utterance 空（纯贴图）→ 首轮不新增任何调用（AC6/G5 守）。
    if not user_text:
        new_knobs = {}  # 纯贴图：首轮不抽（不新增 LLM 调用）
    else:
        new_knobs = await _extract_knobs(state)  # utterance 非空 → 独立纯文本抽取（数量词稳）
    if not new_knobs and url == state.get("image_url") and state.get("knobs") is not None:
        knobs = state.get("knobs")  # 同图重贴且本轮无新配方 → 保留
    else:
        knobs = new_knobs
    return {
        "image_url": url,
        "images_count": int(data.get("images_count") or 1),
        "questions_in_image": int(data.get("questions_in_image") or 1),
        "analysis": analysis,
        "mother_dna": mother_dna,
        "knobs": knobs,
        "shape_defects": [],
        "messages": [],
    }


async def _resolve_grade_code(analysis: dict[str, Any]) -> str | None:
    """两步锚定第一步·定年级 4 位 code：① LLM 年级文案归一（_grade_to_code）优先；
    ② 落空时退回 anchor_subject 按粗考点名反推 grade_code（编码前4位，比 LLM 裸猜准）。
    都拿不准 → None（leaf_pool_for_grade 圈全量叶子，仍受池内校验，不放空锚定）。"""
    gc = _grade_to_code((analysis.get("grade") or {}).get("value"))
    if gc:
        return gc
    coarse = (analysis.get("kp") or {}).get("value")
    if coarse:
        try:
            # 🔴 2026-06-13 整改：反查年级排除复习册候选——同名考点双挂复习册会把 grade_code
            #   反推成 3010/3100/3120（复习册前缀，非教材册年级），出题年级就跑偏成「未知年级」。
            cands = await asyncio.to_thread(
                anchor_subject, coarse, exclude_review_books=True
            )
            if cands and cands[0].get("grade_code"):
                return str(cands[0]["grade_code"])
        except Exception:  # noqa: BLE001 — 名匹配降级失败 → None（全量池兜底）
            pass
    return None


async def classify(state: VariantState, config: RunnableConfig) -> VariantState:
    """② 分类·两步锚定 + DNA 抽取（PRD-C-014 B1）：
      年级 → 年级叶子池 → dna_extract 让 LLM **池内选 id**（禁造词）。
    🔴 池内锚到主 kp → anchored.code = 真知识点 code（dim1KpId 主 kp）+ 抬置信；
      池内无匹配（main_kp=None）→ anchored 缺失 → 低置信 → gate_after_classify 走 clarify
      （根治 C-013 凭 LLM 置信裸放行落 0）；库/网络故障锚定不可用 → 同样 clarify，不 silent-fail。
    🔴 抽出的全维 DNA 存进 mother_dna.dna，由 _mother_facts 穿进 BO（T3）。
    """
    analysis = dict(state.get("analysis") or {})
    kp_node = dict(analysis.get("kp") or {})
    dna_obj = state.get("mother_dna") or {}
    mother_dna = dict(dna_obj)

    # 🔴 复习册开关（批1 step5）：老师本轮文本明确要中考/复习/专题/模考 → 复习册并入锚定池。
    #   否则锚定池只圈教材册（防锚到「新题抢先」等复习册同名节点照样出题的实锚事故）。
    include_review_books = _wants_review_books(
        _latest_human_text(state.get("messages", []))
    )

    # --- 两步锚定第一步：定年级 code ---
    grade_code = await _resolve_grade_code(analysis)

    # --- 第二步：拉年级叶子池（HTTP，故障 → 空池降级走 clarify） ---
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    leaf_pool: list[tuple[str, str]] = []
    client = RuoyiClient(token=token)
    try:
        leaf_pool = await leaf_pool_for_grade(
            grade_code, client, include_review_books=include_review_books
        )
    except Exception as e:  # noqa: BLE001 — 池拉取故障 → 空池（不 silent-fail，转 clarify）
        analysis["_anchor_error"] = str(e)

    # 🔴 首灯「锚定考点」running 已由 analyze 点亮（且可能定时翻过绿）→ 此处不重发 running
    #   防绿→running 回闪；只在锚定真实有结论时发终态 done/warn。

    if not leaf_pool:
        # 库/网络故障或年级池为空 → 锚定不可用 → 不抬置信 → clarify（既有降级路径）
        await client.aclose()
        analysis.setdefault("_anchor_error", "知识点叶子池不可用（库未起/年级未识别）")
        _emit_stage("classify", "锚定考点", "warn", "知识点池不可用，待老师确认")
        # 次灯「解析配方」：锚定不可用也要给个终态，免得灯永远停在 running
        _emit_stage("knobs", "解析配方", "warn", "待老师确认母题后再定配方")
        # 锚定不可用 = 三锚没定死 → mother_confirm.needs_confirm 必 true（与正常路径同口径）。
        early: VariantState = {"analysis": analysis, "mother_confirmed": False, "messages": []}
        early["mother_confirm"] = build_mother_confirm({**state, **early})
        return early

    # --- DNA 抽取（两步锚定第二步：LLM 池内选 id；禁造词由 dna_extract 校验闸把关） ---
    #     首锚主 kp 未知 → tag_pool 空（标签先靠 LLM 自拟），下面 T2 锚定后再注池重选。
    dna = await dna_extract.extract_dna(
        stem=mother_dna.get("stem") or "",
        answer=mother_dna.get("answer") or "",
        analyze=mother_dna.get("solution_skeleton") or "",
        grade=(analysis.get("grade") or {}).get("value") or "",
        leaf_pool=leaf_pool,
        tag_pool=[],
        invoke=_ainvoke_text,  # 🔴 Q1：经 _ainvoke_text → 锚定/DNA 抽取这一步落 trace（label=dna_extract）
        model=settings.variant_model("dna"),  # 按环节分档（默认 nano；缺省回退 dna_extract 自身的 LLM_MODEL_LIGHT）
        include_review_books=include_review_books,  # 批1 step5：复习册闸（二道保险，正常池已剔）
    )

    main_kp = dna.get("main_kp")

    # --- 🔴 T2 标签复用池接线：锚到主 kp 后，按 kp 拉高频标签池，做一次窄调用重选 tags 维 ---
    #     （+1 LLM 调用，仅成功锚定 + 池非空时；拉池失败/池空 → 0 额外调用、保留原 tags 降级）。
    if main_kp and main_kp.get("id"):
        try:
            kp_tag_pool = await tag_pool_for_kp(main_kp["id"], client)
        except Exception as e:  # noqa: BLE001 — 拉池故障 → 空池降级（不卡死出题）
            kp_tag_pool = []
            analysis.setdefault("_tag_pool_error", str(e))
        if kp_tag_pool:
            dna = await dna_extract.refine_tags_with_pool(
                dna, stem=mother_dna.get("stem") or "", tag_pool=kp_tag_pool,
                invoke=_ainvoke_text,  # 🔴 Q1：标签复用窄调同样落 trace（label=dna_tags）
                model=settings.variant_model("dna"),  # 按环节分档（默认 nano）
            )

    await client.aclose()

    # --- 🔴 PRD-C-015 批2·W1' 模型锚定（双轴「怎么解」轴）：母题 DNA 抽完即锚 models ---
    #   按主+副 kp 反查候选（纯只读 ETL，≤8 按 sort）→ gpt-5.4-mini「先解题再选」确认 ≤3（池内选/禁造词）；
    #   无命中→M00 保底（模型维永不为空）；池外名→落待命名池+⚠（不入正式维）；反查库故障→M00+⚠。
    #   写进 mother_dna.dna.models / model_overflow（契约 v2，批1 已留字段位）。
    try:
        m_ref = str((main_kp or {}).get("id") or "") or None
        m_res = await model_anchor.anchor_models(
            dna,
            stem=mother_dna.get("stem") or "",
            answer=mother_dna.get("answer") or mother_dna.get("solution_skeleton") or "",
            invoke=_ainvoke_text,
            model=settings.variant_model("model_confirm"),  # H2：确认档走 gpt-5.4-mini（.env VARIANT_MODEL_MODEL_CONFIRM）
            record_overflow=lambda name, mm: model_anchor.record_overflow_candidate(
                name, mm, question_ref=m_ref
            ),
        )
    except Exception as e:  # noqa: BLE001 — 锚定整体故障也得有 models（M00 兜底，绝不空维/卡死）
        analysis.setdefault("_model_anchor_error", str(e))
        m_res = {"models": [dict(model_anchor.M00)], "model_overflow": [], "model_warn": True,
                 "model_flag": "lookup_unavailable"}
    dna["models"] = m_res.get("models") or [dict(model_anchor.M00)]
    dna["model_overflow"] = m_res.get("model_overflow") or []
    if m_res.get("model_warn"):
        dna["model_warn"] = True
    mother_dna["dna"] = dna

    if main_kp and main_kp.get("id"):
        # 池内锚到真知识点 → anchored.code = 知识点叶子 code（dim1KpId 主 kp）
        kp_node["anchored"] = {
            "id": main_kp["id"],
            "code": str(main_kp["id"]),  # 叶子 id 即层级编码（biz_subject 无独立 code 列）
            "name": main_kp.get("name"),
        }
        if main_kp.get("name"):
            kp_node["value"] = main_kp["name"]
        kp_node["confidence"] = max(float(kp_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["kp"] = kp_node
        # 年级锚定：抬置信（已圈到该年级池且选中其内 id）
        grade_node = dict(analysis.get("grade") or {})
        if grade_code:
            grade_node["code"] = grade_code
        grade_node["confidence"] = max(
            float(grade_node.get("confidence", 0) or 0), CONF_GATE
        )
        analysis["grade"] = grade_node
        # 题型：DNA 抽到的池外校验后题型，回填抬置信
        if dna.get("qtype"):
            qn = dict(analysis.get("qtype") or {})
            qn["value"] = dna["qtype"]
            qn["confidence"] = max(float(qn.get("confidence", 0) or 0), CONF_GATE)
            analysis["qtype"] = qn
    # else: 池内无匹配（main_kp=None）→ anchored 缺失 → 不抬置信 → clarify（不放行出题）

    confirmed = _conf_ok(analysis) and bool(kp_node.get("anchored"))
    kp_name = (analysis.get("kp") or {}).get("value") or "?"
    grade_name = (analysis.get("grade") or {}).get("value") or grade_code or "?"
    _emit_stage(
        "classify",
        "锚定考点",
        "done" if confirmed else "warn",
        f"考点「{kp_name}」·年级「{grade_name}」",
    )
    # 🔴 次灯「解析配方」（改动2）：合并调用 + 锚定真实完成后发 done，带道数（knobs 已由
    #   analyze 抽好随 state 来）。confirmed 才落定配方进 generate；未 confirmed 走 clarify，
    #   配方虽已抽出但本轮不出题 → 给 warn 终态（不停在 running）。
    knobs = state.get("knobs")
    recipe = knobs_desc(knobs) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）"
    _emit_stage(
        "knobs", "解析配方", "done" if confirmed else "warn", recipe
    )
    # 🔴 批3·事实源冻结：classify 是锚定（LLM 来源）写入点，跑完即定死 → 此处置 facts_locked。
    #   confirmed（定死）→ 冻结（此后 LLM 回写被 setter 忽略）；未 confirmed → 不冻结（等老师
    #   补/纠正后重锚，重锚仍是合法锚定路径，不该被冻结挡住）。
    out: VariantState = {
        "analysis": analysis,
        "mother_dna": mother_dna,
        "mother_confirmed": bool(confirmed),
        "facts_locked": bool(confirmed),
        "messages": [],
    }
    # 🔴 PRD-C-015 批1·classify 注入点（缺口5 合并确认闸 + D-merge7 确定性异常门控）：
    #   抽完母题完整 DNA 后，把「年级+主考点三锚置信门控（C-014）」与「守恒维确定性异常」合并算成
    #   mother_confirm（needs_confirm = 有异常 ∨ 三锚没定死）。这是 §3.5 GateCheck→MergedConfirm/
    #   EmitVariants 的算据；批4/5 接 UI 弹合并确认面 / 直接放行。批1 先把状态算齐写进契约 v2。
    out["mother_confirm"] = build_mother_confirm({**state, **out})
    return out


# ---------------------------------------------------------------------------
# 🔴 「定死」硬闸（批2·2026-06-13）：缺锚必停确认，从机制上绝迹「未解析+未知年级进出题」。
# 「定死」三件同时满足：
#   ① 年级学期归一出 4 位教材册 code（grade.code 命中，非复习册前缀）；
#   ② main_kp 锚到该册真叶子（kp.anchored.code 有值，in pool 且非复习册——classify 已校验）；
#   ③ 置信达既有 CONF_GATE（_conf_ok 三锚通过）。
# 任一缺 → 没定死 → 一律停 clarify/确认态（不进 generate）。
# ---------------------------------------------------------------------------
def _pin_status(state: VariantState) -> dict[str, Any]:
    """盘点母题是否「定死」。返回 {pinned, grade_code, grade_text, kp_code, kp_name, kp_book,
    reasons:[缺项]}（reasons 非空 = 没定死，每项 ∈ {grade, kp, confidence}）。"""
    analysis = state.get("analysis") or {}
    g = analysis.get("grade") or {}
    k = analysis.get("kp") or {}
    anchored = (k.get("anchored") or {}) if isinstance(k, dict) else {}
    grade_code = str(g.get("code") or "").strip()
    kp_code = str(anchored.get("code") or "").strip()

    reasons: list[str] = []
    # ① 年级 4 位教材册 code（复习册前缀不算定死的教材年级）
    if not grade_code or _is_review_book(grade_code):
        reasons.append("grade")
    # ② main_kp 锚到真叶子（非复习册）
    if not kp_code or _is_review_book(kp_code):
        reasons.append("kp")
    # ③ 三锚置信
    if not _conf_ok(analysis):
        reasons.append("confidence")

    return {
        "pinned": not reasons,
        "grade_code": grade_code or None,
        "grade_text": g.get("value"),
        "kp_code": kp_code or None,
        "kp_name": anchored.get("name") or k.get("value"),
        "kp_book": _book_name_of(kp_code),
        "reasons": reasons,
    }


def gate_after_classify(state: VariantState) -> Literal["generate", "clarify"]:
    """🔴 定死闸（批2）：定死（年级册 code + main_kp 锚真叶子 + 置信达标）→ 直通 generate；
    没定死（缺任一）→ 一律停 clarify 确认态（不进 generate），从机制上绝迹缺锚出题。

    mother_confirmed 由 classify 按同口径（_conf_ok + anchored）置位，这里用 _pin_status
    再加「年级册 code + 非复习册」收口（mother_confirmed=True 但年级 code 缺/是复习册的边角
    路径也会被本闸拦住，不直通）。"""
    if state.get("mother_confirmed") and _pin_status(state)["pinned"]:
        return "generate"
    return "clarify"


async def clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """没定死 → 回问老师确认（只问不造，进 WAIT 等下一句）。

    🔴 批2：确认态如实回报「年级学期（推断值或未识别）+ 主考点（锚值+所属册 或 未锚定）」，
    复用既有 clarify 聊天气泡协议（book-ui 既有确认/chip 修改能力，不动 book-ui）。
    """
    analysis = state.get("analysis") or {}
    pin = _pin_status(state)
    g = analysis.get("grade") or {}
    k = analysis.get("kp") or {}
    q = analysis.get("qtype") or {}

    # 状态回报：年级学期 + 主考点（锚值+册 或 未锚定）—— 让老师一眼看到缺哪一项
    grade_line = (
        f"年级学期：**{pin['grade_text']}**（已识别）"
        if "grade" not in pin["reasons"] and pin["grade_text"]
        else f"年级学期：**未识别**（看着像「{g.get('value') or '?'}」，请确认是几年级上/下学期）"
    )
    if "kp" not in pin["reasons"] and pin["kp_name"]:
        book = f"·{pin['kp_book']}" if pin["kp_book"] else ""
        kp_line = f"主考点：**{pin['kp_name']}**（已锚定{book}）"
    else:
        kp_line = f"主考点：**未锚定**（粗看是「{k.get('value') or '?'}」，请确认或指正考点）"

    asks: list[str] = []
    if "grade" in pin["reasons"]:
        asks.append(f"年级我没定死（看着像「{g.get('value') or '?'}」），请告诉我是几年级上/下学期？")
    if "kp" in pin["reasons"]:
        asks.append(f"核心考点我没锚准（粗看是「{k.get('value') or '?'}」），对吗？或请指正。")
    if (
        "confidence" in pin["reasons"]
        and float(q.get("confidence", 0) or 0) < CONF_GATE
    ):
        asks.append(f"题型我读着像「{q.get('value') or '?'}」，对吗？")
    if not asks:
        asks.append("我对母题 DNA 还不够确定，请确认下年级/考点/题型再继续。")

    body = (
        "我得先把母题**定死**才能造变式（年级 + 主考点缺一不可）。当前状态：\n\n"
        f"- {grade_line}\n- {kp_line}\n\n"
        "请补充/纠正：\n" + "\n".join(f"- {a}" for a in asks)
    )
    return {"messages": [AIMessage(content=body)]}


# ---------------------------------------------------------------------------
# 🔴 验算载荷契约（PRD-C-012 4a 单一事实源）：GENERATE / REGEN 出题自带 verify_payload
# 与 EXTRACT 事后抽取共用同一段契约文本（提成常量防两份漂移）。
# 注意：本常量以「format 模板片段」形态存在（花括号已双写转义），只能拼进会被
# .format() 的 prompt 模板里使用，不要单独 .format() 它。
# 排版（吃 aigeek 前缀缓存）：契约属固定段，各 prompt 把它排在变动段（题干/facts）之前。
# ---------------------------------------------------------------------------
_PAYLOAD_CONTRACT = """载荷契约（kind 多选一；所有表达式必须是 sympy 可解析的纯 ASCII 数学串：乘号写 *、乘方写 ** 或 ^、分数写 /、根号写 sqrt()；禁止 LaTeX、中文、单位、等号外的标点）。
🔴 表达式硬边界（违反 = 程序直接拒收变 degrade，浪费一次验算机会）：
- 只允许这些函数：sqrt / Abs / Min / Max（比大小、最值题用 Min(...)/Max(...)，绝对值用 Abs()）。
- **禁止任何 Python 语法**：不许 if/else 三元、列表推导、len()/range()、布尔 True/False、比较式（< > ==）。
- 选项/claimed 必须是**可算出的数值或表达式**（如 "-sqrt(25)"、"7/2"），不是真假陈述；某选项本身不是数值（如文字判断、区间）→ 整题输出 kind:none。
- 计数类问题（"有几个是…"）若每项判定都是数值比较可写成 Min/Max/Abs 组合才抽，否则 kind:none——不要发明 Piecewise/Eq/逻辑与。
1. 方程求解: {{"kind":"equation_solve","equations":["x**2-5*x+6=0"],"unknowns":["x"],"claimed":["2","3"]}}
   （claimed = 标准答案申报的全部解；只支持单未知数。🔴 程序按"解集完全相等"判：若题目含舍根/取值范围
   约束——如分式方程验增根后舍去、几何/应用题边长必须为正——标答只保留部分解时，**不要用本 kind**
   （claimed 会少于裸方程全部解被误判 fail）：能数值验保留解就改用 kind 3 numeric，否则输出 kind:none）
2. 表达式等价(化简/展开/因式分解): {{"kind":"expr_equiv","expr_a":"题面原式","expr_b":"标准答案给的结果式"}}
3. 数值计算: {{"kind":"numeric","expr":"3*7/2","claimed":"10.5","tol":1e-6}}
4. 选择题: {{"kind":"choice","ground":{{上述1/2/3任一子载荷}},"options":{{"A":"2","B":"3"}},"claimed_correct":"B"}}
   （options 的值 = 各选项的数学值；ground = 由题干建立的真值载荷；claimed_correct = 标准答案的选项字母）
5. 不等式解集: {{"kind":"inequality_solve","inequality":"2*x-3>1","unknown":["x"],"claimed":"x>2"}}
   （inequality = 题面不等式，须含关系符 < > <= >= ；claimed = 标答解集，同样写成关系式如 "x>2"、"x<=-1"；
   程序按解集相等判，单未知数）
6. 分式方程(含舍根/增根): {{"kind":"rational_roots","equation":"1/(x-1)=2/(x**2-1)","unknowns":["x"],"claimed":["3"]}}
   （🔴 分式方程标答因「使分母为 0 的增根需舍去」只保留部分解时，用本 kind 而非 kind 1：
   程序自动求出裸方程候选解、剔除让任一分母为 0 的增根，按存活解集与 claimed 完全相等判；
   claimed = 标答舍根后保留的解，漏剔增根/把舍掉的根写进 claimed 都会被判 fail）
🔴 应用题（行程/工程/利润等）：直接复用 kind 1 equation_solve —— 列出题面方程组到 equations、
   未知数到 unknowns、标答到 claimed（应用题往往有正负/取值约束只取部分解，此时若 claimed 少于
   裸方程全部解会被误判 fail，应改用 kind 3 numeric 逐解验或输出 kind:none）。"""

# ---------------------------------------------------------------------------
# 🔴 题型结构契约（PRD-C-013 P11 单一事实源）：GENERATE / REGEN / ADD 共用，
# 约束「每道题按其 qtype 长成对的结构」。与验算载荷契约正交：那个管「答案能不能被
# sympy 验」，这个管「题面/选项/答案的形状对不对」。代码级 structure_lint 同口径校验。
# 排版（吃 aigeek 前缀缓存）：本常量属固定段，各 prompt 把它排在变动段（题干/facts）之前。
# 同 _PAYLOAD_CONTRACT，以「format 模板片段」形态存在（无 {占位符}，但与会被 .format()
# 的模板拼接，故文本内若出现花括号须双写——当前无）。
# ---------------------------------------------------------------------------
_QTYPE_CONTRACT = """题型结构契约（每道题必须按其 qtype 字段长成对应结构，违反 = 程序结构 lint 抓出并回炉）：
- 选择：**单一设问** + 恰 4 个选项（A、B、C、D 各一），answer 为选项字母（A/B/C/D 之一）。
  🔴 禁止把多小问 (1)(2)(3) 或 ①②③ 嵌进一道选择题（那是「解答」题的形态，不是选择题）。
- 填空：题干含空位标记（____ 下划线 或 ( ) 括号空），answer 为要填的值。
- 判断：单一陈述句，answer 为「对」或「错」（正确/错误亦可）。
- 解答：**允许** (1)(2)(3) 多小问，answer/solution 分小问作答；这是唯一可含多小问的题型。"""

# 🔴 难度四档 rubric（整改2·2026-06-12·单一事实源）：难度判定并入生题——出题调用同步产出
#   每道题的 difficulty（不再独立走一轮 _grade_difficulty 复评）。rubric 标准本身不变（仍是
#   22-SSOT §2 四档断言），由 GENERATE/REGEN/ADD/revise 出题 prompt 嵌入，让出题时就按 rubric
#   断言难度。难度是「评级」（LLM rubric 断言）≠「判对错」（归闸B sympy，铁律不破）。
#   🔴 本常量将被拼进会 .format() 的 prompt 模板，文本内花括号须双写转义（{{ }}）。
_DIFFICULTY_RUBRIC = """难度四档 rubric（每道题的 difficulty 按下面标准判级，1~4 绝对难度，不是相对母题；难度是评级不是判对错）：
- 4（压轴）：≥2 个真实难点 / 多突破口综合。
- 3（多步综合）：1 个难点，或 考察类型∈{{证明推理·应用建模·探究归纳}}，或 解法骨架含【最难步】构造。
- 2（常规）：无难点 + 考察类型∈{{直接计算·公式套用·性质判定}} + 多步骨架。
- 1（送分）：无难点 +（概念辨析 或 单步骨架）。"""

# 🔴 排版（PRD-C-012 任务3·吃 aigeek 前缀自动缓存）：固定规则/契约段在前，
# 含 {占位符} 的变动段（配方/铁律的考点名、母题 DNA）移到末尾；语义一字不改。
GENERATE_PROMPT = (
    """你是浙教版初中数学命题专家。基于母题 DNA，造 {n} 道举一反三变式。

只输出 JSON 数组(不要解释)，每个元素：
{{"stem":"题干(Markdown+LaTeX)","answer":"标准答案","solution":"完整解析(过程+答案)",
  "qtype":"选择/填空/解答","difficulty":1~4,"level":"normal/hard","injected_kp":"相邻kp名或null",
  "verify_payload":{{...该题的程序验算载荷，契约见下...}}}}

"""
    + _DIFFICULTY_RUBRIC
    + """

格式硬规定（stem/answer/solution 三个字段都遵守）：
- 🔴 题面(stem)与选项里的数学式**一律行内 $...$**，如 $\\sqrt{{2}}$、$x^2-3x+2=0$、$\\frac{{px+a}}{{4}}=2-\\frac{{x+bp}}{{8}}$；
  **严禁** `$$...$$` / `\\[ \\]` / 任何 display 块级公式（会渲染成撑满整行的大号公式、强制换行，破坏阅读）。
- **仅 solution 里多行分步推导**可用 $$...$$（一步一行的竖排演算）；其余单个等式仍优先行内 $...$。
- **禁止**裸 LaTeX 命令、禁止 \\( \\) 定界符。
- 换行用 JSON 标准转义 \\n（一个反斜杠），不要写成 \\\\n。

"""
    + _QTYPE_CONTRACT
    + """

verify_payload 字段（PRD-C-012 4a·出题自带验算载荷：把**这道题自己的题干 + 标准答案**抽成可被 sympy 程序验算的结构化载荷，验算对象 claimed = 该题标准答案）：
"""
    + _PAYLOAD_CONTRACT
    + """
抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。

铁律：
- **主考点 + 年级 硬守恒**：每道题都必须仍考「{kp_name}」、仍在该年级范围内。
- 守{{解题结构, 难度(普通题)}}；只换{{数字, 场景}}。
- 难题 1 道升一档；可综合 1 个相邻知识点(主考点仍守，注入为副点)，填到 injected_kp。
- **答案可程序验算**(PRD-C-012 4c)：answer 优先给可计算的数值/表达式（如 $x_1=2, x_2=3$、
  $3\\sqrt{{2}}$、选项字母），能出数值答案就不要出纯文字表述答案（证明/作图类除外）；
  数字设计成解恰好整洁可验（避免无理数逼近、区间叙述、带单位混排）。

配方(默认)：共 {n} 道 = {n_normal} 道普通(守难度) + {n_hard} 道难题(升一档)。

母题 DNA：
- 主考点(硬守恒，不可改): {kp_name}
- 年级(硬守恒): {grade}
- 题型: {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}"""
)


def _mother_facts(state: VariantState) -> dict:
    analysis = state.get("analysis") or {}
    mdna = state.get("mother_dna") or {}
    dna = mdna.get("dna") or {}  # 🔴 B1：classify 抽的 DNA 契约 v1（嵌在 mother_dna.dna）
    grade_node = analysis.get("grade") or {}
    kp = analysis.get("kp") or {}
    anchored = kp.get("anchored") or {}
    # 🔴 B1 subject_id 改语义 = 科目锚 level1（学段学科册层，跟年级册定，单科目期可固定）：
    #   取年级 4 位 code（如 3071=七上）；缺则回退 _grade_to_code(年级文案)。
    #   知识点叶子 code 不再塞 subject_id —— 那是 dim1KpId（主 kp）的活。
    subject_l1 = grade_node.get("code") or _grade_to_code(grade_node.get("value"))
    return {
        "kp_name": kp.get("value") or "未知考点",
        "grade": grade_node.get("value") or "未知年级",
        "qtype": dna.get("qtype") or (analysis.get("qtype") or {}).get("value") or "解答",
        "stem": mdna.get("stem") or "",
        "skeleton": mdna.get("solution_skeleton") or mdna.get("answer") or "",
        # 🔴 入库用：科目锚 level1（subject_id） + 主 kp 叶子 code（dim1KpId）分级
        "subject_id": subject_l1,  # 科目锚 level1（学段学科册）
        "dim1_kp_id": anchored.get("code"),  # 主 kp 叶子 code（DNA 锚到的真知识点）
        "mother_question_id": mdna.get("mother_question_id"),
        # 🔴 PRD-C-015 批4·缺口10：母题脏（守恒维改）→ persist_items 据此 update 已入库 role=mother 行。
        "mother_dirty": bool(mdna.get("dirty")),
        # 🔴 图母题不在库 → 入库时先把母题(原题)也落库挂血缘，下面这几项给 build_mother_bo 用
        "mother_answer": mdna.get("answer"),
        "mother_solution": mdna.get("solution_skeleton") or mdna.get("answer"),
        "mother_difficulty": dna.get("difficulty") or mdna.get("difficulty"),
        "mother_structure": mdna.get("structure"),
        "kp_confidence": (kp.get("confidence") if isinstance(kp, dict) else None),
        "image_url": state.get("image_url"),
        # 🔴 B1 全维 DNA 穿进 BO（T3）：副 kp/标签/骨架/场景/考察类型/难点 + 锚定审计
        "dna": dna,
    }


# ---------------------------------------------------------------------------
# 🔴 W2 守恒注入（PRD-C-014 B2·T1）：母题 DNA（facts.dna，由 B1 dna_extract 抽）→ 出题
# 硬约束段，GENERATE / REGEN / ADD 三处共用单一事实源。守恒四件套：
#   ① 知识点白名单：解题所需 kp ⊆ 母题{main_kp + secondary_kps}（列 id+名称），白名单为空集
#      则不放行生成（上层降级 clarify/报错，不裸出）——见 _conservation_blocked。
#   ② 考察类型守恒（= 母题 exam_type）。
#   ③ 解法骨架【最难步】基因保留（变式必须保留母题骨架中【】标注的最难步结构）。
#   ④ 表皮必换：数字全换且解整洁（解不整洁宁可换数），场景可换。
# 🔴 文本含 {…} 字面量须双写转义（本段会被拼进 .format() 的 prompt 模板）。
# ---------------------------------------------------------------------------
def _kp_whitelist(dna: dict | None) -> list[tuple[str, str]]:
    """从母题 DNA 取知识点守恒白名单 [(id, name), ...] = 主 kp + 副 kp（去重、去空）。

    纯函数（可单测）。白名单为空 = 母题 DNA 没锚到任何知识点 → 上层据此不放行生成。
    """
    dna = dna or {}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(kp: Any) -> None:
        if not isinstance(kp, dict):
            return
        kid = str(kp.get("id") or "").strip()
        name = str(kp.get("name") or "").strip()
        key = kid or name
        if not key or key in seen:
            return
        seen.add(key)
        out.append((kid, name))

    _add(dna.get("main_kp"))
    for s in dna.get("secondary_kps") or []:
        _add(s)
    return out


def _conservation_blocked(dna: dict | None) -> str | None:
    """🔴 白名单空集守门（T1）：母题 DNA **已抽取但**无任何锚定知识点（白名单空集）→ 返回拦截
    原因（上层降级，不放行生成）；白名单非空 → None（放行）。纯函数、可单测。

    🔴 边界：dna 完全缺失（{} / None，= 库内母题直进 generate 未走 B1 DNA 抽取的合法路径，
    其考点来自 analysis.kp 而非 DNA）→ None（不拦），由既有 mother_confirmed/_conf_ok 闸把关。
    只有「DNA 被抽过（dict 非空）却没锚到任何 kp」才是真·守恒失守，必拦。
    """
    if not dna:
        return None  # 无 DNA = 库内母题合法路径，不归本闸管
    if not _kp_whitelist(dna):
        return "母题 DNA 未锚定任何知识点（守恒白名单为空），无法保证变式不超纲，已暂停生成"
    return None


def _conservation_clause(dna: dict | None) -> str:
    """🔴 W2 守恒硬约束段（GENERATE/REGEN/ADD 共用单一事实源·T1）。返回可拼进 prompt 的文本。

    数据源 = 母题 DNA（facts.dna）：白名单 / 考察类型 / 解法骨架最难步。空字段优雅降级
    （某项缺则该条不注入，绝不输出半截占位）。调用前应已过 _conservation_blocked（白名单非空）。
    """
    dna = dna or {}
    lines: list[str] = ["🔴 守恒硬约束（W2·违反 = 不是合格平行变式）："]

    wl = _kp_whitelist(dna)
    if wl:
        wl_s = "、".join(f"{name}（{kid}）" if kid else name for kid, name in wl)
        lines.append(
            f"① 知识点守恒：解这道变式**所需的知识点必须 ⊆ {{{wl_s}}}**——只能用白名单内的知识点，"
            "禁止引入白名单外的知识点（超纲即废）。"
        )

    exam_type = str(dna.get("exam_type") or "").strip()
    if exam_type:
        lines.append(
            f"② 考察类型守恒：变式的考察类型必须仍是「{exam_type}」（与母题同——不许把"
            "「直接计算」改成「证明推理」之类换赛道）。"
        )

    skeleton = dna.get("skeleton") or []
    hardest = next(
        (str(s) for s in skeleton if "【" in str(s) and "】" in str(s)), None
    )
    if hardest:
        lines.append(
            f"③ 解法骨架【最难步】基因保留：母题骨架的最难一步是「{hardest}」——变式必须保留"
            "这一步**同类挑战**（同样的构造/转化/分类讨论难度），不许把它简化掉或绕开。"
        )
    elif skeleton:
        sk_s = "；".join(str(s) for s in skeleton)
        lines.append(
            f"③ 解法骨架基因保留：变式必须按母题同一解法骨架可解——骨架为「{sk_s}」。"
        )

    lines.append(
        "④ 表皮必换：数字必须**全部换掉**且设计成解依然整洁（解不整洁宁可再换一组数），"
        "场景可换同类；与母题题面几乎相同（只是复读）= 废。"
    )
    # 🔴 难度规则（T3·22-SSOT §2）：normal 与母题同档；hard = 母题档 +1（封顶 4）——
    #    hard 按构造定义就是在同一骨架上**多加一个真实突破口**（一步构造/转化/分类讨论），
    #    不是把数字变丑。
    lines.append(
        "⑤ 难度档：level=\"normal\" 的题与母题同档；level=\"hard\" 的题 = 母题档 +1（封顶 4），"
        "靠在同一骨架上**多加一个真实突破口**（一步构造/转化/分类讨论）升档，不是把数字变丑。"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 🔴 PRD-C-015 批3·W2' 难题注卡（模型卡片注入 GENERATE/REGEN prompt）：
#   注卡条件矩阵（§3.2）：母题难度 ≥3（LLM rubric 档）且命中**非 M00** 模型 → 注卡；
#   难度<3 或仅 M00 → 不注（返回 ""）。卡片文本逐字取词库表（G3）+ 反退化/反表皮缩放约束。
# 🔴 注卡是 prompt 引导（生成侧），绝不混进 pass/fail 判决（铁律：判决只读 sympy）。
#   反查库故障 → fetch_model_cards 抛异常被本函数兜成「不注卡」降级（不卡死出题，C-010 纪律）。
# ---------------------------------------------------------------------------
def _note_card_models(facts: dict) -> list[dict[str, str]]:
    """从母题 DNA.models 取**非 M00**的模型 [{id,name}]（注卡候选）。纯函数。"""
    dna = facts.get("dna") or {}
    out: list[dict[str, str]] = []
    for m in dna.get("models") or []:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "").strip()
        if mid and mid != model_anchor.M00_ID:
            out.append({"id": mid, "name": str(m.get("name") or "").strip()})
    return out


def _should_note_card(facts: dict) -> bool:
    """注卡条件（§3.2 矩阵）：母题难度 ≥ 阈值 且 命中非 M00 模型。纯函数·可单测。"""
    mother_d = _to_int((facts.get("dna") or {}).get("difficulty")) or _to_int(
        facts.get("mother_difficulty")
    )
    if mother_d is None or mother_d < model_anchor.NOTE_CARD_DIFFICULTY_MIN:
        return False
    return bool(_note_card_models(facts))


def _model_cards_clause(facts: dict) -> str:
    """🔴 W2' 注卡段（GENERATE/REGEN 共用单一事实源）。不满足注卡条件 / 取卡失败 → ""（不注）。

    满足条件 → 反查词库表取命中模型整行 → build_model_cards_clause（卡片文本 + 反退化约束）。
    🔴 取卡（反查库）故障 → 降级返回 ""（不注卡，照常出题，绝不卡死；G5/C-010 闸门必有降级路径）。
    """
    if not _should_note_card(facts):
        return ""
    note_models = _note_card_models(facts)
    ids = [m["id"] for m in note_models]
    try:
        card_map = model_anchor.fetch_model_cards(ids)
    except Exception:  # noqa: BLE001 — 反查库故障 → 不注卡降级（不卡死出题）
        return ""
    # 按 DNA.models 顺序取卡（保留锚定时的次序），缺卡的 id 跳过（不半截注入）。
    cards = [card_map[i] for i in ids if i in card_map]
    return model_anchor.build_model_cards_clause(cards, with_anti_degen=True)


def _maybe_note_card_block(facts: dict) -> str:
    """注卡段拼接器：有卡 → 前缀 "\\n\\n" 接进 prompt；无卡（不注） → ""（不留空行）。"""
    clause = _model_cards_clause(facts)
    return ("\n\n" + clause) if clause else ""


# ---------------------------------------------------------------------------
# 🔴 确定上下文硬约束块（整改1·2026-06-12，影响最大）：generate / regen / revise 所有
# 产「题面 + 解析」的 prompt 统一注入此块。内容**全部来自 state 已锚定的确定事实**（考点 /
# 年级学期进度 / 教材版本），不许 LLM 猜。实测痛点：老师明说「7 年级没学二元方程只能用一元
# 一次」，模型生成的解析仍越界用二元 → 本块以「必须遵守」级硬约束压住解题方法不越界。
#
# 排位铁律（与既有经验对齐）：本块措辞为硬约束、放在 prompt 靠前位置，但**绝不挤掉守恒白名单
# 段**（_conservation_clause 仍独立注入、两段共存）；二者正交——守恒段管「考点/考察类型/骨架
# 基因不超纲」，本块管「解题方法不越学生当前进度」。
# 🔴 文本含 {…} 字面量须双写转义（本段会被拼进会 .format() 的 prompt 模板）。
# ---------------------------------------------------------------------------
# 年级 4 位 code 第 4 位 = 学期（1=上 / 2=下）；学段册前缀 → 学段（307=七、308=八、309=九）。
_GRADE_CODE_SEG = {"307": "七年级", "308": "八年级", "309": "九年级"}
_TERM_CODE_SEG = {"1": "上学期", "2": "下学期"}


def _progress_phrase(facts: dict) -> str:
    """从 facts 拼「年级 + 学期」进度短语（如「七年级上学期」）。
    优先用 grade 文案（已含学期），缺学期信息时用 grade code 第 4 位补「上/下学期」。纯函数。"""
    grade = str(facts.get("grade") or "").strip()
    code = str(facts.get("subject_id") or "").strip()
    has_term = any(w in grade for w in ("上", "下", "上学期", "下学期", "上册", "下册"))
    if grade and has_term:
        return grade
    # grade 缺学期 → 用 code 第 4 位补
    if len(code) >= 4:
        seg, term = code[:3], code[3]
        base = _GRADE_CODE_SEG.get(seg) or (grade or "")
        term_cn = _TERM_CODE_SEG.get(term)
        if base and term_cn:
            return f"{base}{term_cn}"
    return grade or "未知进度"


def _context_block(facts: dict) -> str:
    """🔴 确定上下文硬约束块（整改1·单一事实源，generate/regen/revise 共用）。返回可拼进
    prompt 的文本。所有事实取自已锚定的 facts（考点/年级学期/教材版本），LLM 不得猜测。

    - 主考点：锚定叶子名 +（有就带）副 kp 名（知识点路径替代——库无独立 path 字段）。
    - 年级 + 学期 = 教学进度边界：明确「严禁使用该进度之后才学的内容」+ 通用越界例子（不写死
      具体某年级的越界方法，按进度通用表述，避免给低年级讲高年级才有的概念）。
    - 教材版本：settings.TEXTBOOK_VERSION 有值才注入（默认空 → 不注入，绝不编造）。

    纯函数、可单测。空字段优雅降级（某项缺则该条不注入半截占位）。
    """
    dna = facts.get("dna") or {}
    kp_name = str(facts.get("kp_name") or "").strip() or "未知考点"
    progress = _progress_phrase(facts)
    lines: list[str] = [
        "🔴 确定上下文（以下是已锚定的客观事实，**必须遵守**，不得自行更改或猜测）："
    ]
    # ① 主考点（+ 副 kp 当知识点路径补充）
    sec = [
        str(s.get("name")).strip()
        for s in (dna.get("secondary_kps") or [])
        if isinstance(s, dict) and str(s.get("name") or "").strip()
    ]
    if sec:
        lines.append(f"- 主考点：「{kp_name}」；连带知识点：{'、'.join(sec)}。")
    else:
        lines.append(f"- 主考点：「{kp_name}」。")
    # ② 年级 + 学期 = 教学进度边界（硬约束解题方法不越界）
    lines.append(
        f"- 学生当前进度：{progress}。🔴 解题方法**严禁使用该进度之后才学的内容**——"
        "题面、答案、解析里出现的一切方法/概念/工具都必须是学生到这个进度已经学过的。"
        "例如：低年级尚未学到方程组/不等式组/函数图象等更高阶工具时，绝不能用它们来解题或讲解，"
        "必须改用当前进度内的方法（如只用一元一次方程、算术、已学过的几何性质等）。"
    )
    # ③ 教材版本（有真实来源才注入，绝不编造）
    tv = str(getattr(settings, "TEXTBOOK_VERSION", "") or "").strip()
    if tv:
        lines.append(f"- 教材版本：{tv}（命名/记法/方法口径以该版本教材为准）。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 首轮配方旋钮（修 bug：首轮带图附带的文字要求被整条丢弃 → 旋钮成一等公民）
# 链路：generate 入口（knobs 为 None 时）一次受约束 LLM 抽取（KNOBS_PROMPT）
#   → 纯函数 normalize_knobs 钳制 → 存回 state（跨轮保留）
#   → recipe_from_knobs 驱动 GENERATE_PROMPT 配方段
#   → 代码级 shape_check（数量/题型分布/递增单调）不符整组 retry 1 次，仍不符 → 头部 ⚠ 外显
#   （B2·T2：闸A LLM judge 退役后，配方对齐不再注入 judge prompt——闸A 改纯代码三检）。
# 🔴 抽取失败/解析失败 → knobs={} 回落默认，绝不卡死出题（G5）。
# ---------------------------------------------------------------------------
KNOBS_PROMPT = """你是举一反三 agent 的出题配方抽取器（受约束抽取：只抽老师明说的，绝不脑补）。老师贴题目图时附带了下面这句话，请从中抽出出题配方旋钮。

老师的话：
{utterance}

只输出一个 JSON（不要解释、不要 markdown fence）：
{{
  "count": null,            // 要出几道题（正整数，1~8）；没说填 null。🔴 中文数量词必须映射成阿拉伯数字：一/壹/一道/一个→1，两/俩/二/一对/两道→2，仨/三→3，四→4，五→5，六→6，七→7，八→8（"再来两道"=2，"出俩"=2，"一对"=2）
  "difficulty_plan": null,  // 难度安排："increasing"=难度递增/越来越难/一道比一道难；没提难度安排填 null；其它难度要求把老师原话填进来（如"都出难题"）
  "qtype_dist": null,       // 题型配比，如 {{"选择":2,"填空":2,"解答":1}}；题型只能用 选择/填空/解答 三类（应用题/计算题/证明题等都归"解答"）；没说填 null
  "note": ""                // 其余装不进上面旋钮的自由要求原话（如"贴近生活场景"、"数字简单点"）；没有填 ""
}}

硬约束：
- 只抽老师明确说了的；没说的旋钮一律 null/""，绝不脑补默认值。
- qtype_dist 的值必须是正整数；如与 count 看似矛盾也如实抽取，程序会做最终校验。"""

# 题型归一表（normalize_knobs 用）：只认 选择/填空/解答 三类
_QTYPE_ALIAS: dict[str, str] = {
    "选择": "选择",
    "选择题": "选择",
    "单选": "选择",
    "单选题": "选择",
    "填空": "填空",
    "填空题": "填空",
    "解答": "解答",
    "解答题": "解答",
    "计算": "解答",
    "计算题": "解答",
    "应用": "解答",
    "应用题": "解答",
    "证明": "解答",
    "证明题": "解答",
    "大题": "解答",
}
# 归一后要在 note 里保留语义的原始题型词（应用题 → 解答 + note "应用场景"）
_QTYPE_NOTE_HINTS: dict[str, str] = {"应用": "应用场景", "应用题": "应用场景"}

# BUG-001：编辑·regenerate 的 note 里若含「改成X题」这类显式改题型诉求，需从 note 抽出目标
# 题型覆盖 REGEN 出题的 qtype（否则 REGEN_PROMPT 把 qtype 钉死成原题型，改题型形同没改）。
# 纯字符串匹配（零 LLM）：命中题型关键词且 note 出现「改/换/变/成」改动词才算（避免「这是选择题，
# 数字简单点」这类陈述被误判为改题型）。
_QTYPE_KEYWORDS: list[tuple[str, str]] = [
    ("选择", "选择"), ("单选", "选择"),
    ("填空", "填空"),
    ("解答", "解答"), ("计算", "解答"), ("应用", "解答"), ("证明", "解答"), ("大题", "解答"),
]
_QTYPE_CHANGE_VERBS = ("改", "换", "变", "成", "出", "做")


def _qtype_from_note(note: str | None) -> str | None:
    """从编辑 note 里抽出老师要求的目标题型（选择/填空/解答），抽不出返 None。

    纯函数（零 LLM/零 IO，可单测）。命中规则：note 同时含「改/换/变/成…」改动词 + 某题型关键词。
    多个题型关键词命中时取**最后出现**的（更贴近「从X改成Y」的 Y）。
    """
    s = str(note or "")
    if not s or not any(v in s for v in _QTYPE_CHANGE_VERBS):
        return None
    best: tuple[int, str] | None = None
    for kw, canon in _QTYPE_KEYWORDS:
        pos = s.rfind(kw)
        if pos >= 0 and (best is None or pos > best[0]):
            best = (pos, canon)
    return best[1] if best else None


KNOBS_COUNT_MIN, KNOBS_COUNT_MAX = 1, 8
PLAN_INCREASING = "increasing"
_PLAN_INCREASING_WORDS = ("increasing", "递增", "越来越难", "逐题变难", "一道比一道难")
DIFFICULTY_CAP = 4  # 难度封顶（递增计划逐题 +1 的上限；S1.3 由 5→4 对齐绝对 rubric 1-4）


def normalize_knobs(parsed: Any) -> dict[str, Any]:
    """🔴 纯函数钳制（零 LLM/零 IO，可单测）：LLM 抽取产物 → 受约束 knobs dict。

    规则：
    - 非 dict/解析失败 → {}（回落默认配方）。
    - count：宽容转 int，钳到 [1, 8]；非法 → 丢弃。
    - qtype_dist：键过 _QTYPE_ALIAS 归一（应用题/计算题/证明题→解答，且"应用"在 note 保留
      「应用场景」语义）；不认识的题型键丢弃；值须为正整数；同义键合并求和。
    - dist 总和同样钳到 KNOBS_COUNT_MAX（按点名顺序累计到上限封口，截断写进 note 外显）——
      防失控护栏对 dist 路径同等生效，不被「逐项点名」绕过。
    - dist 总和与 count 不一致 → 以 dist 总和为准，且把「数量从 X 调整为 Y」写进 note 外显
      （assemble 头部可见，不静默吞老师的数）。
    - difficulty_plan：命中递增词 → "increasing"；"default"/空 → 丢弃；其余原话保留。
    - note：strip 后非空才保留。
    - 全空 → {}。
    """
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, Any] = {}

    cnt = _to_int(parsed.get("count"))
    if cnt is not None:
        out["count"] = max(KNOBS_COUNT_MIN, min(cnt, KNOBS_COUNT_MAX))

    note_bits: list[str] = []
    raw_dist = parsed.get("qtype_dist")
    if isinstance(raw_dist, dict):
        dist: dict[str, int] = {}
        for k, v in raw_dist.items():
            qt = _QTYPE_ALIAS.get(str(k or "").strip())
            n = _to_int(v)
            if not qt or n is None or n <= 0:
                continue
            dist[qt] = dist.get(qt, 0) + n
            hint = _QTYPE_NOTE_HINTS.get(str(k or "").strip())
            if hint and hint not in note_bits:
                note_bits.append(hint)
        if dist:
            total = sum(dist.values())
            if total > KNOBS_COUNT_MAX:
                # 🔴 防失控护栏（与 count 钳制/ADD_COUNT_MAX 同哲学）：按点名顺序累计到上限封口
                clipped: dict[str, int] = {}
                budget = KNOBS_COUNT_MAX
                for qt, n in dist.items():
                    if budget <= 0:
                        break
                    take = min(n, budget)
                    clipped[qt] = take
                    budget -= take
                note_bits.append(
                    f"题型配比共 {total} 道超单轮上限，已截到 {KNOBS_COUNT_MAX} 道"
                )
                dist = clipped
                total = KNOBS_COUNT_MAX
            out["qtype_dist"] = dist
            if out.get("count") != total:
                if out.get("count") is not None:
                    # count 与配比总和冲突 → dist 为准，但调整必须外显（不静默吞老师的数）
                    note_bits.append(f"按题型配比把数量从 {out['count']} 调整为 {total}")
                out["count"] = total  # dist 总和为准（含 count 缺失时补齐）

    plan = str(parsed.get("difficulty_plan") or "").strip()
    if plan and plan.lower() != "default":
        if any(w in plan.lower() for w in _PLAN_INCREASING_WORDS):
            out["difficulty_plan"] = PLAN_INCREASING
        else:
            out["difficulty_plan"] = plan  # 自由难度要求原话保留（generate/闸A 原样注入）

    note = str(parsed.get("note") or "").strip()
    if note:
        note_bits.append(note)
    if note_bits:
        out["note"] = "；".join(note_bits)

    return out


def recipe_from_knobs(knobs: dict[str, Any] | None, mother_difficulty: Any = None) -> dict[str, Any]:
    """🔴 纯函数：knobs → generate 配方（n/n_normal/n_hard + 老师配方段 spec + 递增预期档位）。

    knobs 空 → 与旧默认完全等价：n=3、2 普 1 难、spec=""（行为不变的回归锚点）。
    递增计划：从母题难度起逐题 +1、封顶 DIFFICULTY_CAP。expected_difficulties 只用于
    prompt 文案 + n_hard 推算；闸A/代码闸的同尺判据 = generate 落在 item 级的
    expected_difficulty 印记 + shape_check(mother_difficulty) 现算（二者公式一致）。
    """
    knobs = knobs or {}
    if not knobs:
        n_normal, n_hard = DEFAULT_SHAPE["normal"], DEFAULT_SHAPE["hard"]
        return {
            "n": n_normal + n_hard,
            "n_normal": n_normal,
            "n_hard": n_hard,
            "spec": "",
            "expected_difficulties": None,
        }

    dist = knobs.get("qtype_dist") or {}
    n = knobs.get("count") or (sum(dist.values()) if dist else 0) or (
        DEFAULT_SHAPE["normal"] + DEFAULT_SHAPE["hard"]
    )
    md = _to_int(mother_difficulty) or 3
    plan = knobs.get("difficulty_plan")
    expected: list[int] | None = None

    lines = [f"- 共 {n} 道（必须恰好 {n} 道，不多不少）。"]
    if dist:
        dist_s = "、".join(f"{k}×{v}" for k, v in dist.items())
        lines.append(f"- 题型配比：{dist_s}（每道题的 qtype 严格按此配比给）。")
    if plan == PLAN_INCREASING:
        expected = [min(md + i, DIFFICULTY_CAP) for i in range(n)]
        d_s = ",".join(str(d) for d in expected)
        lines.append(
            f"- 难度计划：递增 —— 从母题难度({md})起逐题升一档、封顶 {DIFFICULTY_CAP}；"
            f"各题 difficulty 依次为 {d_s}；难度高于母题的填 level=\"hard\"，否则 \"normal\"。"
        )
    elif plan:
        lines.append(f"- 难度要求（老师原话，best-effort 满足）：{plan}")
    if knobs.get("note"):
        lines.append(f"- 其他要求（best-effort 吸收，撞主考点/年级硬守恒的忽略）：{knobs['note']}")

    spec = "\n\n老师指定配方（🔴 优先于上面的默认配方，必须严格满足）：\n" + "\n".join(lines)
    if expected:
        n_hard = sum(1 for d in expected if d > md)
    else:
        n_hard = 1 if n >= 2 else 0
    return {
        "n": n,
        "n_normal": n - n_hard,
        "n_hard": n_hard,
        "spec": spec,
        "expected_difficulties": expected,
    }


def shape_check(
    items: list[dict[str, Any]],
    knobs: dict[str, Any] | None,
    mother_difficulty: Any = None,
) -> list[str]:
    """🔴 纯函数·代码级配方校验：返回缺陷清单（空 = 合格）。knobs 空 → 永远 []（旧行为）。

    规则表：
    - S1 数量：knobs 给了 count（或 dist 推得）→ len(items) 必须相等。
    - S2 题型分布：knobs 给了 qtype_dist → 各题 qtype（过 _QTYPE_ALIAS 归一）计数须逐项相等。
    - S3 递增：difficulty_plan=increasing 时与闸A 同一把尺 —— 给了 mother_difficulty →
      逐项比对预期档 min(md+i, DIFFICULTY_CAP)（让整组 retry 有机会一次修对，而不是代码闸
      放行后闸A 必 warn 且回炉结构性修不动）；母题难度未知 → 退化为单调不减（缺难度按违规算）。
    """
    knobs = knobs or {}
    if not knobs:
        return []
    defects: list[str] = []

    want_n = knobs.get("count")
    if want_n and len(items) != want_n:
        defects.append(f"数量不符：要求 {want_n} 道，实出 {len(items)} 道")

    dist = knobs.get("qtype_dist") or {}
    if dist:
        got: dict[str, int] = {}
        for it in items:
            qt = _QTYPE_ALIAS.get(str(it.get("qtype") or "").strip(), str(it.get("qtype") or "").strip())
            got[qt] = got.get(qt, 0) + 1
        if any(got.get(k, 0) != v for k, v in dist.items()):
            want_s = "、".join(f"{k}×{v}" for k, v in dist.items())
            got_s = "、".join(f"{k}×{v}" for k, v in got.items()) or "(空)"
            defects.append(f"题型分布不符：要求 {want_s}，实出 {got_s}")

    if knobs.get("difficulty_plan") == PLAN_INCREASING and len(items) >= 2:
        diffs = [_to_int(it.get("difficulty")) for it in items]
        md = _to_int(mother_difficulty)
        if md is not None:
            # 与闸A/GENERATE_PROMPT 同一把尺：逐项比对预期档
            expected = [min(md + i, DIFFICULTY_CAP) for i in range(len(items))]
            if diffs != expected:
                defects.append(
                    "要求难度递增但实出难度档与计划不符："
                    f"预期 {','.join(str(d) for d in expected)}，"
                    f"实出 {','.join(str(d) for d in diffs)}"
                )
        else:
            mono = all(
                a is not None and b is not None and b >= a for a, b in zip(diffs, diffs[1:])
            )
            if not mono:
                defects.append(
                    "要求难度递增但实出难度非单调不减：" + ",".join(str(d) for d in diffs)
                )
    return defects


# ---------------------------------------------------------------------------
# 🔴 题型结构 lint（PRD-C-013 P11.3 纯函数·与 shape_check 同位）：单题按其 qtype
# 校验结构对不对（_QTYPE_CONTRACT 的代码侧镜像）。与 shape_check 正交——那个管整组
# 配方（数量/题型分布/递增），这个管单题形态（选择题别长成多小问嵌合体等）。
# 🔴 降级铁律：解析不了一律返回 []（视作合规），绝不卡死出题（G5）。判 verdict 仍归
# sympy（本 lint 不碰答案对错，只碰形状）。
# ---------------------------------------------------------------------------
# 多小问标记：(1)(2)... / ①②...。选择题命中即「嵌合体」缺陷。
# 🔴 对抗审④收紧（PRD-C-013）：旧正则把单个 (1) / 函数记号 f(1)/g(2)/点(1) 误判成多小问嵌合体
#   → 白烧一次结构 REGEN 预算、假缺陷徽章（预算被假阳性吃掉后真 heal/rework 反被跳过）。
#   新口径 = 只在「≥2 个连号小问标记」才判嵌合体：
#   ① 括号数字 (1) 且**左括号前不是 \w**（排除 f(1)/g(2) 函数记号、x(1) 等）——收集编号，
#      含 ≥2 个连续编号（如同时有 1 和 2 / 2 和 3）才算；单个 (1) 不算。
#   ② 圆圈数字 ①②③：含 ≥2 个连续编号才算（单个 ① 不算）。
# 括号小问编号：左括号前非 \w（避免函数记号），(数字) 中文数字一二三四五；用 findall 数命中。
_PAREN_SUBQ_RE = re.compile(r"(?<![\w])[（(]\s*([1-9]|[一二三四五])\s*[）)]")
_CIRCLED_SUBQ_RE = re.compile(r"[①②③④⑤⑥]")
_CN_NUM_ORDER = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
_CIRCLED_ORDER = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6}


def _has_consecutive(nums: list[int]) -> bool:
    """有序编号集合里是否存在两个相邻编号（n 与 n+1 都出现）→ 真多小问嵌合体的判据。"""
    s = set(nums)
    return any((n + 1) in s for n in s)


def _is_multi_subquestion(stem: str) -> bool:
    """🔴 纯函数（对抗审④）：题干是否含「多小问嵌合体」结构（≥2 连号小问标记）。
    单个 (1) / 函数记号 f(1) 一律不算；(1)(2) 连号、①② 连号才算。"""
    paren_nums: list[int] = []
    for g in _PAREN_SUBQ_RE.findall(stem):
        if g.isdigit():
            paren_nums.append(int(g))
        elif g in _CN_NUM_ORDER:
            paren_nums.append(_CN_NUM_ORDER[g])
    if _has_consecutive(paren_nums):
        return True
    circled = [_CIRCLED_ORDER[c] for c in _CIRCLED_SUBQ_RE.findall(stem) if c in _CIRCLED_ORDER]
    return _has_consecutive(circled)
# 选项行：A. / A、 / A) / （A） / A：  —— 用于数选择题选项个数
_OPTION_RE = re.compile(r"(?:^|[\s，,；;])[（(]?\s*([A-D])\s*[）).、、：:]")
# 填空空位：连续下划线 / 全角空格括号 / 中文「填空」括号
_BLANK_RE = re.compile(r"_{2,}|[（(]\s*[）)]|＿{2,}")
# 选择题标答：单个 A-D 字母（去空白/标点后恰好一个字母）
_CHOICE_ANSWER_RE = re.compile(r"^[A-D]$")


def structure_lint(item: dict[str, Any]) -> list[str]:
    """🔴 纯函数·单题结构 lint：按 item['qtype'] 校验题面/选项/答案形态，返回缺陷清单（空=合规）。

    规则（_QTYPE_CONTRACT 的代码镜像）：
    - 选择：① stem 含 (1)(2)/①② 多小问标记 → 嵌合体缺陷；② 选项数 <3 → 选项不足缺陷；
      ③ answer 去噪后非单个 A-D 字母 → 答案形态缺陷。
    - 填空：题干无空位标记（____ / 空括号）→ 缺空位缺陷（宽松，只查空位）。
    - 判断/解答/其它：不约束（解答允许多小问；判断答案形态宽松）。
    🔴 qtype 过 _QTYPE_ALIAS 归一；解析异常 → []（降级，不卡死）。
    """
    try:
        qt = _QTYPE_ALIAS.get(str(item.get("qtype") or "").strip(), str(item.get("qtype") or "").strip())
        stem = str(item.get("stem") or "")
        defects: list[str] = []
        if qt == "选择":
            if _is_multi_subquestion(stem):
                defects.append("选择题混入多小问 (1)(2)/①② —— 应是单一设问 4 选项，不是解答题嵌合体")
            n_opt = len(set(m.group(1) for m in _OPTION_RE.finditer(stem)))
            if n_opt < 3:
                defects.append(f"选择题选项不足（识别到 {n_opt} 个，至少 3 个）")
            ans = re.sub(r"[\s。.，,、；;：:]", "", str(item.get("answer") or "")).upper()
            if not _CHOICE_ANSWER_RE.match(ans):
                defects.append(f"选择题标答非单个选项字母 A-D（实为「{item.get('answer')}」）")
        elif qt == "填空":
            if not _BLANK_RE.search(stem):
                defects.append("填空题题干无空位标记（____ 或 空括号）")
        return defects
    except Exception:  # noqa: BLE001 — lint 解析异常 → 视作合规放行（G5 降级）
        return []


def knobs_desc(knobs: dict[str, Any] | None) -> str:
    """纯函数：knobs → 题组头部人话描述（如「5 道·难度递增·2选择+2填空+1解答」）。空 → ""。"""
    knobs = knobs or {}
    bits: list[str] = []
    if knobs.get("count"):
        bits.append(f"{knobs['count']} 道")
    plan = knobs.get("difficulty_plan")
    if plan == PLAN_INCREASING:
        bits.append("难度递增")
    elif plan:
        bits.append(f"难度「{plan}」")
    dist = knobs.get("qtype_dist") or {}
    if dist:
        bits.append("+".join(f"{v}{k}" for k, v in dist.items()))
    if knobs.get("note"):
        bits.append(str(knobs["note"]))
    return "·".join(bits)


# 🔴 B2·T2：gene_judge_knobs_spec（闸A judge prompt 配方对齐段）随 LLM judge 全链删除——
#    闸A 改纯代码三检（gene_gate_check），不再有 judge prompt 可注入。配方的题型/数量校验
#    仍由 shape_check（纯函数，generate 整组 retry）把关，与闸A 解耦。


def _normalize_generated_item(it: dict[str, Any], facts: dict) -> dict[str, Any]:
    """单题规整 + 净化（generate 全量解析与 P2 流式增量解析共用的单一事实源）。

    🔴 verify_payload（PRD-C-012 4a 出题自带验算载荷）仅当 dict 才随 item 流转——
    item 内部字段：不进 artifact 帧、不入库（_artifact_payload / build_create_bo
    均显式字段白名单，天然不外漏）。
    """
    out: dict[str, Any] = {
        "stem": it.get("stem"),
        "answer": it.get("answer"),
        "solution": it.get("solution"),
        "qtype": it.get("qtype") or facts["qtype"],
        "difficulty": it.get("difficulty"),
        "level": it.get("level") or "normal",
        "injected_kp": it.get("injected_kp") or None,
        # check 待 solve_explain 填（无 check 不许进 assemble）
    }
    if isinstance(it.get("verify_payload"), dict):
        out["verify_payload"] = it["verify_payload"]
    # 🔴 批3·⑦ 反退化载荷（最值/动点构型才有）随 item 流转——item 内部字段，不进帧/不入库。
    if isinstance(it.get("degen_payload"), dict):
        out["degen_payload"] = it["degen_payload"]
    return _sanitize_item(out)


def _parse_generated_items(text: str, facts: dict) -> list[dict[str, Any]]:
    """generate/重试共用：LLM 返回文本 → 规整 items（check 待 solve_explain 填）。"""
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    return [_normalize_generated_item(it, facts) for it in data or [] if isinstance(it, dict)]


def _iter_complete_items(acc: str) -> list[dict[str, Any]]:
    """🔴 纯函数（PRD-C-012 P2 逐题吐出）：从流式累计文本里增量解析「已完整闭合」的 item。

    判定 = 花括号配平（跳过字符串内花括号与转义引号）+ json.loads 成功 + 顶层含 "stem"
    键才算一道；半截题绝不返回。外层包装对象（如 {"items":[...]}）不重复计：已接受
    item 的区间被 consumed 哨兵跳过，且包装对象顶层无 stem 键；item 内嵌套对象
    （如 verify_payload）顶层无 stem 键同样不计。返回按文本出现顺序（= 生成序）。
    """
    out: list[dict[str, Any]] = []
    stack: list[int] = []
    in_str = esc = False
    consumed_end = -1
    for i, ch in enumerate(acc or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if start <= consumed_end:
                continue  # 包着已接受 item 的外层对象 → 不重复计
            try:
                obj = json.loads(acc[start : i + 1])
            except Exception:  # noqa: BLE001 — 闭合但不是合法 JSON → 不算（半截/脏文本）
                continue
            if isinstance(obj, dict) and "stem" in obj:
                out.append(obj)
                consumed_end = i
    return out


async def _extract_knobs(state: VariantState) -> dict[str, Any]:
    """首轮配方旋钮抽取：去掉图 URL 后的人话非空(len>3) → 一次受约束 LLM 抽取 + normalize 钳制。

    🔴 任何失败（LLM 异常/解析失败）→ {} 回落默认配方，绝不卡死出题（G5）。
    """
    user_text = _strip_urls(_latest_human_text(state.get("messages", [])))
    # 🔴 只过滤真空串：「出5道」「来5道」恰好 3 字也是完整数量指令，阈值高了会静默吞掉；
    #   抽取失败本身有 {} 兜底，不靠长度预筛。
    if not user_text:
        return {}
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=KNOBS_PROMPT.format(utterance=user_text))],
            model=settings.variant_model("analyze"),
        )
    except Exception:  # noqa: BLE001 — 旋钮抽取是增强不是关卡
        return {}
    return normalize_knobs(_parse_json(text))


async def generate(state: VariantState, config: RunnableConfig) -> VariantState:
    """③ 造题：配方由首轮旋钮(knobs)驱动，无旋钮走旧默认 3 = 2 普通 + 1 难。

    🔴 入口断言 mother_confirmed 或三锚高置信（防裸奔）。
    🔴 代码级配方校验 shape_check：不符带缺陷反馈整组 retry 1 次，仍不符 → 接受 +
       shape_defects 外显到题组头部（不卡死，G5）。
    """
    if not (state.get("mother_confirmed") or _conf_ok(state.get("analysis") or {})):
        # DNA 闸未过却走到 generate（多入口兜底）→ 拒造，回 clarify 语义
        return {
            "messages": [
                AIMessage(content="母题 DNA 还没确认，我先不造题。请确认年级/考点/题型。")
            ]
        }

    # 🔴 批2·generate 入口防御断言（从机制上绝迹「未解析+未知年级进出题」）：facts 缺年级
    #   或主考点 → 拒绝出题、回确认态。多入口（route_entry 库内母题直进 / patch 重造 / 兜底）
    #   都必过此闸，gate_after_classify 之外的旁路也兜得住。
    _facts_pre = _mother_facts(state)
    _missing_grade = (str(_facts_pre.get("grade") or "").strip() in ("", "未知年级"))
    _missing_kp = (
        str(_facts_pre.get("kp_name") or "").strip() in ("", "未知考点")
        or not _facts_pre.get("dim1_kp_id")
    )
    if _missing_grade or _missing_kp:
        lack = "、".join(
            x for x, m in (("年级", _missing_grade), ("主考点（知识点未锚定）", _missing_kp)) if m
        )
        _emit_stage("generate", "生成题目", "warn", f"母题未定死（缺{lack}）")
        return {
            "messages": [
                AIMessage(
                    content=f"母题还没定死（缺{lack}），我不能出题。请先确认年级 + 主考点知识点，我再造变式。"
                )
            ]
        }

    # 🔴 P13：出题轮入口重置预算（generate→gene_gate→solve_explain→assemble 同轮共享）。
    budget = _budget_bind(state, reset_limit=settings.VARIANT_BUDGET_GENERATE)

    # 🔴 旋钮：新母题轮 analyze 已抽好随 state 来；库内母题直进 generate（不经 analyze）→ 此处兜底抽。
    #   🔴 改动2·阶段灯归属：图片首轮的「解析配方」灯由 classify 真实完成时发 done（带道数）——
    #   此处仅当 knobs 为 None（= 库内母题路径，跳过 analyze/classify）时兜底抽 + 补发该灯，
    #   不在图片路径重复发（防覆盖 classify 已发的终态）。
    knobs = state.get("knobs")
    if knobs is None:
        knobs = await _extract_knobs(state)
        _emit_stage(
            "knobs", "解析配方", "done",
            knobs_desc(knobs) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）",
        )

    facts = _mother_facts(state)
    # 🔴 W2 守恒守门（T1）：母题 DNA 白名单为空集 → 不放行生成，降级回 clarify 语义（不裸出）。
    blocked = _conservation_blocked(facts.get("dna"))
    if blocked:
        _emit_stage("generate", "生成题目", "warn", "母题知识点未锚定")
        return {
            "messages": [
                AIMessage(
                    content=f"{blocked}。请补充确认这道母题考的知识点（年级 + 考点），我再造变式。"
                )
            ]
        }
    mother_d = (state.get("mother_dna") or {}).get("difficulty")
    recipe = recipe_from_knobs(knobs, mother_d)
    _emit_stage("generate", "生成题目", "running", f"{recipe['n']} 道")
    prompt = (
        GENERATE_PROMPT.format(
            n=recipe["n"], n_normal=recipe["n_normal"], n_hard=recipe["n_hard"], **facts
        )
        + "\n\n"
        + _context_block(facts)  # 🔴 整改1：确定上下文硬约束（考点/进度/教材版本，压解题不越界）
        + "\n\n"
        + _conservation_clause(facts.get("dna"))  # 🔴 W2 守恒硬约束注入（T1，与上块正交并存）
        + _maybe_note_card_block(facts)  # 🔴 批3·W2' 难题注卡（难度≥3+非M00 才注，含反退化约束）
        + recipe["spec"]
    )

    # 🔴 思维外放（用户反馈 2026-06-11）：JSON token 对用户是乱码不外放，但流内数
    # "stem" 出现次数 → 思路条实时跳「正在写第 n/N 道」+ 当前题干前几个字，等待不再是黑盒。
    total_n = int(recipe["n"])
    md_i = _to_int(mother_d) or 3
    increasing = bool(knobs) and knobs.get("difficulty_plan") == PLAN_INCREASING

    def _stamp_recipe(item: dict[str, Any], idx: int) -> dict[str, Any]:
        # 🔴 配方印记落 item 级（闸A 改判段只作用于产生该配方的这一轮生成的题）：
        #   from_recipe = 本轮按老师配方生成；expected_difficulty = 递增计划该题预期档
        #   （跟题走，remove 位移/add 追加不会错档；编辑轮新增题无印记 → 不被旧计划误改判）。
        if knobs:
            item["from_recipe"] = True
            if increasing:
                item["expected_difficulty"] = min(md_i + idx, DIFFICULTY_CAP)
        return item

    # ── P2 流内 eager（PRD-C-012）：增量发现完整新题 → 立即 spawn 闸链 task ──
    # 调用成本不增：闸A/闸B 仍是同一批 per-item helper，只是从「流结束后串行」改成
    # 「题一完整就并发跑」（同一 Semaphore(GATE_CONCURRENCY) 限流）。宏观 DAG 零改动：
    # 产物已带 gene+check → 下游 gene_gate/solve_explain 节点天然跳过已判项。
    sem = asyncio.Semaphore(GATE_CONCURRENCY)
    eager_raw: list[dict[str, Any]] = []  # 流内稳收的规整题（生成序；merge/兜底用）
    eager_tasks: list[asyncio.Task] = []
    spawned: dict[int, dict[str, Any]] = {}  # idx → 刚解析出的题（无 check/无 tier，partial 上卡）
    completed: dict[int, dict[str, Any]] = {}  # idx → 已过闸链的题（剔除题带 _dropped 哨兵）
    # 🔴 stale 标志（对抗审修复）：流重启（中转站中途熔断换站从头重流 / 空返回 retry 二次
    # 开流）或 shape 整组 retry 采纳后，已派发的 eager 闸链全部作废 —— 置位后：①不再派发
    # 新 task；②在跑的 task 不再记账/发帧（半发帧误导 FE）；③merge 不消费（use_eager=False）。
    stale = {"flag": False}

    def _cancel_eager() -> None:
        stale["flag"] = True
        for t in eager_tasks:
            t.cancel()

    def _emit_eager_frame() -> None:
        """🔴 P2b-BE 逐题上屏（PRD-C-013）：把「出卡」与「过闸」解耦。按生成序拼累计帧——
        已过闸的题用 completed[idx]（带 tier 定状态），未过闸但已解析的题用 spawned[idx]
        （无 check → _artifact_payload 自然 tier=None，FE 按「无tier帧」先上卡）。

        🔴 稳定 seq 不重压缩（对抗审①修复）：每个 cell 带 `_seq = 题原始生成序 k+1`，整生命周期
        不变（剔除题不让后题 index 前移 → FE 按 seq 原位 merge 不嫁接到别的卡）。剔除题
        （completed[k]._dropped）**显式入帧带 `_dropped:true`**（而非 `continue` 跳过）——
        让 FE 收到显式退场哨兵驱动退场过渡 + 从 mergedItems 移除，而不是靠压缩 index 隐式挤掉
        （后者会残留永删不掉的重复卡）。FE 契约：先收该题无 tier 帧 → 带 tier 帧 →（若被判废）_dropped 帧。"""
        shown: list[dict[str, Any]] = []
        for k in sorted(spawned):
            cell = dict(completed.get(k, spawned[k]))
            cell["_seq"] = k + 1  # 稳定 merge 键：题原始生成序（1-based），剔除题不重压缩
            shown.append(cell)  # 剔除题（_dropped）照样入帧 → _artifact_payload 透传哨兵
        _emit_artifact(
            dict(state, items=shown, knobs=knobs), partial=True, expected_total=total_n
        )

    async def _eager_chain(idx: int, item: dict[str, Any]) -> None:
        """单题闸链（闸A → 闸B）+ 完成即**原位重发**该题帧（同 seq，带 tier）。任何异常静默吞
        （G5：eager 是增强不是关卡，失败的题留给下游节点照旧串行补判）。"""
        try:
            async with sem:
                # B2·T2：闸A 内涵换纯代码三检（facts 直传，judge 配方对齐 facts 组装已退役）。
                judged = await _gene_one_item(item, facts, idx, total_n)
                kept, note = await _check_one_item(judged, facts, idx, total_n)
            if stale["flag"]:
                return  # 本轮 eager 已作废（流重启/整组 retry）→ 不记账、不发作废题的帧
            if kept is None:
                # 4d 方案A：剔除题转哨兵（带 gene 不带 check）→ 下游 solve_explain 收口
                # 进 dropped_notes；不出现在增量帧（本帧起退场）。
                kept = dict(judged)
                kept["_dropped"] = note or "1 道题程序验出标答错误（重生一次仍未过），已剔除"
            completed[idx] = kept
            # 思路条进度（已过闸计数）+ 原位重发该题帧（同 seq，现带 tier 上定状态）
            done_n = len([k for k in completed if not completed[k].get("_dropped")])
            _emit_stage("verify", "程序验算", "running", f"第 {done_n}/{total_n} 道完成")
            _emit_eager_frame()
        except Exception:  # noqa: BLE001 — eager 失败绝不炸 generate；该题留给下游节点补判
            pass

    _seen = {"n": 0}
    _acc_len = {"v": 0}

    def _gen_progress(acc: str) -> None:
        # 🔴 流重启检测（对抗审修复）：中转站首站吐若干 chunk 后熔断换下一站从头重流 /
        # _ainvoke_text 空返回 retry 二次开流时，acc 从头重积（len 回落）。此时 eager_raw
        # 里是死流的题、与新流（最终 text 的事实源）下标天然错位 —— 本轮 eager 全作废
        # （cancel + 静默 + use_eager=False），题目交还下游 gene_gate/solve_explain 节点
        # 照旧补判（宏观 DAG 不变，只损失 eager 增强）。
        if len(acc) < _acc_len["v"] and not stale["flag"]:
            _cancel_eager()
        _acc_len["v"] = len(acc)
        n = min(acc.count('"stem"'), total_n)
        if n > _seen["n"]:
            _seen["n"] = n
            m = re.findall(r'"stem"\s*:\s*"([^"]{0,24})', acc)
            peek = (m[-1].replace("\\n", " ").strip() + "…") if m and m[-1] else ""
            _emit_stage("generate", "生成题目", "running", f"正在写第 {n}/{total_n} 道 {peek}")
        if stale["flag"]:
            return  # eager 已作废 → 只保留进度叙事，不再派发新闸链
        # P2：增量解析已完整闭合的新题（半截题绝不派发）→ 立即起闸链 task
        # 🔴 P2b-BE 逐题上屏：一解析出完整题（stem/answer/solution 齐）就立即上卡——记 spawned
        #   并发**无 tier 的 partial 帧**（出卡与过闸解耦）；闸链 task 跑完再原位重发带 tier 帧。
        try:
            found = _iter_complete_items(acc)
            while len(eager_raw) < len(found):
                idx = len(eager_raw)
                item = _stamp_recipe(_normalize_generated_item(found[idx], facts), idx)
                eager_raw.append(item)
                spawned[idx] = dict(item)  # 无 check → 帧里该题 tier=None（上卡先行）
                _emit_eager_frame()
                eager_tasks.append(
                    asyncio.get_running_loop().create_task(_eager_chain(idx, dict(item)))
                )
        except Exception:  # noqa: BLE001 — 进度/派发是增强不是关卡（G5），绝不打断流
            pass

    try:
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)],
            on_delta=_gen_progress,
            model=settings.variant_model("generate"),
        )
    except BaseException:
        # 🔴 孤儿收口（对抗审修复）：全中转站熔断耗尽等按设计外抛时，已 spawn 的 eager
        # task 必须 cancel + 等待退场——否则每条链最多还有 4-5 次 LLM 调用在后台静默烧完，
        # 并继续向已 error 收尾的流发帧。收口后原样 re-raise，不改失败语义。
        _cancel_eager()
        if eager_tasks:
            await asyncio.gather(*eager_tasks, return_exceptions=True)
        raise
    items = _parse_generated_items(text, facts)
    if len(items) < len(eager_raw):
        # 整体解析比流内增量还少（尾部 JSON 破损等）→ 用流内稳收的题兜底
        items = [dict(it) for it in eager_raw]

    # 🔴 代码级配方校验（数量/题型分布/递增档位）：不符 → 带缺陷反馈整组 retry 1 次。
    #   含首稿解析为空（items=[] 时「要求N道实出0道」也是明确缺陷，值得一次重试）。
    #   🔴 shape 冲突处理（PRD-C-012）：数量类缺陷只能等流结束才判；整组 retry 产出
    #   新 items 时丢弃 eager 结果（接受罕见浪费），retry 题不做流内 eager —— 由下游
    #   gene_gate/solve_explain 节点（同一批 helper + gather 并发）一次性过闸。
    defects = shape_check(items, knobs, mother_d)
    retried = False
    if defects:
        feedback = (
            "\n\n[配方校验反馈] 你上一稿不满足老师指定配方："
            + "；".join(defects)
            + "。请整组重出，严格满足配方（数量/题型配比/难度计划逐项核对后再输出）。"
        )
        try:
            retry_text = await _ainvoke_text(
                [HumanMessage(content=prompt + feedback)],
                model=settings.variant_model("generate"),
            )
            retry_items = _parse_generated_items(retry_text, facts)
        except Exception:  # noqa: BLE001 — 重试失败保留首稿（绝不卡死）
            retry_items = []
        if retry_items:
            items = retry_items
            defects = shape_check(items, knobs, mother_d)
            retried = True
            # 🔴 整组 retry 采纳 = 首稿题全部作废（对抗审修复）：cancel 仍在跑的 eager
            # 闸链（每条链最多还有 4-5 次 LLM 调用，gather 白等可达数十秒）+ 静默其
            # 后续发帧（作废题的 partial 帧只会误导 FE）。retry 失败回退首稿的分支
            # （retried=False）保持现状不取消——首稿仍是最终产物。
            _cancel_eager()

    # eager task 收口（绝不留孤儿任务；retry 整组重出/流重启时结果弃用）
    if eager_tasks:
        await asyncio.gather(*eager_tasks, return_exceptions=True)
    use_eager = bool(eager_tasks) and not retried and not stale["flag"]

    if not items:
        # 两稿皆空/解析失败 → 友好失败收尾（after_generate 走 done→END），绝不静默空轮
        _emit_stage("generate", "生成题目", "warn", "0 道（解析失败）")
        return {
            "items": [],
            "knobs": knobs,
            "shape_defects": defects,
            "llm_call_budget": budget,
            "messages": [
                AIMessage(
                    content="这一轮我没能产出可用的变式题（模型输出解析失败）。"
                    "请再发一次指令（可换种说法），或重贴题目图重试。"
                )
            ],
        }

    # 配方印记（_stamp_recipe 同一公式；eager 已在派发时落印，此处对 retry/未派发尾项
    # 落印 + 对已派发项幂等重写同值）
    if knobs:
        for i, it in enumerate(items):
            _stamp_recipe(it, i)

    if use_eager:
        # 已过闸链的题按生成序回填 + 🔴 同一性校验（对抗审修复）：completed[i] 派生自
        # eager_raw[i]（heal/补题换 stem 不影响——比的是派发时的原稿 stem），仅当它与
        # 全量解析第 i 道是同一道题（stem 一致）才采用。错位场景（模型偶发吐无 stem 的
        # 杂物对象：全量解析保留为 stem=None 而流内增量跳过，两套下标错一位）→ 保留
        # 原稿走下游节点补判（宏观 DAG 不变），绝不把验算徽章嫁接到另一道题上。
        def _same_item(i: int, it: dict[str, Any]) -> bool:
            if i >= len(eager_raw):
                return False
            key = _norm(eager_raw[i].get("stem"))
            return bool(key) and key == _norm(it.get("stem"))

        items = [
            completed[i] if (i in completed and _same_item(i, it)) else it
            for i, it in enumerate(items)
        ]

    _emit_stage("generate", "生成题目", "done", f"{len(items)} 道")
    return {
        "items": items,
        "knobs": knobs,
        "shape_defects": defects,
        # 🔴 新一组题 → 复位手排标记（对抗审③）：上一组的 manual_order 绝不泄漏到新母题/新出题轮。
        "manual_order": False,
        "llm_call_budget": budget,
        "messages": [],
    }


# 🔴 排版（PRD-C-012 任务3）：固定契约段前移，变动段（题干）移末尾；语义一字不改。
SOLVE_PROMPT = """你是严谨的数学阅卷老师。真解下面这道题（不看给定答案，独立算一遍）。

只输出 JSON：
{{"solved_answer":"你独立算出的答案","solution":"完整解题过程(含答案)",
  "kp_name":"这道题实际考的主考点","grade":"这道题适配的年级"}}

格式硬规定：solution 数学式优先行内 $...$；**仅多行分步推导**可用 $$...$$（单个等式仍用行内 $...$）。禁止裸 LaTeX / \\( \\) / \\[ \\] 定界；换行用标准 \\n。

题干：{stem}"""

REGEN_PROMPT = (
    """下面这道变式题，独立解出的答案与题面标答不一致，请**重新出一道**等价变式重做。

主考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
原题干: {stem}
要求：仍考「{kp_name}」、仍在「{grade}」、{level} 难度（目标难度档约 {difficulty}）；换数字/场景使题面与答案自洽；
answer 优先给可计算的数值/表达式（可程序验算），数字设计成解恰好整洁。

只输出 JSON：
{{"stem":"新题干","answer":"标准答案","solution":"完整解析","qtype":"{qtype}","difficulty":1~4,"level":"{level}","injected_kp":{injected_kp},
  "verify_payload":{{...新题的程序验算载荷，契约见下...}}}}
🔴 difficulty 字段（整改2·难度并入生题）：拿下面这张 rubric 对**你重出的这道新题**断一个 1~4 的难度档（不是照抄目标档，是按 rubric 实判）：

"""
    + _DIFFICULTY_RUBRIC
    + """

格式硬规定：🔴 题面(stem)与选项数学式**一律行内 $...$**，**严禁** $$...$$ / \\[ \\] / display 块级公式（撑满整行）；**仅 solution 多行分步推导**可用 $$。禁止裸 LaTeX / \\( \\) 定界；换行用标准 \\n。

"""
    + _QTYPE_CONTRACT
    + """

verify_payload 字段（PRD-C-012 4a·出题自带验算载荷：把**新题的题干 + 标准答案**抽成可被 sympy 程序验算的结构化载荷，验算对象 claimed = 新题标准答案）：
"""
    + _PAYLOAD_CONTRACT
    + """
抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。"""
)


# ---------------------------------------------------------------------------
# 闸B·程序验算（PRD-C-010）：LLM 只负责"人话题 → 结构化载荷"的有界抽取，
# pass/fail 判决只读 math_verify.verify()（纯 sympy，零 LLM）的 verdict。
# ---------------------------------------------------------------------------
# 🔴 排版（PRD-C-012 任务3）：契约固定段前移（与 GENERATE/REGEN 共用 _PAYLOAD_CONTRACT
# 单一事实源），变动段（题型/题干/标答）移末尾；语义一字不改。
EXTRACT_PROMPT = (
    """你是数学验算载荷抽取器。把下面这道题的「题干 + 题面标准答案」抽成可被 sympy 程序验算的结构化载荷 JSON（验算对象 claimed = 题面标准答案）。

"""
    + _PAYLOAD_CONTRACT
    + """

抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ 只输出 {{"kind":"none","reason":"原因"}}。
只输出 JSON（不要解释）。

题型: {qtype}
题干: {stem}
题面标准答案(待验算的 claimed): {answer}
参考·另一次独立解答(仅帮助你理解答案格式，不是验算对象): {solved_answer}"""
)

# 🔴 与 math_verify._HANDLERS 的 kind 名保持一致（PRD-C-013 4b 扩面：+inequality_solve / rational_roots）。
_PAYLOAD_KINDS = {
    "equation_solve",
    "expr_equiv",
    "numeric",
    "choice",
    "inequality_solve",
    "rational_roots",
}

# 题型分流（分流键=题型；题干骨架词兜底）：证明/开放/作图类不进 sympy
_PROOF_QTYPE_RE = re.compile(r"(证明|求证|开放|作图|画图)")
_PROOF_STEM_RE = re.compile(r"(求证|请证明|证明[:：]|尺规作图)")
# 软校验骨架（代码正则级）：证明类题面应含"已知/求证/证明/作图"类结构
_PROOF_SKELETON_RE = re.compile(r"(已知|求证|证明|作图)")


def _is_proof_like(qtype: Any, stem: Any) -> bool:
    """闸B 分流键：题型含 证明/开放/作图 或 题干带求证骨架 → 不进 sympy，走软校验+人审。"""
    return bool(
        _PROOF_QTYPE_RE.search(str(qtype or "")) or _PROOF_STEM_RE.search(str(stem or ""))
    )


def _proof_struct_ok(stem: Any) -> bool:
    """轻量结构软校验：题面是否含「已知/求证/证明」类骨架（不验数学，只验结构）。"""
    s = str(stem or "")
    return len(s) >= 10 and bool(_PROOF_SKELETON_RE.search(s))


def _append_card_note(item: dict, note: str) -> None:
    """把验算标记行追加到题卡可见文本字段(solution)尾部 → UI 渲染 + 入库 analyze 同步可见。"""
    sol = str(item.get("solution") or "").rstrip()
    item["solution"] = f"{sol}\n\n> {note}" if sol else f"> {note}"


def _format_item_stem(item: dict) -> None:
    """🔴 题型模版自动规范（PRD-C-009·BE 镜像 normalize.ts）：就地把 item.stem 跑一遍
    format_by_qtype（选项分行 / 填空 ____ / 判断补括号），规范文本落进 item。

    🔴 落点铁律：必须在 sympy/structure_lint **判决之后**调用——规范是纯排版（可能动选项
    行序/下划线/补括号），绝不能影响判决（判决先跑、规范后做）。cosmetic-only、幂等、
    解析不出原样、永不抛（format_by_qtype 自带 G5 降级）。规范后 item.stem 即 canonical，
    上屏（_artifact_payload 读 stem）/ 入库（build_create_bo 读 stem）/ 会话恢复都吃规范文本。
    """
    stem = item.get("stem")
    if isinstance(stem, str) and stem:
        item["stem"] = format_by_qtype(stem, item.get("qtype"))


_TIER_NOTES = {
    TIER_VERIFIED: NOTE_VERIFIED_OK,
    TIER_SELF_OK: NOTE_SELF_CHECK_OK,
    TIER_PROOF: NOTE_PROOF_REVIEW,
    TIER_BOTH_LOW: NOTE_BOTH_GATES_LOW,
    # TIER_SILENT 无 note（沉默 = 不说话）
}


def _apply_visibility(item: dict) -> None:
    """4d 可见性矩阵（PRD-C-012）：check×gene 真值 → 外显 tier + badge + 题卡 note。

    🔴 只定展示层，真值不动（verify/gene 原样进 aux_tags 审计）。幂等：note 已在
    solution 里不重复追加。⚠ 仅 TIER_BOTH_LOW（双闸皆存疑）；单闸存疑 = 沉默。
    """
    chk = item.get("check") or {}
    gene_low = (item.get("gene") or {}).get("gate") == GENE_GATE_WARN
    if chk.get("review") == REVIEW_PROOF:
        # 证明类：badge=warn 表示结构软校验缺骨架（verify 侧低）
        struct_low = chk.get("badge") == "warn"
        tier = TIER_BOTH_LOW if (struct_low and gene_low) else TIER_PROOF
    elif chk.get("verify") == VERIFY_SYMPY_PASS:
        tier = TIER_VERIFIED  # 强正面（gene 即便 warn 也不外显负面——单闸沉默）
    elif chk.get("self_check") == "match":
        tier = TIER_SELF_OK  # 程序不可验但独立复算一致 → 轻正面
    else:
        # verify 侧低（unverified 且自检不一致）：gene 也低才 ⚠，否则沉默
        tier = TIER_BOTH_LOW if gene_low else TIER_SILENT
    chk["tier"] = tier
    chk["badge"] = "warn" if tier == TIER_BOTH_LOW else "ok"
    item["check"] = chk
    note = _TIER_NOTES.get(tier)
    if note and note not in str(item.get("solution") or ""):
        _append_card_note(item, note)


async def _extract_payload(
    stem: Any, answer: Any, solved_answer: Any, qtype: Any
) -> dict | None:
    """LLM 有界抽取：题干+标答 → 验算载荷 JSON。解析失败带错误反馈 retry，总共最多 2 次。

    返回 None = 抽不成/LLM 异常 → 调用方按 degrade 处理（绝不外抛，G5）。
    """
    base = EXTRACT_PROMPT.format(
        qtype=str(qtype or "解答"),
        stem=str(stem or ""),
        answer=str(answer or ""),
        solved_answer=str(solved_answer or "(无)"),
    )
    feedback = ""
    for _ in range(2):
        try:
            text = await _ainvoke_text(
                [HumanMessage(content=base + feedback)], model=settings.variant_model("solve")
            )
        except Exception:  # noqa: BLE001 — 抽取层异常不许逃逸炸 solve_explain → degrade
            return None
        data = _parse_json(text)
        if isinstance(data, dict):
            kind = data.get("kind")
            if kind in _PAYLOAD_KINDS:
                return data
            if kind == "none":
                return None  # LLM 明确说抽不成 → degrade，不浪费重试
        feedback = (
            "\n\n[错误反馈] 上次输出不是合法载荷 JSON（kind 必须是 "
            "equation_solve/expr_equiv/numeric/choice/inequality_solve/rational_roots/none 之一，"
            "且为合法 JSON）。"
            "请严格按契约重新只输出 JSON。\n上次输出(截断)：" + (text or "")[:300]
        )
    return None


def _payload_claim_consistent(payload: dict, answer: Any) -> bool:
    """🔴 4a 载荷一致性廉价闸（对抗审修复；纯代码非 LLM）：被验对象(claimed) 必须就是
    题面标答——防止模型在同一次调用里把 claimed 写成与 answer 不同的值（或编一套与
    题干无关但自洽的 equations+claimed），sympy PASS 后强正面徽章背书的却不是老师
    看到的那个答案。宽松互含比对（_norm 去空白小写）：
    - numeric：claimed 与标答互含/相等；
    - equation_solve：claimed 各解都出现在标答里，或标答含于 claimed 拼接；
    - choice：claimed_correct 选项字母在标答里，或该选项的值与标答互含；
    - expr_equiv：无 claimed（被验对象=expr_b 结果式，难与人话标答廉价比对）→ 豁免。
    判不一致 → 调用方退回 _extract_payload 兜底（不动判决语义，只收紧
    『被验对象=被展示对象』的绑定）。"""
    kind = payload.get("kind")
    ans = _norm(answer)
    if not ans:
        return True  # 没有展示答案可比 → 不拦（兜底抽取同样无从比）
    if kind == "numeric":
        c = _norm(payload.get("claimed"))
        return bool(c) and (c in ans or ans in c)
    if kind in ("equation_solve", "rational_roots"):
        claimed = payload.get("claimed")
        vals = [_norm(v) for v in claimed] if isinstance(claimed, (list, tuple)) else [_norm(claimed)]
        vals = [v for v in vals if v]
        return bool(vals) and (all(v in ans for v in vals) or ans in ",".join(vals))
    if kind == "inequality_solve":
        # claimed = 解集关系串（如 "x>2"）；标答常含同一关系串/解集描述 → 宽松互含。
        c = _norm(payload.get("claimed"))
        return bool(c) and (c in ans or ans in c)
    if kind == "choice":
        key = _norm(payload.get("claimed_correct"))
        if not key:
            return False
        if key in ans or ans in key:
            return True  # 标答就是选项字母（最常见形态）
        opts = payload.get("options")
        if isinstance(opts, dict):
            for k, v in opts.items():
                if _norm(k) == key:
                    nv = _norm(v)
                    return bool(nv) and (nv in ans or ans in nv)
        return False
    return True  # expr_equiv 等无 claimed 的 kind → 豁免


async def _machine_verify(item: dict, solved_answer: Any) -> dict:
    """程序验算一道题：载荷优先 → 事后抽取兜底 → math_verify.verify（纯 sympy）。永不抛异常。

    🔴 4a 载荷优先（PRD-C-012）：item.verify_payload（出题 LLM 同步产出）是 dict 且
    kind 合法且 claimed 与题面标答一致（_payload_claim_consistent 廉价闸）→ 直接验，
    省一次抽取往返；kind=="none"/缺失/非法/claimed 不一致 → 退回既有
    _extract_payload 事后抽取兜底。判决语义零变化（仍只读 verify() 的 verdict）。
    返回 {"verdict": "pass"|"fail"|"degrade", "detail": str, "computed": str|None}。
    """
    payload: Any = item.get("verify_payload")
    if not (
        isinstance(payload, dict)
        and payload.get("kind") in _PAYLOAD_KINDS
        and _payload_claim_consistent(payload, item.get("answer"))
    ):
        # 🔴 P13：事后抽取载荷是**增强类**调用（载荷优先的兜底），预算耗尽 → 跳过抽取，
        #   按 degrade 降级（sympy 吃不下 → 退回 LLM 自检 fallback，语义与抽取失败一致，不卡死）。
        if _budget_exhausted():
            return {"verdict": math_verify.DEGRADE, "detail": "预算耗尽，跳过载荷抽取", "computed": None}
        payload = await _extract_payload(
            item.get("stem"), item.get("answer"), solved_answer, item.get("qtype")
        )
    if payload is None:
        return {"verdict": math_verify.DEGRADE, "detail": "载荷抽取失败/抽不成", "computed": None}
    try:
        # sympy solve/simplify 偶有耗时 → 丢线程池，不卡事件循环；
        # 🔴 wait_for 墙钟预算（G5 反挂死）：病态载荷把 sympy 拖入无界计算时按 degrade 解锁流程
        return await asyncio.wait_for(
            asyncio.to_thread(math_verify.verify, payload), timeout=VERIFY_TIMEOUT_S
        )
    except TimeoutError:  # py3.11: asyncio.TimeoutError == TimeoutError
        return {
            "verdict": math_verify.DEGRADE,
            "detail": f"验算超时（>{VERIFY_TIMEOUT_S}s），按未验算降级",
            "computed": None,
        }
    except Exception as e:  # noqa: BLE001 — verify 自身永不抛，此处纯保险
        return {"verdict": math_verify.DEGRADE, "detail": f"验算执行异常: {e}", "computed": None}


# ---------------------------------------------------------------------------
# 🔴 PRD-C-015 批3·⑦ 反退化代码闸（纯代数·零 LLM）：变式标答的最优解驻点落动点区间端点
#   （退化构型，机制失效但答案碰巧对）→ 判退化废题 → 上层 REGEN（≤MAX_DEGEN_REGEN，超限弃）。
#   判决只读 math_verify.check_endpoint_degeneracy（纯代数）返回值，**绝不采信 LLM 自评**（铁律）。
#   闸必有降级路径：无 degen_payload / 抽不成 / sympy 算不了 → 视作「不可判·放行」（不卡死，不误杀）。
# ---------------------------------------------------------------------------
async def _degeneracy_verdict(item: dict) -> dict:
    """跑反退化闸：返回 math_verify.check_endpoint_degeneracy 的结果。永不抛。

    🔴 触发条件 = item 带 degen_payload（kind=endpoint_extremum，出题 LLM 对最值/动点构型同步产出）。
    无 degen_payload / 非该 kind → degrade（= 不适用·放行，绝不误判退化）。
    judge 只读代数返回值，零 LLM。sympy 偶有耗时 → 丢线程池 + 墙钟预算（同 _machine_verify）。
    """
    payload = item.get("degen_payload")
    if not (isinstance(payload, dict) and payload.get("kind") == "endpoint_extremum"):
        return {"verdict": math_verify.DEGRADE, "detail": "无反退化载荷（不适用·放行）", "computed": None}
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(math_verify.check_endpoint_degeneracy, payload),
            timeout=VERIFY_TIMEOUT_S,
        )
    except TimeoutError:
        return {"verdict": math_verify.DEGRADE, "detail": f"反退化判定超时（>{VERIFY_TIMEOUT_S}s），降级放行", "computed": None}
    except Exception as e:  # noqa: BLE001 — check_endpoint_degeneracy 本身永不抛，此处纯保险
        return {"verdict": math_verify.DEGRADE, "detail": f"反退化判定异常: {e}", "computed": None}


async def _anti_degen_gate(
    item: dict, facts: dict, idx: int, total: int
) -> tuple[dict, bool]:
    """⑦ 反退化闸（闸B 内·sympy 答案 PASS 之后跑）。返回 (item, dropped)。

    退化（最优点落区间端点）→ REGEN ≤MAX_DEGEN_REGEN 次；重生稿须**非退化 + sympy 重 PASS**才采纳，
    否则超限**弃该变式**（dropped=True，§1⑦「超限则弃」）。非退化 / 不可判（降级）→ 原样放行。
    🔴 判决只读代数返回值；from_edit（老师点名）题不弃不换（老师意志优先，标 ⚠ 注记放行）。
    🔴 预算耗尽 → 跳过 REGEN，标 ⚠ 放行（降级路径，不卡死）。
    """
    res = await _degeneracy_verdict(item)
    if res.get("verdict") != math_verify.DEGENERATE:
        return item, False  # 非退化 / 不可判（degrade）→ 放行（闸门降级路径）

    # 已确认退化构型：老师点名编辑题不动（老师意志优先，仅标注交人审）。
    if item.get("from_edit"):
        _append_card_note(item, "⚠ 反退化闸：最优点落动点区间端点（退化构型），已保留老师原题交人审")
        return item, False

    attempt = 0
    cur = item
    while attempt < MAX_DEGEN_REGEN and not _budget_exhausted():
        attempt += 1
        _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道最优点退化到端点，回炉重生中")
        feedback = (
            f"程序(代数)判定该题为**退化构型**：{res.get('detail')}。"
            "请重新出一道题面与标答自洽的等价变式，务必让最优解（最值取得点）落在**动点定义区间的内部驻点**，"
            "严禁让最优点落在区间端点（如动点与定点重合 / 取临界值使解法机制失效）。"
        )
        draft = await _regen_once(cur, facts, feedback=feedback)
        if not draft:
            break
        # 重生稿须：① 非退化 ② sympy 答案重 PASS（保证答案仍对）→ 才采纳。
        d_res = await _degeneracy_verdict(draft)
        if d_res.get("verdict") == math_verify.DEGENERATE:
            cur = draft  # 仍退化 → 继续重生（受 MAX 上限约束）
            res = d_res
            continue
        resolved = await _solve_one(draft.get("stem", ""))
        if resolved.get("solution"):
            draft["solution"] = resolved.get("solution")
        r_res = await _machine_verify(draft, resolved.get("solved_answer"))
        if r_res.get("verdict") == math_verify.PASS:
            # 采纳非退化重生稿：保留闸A gene 印记，回写 sympy 验算 check。
            draft["gene"] = item.get("gene") or {"gate": GENE_GATE_SKIPPED, "reason": "degen-regen"}
            draft["check"] = {
                "badge": "ok",
                "solved_answer": resolved.get("solved_answer"),
                "verify": VERIFY_SYMPY_PASS,
                "verify_detail": r_res.get("detail"),
                "computed": r_res.get("computed"),
            }
            _apply_visibility(draft)
            return draft, False
        cur = draft  # 重生稿非退化但答案没过 → 视同失败，继续（受 MAX 上限约束）

    # 超限仍退化 / 重生失败 / 预算耗尽 → 弃该变式（§1⑦「超限则弃」）。
    dropped = dict(cur)
    dropped["_dropped"] = "1 道题为退化构型（最优点落动点区间端点，重生仍未脱退化），已剔除"
    return dropped, True


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


def _conservation_ok(solved_kp: str, solved_grade: str, facts: dict) -> bool:
    """主考点 + 年级 守恒校验（重生版仍须过；破则丢弃重生保留原版打⚠）。

    宽松包含匹配：标准考点/年级名 与 solve 回报的实际考点/年级 互含即视为守恒。
    """
    mk, mg = _norm(facts["kp_name"]), _norm(facts["grade"])
    sk, sg = _norm(solved_kp), _norm(solved_grade)
    kp_ok = (not sk) or (mk in sk) or (sk in mk) or (mk[:4] and mk[:4] in sk)
    # 年级守恒：比对编码前2字(如"七年")或互含
    grade_ok = (not sg) or (mg[:2] and mg[:2] in sg) or (mg in sg) or (sg in mg)
    return bool(kp_ok and grade_ok)


# ---------------------------------------------------------------------------
# 🔴 表皮闸纯函数（PRD-C-014 B2·T4）：闸A 三检之②——题干过近 / 数字全同 = 抄母题。
# 纯函数零 LLM 零 IO，可单测；语义搬自 tools/e2_constrained_gen_probe._surface_check。
# 降级铁律：只打 ⚠ flag（返回缺陷描述），不卡死、不剔题——上层据此标记继续（闸门必有降级路径）。
# ---------------------------------------------------------------------------
# 🔴 表皮相似度阈值按题型分（PRD-C-014 AC4「扩样集扫一遍再定死」落锤，依据 tools/c018_result_v2.json
#   §G4_surface_by_qtype）：
#   - 选择/填空：扩样 n=4 max=0.577，0.85 裕量大 → 保持 0.85。
#   - 解答（证明经 _QTYPE_ALIAS 归一进解答）：长多小问共享题干脚手架系统性误警（n=27 max=0.889），
#     放宽到 0.92（实测 max 0.889 过且有余量，仍能抓 >0.92 的真复读）。
_SURFACE_SIM_THRESHOLD = 0.85  # 默认/选择/填空：题干归一化相似度 > 此阈值 = 疑似抄题
_SURFACE_SIM_THRESHOLD_BY_QTYPE: dict[str, float] = {
    "选择": 0.85,
    "填空": 0.85,
    "解答": 0.92,  # 解答/证明长题共享脚手架放宽
}


def _surface_threshold_for_qtype(qtype: Any) -> float:
    """按题型取表皮相似度阈值（题型先过 _QTYPE_ALIAS 归一，证明→解答）。未知题型回默认 0.85。"""
    qt = _QTYPE_ALIAS.get(str(qtype or "").strip(), str(qtype or "").strip())
    return _SURFACE_SIM_THRESHOLD_BY_QTYPE.get(qt, _SURFACE_SIM_THRESHOLD)


def _surface_norm_stem(s: Any) -> str:
    """题干归一化：去空白/LaTeX 定界符/花括号/反斜杠后小写，专供相似度比对（不入库）。"""
    return re.sub(r"[\s$\\{}]+", "", str(s or "")).lower()


def _surface_nums(s: Any) -> list[str]:
    """抽题干里的数字串（含小数），用于「数字与母题全同」检测。"""
    return re.findall(r"\d+(?:\.\d+)?", str(s or ""))


def _surface_check(variant_stem: Any, mother_stem: Any, qtype: Any = None) -> str | None:
    """🔴 表皮距离纯函数（T4 + AC4 按题型定阈值）：题干过近 = 抄母题；数字与母题完全相同也算。

    返回缺陷描述（疑似抄题，上层打 ⚠ flag）或 None（正常变式，表皮已换）。
    阈值按 qtype 取（选择/填空=0.85；解答/证明=0.92），未传或未知 qtype 回默认 0.85。
    母题题干为空 → 无从比对 → None（不误报）。纯函数，可单测。
    """
    m = _surface_norm_stem(mother_stem)
    if not m:
        return None
    v = _surface_norm_stem(variant_stem)
    ratio = difflib.SequenceMatcher(None, v, m).ratio()
    thr = _surface_threshold_for_qtype(qtype)
    if ratio > thr:
        return f"题干与母题相似度{ratio:.2f}>{thr}（疑似抄题，表皮未换）"
    v_nums, m_nums = _surface_nums(variant_stem), _surface_nums(mother_stem)
    if v_nums and v_nums == m_nums:
        return "数字与母题完全相同（表皮没换）"
    return None


async def _solve_one(stem: str) -> dict:
    """真解一道题。🔴 LLM 调用异常吞掉返 {}（与 _extract_payload/_gene_judge_one 契约对齐）：
    瞬时网关抖动绝不外抛炸掉 solve_explain/gene_gate 节点（G5），调用方按"没解出来"降级。"""
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=SOLVE_PROMPT.format(stem=stem or ""))],
            model=settings.variant_model("solve"),
        )
    except Exception:  # noqa: BLE001
        return {}
    solved = _parse_json(text) or {}
    # 🔴 阅卷解析会回写 item["solution"]（solve_explain 三处）—— 出口统一净化，
    # 否则 \( \) / 字面 \n 绕过 _parse_generated_items 的净化直达卡片/入库
    if isinstance(solved, dict) and solved.get("solution"):
        solved["solution"] = _sanitize_rich_text(solved["solution"])
    return solved


async def _regen_once(item: dict, facts: dict, feedback: str | None = None) -> dict | None:
    """REGEN 回炉一次：返回不带 check 的重生草稿（解析失败/LLM 异常返 None）。

    feedback（如 sympy 的 computed/detail，或闸A 结构定向反馈）注回 prompt 的题干段。
    🔴 RC2（PRD-C-013）：item 带 edit_note（老师点名改造的软约束）→ 永远把老师 note 注回
    prompt，feedback 不能把它冲掉（基因/验算失败原因与老师意志并存，老师意志优先）。
    🔴 LLM 调用异常吞掉返 None（G5）：回炉是增强不是关卡，网关抖动时调用方按
    "重生失败 → 保留原版打 ⚠/warn" 的既有降级路径走，绝不炸掉整轮出题。
    """
    stem = str(item.get("stem") or "")
    edit_note = str(item.get("edit_note") or "").strip()
    if edit_note:
        stem = f"{stem}\n\n[老师要求·须保留] {edit_note}"
    if feedback:
        stem = f"{stem}\n\n[程序验算反馈] {feedback}"
    try:
        regen_text = await _ainvoke_text(
            [
                HumanMessage(
                    content=REGEN_PROMPT.format(
                        kp_name=facts["kp_name"],
                        grade=facts["grade"],
                        stem=stem,
                        level=item.get("level") or "normal",
                        qtype=item.get("qtype") or facts["qtype"],
                        difficulty=item.get("difficulty") or 3,
                        injected_kp=json.dumps(item.get("injected_kp"), ensure_ascii=False),
                    )
                    # 🔴 整改1：确定上下文硬约束（回炉重出同样压解题不越界）。
                    + "\n\n"
                    + _context_block(facts)
                    # 🔴 W2 守恒硬约束注入（T1）：回炉重出仍守白名单/考察类型/最难步基因。
                    + "\n\n"
                    + _conservation_clause(facts.get("dna"))
                    # 🔴 批3·W2' 注卡（回炉重出同样照模型卡片，难度≥3+非M00 才注，含反退化约束）。
                    + _maybe_note_card_block(facts)
                )
            ],
            # 🔴 整改4：回炉瘦身 max_tokens 上限压输出失控（出题主调用 generate/add 不动）。
            max_tokens=_regen_max_tokens(),
            # 🔴 回炉 = 出题环节（红线不降档）：走 generate 档（默认 COMPATIBLE_MODEL）。
            model=settings.variant_model("generate"),
        )
    except Exception:  # noqa: BLE001 — 回炉 LLM 异常 → 视同重生失败（G5）
        return None
    regen = _parse_json(regen_text)
    if isinstance(regen, dict) and regen.get("stem"):
        draft = _sanitize_item(
            {
                "stem": regen.get("stem"),
                "answer": regen.get("answer"),
                "solution": regen.get("solution"),
                "qtype": regen.get("qtype") or item.get("qtype"),
                "difficulty": regen.get("difficulty") or item.get("difficulty"),
                "level": regen.get("level") or item.get("level"),
                "injected_kp": regen.get("injected_kp"),
            }
        )
        # 4a：重生稿自带的验算载荷随新题走（旧题载荷绝不沿用——题面已换）
        if isinstance(regen.get("verify_payload"), dict):
            draft["verify_payload"] = regen["verify_payload"]
        # 🔴 批3·⑦：重生稿自带的反退化载荷随新题走（旧题 degen 载荷绝不沿用，题面已换）。
        if isinstance(regen.get("degen_payload"), dict):
            draft["degen_payload"] = regen["degen_payload"]
        # 🔴 RC2：老师 note + 编辑轮印记跟草稿走，下一次 REGEN 仍能注回老师意志、仍只判不回炉
        for k in ("edit_note", "from_edit"):
            if item.get(k):
                draft[k] = item[k]
        return draft
    return None


async def _check_one_item(
    item: dict[str, Any], facts: dict, idx: int, total: int
) -> tuple[dict[str, Any] | None, str | None]:
    """🔴 闸B per-item 协程（P2 单一事实源：solve_explain 节点与 generate 流内 eager 共用）。

    入参 item 视为本协程私有（调用方传副本）；idx/total 只用于思路条叙事编号。
    返回 (item, None)=保留（已带 check + 外显 tier）；(None, dropped 叙事)=4d 方案A 剔除。
    已带 check 的题原样通过（不重判、不发 stage）。判决/降级语义与重排前逐字一致：
    - 证明/开放/作图类 → 不进 sympy，软校验 + 人审标记；
    - sympy pass → sympy_pass；fail → REGEN 回炉 1 次（须 pass+守恒）→ 仍不过剔除；
    - degrade → LLM 独立解自检 fallback（match/mismatch + 自愈 1 次）。
    🔴 判决只读 verify() 的 verdict，永不采信 LLM 自评；任何环节失败降级继续，绝不抛（G5）。
    """
    if item.get("check"):  # 已定状态（如自愈过的补题再次流经）→ 不重复
        return item, None

    # ── 题型结构 lint（P11.3）：先于数学验算，按 qtype 校验形态（选择题别长成多小问
    # 嵌合体/选项不足/标答非字母；填空缺空位）。命中缺陷 → 走既有题级 REGEN 回炉 1 次
    # （与下游 sympy 回炉同一通道，不是整组重试）；回炉清掉缺陷 → 顶位继续往下验算。
    # 🔴 降级（铁律④）：lint 返 []（解析不了/合规）不拦；回炉仍不过 → 标 ⚠ 注记并继续
    # 进 sympy（绝不卡死、不剔除，结构存疑交给老师人审）。
    struct_defects = structure_lint(item)
    if struct_defects:
        # 🔴 P13 预算闸：结构回炉是增强类调用，预算耗尽 → 跳过回炉，直接标 ⚠ 注记继续走
        #   sympy（既有降级路径，绝不卡死）。
        # 🔴 RC2「老师意志优先」补齐结构闸（题组编辑器 reverify 修复）：from_edit 题（老师手动
        #   编辑/点名改造）即便结构 lint 不过也**绝不回炉换题**——回炉会用模型重出覆盖老师改的
        #   内容（reverify 真机实锤：手改题干→结构不匹配旧 qtype→被静默换成另一道题）。只标 ⚠
        #   注记、保留老师原题继续验算，交人审。与下游 FAIL/degrade 的 from_edit 短路语义一致。
        if item.get("from_edit") or _budget_exhausted():
            item["structure_lint"] = {"badge": "warn", "defects": struct_defects}
        else:
            _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道结构不合题型，回炉重生中")
            feedback = (
                "本题结构不符合其题型契约：" + "；".join(struct_defects) + "。"
                "请严格按题型结构契约重新出一道同考点同年级的变式（选择题=单一设问+恰 4 选项+"
                "answer 为选项字母，禁止多小问；填空题题干须含空位 ____）。"
            )
            draft = await _regen_once(item, facts, feedback=feedback)
            if draft and not structure_lint(draft):
                # 回炉稿结构合规 → 顶位（保留闸A gene 印记，下游 sympy 照常验答案）
                draft["gene"] = item.get("gene")
                item = draft
            else:
                # 回炉失败/仍不合规 → 降级：item 保持原样继续往下走 sympy，记结构存疑注记。
                item["structure_lint"] = {"badge": "warn", "defects": struct_defects}
        # 断言：到此处 item 要么结构合规、要么已记 warn 注记（绝不卡死）。
    else:
        # 结构合规留痕（审计：本题进过结构闸且通过）
        item["structure_lint"] = {"badge": "ok", "defects": []}

    # 🔴 P14 叙事修正（RC3·PRD-C-013）：per-item 闸并发跑，旧「第 N/total 道」会让老师误读成
    #   顺序进度（实际三题同时验）。改为「第 N 题验算中」——只点本题号，不带误导性的「/总数」。
    #   编辑轮（from_edit 单题重验）明示「只重验第 N 题」——别让老师把题号读成「全部重验」。
    _emit_stage(
        "verify", "程序验算", "running",
        (f"只重验第 {idx + 1} 题" if item.get("from_edit") else f"第 {idx + 1} 题验算中"),
    )

    # ── 闸B·题型分流：证明/开放/作图 → 不进 sympy，软校验 + 人审标记 ──
    if _is_proof_like(item.get("qtype") or facts["qtype"], item.get("stem")):
        struct_ok = _proof_struct_ok(item.get("stem"))
        item["check"] = {
            "badge": "ok" if struct_ok else "warn",
            "solved_answer": None,
            "review": REVIEW_PROOF,
        }
        _apply_visibility(item)
        return item, None

    solved = await _solve_one(item.get("stem", ""))
    solved_answer = solved.get("solved_answer")
    # solve 产出解析（给老师当判题依据；优先用阅卷解析）
    if solved.get("solution"):
        item["solution"] = solved.get("solution")

    # ── 闸B·程序验算：判决只读 verdict（G1/G2），不再用字符串比对自判 ──
    res = await _machine_verify(item, solved_answer)
    verdict = res.get("verdict")

    if verdict == math_verify.PASS:
        item["check"] = {
            "badge": "ok",
            "solved_answer": solved_answer,
            "verify": VERIFY_SYMPY_PASS,
            "verify_detail": res.get("detail"),
            "computed": res.get("computed"),
        }
        # 🔴 批3·⑦ 反退化闸：答案 sympy PASS ≠ 构型不退化（最优点落区间端点 = 答案碰巧对但机制失效）。
        #   退化 → REGEN（≤MAX_DEGEN_REGEN，超限弃）；非退化/不可判 → 原样放行。判决只读代数零 LLM。
        item, degen_dropped = await _anti_degen_gate(item, facts, idx, total)
        if degen_dropped:
            return None, str(item.get("_dropped") or "退化构型已剔除")
        _apply_visibility(item)
        return item, None

    if verdict == math_verify.FAIL:
        # 🔴 RC2「老师意志优先」补齐闸B（对抗审②修复）：编辑轮老师点名改造的题（item.from_edit，
        #   带 edit_note）即便 sympy 判 FAIL 也**不回炉/不换题/不剔除**——保留老师编辑的原题、
        #   标 ⚠ 注记（verify=fail_after_regen，走 4d both_low/silent 外显）交老师人审。否则
        #   回炉换题会用模型重出覆盖老师意志、剔除路径连 edit_note 一起丢 = 隐性数据丢失，与
        #   闸A from_edit 短路语义矛盾。降级不抛（G5）。
        if item.get("from_edit"):
            item["check"] = {
                "badge": "warn",
                "solved_answer": solved_answer,
                "verify": VERIFY_FAIL_AFTER_REGEN,
                "verify_detail": res.get("detail"),
                "computed": res.get("computed"),
            }
            _apply_visibility(item)
            return item, None
        # sympy 判定标答真错 → 既有回炉机制重生 1 次，computed/detail 注回 prompt
        # 🔴 P13 预算闸：heal/replenish 都是增强类调用，预算耗尽 → 跳过，直接走「剔除不外发」
        #   降级（4d 方案A，本组少一道 dropped 叙事）。绝不卡死、绝不抛。
        healed = None
        if MAX_HEAL >= 1 and not _budget_exhausted():
            _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道回炉重生中")
            feedback = (
                f"程序(sympy)验算判定该题题面标答错误：程序算得 computed={res.get('computed')}；"
                f"详情：{res.get('detail')}。请重新出一道题面与标答自洽、经得起程序验算的等价变式。"
            )
            draft = await _regen_once(item, facts, feedback=feedback)
            if draft:
                resolved = await _solve_one(draft.get("stem", ""))
                if resolved.get("solution"):
                    draft["solution"] = resolved.get("solution")
                # 🔴 重生版仍须过守恒校验 + 程序验算双闸
                cons = _conservation_ok(
                    resolved.get("kp_name", ""), resolved.get("grade", ""), facts
                )
                r_res = await _machine_verify(draft, resolved.get("solved_answer"))
                if cons and r_res.get("verdict") == math_verify.PASS:
                    draft["check"] = {
                        "badge": "ok",
                        "solved_answer": resolved.get("solved_answer"),
                        "verify": VERIFY_SYMPY_PASS,
                        "verify_detail": r_res.get("detail"),
                        "computed": r_res.get("computed"),
                    }
                    # 🔴 闸A 标记随愈合保留（healed 整体替换不丢 gene → 入库 auxTags.gene_gate
                    # 不断档 + 后续编辑轮 gene_gate 不重判/不静默换题）；原版无 gene（如持久化
                    # 旧线程存量题）→ 按既有 skipped 语义留痕（闸A 没判过，REGEN 锁同骨架）。
                    draft["gene"] = item.get("gene") or {
                        "gate": GENE_GATE_SKIPPED,
                        "reason": "healed-in-solve",
                    }
                    healed = draft
        if healed:
            _apply_visibility(healed)
            return healed, None
        # 🔴 整改4（2026-06-12·闸B 回炉松绑·维护者拍板「不能写得这么死」）：
        #   回炉 1 次仍 FAIL → **不再二次回炉、不再补题、不再剔除**——直接标 ⚠（verify=
        #   fail_after_regen，走 4d both_low/silent 外显矩阵）放行，交老师人审。
        #   （铁律「闸门必有降级路径、绝不卡死」本就在；旧 4d 剔除+补题路径单次回炉 95-134s
        #   且输出失控，松绑为「标 ⚠ 放行」减少一整条回炉链。判决仍只读 sympy，不采信 LLM 自评。）
        _emit_stage(
            "verify", "程序验算", "warn",
            f"第 {idx + 1} 道程序验算未过（回炉一次仍未过），已标注存疑交人审",
        )
        item["check"] = {
            "badge": "warn",
            "solved_answer": solved_answer,
            "verify": VERIFY_FAIL_AFTER_REGEN,
            "verify_detail": res.get("detail"),
            "computed": res.get("computed"),
        }
        _apply_visibility(item)
        return item, None

    # ── degrade：sympy 吃不下（载荷抽不成/超范围）→ 保留既有 LLM 自检 fallback ──
    match = _norm(solved_answer) == _norm(item.get("answer"))
    if match:
        item["check"] = {
            "badge": "ok",
            "solved_answer": solved_answer,
            "verify": VERIFY_UNVERIFIED,
            "verify_detail": res.get("detail"),
            "self_check": "match",  # 4d：独立复算一致 → 轻正面（不再打 ⚠ 未经程序验算）
        }
        _apply_visibility(item)
        return item, None

    # 独立解 ≠ 标答（LLM 自检）→ 既有自愈：重生 1 次 → 重解 + 守恒
    # 🔴 P13：degrade 自愈也是增强类调用，预算耗尽 → 跳过自愈，落下方 warn 保留（不抛）。
    # 🔴 RC2「老师意志优先」补齐 degrade 支（题组编辑器 reverify 修复）：from_edit 题不自愈换题，
    #   直接落下方 warn 保留老师原题（与结构闸/FAIL 支一致——reverify 只验不换）。
    healed = None
    if MAX_HEAL >= 1 and not _budget_exhausted() and not item.get("from_edit"):
        _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道回炉重生中")
        draft = await _regen_once(item, facts)
        if draft:
            resolved = await _solve_one(draft.get("stem", ""))
            r_answer = resolved.get("solved_answer")
            r_match = _norm(r_answer) == _norm(draft.get("answer"))
            # 🔴 重生版仍须过守恒校验
            cons = _conservation_ok(
                resolved.get("kp_name", ""), resolved.get("grade", ""), facts
            )
            if r_match and cons:
                draft["solution"] = resolved.get("solution") or draft.get("solution")
                draft["check"] = {
                    "badge": "ok",
                    "solved_answer": r_answer,
                    "verify": VERIFY_UNVERIFIED,
                    "self_check": "match",
                }
                # 🔴 闸A 标记随愈合保留（同 FAIL 自愈路径：不丢 gene、不被编辑轮重判）
                draft["gene"] = item.get("gene") or {
                    "gate": GENE_GATE_SKIPPED,
                    "reason": "healed-in-solve",
                }
                _apply_visibility(draft)
                healed = draft

    if healed:
        return healed, None
    # 守恒破 或 重生仍不过 → 保留（程序没证明它错，只是没把握）。
    # 4d：verify 侧低 → 单闸沉默 / gene 也低 → ⚠（_apply_visibility 矩阵裁决）
    item["check"] = {
        "badge": "warn",
        "solved_answer": solved_answer,
        "verify": VERIFY_UNVERIFIED,
        "verify_detail": res.get("detail"),
        "self_check": "mismatch",
    }
    _apply_visibility(item)
    return item, None


async def solve_explain(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ solve + 闸B 程序验算（PRD-C-010；判决语义见 _check_one_item，单一事实源）。

    P2（PRD-C-012）：循环体已抽成 per-item 协程 _check_one_item，本节点按题
    asyncio.gather + Semaphore(GATE_CONCURRENCY=3) 并发，结果按原下标顺序回填；
    已带 check 的题照旧跳过不重判；generate 流内 eager 剔除的哨兵题
    （item._dropped=叙事）在此收口进 dropped_notes（语义不变：每轮重写）。
    🔴 凡进 items 的题一律过本节点，无 check 不许进 assemble（remove 后旧题带 check 原样通过）。
    """
    budget = _budget_bind(state)  # P13：轮内下游节点携带预算（闸B heal/replenish 受闸）
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    total = len(items)
    sem = asyncio.Semaphore(GATE_CONCURRENCY)

    async def _run(i: int, it: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        item = dict(it)
        # generate 流内 eager 已剔除（4d 方案A）→ 哨兵只收叙事，不外发、不重验
        if item.get("_dropped"):
            return None, str(item["_dropped"])
        if item.get("check"):  # 已定状态 → 不重复（不进并发槽、不发 stage）
            return item, None
        async with sem:
            return await _check_one_item(item, facts, i, total)

    results = await asyncio.gather(*(_run(i, it) for i, it in enumerate(items)))
    out = [it for it, _ in results if it is not None]
    dropped = [note for _, note in results if note]  # 本轮剔除叙事（按原题序）

    if dropped:
        _emit_stage("verify", "程序验算", "done", f"剔除 {len(dropped)} 道，保留 {len(out)} 道")
    else:
        _emit_stage("verify", "程序验算", "done")
    return {"items": out, "dropped_notes": dropped, "llm_call_budget": budget, "messages": []}


# ---------------------------------------------------------------------------
# 闸A·基因闸（验"是不是平行题"）：generate / exec_regenerate / exec_add 产出新变式后、
# solve_explain 之前过本闸。与闸B（答案对不对）正交：本闸只比 DNA 基因，不验数学。
#
# 🔴 B2·T2 退役换血（PRD-C-014）：闸A LLM judge 全链删（GENE_JUDGE_PROMPT / _gene_judge_one /
#    _gene_judge_prompt / _gene_feedback / gene_judge_knobs_spec / _gene_target_qtype* /
#    _gene_facts_for / gene_gate_decision 等），内涵换为**纯代码三检**：
#      ① structure_lint：题型/结构闭集 lint（_QTYPE_CONTRACT 代码镜像，复用 structure_lint）。
#      ② 表皮距离 _surface_check：题干归一化相似度 > 题型阈值（选择/填空 0.85，解答/证明 0.92）
#         或 数字与母题全同 → 判抄题打 ⚠ flag。
#      ③ 守恒透传：W2 注入的考察类型/题型守恒在 item 上的声明性校验，结果作为 flag 透传。
# 🔴 判决铁律不破：闸A 三检全是**结构/表皮形态校验**，不碰答案对错（对错归闸B sympy）。
#    任一检命中 = 打 flag（⚠ warn）**不卡死、不剔题、不回炉**——闸门必有降级路径（铁律④）。
#    干净/三检全过 = gene={gate:"pass"}；命中 = gene={gate:"warn", flags, reason}。
#    judge LLM 已无 → 不再有 skipped 语义的「调用失败放行」（纯代码不会失败，异常一律降级 pass）。
# ---------------------------------------------------------------------------


def _exam_type_conserved(item: dict, facts_i: dict) -> bool | None:
    """③ 考察类型守恒声明性校验（纯函数·可单测）。

    数据源 = 母题 DNA（facts_i.dna.exam_type，W2 注入 prompt 的同一事实源）。
    - item 声明了 exam_type 且与母题 exam_type 不一致 → False（守恒破）。
    - item 未声明 exam_type（绝大多数 generate 产物不回 exam_type 字段）→ None（无从声明性校验，
      不误报；prompt 侧 W2 已硬约束守恒，这里只做「若声明则核」的透传校验）。
    - 母题无 exam_type → None。
    """
    me = str((facts_i.get("dna") or {}).get("exam_type") or "").strip()
    if not me:
        return None
    ve = str(item.get("exam_type") or "").strip()
    if not ve:
        return None
    return ve == me


def _variant_model_ids(item: dict) -> list[str]:
    """从变式 item 取其自身的 models id（W3' 守恒判用）。

    变式当前主路径**继承母题 models**（_item_dna 回退）→ item 多数无独立 models。只有 item
    被单独锚定 / edit-dna 改过 models（批4）时才带 item['models']。无 → []（不参与软警，不误报）。
    """
    out: list[str] = []
    for m in item.get("models") or []:
        if isinstance(m, dict):
            mid = str(m.get("id") or "").strip()
        else:
            mid = str(m or "").strip()
        if mid:
            out.append(mid)
    return out


def _model_conservation_check(item: dict, facts_i: dict) -> dict[str, Any]:
    """④ 🔴 批3·W3' 模型守恒软警（纯函数·零 LLM·**不打回**·可单测）。返回 {warn, out_of_set}。

    守恒集合 = 母题 models ∪ {M00} ∪ 反查候选池（H2 放宽：候选池 = facts_i.dna.models 之外
    DNA 携带的反查候选 model_pool，若有；防 mini 欠选伴生模型致误报）。变式无独立 models →
    继承母题 → 视同守恒（warn=False）。判越界仅打 ⚠ 透传，绝不打回/剔题/回炉（铁律：软警不硬）。
    """
    dna = facts_i.get("dna") or {}
    mother_ids = [
        str(m.get("id") or "").strip()
        for m in (dna.get("models") or [])
        if isinstance(m, dict) and str(m.get("id") or "").strip()
    ]
    # 反查候选池（H2 放宽集）：锚定时若把候选池随 DNA 带（dna.model_pool），并入守恒集合。
    pool = [str(x).strip() for x in (dna.get("model_pool") or []) if str(x).strip()]
    return model_anchor.model_conservation_warn(
        _variant_model_ids(item), mother_ids, candidate_pool_ids=pool
    )


def _qtype_conserved(item: dict, facts_i: dict) -> bool:
    """③ 题型守恒声明性校验（纯函数）：变式 qtype 过别名归一后与母题一致。

    转题型（老师明确把母题改造成别的题型）是合法编辑场景——item 带 from_edit 印记时不算破守恒
    （老师意志优先，由调用方处理）。本函数只比形态：相等 = 守恒。母题 qtype 缺 → 视作守恒（不误报）。
    """
    m_raw = str(facts_i.get("qtype") or "").strip()
    if not m_raw:
        return True
    v_raw = str(item.get("qtype") or "").strip()
    if not v_raw:
        return True
    return _QTYPE_ALIAS.get(v_raw, v_raw) == _QTYPE_ALIAS.get(m_raw, m_raw)


def gene_gate_check(item: dict, facts_i: dict) -> dict[str, Any]:
    """🔴 闸A 纯代码三检（B2·T2，零 LLM / 零 IO，可单测）。

    返回 {gate, flags, reason}：
      - gate=pass：三检全过（含全部 None/无从校验的宽松项）；flags=[]。
      - gate=warn：任一检命中缺陷；flags=命中项标识列表；reason=人话拼接。
    三检（全是形态/表皮校验，不碰答案对错）：
      ① structure_lint：题型结构 lint（选择题别长成多小问嵌合体/选项不足/标答非字母；填空缺空位）。
      ② _surface_check：题干与母题相似度 > 题型阈值（选择/填空 0.85，解答/证明 0.92）或 数字全同 = 抄题。
      ③ 守恒透传：题型守恒（转题型 from_edit 豁免，由调用方判）+ 考察类型守恒（若声明）。
    🔴 降级：任何异常一律视作三检通过（gate=pass），绝不卡死出题（铁律④ + G5）。
    """
    try:
        flags: list[str] = []
        reasons: list[str] = []

        # ① 结构 lint（题型闭集形态）
        struct_defects = structure_lint(item)
        if struct_defects:
            flags.append("structure")
            reasons.extend(struct_defects)

        # ② 表皮距离（防抄母题）
        surface_defect = _surface_check(item.get("stem"), facts_i.get("stem"), item.get("qtype"))
        if surface_defect:
            flags.append("surface")
            reasons.append(surface_defect)

        # ③ 守恒透传（W2 注入的题型/考察类型守恒的声明性校验）
        #    转题型编辑（from_edit）豁免题型守恒——老师明确点名改造，不算破基因。
        if not item.get("from_edit") and not _qtype_conserved(item, facts_i):
            flags.append("qtype_conservation")
            reasons.append(
                f"题型守恒破：变式题型「{item.get('qtype')}」≠ 母题题型「{facts_i.get('qtype')}」"
            )
        if _exam_type_conserved(item, facts_i) is False:
            flags.append("exam_type_conservation")
            reasons.append(
                f"考察类型守恒破：变式「{item.get('exam_type')}」≠ 母题「{(facts_i.get('dna') or {}).get('exam_type')}」"
            )

        # ④ 🔴 批3·W3' 模型守恒软警（纯代码集合判·零 LLM·**不打回**）：变式 models ⊄
        #    母题 models ∪ {M00} ∪ 反查候选池 → 打 model_conservation flag + reason（仅 ⚠ 透传，
        #    上层 _gene_one_item 据 flag 落待命名池）。变式无 models（继承母题）→ 不报（不误杀）。
        cons = _model_conservation_check(item, facts_i)
        if cons.get("warn"):
            flags.append("model_conservation")
            oos = "、".join(cons.get("out_of_set") or [])
            reasons.append(f"解法模型守恒软警：变式用了母题外的解法模型「{oos}」（仅提示·不打回·待审）")

        if not flags:
            return {"gate": GENE_GATE_PASS, "flags": []}
        return {
            "gate": GENE_GATE_WARN,
            "flags": flags,
            "reason": "；".join(reasons) or None,
            # 🔴 透传越界 id（上层落待命名池用；无越界 → 不带键，不污染干净 item.gene）。
            **({"model_out_of_set": cons["out_of_set"]} if cons.get("warn") else {}),
        }
    except Exception:  # noqa: BLE001 — 闸A是增强不是关卡：三检异常一律降级放行（铁律④/G5）
        return {"gate": GENE_GATE_PASS, "flags": []}


async def _gene_one_item(item: dict, facts_i: dict, idx: int, total: int) -> dict:
    """🔴 闸A per-item 协程（P2 单一事实源：gene_gate 节点与 generate 流内 eager 共用）。

    B2·T2 起内涵 = 纯代码三检（gene_gate_check），LLM judge 全链已删：
    - 三检全过 → item.gene={gate:"pass", flags:[]}；
    - 任一检命中 → item.gene={gate:"warn", flags, reason}（**只警示不硬拦、不回炉、不剔题**，
      闸门降级路径；真值随 item.gene 透传进 FE 4d 展示，铁律④）。
    已带 gene 的题原样通过（不重判、不发 stage）。
    🔴 保持 async 签名（节点 gather 并发 + generate eager 链 await 调用契约不变）。
    入参 item 视为本协程私有（调用方传副本）；idx/total 只用于思路条叙事编号。
    """
    if item.get("gene"):  # 已判过 → 不重判（旧题预算保护）
        return item

    # 🔴 P14 叙事修正（RC3·PRD-C-013）：并发闸去掉误导性「/总数」；编辑轮单题明示「只重比第 N 题」。
    _emit_stage(
        "gene_gate", "平行度比对", "running",
        (f"只重比第 {idx + 1} 题" if item.get("from_edit") else f"第 {idx + 1} 题比对中"),
    )
    item["gene"] = gene_gate_check(item, facts_i)
    # 🔴 批3·W3' 越界落待命名池（含题目指针；软警不打回，仅记录可审，G4：写失败不静默）。
    oos = (item.get("gene") or {}).get("model_out_of_set")
    if oos:
        dna_i = facts_i.get("dna") or {}
        mother_ids = [
            str(m.get("id") or "").strip()
            for m in (dna_i.get("models") or [])
            if isinstance(m, dict) and str(m.get("id") or "").strip()
        ]
        ref = str(item.get("stem") or "")[:60] or str(facts_i.get("mother_question_id") or "")
        for name in oos:
            try:
                model_anchor.record_overflow_candidate(name, mother_ids, question_ref=ref)
            except Exception:  # noqa: BLE001 — 待命名池落盘失败不拖垮出题（软警是增强不是关卡）
                pass
    return item


async def gene_gate(state: VariantState, config: RunnableConfig) -> VariantState:
    """闸A 节点：对每道**未判过基因**的变式做纯代码三检（B2·T2，判决语义见 _gene_one_item）。

    P2（PRD-C-012）：循环体抽成 per-item 协程 _gene_one_item，本节点按题 asyncio.gather +
    Semaphore(GATE_CONCURRENCY) 并发，结果按原下标顺序回填（三检纯代码、无 IO，并发只为
    保持与 eager 链调用契约一致；不再有 LLM 调用，无预算消耗回炉）。
    🔴 已带 gene 的题（exec_add 追加时的旧题等）原样通过，不重判。
    """
    budget = _budget_bind(state)  # P13：轮内下游节点携带预算（本节点 T2 后不再花 LLM 预算）
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    total = len(items)
    sem = asyncio.Semaphore(GATE_CONCURRENCY)

    async def _run(i: int, it: dict[str, Any]) -> dict[str, Any]:
        item = dict(it)
        if item.get("gene"):  # 已判过 → 不进并发槽、不发 stage
            return item
        async with sem:
            return await _gene_one_item(item, facts, i, total)

    out = list(await asyncio.gather(*(_run(i, it) for i, it in enumerate(items))))
    _emit_stage("gene_gate", "平行度比对", "done")
    return {"items": out, "llm_call_budget": budget, "messages": []}


def _status_summary(items: list[dict]) -> str:
    """按 4d tier 汇总状态短语（只说好：正面/中性计数，⚠ 仅双闸低）。旧数据无 tier 不计入。"""
    tiers = [(it.get("check") or {}).get("tier") for it in items]
    parts = []
    if n := tiers.count(TIER_VERIFIED):
        parts.append(f"{n} 道程序验算通过")
    if n := tiers.count(TIER_SELF_OK):
        parts.append(f"{n} 道已独立复算一致")
    if n := tiers.count(TIER_PROOF):
        parts.append(f"{n} 道证明类已转人工复核")
    if n := tiers.count(TIER_BOTH_LOW):
        parts.append(f"{n} 道需重点核对（题卡已标注）")
    return "、".join(parts) or f"{len(items)} 道已生成"


# --- P8 难度总评（S1.2）：assemble 前一次 nano call 按绝对 rubric 复评全组难度 -----
# 绝对锚浙教版初中：不与母题相对，让一组题难度可比、入库 difficult/dim4 更准。
# 🔴 难度四档 rubric（B2·T3，事实源 = 22-题目维度-唯一事实源.md §2，2026-06-12 改 LLM rubric 断言）。
#   难度是「评级」归 LLM rubric 断言，对错是「判决」归闸B sympy——两件事不混（铁律不破）。
#   档语义照 SSOT §2 原文：1 送分 / 2 常规 / 3 多步综合 / 4 压轴。
_GRADE_DIFFICULTY_PROMPT = """你是浙教版初中数学难度评定器。下面是同一组变式题（已编号），请拿着下面这张【难度四档 rubric】给每道题断一个 1-4 的难度档（按标准判级，不是相对母题，是绝对难度；难度是评级不是判对错）：

- 4（压轴）：≥2 个真实难点 / 多突破口综合。
- 3（多步综合）：1 个难点，或 考察类型∈{{证明推理·应用建模·探究归纳}}，或 解法骨架含【最难步】构造。
- 2（常规）：无难点 + 考察类型∈{{直接计算·公式套用·性质判定}} + 多步骨架。
- 1（送分）：无难点 +（概念辨析 或 单步骨架）。

只输出 JSON 数组，每项是一个整数难度档，顺序、个数与下面题目严格一一对应，禁止多写少写、禁止解释：
例：[2,2,3,1]

题目：
{items}
"""


def _grade_difficulty_payload(items: list[dict[str, Any]]) -> str:
    """组装评分用题面（题干+答案+解析），编号 1..n。纯函数、可单测。"""
    lines: list[str] = []
    for i, it in enumerate(items, 1):
        stem = str(it.get("stem") or "").strip()
        answer = str(it.get("answer") or "").strip()
        sol = str(it.get("solution") or "").strip()
        lines.append(f"[{i}] 题干：{stem}\n答案：{answer}\n解析：{sol}")
    return "\n\n".join(lines)


async def _grade_difficulty(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """P8 难度总评（S1.2）：一次 nano call 按绝对 rubric 复评全组，覆盖 item['difficulty']。

    ⚠️ 整改2（2026-06-12）已**退出主流程**：难度判定并入生题（GENERATE/REGEN/ADD 出题调用
       同步按嵌入的四档 rubric 产出 difficulty），assemble/revise whole 不再独立调用本函数。
       函数本体保留供单测 + 潜在按需复评用，rubric 与 _DIFFICULTY_RUBRIC 同口径（22-SSOT §2）。

    🔴 G5 降级：items 空 / LLM 异常 / 解析失败 / 个数对不上 → 保留各 item 原 difficulty 值，
    绝不抛、绝不卡死。逐项越界钳到 1-4（与 _clamp_difficult 入库口径一致）。
    返回新 list（不原地改入参）；用 LLM_MODEL_LIGHT 经 _ainvoke_text 的 per-call model 覆盖。
    """
    out = [dict(it) for it in items]
    if not out:
        return out
    try:
        prompt = _GRADE_DIFFICULTY_PROMPT.format(items=_grade_difficulty_payload(out))
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)], model=settings.LLM_MODEL_LIGHT
        )
        parsed = _parse_json(text)
    except Exception:  # noqa: BLE001 — 难度总评是增强不是关卡，失败保留原值
        return out
    if not isinstance(parsed, list) or len(parsed) != len(out):
        return out  # 个数对不上 → 整体降级保留原值（不冒险错位覆盖）
    for it, raw in zip(out, parsed):
        d = _to_int(raw)
        if d is not None:
            it["difficulty"] = max(1, min(DIFFICULTY_CAP, d))  # 越界钳 1-4
        # d 解析不出 → 该题保留原 difficulty（不动）
    return out


def difficulty_consistency_defects(items: list[dict[str, Any]]) -> list[str]:
    """🔴 纯函数·难度一致性（P12.1·PRD-C-013，零 LLM）：难度从闸A 删除后下沉到这里，按
    **组内相对关系**校验 —— level=hard 题的总评难度应 ≥ 同组 level=normal 题（hard 不该比
    normal 还简单）。读 S1 _grade_difficulty 已覆盖的 item['difficulty'] 绝对档值。

    判错维度根因：judge 主观目测对赌 generate 算术申报值（两个噪声源互比，完美平行题被
    「感觉难一点」打回）。改为纯函数比相对关系，不再 LLM 自评、不再因难度回炉，只在 assemble
    头部 warn（铁律④降级：永不卡死、永不剔题）。

    返回缺陷清单（空=一致）。难度缺失/不可解析的题不参与比较（宽松，不误报）。
    """
    normals: list[int] = []
    hards: list[tuple[int, int]] = []  # (1-based 题号, 难度)
    for i, it in enumerate(items, 1):
        d = _to_int(it.get("difficulty"))
        if d is None:
            continue
        lvl = str(it.get("level") or "normal").strip().lower()
        if lvl == "hard":
            hards.append((i, d))
        else:
            normals.append(d)
    if not hards or not normals:
        return []
    max_normal = max(normals)
    bad = [(i, d) for i, d in hards if d < max_normal]
    if not bad:
        return []
    bad_s = "、".join(f"第{i}道(难度{d})" for i, d in bad)
    return [f"难度一致性：标注为「难」的 {bad_s} 总评难度低于同组普通题(最高{max_normal})"]


def _sort_by_difficulty(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """🔴 P9 默认序（PRD-C-013，纯函数零 LLM 可单测）：按总评难度**升序稳定排序**。

    Python sorted 稳定 → 同难度档保持原生成序（不打散 generate 的写题顺序）。难度缺失/
    不可解析 → 钳为 0（排最前，与 _artifact_payload 的 difficulty or 0 同口径），不抛、不丢题。
    返回新 list（不原地改入参）；item 整体随槽位移动，persisted/check/gene 等簿记字段跟题走、
    不错位（seq 由下游 index=i+1 按新序现编）。
    """
    return sorted(items, key=lambda it: _to_int(it.get("difficulty")) or 0)


async def assemble(state: VariantState, config: RunnableConfig) -> VariantState:
    """题组摘要（P1 聊天瘦身·PRD-C-012）：左栏只发摘要头——配方/守恒DNA/状态计数/旋钮提示；
    题干/答案/解析全文**只走右栏 artifact 题卡**，聊天流不再复读（用户拍板 2026-06-11）。

    🔴 整改2（2026-06-12，推翻此前 P8「独立 rubric 复评」设计）：难度判定并入生题——
    GENERATE/REGEN/ADD 出题 prompt 已嵌入四档 rubric，每道题的 difficulty 由出题调用同步产出，
    assemble **不再独立走一轮 _grade_difficulty 复评**（省掉首轮 + 整组重做各一次 nano 调用）。
    缺/非法 difficulty 沿用入库口径兜底（_clamp 到 1~4，缺→2）。P9 排序在下面做。"""
    budget = _budget_bind(state)  # P13：轮内下游节点携带预算（assemble 已无独立难度调用）
    # 🔴 整改2：删除独立难度复评调用；难度随生题产出。仅做缺/非法兜底钳到 1~4（不再额外 LLM 调用）。
    items = [dict(it) for it in (state.get("items") or [])]
    for it in items:
        d = _to_int(it.get("difficulty"))
        it["difficulty"] = max(1, min(DIFFICULTY_CAP, d)) if d is not None else 2  # 缺→2 兜底链
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
    return {"items": items, "llm_call_budget": budget, "messages": [AIMessage(content=head)]}


# ===========================================================================
# 交互层（设计 §6）：多轮 WAIT → parse 判 5 意图 → 三层漏斗分诊
#   修正 / 编辑(remove·regenerate·add) / 确认 / 答疑 / clarify
# ===========================================================================
# --- 受约束分类器（PRD-C-010 G4/FP4）：intent 闭集枚举 + 物理护栏 validate_instruction ---
INTENT_REVISE = "修正"  # 任务口径里的 revise（纠正母题年级/考点本身 → 重锚重造）
INTENT_EDIT = "编辑"  # remove/regenerate/add 细分在 ops.action（任务口径 remove|regenerate|add）
INTENT_CONFIRM = "确认"  # 任务口径里的 confirm
INTENT_QA = "答疑"  # 任务口径里的 qa
INTENT_CLARIFY = "clarify"
# 🔴 整改3（2026-06-12）：解法修正 scope —— 老师约束解题方法/纠正年级进度，但**不要求换题**。
#   实测痛点：「这里是7年级的题目，没学二元方程，只能用一元一次去解题」被误判成「修正(年级)」
#   走 patch 清 items 整组重做（302s）。改为新 scope：题面保留，仅按新约束重写每道题的解析+
#   重跑闸B；某题在新约束下根本无法求解才单题重出题面，其余题不动。不触发整组重出。
INTENT_SOLUTION_ONLY = "解法修正"
VALID_INTENTS = {
    INTENT_REVISE, INTENT_EDIT, INTENT_CONFIRM, INTENT_QA, INTENT_CLARIFY, INTENT_SOLUTION_ONLY,
}
# 🔴 P9（PRD-C-013）：+reorder —— 纯代码 list 重排（零 LLM 改题），与 remove/regenerate/add
# 同走指令通道；越界/缺序/混类 → clarify（永不默认重排）。
EDIT_ACTIONS = {"remove", "regenerate", "add", "reorder"}
ADD_COUNT_MAX = 5  # 与 exec_add 单轮上限同口径（min(n,5)），护栏在源头就钳掉

# 🔴 排版（PRD-C-012 任务3）：分类标准/输出契约/硬约束固定段前移，变动段
# （母题 DNA、老师最新一句话）移末尾（runtime 追加的 17 号无题组语境段照旧排最后）；
# 语义一字不改。
PARSE_PROMPT = """你是举一反三 agent 的指令解析器（受约束分类器：intent 只能从 6 个枚举值里选 1 个，禁止发明新值）。老师正在看一组已出的变式题（共 {n} 道），下面是他的最新一句话。

【分类标准】逐条对照，命中哪条选哪条；都不命中选 "clarify"（其中"编辑"细分 4 种 action）：
1. "答疑" —— 判别：只是要老师**讲解/解惑**某题（问怎么解、为什么，**或祈使式要求讲一遍**），不要求改动任何题。
   - 疑问句例：「为什么第3题选B？」「第2题怎么解？」
   - 🔴 祈使式也算答疑（常见漏判）：「第2题讲一遍」「给学生讲」「按给学生讲的方式讲讲」「再讲讲第1题」「这道题展开说说」——都是要讲解，不是改题。
   - 🔴 承接上一轮 AI 主动提议时尤其要认（见末尾「上一轮 AI 说了什么」）：若 AI 上一句提议「我可以把第N题完整讲一遍」，老师回「好/可以/讲/给学生讲」= 承接该提议 → 答疑（讲第N题），别判 clarify。
2. "编辑"+remove —— 判别：点名删掉某道题，且能给出 1~{n} 内的题号。例：「第2题删掉」→ ops=[{{"action":"remove","index":2}}]
3. "编辑"+regenerate —— 判别：点名**改造/重出某道题**（换数字 / 换一道新题 / **改题型 / 换题型 / 加情境/换场景**），且能给出 1~{n} 内的题号。**改题型与换场景都归这里**（题型、场景都不在硬守恒里，可以改）。把目标题型/场景要求写进 note。
   - 例（换题）：「第1题换一道，数字简单点」→ ops=[{{"action":"regenerate","index":1,"note":"数字简单点"}}]
   - 例（改题型）：「第1题改成选择题」/「这题改填空」→ ops=[{{"action":"regenerate","index":1,"note":"改成选择题"}}]
   - 例（换场景）：「第2题加入杭州元素」/「换成行程问题的情境」→ ops=[{{"action":"regenerate","index":2,"note":"加入杭州元素场景"}}]
4. "编辑"+add —— 判别：要求再加 N 道题（N 为正整数，单轮最多 5）。例：「再来2道难的」→ ops=[{{"action":"add","count":2,"note":"难的"}}]
4b. "编辑"+reorder —— 判别：只调整题目**顺序**（不改题/不增删），且能给出一个**覆盖全部 {n} 道**的新次序。例（共3道）：「把顺序换成 3 1 2」→ ops=[{{"action":"reorder","order":[3,1,2]}}]。order 必须是 1~{n} 的**全排列**（每个题号恰出现一次）；只说「按难度排」「倒过来」这种没给出明确全排列的，**不要自己编 order**，留空 ops、intent 取 "clarify" 让老师给次序。
5. "解法修正" —— 判别：老师**约束解题方法**、或**纠正年级/进度从而限定能用的解法**，但**不要求换题**（题面保留，只改解析/解法）。例：「这里是7年级的题目，没学二元方程，只能用一元一次去解题」→ intent=解法修正，method_constraint=「只能用一元一次方程，不用二元方程」，grade_correction=「七年级」。又例：「解析别用因式分解，改用配方法」→ 解法修正，method_constraint=「改用配方法，不用因式分解」。🔴 关键区分：老师明确改的是**怎么解**（解析/方法），不是**换一道题**——别误判成"修正(整组重锚重做)"或"编辑+regenerate(换题)"。
6. "修正" —— 判别：老师纯纠正母题的年级或考点**本身**、要重新锚定**重出整组**（不是只改解法）。例：「这其实是八年级的二次函数题，重新出」→ mother_correction={{"grade":"八年级","kp":"二次函数"}}。⚠ 若老师只是限定解法（见第5条），优先判"解法修正"，别走整组重做。
7. "确认" —— 判别：老师对这组题满意，要入库/保存/结束。例：「这组可以了，入库吧」
8. "clarify" —— 判别：撞硬守恒（要换主考点）、题号给不出或超出 1~{n}、或意图真说不清。例：「改成考函数的题」（撞守恒换考点）

只输出一个 JSON（不要解释、不要 markdown fence）：
{{
  "intent": "解法修正|修正|编辑|确认|答疑|clarify",   // 6 选 1
  "ops": [                                  // intent=编辑 时的操作列表（其余为空数组）
    {{"action":"remove|regenerate|add|reorder", "index": 1, "count": 1, "order": [3,1,2], "note":"自由约束/旋钮说明(改题型/换场景/数字等要求一律写这里)"}}
  ],
  "comp": "可被旋钮吸收的软约束(超旋钮但 best-effort 能顺的)，没有填 null",
  "extra_constraints": ["其余自由约束句"],
  "mother_correction": {{"grade":null, "kp":null}},  // intent=修正 时老师纠正的年级/考点，否则全 null
  "method_constraint": null,   // intent=解法修正 时老师对解题方法的约束（原话归纳，如"只能用一元一次方程"），否则 null
  "grade_correction": null,    // intent=解法修正 时若老师顺带纠正了年级（如"七年级"），填规范化年级，否则 null
  "confidence": 0.0~1.0
}}

硬约束（违反任何一条，程序护栏会把你的输出整体降级为 clarify）：
- intent 只能是上述 6 个枚举值之一；ops.action 只能是 remove/regenerate/add/reorder。
- remove/regenerate 的 index 从 1 起、必须 ≤ {n}；拿不准题号时 ops 留空、intent 取 "clarify"。
- reorder 的 order 必须是 1~{n} 的全排列（长度={n}、每号恰一次）；给不全/有重复/越界 → ops 留空、intent 取 "clarify"。
- add 的 count 必须是正整数；intent=答疑/确认/修正/解法修正/clarify 时 ops 必须为空数组。
- intent=解法修正 时 method_constraint 必须非空（说清不能用什么/必须用什么）；说不清就 clarify。
- 同一句里不要混多类操作（如又删又排）；混了 → intent 取 "clarify" 请老师分句说。
- 解析不出来 = "clarify"，绝不猜成删题。

母题 DNA（硬守恒，老师不能改这两项，撞它即 clarify 驳回）：
- 主考点: {kp_name}
- 年级: {grade}

上一轮 AI 说了什么（🔴 BUG-006·承接判别用：老师最新一句若在承接/回应 AI 上一句的提议，按上一句语境定意图，比如 AI 提议「我可以讲一遍第2题」、老师回「给学生讲」= 答疑·讲第2题）：
{prev_ai}

老师最新一句话：
{utterance}"""


def _items_brief(items: list[dict]) -> str:
    """给 parse / answer 当上下文：每题 index + 题干前 80 字。"""
    lines = []
    for i, it in enumerate(items):
        stem = (it.get("stem") or "").replace("\n", " ")[:80]
        lines.append(f"第{i + 1}题: {stem}")
    return "\n".join(lines)


def _to_int(v: Any) -> int | None:
    """宽容转 int（str/float 可转则转），失败返 None。bool 不算数字。"""
    if isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def validate_instruction(parsed: Any, current_item_count: int) -> dict[str, Any]:
    """🔴 物理护栏（PRD-C-010 G4/FP4）：把 LLM 分类器输出钳成受约束载荷。

    纯函数（零 LLM、零 IO，可单测）。输入 = _parse_json 的产物（可能为 None/非 dict），
    输出 = 规整后的 pending 雏形（不含 utterance，由 parse_instruction 补）。

    规则表：
    - R0 解析失败/非 dict → 整体降级 clarify（永不默认成 remove）。
    - R1 intent 不在 VALID_INTENTS 白名单 → clarify。
    - R2 ops 白名单清洗：仅保留 action∈EDIT_ACTIONS 的 dict op；index/count 强转 int。
    - R3 remove/regenerate：index 必须给出且 ∈ [1, current_item_count]，
         任何一个越界/缺失 → 整体降级 clarify（让 agent 反问而不是乱删）。
    - R4 add：count 缺失/非法/≤0 → 钳为 1；> ADD_COUNT_MAX → 钳为 ADD_COUNT_MAX。
    - R5 intent=编辑 但 ops 清洗后为空 → clarify。
    - R6 答疑/确认/修正/clarify → ops 强制清空（物理保证答疑/确认带不动编辑 op；
         answer_question 本身 return 不含 items，双保险不破坏）。
    - R7 同句多**类**操作（如 remove+add）→ 整体降级 clarify 请老师分句说：
         执行层（dispatch→exec_*）单轮只走一类分支，混类会被静默丢弃半截，且
         remove 后题号位移会使同句其它 index 失效 —— 与 R3「绝不部分执行」同哲学。
         同类多 op（删第2、3题 / 重出第1、2题）合法保留，exec_* 一次吃完。
    - R8 reorder（P9·PRD-C-013）：order 必须是 1..current_item_count 的**全排列**（长度=N、
         每号恰一次）；缺序/重复/越界/混类 → clarify（永不默认重排）。
    - R9 解法修正（整改3·2026-06-12）：method_constraint 必须非空（说清不能用什么/必须用什么）；
         缺失 → clarify。ops 物理清空（与 R6 同哲学，解法修正带不动编辑 op）。
    """

    def _normalized(p: dict) -> dict[str, Any]:
        mc = p.get("method_constraint")
        gc = p.get("grade_correction")
        return {
            "intent": p.get("intent"),
            "ops": [],
            "knobs": p.get("knobs") if isinstance(p.get("knobs"), dict) else {},
            "comp": p.get("comp"),
            "extra_constraints": (
                list(p.get("extra_constraints"))
                if isinstance(p.get("extra_constraints"), list)
                else []
            ),
            "mother_correction": (
                p.get("mother_correction") if isinstance(p.get("mother_correction"), dict) else {}
            ),
            # 🔴 整改3：解法修正 scope 字段（method_constraint 必填，grade_correction 可选）。
            "method_constraint": (str(mc).strip() if mc and str(mc).strip() else None),
            "grade_correction": (str(gc).strip() if gc and str(gc).strip() else None),
            "confidence": p.get("confidence"),
        }

    def _clarify(base: dict[str, Any] | None = None) -> dict[str, Any]:
        out = base if base is not None else _normalized({})
        out["intent"] = INTENT_CLARIFY
        out["ops"] = []
        return out

    if not isinstance(parsed, dict):  # R0
        return _clarify()

    base = _normalized(parsed)
    intent = base["intent"]
    if intent not in VALID_INTENTS:  # R1
        return _clarify(base)

    # 🔴 整改3·R9：解法修正必须给出 method_constraint（说清不能用什么/必须用什么）；
    #   缺失 → 降级 clarify（不空转一次解法重写）。ops 物理清空（R6 同哲学）。
    if intent == INTENT_SOLUTION_ONLY:
        if not base.get("method_constraint"):
            return _clarify(base)
        return base

    if intent != INTENT_EDIT:  # R6：非编辑意图物理上带不动 ops
        return base

    # intent = 编辑：清洗 ops（R2~R5）
    raw_ops = parsed.get("ops") if isinstance(parsed.get("ops"), list) else []
    ops: list[dict[str, Any]] = []
    for op in raw_ops:
        if not isinstance(op, dict):
            continue
        action = op.get("action")
        if action not in EDIT_ACTIONS:  # R2
            continue
        clean: dict[str, Any] = {"action": action}
        if op.get("note"):
            clean["note"] = str(op.get("note"))
        if action in ("remove", "regenerate"):
            idx = _to_int(op.get("index"))
            if idx is None or not (1 <= idx <= current_item_count):
                return _clarify(base)  # R3：越界/缺号 → 反问，绝不乱删
            clean["index"] = idx
        elif action == "reorder":
            # 🔴 R8（P9·PRD-C-013）：order 必须是 1..N 的**全排列**（长度=N、每号恰一次）。
            #   缺序/重复/越界/混类 → 整体降级 clarify（永不默认重排，与 R3 同哲学）。
            raw_order = op.get("order")
            if not isinstance(raw_order, list) or len(raw_order) != current_item_count:
                return _clarify(base)
            order = [_to_int(x) for x in raw_order]
            if any(x is None for x in order):
                return _clarify(base)
            if sorted(order) != list(range(1, current_item_count + 1)):
                return _clarify(base)  # 非全排列（重复/越界/缺号）→ 反问
            clean["order"] = order
        else:  # add
            cnt = _to_int(op.get("count"))
            if cnt is None or cnt <= 0:
                cnt = 1  # R4 下限
            clean["count"] = min(cnt, ADD_COUNT_MAX)  # R4 上限
        ops.append(clean)
    if not ops:
        return _clarify(base)  # R5
    if len({op["action"] for op in ops}) > 1:
        return _clarify(base)  # R7：混类操作绝不部分执行 → 反问分句
    base["ops"] = ops
    return base


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


# --- 答疑：只问不改（🔴 物理上 return 不含 items） -----------------------------
ANSWER_PROMPT = """你是数学老师，老师对下面这组变式题的某道有疑问，请耐心解惑（讲思路/为什么这么解）。

题组：
{brief}

各题答案/解析摘要：
{detail}

老师的问题：{question}

直接用人话回答（数学式一律 $...$ 包裹）。只解惑，不要改题、不要重出题。"""


async def answer_question(state: VariantState, config: RunnableConfig) -> VariantState:
    """答疑侧循环（设计 §6）：解老师对解析的疑问 → 回等待。

    🔴 代码层物理不写 state.items：return 的 update 不含 'items' 键，即便 parse 误判进此分支也改不了题。
    """
    items = state.get("items") or []
    question = (state.get("pending") or {}).get("utterance") or _latest_human_text(
        state.get("messages", [])
    )
    detail = "\n".join(
        f"第{i + 1}题 答案:{(it.get('answer') or '')[:60]} 解析:{(it.get('solution') or '')[:120]}"
        for i, it in enumerate(items)
    )
    # 🔴 public_stream：答疑是纯人话输出，token 不打 skip_stream → 前端打字机逐字外放
    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ANSWER_PROMPT.format(
                    brief=_items_brief(items), detail=detail, question=question or "(空)"
                )
            )
        ],
        public_stream=True,
    )
    # 🔴 只返回 messages（绝不含 items），并清空 pending
    return {"messages": [AIMessage(content=text or "我没太理解你的疑问，可以再说具体点吗？")], "pending": None}


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
    ops = (state.get("pending") or {}).get("ops") or []
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

    for t in targets:
        old = items[t]
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
                        + (f"\n额外要求：{notes[t]}" if t in notes else ""),
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
            # 配方印记跟题走（与 difficulty 同理：重出仍占原计划槽位，闸A 改判段不丢）
            for k in ("from_recipe", "expected_difficulty"):
                if old.get(k) is not None:
                    new_item[k] = old[k]
            # 🔴 RC2（PRD-C-013）：编辑轮产物打 from_edit 印记 → 重入 gene_gate 时只判不回炉
            # （老师已点名改造，基因闸判不过只标 warn，不重出覆盖老师意志）；老师 note 存 edit_note，
            # 任何下游 REGEN（闸B 验算回炉）都注回，不被失败原因冲掉。
            new_item["from_edit"] = True
            if t in notes:
                new_item["edit_note"] = notes[t]
            items[t] = new_item
    # 🔴 编辑轮清陈旧缺陷外显（同 exec_remove 注释）
    return {
        "items": items,
        "pending": None,
        "shape_defects": [],
        "llm_call_budget": budget,
        "messages": [],
    }


# --- 编辑·add：生成 N 道新题（吸收软约束/旋钮）→ 过 solve_explain ----------------
# 🔴 排版（对抗审修复·比照 GENERATE_PROMPT）：固定段（输出契约/格式/载荷契约）前移
# 吃 aigeek 前缀缓存，变动段（母题 DNA/补充要求）移末尾；并补上 4a verify_payload 契约
# ——编辑轮加题同样出题自带载荷，不再永远回落事后抽取。
ADD_PROMPT = (
    """你是浙教版初中数学命题专家。基于母题 DNA，**新增** {n} 道举一反三变式。

只输出 JSON 数组(不要解释)，每元素：
{{"stem":"题干","answer":"标准答案","solution":"完整解析","qtype":"选择/填空/解答","difficulty":1~4,"level":"normal/hard","injected_kp":"相邻kp名或null",
  "verify_payload":{{...该题的程序验算载荷，契约见下...}}}}

"""
    + _DIFFICULTY_RUBRIC
    + """

格式硬规定：🔴 题面(stem)与选项数学式**一律行内 $...$**，**严禁** $$...$$ / \\[ \\] / display 块级公式（撑满整行）；**仅 solution 多行分步推导**可用 $$。禁止裸 LaTeX / \\( \\) 定界；换行用标准 \\n。

"""
    + _QTYPE_CONTRACT
    + """

verify_payload 字段（PRD-C-012 4a·出题自带验算载荷：把**这道题自己的题干 + 标准答案**抽成可被 sympy 程序验算的结构化载荷，验算对象 claimed = 该题标准答案）：
"""
    + _PAYLOAD_CONTRACT
    + """
抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。

铁律：每道仍考「{kp_name}」、仍在「{grade}」；只换数字/场景（除非老师明确要改难度/题型）。

母题 DNA：
- 主考点(硬守恒): {kp_name}
- 年级(硬守恒): {grade}
- 题型(默认): {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}

老师的补充要求（best-effort 吸收，撞守恒的忽略）：{extra}"""
)


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


# --- 编辑·reorder（P9·PRD-C-013）：纯代码 list 重排 + seq 重编，零 LLM 改题 -----------
def _is_full_permutation(order: Any, n: int) -> bool:
    """order 是否为 1..n 的全排列（长度=n、每号恰一次、无越界/重复/缺失）。"""
    return (
        isinstance(order, list)
        and len(order) == n
        and sorted(int(x) for x in order if isinstance(x, int))  # 全 int 才纳入
        == list(range(1, n + 1))
        and all(isinstance(x, int) for x in order)
    )


def _reorder_items(items: list[dict[str, Any]], order: list[int]) -> list[dict[str, Any]]:
    """🔴 纯重排（零 LLM / 零改题）：按 1-based 全排列 order 挪槽位，整 item 对象跟着走。

    exec_reorder（graph 节点）与 /variant/reorder（端点）共用的单一事实源：check/gene/
    persisted/_seq 等簿记字段随 item 整体搬位、绝不错位。调用方须先用 _is_full_permutation
    校验 order 合法（本函数假定 order 已是 1..len(items) 全排列，不再二次防御）。
    """
    return [items[i - 1] for i in order]


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
SOLUTION_ONLY_PROMPT = """你是浙教版初中数学解题老师。下面这道题的**题面和标准答案保持不变**，老师对解题方法提了新约束，请**只重写解析（解题过程）**，使其严格遵守新约束。

主考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
题型: {qtype}
题干（不要改）: {stem}
标准答案（不要改）: {answer}
当前解析: {solution}

🔴 老师对解题方法的新约束（必须遵守）: {method_constraint}

请判断：在老师的新方法约束下，**这道题的题面**是否还能解出（得到与标准答案一致的结果）？
- 若能解：solvable=true，重写一份只用约束内方法的完整解析（过程 + 答案），不改题面、不改标准答案。
- 若根本无法用约束内方法求解（如题面本身必须用被禁止的方法才能解）：solvable=false，filled reason 说明原因（这种题会被换成另一道符合约束的题，由后续流程处理）。

只输出一个 JSON（不要解释、不要 markdown fence）：
{{"solvable": true, "solution": "重写后的完整解析（只用约束内方法）"}}
或
{{"solvable": false, "reason": "为什么这道题面无法用约束内方法求解"}}

格式硬规定：solution 数学式优先行内 $...$；**仅多行分步推导**可用 $$...$$（单个等式仍用行内 $...$）。禁止裸 LaTeX / \\( \\) / \\[ \\] 定界；换行用标准 \\n。"""


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
            rechecked, _dropped = await _check_one_item(new_it, facts, i, len(items))
            return rechecked if rechecked is not None else new_it
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
        for k in ("from_recipe", "expected_difficulty", "_seq", "persisted", "_persist_id", "level"):
            if it.get(k) is not None:
                draft[k] = it[k]
        draft.pop("check", None)
        rechecked, _dropped = await _check_one_item(draft, facts, i, len(items))
        return rechecked if rechecked is not None else draft

    sem = asyncio.Semaphore(GATE_CONCURRENCY)

    async def _guarded(i: int, it: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _one(i, it)

    new_items = await asyncio.gather(*[_guarded(i, it) for i, it in enumerate(items)])
    for it in new_items:
        _format_item_stem(it)
    _emit_stage("solution", "按新方法重写解析", "done", f"{len(new_items)} 道")
    # 🔴 解法修正改了题集内容（解析/个别题面）→ 清陈旧缺陷外显，回 assemble 收口快照
    update["items"] = list(new_items)
    update["shape_defects"] = []
    return update


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
    return {
        "analysis": analysis,
        "items": [],
        "mother_confirmed": False,
        "facts_locked": False,
        "facts_audit": audit,
        "pending": None,
        "messages": [AIMessage(content="收到修正，我按新的年级/考点重锚并重出这组变式。")],
    }


# BUG-003：clarify 文案归因区分——只有老师真在「换考点/换年级」时才说撞守恒，否则不甩。
#   纯字符串启发式（零 LLM）：utterance 同时含「改/换/变…」改动词 + 考点/年级类对象词，
#   才判为「疑似撞守恒」，给守恒说明；其余一律走「没听懂」分支，不误导。
_CONSERV_OBJECT_WORDS = ("考点", "知识点", "年级", "学段", "年纪")
_CONSERV_CHANGE_VERBS = ("改", "换", "变", "成")


def _looks_like_conservation_hit(utterance: str) -> bool:
    """老师这句是否疑似在动「考点/年级」硬守恒（用于 clarify 文案归因，宁缺毋滥）。"""
    s = str(utterance or "")
    if not s:
        return False
    return any(v in s for v in _CONSERV_CHANGE_VERBS) and any(
        w in s for w in _CONSERV_OBJECT_WORDS
    )


async def ask_clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """三层漏斗第③层 / 答非所问兜底：撞守恒 / 说不清 → 回问（设计 §6，不改 items）。

    🔴 BUG-003：归因二分——真撞守恒（动考点/年级）才说守恒不可改并明示可改维度；否则只说没听懂，
    不甩「可能撞硬守恒」误导（守恒只有考点+年级两项，题型/场景/数量/难度/解法都能改）。
    """
    pending = state.get("pending") or {}
    utterance = pending.get("utterance") or ""
    facts = _mother_facts(state)
    actionable = (
        "你可以这样说：\n"
        "- 删/重出/再加题（如「第 2 题重出」「第 1 题改成选择题」「再来 2 道难的」）\n"
        "- 拨旋钮（数字 / 场景 / 难度 / 题型配比，如「第 2 题加入杭州场景」）\n"
        "- 问解析（如「第 1 题为什么这么解」）\n"
        "- 「这组可以了」入库"
    )
    if _looks_like_conservation_hit(utterance):
        # 真撞守恒：明示守恒=考点+年级两项不可改，其余维度都能改
        body = (
            f"这组变式的硬守恒只有两项：考点「{facts['kp_name']}」+ 年级「{facts['grade']}」"
            "——这两项不能改（换考点/换年级等于换一道母题，请重新贴图）。\n\n"
            "**除这两项外都能改**：题型、场景、数量、难度、解法都可以拨。\n\n"
            + actionable
        )
    else:
        # 真没听懂：不甩守恒（守恒不是被撞的原因），只请老师说具体点
        body = (
            "我没太 get 到你想改什么，能再说具体点吗？\n\n"
            + actionable
        )
    if utterance:
        body += f"\n\n（你刚说的是：「{utterance}」）"
    return {"messages": [AIMessage(content=body)], "pending": None}


# --- 确认入库（设计 §7）：变式+解析经 RuoYi 写老师个人题库，只写不判 -----------
async def persist_to_bank(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ 入库：老师"这组可以了" → 逐题 POST /teacher/question/create（teacher token 定 owner）。

    🔴 只写不再判（质量门已在 solve_explain 闭合）。入库回执（入库 N 道 + 失败标注）。
    🔴 owner 由后端 LoginHelper 定，body 绝不传 createBy。
    """
    items = state.get("items") or []
    facts = _mother_facts(state)
    if not items:
        return {"messages": [AIMessage(content="当前没有可入库的变式题。先贴图举一反三吧。")]}

    # 🔴 PRD-C-015 批4·致命① 入库防脏硬闸（新不变量）：存在 dna_dirty 题 / 母题脏 → 拒绝入库。
    #   与「凡进 items 必过 solve_explain」并列。提示精确到第 N 题（先点重生或撤销）。
    dirty_msg = persist_dirty_guard(state)
    if dirty_msg:
        return {"messages": [AIMessage(content="⚠ 暂不能入库：" + dirty_msg)]}

    # 🔴 入库簿记（PRD-C-011 G5 + 批4 缺口10）：防重判据从「persisted 跳过」改成
    #   「persisted 且 not dirty 才跳过」。dirty 已被上面硬闸拦掉 → 此处 pending = 未入库的题；
    #   重生后再入库走「覆盖原行 update by _persist_id」（persist_items 据 _persist_id 走 update）。
    pending_idx = [
        i for i, it in enumerate(items)
        if not (it.get("persisted") and not it.get("dna_dirty"))
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
        receipts = await persist_items(pending_items, facts, token=token)
    except Exception as e:  # noqa: BLE001 — 登录/网络整体失败 → 友好兜底，不崩
        _emit_stage("persist", "入库", "warn", "连不上题库服务")
        return {
            "messages": [
                AIMessage(content=f"入库时连不上题库服务（book-server :8090 是否在跑？）：{e}")
            ]
        }

    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var_receipts = [r for r in receipts if r.get("role") != "mother"]
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
        new_items[j]["persisted"] = bool(r.get("ok"))
    update: VariantState = {"items": new_items}
    if mother and mother.get("ok") and mother.get("id") is not None:
        # persist_items 的 mother_question_id 回填发生在局部 facts 副本上 → 这里落回 state，
        # 重试/后续入库走「母题已在库」分支，不再重复建母题
        update["mother_dna"] = dict(state.get("mother_dna") or {}, mother_question_id=mother.get("id"))

    # artifact 更新快照（PRD-C-011）：按回写后的 items 组帧（_artifact_payload 读 item.persisted）
    _emit_artifact({**state, "items": new_items})

    lines = [f"## 入库完成 · 变式 {len(pending_items)} 道，成功 {len(ok)} 道"]
    if n_skipped:
        lines.append(f"（另有 {n_skipped} 道此前已收录，本次跳过、未重复入库。）")
    # 母题(原题)入库回执：图母题不在库 → 先落原题挂血缘
    if mother and mother.get("ok"):
        lines.append(f"📌 原题(母题)已一并入库，ID：{mother.get('id')}，变式都挂在它名下（血缘可追）。")
    elif mother and not mother.get("ok"):
        lines.append(f"⚠ 原题入库失败（变式仍已落，血缘暂缺）：{mother.get('error')}")
    if ok:
        ids = [str(r.get("id")) for r in ok if r.get("id") is not None]
        lines.append("已落入你的个人题库（来源标记「举一反三」）。" + (f"变式 ID：{', '.join(ids)}" if ids else ""))
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
    # item 级防重（缺口10）：persisted 且 not dirty → 跳过（已收录、未改过）；
    #   dirty 已被上面拦掉，故走到这里的 persisted 题就是干净的已收录题。
    if target.get("persisted") and not target.get("dna_dirty"):
        return (
            {},
            {"ok": True, "id": target.get("_persist_id"), "role": "variant", "skipped": True},
            None,
        )

    facts = _mother_facts(state)
    receipts = await persist_items([target], facts, token=token)
    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var = next((r for r in receipts if r.get("role") != "mother"), None) or {"ok": False, "error": "无入库回执"}

    update: VariantState = {}
    if var.get("ok"):
        new_items = [dict(it) for it in items]
        new_items[index - 1]["persisted"] = True
        if var.get("id") is not None:
            new_items[index - 1]["_persist_id"] = var.get("id")  # 防重回查用（内部键，不入库）
        update["items"] = new_items
        if mother and mother.get("ok") and mother.get("id") is not None:
            update["mother_dna"] = dict(
                state.get("mother_dna") or {}, mother_question_id=mother.get("id")
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
    if stem is not None:
        it["stem"] = _sanitize_rich_text(stem)
    if answer is not None:
        it["answer"] = _sanitize_rich_text(answer)
    if solution is not None:
        it["solution"] = _sanitize_rich_text(solution)
    it["manual_edited"] = True
    it["from_edit"] = True
    # 🔴 题型模版自动规范（PRD-C-009·BE）：手动编辑回写后顺手规范 stem（净化之后）。
    #   edit-item 本身不跑 sympy 判决（check 置 manual 中性），此处规范无判决可影响——让老师
    #   手改的题立刻是 canonical 上屏，即便未点 reverify 也吃规范文本。cosmetic-only、幂等。
    _format_item_stem(it)
    # check 置中性：手动编辑、验算待重跑（清旧 verify/badge/tier，避免徽章误导）
    it["check"] = {"tier": TIER_MANUAL}
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
        # 🔴 批4·元数据维（meta）：难点是标注/基准属性，改不必重出题面、不进 dirty。
        #   但仍是守恒 4 维之一 → 走冻结 setter 留痕（缺口6）。落 mother_dna.dna.hard_points。
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

    if rclass == "hard_anchor":
        # 🔴 缺口7·硬锚【主考点/年级】改 = 解冻 + 重锚（立即，走既有 patch/classify 路径）：
        #   清 items + mother_confirmed=False + facts_locked=False → after_patch/after_classify
        #   触发重锚重造。不进 dirty 攒批（与软重生维分路）。
        #   ⚠ 注意：硬锚立即重锚是「整组重出」语义，本道的 manual 改动已写进 analysis/dna 留痕，
        #   classify 会按新锚重抽 → 此处不保留旧 items。
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

    # 🔴 D-merge8·母题守恒维改（secondary_kps/exam_type/skeleton/hard_points 母题级）→ 母题脏 +
    #   下游所有变式标 dirty 不自动重出（并入待重生集合）。注意 hard_points 虽是 meta（自身不脏），
    #   但作为母题守恒基准维改了仍要波及下游（基准变了）；secondary_kps 同理（meta 但母题级守恒）。
    if field in _DNA_CONSERVE_FIELDS and field != "main_kp":
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
REVISE_FIELD_PROMPT = (
    """你是浙教版初中数学命题专家。老师要求**只重写这道变式题的「{target_cn}」这一维**，其余维度
（题型/答案/知识点/难度等）一律**保持不动**。

母题考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
本题现状：
- 题干: {stem}
- 题型: {qtype}
- 当前{target_cn}: {current}

老师的修改要求：{instruction}

只输出一个 JSON（不要解释），仅含被改维：
{{"{target_key}": "重写后的{target_cn}文本"}}
🔴 只改「{target_cn}」，不要顺手改题型/答案/知识点；{target_cn}是纯文本维，改它**不影响**本题的标准答案与判分。"""
)

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
        # 配方印记跟题走（重出仍占原计划槽位）
        for k in ("from_recipe", "expected_difficulty", "_seq", "persisted", "_persist_id"):
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
        new_items[index - 1] = final
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
_REWRITE_SOLVE_PROMPT = (
    """你是浙教版初中数学命题专家。老师改了这道题的**解法基准（解法骨架 / 解题模型）**，
要求按新基准**重写解析（solution）**，但**题干、标准答案、题型一律不动**（只换"怎么解"的写法，
不换"题目"和"答案"）。

母题考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
题干(不动): {stem}
标准答案(不动): {answer}
新解法骨架(老师定): {skeleton}
新解题模型(老师定): {models}

只输出一个 JSON（不要解释）：
{{"solution": "按新解法骨架/模型重写的解析全文（要能推出上面那个标准答案）"}}
🔴 重写后的解析必须仍然推得出题面给定的标准答案；不要改题目、不要改答案。"""
)


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
    state: VariantState, indexes: list[int] | None = None
) -> tuple[VariantState, dict[str, Any], str | None]:
    """🔴 手动「重生」入口（D-merge6/8 + 缺口12）：对待重生集合（dna_dirty 题）一次性重出。

    - indexes=None → 全待重生集合（dirty_item_indexes）；给定 indexes → 只重生这些（仍须是 dirty 题）。
    - 每道 dirty 题：① 存 regen_snapshot（缺口12 撤销用）② 按脏维分流重出：
        · 含软重生维（题型/难度/考察类型/场景）→ _regen_once 重出整题 + _check_one_item 闸B（批3
          反退化闸自动复用）；
        · 仅重写解析维（骨架/models）脏 → _rewrite_solve_once 只重写解析 + _check_one_item 闸B 重验。
      ③ 保留手改不被母题基准覆盖（_merge_preserve_manual，D-merge8）④ 清 dirty。
    - 🔴 重生后该题 persisted 保留但因内容已变 → 入库走「覆盖原行 update by _persist_id」（缺口10，
      persist_items 据 _persist_id 走 update_question；见 persist_to_bank/persist_items）。
    返回 (update, result{regenerated:[idx], failed:[{idx,error}]}, error)。无 dirty → (空, {...}, None)。
    """
    items = list(state.get("items") or [])
    facts = _mother_facts(state)
    if not items:
        return {}, {"regenerated": [], "failed": []}, None
    targets = indexes if indexes else dirty_item_indexes(items)
    # 过滤：只重生确实 dirty 的（防误触发非脏题重出）
    targets = [n for n in targets if 1 <= n <= len(items) and items[n - 1].get("dna_dirty")]
    if not targets:
        return {}, {"regenerated": [], "failed": []}, None

    new_items = [dict(it) for it in items]
    regenerated: list[int] = []
    failed: list[dict[str, Any]] = []

    for n in targets:
        old = new_items[n - 1]
        snapshot = snapshot_item(old)
        dirty_dims = list(old.get("dirty_dims") or [])
        mother_dims = list(old.get("mother_dirty_dims") or [])
        # 本题脏维涉及软重生维 → 整题重出；否则（仅 rewrite_solve 维脏）→ 只重写解析。
        all_dims = set(dirty_dims) | set(mother_dims)
        need_full_regen = any(d in _SOFT_REGEN_FIELDS for d in all_dims)

        try:
            if need_full_regen:
                seed = dict(old)
                seed["from_edit"] = True  # 闸B 见 from_edit：FAIL 不回炉换题，保留打 ⚠
                draft = await _regen_once(seed, facts, feedback=None)
                if not draft:
                    failed.append({"index": n, "error": "重出失败（模型未返回有效题目），已保留原题"})
                    continue
                draft["from_edit"] = True
                # 配方印记 + 入库簿记跟题走（重出仍占原槽位；_persist_id 留着 → 入库走覆盖）
                for k in ("from_recipe", "expected_difficulty", "_seq", "persisted", "_persist_id", "level"):
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
        new_items[n - 1] = final
        regenerated.append(n)

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
    return {"items": new_items}, restored, None


# --- 输入边界兜底（设计 §6）：没图/无在途母题/无题组 → 催图 ------------------
async def require_login(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'auth' 分支落点：登录态缺失 → 拒入图（teacher_id 绑死硬闸的提示面）。"""
    return {
        "messages": [
            AIMessage(
                content=(
                    "🔒 登录态缺失或已过期，举一反三需要绑定到你的账号才能使用"
                    "（对话记录与入库的题都归属到你本人）。请重新登录平台后再试。"
                )
            )
        ]
    }


async def ask_for_image(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'ask' 分支落点：首轮无图无母题无题组，催老师贴题图。

    🔴 必须是真节点（不能直连 END）—— 否则首轮没有任何节点产消息，回复为空，
    '没图催' 提示从未触发（PRD-C-009 G15 红的 root cause）。
    """
    return {
        "messages": [
            AIMessage(
                content=(
                    "我还没看到题目图。请先贴一张题目图的 OSS URL，我才能开始举一反三。\n\n"
                    "（贴图后我会读图、锚定年级/考点/题型，再按你要的数量出变式题。）"
                )
            )
        ]
    }


# ---------------------------------------------------------------------------
# 图（StateGraph）
# ---------------------------------------------------------------------------
graph = StateGraph(VariantState)
graph.add_node("analyze", analyze)
graph.add_node("classify", classify)
graph.add_node("clarify", clarify)
graph.add_node("generate", generate)
graph.add_node("gene_gate", gene_gate)  # 闸A·基因闸（新变式 → 平行度比对 → 闸B）
graph.add_node("solve_explain", solve_explain)
graph.add_node("assemble", assemble)
# 交互层节点（多轮 WAIT 后的下一句）
graph.add_node("parse_instruction", parse_instruction)
graph.add_node("answer_question", answer_question)
graph.add_node("exec_remove", exec_remove)
graph.add_node("exec_regenerate", exec_regenerate)
graph.add_node("exec_add", exec_add)
graph.add_node("exec_reorder", exec_reorder)  # P9·指令排序（纯代码重排，不过 assemble）
graph.add_node("exec_solution_only", exec_solution_only)  # 整改3·解法修正（题面留只改解析+重跑闸B）
graph.add_node("patch", patch)
graph.add_node("ask_clarify", ask_clarify)
graph.add_node("persist_to_bank", persist_to_bank)
graph.add_node("ask_for_image", ask_for_image)
graph.add_node("require_login", require_login)

graph.set_conditional_entry_point(
    route_entry,
    {
        "analyze": "analyze",
        "generate": "generate",
        "parse": "parse_instruction",
        # 🔴 'ask' 必落真节点（ask_for_image），不能直连 END —— 否则首轮无节点产消息，回复为空
        "ask": "ask_for_image",
        # 🔴 身份硬闸：无登录态 → 提示重登（同上，必落真节点）
        "auth": "require_login",
    },
)
graph.add_edge("ask_for_image", END)
graph.add_edge("require_login", END)

# analyze：非题目图/读图失败 → 直接 END（已吐友好报错）；成功 → classify
def after_analyze(state: VariantState) -> Literal["classify", "done"]:
    # analyze 友好报错时会塞 messages（且未产 analysis）→ 结束本轮等待
    if not state.get("analysis"):
        return "done"
    return "classify"


graph.add_conditional_edges("analyze", after_analyze, {"classify": "classify", "done": END})
graph.add_conditional_edges(
    "classify", gate_after_classify, {"generate": "generate", "clarify": "clarify"}
)
graph.add_edge("clarify", END)


# generate：裸奔兜底时只吐消息、无 items → 结束；正常 → 闸A 基因闸 → 闸B solve_explain
def after_generate(state: VariantState) -> Literal["gene_gate", "done"]:
    if not state.get("items"):
        return "done"
    return "gene_gate"


graph.add_conditional_edges(
    "generate", after_generate, {"gene_gate": "gene_gate", "done": END}
)
graph.add_edge("gene_gate", "solve_explain")
graph.add_edge("solve_explain", "assemble")
graph.add_edge("assemble", END)

# --- 交互层路由（设计 §3 mermaid：WAIT → parse → 5 意图分诊） ----------------
graph.add_conditional_edges(
    "parse_instruction",
    route_after_parse,
    {
        "patch": "patch",
        "dispatch": "dispatch",  # 编辑意图 → 三层漏斗节点收口（remove/regenerate/add）
        "answer": "answer_question",
        "save": "persist_to_bank",
        "solution_only": "exec_solution_only",  # 整改3·解法修正（题面留只改解析）
        "ask_clarify": "ask_clarify",
    },
)


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


# 编辑意图先经 dispatch 漏斗：把 route_after_parse 的 "dispatch" 桥到三原语。
# 用一个轻量调度节点统一收口（避免 route_after_parse 直连 exec_remove 误派）。
graph.add_node("dispatch", lambda state: {"messages": []})
graph.add_conditional_edges(
    "dispatch",
    route_dispatch,
    {
        "exec_remove": "exec_remove",
        "exec_regenerate": "exec_regenerate",
        "exec_add": "exec_add",
        "exec_reorder": "exec_reorder",
        "ask_clarify": "ask_clarify",
    },
)

# 三原语收口：regenerate/add 产**新变式** → 先过闸A基因闸再到闸B；
# remove 只删不产新题 → 直连 solve_explain（旧题带 check+gene 双标，两闸都原样通过）。
graph.add_edge("exec_remove", "solve_explain")
graph.add_edge("exec_regenerate", "gene_gate")
graph.add_edge("exec_add", "gene_gate")
# 🔴 reorder 只挪槽位、不产新题、不重判 → 直连 END（已自发 artifact 整帧；**不过 assemble**，
# 否则 assemble 的默认难度升序排序会覆盖老师手排）。
graph.add_edge("exec_reorder", END)
# 整改3·解法修正：节点内已逐题重写解析 + 重跑闸B（每题 check 已定）→ 过 assemble 收口快照
# （刷新头部 chip/状态计数 + 题型规范 + artifact 整帧）。
graph.add_edge("exec_solution_only", "assemble")

# 答疑/clarify → END（不改 items，回等待下一句）
graph.add_edge("answer_question", END)
graph.add_edge("ask_clarify", END)
graph.add_edge("persist_to_bank", END)


# patch：改了硬锚（清 items + mother_confirmed=False）→ 重锚重造走 classify；
#        没改（仅回问消息，items 仍在）→ END 等下一句。
def after_patch(state: VariantState) -> Literal["classify", "done"]:
    if state.get("mother_confirmed") is False and not state.get("items"):
        return "classify"
    return "done"


graph.add_conditional_edges("patch", after_patch, {"classify": "classify", "done": END})

# 🔴 不在此 compile checkpointer：service lifespan 注入 saver（按 thread_id 持久 state）
variant = graph.compile()
variant.name = "variant"
