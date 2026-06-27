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
  LLM 外呼中转走默认（别套 trust_env=False，那是治本地 localhost 的）。

checkpointer 不在此 compile（service lifespan 注入 saver；多轮 state 按 thread_id 持久）。
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import difflib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config
from langgraph.config import get_stream_writer
from langgraph.graph import END, MessagesState, StateGraph

from agents import (
    conv_trace,
    dna_extract,
    math_verify,
    model_anchor,
    mother_opus,
    mother_precheck,
)
# 🔴 PRD-C-100 B1a 塌缩入口节点（懒导入 variant 防循环：variant_entry 模块体不 import variant，
#   仅其节点函数运行期反向用 variant 机具）。此处 import 仅取节点函数挂图，不触发循环。
from agents.variant_entry import mother_opus_entry
from agents.qtype_format import format_by_qtype
from agents.variant_support import (
    _is_review_book,
    anchor_subject,
    chapter_name_for_id,
    leaf_pool_for_grade,
    record_link_manifest,
)
# 🔴 PRD-C-104 B2a：shared/ 抽出的引擎内部 helper（纯搬零改），re-export 回本模块面 →
#   service.py / variant_entry.py 仍按 variant.X / V.X 解析得到，外部零感。
from agents.variant.shared.sanitize import (
    _PAREN_MATH_RE,
    _BRACKET_MATH_RE,
    _LITERAL_NL_RE,
    _MATH_SPLIT_RE,
    _BARE_SPACING_RE,
    _GLUED_GEOM_RE,
    _BARE_DEGREE_RE,
    _fix_glued_inside_math,
    _strip_bare_spacing_outside_math,
    _sanitize_rich_text,
    _sanitize_item,
    join_skeleton,
)
from agents.variant.shared.budget import (
    _budget_ctx,
    _budget_begin,
    _budget_tick,
    _budget_exhausted,
    _budget_bind,
)
from agents.variant.shared.ruoyi import (
    RuoyiClient,
    persist_items,
    build_mother_bo,
    build_create_bo,
    build_update_bo,
    build_block_json,
)
# 🔴 PRD-C-104 B2b：llm 出口簇 + 纯 SSE 发射器抽到 shared/（contextvar 命脉），re-export 回本模块面。
from agents.variant.shared.llm import (
    _content_text,
    _LLM_TRACE_ENABLED,
    _LLM_TRACE_PATH,
    _TRACE_MARKERS,
    _msg_text,
    _trace_label,
    _serialize_request,
    _trace_llm,
    _ainvoke_text,
    _JSON_FENCE,
    _parse_json,
)
from agents.variant.shared.emit import (
    _emit_stage,
    _emit_error,
    _emit_need_confirm,
    _emit_reject,
    _emit_reasoning,
    _emit_richtext_stem,
)
from core import get_model, relay_pool, settings
from core import difficulty  # 🔴 PRD-C-103 WS1：确定性难度判档（grade_observed，表反控的代码点）

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

# P13 预算闸（PRD-C-013）：state 级 LLM 调用计数器（per-round 重置）。
#   🔴 PRD-C-104 B2a：_budget_ctx / _budget_begin / _budget_tick / _budget_exhausted /
#   _budget_bind 已抽到 agents.variant.shared.budget（纯搬零改），顶部 re-export 回本模块。

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
# 🔴 BUG-09 手动验算（2026-06-19）：auto_verify=False（产品默认）下生成不自动跑 sympy/回炉，
#   每题挂 pending（待老师手动点验算）。判分铁律不破——pending 只是「还没验」，不是判结果；
#   老师点 /variant/verify-one 或 /variant/reverify 才真跑 sympy（判决仍只读 verdict）。
VERIFY_PENDING = "pending"  # 待手动验算（auto_verify=False 生成态；非判分态）

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
TIER_PENDING = "pending"  # 中性·待手动验算：auto_verify=False 生成态（FE 渲染「待验算」徽章 + 验算按钮）

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

# 🔴 PRD-C-017 B5·阶段灯中性态（status 枚举新增）：正常暂停（等老师确认年级章 / 等老师点
#   「开始举一反三」）不是告警，发 "await"（FE 配中性色 + "待确认/待开始"文案），不发 warn、
#   不发"已中断"。带图打回(reject) 那个 warn 保留（真要拦）。FE 需对 "await" 配色。
STAGE_AWAIT = "await"
# 🔴 BUG-01/02（2026-06-19）·状态枚举常量化（避免散落字面量）：figure/review 两节点新增帧用它们。
STAGE_RUNNING = "running"
STAGE_DONE = "done"


# 🔴 BUG-09（2026-06-19）：程序验算「何时跑」开关——auto_verify。
#   - 产品默认 = 手动（False）：生成秒到就绪、每题 pending、老师按需点验算（service 入口注入 False）。
#   - 节点级缺省 = True：直调节点（单测 / 未走 service 入口的内部路径）不传该键时保持既有自动验算行为，
#     不破现有测试与回炉/守恒链。判分铁律不破——本开关只改「何时验」，不改「怎么判」（仍只读 sympy verdict）。
def _auto_verify_on(config: RunnableConfig | None = None) -> bool:
    """读 auto_verify（config.configurable 优先，回退 graph 上下文 ensure_config()）。
    键缺省 → True（节点级自动验算，兼容直调）；service 入口对真实请求显式注入 False = 手动。"""
    try:
        conf = ((config or {}).get("configurable") or {}) if config else {}
    except Exception:  # noqa: BLE001
        conf = {}
    if "auto_verify" not in conf:
        try:
            conf = (ensure_config() or {}).get("configurable", {}) or {}
        except Exception:  # noqa: BLE001
            conf = {}
    if "auto_verify" not in conf:
        return True  # 节点级缺省自动（兼容直调单测/内部路径）
    return bool(conf.get("auto_verify"))


# 🔴 PRD-C-103 WS4·AC10/D5：sympy 硬门总闸。默认 **关**（settings.VARIANT_SYMPY_GATE_ON=False）。
#   关 → sympy 验算照算（产 check 徽章/computed/解析），但 **fail/退化构型一律不剔除变式、不卡流程**，
#       降级标 ⚠ 放行交人审（线上正确性兜底走人工审核）。
#   开 → 恢复旧硬门语义（退化构型超限剔除）。config.configurable.sympy_gate 可逐请求覆盖（优先于
#       settings 默认），便于单测/回归在不改全局 .env 的前提下临时开硬门验旧行为。
#   判分铁律不破：只改「sympy 判出 fail/退化时是否硬拦」，不改「怎么判」（仍只读 verdict）。
def _sympy_gate_on(config: RunnableConfig | None = None) -> bool:
    """sympy 是否作硬门（剔除退化构型）。config.configurable.sympy_gate 优先，缺省回退
    settings.VARIANT_SYMPY_GATE_ON（默认 False=不硬门）。"""
    try:
        conf = ((config or {}).get("configurable") or {}) if config else {}
    except Exception:  # noqa: BLE001
        conf = {}
    if "sympy_gate" not in conf:
        try:
            conf = (ensure_config() or {}).get("configurable", {}) or {}
        except Exception:  # noqa: BLE001
            conf = {}
    if "sympy_gate" in conf:
        return bool(conf.get("sympy_gate"))
    return bool(getattr(settings, "VARIANT_SYMPY_GATE_ON", False))


# ---------------------------------------------------------------------------
# 🔴 PRD-C-104 B1：state 契约（merge reducer + VariantState）已抽到 variant/state.py，
#    本处 re-export 保 service.py / variant_entry.py / 本模块内其余引用零感（纯搬零改）。
# ---------------------------------------------------------------------------
from agents.variant.state import (  # noqa: E402
    VariantState,
    merge_items,
    _item_stem_norm,
    _ITEMS_PRESERVE_ALWAYS,
    _ITEMS_PRESERVE_IF_SAME_STEM,
)

# ---------------------------------------------------------------------------
# 🔴 PRD-C-104 B1：prompt 字符串常量已抽到 variant/prompts.py，本处 re-export 保零感。
#    须在 GENERATE/REGEN/EXTRACT/ADD 等拼接型 prompt 使用 _PAYLOAD_CONTRACT 等片段前导入。
# ---------------------------------------------------------------------------
from agents.variant.prompts import (  # noqa: E402
    ANALYZE_PROMPT,
    _PAYLOAD_CONTRACT,
    _QTYPE_CONTRACT,
    _DIFFICULTY_RUBRIC,
    _FIGURE_SPEC_CONTRACT,
    KNOBS_PROMPT,
    SOLVE_PROMPT,
    _GRADE_DIFFICULTY_PROMPT,
    PARSE_PROMPT,
    ANSWER_PROMPT,
    SOLUTION_ONLY_PROMPT,
    REVISE_FIELD_PROMPT,
    _REWRITE_SOLVE_PROMPT,
)


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
#   hard_anchor  【年级】→ 改 = 立即解冻重锚（不进 dirty，缺口7）
#   soft_regen   【主考点/题型/难度/考察类型/场景】→ 改 = 标 dirty，点「重生」_regen_once 重出
#                （main_kp 见 A-2/契约C4：母题级守恒维，改 = 标全组 dirty + 可回退，不自动重出）
#   rewrite_solve【解法骨架/models】→ 改 = 重写解析过闸B，置 dirty 直到重写完（D-merge9）
#   meta         【标签/副考点】→ 改 = 只标注即时生效，不进 dirty
# 🔴 维名用 edit-dna 契约维 key（main_kp/grade/qtype/difficulty/exam_type/scene/skeleton/
#    models/tags/secondary_kps）。批1 仅建映射；批4 据此驱动 dirty/重生。
REGEN_CLASS: dict[str, str] = {
    # 🔴 A-2/契约C4（PRD-A-018）：main_kp 由 hard_anchor → soft_regen（单一真相，前后端逐字一致）。
    #   改主考点 = 标 dirty + await 显式重生 + 可回退（与其余 soft_regen 维一致），不再走「清 items
    #   立即整组重锚」的 hard_anchor 语义，也不靠 BE 特判补丁（旧 main_kp 特例分支已删）。
    "main_kp": "soft_regen",
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
# 🔴 PRD-C-017 B5-fix6·此集合只管「冻结 setter + 留痕」（_dna_fact_edit 据它放行写入）；
#   「改了是否波及下游 dirty」另由 _MOTHER_DIRTY_PROP_FIELDS 决定（hard_points 在此但不波及）。
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


def mark_content_dirty_if_persisted(item: dict[str, Any]) -> None:
    """🔴 PRD-A-018 M1③/F16：已入库题(persisted)的内容(题面/答案/解析)被编辑/重生/重写解析后，
    标 `_content_dirty=True` → 让 persist_to_bank / persist_one_to_bank 不再「已入库就跳过」，
    而是放行走「覆盖原行 update by _persist_id」（库行内容同步更新，不重复落行）。

    - 仅对**已入库**题打标（未入库题首次入库本就走 create，无需标；避免污染未入库流）。
    - 与 dna_dirty 解耦：dna_dirty=改维待重生→拒入库；_content_dirty=内容已编辑待覆盖→允许覆盖入库。
    - `_content_dirty` 是内部键，不入库（build_create_bo/build_update_bo 白名单挡）；落库时清。
    """
    if item.get("persisted"):
        item["_content_dirty"] = True


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
# secondary_kps / exam_type 是守恒白名单/考察类型基准；skeleton 是基因基准。
# 🔴 PRD-C-017 B5-fix6·hard_points 已剔除：难点是「只标注」维（meta），重生 prompt 不读它，
#   作为母题基准波及下游 = 空转重出（结果等价）。保留在 _DNA_CONSERVE_FIELDS（冻结 setter 留痕）
#   但不在波及/基准集合里。
_MOTHER_BASELINE_DIMS: tuple[str, ...] = ("secondary_kps", "exam_type", "skeleton")

# 🔴 PRD-C-017 B5-fix6·母题守恒维「真正会波及下游 dirty」的集合 = 守恒维 − main_kp（硬锚走解冻重锚）
#   − hard_points（纯标注，不波及）。edit_dna_state 据此置 mother_dirty + 标下游变式 dirty。
_MOTHER_DIRTY_PROP_FIELDS: tuple[str, ...] = tuple(
    f for f in _DNA_CONSERVE_FIELDS if f not in ("main_kp", "hard_points")
)


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


# 🔴 PRD-C-104 B2b：_content_text / _LLM_TRACE_* / _TRACE_MARKERS / _msg_text / _trace_label /
#   _serialize_request / _trace_llm / _ainvoke_text / _JSON_FENCE / _parse_json 已抽到
#   agents.variant.shared.llm（纯搬零改，_LLM_TRACE_PATH 适配 parents[3]）→ 顶部 re-export 回本模块。


# --- 富文本净化（用户反馈 2026-06-11：解析裸字符不渲染的根因） -----------------
#   🔴 PRD-C-104 B2a：正则常量 + _fix_glued_inside_math / _strip_bare_spacing_outside_math /
#   _sanitize_rich_text / _sanitize_item / join_skeleton 已抽到 agents.variant.shared.sanitize
#   （纯搬零改），顶部 re-export 回本模块。


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
# 🔴 F14/PRD-A-018：route_entry 用 _extract_image_url 当「跨轮新图=新母题」优先级最高的判据。
#   若裸认任意 http(s) 串，老师在编辑指令里夹一条普通参考链接（『第2题参考 https://example.com 改难点』）
#   就会被劫持成新母题、整组题被清。收紧成只认**图片 URL**：① 路径以图片后缀结尾（允许 ?query），
#   或 ② 来自图床白名单域名（OSS/对象存储——这些 URL 常无扩展名却是题图）。普通参考链接（无图片后缀、
#   非图床域名）不再被当题图，落 parse 让老师正常编辑，不清组。
_IMG_SUFFIX_RE = re.compile(
    r"https?://[^\s)>'\"]+\.(?:png|jpg|jpeg|webp)(?:\?[^\s)>'\"]*)?",
    re.IGNORECASE,
)
# 图床/对象存储域名白名单（无扩展名的题图 URL 兜底，如 OSS 签名直链）。新增图床在此加。
_IMG_HOST_HINTS = ("aliyuncs.com", "myqcloud.com", "qiniucdn.com", "oss")


def _extract_image_url(text: str) -> str | None:
    """从用户消息抽 OSS 题图 URL（MVP 贴 URL，file_uploader future）。

    🔴 F14：只认图片 URL —— 图片后缀（.png/.jpg/.jpeg/.webp，允许 ?query）优先；其次图床白名单域名。
    普通 http 链接（编辑指令里夹的参考链接）一律不认，避免被路由劫持成新母题清整组。
    """
    t = text or ""
    m = _IMG_SUFFIX_RE.search(t)
    if m:
        return m.group(0)
    # 无图片后缀 → 仅当 URL 域名命中图床白名单才认（OSS 签名直链常无扩展名）。
    for cand in _URL_RE.findall(t):
        host = cand.split("//", 1)[-1].split("/", 1)[0].lower()
        if any(hint in host for hint in _IMG_HOST_HINTS):
            return cand
    return None


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
# 🔴 PRD-C-104 B2b：纯 SSE 发射器 _emit_stage / _emit_error / _emit_need_confirm / _emit_reject /
#   _emit_reasoning / _emit_richtext_stem 已抽到 agents.variant.shared.emit（纯搬零改）→ 顶部
#   re-export 回本模块。_artifact_payload / _emit_artifact / _emit_mother_card 依赖 stage1/mother
#   簇 helper（本批不搬），留在本模块以免循环 import（见 shared/emit.py 抬头说明）。


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


# 🔴 PRD-C-104 B4：_item_dna 已抽到 stage1_anchor/label.py（末尾 re-export，纯搬零改）。


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
            # 🔴 BUG-09（2026-06-19）：手动验算态机器抓手——pending=待老师手动点验算（auto_verify=False
            #   生成态），done=已过验算/已定状态（含 sympy_pass/self_ok/proof/manual 等）。FE 据它渲染
            #   「待验算」徽章 + 验算按钮（点 → /variant/verify-one 或 /variant/reverify）。
            "verify_status": (
                "pending"
                if (chk.get("verify") == VERIFY_PENDING or chk.get("tier") == TIER_PENDING)
                else ("done" if chk else "pending")
            ),
            # 4d 外显层级（FE 徽章唯一依据；旧线程恢复无 tier → FE 按「只说好」兜底）
            "tier": chk.get("tier") or None,
            # 🔴 PRD-A-017 R1·验算可查真证据透传（FE 验算徽章展开层）：math_verify.verify() 纯 sympy
            #   算出的逐步核对话术 detail（如 computed=46.0, claimed=46.0, tol=1e-6: within tolerance）
            #   + 真算出的解集/真值 computed（如 [46]）。证明/开放/作图类只走 review、无 sympy 证据
            #   → chk 无这俩键 → None（如实留空不伪造，禁假数据铁律 §0.5）。零核心逻辑改、零编造。
            "verify_detail": chk.get("verify_detail") or None,
            "verify_computed": chk.get("computed") or None,
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
        # 🔴 PRD-C-100 BC2：变式配图 OSS url（FE compose_variant_figure 产 base64 → uploadMotherImage
        #   传 OSS → 经 /variant/set-figure-url 回写 state.items[i].figure_url）。入库时
        #   build_create_bo 据它产 A-015 image 块；透传给 FE 用于会话恢复后保持配图态。缺则 None。
        cell["figure_url"] = str(it.get("figure_url") or "") or None
        # 🔴 PRD-A-021 R2b·U1：生成态配图 PNG base64 透传（会话恢复/刷新重建配图显示态）。
        #   旧实现生成态 base64 仅活在 FE 内存 variantFigures[idx].png，切 tab/刷新即丢且只在入库才
        #   传 OSS。现 FE 造图认账后回写 state（经 set-figure-url 带 figure_base64）→ 落 checkpoint →
        #   /variant/artifact 恢复时随帧透出，FE 无 OSS url 也能从 base64 重建配图。缺 → None。
        cell["figure_base64"] = str(it.get("figure_base64") or "") or None
        # 🔴 PRD-A-018 治本A·figure_spec 透传（出题节点产的配图自然语言描述）：随帧上屏 + 会话恢复保留，
        #   供 compose 从 state 取来照画（service /variant/compose-figure 自取，FE 不必传）。展示/内部键，
        #   不入 biz_question 旧字段（同 figure_url；build_create_bo 显式白名单天然不外漏）。缺则 None。
        # 🔴 round4：figure_spec 现可为 str（旧）或 dict（新 {"layout":..,"angle_labels":[..]}）——
        #   原样透传（dict 不 str 化），FE/会话恢复保留结构、compose 取来按类型分发。缺/空 → None。
        _cell_fs = it.get("figure_spec")
        if isinstance(_cell_fs, dict) and (_cell_fs.get("layout") or _cell_fs.get("angle_labels")):
            cell["figure_spec"] = _cell_fs
        else:
            cell["figure_spec"] = (str(_cell_fs).strip() or None) if isinstance(_cell_fs, str) else None
        # 🔴 PRD-C-100 BC3：已入库题在库雪花 id（= _persist_id 内部簿记键，persist_one/全部入库后回写）。
        #   FE 据它把已入库变式 round-trip 进 A-015 网格编辑器（/question/editor/:id 按 questionId 载 blockJson
        #   编辑 → /teacher/question/update-block 存）。未入库题为 None（无 questionId 不能进编辑器）。
        cell["question_id"] = (
            str(it.get("_persist_id")) if it.get("_persist_id") not in (None, "") else None
        )
        # 🔴 PRD-C-100 BC3：本题被老师手动排版过（A-015 网格编辑器存过 blockJson）→ 卡显「手动排版」印记 +
        #   重生前二次确认（会覆盖手改布局）。复用 manual_edited 印记语义（与内容编辑共用一面旗，
        #   都表「老师亲手改过，重生需确认」）；manual_block 单独标「排版」态供文案区分。
        cell["manual_block"] = bool(it.get("manual_block"))
        out_items.append(cell)
    # 🔴 批4·组级重生态：mother_dirty（母题守恒维改）+ regen_pending（待重生集合 1-based 题号）。
    #   FE 据 regen_pending 非空 → 「重生」按钮可点 + 入库按钮禁用（致命① dirty 拒入库视觉）。
    mother_dirty = bool((state.get("mother_dna") or {}).get("dirty"))
    regen_pending = dirty_item_indexes(items)
    # 🔴 批5·合并确认面（G10/G11）数据源：透传 mother_confirm（flags/needs_confirm/
    #   confirmed_dims/audit_ref），FE pickMotherConfirm 解析弹合并确认面。缺省 → None
    #   （旧 FE 不读不坏，向后兼容）。
    mother_confirm = state.get("mother_confirm") or None
    # 🔴 2026-06-17：母题卡专帧 sticky 化（修「母题入库·题面尚未产出」根因）。原 mother_card 只在
    #   classify 经 _emit_mother_card 发一次（turn1）；turn2「开始举一反三」的 assemble/persist 帧
    #   走 _artifact_payload 不带 mother_card → FE artifact.value 整帧替换后 header.mother_card=null
    #   → pickMotherCard 路①失效、mc.stem 丢 → 母题入库被拦。修法：每帧都附 _build_mother_card(state)
    #   （纯函数；无 mother_dna→None，FE 兼容 null 不回归）。让母题专帧贯穿全生命周期、stem 永在。
    mother_card = _build_mother_card(state)
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
            # 🔴 母题专帧 sticky：每帧附母题卡（含 stem），FE pickMotherCard 路①全程可用
            "mother_card": mother_card,
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
# 🔴 PRD-C-017 B3.5·母题卡专帧（mother_card「先出」）——契约 §10 / AC4 / G12。
# 心智：classify(opus 解题打标) → generate(变式) **直连**，旧路母题 DNA 只能从变式
#   items[0].dna（组级共享）外显，母题做不到真正「先出」，且母题题面/solved_answer/副考点 id/
#   anchor 章 id/need_anchor_review 在 items 里缺失 → 母题入库被拦。本帧在 classify 末尾
#   （opus 产出 mother_dna 之后、generate 之前）单发一帧，把母题卡全字段提前外显。
#
# 🔴 帧机制 = 复用 _emit_artifact 的 custom 帧 header 透传：放进 artifact.header.mother_card
#   （与 mother_confirm 同载体），FE pickMotherCard 路①优先读 header.mother_card。
# 🔴 时序「先出」= 在 classify return 前 emit；classify→generate(出 items) 是后续节点，
#   故本帧必早于任何变式 item 帧。
# 🔴 复用 classify 已产出的 mother_dna（含 opus dna/解答/锚定），绝不重调 opus。
# ---------------------------------------------------------------------------
def _mother_chapter_name(state: VariantState) -> str | None:
    """🔴 BUG-08（2026-06-19）：从 state 取母题章名（纯函数·零 IO，可单测）。

    源优先级（皆已由 classify/entry 异步节点用 chapter_name_for_id / opus 判定回写进 state）：
      ① state.confirmed_chapter_name（确认章人话名，classify 回写）；
      ② analysis.chapter.value（opus 判定章文本 / classify 反查章名）。
    都空 → None（调用方据此不加 chapter_name，绝不伪造）。
    """
    direct = str(state.get("confirmed_chapter_name") or "").strip()
    if direct:
        return direct
    analysis = state.get("analysis") or {}
    chap = analysis.get("chapter")
    if isinstance(chap, dict):
        v = str(chap.get("value") or "").strip()
        if v:
            return v
    return None


def _build_mother_card(state: VariantState) -> dict[str, Any] | None:
    """组母题卡 payload（契约 §10）。纯函数·零 IO（可单测）。

    数据源 = state.mother_dna（B1 classify 写入：stem/answer/analysis/solution_skeleton/
    solved_answer + dna 契约 v1 + need_anchor_review）+ analysis（年级/锚定）+ confirmed_chapter_id。
    无 mother_dna（库内母题直进 generate / 早退路径）→ 返回 None（调用方不发帧，FE 走 items[0] 兜底）。

    🔴 dna 子对象键名对齐 FE pickDna：main_kp=考点名(str)、main_kp_id=叶子 id、
       secondary_kps=[{id,name}]（FE strArr 取 name、顶层抠 id 作 secondaryKpIds）、
       exam_type/skeleton/hard_points/tags/scene/models。母题骨架 list → 换行拼 str（与 _item_dna 同）。
    """
    mdna = state.get("mother_dna") or {}
    if not mdna:
        return None
    dna = mdna.get("dna") or {}
    if not isinstance(dna, dict):
        dna = {}
    analysis = state.get("analysis") or {}
    grade_node = analysis.get("grade") or {}

    main_kp_obj = dna.get("main_kp") or {}
    if not isinstance(main_kp_obj, dict):
        main_kp_obj = {}
    main_kp_name = str(main_kp_obj.get("name") or "") or None
    main_kp_id = str(main_kp_obj.get("id") or "") or None

    secondary_kps = _norm_secondary_kps(dna.get("secondary_kps"))

    # 母题骨架：dna.skeleton 是 list[str]（opus 步骤序列）→ FE 要 str，换行拼（与 _item_dna 同）。
    skeleton_raw = dna.get("skeleton")
    if isinstance(skeleton_raw, list):
        skeleton = "\n".join(str(s) for s in skeleton_raw if str(s).strip())
    else:
        skeleton = str(skeleton_raw or "")
    # 母题题面/解答优先取 opus 富文本骨架字段，其次 mother_dna.solution_skeleton。
    solution_skeleton = mdna.get("solution_skeleton") or skeleton or None

    models = [
        {"id": str(m.get("id") or ""), "name": str(m.get("name") or "")}
        for m in (dna.get("models") or [])
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    ]

    difficulty = dna.get("difficulty")
    if not isinstance(difficulty, int):
        md = mdna.get("difficulty")
        difficulty = md if isinstance(md, int) else None

    # 🔴 PRD-C-017 B5 问题3·答案核齐根因修：opus 把标准答案放 richText.answer（→ mdna.answer），
    #   solvedAnswer（→ mdna.solved_answer）是它的"解出值"草稿、常为空或更简略。旧版母题卡只外显
    #   solved_answer 顶层 → opus solvedAnswer 留空时母题卡"答案为空"（真机症状）。修法：① 顶层新增
    #   `answer`（标准答案，FE 优先映射它）；② solved_answer 兜底回退 answer（两者择有值者），保证
    #   "答案"区永不空（只要 opus 给了 richText.answer 或 solvedAnswer 任一）。
    std_answer = str(mdna.get("answer") or "") or None
    solved_answer = str(mdna.get("solved_answer") or "") or None

    # need_anchor_review：闸B 留空标记（mother_dna 或 dna 任一标了即为真）。
    need_review = bool(mdna.get("need_anchor_review") or dna.get("need_anchor_review"))

    # anchor 章 id：B2 确认章优先（confirmed_chapter_id），缺则年级册 code 前缀。
    chapter_id = (
        str(state.get("confirmed_chapter_id") or "").strip()
        or str(grade_node.get("code") or "").strip()
        or None
    )
    grade_book_id = str(grade_node.get("code") or "").strip() or None
    confidence = grade_node.get("confidence")

    return {
        # 顶层（FE pickMotherCard 直读）
        "stem": str(mdna.get("stem") or "") or None,  # 🔴 入库靠它，缺则入库被拦
        "analysis": str(mdna.get("analysis") or "") or None,  # 🔴 opus 解析富文本（入库存它，非骨架顶替）
        "solution_skeleton": solution_skeleton,
        # 🔴 B5 问题3：answer=标准答案（FE 优先映射）；solved_answer 兜底回退 answer（永不空）。
        "answer": std_answer,
        "solved_answer": solved_answer or std_answer,
        "qtype": str(dna.get("qtype") or "") or None,
        "difficulty": difficulty if isinstance(difficulty, int) else None,
        "main_kp": main_kp_name,  # anchorKp（考点名）
        "need_anchor_review": need_review,
        # 🔴 PRD-C-100 B3/B6：带图母题钩子——FE 据 mother_has_figure 自动调 crop_mother_figure
        #   （/variant/compose-figure mode=crop_mother，传 mother_image_url）显示母题切图（AC4）。
        "mother_has_figure": bool(state.get("mother_has_figure")),
        "mother_image_url": str(state.get("image_url") or "") or None,
        # 🔴 PRD-A-018 A-22：透传母题入库态（恢复历史会话据此重建，免显「未入库」/可重复入库）。
        #   雪花 id 发 string 防 JS Number 精度丢失（FE pickMotherCard 读 mother_question_id/persisted）。
        #   未入库 → None/False（禁假数据，FE 防御性兜 null）。
        "mother_question_id": (
            str(mdna.get("mother_question_id")) if mdna.get("mother_question_id") else None
        ),
        "persisted": bool(mdna.get("mother_question_id")),
        # 10 维 DNA（FE pickDna 解析；键名对齐 _item_dna）
        "dna": {
            "main_kp": main_kp_name,
            "main_kp_id": main_kp_id,
            "secondary_kps": secondary_kps,  # [{id,name}]：FE 取 name + 顶层抠 id
            "qtype": str(dna.get("qtype") or "") or None,
            "exam_type": str(dna.get("exam_type") or "") or None,
            "difficulty": difficulty if isinstance(difficulty, int) else None,
            "scenario": str(dna.get("scene") or "") or None,
            "hard_point_count": int(dna.get("hard_point_count") or len(dna.get("hard_points") or [])),
            "breakthrough_points": [str(h) for h in (dna.get("hard_points") or []) if str(h).strip()],
            "hard_points": [str(h) for h in (dna.get("hard_points") or []) if str(h).strip()],
            "skeleton": skeleton or None,
            "models": models,
            "tags": [str(t) for t in (dna.get("tags") or []) if str(t).strip()],
        },
        # 锚定（FE anchor.chapter_id → anchorChapterId）
        "anchor": {
            "grade_book_id": grade_book_id,
            # 🔴 2026-06-17：补年级册名(八年级下册)，FE 显示它而非 ID(3082)——帧原先只发 code
            "grade_book_name": str(grade_node.get("value") or "").strip() or None,
            "chapter_id": chapter_id,
            # 🔴 BUG-08（2026-06-19）：母题卡回灌章名（FE 显示章名而非 chapter_id）。源 = classify/entry
            #   节点已用 chapter_name_for_id / opus 判定回写进 state（analysis.chapter.value 或顶层
            #   chapter_name）。空就不加（不伪造，禁假数据）。
            **({"chapter_name": _chapter_name} if (_chapter_name := _mother_chapter_name(state)) else {}),
            "confidence": confidence if isinstance(confidence, (int, float)) else None,
            "need_anchor_review": need_review,
        },
    }


def _emit_mother_card(state: VariantState) -> None:
    """发母题卡专帧（mother_card「先出」·AC4/G12）。机制 = custom 帧 header.mother_card 透传
    （复用 _emit_artifact 的 header 载体，FE pickMotherCard 路①）。items 留空 → 本帧只携母题卡，
    早于任何变式 item 帧。同 _emit_stage 双层静默吞：无 runtime context / 组卡为空 → no-op。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    try:
        card = _build_mother_card(state)
        if not card:
            return  # 无 mother_dna（库内母题直进 generate）→ 不发，FE 走 items[0] 兜底
        writer(
            ChatMessage(
                content=[{"artifact": {"items": [], "header": {"mother_card": card}}}],
                role="custom",
            )
        )
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点（母题卡是增强不是关卡）
        pass


def _emit_figure_stage(state: VariantState) -> None:
    """🔴 BUG-01/02（2026-06-19）·「母题切图」节点专帧（key="figure"）：母题就绪、figure 判定之后发。

    🔴 C2/A-11（PRD-A-018）·figure key 三义拆开（治标·不入 graph）：母题切图灯的**唯一真值 key =
       `figure-mother`**，由宿主（FE）在真实调 crop_mother_figure 切图后 push。本图内节点**不得**抢发
       一帧 done 压过真实切图：
      - 带图母题（mother_has_figure=True）：真切图还没发生（那是宿主 POST crop 的事）→ **不发** done
        「已切出」（旧实现这帧是假完成，会让状态条母题切图灯先绿、压过随后宿主推的 figure-mother 真值）。
        本节点只在 state 里有 mother_has_figure 钩子，FE 据它自动发起切图，灯由 figure-mother 帧驱动。
      - 纯文本/无图母题（mother_has_figure=False）：确实无图可切 → 发 STAGE_DONE 跳过态，且落在
        `figure-mother` key（与 FE 唯一切图灯 key 对齐），detail「无图可切」。**绝不**在 mother_has_figure
        误判 false 时显「无需切图」压过真实切图——故只在确无图钩子时发跳过，发于唯一 key 不再产生矛盾帧。
    🔴 独立于 classify 三锚帧（不与「锚定考点」共 key）。同 _emit_stage 双层静默吞：无 runtime context → no-op。"""
    if bool(state.get("mother_has_figure")):
        # 带图 → 切图是宿主真活，本节点不抢发假 done；灯等宿主 figure-mother 真值帧。
        return
    # 纯文本 → 无图可切，发跳过态于唯一切图灯 key（figure-mother），不压真实切图。
    _emit_stage("figure-mother", "母题切图", STAGE_DONE, "纯文本母题，无图可切")


# ---------------------------------------------------------------------------
# Router（入口分诊：登录? 有图? 在途母题? 库内母题跳 analyze/classify）
# ---------------------------------------------------------------------------
def _editor_op(config: RunnableConfig | None) -> dict[str, Any] | None:
    """🔴 PRD-A-021 S1：从 config.configurable 取「编辑/验算」结构化 op（让编辑走 graph 发真帧）。

    形如 {"kind": "revise"|"regen"|"edit-item"|"reverify", "index": int, ...}。缺省/非 dict → None
    （回退既有自然语言/分诊路径）。通道B 端点若想发真状态帧，可经 /stream 带 agent_config.editor_op
    进 graph（editor_entry 节点应用 op + 清 check → 下游 solve_explain 重验并发「程序验算」真帧）。
    🔴 verify-one（无状态、不依赖 thread state 的纯验算）**不**走此入口（仍是独立端点，见任务约束）。
    """
    conf = ((config or {}).get("configurable") or {}) if config else {}
    op = conf.get("editor_op")
    if isinstance(op, dict) and op.get("kind") in ("revise", "regen", "edit-item", "reverify"):
        return op
    return None


# 🔴 PRD-A-021 R2a·闸4（BUG-04）·读图低置信前置闸阈值（用户拍板 0.40）。
LOWCONF_BLOCK_THRESHOLD = 0.40


def _entry_read_lowconf(state: VariantState) -> bool:
    """母题入口读图是否「极低置信 / 章未判出」（闸4 拦截判据）。读 entry_decision 快照
    （mother_opus_entry 入口轮写），置信 < 0.40 或 章为空 → True。无 entry_decision（旧线程/
    回退入口）→ False（不拦，向后兼容）。"""
    dec = state.get("entry_decision")
    if not isinstance(dec, dict):
        return False
    try:
        conf = float(dec.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    chapter_empty = not str(dec.get("chapter") or "").strip()
    return conf < LOWCONF_BLOCK_THRESHOLD or chapter_empty


def _should_lowconf_block(state: VariantState) -> bool:
    """闸4 是否应在本 resume 轮拦截：读图极低置信 且 尚未拦过一次（_lowconf_blocked=False）。
    已拦过（老师坚持再确认）→ 不再拦，放行进 classify（防永久卡死）。"""
    return _entry_read_lowconf(state) and not state.get("_lowconf_blocked")


def route_entry(
    state: VariantState, config: RunnableConfig
) -> Literal[
    "mother_opus_entry", "parse", "generate", "ask", "auth", "classify",
    "editor_entry", "entry_lowconf_block",
]:
    # 🔴 身份硬闸（用户拍板 2026-06-11）：每次对话绑死登录老师。token 缺失/解不出 userId
    # → 一步不走（不进任何 LLM 节点，conv_trace 也不会产生无主行；表级 NOT NULL 双保险）。
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    if conv_trace.teacher_id_from_token(token) is None:
        return "auth"
    # 🔴 PRD-A-021 S1：结构化编辑/重生 op（经 /stream 带 editor_op 进 graph）优先于自然语言分诊——
    #   有 op 且有题组在手 → editor_entry（应用 op + 清 check → solve_explain 发真「程序验算」帧，治 F1
    #   通道B 静默 no-op）。无题组（op 无对象）则不拦，落既有路径（防把空轮误导进编辑）。
    if _editor_op(config) is not None and (state.get("items") or state.get("mother_dna")):
        return "editor_entry"
    url = _extract_image_url(_latest_human_text(state.get("messages", [])))
    # 🔴 PRD-C-100 B1a：跨轮新图 = 新母题 → 走塌缩入口 mother_opus_entry（opus 一把判章+解题+打标），
    #   替代旧 analyze→mother_precheck→classify 三节点链（控制流重写）。
    if url:
        return "mother_opus_entry"
    # 🔴 PRD-C-017 B2·母题确认 resume（复用 chat-resume，不引 interrupt）：上一轮 mother_precheck
    #   发了 needConfirm 停在等确认（awaiting_mother_confirm），本轮老师**经 config 回传确认章 id**
    #   （confirmed_chapter_id）→ 直奔 classify（带确认章接闸B）。老师若改成纯文字纠正（没回 id）→
    #   落下面 parse 分诊（既有在途母题 mother_correction → patch 重锚路径），不在此拦。
    if state.get("awaiting_mother_confirm"):
        cfg = (config or {}).get("configurable") or {}
        if cfg.get("confirmed_chapter_id"):
            # 🔴 R2a·闸4（BUG-04）：进 classify 前置闸——读图极低置信/章未判出 → 拦一次建议换图，
            #   不烧 opus token。老师坚持（再确认同一章）→ 第二轮 _lowconf_blocked 已 True，放行。
            if _should_lowconf_block(state):
                return "entry_lowconf_block"
            return "classify"
    # 🔴 PRD-C-017 B5·母题卡硬停闸 resume（复用 chat-resume，不引 interrupt）：上一轮 classify
    #   解出 mother_dna + 发母题卡帧后停在 awaiting_mother_review 等老师点「开始举一反三」。本轮
    #   老师点了 → FE 经 config 回传 start_variants=True → 已有 mother_dna（checkpointer 持久 thread
    #   state）→ **直奔 generate**（不重跑 classify、不重调 opus，复用 state.mother_dna）。
    #   🔴 即使老师改了母题 DNA 再点开始（既有 dirty/patch 逻辑会清 items + mother_confirmed=False
    #   走 classify 重锚），此处只在「已有 mother_dna 且未出题」时直奔 generate，不破 B3.6 edit→regen。
    if state.get("awaiting_mother_review") and not state.get("items"):
        cfg = (config or {}).get("configurable") or {}
        if cfg.get("start_variants") and state.get("mother_dna"):
            return "generate"
        # 🔴 BUG-01 R1·#4：高置信 await_review 态下老师改章（FE 经 config 回传 confirmed_chapter_id）
        #   → 也走重锚（同低置信 awaiting_mother_confirm 那条），重入 classify（_reanchor_reuse_first_solve
        #   复用首解、重发 classify done 帧），别落 parse 僵住（旧实现只认低置信确认章，高置信改章静默回
        #   parse、classify 帧不刷 = 状态条卡死）。confirmed_chapter_id 在 → 重锚优先于 parse 分诊。
        if cfg.get("confirmed_chapter_id"):
            if _should_lowconf_block(state):  # 🔴 闸4：高置信 await_review 改章 resume 同样前置拦截
                return "entry_lowconf_block"
            return "classify"
        # 🔴 停在 review 但老师没点开始（发了别的话/改 DNA）→ 落 parse 分诊（既有母题纠正/
        #   答疑路径），**绝不**掉进下面「mother_confirmed → 自动 generate」把硬停闸架空。
        return "parse"
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
        # 🔴 B2：新母题轮重置前置判态（旧母题的 needConfirm/reject/确认 id 绝不带进新图）
        "mother_precheck": None,
        "awaiting_mother_confirm": False,
        "mother_rejected": False,
        "confirmed_chapter_id": None,
        "confirmed_grade_book_id": None,
        # 🔴 M4/PRD-A-018·退役 analyze 入口同步清旧轮 items（与 mother_opus_entry.base_out 对齐）：
        #   回退旧入口时同 thread 第二张图也不得带旧变式，否则「开始举一反三」误落 parse 不出题。
        "items": [],
        "manual_order": False,
        "dropped_notes": [],
        "main_kp_prev": None,
        "messages": [],
    }


# ---------------------------------------------------------------------------
# 🔴 PRD-C-017 B2·母题 nano 前置判（analyze 后、classify 前）：
#   ① 读母题原图判「年级册 + 章」（+ 候选）→ 发 needConfirm，**无条件停**等老师确认（复用
#      clarify→END chat-resume，不引 LangGraph interrupt）。候选为空也发（让老师全手选）。
#   ② 顺手判题面**是否含图形/图表/几何图**（拍照纯文本题不算）→ 含图 → 发 reject 终止流程
#      （不发 needConfirm、不调 opus、不出变式）。false-positive 偏保守（拿不准当纯文本放行）。
# ---------------------------------------------------------------------------
async def mother_precheck_node(state: VariantState, config: RunnableConfig) -> VariantState:
    """nano 前置判 年级册+章+带图。带图 → reject 终止；否则 → needConfirm 停等确认。"""
    analysis = dict(state.get("analysis") or {})
    image_url = state.get("image_url") or ""
    grade_hint = (analysis.get("grade") or {}).get("value")

    _emit_stage("classify", "锚定考点", "running", "判定年级与章…")
    try:
        pre = await mother_precheck.precheck_judge(
            image_url=image_url,
            invoke=_ainvoke_text,
            model=settings.LLM_MODEL_LIGHT,  # gpt-5.4-nano（前置判轻活，复用 B0 探针 nano）
            grade_text_hint=grade_hint,
            parse_json=_parse_json,
            max_tokens=settings.VARIANT_MAX_TOKENS,
        )
    except Exception as e:  # noqa: BLE001 — 前置判失败 → 不静默放行带图、不静默进 opus；
        # 退化为「全手选」needConfirm（候选空 + has_figure=False 偏保守），让老师确认后再走 classify。
        analysis["_precheck_error"] = str(e)
        pre = {
            "grade_book": grade_hint or "", "chapter": "",
            "grade_candidates": [], "chapter_candidates": [],
            "has_figure": False, "confidence": 0.0,
        }

    # --- ① 带图打回（G13）：含图 → reject 终止流程（不发 needConfirm、不调 opus、不出变式） ---
    if pre.get("has_figure") is True:
        _emit_stage("classify", "锚定考点", "warn", "题面含图形，举一反三暂不支持")
        _emit_reject("with_figure", "举一反三暂不支持带图题")
        return {
            "analysis": analysis,
            "mother_precheck": pre,
            "mother_rejected": True,
            "awaiting_mother_confirm": False,
            "mother_confirmed": False,
            "messages": [AIMessage(content="这道题题面含图形（几何图/函数图象/统计图等），举一反三暂不支持带图题，请换一道纯文字题。")],
        }

    # --- ② 必停确认（G1）：无条件发 needConfirm（候选空也发），停等老师确认（clarify→END resume） ---
    payload = {
        "grade_book": {"id": "", "name": pre.get("grade_book") or ""},
        "chapter": {"id": "", "name": pre.get("chapter") or ""},
        "grade_candidates": [{"id": "", "name": n} for n in (pre.get("grade_candidates") or [])],
        "chapter_candidates": [{"id": "", "name": n} for n in (pre.get("chapter_candidates") or [])],
        "confidence": pre.get("confidence"),
    }
    _emit_need_confirm(payload)
    # 🔴 B5 问题1·阶段灯中性态：等老师确认年级章是**正常暂停**不是告警 → 发 STAGE_AWAIT（"await"），
    #   不发 warn、不发"已中断"（带图打回那个 warn 在上面保留，那是真要拦）。
    _emit_stage("classify", "锚定考点", STAGE_AWAIT, "请确认年级与章后继续")
    # 🔴 B5-fix3 问题1 收尾：「解析配方」灯在 analyze 发了 running 后、流程在此 needConfirm 处停下
    #   等老师确认（route END），**从未拿到终态**——FE settleStages 把残留 running 渲成 warn+「已中断」
    #   误告警。这里与「锚定考点」await 配对补发 knobs 中性 await，让正常暂停两灯齐齐中性、消除误告警。
    #   （真配方在老师确认章后由 classify 发 done/warn 终态。）
    _emit_stage("knobs", "解析配方", STAGE_AWAIT, "待确认年级章后定配方")
    grade_line = pre.get("grade_book") or "（未判出，请手选）"
    chapter_line = pre.get("chapter") or "（未判出，请手选）"
    body = (
        "我先判了一下母题的范围，**请确认年级册与章**再继续举一反三：\n\n"
        f"- 年级册：**{grade_line}**\n- 章：**{chapter_line}**\n\n"
        "确认无误请回复「确认」，需要修改请直接告诉我正确的年级/章。"
    )
    return {
        "analysis": analysis,
        "mother_precheck": pre,
        "awaiting_mother_confirm": True,
        "mother_rejected": False,
        "mother_confirmed": False,
        "messages": [AIMessage(content=body)],
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


# 🔴 PRD-C-104 B4：classify 已抽到 stage1_anchor/label.py（末尾 re-export，纯搬零改）。


# 🔴 PRD-C-104 B4：_reanchor_reuse_first_solve 已抽到 stage1_anchor/label.py（末尾 re-export，纯搬零改）。


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


def gate_after_classify(state: VariantState) -> Literal["await_review", "clarify"]:
    """🔴 定死闸（批2）+ 母题卡硬停闸（B5）：定死（年级册 code + main_kp 锚真叶子 + 置信达标）
    → **不再直通 generate**，改走 await_review（置 awaiting_mother_review + END，母题卡已先出，
    等老师点「开始举一反三」再经 route_entry resume 直奔 generate）；没定死（缺任一）→ 一律停
    clarify 确认态。从机制上①绝迹缺锚出题；②母题卡先出后必停、不自动造变式（B5 AC）。

    🔴 B5 前本闸 pinned → "generate" 直连，变式立刻自动生成。现在 pinned → "await_review"
    硬停，让老师 review 母题卡后主动触发。resume 路径（start_variants）绕开 classify/本闸
    （route_entry 直接 → generate），故本闸的 "generate" 出口已退役（只剩 await_review/clarify）。

    mother_confirmed 由 classify 按同口径（_conf_ok + anchored）置位，这里用 _pin_status
    再加「年级册 code + 非复习册」收口（mother_confirmed=True 但年级 code 缺/是复习册的边角
    路径也会被本闸拦住，不直通）。"""
    if state.get("mother_confirmed") and _pin_status(state)["pinned"]:
        return "await_review"
    return "clarify"


async def await_mother_review(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 PRD-C-017 B5·母题卡硬停闸：classify 已解出 mother_dna + 发了母题卡专帧（_emit_mother_card），
    本节点置 awaiting_mother_review=True 并 END，**不流向 generate**——母题卡先出后流程停下，等老师
    点「开始举一反三」（FE 经 config 回传 start_variants=True）下一轮经 route_entry resume 直奔 generate。

    🔴 阶段灯中性态（B5 问题1）：发 STAGE_AWAIT（"await"，非 warn / 非"已中断"），文案「母题已就绪，
       点『开始举一反三』生成变式」。这是正常暂停不是告警。
    🔴 BUG-01（2026-06-19）·「确认母题」节点改由 review key 驱动，**不再复用 classify key**：
       旧实现这里发 classify=await，把前面 finalize/classify 已发的 classify=done 覆盖回退成 await →
       「读图锚定」节点倒退「需人工」（三节点自相矛盾根因）。现改发 review=await（专用 key），
       classify=done 不被动；「确认母题」节点 FE 改读 review，不再读 knobs（knobs 仍发但语义=配方非确认）。
    🔴 不重调任何 LLM（母题卡已在 classify 备齐）；checkpointer 跨本次暂停持久 mother_dna（thread state）。
    """
    _emit_stage("review", "确认母题", STAGE_AWAIT,
                "待老师确认母题，点『开始举一反三』生成变式")
    return {
        "awaiting_mother_review": True,
        # 留痕：母题确认环节已过（route 不再当成在途确认）；review 是新的暂停态。
        "awaiting_mother_confirm": False,
        "messages": [
            AIMessage(content="请确认母题无误（见上方母题卡）：确认无误就点「开始举一反三」，开始准备生成变式；若要修改母题，直接告诉我。")
        ],
    }


async def entry_lowconf_block(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 PRD-A-021 R2a·闸4（BUG-04）·读图低置信前置闸（resume 轮·classify 之前）：母题读图置信
    极低（< 0.40）/ 章未判出 → 拦截一次，建议老师换张清晰的图，**不进 classify、不烧 opus token**。

    🔴 一次性拦截（防永久卡死）：置 _lowconf_blocked=True。老师若坚持（再回传 confirmed_chapter_id），
       route_entry 见 _lowconf_blocked=True → _should_lowconf_block False → 放行进 classify 正常出题。
    🔴 阶段灯中性 await（不是 warn/已中断）：这是「建议换图」的暂停，不是流程出错。
    🔴 保持 awaiting_mother_confirm=True，让老师下一句（坚持确认 / 换图 URL）能继续被 route 接住。
    """
    dec = state.get("entry_decision") if isinstance(state.get("entry_decision"), dict) else {}
    try:
        conf = float((dec or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    _emit_stage("classify", "锚定考点", STAGE_AWAIT,
                "这张图可能不适合做母题·建议换张清晰的图")
    _emit_stage("knobs", "解析配方", STAGE_AWAIT, "待换图或确认后定配方")
    body = (
        "⚠ 我对这张图的读图把握很低"
        f"（置信约 {conf:.2f}{'、且没判出具体章' if not str((dec or {}).get('chapter') or '').strip() else ''}）"
        "——**可能这张图不太适合做母题**（拍得不清 / 不是标准题图 / 含大量图形）。\n\n"
        "建议：\n"
        "- **换一张更清晰的题目图**（直接贴新图的 OSS URL，我重新读）；\n"
        "- 若你确认就用这张图、按你选的章继续 → **再回复一次「确认」**，我照常出变式。"
    )
    return {
        # 拦过一次（坚持再确认即放行，不二次拦）。awaiting_mother_confirm 保持 True 让 resume 续接。
        "_lowconf_blocked": True,
        "awaiting_mother_confirm": True,
        "awaiting_mother_review": False,
        "messages": [AIMessage(content=body)],
    }


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
        _qtype_guess = q.get('value') or "新定义/非常规题，按解答处理可以吗？"
        asks.append(f"题型我没把准（看着像「{_qtype_guess}」），对吗？")
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

# ---------------------------------------------------------------------------
# 🔴 题型结构契约（PRD-C-013 P11 单一事实源）：GENERATE / REGEN / ADD 共用，
# 约束「每道题按其 qtype 长成对的结构」。与验算载荷契约正交：那个管「答案能不能被
# sympy 验」，这个管「题面/选项/答案的形状对不对」。代码级 structure_lint 同口径校验。
# 排版（吃 aigeek 前缀缓存）：本常量属固定段，各 prompt 把它排在变动段（题干/facts）之前。
# 同 _PAYLOAD_CONTRACT，以「format 模板片段」形态存在（无 {占位符}，但与会被 .format()
# 的模板拼接，故文本内若出现花括号须双写——当前无）。
# ---------------------------------------------------------------------------

# 🔴 难度四档 rubric（整改2·2026-06-12·单一事实源）：难度判定并入生题——出题调用同步产出
#   每道题的 difficulty（不再独立走一轮 _grade_difficulty 复评）。rubric 标准本身不变（仍是
#   22-SSOT §2 四档断言），由 GENERATE/REGEN/ADD/revise 出题 prompt 嵌入，让出题时就按 rubric
#   断言难度。难度是「评级」（LLM rubric 断言）≠「判对错」（归闸B sympy，铁律不破）。
#   🔴 本常量将被拼进会 .format() 的 prompt 模板，文本内花括号须双写转义（{{ }}）。

# 🔴 配图描述契约（PRD-A-018 治本A·单一事实源）：出题时（上下文最全——有母题题面+母题配图+
#   变式题面）顺手为每道变式产一份**自然语言配图描述 figure_spec**，下游 compose 直接照它翻
#   GeoGebra（不再现场逆推构型，根治「逆推歧义」+ 大降耗时）。figure_spec 是**自然语言**，不是
#   GeoGebra 命令——说清「这道图有哪些点、大致布局、标哪些角/度数、什么旋转/平移/对称到哪、
#   哪些虚线」即可，把"画什么"想清楚，"怎么翻成命令"留给 compose。
#   🔴 与 GENERATE 同以「format 模板片段」形态存在（无 {占位符}；文本内若出现花括号须双写——当前无）。

# 🔴 排版（PRD-C-012 任务3·吃 aigeek 前缀自动缓存）：固定规则/契约段在前，
# 含 {占位符} 的变动段（配方/铁律的考点名、母题 DNA）移到末尾；语义一字不改。
# 🔴 PRD-C-104 B3a：GENERATE_PROMPT 已抽到 stage2_variant/prompts.py（末尾 re-export）。


def _figure_type_gate_block(state: VariantState, facts: dict) -> str:
    """🔴 PRD-A-021 R3b·章节×图型定型闸（GENERATE 落点①）：据母题章节名 + 主考点名查
    biz_chapter_figure_map 取「允许图型集」，渲成约束段拼到 GENERATE_PROMPT **末尾**
    （护 aigeek 前缀缓存，约束段是变动尾段）。让出题写 figure_spec 时**只在允许图型内**描述配图。

    🔴 逃生：章节/考点取不到 / 无匹配 / 表读不到 → 允许集空 → 返回 ""（不拼约束，自由发挥）。
    纯 best-effort：任何异常一律返回 ""（定型闸是增强非关卡，绝不卡住出题）。
    """
    try:
        from agents.figure import chapter_figure
        chapter = _mother_chapter_name(state)
        kp = str(facts.get("kp_name") or "").strip() or None
        allowed = chapter_figure.allowed_figure_types(chapter, kp)
        return chapter_figure.constraint_clause(allowed)
    except Exception:  # noqa: BLE001
        return ""


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
        # 🔴 PRD-A-022 批1：母题草稿是否已发布（autodraft 落母题草稿后此为 False；publish promote 后置 True）。
        #   persist_items publish 据此决定「promote 母题草稿」还是「跳过」（已发布幂等不重 promote）。
        "mother_published": bool(mdna.get("mother_published")),
        # 🔴 PRD-C-015 批4·缺口10：母题脏（守恒维改）→ persist_items 据此 update 已入库 role=mother 行。
        "mother_dirty": bool(mdna.get("dirty")),
        # 🔴 图母题不在库 → 入库时先把母题(原题)也落库挂血缘，下面这几项给 build_mother_bo 用
        "mother_answer": mdna.get("answer"),
        "mother_solution": mdna.get("solution_skeleton") or mdna.get("answer"),
        "mother_difficulty": dna.get("difficulty") or mdna.get("difficulty"),
        "mother_structure": mdna.get("structure"),
        "kp_confidence": (kp.get("confidence") if isinstance(kp, dict) else None),
        "image_url": state.get("image_url"),
        # 🔴 PRD-A-022 批2·D8：母题切图 OSS url（build_mother_bo 入库优先取它，缺则不带图）。
        "mother_figure_url": state.get("mother_figure_url"),
        # 🔴 B1 全维 DNA 穿进 BO（T3）：副 kp/标签/骨架/场景/考察类型/难点 + 锚定审计
        "dna": dna,
    }


# ---------------------------------------------------------------------------
# 🔴 PRD-C-103 WS1·母题起步档查表（AC2）：母题 md = 锚定模型查表(tier/freq) 经 grade_observed
#   确定算，替代 mother_opus dim8 的 LLM 自评 difficulty。喂进 recipe_from_knobs 的 md（md+i 递增
#   逻辑 L3695 不动，只换 md 来源）。难度永不取 LLM 自评（铁律：pass/fail 只读 grade_observed）。
# ---------------------------------------------------------------------------
# 🔴 PRD-C-104 B3a：_dna_factors_for_grade 已抽到 stage2_variant/difficulty.py（末尾 re-export）。


def mother_md_from_table(state: VariantState) -> int | None:
    """🔴 WS1·AC2：母题起步档 = 锚定模型查表 经 grade_observed 确定算（替 dim8 LLM 自评）。

    返回 level（1..4）。降级（契约 §异常）：
      - 无锚定模型 / 抽不出 K/R → grade_observed 自带哨兵兜底（仍返回 level，不报错、不返回 None）。
      - 仅当 mother_dna/dna 完全缺失（库内母题旧线程无 DNA）→ 返回 None，由调用方回退旧 dim8 值。
    """
    mdna = state.get("mother_dna") or {}
    dna = mdna.get("dna") or {}
    if not dna:
        return None  # 无 DNA（库内母题等）→ 调用方回退原 difficulty 值
    f = _dna_factors_for_grade(
        dna, stem=mdna.get("stem") or "", analysis_text=mdna.get("solution_skeleton") or ""
    )
    bill = difficulty.grade_observed(
        model_hits=f["model_hits"], K=f["K"], R=f["R"], D=f["D"], G=f["G"],
        high_strategies=f["high_strategies"],
    )
    return bill.get("level")


# 解析分步：把变式 solution 文本拆成步骤行（R 推理链步数的确定性来源）。
_SOLUTION_STEP_RE = re.compile(r"\n|；|;|。(?=\S)|步骤|第[一二三四五六七八九1-9]步")


def _solution_steps(solution: str) -> list[str]:
    """把变式解析文本拆成步骤行（去空），作 R 的确定性近似（变式无 JSON skeleton 时用）。"""
    if not solution:
        return []
    parts = [p.strip() for p in _SOLUTION_STEP_RE.split(solution) if p and str(p).strip()]
    return parts


# 🔴 PRD-C-104 B3a：grade_variant_item 已抽到 stage2_variant/difficulty.py（末尾 re-export）。


def variant_trace_block(item: dict, knobs: dict | None) -> dict[str, Any]:
    """🔴 WS3·AC9 纯函数（零 LLM/零 IO，可单测）：单道变式 → biz_variation_trace 喂料块。

    填真值（批2 的 method=forward-gen/similarity=None 兜底 → 本函数补真）：
      - operator/method = 变式系数轴落带的代表算子（knobs.operator_band.operator），缺 → 'forward-gen'。
      - similarity/variation_degree = 变式系数（knobs.operator_band.similarity 或 knobs.variant_coeff），
        缺 → 默认 0.7（VARIANT_COEFF_DEFAULT，= 默认变式系数，仍是确定值非 None）。
      - target_level = 难度轴目标档（knobs.difficulty_target），缺 → None（=keep 母题档，无显式目标）。
      - actual_level = 该变式确定判档 level（item.difficulty_bill.level，assemble 落的 grade_observed 真值）。
      - retries = 该题回炉次数（item._regen_count，未追踪 → 0）。
    🔴 method/variation_degree 有 DB 列（落库真值）；target/actual/retries 库无列、仅落 manifest 供审计。
    created_by = 'forward-gen'（举一反三正向生成，区别于打标 reverse-dna）。
    """
    knobs = knobs or {}
    op_band = knobs.get("operator_band") if isinstance(knobs.get("operator_band"), dict) else {}
    operator = str(op_band.get("operator") or "forward-gen").strip() or "forward-gen"
    similarity = op_band.get("similarity")
    if similarity is None:
        similarity = knobs.get("variant_coeff")
    if similarity is None:
        similarity = VARIANT_COEFF_DEFAULT
    bill = item.get("difficulty_bill") or {}
    actual_level = bill.get("level")
    if actual_level is None:
        actual_level = _to_int(item.get("difficulty"))
    target_level = _to_int(knobs.get("difficulty_target"))  # None = keep（母题档，无显式目标）
    retries = _to_int(item.get("_regen_count")) or 0
    return {
        "operator": operator,
        "method": operator,
        "similarity": round(float(similarity), 2),
        "similarity_band": (op_band.get("band") if op_band else None),
        "target_level": target_level,
        "actual_level": actual_level,
        "retries": retries,
        "created_by": "forward-gen",
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
# 🔴 B3·变式多样性重组（PRD-A-021 R3a）：松绑守恒段④「只换数字」导致的「变式太像」。
#   根因 = DNA 绑死、无多样性轴 → 只能换数字。方案（用户拍板）：**骨架四维锁死**
#   （题型 / 主考点 / 模型 / 难点），多样性来自**副考点 + 高频标签 + 场景**重组。
#
# 注入点语义（审计订正）：
#   - 只注 GENERATE（首轮出题）。REGEN 是单题等价保型（注多样性自相矛盾）、ADD 来源未定，都不注。
#   - scene 是**组级共享维**（B5-fix5「改场景全组生效」契约）→ 整组一个场景，**不给每道变式各带
#     scene_hint**（会和组级契约打架）。多样性主载体 = **per-variant 的副考点/标签子集差异**。
#   - 多样性主载体 = 复用**母题自身已有的副考点 + 标签**（10 维 DNA 已含 secondary_kps + tags，
#     解题打标产出）—— 每道变式强调不同的副考点/标签子集即可拉开差异，**无需新 DB 查询**。
#   - 稀疏数据优雅降级：母题副考点/标签太少（去重后 < 2 个差异化锚）→ 不做 per-variant 硬分派，
#     退回「鼓励情境细节差异」软指令（绝不报错 / 绝不空转）。
#
# 🔴 守恒红线：本段**只在副考点切入角度 / 标签子集 / 场景细节**维放开多样性；题型 / 主考点 /
#   模型 / 难点四维仍由 _conservation_clause + _model_cards_clause + _context_block 锁死。
#   本段反复申明「骨架四维不许动、副考点必须仍在白名单内」，不给 opus 超纲/换骨架的口子。
# 🔴 文本含 {…} 字面量须双写转义（本段会被拼进 .format() 的 prompt 模板）。
# ---------------------------------------------------------------------------
def _diversity_anchors(dna: dict | None) -> list[str]:
    """从母题 DNA 取「多样性差异化锚」候选 = 副考点名 + 高频标签（去重、去空、保序）。纯函数·可单测。

    这些是**副考点切入角度 / 标签子集**的素材：每道变式强调不同的子集 → 拉开「考法切入」差异，
    而不只换数字。**只取母题自带的**（secondary_kps + tags，解题打标已产出）—— 不查 DB（审计定）。
    """
    dna = dna or {}
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: Any) -> None:
        name = str(s or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)

    for s in _norm_secondary_kps(dna.get("secondary_kps")):
        _add(s.get("name"))
    for t in dna.get("tags") or []:
        _add(t)
    return out


def _diversity_clause(dna: dict | None, n: int) -> str:
    """🔴 B3 多样性重组段（**只 GENERATE 注入**·T1 单一事实源）。返回可拼进 .format() prompt 的文本。

    机制：把母题自带的副考点 + 标签当差异化锚池，给 n 道变式做 **round-robin 子集分派** ——
    每道强调一个不同的副考点/标签切入角度（per-variant 差异轴），同时反复锁死骨架四维。
    锚池 < 2 个 或 n < 2 → 优雅降级为软指令（不分派、不报错）。

    🔴 锚池里的「副考点」全部来自母题 DNA 白名单（secondary_kps），故强调它们**不会超纲**——
       与守恒段①白名单正交（守恒段管「⊆ 白名单」，本段管「在白名单内换不同切入角度」）。
    """
    n = int(n or 0)
    anchors = _diversity_anchors(dna)
    lines: list[str] = [
        "🔴 变式多样性（B3·避免「几乎只换数字」—— 变式之间必须有真实的考法切入差异）："
    ]
    # 骨架四维锁死申明（无论稀疏与否都喊，堵住松绑后的超纲/换骨架口子）
    lines.append(
        "⛔ 锁死四维（绝不许借「多样性」之名动）：**题型 / 主考点 / 解题模型 / 难点**四维全组一致、"
        "与母题守恒（见上方守恒段与确定上下文块）。多样性**只在「副考点切入角度 / 标签子集 / "
        "场景情境细节」**上做，不许换赛道、不许超纲、不许简化最难步。"
    )
    if n >= 2 and len(anchors) >= 2:
        # per-variant round-robin 子集分派：每道题点名一个不同的差异化锚（副考点/标签切入角度）
        assign = [anchors[i % len(anchors)] for i in range(n)]
        bullet = "\n".join(
            f"  - 第 {i + 1} 道：突出从「{a}」这个**副考点/角度**切入"
            "（在主考点不变、白名单内组织题面，让这道题的考查侧重明显区别于其它变式）。"
            for i, a in enumerate(assign)
        )
        lines.append(
            "① per-variant 切入角度分派（每道强调不同的副考点/标签子集，拉开「考法」差异，"
            "**不是**只换数字）：\n" + bullet
        )
        lines.append(
            "② 即便分到同一锚，也要在**情境/设问方式/数据组织**上与其它变式明显不同；"
            "禁止 n 道题题面骨架雷同只有数字不同。"
        )
    else:
        # 稀疏降级：锚不够分派 → 软指令，鼓励情境/设问差异（仍守四维），绝不报错/空转
        if anchors:
            anchors_s = "、".join(anchors)
            lines.append(
                f"① 母题可用的副考点/标签较少（{anchors_s}）—— 在主考点不变、白名单内，"
                "尽量让每道变式从不同的副考点/标签角度或不同情境切入，避免题面只有数字不同。"
            )
        else:
            lines.append(
                "① 母题副考点/标签信息有限 —— 请在主考点与骨架四维不变的前提下，"
                "让每道变式在**情境设定 / 设问方式 / 数据组织**上彼此明显不同，避免只换数字的"
                "「克隆题」（仍守上方全部守恒约束）。"
            )
    # scene 组级申明：不给每道各带场景（与 B5-fix5 组级契约一致）
    scene = str((dna or {}).get("scene") or "").strip()
    if scene:
        lines.append(
            f"③ 场景是**整组共享**维（当前组场景：「{scene}」）—— 全组统一在此场景下，"
            "**不要**每道题各换一个不同的大场景（多样性靠上面的副考点/角度差异，不靠拆散组级场景）。"
        )
    return "\n".join(lines)


def _maybe_diversity_block(dna: dict | None, n: int) -> str:
    """多样性段拼接器：n<2（单题无「彼此差异」可言）→ ""（不注）；否则前缀换行接进 prompt。"""
    if int(n or 0) < 2:
        return ""
    return "\n\n" + _diversity_clause(dna, n)


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

# ---------------------------------------------------------------------------
# 🔴 PRD-C-103 WS3·双旋钮（AC8/AC9）：变式系数轴（相似度→选算子带）+ 难度轴（目标档移 md）。
#   事实源 = `举一反三策略-定稿.md §二/§三`（9 算子 + 相似度系数带）。两轴独立：
#     · 变式系数（similarity，0–1，默认 0.7）：管"像不像母题" —— 落在某相似度带 → 选该带算子。
#     · 难度（difficulty，'keep' 或目标档 1–4，默认 keep）：管"难不难" —— 相对母题档移 md。
#   🔴 与「线性/不上大图」一致：两轴都做成纯函数 + 配方段注入 + trace 落真值，不改宏观 DAG。
#   🔴 母题 variationProfile（C/D 层）尚未由 mother_opus 产出（落地顺序 item4，future）→ 本轮
#      算子选择走「相似度带 → 代表算子」确定映射（不依赖 profile 可用性），profile 上线后再收窄。
# ---------------------------------------------------------------------------
VARIANT_COEFF_DEFAULT = 0.7  # 变式系数默认（策略定稿 §二）
# 9 算子相似度系数带（策略定稿 §三）：(算子名, 带下限, 带上限)。系数落区间 → 选代表算子。
#   高仿带(0.8~1.0)=数值；中变带(0.4~0.7)=结构/情境/条件增删；远迁带(0.2~0.4)=逆向/推广/升维/分类。
_OPERATOR_BANDS: list[tuple[str, float, float, str]] = [
    # (代表算子, 下限, 上限, 相似度带名)
    ("数值", 0.75, 1.01, "高"),       # ①数值(0.8~0.9) 高仿
    ("结构", 0.45, 0.75, "中"),       # ②结构 / ③情境 / ⑧条件增删 中变（代表取「结构」）
    ("推广一般化", 0.0, 0.45, "低"),  # ⑤逆向/⑥推广/⑦升维/⑨分类 远迁（代表取「推广一般化」）
]
# 算子 → 出题人话指令（注入 GENERATE 配方段，让"像不像"随系数真变）。
_OPERATOR_GUIDANCE: dict[str, str] = {
    "数值": "高仿母题（只换数字/表皮，结构与考法尽量贴母题，像不像≈0.8+）",
    "结构": "中度变式（可改结构/情境/增删条件，保留母题核心考法，像不像≈0.5）",
    "推广一般化": "远迁变式（可逆向/推广/升维/分类讨论化，考点守恒但形态大变，像不像≈0.3）",
}


def operator_band_from_similarity(coeff: Any) -> dict[str, Any] | None:
    """🔴 WS3 纯函数（零 LLM/零 IO，可单测）：变式系数 → {operator, similarity, band, guidance}。

    coeff 缺/非数 → None（调用方回落默认配方，不注入算子段 = 旧行为）。
    coeff 钳到 [0,1]；落在某相似度系数带 → 取该带代表算子 + 出题人话指令 + 高/中/低带名。
    similarity = 钳后的系数本身（落 trace.variation_degree；method = operator）。
    """
    try:
        c = float(coeff)
    except (TypeError, ValueError):
        return None
    c = max(0.0, min(1.0, c))
    for op, lo, hi, band in _OPERATOR_BANDS:
        if lo <= c < hi:
            return {
                "operator": op,
                "similarity": round(c, 2),
                "band": band,
                "guidance": _OPERATOR_GUIDANCE.get(op, ""),
            }
    # 兜底（理论不达，区间已覆盖 0~1）：归高仿带
    return {
        "operator": "数值",
        "similarity": round(c, 2),
        "band": "高",
        "guidance": _OPERATOR_GUIDANCE["数值"],
    }


def normalize_two_knobs(conf: dict[str, Any] | None) -> dict[str, Any]:
    """🔴 WS3 纯函数：config.configurable 的双旋钮原值 → 受约束 knobs 增量段。

    读两键（FE 经 agent_config 透传，见 book-ui streamVariant）：
      - `variant_similarity`（变式系数 0–1）→ out['variant_coeff'] + out['operator_band']（带代表算子）。
      - `difficulty_target`（'keep' 或目标档 1–4）→ out['difficulty_target']（int）；'keep'/缺 → 不设。
    任一缺/非法 → 该轴不设键（回落默认：相似度走 0.7 由调用方补、难度走 keep=母题档 md+i）。
    纯函数、零 IO，可单测。返回的段会并进 state['knobs']（与 LLM 抽的 count/qtype 不冲突）。
    """
    out: dict[str, Any] = {}
    conf = conf or {}
    sim = conf.get("variant_similarity")
    if sim is None:
        sim = conf.get("variantSimilarity")  # camelCase 容错
    band = operator_band_from_similarity(sim) if sim is not None else None
    if band is not None:
        out["variant_coeff"] = band["similarity"]
        out["operator_band"] = band
    dt = conf.get("difficulty_target")
    if dt is None:
        dt = conf.get("difficultyTarget")
    if dt is not None and str(dt).strip().lower() not in ("keep", "", "none", "null"):
        tgt = _to_int(dt)
        if tgt is not None:
            out["difficulty_target"] = max(1, min(DIFFICULTY_CAP, tgt))
    return out


# 🔴 PRD-C-104 B3b：normalize_knobs 已抽到 stage2_variant/generate.py（末尾 re-export）。


# 🔴 PRD-C-104 B3b：recipe_from_knobs 已抽到 stage2_variant/generate.py（末尾 re-export）。


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
        # 🔴 A-1/M6 态②：qtype_partial 时 dist 是"部分约束"（各题型最小值），用 ≥ 判而非逐项相等。
        partial = bool(knobs.get("qtype_partial"))
        got: dict[str, int] = {}
        for it in items:
            qt = _QTYPE_ALIAS.get(str(it.get("qtype") or "").strip(), str(it.get("qtype") or "").strip())
            got[qt] = got.get(qt, 0) + 1
        if partial:
            bad = any(got.get(k, 0) < v for k, v in dist.items())
        else:
            bad = any(got.get(k, 0) != v for k, v in dist.items())
        if bad:
            sep = "≥" if partial else "×"
            want_s = "、".join(f"{k}{sep}{v}" for k, v in dist.items())
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
    # 🔴 PRD-A-018 治本A·figure_spec（出题节点产的「画什么」自然语言描述，下游 compose 照此翻
    #   GeoGebra，不再现场逆推构型）：仅当 LLM 给了非空字符串才落 item（内部/展示键，不入 biz_question
    #   旧字段——build_create_bo/_artifact_payload 显式白名单天然不外漏；参照 figure_url 处理）。
    #   缺字段/空串/纯代数题给空串 → 不落该键 → 下游退回从 stem 现推（向后兼容，旧线程无此键不崩）。
    #   🔴 round4 半结构化：figure_spec 现可为 ① 字符串（旧形态，自然语言）或 ② dict（新形态
    #   {"layout":..., "angle_labels":[{"angle":"∠BAC","label":"20°"}]}）—— 标注决策结构化无损下传。
    #   两种形态都原样落 item（dict 的空判 = layout/angle_labels 均空才算空）；compose 端按类型分发。
    _fig_spec = it.get("figure_spec")
    if isinstance(_fig_spec, str) and _fig_spec.strip():
        out["figure_spec"] = _fig_spec.strip()
    elif isinstance(_fig_spec, dict):
        _layout = str(_fig_spec.get("layout") or "").strip()
        _al = _fig_spec.get("angle_labels")
        _al = _al if isinstance(_al, list) else []
        # 仅当至少有 layout 或 angle_labels 才落（纯空对象视同无 spec → 下游退回现推）
        if _layout or _al:
            out["figure_spec"] = {"layout": _layout, "angle_labels": _al}
    if isinstance(it.get("verify_payload"), dict):
        out["verify_payload"] = it["verify_payload"]
    # 🔴 批3·⑦ 反退化载荷（最值/动点构型才有）随 item 流转——item 内部字段，不进帧/不入库。
    if isinstance(it.get("degen_payload"), dict):
        out["degen_payload"] = it["degen_payload"]
    return _sanitize_item(out)


# 🔴 PRD-C-104 B3b：_parse_generated_items 已抽到 stage2_variant/generate.py（末尾 re-export）。


def _extract_items(data: Any) -> list[Any] | None:
    """从已解析对象取 items 数组：顶层是数组直接用；是 {"items":[...]} 取之；否则 None。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("items")
        return items if isinstance(items, list) else None
    return None


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
            seg = acc[start : i + 1]
            try:
                obj = json.loads(seg)
            except Exception:  # noqa: BLE001 — 闭合但 json.loads 不过
                # 🔴 PRD-A-023 B8：未转义引号致单题 json.loads 炸 → eager 上屏 0 道、右栏不增量。
                #   套确定性引号修复再试一次（闭合花括号已配平，仅引号脏 → 修复后多可解）；
                #   仍失败才算半截/脏文本不计。
                obj = None
                try:
                    from agents.variant_entry import _repair_json_quotes  # 懒导入防循环
                    repaired = _repair_json_quotes(seg)
                    if repaired and repaired != seg:
                        obj = json.loads(repaired)
                except Exception:  # noqa: BLE001 — 修复亦失败 → 真半截/脏文本，跳过
                    obj = None
                if not isinstance(obj, dict):
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


# 🔴 PRD-C-104 B3b：generate 已抽到 stage2_variant/generate.py（含内嵌闭包；末尾 re-export）。


# 🔴 排版（PRD-C-012 任务3）：固定契约段前移，变动段（题干）移末尾；语义一字不改。

# 🔴 PRD-C-104 B3a：REGEN_PROMPT 已抽到 stage2_variant/prompts.py（末尾 re-export）。


# B5-fix4（PRD-C-017）：老师在变式卡显式把题型从 A 改成 B（edit-dna field=qtype）→ 该题
# dirty_dims 含 "qtype"。REGEN_PROMPT 本质是「等价变式·换数字保型重出」，传进去的新 qtype
# 只是弱信号被模型无视（沿用原题面结构 = 改了个寂寞）。检测到题型脏时，往 REGEN_PROMPT 拼
# 这段强指令（压过"等价变式保型"框架），让模型真按新题型的标准结构（_QTYPE_CONTRACT）重构题面。
# 🔴 只换题型一维：主考点/年级/难度/考察类型/骨架基因仍守恒（守恒注入不动）。
def _qtype_change_clause(new_qtype: str) -> str:
    return (
        f"\n\n🔴🔴 老师显式要求改题型 → 必须把本题**重构**为【{new_qtype}】题型（这条优先级高于上面的"
        f"「等价变式·换数字保型」框架）：按【{new_qtype}】题型的标准结构（见上方题型结构契约）**重组题面**，"
        f"**不是**换数字保留原题型形态。例如：填空→解答：去掉 `____` 空位，改成「求…的值，写出完整解题过程」；"
        f"选择→填空：去掉 ABCD 选项改成空位填值；选择/填空→解答：展开为带完整推导的解答。"
        f"主考点/年级/难度/考察类型/解法骨架最难步仍守恒，**只换题型形态** + 配套调整 stem/answer/solution/"
        f"qtype/verify_payload 的结构以匹配【{new_qtype}】。输出 JSON 的 qtype 字段**必须**= 「{new_qtype}」。"
    )


# B5-fix5（PRD-C-017）：老师在变式卡显式把场景从 A 改成 B（edit-dna field=scene）→ 落
# mother_dna.dna.scene（场景是组级共享维，对全组生效）。REGEN_PROMPT 本质是「等价变式·换数字
# 保型重出」，守恒段④只说「场景可换同类」= 让模型随机换/沿用原场景，老师指定的新场景从不传达
# 给模型（= 改了个寂寞，与 qtype 同构坑）。检测到场景脏时往 REGEN_PROMPT 拼这段强指令（压过
# 「等价变式保型 + 守恒段④随机换场景」），让模型真把题面改写到老师指定的场景下。
# 🔴 只换场景表皮一维：考点/年级/题型/难度/考察类型/解法骨架最难步仍守恒（守恒注入不动）。
def _scene_change_clause(new_scene: str) -> str:
    return (
        f"\n\n🔴🔴 老师显式指定了新场景【{new_scene}】 → 必须把本题的**题面叙述改写到这个场景下**"
        f"（这条优先级高于上面的「等价变式·换数字保型」框架与守恒段「场景可换同类」——**不是**随机换场景、"
        f"**不是**保留原场景）：把题干的背景/角色/情境替换成【{new_scene}】，并配套调整 stem 的叙述措辞，"
        f"让整道题读起来就发生在【{new_scene}】里。主考点/年级/题型/难度/考察类型/解法骨架最难步仍守恒，"
        f"**只换场景表皮**（数值与考查结构不变）。输出 JSON 若含 scene/场景字段则**必须**= 「{new_scene}」。"
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


# 🔴 PRD-C-104 B3b：_anti_degen_gate 已抽到 stage2_variant/gates.py（末尾 re-export）。


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


# 🔴 PRD-C-104 B3b：_conservation_ok 已抽到 stage2_variant/gates.py（末尾 re-export）。


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


# 🔴 PRD-C-104 B3b：_surface_check 已抽到 stage2_variant/gates.py（末尾 re-export）。


# 🔴 PRD-C-104 B4：_solve_one 已抽到 stage1_anchor/solve.py（末尾 re-export，纯搬零改）。


# 🔴 PRD-C-104 B3b：_regen_once 已抽到 stage2_variant/gates.py（末尾 re-export）。


# 🔴 PRD-C-104 B4：_check_one_item 已抽到 stage1_anchor/solve.py（末尾 re-export，纯搬零改）。


# 🔴 PRD-C-104 B3b：solve_explain 已抽到 stage2_variant/gates.py（末尾 re-export）。


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


# 🔴 PRD-C-104 B3b：_model_conservation_check 已抽到 stage2_variant/gates.py（末尾 re-export）。


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


# 🔴 PRD-C-104 B3b：gene_gate_check 已抽到 stage2_variant/gates.py（末尾 re-export）。


# 🔴 PRD-C-104 B4：_gene_one_item 已抽到 stage1_anchor/solve.py（末尾 re-export，纯搬零改）。


# 🔴 PRD-C-104 B3b：gene_gate 已抽到 stage2_variant/gates.py（末尾 re-export）。


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


def _grade_difficulty_payload(items: list[dict[str, Any]]) -> str:
    """组装评分用题面（题干+答案+解析），编号 1..n。纯函数、可单测。"""
    lines: list[str] = []
    for i, it in enumerate(items, 1):
        stem = str(it.get("stem") or "").strip()
        answer = str(it.get("answer") or "").strip()
        sol = str(it.get("solution") or "").strip()
        lines.append(f"[{i}] 题干：{stem}\n答案：{answer}\n解析：{sol}")
    return "\n\n".join(lines)


# 🔴 PRD-C-104 B3a：_grade_difficulty 已抽到 stage2_variant/difficulty.py（末尾 re-export）。


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


# 🔴 PRD-C-104 B3b：assemble 已抽到 stage2_variant/assemble.py（末尾 re-export）。


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


# --- 答疑：只问不改（🔴 物理上 return 不含 items） -----------------------------


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
    # 🔴 PRD-A-018 G10-4：配图相关诉求（补图/配图/图没画完/图歪）单独引导——配图是变式卡上
    #   逐题手动点的（不进 graph、agent 不直接画），别甩通用兜底让老师以为没听懂。
    _FIGURE_HINT_WORDS = ("图片", "配图", "图形", "画图", "图没", "补图", "切图", "图歪", "没画", "没出来", "图不")
    if utterance and any(w in utterance for w in _FIGURE_HINT_WORDS):
        body = (
            "配图是在变式卡上逐题手动点的（我这边不直接画图）：\n"
            "- 每道变式正文下方有「🖼 配图 / 🖼 重新配图」按钮，点它给这道题画图；\n"
            "- 图歪了 / 没画全 → 点「图歪了？重新生成」，补一句图形描述（说清要画哪些点 / 角 / 线）再画；\n"
            "- 母题图在母题卡左侧，点「重新切图」可重切。\n\n"
            "（几何题偶尔画不出会标「⚠ 待补图」，点上面的按钮重试或补一句描述即可。）"
        )
    elif _looks_like_conservation_hit(utterance):
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

# ---------------------------------------------------------------------------
# 🔴 PRD-C-104 B4：stage1_anchor/solve.py re-export（纯搬零改）。
#   🔴 必须先于 stage2_variant.gates re-export —— gates.py 顶部 import _solve_one /
#   _check_one_item / _gene_one_item（这三件），故本块须把它们绑回 facade 后 gates 才装载得了。
#   solve.py 对 gates 内符号（_regen_once/_anti_degen_gate/_conservation_ok/gene_gate_check）
#   走函数体内延迟 import（调用期解析）→ 破除 stage1↔stage2 装载期循环。
# ---------------------------------------------------------------------------
from agents.variant.stage1_anchor.solve import (  # noqa: E402
    _check_one_item,
    _gene_one_item,
    _solve_one,
)
# 🔴 PRD-C-104 B4：stage1_anchor/label.py re-export（纯搬零改）。
#   classify/_reanchor 体内延迟 import _emit_mother_card/_emit_figure_stage（mother_card.py，
#   稍后 re-export）→ 调用期解析；_item_dna 须先于 mother_card re-export（_artifact_payload 调它）。
from agents.variant.stage1_anchor.label import (  # noqa: E402
    _item_dna,
    _reanchor_reuse_first_solve,
    classify,
)
# 🔴 PRD-C-104 B3a：stage2_variant/{prompts,difficulty}.py re-export（纯搬零改）。
#   置于此处（所有 __init__ 依赖已定义、图 wiring 之前）→ 子模块顶部 from agents.variant import
#   取依赖时本模块命名空间已就绪，无循环；调用方按 variant.X / V.X 仍解析得到。
# ---------------------------------------------------------------------------
from agents.variant.stage2_variant.prompts import (  # noqa: E402
    GENERATE_PROMPT,
    REGEN_PROMPT,
)
from agents.variant.stage2_variant.difficulty import (  # noqa: E402
    _dna_factors_for_grade,
    grade_variant_item,
    _grade_difficulty,
)
from agents.variant.stage2_variant.generate import (  # noqa: E402
    normalize_knobs,
    recipe_from_knobs,
    _parse_generated_items,
    generate,
)
from agents.variant.stage2_variant.gates import (  # noqa: E402
    _anti_degen_gate,
    _conservation_ok,
    _surface_check,
    _regen_once,
    solve_explain,
    _model_conservation_check,
    gene_gate_check,
    gene_gate,
)
from agents.variant.stage2_variant.assemble import (  # noqa: E402
    assemble,
)

graph = StateGraph(VariantState)
# 🔴 PRD-C-100 B1a 塌缩入口：mother_opus_entry 替代 analyze+mother_precheck（新图入口）。
#   🔴 PRD-A-021 R4·F19：旧 analyze/mother_precheck 退役节点 + 其 edge + path_map 死映射已删
#   （route_entry 不再路由到它们，节点不可达 = 编译期死岛）。两函数体仍保留 = 防引用 / 留档，
#   但不再注册进图。注意：`analyze` 字符串在 variant_model("analyze") 模型槽 + conv_trace marker
#   处仍 LIVE，未动。classify 保留 = 低置信确认 resume 的「+1 次 opus 池注入重锚」路径（D3）。
graph.add_node("mother_opus_entry", mother_opus_entry)
graph.add_node("classify", classify)
graph.add_node("await_review", await_mother_review)  # B5·母题卡硬停闸（置 awaiting_mother_review + END）
graph.add_node("clarify", clarify)
graph.add_node("entry_lowconf_block", entry_lowconf_block)  # 🔴 R2a·闸4·读图低置信前置拦截（建议换图，不进 classify）
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
graph.add_node("editor_entry", editor_entry)  # 🔴 PRD-A-021 S1·结构化编辑/重生入口（应用 op + 清 check → 下游 solve_explain 发真帧）
graph.add_node("patch", patch)
graph.add_node("ask_clarify", ask_clarify)
graph.add_node("persist_to_bank", persist_to_bank)
graph.add_node("ask_for_image", ask_for_image)
graph.add_node("require_login", require_login)

graph.set_conditional_entry_point(
    route_entry,
    {
        # 🔴 PRD-C-100 B1a：新图入口 → 塌缩节点（替 analyze）
        "mother_opus_entry": "mother_opus_entry",
        # 🔴 PRD-A-021 R4·F19：'analyze':'analyze' 死映射已删（route_entry 永不返回 'analyze'，
        #   节点也已退役不再注册 → 留着会 path_map 目标悬空编译报错）。
        "generate": "generate",
        "parse": "parse_instruction",
        # 🔴 B2·母题确认 resume（config 回传确认章 id）→ 直奔 classify（带确认章接闸B）
        "classify": "classify",
        # 🔴 PRD-A-021 S1·结构化编辑/重生 op（经 /stream 带 editor_op）→ editor_entry
        "editor_entry": "editor_entry",
        # 🔴 'ask' 必落真节点（ask_for_image），不能直连 END —— 否则首轮无节点产消息，回复为空
        "ask": "ask_for_image",
        # 🔴 身份硬闸：无登录态 → 提示重登（同上，必落真节点）
        "auth": "require_login",
        # 🔴 PRD-A-021 R2a·闸4（BUG-04）：读图极低置信 resume → 前置拦截建议换图（不进 classify）
        "entry_lowconf_block": "entry_lowconf_block",
    },
)
graph.add_edge("ask_for_image", END)
graph.add_edge("require_login", END)

# 🔴 PRD-C-100 B1a·塌缩入口出口路由：
#   低置信弹窗（awaiting_mother_confirm）→ END 等确认（下一轮 route_entry 见 confirmed_chapter_id
#     → classify 池注入重锚 +1 次 opus，D3）；错误早退（_entry_finalized=False）→ END（消息已发）；
#   高置信 finalize → 复用 gate_after_classify（定死闸）→ await_review 硬停 / clarify。
def after_mother_entry(state: VariantState) -> Literal["await_review", "clarify", "done"]:
    if state.get("awaiting_mother_confirm"):
        return "done"
    if not state.get("_entry_finalized"):
        return "done"
    return gate_after_classify(state)


graph.add_conditional_edges(
    "mother_opus_entry",
    after_mother_entry,
    {"await_review": "await_review", "clarify": "clarify", "done": END},
)


# 🔴 PRD-A-021 R4·F19：旧 analyze→mother_precheck 退役链的 after_analyze 路由 + 两条 edge
#   （analyze→{mother_precheck,END}、mother_precheck→END）已删（节点已退役，留着 = 引用未知节点
#   编译报错）。新图入口走 mother_opus_entry（见上 after_mother_entry）。
# 🔴 B5·classify 不再直连 generate：定死 → await_review（母题卡硬停闸，置 awaiting_mother_review
#   + END，等老师点「开始举一反三」经 route_entry resume → generate）；没定死 → clarify。
graph.add_conditional_edges(
    "classify", gate_after_classify, {"await_review": "await_review", "clarify": "clarify"}
)
graph.add_edge("await_review", END)
graph.add_edge("clarify", END)
graph.add_edge("entry_lowconf_block", END)  # 🔴 R2a·闸4·拦截后 END（等老师换图 / 坚持确认）


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

# 🔴 PRD-A-021 S1·editor_entry 出口：内容变动（清了 check）→ solve_explain 重验 + 发真「程序验算」帧
#   → assemble 收口快照（solve_explain→assemble 既有边）；无未判题/无题 → END（已自带 check / 友好提示）。
graph.add_conditional_edges(
    "editor_entry", after_editor_entry, {"solve_explain": "solve_explain", "done": END}
)

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
