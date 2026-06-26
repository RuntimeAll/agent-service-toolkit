# -*- coding: utf-8 -*-
"""PRD-C-102 批1·确定性难度评级模块（4 档：送分/巩固/中档/压轴）。

🔴 架构铁律（PRD-C-102 §11 / codeplace-C/CLAUDE.md §4）：
- **纯确定性函数**：零 LLM、零网络、零全局可变状态。本模块禁 import openai / anthropic /
  langchain / core.llm；只吃「确定性可抽因子」，绝不读 LLM 自评难度字段（dim4_difficulty /
  difficulty_reason）。pass/fail 与档位判决只采信代码规则，永不采信 LLM 自评。
- 一切闸门有降级路径：解析抽不出 K/R → 用题面字数哨兵 L 兜底判档、标 ⚠（warn），绝不报错。

公开入口（单一）：``grade(dna: dict) -> dict``
  输入 dna = 一道题的「确定性原料」聚合（由调用方从库/HTTP 组装，本模块不连库）：
    {
      "model_kind_hits": [{"modelId","name","model_kind","link_count"}],  # 题→biz_question_model→biz_solution_model
      "skeleton": [...]            # solution_skeleton（JSON 数组，已 parse）
      "analysis_text": str,        # 解析纯文本（由 analyze_block_json 抽出）
      "stem_text": str,            # 题面纯文本
      "kp_count": int|None,        # KG 锚定的不同知识点数（可空，空则从 analysis 推 K）
      "verify_kind": str|None, "dna_type": str|None,  # 几何标识（G 用）
    }
  输出账单 = {
      "level": 1..4, "levelName": str,
      "modelHits": [{modelId,name,tier,freqBand}], "proxy": bool,
      "K":int,"R":int,"D":int,"T":int,"G":int,"L":int,
      "rule": "L4(a)"|..., "warn": bool, "notes": [str]
  }

人读事实源 = PRD-C-102 §3.1（4 档定义）/ §3.3（7 因子）。改判档口径先改 PRD 再改本模块。
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 模型轴常量（PRD §3.2 / §10）
# ---------------------------------------------------------------------------
TIER_BASE = 1   # 基础阶
TIER_HIGH = 2   # 高阶
FREQ_LOW = 1    # 一次性低频
FREQ_HIGH = 2   # 高频通用

# 🔴 PRD-C-103 WS1 订正（2026-06-26）：V911 **已 apply**，biz_solution_model.difficulty_tier/
#   freq_band 列已存在且已填。本函数 = grade()(regex 重打 868 题路径) 在「模型对象没带表真值
#   tier_int」时的 model_kind 代理兜底，**不是举一反三主链取值入口**（主链走 grade_observed →
#   _resolve_tier，优先吃表列 tier_int）。代理仍保留作降级路径（契约 §异常：tier 列缺→代理+⚠）。
def model_tier(model_kind: str | None) -> int:
    """模型难度阶代理（降级用）：gold→高阶(2)，derived/其余→基础阶(1)。"""
    return TIER_HIGH if (model_kind or "").strip().lower() == "gold" else TIER_BASE


# 考频带阈值（PRD §3.2「按阈值分带」）：链接数 ≥ 阈值 = 高频通用，否则一次性低频。
# 🔴 初值 8（任务约定）。本批 868 题模型链接数最大仅 3（见交付报告校准线索），故现阶段所有
#   已链模型都落「一次性低频」——这是 V911 阈值待用真实链接分布重标的已知校准点，不是 bug。
FREQ_BAND_THRESHOLD = 8


def freq_band(link_count: int | None) -> int:
    """考频带代理：链接数 ≥ 阈值 → 高频通用(2)，否则一次性低频(1)。"""
    try:
        n = int(link_count or 0)
    except (TypeError, ValueError):
        n = 0
    return FREQ_HIGH if n >= FREQ_BAND_THRESHOLD else FREQ_LOW


# ---------------------------------------------------------------------------
# 高阶策略关键词（MODEL 轴代理：没链模型的题用解析里的高阶策略关键词探测，标 proxy:true）
# PRD §3.1 任务约定的「分类讨论/构造/转化/数形结合/整体代换/换元/裂项/归纳/新定义/阅读迁移」。
# ---------------------------------------------------------------------------
HIGH_STRATEGY_KEYWORDS: tuple[str, ...] = (
    "分类讨论", "分类", "构造", "转化", "化归", "数形结合", "整体代换", "整体思想",
    "换元", "裂项", "归纳", "新定义", "阅读迁移", "迁移", "待定系数",
    "数学归纳", "反证", "极端", "赋值",
)

# K（知识点综合度）：解析中独立依据连接词（PRD §3.3「由…得/根据…/因为…所以」独立依据数）。
K_DEP_PATTERNS: tuple[str, ...] = (
    "由", "根据", "因为", "所以", "因此", "故", "可得", "得到", "又", "则",
)

# T（陷阱/隐含约束）：仅档内微调（PRD §3.3）。
TRAP_KEYWORDS: tuple[str, ...] = (
    "舍去", "排除", "检验", "范围", "双值", "两解", "端点", "不符", "不成立",
    "舍", "增根", "取等", "讨论",
)

# G（数形结合）：题干含「如图」或 verify_kind/dna_type 标几何（仅微调）。
GEO_HINTS: tuple[str, ...] = ("如图", "图所示", "图中", "几何", "图形")

# 去 LaTeX 噪声用：剥 $...$ 分隔符 / \cmd / 花括号，留可视字符算字数（L 哨兵）。
_LATEX_NOISE_RE = re.compile(r"\\[a-zA-Z]+|[\${}^_\\]|\\left|\\right")
_WS_RE = re.compile(r"\s+")

# 递进引用（D）：后问引用前问才计入递进。
PROGRESSION_REFS: tuple[str, ...] = ("由(1)", "由（1）", "同理", "由第", "结合(1)", "在(1)", "根据(1)", "由上")

# 小问编号模式（D 计数）。
_SUBQ_PARENS = re.compile(r"[（(]\s*([1-9１-９])\s*[）)]")
_SUBQ_CIRCLED = re.compile(r"[①②③④⑤⑥]")


# ---------------------------------------------------------------------------
# 因子抽取（全部确定性）
# ---------------------------------------------------------------------------
def _strip_latex(text: str) -> str:
    s = _LATEX_NOISE_RE.sub("", text or "")
    return _WS_RE.sub("", s)


def _dep_count(analysis_text: str) -> int:
    """解析中独立依据连接词计数（「由/根据/因为…所以/故/可得」），封顶 8 防长解析虚高。"""
    t = analysis_text or ""
    if not t:
        return 0
    hits = sum(t.count(kw) for kw in K_DEP_PATTERNS)
    return min(hits, 8)


def extract_K(analysis_text: str, kp_count: int | None) -> int:
    """K 知识点综合度（PRD §3.3：解析独立依据数 或 KG 锚定的不同知识点数，取你能稳定抽的）。

    🔴 本批数据特性：biz_question_knowledge 每题恒锚 1 个主知识点（kp 分布 max=avg=1），
    单读 kp_count 会把 K 永久钉死在 1、L4(a) 的 K≥3 条件永不触发。故 kp_count>=2 才采信
    KG 计数（真多锚），kp_count<=1 退回解析独立依据数（真综合度信号）。两者取较大值，
    让「单锚但解析多步依据」的综合题也能抬 K。"""
    dep = _dep_count(analysis_text)
    kp = int(kp_count) if (kp_count is not None and kp_count > 0) else 0
    if kp >= 2:
        return max(kp, dep)
    return dep


def extract_R(skeleton: Any) -> int:
    """R 推理链步数 = solution_skeleton 步骤条数（JSON 数组长度）。"""
    if isinstance(skeleton, (list, tuple)):
        return len(skeleton)
    return 0


def extract_D(stem_text: str, analysis_text: str) -> int:
    """D 递进小问深度：题干小问数；后问引用前问（解析含「由(1)/同理」）才计入递进。

    返回值语义：
      - 0/1 = 单问或平行多问（不计递进）→ D<2；
      - 2   = 存在递进小问（≥2 小问且后问引用前问）→ D≥2 触发中档/压轴条件。
    """
    stem = stem_text or ""
    n_paren = len(set(_SUBQ_PARENS.findall(stem)))
    n_circ = len(_SUBQ_CIRCLED.findall(stem))
    n_sub = max(n_paren, n_circ)
    if n_sub < 2:
        return min(n_sub, 1)  # 0 或 1
    # ≥2 小问：仅当解析体现「后问引用前问」才判递进（D=2），否则平行多问按 1。
    blob = (analysis_text or "") + stem
    if any(ref in blob for ref in PROGRESSION_REFS):
        return 2
    return 1


def extract_T(analysis_text: str, stem_text: str) -> int:
    """T 陷阱/隐含约束（仅档内微调）：解析+题干命中陷阱词去重计数。"""
    blob = (analysis_text or "") + (stem_text or "")
    return sum(1 for kw in TRAP_KEYWORDS if kw in blob)


def extract_G(stem_text: str, verify_kind: str | None, dna_type: str | None) -> int:
    """G 数形结合（仅微调）：题干含「如图」或 verify_kind/dna_type 标几何 → 1，否则 0。"""
    stem = stem_text or ""
    if any(h in stem for h in GEO_HINTS):
        return 1
    vk = (verify_kind or "") + (dna_type or "")
    if "几何" in vk or "图" in vk:
        return 1
    return 0


def extract_L(stem_text: str) -> int:
    """L 题面字数：去 LaTeX 噪声后字符数（单调兜底哨兵）。"""
    return len(_strip_latex(stem_text or ""))


def _proxy_high_strategy_hits(analysis_text: str, stem_text: str) -> list[str]:
    """没链模型时的 MODEL 代理：从解析/题面探测高阶策略关键词。返回命中词列表。"""
    blob = (analysis_text or "") + (stem_text or "")
    return [kw for kw in HIGH_STRATEGY_KEYWORDS if kw in blob]


# ---------------------------------------------------------------------------
# 判档（PRD §3.1：L4>L3>L2>L1 取最高命中，规则编号写进账单 rule 字段）
# ---------------------------------------------------------------------------
def _decide_level(
    *, high_hits: int, high_freqhigh_hits: int, low_freq_model_hits: int,
    K: int, R: int, D: int, has_high_model: bool,
) -> tuple[int, str]:
    """按 PRD §3.1 规则表落档，返回 (level, rule)。L4>L3>L2>L1。

    参数：
      high_hits          命中高阶模型总数（含代理探测计 1 个/题）
      high_freqhigh_hits 命中「高阶且高频通用」模型数（L3 主条件）
      low_freq_model_hits 命中「一次性/低频」模型数（L4 关键刀）
      has_high_model     是否有任一高阶模型命中（= high_hits>=1）
    """
    # --- L4 压轴 ---
    # (高阶模型≥1 且 K≥3 且 R≥4) 或 高阶模型≥2 或 命中一次性低频模型 或 (D≥2 且高阶模型≥1)
    if has_high_model and K >= 3 and R >= 4:
        return 4, "L4(a)"
    if high_hits >= 2:
        return 4, "L4(b)"
    if low_freq_model_hits >= 1:
        return 4, "L4(c)"
    # 🔴 L4(d) 对齐 rubric 正本（难度评级 skill §四，2026-06-26 修）：D≥2 递进 + 真综合
    #   = (高阶模型≥2 或 K≥3)。**仅 D≥2 + 单个高阶/proxy → 归 L3(b)，不够 L4**（旧代码漏掉
    #   这条限定、误把"递进+1高阶"全判 L4，导致 L4 占比虚高 39%）。
    if D >= 2 and (high_hits >= 2 or K >= 3):
        return 4, "L4(d)"

    # --- L3 中档 ---
    # 命中 ≥1 高阶模型(且该模型高频通用) 或 D≥2 递进
    if high_freqhigh_hits >= 1:
        return 3, "L3(a)"
    if D >= 2:
        return 3, "L3(b)"

    # --- L2 巩固 ---
    # 仅基础阶模型 + 一道弯（R∈{2,3} 且 (K≥2 或 G 或 T≥1)）—— 见调用处把 G/T 折进 one_turn。
    # 这里仅按 R 与「一道弯」布尔判（one_turn 由上层算好传 R/K，G/T 已并入）。
    # 实际条件在 grade() 里组装；此函数只处理 model 主轴 + K/R/D。

    # --- L1 送分 ---
    return 0, ""  # 占位：L2/L1 在 grade() 内按完整因子裁（见下）


def grade(dna: dict) -> dict:
    """确定性难度评级。输入见模块 docstring；输出 = 理由账单 dict。

    🔴 纯确定性、无 LLM、无网络。降级：解析抽不出 K/R（R==0 且 K==0）→ 用 L 哨兵兜底判档、warn=True。
    """
    notes: list[str] = []
    warn = False

    model_hits_raw = dna.get("model_kind_hits") or []
    skeleton = dna.get("skeleton")
    analysis_text = dna.get("analysis_text") or ""
    stem_text = dna.get("stem_text") or ""
    kp_count = dna.get("kp_count")
    verify_kind = dna.get("verify_kind")
    dna_type = dna.get("dna_type")

    # ---- 因子抽取 ----
    K = extract_K(analysis_text, kp_count)
    R = extract_R(skeleton)
    D = extract_D(stem_text, analysis_text)
    T = extract_T(analysis_text, stem_text)
    G = extract_G(stem_text, verify_kind, dna_type)
    L = extract_L(stem_text)

    # ---- MODEL 轴 ----
    model_hits: list[dict] = []
    proxy = False
    high_hits = 0
    high_freqhigh_hits = 0
    low_freq_model_hits = 0

    if model_hits_raw:
        for m in model_hits_raw:
            tier = model_tier(m.get("model_kind"))
            fb = freq_band(m.get("link_count"))
            model_hits.append({
                "modelId": m.get("modelId") or m.get("model_id"),
                "name": m.get("name"),
                "tier": tier,
                "freqBand": fb,
            })
            if tier == TIER_HIGH:
                high_hits += 1
                if fb == FREQ_HIGH:
                    high_freqhigh_hits += 1
                else:
                    low_freq_model_hits += 1
    else:
        # 没链模型（868 里约 850 道）：用高阶策略关键词探测作 MODEL 代理。
        proxy = True
        kws = _proxy_high_strategy_hits(analysis_text, stem_text)
        if kws:
            # 代理命中 → 视作 1 个高阶模型（无 freq 信息，保守按高频通用，不直接拉到 L4）。
            high_hits = 1
            high_freqhigh_hits = 1
            notes.append(f"proxy 高阶策略命中: {','.join(kws[:5])}（待 B2 打标补链接）")
        else:
            notes.append("无模型命中（proxy 未探测到高阶策略）")

    has_high_model = high_hits >= 1

    # ---- 降级哨兵：解析抽不出 K/R ----
    if R == 0 and K == 0:
        warn = True
        # L 哨兵兜底（PRD §3.3：<80 几乎非 L4，>200 几乎非 L1）。
        if L < 80:
            level, rule = 1, "L1(degrade-L)"
        elif L > 200:
            level, rule = 3, "L3(degrade-L)"
        else:
            level, rule = 2, "L2(degrade-L)"
        notes.append(f"⚠ 解析抽不出 K/R，按题面字数 L={L} 哨兵兜底判档")
        return _bill(level, model_hits, proxy, K, R, D, T, G, L, rule, warn, notes)

    # ---- 主判档：L4 > L3 ----
    level, rule = _decide_level(
        high_hits=high_hits, high_freqhigh_hits=high_freqhigh_hits,
        low_freq_model_hits=low_freq_model_hits, K=K, R=R, D=D, has_high_model=has_high_model,
    )

    if level == 0:
        # ---- L2 巩固：仅基础阶模型 + 一道弯（R∈{2,3} 且 (K≥2 或 G 或 T≥1)）----
        one_turn = R in (2, 3) and (K >= 2 or G >= 1 or T >= 1)
        if one_turn:
            level, rule = 2, "L2(a)"
        else:
            # ---- L1 送分：无高阶 且 R≤2 且 K≤1（其余）----
            level, rule = 1, "L1(a)"

    return _bill(level, model_hits, proxy, K, R, D, T, G, L, rule, warn, notes)


_LEVEL_NAMES = {1: "送分", 2: "巩固", 3: "中档", 4: "压轴"}
# 🔴 对齐 sys_dict biz_question_difficulty（value 1-4 = 基础/中等/较难/压轴）：账单 level 整数
#    直接 = 字典 value，写库写整数即自动对齐前台字典；levelDictLabel 仅人读留痕。
_LEVEL_DICT_LABEL = {1: "基础", 2: "中等", 3: "较难", 4: "压轴"}


# ---------------------------------------------------------------------------
# grade_observed —— 吃 labeler 观察的因子直接判档（第 4 步打标专用入口）
#
# 与 grade(dna) 的区别：grade 走 regex 从解析/题面【重抽】因子（重打已入库 868 题用，
# 历史 LLM 自评不可信时的兜底）；grade_observed 直接信 labeler 多模态读详解判的因子
# （tier/freqHint 由 labeler 看详解实判，比正则探测准 —— SKILL §3 / CLAUDE.md §4③：
# pass/fail 判决只读工具返回值，但「客观因子观察」labeler 读详解比 regex 准），不再 regex 重抽。
# 复用同一套 _decide_level 判档规则（L4>L3>L2>L1）。grade() 完全不动（regex 路径并存）。
# ---------------------------------------------------------------------------
def _tier_int(tier: Any) -> int:
    """labeler 的 tier 文本（高阶/基础）→ 阶整数。"""
    return TIER_HIGH if str(tier or "").strip() == "高阶" else TIER_BASE


def _freq_int(freq_hint: Any) -> int:
    """labeler 的 freqHint 文本（通法/一次性）→ 考频带整数。"""
    return FREQ_HIGH if str(freq_hint or "").strip() == "通法" else FREQ_LOW


# 🔴 PRD-C-103 WS1：模型轴优先吃「控制面板表真值整数」（biz_solution_model.difficulty_tier/
#   freq_band），缺则回退 labeler 文本（高阶/通法），再缺退默认基础阶/低频。这是「改表→反控难度」
#   的取值入口：锚定模型带 tier_int/freq_int（由 model_anchor 反查 SQL 选出）时直读表，确保改表生效。
def _resolve_tier(m: dict) -> int:
    """模型 → 阶整数：tier_int(表真值，1/2) > tier 文本(高阶/基础) > 默认基础阶。"""
    ti = m.get("tier_int")
    if ti in (TIER_BASE, TIER_HIGH):
        return int(ti)
    if m.get("tier") is not None:
        return _tier_int(m.get("tier"))
    return TIER_BASE


def _resolve_freq(m: dict) -> int:
    """模型 → 考频带整数：freq_int(表真值，1/2) > freqHint 文本(通法/一次性) > 默认低频。"""
    fi = m.get("freq_int")
    if fi in (FREQ_LOW, FREQ_HIGH):
        return int(fi)
    if m.get("freqHint") is not None:
        return _freq_int(m.get("freqHint"))
    return FREQ_LOW


def grade_observed(
    *,
    model_hits: list[dict] | None,
    K: int,
    R: int,
    D: int,
    G: int = 0,
    high_strategies: list[str] | None = None,
) -> dict:
    """据 labeler 观察的因子判档（不再 regex 重抽）。输出账单同 grade()。

    🔴 **无陷阱 T 维**（维护者裁定砍掉）：判档只用 模型轴 + K/R/D/G，不再用 T 做 L2 一道弯微调。
    model_hits = labeler models[] = [{id,name,isNew,tier(高阶/基础),freqHint(通法/一次性),...}]。
    - tier/freq 来自 labeler（读详解实判），本函数只做整数化 + 计数 + 套 _decide_level 规则。
    - 没命中模型但 high_strategies 非空 → 与 grade() 同口径，视作 1 个高阶高频模型代理（proxy=true）。
    - 降级：R==0 且 K==0（labeler 没给可用因子）→ warn=True、按 L2 兜底（无题面字数 L 哨兵可用）。
    """
    notes: list[str] = []
    warn = False
    K = int(K or 0)
    R = int(R or 0)
    D = int(D or 0)
    G = int(G or 0)
    T = 0  # 无 T 维：账单字段保留（与 grade() 账单同形），恒 0 不参与判档

    model_hits_in = model_hits or []
    out_hits: list[dict] = []
    high_hits = 0
    high_freqhigh_hits = 0
    low_freq_model_hits = 0
    proxy = False

    for m in model_hits_in:
        # 🔴 WS1：优先吃表真值整数 tier_int/freq_int（锚定模型带），缺则回退 labeler 文本。
        tier = _resolve_tier(m)
        fb = _resolve_freq(m)
        out_hits.append({
            "modelId": m.get("id") or m.get("modelId"),
            "name": m.get("name"),
            "tier": tier,
            "freqBand": fb,
        })
        if tier == TIER_HIGH:
            high_hits += 1
            if fb == FREQ_HIGH:
                high_freqhigh_hits += 1
            else:
                low_freq_model_hits += 1

    # 🔴 高阶策略命中永远能抬档（修 2026-06-26）：highStrategies 是高阶信号，**不被基础模型淹没**。
    #   旧逻辑只在 models 为空时才看 highStrategies → 一道「分类讨论+基础模型」的题高阶信号被丢、误掉 L2。
    #   现在：只要 highStrategies 非空且尚无高阶模型命中，补 1 个高阶高频代理（proxy=true），抬到 L3。
    kws = [k for k in (high_strategies or []) if str(k).strip()]
    if kws and high_hits == 0:
        proxy = True
        high_hits = 1
        high_freqhigh_hits = 1
        notes.append(f"proxy 高阶策略命中: {','.join(kws[:5])}")
    elif not model_hits_in and not kws:
        notes.append("无模型命中（labeler 未观察到高阶策略）")

    has_high_model = high_hits >= 1

    # 降级：labeler 没给可用因子
    if R == 0 and K == 0:
        warn = True
        level, rule = 2, "L2(degrade-observed)"
        notes.append("⚠ labeler 未给可用 K/R 因子，兜底 L2")
        return _bill(level, out_hits, proxy, K, R, D, T, G, 0, rule, warn, notes)

    # 主判档（L4>L3，与 grade() 同规则表）
    level, rule = _decide_level(
        high_hits=high_hits, high_freqhigh_hits=high_freqhigh_hits,
        low_freq_model_hits=low_freq_model_hits, K=K, R=R, D=D, has_high_model=has_high_model,
    )
    if level == 0:
        # L2 巩固两条路：
        #   (a) 一道弯：仅基础阶模型 + R∈{2,3} 且 (K≥2 或 G)。**不再用 T**（维护者裁定）。
        #   (b) 繁琐多步计算：无高阶模型但纯长链硬算 R≥4 → 巩固档不是送分
        #       （维护者裁定 2026-06-26：多步计算虽无高阶套路，落 L2 中等）。
        one_turn = R in (2, 3) and (K >= 2 or G >= 1)
        long_chain = R >= 4
        if one_turn:
            level, rule = 2, "L2(a)"
        elif long_chain:
            level, rule = 2, "L2(b-longchain)"
        else:
            level, rule = 1, "L1(a)"

    return _bill(level, out_hits, proxy, K, R, D, T, G, 0, rule, warn, notes)


def _bill(level, model_hits, proxy, K, R, D, T, G, L, rule, warn, notes) -> dict:
    """组装理由账单（PRD §10 契约 + AC3）。"""
    return {
        "level": level,
        "levelName": _LEVEL_NAMES.get(level, "?"),
        "levelDictLabel": _LEVEL_DICT_LABEL.get(level, "?"),
        "modelHits": model_hits,
        "proxy": proxy,
        "K": K, "R": R, "D": D, "T": T, "G": G, "L": L,
        "rule": rule,
        "warn": warn,
        "notes": notes,
    }
