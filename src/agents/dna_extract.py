"""PRD-C-014 B1 · 单题 DNA 抽取器（独立可复用模块）。

把一道题（题面 + 年级 + 年级叶子池 + 标签复用池）抽成 **DNA 契约 v1**：
  main_kp / secondary_kps / qtype / exam_type / skeleton / hard_points / tags /
  scene / difficulty / flags（校验标记随行）。

事实源：
  - 维度 SSOT = claude-code-sign/22-题目维度-唯一事实源.md（两步锚定、禁造词、难点克制、
    标签复用、考察类型闭集 10 种、难度四档 LLM rubric）。
  - prompt 语义搬自 tools/e1_dna_probe.py（实测 81.5% 一次过）；模块化重排、生产差异：
    ① LLM 走 core.relay_pool（per-call 覆盖用 nano 档），不直读 .env RELAY_POOL；
    ② 难度改 LLM 四档 rubric 断言（22-SSOT §2，2026-06-12），不再代码从难点数派生；
    ③ 叶子池/标签池由调用方注入（生产走 RuoYi lazyTree/tagsByKp，单测走 fixture），
       本模块不直读 tsv、不直连库。

🔴 铁律（22-SSOT）：
  - 两步锚定：年级 → 叶子池 → LLM 只能从池里选 id；**禁造词**=输出 id 必须在池内，
    否则按越界处理（主 kp 越界 = anchored.code 缺失 → 上层走 clarify；副 kp 越界 = 丢弃 +flag）。
  - 难点克制：基础题必空（prompt 红线「宁空不凑」），个数代码重算不信 LLM 自报。
  - 标签优先复用池内词；考察类型 ∈ 闭集 10；难度由 LLM 拿 rubric 断言（评级非判对错）。
  - DNA 抽取是「打标」不是「判对错」——判对错归闸B sympy，本模块不碰。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from core import relay_pool, settings

# ---------------------------------------------------------------------------
# 闭集 / 常量（22-SSOT §1）
# ---------------------------------------------------------------------------
EXAM_TYPES: list[str] = [
    "概念辨析", "直接计算", "公式套用", "性质判定", "证明推理",
    "应用建模", "作图", "探究归纳", "阅读理解迁移", "纠错",
]
QTYPES: list[str] = ["选择", "填空", "解答"]
SECONDARY_KP_MAX = 3  # 副 kp ≤3（22-SSOT §1 #3）
TAGS_MIN, TAGS_MAX = 3, 6  # 标签 3~6（22-SSOT §1 #9）
SCENE_MAX_LEN = 64  # 场景 ≤64 字
DIFFICULTY_MIN, DIFFICULTY_MAX = 1, 4  # 难度四档（22-SSOT §2，1~4 对齐绝对 rubric）
DIFFICULTY_FALLBACK = 2  # 难度缺/非法 → 常规档兜底（NOT NULL 链最后保险）

# 校验标记（随 DNA 输出，上层透传进 auxTags / need_anchor_review）
FLAG_MAIN_KP_OOB = "main_kp_oob"  # 主 kp 越界（锚定失败）→ 上层走 clarify
FLAG_SECONDARY_KP_OOB = "secondary_kp_oob"  # 副 kp 越界（已丢弃该项）
FLAG_EXAM_TYPE_OOB = "exam_type_oob"  # 考察类型出闭集（已置 None）
FLAG_QTYPE_OOB = "qtype_oob"  # 题型出闭集（已尽力归一/置 None）
FLAG_DIFFICULTY_FALLBACK = "difficulty_fallback"  # 难度缺/非法 → 兜底 2
FLAG_TAG_POOL_EMPTY = "tag_pool_empty"  # 复用池拉空（降级继续，T4）
FLAG_LLM_PARSE_FAIL = "llm_parse_fail"  # LLM 返回非 JSON / 解析失败 → 空 DNA 兜底
FLAG_LLM_ERROR = "llm_error"  # LLM 调用异常 → 空 DNA 兜底

# 题型别名归一（与 variant_support.QTYPE_MAP 同口径的中文侧）
_QTYPE_ALIAS: dict[str, str] = {
    "选择": "选择", "选择题": "选择", "单选": "选择", "单选题": "选择",
    "填空": "填空", "填空题": "填空",
    "解答": "解答", "解答题": "解答", "计算": "解答", "计算题": "解答",
    "应用": "解答", "应用题": "解答", "证明": "解答", "证明题": "解答", "大题": "解答",
}

# ---------------------------------------------------------------------------
# prompt（语义搬自 e1_dna_probe.E1_PROMPT；难度改 rubric 断言 + 复用池可空兜底）
# 排版：固定规则段在前（吃前缀缓存），变动段（年级/题面/池）在后。
# ---------------------------------------------------------------------------
# 🔴 本常量将被拼进会 .format() 的 DNA_PROMPT，故文本内的花括号须双写转义（{{ }}）。
_DIFFICULTY_RUBRIC = """难度四档 rubric（按下面标准判级，不裸问「几星」；难度是评级不是判对错）：
- 4（压轴）：≥2 个真实难点 / 多突破口综合。
- 3（多步综合）：1 个难点，或 考察∈{{证明推理·应用建模·探究归纳}}，或 骨架含【最难步】构造。
- 2（常规）：无难点 + 考察∈{{直接计算·公式套用·性质判定}} + 多步骨架。
- 1（送分）：无难点 +（概念辨析 或 单步骨架）。"""

DNA_PROMPT = (
    """你是浙教版初中数学命题专家。对下面这道题做**打标式 DNA 抽取**，只输出一个 JSON。

"""
    + _DIFFICULTY_RUBRIC
    + """

考察类型闭集（exam_type 只能取其一）：{exam_types}
题型闭集（qtype 只能取其一）：选择 / 填空 / 解答

输出 JSON 结构（不要任何解释文字、不要 markdown fence）：
{{
  "main_kp": {{"id": "池内id", "name": "池内名"}},
  "secondary_kps": [{{"id": "...", "name": "..."}}],   // 0~3 个，解这道题连带必须用到的其他知识点；没有就空数组
  "qtype": "选择/填空/解答 之一",
  "exam_type": "上述闭集之一",
  "skeleton": ["步骤1", "步骤2", ...],                  // 解法骨架；把最难的那一步用【】整步包住（至多一处），如 "【构造全等三角形】"
  "hard_points": ["..."],                               // 🔴 难点=让题目变质的突破口，**只有进阶题才有**。
                                                        // 基础知识考察/纯套公式/直接计算/概念辨析的题必须给空数组 []，宁空不凑。
  "tags": ["...", "..."],                               // 3~6 个检索标签（求什么/用什么定理/什么方法/什么场景）
  "scene": "纯代数 或 一句话场景描述（≤64 字）",
  "difficulty": 1                                       // 按上面 rubric 判的难度档（1~4 整数）
}}

🔴 硬约束（违反 = 程序按越界处理）：
- main_kp / secondary_kps 的 id **只能从下面【知识点候选池】里选**，禁止造词、禁止超纲。
- tags 优先从【标签复用池】里复用；确实没有贴切的才允许造新词，新词必须像池内词一样短。
- 难点克制：宁空不凑——基础题 hard_points 必须是空数组。

【这道题】年级：{grade}

题干：{stem}
标准答案：{answer}
解析：{analyze}

【知识点候选池】（该年级全部叶子知识点，主/副知识点只能从池里选 id）：
{kp_pool}

【标签复用池】（线上高频标签，优先复用；为空则按上面规则自拟）：
{tag_pool}"""
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> Any:
    """剥 markdown fence + JSON 容错（与 variant._parse_json 同范式）。"""
    text = (text or "").strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                pass
    return None


def _norm_qtype(raw: Any) -> str | None:
    s = str(raw or "").strip()
    return _QTYPE_ALIAS.get(s) or (s if s in QTYPES else None)


def _clamp_difficulty(raw: Any) -> int | None:
    try:
        d = int(raw)
    except (TypeError, ValueError):
        return None
    if DIFFICULTY_MIN <= d <= DIFFICULTY_MAX:
        return d
    return max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, d))


def _truncate_scene(raw: Any) -> str:
    s = str(raw or "").strip()
    return s[:SCENE_MAX_LEN] if len(s) > SCENE_MAX_LEN else s


def _build_prompt(
    *, stem: str, answer: str, analyze: str, grade: str,
    leaf_pool: list[tuple[str, str]], tag_pool: list[str],
) -> str:
    kp_pool_text = "\n".join(f"{pid} {name}" for pid, name in leaf_pool) or "（空）"
    tag_pool_text = "、".join(tag_pool) if tag_pool else "（无，请按规则自拟标签）"
    return DNA_PROMPT.format(
        grade=grade or "未知年级",
        stem=stem or "",
        answer=answer or "",
        analyze=analyze or "",
        kp_pool=kp_pool_text,
        tag_pool=tag_pool_text,
        exam_types="/".join(EXAM_TYPES),
    )


def _validate(raw: dict[str, Any], pool_ids: set[str], tag_pool: list[str]) -> dict[str, Any]:
    """代码校验闸：池内校验 / 越界处置 / 闭集校验 / 难度兜底 / 标签复用统计。

    🔴 禁造词铁律：主 kp 越界 → anchored=None（上层据此走 clarify，不放行出题）；
       副 kp 越界 → 丢弃该项 +flag（不报错）。
    🔴 难点个数代码重算（len(hard_points)），不信 LLM 自报。
    """
    flags: list[str] = []

    # --- 主 kp（池内才算锚定成功） ---
    mk = raw.get("main_kp") or {}
    mk_id = str(mk.get("id") or "").strip()
    if mk_id and mk_id in pool_ids:
        main_kp = {"id": mk_id, "name": mk.get("name")}
    else:
        main_kp = None  # 越界/缺失 = 锚定失败
        flags.append(FLAG_MAIN_KP_OOB)

    # --- 副 kp（池内 + ≤3，越界丢弃+flag） ---
    secondary_kps: list[dict[str, Any]] = []
    for s in raw.get("secondary_kps") or []:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or "").strip()
        if sid and sid in pool_ids:
            if sid == mk_id:
                continue  # 副 kp 不重复主 kp
            secondary_kps.append({"id": sid, "name": s.get("name")})
        else:
            flags.append(FLAG_SECONDARY_KP_OOB)
        if len(secondary_kps) >= SECONDARY_KP_MAX:
            break

    # --- 题型（闭集归一） ---
    qtype = _norm_qtype(raw.get("qtype"))
    if qtype is None and raw.get("qtype"):
        flags.append(FLAG_QTYPE_OOB)

    # --- 考察类型（闭集） ---
    exam_type = raw.get("exam_type")
    if exam_type not in EXAM_TYPES:
        flags.append(FLAG_EXAM_TYPE_OOB)
        exam_type = None

    # --- 骨架 ---
    skeleton = [str(x) for x in (raw.get("skeleton") or []) if str(x).strip()]

    # --- 难点（克制；个数代码重算） ---
    hard_points = [str(x) for x in (raw.get("hard_points") or []) if str(x).strip()]

    # --- 标签（3~6，优先复用） ---
    tags = [str(t).strip() for t in (raw.get("tags") or []) if str(t).strip()]
    tags = tags[:TAGS_MAX]
    if tag_pool:
        pool_set = set(tag_pool)
        reused = sum(1 for t in tags if t in pool_set)
    else:
        reused = 0
        flags.append(FLAG_TAG_POOL_EMPTY)

    # --- 场景 ---
    scene = _truncate_scene(raw.get("scene"))

    # --- 难度（LLM rubric 断言；缺/非法兜底 2，与难点不再代码派生） ---
    difficulty = _clamp_difficulty(raw.get("difficulty"))
    if difficulty is None:
        difficulty = DIFFICULTY_FALLBACK
        flags.append(FLAG_DIFFICULTY_FALLBACK)

    return {
        "main_kp": main_kp,
        "secondary_kps": secondary_kps,
        "qtype": qtype,
        "exam_type": exam_type,
        "skeleton": skeleton,
        "hard_points": hard_points,
        "hard_point_count": len(hard_points),  # 代码重算，不信 LLM
        "tags": tags,
        "tag_reused_count": reused,  # 复用统计（G5b 复用率监控用）
        "scene": scene,
        "difficulty": difficulty,
        "flags": flags,
    }


def empty_dna(flags: list[str] | None = None) -> dict[str, Any]:
    """空 DNA 兜底（LLM 失败/解析失败时返回；锚定失败 → main_kp=None 触发上层 clarify）。"""
    return {
        "main_kp": None,
        "secondary_kps": [],
        "qtype": None,
        "exam_type": None,
        "skeleton": [],
        "hard_points": [],
        "hard_point_count": 0,
        "tags": [],
        "tag_reused_count": 0,
        "scene": "",
        "difficulty": DIFFICULTY_FALLBACK,
        "flags": list(flags or []),
    }


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
async def extract_dna(
    *,
    stem: str,
    answer: str = "",
    analyze: str = "",
    grade: str,
    leaf_pool: list[tuple[str, str]],
    tag_pool: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """单题 DNA 抽取（DNA 契约 v1）。

    参数：
      stem/answer/analyze：题面三要素（answer/analyze 可空）。
      grade：年级文案（仅入 prompt 提示；叶子池由调用方按年级备好）。
      leaf_pool：该年级叶子知识点池 [(id, name), ...]——LLM 只能从池内选 id（两步锚定第二步）。
      tag_pool：标签复用池（该 kp 高频词，可空；空则降级继续 +FLAG_TAG_POOL_EMPTY）。
      model：per-call 模型覆盖（默认走 settings.LLM_MODEL_LIGHT = nano 降本；锚定/抽取轻活）。

    返回 DNA 契约 v1 dict（键见 empty_dna）：
      main_kp(池内真 id 或 None) / secondary_kps(≤3 池内 id) / qtype / exam_type /
      skeleton / hard_points / hard_point_count / tags / tag_reused_count / scene /
      difficulty(1~4) / flags(校验标记随行)。

    🔴 锚定失败（main_kp 越界/缺失）→ main_kp=None + FLAG_MAIN_KP_OOB，**不报错**：
       上层（variant.classify）据此走 clarify，不放行出题（根治 C-013 凭 LLM 置信裸放行）。
    🔴 LLM 调用/解析异常 → 返回 empty_dna(+flag)，绝不抛（DNA 抽取是增强不卡死）。
    """
    tag_pool = list(tag_pool or [])
    pool_ids = {str(pid) for pid, _ in leaf_pool}
    prompt = _build_prompt(
        stem=stem, answer=answer, analyze=analyze, grade=grade,
        leaf_pool=leaf_pool, tag_pool=tag_pool,
    )
    chosen_model = model if model is not None else settings.LLM_MODEL_LIGHT

    try:
        # 走 relay_pool（熔断转移）；JSON 中间产物 → skip_stream（不外放 token 流）。
        resp, _relay, _model_used, _fb = await relay_pool.ainvoke_failover(
            [HumanMessage(content=prompt)],
            max_tokens=settings.VARIANT_MAX_TOKENS,
            tags=["skip_stream"],
            model=chosen_model,
        )
    except Exception:  # noqa: BLE001 — 抽取是增强，调用失败 → 空 DNA 兜底（触发上层 clarify）
        return empty_dna([FLAG_LLM_ERROR, FLAG_MAIN_KP_OOB])

    text = _content_text(resp)
    raw = _parse_json(text)
    if not isinstance(raw, dict):
        return empty_dna([FLAG_LLM_PARSE_FAIL, FLAG_MAIN_KP_OOB])

    return _validate(raw, pool_ids, tag_pool)


# ---------------------------------------------------------------------------
# 🔴 G5b 标签复用池接线（PRD-C-014 T2）：首锚抽 DNA 时主 kp 未知 → tag_pool 只能空，
# 标签全靠 LLM 自拟+exact 撞，实测复用率仅 76.7%（设计本意 ≥80%，PRD §3.5/H6）。
# 修法（方案 a，单一额外 LLM 调用、改动最小）：extract_dna 先锚到 main_kp，**之后**按 kp 拉
# 「该 kp 高频标签池」，做一次**只重选 tags 维**的窄 LLM 调用，从池里复用，merge 回 DNA。
#   - 触发条件：main_kp 锚定成功 + 该 kp 标签池非空（否则照旧空池降级 + flag，不卡死）。
#   - 调用预算：仅成功锚定路径 +1 次窄调用（拉池失败/池空 → 0 次额外调用）。
# ---------------------------------------------------------------------------
_TAGS_REFINE_PROMPT = (
    """你是浙教版初中数学检索标签师。下面给一道题 + 它锚定的核心考点 + 该考点【线上高频标签复用池】。
请为这道题挑 3~6 个**最贴切**的检索标签（求什么/用什么定理/什么方法/什么场景）。

🔴 硬约束：
- **优先从【标签复用池】里复用**贴切的词；池里实在没有合适的，才允许补少量新词（新词要像池内词一样短）。
- 只输出一个 JSON：{{"tags": ["...", "..."]}}，不要解释、不要 markdown fence。

【这道题】题干：{stem}
当前核心考点：{main_kp_name}
当前已抽标签（供参考，可替换）：{cur_tags}

【标签复用池】（线上高频，优先复用）：
{tag_pool}"""
)


async def refine_tags_with_pool(
    dna: dict[str, Any],
    *,
    stem: str,
    tag_pool: list[str],
    model: str | None = None,
) -> dict[str, Any]:
    """🔴 T2：拿 kp 专属标签池重选 tags 维，merge 回已抽的 DNA（不动其余维度）。

    入参 dna = extract_dna 产物（已含 main_kp/tags/flags）。tag_pool = 该 main_kp 的高频标签池。
    返回新 dict（浅拷贝改 tags / tag_reused_count / flags）：
      - tag_pool 为空 → 原样返回（调用方应在池空时直接跳过，不该走到这里；双保险）。
      - 窄 LLM 调用失败/解析失败 → 原样返回（**保留原 tags**，降级不卡死，铁律④）。
      - 成功 → tags 重选自池、复用统计重算、去掉 FLAG_TAG_POOL_EMPTY。
    """
    pool = [t for t in (tag_pool or []) if str(t).strip()]
    if not pool:
        return dna

    cur_tags = list(dna.get("tags") or [])
    main_kp_name = ((dna.get("main_kp") or {}).get("name")) or "（未知）"
    prompt = _TAGS_REFINE_PROMPT.format(
        stem=stem or "",
        main_kp_name=main_kp_name,
        cur_tags="、".join(cur_tags) if cur_tags else "（无）",
        tag_pool="、".join(pool),
    )
    chosen_model = model if model is not None else settings.LLM_MODEL_LIGHT
    try:
        resp, _relay, _model_used, _fb = await relay_pool.ainvoke_failover(
            [HumanMessage(content=prompt)],
            max_tokens=settings.VARIANT_MAX_TOKENS,
            tags=["skip_stream"],
            model=chosen_model,
        )
    except Exception:  # noqa: BLE001 — 窄调用失败 → 保留原 tags 降级（不卡死）
        return dna

    raw = _parse_json(_content_text(resp))
    if not isinstance(raw, dict):
        return dna
    new_tags = [str(t).strip() for t in (raw.get("tags") or []) if str(t).strip()][:TAGS_MAX]
    if not new_tags:
        return dna  # LLM 没给标签 → 保留原 tags（不退化成空）

    pool_set = set(pool)
    out = dict(dna)
    out["tags"] = new_tags
    out["tag_reused_count"] = sum(1 for t in new_tags if t in pool_set)
    # 池非空且已重选 → 去掉空池 flag（避免误报 76.7% 复用率的根因 flag）
    out["flags"] = [f for f in (dna.get("flags") or []) if f != FLAG_TAG_POOL_EMPTY]
    return out


def _content_text(resp: Any) -> str:
    """取 LLM resp 文本（思考型 content 可能是 parts list；只取 text，不外放 reasoning）。"""
    c = getattr(resp, "content", resp)
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
