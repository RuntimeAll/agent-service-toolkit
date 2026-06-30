# -*- coding: utf-8 -*-
r"""题目打标 labeler（整书录入第 4 步 · 装配式 · 多模态）。

正本 = `.claude/skills/题目打标/SKILL.md`（§2 装配式 prompt + 6 块输出 schema）。
心智照搬 `agents/mother_opus.py`：多模态 HumanMessage（text + image_url）+ response_format
json_schema + 低温 + 超时 + 「机器重算」（hardPointCount=len 等，不信 LLM 自报）。**与 mother 的根本
区别：labeler 读已有详解打标、绝不重新解题**（费 token 且可能解错；SKILL §2 / §0 铁律）。

本模块全是纯函数 + 一个 LLM 调用包装（`invoke` 注入，便于单测桩与批量/在线共用）：
  - LABEL_SCHEMA / RESPONSE_FORMAT：6 块 json_schema（难度**不进 schema**，交 difficulty.py 确定性算）。
  - build_label_prompt：装配式 prompt（读详解不解题 / 模型阶看详解实判不被 gold 标签骗 /
    稀有度看通法 vs 现造 / 新招式 isNew 提议不入库）。
  - label_one：1 次多模态中转调用（🔴 配图必须 base64 内联，sui-xiang 逆向渠道不抓远程 URL）。
  - parse_label：原始文本 → 结构化打标 dict（剥 markdown fence + JSON 容错 + 机器重算）。
  - label_and_grade：调 core.difficulty.grade_observed 出难度档 + 账单。

难度由代码算（SKILL §3）：labeler 只产 factors（K/R/D/T/G + 模型命中），永不写 LLM 自评难度数字。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from core import difficulty
from core.contracts import EXAM_TYPES  # 收口到契约模块；re-export，与 dna_extract 同源，labeler 侧机器归一用

# ---------------------------------------------------------------------------
# 调用参数（仿 mother_opus：低温稳 JSON、读图慢留超时头）
# ---------------------------------------------------------------------------
LABEL_TEMPERATURE = 0.1
LABEL_TIMEOUT_S = 180.0
# 🔴 题型 = 系统字典 sys_dict_data[dict_type='biz_question_type'] 闭集（录入步17 seed 的 8 类）。
#   labeler **不自由生成题型**，只能从注入的字典 label 里选（与 EXAM_TYPES/leaf_pool 同注入模式）。
#   调用方（批量/在线）从库查出 [(value,label)] 传 build_label_prompt(qtype_dict=...) + parse_label(qtype_dict=)。
#   缺字典时回退本兜底集（防离线单测/库不可达，但生产必注入）。
QTYPE_DICT_FALLBACK: list[str] = ["选择题", "判断题", "应用题", "填空题", "解答题", "作图题", "计算题", "证明题"]

# variationProfile 九算子（SKILL §2③ / 举一反三策略 9 算子）。
VARIATION_OPS: list[str] = [
    "数值变换", "结构变换", "情境变换", "条件增删", "逆向变换",
    "推广变换", "升维变换", "分类讨论化", "定值化",
]
# 🔴 「会把题变成另一类更难题」的算子默认不可用（除非确能产同型变式）。变式系数旋钮要选出
#    「像母题的题」而非「另出一道难题」（维护者裁定·prompt 注意力修正 §3）。
VARIATION_OPS_DEFAULT_OFF: set[str] = {"升维变换", "分类讨论化", "推广变换"}

TIERS = {"高阶", "基础"}
FREQ_HINTS = {"通法", "一次性"}

# ---------------------------------------------------------------------------
# 6 块输出 json_schema（SKILL §2 输出结构：models/factors/variationProfile/material/
# questionType/dna）。🔴 难度不进 schema —— 由 difficulty.py 据 factors+models 确定性算。
# ---------------------------------------------------------------------------
LABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # ① 解题模型命中（采用候选 id / isNew 提议新；tier/freq 看详解实判，不被 gold 标签骗）
        "models": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "命中候选模型 id；isNew 提议新模型时留空"},
                    "name": {"type": "string"},
                    "isNew": {"type": "boolean", "description": "候选里没有、提议新模型=true（不入库，批后归并转正）"},
                    "tier": {"type": "string", "enum": ["高阶", "基础"], "description": "看详解实际招式判，不被 gold 标签骗"},
                    "freqHint": {"type": "string", "enum": ["通法", "一次性"], "description": "稀有度：可复用通法 vs 现造一次性"},
                    "triggerFeature": {"type": "string", "description": "触发特征（isNew 时给）"},
                    "action": {"type": "string", "description": "招式结论（isNew 时给）"},
                },
                "required": ["name", "isNew", "tier", "freqHint"],
            },
        },
        # ② 难度因子（喂 difficulty.py，不判难度档）。🔴 不含「陷阱 T」维（维护者裁定砍掉，
        #    防 LLM 为填满字段硬凑陷阱；难度判档也不再用 T）。
        "factors": {
            "type": "object",
            "properties": {
                "highStrategies": {"type": "array", "items": {"type": "string"},
                                   "description": "高阶策略命中名（分类讨论/构造/数形结合/整体代换/换元/裂项/归纳…），无则空数组"},
                "K": {"type": "integer", "description": "知识点综合度（独立依据/锚定知识点数）"},
                "R": {"type": "integer", "description": "推理链步数（解法骨架步数）"},
                "D": {"type": "integer", "description": "递进小问深度（0/1=平行，2=后问引用前问）"},
                "G": {"type": "integer", "description": "数形结合（含图/几何=1 否则 0）"},
            },
            "required": ["highStrategies", "K", "R", "D", "G"],
        },
        # ③ variationProfile 九算子各 {可用,风险}。只标「产同类变式（认得出同型题）」的算子可用；
        #    升维/分类讨论化/推广 这类会变成「另一类更难题」的默认 usable=false（除非确能产同型）。
        "variationProfile": {
            "type": "object",
            "properties": {op: {
                "type": "object",
                "properties": {
                    "usable": {"type": "boolean", "description": "能产出同类（同型）变式才 true"},
                    "risk": {"type": "string", "description": "低/中/高 或一句话风险说明"},
                },
                "required": ["usable", "risk"],
            } for op in VARIATION_OPS},
            "required": VARIATION_OPS,
        },
        # ④ material 变式生成原料。parametricSlots = 结构化对象数组（不 str 化）。
        "material": {
            "type": "object",
            "properties": {
                "parametricSlots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slot": {"type": "string", "description": "可参数化槽位名"},
                            "current": {"type": "string", "description": "母题当前取值"},
                            "constraint": {"type": "string", "description": "可变范围/约束"},
                        },
                        "required": ["slot"],
                    },
                    "description": "可参数化槽位（结构化对象，非字符串）",
                },
                "modelingFrame": {"type": "string", "description": "建模框架/母题骨架一句话"},
                "conditions": {"type": "array", "items": {"type": "string"},
                               "description": "题设条件清单（守恒/可增删依据）"},
            },
            "required": ["parametricSlots", "modelingFrame", "conditions"],
        },
        # ⑤ questionType（读详解判，闭集 = 注入的题型字典 label，不自由生成）
        "questionType": {"type": "string", "description": "题型字典闭集之一（见 prompt 注入列表）"},
        # ⑥ dna 其余维（读详解产）
        "dna": {
            "type": "object",
            "properties": {
                "solutionSkeleton": {"type": "array", "items": {"type": "string"},
                                     "description": "解题骨架步骤序列；最难一步用【】整步包住"},
                "hardPoints": {"type": "array", "items": {"type": "string"}, "description": "认知难点（克制·宁空不凑；基础/直接计算/送分题必空）"},
                "breakthroughPoints": {"type": "array", "items": {"type": "string"}, "description": "突破口（克制·宁空不凑；基础/直接计算/送分题必空）"},
                "assessmentType": {"type": "string", "description": "考察类型闭集10之一"},
                "scenario": {"type": "string", "description": "一句话场景 或 纯代数"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "3~6 检索标签，禁近义增生"},
            },
            "required": ["solutionSkeleton", "hardPoints", "breakthroughPoints",
                         "assessmentType", "scenario", "tags"],
        },
    },
    "required": ["models", "factors", "variationProfile", "material", "questionType", "dna"],
}

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "question_label", "schema": LABEL_SCHEMA, "strict": False},
}


# ---------------------------------------------------------------------------
# 装配式 prompt（SKILL §2：读详解不解题 + 模型阶/稀有度判别铁律 + isNew 提议）
# ---------------------------------------------------------------------------
def build_label_prompt(
    *,
    stem: str,
    analysis: str,
    answer: str = "",
    candidate_models: list[dict[str, Any]] | None = None,
    knowledge_points: list[dict[str, Any]] | None = None,
    grade_text: str = "",
    source_marks: str = "",
    qtype_dict: list[str] | None = None,
) -> str:
    """组装配式打标 prompt（多模态文本部分；配图由 label_one 以 image_url 块附加）。

    candidate_models：[{id,name,category?,model_kind?,trigger_feature?,action_conclusion?}]（本章 gold + 通用）。
    knowledge_points：[{id,name}]（KG 锚定主 kp + 邻居）。
    qtype_dict：系统题型字典 label 闭集（sys_dict_data biz_question_type）；缺则回退兜底集。
    """
    cms = candidate_models or []
    kps = knowledge_points or []
    qtypes = qtype_dict or QTYPE_DICT_FALLBACK

    def _model_row(m: dict[str, Any]) -> str:
        kind = (m.get("model_kind") or "").strip()
        kind_tag = f"[{kind}]" if kind else ""
        trig = (m.get("trigger_feature") or "").strip()
        act = (m.get("action_conclusion") or "").strip()
        cat = (m.get("category") or "").strip()
        extra = []
        if cat:
            extra.append(f"类:{cat}")
        if trig:
            extra.append(f"触发:{trig}")
        if act:
            extra.append(f"招式:{act}")
        tail = f"（{' '.join(extra)}）" if extra else ""
        return f"· {m.get('id', '')} {m.get('name', '')}{kind_tag}{tail}"

    model_block = "\n".join(_model_row(m) for m in cms) or "（无候选模型，simple 题 models 留空数组）"
    kp_block = "、".join(f"{k.get('id', '')} {k.get('name', '')}" for k in kps) or "（无）"
    ops_line = "/".join(VARIATION_OPS)
    off_line = "/".join(sorted(VARIATION_OPS_DEFAULT_OFF))
    exam_types = "/".join(EXAM_TYPES)
    qtype_line = " / ".join(qtypes)
    grade_line = f"本题年级：{grade_text}。" if grade_text else ""
    marks_line = f"【教辅标记】{source_marks}\n" if source_marks else ""

    return f"""你是初中数学「解题模型 + 变式」打标专家。下面给你一道**已入库**题的题干、配图、已有标准答案与详解，及该题相关候选解题模型与知识点。{grade_line}
🔴 **请【读详解】完成打标，禁止自己重新解题**（题已被解开、详解可信；重解费 token 且可能解错，详解里的真实招式才是打标依据）。

================ 🔴 克制铁律（最重要·别为填满字段而注水） ================
打标的价值在「准」不在「满」。绝大多数字段**宁空不凑**：
- **难点/突破口**：基础题、纯套公式、直接计算、概念辨析、送分题 → hardPoints **必须留空数组**、breakthroughPoints **必须留空数组**。**只有真正存在「不想到就做不出」的认知难点时才填**，且照实写、有几个写几个，绝不为凑数硬编。（反例：一道「去分母解一元一次方程」是直接计算，不许编出 3 个难点 3 个突破口。）
- **解题模型**：详解真有可复用套路才给；纯计算/套定义题 models 留空数组，不硬安模型。
- **高阶策略 highStrategies**：详解真用了才填；常规计算无则空数组。
- 评判标准：你填的每一项都要经得起「这题真有这个吗」的反问。

================ 🔴 模型匹配铁律（你只【匹配模型】，不判难度——难度阶由系统模型表确定） ================
1. **你的任务 = 判这题命中了候选里的哪个/哪些解题模型，不是评难度**。难度档由系统按【模型表里
   该模型的难度阶】确定性算，**你不打难度档、不给星级、不判 tier**（命中候选 id 即可，tier 以模型表为准、会覆盖你填的）。
2. **命中候选 → 填其 id**；候选里没有、但详解确有【可跨题复用的解题套路】→ isNew=true 提议
   （给 name/triggerFeature/action + 建议 tier，**仅供人工转正参考**，本步不入库、不影响本题难度）。
3. 🔴 **基础运算技巧 ≠ 解题模型，别硬往高阶模型上套（这是难度虚高的头号来源）**：
   提取公因数、凑整、乘法分配律/交换律/结合律、通分、约分、去括号、移项、合并同类项、直接代入求值——
   这些是**基本运算操作**，不是解题模型。一道靠这些就能做的计算题，**models 留空数组**。
   （反例：「7⅓×(-5)+7×(-7⅓)-12×7⅓ 提出公因数 7⅓」是分配律逆用 = 基础运算技巧，**绝不是「整体代换」**；
    只有「把反复出现的复杂组合式设为一个整体、一次代入」才算真整体代换。）
4. **freqHint 稀有度**（仅 isNew 提议时填，转正参考）：可跨题复用 = 通法 / 本题现凑 = 一次性。

================ 题目 ================
【题干】{stem or "（见配图）"}
【标准答案】{answer or "（见详解）"}
【标准详解】{analysis or "（无，谨慎打标）"}
{marks_line}================ 候选解题模型（本章 gold + 通用） ================
{model_block}

================ 相关知识点（KG 锚定 + 邻居） ================
{kp_block}

================ 题型字典（闭集·只选不造） ================
🔴 questionType **只能从下面字典里原样选一个**（系统已定，禁自由生成/改写）：{qtype_line}

================ 输出（6 块，一个 JSON，不要解释、不要 markdown fence） ================
① models[]：**匹配**命中的模型。命中候选→填 id+name+isNew=false（tier/freqHint 可空，以模型表为准）；
   候选没有但确有可复用套路→isNew=true+name+triggerFeature+action+建议tier(转正参考)。
   **纯基础运算技巧/套公式 → 空数组（见匹配铁律3）**。🔴 你不判难度档。
② factors{{highStrategies[], K, R, D, G}}：客观因子（K/R/D/G 喂确定性难度算法；**你不判难度档/不给星级**）。**不含陷阱 T 维**。
   - highStrategies：详解里**真正动用**的解题策略名（仅作描述/检索，**不直接定难度**——难度看①匹配的模型表），无则空数组。**基础运算技巧（提公因数/凑整/通分…）不算策略、不写这里**。
   - K=知识点综合度（解析独立依据数 或 锚定知识点数）；R=推理链步数(=solutionSkeleton 步数)；D=递进小问深度(0/1平行,2=后问引用前问)；G=数形结合(含图/几何=1否则0)。
③ variationProfile：九算子各 {{usable:bool, risk:"低/中/高 或说明"}} —— {ops_line}。
   🔴 **只标能产出「同类变式」（一眼认得出是同型题）的算子 usable=true**。**{off_line} 这类会把题变成「另一类更难的题」的，默认 usable=false**（除非你确信它仍产同型变式）。目标=旋钮选出来的是「像母题的题」，不是「另出一道难题」。
④ material{{parametricSlots[], modelingFrame, conditions[]}}：变式生成原料。parametricSlots 是**结构化对象数组** [{{"slot":"槽位名","current":"当前值","constraint":"可变范围"}}]，**不要写成字符串**。
⑤ questionType：从上面题型字典里**原样选一个**（不自由生成）。
⑥ dna{{solutionSkeleton[](最难一步用【】整步包住), hardPoints[], breakthroughPoints[], assessmentType, scenario, tags[]}}：
   - assessmentType：闭集10选1 = {exam_types}。
   - hardPoints/breakthroughPoints：见上「克制铁律」——基础/纯套公式/直接计算/概念辨析/送分题 **必空数组**。
   - tags：3~6 检索标签，禁近义增生（同义只留一个）。
   - scenario：一句话场景 或 "纯代数"。

JSON 结构：
{{
  "models": [{{"id": "DZ06 或 空", "name": "模型名", "isNew": false, "tier": "高阶/基础", "freqHint": "通法/一次性", "triggerFeature": "", "action": ""}}],
  "factors": {{"highStrategies": [], "K": 1, "R": 2, "D": 0, "G": 0}},
  "variationProfile": {{ {", ".join(f'"{op}": {{"usable": false, "risk": "低"}}' for op in VARIATION_OPS)} }},
  "material": {{"parametricSlots": [{{"slot": "", "current": "", "constraint": ""}}], "modelingFrame": "", "conditions": []}},
  "questionType": "从题型字典选一个",
  "dna": {{"solutionSkeleton": [], "hardPoints": [], "breakthroughPoints": [], "assessmentType": "上述闭集之一", "scenario": "", "tags": []}}
}}"""


# ---------------------------------------------------------------------------
# 多模态中转调用（1 次）。🔴 配图必须 base64 内联（sui-xiang 逆向渠道不抓远程 URL）
# ---------------------------------------------------------------------------
async def label_one(
    *,
    stem: str,
    analysis: str,
    image_b64_list: list[str] | None,
    candidate_models: list[dict[str, Any]] | None,
    knowledge_points: list[dict[str, Any]] | None,
    invoke: Any,
    model: str,
    answer: str = "",
    grade_text: str = "",
    source_marks: str = "",
    qtype_dict: list[str] | None = None,
    max_tokens: int | None = None,
) -> str:
    """一次多模态中转调用打标。返回原始文本（调用方 parse_label）。

    image_b64_list：每项 = 纯 base64 串（不含 data: 前缀）或已带 data: 前缀的 url；
      本函数统一封成 `data:image/...;base64,...` 内联块。**绝不传远程 URL**（逆向渠道不抓）。
    qtype_dict：系统题型字典 label 闭集（注入 prompt，questionType 只能从中选）。
    invoke：异步可调用 `invoke(messages, model=, temperature=, response_format=, timeout=, max_tokens=) -> str`。
      在线走 variant._ainvoke_text；批量走 c102_label_batch 的直连中转 invoke。
    🔴 不吞异常：超时/失败由调用方接住（与 mother 同纪律）。
    """
    prompt = build_label_prompt(
        stem=stem, analysis=analysis, answer=answer,
        candidate_models=candidate_models, knowledge_points=knowledge_points,
        grade_text=grade_text, source_marks=source_marks, qtype_dict=qtype_dict,
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in image_b64_list or []:
        if not b64:
            continue
        url = b64 if str(b64).startswith("data:") else f"data:image/png;base64,{b64}"
        content.append({"type": "image_url", "image_url": {"url": url}})

    msg = HumanMessage(content=content)
    kw: dict[str, Any] = dict(
        model=model,
        temperature=LABEL_TEMPERATURE,
        response_format=RESPONSE_FORMAT,
        timeout=LABEL_TIMEOUT_S,
    )
    if max_tokens and max_tokens > 0:
        kw["max_tokens"] = max_tokens
    return await invoke([msg], **kw)


# ---------------------------------------------------------------------------
# parse（剥 markdown fence + JSON 容错 + 机器重算，仿 opus_to_dna）
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fence(raw: str) -> str:
    s = (raw or "").strip()
    s = _FENCE_RE.sub("", s)
    return s.strip()


def _loads_lenient(raw: str) -> dict[str, Any]:
    """容错 JSON：剥 fence → 直接 loads → 抠第一个 {...} 平衡块再 loads。"""
    s = _strip_fence(raw)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        pass
    # 抠首个平衡花括号块
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(s[start:i + 1])
                    return v if isinstance(v, dict) else {}
                except Exception:
                    return {}
    return {}


def _norm_qtype(raw: Any, qtype_dict: list[str] | None = None) -> str | None:
    """questionType 归一到注入的题型字典闭集（label 原样）。

    精确命中字典 label → 取之；否则做最小别名容错（去「题」字后缀匹配、计算↔计算题等），
    仍不中 → None（由调用方按 dim2 兜底，绝不放行字典外的自由词）。
    """
    s = str(raw or "").strip()
    pool = qtype_dict or QTYPE_DICT_FALLBACK
    if s in pool:
        return s
    # 别名容错：补「题」字后再匹配；去「题」字匹配
    cand = s if s.endswith("题") else (s + "题")
    if cand in pool:
        return cand
    base = s[:-1] if s.endswith("题") else s
    for p in pool:
        if p == base or p[:-1] == base:
            return p
    return None


def _int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _str_list(raw: Any, *, cap: int | None = None) -> list[str]:
    out = [str(x).strip() for x in (raw or []) if str(x).strip()]
    return out[:cap] if cap else out


def parse_label(raw: str, *, qtype_dict: list[str] | None = None) -> dict[str, Any]:
    """原始文本 → 结构化打标 dict（6 块归一 + 机器重算，仿 opus_to_dna）。

    机器重算（不信 LLM 自报）：
      - models[].tier/freqHint 闭集归一；isNew 项 id 强制清空。
      - factors K/R/D/G 取整（**无 T 维**）；R 优先用 solutionSkeleton 长度回填（骨架是 R 的事实源）。
      - hardPointCount = len(breakthroughPoints)（不取 LLM 自报）。
      - questionType 归一到注入的题型字典闭集；assessmentType 归一 EXAM_TYPES；tags 截 6；scenario 截 64。
      - variationProfile：升维/分类讨论化/推广 缺省补 usable=false（默认 off，防瞎跑成另一类题）。
      - parametricSlots 保结构（对象数组，不 str 化）。
      - 🔴 克制兜底：assessmentType=直接计算 且 models=[] 且 highStrategies=[] → 强制清空难点/突破口。
    parse 失败 → 返回 {"_parse_ok": False, ...} 兜底空结构（不抛，让批量继续）。
    """
    obj = _loads_lenient(raw)
    if not obj:
        return {
            "_parse_ok": False,
            "models": [], "factors": {"highStrategies": [], "K": 0, "R": 0, "D": 0, "G": 0},
            "variationProfile": {}, "material": {"parametricSlots": [], "modelingFrame": "", "conditions": []},
            "questionType": None,
            "dna": {"solutionSkeleton": [], "hardPoints": [], "breakthroughPoints": [],
                    "hardPointCount": 0, "assessmentType": None, "scenario": "", "tags": []},
        }

    # ① models
    models: list[dict[str, Any]] = []
    for m in obj.get("models") or []:
        if not isinstance(m, dict):
            continue
        is_new = bool(m.get("isNew"))
        mid = "" if is_new else str(m.get("id") or "").strip()
        name = str(m.get("name") or "").strip()
        if not name and not mid:
            continue
        tier = str(m.get("tier") or "").strip()
        tier = tier if tier in TIERS else "基础"
        freq = str(m.get("freqHint") or "").strip()
        freq = freq if freq in FREQ_HINTS else "一次性"
        models.append({
            "id": mid, "name": name, "isNew": is_new, "tier": tier, "freqHint": freq,
            "triggerFeature": str(m.get("triggerFeature") or "").strip(),
            "action": str(m.get("action") or "").strip(),
        })

    # ② factors
    f = obj.get("factors") or {}
    dna_raw = obj.get("dna") or {}
    skeleton = _str_list(dna_raw.get("solutionSkeleton"))
    # R 事实源 = 骨架步数（LLM 自报 R 与骨架不符时以骨架为准）
    r_skel = len(skeleton)
    factors = {
        "highStrategies": _str_list(f.get("highStrategies")),
        "K": _int(f.get("K")),
        "R": r_skel if r_skel else _int(f.get("R")),
        "D": _int(f.get("D")),
        "G": _int(f.get("G")),
    }

    # ③ variationProfile（保留 LLM 形态，缺算子补默认 usable=false；
    #    升维/分类讨论化/推广 即使 LLM 漏给也补 False，但 LLM 显式给 True 时尊重）
    vp_raw = obj.get("variationProfile") or {}
    variation_profile: dict[str, Any] = {}
    for op in VARIATION_OPS:
        v = vp_raw.get(op)
        if isinstance(v, dict):
            variation_profile[op] = {"usable": bool(v.get("usable")), "risk": str(v.get("risk") or "").strip()}
        else:
            # 缺省：默认 off 集补 False（与默认 on 集同为 False，但语义上前者是「会变难题」的克制项）
            variation_profile[op] = {"usable": False, "risk": ""}

    # ④ material（parametricSlots 保结构对象，不 str 化）
    mat_raw = obj.get("material") or {}
    slots: list[dict[str, Any]] = []
    for s in mat_raw.get("parametricSlots") or []:
        if isinstance(s, dict):
            slot = {
                "slot": str(s.get("slot") or s.get("name") or "").strip(),
                "current": str(s.get("current") or s.get("currentValue") or "").strip(),
                "constraint": str(s.get("constraint") or "").strip(),
            }
            if slot["slot"] or slot["current"]:
                slots.append(slot)
        elif str(s).strip():
            # 兼容 LLM 偶发吐字符串：塞进 slot 名
            slots.append({"slot": str(s).strip(), "current": "", "constraint": ""})
    material = {
        "parametricSlots": slots,
        "modelingFrame": str(mat_raw.get("modelingFrame") or "").strip(),
        "conditions": _str_list(mat_raw.get("conditions")),
    }

    # ⑤ questionType（归一到注入的题型字典闭集；缺则用 dim2 题型由调用方补）
    qtype = _norm_qtype(obj.get("questionType"), qtype_dict)

    # ⑥ dna 机器重算
    breakthrough = _str_list(dna_raw.get("breakthroughPoints"))
    hard_points = _str_list(dna_raw.get("hardPoints"))
    exam_type = str(dna_raw.get("assessmentType") or "").strip()
    exam_type = exam_type if exam_type in EXAM_TYPES else None
    scene = str(dna_raw.get("scenario") or "").strip()[:64]
    tags = _str_list(dna_raw.get("tags"), cap=6)

    # 🔴 克制兜底（防 LLM 不听话）：直接计算 + 无模型 + 无高阶策略 → 强制清空难点/突破口
    if exam_type == "直接计算" and not models and not factors["highStrategies"]:
        hard_points = []
        breakthrough = []

    dna = {
        "solutionSkeleton": skeleton,
        "hardPoints": hard_points,
        "breakthroughPoints": breakthrough,
        "hardPointCount": len(breakthrough),  # 机器重算，不信自报
        "assessmentType": exam_type,
        "scenario": scene,
        "tags": tags,
    }

    return {
        "_parse_ok": True,
        "models": models,
        "factors": factors,
        "variationProfile": variation_profile,
        "material": material,
        "questionType": qtype,
        "dna": dna,
    }


# ---------------------------------------------------------------------------
# 难度（确定性算 · 交 difficulty.grade_observed，不让 LLM 判档）
# ---------------------------------------------------------------------------
def _enrich_models_tier_from_table(
    models: list[dict[str, Any]], candidate_models: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """🔴 PRD-C-103 WS2·AC7 打标口径统一：命中已有模型（有 id、非 isNew）的 tier/freq 以**模型表
    真值**为准（candidate_models 由 lookup 带 tier_int/freq_int），覆盖 LLM 文本 tier。

    与 model_anchor.confirm_models 同口径：grade_observed._resolve_tier 优先吃 tier_int(表) → 不论
    走举一反三锚定还是批处理打标，难度都由同一张表驱动（不采信 LLM 自评 tier）。临时模型(isNew)
    无表行 → 保留 LLM 临时 tier 文本（待审草案，转正脚本落表后即成表真值）。
    """
    if not candidate_models:
        return models
    by_id = {str(c.get("id")): c for c in candidate_models if c.get("id")}
    out: list[dict[str, Any]] = []
    for m in models or []:
        m2 = dict(m)
        mid = str(m2.get("id") or "").strip()
        if mid and not m2.get("isNew") and mid in by_id:
            c = by_id[mid]
            if c.get("tier_int") is not None:
                m2["tier_int"] = c.get("tier_int")  # 表真值整数（_resolve_tier 优先吃）
            if c.get("freq_int") is not None:
                m2["freq_int"] = c.get("freq_int")
        out.append(m2)
    return out


def label_and_grade(
    parsed: dict[str, Any],
    candidate_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """据 parse_label 的 factors + models 调 difficulty.grade_observed 出难度档 + 账单。

    返回原 parsed 浅拷贝并挂上 `difficulty` 块（{level,levelName,rule,...}）。
    🔴 难度永不取 LLM 自评：labeler 只产 factors，档由 difficulty.py 据 model tier/freq + K/R/D 算。
    🔴 WS2·AC7：传 candidate_models（同 prompt 那份，带表真值 tier_int/freq_int）→ 命中已有模型
       的 tier/freq 以表为准（与 model_anchor 锚定路径同口径），不采信 LLM 自评 tier。
    """
    out = dict(parsed)
    factors = parsed.get("factors") or {}
    model_hits = _enrich_models_tier_from_table(parsed.get("models") or [], candidate_models)
    out["models"] = model_hits  # 回写 enriched（带表真值）供下游落链/落库
    bill = difficulty.grade_observed(
        model_hits=model_hits,
        K=_int(factors.get("K")),
        R=_int(factors.get("R")),
        D=_int(factors.get("D")),
        G=_int(factors.get("G")),
        high_strategies=factors.get("highStrategies") or [],
    )
    out["difficulty"] = bill
    return out
