# -*- coding: utf-8 -*-
r"""PRD-C-100 B1a · 塌缩入口：opus 一把（读图判年级册+章 + has_figure + 置信 + 解题 + 10维DNA打标）。

🔴 心智（PRD §1.1 + D3）：把旧入口三节点 `analyze(gpt5.4) + mother_precheck(nano) + classify(opus)`
   合并重写为单入口节点 `mother_opus_entry`：
   - **高置信（≥0.80 且无强歧义章）→ 1 次 opus 全 done**（读图判章+解题+10维DNA → 代码锚定 →
     母题卡先出 → 硬停 await_review）。
   - **低置信（<0.80 或 ≥2 强候选章）→ 弹窗确认年级册+章 → resume 走既有 `classify`（+1 次 opus
     按确认章重锚打标，池注入 build_mother_prompt）**。D3「高置信1次/低置信+1重锚」。
   - **带图不再打回**（反转 C-017 G13）：has_figure 仅如实记录，不 reject；母题卡照出，带图切图归 B3。

🔴 控制流重写边界（铁律）：本模块只动**入口段**；变式四节点（generate/gene_gate/solve_explain/
   assemble）+ 闸A基因/闸B sympy 判决**字节级不动**。母题闸A（validate_rich_text）/闸B（anchor_to_chapter）
   仍走 mother_opus.py 的**同一纯函数**（高置信路径在此调用，逻辑与 classify 一致）。

🔴 复用而非重造：opus 调用走 variant._ainvoke_text（自带 conv_trace 计费 + relay 熔断转移）；
   DNA 归一/闸A/闸B = mother_opus.opus_to_dna/validate_rich_text/anchor_to_chapter；模型锚 = model_anchor；
   母题卡帧/合并确认/事实冻结 = variant._emit_mother_card/build_mother_confirm（懒导入 variant 防循环）。

🔴 max_tokens 护栏（B0 H5 实测定 12288，见 settings.MOTHER_OPUS_MAX_TOKENS）：母题节点宽护栏防失控
   不截断（实测峰值 5158，零截断）。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents import dna_extract, model_anchor, mother_opus

# 🔴 PRD-C-104 B2b·解循环依赖：迁进 variant/shared/ 的符号改从 shared 直引（shared 反向不 import
#   variant_entry → 循环断在 shared 层）。仍留在 variant/__init__.py 的符号继续走懒导入 `V.*`。
from agents.variant.shared.llm import _ainvoke_text, _parse_json
from agents.variant.shared.sanitize import _sanitize_rich_text, join_skeleton
from agents.variant.shared.emit import (
    _emit_stage,
    _emit_error,
    _emit_need_confirm,
    _emit_reasoning,
    _emit_richtext_stem,
)
from agents.variant.shared.ruoyi import RuoyiClient

# D1：条件 confirm 触发阈值（置信 < 0.80 或 ≥2 强候选章 → 弹窗）。
CONF_CONFIRM_THRESHOLD = 0.80
# 🔴 BUG-04（2026-06-19）：读图置信「极低」阈值——低于它提示「图可能不适合做母题，建议换清晰图」。
#   与 CONF_CONFIRM_THRESHOLD 分层：0.40~0.80 = 普通待确认；< 0.40 = 加换图建议（不改判定逻辑）。
LOW_CONF_HINT_THRESHOLD = 0.40

# 6 个浙教版教材册（与 analyze 旧闭集对齐，防口径漂移）。
_GRADE_BOOKS = [
    "七年级上册", "七年级下册", "八年级上册", "八年级下册", "九年级上册", "九年级下册",
]


# ---------------------------------------------------------------------------
# opus 一把 schema（在 mother_opus.MOTHER_SCHEMA 上加「判年级册+章+置信+候选」头）
# ---------------------------------------------------------------------------
def _entry_schema() -> dict[str, Any]:
    base = mother_opus.MOTHER_SCHEMA
    props = dict(base["properties"])
    props["gradeBook"] = {"type": "string", "description": "6 册之一或空串"}
    props["chapter"] = {"type": "string", "description": "章名（如:第2章 一元二次方程），判不出留空"}
    props["gradeCandidates"] = {"type": "array", "items": {"type": "string"},
                                "description": "强候选年级册（拿不准时给 1~2 个）"}
    props["chapterCandidates"] = {"type": "array", "items": {"type": "string"},
                                  "description": "强候选章（≥2 个=章歧义，触发确认）"}
    props["confidence"] = {"type": "number", "description": "对年级册+章判定的整体置信 0~1"}
    return {
        "type": "object",
        "properties": props,
        "required": list(base["required"]) + ["gradeBook", "confidence"],
    }


ENTRY_SCHEMA = _entry_schema()
RESPONSE_FORMAT_ENTRY: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "mother_entry", "schema": ENTRY_SCHEMA, "strict": False},
}


# ---------------------------------------------------------------------------
# opus 一把 prompt（判章 + 解题 + 开集 10 维打标；无叶子池注入，kp 开集捕获，代码后锚）
#
# 🔴 PRD-C-100 B2 缓存接缝（D7/AC8）：prompt 严格分层
#   稳定前缀(system, ENTRY_SYSTEM_PREFIX，字节级稳定、无 teacher_id/时间戳/随机/utterance)
#   ‖ 变量后缀(user message：题图 + query + **teacher 记忆后置**)。
#   aigeek 自动缓存（无需手打 cache_control）；teacher 记忆**必须在后缀**（per-teacher，进前缀毁
#   多用户共享缓存——多用户接缝预留）。entry 前缀 ~1500 token < opus 4096 缓存门槛（单用户期不命中，
#   D7 接缝优先、省钱推多用户期；conv_trace.cached_tokens 记录对账）。画图链/图片重生不挂缓存。
# ---------------------------------------------------------------------------
def _build_entry_system_prefix(*, sentinel: bool = False, preset: bool = False) -> str:
    """稳定系统前缀（模块加载期算一次，字节级稳定）。仅依赖闭集常量（EXAM_TYPES/_GRADE_BOOKS），
    绝不内插 utterance/teacher_id/时间戳/随机。

    🔴 R2b·U8：sentinel=True → richText 三段走免转义哨兵框（settings.MOTHER_RICHTEXT_SENTINEL 开时用）；
       sentinel=False（默认）→ 旧式整 JSON（response_format 硬锁 10 维，字节级不变）。
    🔴 preset=True（用户拍板·两 prompt 隔离）：老师已在系统中定好年级 → **砍掉①判年级整段**，让模型把
       全部注意力放在读图 + 解题（年级册由人类消息后置给出）。四版（json/sentinel × preset/非preset）
       均模块加载期各算一次缓存（前缀字节稳定 = aigeek 缓存友好），运行期按 settings + preset 选。"""
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    books = "、".join(_GRADE_BOOKS)
    # 解题纪律（通用·preset/非preset 共用；不掺单一题型操作）。
    solve_discipline = (
        "（一步步算到最终答案，不抄图、不跳步，关键步代入原题验算）。🔴 **解题纪律**：含图先看清图中"
        "图形/标注/已知量再列式；条件拿不准时取最自然的一种解释解到底、不中途乱换；验算通过的结果即唯一终答，"
        "不要因「感觉不对/不合常理」推翻已验算的正确结论、反复改答案。"
    )
    if preset:
        head = (
            "你是浙教版初中数学**解题专家** + 题库打标师。🔴 本题**年级册已由老师在系统中确定**"
            "（下方人类消息给出），你**绝不再自行判定年级册**——gradeBook 直接照老师给的原样填、confidence 给 1.0、"
            "gradeCandidates / chapterCandidates 留空数组（章 chapter 可据题如实填）。把全部精力放在两件事"
            "（🔴 母题解题是地基，③打标全建立在②你解对的基础上）：\n"
            f"② **像严谨的解题者那样真正把题解出来**{solve_discipline}\n"
            "③ **据你②解出的答案做 10 维 DNA 打标**（母题是所有变式的基准，解错则全变式跟着错——务必稳准）。"
        )
        grade_block = ""
        stage_lock_who = "老师确定的"
    else:
        head = (
            "你是浙教版初中数学**解题专家** + 题库打标师。看这张母题图，按顺序做三件事并一次输出"
            "（🔴 母题解题是地基，③打标全建立在②你解对的基础上）：\n"
            "① **判年级册 + 章 + 置信度**（判定母题属于哪个教材册、哪一章，给整体置信 0~1，拿不准给候选）；\n"
            f"② **像严谨的解题者那样真正把题解出来**{solve_discipline}\n"
            "③ **据你②解出的答案做 10 维 DNA 打标**（母题是所有变式的基准，解错则全变式跟着错——务必稳准）。"
        )
        grade_block = (
            "================ 判年级册 + 章（闭集 + 候选） ================\n"
            f"- gradeBook **只能从这 6 册选一个**：{books}。判不出 → 留空串 \"\"、confidence 给低、gradeCandidates 给你最可能的 1~2 个。\n"
            "- chapter：章名（如「第2章 一元二次方程」）；判不出留空。**章有歧义（说不清是哪一章）→ chapterCandidates 给 ≥2 个**。\n"
            "- confidence：对「年级册+章」判定的整体把握 0~1（不是解题把握）。**没十足把握就给低（<0.8），别硬撑高分。**\n\n"
        )
        stage_lock_who = "你判定"
    tmpl = f"""{head}
{grade_block}================ 学段锁死红线 ================
🔴 解法不得超出{stage_lock_who}年级的进度（不许用更高年级才学的定理/方法绕过）；考点落在该年级/章范围内。

================ 10 维逐维规则（开集打标，kp 写真实考点名） ================
1. primaryKp 主考点：写**真实考点名**（id 留空串，由系统后锚到题库叶子）：{{"id":"","name":"一元二次方程的根的判别式"}}。
2. secondaryKps 副考点 0~3：与主不同体系才算，同样 {{"id":"","name":"..."}}；没有就空数组。
3. qtype 题型：选择/填空/解答 之一（闭集）。
4. assessmentType 考察类型：**闭集10选1** = {exam_types}。
5. solutionSkeleton 解法骨架：解题步骤序列；**最难的那一步用【】整步包住**（至多一处）。如实反映你解题真实路径，是变式守恒基因。
6. hardPointCount + breakthroughPoints 难点（克制·宁空不凑）：基础/纯套公式/直接计算/概念辨析/送分题 → breakthroughPoints **必空**、hardPointCount=0；🔴 hardPointCount **必须等于** breakthroughPoints 数组长度。
7. scenario 场景：一句话场景 或 "纯代数"。
8. difficulty 难度四档（按构造断言）：1★送分(无难点+概念辨析/单步)；2★★常规(无难点+{{直接计算·公式套用·性质判定}}+多步)；3★★★(1难点 或 {{证明推理·应用建模·探究归纳}} 或 骨架含【最难步】)；4★压轴(≥2难点 或 多突破口综合)。
9. tags 标签 3~6：检索标签（求什么/用什么定理/什么方法/什么场景）；禁近义增生。
10. modelCandidates 解题模型（克制）：真有可复用套路才给候选名（简单题空数组），只给名不给 M-id。

================ 富文本红线（题面/答案/解析） ================
🔴 数学式用行内 $...$；换行用真实换行（直接回车，不要写字面 \\n）；禁裸 LaTeX 命令、禁 \\( \\) / \\[ \\] 定界；LaTeX 括号/命令参数配对完整（下游有机器闸逐项检）。
🔴 **analysis/解析 = 给学生看的最终干净解法,不是草稿**：数学式/符号优先用 $LaTeX$ 表达、推导紧凑；**只写正确的最终推导链,禁止写试错/回头/『等等重新算』/反复估算/自我纠正等思考过程**（那些在你心里算，别外吐）。低冗余、省 token、一遍到底。
🔴 has_figure：题面真含图形/图表/几何图填 true；只是拍照的纯文本题填 false。
{{rich_output_block}}"""
    block = (_RICH_BLOCK_SENTINEL if sentinel else _RICH_BLOCK_JSON).replace(
        "{{", "{"
    ).replace("}}", "}")
    return tmpl.replace("{rich_output_block}", block)


# richText + 输出格式段（两版：免转义哨兵框 / 旧式整 JSON），按 settings.MOTHER_RICHTEXT_SENTINEL 选。
_RICH_BLOCK_JSON = """
================ 输出（只输出一个 JSON，不要解释、不要 markdown fence） ================
{{
  "gradeBook": "六册之一 或 空串",
  "chapter": "章名 或 空串",
  "gradeCandidates": [],
  "chapterCandidates": [],
  "confidence": 0.0,
  "has_figure": true/false,
  "richText": {{"stem": "题干(Markdown+行内$LaTeX$)", "answer": "标准答案", "analysis": "解析(含解题过程)"}},
  "solvedAnswer": "你一步步解出的最终答案",
  "dna": {{
    "primaryKp": {{"id": "", "name": "真实考点名"}},
    "secondaryKps": [],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2(最难一步用【】整步包住)"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["3~6个检索标签"],
    "modelCandidates": []
  }}
}}"""

_RICH_BLOCK_SENTINEL = """
================ 🔴 免转义哨兵框（题面/答案/解析三段——务必照此格式，防 JSON 转义坏整对象） ================
题面/答案/解析含 $LaTeX$、换行、<table> 等天然带 JSON 元字符（反斜杠/引号），塞进 JSON 字符串极易因一处转义坏 → 整个对象解析失败。**所以这三段不要塞进 JSON 字符串**，改用下面的哨兵框**原文**输出（框内随便写 LaTeX/换行/HTML，不用转义）：
⟦STEM⟧
（题干原文：Markdown + 行内 $LaTeX$ + 必要时 <table>，直接换行，无需任何转义）
⟦/STEM⟧
⟦ANSWER⟧
（标准答案原文）
⟦/ANSWER⟧
⟦ANALYSIS⟧
（解析原文：最终干净解法，不写草稿/试错）
⟦/ANALYSIS⟧
🔴 三段各自独立成框，缺一不可；框标签 ⟦STEM⟧/⟦/STEM⟧ 照抄、不要改写。

================ 输出格式（先一个 JSON，紧跟三个哨兵框；不要 markdown fence、不要前言） ================
先输出结构化 JSON（richText 三段在 JSON 里留空串占位，真内容走下方哨兵框）：
{{
  "gradeBook": "六册之一 或 空串",
  "chapter": "章名 或 空串",
  "gradeCandidates": [],
  "chapterCandidates": [],
  "confidence": 0.0,
  "has_figure": true/false,
  "richText": {{"stem": "", "answer": "", "analysis": ""}},
  "solvedAnswer": "你一步步解出的最终答案(简短，一行)",
  "dna": {{
    "primaryKp": {{"id": "", "name": "真实考点名"}},
    "secondaryKps": [],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2(最难一步用【】整步包住)"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["3~6个检索标签"],
    "modelCandidates": []
  }}
}}
紧接着 JSON **之后**输出三个哨兵框（⟦STEM⟧…⟦/STEM⟧、⟦ANSWER⟧…⟦/ANSWER⟧、⟦ANALYSIS⟧…⟦/ANALYSIS⟧），框内放三段富文本原文。"""


# 两版前缀模块加载期各算一次（_build_entry_system_prefix 内部已注入对应 rich_output_block）。
# 🔴 _build_entry_system_prefix() 无参（=sentinel=False）返回的就是 JSON 版完整前缀（与本量字节级
#   相同，test_prefix_idempotent 据此校验）。
ENTRY_SYSTEM_PREFIX_JSON: str = _build_entry_system_prefix(sentinel=False)
ENTRY_SYSTEM_PREFIX_SENTINEL: str = _build_entry_system_prefix(sentinel=True)
# 🔴 preset 版（老师已定年级 → 砍①判年级，专注读图解题）：四版各模块加载期算一次缓存。
ENTRY_SYSTEM_PREFIX_JSON_PRESET: str = _build_entry_system_prefix(sentinel=False, preset=True)
ENTRY_SYSTEM_PREFIX_SENTINEL_PRESET: str = _build_entry_system_prefix(sentinel=True, preset=True)


def _entry_prefix(preset: bool = False) -> str:
    """运行期按 settings + preset 选前缀（四版均模块加载期算好，字节稳定 = 缓存友好）。
    preset=True（老师已定年级）→ 砍①判年级的独立前缀。"""
    from core import settings as _settings  # 注意：core 重导出的是 Settings 实例（非模块）
    _sent = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False)
    if preset:
        return ENTRY_SYSTEM_PREFIX_SENTINEL_PRESET if _sent else ENTRY_SYSTEM_PREFIX_JSON_PRESET
    return ENTRY_SYSTEM_PREFIX_SENTINEL if _sent else ENTRY_SYSTEM_PREFIX_JSON


# 🔴 兼容旧引用（单测 / build_entry_prompt 壳）：默认指向 JSON 版（旧行为字节级一致）。
ENTRY_SYSTEM_PREFIX: str = ENTRY_SYSTEM_PREFIX_JSON


# ===========================================================================
# 🔴 PRD-A-021 R5（用户拍板 2026-06-22「拆两轮：①解题 ②富文本打标」）：把母题单调用拆成
#   R1 解题轮（纯解题·流式·宁慢求准）+ R2 结构化打标轮（接 R1 权威解答·忠实富文本+10维打标+切图）。
#   动机：单调用让模型边解题边打标边判年级，注意力过载 → 解题崩坏（已验算出 1 却硬填幻觉 12）、
#   单轮 117s 黑屏干等。拆开后 R1 全部注意力放解题（public_stream 流式吐到对话气泡=看得见思路），
#   R2 只忠实整理+打标（不重解、不改答案），解题对错与结构化解耦。
# ===========================================================================
# R1 解题轮系统前缀（字节稳定·三段式：角色 + 通用解题纪律 + 输出规范；不掺任何单一题型特化）。
_SOLVE_SYSTEM_PREFIX: str = (
    "你是浙教版初中数学解题专家。把这道题准确解出来（含图先看清图）。\n\n"
    "【解题纪律】\n"
    "- 读图完整：看清目标量由哪些部分组成、别漏（可能不止一块）。\n"
    "- 关键步代入原题验算；验算通过即为终答，不因「感觉不对」反复改答案。\n"
    "- 解法不超出给定年级学段。\n\n"
    "【输出】\n"
    "先写干净解题过程（数学式用 $...$），最后一行：\n"
    "【最终答案】<最简短答案，一行>"
)


def build_solve_messages(
    *, image_url: str, utterance: str | None = None, teacher_memory: str | None = None,
    preset_grade_book: str | None = None, preset_chapter: str | None = None,
    model_toolbox: str | None = None,
) -> list[Any]:
    """R1 解题轮消息：稳定 system 前缀（解题纪律）‖ 变量后缀（题图 + 年级/章学段约束 + 背景语境 + 记忆）。

    🔴 PRD-C-106 B1①·带料解题：model_toolbox（year-level 解题模型工具箱 clause，
       model_anchor.build_toolbox_clause 产出）非空 → 注入变量后缀当「可用解题大招工具箱」，让 opus
       带着工具箱解题（认得出该用哪个套路）。空/None → 不注入（裸解降级，不卡死）。
    """
    from langchain_core.messages import SystemMessage

    parts: list[dict[str, Any]] = []
    var_text_segs: list[str] = []
    if model_toolbox and model_toolbox.strip():
        var_text_segs.append(
            f"【🔴 可用解题大招工具箱（带料解题·解题时认得出该用哪个套路就用）】\n{model_toolbox}"
        )
    if preset_grade_book or preset_chapter:
        # 🔴 老师已定年级册 + 章 → 都注入学段约束，解题方法收窄到该章进度（之前漏注章）。
        _scope = "、".join(
            x for x in (
                f"年级册 = 「{preset_grade_book}」" if preset_grade_book else "",
                f"章 = 「{preset_chapter}」" if preset_chapter else "",
            ) if x
        )
        var_text_segs.append(
            f"【🔴 老师已确定的范围】本题{_scope}。**解题方法必须落在该章/该册进度内**，"
            "不许用更高年级或本章之后才学的定理/方法绕过——按老师定的章的解法来解。"
        )
    if utterance:
        var_text_segs.append(f"【老师附带的背景语境（参考，不影响解题本身）】{utterance}")
    if teacher_memory:
        var_text_segs.append(f"【该老师的偏好/纠正记忆（参考，不强制）】\n{teacher_memory}")
    if var_text_segs:
        parts.append({"type": "text", "text": "\n".join(var_text_segs)})
    parts.append({"type": "image_url", "image_url": {"url": image_url}})
    return [SystemMessage(content=_SOLVE_SYSTEM_PREFIX), HumanMessage(content=parts)]


# ===========================================================================
# 🔴 R6·富文本化轮（独立异步任务·与 R1 解题并发·走 sui-xiang 默认网关）：
#   用户 2026-06-22 拍板的三步编排第①步。职责 = **只忠于原文誊抄母题题面**为富文本
#   （LaTeX/必要 HTML），让题目能被系统索引/识别 + 给一个干净题面。
#   🔴 极简无噪音（仓库「忠于原文·禁单题型噪音」原则）：禁解题、禁打标、禁补漏、禁判章、
#      禁输出题面之外任何东西。relay=None（默认 sui-xiang，不走 MOTHER_SOLVE_RELAY）。
#   产物作为最终 entry.stem 的权威来源（非空时覆盖 R2 誊抄的 stem）。
# ===========================================================================
_RICHTEXT_SYSTEM_PREFIX: str = (
    "你是题面誊抄员。把图中这道题的**题面**一字不差地誊抄成富文本，**只做誊抄这一件事**。\n\n"
    "【铁律】\n"
    "- 忠于原文：照图中原题逐字誊抄题面，不要解题、不要补条件、不要改写、不要省略。\n"
    "- 数学式用行内 $...$；换行用真实换行（直接回车，不要写字面 \\n）。\n"
    "- 禁裸 LaTeX 命令、禁 \\( \\) / \\[ \\] 定界；LaTeX 括号/命令参数配对完整。\n"
    "- 题面含表格 → 用 <table> 还原，不要拍平成文字。\n\n"
    "【输出】\n"
    "只输出题面富文本本身，不写任何前言/解题/答案/分类/标签/说明。"
)


def build_richtext_messages(
    *, image_url: str, utterance: str | None = None, teacher_memory: str | None = None,
) -> list[Any]:
    """R6 富文本化轮消息：极简稳定 system 前缀（只誊抄题面）‖ 变量后缀（题图）。

    🔴 极简无噪音：不注入解题纪律 / 年级章约束 / 打标规则；utterance/记忆仅作弱背景（多数轮为空）。
    """
    from langchain_core.messages import SystemMessage

    parts: list[dict[str, Any]] = []
    # 富文本化只誊抄题面，utterance/记忆与题面誊抄无关 → 默认不注入（保持 prompt 极简无噪音）。
    parts.append({"type": "image_url", "image_url": {"url": image_url}})
    return [SystemMessage(content=_RICHTEXT_SYSTEM_PREFIX), HumanMessage(content=parts)]


_FINAL_ANSWER_RE = re.compile(r"【最终答案】\s*(.+?)\s*$", re.MULTILINE)


def _extract_solved_answer(text: str) -> str:
    """从 R1 解题正文抠出【最终答案】标记内容（取最后一处）；无标记 → 空串（R2 退兜底，不崩）。"""
    if not text:
        return ""
    hits = _FINAL_ANSWER_RE.findall(text)
    return hits[-1].strip() if hits else ""


def _strip_final_answer_marker(text: str) -> str:
    """喂 R2 的 R1 解答正文：去掉【最终答案】标记行（最终答案另以结构化字段单独给 R2）。"""
    if not text:
        return ""
    return _FINAL_ANSWER_RE.sub("", text).strip()


def build_entry_messages(
    *, image_url: str, utterance: str | None = None, teacher_memory: str | None = None,
    preset_grade_book: str | None = None,
) -> list[Any]:
    """B2 缓存接缝：稳定 system 前缀 ‖ 变量 user 后缀（题图 + query + teacher 记忆**后置**）。

    🔴 teacher_memory 必须在后缀（per-teacher，进前缀毁多用户共享缓存）；B4 记忆层注入填这里。
    🔴 前缀 = ENTRY_SYSTEM_PREFIX（字节稳定）；后缀变量随请求变。aigeek 自动缓存吃前缀。
    """
    from langchain_core.messages import SystemMessage  # 局部 import（顶层已 import HumanMessage）

    parts: list[dict[str, Any]] = []
    var_text_segs: list[str] = []
    _preset = bool(preset_grade_book)
    if _preset:
        # 🔴 preset（用户拍板·两 prompt 隔离）：年级册由老师确定，注入人类消息后缀（per-request，不进缓存
        #   前缀）；系统前缀已用 preset 版（砍①判年级）。让模型直接采用、专注读图解题。
        var_text_segs.append(
            f"【🔴 老师已确定】本题年级册 = 「{preset_grade_book}」。gradeBook 直接填它、confidence=1.0、"
            "gradeCandidates/chapterCandidates 留空，**不要再自行判定年级册**；专注严谨读图解题。"
        )
    if utterance:
        # 🔴 R2b·U8：哨兵模式输出 = JSON + 三哨兵框（不是「只一个 JSON」），措辞按模式区分，
        #   否则与 system 段的哨兵框指令打架。两版都强调「不据此出变式/不输出数组/不写前言」。
        from core import settings as _settings  # core 重导出的是 Settings 实例（非模块）
        _sentinel = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False)
        _fmt_hint = (
            "按系统约定的「JSON + 三哨兵框」格式输出"
            if _sentinel else "**只输出一个 JSON 对象**"
        )
        var_text_segs.append(
            "【老师附带要求（仅作背景语境参考，不抽配方）：本步只给"
            f"母题本身打标，{_fmt_hint}，不要据此出变式、不要输出数组或多个对象、不要写前言】"
            f"{utterance}"
        )
    if teacher_memory:  # B4 记忆注入（后置，不进缓存前缀）
        var_text_segs.append(f"【该老师的偏好/纠正记忆（参考，不强制）】\n{teacher_memory}")
    if var_text_segs:
        parts.append({"type": "text", "text": "\n".join(var_text_segs)})
    parts.append({"type": "image_url", "image_url": {"url": image_url}})
    return [SystemMessage(content=_entry_prefix(preset=_preset)), HumanMessage(content=parts)]


# ===========================================================================
# 🔴 R5·R2 结构化打标轮（解题已由 R1 完成，权威解答经人类消息注入）：忠实整理富文本 + 10维打标。
#   与 R1 隔离=R2 系统前缀**不含解题纪律**（不再诱导重解），只讲「忠实整理 + 打标 + 判章」。
#   richText.analysis = 把 R1 已给的正确解法理顺成给学生看的干净链路（忠实，不重推、不改答案）。
# ===========================================================================
def _build_struct_label_prefix(*, sentinel: bool = False, preset: bool = False) -> str:
    """R2 结构化打标系统前缀（字节稳定，四版 sentinel×preset 各模块加载期算一次）。
    preset=True → 砍①判年级（老师已定）；解题始终已给（R1 权威解答），故无解题纪律段。"""
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    books = "、".join(_GRADE_BOOKS)
    if preset:
        head = (
            "你是浙教版初中数学题库打标师。年级册已由老师确定、解题已由解题专家完成（下方权威解答），不必重解。"
            "gradeBook 照老师给的填、confidence=1.0、gradeCandidates/chapterCandidates 留空（章可如实填）。做两件事：\n"
            "② 富文本三段：stem(题面) 照图中原题誊抄；answer/analysis 据已给解答整理；solvedAnswer = 已给最终答案。\n"
            "③ 据已解出的答案做 10 维 DNA 打标（母题是所有变式的基准）。"
        )
        grade_block = ""
    else:
        head = (
            "你是浙教版初中数学题库打标师。解题已由解题专家完成（下方权威解答），不必重解。按顺序做三件事：\n"
            "① 判年级册 + 章 + 置信度（属哪册哪章，给 0~1 置信，拿不准给候选）；\n"
            "② 富文本三段：stem(题面) 照图中原题誊抄；answer/analysis 据已给解答整理；solvedAnswer = 已给最终答案。\n"
            "③ 据已解出的答案做 10 维 DNA 打标（母题是所有变式的基准）。"
        )
        grade_block = (
            "================ 判年级册 + 章（闭集 + 候选） ================\n"
            f"- gradeBook 只从这 6 册选一个：{books}。判不出留空串、gradeCandidates 给最可能的 1~2 个。\n"
            "- chapter：章名（如「第2章 一元二次方程」）；判不出留空。章有歧义 → chapterCandidates 给 ≥2 个。\n"
            "- confidence：对年级册+章判定的把握 0~1，没把握就给低。\n\n"
        )
    tmpl = f"""{head}
{grade_block}================ 10 维逐维规则（开集打标，kp 写真实考点名） ================
1. primaryKp 主考点：写**真实考点名**（id 留空串，由系统后锚到题库叶子）：{{"id":"","name":"一元二次方程的根的判别式"}}。
2. secondaryKps 副考点 0~3：与主不同体系才算，同样 {{"id":"","name":"..."}}；没有就空数组。
3. qtype 题型：选择/填空/解答 之一（闭集）。
4. assessmentType 考察类型：**闭集10选1** = {exam_types}。
5. solutionSkeleton 解法骨架：**据已给权威解答**抽解题步骤序列；**最难的那一步用【】整步包住**（至多一处）。是变式守恒基因。
6. hardPointCount + breakthroughPoints 难点（克制·宁空不凑）：基础/纯套公式/直接计算/概念辨析/送分题 → breakthroughPoints **必空**、hardPointCount=0；🔴 hardPointCount **必须等于** breakthroughPoints 数组长度。
7. scenario 场景：一句话场景 或 "纯代数"。
8. difficulty 难度四档（按构造断言）：1★送分(无难点+概念辨析/单步)；2★★常规(无难点+{{直接计算·公式套用·性质判定}}+多步)；3★★★(1难点 或 {{证明推理·应用建模·探究归纳}} 或 骨架含【最难步】)；4★压轴(≥2难点 或 多突破口综合)。
9. tags 标签 3~6：检索标签（求什么/用什么定理/什么方法/什么场景）；禁近义增生。
10. modelCandidates 解题模型（克制）：真有可复用套路才给候选名（简单题空数组），只给名不给 M-id。

================ 富文本红线（题面/答案/解析） ================
🔴 数学式用行内 $...$；换行用真实换行（直接回车，不要写字面 \\n）；禁裸 LaTeX 命令、禁 \\( \\) / \\[ \\] 定界；LaTeX 括号/命令参数配对完整（下游有机器闸逐项检）。
🔴 analysis 只写干净的最终推导链，禁止写试错/回头/反复估算等草稿过程。
🔴 has_figure：题面真含图形/图表/几何图填 true；只是拍照的纯文本题填 false。
{{rich_output_block}}"""
    block = (_RICH_BLOCK_SENTINEL if sentinel else _RICH_BLOCK_JSON).replace(
        "{{", "{"
    ).replace("}}", "}")
    return tmpl.replace("{rich_output_block}", block)


STRUCT_SYSTEM_PREFIX_JSON: str = _build_struct_label_prefix(sentinel=False)
STRUCT_SYSTEM_PREFIX_SENTINEL: str = _build_struct_label_prefix(sentinel=True)
STRUCT_SYSTEM_PREFIX_JSON_PRESET: str = _build_struct_label_prefix(sentinel=False, preset=True)
STRUCT_SYSTEM_PREFIX_SENTINEL_PRESET: str = _build_struct_label_prefix(sentinel=True, preset=True)


def _struct_prefix(preset: bool = False) -> str:
    """运行期按 settings + preset 选 R2 结构化打标前缀。"""
    from core import settings as _settings
    _sent = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False)
    if preset:
        return STRUCT_SYSTEM_PREFIX_SENTINEL_PRESET if _sent else STRUCT_SYSTEM_PREFIX_JSON_PRESET
    return STRUCT_SYSTEM_PREFIX_SENTINEL if _sent else STRUCT_SYSTEM_PREFIX_JSON


def build_struct_messages(
    *, image_url: str, solved_solution: str, solved_answer: str,
    utterance: str | None = None, teacher_memory: str | None = None,
    preset_grade_book: str | None = None, preset_chapter: str | None = None,
    model_toolbox: str | None = None,
) -> list[Any]:
    """R2 结构化打标消息：稳定 system 前缀 ‖ 变量后缀（题图 + 🔴R1 权威解答 + 年级/章 + 背景 + 记忆）。

    solved_solution = R1 解题正文（已去【最终答案】标记）；solved_answer = R1 抠出的最终答案。
    🔴 题面(stem) 忠实誊抄**图中原题**（不是从解答倒推）；答案/解析据 R1 权威解答整理。
    🔴 PRD-C-106 B1①·带料：model_toolbox 非空 → 注入工具箱，让 opus 在 modelCandidates 里**照工具箱
       名原样填**它真正用到的模型（下游 anchor_models_from_names 纯代码映射 M-id，消第二次 LLM 解题）。
    """
    from langchain_core.messages import SystemMessage

    _preset = bool(preset_grade_book)
    parts: list[dict[str, Any]] = []
    var_text_segs: list[str] = []
    if model_toolbox and model_toolbox.strip():
        var_text_segs.append(
            "【🔴 可用解题大招工具箱（带料打标·modelCandidates 照下面名称原样填你解题真正用到的，"
            f"没用到留空数组、绝不硬凑）】\n{model_toolbox}"
        )
    # 🔴 忠于原文头条铁律（用户反馈「忠于原文没做到·原文没配对」）：题面来源 = 图中原题，不是解答倒推。
    # 🔴 R1 权威解答（核心注入）：stem 照图誊抄 / answer·analysis 照此解答整理（忠于原文规则在 system 已给，此处只提一句不重复）。
    _ans_line = f"最终答案 = {solved_answer}\n\n" if solved_answer else ""
    var_text_segs.append(
        "【🔴 权威解答（题面照图誊抄；answer/analysis 照此整理，绝不重解、绝不改答案）】"
        f"\n{_ans_line}解题过程：\n{solved_solution}"
    )
    if _preset or preset_chapter:
        _scope = "、".join(
            x for x in (
                f"年级册 = 「{preset_grade_book}」" if preset_grade_book else "",
                f"章 = 「{preset_chapter}」" if preset_chapter else "",
            ) if x
        )
        var_text_segs.append(
            f"【🔴 老师已确定的范围】本题{_scope}。gradeBook 直接填年级册名、confidence=1.0、"
            "gradeCandidates/chapterCandidates 留空，不要再自行判定年级册；打标考点落在该章范围内。"
        )
    if utterance:
        from core import settings as _settings
        _sentinel = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False)
        _fmt_hint = (
            "按系统约定的「JSON + 三哨兵框」格式输出"
            if _sentinel else "**只输出一个 JSON 对象**"
        )
        var_text_segs.append(
            "【老师附带要求（仅作背景语境参考，不抽配方）：本步只给"
            f"母题本身打标，{_fmt_hint}，不要据此出变式、不要输出数组或多个对象、不要写前言】"
            f"{utterance}"
        )
    if teacher_memory:
        var_text_segs.append(f"【该老师的偏好/纠正记忆（参考，不强制）】\n{teacher_memory}")
    parts.append({"type": "text", "text": "\n\n".join(var_text_segs)})
    parts.append({"type": "image_url", "image_url": {"url": image_url}})
    return [SystemMessage(content=_struct_prefix(preset=_preset)), HumanMessage(content=parts)]


# ===========================================================================
# 🔴 PRD-C-107 B1·阶段一「一条连续对话」构件（全 sui-xiang，图发一次，messages 累积）
#   定稿 §「形态」：阶段一 = 一条连续对话（消息历史累积），不再是 3 次独立调用。
#     SYSTEM 发一次 → turn1 誊抄+初判+解析老师信息 → (确认闸) → turn2 双料闭集注入+解题
#     → turn3 打标。母题图只在 turn1 发一次，后续轮靠对话历史。
#   🔴 与旧 R6（build_richtext/solve/struct 三独立调用）并存：旧函数留作回退/单测；mother_opus_entry
#      在 settings.C107_CONTINUOUS（默认开）时走本连续对话，否则走旧 R6（决策表 fallback 路径）。
#   🔴 闭集注入（改 C-106 开集后锚）：叶子池进对话、LLM 选真 id；代码后锚 _match_kp_in_pool 仍兜底。
#   🔴 难度表驱动：模型工具箱（biz_solution_model）注入；命中模型的 tier/freq 由 anchor_models_from_names
#      映回（绝不 LLM 自评），与 C-106 链路一致、字节不动。
# ===========================================================================
# 系统提示（发一次·字节稳定·定稿 §「系统提示」原话）：
_STAGE1_SYSTEM_PREFIX: str = (
    "你是浙教版初中数学老师，针对老师贴来的这道母题，依次做：判年级章 →（必要时确认）→ 誊抄 → 解题 → 打标。"
    "我会分几轮逐步引导你，请按每轮的指示只做那一轮的事、并带上前面轮次的结论。\n\n"
    "【铁律】\n"
    "- 拿不准年级/章 → 给候选并用一句人话说明为什么拿不准（reason），别硬撑高置信。\n"
    "- 考点只从我给你的「知识点叶子池」里选真 id，绝不自造 id；池里没有的考点只写名、id 留空（系统会后锚）。\n"
    "- 解法不超出该年级学段进度；关键步代入原题验算；答案据你的解题，验算通过即为终答，不反复改。\n"
    "- 解题大招从我给的「工具箱」里按名认领（modelCandidates 照工具箱名原样填你真正用到的，没用到留空、绝不硬凑）。"
)


def _build_leaf_pool_clause(leaf_pool: list[tuple[str, str]] | None, *, limit: int = 120) -> str:
    """🔴 B1·知识点叶子池闭集注入：[(id,name)...] → 可注入对话的「考点候选(真 id)」段（纯函数·可单测）。

    闭集精神 = LLM 只能从池内选真 id（改 C-106「开集自由命名、id 留空、代码后锚」）；代码后锚
    （_match_kp_in_pool）仍当兜底（LLM 选歪/没选 → 代码纠）。空池 → 返回 ""（上层降级裸标，不卡死）。
    limit 防超长池吞 token（按 BE 给序取前 N；连续对话只发一次，cache 高，成本可控）。
    """
    pool = [(str(i).strip(), str(n).strip()) for i, n in (leaf_pool or []) if str(i).strip() and str(n).strip()]
    if not pool:
        return ""
    pool = pool[:limit]
    lines = [
        "🔴 知识点叶子池（本年级/章的真实考点候选 + 真 id）——主考点/副考点**只能从这里选**，"
        "primaryKp.id / secondaryKps[].id 照填池内真 id，名也照池内名；池里实在没有的考点才只写名、id 留空："
    ]
    lines.extend(f"- [{i}] {n}" for i, n in pool)
    return "\n".join(lines)


def build_stage1_turn1_messages(
    *, image_url: str, utterance: str | None = None, teacher_memory: str | None = None,
    preset_grade_book: str | None = None, preset_chapter: str | None = None,
) -> list[Any]:
    """B1·turn1（誊抄 + 初判年级章 + 解析老师附带信息）= 连续对话首轮（唯一带图的轮）。

    产出（让 opus 一次给齐，下游解析）：① 富文本题面 ② 年级初判(置信+候选) ③ reason(拿不准的人话)。
    🔴 图只在本轮发；后续 solve/label 轮纯文本、靠对话历史。SYSTEM = _STAGE1_SYSTEM_PREFIX（发一次）。
    """
    from langchain_core.messages import SystemMessage

    var_text_segs: list[str] = []
    if preset_grade_book or preset_chapter:
        _scope = "、".join(
            x for x in (
                f"年级册 = 「{preset_grade_book}」" if preset_grade_book else "",
                f"章 = 「{preset_chapter}」" if preset_chapter else "",
            ) if x
        )
        var_text_segs.append(
            f"【🔴 老师已确定的范围】本题{_scope}。年级/章直接采用、confidence=1.0、候选留空，不必再判。"
        )
    if utterance:
        var_text_segs.append(f"【老师附带的话（请从中接住年级/章/指定解法等信息，只接明说的、不脑补）】{utterance}")
    if teacher_memory:
        var_text_segs.append(f"【该老师的偏好/纠正记忆（参考，不强制）】\n{teacher_memory}")
    var_text_segs.append(
        "【本轮（turn1）只做三件事，先别解题、先别打标】\n"
        "1. 把题面一字不差誊抄成富文本（行内 $...$，真实换行，禁裸 LaTeX 命令/定界符）。\n"
        "2. 判这道题的年级册 + 章 + 置信度 0~1，拿不准给 gradeCandidates / chapterCandidates。\n"
        "3. 若置信不高，用一句人话写明为什么拿不准（reason）。\n"
        "只输出一个 JSON（不要 markdown fence）："
        '{"stem":"题面富文本","gradeBook":"六册之一或空串","chapter":"章名或空串",'
        '"gradeCandidates":[],"chapterCandidates":[],"confidence":0.0,'
        '"reason":"拿不准就写人话原因，高置信留空","has_figure":true/false}'
    )
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "\n\n".join(var_text_segs)},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    return [SystemMessage(content=_STAGE1_SYSTEM_PREFIX), HumanMessage(content=parts)]


def build_stage1_solve_turn(
    *, leaf_pool: list[tuple[str, str]] | None, model_toolbox: str | None = None,
    anchored_grade: str | None = None, anchored_chapter: str | None = None,
    teacher_model: str | None = None,
) -> list[Any]:
    """B1·turn2（双料闭集注入 + 解题）= 连续对话第二轮（纯文本·无图，靠历史看题面）。

    🔴 双料 = 知识点叶子池（_build_leaf_pool_clause·闭集真 id）+ 解题大招工具箱（model_toolbox）。
    🔴 接住老师指定模型（teacher_model）→ 优先按此解法解。返回单条 HumanMessage（list 形态便于累积拼接）。
    """
    segs: list[str] = []
    if anchored_grade or anchored_chapter:
        _scope = "、".join(
            x for x in (
                f"年级·{anchored_grade}" if anchored_grade else "",
                f"章·{anchored_chapter}" if anchored_chapter else "",
            ) if x
        )
        segs.append(f"〔锚定范围：{_scope}〕解法必须落在该范围进度内。")
    leaf_clause = _build_leaf_pool_clause(leaf_pool)
    if leaf_clause:
        segs.append(leaf_clause)
    if model_toolbox and model_toolbox.strip():
        segs.append(model_toolbox)
    if teacher_model:
        segs.append(f"🔴 老师指定优先用解法/模型：「{teacher_model}」——若适用就按它解；不适用再换。")
    segs.append(
        "【本轮（turn2）只做一件事：解题】对照上面誊抄的题面（和工具箱里的大招），一步步把题准确解出来。"
        "关键步代入原题验算。先写干净解题过程（数学式 $...$），最后一行：\n【最终答案】<最简短答案，一行>"
    )
    return [HumanMessage(content="\n\n".join(segs))]


def build_stage1_label_turn(
    *, leaf_pool: list[tuple[str, str]] | None, model_toolbox: str | None = None,
    sentinel: bool | None = None,
) -> list[Any]:
    """B1·turn3（打标）= 连续对话第三轮（纯文本·无图，靠历史看题面+解题）。

    🔴 闭集：考点从叶子池选真 id（_build_leaf_pool_clause 已在 turn2 注入·历史可见，这里再点一句强化）；
       模型从工具箱填名（model_toolbox 历史可见）。输出沿用 R2 的 JSON / 哨兵框格式（settings 决定）。
    """
    from core import settings as _settings
    _sent = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False) if sentinel is None else sentinel
    leaf_clause = _build_leaf_pool_clause(leaf_pool)
    segs: list[str] = []
    if leaf_clause:
        segs.append(leaf_clause)
    if model_toolbox and model_toolbox.strip():
        segs.append(model_toolbox)
    fmt_hint = (
        "按系统约定的「JSON + 三哨兵框（⟦STEM⟧/⟦ANSWER⟧/⟦ANALYSIS⟧）」格式输出"
        if _sent else "只输出一个 JSON 对象（不要 markdown fence、不要前言）"
    )
    segs.append(
        "【本轮（turn3）只做一件事：打标】据你上面解出的答案，给这道母题做 10 维 DNA 打标。\n"
        "🔴 primaryKp / secondaryKps 的 id **从知识点叶子池选真 id**（池里没有才只写名、id 留空）；\n"
        "🔴 modelCandidates **照工具箱名原样填**你解题真正用到的模型（没用到留空数组、绝不硬凑）；\n"
        "🔴 stem 照题面誊抄、answer/analysis 据已解出的过程整理、solvedAnswer = 你的最终答案；\n"
        "🔴 难度/难点据构造如实断言（系统会按命中模型的表 tier 复核，你只如实标、不自评难度旋钮）。\n"
        f"{fmt_hint}，10 维字段（primaryKp/secondaryKps/qtype/assessmentType/solutionSkeleton/"
        "hardPointCount/breakthroughPoints/scenario/difficulty/tags/modelCandidates）+ richText 三段 + "
        "gradeBook/chapter/confidence/has_figure/solvedAnswer 齐全。"
    )
    return [HumanMessage(content="\n\n".join(segs))]


# ---------------------------------------------------------------------------
# 🔴 PRD-C-107 B1·接住老师附带信息（解析 utterance 抽 grade/chapter/model）+ 确认带 reason
#   定稿 §「确认闸」「接住老师信息」：老师打字年级章 = 等同 preset（跳确认）；指定模型 = 注入优先用。
#   🔴 用确定性解析（正则·闭集匹配），不耗 LLM、不引漂移（铁律：代码后锚精神；解析失败 → 退默认，不卡死）。
# ---------------------------------------------------------------------------
# 年级关键词 → 标准年级册名（六册闭集），覆盖「八下/8下/八年级下/八年级下册」等口语。
_GRADE_UTTER_RE = re.compile(
    r"(七|八|九|7|8|9)\s*(年级)?\s*(上|下)\s*(学期|册)?"
)
_GRADE_CN_MAP = {"7": "七", "8": "八", "9": "九"}


def _extract_teacher_intent(utterance: str | None) -> dict[str, Any]:
    """🔴 B1·接住老师 utterance：抽 {grade, chapter, model}（确定性解析·只接明说的·不脑补）。

    - grade：命中「七/八/九 + 上/下」→ 标准册名「X年级Y册」；没命中 → 空。
    - chapter：命中「第N章...」片段 → 原样截取；没命中 → 空。
    - model：命中「用/按/走 ...法/...模型/...定理」→ 截取该解法名；没命中 → 空。
    任何字段拿不准一律空（不脑补默认）；整体失败 → {}（上层退默认 spec，绝不卡死）。
    """
    out: dict[str, Any] = {}
    u = str(utterance or "").strip()
    if not u:
        return out
    try:
        m = _GRADE_UTTER_RE.search(u)
        if m:
            g = _GRADE_CN_MAP.get(m.group(1), m.group(1))
            term = m.group(3)  # 上/下
            out["grade"] = f"{g}年级{term}册"
        cm = re.search(r"第\s*[一二三四五六七八九十\d]+\s*章[^，。,；;\n]{0,20}", u)
        if cm:
            out["chapter"] = cm.group(0).strip()
        # 解法/模型：「用判别式法」「按配方法」「走数轴折叠」「韦达定理」等。
        #   🔴 先剥掉已命中的年级/章片段，防贪婪匹配把「八下用判别式法」整段当模型名（边缘坑实测）。
        u_for_model = u
        if m:
            u_for_model = u_for_model.replace(m.group(0), " ")
        if cm:
            u_for_model = u_for_model.replace(cm.group(0), " ")
        # 优先取触发词（用/按/走/套/以）之后的解法名（更干净）；无触发词再退裸名匹配。
        mm = re.search(r"(?:用|按|走|套|以)\s*([一-龥A-Za-z]{2,10}(?:法|模型|定理|公式))", u_for_model)
        if not mm:
            mm = re.search(r"([一-龥A-Za-z]{2,10}(?:法|模型|定理|公式))", u_for_model)
        if mm:
            name = mm.group(1).strip()
            # 过滤无意义命中（如「方法」「办法」本身不是模型名）
            if name not in ("方法", "办法", "解法", "做法", "用法"):
                out["model"] = name
    except Exception:  # noqa: BLE001 — 解析永不卡死，退已抽到的部分
        pass
    return out


def _teacher_intent_skips_confirm(intent: dict[str, Any] | None) -> bool:
    """老师打字给了年级（≈ 老师亲选范围）→ 等同 preset，跳确认闸（定稿 §确认闸·接住）。
    只认 grade（章可空——年级定了下游能圈年级池）。没给 → False（维持原确认路径）。"""
    return bool((intent or {}).get("grade"))


def _build_confirm_payload(decision: dict[str, Any]) -> dict[str, Any]:
    """🔴 B1·确认弹窗 payload（端出 reason）。在既有 needConfirm 契约上**增量加 reason**（FE 容忍未知键，
    B4 再渲染）。承接 decide_confirm 产出的人话 reason，让老师看到「为什么要确认」。"""
    return {
        "grade_book": {"id": "", "name": decision.get("grade_book") or ""},
        "chapter": {"id": "", "name": decision.get("chapter") or ""},
        "grade_candidates": [{"id": "", "name": n} for n in (decision.get("grade_candidates") or [])],
        "chapter_candidates": [{"id": "", "name": n} for n in (decision.get("chapter_candidates") or [])],
        "confidence": float(decision.get("confidence") or 0.0),
        "reason": str(decision.get("reason") or "").strip(),  # 🆕 B1：端出人话原因
    }


# 🔴 PRD-C-105 A（D3）：喂模型前下采样上限。Claude 服务端本就把图降到长边 ≤1568px / ≤~1.15MP，
#   超过这个尺寸的字节全是白送（转富文本/解题慢、撞 180s、"图没过去"）。长边 >1568 → 等比缩到 1568，
#   对模型无损（它内部也降到这）。≤1568 的原样不缩（不做任何重编码，零损耗）。
_MODEL_IMG_MAX_EDGE = 1568


def _downsample_image_bytes(raw: bytes, ctype: str) -> tuple[bytes, str]:
    """🔴 PRD-C-105 A：把喂模型那一份临时拷贝下采样到长边 ≤1568px（对模型无损·提速）。

    入参 = 下载到的原图字节 + content-type；返回 (字节, content-type)。
    - 长边 ≤1568px → **原样返回原字节**（不重编码，零损耗、零失真）。
    - 长边 >1568px → 等比缩到长边=1568：照片/JPEG 源 → JPEG q90；PNG/透明 → 保 PNG。
    - Pillow 任何异常 → 回退原字节（绝不因压缩失败让解题崩；失败打 log）。
    🔴 只压"喂 LLM 前的临时拷贝"——OSS 原图 / 入库母题图 / FE 展示全不碰（本函数只在喂模型前调）。
    """
    try:
        import io as _io

        from PIL import Image as _Image

        im = _Image.open(_io.BytesIO(raw))
        im.load()
        w, h = im.size
        if max(w, h) <= _MODEL_IMG_MAX_EDGE:
            return raw, ctype  # 已够小：原样不缩，不重编码（零损耗）
        scale = _MODEL_IMG_MAX_EDGE / float(max(w, h))
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        im = im.resize(new_size, _Image.LANCZOS)
        # 透明/PNG 源 → 保 PNG（不丢 alpha）；其余（照片/JPEG）→ JPEG q90。
        src_png = (im.mode in ("RGBA", "LA", "P")) or ("png" in (ctype or "").lower())
        buf = _io.BytesIO()
        if src_png:
            im.save(buf, format="PNG", optimize=True)
            out_ctype = "image/png"
        else:
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=90)
            out_ctype = "image/jpeg"
        return buf.getvalue(), out_ctype
    except Exception as e:  # noqa: BLE001 — 压缩失败 → 回退原字节（绝不让下采样阻断解题）
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "C-105 A 图片下采样失败，回退原字节 base64：%s", e
        )
        return raw, ctype


async def _to_b64_data_url(url: str) -> str:
    """sui-xiang(kiro 逆向站)不抓远程图 URL → 母题图必须 base64 内嵌。下载 OSS 图 → data URL。
    已是 data: 直接返回；下载失败 → 原样返回 url（aigeek failover 仍可用远程 URL，不破熔断备用）。

    🔴 PRD-C-105 A（D3）：下载后、base64 前先经 _downsample_image_bytes 把长边压到 ≤1568px
    （只压这份喂模型的临时拷贝，OSS/展示/入库原图不碰）。≤1568 原样、>1568 等比缩、失败回退原字节。"""
    if not url or url.startswith("data:"):
        return url
    try:
        import base64 as _b64

        import httpx as _httpx
        async with _httpx.AsyncClient(timeout=30, trust_env=False) as c:
            resp = await c.get(url)
            resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "image/png").split(";")[0].strip() or "image/png"
        # C-105 A：喂模型前下采样（≤1568 原样不缩；只压这份临时拷贝）。
        content, ctype = _downsample_image_bytes(resp.content, ctype)
        return f"data:{ctype};base64,{_b64.b64encode(content).decode()}"
    except Exception:  # noqa: BLE001 — 下载失败 → 原 url（aigeek 兜底能用远程 URL）
        return url


def _repair_json_quotes(text: str) -> str:
    """确定性修未转义引号（opus 复杂几何题高发，conv_trace id=1399：字符串值内写了 ASCII 双引号
    如 关于"...的映射" 未转义 → json.loads 断）。字符级扫描：字符串内遇 " 看下一个非空白字符——
    是 :,}] 或结尾 → 闭合引号(留)；否则 = 内容引号 → 转义。对本就合法的 JSON 无副作用
    （内容引号已是 \\" escape pair，扫描跳过；结构引号后必跟 :,}]）。"""
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    body = (m.group(1) if m else text).strip()
    out: list[str] = []
    in_str = False
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            i += 1
            continue
        if ch == "\\":  # escape pair：原样保留两字符
            out.append(ch)
            if i + 1 < n:
                out.append(body[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and body[j] in " \t\r\n":
                j += 1
            nxt = body[j] if j < n else ""
            if nxt in ":,}]" or nxt == "":  # 闭合引号
                out.append(ch)
                in_str = False
            else:  # 内容引号 → 转义
                out.append('\\"')
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


async def _repair_entry_json(broken: str, V: Any = None) -> Any:
    """母题 opus 非法 JSON 两级修复：① 确定性修未转义引号（覆盖 id=1399 失败模式，无副作用、不花钱）；
    ② 仍失败 → 文本级 LLM 修复兜底（C-010 §② 失败带反馈 retry，不重读图）。返回解析对象（失败→None）。

    🔴 V = variant 模块句柄（懒导入注入，防循环 + 修原模块全局 `V` 未定义 NameError 隐患——
       原实现里 _repair_json_via_llm 引用模块级 `V` 但 variant_entry 从不在模块级 import variant，
       LLM 兜底层一直静默 NameError→None=死档；显式传 V 后 LLM 兜底真正生效）。
    """
    import json as _json
    try:
        return _json.loads(_repair_json_quotes(broken))
    except Exception:  # noqa: BLE001 — 确定性修不了 → LLM 兜底
        pass
    if V is None:
        from agents import variant as V  # 懒导入防循环（仅 LLM 兜底层需要）
    return await _repair_json_via_llm(broken, V)


async def _repair_json_via_llm(broken: str, V: Any) -> Any:
    """文本级 LLM 修复兜底：喂坏文本让模型只修语法、内容一字不改。返回解析后的对象（失败→None）。"""
    from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM

    sys_p = (
        "你是 JSON 修复器。下面文本本应是合法 JSON，但有语法错误（最常见：字符串值内出现"
        '未转义的 ASCII 双引号 " → 破坏 JSON 定界）。请输出**修正后的严格合法 JSON**：'
        "保持所有字段/内容/数学式 $LaTeX$ 一字不改，**只修语法**——字符串内的 ASCII 引号转义为 "
        '\\" 或改成中文「」。只输出 JSON 本身，不要解释、不要 markdown fence。'
    )
    try:
        fixed = await _ainvoke_text(
            [_SM(content=sys_p), _HM(content=broken)],
            model=None, max_tokens=8192, temperature=0.0,
        )
        return _parse_json(fixed)
    except Exception:  # noqa: BLE001 — 修复失败 → 上层走原 parse_fail
        return None


def _unwrap_obj(parsed: Any) -> Any:
    """opus 偶吐数组（id=1383/1384）→ 取首个 dict 元素；否则原样返回。"""
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    return parsed


# ---------------------------------------------------------------------------
# 🔴 PRD-A-021 R2b·U8·免转义哨兵框（治「richText 含 $LaTeX$+\n+<table> 天然带 JSON 元字符
#   → 一处转义坏 json.loads 整对象断 → +100s LLM 修复往返 → 仍不行 +几百s 重读图」惩罚链）
#
# 根因：stem/answer/analysis 三段富文本是破损重灾区（反斜杠/引号/换行/HTML 全往 JSON 字符串里塞）。
# 方案：让 opus 把这三段**移出 JSON 字符串**，用唯一哨兵分隔符切出来（原文，不转义）；其余结构化
#   字段（判章/dna/has_figure/solvedAnswer）仍正常 JSON（它们短、无元字符、转义稳）。
# 解析：先按哨兵正则抠出三段原文 → 从文本里删掉哨兵块 → 剩下的瘦 JSON（richText 三段为空占位）
#   干净 json.loads → 把抠出的三段塞回 parsed["richText"]。三段彼此独立：一段哨兵缺/坏不连累另两段。
# 哨兵选 ⟦⟧ 双角括号 + 全大写英文标签（U+27E6/U+27E7，数学/中文题面/LaTeX/HTML 都不会出现），
#   配 json 字段名 richText_blocks 关联。无哨兵（旧式整 JSON / mock 测试）→ 提取器返回 None，
#   调用方回退既有 _parse_json 路径（向后兼容，0 行为变更）。
# ---------------------------------------------------------------------------
RICHTEXT_SENTINELS: dict[str, str] = {"stem": "STEM", "answer": "ANSWER", "analysis": "ANALYSIS"}
# ⟦TAG⟧ ... ⟦/TAG⟧（DOTALL，非贪婪；段内含换行/反斜杠/引号/HTML 原样捕获，零转义）
_SENTINEL_RE = {
    key: re.compile(r"⟦" + tag + r"⟧(.*?)⟦/" + tag + r"⟧", re.DOTALL)
    for key, tag in RICHTEXT_SENTINELS.items()
}
_ANY_SENTINEL_RE = re.compile(
    r"⟦/?(?:" + "|".join(RICHTEXT_SENTINELS.values()) + r")⟧", re.DOTALL
)


def extract_sentinel_richtext(text: str) -> dict[str, Any] | None:
    """免转义哨兵框解析：从 opus 文本抠出 ⟦STEM⟧/⟦ANSWER⟧/⟦ANALYSIS⟧ 三段原文 + 解析剩余瘦 JSON。

    返回合并后的 dict（含 richText 三段 + 结构化字段），或 None（无任何哨兵 = 旧式整 JSON，
    调用方回退既有 _parse_json）。三段彼此独立：缺一段则该段为空字符串、不影响另两段/结构化字段。
    🔴 纯函数（除正则）可单测；不调 LLM、不抛（解不出剩余 JSON 仍返回带三段的 dict，结构字段尽力而为）。
    """
    if not text or "⟦" not in text:
        return None
    seg: dict[str, str] = {}
    hit = False
    for key, rx in _SENTINEL_RE.items():
        m = rx.search(text)
        if m:
            hit = True
            seg[key] = m.group(1).strip()
    if not hit:
        return None
    # 从文本里删掉哨兵块（连同标签）→ 剩下结构化瘦 JSON。先删整块，再清残留孤立标签。
    stripped = text
    for rx in _SENTINEL_RE.values():
        stripped = rx.sub("", stripped)
    stripped = _ANY_SENTINEL_RE.sub("", stripped)
    # 解析剩余瘦 JSON（结构化字段）；解不出 → 空 dict（仍交付三段富文本，宁丢结构维不丢题面）。
    base: dict[str, Any] = {}
    try:
        import json as _json
        s, e = stripped.find("{"), stripped.rfind("}")
        if s >= 0 and e > s:
            parsed = _json.loads(_repair_json_quotes(stripped[s : e + 1]))
            if isinstance(parsed, dict):
                base = parsed
    except Exception:  # noqa: BLE001 — 结构化段坏：三段富文本仍交付（U8 宁丢结构维不丢题面）
        base = {}
    # 三段塞回 richText（覆盖瘦 JSON 里的空占位；段缺则不写，留瘦 JSON 占位/缺省）
    rich = dict(base.get("richText") or {}) if isinstance(base.get("richText"), dict) else {}
    for key in RICHTEXT_SENTINELS:
        if key in seg:
            rich[key] = seg[key]
    base["richText"] = rich
    return base


async def parse_or_repair_entry(opus_text: str, V: Any) -> Any:
    """母题 opus 文本 → dict 的「解析 + 自愈」单一口径（B2 抽取，入口/classify 重锚共用）。
    🔴 R2b·U8 免转义哨兵框优先：opus 用哨兵框分隔 richText 三段 → extract_sentinel_richtext 原文
       抠段 + 瘦 JSON 解结构 → 绕开三段富文本转义坑（不再因一处转义坏整对象断）。无哨兵则回退：
    ① _parse_json（含数组解包）→ ② 失败则 _repair_entry_json（确定性引号修复 + LLM 兜底）→ 数组解包。
    返回 dict（成功）或 None（彻底解不出）。**只修解析、不重调 opus**（重调由调用方循环管）。"""
    # R2b·U8：哨兵框优先（命中即免转义抠段，根治三段富文本破损）
    sent = extract_sentinel_richtext(opus_text)
    if isinstance(sent, dict):
        return sent
    parsed = _unwrap_obj(_parse_json(opus_text))
    if isinstance(parsed, dict):
        return parsed
    parsed = _unwrap_obj(await _repair_entry_json(opus_text, V))
    return parsed if isinstance(parsed, dict) else None


async def solve_and_label_resilient(
    *, image_url: str, prompt: str, V: Any, model: str, max_tokens: int | None = None,
    on_progress: Any = None,
) -> Any:
    """classify 重锚路径的 opus 调用 + 解析自愈网（B2：复用入口 mother_opus_entry 的「重试≤2 +
    parse_or_repair_entry」同口径，根治『重锚 opus 坏 JSON 单次 _parse_json 早退 → 死循环』不对称）。

    🔴 与入口 mother_opus_entry（variant_entry.py:383-426）逐项同口径：
       - 至多 2 次尝试（调用异常重试 / 解析+修复仍失败重读母题一次）；
       - 每次返回先走 parse_or_repair_entry（确定性引号修复 + LLM 兜底）；
       - 调用走 mother_opus.solve_and_label（落 conv_trace + relay 熔断），不静默退 gpt。

    返回 dict（成功）；彻底失败抛 _SolveLabelError（携带 last_exc / 是否纯解析失败），
    由 classify 接住走可前进的 needs_confirm 降级（不卡死、不无限回环）。
    on_progress(stage_text) 可选：用于喂阶段灯文案（与入口 _emit_stage 同节奏）。
    """
    from agents import mother_opus  # 局部 import（与 variant_entry 顶层一致风格）

    last_exc: Exception | None = None
    for _attempt in range(2):
        try:
            opus_text = await mother_opus.solve_and_label(
                image_url=image_url or "", prompt=prompt, invoke=_ainvoke_text,
                model=model, max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001 — opus 调用异常（超时/全站失败）→ 重试一次
            last_exc = e
            if _attempt == 0:
                if on_progress:
                    on_progress("母题重锚读图重试中…")
                continue
            raise _SolveLabelError(parse_only=False, last_exc=e) from e
        parsed = await parse_or_repair_entry(opus_text, V)
        if isinstance(parsed, dict):
            return parsed
        if _attempt == 0:  # 解析+修复仍失败 → 重读母题一次（截断/坏 JSON 多为瞬时）
            if on_progress:
                on_progress("重锚解析失败，重读母题…")
    raise _SolveLabelError(parse_only=(last_exc is None), last_exc=last_exc)


class _SolveLabelError(Exception):
    """solve_and_label_resilient 自愈网全耗尽后的可控失败信号（携带是否纯解析失败 + 原异常）。"""

    def __init__(self, *, parse_only: bool, last_exc: Exception | None = None) -> None:
        self.parse_only = parse_only
        self.last_exc = last_exc
        super().__init__("mother opus solve+label exhausted self-heal")


def build_entry_prompt(*, utterance: str | None = None) -> str:
    """兼容壳（单测/旧调用）：返回稳定前缀（+ utterance 仅作可读拼接，真实调用走 build_entry_messages
    把 utterance 放变量后缀，不进缓存前缀）。"""
    if utterance:
        return ENTRY_SYSTEM_PREFIX + f"\n【老师附带要求】{utterance}"
    return ENTRY_SYSTEM_PREFIX


# ---------------------------------------------------------------------------
# D1 条件 confirm 判定（置信 < 0.80 或 ≥2 强候选章 → 弹窗）
# ---------------------------------------------------------------------------
def decide_confirm(entry: dict[str, Any]) -> dict[str, Any]:
    """据 opus 一把输出判是否需要弹窗确认。返回
       {needs_confirm:bool, reason:str, grade_book, chapter, grade_candidates, chapter_candidates, confidence}。
    D1：confidence < 0.80 → 确认；chapterCandidates 去重后 ≥2 → 章歧义 → 确认；gradeBook 空 → 确认。
    """
    conf = entry.get("confidence")
    try:
        conf_f = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        conf_f = 0.0
    grade_book = str(entry.get("gradeBook") or "").strip()
    chapter = str(entry.get("chapter") or "").strip()
    chap_cands = [str(c).strip() for c in (entry.get("chapterCandidates") or []) if str(c).strip()]
    chap_cands = list(dict.fromkeys(chap_cands))  # 去重保序
    grade_cands = [str(c).strip() for c in (entry.get("gradeCandidates") or []) if str(c).strip()]
    grade_cands = list(dict.fromkeys(grade_cands))

    reasons: list[str] = []
    if not grade_book:
        reasons.append("年级册判不出")
    if conf_f < CONF_CONFIRM_THRESHOLD:
        reasons.append(f"置信{conf_f:.2f}<{CONF_CONFIRM_THRESHOLD}")
    if len(chap_cands) >= 2:
        reasons.append(f"{len(chap_cands)}个强候选章歧义")
    return {
        "needs_confirm": bool(reasons),
        "reason": "；".join(reasons) or "高置信直过",
        "grade_book": grade_book,
        "chapter": chapter,
        "grade_candidates": grade_cands or ([grade_book] if grade_book else []),
        "chapter_candidates": chap_cands or ([chapter] if chapter else []),
        "confidence": conf_f,
    }


# ---------------------------------------------------------------------------
# 🔴 PRD-A-021 R2a·闸1（B5）·变式入口可选预设年级/章注入
#   老师在 FE 已选好母题范围（年级册 + 章 id）→ 经 config.configurable 传入：
#     preset_grade_book：年级册**人话名**（如「八年级下册」），可空；
#     preset_chapter_id：章节叶子/章 id（biz_subject id，4 位=年级册 / 7 位=章 / 完整=叶子），可空。
#   命中任一 → 视作「老师已定范围」：**跳过 classify-grade 那次 LLM 判定**（直接用预设当 grade_book）
#   + **不弹「确定范围」确认闸**（needs_confirm 强制 False），以老师选择为准。
#   🔴 防空跑（铁律）：预设须保证后续 solve/dna 有合法范围——
#     · preset_chapter_id 在 → 作 confirmed_chapter_id 注入，_finalize_high_conf 用其前 4 位当
#       grade_code 圈池、用其作闸B 锚定前缀（与 classify 确认章路径同口径）；
#     · 仅给 preset_grade_book（无 chapter_id）→ 回退用 grade_book 名归一 grade_code 圈年级池
#       （anchor 前缀退年级册），仍是合法范围。
#   没传任何预设 → 返回 None，入口维持原 classify + 确认闸路径，行为字节级不变。
# ---------------------------------------------------------------------------
def read_preset(config: RunnableConfig | None) -> dict[str, str] | None:
    """从 config.configurable 取老师预设范围。返回
       {grade_book:str, chapter_id:str, chapter_name:str}（grade_book/chapter_id 至少一个非空才返回）；
       都没传 → None。
    🔴 chapter_name（老师在选择器亲选的章人话名，如「第5章 一元一次方程」）→ 注入 R1/R2 prompt 学段约束：
       之前只注年级册、漏了章，模型解题/打标缺章语境（用户反馈「定下的章节没注入」）。"""
    conf = ((config or {}).get("configurable") or {}) if config else {}
    grade_book = str(conf.get("preset_grade_book") or "").strip()
    chapter_id = str(conf.get("preset_chapter_id") or "").strip()
    chapter_name = str(conf.get("preset_chapter_name") or "").strip()
    if not grade_book and not chapter_id:
        return None
    return {"grade_book": grade_book, "chapter_id": chapter_id, "chapter_name": chapter_name}


def _match_kp_in_pool(name: str, leaf_pool: list[tuple[str, str]]) -> str | None:
    """开集 kp 名 → 年级叶子池 id（高置信路径后锚）。精确名匹配优先，退包含匹配（最短名优先=最细叶子）。
    锚不到 → None（mother_opus.anchor_to_chapter 据此走「宁空不凑」need_anchor_review）。"""
    name = (name or "").strip()
    if not name:
        return None
    # 精确
    for pid, pname in leaf_pool:
        if str(pname).strip() == name:
            return str(pid)
    # 包含（叶子名 ⊆ opus名 或 opus名 ⊆ 叶子名）；多命中取叶子名最短（最具体）
    cands = [
        (str(pid), str(pname))
        for pid, pname in leaf_pool
        if name and (name in str(pname) or str(pname) in name)
    ]
    if cands:
        cands.sort(key=lambda x: len(x[1]))
        return cands[0][0]
    return None


# ---------------------------------------------------------------------------
# 🔴 PRD-C-107 B1·阶段一连续对话编排（一条 messages 累积线，全 sui-xiang，图发一次）
#   产出与旧 R6 完全同形（richtext_stem / solved_text / solved_answer / solved_solution / entry），
#   让下游 _finalize_high_conf / 低置信暂存路径字节不动。失败 → 返回 early_return dict（节点直接 return）。
# ---------------------------------------------------------------------------
async def _run_stage1_continuous(
    *, img_for_llm: str, user_text: str, teacher_memory: str | None,
    preset_grade: str | None, preset_chapter: str | None,
    teacher_intent: dict[str, Any], opus_model: str | None,
    url: str, config: RunnableConfig, V: Any,
) -> tuple[bool, dict[str, Any] | None, str, str, str, str, Any]:
    """阶段一连续对话三轮（turn1 誊抄+初判 → turn2 双料闭集+解题 → turn3 打标）。

    返回 (ok, early_return, richtext_stem, solved_text, solved_answer, solved_solution, entry)。
    ok=False 时 early_return 是节点应直接 return 的 dict（消息/错误帧已发）。
    """
    _empty = ("", "", "", "", None)
    _sentinel_mode = getattr(V.settings, "MOTHER_RICHTEXT_SENTINEL", False)

    # ===== turn1：誊抄 + 初判年级章 + 接住老师信息（唯一带图轮）=====
    _emit_stage("classify", "锚定考点", "running", "① 读题、判年级章中…")
    _emit_stage("richtext", "富文本化", "running", "誊抄题面为富文本…")
    t1_msgs = build_stage1_turn1_messages(
        image_url=img_for_llm, utterance=user_text or None, teacher_memory=teacher_memory,
        preset_grade_book=preset_grade, preset_chapter=preset_chapter,
    )
    t1_json: dict[str, Any] | None = None
    t1_raw = ""
    for _attempt in range(2):
        try:
            t1_raw = await _ainvoke_text(
                t1_msgs, model=opus_model,
                max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
                temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
                timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
                prefer_relay=None,  # 🔴 全程 sui-xiang
                trace_label="mother_entry",  # 🔴 turn1=誊抄+判年级章（消费记录分桶）
            )
        except Exception:  # noqa: BLE001 — turn1 调用异常 → 重试一次
            if _attempt == 0:
                _emit_stage("classify", "锚定考点", "running", "读题重试中…")
                continue
            break
        _p = _parse_json(t1_raw)
        if isinstance(_p, list) and _p and isinstance(_p[0], dict):
            _p = _p[0]
        if isinstance(_p, dict):
            t1_json = _p
            break
    if not isinstance(t1_json, dict):
        _emit_stage("classify", "锚定考点", "error", "读题/判年级失败（opus 超时/坏返回）")
        _emit_stage("richtext", "富文本化", "warn", "誊抄失败")
        _emit_error("mother_turn1_failed", "母题读题失败了（opus 超时或返回异常），请重试或换更清晰的图。")
        return (False, {
            "image_url": url, "_entry_finalized": False,
            "messages": [AIMessage(content="母题读题失败了（opus 超时或返回异常），请重试或换一张更清晰的题目图。")],
        }, *_empty)

    richtext_stem = _sanitize_rich_text(str(t1_json.get("stem") or "").strip())
    if richtext_stem:
        _emit_richtext_stem(richtext_stem)
        _emit_stage("richtext", "富文本化", "done", "题面已整理为富文本")
    else:
        _emit_stage("richtext", "富文本化", "warn", "题面誊抄空，回退打标轮题面")

    # turn1 的年级初判（接住老师 utterance / preset 优先；都没有 → 用 opus 初判圈池）
    grade_for_pool = (
        preset_grade or teacher_intent.get("grade")
        or str(t1_json.get("gradeBook") or "").strip() or None
    )
    grade_code_pool = V._grade_to_code(grade_for_pool) if grade_for_pool else None
    # preset 章 id 前 4 位优先（与 _finalize_high_conf 同口径）
    _pre_chap = str((read_preset(config) or {}).get("chapter_id") or "").strip()
    if _pre_chap and len(_pre_chap) >= 4:
        grade_code_pool = _pre_chap[:4]

    # ===== 双料备料（闭集叶子池 + 模型工具箱，只读 ETL；缺位 → 空注入裸解，铁律④不卡死）=====
    include_review = V._wants_review_books(user_text)
    leaf_pool: list[tuple[str, str]] = []
    model_toolbox_clause = ""
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    if grade_code_pool:
        _client = RuoyiClient(token=token)
        try:
            leaf_pool = await V.leaf_pool_for_grade(
                grade_code_pool, _client, include_review_books=include_review
            )
        except Exception:  # noqa: BLE001 — 池故障 → 空池（裸标降级，后锚仍在）
            leaf_pool = []
        finally:
            await _client.aclose()
        try:
            _tb = model_anchor.toolbox_for_grade(grade_code_pool)
            model_toolbox_clause = model_anchor.build_toolbox_clause(_tb)
        except Exception:  # noqa: BLE001 — 工具箱故障 → 空（裸解降级）
            model_toolbox_clause = ""

    # ===== turn2：双料闭集注入 + 解题（累积·纯文本·无图）=====
    _emit_stage("classify", "锚定考点", "running", "② 解题中（带考点+大招，一步步算）…")
    _emit_stage("solve", "解题", "running", "一步步算，求稳准…")
    convo: list[Any] = [*t1_msgs, AIMessage(content=t1_raw)]
    solve_turn = build_stage1_solve_turn(
        leaf_pool=leaf_pool, model_toolbox=model_toolbox_clause,
        anchored_grade=grade_for_pool, anchored_chapter=preset_chapter,
        teacher_model=teacher_intent.get("model"),
    )
    convo_solve = [*convo, *solve_turn]
    solved_text = ""
    solve_exc: Exception | None = None
    for _attempt in range(2):
        try:
            solved_text = await _ainvoke_text(
                convo_solve, model=opus_model,
                max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
                temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
                timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
                public_stream=True,  # 解题正文流式吐前端（看得见思路）
                on_reasoning=_emit_reasoning,
                prefer_relay=None,  # 🔴 全程 sui-xiang（赌注：连续上下文压住漂移；退化则 C107_CONTINUOUS=0）
                trace_label="mother_solve",  # 🔴 turn2=解题（消费记录分桶）
            )
        except Exception as e:  # noqa: BLE001
            solve_exc = e
            if _attempt == 0:
                _emit_stage("classify", "锚定考点", "running", "解题重试中…")
                continue
            break
        if solved_text and solved_text.strip():
            solve_exc = None
            break
        if _attempt == 0:
            _emit_stage("classify", "锚定考点", "running", "解题空返回，重算一次…")
    if not (solved_text and solved_text.strip()):
        _emit_stage("classify", "锚定考点", "error", "母题解题失败（opus 超时/空返回）")
        _emit_stage("solve", "解题", "warn", "母题解题失败（opus 超时/空返回）")
        _emit_error("mother_solve_failed",
                      f"母题解题失败（{str(solve_exc)[:80] if solve_exc else '空返回'}），请重试或换更清晰的图。")
        return (False, {
            "image_url": url, "_entry_finalized": False,
            "messages": [AIMessage(content="母题解题失败了（opus 超时或空返回），请重试或换一张更清晰的题目图。")],
        }, *_empty)
    _emit_stage("solve", "解题", "done", "母题已解出")
    solved_answer = _extract_solved_answer(solved_text)
    solved_solution = _strip_final_answer_marker(solved_text)

    # ===== turn3：打标（累积·纯文本·无图）=====
    _emit_stage("classify", "锚定考点", "running", "③ 富文本整理 + 打标分类…")
    _emit_stage("label", "深度解析", "running", "深度解析 + 10 维打标分类…")
    label_turn = build_stage1_label_turn(
        leaf_pool=leaf_pool, model_toolbox=model_toolbox_clause, sentinel=_sentinel_mode,
    )
    convo_label = [*convo_solve, AIMessage(content=solved_text), *label_turn]
    entry: Any = None
    opus_exc: Exception | None = None
    _rf = None if _sentinel_mode else RESPONSE_FORMAT_ENTRY
    for _attempt in range(2):
        try:
            opus_text = await _ainvoke_text(
                convo_label, model=opus_model,
                max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
                temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
                response_format=_rf,
                timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
                prefer_relay=None,  # 🔴 全程 sui-xiang
                trace_label="mother_label",  # 🔴 turn3=10维打标（消费记录分桶）
            )
        except Exception as e:  # noqa: BLE001
            opus_exc = e
            if _attempt == 0:
                _emit_stage("classify", "锚定考点", "running", "打标重试中…")
                continue
            break
        parsed = extract_sentinel_richtext(opus_text)
        if not isinstance(parsed, dict):
            parsed = _parse_json(opus_text)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            _emit_stage("classify", "锚定考点", "running", "解析修复中…")
            parsed = await parse_or_repair_entry(opus_text, V)
        if isinstance(parsed, dict):
            entry, opus_exc = parsed, None
            break
        if _attempt == 0:
            _emit_stage("classify", "锚定考点", "running", "解析失败，重整一次…")
    if not isinstance(entry, dict):
        _emit_stage("classify", "锚定考点", "error", "母题打标解析失败")
        _emit_stage("label", "深度解析", "warn", "深度解析/打标失败")
        _emit_error("mother_opus_failed",
                      f"母题打标失败（{str(opus_exc)[:80] if opus_exc else '解析失败'}），请重试。")
        return (False, {
            "image_url": url, "_entry_finalized": False,
            "messages": [AIMessage(content="母题打标结果没解析出来，请重试。")],
        }, *_empty)
    _emit_stage("label", "深度解析", "done", "深度解析 + 10 维打标完成")

    # turn1 富文本题面权威（非空覆盖 turn3 stem，与 R6 同口径）
    if not richtext_stem:
        _rt3 = entry.get("richText") or {}
        if isinstance(_rt3, dict) and _rt3.get("stem"):
            richtext_stem = _sanitize_rich_text(str(_rt3.get("stem")))
    return (True, None, richtext_stem, solved_text, solved_answer, solved_solution, entry)


# ---------------------------------------------------------------------------
# 入口节点（懒导入 variant 防循环：variant 模块加载期 import 本模块挂图，本节点运行期才反向用 variant 机具）
# ---------------------------------------------------------------------------
async def mother_opus_entry(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """B1a 塌缩入口节点：opus 一把（判章+解题+10维打标）→ 高置信直过/低置信弹确认。

    🔴 PRD-C-107 B1：settings.C107_CONTINUOUS 开（默认）→ 阶段一走「一条连续对话」
       （_run_stage1_continuous，全 sui-xiang、图发一次、双料闭集注入、接住老师信息）；
       关 → 回退旧 R6 三独立调用（决策表 fallback）。两路产出同形 → 下游 finalize 字节不动。"""
    from agents import variant as V  # 懒导入防循环（运行期才用）

    url = V._extract_image_url(V._latest_human_text(state.get("messages", []))) or state.get(
        "image_url"
    )
    if not url:
        return {"_entry_finalized": False,
                "messages": [AIMessage(content="请先贴一张题目图的 OSS URL，我才能开始举一反三。")]}

    # 🔴 B5 单一全局日预算护栏（G7）：当日花费超阈值 → 拦截母题 opus 一把（不调，不静默烧钱）。
    from agents import cost_guard
    # 🔴 P5：护栏读库丢线程池（async 版），慢库不卡 asyncio loop / 不拖垮并发 SSE。
    if await cost_guard.is_budget_exceeded_async():
        # 已超 → 取一次 status 拼提示文案（缓存 ~60s，几乎不二次查库）；放线程池保险。
        st = await cost_guard.budget_status_async()
        _emit_stage("classify", "锚定考点", "warn", "今日 AI 额度已用尽")
        _emit_error("budget_exceeded",
                      f"今日 AI 额度已用尽（已用 ¥{st['spend']:.2f}/¥{st['limit']:.2f}），请稍后或明日再试。")
        return {
            "_entry_finalized": False, "image_url": url,
            "messages": [AIMessage(content="今日 AI 额度已用尽，举一反三暂停以控成本，请稍后或明日再试。")],
        }

    user_text = V._strip_urls(V._latest_human_text(state.get("messages", [])))
    # B2 缓存接缝：稳定 system 前缀 ‖ 变量 user 后缀（题图+query+teacher记忆后置）。
    # 🔴 B4 记忆注入（走 RuoYi HTTP，只取 enabled，停用不注入 G14/G9）；放变量后缀（多用户接缝，
    #    不进缓存稳定前缀）；best-effort（记忆故障绝不卡 mother 主流程）。
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    teacher_memory: str | None = None
    try:
        from agents import teacher_memory as TM
        _mem_client = RuoyiClient(token=token)
        teacher_memory = await TM.fetch_memory_block(_mem_client)
        await _mem_client.aclose()
    except Exception:  # noqa: BLE001 — 记忆拉取失败 → 不注入，继续
        teacher_memory = None
    # 🔴 2026-06-17：sui-xiang(kiro 逆向站)主站不抓远程图 URL，母题图必须 base64 内嵌；
    #   下载失败 → 原 url 兜底（aigeek failover 仍可用远程 URL，不破熔断备用路径）。
    img_for_llm = await _to_b64_data_url(url)
    _preset_for_prompt = read_preset(config)
    _preset_grade = (_preset_for_prompt or {}).get("grade_book")
    _preset_chapter = (_preset_for_prompt or {}).get("chapter_name")  # 🔴 老师定的章人话名，注入 R1/R2
    opus_model = V.settings.variant_model("mother_solve_label")  # fail-fast 已锁 opus

    # 🔴 PRD-C-107 B1·接住老师附带信息（解析 utterance 抽 grade/chapter/model；确定性、不耗 LLM）。
    teacher_intent = _extract_teacher_intent(user_text)

    # =========================================================================
    # 🔴 PRD-C-107 B1·分支：连续对话（默认）vs 旧 R6 三独立调用（fallback，C107_CONTINUOUS=0 切回）。
    #   两路产出同形（richtext_stem/solved_text/solved_answer/solved_solution/entry），下游字节不动。
    # =========================================================================
    _use_continuous = getattr(V.settings, "C107_CONTINUOUS", True)
    if _use_continuous:
        ok_c, early_c, richtext_stem, solved_text, solved_answer, solved_solution, entry = (
            await _run_stage1_continuous(
                img_for_llm=img_for_llm, user_text=user_text, teacher_memory=teacher_memory,
                preset_grade=_preset_grade, preset_chapter=_preset_chapter,
                teacher_intent=teacher_intent, opus_model=opus_model,
                url=url, config=config, V=V,
            )
        )
        if not ok_c:
            return early_c  # type: ignore[return-value]
    else:
        # 🔴 PRD-C-106 B1①·带料解题工具箱：解题前备「该年级全量模型名单」注入 R1 解题 + R2 打标 prompt。
        #   年级前缀来源（解题前能定的）：预设章 id 前 4 位 → 3 位根 / 预设年级名归一。非预设贴图（grade 由
        #   opus 在 R1 现判）→ 解题前拿不到 grade → 空工具箱裸解（不卡死），但下游 anchor_models_from_names
        #   仍按 opus 给的名纯代码映射（带料缺位不影响消重复解题 + 诚实三态）。库故障 → 空工具箱（降级）。
        _toolbox_grade_code: str | None = None
        _pre = _preset_for_prompt or {}
        _pre_chap = str(_pre.get("chapter_id") or "").strip()
        if _pre_chap and len(_pre_chap) >= 4:
            _toolbox_grade_code = _pre_chap[:4]
        elif _preset_grade:
            _toolbox_grade_code = V._grade_to_code(_preset_grade) or None
        model_toolbox_clause = ""
        if _toolbox_grade_code:
            try:
                _tb = model_anchor.toolbox_for_grade(_toolbox_grade_code)
                model_toolbox_clause = model_anchor.build_toolbox_clause(_tb)
            except Exception:  # noqa: BLE001 — 工具箱备料失败 → 裸解降级（绝不卡母题主链）
                model_toolbox_clause = ""

        # =========================================================================
        # 🔴 R6 三步编排（用户 2026-06-22 拍板）：① 富文本化(sui-xiang·只誊抄题面) ∥ ② R1解题(aigeek)
        #   **并发跑**（asyncio.gather，富文本化不阻塞解题）；两者完成后 → ③ R2打标(sui-xiang)。
        #   富文本化跑完即 emit 一帧让 FE 用富文本题面替换原图占位；其结果作为最终 entry.stem 权威来源。
        #   富文本化失败/空 → 回退用 R2 的 stem（绝不卡死主链；只有 R1 解题失败才硬终止）。
        # =========================================================================
        # 🔴 R6 进度条新 stage 帧契约（2026-06-22 拍板，定题中阶段对齐重构后母题编排）：
        #   母题阶段不再都塞单一 classify key，拆成各轮独立 key（进度条真值源）：
        #     · richtext（富文本化·异步组）：_run_richtext 开始 running → 跑完 done（失败 warn，不卡主链）。
        #     · solve（解题·主步骤）：_run_solve 开始 running → 跑完 done（失败已硬终止）。
        #     · label（深度解析/打标·主步骤）：R2 轮 running → done。
        #     · figure-mother（切图·异步组）：FE 经 upsertFigureStage 自发，toolkit 不发。
        #     · review（确认母题·闸）：await_mother_review 发 await → 老师点开始 done（沿用既有）。
        #   旧 classify 帧继续发（向后兼容 + 仍驱动「读图锚定」legacy/低置信路径），新 key 是 dingNodes 真值源。
        _emit_stage("classify", "锚定考点", "running", "① 解题中（一步步算，求稳准）…")
        _emit_stage("richtext", "富文本化", "running", "誊抄题面为富文本（异步·并行）…")
        _emit_stage("solve", "解题", "running", "一步步算，求稳准…")

        # --- ① 富文本化任务（独立异步·sui-xiang 默认网关 = prefer_relay=None；不走 MOTHER_SOLVE_RELAY） ---
        async def _run_richtext() -> str:
            rt_messages = build_richtext_messages(
                image_url=img_for_llm, utterance=user_text or None, teacher_memory=teacher_memory,
            )
            try:
                txt = await _ainvoke_text(
                    rt_messages, model=opus_model,
                    max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
                    temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
                    timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
                    prefer_relay=None,  # 🔴 默认走 sui-xiang（RELAY_POOL[0]），不走 aigeek
                )
            except Exception:  # noqa: BLE001 — 富文本化失败绝不卡主链，回退 R2 stem
                _emit_stage("richtext", "富文本化", "warn", "富文本化失败，回退 R2 题面（不影响出题）")
                return ""
            stem = _sanitize_rich_text((txt or "").strip())
            if not (stem and str(stem).strip()):
                _emit_stage("richtext", "富文本化", "warn", "富文本化空返回，回退 R2 题面（不影响出题）")
                return ""
            # 轻量验证（非空已过；富文本机器检失败仅告警、不阻塞、不丢结果）。
            try:
                _rt_chk = mother_opus.validate_rich_text({"stem": stem})
                if not _rt_chk["ok"]:
                    _emit_stage("classify", "锚定考点", "running",
                                  f"题面富文本机器检 {len(_rt_chk['issues'])} 处小问题（不阻塞）")
            except Exception:  # noqa: BLE001
                pass
            # 早帧：FE 占位区把原图替换成富文本题面（解题/打标还在跑时就能读到干净题面）。
            _emit_richtext_stem(str(stem))
            _emit_stage("richtext", "富文本化", "done", "题面已整理为富文本")
            return str(stem)

        # --- ② R1 解题轮（纯解题·aigeek·public_stream 流式吐对话气泡=看得见思路·宁慢求准） ---
        async def _run_solve() -> tuple[str, Exception | None]:
            solve_messages = build_solve_messages(
                image_url=img_for_llm, utterance=user_text or None, teacher_memory=teacher_memory,
                preset_grade_book=_preset_grade, preset_chapter=_preset_chapter,
                model_toolbox=model_toolbox_clause,  # 🔴 B1①·带料解题工具箱
            )
            _txt = ""
            _exc: Exception | None = None
            for _attempt in range(2):  # 解题轮失败/空返回重试一次（治瞬时截断/空返回）
                try:
                    _txt = await _ainvoke_text(
                        solve_messages, model=opus_model,
                        max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
                        temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
                        timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
                        public_stream=True,  # 🔴 解题正文流式吐前端打字机（看得见思路·不干等 117s）
                        on_reasoning=_emit_reasoning,
                        prefer_relay=V.settings.MOTHER_SOLVE_RELAY,  # 🔴 母题解题轮走 aigeek（仍 opus·只换网关）
                    )
                except Exception as e:  # noqa: BLE001 — 解题轮调用异常（超时/全站失败）
                    _exc = e
                    if _attempt == 0:
                        _emit_stage("classify", "锚定考点", "running", "母题读图解题重试中…")
                        continue
                    break
                if _txt and _txt.strip():
                    _exc = None
                    break
                if _attempt == 0:
                    _emit_stage("classify", "锚定考点", "running", "解题空返回，重读一次…")
            return _txt, _exc

        # 并发：富文本化 ∥ 解题（gather；富文本化内已吞异常，return_exceptions 兜底解题侧不被波及）。
        _rt_res, _solve_res = await asyncio.gather(
            _run_richtext(), _run_solve(), return_exceptions=True,
        )
        richtext_stem = _rt_res if isinstance(_rt_res, str) else ""
        if isinstance(_solve_res, tuple):
            solved_text, solve_exc = _solve_res
        else:  # _run_solve 抛了未捕获异常（理论不至于，gather 兜底）
            solved_text, solve_exc = "", (_solve_res if isinstance(_solve_res, Exception) else None)

        if not (solved_text and solved_text.strip()):
            _emit_stage("classify", "锚定考点", "error", "母题解题失败（opus 超时/空返回）")
            _emit_stage("solve", "解题", "warn", "母题解题失败（opus 超时/空返回）")
            _emit_error("mother_solve_failed",
                          f"母题解题失败（{str(solve_exc)[:80] if solve_exc else '空返回'}），请重试或换更清晰的图。")
            return {
                "image_url": url, "_entry_finalized": False,
                "messages": [AIMessage(content="母题解题失败了（opus 超时或空返回），请重试或换一张更清晰的题目图。")],
            }
        _emit_stage("solve", "解题", "done", "母题已解出")
        solved_answer = _extract_solved_answer(solved_text)
        solved_solution = _strip_final_answer_marker(solved_text)

        # =========================================================================
        # 🔴 R5·R2 结构化打标轮（接 R1 权威解答 → 忠实富文本三段 + 10 维 DNA 打标 + 判章；绝不重解）。
        #   solvedAnswer 终钉 = R1 抠出的最终答案（R2 只整理打标，无权改答案）。
        # =========================================================================
        _emit_stage("classify", "锚定考点", "running", "② 富文本整理 + 打标分类…")
        _emit_stage("label", "深度解析", "running", "深度解析 + 10 维打标分类…")
        struct_messages = build_struct_messages(
            image_url=img_for_llm, solved_solution=solved_solution, solved_answer=solved_answer,
            utterance=user_text or None, teacher_memory=teacher_memory,
            preset_grade_book=_preset_grade, preset_chapter=_preset_chapter,
            model_toolbox=model_toolbox_clause,  # 🔴 B1①·带料打标(opus 从工具箱选 modelCandidates)
        )
        entry = None
        opus_exc: Exception | None = None
        # 🔴 R2b·U8：哨兵模式 = 输出 JSON + 尾随哨兵框 → **不能下发 response_format**（json_schema 强制
        #   整段合法 JSON、拒尾随文本）；关时维持旧式整 JSON + response_format 硬锁 10 维。
        _sentinel_mode = getattr(V.settings, "MOTHER_RICHTEXT_SENTINEL", False)
        _rf = None if _sentinel_mode else RESPONSE_FORMAT_ENTRY
        for _attempt in range(2):
            try:
                opus_text = await _ainvoke_text(
                    struct_messages, model=opus_model,
                    max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
                    temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
                    response_format=_rf,
                    timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
                    prefer_relay=None,  # 🔴 R6（2026-06-22 拍板）：打标轮改走 sui-xiang（默认网关·仍 opus·只换网关）
                )
            except Exception as e:  # noqa: BLE001 — 结构化轮调用异常
                opus_exc = e
                if _attempt == 0:
                    _emit_stage("classify", "锚定考点", "running", "富文本打标重试中…")
                    continue
                break
            # 🔴 免转义哨兵框优先：抠三段 richText 原文 + 瘦 JSON 解结构（绕三段富文本转义坑）。
            parsed = extract_sentinel_richtext(opus_text)
            if not isinstance(parsed, dict):
                parsed = _parse_json(opus_text)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                _emit_stage("classify", "锚定考点", "running", "解析修复中…")
                parsed = await parse_or_repair_entry(opus_text, V)
            if isinstance(parsed, dict):
                entry, opus_exc = parsed, None
                break
            if _attempt == 0:  # 解析+修复仍失败 → 重整一次（截断/坏 JSON 多为瞬时）
                _emit_stage("classify", "锚定考点", "running", "解析失败，重整一次…")
        if not isinstance(entry, dict):
            if opus_exc is not None:
                _emit_stage("classify", "锚定考点", "error", "母题读图解题失败（opus 超时/异常）")
                _emit_stage("label", "深度解析", "warn", "深度解析/打标失败（opus 超时/异常）")
                _emit_error("mother_opus_failed", f"母题读图解题失败（{str(opus_exc)[:80]}），请重试或换更清晰的图。")
                return {
                    "image_url": url, "_entry_finalized": False,
                    "messages": [AIMessage(content="母题读图解题失败了（opus 超时或异常），请重试或换一张更清晰的题目图。")],
                }
            _emit_stage("classify", "锚定考点", "error", "母题富文本打标解析失败")
            _emit_stage("label", "深度解析", "warn", "深度解析/打标结果解析失败")
            _emit_error("mother_opus_parse_fail", "母题富文本打标结果解析失败，请重试。")
            return {
                "image_url": url, "_entry_finalized": False,
                "messages": [AIMessage(content="母题富文本打标结果没解析出来，请重试。")],
            }
        _emit_stage("label", "深度解析", "done", "深度解析 + 10 维打标完成")

    # 🔴 R6（2026-06-22 拍板）：富文本化轮（sui-xiang·只誊抄题面）的结果 = 最终 entry.stem 权威源。
    #   非空 → 覆盖 R2 誊抄的 stem（下游 _finalize_high_conf / early_dna 统一从 entry.richText.stem
    #   回填 mother_dna["stem"]，故只需在此处覆盖一处）；为空 → 保留 R2 的 stem 作兜底（绝不卡死）。
    if richtext_stem and richtext_stem.strip():
        _rich = entry.get("richText")
        if not isinstance(_rich, dict):
            _rich = {}
        _rich["stem"] = richtext_stem
        entry["richText"] = _rich

    # 🔴 R5：solvedAnswer 终钉 = R1 解题轮抠出的最终答案（R2 只整理打标，无权改答案）——
    #   根治「推导出 1 却硬填幻觉 12」的答案-推导矛盾（解题已在 R1 隔离完成、答案不再被打标轮改写）。
    if solved_answer:
        entry["solvedAnswer"] = solved_answer

    has_figure = bool(entry.get("has_figure"))  # 🔴 仅记录，不 reject（反转 C-017 带图打回）
    decision = decide_confirm(entry)

    # 🔴 R2a·闸1（B5）：老师预设了年级/章 → 以老师选择为准，跳过确认闸（needs_confirm=False）+
    #   把预设范围覆盖进 decision（不再采信 opus 判的年级/章；opus 的解题/DNA 仍复用）。
    #   preset_chapter_id 经 base_out.confirmed_chapter_id 注入 → _finalize_high_conf 用它当
    #   grade_code 前缀 + 闸B 锚定前缀（与 classify 确认章路径同口径，不空跑）。
    preset = read_preset(config)
    if preset:
        if preset.get("grade_book"):
            decision["grade_book"] = preset["grade_book"]
            decision["grade_candidates"] = [preset["grade_book"]]
        decision["needs_confirm"] = False
        decision["confidence"] = max(float(decision.get("confidence") or 0.0), CONF_CONFIRM_THRESHOLD)
        decision["reason"] = "老师已预设年级/章（跳过确认）"

    # 🔴 PRD-C-107 B1·接住老师打字的年级（= 等同 preset，跳确认闸）：老师在 utterance 里明说了年级
    #   （如「八下」）但没在 picker 选 preset → 也视作老师已定范围，覆盖 decision 年级 + 跳确认。
    #   只认 grade（年级定了下游能圈年级池）；章/模型作为软信息在解题轮已注入，不强制跳确认依据。
    elif _teacher_intent_skips_confirm(teacher_intent):
        _t_grade = teacher_intent.get("grade")
        if _t_grade:
            decision["grade_book"] = _t_grade
            decision["grade_candidates"] = [_t_grade]
        decision["needs_confirm"] = False
        decision["confidence"] = max(float(decision.get("confidence") or 0.0), CONF_CONFIRM_THRESHOLD)
        decision["reason"] = "老师已在对话里指明年级（跳过确认）"

    # 出题配方旋钮（与 analyze 同口径）：utterance 非空 → 独立纯文本抽取（数量词稳）；纯贴图不抽。
    knobs: dict[str, Any] = {}
    if user_text:
        try:
            knobs = await V._extract_knobs({**state, "image_url": url, "messages": state.get("messages", [])})
        except Exception:  # noqa: BLE001
            knobs = {}

    # 新母题轮基础 state（重置在途母题态 + 暂存 opus 一把输出供 confirm-resume 复用）
    base_out: dict[str, Any] = {
        "image_url": url,
        "images_count": 1,
        "questions_in_image": 1,
        "knobs": knobs,
        "shape_defects": [],
        "mother_precheck": None,
        "awaiting_mother_confirm": False,
        "mother_rejected": False,
        # 🔴 R2a·闸1（B5）：预设章 id → 落 confirmed_chapter_id（= 老师确认章语境，_finalize_high_conf
        #   据此用预设章前缀圈池/锚定；预设仅给年级册名时为空串，退年级册 code 圈池）。
        "confirmed_chapter_id": (preset.get("chapter_id") or None) if preset else None,
        "confirmed_grade_book_id": None,
        # 🔴 M4/PRD-A-018·新母题入口必清旧轮残留 items（与 mother_dna 被本轮覆盖对齐）：
        #   同一 thread（会话级，贴新图不换 thread——PF-1 实测）先对图A 出过变式（items 非空）后直接贴图B
        #   时，旧实现 base_out 从不清 items → 点「开始举一反三」时 route_entry 守卫 `not items` 为 False →
        #   落 parse 把「开始」当编辑指令乱分诊 → 新母题永不出变式。此处统一清，所有入口分支（needs_confirm/
        #   空池 early/not-confirmed picker/confirmed）spread base_out 都吃到。连带清手排标记/剔除叙事/
        #   改主考点撤销快照（都属上一母题轮的残留，不得泄漏到新母题）。
        "items": [],
        "manual_order": False,
        "dropped_notes": [],
        "main_kp_prev": None,
        # 🔴 R2a·闸3/闸4：新母题轮清上一轮的闸断标记（章冲突闸断 / 读图低置信拦截一次性标记），
        #   否则同 thread 第二张图带 stale True → 该拦的不拦 / 该闸的不闸。
        "_bug03_gated_chapter": None,
        "_lowconf_blocked": False,
        # B1a：暂存 opus 一把判定（has_figure 给 B3 切图判定；entry_opus 给 confirm-resume 兜底）
        "mother_has_figure": has_figure,
        "entry_decision": decision,
        "_entry_finalized": True,  # after_mother_entry 路由信号（高置信/弹窗均算「本轮入口处理过」）
    }

    # ---- 低置信 / 章歧义 → 弹窗确认（resume 走既有 classify 池注入重锚 +1 次 opus） ----
    if decision["needs_confirm"]:
        # 🔴 PRD-C-107 B1·确认弹窗端出 reason（人话原因，让老师看到「为什么要确认」）。
        #   _build_confirm_payload 在既有 needConfirm 契约上增量加 reason（FE 容忍未知键，B4 渲染）。
        payload = _build_confirm_payload(decision)
        _emit_need_confirm(payload)
        # 🔴 BUG-04（2026-06-19）·读图置信极低提示：confidence 极低（< LOW_CONF_HINT_THRESHOLD）时，
        #   多半是图本身不清/非标准题图——状态条 detail + 母题确认气泡里明确建议换清晰图，别让老师在读不清
        #   的图上白点「开始举一反三」。复用既有低置信分支加文案（不改判定逻辑、不新增闸）。
        _very_low = decision["confidence"] < LOW_CONF_HINT_THRESHOLD
        _await_detail = (
            "这张图可能不够清晰·建议换张清晰的图，或确认年级与章后继续"
            if _very_low else "请确认年级与章后继续"
        )
        _emit_stage("classify", "锚定考点", V.STAGE_AWAIT, _await_detail)
        _emit_stage("knobs", "解析配方", V.STAGE_AWAIT, "待确认年级章后定配方")
        grade_line = decision["grade_book"] or "（未判出，请手选）"
        chapter_line = decision["chapter"] or "（未判出，请手选）"
        _low_conf_note = (
            "\n\n⚠️ 我对这张图的读图把握很低——**可能这张图不太适合做母题**（拍得不清/不是标准题图）。"
            "建议换一张更清晰的题目图，或先确认年级与章后再继续。"
            if _very_low else ""
        )
        body = (
            f"我读图判了一下母题范围（{decision['reason']}），**请确认年级册与章**再继续举一反三：\n\n"
            f"- 年级册：**{grade_line}**\n- 章：**{chapter_line}**\n\n"
            "确认无误请回复「确认」，需要修改请直接告诉我正确的年级/章。"
            f"{_low_conf_note}"
        )
        # 暂存 opus 一把的 richText/solve/dna 供 classify-resume 复用（避免 confirm 后白丢这次解题）
        prov_dna = dict(state.get("mother_dna") or {})
        rich = entry.get("richText") or {}
        if isinstance(rich, dict):
            if rich.get("stem"):
                prov_dna["stem"] = _sanitize_rich_text(rich.get("stem"))
            if rich.get("answer"):
                prov_dna["answer"] = _sanitize_rich_text(rich.get("answer"))
            if rich.get("analysis"):
                prov_dna["analysis"] = _sanitize_rich_text(rich.get("analysis"))
        # 🔴 PRD-C-100 B2 死循环根治：低置信暂存与高置信路径对齐——补 mother_solve_source="opus"
        #   + 完整 dna（含 main_kp/skeleton）。否则 confirm 后 classify 的 _reuse_ok（variant.py:1879）
        #   判 False → 重锚不复用首解 → 重调 opus 二次读图、niche 题偶发坏 JSON → 退回 needs_confirm
        #   → 又弹确认框 = 死循环。dna 归一走与高置信路径同一函数 mother_opus.opus_to_dna(entry)。
        prov_dna_obj = mother_opus.opus_to_dna(entry)
        prov_skeleton = prov_dna_obj.get("skeleton") or []
        if prov_skeleton:
            prov_dna["solution_skeleton"] = join_skeleton(prov_skeleton)  # P8 逐行净化
        prov_solved = entry.get("solvedAnswer")
        if prov_solved:
            prov_dna["solved_answer"] = _sanitize_rich_text(prov_solved)
        prov_dna["dna"] = prov_dna_obj
        prov_dna["mother_solve_source"] = "opus"
        # 🔴 R2a·闸2（B5b）首解范围指纹（写点①·低置信路）：记首解所在年级册 4 位 code 前缀。
        #   低置信路常无法判出年级册（decision.grade_book 空）→ 空 fp。空 fp 的比较策略见 _should_resolve
        #   （空 fp 不判「册变」，只靠「首解没锚牢」子集触发重解，不撞 B2 死循环）。
        prov_dna["_solve_range_fp"] = V._grade_to_code(decision.get("grade_book")) or ""
        return {
            **base_out,
            "analysis": {
                "grade": {"value": decision["grade_book"], "confidence": decision["confidence"]},
                "subject": "数学",
                "kp": {"value": (entry.get("dna") or {}).get("primaryKp", {}).get("name")
                       if isinstance((entry.get("dna") or {}).get("primaryKp"), dict) else None,
                       "confidence": decision["confidence"]},
                "qtype": {"value": (entry.get("dna") or {}).get("qtype"), "confidence": decision["confidence"]},
                # 🔴 BUG-08：opus 判定章文本（空则省略键）→ 母题卡 anchor.chapter_name 回灌。
                **({"chapter": {"id": "", "value": str(decision["chapter"]).strip()}}
                   if str(decision.get("chapter") or "").strip() else {}),
            },
            "mother_dna": prov_dna,
            "awaiting_mother_confirm": True,
            "mother_confirmed": False,
            "messages": [AIMessage(content=body)],
        }

    # ---- 高置信 → 1 次 opus 全 done：代码后锚 + 母题卡先出 + 硬停 ----
    out = await _finalize_high_conf(state, config, entry, decision, base_out, V)
    return out


async def _finalize_high_conf(
    state: dict[str, Any], config: RunnableConfig, entry: dict[str, Any],
    decision: dict[str, Any], base_out: dict[str, Any], V: Any,
) -> dict[str, Any]:
    """高置信路径终结：grade_code → 年级叶子池 → 开集 kp 后锚（闸B）→ model_anchor → 母题卡 → await_review。

    🔴 闸A/闸B 走 mother_opus 同一纯函数（与 classify 一致逻辑，不另造判决）。
    🔴 不再调 opus（复用一把输出）—— 这是「高置信1次全done」的兑现。
    """
    analysis: dict[str, Any] = {
        "grade": {"value": decision["grade_book"], "confidence": max(decision["confidence"], V.CONF_GATE)},
        "subject": "数学",
        "kp": {"value": None, "confidence": 0},
        "qtype": {"value": None, "confidence": 0},
    }
    # 🔴 BUG-08（2026-06-19）：opus 判定的章文本落进 analysis.chapter → _build_mother_card 回灌
    #   母题卡 anchor.chapter_name（空就不落，不伪造）。
    if str(decision.get("chapter") or "").strip():
        analysis["chapter"] = {"id": "", "value": str(decision["chapter"]).strip()}
    mother_dna = dict(state.get("mother_dna") or {})

    include_review_books = V._wants_review_books(V._latest_human_text(state.get("messages", [])))

    # 🔴 R2a·闸1（B5）：预设章 id（base_out.confirmed_chapter_id，老师已选范围）= 锚定事实源，
    #   压过 opus 图读 grade。前 4 位 = 年级册 code（与 classify B4-fix 确认章驱动同口径），
    #   并作闸B 锚定前缀（收窄到预设章）。无预设章 → 退 opus gradeBook 归一 / 粗考点反查。
    preset_chapter_id = str(base_out.get("confirmed_chapter_id") or "").strip() or None

    # 年级 code：预设章前 4 位优先；否则 opus 判的 gradeBook 归一 4 位 code（落空退粗考点反查）
    if preset_chapter_id and len(preset_chapter_id) >= 4:
        grade_code = preset_chapter_id[:4]
    else:
        grade_code = V._grade_to_code(decision["grade_book"])
        if not grade_code:
            grade_code = await V._resolve_grade_code(analysis)
    if grade_code:
        analysis["grade"]["code"] = grade_code

    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    client = RuoyiClient(token=token)
    leaf_pool: list[tuple[str, str]] = []
    try:
        leaf_pool = await V.leaf_pool_for_grade(
            grade_code, client, include_review_books=include_review_books
        )
    except Exception as e:  # noqa: BLE001 — 池故障 → 空池降级（与 classify 同口径，转 needs_confirm）
        analysis["_anchor_error"] = str(e)

    if not leaf_pool:
        await client.aclose()
        analysis.setdefault("_anchor_error", "知识点叶子池不可用（库未起/年级未识别）")
        _emit_stage("classify", "锚定考点", "warn", "知识点池不可用，待老师确认")
        _emit_stage("knobs", "解析配方", "warn", "待老师确认母题后再定配方")
        # 🔴 M3/PRD-A-018·高置信 + 叶子池空 = 静默死态修复：对齐 not-confirmed picker 分支（769-802）。
        #   旧实现这里只发 mother_confirm 数据、不置 awaiting_mother_confirm → BE 路由态没置 → 下一轮老师
        #   打字补年级/章时 route_entry（awaiting_mother_confirm=False + items 空 + mother_dna 空）落兜底
        #   'ask' → 催「请先贴图」吞掉老师的话、整道母题丢失。修法：① 置 awaiting_mother_confirm=True +
        #   emit need_confirm（FE 弹章 picker），让老师经「确认年级章/文本纠正」前进；② 把已解出的 opus 富文本
        #   /DNA 回填进 mother_dna（即便锚不到叶子，stem/answer/skeleton/dna 仍保留供 confirm 后 classify
        #   走 parse/重锚复用首解，不白丢这次解题）。
        early_dna = dict(state.get("mother_dna") or {})
        _rich = entry.get("richText") or {}
        if isinstance(_rich, dict):
            if _rich.get("stem"):
                early_dna["stem"] = _sanitize_rich_text(_rich.get("stem"))
            if _rich.get("answer"):
                early_dna["answer"] = _sanitize_rich_text(_rich.get("answer"))
            if _rich.get("analysis"):
                early_dna["analysis"] = _sanitize_rich_text(_rich.get("analysis"))
        _early_dna_obj = mother_opus.opus_to_dna(entry)
        _early_skeleton = _early_dna_obj.get("skeleton") or []
        if _early_skeleton:
            early_dna["solution_skeleton"] = join_skeleton(_early_skeleton)
        _early_solved = entry.get("solvedAnswer")
        if _early_solved:
            early_dna["solved_answer"] = _sanitize_rich_text(_early_solved)
        early_dna["dna"] = _early_dna_obj
        early_dna["mother_solve_source"] = "opus"
        # 🔴 R2a·闸2（B5b）首解范围指纹（写点②·高置信空池 early 路）：记首解年级册 4 位 code。
        early_dna["_solve_range_fp"] = str(grade_code or "")
        picker_payload = {
            "grade_book": {"id": grade_code or "", "name": decision.get("grade_book") or ""},
            "chapter": {"id": "", "name": str(decision.get("chapter") or "").strip()},
            "grade_candidates": [{"id": grade_code or "", "name": decision.get("grade_book")}]
                                if decision.get("grade_book") else [],
            "chapter_candidates": [{"id": "", "name": n}
                                   for n in (decision.get("chapter_candidates") or []) if n],
            "confidence": float(decision.get("confidence") or 0.0),
        }
        _emit_need_confirm(picker_payload)
        early: dict[str, Any] = {
            **base_out,
            "analysis": analysis,
            "mother_dna": early_dna,
            "mother_confirmed": False,
            "facts_locked": False,
            "awaiting_mother_confirm": True,   # resume → route_entry → classify 重锚（带确认章），不再死态
            "awaiting_mother_review": False,    # 清 stale review，防「开始举一反三」误路由回 parse
            "messages": [AIMessage(content=(
                "我读出了这道母题，但暂时没能连到题库的知识点章节（库可能没起，或年级没识别出来）。"
                "**请确认母题所属的年级与章**，我再据此锚定考点、出变式（确认无误回复「确认」，"
                "需要修改请直接告诉我正确的年级/章）。"
            ))],
        }
        early["mother_confirm"] = V.build_mother_confirm({**state, **early})
        V._emit_mother_card({**state, **early})  # 母题卡仍先出（卡出了但未定死，等老师定章）
        V._emit_figure_stage({**state, **early})
        return early

    # opus 富文本回填 mother_dna（题面/答案/解析）
    rich = entry.get("richText") or {}
    if isinstance(rich, dict):
        if rich.get("stem"):
            mother_dna["stem"] = _sanitize_rich_text(rich.get("stem"))
        if rich.get("answer"):
            mother_dna["answer"] = _sanitize_rich_text(rich.get("answer"))
        if rich.get("analysis"):
            mother_dna["analysis"] = _sanitize_rich_text(rich.get("analysis"))

    # DNA 归一（mother_opus.opus_to_dna，与 classify 同函数）
    dna = mother_opus.opus_to_dna(entry)
    skeleton_lines = dna.get("skeleton") or []
    if skeleton_lines:
        mother_dna["solution_skeleton"] = join_skeleton(skeleton_lines)  # P8 逐行净化
    solved = entry.get("solvedAnswer")
    if solved:
        mother_dna["solved_answer"] = _sanitize_rich_text(solved)
    mother_dna["mother_solve_source"] = "opus"
    # 🔴 R2a·闸2（B5b）首解范围指纹（写点②·高置信成功路同口径）：记首解年级册 4 位 code。
    mother_dna["_solve_range_fp"] = str(grade_code or "")

    # 🔴 开集 kp 名 → 年级叶子池后锚（高置信路径专属；id 落定后交闸B 校验前缀）
    main_kp_obj = dna.get("main_kp") or {}
    if not (main_kp_obj.get("id") or "").strip() and main_kp_obj.get("name"):
        matched = _match_kp_in_pool(main_kp_obj["name"], leaf_pool)
        if matched:
            dna["main_kp"] = {"id": matched, "name": main_kp_obj["name"]}
    for s in dna.get("secondary_kps") or []:
        if not (s.get("id") or "").strip() and s.get("name"):
            mid = _match_kp_in_pool(s["name"], leaf_pool)
            if mid:
                s["id"] = mid

    # 闸A 富文本机器验证（G10，同 classify）
    rt_check = mother_opus.validate_rich_text(
        rich if isinstance(rich, dict) else {}, has_table=bool(entry.get("has_table")),
    )
    if not rt_check["ok"]:
        analysis["_richtext_issues"] = rt_check["issues"]
        mother_dna["need_richtext_review"] = True
        _emit_stage("classify", "锚定考点", "warn",
                      f"母题富文本机器检发现 {len(rt_check['issues'])} 处问题，待人工复核")

    # 闸B 锚定·宁空不凑（G11，同 classify）：预设章 id（老师选范围）优先收窄前缀；无预设 → 年级册 4 位
    #   code（高置信无确认章路径，章为 opus 判定文本、不收窄前缀，行为字节级不变）。
    chapter_id = preset_chapter_id or grade_code
    dna = mother_opus.anchor_to_chapter(
        dna, chapter_id=chapter_id, leaf_pool=leaf_pool, include_review_books=include_review_books,
    )
    if dna.get("need_anchor_review"):
        mother_dna["need_anchor_review"] = True
    main_kp = dna.get("main_kp") if (dna.get("main_kp") or {}).get("id") else None

    await client.aclose()

    # 🔴 PRD-C-106 B1②③·模型对齐 = 纯代码（消重复解题 + 诚实三态）：opus 带料解题已在 modelCandidates
    #   选了模型名 → anchor_models_from_names 纯代码映射 M-id+tier/freq（**不再调 confirm_models 第二次
    #   LLM 解题**）。真无考模型 → models:[] + model_flag="no_model"（去 M00 兜底，难度走降级）。
    try:
        m_ref = str((main_kp or {}).get("id") or "") or None
        m_res = model_anchor.anchor_models_from_names(
            dna.get("model_candidates") or [],
            grade_code=grade_code,  # 🔴 PRD-C-105 B：按年级全量召回集对齐
        )
        # 池外名留痕待命名池（软警·不打回；与旧 anchor_models 同副作用）。
        for _nm in (m_res.get("model_overflow") or []):
            try:
                model_anchor.record_overflow_candidate(_nm, [], question_ref=m_ref)
            except Exception:  # noqa: BLE001
                pass
    except Exception as e:  # noqa: BLE001 — 对齐整体故障 → 诚实留空 + ⚠（绝不 M00 假装有、不卡死）
        analysis.setdefault("_model_anchor_error", str(e))
        m_res = {"models": [], "model_overflow": [], "model_warn": True,
                 "model_flag": "lookup_unavailable"}
    dna["models"] = m_res.get("models") or []  # 🔴 诚实三态：无模型留空，绝不 M00
    dna["model_overflow"] = m_res.get("model_overflow") or []
    dna["temp_models"] = m_res.get("temp_models") or []
    dna["model_flag"] = m_res.get("model_flag")  # no_model/overflow/lookup_unavailable/None
    if m_res.get("model_warn"):
        dna["model_warn"] = True
    mother_dna["dna"] = dna

    # 锚到真叶子 → 抬三锚置信（同 classify）
    if main_kp and main_kp.get("id"):
        analysis["kp"] = {
            "value": main_kp.get("name"),
            "confidence": max(float(analysis["kp"].get("confidence", 0) or 0), V.CONF_GATE),
            "anchored": {"id": main_kp["id"], "code": str(main_kp["id"]), "name": main_kp.get("name")},
        }
        if grade_code:
            analysis["grade"]["code"] = grade_code
        analysis["grade"]["confidence"] = max(float(analysis["grade"].get("confidence", 0) or 0), V.CONF_GATE)
        if dna.get("qtype"):
            analysis["qtype"] = {"value": dna["qtype"],
                                 "confidence": max(float(analysis["qtype"].get("confidence", 0) or 0), V.CONF_GATE)}

    confirmed = V._conf_ok(analysis) and bool((analysis.get("kp") or {}).get("anchored"))
    kp_name = (analysis.get("kp") or {}).get("value") or "?"
    grade_name = (analysis.get("grade") or {}).get("value") or grade_code or "?"
    # 🔴 R2a·闸1(B5) 隔离修复（用户反馈「选了年级-章节还弹确认弹窗·两个 prompt 要隔离」+「为什么没定死」）：
    #   老师已预设章范围 → 即便 opus 考点没锚到库里真叶子（范围内未命中），也**绝不再弹确认**。
    #   复用 _reanchor_reuse_first_solve 的 graceful 降级（variant.py:2569-2578）：把 main_kp 锚到**所选
    #   章节点本身**（preset_chapter_id，= 老师亲选范围、非凭空造叶子）+ need_anchor_review=True（母题卡
    #   显「锚定待人审」），并写 analysis.kp.anchored.code → 让出题闸（generate 查 dim1_kp_id=anchored.code,
    #   variant.py:3068/3977）放行、母题「定死」可出题。老师后续仍可经 DNA「改考点」修正到真叶子。
    #   只在「考点未锚到叶子」这一种 not-confirmed 上放行；叶子池整体不可用（上面 881 早退）不在此列。
    # 🔴 用户反馈（2026-06-22）「选了年级章节还弹窗（因为系统里面默认选择了）」：preset 必须**无条件**
    #   隔离确认闸——即便 opus 这轮解题崩坏、primaryKp 名都没解出（_pkp 空），也不再弹。退而求其次用
    #   预设章自身名兜底（chapter 文本 / 预设章名），绝不因 _pkp 空回退到弹窗（那等于没隔离）。
    if not confirmed and preset_chapter_id:
        _pkp = (
            (dna.get("main_kp") or {}).get("name")
            or main_kp_obj.get("name")
            or str(decision.get("chapter") or "").strip()
            or "（待人审锚定考点）"
        )
        dna["main_kp"] = {"id": preset_chapter_id, "name": _pkp}  # 章级锚定（老师亲选章，非造叶子）
        dna["need_anchor_review"] = True
        mother_dna["need_anchor_review"] = True
        mother_dna["dna"] = dna  # 回写（dna 同引用，显式确保下游 dim1_kp_id 取到章级锚）
        analysis["kp"] = {
            "value": _pkp,
            "confidence": max(float((analysis.get("kp") or {}).get("confidence", 0) or 0), V.CONF_GATE),
            "anchored": {"id": preset_chapter_id, "code": str(preset_chapter_id), "name": _pkp},
        }
        confirmed = True
        kp_name = _pkp
    _emit_stage("classify", "锚定考点", "done" if confirmed else "warn",
                  f"考点「{kp_name}」·年级「{grade_name}」")
    recipe = V.knobs_desc(base_out.get("knobs")) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）"
    _emit_stage("knobs", "解析配方", "done" if confirmed else "warn", recipe)

    # 🔴 PRD-C-100 B1·锚不到叶子 → 弹真章树 picker，不走 clarify 死胡同：
    #   高置信路径若主考点 opus 给的是开集名、_match_kp_in_pool/闸B 都锚不到年级章内真叶子
    #   （need_anchor_review）→ confirmed=False。旧实现此时仍走 awaiting_mother_confirm=False →
    #   after_mother_entry→gate_after_classify 返 clarify、不置 awaiting_mother_review →「开始举一反三」
    #   静默回 parse、永不出变式（HANDOFF 待校准 #3 的真实后果）。
    #   对齐 D1（置信低/章歧义→弹窗确认年级章）：confirmed 不成立 → 导向 needs_confirm（弹真章树
    #   picker，await 老师确认章），老师选定章 → route_entry 见 confirmed_chapter_id → classify 按
    #   确认章重锚（B2 自愈网兜底）→ 正常出变式。**绝不为出变式强凑错章**（闸B 宁空不凑精神保留），
    #   是「问老师定章」不是「瞎锚」。母题卡照出（卡先出不变），只是把死胡同换成可前进的确认面。
    if not confirmed:
        decision_l = dict(decision)
        decision_l["grade_book"] = decision.get("grade_book") or grade_name or ""
        chap_text = (analysis.get("chapter") or {}).get("value") if isinstance(
            analysis.get("chapter"), dict
        ) else None
        picker_payload = {
            "grade_book": {"id": grade_code or "", "name": decision_l["grade_book"]},
            "chapter": {"id": "", "name": chap_text or decision.get("chapter") or ""},
            "grade_candidates": [{"id": grade_code or "", "name": decision_l["grade_book"]}]
                                if decision_l["grade_book"] else [],
            "chapter_candidates": [{"id": "", "name": n}
                                   for n in (decision.get("chapter_candidates") or []) if n],
            "confidence": float(decision.get("confidence") or 0.0),
        }
        _emit_need_confirm(picker_payload)
        out_nc: dict[str, Any] = {
            **base_out,
            "analysis": analysis,
            "mother_dna": mother_dna,
            "mother_confirmed": False,
            "facts_locked": False,
            "awaiting_mother_confirm": True,   # resume → route_entry → classify 重锚（带确认章）
            "awaiting_mother_review": False,    # 清 stale review，防「开始举一反三」误路由回 parse
            "messages": [AIMessage(content=(
                f"母题考点「{kp_name}」我没能锚到题库里的具体章节叶子（{grade_name} 范围内未命中）。"
                "为避免锚错章串题，**请确认母题所属的年级与章**，我再据此重新锚定考点、出变式（"
                "确认无误回复「确认」，需要修改请直接告诉我正确的年级/章）。"
            ))],
        }
        out_nc["mother_confirm"] = V.build_mother_confirm({**state, **out_nc})
        V._emit_mother_card({**state, **out_nc})  # 母题卡仍先出（卡出了但未定死，等老师定章）
        V._emit_figure_stage({**state, **out_nc})  # 🔴 BUG-02：「母题切图」节点据 mother_has_figure 发 done
        return out_nc

    out: dict[str, Any] = {
        **base_out,
        "analysis": analysis,
        "mother_dna": mother_dna,
        "mother_confirmed": bool(confirmed),
        "facts_locked": bool(confirmed),
        "awaiting_mother_confirm": False,
        "messages": [],
    }
    out["mother_confirm"] = V.build_mother_confirm({**state, **out})
    V._emit_mother_card({**state, **out})  # 母题卡先出（早于变式）
    V._emit_figure_stage({**state, **out})  # 🔴 BUG-02：「母题切图」节点据 mother_has_figure 发 done
    return out
