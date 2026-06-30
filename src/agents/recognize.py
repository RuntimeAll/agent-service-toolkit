# -*- coding: utf-8 -*-
r"""PRD-A-002 路A · 录题「框选识别」无状态端点核心。

🔴 心智：老师在一张题/卷照片上手动框出一道题区，把框区图丢给 opus 多模态，一次产出
   「去手写的印刷体原题富文本」(+ 可选解题 + 10 维 DNA 打标)。本模块全是纯函数 + 一次
   LLM 调用包装，无状态（不读会话、不读 DB、不落库）——落库走 book-server /teacher/ingest/**。

设计铁则（对应 PRD-A-002 全局铁则）：
  - R3 零外部 OCR：OCR 由 opus 多模态自做（本模块），不接 TextIn/YOLO。
  - R2 人手框：本模块只识别「老师框好的单个题区图」，不做自动切图/划区。
  - R1 解题=自动打标：solve=true 时一次产出 答案/解析 + 10 维 DNA（不设独立打标开关）。
  - R5 答案只信 sympy：solve 后用 verify_one_stem 程序验算，verdict 透传，不臆造。
  - R4 模型锁 opus：走 _ainvoke_text（relay 池 claude-opus-4-8 + 熔断备站）。

🔴 prompt 通用（记忆 feedback_prompt_general_no_noise）：角色+输入规范+输出规范(系统格式)，
   不塞教材版本/母题/单题型噪音。复用 mother_opus 的纯归一/校验 helper，但**不复用其母题特化
   prompt**（那带 leaf_pool 锚定 + 浙教版红线，对通用录题是噪音）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract, mother_opus
from core.settings import settings

# opus 读图调用上限（H4：带图可达数百秒；录题单题区图通常远小于母题整卷）。
RECOGNIZE_TIMEOUT_S = 180.0
RECOGNIZE_TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# 通用 prompt（角色 + 输入规范 + 输出规范）—— solve 开关条件拼解题/DNA 段
# ---------------------------------------------------------------------------
def build_recognize_prompt(*, solve: bool, grade_hint: str | None = None) -> str:
    """组录题识别 prompt（通用，无单题型噪音）。

    solve=False → 纯识别（去手写）→ 只产 has_figure/stem/qtype/options。
    solve=True  → 识别 + 真解题 + 10 维 DNA 打标（R1）。
    grade_hint：可选学段提示（如「七年级」），只用于约束解法不超纲，缺省不约束。
    """
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    grade_line = (
        f"学段提示：{grade_hint}（解法不超出本学段进度）。" if grade_hint else ""
    )

    base_rules = """你是数学题目识别专家。这是老师在试卷/教辅上**框出的一道题区照片**，把其中【印刷体原题】转成干净富文本。
🔴 识别铁则：
- 只保留印刷体原题；手写笔迹、铅笔作答、批改符号(对勾/叉/红笔)、涂改一律**去除**，绝不进题干。
- 数学式一律用行内 $...$；换行用 \\n；LaTeX 命令参数配对完整；禁 \\(..\\) / \\[..\\] 定界（只用 $）。
- 题面含表格 → 用 <table> 还原结构，不拍平成一行。
- 选择题：把选项逐项放进 options 数组（每项**不含** "A." 前缀），题干 stem **不重复**选项文本；非选择题 options 留空数组。
- has_figure：题面真含几何图/函数图/图表填 true，纯文字题填 false（本端点不切图，仅标记）。
- need_grading：框区内若含**学生手写作答 / 铅笔笔迹 / 批改痕迹（对勾叉、红笔）**填 true（该题可批改），纯印刷体无作答填 false。
- 看不清/框内无完整印刷体题 → stem 留空串、has_figure=false（下游会兜底提示重框，绝不编造题目）。"""

    if not solve:
        output = """================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{
  "has_figure": true/false,
  "need_grading": true/false,
  "stem": "题干(Markdown+行内$LaTeX$，已去手写)",
  "qtype": "选择/填空/解答",
  "options": ["选项A正文", "选项B正文"]
}"""
        return f"{base_rules}\n{grade_line}\n\n{output}"

    solve_rules = f"""
🔴 解题纪律（solve）：像严谨解题者一样**真正把题解出来**（一步步算到最终答案，不抄图、不跳步、不臆造）；
   信任自己的正确推导，不为凑预设答案来回改；与直觉不符最多复核一遍。据你的解答做 10 维 DNA 打标。
- assessmentType 考察类型：**闭集10选1** = {exam_types}。
- solutionSkeleton 解法骨架：解题步骤序列，最难一步用【】整步包住（至多一处）。
- hardPointCount **必须等于** breakthroughPoints 数组长度（基础/套公式题→空数组、计 0）。
- difficulty 难度 1~4：1 送分 / 2 常规多步 / 3 含1难点或证明探究 / 4 压轴多难点。
- tags 检索标签 3~6（禁近义增生）；primaryKp 主考点只给 name（不强求 id）；secondaryKps 0~3 个。
- modelCandidates 解题模型候选名（真有可复用套路才给，简单题空数组）。"""

    output = f"""================ 输出（只输出一个 JSON，无解释、无 markdown fence） ================
{{
  "has_figure": true/false,
  "need_grading": true/false,
  "stem": "题干(Markdown+行内$LaTeX$，已去手写)",
  "qtype": "选择/填空/解答",
  "options": ["选项A正文", "选项B正文"],
  "answer": "标准答案",
  "analysis": "解析(干净最终解法，不写草稿/试错)",
  "solvedAnswer": "你一步步解出的最终答案(简短)",
  "dna": {{
    "primaryKp": {{"name": "主考点名"}},
    "secondaryKps": [{{"name": "副考点名"}}],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["检索标签"],
    "modelCandidates": []
  }}
}}"""
    return f"{base_rules}\n{solve_rules}\n{grade_line}\n\n{output}"


# ---------------------------------------------------------------------------
# JSON 兜底解析（剥 markdown fence + 抓首个完整 {...}）
# ---------------------------------------------------------------------------
def _repair_invalid_escapes(s: str) -> str:
    r"""修非法 JSON 转义：JSON 字符串里只允许 \" \\ \/ \b \f \n \r \t \uXXXX。

    🔴 LLM 输出数学题常带 LaTeX（$a \parallel b$、\angle、\frac、\triangle…），其中 \p \a \f...
    多数不是合法 JSON 转义 → json.loads 直接报 "Invalid \escape" / "Expecting ',' delimiter"
    （实测整卷拆题踩）。把**非法的单反斜杠** \X 翻倍成 \\X（X 不是合法转义引导符时），
    使 LaTeX 命令在 JSON 里成为字面反斜杠，解析通过、内容不丢。合法转义（\n \" \\ \uXXXX）不动。
    """
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s)


def _loads_lax(s: str) -> Any:
    """json.loads 容错版：先原样，失败再修非法转义重试。"""
    try:
        return json.loads(s)
    except Exception:
        return json.loads(_repair_invalid_escapes(s))


def parse_json_lax(text: str) -> dict[str, Any]:
    """从 LLM 文本里抠出 JSON 对象（容错 fence / 前后噪音 / LaTeX 非法转义）。失败抛 ValueError。"""
    s = (text or "").strip()
    # 去 ```json ... ``` fence
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        return _loads_lax(s)
    except Exception:
        pass
    # 抓第一个 { 到与之配对的 }（栈匹配，跳过字符串内花括号）
    start = s.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return _loads_lax(s[start : i + 1])
    raise ValueError("unbalanced JSON braces")


def _normalize_image_ref(image_url: str | None, image_base64: str | None) -> str:
    """归一图片入参 → LLM image_url 可吃的 url（https 直传 / 裸 base64 包 data uri）。"""
    if image_url:
        return image_url.strip()
    b64 = (image_base64 or "").strip()
    if not b64:
        raise ValueError("image_url / image_base64 至少给一个")
    if b64.startswith("data:"):
        return b64
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# 主入口：一次 opus 多模态调用 + (solve 时) sympy 验算 + DNA 归一
# ---------------------------------------------------------------------------
async def recognize(
    *,
    image_url: str | None = None,
    image_base64: str | None = None,
    solve: bool = False,
    grade_hint: str | None = None,
    invoke: Any,
) -> dict[str, Any]:
    """识别单个框区题图 → 富文本题 (+ 可选解题/DNA/验算)。无状态。

    invoke = variant._ainvoke_text（落 trace + relay 池熔断）。返回前端可直渲的结构：
      {ok, has_figure, stem, qtype, options, answer, analysis, solved_answer,
       dna, verify, richtext_issues, error}
    solve=False 时 answer/analysis/dna/verify 为空/None。永不抛到端点外（异常收口为 ok=False）。
    """
    img = _normalize_image_ref(image_url, image_base64)
    prompt = build_recognize_prompt(solve=solve, grade_hint=grade_hint)
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": img}},
    ])

    raw = await invoke(
        [msg],
        model=settings.LLM_MODEL_HEAVY,
        temperature=RECOGNIZE_TEMPERATURE,
        timeout=RECOGNIZE_TIMEOUT_S,
    )
    data = parse_json_lax(raw)

    stem = str(data.get("stem") or "").strip()
    has_figure = bool(data.get("has_figure"))
    need_grading = bool(data.get("need_grading"))
    qtype = dna_extract._norm_qtype(data.get("qtype"))
    options = [str(o).strip() for o in (data.get("options") or []) if str(o).strip()]

    out: dict[str, Any] = {
        "ok": bool(stem),
        "has_figure": has_figure,
        "need_grading": need_grading,
        "stem": stem,
        "qtype": qtype,
        "options": options,
        "answer": "",
        "analysis": "",
        "solved_answer": "",
        "dna": None,
        "verify": None,
        "richtext_issues": [],
        "error": None if stem else "未识别到完整印刷体题目（请重框或手动输入）",
    }

    if not solve:
        return out

    # ---- solve 段：答案/解析 + DNA 归一(R1) + sympy 验算(R5) ----
    answer = str(data.get("answer") or "").strip()
    analysis = str(data.get("analysis") or "").strip()
    solved_answer = str(data.get("solvedAnswer") or "").strip()
    out["answer"] = answer
    out["analysis"] = analysis
    out["solved_answer"] = solved_answer

    # DNA 归一（复用 mother_opus.opus_to_dna：克制重算 hardPointCount/clamp 难度/闭集归一）
    try:
        out["dna"] = mother_opus.opus_to_dna(data)
    except Exception as e:  # noqa: BLE001
        out["dna"] = None
        out["error"] = f"DNA 归一异常: {str(e)[:80]}"

    # 闸A 富文本机器校验（坏 LaTeX/缺表 → issues，标问题不拦死）
    try:
        rv = mother_opus.validate_rich_text(
            {"stem": stem, "answer": answer, "analysis": analysis}
        )
        out["richtext_issues"] = rv.get("issues") or []
    except Exception:  # noqa: BLE001
        out["richtext_issues"] = []

    # R5 sympy 验算（verdict 透传：pass 标答自洽 / fail 标答错 / degrade 转人工，永不臆造）
    if answer:
        try:
            from agents.variant import verify_one_stem
            out["verify"] = await verify_one_stem(
                stem, answer, qtype=qtype, options=options or None
            )
        except Exception as e:  # noqa: BLE001
            out["verify"] = {"verdict": "degrade", "detail": f"验算异常: {str(e)[:80]}", "computed": None}

    return out
