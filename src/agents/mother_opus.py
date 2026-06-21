# -*- coding: utf-8 -*-
r"""PRD-C-017 B1 · 母题 opus 合并「解题 + 10 维 DNA 打标」+ 闸A（富文本机器验证）+ 闸B（锚定宁空不凑）。

🔴 心智：母题是「一对多放大器」，母题歪则所有变式全歪。本模块把母题前段从「nano 出题前打标 +
   抄图骨架」改成「opus(claude-opus-4-8) 多模态直读母题原图、一次合并产出 题面富文本 + 解答 +
   10 维 DNA」。母题侧零机器验证（决策表 06-15 去 sympy）→ opus 是唯一安全网，故：
   - 调用走 settings.variant_model("mother_solve_label")（启动期 fail-fast 已锁 opus，见 settings）；
   - response_format json_schema 硬锁 10 维（F3 实测中转支持）；低温 0.1（M9）；超时 ≤180s（H4）。
   - 失败/超时 → 上层走 SSE error，绝不静默退 gpt-5.4。

本模块全是纯函数 + 一个 LLM 调用包装（invoke 注入，便于单测桩）：
  - MOTHER_SCHEMA / build_mother_prompt：schema + PREFIX 三注入 prompt（事实源=B0 探针 + §一规则）。
  - solve_and_label：opus 一次合并调用（多模态）。
  - opus_to_dna：opus 输出 → 内部 DNA 契约 v1（与 dna_extract 产物同形）。
  - validate_rich_text（闸A）：富文本机器逐项检（LaTeX 配对/缺参/孤立 \ / $·{} 不配对/有表必 table）。
  - anchor_to_chapter（闸B）：opus 主考点锚到「确认章 id 前缀内的叶子」；锚不到 → 留空 +
    need_anchor_review=1（宁空不凑，绝不锚粗/造叶子/回退全量池硬塞）。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from agents import dna_extract

# opus 母题调用上限（H4：纯文本 ~21s，带图可达数百秒；带图本卡 G13 打回不到此处）。
MOTHER_OPUS_TIMEOUT_S = 180.0
MOTHER_OPUS_TEMPERATURE = 0.1

# 10 维 DNA json_schema（B0 探针 MOTHER_SCHEMA，required 锁字段 = 不偷工不漏维硬保证）。
MOTHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "has_figure": {"type": "boolean", "description": "题面是否含图形/图表/几何图（拍照纯文本题=false）"},
        "richText": {
            "type": "object",
            "properties": {
                "stem": {"type": "string"},
                "answer": {"type": "string"},
                "analysis": {"type": "string"},
            },
            "required": ["stem", "answer", "analysis"],
        },
        "solvedAnswer": {"type": "string", "description": "opus 真解出的最终答案"},
        "dna": {
            "type": "object",
            "properties": {
                "primaryKp": {"type": "object", "properties": {
                    "id": {"type": "string"}, "name": {"type": "string"}}},
                "secondaryKps": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "name": {"type": "string"}}}},
                "qtype": {"type": "string"},
                "assessmentType": {"type": "string"},
                "solutionSkeleton": {"type": "array", "items": {"type": "string"}},
                "hardPointCount": {"type": "integer"},
                "breakthroughPoints": {"type": "array", "items": {"type": "string"}},
                "scenario": {"type": "string"},
                "difficulty": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "modelCandidates": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "primaryKp", "secondaryKps", "qtype", "assessmentType",
                "solutionSkeleton", "hardPointCount", "breakthroughPoints",
                "scenario", "difficulty", "tags", "modelCandidates",
            ],
        },
    },
    "required": ["has_figure", "richText", "solvedAnswer", "dna"],
}

# response_format 封装（中转 OpenAI-compatible json_schema）。
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "mother_label", "schema": MOTHER_SCHEMA, "strict": False},
}


# ---------------------------------------------------------------------------
# PREFIX 三注入 + 逐维规则 prompt（§一规则源：label_spec §3 十维 + §4 难度四档 + §1 学段锁死）
# ---------------------------------------------------------------------------
def build_mother_prompt(
    *,
    grade_text: str,
    chapter_text: str | None,
    leaf_pool: list[tuple[str, str]],
    model_vocab: list[str] | None = None,
) -> str:
    """组母题 opus 合并解题+打标 prompt（PREFIX 三注入 + 逐维规则）。

    ① 规则书（10 维逐维 + 难度四档 rubric + 难点克制 + 考察类型闭集10 + 标签禁近义）内联；
    ② 知识点叶子池（确认年级/章范围内，主/副 kp 只能锚池内 id、禁造词、越界 needReview）；
    ③ 模型词库快照（只读命名参考，简单题空数组）；+ 学段锁死红线（解法不超本年级进度）。

    leaf_pool 走 toolkit 既有 lazyTree 取数（调用方备好，本函数只渲染），不新写 pymysql。
    """
    def _row(pid: Any, name: Any) -> str:
        book = dna_extract._book_of(pid)
        return f"{pid} {name}（册：{book}）" if book else f"{pid} {name}"

    from core.settings import settings as _settings
    _sentinel = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False)
    kp_pool_text = "\n".join(_row(pid, name) for pid, name in leaf_pool) or "（空）"
    vocab_text = "、".join(model_vocab or []) or "（无快照，简单题模型候选留空数组）"
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    chapter_line = f"确认章：{chapter_text}" if chapter_text else "确认章：（未细化到章，按年级册范围）"

    # 🔴 R2b·U8 输出段两版：哨兵框（richText 三段走 ⟦STEM⟧/⟦ANSWER⟧/⟦ANALYSIS⟧ 原文，绕 JSON 转义）
    #   / 旧式整 JSON。按 settings.MOTHER_RICHTEXT_SENTINEL 选；解析侧 parse_or_repair_entry 通吃两版。
    if _sentinel:
        output_block = """================ 输出（先一个 JSON，紧跟三个哨兵框；不要解释、不要 markdown fence） ================
题面/答案/解析含 $LaTeX$/换行/<table> 天然带 JSON 元字符，**别塞进 JSON 字符串**（一处转义坏整对象断），改用哨兵框**原文**输出（框内随便写 LaTeX/换行/HTML，不转义）。先输出 JSON（richText 三段留空串占位）：
{
  "has_figure": true/false,
  "richText": {"stem": "", "answer": "", "analysis": ""},
  "solvedAnswer": "你一步步解出的最终答案(简短)",
  "dna": {
    "primaryKp": {"id": "池内叶子id 或 空串", "name": "考点名"},
    "secondaryKps": [{"id": "池内id", "name": "名"}],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2(最难一步用【】整步包住)"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["3~6个检索标签,禁近义增生"],
    "modelCandidates": []
  }
}
紧接着 JSON **之后**输出三个哨兵框：
⟦STEM⟧
（题干原文：Markdown+行内$LaTeX$+必要时<table>，直接换行，无需转义）
⟦/STEM⟧
⟦ANSWER⟧
（标准答案原文）
⟦/ANSWER⟧
⟦ANALYSIS⟧
（解析原文：最终干净解法，不写草稿/试错）
⟦/ANALYSIS⟧
🔴 三段各自独立成框、缺一不可；框标签照抄、不改写。"""
    else:
        output_block = """================ 输出 ================
只输出一个 JSON（不要解释、不要 markdown fence），结构：
{
  "has_figure": true/false,
  "richText": {"stem": "题干(Markdown+行内$LaTeX$)", "answer": "标准答案", "analysis": "解析(含解题过程)"},
  "solvedAnswer": "你一步步解出的最终答案",
  "dna": {
    "primaryKp": {"id": "池内叶子id 或 空串", "name": "考点名"},
    "secondaryKps": [{"id": "池内id", "name": "名"}],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2(最难一步用【】整步包住)"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["3~6个检索标签,禁近义增生"],
    "modelCandidates": []
  }
}"""

    return f"""你是浙教版初中数学命题专家 + 题库打标师。看这张母题图，**先真正把题解出来**（一步步算到最终答案，不许抄图、不许跳步），**再据你的解答**做 10 维 DNA 打标。母题是所有变式的基准，解错则全部变式跟着错——务必稳准。

================ 学段锁死红线 ================
本题年级：{grade_text or "未知年级"}。{chapter_line}
🔴 解法不得超出本年级进度（不许用更高年级才学的定理/方法绕过）；考点必须落在本年级/章范围内。

================ 知识点叶子池（闭集·只选不造） ================
🔴 primaryKp / secondaryKps 的 id **只能从下面池里选**（禁造词）；池内确实没有贴切叶子 → primaryKp 留 {{"id":"","name":"真实考点名"}}（id 空、name 如实写），由下游标 needReview，**绝不锚粗、绝不造叶子、绝不硬塞最近的**。
{kp_pool_text}

================ 模型词库快照（只读命名参考） ================
{vocab_text}

================ 10 维逐维规则 ================
1. primaryKp 主考点：锚池内叶子（id+name 单值）；锚不到留 id 空 + 真实考点名。
2. secondaryKps 副考点 0~3：与主不同体系才算、锚池内 id；没有就空数组。
3. qtype 题型：选择/填空/解答 之一（闭集）。
4. assessmentType 考察类型：**闭集10选1** = {exam_types}。
5. solutionSkeleton 解法骨架：解题步骤序列；**最难的那一步用【】整步包住**（至多一处，如 "【构造全等三角形】"）。这是变式守恒基因，必须如实反映你解题真实路径。
6. hardPointCount + breakthroughPoints 难点（克制·宁空不凑）：基础/纯套公式/直接计算/概念辨析/送分题 → breakthroughPoints **必空**、hardPointCount=0；🔴 hardPointCount **必须等于** breakthroughPoints 数组长度（不许自报数）。
7. scenario 场景：一句话场景描述 或 "纯代数"。
8. difficulty 难度四档 rubric（**按构造断言，不裸问几星**）：
   - 1 ★（送分）：无难点 + (概念辨析 或 单步)。
   - 2 ★★（常规）：常规无难点 + 考察∈{{直接计算·公式套用·性质判定}} + 多步。
   - 3 ★★★：1 难点 或 考察∈{{证明推理·应用建模·探究归纳}} 或 骨架含【最难步】。
   - 4 ★（压轴）：≥2 难点 或 多突破口综合。
9. tags 标签 3~6：检索标签（求什么/用什么定理/什么方法/什么场景）；**禁近义增生**（同义只留一个）。
10. modelCandidates 解题模型（克制）：真有可复用套路才给候选名（简单题空数组）；只吐候选名不给 M-id。

================ 富文本红线（题面/答案/解析） ================
🔴 数学式用行内 $...$；换行用标准 \\n；禁裸 LaTeX 命令、禁 \\( \\) / \\[ \\] 定界；LaTeX 括号/命令参数必须配对完整（下游有机器闸逐项检，坏 LaTeX 会被打回）。
🔴 has_figure：题面真含图形/图表/几何图填 true；只是拍照的纯文本题填 false。

{output_block}"""


# ---------------------------------------------------------------------------
# opus 合并调用（多模态：image_url + response_format + 低温 + 超时）
# ---------------------------------------------------------------------------
async def solve_and_label(
    *,
    image_url: str,
    prompt: str,
    invoke: Any,
    model: str,
    max_tokens: int | None = None,
) -> str:
    """opus 一次多模态合并调用（解题 + 10 维打标）。返回原始文本（调用方 parse）。

    🔴 invoke = variant._ainvoke_text（落 trace/conv_trace + 走 relay 池熔断转移）。
       传 model（mother_solve_label=opus）+ 低温 0.1 + response_format(10维schema) + timeout≤180s。
    🔴 不吞异常：超时/失败由调用方接住走 SSE error（母题侧零机器验证，绝不静默退 gpt-5.4）。
    """
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    # 🔴 R2b·U8：哨兵模式（build_mother_prompt 已含哨兵框指令）→ 不下发 response_format
    #   （json_schema 拒尾随哨兵文本）；关时维持旧式整 JSON + response_format 硬锁（字节级不变）。
    from core.settings import settings as _settings
    _sentinel = getattr(_settings, "MOTHER_RICHTEXT_SENTINEL", False)
    kw: dict[str, Any] = dict(
        model=model,
        temperature=MOTHER_OPUS_TEMPERATURE,
        timeout=MOTHER_OPUS_TIMEOUT_S,
    )
    if not _sentinel:
        kw["response_format"] = RESPONSE_FORMAT
    if max_tokens and max_tokens > 0:
        kw["max_tokens"] = max_tokens
    return await invoke([msg], **kw)


# ---------------------------------------------------------------------------
# opus 输出 → 内部 DNA 契约 v1（与 dna_extract 产物同形，供 _mother_facts / 守恒注入复用）
# ---------------------------------------------------------------------------
def _kp_obj(raw: Any) -> dict[str, str] | None:
    """opus 的 kp 项（{id,name}）→ 归一 dict；id/name 全空 → None。"""
    if not isinstance(raw, dict):
        # 兼容裸字符串（opus 偶尔吐 name 串）
        name = str(raw or "").strip()
        return {"id": "", "name": name} if name else None
    kid = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not kid and not name:
        return None
    return {"id": kid, "name": name}


def opus_to_dna(opus: dict[str, Any]) -> dict[str, Any]:
    """opus 合并输出 → 内部 DNA 契约 v1（键与 dna_extract.empty_dna / _validate 同形）。

    🔴 锚定/闭集校验留给 anchor_to_chapter（闸B）+ 后续 model_anchor；本函数只做形态归一 +
       克制类机器重算（hard_point_count = len(breakthroughPoints) 不信自报、难度 clamp、
       qtype/exam_type 闭集归一、tags 截断）。primaryKp 原样带过（含 opus 选的 id，闸B 再校验前缀）。
    """
    dna_raw = opus.get("dna") or {}
    if not isinstance(dna_raw, dict):
        dna_raw = {}

    main_kp = _kp_obj(dna_raw.get("primaryKp"))
    secondary_kps: list[dict[str, str]] = []
    for s in dna_raw.get("secondaryKps") or []:
        obj = _kp_obj(s)
        if obj and obj.get("id"):  # 副 kp 需 id 才算（与 dna_extract 同口径）
            secondary_kps.append(obj)
        if len(secondary_kps) >= dna_extract.SECONDARY_KP_MAX:
            break

    qtype = dna_extract._norm_qtype(dna_raw.get("qtype"))

    exam_type = dna_raw.get("assessmentType")
    if exam_type not in dna_extract.EXAM_TYPES:
        exam_type = None

    skeleton = [str(x) for x in (dna_raw.get("solutionSkeleton") or []) if str(x).strip()]
    hard_points = [str(x) for x in (dna_raw.get("breakthroughPoints") or []) if str(x).strip()]
    tags = [str(t).strip() for t in (dna_raw.get("tags") or []) if str(t).strip()][
        : dna_extract.TAGS_MAX
    ]
    scene = dna_extract._truncate_scene(dna_raw.get("scenario"))
    difficulty = dna_extract._clamp_difficulty(dna_raw.get("difficulty"))
    if difficulty is None:
        difficulty = dna_extract.DIFFICULTY_FALLBACK
    model_candidates = [str(m).strip() for m in (dna_raw.get("modelCandidates") or []) if str(m).strip()]

    return {
        "main_kp": main_kp,
        "secondary_kps": secondary_kps,
        "qtype": qtype,
        "exam_type": exam_type,
        "skeleton": skeleton,
        "hard_points": hard_points,
        "hard_point_count": len(hard_points),  # 代码重算，不信 opus 自报
        "tags": tags,
        "scene": scene,
        "difficulty": difficulty,
        "model_candidates": model_candidates,  # 候选名（model_anchor 锚正式 M-id 用）
        "flags": [],
    }


# ---------------------------------------------------------------------------
# 闸A · 富文本机器验证（G10，非 LLM）
# ---------------------------------------------------------------------------
_LATEX_PAREN_CMD = re.compile(r"\\(frac|sqrt|binom|overline|underline|vec|hat|dot)\b")


def _try_katex_render(s: str) -> str | None:
    """KaTeX 试渲染（若装了 katex/python wrapper）；未装 → None（跳过该项，不误报）。"""
    try:
        import katex  # type: ignore
    except Exception:  # noqa: BLE001 — 环境无 katex → 不做此项（其余机器检仍生效）
        return None
    # 抽行内 $...$ 段逐个试渲染
    for m in re.finditer(r"\$([^$]+)\$", s):
        expr = m.group(1)
        try:
            katex.render(expr)  # type: ignore
        except Exception as e:  # noqa: BLE001
            return f"KaTeX 渲染失败：{expr[:40]}（{str(e)[:60]}）"
    return None


def _check_one_field(name: str, s: str, *, expect_table: bool = False) -> list[str]:
    issues: list[str] = []
    if not s:
        return issues

    # ① $ 配对（行内/块级数学定界）
    if s.count("$") % 2 != 0:
        issues.append(f"{name}：$ 定界符不配对（奇数个）")

    # ② {} 配对
    if s.count("{") != s.count("}"):
        issues.append(f"{name}：花括号 {{}} 不配对")

    # ③ \frac / \sqrt 等命令缺参（命令后未跟 { ）
    for m in _LATEX_PAREN_CMD.finditer(s):
        tail = s[m.end():].lstrip()
        if not tail.startswith("{") and not (m.group(1) == "sqrt" and tail.startswith("[")):
            issues.append(f"{name}：\\{m.group(1)} 缺参数（命令后未跟 {{}}）")

    # ④ 孤立 \（反斜杠后不是字母/已知 LaTeX 转义/定界，疑似裸残留）。
    #    放行：字母命令（\frac 等）、定界/转义（\$ \% \& \# \_ \{ \} \[ \] \( \)）、
    #    LaTeX 间距/换行命令（\, \; \! \: \> \  \\）—— 这些是合法 LaTeX，误报会把好题挡死。
    for m in re.finditer(r"\\(?![a-zA-Z\\$%&#_{}\[\]() ,;!:>])", s):
        nxt = s[m.end():m.end() + 1]
        if nxt not in ("\n", ""):
            issues.append(f"{name}：孤立反斜杠 \\（后跟「{nxt}」）")
            break  # 报一处即可

    # ⑤ 残留 \( \) / \[ \] 定界（前端 katex 只认 $，B1 prompt 已禁，机器兜底）
    if "\\(" in s or "\\)" in s or "\\[" in s or "\\]" in s:
        issues.append(f"{name}：残留 \\(...\\) / \\[...\\] 定界（应为 $...$）")

    # ⑥ KaTeX 试渲染（装了才查）
    katex_err = _try_katex_render(s)
    if katex_err:
        issues.append(f"{name}：{katex_err}")

    # ⑦ 原图有表则富文本须有 <table> 不拍平
    if expect_table and "<table" not in s.lower():
        issues.append(f"{name}：原图含表格但富文本无 <table>（被拍平）")

    return issues


def validate_rich_text(
    rich: dict[str, Any] | None, *, has_table: bool = False
) -> dict[str, Any]:
    """闸A：母题富文本机器逐项检（非 LLM）。返回 {ok:bool, issues:[...]}。

    检 stem/answer/analysis 三段：LaTeX $ 配对 / {} 配对 / \\frac·\\sqrt 缺参 / 孤立 \\ /
    残留 \\(..\\) 定界 / KaTeX 试渲染（装了才查）/ 原图有表则须有 <table> 不拍平。
    🔴 坏 LaTeX/缺表 → ok=False + issues（标问题，不直接放行）；调用方据此标 need_richtext_review。
    纯函数（除可选 katex），可单测。
    """
    rich = rich or {}
    issues: list[str] = []
    # 表格只需在题面(stem)出现一次即可（答案/解析未必复述表）
    issues += _check_one_field("题面", str(rich.get("stem") or ""), expect_table=has_table)
    issues += _check_one_field("答案", str(rich.get("answer") or ""))
    issues += _check_one_field("解析", str(rich.get("analysis") or ""))
    return {"ok": not issues, "issues": issues}


# ---------------------------------------------------------------------------
# 闸B · 锚定·宁空不凑（G11）
# ---------------------------------------------------------------------------
def anchor_to_chapter(
    dna: dict[str, Any],
    *,
    chapter_id: str | None,
    leaf_pool: list[tuple[str, str]],
    include_review_books: bool = False,
) -> dict[str, Any]:
    """闸B：把 opus 主考点（main_kp）锚到「确认章 id 前缀内的叶子」。

    规则（宁空不凑）：
      - chapter_id 给定 → 主 kp 的 id 必须 in leaf_pool **且** 以 chapter_id 为前缀 → 锚上；
      - 否则（id 空/不在池/越界前缀/复习册未开放）→ main_kp.id 留空 + need_anchor_review=1，
        **保留 opus 给的考点 name**（开集如实捕获真实考点），绝不锚粗/造叶子/回退全量池硬塞。
      - 副 kp 同理逐项校验（越界丢弃该项，不报错）。

    返回新 dna（浅拷贝改 main_kp/secondary_kps + need_anchor_review + flags），纯函数可单测。
    """
    out = dict(dna)
    pool_ids = {str(pid) for pid, _ in leaf_pool}
    chap = str(chapter_id or "").strip()
    flags = list(out.get("flags") or [])

    def _anchored_ok(kid: str) -> bool:
        if not kid or kid not in pool_ids:
            return False
        if not include_review_books and dna_extract._is_review_id(kid):
            return False
        if chap and not kid.startswith(chap):
            return False
        return True

    mk = out.get("main_kp") or {}
    mk_id = str(mk.get("id") or "").strip()
    mk_name = str(mk.get("name") or "").strip()
    if _anchored_ok(mk_id):
        out["main_kp"] = {"id": mk_id, "name": mk_name}
        need_review = False
    else:
        # 宁空不凑：id 留空、保留真实考点名（开集捕获）
        out["main_kp"] = {"id": "", "name": mk_name} if mk_name else None
        need_review = True
        flags.append(dna_extract.FLAG_MAIN_KP_OOB)
        if mk_id and dna_extract._is_review_id(mk_id) and not include_review_books:
            flags.append(dna_extract.FLAG_MAIN_KP_REVIEW_OOB)

    # 副 kp 逐项校验（越界丢弃）
    sec_in: list[dict[str, str]] = []
    for s in out.get("secondary_kps") or []:
        sid = str((s or {}).get("id") or "").strip()
        if _anchored_ok(sid) and sid != mk_id:
            sec_in.append({"id": sid, "name": str((s or {}).get("name") or "")})
        else:
            flags.append(dna_extract.FLAG_SECONDARY_KP_OOB)
        if len(sec_in) >= dna_extract.SECONDARY_KP_MAX:
            break
    out["secondary_kps"] = sec_in

    out["need_anchor_review"] = need_review
    out["flags"] = flags
    return out
