"""题型模版自动规范 formatter（BE 纯函数；镜像 FE SSOT normalize.ts）。

🔴 SSOT = book-ui/src/views/variant/normalize.ts（normalizeChoice/normalizeBlanks/
normalizeJudge）。两轨（BE formatter ↔ FE 结构化编辑器）必须产出/解析同一套 canonical
题干格式，否则 BE 规范 ↔ FE 解析对不上。本模块逐条对齐那份文件，行为一字不差。

边界（与 normalize.ts 完全一致，必须遵守）：
  规范是**纯排版（cosmetic）**——只动换行 / 下划线 / 补括号 / 选项分行，
  **绝不改数学语义、绝不动 $...$ 内的 LaTeX、不重排选项、不改字**。
  解析不出（拿不准是选择/填空/判断）→ 原样返回，绝不报错（降级 G5）。

三类处理（镜像 normalize.ts）：
  ① 选择题：每个选项（A. / B、/ C）独立成行（题干在前，选项各占一行）；
     存储一行一项（2×2 是 FE 渲染层的事，不写进存储文本）。
  ② 填空题：连续下划线 / 全角下划线 / 连续全角空格占位 → 标准下划线「____」(4 个半角)。
  ③ 判断题：题干尾若无判断括号 → 补「（  ）」。
题型判定优先用 qtype（BE 给的题型名），不可靠时按题干文本特征兜底。

幂等：已规范的文本再跑一次结果不变（防多轮应用漂移）。
"""

from __future__ import annotations

import re
from typing import Any

# 标准填空下划线（4 个半角下划线）——与 normalize.ts 的 BLANK 常量一致。
BLANK = "____"

# 选项前缀：行首/分隔后的 A-H + 分隔符（. ． 、 : ： ) ）都算）。
# 镜像 normalize.ts 的 OPTION_RE（仅用于题型兜底判定，不做全局替换）。
_OPTION_MARK = r"[A-H]\s*[.．、:：)）]"

# isChoice 文本兜底：含至少 A、B 两个选项标记（前面是行首/空白/中文标点）。
_CHOICE_A_RE = re.compile(r"(^|[\s，。；、])A\s*[.．、:：)）]")
_CHOICE_B_RE = re.compile(r"(^|[\s，。；、])B\s*[.．、:：)）]")

# isBlank 文本兜底：含连续（≥2）下划线 / 全角下划线占位。
_BLANK_TEXT_RE = re.compile(r"_{2,}|＿{2,}")

# normalizeBlanks：连续下划线 / 全角下划线统一长度；3+ 全角空格当填空占位。
_BLANK_RUN_RE = re.compile(r"[_＿]{2,}")
_FULLWIDTH_SPACE_RUN_RE = re.compile(r"　{3,}")

# 选项标记（标记后须紧跟非空白字符，避免误切空标记）。
# _FIRST_OPT_RE：定位第一个选项 → 切题干/选项区；_CHOICE_SPLIT_LOOKAHEAD_RE：零宽前瞻切各选项。
_FIRST_OPT_RE = re.compile(r"[A-H]\s*[.．、:：)）]\s*\S")
_CHOICE_SPLIT_LOOKAHEAD_RE = re.compile(r"(?=[A-H]\s*[.．、:：)）]\s*\S)")
# 行首选项标记（去空行收口时判一行是不是选项行）。
_OPTION_LINE_RE = re.compile(r"^[A-H]\s*[.．、:：)）]")

# 选项块抠「标签字母 + 正文」（去重/封顶/规整用）。镜像 variant_support._OPT_LABEL_RE。
_OPT_CHUNK_RE = re.compile(r"^\s*[（(]?\s*([A-H])\s*[）).．、:：]\s*(.*)$", re.DOTALL)
# 规整后重排用的字母序（A-H，最多 8 项；绝不产出 ? 标签）。
_CHOICE_LETTERS = "ABCDEFGH"


def dedup_cap_reorder_choice(chunks: list[str]) -> list[str]:
    """A2 SSOT：选项块去重 + 封顶 + 顺序规整（FE/BE/入库 一字不差）。

    输入 = 切出的各选项原文块（每块含「A. 甲」式标记 + 正文）。规则：
      ① 按 label 去重：同一字母 label 只保留**第一次**出现，后续重复 label 丢弃；
      ② 封顶：去重后最多取前 len(_CHOICE_LETTERS)=8 项（绝不产出/落库 label=? 的项）；
      ③ 顺序规整：去重封顶后按 A/B/C/D… **连续重排** label（原 label 跳号/乱序也规整成连续）。
    返回规整后的选项块列表（每块形如 "A. 正文"，正文原样不动·cosmetic-only）。
    解析不出 label 的块按出现顺序保留正文，分配下一个连续字母（降级不丢内容）。
    幂等：已是连续合法 A.B.C.D 的入参再跑结果不变。
    """
    seen: set[str] = set()
    contents: list[str] = []
    for chunk in chunks:
        c = chunk.strip()
        if not c:
            continue
        m = _OPT_CHUNK_RE.match(c)
        if m:
            label = m.group(1)
            if label in seen:
                continue  # 重复 label → 丢弃（保留首次）
            seen.add(label)
            contents.append((m.group(2) or "").strip())
        else:
            # 抠不出 label（罕见）→ 正文原样留，占下一个连续字母位（降级不丢内容）。
            contents.append(c)
    # 封顶到字母序长度，按连续字母重排 label。
    contents = contents[: len(_CHOICE_LETTERS)]
    return [f"{_CHOICE_LETTERS[i]}. {content}" for i, content in enumerate(contents)]

# normalizeJudge：题干尾已含判断括号（半/全角空括号或带对错符）→ 不重复补。
_JUDGE_EMPTY_PAREN_RE = re.compile(r"[（(]\s*[）)]\s*$")
_JUDGE_FILLED_PAREN_RE = re.compile(r"[（(].{0,3}[）)]\s*$")
_TRAILING_WS_RE = re.compile(r"\s+$")


def _is_choice(stem: str, qtype: str) -> bool:
    if re.search(r"选择|单选|多选", qtype):
        return True
    return bool(_CHOICE_A_RE.search(stem)) and bool(_CHOICE_B_RE.search(stem))


def _is_blank(stem: str, qtype: str) -> bool:
    if re.search(r"填空", qtype):
        return True
    return bool(_BLANK_TEXT_RE.search(stem))


def _is_judge(stem: str, qtype: str) -> bool:
    return bool(re.search(r"判断|对错|正误", qtype))


def _normalize_blanks(text: str) -> str:
    """填空：连续下划线 / 全角下划线 / 连续全角空格占位 → 统一标准下划线。"""
    text = _BLANK_RUN_RE.sub(BLANK, text)  # 已有的连续下划线统一长度
    text = _FULLWIDTH_SPACE_RUN_RE.sub(BLANK, text)  # 3+ 全角空格当填空占位
    return text


def _normalize_choice(text: str) -> str:
    """选择题：题干 + 每个选项独立成段（只调换行，不重排、不改内容）。

    🔴 解析-重建式（幂等核心）：原"正则插换行"做法镜像 JS replace 的零宽双替换副产物，
    选项间空行数不一致（首项单 \\n、其余 \\n\\n），再跑一次还会给首项补一个空行 →
    **非幂等**（formatter 在 assemble/edit/reverify 多处应用，漂移不可接受）。改为：
      ① 定位第一个选项标记 → 切出题干 head + 选项区 region；
      ② region 按选项标记前瞻切成各选项、去首尾空白；
      ③ 重建 = head + 选项各占一段，统一用空行分隔（markdown 单 \\n 不换行，须 \\n\\n
         才渲染成独立行）。
    canonical = "题干（ ）\\n\\nA. 甲\\n\\nB. 乙\\n\\nC. 丙\\n\\nD. 丁"——所有选项间距一致，
    再跑一次：head/选项原样切出、原样重建 → 不动点（幂等）。cosmetic-only：不重排、不改字。
    """
    m = _FIRST_OPT_RE.search(text)
    if not m:
        return text  # 没找到选项标记 → 原样（降级，不臆断）
    head = text[: m.start()].rstrip()
    region = text[m.start() :]
    raw = [p for p in _CHOICE_SPLIT_LOOKAHEAD_RE.split(region) if p.strip()]
    if not raw:
        return text
    # A2 SSOT：去重 label + 封顶到合法字母序 + 连续重排（与 FE/入库 一字不差）。
    opts = dedup_cap_reorder_choice(raw)
    if not opts:
        return text
    body = "\n\n".join(opts)
    return f"{head}\n\n{body}" if head else body


def _normalize_judge(text: str) -> str:
    """判断题：题干尾若无判断括号则补「（  ）」。"""
    trimmed = _TRAILING_WS_RE.sub("", text)
    if _JUDGE_EMPTY_PAREN_RE.search(trimmed) or _JUDGE_FILLED_PAREN_RE.search(trimmed):
        return text  # 已含判断括号 → 不重复补
    return f"{trimmed}（  ）"


def format_by_qtype(stem: Any, qtype: Any = "") -> Any:
    """规范排版主入口（纯函数·cosmetic-only·幂等·永不抛）。

    镜像 normalize.ts.normalizeStem：
      选择题 → normalizeBlanks → normalizeChoice；
      判断题 → normalizeBlanks → normalizeJudge；
      填空题 → normalizeBlanks；
      判定不出 → 保守只做下划线统一（无下划线则原样）。

    @param stem  题干原文（markdown + 可能内联 LaTeX）
    @param qtype BE 给的题型名（可空）
    @returns 规范化后的题干；判定不出 / 无需调整 / 空 / 异常 → 原样返回（绝不抛）。
    """
    if not isinstance(stem, str) or not stem.strip():
        return stem
    qt = str(qtype or "")
    try:
        if _is_choice(stem, qt):
            # 选择题：先统一可能的填空占位（题干里也可能有空），再拆选项行。
            return _normalize_choice(_normalize_blanks(stem))
        if _is_judge(stem, qt):
            return _normalize_judge(_normalize_blanks(stem))
        if _is_blank(stem, qt):
            return _normalize_blanks(stem)
        # 判定不出题型：保守只做下划线统一（无下划线则原样）。
        return _normalize_blanks(stem)
    except Exception:  # noqa: BLE001 — 任何异常都降级原样返回（G5），绝不炸主流程。
        return stem
