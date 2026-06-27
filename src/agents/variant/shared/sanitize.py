"""variant 引擎 · 富文本净化层（PRD-C-104 B2 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出（约 934-1004 行）：
  - 正则常量 _PAREN_MATH_RE / _BRACKET_MATH_RE / _LITERAL_NL_RE / _MATH_SPLIT_RE
    / _BARE_SPACING_RE / _GLUED_GEOM_RE / _BARE_DEGREE_RE
  - 函数 _fix_glued_inside_math / _strip_bare_spacing_outside_math / _sanitize_rich_text
    / _sanitize_item / join_skeleton

🔴 行为零改：内容逐字搬，仅补本模块所需 import（re/Any）。
   __init__.py 顶部 re-export 这些符号 → service.py / variant_entry.py 零感。
"""

from __future__ import annotations

import re
from typing import Any


# --- 富文本净化（用户反馈 2026-06-11：解析裸字符不渲染的根因） -----------------
# LLM（gpt-5.4）产 JSON 时两类脏输出：① LaTeX 用 \( \) / \[ \] 定界（前端
# markdown-it-katex 只认 $/$$，且 markdown 会把 \( 的反斜杠当转义吃掉）；② 把换行
# 写成双反斜杠 → 解出字面 \n 两个字符。统一在解析边界净化，入库/快照/气泡三处共净。
_PAREN_MATH_RE = re.compile(r"\\\(\s*(.+?)\s*\\\)", re.DOTALL)
_BRACKET_MATH_RE = re.compile(r"\\\[\s*(.+?)\s*\\\]", re.DOTALL)
# 字面 \n 后跟小写字母 = 可能是 LaTeX 命令（\neq \nabla \newline \nu …），不动；其余视为换行
_LITERAL_NL_RE = re.compile(r"\\n(?![a-z])")

# 🔴 A1（2026-06-18）：opus 把选项写成一行 `A. 37° \quad B. 53° …`，\quad 在 $...$ **外**
#   → KaTeX 不渲染、裸露。把出现在 $...$ 外的裸 LaTeX 间距命令替成普通空格；$...$ 内的不动
#   （留给 KaTeX）。FE mathNormalize.ts 同口径，两边一致。
# 切分 $$...$$ / $...$ 数学段：偶数下标 = 段外文本，奇数下标 = 数学段（含定界符）。
_MATH_SPLIT_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\$[^\n$]+?\$)")
# 裸间距命令：\quad \qquad \, \; \! \:（后接非字母边界，防误伤 \quadword 之类）
_BARE_SPACING_RE = re.compile(r"\\(?:qquad|quad)(?![a-zA-Z])|\\[,;!:]")
# 🔴 2026-06-21（PRD-A-018 用户终审，DB 挖真值定位）：LLM 产 $...$ 时**闭合 $ 前留空格**，如
#   `$\angle 1 = 44^\circ $`。markdown-it-katex 要求闭合 $ 前非空白（防误匹配货币）→ 整段不被识别
#   为公式、裸显示源码。**真根因** = trim 掉 $...$ 内首尾空白。兼带防御：粘连几何命令补空格、裸 °→^\circ。
#   与 FE mathNormalize.ts fixGluedInsideMath 同口径。
_GLUED_GEOM_RE = re.compile(r"\\(angle|triangle|parallel|nparallel|perp|cong|simeq|odot)(?=[A-Z])")
_BARE_DEGREE_RE = re.compile("°")


def _fix_glued_inside_math(s: str) -> str:
    """修 $...$ **内**常见 LLM LaTeX 脏写：trim 首尾空白(真根因)、粘连几何命令补空格、裸 °→^\\circ。段外不碰。"""
    parts = _MATH_SPLIT_RE.split(s)
    for i in range(1, len(parts), 2):  # 奇数下标 = 数学段（含定界符）
        seg = parts[i]
        dd = seg.startswith("$$")
        inner = seg[2:-2] if dd else seg[1:-1]
        fixed = _BARE_DEGREE_RE.sub(r"^\\circ ", _GLUED_GEOM_RE.sub(r"\\\1 ", inner)).strip()
        if fixed:
            parts[i] = f"$${fixed}$$" if dd else f"${fixed}$"
    return "".join(parts)


def _strip_bare_spacing_outside_math(s: str) -> str:
    """把 $...$ 外的裸 LaTeX 间距命令（\\quad \\qquad \\, \\; \\! \\:）替成普通空格；段内原样。"""
    parts = _MATH_SPLIT_RE.split(s)
    for i in range(0, len(parts), 2):  # 偶数下标 = 数学段外文本
        if parts[i]:
            parts[i] = _BARE_SPACING_RE.sub(" ", parts[i])
    return "".join(parts)


def _sanitize_rich_text(s: Any) -> Any:
    """LLM 产出的 stem/answer/solution 净化：\\(..\\)→$..$、\\[..\\]→$$..$$、字面 \\n→换行、
    $...$ 外裸间距命令(\\quad 等)→空格。"""
    if not isinstance(s, str) or not s:
        return s
    s = _BRACKET_MATH_RE.sub(lambda m: f"$${m.group(1)}$$", s)
    s = _PAREN_MATH_RE.sub(lambda m: f"${m.group(1)}$", s)
    s = _LITERAL_NL_RE.sub("\n", s)
    return _fix_glued_inside_math(_strip_bare_spacing_outside_math(s))


def _sanitize_item(it: dict[str, Any]) -> dict[str, Any]:
    """就地净化一道题的富文本字段，返回原 dict（链式用）。"""
    for k in ("stem", "answer", "solution"):
        it[k] = _sanitize_rich_text(it.get(k))
    return it


def join_skeleton(lines: Any) -> str:
    """P8（2026-06-18）：骨架步骤序列逐行净化后换行拼接 → 落 solution_skeleton/analyze。
    解决图母题入库 analyze = 未净化 skeleton（裸 \\(..\\)/裸间距命令）裸露。"""
    out = []
    for s in lines or []:
        out.append(_sanitize_rich_text(str(s)))
    return "\n".join(out)
