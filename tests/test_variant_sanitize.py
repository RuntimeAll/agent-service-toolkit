"""富文本净化（用户反馈 2026-06-11：解析裸字符不渲染）单测。

根因：LLM 产 JSON 时 ① LaTeX 用 \\( \\) / \\[ \\] 定界（FE markdown-it-katex 只认 $）；
② 换行写成双反斜杠 → 解出字面 \\n 两字符。净化在解析边界统一做。
"""

import pytest

from agents.variant import (
    _sanitize_item,
    _sanitize_rich_text,
    _strip_bare_spacing_outside_math,
    join_skeleton,
)


class TestSanitizeRichText:
    def test_paren_delimiter_to_dollar(self):
        assert _sanitize_rich_text(r"解 \(x^2-3x+2=0\) 得") == "解 $x^2-3x+2=0$ 得"

    def test_bracket_delimiter_to_display_dollar(self):
        assert _sanitize_rich_text(r"\[ \frac{1}{2} \]") == r"$$\frac{1}{2}$$"

    def test_literal_backslash_n_to_newline(self):
        # 字面 \n（反斜杠+n 两字符）后跟非小写字母 → 真换行
        assert _sanitize_rich_text(r"第一步\n\n(1) 化简") == "第一步\n\n(1) 化简"

    def test_latex_commands_starting_with_n_preserved(self):
        # \neq \nabla 等 LaTeX 命令（\n 后跟小写字母）不能被劈成换行
        assert _sanitize_rich_text(r"$a \neq b$") == r"$a \neq b$"
        assert _sanitize_rich_text(r"$\nabla f$") == r"$\nabla f$"

    def test_mixed_real_case(self):
        # 模拟 DB 实测脏样本：字面 \n + \( \) 混排
        src = r"先化简\n\n\(\sqrt{18}=3\sqrt{2}\)，故答案 \(9\sqrt{2}\)"
        out = _sanitize_rich_text(src)
        assert out == "先化简\n\n$\\sqrt{18}=3\\sqrt{2}$，故答案 $9\\sqrt{2}$"

    def test_non_string_passthrough(self):
        assert _sanitize_rich_text(None) is None
        assert _sanitize_rich_text(3) == 3
        assert _sanitize_rich_text("") == ""

    def test_dollar_already_clean_untouched(self):
        s = "已规范：$x^2$ 与 $$\\frac{a}{b}$$\n换行也真"
        assert _sanitize_rich_text(s) == s


class TestSanitizeItem:
    def test_sanitizes_three_fields_in_place(self):
        it = {
            "stem": r"\(x+1=2\)",
            "answer": r"x=1",
            "solution": r"移项\n得 \(x=1\)",
            "qtype": "填空",
        }
        out = _sanitize_item(it)
        assert out is it  # 就地修改，链式返回原 dict
        assert it["stem"] == "$x+1=2$"
        assert it["solution"] == "移项\n得 $x=1$"
        assert it["qtype"] == "填空"  # 其余字段不动

    def test_missing_fields_ok(self):
        it = {"stem": None}
        _sanitize_item(it)
        assert it["stem"] is None
        assert it["answer"] is None


class TestBareSpacingOutsideMath:
    """A1（2026-06-18）：$...$ 外裸 LaTeX 间距命令 → 空格；段内原样交 KaTeX。"""

    def test_bare_quad_outside_stripped(self):
        out = _sanitize_rich_text(r"A. 37° \quad B. 53° \quad C. 60°")
        assert "\\quad" not in out
        assert "A. 37°" in out and "B. 53°" in out

    def test_quad_inside_math_preserved(self):
        # $...$ 内的 \quad 不动，留给 KaTeX
        assert _sanitize_rich_text(r"$x \quad y$") == r"$x \quad y$"

    def test_mixed_inside_and_outside(self):
        # 题目自检样例：外面 \quad 清掉、$x \quad y$ 完整保留
        out = _sanitize_rich_text(r"A. 37° \quad B. 53° $x \quad y$")
        assert r"$x \quad y$" in out
        # 段外的 \quad（B 后面那个）没了
        before_math = out.split("$x")[0]
        assert "\\quad" not in before_math

    def test_all_spacing_commands(self):
        out = _strip_bare_spacing_outside_math(r"a \quad b \qquad c \, d \; e \! f \: g")
        for cmd in (r"\quad", r"\qquad", r"\,", r"\;", r"\!", r"\:"):
            assert cmd not in out
        assert all(ch in out for ch in "abcdefg")

    def test_display_math_span_preserved(self):
        # $$...$$ 段内 \quad 也不动
        assert _strip_bare_spacing_outside_math(r"$$a \quad b$$ \quad c") == r"$$a \quad b$$   c"

    def test_no_false_positive_on_word(self):
        # \quadword 不是裸 \quad（后接字母）→ 不动
        assert _strip_bare_spacing_outside_math(r"\quadword") == r"\quadword"


class TestJoinSkeleton:
    """P8：骨架逐行净化后拼接 → analyze 不裸露。"""

    def test_each_line_sanitized(self):
        out = join_skeleton([r"第一步 \(x=1\)", r"第二步 A \quad B"])
        assert out == "第一步 $x=1$\n第二步 A   B"

    def test_empty_and_none(self):
        assert join_skeleton([]) == ""
        assert join_skeleton(None) == ""

    def test_non_str_lines(self):
        assert join_skeleton([1, 2]) == "1\n2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
