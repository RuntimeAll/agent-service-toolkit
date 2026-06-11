"""题型模版自动规范 formatter 单测（PRD-C-009·BE）。

🔴 SSOT = book-ui/src/views/variant/normalize.ts。本测逐条对齐其
normalizeChoice/normalizeBlanks/normalizeJudge 行为，并守住 cosmetic-only 铁律：
  - 选择题选项一行一项、题干（ ）收尾不被破坏；
  - 填空 ____ 统一；判断补（  ）；
  - 幂等（再跑一次结果不变）；
  - LaTeX（$...$ 内）一字不动；
  - 解析不出 / 空 → 原样返回，绝不抛。
"""

import pytest

from agents.qtype_format import BLANK, format_by_qtype


class TestChoice:
    def test_inline_options_split_to_lines(self):
        # 内联选项 → 每个独立成行；题干（ ）收尾保留。
        # 🔴 canonical（幂等·解析-重建）：题干 + 各选项独立成段、统一空行分隔（markdown 须
        #   \n\n 才渲染成独立行），即 "题干（ ）\n\nA. 甲\n\nB. 乙\n\nC. 丙\n\nD. 丁"。
        stem = "下列哪个是正确的（  ）A. 甲 B. 乙 C. 丙 D. 丁"
        out = format_by_qtype(stem, "选择题")
        assert out == "下列哪个是正确的（  ）\n\nA. 甲\n\nB. 乙\n\nC. 丙\n\nD. 丁"
        lines = out.split("\n")
        assert lines[0] == "下列哪个是正确的（  ）"
        # 每个选项独立成行（一行一项，2×2 是 FE 渲染层的事）
        opt_lines = [ln for ln in lines if ln and ln[0] in "ABCD"]
        assert opt_lines == ["A. 甲", "B. 乙", "C. 丙", "D. 丁"]

    def test_text_fallback_detect_choice_without_qtype(self):
        # qtype 缺失 → 文本兜底（含 A./B. 两个标记）判为选择题。
        stem = "选项判断 A. 对 B. 错"
        out = format_by_qtype(stem, "")
        assert "\nA. 对" in out
        assert "\nB. 错" in out

    def test_various_separators(self):
        # 分隔符 . 、 ： ) 都算选项标记。
        stem = "题（）A、一 B：二 C) 三 D. 四"
        out = format_by_qtype(stem, "单选")
        opt_lines = [ln for ln in out.split("\n") if ln and ln[0] in "ABCD"]
        assert opt_lines == ["A、一", "B：二", "C) 三", "D. 四"]

    def test_choice_idempotent(self):
        stem = "题干（  ）A. 甲 B. 乙 C. 丙 D. 丁"
        once = format_by_qtype(stem, "选择题")
        twice = format_by_qtype(once, "选择题")
        assert once == twice

    def test_canonical_output_is_fixed_point(self):
        # 规范输出（题干 + 各选项统一空行分隔）= 不动点：再跑一次结果不变（幂等核心）。
        canonical = "题干（  ）\n\nA. 甲\n\nB. 乙\n\nC. 丙\n\nD. 丁"
        assert format_by_qtype(canonical, "选择题") == canonical


class TestBlank:
    def test_underscore_run_normalized(self):
        stem = "圆周率约等于______（保留两位）"
        out = format_by_qtype(stem, "填空题")
        assert BLANK in out
        assert "______" not in out

    def test_fullwidth_underscore_normalized(self):
        stem = "答案是＿＿＿＿＿"
        out = format_by_qtype(stem, "填空题")
        assert BLANK in out
        assert "＿" not in out

    def test_fullwidth_spaces_as_blank(self):
        # 3+ 全角空格当填空占位。
        stem = "结果为　　　　元"
        out = format_by_qtype(stem, "填空")
        assert BLANK in out

    def test_blank_text_fallback_no_qtype(self):
        stem = "x 的值是____。"
        out = format_by_qtype(stem, "")
        assert BLANK in out

    def test_blank_idempotent(self):
        stem = "x 的值是______。"
        once = format_by_qtype(stem, "填空题")
        twice = format_by_qtype(once, "填空题")
        assert twice == once
        assert once == "x 的值是____。"


class TestJudge:
    def test_append_paren_when_missing(self):
        stem = "三角形内角和是 180 度。"
        out = format_by_qtype(stem, "判断题")
        assert out == "三角形内角和是 180 度。（  ）"

    def test_no_duplicate_when_paren_present(self):
        stem = "三角形内角和是 180 度（  ）"
        out = format_by_qtype(stem, "判断题")
        assert out == stem  # 已含判断括号 → 不重复补

    def test_judge_with_filled_paren_unchanged(self):
        stem = "命题为真（√）"
        out = format_by_qtype(stem, "对错")
        assert out == stem

    def test_judge_idempotent(self):
        stem = "勾股定理对任意三角形成立。"
        once = format_by_qtype(stem, "正误")
        twice = format_by_qtype(once, "正误")
        assert twice == once
        assert once.endswith("（  ）")


class TestLatexAndDegrade:
    def test_latex_in_choice_untouched(self):
        # $...$ 内的 LaTeX 一字不动（绝不改数学语义）。
        stem = r"解 $x^2-3x+2=0$ 得（  ）A. $x=1$ B. $x=2$ C. $x=3$ D. $x=4$"
        out = format_by_qtype(stem, "选择题")
        assert "$x^2-3x+2=0$" in out
        assert "$x=1$" in out and "$x=2$" in out
        # 选项仍分行（canonical：选项间夹空行）
        opt_lines = [ln for ln in out.split("\n") if ln and ln[0] in "ABCD"]
        assert opt_lines == ["A. $x=1$", "B. $x=2$", "C. $x=3$", "D. $x=4$"]

    def test_latex_in_blank_untouched(self):
        stem = r"已知 $\frac{1}{2}+\frac{1}{3}$ 的值为______"
        out = format_by_qtype(stem, "填空题")
        assert r"$\frac{1}{2}+\frac{1}{3}$" in out
        assert BLANK in out

    def test_solve_question_not_forced(self):
        # 解答题不强排，保留多小问原样。
        stem = "(1) 求函数定义域；(2) 求最大值。"
        out = format_by_qtype(stem, "解答题")
        assert out == stem

    def test_undetermined_qtype_underscore_only(self):
        # 判定不出题型 → 保守只做下划线统一（无下划线则原样）。
        stem = "这是一段普通描述，没有任何题型特征。"
        assert format_by_qtype(stem, "解答") == stem

    def test_empty_and_non_string_returned_as_is(self):
        assert format_by_qtype("", "选择题") == ""
        assert format_by_qtype("   ", "选择题") == "   "
        assert format_by_qtype(None, "选择题") is None
        assert format_by_qtype(123, "填空题") == 123

    def test_never_raises_returns_original(self):
        # 任意题型名 / 奇异输入都不抛。
        assert format_by_qtype("纯文本", "莫名其妙的题型") == "纯文本"


class TestQtypePriorityOverText:
    def test_judge_qtype_wins_over_blank_text(self):
        # qtype=判断 优先：含下划线也按判断处理（补括号 + 下划线统一）。
        stem = "结果为____"
        out = format_by_qtype(stem, "判断题")
        assert out.endswith("（  ）")
        assert BLANK in out
