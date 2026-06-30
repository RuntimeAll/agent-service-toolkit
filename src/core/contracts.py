# -*- coding: utf-8 -*-
"""跨工作流「DNA 契约词表」单一真相源（2026-06-30 魔法值收口）。

🔴 这是 FE↔toolkit↔Python 的**代码契约**词表（题型 / 考察类型 / 难度值域），**不是 RuoYi 字典**——
   toolkit 不直连库、也绝不依赖「用户可编辑的字典」来定 DNA 契约（用户改字典不该能改坏引擎）。
   打标 / 解题 / 变式三条线统一从这里取，杜绝各模块各写一份、防漂移。

🔴 纯叶子模块铁律：只 import stdlib / typing，**绝不 import agents.\* 或 core 内其它业务模块**
   （否则成循环 import）。这样任何模块都能安全 `from core.contracts import ...`。
"""

from __future__ import annotations

# 考察类型（dim2）闭集 10（22-SSOT §1）。原 dna_extract.py / labeler.py 各定义一份完全相同的，
#   现收口到此唯一真相源；两处改为 re-export，存量 `dna_extract.EXAM_TYPES` 引用零改仍可用。
EXAM_TYPES: list[str] = [
    "概念辨析", "直接计算", "公式套用", "性质判定", "证明推理",
    "应用建模", "作图", "探究归纳", "阅读理解迁移", "纠错",
]

# 题型三类短名（misikt 真实 3 种的 DNA 短名）。
QTYPES: list[str] = ["选择", "填空", "解答"]

# 题型别名归一表：各种中文写法 → 三类短名（值域 = {选择, 填空, 解答}）。
#   原 dna_extract._QTYPE_ALIAS 与 variant/__init__._QTYPE_ALIAS 两份**逐字相同**，收口到此。
#   🔴 与下方 QTYPE_TO_MISIKT（→入库码）**值域不同，绝不合并**：一个出短名、一个出 misikt 整数码。
QTYPE_ALIAS: dict[str, str] = {
    "选择": "选择", "选择题": "选择", "单选": "选择", "单选题": "选择",
    "填空": "填空", "填空题": "填空",
    "解答": "解答", "解答题": "解答", "计算": "解答", "计算题": "解答",
    "应用": "解答", "应用题": "解答", "证明": "解答", "证明题": "解答", "大题": "解答",
}

# 题型中文名 → misikt CreateQuestionBo.questionType 入库码（1=选择 / 4=填空 / 5=解答系）。
#   原 variant_support.QTYPE_MAP，收口到此。🔴 值域 = {1, 4, 5}，与 QTYPE_ALIAS（短名）不同，别混。
#   未命中（如「应用题」）由调用方回退 DEFAULT_QTYPE（=5），与原 QTYPE_MAP.get(s, DEFAULT_QTYPE) 行为一致。
QTYPE_TO_MISIKT: dict[str, int] = {
    "选择": 1, "选择题": 1,
    "填空": 4, "填空题": 4,
    "解答": 5, "解答题": 5, "简答": 5, "简答题": 5,
    "计算": 5, "计算题": 5, "证明": 5, "证明题": 5,
}
