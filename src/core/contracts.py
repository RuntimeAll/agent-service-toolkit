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
