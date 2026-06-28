"""variant 引擎 · stage2_variant prompt 常量层（PRD-C-104 B3a 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出变式生成/重生的两个**拼接型** prompt：
  GENERATE_PROMPT（造 N 道变式）/ REGEN_PROMPT（闸B 验算不一致 → 重出一道）。

🔴 行为零改：字符串字面量逐字搬。二者由 4 个契约/rubric 片段拼接而成
   （_FIGURE_SPEC_CONTRACT / _DIFFICULTY_RUBRIC / _QTYPE_CONTRACT / _PAYLOAD_CONTRACT），
   这些片段已在 B1 抽到 variant/prompts.py —— 本模块从那里 import 进来拼接，
   断掉「拼接型 prompt 在 __init__ 引用 variant/prompts 片段」的跨模块临时接缝（§B3）。
   __init__.py 末尾 re-export GENERATE_PROMPT / REGEN_PROMPT → 调用方零感。
"""

from __future__ import annotations

from agents.variant.prompts import (
    _FIGURE_SPEC_CONTRACT,
    _DIFFICULTY_RUBRIC,
    _QTYPE_CONTRACT,
    _PAYLOAD_CONTRACT,
)


GENERATE_PROMPT = (
    """你是浙教版初中数学命题专家。基于母题 DNA，造 {n} 道举一反三变式。

只输出 JSON 数组(不要解释)，每个元素：
{{"stem":"题干(Markdown+LaTeX)","answer":"标准答案","solution":"完整解析(过程+答案)",
  "qtype":"选择/填空/解答","difficulty":1~4,"level":"normal/hard","injected_kp":"相邻kp名或null",
  "figure_spec":{{"layout":"...","angle_labels":[...]}} 或 "" （配图决策对象/空串，契约见下；纯代数题给 ""）,
  "verify_payload":{{...该题的程序验算载荷，契约见下...}}}}

"""
    + _FIGURE_SPEC_CONTRACT
    + """

"""
    + _DIFFICULTY_RUBRIC
    + """

格式硬规定（stem/answer/solution 三个字段都遵守）：
- 🔴 题面(stem)与选项里的数学式**一律行内 $...$**，如 $\\sqrt{{2}}$、$x^2-3x+2=0$、$\\frac{{px+a}}{{4}}=2-\\frac{{x+bp}}{{8}}$；
  **严禁** `$$...$$` / `\\[ \\]` / 任何 display 块级公式（会渲染成撑满整行的大号公式、强制换行，破坏阅读）。
- **仅 solution 里多行分步推导**可用 $$...$$（一步一行的竖排演算）；其余单个等式仍优先行内 $...$。
- **禁止**裸 LaTeX 命令、禁止 \\( \\) 定界符。
- 🔴 选项间距禁用 `\\quad`/`\\qquad`/`\\,` 等 LaTeX 间距命令，**选项各自成项**（用换行或并列文本分隔），
  绝不写成 `A. 37° \\quad B. 53°` 这种一行内 $...$ 外裸 \\quad（渲染层不认、会裸露）。
- 换行用 JSON 标准转义 \\n（一个反斜杠），不要写成 \\\\n。

"""
    + _QTYPE_CONTRACT
    + """

verify_payload 字段（PRD-C-012 4a·出题自带验算载荷：把**这道题自己的题干 + 标准答案**抽成可被 sympy 程序验算的结构化载荷，验算对象 claimed = 该题标准答案）：
"""
    + _PAYLOAD_CONTRACT
    + """
抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。

铁律：
- **主考点 + 年级 硬守恒**：每道题都必须仍考「{kp_name}」、仍在该年级范围内。
- 守{{解题结构, 难度(普通题)}}；只换{{数字, 场景}}。
- 难题 1 道升一档；可综合 1 个相邻知识点(主考点仍守，注入为副点)，填到 injected_kp。
- **答案可程序验算**(PRD-C-012 4c)：answer 优先给可计算的数值/表达式（如 $x_1=2, x_2=3$、
  $3\\sqrt{{2}}$、选项字母），能出数值答案就不要出纯文字表述答案（证明/作图类除外）；
  数字设计成解恰好整洁可验（避免无理数逼近、区间叙述、带单位混排）。

配方(默认)：共 {n} 道 = {n_normal} 道普通(守难度) + {n_hard} 道难题(升一档)。

母题 DNA：
- 主考点(硬守恒，不可改): {kp_name}
- 年级(硬守恒): {grade}
- 题型: {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}"""
)


# 🔴 PRD-C-106 B3·per-variant 单题 prompt（阶段二 fan-out 每道子上下文各调一次）：
#   与 GENERATE_PROMPT 同契约（figure_spec / 难度 rubric / 题型 / payload / 格式硬规定），
#   但**只出一道**，且把本道派工(变式系数/算子/难度档)注入指令——让每道按各自 knob 真分级。
#   占位符：{kp_name}/{grade}/{qtype}/{stem}/{skeleton}（来自 facts，与 GENERATE 同源）
#   + {seq}/{total}/{coeff}/{operator}/{op_guidance}/{difficulty_line}（来自 PLAN spec）。
GENERATE_ONE_PROMPT = (
    """你是浙教版初中数学命题专家。基于母题 DNA，造**恰好 1 道**举一反三变式（这是一组 {total} 道里的第 {seq} 道）。

只输出**单个 JSON 对象**(不要解释、不要数组)：
{{"stem":"题干(Markdown+LaTeX)","answer":"标准答案","solution":"完整解析(过程+答案)",
  "qtype":"选择/填空/解答","difficulty":1~4,"level":"normal/hard","injected_kp":"相邻kp名或null",
  "figure_spec":{{"layout":"...","angle_labels":[...]}} 或 "" （配图决策对象/空串，契约见下；纯代数题给 ""）,
  "verify_payload":{{...该题的程序验算载荷，契约见下...}}}}

🔴 本道派工（变式系数 + 算子 + 难度，必须严格按此出）：
- 变式系数 {coeff}（{operator}）：{op_guidance}
{difficulty_line}

"""
    + _FIGURE_SPEC_CONTRACT
    + """

"""
    + _DIFFICULTY_RUBRIC
    + """

格式硬规定（stem/answer/solution 三个字段都遵守）：
- 🔴 题面(stem)与选项里的数学式**一律行内 $...$**，如 $\\sqrt{{2}}$、$x^2-3x+2=0$；
  **严禁** `$$...$$` / `\\[ \\]` / 任何 display 块级公式（撑满整行、强制换行，破坏阅读）。
- **仅 solution 里多行分步推导**可用 $$...$$；其余单个等式仍优先行内 $...$。
- **禁止**裸 LaTeX 命令、禁止 \\( \\) 定界符。
- 🔴 选项间距禁用 `\\quad`/`\\qquad`/`\\,` 等命令，**选项各自成项**。
- 换行用 JSON 标准转义 \\n（一个反斜杠）。

"""
    + _QTYPE_CONTRACT
    + """

verify_payload 字段（把**这道题自己的题干 + 标准答案**抽成可被 sympy 程序验算的结构化载荷，验算对象 claimed = 该题标准答案）：
"""
    + _PAYLOAD_CONTRACT
    + """
抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。

铁律：
- **主考点 + 年级 硬守恒**：必须仍考「{kp_name}」、仍在该年级范围内。
- 守{{解题结构（普通题）}}；按上面派工的变式系数/算子改{{数字, 场景, 结构}}。
- 可综合 1 个相邻知识点(主考点仍守，注入为副点)，填到 injected_kp。
- **答案可程序验算**：answer 优先给可计算的数值/表达式，数字设计成解恰好整洁可验。

母题 DNA：
- 主考点(硬守恒，不可改): {kp_name}
- 年级(硬守恒): {grade}
- 题型: {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}"""
)


REGEN_PROMPT = (
    """下面这道变式题，独立解出的答案与题面标答不一致，请**重新出一道**等价变式重做。

主考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
原题干: {stem}
要求：仍考「{kp_name}」、仍在「{grade}」、{level} 难度（目标难度档约 {difficulty}）；换数字/场景使题面与答案自洽；
answer 优先给可计算的数值/表达式（可程序验算），数字设计成解恰好整洁。

只输出 JSON：
{{"stem":"新题干","answer":"标准答案","solution":"完整解析","qtype":"{qtype}","difficulty":1~4,"level":"{level}","injected_kp":{injected_kp},
  "verify_payload":{{...新题的程序验算载荷，契约见下...}}}}
🔴 difficulty 字段（整改2·难度并入生题）：拿下面这张 rubric 对**你重出的这道新题**断一个 1~4 的难度档（不是照抄目标档，是按 rubric 实判）：

"""
    + _DIFFICULTY_RUBRIC
    + """

格式硬规定：🔴 题面(stem)与选项数学式**一律行内 $...$**，**严禁** $$...$$ / \\[ \\] / display 块级公式（撑满整行）；**仅 solution 多行分步推导**可用 $$。禁止裸 LaTeX / \\( \\) 定界；换行用标准 \\n。

"""
    + _QTYPE_CONTRACT
    + """

verify_payload 字段（PRD-C-012 4a·出题自带验算载荷：把**新题的题干 + 标准答案**抽成可被 sympy 程序验算的结构化载荷，验算对象 claimed = 新题标准答案）：
"""
    + _PAYLOAD_CONTRACT
    + """
抽不成（文字应用题难建模/几何图形/证明/答案含区间或单位等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。"""
)
