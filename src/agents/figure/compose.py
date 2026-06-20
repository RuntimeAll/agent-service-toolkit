# -*- coding: utf-8 -*-
r"""PRD-C-100 B3 带图管线 · service 层后处理（D14：不进变式 StateGraph，四节点字节不动）。

三件事（人在回路）：
  B3.1 母题切图 `crop_mother_figure`：带图母题 → figure-crop 检测+裁 → PNG base64（母题图直贴）。
  B3.2 变式造图一轮直出 `compose_variant_figure`：opus 翻 GeoGebra 命令（不调 verify_figure 回炉 D12）
       → mathfig 渲染 → PNG base64。⚙模板（geogebra_samples）+ needs_figure 降级。
  B3.3 图片重生：同 compose_variant_figure 带 correction_prompt（老师发修正提示词 → 单图重造）。

🔴 铁律：图全链 PNG 无损；渲染失败 try/except 降级（needs_figure ⚠ 外显，不掐 SSE/不卡流程 G11）；
   opus 翻命令走 invoke(=_ainvoke_text) 落 conv_trace（label=figure_geogebra，G6 opus 调用不漏计）；
   画图链/图片重生**不挂缓存**（低频+每次变，开缓存更贵）。
🔴 OSS：本模块只产 PNG base64（人在回路展示，省 OSS spam）；入库时 FE 传 OSS（uploadMotherImage 既有）
   → A-015 image 块。本模块不耦合 OSS。
"""
from __future__ import annotations

import base64
import os
import tempfile
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from agents.figure import geogebra_samples, mathfig_render

# 🔴 单元2 标定（2026-06-18 实测）：配图默认 fig_scale=0.7（缩画布让标签相对放大；无头渲染下
#   fontSize 参数无效，figScale 是唯一杠杆——见本地引擎 geogebra/render.js:81-92 注释）。
#   标定对比（三角形/圆/坐标系各渲 1/0.7/0.6）：0.7 标签清晰可读、无碰撞且图形仍充满画布；
#   0.6 标签再大一点但图形向中心缩、边距浪费；1 标签偏小。取 0.7 为甜点。
_FIGURE_DEFAULT_SCALE = 0.7
# 🔴 单元2：默认隐圆点标记（point_size=0 只留字母标签，贴合教材原图，去多余点）；
#   非写死——若题目本要标实心点，opus 可在 spec 里显式给 point_size（>0）覆盖此默认。
_FIGURE_DEFAULT_POINT_SIZE = 0

# opus 翻 GeoGebra 命令 system 前缀（稳定，喂样例 + 命令语义坑）。画图链不挂缓存，但 prompt 仍分层清晰。
_GEO_SYSTEM = (
    "你是中小学数学配图助手。把给定的「变式题面 + 解答」翻译成一组 GeoGebra evalCommand 命令"
    "（每行一条，能被 GeoGebra Math Apps 执行渲染成题目配图）。\n"
    "🔴🔴🔴 唯一任务 = 忠实「画图」，绝不「解题」（铁律，放最前必守）：\n"
    "  你画的是**给学生做的题目图**（不是解题图、不是答案图）。必须**简洁明了、最小忠实集、宁少勿多**："
    "只画题目**给定**的几何构型（点/线/射线/必要标注）与**题目明确给出的已知**——"
    "**除必要内容一概不画、不做多余的事**。\n"
    "  🔴 **必要性闸（每画一个元素——点/线/弧/标注——前先自问，不通过就不写这条命令）**：\n"
    "     「这个是**必须**画的吗？**不画它，学生还能看懂这道题吗？**」——**能看懂，就不画。**\n"
    "     只画「不画就看不懂题目」的元素。画弧线本身没问题，但画前同样要过这道闸：这道弧必要吗？\n"
    "     —— 被平分/被分割角的各半子角弧、题目没问到的角弧，**学生不看也懂题**（平分关系两条射线已表达清楚）"
    "→ 属多余，**连弧带标度数一律不画**（不是只去掉度数、弧也不画）。\n"
    "  ⛔ **绝不标注/画出需要学生求解的角、推导出的中间角度值**（那是答案，画出来=泄题）；\n"
    "  ⛔ **被平分/被分割的角，不要把它的每一半子角画出来**（连弧都不画）——平分关系**只用射线表示**"
    "（必要时在两半边加等长刻度记号），既不给各半子角画弧、更不标度数。\n"
    "  ⛔ **不要去求解题目的答案**（不算 ∠M′NP 等于几度、不算边长、不验证结论）。\n"
    "  ⛔ **不要纠结、不要推断题面没明说的构型关系**（如「P 是否在某射线上」「某点是否共线」——"
    "题面给了角度/点/变换就照给的画，没给的别推、别脑补、别来回论证）。\n"
    "  ⛔ **不要做任何数学推导/反复推演**——题面写「绕 N 逆时针旋转 50°」就 Rotate(…,50°,N)，"
    "题面标「∠MNO=20°」就把 O 放在与 NM 夹 20° 处，到此为止。\n"
    "  ✅ 把「画图」当**纯翻译**：题面文字 → GeoGebra 命令，一一对应，不增不减、不推理。"
    "你越省去无谓推演，出图越快越准。\n"
    "🔴 强制工作流（必走三步，治「首次造图漏元素」——题面要标的角/记号别漏画）：\n"
    "  第一步·列清单：先逐一列出题面/解答里**明确提到的所有几何对象**——点、角"
    "（含「角1/角2/∠1/∠2」这类**要求标注**的角）、线段、辅助线、记号（直角/相等/平行/角度数）"
    "——形成元素清单（在脑中或注释里过一遍，不外放到 JSON）。\n"
    "  第二步·逐项落命令：保证清单里**每一项都有对应的 GeoGebra 命令或标注**"
    "（标注角用 Angle(...)，点序见下；要标的角必须有 Angle 命令，绝不只画线不标角）。\n"
    "  第三步·交付前自检：对照第一步的元素清单**逐项核对**，确认**无遗漏**"
    "（题面要的角/记号都画了）、也**无擅自新增**题面不存在的元素（忠实原图最小集——"
    "原图没有的圆点/角弧/直角小方块/辅助线一律不加）。\n"
    "🔴 命令语义坑（必守，防返工）：① 自由点直接 A=(2,3)，**别用 Point((2,3))**（会失败）；"
    "② 派生点（交点/垂足/中点/旋转像）用 Intersect/Midpoint/PerpendicularLine/Rotate/Reflect/Translate"
    "让引擎算，别手填坐标——垂足用 ClosestPoint(Line(A,E),B)（求 B 到 AE 的垂足）、"
    "三点定圆用 Circle(A,B,C)（过三点画圆）、圆心用 Center(c)、垂直平分线用 PerpendicularBisector(B,N)；"
    "③ **别在 commands 里写 SetLineStyle**（会把对象渲染没）——虚线走 dashed 字段；"
    "④ 变换题的「像」、立体隐藏棱放进 dashed；⑤ 关键点放进 vals 自核；"
    "⑦ 🔴【多余点硬约束】**派生/中间点（Intersect/Midpoint/垂足 ClosestPoint/PerpendicularBisector "
    "等构造出来、仅作中间计算、题面不要求标的点）默认必须放进 hide**——它们只是引擎算坐标的脚手架，"
    "不是题目要展示的点，漏 hide 会画出题目不需要的多余点。只有题面/解答**明确要标注的点**（顶点、"
    "圆上要点名的点、坐标系上要标的点）才留可见；其余派生点一律进 hide。"
    "（注：圆点标记默认已隐藏、只留字母标签；若某点题目本要画成实心圆点，可在 JSON 顶层给 "
    '"point_size":4 之类正值覆盖。）'
    "⑥ 🔴 标注角的点序坑：Angle(P,V,Q) 是**有向角**（V 是顶点，从 V→P 逆时针扫到 V→Q），"
    "点序须让扫角 ≤180°，否则画成反向优角（曾把直角渲成 270° 大半圆）——"
    "标 ∠BAC 写 Angle(B,A,C)（顶点 A 在中间），扫出来大于平角就把首尾两点调换。\n"
    "🔴🔴 角度标注（默认只画弧不标度数；只有题目已知的角才标，治「泄答案 + 丑小数 + 套嵌成团」根因，必守）：\n"
    "  ⓪ 【默认：Angle() 只画弧、不标度数；只有题目已知角才标，且走 angle_labels】"
    "本渲染器默认对每个 Angle(P,V,Q) **只画一道弧、不显任何度数文字**。"
    "🔴 **只有题目里明确给出（已知）的角度才标出来**——把那个角放进顶层 angle_labels 字段映射要显示的文字："
    "\"angle_labels\":{\"a1\":\"20°\",\"a2\":\"α\"}（a1 标「20°」、a2 标「α」）。"
    "🔴🔴 **绝不要标注/画弧标度数给「学生要求解的角」或「推导出的中间角度」**（那是答案，标=泄题）；"
    "**被平分/分割的角不要给各半子角各标度数**（平分只用射线表示，必要时加等长刻度记号）。"
    "🔴 **绝不要写 Text(\"20°\",(x,y)) 手放任何角度数字**（坐标全靠猜必放歪、还和标签双标）——"
    "要标的已知角走 angle_labels，其余角只画弧（或干脆不画那道弧）。"
    "同一顶点有多个角时，引擎自动把各角弧按大小**递增半径错开**，你不用操心半径。\n"
    "  ⓪″ 【角的点序——别画成反向优角/整圈】Angle(P,V,Q) 是有向角（V 顶点，V→P 逆时针扫到 V→Q）。"
    "**扫角必须 ≤180°**，否则渲成反向优角甚至近一整圈大圆（如旋转角 50° 写反点序会扫成 310° 画出大圆圈）。"
    "若某角应是锐角/钝角却扫超平角，**把首尾两点 P、Q 调换**即可。\n"
    "  ⓪′ 【angle_labels 就是「标哪些已知角」的唯一开关】只把**题目已知的角**放进 angle_labels，"
    "值写题面给的文字（实测度数 30°、或符号 α/x°——以题面为准，不要写引擎算出的丑小数）。"
    "不放进 angle_labels 的角默认不标度数（只画弧）。**该求解的角、各半子角一律不要放进 angle_labels。**\n"
    "🔴🔴 文字/公式标注（治「公式插不进、撇号点名乱」根因，必守）：\n"
    "  ⓐ 【Unicode 不 LaTeX】图里要写公式/数学符号一律用 **Unicode 字符**直接放进 Text(\"...\")——"
    "本无头渲染器**不认 LaTeX 宏**（写 Text(\"\\\\frac{1}{2}\",pt,true) 会原样印出反斜杠 \\frac 乱码，"
    "$...$ 也不行）。改用 Unicode：根号 √、角 ∠、度 °、撇号 ′、平方 ²、立方 ³、下标 ₁₂₃、"
    "平行 ∥、垂直 ⊥、全等 ≅、相似 ∽、三角形 △、乘 ×、除 ÷、正负 ±、≤ ≥ ≈ π。"
    "如 Text(\"∠1=90°\",(x,y))、Text(\"a²+b²=c²\",(x,y))、Text(\"AB∥CD\",(x,y))、"
    "Text(\"x₁+x₂=-b/a\",(x,y))——分式用 a/b 斜杠写、别用 \\frac。\n"
    "  ⓑ 【撇号点名走 relabel，别手放 Text】旋转/对称/平移的「像」点（A′/B′/C′）——GeoGebra "
    "标识符**不能含撇号**，只能命名 Ap/Bp/Cp（或 A1/A2）。要让图上显示 A′/B′，**用顶层 relabel "
    "字段**把内部点名映射到撇号显示文字：\"relabel\":{\"Ap\":\"A′\",\"Bp\":\"B′\"}。🔴 **绝不要再手放 "
    "Text(\"A'\",(x,y)) 撇号标签**——那样会和自动标签的内部名（Ap）双标打架、且坐标全靠猜必偏。"
    "relabel 让自动标签直接印 A′（位置自动算，不偏不撞）。\n"
    "🔴 只输出一个 JSON（不要解释、不要 markdown fence）：\n"
    '{"commands":["...","..."],"dashed":["对象名"],"hide":["辅助对象名"],"vals":["关键点名"],'
    '"relabel":{"Ap":"A′","Bp":"B′"},"angle_labels":{"a2":"30°"},"axes":false,"needs_figure":false}\n'
    "  （relabel 仅旋转/对称/平移有像点要标撇号时给；无撇号点的题省略此字段。"
    "angle_labels 放**题目已知、需在图上标出的角**（key=角对象名,value=显示文字）；"
    "不放的角默认只画弧不标度数——该求解的角/各半子角绝不放进来。无已知角要标时省略此字段。）\n"
    "🔴 若此题**不适合/不需要配图**（纯代数无几何意义、或你无法可靠构造），把 needs_figure 设 true、"
    "commands 留空数组（降级，不硬画错图）。\n\n"
    + geogebra_samples.samples_prompt_block()
)


def _png_to_b64(png_path: str) -> str | None:
    try:
        with open(png_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 配图人在回路「主动引导」信号（service 后处理 D14，不进四节点）
# ---------------------------------------------------------------------------
# 现状：配图画不准/画不出只被动等老师发修正词，不主动提示「补一句图形描述再画」；
# 方向元素（旋转/箭头/镜像）自报成功、无人工确认。本单元在配图返回里**附加可选信号**：
#   need_user_desc=True  → 引导老师补一句图形描述（说清要画哪些点/角/线/标注）。
#   direction_review=True → 含方向元素（旋转/箭头/镜像），引导老师确认方向是否正确。
# 🔴 信号为**可选附加字段**，FE 没读到也不崩；needs_figure 降级路径不破坏（题照常交付）。
_FIGURE_KEYWORDS = (
    "角", "三角形", "圆", "折叠", "垂直", "垂足", "平行", "旋转", "对称", "镜像",
    "数轴", "坐标", "图象", "图像", "扇形", "弧", "象限", "网格", "立体", "三视图",
    "正方体", "长方体", "棱", "梯形", "矩形", "菱形", "平行四边形", "抛物线",
)
# 🔴 PRD-A-018 bug#4（2026-06-20 用户反馈「方向待确认提示太泛」）：方向待确认**只在图里真画了
#   方向箭头**(GeoGebra Vector / 箭头符号)时才触发。旧逻辑按题面/答案/命令里「旋转/平移/镜像」关键词
#   泛触发 → 任何变换题(哪怕只出虚线像、图中无箭头)都弹「方向待确认」= 噪音(用户实测旋转题无箭头也弹)。
#   收窄判据：纯出虚线像的旋转/平移/对称题(无箭头) **不**提示；唯有 commands 里含**可见**的 Vector(...)/
#   箭头(明确画了表方向的箭头)才提示老师确认方向。判据只看 commands(图本身)，不看题面文字。
# 🔴 round3 再收窄(2026-06-20 trace 1953)：UnitVector(Vector(O,M)) 这种**隐藏辅助构造**(只为算方向、
#   放进 hide 不画出来)被旧 "vector(" 子串误判成方向箭头 → 无箭头的平分射线题也弹「方向待确认」。
#   两层收窄：① 命中前先剔除 unitvector( 与所有 Vector(...) 的内嵌实参里的 vector(，只认**真画箭头**的
#   顶层 Vector 赋值；② 该 Vector 对象若在 hide 里(辅助构造,不可见)不算箭头。
_ARROW_CMD_KEYWORDS = ("→", "箭头")
_DESC_HINT = "如需配图，请补一句图形描述（说清要画哪些点/角/线/标注），我再据此重画。"
_DIRECTION_HINT = "本图含方向箭头，请确认旋转/平移方向是否正确；如不对，补一句说明我来重画。"


def _has_keyword(text: str | None, keywords: tuple[str, ...]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any((kw.lower() in low) for kw in keywords)


def _hit_direction(
    stem: str | None, answer: str | None, commands: Any, hide: Any = None,
) -> bool:
    """方向待确认命中：**仅当 commands 里真画了「可见的」方向箭头**(顶层 Vector(...) 且未隐藏 / 箭头符号)
       时返 True。
    🔴 bug#4 收窄（2026-06-20）：旧版按「旋转/平移/镜像」关键词泛触发，任何变换题(哪怕只出虚线像)都弹噪音。
    🔴 round3 再收窄（trace 1953）：旧 "vector(" 子串把 UnitVector(Vector(O,M)) 这类**隐藏辅助构造**误判成
       箭头 → 无箭头的平分射线题也弹「方向待确认」。现三层收窄，只认**真画出来的箭头对象**：
         ① 只认**顶层 Vector(...) 赋值**（RHS 以 Vector( 开头，如 v=Vector(A,B)）——嵌在别的函数实参里的
            vector(（如 UnitVector(Vector(O,M)) 的内层、或 +Vector(...) 运算项）一律不算（那不是独立箭头对象）；
         ② 排除 UnitVector（不是箭头）；
         ③ 该 Vector 对象名若在 hide 里(辅助构造、不可见)不算箭头。"""
    if not isinstance(commands, list):
        return False
    hide_set = {str(h).strip() for h in hide} if isinstance(hide, list) else set()
    for c in commands:
        s = str(c)
        # 箭头符号直接命中（少见，opus 一般不会塞）
        if "→" in s or "箭头" in s:
            return True
        # 只认「name=Vector(...)」形态（RHS 顶层就是 Vector 调用）= 真画出来的独立箭头对象。
        if "=" not in s:
            continue
        name, rhs = s.split("=", 1)
        name = name.strip()
        rhs = rhs.strip().lower()
        if rhs.startswith("vector(") and (name not in hide_set):
            return True
    return False


def _append_reason(base: str | None, extra: str) -> str:
    base = (base or "").strip()
    return f"{base} {extra}".strip() if base else extra


async def compose_variant_figure(
    *,
    stem: str,
    answer: str | None = None,
    invoke: Any,
    parse_json: Any,
    correction_prompt: str | None = None,
    prev_commands: list[str] | None = None,
    item_id: str | None = None,
    model: str | None = None,
    figure_spec: str | None = None,
) -> dict[str, Any]:
    """变式造图一轮直出（+ 图片重生）。返回
       {item_id, ok, png_base64?, commands, dashed, vals, needs_figure, warnings, reason?}。

    🔴 invoke = variant._ainvoke_text（落 conv_trace label=figure_geogebra）；parse_json = variant._parse_json。
    🔴 correction_prompt 非空 = 图片重生（老师修正提示词，单图重造，人在回路 D12）。
    🔴 PRD-C-100 C：图片重生带上一版 GeoGebra commands（prev_commands）+ 原题上下文 →
       opus 在「上一版配图命令的基础上按修正要求调整」（增量修改，而非从零重画，保证继承上一版）。
       仅当 correction_prompt 与 prev_commands 同时非空才走增量分支；首次造图（无 prev_commands）维持原行为。
    🔴 PRD-A-018 治本A·figure_spec（出题节点产的「画什么」自然语言配图描述）：非空时作为**权威画图依据**
       喂给 opus —— opus 只需「照 figure_spec 翻成 GeoGebra 命令」，不再现场从题面**逆推**几何构型
       （根治「逆推歧义」+ 大降耗时）。figure_spec 为空/缺省 → 退回原行为（从 stem/answer 现推），向后兼容。
       图片重生（correction_prompt 在）时 figure_spec 仍作为「这道图本该是什么」的底图描述一并喂入。
    🔴 任何失败 → needs_figure=True（降级，不抛、不掐流程 G11）。
    """
    # 🔴 B5 预算护栏（G7）：当日花费超阈值 → 造图降级（needs_figure，不调翻命令，题照常交付）。
    from agents import cost_guard
    # 🔴 P5：护栏读库丢线程池（async 版），慢库不卡 asyncio loop / 不拖垮并发 SSE。
    if await cost_guard.is_budget_exceeded_async():
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": "今日 AI 额度已用尽，配图暂缓（可明日重试或手动配图）", "commands": [], "warnings": []}

    user_segs = [f"【变式题面】\n{stem}"]
    if answer:
        user_segs.append(f"【标准答案/解答】\n{answer}")
    # 🔴 PRD-A-018 治本A：figure_spec 非空 = 出题节点已把「这道图画什么」想清楚（自然语言描述）。
    #   把它作为**权威画图依据**放在最前位重的段——指令 opus「照这份描述翻 GeoGebra，别再从题面逆推
    #   构型」。题面/答案仍附在后供消歧（如描述里某点坐标含糊时回看题面），但「画什么」以 spec 为准。
    spec_clean = (figure_spec or "").strip()
    if spec_clean:
        user_segs.append(
            "【🔴 配图描述（权威·画什么以此为准）】\n"
            "下面是出题时（上下文最全）已经写好的本图自然语言描述——你的任务是**照这份描述忠实翻译成 "
            "GeoGebra 命令**，不要再从题面逆推几何构型、不要纠结题面没明说的关系。描述里说画哪些点/角/"
            "度数/旋转平移对称/虚线，就照画哪些；题面与答案附在上方仅供消歧参考。\n"
            f"{spec_clean}"
        )
    # 🔴 PRD-C-100 C：图片重生 = correction_prompt + prev_commands 都在 → 增量修改（带上一版命令 + 原题上下文）。
    #   把上一版 GeoGebra commands 原样喂回，指令改成「在下面这版配图命令的基础上，按修正要求调整」，
    #   让 opus 继承上一版骨架做增量改动，而非丢掉上下文从零重画（跑偏的根因）。
    incremental_regen = bool(correction_prompt and prev_commands)
    if incremental_regen:
        prev_block = "\n".join(str(c) for c in (prev_commands or []) if str(c).strip())
        user_segs.append(f"【上一版配图 GeoGebra 命令（基准，在此之上调整）】\n{prev_block}")
        user_segs.append(
            "【老师修正要求（🔴 在上面这版配图命令的基础上，按本要求调整——而非从零重画，"
            f"保留与修正无关的部分、只改需要改的）】\n{correction_prompt}"
        )
    elif correction_prompt:  # 图片重生但无上一版命令（首版即降级 / 老会话）→ 退回原行为（从修正词重画）
        user_segs.append(f"【老师修正要求（按此重新构造配图）】\n{correction_prompt}")
    messages = [
        SystemMessage(content=_GEO_SYSTEM),
        HumanMessage(content="\n\n".join(user_segs)),
    ]
    # opus 翻命令（走 _ainvoke_text 落 conv_trace；label 由 system 头 "GeoGebra evalCommand" 决定，
    # B1b trace marker 已加 figure_geogebra 钩子见 variant._TRACE_MARKERS 同批补）。
    try:
        # model 默认 None → 调用方（service 端点）传 settings.VARIANT_MODEL_FIGURE（opus）；
        #   传了 model 才走 per-call 覆盖（温度 0.2，timeout 留默认；造图链不挂缓存）。
        text = await invoke(messages, model=model, max_tokens=4096, temperature=0.2)
    except Exception as e:  # noqa: BLE001 — opus 翻命令失败 → needs_figure 降级
        return {"item_id": item_id, "ok": False, "needs_figure": True,
                "reason": f"opus 翻命令失败: {str(e)[:80]}", "commands": [], "warnings": []}

    data = parse_json(text)
    if not isinstance(data, dict) or data.get("needs_figure") is True or not data.get("commands"):
        # 🔴 主动引导（触发条件 1+2）：opus 自评 needs_figure / 给的 commands 为空。
        #   只要题面/答案含图形关键词（说明该题本应有图）→ 标 need_user_desc + 引导老师补描述。
        #   纯代数无几何意义题（无图形关键词）→ 不主动催（避免对不需配图的题误提示）。
        cmds = (data or {}).get("commands", []) if isinstance(data, dict) else []
        want_fig = _has_keyword(stem, _FIGURE_KEYWORDS) or _has_keyword(answer, _FIGURE_KEYWORDS)
        # 🔴 PRD-A-018 RED#1：needs_figure 真值 = 本题「本应有图」(含几何关键词 want_fig)。
        #   纯代数/无几何意义题 opus 判「不适合配图」→ want_fig=False → needs_figure=False
        #   （= 无需配图，非「待补图」）。FE 据此把该题配图灯跳过(done)、不污染配图节点成「异常·待补图」、
        #   不挂住题组就绪（治 A-018 C2 把就绪绑配图后纯代数题就绪被假阳性配图卡死的回归）。
        #   仅 want_fig=True（本应有图却没画出）才 needs_figure=True + need_user_desc 引导补描述。
        out: dict[str, Any] = {
            "item_id": item_id, "ok": False, "needs_figure": bool(want_fig),
            "reason": "opus 判定不适合配图或未给命令", "commands": cmds, "warnings": [],
        }
        if want_fig:
            out["need_user_desc"] = True
            out["reason"] = _append_reason(out["reason"], _DESC_HINT)
        return out

    # 🔴 PRD-C-100 B3·渲染异常兜住（D9 图失败照常交付 / G11 不卡流程）：mathfig_render.render
    #   本应自兜异常返 ok:False，但作为「造图后处理」最后一道闸，这里再加一层 try/except——任何
    #   渲染层冒泡（子进程崩/PNG IO/未预期异常）一律降级 needs_figure，绝不让带图变式整条 stream
    #   被一道图的渲染异常掐死（变式正文先出、图作后处理，图失败照常交付带⚠待补图）。
    # 🔴 单元2：默认放大标签（fig_scale=0.7）+ 隐多余圆点（point_size=0，只留字母）；
    #   二者均可被 opus 在 spec 显式给值覆盖（如题目本要标实心点 → point_size>0），不写死。
    fig_scale = data.get("fig_scale")
    fig_scale = float(fig_scale) if fig_scale else _FIGURE_DEFAULT_SCALE
    point_size = data.get("point_size")
    point_size = float(point_size) if point_size is not None else _FIGURE_DEFAULT_POINT_SIZE
    # 🔴 relabel（{内部点名:"显示文字"}，如 {"Ap":"A′"}）：opus 把旋转/对称像点（命名 Ap/Bp，
    #   GeoGebra 标识符不能含撇号）映射到数学正确的撇号显示文字 A′/B′。让自动标签印 A′/B′
    #   而非内部名 Ap/Bp，opus 不必再手放 Text 撇号标签（治「Ap 与 A′ 双标签打架」根因）。
    relabel = data.get("relabel")
    relabel = dict(relabel) if isinstance(relabel, dict) else None
    # 🔴 angle_labels（{角对象名:"显示文字"}）：round3 默认翻转——Angle() 默认**只画弧、不标度数**
    #   （配图=给学生做的题，不泄答案、不冒丑小数）。**只有题目已知的角**才放进 angle_labels 标出（如
    #   {"a1":"20°"}）；不放的角无标签。该求解的角/被平分各半子角绝不放进来。同顶点多角弧仍自动递增半径错开。
    angle_labels = data.get("angle_labels")
    angle_labels = dict(angle_labels) if isinstance(angle_labels, dict) else None
    try:
        r = mathfig_render.render(
            list(data.get("commands") or []),
            dashed=data.get("dashed"), hide=data.get("hide"), vals=data.get("vals"),
            axes=bool(data.get("axes")), stem=f"variant_{item_id or 'fig'}",
            fig_scale=fig_scale, point_size=point_size, relabel=relabel,
            angle_labels=angle_labels,
        )
    except Exception as e:  # noqa: BLE001 — 渲染冒泡 → needs_figure 降级（不抛、不卡流程）
        # 触发条件 3·渲染失败 → 引导补描述/重试（need_user_desc）。
        return {"item_id": item_id, "ok": False, "needs_figure": True, "need_user_desc": True,
                "reason": _append_reason(f"造图异常: {str(e)[:80]}", _DESC_HINT),
                "commands": data.get("commands"), "warnings": []}
    if not r.get("ok") or not r.get("png_path"):
        return {"item_id": item_id, "ok": False, "needs_figure": True, "need_user_desc": True,
                "reason": _append_reason(r.get("error") or "渲染失败", _DESC_HINT),
                "commands": data.get("commands"), "warnings": r.get("warnings", [])}
    b64 = _png_to_b64(r["png_path"])
    if not b64:
        return {"item_id": item_id, "ok": False, "needs_figure": True, "need_user_desc": True,
                "reason": _append_reason("PNG 读取失败", _DESC_HINT),
                "commands": data.get("commands"), "warnings": []}
    # 成功路径：触发条件 4·方向元素 → 标 direction_review + 引导老师确认方向（图照常交付）。
    out: dict[str, Any] = {
        "item_id": item_id, "ok": True, "needs_figure": False, "png_base64": b64,
        "commands": data.get("commands"), "dashed": data.get("dashed"),
        "vals": r.get("vals", {}), "warnings": r.get("warnings", []),
    }
    if _hit_direction(stem, answer, data.get("commands"), data.get("hide")):
        out["direction_review"] = True
        out["reason"] = _DIRECTION_HINT
    return out


async def crop_mother_figure(
    image_url: str, *, conf: float = 0.2,
) -> dict[str, Any]:
    """母题切图（B3.1）：下载母题图 → figure-crop 检测+裁 → 所有 figure 的 PNG base64。
       返回 {ok, figures:[{png_base64,bbox,conf},...], png_base64?, bbox?, conf?, n_figures, needs_figure, reason?}。
    🔴 纯文字/公式题（figures=[]）→ needs_figure=False + ok=False（母题无图，正常，不降级告警）。
    🔴 任何 IO/模型失败 → needs_figure=True（降级，不抛）。
    🔴 单元3（PRD-C-100 bug-002 三轮，升方案 a「真切全部图」）：多图母题不再只回 figs[0]——
       遍历全部 figs 各转 base64，回 `figures` 数组（每项 {png_base64,bbox,conf}）。
       **同时保留 `png_base64`/`bbox`/`conf` = figs[0]（兼容老 FE：未升多图渲染前老 FE 仍读这俩字段不崩）**，
       并保留 `n_figures` 真实张数。新 FE 消费 `figures` 数组做 v-for 多图渲染。
       某张读盘失败则该项跳过（不整体降级，只要至少 1 张成功就 ok=True）；全部读失败才 needs_figure。
    """
    tmp_path = None
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(resp.content)
        from agents.figure.figure_crop import detect_and_crop  # 进程内 import（torch 懒加载）
        det = detect_and_crop(tmp_path, conf=conf)
        figs = det.get("figures") or []
        if not figs:
            return {"ok": False, "needs_figure": False, "n_figures": 0, "figures": [],
                    "reason": "母题无图形（纯文字/公式题）"}
        n = len(figs)
        # 单元3：遍历全部图各转 base64（某张读盘失败则跳过，不整体降级）。
        out_figs: list[dict[str, Any]] = []
        for f in figs:
            b64 = _png_to_b64(f.get("crop_path"))
            if not b64:
                continue
            out_figs.append({"png_base64": b64, "bbox": f.get("bbox"), "conf": f.get("conf")})
        if not out_figs:  # 全部读失败才降级
            return {"ok": False, "needs_figure": True, "n_figures": n, "figures": [],
                    "reason": "切图 PNG 读取失败"}
        # 部分图读失败时外显提示（切出张数 < 检出张数）。
        n_ok = len(out_figs)
        reason = (
            f"本题检出 {n} 图、成功切出 {n_ok} 张；如缺图可手动重切"
            if n_ok < n else None
        )
        first = out_figs[0]
        return {
            "ok": True, "needs_figure": False, "n_figures": n,
            "figures": out_figs,                       # 新契约：多图数组（FE v-for）
            "png_base64": first["png_base64"],         # 兼容老 FE：=figs[0]
            "bbox": first.get("bbox"), "conf": first.get("conf"),
            "reason": reason,
        }
    except Exception as e:  # noqa: BLE001 — 下载/检测失败 → 降级
        return {"ok": False, "needs_figure": True, "figures": [], "reason": f"母题切图失败: {str(e)[:80]}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
