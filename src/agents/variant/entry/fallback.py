"""variant 引擎 · entry/fallback.py 兜底节点（PRD-C-104 B5 抽出，纯搬零改）。

route_entry 各兜底分支落点 + 低置信前置拦截 + 没定死回问：
- entry_lowconf_block：R2a·闸4·读图低置信前置拦截（不进 classify、不烧 opus token）。
- clarify：母题没定死 → 回问老师确认（只问不造，进 WAIT 等下一句）。
- require_login：route_entry 'auth' 分支落点（登录态缺失拒入图）。
- ask_for_image：route_entry 'ask' 分支落点（首轮无图无母题无题组，催贴题图）。

🔴 行为零改：提示文案/状态置位/阶段灯逐字保持。图 wiring 节点名（clarify/entry_lowconf_block/
   require_login/ask_for_image）不变 → 拓扑零改。
🔴 strangler：顶部 from agents.variant import 取依赖（_emit_stage/_pin_status/常量），运行期解析。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾导入）
    CONF_GATE,
    STAGE_AWAIT,
    VariantState,
    _emit_stage,
    _pin_status,
)


async def entry_lowconf_block(state: VariantState, config: RunnableConfig) -> VariantState:
    """🔴 PRD-A-021 R2a·闸4（BUG-04）·读图低置信前置闸（resume 轮·classify 之前）：母题读图置信
    极低（< 0.40）/ 章未判出 → 拦截一次，建议老师换张清晰的图，**不进 classify、不烧 opus token**。

    🔴 一次性拦截（防永久卡死）：置 _lowconf_blocked=True。老师若坚持（再回传 confirmed_chapter_id），
       route_entry 见 _lowconf_blocked=True → _should_lowconf_block False → 放行进 classify 正常出题。
    🔴 阶段灯中性 await（不是 warn/已中断）：这是「建议换图」的暂停，不是流程出错。
    🔴 保持 awaiting_mother_confirm=True，让老师下一句（坚持确认 / 换图 URL）能继续被 route 接住。
    """
    dec = state.get("entry_decision") if isinstance(state.get("entry_decision"), dict) else {}
    try:
        conf = float((dec or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    _emit_stage("classify", "锚定考点", STAGE_AWAIT,
                "这张图可能不适合做母题·建议换张清晰的图")
    _emit_stage("knobs", "解析配方", STAGE_AWAIT, "待换图或确认后定配方")
    body = (
        "⚠ 我对这张图的读图把握很低"
        f"（置信约 {conf:.2f}{'、且没判出具体章' if not str((dec or {}).get('chapter') or '').strip() else ''}）"
        "——**可能这张图不太适合做母题**（拍得不清 / 不是标准题图 / 含大量图形）。\n\n"
        "建议：\n"
        "- **换一张更清晰的题目图**（直接贴新图的 OSS URL，我重新读）；\n"
        "- 若你确认就用这张图、按你选的章继续 → **再回复一次「确认」**，我照常出变式。"
    )
    return {
        # 拦过一次（坚持再确认即放行，不二次拦）。awaiting_mother_confirm 保持 True 让 resume 续接。
        "_lowconf_blocked": True,
        "awaiting_mother_confirm": True,
        "awaiting_mother_review": False,
        "messages": [AIMessage(content=body)],
    }


async def clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """没定死 → 回问老师确认（只问不造，进 WAIT 等下一句）。

    🔴 批2：确认态如实回报「年级学期（推断值或未识别）+ 主考点（锚值+所属册 或 未锚定）」，
    复用既有 clarify 聊天气泡协议（book-ui 既有确认/chip 修改能力，不动 book-ui）。
    """
    analysis = state.get("analysis") or {}
    pin = _pin_status(state)
    g = analysis.get("grade") or {}
    k = analysis.get("kp") or {}
    q = analysis.get("qtype") or {}

    # 状态回报：年级学期 + 主考点（锚值+册 或 未锚定）—— 让老师一眼看到缺哪一项
    grade_line = (
        f"年级学期：**{pin['grade_text']}**（已识别）"
        if "grade" not in pin["reasons"] and pin["grade_text"]
        else f"年级学期：**未识别**（看着像「{g.get('value') or '?'}」，请确认是几年级上/下学期）"
    )
    if "kp" not in pin["reasons"] and pin["kp_name"]:
        book = f"·{pin['kp_book']}" if pin["kp_book"] else ""
        kp_line = f"主考点：**{pin['kp_name']}**（已锚定{book}）"
    else:
        kp_line = f"主考点：**未锚定**（粗看是「{k.get('value') or '?'}」，请确认或指正考点）"

    asks: list[str] = []
    if "grade" in pin["reasons"]:
        asks.append(f"年级我没定死（看着像「{g.get('value') or '?'}」），请告诉我是几年级上/下学期？")
    if "kp" in pin["reasons"]:
        asks.append(f"核心考点我没锚准（粗看是「{k.get('value') or '?'}」），对吗？或请指正。")
    if (
        "confidence" in pin["reasons"]
        and float(q.get("confidence", 0) or 0) < CONF_GATE
    ):
        _qtype_guess = q.get('value') or "新定义/非常规题，按解答处理可以吗？"
        asks.append(f"题型我没把准（看着像「{_qtype_guess}」），对吗？")
    if not asks:
        asks.append("我对母题 DNA 还不够确定，请确认下年级/考点/题型再继续。")

    body = (
        "我得先把母题**定死**才能造变式（年级 + 主考点缺一不可）。当前状态：\n\n"
        f"- {grade_line}\n- {kp_line}\n\n"
        "请补充/纠正：\n" + "\n".join(f"- {a}" for a in asks)
    )
    return {"messages": [AIMessage(content=body)]}


# --- 输入边界兜底（设计 §6）：没图/无在途母题/无题组 → 催图 ------------------
async def require_login(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'auth' 分支落点：登录态缺失 → 拒入图（teacher_id 绑死硬闸的提示面）。"""
    return {
        "messages": [
            AIMessage(
                content=(
                    "🔒 登录态缺失或已过期，举一反三需要绑定到你的账号才能使用"
                    "（对话记录与入库的题都归属到你本人）。请重新登录平台后再试。"
                )
            )
        ]
    }


async def ask_for_image(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'ask' 分支落点：首轮无图无母题无题组，催老师贴题图。

    🔴 必须是真节点（不能直连 END）—— 否则首轮没有任何节点产消息，回复为空，
    '没图催' 提示从未触发（PRD-C-009 G15 红的 root cause）。
    """
    return {
        "messages": [
            AIMessage(
                content=(
                    "我还没看到题目图。请先贴一张题目图的 OSS URL，我才能开始举一反三。\n\n"
                    "（贴图后我会读图、锚定年级/考点/题型，再按你要的数量出变式题。）"
                )
            )
        ]
    }
