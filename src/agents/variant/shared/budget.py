"""variant 引擎 · LLM 调用预算闸（PRD-C-104 B2 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出（约 86-137 行）：
  - _budget_ctx（contextvar 计数器）
  - _budget_begin / _budget_tick / _budget_exhausted / _budget_bind

🔴 行为零改：内容逐字搬，仅补本模块所需 import（contextvars + VariantState 类型注解）。
   `from __future__ import annotations` 让注解变字符串，VariantState 仅作类型提示无运行期副作用。
   __init__.py 顶部 re-export 这些符号 → 各调用点零感。
"""

from __future__ import annotations

import contextvars

from agents.variant.state import VariantState

# ---------------------------------------------------------------------------
# P13 预算闸（PRD-C-013）：state 级 LLM 调用计数器（per-round 重置）。
# 🔴 铁律：是「超限跳过增强类调用」不是「LLM 决定流程」——宏观 DAG 一字不破。
#   核心链（parse/generate 首稿/grade 难度总评/solve 真解）永不跳；只有**增强类**调用
#   （闸A rework 回炉 / 闸B heal 回炉 / replenish 补题 / extract 兜底抽载荷）在超限后
#   跳过，落既有 G5 降级路径（标 ⚠ / 保留原题，绝不卡死）。
# 实现：contextvar 持一个 {"used":int,"limit":int} 计数器（per graph round 由出题/编辑节点
#   入口 _budget_begin 重置）。_ainvoke_text 每次成功调用 _budget_tick()+1；增强类调用点
#   先问 _budget_exhausted() 再决定跳不跳。contextvar 天然随 asyncio task 复制传播 →
#   eager 并发子 task / gather 并发都共享同一计数器（同一轮预算），单测直调节点（无
#   begin）时 _budget 为 None → 永不超限（行为回退到老逻辑，零侵入）。
# ---------------------------------------------------------------------------
_budget_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "variant_llm_budget", default=None
)


def _budget_begin(limit: int) -> None:
    """出题/编辑轮入口重置预算（per-round）。limit≤0 视为不设限（关闸）。"""
    _budget_ctx.set({"used": 0, "limit": int(limit)} if limit and limit > 0 else None)


def _budget_tick() -> None:
    """记一次成功 LLM 调用（_ainvoke_text 内部唯一调用点）。无预算上下文 → no-op。"""
    b = _budget_ctx.get()
    if b is not None:
        b["used"] += 1


def _budget_exhausted() -> bool:
    """增强类调用点的闸：True=预算已耗尽，本次增强调用应跳过走降级。无预算 → 永 False。"""
    b = _budget_ctx.get()
    return b is not None and b["used"] >= b["limit"]


def _budget_bind(state: VariantState, *, reset_limit: int | None = None) -> dict | None:
    """节点入口绑定预算到 contextvar，返回 live 计数器 dict（节点须把它放回返回值 state，
    used 才能跨节点累计——LangGraph 每个 superstep 用新 copy_context，contextvar 不跨节点存活，
    预算的**事实源是 state.llm_call_budget**，contextvar 只是给无 state 视野的 _ainvoke_text 记账）。

    - reset_limit 非 None（出题/编辑轮**入口**节点）→ 本轮重置 {"used":0,"limit":reset_limit}；
      limit≤0 视为关闸（返回 None，永不超限）。
    - reset_limit=None（轮内下游节点 gene_gate/solve_explain/assemble/exec_*）→ 从 state 携带；
      state 无簿记（单测直调 / 旧线程恢复）→ None（关闸，行为回退老逻辑）。
    """
    if reset_limit is not None:
        b = {"used": 0, "limit": int(reset_limit)} if reset_limit > 0 else None
    else:
        carried = state.get("llm_call_budget")
        b = dict(carried) if isinstance(carried, dict) and "limit" in carried else None
    _budget_ctx.set(b)
    return b
