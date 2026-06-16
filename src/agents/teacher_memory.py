# -*- coding: utf-8 -*-
"""PRD-C-100 B4 全局记忆层 · toolkit 侧（注入 + 自动写，全走 RuoYi HTTP，不直连 MySQL）。

🔴 铁律：数据归 RuoYi、Python 不直连 MySQL（记忆走 HTTP，RuoyiClient.list_ai_memory/add_ai_memory）。
🔴 简单优先（D16）：纯确定性直存（confirm改章/改DNA/入库 按规则写），不 LLM 提炼（推下轮）。
🔴 多用户接缝（D7/D16）：记忆是 per-teacher → build prompt 时放**变量后缀**（build_entry_messages 的
   teacher_memory 槽），绝不进缓存稳定前缀（进前缀毁多用户共享）。
🔴 注入只用 enabled 记忆（停用不注入，G14）；全链路 best-effort（记忆故障绝不卡 mother 主流程）。
"""
from __future__ import annotations

from typing import Any

_TYPE_LABEL = {"偏好": "偏好", "纠正": "纠正", "习惯": "习惯"}


def format_memory_block(rows: list[dict[str, Any]]) -> str | None:
    """enabled 记忆行 → prompt 注入块（放变量后缀）。空 → None（不注入空块）。"""
    rows = [r for r in (rows or []) if isinstance(r, dict) and r.get("memValue")]
    if not rows:
        return None
    # 仅注入 enabled（端点 enabledOnly 已过滤；双保险再过一次）
    rows = [r for r in rows if r.get("enabled") in (1, True, None)]
    if not rows:
        return None
    lines: list[str] = []
    for r in rows:
        t = _TYPE_LABEL.get(str(r.get("memType") or ""), "记忆")
        key = str(r.get("memKey") or "").strip()
        val = str(r.get("memValue") or "").strip()
        lines.append(f"- [{t}] {key}：{val}" if key else f"- [{t}] {val}")
    return "\n".join(lines)


async def fetch_memory_block(client: Any) -> str | None:
    """拉该老师 enabled 记忆 → 注入块。client = RuoyiClient（带老师 token）。失败 → None（降级）。"""
    try:
        rows = await client.list_ai_memory(enabled_only=True)
    except Exception:  # noqa: BLE001
        return None
    return format_memory_block(rows)


# === 确定性自动写（D10：confirm改章/改DNA/入库 各按规则写）=================
async def write_correction_grade_chapter(
    client: Any, *, grade_book: str | None, chapter: str | None,
) -> None:
    """confirm 改章/纠正年级 → 写「纠正」记忆（确定性）。best-effort。"""
    if grade_book:
        await client.add_ai_memory(
            mem_type="纠正", mem_key="常教年级册", mem_value=grade_book, source="自动")
    if chapter:
        await client.add_ai_memory(
            mem_type="纠正", mem_key="常确认章", mem_value=chapter, source="自动")


async def write_preference_on_persist(
    client: Any, *, grade_book: str | None, qtype: str | None,
) -> None:
    """入库 → 写「偏好」记忆（常教年级/常出题型，确定性）。best-effort。"""
    if grade_book:
        await client.add_ai_memory(
            mem_type="偏好", mem_key="常教年级册", mem_value=grade_book, source="自动")
    if qtype:
        await client.add_ai_memory(
            mem_type="偏好", mem_key="常出题型", mem_value=qtype, source="自动")
