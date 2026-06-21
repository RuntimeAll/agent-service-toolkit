# -*- coding: utf-8 -*-
r"""PRD-A-021 R3b·章节×图型定型闸：按母题章节/考点取「允许图型集」约束 opus 造图。

治「数轴乱画 / 图型有限」根因 —— 造图链全程零类型约束。方案 = 建「章节 → 允许图型集」
定型闸：compose 先据母题章节取允许图型集，让 opus **只在该型集内**翻 GeoGebra 命令。

数据源 = 表 `biz_chapter_figure_map`（dev 库 miskt_data2，book-server 建；本模块**纯只读 ETL**，
  与 model_anchor.lookup_candidates 同精神，是架构允许的唯一直连例外）：
  - chapter_keyword varchar：章节/考点主题关键词
  - figure_type     varchar：图型 code（见 KNOWN_FIGURE_TYPES）
  多对多（一 keyword 多行）。

读法（allowed_figure_types）：
  取母题**章节名 + 考点名**两路文本，对每行 chapter_keyword 做**包含匹配**（章节/考点名 contains
  keyword）→ 命中行的 figure_type 并集 = 允许图型集。进程内缓存全表（小表，首次查时加载一次）。

🔴 逃生窗口（PRD-A-021 用户拍板，必守）：
  - 章节**无匹配图型**（允许集为空 / 取不到章节/考点）→ **不约束**，回退 opus 自由翻命令。
  - 表读不到（dev 库还没 apply 迁移 / 库未起 / 表不存在）→ 缓存记 unavailable，allowed 一律空集
    → 全部走逃生（绝不因没映射就画不出图）。
🔴 误命中防护：包含匹配天然有「圆 ⊂ 圆柱」类误命中风险——首批从简（直接子串包含），
  误命中风险记进交付物，靠 seed 关键词选词 + 后续精化（首版不做分词/边界匹配）。
"""
from __future__ import annotations

import logging
from typing import Any

import pymysql

from core import settings

logger = logging.getLogger(__name__)

# 已知图型 code（与 seed/book-server 对齐；仅作人读参照 + 可选校验，不强校验防新增图型被本模块挡掉）。
KNOWN_FIGURE_TYPES: frozenset[str] = frozenset({
    "number_line", "cartesian", "line_func", "parabola", "hyperbola",
    "triangle", "congruent_pair", "quadrilateral", "circle", "angles_lines",
    "transform", "solid", "stat_chart", "ruler_compass",
})

# 图型 code → 人读中文名（拼进 prompt 给 opus 看，比裸 code 易懂）。
_FIGURE_TYPE_CN: dict[str, str] = {
    "number_line": "数轴",
    "cartesian": "平面直角坐标系",
    "line_func": "一次函数图象",
    "parabola": "抛物线/二次函数图象",
    "hyperbola": "反比例函数图象",
    "triangle": "三角形",
    "congruent_pair": "全等/相似三角形对",
    "quadrilateral": "四边形",
    "circle": "圆",
    "angles_lines": "角与直线（相交线/平行线）",
    "transform": "图形变换（平移/旋转/对称/折叠）",
    "solid": "立体图形/三视图",
    "stat_chart": "统计图表",
    "ruler_compass": "尺规作图",
}

# ---------------------------------------------------------------------------
# 进程内缓存：全表 [(keyword, figure_type), ...]。None = 未加载；[] = 加载过但表空/不可用。
# unavailable 标记区分「表真空」与「读不到」（两者都走逃生，但日志/语义不同）。
# ---------------------------------------------------------------------------
_MAP_CACHE: list[tuple[str, str]] | None = None
_MAP_UNAVAILABLE: bool = False


def _db_kwargs() -> dict:
    """复用 model_anchor / variant_support 的同一只读连接配置（dev 库 miskt_data2 @ :3307）。"""
    pwd = settings.VARIANT_DB_PASSWORD
    return dict(
        host=settings.VARIANT_DB_HOST,
        port=settings.VARIANT_DB_PORT,
        user=settings.VARIANT_DB_USER,
        password=pwd.get_secret_value() if pwd else "",
        database=settings.VARIANT_DB_NAME,
        charset="utf8mb4",
    )


def _load_map() -> list[tuple[str, str]]:
    """加载全表到进程内缓存（纯只读 ETL）。返回 [(keyword_lower, figure_type), ...]。

    🔴 任何异常（库未起 / 表不存在 / 网络）→ 记 _MAP_UNAVAILABLE=True + 返回 []（逃生，不抛）。
       表存在但为空 → 缓存 []（unavailable=False，语义「有表无映射」，同样逃生）。
    """
    global _MAP_CACHE, _MAP_UNAVAILABLE
    if _MAP_CACHE is not None:
        return _MAP_CACHE
    rows: list[tuple[str, str]] = []
    try:
        conn = pymysql.connect(**_db_kwargs())
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT chapter_keyword, figure_type FROM biz_chapter_figure_map"
            )
            for kw, ft in cur.fetchall():
                kw_s = str(kw or "").strip()
                ft_s = str(ft or "").strip()
                if kw_s and ft_s:
                    rows.append((kw_s.lower(), ft_s))
        finally:
            conn.close()
        _MAP_UNAVAILABLE = False
        _MAP_CACHE = rows
        logger.info("chapter_figure_map loaded: %d rows", len(rows))
    except Exception as e:  # noqa: BLE001 — 表未 apply / 库故障 → 逃生（不约束）
        _MAP_UNAVAILABLE = True
        _MAP_CACHE = []
        logger.info("chapter_figure_map unavailable, escape (no constraint): %s", str(e)[:120])
    return _MAP_CACHE


def reset_cache() -> None:
    """清进程内缓存（单测/热更用；生产长驻进程不调）。"""
    global _MAP_CACHE, _MAP_UNAVAILABLE
    _MAP_CACHE = None
    _MAP_UNAVAILABLE = False


def _seed_cache_for_test(rows: list[tuple[str, str]]) -> None:
    """单测注入桩表（绕真 DB）。rows = [(keyword, figure_type), ...]。"""
    global _MAP_CACHE, _MAP_UNAVAILABLE
    _MAP_CACHE = [(str(k).strip().lower(), str(v).strip()) for k, v in rows if str(k).strip() and str(v).strip()]
    _MAP_UNAVAILABLE = False


def allowed_figure_types(chapter_name: str | None, kp_name: str | None = None) -> set[str]:
    """定型闸核心：母题章节名 + 考点名 → 允许图型集（figure_type code 并集）。

    匹配 = **包含匹配**：对每行 chapter_keyword，若 (章节名 + 考点名) 拼起来的文本里**包含**该
      keyword（子串），则该行 figure_type 入并集。大小写不敏感（统一 lower）。

    🔴 逃生窗口（返回空集即「不约束」）：
      - chapter_name 与 kp_name 都空 → 空集。
      - 表不可用 / 表空 → 空集。
      - 有文本但无 keyword 命中 → 空集。
    上层据「空集 = 自由发挥（不拼约束）」处置，绝不因没映射而画不出图。
    """
    text = " ".join(
        s.strip().lower() for s in (chapter_name, kp_name) if s and str(s).strip()
    )
    if not text:
        return set()
    rows = _load_map()
    if not rows:
        return set()
    allowed: set[str] = set()
    for kw, ft in rows:
        if kw and kw in text:
            allowed.add(ft)
    return allowed


def constraint_clause(allowed: set[str]) -> str:
    """把允许图型集渲成可拼进造图 prompt 的「定型约束」段（纯函数·可单测）。

    🔴 逃生：allowed 为空 → 返回 ""（不拼任何约束，opus 自由翻命令）。
    非空 → 列出允许图型（中文名 + code），要求 opus **只在这些图型内**选择并翻命令。
    🔴 留自由度（数轴别画死）：约束的是「选哪类图」，不规定该类图里具体怎么画——
       opus 在允许图型内仍自由构造命令。
    """
    if not allowed:
        return ""
    # 稳定排序（让 prompt 文本可复现、护缓存）。
    names = []
    for ft in sorted(allowed):
        cn = _FIGURE_TYPE_CN.get(ft)
        names.append(f"{cn}（{ft}）" if cn else ft)
    return (
        "\n\n【🔴 本题章节定型约束（按母题章节/考点限定可画图型，治数轴乱画/图型乱选）】\n"
        "本题章节/考点允许的配图图型**仅限**以下几类：" + "、".join(names) + "。\n"
        "🔴 你**只能在这些图型范围内**选择并翻成 GeoGebra 命令，**绝不画其他类型的图**"
        "（如不在允许集里就别画坐标系/数轴/某种曲线）。在允许图型内，具体点/线/角怎么布置由你按"
        "题面合理构造（保留必要自由度，别画死）。"
    )
