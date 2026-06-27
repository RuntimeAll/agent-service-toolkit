"""variant 引擎 · RuoYi 客户端门面（PRD-C-104 B2，re-export 转发，零逻辑改）。

🔴 决策（B2a）= **re-export 门面**，不真搬：
  variant_support.py 是 1269 行紧耦合文件——RuoyiClient / persist_items / build_*_bo
  与留在 variant_support 的锚定函数（anchor_subject / chapter_name_for_id /
  leaf_pool_for_grade / _is_review_book）互相依赖大量同文件 helper（_apply_labels /
  compute_source_hash / record_link_manifest 等）+ 模块级常量。真搬会造成跨文件 helper
  大迁移或 variant_support ↔ shared/ruoyi 循环 import。
  按 PRD-C-104 架构设计.md「纯搬零改 + 最低 churn」原则，本模块仅做转发：
  「shared 层有 ruoyi 门面、外部从 shared 引」达标，variant_support.py 物理不动、零回归。
"""

from __future__ import annotations

from agents.variant_support import (
    RuoyiClient,
    RuoyiError,
    build_block_json,
    build_create_bo,
    build_mother_bo,
    build_update_bo,
    persist_items,
)

__all__ = [
    "RuoyiClient",
    "RuoyiError",
    "persist_items",
    "build_mother_bo",
    "build_create_bo",
    "build_update_bo",
    "build_block_json",
]
