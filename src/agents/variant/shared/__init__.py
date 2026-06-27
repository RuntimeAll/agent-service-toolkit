"""variant 引擎内部共享 helper 子包（PRD-C-104 B2 抽出，纯搬零改）。

从 `variant/__init__.py` / `variant_support.py` 原样剪出的 god-helper：
  - sanitize.py：富文本净化（纯函数，仅依赖 stdlib re）
  - budget.py：LLM 调用预算闸（contextvar 计数器）
  - ruoyi.py：RuoyiClient + persist_items + build_*_bo 门面（re-export 自 variant_support）

🔴 行为零改：内容逐字搬/转发，仅补本模块所需 import。
   __init__.py 顶部 re-export 这些符号 → service.py / variant_entry.py 零感。
"""
