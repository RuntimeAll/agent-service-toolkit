"""variant 引擎 · entry 子包（PRD-C-104 B5：入口/路由/兜底/多轮交互）。

本批（B5-A）落 fallback.py（兜底节点）+ interact.py（多轮交互节点）。
route.py / mother_entry.py / graph wiring 收口留 B5-B。
各模块走 strangler（顶部 from agents.variant import 取依赖），__init__.py 末尾 re-export
+ 图 wiring 引节点名 → 拓扑零改。
"""
