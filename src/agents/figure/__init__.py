# -*- coding: utf-8 -*-
"""PRD-C-100 带图配图子包（B-mathfig vendor + B3 带图管线）。

内嵌（去 MCP/不做独立服务，见 artifacts/python-langgraph-健康+集成可行性.md §四/五）：
  - figure_crop  ：DocLayout-YOLO 母题切图（vendor 自 codeplace-B/qbank-labeler，去 MCP 壳，
                   模型单例只加载一次；torch 全栈进 toolkit 进程，numpy 2.3 兼容 B0 已实测）。
  - mathfig_render：opus 翻 GeoGebra 命令 → mathfig.render_geogebra 一轮直出（进程内 import +
                   node 子进程隔离）。**图全链 PNG 无损**；渲染失败包 try/except 降级（标 ⚠ 继续，
                   绝不冒泡掐 SSE，见 §四护栏3）。
  - geogebra_samples：6 图型（旋转/平移/对称/折叠/伪3D/三视图）canonical 命令样例（B-mathfig
                   render 抽验已 0-fail 验过），喂 B3 opus 翻命令 prompt。
"""
