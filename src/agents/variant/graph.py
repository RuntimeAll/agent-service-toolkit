"""variant 引擎 · graph.py：StateGraph wiring 收口（PRD-C-104 B5 抽出，纯搬零改）。

把 __init__.py 末尾的图装配（StateGraph(VariantState) + 全部 add_node/add_edge/
add_conditional_edges/set_conditional_entry_point + variant = graph.compile()）整块搬来。
节点名/边/条件一字不改 → 25 节点 / 46 边不变。

🔴 strangler：顶部 from agents.variant import 取全部节点函数 + 路由函数 + VariantState
   （此时 __init__.py 已跑完所有 re-export，符号全就绪）。__init__.py 末尾
   `from agents.variant.graph import graph, variant` 把 compiled graph 导回 facade，
   保 service.py / agents.variant 取图的路径（`from agents.variant import variant`）不变。
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、所有 re-export 之后导入）
    VariantState,
    after_editor_entry,
    after_generate,
    after_mother_entry,
    after_patch,
    answer_question,
    ask_clarify,
    ask_for_image,
    assemble,
    await_mother_review,
    classify,
    clarify,
    editor_entry,
    entry_lowconf_block,
    exec_add,
    exec_regenerate,
    exec_remove,
    exec_reorder,
    exec_solution_only,
    gate_after_classify,
    gene_gate,
    generate,
    mother_opus_entry,
    parse_instruction,
    patch,
    persist_to_bank,
    require_login,
    route_after_parse,
    route_dispatch,
    route_entry,
    solve_explain,
)

graph = StateGraph(VariantState)
# 🔴 PRD-C-100 B1a 塌缩入口：mother_opus_entry 替代 analyze+mother_precheck（新图入口）。
#   🔴 PRD-A-021 R4·F19：旧 analyze/mother_precheck 退役节点 + 其 edge + path_map 死映射已删
#   （route_entry 不再路由到它们，节点不可达 = 编译期死岛）。两函数体仍保留 = 防引用 / 留档，
#   但不再注册进图。注意：`analyze` 字符串在 variant_model("analyze") 模型槽 + conv_trace marker
#   处仍 LIVE，未动。classify 保留 = 低置信确认 resume 的「+1 次 opus 池注入重锚」路径（D3）。
graph.add_node("mother_opus_entry", mother_opus_entry)
graph.add_node("classify", classify)
graph.add_node("await_review", await_mother_review)  # B5·母题卡硬停闸（置 awaiting_mother_review + END）
graph.add_node("clarify", clarify)
graph.add_node("entry_lowconf_block", entry_lowconf_block)  # 🔴 R2a·闸4·读图低置信前置拦截（建议换图，不进 classify）
graph.add_node("generate", generate)
graph.add_node("gene_gate", gene_gate)  # 闸A·基因闸（新变式 → 平行度比对 → 闸B）
graph.add_node("solve_explain", solve_explain)
graph.add_node("assemble", assemble)
# 交互层节点（多轮 WAIT 后的下一句）
graph.add_node("parse_instruction", parse_instruction)
graph.add_node("answer_question", answer_question)
graph.add_node("exec_remove", exec_remove)
graph.add_node("exec_regenerate", exec_regenerate)
graph.add_node("exec_add", exec_add)
graph.add_node("exec_reorder", exec_reorder)  # P9·指令排序（纯代码重排，不过 assemble）
graph.add_node("exec_solution_only", exec_solution_only)  # 整改3·解法修正（题面留只改解析+重跑闸B）
graph.add_node("editor_entry", editor_entry)  # 🔴 PRD-A-021 S1·结构化编辑/重生入口（应用 op + 清 check → 下游 solve_explain 发真帧）
graph.add_node("patch", patch)
graph.add_node("ask_clarify", ask_clarify)
graph.add_node("persist_to_bank", persist_to_bank)
graph.add_node("ask_for_image", ask_for_image)
graph.add_node("require_login", require_login)

graph.set_conditional_entry_point(
    route_entry,
    {
        # 🔴 PRD-C-100 B1a：新图入口 → 塌缩节点（替 analyze）
        "mother_opus_entry": "mother_opus_entry",
        # 🔴 PRD-A-021 R4·F19：'analyze':'analyze' 死映射已删（route_entry 永不返回 'analyze'，
        #   节点也已退役不再注册 → 留着会 path_map 目标悬空编译报错）。
        "generate": "generate",
        "parse": "parse_instruction",
        # 🔴 B2·母题确认 resume（config 回传确认章 id）→ 直奔 classify（带确认章接闸B）
        "classify": "classify",
        # 🔴 PRD-A-021 S1·结构化编辑/重生 op（经 /stream 带 editor_op）→ editor_entry
        "editor_entry": "editor_entry",
        # 🔴 'ask' 必落真节点（ask_for_image），不能直连 END —— 否则首轮无节点产消息，回复为空
        "ask": "ask_for_image",
        # 🔴 身份硬闸：无登录态 → 提示重登（同上，必落真节点）
        "auth": "require_login",
        # 🔴 PRD-A-021 R2a·闸4（BUG-04）：读图极低置信 resume → 前置拦截建议换图（不进 classify）
        "entry_lowconf_block": "entry_lowconf_block",
    },
)
graph.add_edge("ask_for_image", END)
graph.add_edge("require_login", END)

# 🔴 PRD-C-104 B5：after_mother_entry（塌缩入口出口路由）已抽到 entry/route.py（纯搬零改），
#   顶部 re-export 回本模块；图 wiring 引用零感。
graph.add_conditional_edges(
    "mother_opus_entry",
    after_mother_entry,
    {"await_review": "await_review", "clarify": "clarify", "done": END},
)


# 🔴 PRD-A-021 R4·F19：旧 analyze→mother_precheck 退役链的 after_analyze 路由 + 两条 edge
#   （analyze→{mother_precheck,END}、mother_precheck→END）已删（节点已退役，留着 = 引用未知节点
#   编译报错）。新图入口走 mother_opus_entry（见上 after_mother_entry）。
# 🔴 B5·classify 不再直连 generate：定死 → await_review（母题卡硬停闸，置 awaiting_mother_review
#   + END，等老师点「开始举一反三」经 route_entry resume → generate）；没定死 → clarify。
graph.add_conditional_edges(
    "classify", gate_after_classify, {"await_review": "await_review", "clarify": "clarify"}
)
graph.add_edge("await_review", END)
graph.add_edge("clarify", END)
graph.add_edge("entry_lowconf_block", END)  # 🔴 R2a·闸4·拦截后 END（等老师换图 / 坚持确认）


# 🔴 PRD-C-104 B5：after_generate（generate 裸奔兜底 → done / 正常 → 闸A）已抽到 entry/route.py
#   （纯搬零改），顶部 re-export 回本模块。
graph.add_conditional_edges(
    "generate", after_generate, {"gene_gate": "gene_gate", "done": END}
)
graph.add_edge("gene_gate", "solve_explain")
graph.add_edge("solve_explain", "assemble")
graph.add_edge("assemble", END)

# --- 交互层路由（设计 §3 mermaid：WAIT → parse → 5 意图分诊） ----------------
graph.add_conditional_edges(
    "parse_instruction",
    route_after_parse,
    {
        "patch": "patch",
        "dispatch": "dispatch",  # 编辑意图 → 三层漏斗节点收口（remove/regenerate/add）
        "answer": "answer_question",
        "save": "persist_to_bank",
        "solution_only": "exec_solution_only",  # 整改3·解法修正（题面留只改解析）
        "ask_clarify": "ask_clarify",
    },
)


# 编辑意图先经 dispatch 漏斗：把 route_after_parse 的 "dispatch" 桥到三原语。
# 用一个轻量调度节点统一收口（避免 route_after_parse 直连 exec_remove 误派）。
graph.add_node("dispatch", lambda state: {"messages": []})
graph.add_conditional_edges(
    "dispatch",
    route_dispatch,
    {
        "exec_remove": "exec_remove",
        "exec_regenerate": "exec_regenerate",
        "exec_add": "exec_add",
        "exec_reorder": "exec_reorder",
        "ask_clarify": "ask_clarify",
    },
)

# 三原语收口：regenerate/add 产**新变式** → 先过闸A基因闸再到闸B；
# remove 只删不产新题 → 直连 solve_explain（旧题带 check+gene 双标，两闸都原样通过）。
graph.add_edge("exec_remove", "solve_explain")
graph.add_edge("exec_regenerate", "gene_gate")
graph.add_edge("exec_add", "gene_gate")
# 🔴 reorder 只挪槽位、不产新题、不重判 → 直连 END（已自发 artifact 整帧；**不过 assemble**，
# 否则 assemble 的默认难度升序排序会覆盖老师手排）。
graph.add_edge("exec_reorder", END)
# 整改3·解法修正：节点内已逐题重写解析 + 重跑闸B（每题 check 已定）→ 过 assemble 收口快照
# （刷新头部 chip/状态计数 + 题型规范 + artifact 整帧）。
graph.add_edge("exec_solution_only", "assemble")

# 🔴 PRD-A-021 S1·editor_entry 出口：内容变动（清了 check）→ solve_explain 重验 + 发真「程序验算」帧
#   → assemble 收口快照（solve_explain→assemble 既有边）；无未判题/无题 → END（已自带 check / 友好提示）。
graph.add_conditional_edges(
    "editor_entry", after_editor_entry, {"solve_explain": "solve_explain", "done": END}
)

# 答疑/clarify → END（不改 items，回等待下一句）
graph.add_edge("answer_question", END)
graph.add_edge("ask_clarify", END)
graph.add_edge("persist_to_bank", END)


graph.add_conditional_edges("patch", after_patch, {"classify": "classify", "done": END})

# 🔴 不在此 compile checkpointer：service lifespan 注入 saver（按 thread_id 持久 state）
variant = graph.compile()
variant.name = "variant"
