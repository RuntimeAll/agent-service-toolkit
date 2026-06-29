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
    compress,  # 🔴 PRD-C-106 B2·压缩闸（STOP1 后、stage2 前固化 MotherCoreRef）
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
    intent_triage,  # 🔴 PRD-C-108 B1·薄意图层节点（route_entry_v2 对话型歧义态进它跑 LLM 分诊）
    mother_opus_entry,
    parse_instruction,
    patch,
    persist_to_bank,
    require_login,
    route_after_parse,
    route_after_triage,  # 🔴 PRD-C-108 B1·意图 → 现有节点映射 + 低置信回退 route_entry
    route_dispatch,
    route_entry_v2,  # 🔴 PRD-C-108 B1·新 conditional entry point（罩在 route_entry 之上）
    solve_explain,
)

# 🔴 PRD-C-108 B1：route_entry 不再被 graph.py 直接引用（route_entry_v2 内部包它当兜底）；
#   conditional entry point 改用 route_entry_v2，route_entry 函数体零改、仍当低置信回退兜底。

graph = StateGraph(VariantState)
# 🔴 PRD-C-100 B1a 塌缩入口：mother_opus_entry 替代 analyze+mother_precheck（新图入口）。
#   🔴 PRD-A-021 R4·F19：旧 analyze/mother_precheck 退役节点 + 其 edge + path_map 死映射已删
#   （route_entry 不再路由到它们，节点不可达 = 编译期死岛）。两函数体仍保留 = 防引用 / 留档，
#   但不再注册进图。注意：`analyze` 字符串在 variant_model("analyze") 模型槽 + conv_trace marker
#   处仍 LIVE，未动。classify 保留 = 低置信确认 resume 的「+1 次 opus 池注入重锚」路径（D3）。
graph.add_node("mother_opus_entry", mother_opus_entry)
# 🔴 PRD-C-108 B1·薄意图层节点：route_entry_v2 把「对话型纯文本歧义态」分到这里跑 LLM 分诊，
#   写 state.intent_decision 后由 route_after_triage 确定性派到现有节点（节点零改）。
graph.add_node("intent_triage", intent_triage)
graph.add_node("classify", classify)
graph.add_node("await_review", await_mother_review)  # B5·母题卡硬停闸（置 awaiting_mother_review + END）
graph.add_node("clarify", clarify)
graph.add_node("entry_lowconf_block", entry_lowconf_block)  # 🔴 R2a·闸4·读图低置信前置拦截（建议换图，不进 classify）
graph.add_node("compress", compress)  # 🔴 PRD-C-106 B2·压缩闸（阶段边界：固化母题核心参照）
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
    # 🔴 PRD-C-108 B1·意图层入口（wrap-not-rewrite）：对话型歧义态 → intent_triage（LLM 听懂意图），
    #   其余一切（auth/编辑 op/新图/按钮 resume/确认章 resume/低置信闸4/库内母题直造）→ 原 route_entry
    #   确定性分诊（函数体零改）。route_entry_v2 内部直接调 route_entry，返回值集合 = route_entry ∪ {intent_triage}。
    route_entry_v2,
    {
        # 🔴 PRD-C-108 B1·意图层节点（对话型歧义态落点）
        "intent_triage": "intent_triage",
        # 🔴 PRD-C-100 B1a：新图入口 → 塌缩节点（替 analyze）
        "mother_opus_entry": "mother_opus_entry",
        # 🔴 PRD-A-021 R4·F19：'analyze':'analyze' 死映射已删（route_entry 永不返回 'analyze'，
        #   节点也已退役不再注册 → 留着会 path_map 目标悬空编译报错）。
        # 🔴 PRD-C-106 B2·阶段边界压缩闸：route_entry 仍返回 "generate"（frozen 行为不动），
        #   但路径映射把它**先经 compress 节点**（固化 MotherCoreRef）再 → generate。
        #   compress→generate 直连（见下方 add_edge），阶段二只读固化参照（AC3/G3 隔离）。
        #   所有进 generate 的入口（STOP1 resume / 库内母题直进 / mother_confirmed 直造）统一过 compress；
        #   facts_from_ref 对无参照旁路有降级，过 compress 只增益不破。
        "generate": "compress",
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

# 🔴 PRD-C-108 B1·意图层出口：route_after_triage 把意图分诊（state.intent_decision）确定性映射到现有
#   节点。高置信 → 按意图派（开始→generate / 调整母题·编辑·答疑·新任务→parse / 确认范围→await_review）；
#   母题存疑 → await_review（即便"开始"也先停母题卡解决疑点·AC2）；低置信/无意图 → route_after_triage
#   内部直接回退 route_entry（返回值落同一组 path_map），故映射目标 = route_entry ∪ {await_review}。
#   "generate" 同入口口径先经 compress 固化 MotherCoreRef（阶段边界一致，不旁路 C-106 压缩闸）。
graph.add_conditional_edges(
    "intent_triage",
    route_after_triage,
    {
        "generate": "compress",          # 开始出题（母题立住）→ 经 compress → generate
        "await_review": "await_review",  # 确认范围 / 母题存疑停等 → 母题卡硬停（AC2）
        "parse": "parse_instruction",    # 调整母题 / 编辑变式 / 答疑 / 新任务 → 既有 parse 分诊
        "classify": "classify",          # 回退·结构化确认章 resume
        "mother_opus_entry": "mother_opus_entry",  # 回退·新图（理论不可达，意图层不接图轮）
        "editor_entry": "editor_entry",  # 回退·结构化编辑 op
        "ask": "ask_for_image",          # 回退·催图
        "auth": "require_login",         # 回退·登录硬闸
        "entry_lowconf_block": "entry_lowconf_block",  # 回退·低置信闸4
    },
)

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


# 🔴 PRD-C-106 B2·压缩闸 → 阶段二：compress 固化 MotherCoreRef 后直连 generate（阶段边界）。
#   compress 只写 state.mother_core_ref、不分支（异常路径 incomplete=true 由 generate 入口防御断言收口
#   回确认态，不在此分流）→ 单一无条件边。
graph.add_edge("compress", "generate")

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
