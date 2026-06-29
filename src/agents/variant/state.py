"""variant 引擎 · State 契约层（PRD-C-104 B1 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出：
  - items merge reducer（_ITEMS_PRESERVE_*/_item_stem_norm/merge_items，PRD-A-021 S4）
  - VariantState（MessagesState 子类，TypedDict 契约）

🔴 行为零改：内容逐字搬，仅补本模块所需 import（Annotated/Any/MessagesState）。
   __init__.py 顶部 re-export 这些符号 → service.py / variant_entry.py 零感。
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState

# ---------------------------------------------------------------------------
# 🔴 PRD-A-021 S4·items merge reducer（治本：杜绝 last-writer-wins 全量覆盖丢 figure_url/手改）
# ---------------------------------------------------------------------------
# 背景根因：旧 VariantState.items 无 reducer = LangGraph 默认 last-writer-wins 全量覆盖。任何节点
#   （通道A graph 内）或 aupdate_state（通道B /variant/* graph 外）写 {"items": full_list} 都整组替换。
#   于是「通道B 设了 figure_url → 通道A 某节点回写整组（拿的是更早的快照 / 重组时漏带 figure_url）」
#   → figure_url/手改静默丢失（跨轮非并发也丢，是最普遍的丢图通道）。
#
# 修法 = 给 items 挂「按稳定 id 合并 + 同时支持显式清空」的 reducer：
#   ① 右操作数（new = 本次写入）是**权威集合与顺序**：只有 new 里的题留下（不复活 old 里被删的题），
#      顺序完全照 new（reorder/默认序都不被 reducer 扰动）。→ 「按 id 合并」是**在 new 的骨架上回填**，
#      不是并集。
#   ② 稳定 id = item["_seq"]（§P2b「题原始生成序」既有稳定 merge 键）。new 里缺 _seq 的题，reducer
#      就地补 _seq（首次全量集落库即补齐，此后跨轮稳定）→ 后续 backfill 一律能按 _seq 命中、不靠位置。
#   ③ 回填策略两类（只在 new 缺该键时回填，绝不覆盖 new 已带的新值）：
#      - PRESERVE_ALWAYS（簿记/手改印记）：按 _seq 命中即回填（入库 id/已入库标记/手改印记等，
#        节点重组时漏带也不丢）。
#      - PRESERVE_IF_SAME_STEM（figure_url）：**仅当 old/new 同 _seq 且题面(stem)未变**才回填——
#        题面变了（重生/换题）= 旧配图对新内容失效，绝不把过期图嫁接到新题面上。
#   ④ 显式「整体替换/清空」= **空列表语义约定**：new 为 []（或非 list 容错）→ 直接返回 []（真清空）。
#      所有「清 items」调用（mother_opus_entry.base_out / analyze 退役入口 / patch 改硬锚 / generate
#      裸奔兜底）写的就是 "items": []，本 reducer 据此真清空，**不需另设哨兵**（[] 在本域永远=清空，
#      无「[] 表示合并空集保留旧组」的歧义——已逐处核对全部 items 赋值点确认）。
# 🔴 红线：本 reducer 不改 M3（高置信空池 picker 兜底，走 generate 正常出题）/ M4（新母题清旧 items=
#   写 []，reducer 清空）/ B2（niche 重锚不死循环，走 classify/控制流，与 items 合并正交）的任何行为；
#   它只在「整组覆盖」这一步把 figure_url/手改/簿记按 id 安全续上，对清空/替换/重排字节级透明。
_ITEMS_PRESERVE_ALWAYS: tuple[str, ...] = (
    "_persist_id",      # 在库雪花 id（防重落库 / 覆盖入库键）——重组漏带会致重复落行
    "persisted",        # 已入库标记
    "_draft_id",        # 🔴 PRD-A-022 批1：assemble 自动落的草稿行雪花 id（status=0，未发布）——
    #                     跨轮重组（排序/编辑非替换路径）漏带会致 assemble 重复落草稿；与 _persist_id
    #                     同口径走 PRESERVE_ALWAYS 续上。重生（换题）路径显式 strip（不 carry 旧草稿 id），
    #                     由 assemble 对新题重落新草稿，与本续上正交（new 已带=不覆盖，new 缺才回填）。
    "_content_dirty",   # 已入库题内容已改待覆盖
    "question_id",      # FE 进 A-015 编辑器用
    "manual_edited",    # 老师手改印记（重生前二次确认 + 闸B 不回炉）
    "manual_block",     # 老师手动排版印记
    # 🔴 PRD-C-107 B2·per-variant 压缩 memo（progress 修订记录2）：每道变式的「接着聊」续聊载荷
    #   = {spec(seq/coeff/operator/difficulty_target/qtype) + 产物摘要(stem/answer) + 1-2 行算子/系数
    #   理由}。**不开 per-variant thread_id**（避 A3 checkpoint 竞态 + 守 langgraph 状态最小）——
    #   memo 入 items[i]、跨轮重组靠本 reducer 按 _seq PRESERVE_ALWAYS 续上（节点重组漏带也不丢）。
    #   「调整=接着聊」用 [MotherCoreRef facts + 该道 memo + 新指令] 重建续聊上下文（看得见自己 v1）。
    #   重出/新增不读旧 memo（重出丢弃重写新 memo；新增 spawn 新 memo）。
    "_variant_memo",
)
_ITEMS_PRESERVE_IF_SAME_STEM: tuple[str, ...] = (
    "figure_url",       # 变式配图 OSS url：仅题面未变才续（题面变=旧图失效，不嫁接）
    # 🔴 PRD-A-021 R2b·U1：生成态配图 PNG base64。旧实现配图 base64 仅活在 FE 内存
    #   （variantFigures[idx].png），刷新/切 tab 即丢，且只在入库动作才传 OSS——生成态配图
    #   永不落 checkpoint。现把 base64 也按 figure_url 同口径走 reducer 续上（仅题面未变才续，
    #   题面变=重生=旧图失效不嫁接）→ FE 造图后写一次 state，刷新即从 checkpoint 取回。
    "figure_base64",
)


def _item_stem_norm(it: Any) -> str:
    """题面归一（仅供 reducer 判「题面是否变了」；与 _norm 同口径但本函数定义早于 _norm，独立实现）。"""
    if not isinstance(it, dict):
        return ""
    return " ".join(str(it.get("stem") or "").split())


def merge_items(old: Any, new: Any) -> list[dict[str, Any]]:
    """items 通道 reducer（见上方设计块）。new=本次写入（权威集合+序），old=已落库前态。

    - new 非 list → 容错返回 old（不让坏写炸状态；理论上不该发生）。
    - new == [] → 显式清空（M4 新母题/patch 改硬锚/裸奔兜底都靠它真清空）。
    - 否则在 new 骨架上：按 _seq 命中 old 同题 → 回填 PRESERVE_ALWAYS（缺即补）；题面未变再回填
      figure_url。new 缺 _seq 的题就地补 _seq（位置序兜底，仅 old 全无 _seq 的纯首轮场景才用位置匹配）。
    """
    if not isinstance(new, list):
        return old if isinstance(old, list) else []
    if len(new) == 0:
        return []  # 🔴 显式清空（空列表语义约定）
    old_list = old if isinstance(old, list) else []
    old_by_seq: dict[Any, dict[str, Any]] = {
        o["_seq"]: o
        for o in old_list
        if isinstance(o, dict) and o.get("_seq") is not None
    }
    any_old_seq = bool(old_by_seq)
    out: list[dict[str, Any]] = []
    for pos, n in enumerate(new):
        if not isinstance(n, dict):
            out.append(n)  # 非 dict 原样保留（不该发生，纯防御）
            continue
        m = dict(n)
        seq = m.get("_seq")
        if seq is not None:
            match = old_by_seq.get(seq)
        elif not any_old_seq and pos < len(old_list) and isinstance(old_list[pos], dict):
            # 纯首轮兜底：old 整组都没 _seq（理论上仅极早期）→ 退化为位置匹配；一旦补了 _seq
            # 此分支后续不再走（稳定键优先），避免 reorder 后位置错配。
            match = old_list[pos]
        else:
            match = None
        if isinstance(match, dict):
            for k in _ITEMS_PRESERVE_ALWAYS:
                if k not in m and k in match:
                    m[k] = match[k]
            if _item_stem_norm(m) == _item_stem_norm(match):
                for k in _ITEMS_PRESERVE_IF_SAME_STEM:
                    if k not in m and k in match:
                        m[k] = match[k]
        if m.get("_seq") is None:
            m["_seq"] = pos + 1  # 就地补稳定键（此后跨轮按 _seq 命中，不靠位置）
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# State（设计 prompt 指定结构）
# ---------------------------------------------------------------------------
class VariantState(MessagesState, total=False):
    image_url: str | None
    # 🔴 PRD-A-022 批2·D8：母题「切图」OSS https url（toolkit crop_mother_figure 产 → FE 上 OSS
    #   → 经 /variant/set-mother-figure 回写）。build_mother_bo 入库优先取它（切图），缺则不带图
    #   （D8 不退原图兜底）。语义同变式 figure_url，但这是顶层 scalar、单次写定，无需 reducer。
    mother_figure_url: str | None
    images_count: int
    questions_in_image: int
    # analysis：年级/考点(kp)/题型(qtype) 各带置信
    analysis: dict[str, Any]
    mother_dna: dict[str, Any]
    mother_confirmed: bool
    # 🔴 PRD-C-017 B2·母题 nano 前置判（年级册+章+带图）。analyze 后、classify 前由 mother_precheck
    #   节点写。awaiting_mother_confirm=True = 已发 needConfirm 停下等老师确认（复用 clarify→END
    #   chat-resume，不引 LangGraph interrupt）。下一轮老师确认（confirmed_chapter_id 经 config 回传）→
    #   route 直奔 classify。mother_rejected=True = 带图打回（终止流程，不调 opus、不出变式）。
    #   confirmed_chapter_id/confirmed_grade_book_id = 老师确认后的章/年级册 id（接 classify 闸B）。
    mother_precheck: dict[str, Any] | None
    awaiting_mother_confirm: bool
    mother_rejected: bool
    confirmed_chapter_id: str | None
    confirmed_grade_book_id: str | None
    # 🔴 PRD-C-017 B5·母题卡硬停闸：classify 解出 mother_dna + 发母题卡帧后，置 True 并 END
    #   （不自动流向 generate）。下一轮老师点「开始举一反三」→ FE 经 config.configurable 回传
    #   start_variants=True → route_entry 见信号 + 已有 mother_dna（checkpointer 持久 thread state）
    #   → 直奔 generate（不重跑 classify、不重调 opus，复用 state.mother_dna）。
    awaiting_mother_review: bool
    # 🔴 BUG-A（PRD-C-107 收尾）：母题卡态老师【指定/纠正解法】或【要求重解】时，patch 写入解法要求，
    #   classify 全量重 solve 时把它注入母题 opus prompt（让重解按老师指定的解法走）。重解后 classify 清空。
    _resolve_method_hint: str | None
    # 🔴 PRD-C-106 B2·母题核心参照（MotherCoreRef，契约 §10）：阶段一 → 阶段二的唯一传递物。
    #   compress 节点（STOP1 后、generate 前）把阶段一锚定产物固化成一个不可变参照对象
    #   {version, incomplete, dna(结构化), summary(人话), stem/answer/solution, figure, facts}。
    #   阶段二 generate + B3 fan-out 子任务**只读它**（facts_from_ref），不再各自 _mother_facts(state)
    #   重算（B0 坑2：并发读脏）。无 reducer：单次写定（compress 一次性 return），无并发写。
    #   缺省（库内母题直进 / 旧线程 / 未过 compress 的旁路）→ facts_from_ref 回退 _mother_facts(state)。
    mother_core_ref: dict[str, Any] | None
    # items[{stem, answer, solution, qtype, difficulty, level, injected_kp?,
    #        check:{badge:ok|warn, solved_answer}  ← 闸B(solve_explain)填,
    #        gene:{gate:pass|warn|skipped, reason?} ← 闸A(gene_gate)填}]
    # 🔴 PRD-A-021 S4：挂 merge_items reducer（按 _seq 稳定 id 合并 + 空列表显式清空），治本丢
    #   figure_url/手改/簿记的 last-writer-wins 全量覆盖。语义/红线见 merge_items 上方设计块。
    items: Annotated[list[dict[str, Any]], merge_items]
    history: list[dict[str, Any]]
    # 交互层：parse_instruction 的解析结果（intent/ops/knobs/...），路由后各分支消费并清空
    pending: dict[str, Any] | None
    # 🔴 首轮配方旋钮（设计 §5 五旋钮的"首轮接线"）：None=还没抽过；{}=抽过但老师没提(走默认)。
    # {"count": int, "difficulty_plan": "increasing"|str, "qtype_dist": {"选择":2,...}, "note": str}
    # analyze（新母题轮）负责抽取/重置：新图新要求 → 重抽覆盖；同图重贴无新要求 → 保留；
    # 新图无要求 → 重置 {}（旧母题配方绝不泄漏到新母题）。generate 仅对库内母题路径兜底抽。
    knobs: dict[str, Any] | None
    # generate 的代码级配方校验缺陷清单（整组 retry 1 次后仍不符 → assemble 头部外显 ⚠）
    shape_defects: list[str]
    # 🔴 PRD-C-107 B2·统一意图层（契约 §10 intent spec 读模型）：按钮路径=default_intent_spec
    #   (generate 写)、打字/后续命令=build_intent_spec(parse_instruction 写进 pending.intent_spec)。
    #   单次写定、纯视图（不参与路由），供 trace/FE/单测断言「这次是按钮默认还是打字哪种意图」。
    intent_spec: dict[str, Any] | None
    # 🔴 BUG-01（2026-06-19）·改主考点可回退：edit-dna 改 main_kp 前的旧考点快照
    #   {"main_kp": {id,name}|None, "kp": analysis.kp 旧值|None}，FE 据它给「撤销改考点」入口。
    #   不破坏 items（改主考点不再清 items）→ 撤销 = 把主考点改回旧值即可，变式都还在。
    main_kp_prev: dict[str, Any] | None
    # 4d 方案A（PRD-C-012）：本轮被剔除题的叙事（sympy 证实标答错且重生未果 → 不外发），
    # solve_explain 每轮重写（非累计），assemble 摘要外显「本组少 N 道」
    dropped_notes: list[str]
    # 🔴 P9 手排 sticky（对抗审③·PRD-C-013）：exec_reorder 置 True，标记老师已手动排过序。
    # assemble 见 True 时**跳过** _sort_by_difficulty（默认难度升序排序），不静默重排覆盖手排；
    # 改变题集的编辑（exec_add/exec_remove）清掉该标记（题集变了，手排次序失效，回默认序）。
    # exec_regenerate 是原位改单题不变序 → 不清（手排保留）。
    manual_order: bool
    # 🔴 P13 预算闸（PRD-C-013）：state 级 LLM 调用计数器 {"used":int,"limit":int}。
    # 出题/编辑轮**入口**节点（generate / parse_instruction）按 settings 重置；轮内下游节点
    # （gene_gate/solve_explain/assemble/exec_*）从 state 携带、累计 used，并把它放回返回值
    # state（跨 superstep 保活）。超限后增强类调用跳过走 G5 降级（_budget_exhausted）。
    llm_call_budget: dict[str, int] | None
    # 🔴 批3（2026-06-13）·事实源冻结：每批次一份「老师锚准的事实源」（年级学期/主考点/DNA
    #   基础元素 = analysis.grade/kp + mother_dna.dna）。定死/确认时 facts_locked 置位 → 此后
    #   **只许老师指令改、LLM 输出不许反向覆盖**。每次老师修正记一条 audit（字段/旧值/新值/
    #   指令原文）。不搞重型版本系统，setter 统一收口写入。
    facts_locked: bool
    facts_audit: list[dict[str, Any]]
    # ===================================================================
    # 🔴 PRD-C-015 批1·DNA 契约 v2 增量（在 C-014 v1 上加，不动既有字段；一次升 v2 无 v1.5）
    # ===================================================================
    # (a) 模型维（原双轴；批1 仅建字段占位，批2 才真填）：
    #   mother_dna.dna.models = [{id,name}]（1~3 项，非空）由批2 model_anchor 写；
    #   mother_dna.dna.model_overflow = [str]（池外名，⚠/待命名池用）。批1 不动 dna_extract 产物，
    #   字段缺省即「未抽」（None/缺），下游容缺。
    # (b) 母题守恒确认状态（块①·D-merge7 确定性异常门控·非置信非硬锁 + 缺口5 合并闸）：
    #   mother_confirm = {
    #     flags: [str],            # 确定性异常标记（FLAG_SECONDARY_KP_OOB/EXAM_TYPE_OOB/SKELETON_EMPTY），无逐维置信分
    #     needs_confirm: bool,     # (flags 非空) ∨ (年级+主考点三锚没定死) → 弹合并确认面；否则直接放行
    #     confirmed_dims: [str],   # 老师已过/改的守恒维（留痕用）
    #     audit_ref: int | None,   # 指向 facts_audit 的索引（不另起审计表）
    #   }
    mother_confirm: dict[str, Any]
    # (c) DNA 改→重生四分流·待重生态·防脏·快照（块③·批1 仅建字段，批4 才接真重生逻辑）：
    #   - regen_class 是约定/枚举映射（哪维属哪类）= 模块级常量 REGEN_CLASS（前后端共用，不逐题落库）。
    #   - items[i].dna_dirty: bool（重生维/骨架/models 改置位；纯元数据维改不置位）→ 致命①入库硬闸。
    #   - items[i].regen_snapshot: 重生前快照（缺口12 撤销重生，批4 真用）。
    #   - mother_dna.dirty: bool（母题守恒维改置位）。
    #   - regen_dirty: 会话态，已改未重生的重生维列表（打角标；待重生集合=变式自身脏∪母题脏波及）。
    regen_dirty: list[str]
    # 🔴 PRD-C-100 B1a 塌缩入口（mother_opus_entry）增量：
    #   mother_has_figure：opus 一把判定题面是否含图（带图不再 reject，给 B3 切图管线判定钩子）。
    #   entry_decision：D1 条件 confirm 判据快照（needs_confirm/reason/候选/置信）。
    #   _entry_finalized：本轮入口是否走到 finalize（高置信路径）= after_mother_entry 路由信号
    #     （错误早退显式置 False，避免 checkpointer 跨轮 stale True 误路由到 await_review）。
    mother_has_figure: bool
    entry_decision: dict[str, Any]
    _entry_finalized: bool
    # 🔴 PRD-A-021 R2a·闸3（BUG-03）：复用首解强锚路径下「老师选定章↔AI 判主考点冲突」（章里锚不到
    #   主考点真叶子）→ 闸断/强确认（不静默强锚放行）。记录已就该冲突闸断过的章 id；老师**再次确认
    #   同一章**（坚持）→ 接受强锚放行，不再二次闸断（防死循环）；改成别的章 → 重新走锚定。
    _bug03_gated_chapter: str | None
    # 🔴 PRD-C-107 BUG-1·防御兜底：classify 重 solve 路径（无首解产物可复用时）对某确认章 solve 失败过
    #   一次即记此 id；老师再确认同章 → 不再重 solve（已证反复坏 JSON），改 graceful 降级（锚确认章 +
    #   待人审 + confirmed=True 照常出题），杜绝「确认→重 solve→失败→picker→再确认」无限回环。
    _resolve_failed_chapter: str | None
    # 🔴 PRD-A-021 R2a·闸4（BUG-04）：母题读图置信极低（< 0.40）/ 章未判出 → resume 轮在 classify
    #   **之前**前置拦截，建议换清晰图，不进 classify 烧 opus token。置 True = 已拦过一次；老师坚持
    #   （再回传 confirmed_chapter_id）→ 放行进 classify（防永久卡死）。
    _lowconf_blocked: bool
