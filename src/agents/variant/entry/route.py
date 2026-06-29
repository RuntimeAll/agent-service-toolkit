"""variant 引擎 · entry/route.py 入口路由族（PRD-C-104 B5 抽出，纯搬零改）。

图的路由真身（read state + config → 下一节点字符串，全程同步纯函数 / 无 LLM）：
- route_entry：入口分诊（身份硬闸 / editor_op / 新图 / 各 awaiting_* resume / 库内母题 / 催图），
  frozen 行为最密集（满是 BUG-编号历史补丁分支），逐字节搬、一个 config 信号判断不漏。
- gate_after_classify：定死闸 + 母题卡硬停闸（pinned → await_review / 否则 clarify）。
- after_mother_entry：塌缩入口出口路由（awaiting_mother_confirm / 错误早退 → done；高置信 → 复用 gate_after_classify）。
- after_generate：generate 出口（无 items → done / 否则 gene_gate）。
- _editor_op / _entry_read_lowconf / _should_lowconf_block + LOWCONF_BLOCK_THRESHOLD：route 专属 helper。

🔴 行为零改：C 线 resume = →END + checkpointer 持久 + 下轮 config 回传 resume（**不引 LangGraph
   interrupt**），route_entry 的路由决策即「resume 后走哪一步」的全部真相，逐位复现、已知行为照搬不修。
🔴 图 wiring 节点名/边/条件不变 → 拓扑零改。
🔴 strangler：顶部 from agents.variant import 取依赖（conv_trace / _extract_image_url /
   _latest_human_text / _pin_status / VariantState）。route re-export 置于 __init__.py **所有
   re-export 之后、图 wiring 之前**（wiring 引 route_entry/gate_after_classify/after_* 即就绪）。
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、所有 re-export 之后导入）
    VariantState,
    _extract_image_url,
    _latest_human_text,
    _pin_status,
    conv_trace,
)


# ---------------------------------------------------------------------------
# Router（入口分诊：登录? 有图? 在途母题? 库内母题跳 analyze/classify）
# ---------------------------------------------------------------------------
def _editor_op(config: RunnableConfig | None) -> dict[str, Any] | None:
    """🔴 PRD-A-021 S1：从 config.configurable 取「编辑/验算」结构化 op（让编辑走 graph 发真帧）。

    形如 {"kind": "revise"|"regen"|"edit-item"|"reverify", "index": int, ...}。缺省/非 dict → None
    （回退既有自然语言/分诊路径）。通道B 端点若想发真状态帧，可经 /stream 带 agent_config.editor_op
    进 graph（editor_entry 节点应用 op + 清 check → 下游 solve_explain 重验并发「程序验算」真帧）。
    🔴 verify-one（无状态、不依赖 thread state 的纯验算）**不**走此入口（仍是独立端点，见任务约束）。
    """
    conf = ((config or {}).get("configurable") or {}) if config else {}
    op = conf.get("editor_op")
    if isinstance(op, dict) and op.get("kind") in ("revise", "regen", "edit-item", "reverify"):
        return op
    return None


# 🔴 PRD-A-021 R2a·闸4（BUG-04）·读图低置信前置闸阈值（用户拍板 0.40）。
LOWCONF_BLOCK_THRESHOLD = 0.40


def _entry_read_lowconf(state: VariantState) -> bool:
    """母题入口读图是否「极低置信 / 章未判出」（闸4 拦截判据）。读 entry_decision 快照
    （mother_opus_entry 入口轮写），置信 < 0.40 或 章为空 → True。无 entry_decision（旧线程/
    回退入口）→ False（不拦，向后兼容）。"""
    dec = state.get("entry_decision")
    if not isinstance(dec, dict):
        return False
    try:
        conf = float(dec.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    chapter_empty = not str(dec.get("chapter") or "").strip()
    return conf < LOWCONF_BLOCK_THRESHOLD or chapter_empty


def _should_lowconf_block(state: VariantState) -> bool:
    """闸4 是否应在本 resume 轮拦截：读图极低置信 且 尚未拦过一次（_lowconf_blocked=False）。
    已拦过（老师坚持再确认）→ 不再拦，放行进 classify（防永久卡死）。"""
    return _entry_read_lowconf(state) and not state.get("_lowconf_blocked")


def route_entry(
    state: VariantState, config: RunnableConfig
) -> Literal[
    "mother_opus_entry", "parse", "generate", "ask", "auth", "classify",
    "editor_entry", "entry_lowconf_block",
]:
    # 🔴 身份硬闸（用户拍板 2026-06-11）：每次对话绑死登录老师。token 缺失/解不出 userId
    # → 一步不走（不进任何 LLM 节点，conv_trace 也不会产生无主行；表级 NOT NULL 双保险）。
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    if conv_trace.teacher_id_from_token(token) is None:
        return "auth"
    # 🔴 PRD-A-021 S1：结构化编辑/重生 op（经 /stream 带 editor_op 进 graph）优先于自然语言分诊——
    #   有 op 且有题组在手 → editor_entry（应用 op + 清 check → solve_explain 发真「程序验算」帧，治 F1
    #   通道B 静默 no-op）。无题组（op 无对象）则不拦，落既有路径（防把空轮误导进编辑）。
    if _editor_op(config) is not None and (state.get("items") or state.get("mother_dna")):
        return "editor_entry"
    url = _extract_image_url(_latest_human_text(state.get("messages", [])))
    # 🔴 PRD-C-108 B2②·题面已在手 → 同图重发不重读 turn1（AC5，承 BUG-1「复用首解、别重读图」精神）。
    #   场景：老师纠正母题（改解法/重解/换考点）时 FE 仍把原 OSS 图 URL 带在消息里 → 旧逻辑认作
    #   「跨轮新图」→ mother_opus_entry 重跑 turn1 读图 + base_out 清 items（既浪费 ~18k token 重读
    #   同一张已誊抄的题，又把已出的题组清掉）。判据 = 本轮图 URL **等于**已解出母题的 image_url
    #   且首解产物在手（mother_dna 有 opus 首解 stem）→ 这不是新母题，是同图 + 纠正话 → **不进重读
    #   入口**，落下面既有 awaiting_*/parse 纠正路径（classify 的 _reanchor_reuse_first_solve 复用首解、
    #   不重 solve）。仅「同图」才跳过；老师真贴**新图** → url ≠ state.image_url → 照常进 mother_opus_entry
    #   重读（首次见图必读，边界诚实不破）。无 mother_dna（还没解出过）→ 不是「题面在手」→ 照常进。
    _same_img_in_hand = bool(
        url
        and str(state.get("image_url") or "").strip() == str(url).strip()
        and isinstance(state.get("mother_dna"), dict)
        and str((state.get("mother_dna") or {}).get("stem") or "").strip()
    )
    # 🔴 PRD-C-100 B1a：跨轮新图 = 新母题 → 走塌缩入口 mother_opus_entry（opus 一把判章+解题+打标），
    #   替代旧 analyze→mother_precheck→classify 三节点链（控制流重写）。
    if url and not _same_img_in_hand:
        return "mother_opus_entry"
    # 🔴 PRD-C-017 B2·母题确认 resume（复用 chat-resume，不引 interrupt）：上一轮 mother_precheck
    #   发了 needConfirm 停在等确认（awaiting_mother_confirm），本轮老师**经 config 回传确认章 id**
    #   （confirmed_chapter_id）→ 直奔 classify（带确认章接闸B）。老师若改成纯文字纠正（没回 id）→
    #   落下面 parse 分诊（既有在途母题 mother_correction → patch 重锚路径），不在此拦。
    if state.get("awaiting_mother_confirm"):
        cfg = (config or {}).get("configurable") or {}
        if cfg.get("confirmed_chapter_id"):
            # 🔴 R2a·闸4（BUG-04）：进 classify 前置闸——读图极低置信/章未判出 → 拦一次建议换图，
            #   不烧 opus token。老师坚持（再确认同一章）→ 第二轮 _lowconf_blocked 已 True，放行。
            if _should_lowconf_block(state):
                return "entry_lowconf_block"
            return "classify"
    # 🔴 PRD-C-017 B5·母题卡硬停闸 resume（复用 chat-resume，不引 interrupt）：上一轮 classify
    #   解出 mother_dna + 发母题卡帧后停在 awaiting_mother_review 等老师点「开始举一反三」。本轮
    #   老师点了 → FE 经 config 回传 start_variants=True → 已有 mother_dna（checkpointer 持久 thread
    #   state）→ **直奔 generate**（不重跑 classify、不重调 opus，复用 state.mother_dna）。
    #   🔴 即使老师改了母题 DNA 再点开始（既有 dirty/patch 逻辑会清 items + mother_confirmed=False
    #   走 classify 重锚），此处只在「已有 mother_dna 且未出题」时直奔 generate，不破 B3.6 edit→regen。
    if state.get("awaiting_mother_review") and not state.get("items"):
        cfg = (config or {}).get("configurable") or {}
        if cfg.get("start_variants") and state.get("mother_dna"):
            return "generate"
        # 🔴 BUG-01 R1·#4：高置信 await_review 态下老师改章（FE 经 config 回传 confirmed_chapter_id）
        #   → 也走重锚（同低置信 awaiting_mother_confirm 那条），重入 classify（_reanchor_reuse_first_solve
        #   复用首解、重发 classify done 帧），别落 parse 僵住（旧实现只认低置信确认章，高置信改章静默回
        #   parse、classify 帧不刷 = 状态条卡死）。confirmed_chapter_id 在 → 重锚优先于 parse 分诊。
        if cfg.get("confirmed_chapter_id"):
            if _should_lowconf_block(state):  # 🔴 闸4：高置信 await_review 改章 resume 同样前置拦截
                return "entry_lowconf_block"
            return "classify"
        # 🔴 停在 review 但老师没点开始（发了别的话/改 DNA）→ 落 parse 分诊（既有母题纠正/
        #   答疑路径），**绝不**掉进下面「mother_confirmed → 自动 generate」把硬停闸架空。
        return "parse"
    # 库内母题（已确认 DNA）、还没出题 → 直接造（跳 analyze/classify）
    if state.get("mother_confirmed") and state.get("mother_dna") and not state.get("items"):
        return "generate"
    # 老会话·纯文字：已出题组 或 🔴 在途母题（已分析停在 clarify 等老师答年级/考点）
    # → parse 分诊。修 17 号多轮路由漏洞：旧版要求有 items 才进 parse，把「clarify 的
    # 回答」漏成催图（root cause 见 claude-code-sign/17-route_entry-多轮路由漏洞-修复任务.md §2）。
    if state.get("items") or state.get("mother_dna"):
        return "parse"
    # 没图、无在途母题、无题组 → 催图（设计 §6 输入边界兜底）
    return "ask"


def gate_after_classify(state: VariantState) -> Literal["await_review", "clarify"]:
    """🔴 定死闸（批2）+ 母题卡硬停闸（B5）：定死（年级册 code + main_kp 锚真叶子 + 置信达标）
    → **不再直通 generate**，改走 await_review（置 awaiting_mother_review + END，母题卡已先出，
    等老师点「开始举一反三」再经 route_entry resume 直奔 generate）；没定死（缺任一）→ 一律停
    clarify 确认态。从机制上①绝迹缺锚出题；②母题卡先出后必停、不自动造变式（B5 AC）。

    🔴 B5 前本闸 pinned → "generate" 直连，变式立刻自动生成。现在 pinned → "await_review"
    硬停，让老师 review 母题卡后主动触发。resume 路径（start_variants）绕开 classify/本闸
    （route_entry 直接 → generate），故本闸的 "generate" 出口已退役（只剩 await_review/clarify）。

    mother_confirmed 由 classify 按同口径（_conf_ok + anchored）置位，这里用 _pin_status
    再加「年级册 code + 非复习册」收口（mother_confirmed=True 但年级 code 缺/是复习册的边角
    路径也会被本闸拦住，不直通）。

    🔴 PRD-C-109 fix·确认收口（AC4/§2.2①）：母题已 mother_endorsed（老师明确背书）+ 母题已立住
       （有 mother_dna）→ 即便重锚后未 pin（degraded/锚不到叶子），也走 await_review（保持就绪、
       待人审），**不再 clarify 重弹年级章确认**——用户确认 > 代码硬锚，多确认闸收口成一个。
       承 B3 _reanchor/_bounded 的 endorsed 旁路，把它补到 classify 出口闸（fresh-solve 降级也覆盖）。"""
    if state.get("mother_confirmed") and _pin_status(state)["pinned"]:
        return "await_review"
    if state.get("mother_endorsed") and isinstance(state.get("mother_dna"), dict) and state.get("mother_dna"):
        return "await_review"
    return "clarify"


# 🔴 PRD-C-100 B1a·塌缩入口出口路由：
#   低置信弹窗（awaiting_mother_confirm）→ END 等确认（下一轮 route_entry 见 confirmed_chapter_id
#     → classify 池注入重锚 +1 次 opus，D3）；错误早退（_entry_finalized=False）→ END（消息已发）；
#   高置信 finalize → 复用 gate_after_classify（定死闸）→ await_review 硬停 / clarify。
def after_mother_entry(state: VariantState) -> Literal["await_review", "clarify", "done"]:
    if state.get("awaiting_mother_confirm"):
        return "done"
    if not state.get("_entry_finalized"):
        return "done"
    return gate_after_classify(state)


# generate：裸奔兜底时只吐消息、无 items → 结束；正常 → 闸A 基因闸 → 闸B solve_explain
def after_generate(state: VariantState) -> Literal["gene_gate", "done"]:
    if not state.get("items"):
        return "done"
    return "gene_gate"
