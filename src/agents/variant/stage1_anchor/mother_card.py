"""variant 引擎 · stage1_anchor 母题卡 + emit 三件（PRD-C-104 B4 抽出，纯搬零改）。

stage1 母题卡组帧/发射件：
- _artifact_payload：纯组帧（state → artifact 契约 dict，零 IO 可单测）。
- _emit_artifact：发 artifact 快照帧（FE 题卡数据源；含 P2 增量帧）。
- _mother_chapter_name：从 state 取母题章名（纯函数）。
- _build_mother_card：组母题卡 payload（契约 §10，纯函数零 IO）。
- _emit_mother_card：发母题卡专帧（mother_card「先出」·AC4/G12）。
- mother_md_from_table：母题起步档 = 锚定模型查表 grade_observed 确定算（WS1·AC2）。

🔴 行为零改：组帧/发帧/查表逐字保持；母题难度仍走表驱动 grade_observed（非 LLM 自评）。
🔴 一处必要的装载适配（strangler）：mother_md_from_table 用 _dna_factors_for_grade（在
   stage2_variant/difficulty.py，其 re-export 晚于本模块）→ 体内延迟 import（调用期解析）。
   本模块 re-export 须**先于** stage2 generate/assemble（二者顶部 import _emit_artifact）。
   _artifact_payload 顶部 import _item_dna（stage1_anchor/label，其 re-export 先于本模块）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.config import get_stream_writer

from core import difficulty  # 🔴 PRD-C-103 WS1：确定性难度判档（grade_observed，表反控的代码点）

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、stage2 generate 之前导入）
    TIER_PENDING,
    VERIFY_PENDING,
    VariantState,
    _item_dna,
    _mother_facts,
    _norm_secondary_kps,
    _to_int,
    dirty_item_indexes,
    knobs_desc,
)


def _artifact_payload(
    state: VariantState, persisted_flags: list[bool] | None = None
) -> dict[str, Any]:
    """纯组帧（零 IO 可单测）：state.items/check/gene/knobs/analysis → artifact 契约 dict。"""
    items = state.get("items") or []
    facts = _mother_facts(state)
    out_items: list[dict[str, Any]] = []
    for i, it in enumerate(items):
        chk = it.get("check") or {}
        # 🔴 P2b 稳定 seq（PRD-C-013）：item 自带 `_seq`（generate 流内 eager 落的「题原始
        #   生成序」，整生命周期不变、剔除题不重压缩）优先；缺省（assemble/入库/会话恢复等
        #   定稿全量帧）回退 index=i+1（一期等价，单键不分叉）。FE pickArtifact 按 seq 原位 merge。
        seq = _to_int(it.get("_seq")) or (i + 1)
        cell: dict[str, Any] = {
            "index": i + 1,
            "seq": seq,
            "stem": str(it.get("stem") or ""),
            "answer": str(it.get("answer") or ""),
            "solution": str(it.get("solution") or ""),
            "qtype": str(it.get("qtype") or ""),
            "difficulty": _to_int(it.get("difficulty")) or 0,
            "level": str(it.get("level") or "normal"),
            # 🔴 verify 与 review 互斥不同键：证明类只有 review（proof_needs_human）
            "verify": chk.get("verify") or chk.get("review") or None,
            # 🔴 BUG-09（2026-06-19）：手动验算态机器抓手——pending=待老师手动点验算（auto_verify=False
            #   生成态），done=已过验算/已定状态（含 sympy_pass/self_ok/proof/manual 等）。FE 据它渲染
            #   「待验算」徽章 + 验算按钮（点 → /variant/verify-one 或 /variant/reverify）。
            "verify_status": (
                "pending"
                if (chk.get("verify") == VERIFY_PENDING or chk.get("tier") == TIER_PENDING)
                else ("done" if chk else "pending")
            ),
            # 4d 外显层级（FE 徽章唯一依据；旧线程恢复无 tier → FE 按「只说好」兜底）
            "tier": chk.get("tier") or None,
            # 🔴 PRD-A-017 R1·验算可查真证据透传（FE 验算徽章展开层）：math_verify.verify() 纯 sympy
            #   算出的逐步核对话术 detail（如 computed=46.0, claimed=46.0, tol=1e-6: within tolerance）
            #   + 真算出的解集/真值 computed（如 [46]）。证明/开放/作图类只走 review、无 sympy 证据
            #   → chk 无这俩键 → None（如实留空不伪造，禁假数据铁律 §0.5）。零核心逻辑改、零编造。
            "verify_detail": chk.get("verify_detail") or None,
            "verify_computed": chk.get("computed") or None,
            "gene": (it.get("gene") or {}).get("gate") or None,
            # persisted：flags 优先（persist 节点按回执现算）；否则读 item 簿记
            # （persist_to_bank 成功后回写 state.items[i].persisted → 后续编辑轮
            #  assemble 重发快照时「已收录」徽章不回退，G5 二次入库不重复落行）
            "persisted": (
                bool(persisted_flags[i])
                if persisted_flags is not None and i < len(persisted_flags)
                else bool(it.get("persisted"))
            ),
            # 🔴 PRD-C-014 B4·FE DNA 面板数据源（键名钉死，FE pickDna 解析）：组级维度共享 +
            #   item 级（hard_points/manual_edited）覆盖。DNA 未抽取 → 空值/空数组兜底，不崩。
            "dna": _item_dna(it, facts),
        }
        # 🔴 P2b 退场哨兵（PRD-C-013）：剔除题显式带 `_dropped: true`（字段白名单透传），
        #   驱动 FE upsertIncremental 走退场过渡（is-dropping/scheduleDropRemoval）后从
        #   mergedItems 移除——不再靠「压缩 index 隐式挤掉」，避免后题 index 前移嫁接错卡。
        if it.get("_dropped"):
            cell["_dropped"] = True
        # 🔴 PRD-C-015 批4·DNA 改→重生态透传给 FE（批5 渲染角标/重生·撤销按钮/dirty 入库拦截）：
        #   dna_dirty=本题待重生；dirty_dims=哪些维脏（驱动逐维角标）；can_undo=有重生快照可撤销。
        cell["dna_dirty"] = bool(it.get("dna_dirty"))
        cell["dirty_dims"] = list(it.get("dirty_dims") or [])
        cell["mother_dirty_dims"] = list(it.get("mother_dirty_dims") or [])
        cell["can_undo_regen"] = isinstance(it.get("regen_snapshot"), dict)
        # 🔴 PRD-C-100 BC2：变式配图 OSS url（FE compose_variant_figure 产 base64 → uploadMotherImage
        #   传 OSS → 经 /variant/set-figure-url 回写 state.items[i].figure_url）。入库时
        #   build_create_bo 据它产 A-015 image 块；透传给 FE 用于会话恢复后保持配图态。缺则 None。
        cell["figure_url"] = str(it.get("figure_url") or "") or None
        # 🔴 PRD-A-021 R2b·U1：生成态配图 PNG base64 透传（会话恢复/刷新重建配图显示态）。
        #   旧实现生成态 base64 仅活在 FE 内存 variantFigures[idx].png，切 tab/刷新即丢且只在入库才
        #   传 OSS。现 FE 造图认账后回写 state（经 set-figure-url 带 figure_base64）→ 落 checkpoint →
        #   /variant/artifact 恢复时随帧透出，FE 无 OSS url 也能从 base64 重建配图。缺 → None。
        cell["figure_base64"] = str(it.get("figure_base64") or "") or None
        # 🔴 PRD-A-018 治本A·figure_spec 透传（出题节点产的配图自然语言描述）：随帧上屏 + 会话恢复保留，
        #   供 compose 从 state 取来照画（service /variant/compose-figure 自取，FE 不必传）。展示/内部键，
        #   不入 biz_question 旧字段（同 figure_url；build_create_bo 显式白名单天然不外漏）。缺则 None。
        # 🔴 round4：figure_spec 现可为 str（旧）或 dict（新 {"layout":..,"angle_labels":[..]}）——
        #   原样透传（dict 不 str 化），FE/会话恢复保留结构、compose 取来按类型分发。缺/空 → None。
        _cell_fs = it.get("figure_spec")
        if isinstance(_cell_fs, dict) and (_cell_fs.get("layout") or _cell_fs.get("angle_labels")):
            cell["figure_spec"] = _cell_fs
        else:
            cell["figure_spec"] = (str(_cell_fs).strip() or None) if isinstance(_cell_fs, str) else None
        # 🔴 PRD-C-100 BC3：已入库题在库雪花 id（= _persist_id 内部簿记键，persist_one/全部入库后回写）。
        #   FE 据它把已入库变式 round-trip 进 A-015 网格编辑器（/question/editor/:id 按 questionId 载 blockJson
        #   编辑 → /teacher/question/update-block 存）。未入库题为 None（无 questionId 不能进编辑器）。
        cell["question_id"] = (
            str(it.get("_persist_id")) if it.get("_persist_id") not in (None, "") else None
        )
        # 🔴 PRD-C-100 BC3：本题被老师手动排版过（A-015 网格编辑器存过 blockJson）→ 卡显「手动排版」印记 +
        #   重生前二次确认（会覆盖手改布局）。复用 manual_edited 印记语义（与内容编辑共用一面旗，
        #   都表「老师亲手改过，重生需确认」）；manual_block 单独标「排版」态供文案区分。
        cell["manual_block"] = bool(it.get("manual_block"))
        out_items.append(cell)
    # 🔴 批4·组级重生态：mother_dirty（母题守恒维改）+ regen_pending（待重生集合 1-based 题号）。
    #   FE 据 regen_pending 非空 → 「重生」按钮可点 + 入库按钮禁用（致命① dirty 拒入库视觉）。
    mother_dirty = bool((state.get("mother_dna") or {}).get("dirty"))
    regen_pending = dirty_item_indexes(items)
    # 🔴 批5·合并确认面（G10/G11）数据源：透传 mother_confirm（flags/needs_confirm/
    #   confirmed_dims/audit_ref），FE pickMotherConfirm 解析弹合并确认面。缺省 → None
    #   （旧 FE 不读不坏，向后兼容）。
    mother_confirm = state.get("mother_confirm") or None
    # 🔴 2026-06-17：母题卡专帧 sticky 化（修「母题入库·题面尚未产出」根因）。原 mother_card 只在
    #   classify 经 _emit_mother_card 发一次（turn1）；turn2「开始举一反三」的 assemble/persist 帧
    #   走 _artifact_payload 不带 mother_card → FE artifact.value 整帧替换后 header.mother_card=null
    #   → pickMotherCard 路①失效、mc.stem 丢 → 母题入库被拦。修法：每帧都附 _build_mother_card(state)
    #   （纯函数；无 mother_dna→None，FE 兼容 null 不回归）。让母题专帧贯穿全生命周期、stem 永在。
    mother_card = _build_mother_card(state)
    return {
        "items": out_items,
        "header": {
            "recipe": knobs_desc(state.get("knobs")) or None,
            "kp": facts["kp_name"] if facts["kp_name"] != "未知考点" else None,
            "grade": facts["grade"] if facts["grade"] != "未知年级" else None,
            # 批4·母题脏 + 待重生集合（FE 批5 用；旧 FE 不读 header 这俩键也不坏）
            "mother_dirty": mother_dirty,
            "regen_pending": regen_pending,
            # 批5·合并确认面（G10/G11）：母题确认状态（needs_confirm 时 FE 弹面）
            "mother_confirm": mother_confirm,
            # 🔴 母题专帧 sticky：每帧附母题卡（含 stem），FE pickMotherCard 路①全程可用
            "mother_card": mother_card,
        },
    }


def _emit_artifact(
    state: VariantState,
    persisted_flags: list[bool] | None = None,
    *,
    partial: bool = False,
    expected_total: int | None = None,
) -> None:
    """发 artifact 快照帧（FE 题卡数据源）。任何异常静默吞，artifact 是增强不是关卡。

    🔴 P2 增量帧（PRD-C-012）：partial=True 时帧带 {"partial": true, "expected_total": N}
    （每题闸链完成发一帧，items=按生成序已完成的题）；assemble 定稿帧不带 partial 键
    （FE 老逻辑只认最后一帧也不坏，向后兼容）。
    """
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    try:
        payload = _artifact_payload(state, persisted_flags)
        if partial:
            payload["partial"] = True
            if expected_total is not None:
                payload["expected_total"] = int(expected_total)
        writer(ChatMessage(content=[{"artifact": payload}], role="custom"))
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点
        pass


# ---------------------------------------------------------------------------
# 🔴 PRD-C-017 B3.5·母题卡专帧（mother_card「先出」）——契约 §10 / AC4 / G12。
# 心智：classify(opus 解题打标) → generate(变式) **直连**，旧路母题 DNA 只能从变式
#   items[0].dna（组级共享）外显，母题做不到真正「先出」，且母题题面/solved_answer/副考点 id/
#   anchor 章 id/need_anchor_review 在 items 里缺失 → 母题入库被拦。本帧在 classify 末尾
#   （opus 产出 mother_dna 之后、generate 之前）单发一帧，把母题卡全字段提前外显。
#
# 🔴 帧机制 = 复用 _emit_artifact 的 custom 帧 header 透传：放进 artifact.header.mother_card
#   （与 mother_confirm 同载体），FE pickMotherCard 路①优先读 header.mother_card。
# 🔴 时序「先出」= 在 classify return 前 emit；classify→generate(出 items) 是后续节点，
#   故本帧必早于任何变式 item 帧。
# 🔴 复用 classify 已产出的 mother_dna（含 opus dna/解答/锚定），绝不重调 opus。
# ---------------------------------------------------------------------------
def _mother_chapter_name(state: VariantState) -> str | None:
    """🔴 BUG-08（2026-06-19）：从 state 取母题章名（纯函数·零 IO，可单测）。

    源优先级（皆已由 classify/entry 异步节点用 chapter_name_for_id / opus 判定回写进 state）：
      ① state.confirmed_chapter_name（确认章人话名，classify 回写）；
      ② analysis.chapter.value（opus 判定章文本 / classify 反查章名）。
    都空 → None（调用方据此不加 chapter_name，绝不伪造）。
    """
    direct = str(state.get("confirmed_chapter_name") or "").strip()
    if direct:
        return direct
    analysis = state.get("analysis") or {}
    chap = analysis.get("chapter")
    if isinstance(chap, dict):
        v = str(chap.get("value") or "").strip()
        if v:
            return v
    return None


def _build_mother_card(state: VariantState) -> dict[str, Any] | None:
    """组母题卡 payload（契约 §10）。纯函数·零 IO（可单测）。

    数据源 = state.mother_dna（B1 classify 写入：stem/answer/analysis/solution_skeleton/
    solved_answer + dna 契约 v1 + need_anchor_review）+ analysis（年级/锚定）+ confirmed_chapter_id。
    无 mother_dna（库内母题直进 generate / 早退路径）→ 返回 None（调用方不发帧，FE 走 items[0] 兜底）。

    🔴 dna 子对象键名对齐 FE pickDna：main_kp=考点名(str)、main_kp_id=叶子 id、
       secondary_kps=[{id,name}]（FE strArr 取 name、顶层抠 id 作 secondaryKpIds）、
       exam_type/skeleton/hard_points/tags/scene/models。母题骨架 list → 换行拼 str（与 _item_dna 同）。
    """
    mdna = state.get("mother_dna") or {}
    if not mdna:
        return None
    dna = mdna.get("dna") or {}
    if not isinstance(dna, dict):
        dna = {}
    analysis = state.get("analysis") or {}
    grade_node = analysis.get("grade") or {}

    main_kp_obj = dna.get("main_kp") or {}
    if not isinstance(main_kp_obj, dict):
        main_kp_obj = {}
    main_kp_name = str(main_kp_obj.get("name") or "") or None
    main_kp_id = str(main_kp_obj.get("id") or "") or None

    secondary_kps = _norm_secondary_kps(dna.get("secondary_kps"))

    # 母题骨架：dna.skeleton 是 list[str]（opus 步骤序列）→ FE 要 str，换行拼（与 _item_dna 同）。
    skeleton_raw = dna.get("skeleton")
    if isinstance(skeleton_raw, list):
        skeleton = "\n".join(str(s) for s in skeleton_raw if str(s).strip())
    else:
        skeleton = str(skeleton_raw or "")
    # 母题题面/解答优先取 opus 富文本骨架字段，其次 mother_dna.solution_skeleton。
    solution_skeleton = mdna.get("solution_skeleton") or skeleton or None

    models = [
        {"id": str(m.get("id") or ""), "name": str(m.get("name") or "")}
        for m in (dna.get("models") or [])
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    ]
    # 🔴 PRD-C-106 B1③·诚实三态展示态：去 M00 兜底后，无考模型 → models:[] + model_flag="no_model"。
    #   把 flag 透传给 FE，让母题卡模型行渲染「无考模型」明示态（而非空白/报错）。有模型 → flag=None
    #   → FE 渲染真模型 summary。model_flag 缺省（旧线程/库内母题无此键）→ None，FE 兼容兜底。
    model_flag = dna.get("model_flag")
    if model_flag is None and not models:
        # 兜底：无 flag 但 models 空（旧线程恢复 / 上游没写 flag）→ 视作无考模型态（不留模糊空白）。
        model_flag = "no_model"
    no_model = (model_flag == "no_model") or (not models)

    # 🔴 PRD-C-105 G1（D5）：母题难度改表驱动同源——优先读 mother_md_from_table(state) 算出的
    #   表驱动档（grade_observed，与变式侧 grade_variant_item 同源），不再显 dna.difficulty（opus dim8
    #   LLM 自评）。与「难度全线表驱动、不采信 LLM 自评」铁律一致；FE 不动（母题卡本就只读展示难度）。
    #   降级：表驱动拿不到（库内母题无 DNA → mother_md_from_table 返 None / 任何异常）→ 回退旧
    #   dna.difficulty / mdna.difficulty（不崩、不留空白）。
    difficulty = None
    try:
        difficulty = mother_md_from_table(state)
    except Exception:  # noqa: BLE001 — 表驱动判档异常 → 回退旧自评值（绝不因判档崩了组不出母题卡）
        difficulty = None
    if not isinstance(difficulty, int):
        difficulty = dna.get("difficulty")
        if not isinstance(difficulty, int):
            md = mdna.get("difficulty")
            difficulty = md if isinstance(md, int) else None

    # 🔴 PRD-C-017 B5 问题3·答案核齐根因修：opus 把标准答案放 richText.answer（→ mdna.answer），
    #   solvedAnswer（→ mdna.solved_answer）是它的"解出值"草稿、常为空或更简略。旧版母题卡只外显
    #   solved_answer 顶层 → opus solvedAnswer 留空时母题卡"答案为空"（真机症状）。修法：① 顶层新增
    #   `answer`（标准答案，FE 优先映射它）；② solved_answer 兜底回退 answer（两者择有值者），保证
    #   "答案"区永不空（只要 opus 给了 richText.answer 或 solvedAnswer 任一）。
    std_answer = str(mdna.get("answer") or "") or None
    solved_answer = str(mdna.get("solved_answer") or "") or None

    # need_anchor_review：闸B 留空标记（mother_dna 或 dna 任一标了即为真）。
    need_review = bool(mdna.get("need_anchor_review") or dna.get("need_anchor_review"))

    # anchor 章 id：B2 确认章优先（confirmed_chapter_id），缺则年级册 code 前缀。
    chapter_id = (
        str(state.get("confirmed_chapter_id") or "").strip()
        or str(grade_node.get("code") or "").strip()
        or None
    )
    grade_book_id = str(grade_node.get("code") or "").strip() or None
    confidence = grade_node.get("confidence")

    return {
        # 顶层（FE pickMotherCard 直读）
        "stem": str(mdna.get("stem") or "") or None,  # 🔴 入库靠它，缺则入库被拦
        "analysis": str(mdna.get("analysis") or "") or None,  # 🔴 opus 解析富文本（入库存它，非骨架顶替）
        "solution_skeleton": solution_skeleton,
        # 🔴 B5 问题3：answer=标准答案（FE 优先映射）；solved_answer 兜底回退 answer（永不空）。
        "answer": std_answer,
        "solved_answer": solved_answer or std_answer,
        "qtype": str(dna.get("qtype") or "") or None,
        "difficulty": difficulty if isinstance(difficulty, int) else None,
        "main_kp": main_kp_name,  # anchorKp（考点名）
        "need_anchor_review": need_review,
        # 🔴 PRD-C-100 B3/B6：带图母题钩子——FE 据 mother_has_figure 自动调 crop_mother_figure
        #   （/variant/compose-figure mode=crop_mother，传 mother_image_url）显示母题切图（AC4）。
        "mother_has_figure": bool(state.get("mother_has_figure")),
        "mother_image_url": str(state.get("image_url") or "") or None,
        # 🔴 PRD-A-018 A-22：透传母题入库态（恢复历史会话据此重建，免显「未入库」/可重复入库）。
        #   雪花 id 发 string 防 JS Number 精度丢失（FE pickMotherCard 读 mother_question_id/persisted）。
        #   未入库 → None/False（禁假数据，FE 防御性兜 null）。
        "mother_question_id": (
            str(mdna.get("mother_question_id")) if mdna.get("mother_question_id") else None
        ),
        "persisted": bool(mdna.get("mother_question_id")),
        # 10 维 DNA（FE pickDna 解析；键名对齐 _item_dna）
        "dna": {
            "main_kp": main_kp_name,
            "main_kp_id": main_kp_id,
            "secondary_kps": secondary_kps,  # [{id,name}]：FE 取 name + 顶层抠 id
            "qtype": str(dna.get("qtype") or "") or None,
            "exam_type": str(dna.get("exam_type") or "") or None,
            "difficulty": difficulty if isinstance(difficulty, int) else None,
            "scenario": str(dna.get("scene") or "") or None,
            "hard_point_count": int(dna.get("hard_point_count") or len(dna.get("hard_points") or [])),
            "breakthrough_points": [str(h) for h in (dna.get("hard_points") or []) if str(h).strip()],
            "hard_points": [str(h) for h in (dna.get("hard_points") or []) if str(h).strip()],
            "skeleton": skeleton or None,
            "models": models,
            # 🔴 B1③·诚实三态：model_flag=no_model → FE 母题卡模型行出「无考模型」(非空白/非 M00)。
            "model_flag": model_flag,
            "no_model": bool(no_model),
            "tags": [str(t) for t in (dna.get("tags") or []) if str(t).strip()],
        },
        # 锚定（FE anchor.chapter_id → anchorChapterId）
        "anchor": {
            "grade_book_id": grade_book_id,
            # 🔴 2026-06-17：补年级册名(八年级下册)，FE 显示它而非 ID(3082)——帧原先只发 code
            "grade_book_name": str(grade_node.get("value") or "").strip() or None,
            "chapter_id": chapter_id,
            # 🔴 BUG-08（2026-06-19）：母题卡回灌章名（FE 显示章名而非 chapter_id）。源 = classify/entry
            #   节点已用 chapter_name_for_id / opus 判定回写进 state（analysis.chapter.value 或顶层
            #   chapter_name）。空就不加（不伪造，禁假数据）。
            **({"chapter_name": _chapter_name} if (_chapter_name := _mother_chapter_name(state)) else {}),
            "confidence": confidence if isinstance(confidence, (int, float)) else None,
            "need_anchor_review": need_review,
        },
    }


def _emit_mother_card(state: VariantState) -> None:
    """发母题卡专帧（mother_card「先出」·AC4/G12）。机制 = custom 帧 header.mother_card 透传
    （复用 _emit_artifact 的 header 载体，FE pickMotherCard 路①）。items 留空 → 本帧只携母题卡，
    早于任何变式 item 帧。同 _emit_stage 双层静默吞：无 runtime context / 组卡为空 → no-op。"""
    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 — 无 runtime context（单测直调节点）→ 静默 no-op
        return
    try:
        card = _build_mother_card(state)
        if not card:
            return  # 无 mother_dna（库内母题直进 generate）→ 不发，FE 走 items[0] 兜底
        writer(
            ChatMessage(
                content=[{"artifact": {"items": [], "header": {"mother_card": card}}}],
                role="custom",
            )
        )
    except Exception:  # noqa: BLE001 — 发送失败绝不炸节点（母题卡是增强不是关卡）
        pass


def mother_md_from_table(state: VariantState) -> int | None:
    """🔴 WS1·AC2：母题起步档 = 锚定模型查表 经 grade_observed 确定算（替 dim8 LLM 自评）。

    返回 level（1..4）。降级（契约 §异常）：
      - 无锚定模型 / 抽不出 K/R → grade_observed 自带哨兵兜底（仍返回 level，不报错、不返回 None）。
      - 仅当 mother_dna/dna 完全缺失（库内母题旧线程无 DNA）→ 返回 None，由调用方回退旧 dim8 值。
    """
    # 🔴 PRD-C-104 B4：_dna_factors_for_grade 在 stage2_variant/difficulty.py（其 re-export 晚于
    #   本模块）→ 体内延迟取（调用期已就绪），破除 mother_card↔stage2.difficulty 装载期循环。
    from agents.variant import _dna_factors_for_grade  # noqa: E402

    mdna = state.get("mother_dna") or {}
    dna = mdna.get("dna") or {}
    if not dna:
        return None  # 无 DNA（库内母题等）→ 调用方回退原 difficulty 值
    f = _dna_factors_for_grade(
        dna, stem=mdna.get("stem") or "", analysis_text=mdna.get("solution_skeleton") or ""
    )
    bill = difficulty.grade_observed(
        model_hits=f["model_hits"], K=f["K"], R=f["R"], D=f["D"], G=f["G"],
        high_strategies=f["high_strategies"],
    )
    return bill.get("level")
