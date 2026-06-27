"""variant 引擎 · stage1_anchor 打标/锚定（PRD-C-104 B4 抽出，纯搬零改）。

stage1 母题打标/锚定共用件：
- _item_dna：每个 artifact item 的嵌套 DNA 对象（FE DNA 面板数据源）。
- classify：② 分类·两步锚定 + DNA 抽取（母题 opus 合并解题+10维打标 / 重锚复用分流）。
- _reanchor_reuse_first_solve：PRD-C-100 B2 重锚复用母题首解（只换锚不重 solve，根治死循环）。

🔴 行为零改：判决/锚定/降级语义逐字保持（宁空不凑 G11、B2 死循环护红线、闸3 冲突闸断）。
🔴 两处必要的装载适配（非逻辑改动，strangler 同 B3 _LLM_TRACE_PATH 适配）：
  ① classify 内 solve_and_label_resilient(V=...) 原传 sys.modules[__name__]（彼时 __name__==
     "agents.variant"）→ 本模块 __name__ 变 stage1_anchor.label，故显式写 sys.modules["agents.variant"]
     （= 同一 facade 对象，V 行为字节级不变）。
  ② _emit_mother_card / _emit_figure_stage 在 stage1_anchor/mother_card.py（晚于本模块 re-export）
     → classify / _reanchor 体内**延迟 import**（调用期已就绪），破除 label↔mother_card 装载期循环。
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from agents import mother_opus, mother_precheck, model_anchor  # noqa: E402

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾导入）
    CONF_GATE,
    STAGE_AWAIT,
    RuoyiClient,
    VariantState,
    _ainvoke_text,
    _conf_ok,
    _emit_need_confirm,
    _emit_stage,
    _latest_human_text,
    _norm_secondary_kps,
    _resolve_grade_code,
    _sanitize_rich_text,
    _wants_review_books,
    build_mother_confirm,
    chapter_name_for_id,
    join_skeleton,
    knobs_desc,
    leaf_pool_for_grade,
    settings,
)


def _item_dna(it: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """🔴 PRD-C-014 B4·FE DNA 面板数据源：每个 artifact item 的嵌套 `dna` 对象（键名钉死，
    FE pickDna 按此解析）。组级维度（main_kp/secondary_kps/exam_type/tags/scene/skeleton）来自
    母题 DNA（facts.dna，整组共享）；item 级（hard_points/manual_edited）优先取 item。

    DNA 未抽取（库内母题直进 generate 路径，facts.dna 为空）→ 各键给空值/空数组，绝不崩。
    """
    dna = facts.get("dna") or {}
    main_kp = dna.get("main_kp") or {}
    if not isinstance(main_kp, dict):
        main_kp = {}
    # skeleton：item 级覆盖（revise target=skeleton 落 it["skeleton"]）优先，缺则回退母题
    # dna.skeleton（与 hard_points 同模式·G12a）。母题骨架是守恒基因，老师改单题骨架只覆盖本题。
    sk_item = it.get("skeleton")
    if sk_item is not None and str(sk_item).strip():
        skeleton = str(sk_item)
    else:
        # 母题 DNA 里 skeleton 是 list[str]（dna_extract 抽的步骤序列）→ FE 要 str，按换行拼。
        skeleton_raw = dna.get("skeleton")
        if isinstance(skeleton_raw, list):
            skeleton = "\n".join(str(s) for s in skeleton_raw if str(s).strip())
        else:
            skeleton = str(skeleton_raw or "")
    # hard_points：item 级有则用 item 的，否则母题 DNA 的。
    hp_raw = it.get("hard_points")
    if hp_raw is None:
        hp_raw = dna.get("hard_points")
    hard_points = [str(h) for h in (hp_raw or []) if str(h).strip()]
    # 🔴 PRD-C-015 批2·models 维（双轴「怎么解」轴）：母题级，整组共享 facts.dna.models。
    #   item 级可被 edit-dna 覆盖（it["models"]，批4 接重写解析）；缺则回退母题 DNA。非空（M00 兜底）。
    models_raw = it.get("models") if it.get("models") is not None else dna.get("models")
    models = [
        {"id": str(m.get("id") or ""), "name": str(m.get("name") or "")}
        for m in (models_raw or [])
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    ]
    return {
        "main_kp": str(main_kp.get("name") or "") or None,
        "main_kp_id": str(main_kp.get("id") or "") or None,
        "secondary_kps": _norm_secondary_kps(dna.get("secondary_kps")),
        "exam_type": str(dna.get("exam_type") or "") or None,
        "tags": [str(t) for t in (dna.get("tags") or []) if str(t).strip()],
        "scene": str(dna.get("scene") or "") or None,
        "skeleton": skeleton or None,
        "hard_points": hard_points,
        # 双轴模型维（批2）：models 非空（M00 兜底）；model_overflow/model_warn 给 FE 标 ⚠（批5 渲染）。
        "models": models,
        "model_overflow": [str(x) for x in (dna.get("model_overflow") or []) if str(x).strip()],
        "model_warn": bool(dna.get("model_warn")),
        "manual_edited": bool(it.get("manual_edited")),
    }


async def classify(state: VariantState, config: RunnableConfig) -> VariantState:
    """② 分类·两步锚定 + DNA 抽取（PRD-C-014 B1）：
      年级 → 年级叶子池 → dna_extract 让 LLM **池内选 id**（禁造词）。
    🔴 池内锚到主 kp → anchored.code = 真知识点 code（dim1KpId 主 kp）+ 抬置信；
      池内无匹配（main_kp=None）→ anchored 缺失 → 低置信 → gate_after_classify 走 clarify
      （根治 C-013 凭 LLM 置信裸放行落 0）；库/网络故障锚定不可用 → 同样 clarify，不 silent-fail。
    🔴 抽出的全维 DNA 存进 mother_dna.dna，由 _mother_facts 穿进 BO（T3）。
    """
    # 🔴 PRD-C-104 B4：_emit_mother_card / _emit_figure_stage 在 stage1_anchor/mother_card.py
    #   （晚于本模块 re-export）→ 体内延迟取（调用期已就绪），破 label↔mother_card 装载期循环。
    from agents.variant import _emit_figure_stage, _emit_mother_card  # noqa: E402

    analysis = dict(state.get("analysis") or {})
    kp_node = dict(analysis.get("kp") or {})
    dna_obj = state.get("mother_dna") or {}
    mother_dna = dict(dna_obj)

    # 🔴 复习册开关（批1 step5）：老师本轮文本明确要中考/复习/专题/模考 → 复习册并入锚定池。
    #   否则锚定池只圈教材册（防锚到「新题抢先」等复习册同名节点照样出题的实锚事故）。
    include_review_books = _wants_review_books(
        _latest_human_text(state.get("messages", []))
    )

    # --- 🔴 PRD-C-017 B4-fix·确认章驱动 grade_code（修 AC2 零作用 critical bug） ---
    #   老师确认章 = 锚定事实源，压过 analyze 图读。纯文本母题题面常无年级标记 → analyze 瞎猜
    #   grade（且人教/浙教版错位，如把浙教八下一元二次方程判成七上/九上）→ leaf_pool 用错册圈池 →
    #   opus 找不到叶子 → 闸B 锚空 + grade_code 错 → pin 三锚全缺 → 卡 clarify、不出变式。
    #   biz_subject id 编码：根=4 位（年级册 level1，如 3082=浙教八下），章=7 位（level2，如
    #   3082002），叶子=完整 id（如 3082002001004）。确认章前 4 位 = 年级册 code。
    #   确认章存在 → 用其前 4 位作 grade_code、并同步抬 analysis.grade.code/confidence（pin 闸
    #   _pin_status 读 grade.code），让 leaf_pool 用对册圈池。无确认章（理论上 B2 必停确认后不该
    #   出现，防御性）→ 回退原 analyze grade 行为（不回归 B1）。
    confirmed_chapter_id = (
        ((config or {}).get("configurable") or {}).get("confirmed_chapter_id")
        or state.get("confirmed_chapter_id")
    )
    confirmed_chapter_id = str(confirmed_chapter_id).strip() if confirmed_chapter_id else None

    # --- 两步锚定第一步：定年级 code（确认章前缀优先，老师背书压过 analyze 图读） ---
    if confirmed_chapter_id and len(confirmed_chapter_id) >= 4:
        grade_code = confirmed_chapter_id[:4]  # 确认章前 4 位 = 年级册 code
        # 同步抬 analysis.grade.code/confidence —— pin 闸 _pin_status 读 grade.code，
        #   且 leaf_pool_for_grade 用此 grade_code 圈本册叶子（不再信 analyze 误读的 grade）。
        grade_node = dict(analysis.get("grade") or {})
        grade_node["code"] = grade_code
        grade_node["confidence"] = max(float(grade_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["grade"] = grade_node
    else:
        grade_code = await _resolve_grade_code(analysis)  # 旧路径兜底（无确认章的降级线程）

    # --- 第二步：拉年级叶子池（HTTP，故障 → 空池降级走 clarify） ---
    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    leaf_pool: list[tuple[str, str]] = []
    client = RuoyiClient(token=token)

    # --- 🔴 PRD-C-017 B5 问题2附带·确认后年级**人话名**同步 ---
    #   B4-fix 已让 grade.code 吃确认章前 4 位，但 grade.value（阶段灯/气泡显示的年级册名）仍是
    #   analyze 误读值（如确认八上却显示「七年级上学期」）。这里按确认章前 4 位 = 年级册 level1
    #   节点 id，走 lazyTree 反查册名（chapter_name_for_id 整树压平命中），同步成老师确认的年级册名。
    #   FE 回传了 grade_book_name（config.configurable）则优先用它（免一次树查）。拉不到 → 不改
    #   （宁可留旧值也不抹空，不破现有显示）。仅在确认章驱动 grade_code 时同步（无确认章不动）。
    if confirmed_chapter_id and len(confirmed_chapter_id) >= 4:
        cfg_conf = (config or {}).get("configurable") or {}
        gb_name = str(cfg_conf.get("grade_book_name") or "").strip() or None
        if not gb_name:
            gb_name = await chapter_name_for_id(grade_code, client)
        if gb_name:
            grade_node = dict(analysis.get("grade") or {})
            grade_node["value"] = gb_name  # 人话名同步成老师确认的年级册（阶段灯/气泡显示用它）
            analysis["grade"] = grade_node

    try:
        leaf_pool = await leaf_pool_for_grade(
            grade_code, client, include_review_books=include_review_books
        )
    except Exception as e:  # noqa: BLE001 — 池拉取故障 → 空池（不 silent-fail，转 clarify）
        analysis["_anchor_error"] = str(e)

    # 🔴 首灯「锚定考点」running 已由 analyze 点亮（且可能定时翻过绿）→ 此处不重发 running
    #   防绿→running 回闪；只在锚定真实有结论时发终态 done/warn。

    if not leaf_pool:
        # 库/网络故障或年级池为空 → 锚定不可用 → 不抬置信 → clarify（既有降级路径）
        await client.aclose()
        analysis.setdefault("_anchor_error", "知识点叶子池不可用（库未起/年级未识别）")
        _emit_stage("classify", "锚定考点", "warn", "知识点池不可用，待老师确认")
        # 次灯「解析配方」：锚定不可用也要给个终态，免得灯永远停在 running
        _emit_stage("knobs", "解析配方", "warn", "待老师确认母题后再定配方")
        # 锚定不可用 = 三锚没定死 → mother_confirm.needs_confirm 必 true（与正常路径同口径）。
        early: VariantState = {"analysis": analysis, "mother_confirmed": False, "messages": []}
        early["mother_confirm"] = build_mother_confirm({**state, **early})
        return early

    # --- 🔴 PRD-C-017 B1：母题 opus 合并「解题 + 10 维 DNA 打标」（直读原图，一次调用） ---
    #   取代旧「nano dna_extract 抄图打标」。opus 直读母题图、在确认年级/章范围叶子池内打细 kp，
    #   一次输出 题面富文本(stem/answer/analysis) + 解答 + 10 维 DNA。母题侧零机器验证（决策表去 sympy）
    #   → opus 是唯一安全网：走 mother_solve_label 档（启动期 fail-fast 已锁 opus）+ 低温 0.1 +
    #   response_format 硬锁 schema + 超时 ≤180s；超时/失败 → SSE error，绝不静默退 gpt-5.4。
    #
    #   🔴 PRD-C-017 B2·闸B 章范围参数化：老师确认章 id（confirmed_chapter_id，经 config/state 回传）
    #   优先 → 收窄到章前缀；老师没给（旧线程/降级）→ 回退年级册 4 位 code（grade_code）作前缀。
    #   anchor_to_chapter 按此前缀做越界校验，opus 选的叶子 id 必以确认章 id 为前缀，越界拒/降级标注。
    #   🔴 confirmed_chapter_id 已在 classify 开头（B4-fix）解析并用于驱动 grade_code，此处直接复用。
    chapter_id = confirmed_chapter_id or grade_code  # B2：确认章 id 优先；回退年级册前缀

    # --- 🔴 M7（PRD-C-017 B2）·聚合/复习章排除：确认章若是册内 level2「中考一轮复习/期末专题/
    #   专题」（跨章大杂烩，误锚=制造想消灭的跨章串题 L-03 critical）→ 不拿它当锚定范围（降回年级册
    #   前缀），并标记。chapter_name_for_id 走既有 lazyTree 反查章名（拉不到 → 不当聚合章处置，
    #   宁可不排除也不误排）。
    chapter_text = (analysis.get("chapter") or {}).get("value") if isinstance(
        analysis.get("chapter"), dict
    ) else None
    if confirmed_chapter_id:
        chap_name = await chapter_name_for_id(confirmed_chapter_id, client)
        if chap_name:
            chapter_text = chapter_text or chap_name
        if mother_precheck.is_aggregation_chapter_name(chap_name):
            # 聚合章 → 排除其作锚定范围（降回年级册前缀），标记 + 告知
            analysis["_aggregation_chapter_excluded"] = chap_name or confirmed_chapter_id
            chapter_id = grade_code  # 不以聚合章 id 为前缀锚（防跨章串题）
            _emit_stage("classify", "锚定考点", "warn",
                        f"「{chap_name}」是聚合/复习章，已按年级册范围锚定（防跨章串题）")
    # 🔴 BUG-08（2026-06-19）：把章名落进 analysis.chapter → 随 classify 返回值进 state →
    #   _build_mother_card 回灌母题卡 anchor.chapter_name（空就不落，不抹既有值）。
    if chapter_text:
        analysis["chapter"] = {
            **(analysis.get("chapter") if isinstance(analysis.get("chapter"), dict) else {}),
            "id": confirmed_chapter_id or "",
            "value": chapter_text,
        }

    # --- 🔴 PRD-C-100 B2·重锚复用母题首解（root cause 根治 niche 考点重锚反复解析失败死循环） ---
    #   现象：niche 考点母题（如韦达定理）首解（mother_opus_entry）opus **已成功**解题+10维打标
    #   （母题卡显示了解题/骨架/DNA），但 _match_kp_in_pool/闸B 锚不到年级章内叶子（need_anchor_review）
    #   → confirmed=False → 弹真章树 picker、停 awaiting_mother_confirm。老师选定章确认后 route 回
    #   classify 重锚——旧实现这里**无条件重调 opus 解题打标**（solve_and_label_resilient），把已
    #   成功的首解产物（state.mother_dna 里的 stem/answer/analysis/solved_answer/skeleton/dna）整个丢掉、
    #   对同一道难图重新读图解题。opus 对该 niche 几何/压轴母题二次读图偶发坏 JSON（自愈网耗尽）→
    #   _SolveLabelError → 退回 picker → 老师再确认 → 又重 solve → 又失败 = 反复（Round B PARTIAL）。
    #   根因 = **重锚不该重 solve**：母题首解一次性成功就该一次性定稿，重锚只是「换锚定章」=
    #   纯代码重跑闸B（anchor_to_chapter）+ 模型锚，不碰 opus。这样 niche 考点首解成功过 → 重锚必成。
    #   复用前置（全满足才走）：① 有确认章（confirmed_chapter_id，= 重锚语境，非首图首解）；
    #   ② 首解来源是 opus（mother_solve_source=="opus"，排除 analyze 抄图骨架）；③ 首解富文本/DNA 还在
    #   state（stem 非空 + dna 是 dict + dna.main_kp 有 name）。任一不满足 → 落回原 opus 重 solve 路径
    #   （首解产物已丢/库内母题/异常态——此时重 solve 是唯一选项，仍吃下方自愈网 + graceful 降级）。
    _prev_dna = (mother_dna.get("dna") if isinstance(mother_dna.get("dna"), dict) else None) or {}
    _prev_main_kp = _prev_dna.get("main_kp") if isinstance(_prev_dna.get("main_kp"), dict) else None
    _reuse_ok = bool(
        confirmed_chapter_id
        and mother_dna.get("mother_solve_source") == "opus"
        and str(mother_dna.get("stem") or "").strip()
        and _prev_dna
        and _prev_main_kp
        and str(_prev_main_kp.get("name") or "").strip()
    )
    # 🔴 R2a·闸2（B5b）·range-fingerprint：范围错位才重解（窄子集，绝不全量重解撞 B2 死循环）。
    #   _reuse_ok 成立时再过一道「该不该刷新首解」判据——满足其一才**放弃复用、走全量重 solve**：
    #     (a) 册变：fp 非空 且 确认章前 4 位 ≠ 首解范围指纹（_solve_range_fp），= 老师把范围换到**另一
    #         年级册**，首解 solve/DNA 按旧册解的（学段进度/解法范围已失真）→ 须按新册重解；
    #     (b) 首解没锚牢 **且 是同册再锚**：首解 need_anchor_review / main_kp 锚空——但**仅当不是
    #         B2 复用语境时**才重解。
    #   🔴🔴 B2 死循环根治不可破（红线）：_reanchor_reuse_first_solve 的全部存在意义 = 首解 opus 成功
    #     产 DNA（_reuse_ok 已保证 stem + main_kp 有 name）但**锚不到叶子**（need_anchor_review）→ 老师
    #     在**同册**选定章后**复用首解 + 只换锚**，绝不重 solve（重 solve niche 题会坏 JSON → 死循环）。
    #     因此「首解没锚牢」在**同册**路径下恰恰是 B2 要复用的场景，**不得**据此重解。故 (b) 仅在「换册」
    #     时与 (a) 合流——换册后首解锚定本就作废，重解顺理成章。同册（含空 fp）一律走复用，B2 字节级不变。
    #   🔴 空 fp 策略（核心盲点）：纯文字母题首解判不出年级册 → _solve_range_fp 空串。**空 fp 不判册变**
    #     （否则非空确认章前缀必 ≠ 空 fp → 每次重锚都全量回退 = 撞 B2 死循环）。空 fp → _book_changed=False
    #     → 走复用（B2 同册路径），靠 _reanchor 内 graceful 降级兜锚不牢，不重解。
    _resolve_needed = False
    if _reuse_ok:
        _fp = str(mother_dna.get("_solve_range_fp") or "").strip()
        _new_book = (confirmed_chapter_id or "")[:4] if confirmed_chapter_id else ""
        # (a)+(b) 合流：仅「换册」触发重解（空 fp / 同册一律复用，护 B2）。换册后首解锚定作废，
        #   连带覆盖「首解没锚牢」——新册重解一次更稳，且不在 B2 同册复用路径上，不会引死循环。
        _resolve_needed = bool(_fp) and bool(_new_book) and _new_book != _fp
    if _reuse_ok and not _resolve_needed:
        return await _reanchor_reuse_first_solve(
            state=state, analysis=analysis, mother_dna=mother_dna, prev_dna=_prev_dna,
            grade_code=grade_code, chapter_id=chapter_id, leaf_pool=leaf_pool,
            confirmed_chapter_id=confirmed_chapter_id, include_review_books=include_review_books,
            knobs=state.get("knobs"),
        )

    image_url = state.get("image_url")
    opus_model = settings.variant_model("mother_solve_label")  # G3：母题必命中 opus（fail-fast 已锁）
    prompt = mother_opus.build_mother_prompt(
        grade_text=(analysis.get("grade") or {}).get("value") or grade_code or "",
        chapter_text=chapter_text,
        leaf_pool=leaf_pool,
        model_vocab=None,  # 模型词库快照（只读命名参考）；现阶段缺省，model_anchor 步另锚正式 M-id
    )
    # 🔴 PRD-C-100 B2·重锚自愈网（复用入口 mother_opus_entry 同口径，根治死循环）：旧实现这里只
    #   做「单次 solve_and_label + 单次 _parse_json」——opus 偶发坏 JSON（markdown fence/截断/未转义
    #   引号）即早退「没解析出来」，且不清确认态 → 老师再点开始仍走 parse 兜底 = 死循环。入口节点对
    #   同一 opus 有「重试≤2 + parse_or_repair_entry（确定性引号修复 + LLM 兜底）」自愈网，重锚没有 =
    #   不对称。此处复用 variant_entry.solve_and_label_resilient（与入口逐项同口径），不另造。
    from agents import variant_entry as _VE  # 懒导入防循环（variant_entry 顶层 import 本模块挂图）
    try:
        opus_data = await _VE.solve_and_label_resilient(
            # 🔴 PRD-C-104 B4：原 V=sys.modules[__name__]（彼时 __name__=="agents.variant"）→
            #   本模块 __name__ 变 stage1_anchor.label，显式取 facade 模块对象（V 行为字节级不变）。
            image_url=image_url or "", prompt=prompt, V=sys.modules["agents.variant"],
            model=opus_model,
            on_progress=lambda t: _emit_stage("classify", "锚定考点", "running", t),
        )
    except _VE._SolveLabelError as se:  # 自愈网耗尽（超时/全站失败 或 坏 JSON 重读仍解不出）
        await client.aclose()
        # 🔴 不卡死、不无限回环：导向可前进的 needs_confirm（弹真章树 picker，让老师重定章后再来），
        #   而非保留 stale unconfirmed 态让「开始举一反三」静默回 parse。清在途 review 态防误路由。
        reason = "opus 超时/异常" if not se.parse_only else "结果反复解析失败"
        analysis["_mother_opus_error"] = (str(se.last_exc)[:120] if se.last_exc else "opus 返回非 JSON（自愈仍失败）")
        _emit_stage("classify", "锚定考点", "warn", f"母题解题打标{reason}，请确认年级章后重试")
        _emit_stage("knobs", "解析配方", "warn", "待确认年级章后再定配方")
        grade_name = (analysis.get("grade") or {}).get("value") or grade_code or ""
        chap_name = chapter_text or ""
        _emit_need_confirm({
            "grade_book": {"id": grade_code or "", "name": grade_name},
            "chapter": {"id": confirmed_chapter_id or "", "name": chap_name},
            "grade_candidates": [{"id": grade_code or "", "name": grade_name}] if grade_name else [],
            "chapter_candidates": [{"id": confirmed_chapter_id or "", "name": chap_name}] if chap_name else [],
            "confidence": 0.0,
        })
        early: VariantState = {
            "analysis": analysis,
            "mother_confirmed": False,
            "facts_locked": False,
            "awaiting_mother_confirm": True,   # resume 走 route_entry → classify 重锚（新一次 opus）
            "awaiting_mother_review": False,    # 清 stale review，防「开始举一反三」误路由
            "messages": [AIMessage(content=(
                f"母题解题打标{reason}了。我已读出年级章范围，**请确认年级与章**后我再重试一次解题打标（"
                "确认无误回复「确认」，需要修改请直接告诉我正确的年级/章）。"
            ))],
        }
        early["mother_confirm"] = build_mother_confirm({**state, **early})
        return early

    # opus 富文本回填 mother_dna（题面/答案/解析 + 解答骨架进 _mother_facts 的来源·G4）
    rich = opus_data.get("richText") or {}
    if isinstance(rich, dict):
        if rich.get("stem"):
            mother_dna["stem"] = _sanitize_rich_text(rich.get("stem"))
        if rich.get("answer"):
            mother_dna["answer"] = _sanitize_rich_text(rich.get("answer"))
        if rich.get("analysis"):
            mother_dna["analysis"] = _sanitize_rich_text(rich.get("analysis"))
    # 🔴 G4：opus 解答骨架 = 守恒基准。solution_skeleton 写 opus 骨架（变式守恒注入引用它），
    #   solved_answer 单列。两者源 = opus 解答，**非 analyze 抄图骨架**。
    dna = mother_opus.opus_to_dna(opus_data)
    skeleton_lines = dna.get("skeleton") or []
    if skeleton_lines:
        mother_dna["solution_skeleton"] = join_skeleton(skeleton_lines)  # P8 逐行净化
    solved = opus_data.get("solvedAnswer")
    if solved:
        mother_dna["solved_answer"] = _sanitize_rich_text(solved)
    mother_dna["mother_solve_source"] = "opus"  # G4 断言锚点：骨架来源 = opus 解答
    # 🔴 R2a·闸2（B5b）首解范围指纹（写点③·classify opus-resolve 路）：记首解年级册 4 位 code。
    #   此处 grade_code 已由确认章前 4 位 / _resolve_grade_code 定出（首解真实落册）。
    mother_dna["_solve_range_fp"] = str(grade_code or "")

    # --- 🔴 闸A 富文本机器验证（G10，非 LLM）：坏 LaTeX/缺表 → 标问题（不直接放行） ---
    rt_check = mother_opus.validate_rich_text(
        rich if isinstance(rich, dict) else {},
        has_table=bool(opus_data.get("has_table")),
    )
    if not rt_check["ok"]:
        analysis["_richtext_issues"] = rt_check["issues"]
        mother_dna["need_richtext_review"] = True
        _emit_stage("classify", "锚定考点", "warn",
                    f"母题富文本机器检发现 {len(rt_check['issues'])} 处问题，待人工复核")

    # --- 🔴 闸B 锚定·宁空不凑（G11）：opus 主考点锚到「确认章 id 前缀内的叶子」，锚不到留空 ---
    dna = mother_opus.anchor_to_chapter(
        dna, chapter_id=chapter_id, leaf_pool=leaf_pool,
        include_review_books=include_review_books,
    )
    if dna.get("need_anchor_review"):
        mother_dna["need_anchor_review"] = True

    main_kp = dna.get("main_kp") if (dna.get("main_kp") or {}).get("id") else None

    await client.aclose()

    # --- 🔴 PRD-C-015 批2·W1' 模型锚定（双轴「怎么解」轴）：母题 DNA 抽完即锚 models ---
    #   按主+副 kp 反查候选（纯只读 ETL，≤8 按 sort）→ gpt-5.4-mini「先解题再选」确认 ≤3（池内选/禁造词）；
    #   无命中→M00 保底（模型维永不为空）；池外名→落待命名池+⚠（不入正式维）；反查库故障→M00+⚠。
    #   写进 mother_dna.dna.models / model_overflow（契约 v2，批1 已留字段位）。
    try:
        m_ref = str((main_kp or {}).get("id") or "") or None
        m_res = await model_anchor.anchor_models(
            dna,
            stem=mother_dna.get("stem") or "",
            answer=mother_dna.get("answer") or mother_dna.get("solution_skeleton") or "",
            invoke=_ainvoke_text,
            model=settings.variant_model("model_confirm"),  # H2：确认档走 gpt-5.4-mini（.env VARIANT_MODEL_MODEL_CONFIRM）
            record_overflow=lambda name, mm: model_anchor.record_overflow_candidate(
                name, mm, question_ref=m_ref
            ),
        )
    except Exception as e:  # noqa: BLE001 — 锚定整体故障也得有 models（M00 兜底，绝不空维/卡死）
        analysis.setdefault("_model_anchor_error", str(e))
        m_res = {"models": [dict(model_anchor.M00)], "temp_models": [], "model_overflow": [],
                 "model_warn": True, "model_flag": "lookup_unavailable"}
    dna["models"] = m_res.get("models") or [dict(model_anchor.M00)]
    dna["model_overflow"] = m_res.get("model_overflow") or []
    # 🔴 WS2·AC5：临时模型随 DNA 透传（供 WS2 落链/转正脚本消费；参与判档已在 models 内体现）。
    dna["temp_models"] = m_res.get("temp_models") or []
    if m_res.get("model_warn"):
        dna["model_warn"] = True
    mother_dna["dna"] = dna

    if main_kp and main_kp.get("id"):
        # 池内锚到真知识点 → anchored.code = 知识点叶子 code（dim1KpId 主 kp）
        kp_node["anchored"] = {
            "id": main_kp["id"],
            "code": str(main_kp["id"]),  # 叶子 id 即层级编码（biz_subject 无独立 code 列）
            "name": main_kp.get("name"),
        }
        if main_kp.get("name"):
            kp_node["value"] = main_kp["name"]
        kp_node["confidence"] = max(float(kp_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["kp"] = kp_node
        # 年级锚定：抬置信（已圈到该年级池且选中其内 id）
        grade_node = dict(analysis.get("grade") or {})
        if grade_code:
            grade_node["code"] = grade_code
        grade_node["confidence"] = max(
            float(grade_node.get("confidence", 0) or 0), CONF_GATE
        )
        analysis["grade"] = grade_node
        # 题型：DNA 抽到的池外校验后题型，回填抬置信
        if dna.get("qtype"):
            qn = dict(analysis.get("qtype") or {})
            qn["value"] = dna["qtype"]
            qn["confidence"] = max(float(qn.get("confidence", 0) or 0), CONF_GATE)
            analysis["qtype"] = qn
    # else: 池内无匹配（main_kp=None）→ anchored 缺失 → 不抬置信 → clarify（不放行出题）

    confirmed = _conf_ok(analysis) and bool(kp_node.get("anchored"))
    kp_name = (analysis.get("kp") or {}).get("value") or "?"
    grade_name = (analysis.get("grade") or {}).get("value") or grade_code or "?"
    _emit_stage(
        "classify",
        "锚定考点",
        "done" if confirmed else "warn",
        f"考点「{kp_name}」·年级「{grade_name}」",
    )
    # 🔴 次灯「解析配方」（改动2）：合并调用 + 锚定真实完成后发 done，带道数（knobs 已由
    #   analyze 抽好随 state 来）。confirmed 才落定配方进 generate；未 confirmed 走 clarify，
    #   配方虽已抽出但本轮不出题 → 给 warn 终态（不停在 running）。
    knobs = state.get("knobs")
    recipe = knobs_desc(knobs) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）"
    _emit_stage(
        "knobs", "解析配方", "done" if confirmed else "warn", recipe
    )
    # 🔴 批3·事实源冻结：classify 是锚定（LLM 来源）写入点，跑完即定死 → 此处置 facts_locked。
    #   confirmed（定死）→ 冻结（此后 LLM 回写被 setter 忽略）；未 confirmed → 不冻结（等老师
    #   补/纠正后重锚，重锚仍是合法锚定路径，不该被冻结挡住）。
    out: VariantState = {
        "analysis": analysis,
        "mother_dna": mother_dna,
        "mother_confirmed": bool(confirmed),
        "facts_locked": bool(confirmed),
        # 🔴 B2：进入 classify = 母题确认环节已过（resume 或首图无确认阻断）→ 清确认等待态，
        #   留痕本次确认章 id（接闸B 用），防下一轮再被 route 当成在途确认。
        "awaiting_mother_confirm": False,
        "confirmed_chapter_id": confirmed_chapter_id,
        "messages": [],
    }
    # 🔴 PRD-C-015 批1·classify 注入点（缺口5 合并确认闸 + D-merge7 确定性异常门控）：
    #   抽完母题完整 DNA 后，把「年级+主考点三锚置信门控（C-014）」与「守恒维确定性异常」合并算成
    #   mother_confirm（needs_confirm = 有异常 ∨ 三锚没定死）。这是 §3.5 GateCheck→MergedConfirm/
    #   EmitVariants 的算据；批4/5 接 UI 弹合并确认面 / 直接放行。批1 先把状态算齐写进契约 v2。
    out["mother_confirm"] = build_mother_confirm({**state, **out})
    # 🔴 PRD-C-017 B3.5·母题卡「先出」专帧（AC4/G12）：opus 已产出 mother_dna（含解答/10维DNA/锚定），
    #   在 classify return 前（generate 出 items 之前）单发一帧 header.mother_card，让母题卡早于变式
    #   渲染、且携全字段（stem/solved_answer/副考点 id/anchor 章 id/need_anchor_review）供入库。
    #   复用 mother_dna（绝不重调 opus）；{**state,**out} 让组卡读到本轮最终 mother_dna/confirmed_chapter_id。
    _emit_mother_card({**state, **out})
    # 🔴 BUG-02：「母题切图」节点专帧（母题就绪、figure 判定后）——据 mother_has_figure 发 done。
    _emit_figure_stage({**state, **out})
    return out


async def _reanchor_reuse_first_solve(
    *,
    state: VariantState,
    analysis: dict[str, Any],
    mother_dna: dict[str, Any],
    prev_dna: dict[str, Any],
    grade_code: str | None,
    chapter_id: str | None,
    leaf_pool: list[tuple[str, str]],
    confirmed_chapter_id: str | None,
    include_review_books: bool,
    knobs: Any,
) -> VariantState:
    """🔴 PRD-C-100 B2·重锚复用母题首解（不重调 opus）：母题首解（mother_opus_entry）opus 已
    成功产出题面富文本 + 解答 + 10 维 DNA（都在 state.mother_dna），但当时锚不到年级章内叶子 →
    弹 picker。老师确认章后本函数**只换锚定章**：拿已有 DNA 的 main_kp **名**在确认章收窄后的
    leaf_pool 内重锚（_match_kp_in_pool + 闸B anchor_to_chapter）+ 重跑模型锚 → 抬置信 → 母题卡。
    **绝不重读图/重解题**——niche 考点首解成功过就不会在重锚再失败（根治 Round B 反复解析失败）。

    graceful 降级（铁律④·闸门必有降级路径）：老师已**显式确认章**，若该确认章收窄池里仍无贴切
    叶子（极端 niche），不再退回 picker 卡死，而是把 main_kp 锚到**确认章节点本身**（chapter_id，
    = 老师亲选范围，非凭空造叶子）+ need_anchor_review=True（标待人审）→ confirmed=True → 照常出
    变式（守恒注入用 kp 名 + 确认章范围，仍不超纲）。这是「问过老师后按其选定范围出题」，不是「瞎锚」。
    """
    # 🔴 PRD-C-104 B4：_emit_mother_card / _emit_figure_stage 体内延迟取（破 label↔mother_card 循环）。
    from agents.variant import _emit_figure_stage, _emit_mother_card  # noqa: E402
    from agents import model_anchor
    from agents import variant_entry as _VE

    mother_dna = dict(mother_dna)
    analysis = dict(analysis)
    kp_node = dict(analysis.get("kp") or {})

    # 复用首解 DNA（深拷一份再改锚定字段，不污染 state 原对象）。
    dna = dict(prev_dna)
    main_kp_name = str((dna.get("main_kp") or {}).get("name") or "").strip()

    # ① 名 → 确认章收窄池重锚 id（首解 main_kp.id 可能空/越界，按名在新池重找）。
    matched = _VE._match_kp_in_pool(main_kp_name, leaf_pool) if main_kp_name else None
    if matched:
        dna["main_kp"] = {"id": matched, "name": main_kp_name}
    # 副 kp 同理按名在新池补 id（锚不到留原样，闸B 会逐项校验丢越界）。
    for s in dna.get("secondary_kps") or []:
        if isinstance(s, dict) and not str(s.get("id") or "").strip() and s.get("name"):
            mid = _VE._match_kp_in_pool(str(s["name"]).strip(), leaf_pool)
            if mid:
                s["id"] = mid

    # ② 闸B 锚定·宁空不凑（与 classify 同一纯函数，逻辑一致）。
    dna = mother_opus.anchor_to_chapter(
        dna, chapter_id=chapter_id, leaf_pool=leaf_pool,
        include_review_books=include_review_books,
    )
    main_kp = dna.get("main_kp") if (dna.get("main_kp") or {}).get("id") else None

    # ③ graceful 降级：老师已确认章，仍锚不到叶子（极端 niche）→ 锚到确认章节点本身 + 待人审，
    #    不退回 picker 卡死（铁律④）。chapter_id 优先用确认章 id（老师亲选范围）。
    degraded = False
    if main_kp is None and main_kp_name:
        fallback_chap = str(confirmed_chapter_id or chapter_id or grade_code or "").strip()
        if fallback_chap:
            dna["main_kp"] = {"id": fallback_chap, "name": main_kp_name}
            dna["need_anchor_review"] = True
            main_kp = dna["main_kp"]
            degraded = True

    if dna.get("need_anchor_review"):
        mother_dna["need_anchor_review"] = True

    # ④ 模型锚（双轴·与 classify 同口径）：M00 兜底，故障不空维。
    try:
        m_ref = str((main_kp or {}).get("id") or "") or None
        m_res = await model_anchor.anchor_models(
            dna,
            stem=mother_dna.get("stem") or "",
            answer=mother_dna.get("answer") or mother_dna.get("solution_skeleton") or "",
            invoke=_ainvoke_text,
            model=settings.variant_model("model_confirm"),
            record_overflow=lambda name, mm: model_anchor.record_overflow_candidate(
                name, mm, question_ref=m_ref
            ),
        )
    except Exception as e:  # noqa: BLE001 — 锚定整体故障也得有 models（M00 兜底）
        analysis.setdefault("_model_anchor_error", str(e))
        m_res = {"models": [dict(model_anchor.M00)], "temp_models": [], "model_overflow": [],
                 "model_warn": True, "model_flag": "lookup_unavailable"}
    dna["models"] = m_res.get("models") or [dict(model_anchor.M00)]
    dna["model_overflow"] = m_res.get("model_overflow") or []
    dna["temp_models"] = m_res.get("temp_models") or []  # WS2·AC5 临时模型透传
    if m_res.get("model_warn"):
        dna["model_warn"] = True
    mother_dna["dna"] = dna

    # ⑤ 锚到 kp（真叶子 或 降级锚到确认章节点）→ 抬三锚置信（与 classify 同口径）。
    if main_kp and main_kp.get("id"):
        kp_node["anchored"] = {
            "id": main_kp["id"],
            "code": str(main_kp["id"]),
            "name": main_kp.get("name"),
        }
        if main_kp.get("name"):
            kp_node["value"] = main_kp["name"]
        kp_node["confidence"] = max(float(kp_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["kp"] = kp_node
        grade_node = dict(analysis.get("grade") or {})
        if grade_code:
            grade_node["code"] = grade_code
        grade_node["confidence"] = max(float(grade_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["grade"] = grade_node
        if dna.get("qtype"):
            qn = dict(analysis.get("qtype") or {})
            qn["value"] = dna["qtype"]
            qn["confidence"] = max(float(qn.get("confidence", 0) or 0), CONF_GATE)
            analysis["qtype"] = qn

    kp_name = (analysis.get("kp") or {}).get("value") or main_kp_name or "?"
    grade_name = (analysis.get("grade") or {}).get("value") or grade_code or "?"

    # 🔴 PRD-A-021 R2a·闸3（BUG-03）·锚定章↔主考点冲突闸断（取代旧「⚠ 警告 + 强锚放行」）：
    #   degraded = 老师选定章里锚不到主考点真叶子 = 手选章与 AI 判主考点不一致。旧实现静默强锚到章节点
    #   + confirmed=True → 出一组锚错章的变式。改向：**首次冲突 → 闸断/强确认**（不放行出题），把冲突
    #   播给老师，让其「再点确认（坚持按此章出）」或「换正确的章」。**只收窄到此「复用首解强锚」危险路**——
    #   fresh classify 路锚不到本就走 confirmed=False→clarify（安全），不在此被误伤。
    #   防死循环：老师**第二次确认同一章**（FE 再回传同一 confirmed_chapter_id）→ 视为坚持 → 接受强锚
    #   放行（_bug03_gated_chapter 记过该章，本轮等于）。换别的章 → 重新锚定（_should_resolve/复用再判）。
    _bug03_insisted = bool(
        degraded
        and confirmed_chapter_id
        and str(state.get("_bug03_gated_chapter") or "").strip() == str(confirmed_chapter_id).strip()
    )
    if degraded and not _bug03_insisted:
        _emit_stage("classify", "锚定考点", "warn",
                    f"所选章里没有主考点「{kp_name}」的具体叶子——请确认是否选错章")
        _emit_stage("knobs", "解析配方", STAGE_AWAIT, "待确认章后定配方")
        _emit_need_confirm({
            "grade_book": {"id": grade_code or "", "name": grade_name},
            "chapter": {"id": confirmed_chapter_id or "", "name": ""},
            "grade_candidates": [{"id": grade_code or "", "name": grade_name}] if grade_name else [],
            "chapter_candidates": [],
            "confidence": 0.0,
        })
        gated: VariantState = {
            "analysis": analysis,
            "mother_dna": mother_dna,
            "mother_confirmed": False,
            "facts_locked": False,
            "awaiting_mother_confirm": True,   # resume → route_entry → classify 重锚（带确认章）
            "awaiting_mother_review": False,    # 清 stale review，防「开始举一反三」误路由
            "confirmed_chapter_id": confirmed_chapter_id,
            "_bug03_gated_chapter": confirmed_chapter_id,  # 记过此章，老师再确认同章即放行（防死循环）
            "messages": [AIMessage(content=(
                f"⚠ 你选的这一章里**没有**主考点「{main_kp_name}」对应的知识点叶子——可能选错了章。\n\n"
                "请核对：\n"
                "- 若**确实是这一章**（就按此章范围出题，标「锚定待人审」）→ 请**再回复一次「确认」**；\n"
                "- 若**选错了章** → 直接告诉我正确的章，我重新锚定。"
            ))],
        }
        gated["mother_confirm"] = build_mother_confirm({**state, **gated})
        _emit_mother_card({**state, **gated})  # 母题卡仍先出（复用首解全字段，等老师定章）
        _emit_figure_stage({**state, **gated})
        return gated

    confirmed = _conf_ok(analysis) and bool(kp_node.get("anchored"))
    _detail = f"考点「{kp_name}」·年级「{grade_name}」（复用母题首解，未重解）"
    if degraded:
        # 老师二次确认同章（坚持）→ 接受强锚：仍按所选章范围出题，标待人审（铁律④不卡死）。
        _detail += f"·⚠ 主考点「{kp_name}」未落到所选章的具体叶子（按所选章范围锚定·待人审）"
    _emit_stage("classify", "锚定考点", "done" if confirmed else "warn", _detail)
    recipe = knobs_desc(knobs) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）"
    _emit_stage("knobs", "解析配方", "done" if confirmed else "warn", recipe)

    # 🔴 BUG-03：降级（手选章锚不到主考点叶子）→ 气泡里把冲突说清（不静默放行），但仍按所选章出题。
    _conflict_msgs = (
        [AIMessage(content=(
            f"提示：主考点「{main_kp_name}」没能落到你选的章里的具体知识点叶子——可能这道母题的主考点"
            f"不在该章。我先按你选定的章范围出变式（已标「锚定待人审」），若你觉得选错了章，直接告诉我正确的章。"
        ))]
        if degraded else []
    )
    out: VariantState = {
        "analysis": analysis,
        "mother_dna": mother_dna,
        "mother_confirmed": bool(confirmed),
        "facts_locked": bool(confirmed),
        "awaiting_mother_confirm": False,
        "confirmed_chapter_id": confirmed_chapter_id,
        # 闸3 放行后清闸断标记（下次别的纠正不带 stale）。
        "_bug03_gated_chapter": None,
        "messages": _conflict_msgs,
    }
    out["mother_confirm"] = build_mother_confirm({**state, **out})
    _emit_mother_card({**state, **out})  # 母题卡仍先出（复用首解的全字段）
    _emit_figure_stage({**state, **out})  # 🔴 BUG-02：「母题切图」节点据 mother_has_figure 发 done
    return out
