"""variant 引擎 · stage2_variant 双闸层（PRD-C-104 B3b 抽出，纯搬零改）。

闸A·基因闸：gene_gate / gene_gate_check / _model_conservation_check。
闸B·验算：solve_explain + _anti_degen_gate / _surface_check /
  _conservation_ok / _regen_once。

🔴 行为零改：pass/fail 判决只读 math_verify verdict（铁律）。
_anti_degen_gate 调 _solve_one（stage1 解题机器，跨阶段共享件**不搬**，留 __init__）、
_regen_once 用 REGEN_PROMPT（同包 prompts）；其余依赖从 facade agents.variant 运行期取。
"""

from __future__ import annotations

import asyncio
import difflib
import json
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents import math_verify, model_anchor  # noqa: E402

from agents.variant.stage2_variant.prompts import REGEN_PROMPT  # noqa: E402
from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾导入）
    GATE_CONCURRENCY,
    GENE_GATE_PASS,
    GENE_GATE_SKIPPED,
    GENE_GATE_WARN,
    MAX_DEGEN_REGEN,
    STAGE_AWAIT,
    TIER_PENDING,
    VERIFY_PENDING,
    VariantState,
    VERIFY_SYMPY_PASS,
    _ainvoke_text,
    _append_card_note,
    _apply_visibility,
    _auto_verify_on,
    _budget_bind,
    _budget_exhausted,
    _check_one_item,
    _conservation_clause,
    _context_block,
    _degeneracy_verdict,
    _emit_stage,
    _exam_type_conserved,
    _format_item_stem,
    _gene_one_item,
    _machine_verify,
    _maybe_note_card_block,
    _mother_facts,
    _norm,
    _parse_json,
    _qtype_change_clause,
    _qtype_conserved,
    _regen_max_tokens,
    _sanitize_item,
    _scene_change_clause,
    _solve_one,
    _surface_norm_stem,
    _surface_nums,
    _surface_threshold_for_qtype,
    _sympy_gate_on,
    _variant_model_ids,
    settings,
    structure_lint,
)


async def _anti_degen_gate(
    item: dict, facts: dict, idx: int, total: int
) -> tuple[dict, bool]:
    """⑦ 反退化闸（闸B 内·sympy 答案 PASS 之后跑）。返回 (item, dropped)。

    退化（最优点落区间端点）→ REGEN ≤MAX_DEGEN_REGEN 次；重生稿须**非退化 + sympy 重 PASS**才采纳，
    否则超限**弃该变式**（dropped=True，§1⑦「超限则弃」）。非退化 / 不可判（降级）→ 原样放行。
    🔴 判决只读代数返回值；from_edit（老师点名）题不弃不换（老师意志优先，标 ⚠ 注记放行）。
    🔴 预算耗尽 → 跳过 REGEN，标 ⚠ 放行（降级路径，不卡死）。
    """
    res = await _degeneracy_verdict(item)
    if res.get("verdict") != math_verify.DEGENERATE:
        return item, False  # 非退化 / 不可判（degrade）→ 放行（闸门降级路径）

    # 🔴 PRD-C-103 WS4·AC10/D5：sympy 硬门关（默认）→ 退化构型也**不剔除、不回炉**，标 ⚠ 放行交人审
    #   （线上正确性靠人工审核兜底；sympy 仍算出退化判定供徽章/审计，但绝不卡流程到 assemble）。
    if not _sympy_gate_on(config=None):
        _append_card_note(item, "⚠ 反退化闸：检出退化构型（sympy 硬门已关），已放行交人审")
        return item, False

    # 已确认退化构型：老师点名编辑题不动（老师意志优先，仅标注交人审）。
    if item.get("from_edit"):
        _append_card_note(item, "⚠ 反退化闸：最优点落动点区间端点（退化构型），已保留老师原题交人审")
        return item, False

    attempt = 0
    cur = item
    while attempt < MAX_DEGEN_REGEN and not _budget_exhausted():
        attempt += 1
        _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道最优点退化到端点，回炉重生中")
        feedback = (
            f"程序(代数)判定该题为**退化构型**：{res.get('detail')}。"
            "请重新出一道题面与标答自洽的等价变式，务必让最优解（最值取得点）落在**动点定义区间的内部驻点**，"
            "严禁让最优点落在区间端点（如动点与定点重合 / 取临界值使解法机制失效）。"
        )
        draft = await _regen_once(cur, facts, feedback=feedback)
        if not draft:
            break
        # 重生稿须：① 非退化 ② sympy 答案重 PASS（保证答案仍对）→ 才采纳。
        d_res = await _degeneracy_verdict(draft)
        if d_res.get("verdict") == math_verify.DEGENERATE:
            cur = draft  # 仍退化 → 继续重生（受 MAX 上限约束）
            res = d_res
            continue
        resolved = await _solve_one(draft.get("stem", ""))
        if resolved.get("solution"):
            draft["solution"] = resolved.get("solution")
        r_res = await _machine_verify(draft, resolved.get("solved_answer"))
        if r_res.get("verdict") == math_verify.PASS:
            # 采纳非退化重生稿：保留闸A gene 印记，回写 sympy 验算 check。
            draft["gene"] = item.get("gene") or {"gate": GENE_GATE_SKIPPED, "reason": "degen-regen"}
            draft["check"] = {
                "badge": "ok",
                "solved_answer": resolved.get("solved_answer"),
                "verify": VERIFY_SYMPY_PASS,
                "verify_detail": r_res.get("detail"),
                "computed": r_res.get("computed"),
            }
            _apply_visibility(draft)
            return draft, False
        cur = draft  # 重生稿非退化但答案没过 → 视同失败，继续（受 MAX 上限约束）

    # 超限仍退化 / 重生失败 / 预算耗尽 → 弃该变式（§1⑦「超限则弃」）。
    dropped = dict(cur)
    dropped["_dropped"] = "1 道题为退化构型（最优点落动点区间端点，重生仍未脱退化），已剔除"
    return dropped, True


def _conservation_ok(solved_kp: str, solved_grade: str, facts: dict) -> bool:
    """主考点 + 年级 守恒校验（重生版仍须过；破则丢弃重生保留原版打⚠）。

    宽松包含匹配：标准考点/年级名 与 solve 回报的实际考点/年级 互含即视为守恒。
    """
    mk, mg = _norm(facts["kp_name"]), _norm(facts["grade"])
    sk, sg = _norm(solved_kp), _norm(solved_grade)
    kp_ok = (not sk) or (mk in sk) or (sk in mk) or (mk[:4] and mk[:4] in sk)
    # 年级守恒：比对编码前2字(如"七年")或互含
    grade_ok = (not sg) or (mg[:2] and mg[:2] in sg) or (mg in sg) or (sg in mg)
    return bool(kp_ok and grade_ok)


def _surface_check(variant_stem: Any, mother_stem: Any, qtype: Any = None) -> str | None:
    """🔴 表皮距离纯函数（T4 + AC4 按题型定阈值）：题干过近 = 抄母题；数字与母题完全相同也算。

    返回缺陷描述（疑似抄题，上层打 ⚠ flag）或 None（正常变式，表皮已换）。
    阈值按 qtype 取（选择/填空=0.85；解答/证明=0.92），未传或未知 qtype 回默认 0.85。
    母题题干为空 → 无从比对 → None（不误报）。纯函数，可单测。
    """
    m = _surface_norm_stem(mother_stem)
    if not m:
        return None
    v = _surface_norm_stem(variant_stem)
    ratio = difflib.SequenceMatcher(None, v, m).ratio()
    thr = _surface_threshold_for_qtype(qtype)
    if ratio > thr:
        return f"题干与母题相似度{ratio:.2f}>{thr}（疑似抄题，表皮未换）"
    v_nums, m_nums = _surface_nums(variant_stem), _surface_nums(mother_stem)
    if v_nums and v_nums == m_nums:
        return "数字与母题完全相同（表皮没换）"
    return None


async def _regen_once(item: dict, facts: dict, feedback: str | None = None) -> dict | None:
    """REGEN 回炉一次：返回不带 check 的重生草稿（解析失败/LLM 异常返 None）。

    feedback（如 sympy 的 computed/detail，或闸A 结构定向反馈）注回 prompt 的题干段。
    🔴 RC2（PRD-C-013）：item 带 edit_note（老师点名改造的软约束）→ 永远把老师 note 注回
    prompt，feedback 不能把它冲掉（基因/验算失败原因与老师意志并存，老师意志优先）。
    🔴 LLM 调用异常吞掉返 None（G5）：回炉是增强不是关卡，网关抖动时调用方按
    "重生失败 → 保留原版打 ⚠/warn" 的既有降级路径走，绝不炸掉整轮出题。
    """
    stem = str(item.get("stem") or "")
    edit_note = str(item.get("edit_note") or "").strip()
    if edit_note:
        stem = f"{stem}\n\n[老师要求·须保留] {edit_note}"
    if feedback:
        stem = f"{stem}\n\n[程序验算反馈] {feedback}"
    # B5-fix4：老师显式改了题型（edit-dna field=qtype → dirty_dims 含 "qtype"）→ 注强指令，
    # 让重生按新题型重构题面（压过"等价变式保型"框架）；没改题型则该串为空，保型路完全不变。
    new_qtype = item.get("qtype") or facts["qtype"]
    qtype_change_clause = (
        _qtype_change_clause(new_qtype)
        if "qtype" in (item.get("dirty_dims") or [])
        else ""
    )
    # B5-fix5：老师显式改了场景（edit-dna field=scene → dirty_dims 含 "scene"，落
    # mother_dna.dna.scene = 组级共享，全组重生都带新场景）→ 注强指令，让重生把题面
    # 改写到指定场景（压过"等价变式保型 + 守恒段随机换场景"）；没改场景则该串为空，保型路完全不变。
    new_scene = (facts.get("dna") or {}).get("scene")
    scene_change_clause = (
        _scene_change_clause(new_scene)
        if "scene" in (item.get("dirty_dims") or []) and str(new_scene or "").strip()
        else ""
    )
    try:
        regen_text = await _ainvoke_text(
            [
                HumanMessage(
                    content=REGEN_PROMPT.format(
                        kp_name=facts["kp_name"],
                        grade=facts["grade"],
                        stem=stem,
                        level=item.get("level") or "normal",
                        qtype=item.get("qtype") or facts["qtype"],
                        difficulty=item.get("difficulty") or 3,
                        injected_kp=json.dumps(item.get("injected_kp"), ensure_ascii=False),
                    )
                    # 🔴 整改1：确定上下文硬约束（回炉重出同样压解题不越界）。
                    + "\n\n"
                    + _context_block(facts)
                    # 🔴 W2 守恒硬约束注入（T1）：回炉重出仍守白名单/考察类型/最难步基因。
                    + "\n\n"
                    + _conservation_clause(facts.get("dna"))
                    # 🔴 批3·W2' 注卡（回炉重出同样照模型卡片，难度≥3+非M00 才注，含反退化约束）。
                    + _maybe_note_card_block(facts)
                    # 🔴 B5-fix4·题型改造强指令（老师显式改题型才注，压过"等价变式保型"框架；
                    #   未改题型时为空串，保型重出路完全不变）。放末尾 = 最末读到优先级最高。
                    + qtype_change_clause
                    # 🔴 B5-fix5·场景改写强指令（老师显式改场景才注，压过"等价变式保型 + 守恒段
                    #   随机换场景"；未改场景时为空串，保型/随机换场景路完全不变）。与 qtype 并列同注。
                    + scene_change_clause
                )
            ],
            # 🔴 整改4：回炉瘦身 max_tokens 上限压输出失控（出题主调用 generate/add 不动）。
            max_tokens=_regen_max_tokens(),
            # 🔴 回炉 = 出题环节（红线不降档）：走 generate 档（默认 COMPATIBLE_MODEL）。
            model=settings.variant_model("generate"),
        )
    except Exception:  # noqa: BLE001 — 回炉 LLM 异常 → 视同重生失败（G5）
        return None
    regen = _parse_json(regen_text)
    if isinstance(regen, dict) and regen.get("stem"):
        draft = _sanitize_item(
            {
                "stem": regen.get("stem"),
                "answer": regen.get("answer"),
                "solution": regen.get("solution"),
                "qtype": regen.get("qtype") or item.get("qtype"),
                "difficulty": regen.get("difficulty") or item.get("difficulty"),
                "level": regen.get("level") or item.get("level"),
                "injected_kp": regen.get("injected_kp"),
            }
        )
        # 4a：重生稿自带的验算载荷随新题走（旧题载荷绝不沿用——题面已换）
        if isinstance(regen.get("verify_payload"), dict):
            draft["verify_payload"] = regen["verify_payload"]
        # 🔴 批3·⑦：重生稿自带的反退化载荷随新题走（旧题 degen 载荷绝不沿用，题面已换）。
        if isinstance(regen.get("degen_payload"), dict):
            draft["degen_payload"] = regen["degen_payload"]
        # 🔴 RC2：老师 note + 编辑轮印记跟草稿走，下一次 REGEN 仍能注回老师意志、仍只判不回炉
        for k in ("edit_note", "from_edit"):
            if item.get(k):
                draft[k] = item[k]
        return draft
    return None


async def solve_explain(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ solve + 闸B 程序验算（PRD-C-010；判决语义见 _check_one_item，单一事实源）。

    P2（PRD-C-012）：循环体已抽成 per-item 协程 _check_one_item，本节点按题
    asyncio.gather + Semaphore(GATE_CONCURRENCY=3) 并发，结果按原下标顺序回填；
    已带 check 的题照旧跳过不重判；generate 流内 eager 剔除的哨兵题
    （item._dropped=叙事）在此收口进 dropped_notes（语义不变：每轮重写）。
    🔴 凡进 items 的题一律过本节点，无 check 不许进 assemble（remove 后旧题带 check 原样通过）。
    """
    budget = _budget_bind(state)  # P13：轮内下游节点携带预算（闸B heal/replenish 受闸）
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    total = len(items)
    sem = asyncio.Semaphore(GATE_CONCURRENCY)

    # 🔴 BUG-09（2026-06-19）：手动验算（auto_verify=False，产品默认）—— 跳过 _solve_one+_machine_verify
    #   +回炉+反退化闸，每题挂 pending（待老师手动点验算），**绝不因未验算被剔除**（pending≠dropped）。
    #   终态非阻塞「done」帧，流程秒到「题组就绪」不挂起。判分铁律不破：pending 只是「还没验」，
    #   老师点 /variant/verify-one 或 /variant/reverify 才真跑 sympy（判决仍只读 verdict）。
    if not _auto_verify_on(config):
        out: list[dict[str, Any]] = []
        for it in items:
            item = dict(it)
            if item.get("_dropped"):
                # 极端：上游已带剔除哨兵（手动模式下 generate 不产生，仅兜底）→ 转 pending 保留不剔
                item.pop("_dropped", None)
            if not item.get("check"):
                item["check"] = {
                    "badge": "ok",
                    "solved_answer": None,
                    "verify": VERIFY_PENDING,
                    "tier": TIER_PENDING,
                }
            _format_item_stem(item)  # 题型模版规范（与 assemble 同口径，幂等）
            out.append(item)
        # 🔴 C1/A-10/M2（PRD-A-018）：手动模式（auto_verify=False，产品默认）下程序验算是「老师自选可选
        #   旁挂步」，**不再发 done**（否则状态条把「程序验算」判完成绿，与每题 check.tier=pending 自相矛盾，
        #   还会把验算挂进「题组就绪/全部完成」必经链）。改发未完成态 STAGE_AWAIT（中性·可选），detail
        #   「待老师自选验算」。每题 check 仍 {verify:pending, tier:pending} 不变（FE 渲染「待验算」徽章 +
        #   验算按钮）。FE 把「程序验算」从 coreDone/allDone 必经链摘出，题组就绪只依赖 generate+gene_gate(+真配图)。
        _emit_stage("verify", "程序验算", STAGE_AWAIT, "待老师自选验算")
        return {"items": out, "dropped_notes": [], "llm_call_budget": budget, "messages": []}

    async def _run(i: int, it: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        item = dict(it)
        # generate 流内 eager 已剔除（4d 方案A）→ 哨兵只收叙事，不外发、不重验
        if item.get("_dropped"):
            return None, str(item["_dropped"])
        if item.get("check"):  # 已定状态 → 不重复（不进并发槽、不发 stage）
            return item, None
        async with sem:
            return await _check_one_item(item, facts, i, total)

    results = await asyncio.gather(*(_run(i, it) for i, it in enumerate(items)))
    out = [it for it, _ in results if it is not None]
    dropped = [note for _, note in results if note]  # 本轮剔除叙事（按原题序）

    if dropped:
        _emit_stage("verify", "程序验算", "done", f"剔除 {len(dropped)} 道，保留 {len(out)} 道")
    else:
        _emit_stage("verify", "程序验算", "done")
    return {"items": out, "dropped_notes": dropped, "llm_call_budget": budget, "messages": []}


def _model_conservation_check(item: dict, facts_i: dict) -> dict[str, Any]:
    """④ 🔴 批3·W3' 模型守恒软警（纯函数·零 LLM·**不打回**·可单测）。返回 {warn, out_of_set}。

    守恒集合 = 母题 models ∪ {M00} ∪ 反查候选池（H2 放宽：候选池 = facts_i.dna.models 之外
    DNA 携带的反查候选 model_pool，若有；防 mini 欠选伴生模型致误报）。变式无独立 models →
    继承母题 → 视同守恒（warn=False）。判越界仅打 ⚠ 透传，绝不打回/剔题/回炉（铁律：软警不硬）。
    """
    dna = facts_i.get("dna") or {}
    mother_ids = [
        str(m.get("id") or "").strip()
        for m in (dna.get("models") or [])
        if isinstance(m, dict) and str(m.get("id") or "").strip()
    ]
    # 反查候选池（H2 放宽集）：锚定时若把候选池随 DNA 带（dna.model_pool），并入守恒集合。
    pool = [str(x).strip() for x in (dna.get("model_pool") or []) if str(x).strip()]
    return model_anchor.model_conservation_warn(
        _variant_model_ids(item), mother_ids, candidate_pool_ids=pool
    )


def gene_gate_check(item: dict, facts_i: dict) -> dict[str, Any]:
    """🔴 闸A 纯代码三检（B2·T2，零 LLM / 零 IO，可单测）。

    返回 {gate, flags, reason}：
      - gate=pass：三检全过（含全部 None/无从校验的宽松项）；flags=[]。
      - gate=warn：任一检命中缺陷；flags=命中项标识列表；reason=人话拼接。
    三检（全是形态/表皮校验，不碰答案对错）：
      ① structure_lint：题型结构 lint（选择题别长成多小问嵌合体/选项不足/标答非字母；填空缺空位）。
      ② _surface_check：题干与母题相似度 > 题型阈值（选择/填空 0.85，解答/证明 0.92）或 数字全同 = 抄题。
      ③ 守恒透传：题型守恒（转题型 from_edit 豁免，由调用方判）+ 考察类型守恒（若声明）。
    🔴 降级：任何异常一律视作三检通过（gate=pass），绝不卡死出题（铁律④ + G5）。
    """
    try:
        flags: list[str] = []
        reasons: list[str] = []

        # ① 结构 lint（题型闭集形态）
        struct_defects = structure_lint(item)
        if struct_defects:
            flags.append("structure")
            reasons.extend(struct_defects)

        # ② 表皮距离（防抄母题）
        surface_defect = _surface_check(item.get("stem"), facts_i.get("stem"), item.get("qtype"))
        if surface_defect:
            flags.append("surface")
            reasons.append(surface_defect)

        # ③ 守恒透传（W2 注入的题型/考察类型守恒的声明性校验）
        #    转题型编辑（from_edit）豁免题型守恒——老师明确点名改造，不算破基因。
        if not item.get("from_edit") and not _qtype_conserved(item, facts_i):
            flags.append("qtype_conservation")
            reasons.append(
                f"题型守恒破：变式题型「{item.get('qtype')}」≠ 母题题型「{facts_i.get('qtype')}」"
            )
        if _exam_type_conserved(item, facts_i) is False:
            flags.append("exam_type_conservation")
            reasons.append(
                f"考察类型守恒破：变式「{item.get('exam_type')}」≠ 母题「{(facts_i.get('dna') or {}).get('exam_type')}」"
            )

        # ④ 🔴 批3·W3' 模型守恒软警（纯代码集合判·零 LLM·**不打回**）：变式 models ⊄
        #    母题 models ∪ {M00} ∪ 反查候选池 → 打 model_conservation flag + reason（仅 ⚠ 透传，
        #    上层 _gene_one_item 据 flag 落待命名池）。变式无 models（继承母题）→ 不报（不误杀）。
        cons = _model_conservation_check(item, facts_i)
        if cons.get("warn"):
            flags.append("model_conservation")
            oos = "、".join(cons.get("out_of_set") or [])
            reasons.append(f"解法模型守恒软警：变式用了母题外的解法模型「{oos}」（仅提示·不打回·待审）")

        if not flags:
            return {"gate": GENE_GATE_PASS, "flags": []}
        return {
            "gate": GENE_GATE_WARN,
            "flags": flags,
            "reason": "；".join(reasons) or None,
            # 🔴 透传越界 id（上层落待命名池用；无越界 → 不带键，不污染干净 item.gene）。
            **({"model_out_of_set": cons["out_of_set"]} if cons.get("warn") else {}),
        }
    except Exception:  # noqa: BLE001 — 闸A是增强不是关卡：三检异常一律降级放行（铁律④/G5）
        return {"gate": GENE_GATE_PASS, "flags": []}


async def gene_gate(state: VariantState, config: RunnableConfig) -> VariantState:
    """闸A 节点：对每道**未判过基因**的变式做纯代码三检（B2·T2，判决语义见 _gene_one_item）。

    P2（PRD-C-012）：循环体抽成 per-item 协程 _gene_one_item，本节点按题 asyncio.gather +
    Semaphore(GATE_CONCURRENCY) 并发，结果按原下标顺序回填（三检纯代码、无 IO，并发只为
    保持与 eager 链调用契约一致；不再有 LLM 调用，无预算消耗回炉）。
    🔴 已带 gene 的题（exec_add 追加时的旧题等）原样通过，不重判。
    """
    budget = _budget_bind(state)  # P13：轮内下游节点携带预算（本节点 T2 后不再花 LLM 预算）
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    total = len(items)
    sem = asyncio.Semaphore(GATE_CONCURRENCY)

    async def _run(i: int, it: dict[str, Any]) -> dict[str, Any]:
        item = dict(it)
        if item.get("gene"):  # 已判过 → 不进并发槽、不发 stage
            return item
        async with sem:
            return await _gene_one_item(item, facts, i, total)

    out = list(await asyncio.gather(*(_run(i, it) for i, it in enumerate(items))))
    _emit_stage("gene_gate", "平行度比对", "done")
    return {"items": out, "llm_call_budget": budget, "messages": []}
