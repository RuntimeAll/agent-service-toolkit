"""variant 引擎 · stage2_variant 变式生成层（PRD-C-104 B3b 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出变式「造题」积木：
  - normalize_knobs / recipe_from_knobs：旋钮钳制 + 配方推导（纯函数）。
  - _parse_generated_items：LLM 文本 → 规整 items。
  - generate：③ 造题节点（含内嵌 _stamp_recipe/_eager_chain/_gen_progress 闭包）。

🔴 行为零改·命脉保全：_eager_chain 经 asyncio.create_task 派发
（PEP567 自动复制 contextvar）→ get_stream_writer 流式不破；items +
merge_items reducer 是 state.py 契约，只读不复制。GENERATE_PROMPT 从同包
stage2_variant.prompts 取；其余依赖从 facade agents.variant 运行期取。
__init__.py 末尾 re-export 4 函数 → 调用方/图 wiring 零感。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.variant.stage2_variant.prompts import (  # noqa: E402
    GENERATE_ONE_PROMPT,
)
from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾导入）
    DEFAULT_SHAPE,
    DIFFICULTY_CAP,
    GATE_CONCURRENCY,
    KNOBS_COUNT_MAX,
    KNOBS_COUNT_MIN,
    PLAN_INCREASING,
    STAGE_DONE,
    TIER_PENDING,
    VARIANT_COEFF_DEFAULT,
    VERIFY_PENDING,
    VariantState,
    _PLAN_INCREASING_WORDS,
    _QTYPE_ALIAS,
    _QTYPE_NOTE_HINTS,
    _ainvoke_text,
    _auto_verify_on,
    _budget_bind,
    _check_one_item,
    _conf_ok,
    _conservation_blocked,
    _conservation_clause,
    _context_block,
    _emit_artifact,
    _emit_stage,
    _extract_items,
    _extract_knobs,
    _facts_log,
    _figure_type_gate_block,
    _gene_one_item,
    _maybe_diversity_block,
    _maybe_note_card_block,
    _normalize_generated_item,
    _parse_json,
    _to_int,
    knobs_desc,
    mother_md_from_table,
    normalize_two_knobs,
    plan_variant_specs,
    settings,
    shape_check,
)


def normalize_knobs(parsed: Any) -> dict[str, Any]:
    """🔴 纯函数钳制（零 LLM/零 IO，可单测）：LLM 抽取产物 → 受约束 knobs dict。

    规则：
    - 非 dict/解析失败 → {}（回落默认配方）。
    - count：宽容转 int，钳到 [1, 8]；非法 → 丢弃。
    - qtype_dist：键过 _QTYPE_ALIAS 归一（应用题/计算题/证明题→解答，且"应用"在 note 保留
      「应用场景」语义）；不认识的题型键丢弃；值须为正整数；同义键合并求和。
    - dist 总和同样钳到 KNOBS_COUNT_MAX（按点名顺序累计到上限封口，截断写进 note 外显）——
      防失控护栏对 dist 路径同等生效，不被「逐项点名」绕过。
    - dist 总和与 count 不一致 → 以 dist 总和为准，且把「数量从 X 调整为 Y」写进 note 外显
      （assemble 头部可见，不静默吞老师的数）。
    - difficulty_plan：命中递增词 → "increasing"；"default"/空 → 丢弃；其余原话保留。
    - note：strip 后非空才保留。
    - 全空 → {}。
    """
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, Any] = {}

    cnt = _to_int(parsed.get("count"))
    if cnt is not None:
        out["count"] = max(KNOBS_COUNT_MIN, min(cnt, KNOBS_COUNT_MAX))

    note_bits: list[str] = []
    raw_dist = parsed.get("qtype_dist")
    if isinstance(raw_dist, dict):
        dist: dict[str, int] = {}
        for k, v in raw_dist.items():
            qt = _QTYPE_ALIAS.get(str(k or "").strip())
            n = _to_int(v)
            if not qt or n is None or n <= 0:
                continue
            dist[qt] = dist.get(qt, 0) + n
            hint = _QTYPE_NOTE_HINTS.get(str(k or "").strip())
            if hint and hint not in note_bits:
                note_bits.append(hint)
        if dist:
            total = sum(dist.values())
            if total > KNOBS_COUNT_MAX:
                # 🔴 防失控护栏（与 count 钳制/ADD_COUNT_MAX 同哲学）：按点名顺序累计到上限封口
                clipped: dict[str, int] = {}
                budget = KNOBS_COUNT_MAX
                for qt, n in dist.items():
                    if budget <= 0:
                        break
                    take = min(n, budget)
                    clipped[qt] = take
                    budget -= take
                note_bits.append(
                    f"题型配比共 {total} 道超单轮上限，已截到 {KNOBS_COUNT_MAX} 道"
                )
                dist = clipped
                total = KNOBS_COUNT_MAX
            # 🔴 A-1/M6：dist 总和不再无条件压 count。三态分流（cnt = 老师显式给的 count）：
            #   ① 单题型隐含1（total==1 且唯一键）且老师没显式给 count
            #      → 不压成1道，回落默认道数，把该题型作约束施加到全部默认道数
            #        （dist 改写成 {该题型: 默认道数}，shape_check 仍逐项相等可过）。
            #   ② 老师显式 count > sum(dist)
            #      → 以 count 为准、dist 当"部分约束"（各题型最小值，缺额其余题型自由）。
            #        标 qtype_partial，shape_check 用 ≥ 判、generate 提示缺额自由。
            #   ③ 其余（count 缺失但 dist 非单题型隐含1 / count==sum(dist)）
            #      → dist 总和为准（保持旧行为）。
            default_count = DEFAULT_SHAPE["normal"] + DEFAULT_SHAPE["hard"]
            single_implicit_one = total == 1 and len(dist) == 1
            if cnt is None and single_implicit_one:
                # 态①：单题型无显式数量 → 不压1道，回落默认并把题型铺满默认道数
                only_qt = next(iter(dist.keys()))
                dist = {only_qt: default_count}
                out["qtype_dist"] = dist
                out["count"] = default_count
            elif cnt is not None and out.get("count", 0) > total:
                # 态②：显式 count 大于配比总和 → count 为准，dist 当部分约束
                out["qtype_dist"] = dist
                out["qtype_partial"] = True
                note_bits.append(
                    f"按 {out['count']} 道出，其中 "
                    + "、".join(f"{k}×{v}" for k, v in dist.items())
                    + "，缺额由其余题型自由补"
                )
                # out['count'] 保持老师显式值（已在 cnt 钳制时写入）
            else:
                # 态③：dist 总和为准（含 count 缺失非单题型隐含1、或 count==sum(dist)）
                out["qtype_dist"] = dist
                if out.get("count") != total:
                    if out.get("count") is not None:
                        # count 与配比总和冲突 → dist 为准，但调整必须外显（不静默吞老师的数）
                        note_bits.append(f"按题型配比把数量从 {out['count']} 调整为 {total}")
                    out["count"] = total  # dist 总和为准（含 count 缺失时补齐）

    plan = str(parsed.get("difficulty_plan") or "").strip()
    if plan and plan.lower() != "default":
        if any(w in plan.lower() for w in _PLAN_INCREASING_WORDS):
            out["difficulty_plan"] = PLAN_INCREASING
        else:
            out["difficulty_plan"] = plan  # 自由难度要求原话保留（generate/闸A 原样注入）

    note = str(parsed.get("note") or "").strip()
    if note:
        note_bits.append(note)
    if note_bits:
        out["note"] = "；".join(note_bits)

    return out


def recipe_from_knobs(knobs: dict[str, Any] | None, mother_difficulty: Any = None) -> dict[str, Any]:
    """🔴 纯函数：knobs → generate 配方（n/n_normal/n_hard + 老师配方段 spec + 递增预期档位）。

    knobs 空 → 与旧默认完全等价：n=3、2 普 1 难、spec=""（行为不变的回归锚点）。
    递增计划：从母题难度起逐题 +1、封顶 DIFFICULTY_CAP。expected_difficulties（复数）只用于
    prompt 文案 + n_hard 推算；闸A/代码闸的同尺判据 = shape_check(mother_difficulty) 现算。
    （R4·F17：item 级 expected_difficulty 单数印记原只写不读、已移除，不再是判据来源。）
    """
    knobs = knobs or {}
    if not knobs:
        n_normal, n_hard = DEFAULT_SHAPE["normal"], DEFAULT_SHAPE["hard"]
        return {
            "n": n_normal + n_hard,
            "n_normal": n_normal,
            "n_hard": n_hard,
            "spec": "",
            "expected_difficulties": None,
        }

    dist = knobs.get("qtype_dist") or {}
    n = knobs.get("count") or (sum(dist.values()) if dist else 0) or (
        DEFAULT_SHAPE["normal"] + DEFAULT_SHAPE["hard"]
    )
    md = _to_int(mother_difficulty) or 3
    # 🔴 WS3·难度轴（AC8）：difficulty_target 给了目标档（1–4）→ 把起步档 md 移到目标档，
    #   md+i 递增逻辑不动（只换 md 输入）。'keep'/缺 → md 保持母题档（旧行为）。
    #   两轴独立：变式系数(operator_band)管"像不像"，难度轴只移 md 管"难不难"。
    dtgt = _to_int(knobs.get("difficulty_target"))
    if dtgt is not None and 1 <= dtgt <= DIFFICULTY_CAP:
        md = dtgt
    plan = knobs.get("difficulty_plan")
    expected: list[int] | None = None

    lines = [f"- 共 {n} 道（必须恰好 {n} 道，不多不少）。"]
    # 🔴 WS3·变式系数轴（AC8）：operator_band 给了相似度带 → 注入算子人话指令（让"像不像"随系数变）。
    op_band = knobs.get("operator_band")
    if isinstance(op_band, dict) and op_band.get("guidance"):
        lines.append(
            f"- 变式幅度（变式系数 {op_band.get('similarity')}，"
            f"{op_band.get('band')}相似度带·算子「{op_band.get('operator')}」）：{op_band['guidance']}。"
        )
    if dist:
        dist_s = "、".join(f"{k}×{v}" for k, v in dist.items())
        if knobs.get("qtype_partial"):
            # 🔴 A-1/M6 态②：部分约束 —— 这些题型是最低道数，缺额其余题型自由补
            lines.append(
                f"- 题型要求（至少）：{dist_s}；其余 {n - sum(dist.values())} 道题型自由（选择/填空/解答均可）。"
            )
        else:
            lines.append(f"- 题型配比：{dist_s}（每道题的 qtype 严格按此配比给）。")
    if plan == PLAN_INCREASING:
        expected = [min(md + i, DIFFICULTY_CAP) for i in range(n)]
        d_s = ",".join(str(d) for d in expected)
        lines.append(
            f"- 难度计划：递增 —— 从母题难度({md})起逐题升一档、封顶 {DIFFICULTY_CAP}；"
            f"各题 difficulty 依次为 {d_s}；难度高于母题的填 level=\"hard\"，否则 \"normal\"。"
        )
    elif plan:
        lines.append(f"- 难度要求（老师原话，best-effort 满足）：{plan}")
    if knobs.get("note"):
        lines.append(f"- 其他要求（best-effort 吸收，撞主考点/年级硬守恒的忽略）：{knobs['note']}")

    spec = "\n\n老师指定配方（🔴 优先于上面的默认配方，必须严格满足）：\n" + "\n".join(lines)
    if expected:
        n_hard = sum(1 for d in expected if d > md)
    else:
        n_hard = 1 if n >= 2 else 0
    return {
        "n": n,
        "n_normal": n - n_hard,
        "n_hard": n_hard,
        "spec": spec,
        "expected_difficulties": expected,
    }


def _parse_generated_items(text: str, facts: dict) -> list[dict[str, Any]]:
    """generate/重试共用：LLM 返回文本 → 规整 items（check 待 solve_explain 填）。

    🔴 PRD-A-023 B4/B8：opus 在「新定义题」术语处写**未转义 ASCII 双引号**（如 `的"$k$ 倍点"`）
       破坏 JSON 字符串 → 裸 `json.loads` 炸 → 0 道。母题入口已有确定性解药
       `_repair_json_quotes`（无副作用、不花钱、纯字符级扫描修引号），此处接进：
       直接 parse 解出空时，先修引号再 parse 一次（参照 _repair_entry_json 用法）。
    """
    data = _parse_json(text)
    extracted = _extract_items(data)
    if extracted:
        return [_normalize_generated_item(it, facts) for it in extracted if isinstance(it, dict)]
    # 直接 parse 为空 → 引号修复后再试一次（确定性，懒导入防循环）
    try:
        from agents.variant_entry import _repair_json_quotes  # 懒导入防循环
        repaired = _repair_json_quotes(text or "")
    except Exception:  # noqa: BLE001 — 修复是兜底不是关卡，失败回落空
        repaired = ""
    if repaired and repaired != (text or ""):
        data2 = _parse_json(repaired)
        extracted2 = _extract_items(data2)
        if extracted2:
            _facts_log.warning(
                "parse_generated_items: 引号修复救活 %d 道（修复前 0 道，根因=未转义引号）",
                len(extracted2),
            )
            return [_normalize_generated_item(it, facts) for it in extracted2 if isinstance(it, dict)]
    # 修复后仍空 → 区分「引号修复后仍空=多半截断」vs「修复前就空」便于后续观测
    if text and text.strip():
        _facts_log.warning(
            "parse_generated_items: 解析为 0 道（引号修复亦未救活，疑似 JSON 截断/半截；text len=%d）",
            len(text),
        )
    return []


def _parse_one_item(text: str, facts: dict) -> dict[str, Any] | None:
    """🔴 B3·per-variant 单题解析：GENERATE_ONE_PROMPT 出**单个 JSON 对象**（不是数组）。

    宽容：① 顶层是单个含 stem 的对象 → 直接规整；② 模型偶发吐 [obj] / {"items":[obj]}
    （没完全照单题契约）→ 复用 _parse_generated_items 取第一道。③ 未转义引号 → 复用其引号修复
    路径。解不出 → None（调用方按单道失败容缺，G5）。
    """
    data = _parse_json(text)
    if isinstance(data, dict) and (data.get("stem") or data.get("answer")):
        norm = _normalize_generated_item(data, facts)
        return norm if norm.get("stem") else None
    # 退路：当数组/包装对象解（含引号修复）→ 取第一道
    items = _parse_generated_items(text, facts)
    return items[0] if items else None


async def generate(state: VariantState, config: RunnableConfig) -> VariantState:
    """③ 造题：配方由首轮旋钮(knobs)驱动，无旋钮走旧默认 3 = 2 普通 + 1 难。

    🔴 入口断言 mother_confirmed 或三锚高置信（防裸奔）。
    🔴 代码级配方校验 shape_check：不符带缺陷反馈整组 retry 1 次，仍不符 → 接受 +
       shape_defects 外显到题组头部（不卡死，G5）。
    """
    if not (state.get("mother_confirmed") or _conf_ok(state.get("analysis") or {})):
        # DNA 闸未过却走到 generate（多入口兜底）→ 拒造，回 clarify 语义
        return {
            "messages": [
                AIMessage(content="母题 DNA 还没确认，我先不造题。请确认年级/考点/题型。")
            ]
        }

    # 🔴 PRD-C-106 B2·隔离接缝：阶段二取 facts 统一走 facts_from_ref —— 优先读 compress 固化的
    #   frozen MotherCoreRef.facts（AC3/G3：阶段二只读参照、不继承阶段一对话；并发不读脏，B0 坑2），
    #   缺参照（旧线程/库内母题旁路）回退 _mother_facts(state)（旧行为，向后兼容）。
    #   🔴 facts_from_ref 在 compress.py（re-export 晚于本模块）→ 体内延迟 import（调用期已就绪）。
    from agents.variant import facts_from_ref  # noqa: E402

    # 🔴 批2·generate 入口防御断言（从机制上绝迹「未解析+未知年级进出题」）：facts 缺年级
    #   或主考点 → 拒绝出题、回确认态。多入口（route_entry 库内母题直进 / patch 重造 / 兜底）
    #   都必过此闸，gate_after_classify 之外的旁路也兜得住。
    _facts_pre = facts_from_ref(state)
    _missing_grade = (str(_facts_pre.get("grade") or "").strip() in ("", "未知年级"))
    _missing_kp = (
        str(_facts_pre.get("kp_name") or "").strip() in ("", "未知考点")
        or not _facts_pre.get("dim1_kp_id")
    )
    if _missing_grade or _missing_kp:
        lack = "、".join(
            x for x, m in (("年级", _missing_grade), ("主考点（知识点未锚定）", _missing_kp)) if m
        )
        _emit_stage("generate", "生成题目", "warn", f"母题未定死（缺{lack}）")
        return {
            "messages": [
                AIMessage(
                    content=f"母题还没定死（缺{lack}），我不能出题。请先确认年级 + 主考点知识点，我再造变式。"
                )
            ]
        }

    # 🔴 C3/A-24（PRD-A-018）：进入出题（generate 起跑 / resume 经 route_entry 直奔 generate）时补发一帧
    #   review=STAGE_DONE，让「确认母题」节点善终（之前停在 review=await 等老师点「开始」；老师点了进 generate
    #   后若不补 done，定题三节点里「确认母题」永远停 await、状态条不收口）。await_mother_review 仍发 await
    #   （variant.py:2495 不动）——那是「等老师确认」的暂停态；本帧是「老师已确认、开始出题」的善终态。
    _emit_stage("review", "确认母题", STAGE_DONE, "母题已确认，开始生成变式")

    # 🔴 P13：出题轮入口重置预算（generate→gene_gate→solve_explain→assemble 同轮共享）。
    budget = _budget_bind(state, reset_limit=settings.VARIANT_BUDGET_GENERATE)

    # 🔴 旋钮：新母题轮 analyze 已抽好随 state 来；库内母题直进 generate（不经 analyze）→ 此处兜底抽。
    #   🔴 改动2·阶段灯归属：图片首轮的「解析配方」灯由 classify 真实完成时发 done（带道数）——
    #   此处仅当 knobs 为 None（= 库内母题路径，跳过 analyze/classify）时兜底抽 + 补发该灯，
    #   不在图片路径重复发（防覆盖 classify 已发的终态）。
    knobs = state.get("knobs")
    if knobs is None:
        knobs = await _extract_knobs(state)
        _emit_stage(
            "knobs", "解析配方", "done",
            knobs_desc(knobs) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）",
        )

    # 🔴 PRD-C-103 WS3·双旋钮（AC8）：FE 经 agent_config 透传 variant_similarity / difficulty_target
    #   → config.configurable → 这里并进 knobs（与 LLM 抽的 count/qtype 正交，不冲突）。
    #   缺/keep → 不设键（回落默认：相似度 0.7 / 难度=母题档 md+i）。纯函数钳制，零 LLM。
    two = normalize_two_knobs((config or {}).get("configurable") or {})
    if two:
        knobs = {**(knobs or {}), **two}

    facts = facts_from_ref(state)  # 🔴 B2：同上——阶段二吃固化参照，不重算（隔离 + 不读脏）
    # 🔴 W2 守恒守门（T1）：母题 DNA 白名单为空集 → 不放行生成，降级回 clarify 语义（不裸出）。
    blocked = _conservation_blocked(facts.get("dna"))
    if blocked:
        _emit_stage("generate", "生成题目", "warn", "母题知识点未锚定")
        return {
            "messages": [
                AIMessage(
                    content=f"{blocked}。请补充确认这道母题考的知识点（年级 + 考点），我再造变式。"
                )
            ]
        }
    # 🔴 PRD-C-103 WS1·AC2：母题起步档 md = 锚定模型查表 经 grade_observed 确定算
    #   （替代 mother_opus dim8 的 LLM 自评 difficulty）。md+i 递增逻辑（recipe_from_knobs L3695）
    #   保留不动，只换 md 来源。降级：无 DNA（库内母题旧线程）→ 回退原 dim8/difficulty 值，不卡死。
    mother_d = mother_md_from_table(state)
    if mother_d is None:
        mother_d = (state.get("mother_dna") or {}).get("difficulty") \
            or (state.get("mother_dna") or {}).get("dna", {}).get("difficulty")
    recipe = recipe_from_knobs(knobs, mother_d)
    total_n = int(recipe["n"])
    _emit_stage("generate", "生成题目", "running", f"{total_n} 道")

    # ── 🔴 PRD-C-106 B3·共享前缀（每道子上下文共用的母题约束块，组一次） ──
    #   facts/守恒/上下文/难题注卡/多样性/图型闸——这些是「母题硬约束」组级共享，不随 per-variant
    #   knob 变。per-variant 子 prompt = GENERATE_ONE_PROMPT(本道派工) + 本共享前缀。
    _shared_suffix = (
        "\n\n"
        + _context_block(facts)  # 整改1：确定上下文硬约束（考点/进度/教材版本）
        + "\n\n"
        + _conservation_clause(facts.get("dna"))  # W2 守恒硬约束（T1）
        + _maybe_note_card_block(facts)  # 批3·W2' 难题注卡
        + _maybe_diversity_block(facts.get("dna"), total_n)  # R3a 多样性重组
        + _figure_type_gate_block(state, facts)  # R3b 章节×图型定型闸
    )

    # 🔴 母题原图（多模态）：每道子任务都附母题图写 figure_spec（向后兼容缺图退纯文本）。
    _mother_img_url = (facts.get("image_url") or "").strip() if facts.get("image_url") else ""

    # ── 🔴 子件1·PLAN 派工：为每道分配 变式系数(基准+带内浮动) + 轮换算子 + 难度(md+i 递增) ──
    _base_coeff = (knobs or {}).get("variant_coeff")
    if _base_coeff is None:
        _base_coeff = VARIANT_COEFF_DEFAULT
    specs = plan_variant_specs(
        total_n, _base_coeff, recipe.get("expected_difficulties"), mother_d
    )

    _auto_verify = _auto_verify_on(config)
    sem = asyncio.Semaphore(GATE_CONCURRENCY)  # 🔴 节点内 gather 限流=3（红线：不用 Send）
    results: dict[int, dict[str, Any]] = {}  # seq(1-based) → 已过闸的题（剔除题带 _dropped）
    spawned: dict[int, dict[str, Any]] = {}  # seq → 已生成未过闸的题（partial 上卡）

    def _emit_fanout_frame() -> None:
        """🔴 子件3·emit 重写：fan-out 后「数单流 stem 次数」失效 → 改「每道按 seq 归属」。
        按 seq 拼累计帧：已过闸用 results[seq]，未过闸已生成用 spawned[seq]（tier=None 先上卡）。
        seq = 道序（PLAN 派工固定），整生命周期不变 → FE 按 _seq 原位 merge 不串台、不嫁接。
        剔除题（_dropped）显式入帧（退场哨兵）。任何异常静默吞（帧是增强不是关卡）。"""
        shown: list[dict[str, Any]] = []
        for seq in sorted(spawned):
            cell = dict(results.get(seq, spawned[seq]))
            cell["_seq"] = seq  # 稳定 merge 键 = 道序（1-based）
            shown.append(cell)
        _emit_artifact(
            dict(state, items=shown, knobs=knobs), partial=True, expected_total=total_n
        )

    def _build_one_prompt(spec: dict[str, Any]) -> str:
        seq = spec["seq"]
        diff = spec.get("difficulty")
        if isinstance(diff, int):
            lvl = "hard" if (mother_d is not None and diff > int(mother_d)) else "normal"
            difficulty_line = (
                f"- 目标难度档：{diff}（{lvl}）；难度高于母题填 level=\"hard\"，否则 \"normal\"。"
            )
        else:
            difficulty_line = "- 难度：守母题难度（普通题）。"
        head = GENERATE_ONE_PROMPT.format(
            seq=seq,
            total=total_n,
            coeff=spec["coeff"],
            operator=spec["operator"],
            op_guidance=spec["guidance"],
            difficulty_line=difficulty_line,
            **facts,
        )
        return head + _shared_suffix

    async def _gen_one(spec: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        """🔴 子件2·per-variant 独立子上下文：吃同一份 frozen facts + 本道 spec，
        各自一次 LLM 出**一道**变式 → 闸链（闸A，自动模式再闸B）→ 返回 (seq, kept|None)。
        任何异常静默吞（G5：单道失败不炸整组，返 None 由组级容缺）。"""
        seq = spec["seq"]
        idx0 = seq - 1
        try:
            prompt = _build_one_prompt(spec)
            if _mother_img_url:
                human = HumanMessage(content=[
                    {"type": "text", "text": prompt
                        + "\n\n🔴 上方附了**母题原配图**。写 figure_spec 时对照这张母题图判断"
                          "几何构型与标注，别凭题干文字脑补。"},
                    {"type": "image_url", "image_url": {"url": _mother_img_url}},
                ])
            else:
                human = HumanMessage(content=prompt)
            async with sem:
                _emit_stage(
                    "generate", "生成题目", "running",
                    f"正在写第 {seq}/{total_n} 道（变式系数 {spec['coeff']}·{spec['operator']}）",
                )
                text = await _ainvoke_text(
                    [human],
                    model=settings.variant_model("generate"),
                    timeout=settings.VARIANT_TIMEOUT_GENERATE,
                )
                parsed = _parse_one_item(text, facts)
                if not parsed:
                    _facts_log.warning("generate fan-out: 第 %d 道解析为 0 道（疑似截断/空返）", seq)
                    return seq, None
                item = parsed
                item["_seq"] = seq  # 稳定道序（PLAN 派工，整生命周期不变）
                # 派工印记（trace/外显用；与策略定稿一致，不影响判决）
                item["variant_coeff"] = spec["coeff"]
                item["variant_operator"] = spec["operator"]
                spawned[seq] = dict(item)  # 先上卡（无 tier）
                _emit_fanout_frame()
                # 闸链：闸A 纯代码三检；自动模式再过闸B（sympy 验算/回炉），手动模式挂 pending
                judged = await _gene_one_item(item, facts, idx0, total_n)
                if _auto_verify:
                    kept, note = await _check_one_item(judged, facts, idx0, total_n)
                    if kept is None:
                        kept = dict(judged)
                        kept["_dropped"] = note or "程序验出标答错误（重生未过），已剔除"
                else:
                    kept = dict(judged)
                    if not kept.get("check"):
                        kept["check"] = {
                            "badge": "ok", "solved_answer": None,
                            "verify": VERIFY_PENDING, "tier": TIER_PENDING,
                        }
                kept["_seq"] = seq
            results[seq] = kept
            done_n = len([s for s in results if not results[s].get("_dropped")])
            _detail = (
                f"第 {done_n}/{total_n} 道完成" if _auto_verify
                else f"第 {done_n}/{total_n} 道就绪（待手动验算）"
            )
            _emit_stage("verify", "程序验算", "running", _detail)
            _emit_fanout_frame()
            return seq, kept
        except (TimeoutError, asyncio.TimeoutError):
            _emit_stage("generate", "生成题目", "warn", f"第 {seq} 道生成超时，跳过")
            return seq, None
        except Exception:  # noqa: BLE001 — 单道失败不炸整组（G5），由组级容缺
            _facts_log.warning("generate fan-out: 第 %d 道异常，跳过", seq, exc_info=True)
            return seq, None

    # 🔴 子件0 红线：节点内 asyncio.gather(Semaphore=3) fan-out，绝不用 LangGraph Send。
    gathered = await asyncio.gather(*(_gen_one(s) for s in specs), return_exceptions=True)

    # 按道序(seq)装配 → 一次性 return 全组（保 merge_items reducer「new=权威全集」语义）
    items: list[dict[str, Any]] = []
    for r in gathered:
        if isinstance(r, BaseException):
            continue
        seq, kept = r
        if kept is not None:
            items.append(kept)
    items.sort(key=lambda it: it.get("_seq") or 0)

    if not items:
        # 全道失败/解析失败 → 友好失败收尾（after_generate 走 done→END），绝不静默空轮
        _emit_stage("generate", "生成题目", "warn", "0 道（解析失败）")
        return {
            "items": [],
            "knobs": knobs,
            "shape_defects": [],
            "llm_call_budget": budget,
            "messages": [
                AIMessage(
                    content="这一轮我没能产出可用的变式题（模型输出解析失败）。"
                    "请再发一次指令（可换种说法），或重贴题目图重试。"
                )
            ],
        }

    # 🔴 配方校验（数量/题型分布/递增档位）：fan-out 已按 PLAN 逐道派系数/算子/难度，整组 retry
    #   不再适用（每道独立子上下文，无「整组重出一个 prompt」概念）；缺额/数量不符只外显缺陷，
    #   由 assemble 头部 ⚠ 呈现（不卡死，G5）。质量阈值由维护者自测（AC7 口径）。
    defects = shape_check(items, knobs, mother_d)

    _emit_stage("generate", "生成题目", "done", f"{len(items)} 道")
    return {
        "items": items,
        "knobs": knobs,
        "shape_defects": defects,
        # 🔴 新一组题 → 复位手排标记：上一组的 manual_order 绝不泄漏到新母题/新出题轮。
        "manual_order": False,
        "llm_call_budget": budget,
        "messages": [],
    }
