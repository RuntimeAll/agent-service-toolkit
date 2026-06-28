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
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.variant.stage2_variant.prompts import GENERATE_PROMPT  # noqa: E402
from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾导入）
    DEFAULT_SHAPE,
    DIFFICULTY_CAP,
    GATE_CONCURRENCY,
    KNOBS_COUNT_MAX,
    KNOBS_COUNT_MIN,
    PLAN_INCREASING,
    STAGE_DONE,
    TIER_PENDING,
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
    _iter_complete_items,
    _maybe_diversity_block,
    _maybe_note_card_block,
    _mother_facts,
    _norm,
    _normalize_generated_item,
    _parse_json,
    _to_int,
    knobs_desc,
    mother_md_from_table,
    normalize_two_knobs,
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
    _emit_stage("generate", "生成题目", "running", f"{recipe['n']} 道")
    prompt = (
        GENERATE_PROMPT.format(
            n=recipe["n"], n_normal=recipe["n_normal"], n_hard=recipe["n_hard"], **facts
        )
        + "\n\n"
        + _context_block(facts)  # 🔴 整改1：确定上下文硬约束（考点/进度/教材版本，压解题不越界）
        + "\n\n"
        + _conservation_clause(facts.get("dna"))  # 🔴 W2 守恒硬约束注入（T1，与上块正交并存）
        + _maybe_note_card_block(facts)  # 🔴 批3·W2' 难题注卡（难度≥3+非M00 才注，含反退化约束）
        # 🔴 B3·R3a 多样性重组（PRD-A-021）：仅 GENERATE 注入；副考点/标签 per-variant 子集差异，
        #   场景保持组级，骨架四维仍锁。n<2 → 不注（_maybe_diversity_block 内判）。
        + _maybe_diversity_block(facts.get("dna"), recipe["n"])
        + recipe["spec"]
        # 🔴 PRD-A-021 R3b·章节×图型定型闸（落点①）：约束段拼最末（护 aigeek 前缀缓存）。
        #   无章节/无映射/表读不到 → 返回 ""（逃生，不约束）。
        + _figure_type_gate_block(state, facts)
    )

    # 🔴 PRD-A-018 round4·治本P0「出题写 figure_spec 时手里有图」：把母题原图作为多模态 image 一并
    #   喂给 generate（仿 analyze 的 ANALYZE_PROMPT 多模态写法 variant.py:1756-1761）。出题节点此刻
    #   **真看着母题图**写 figure_spec（含结构化标注决策），不再凭母题题干文字+骨架脑补几何构型。
    #   relay/opus 支持 vision（analyze 已用同套 list-content HumanMessage）。
    #   🔴 向后兼容：image_url 可能缺（无图母题/纯代数母题/旧线程）→ 退回纯文本 HumanMessage，不崩。
    #   on_delta 流内进度回调对 list-content / str-content 一视同仁（数 acc 里 "stem" 次数），不受影响。
    _mother_img_url = (facts.get("image_url") or "").strip() if facts.get("image_url") else ""
    if _mother_img_url:
        _gen_human = HumanMessage(
            content=[
                {"type": "text", "text": prompt
                    + "\n\n🔴 上方附了**母题原配图**。写每道变式的 figure_spec 时**对照这张母题图**"
                      "判断几何构型与标注（哪些点/角/线、标哪些已知角=什么文字），别再凭题干文字脑补。"},
                {"type": "image_url", "image_url": {"url": _mother_img_url}},
            ]
        )
    else:
        _gen_human = HumanMessage(content=prompt)

    # 🔴 思维外放（用户反馈 2026-06-11）：JSON token 对用户是乱码不外放，但流内数
    # "stem" 出现次数 → 思路条实时跳「正在写第 n/N 道」+ 当前题干前几个字，等待不再是黑盒。
    total_n = int(recipe["n"])
    # 🔴 PRD-A-021 R4·F17：md_i / increasing 原仅供已退役的 from_recipe/expected_difficulty
    #   印记计算，随死印记一并移除（无其它消费方）。

    def _stamp_recipe(item: dict[str, Any], idx: int) -> dict[str, Any]:
        # 🔴 PRD-A-021 R4·F17：from_recipe / expected_difficulty 为只写不读的死印记
        #   （全仓无任何读方，仅在各处 strip 元组里被 pop 掉）→ 已停写。本 shim 现为 identity，
        #   保留以不扰动两处调用点（4186 仍承担 _normalize 包装、4308 的幂等回填语义）。
        #   注意：recipe 级 `expected_difficulties`（复数·喂 prompt）是另一个 LIVE 键，未动。
        return item

    # ── P2 流内 eager（PRD-C-012）：增量发现完整新题 → 立即 spawn 闸链 task ──
    # 调用成本不增：闸A/闸B 仍是同一批 per-item helper，只是从「流结束后串行」改成
    # 「题一完整就并发跑」（同一 Semaphore(GATE_CONCURRENCY) 限流）。宏观 DAG 零改动：
    # 产物已带 gene+check → 下游 gene_gate/solve_explain 节点天然跳过已判项。
    # 🔴 BUG-09（2026-06-19）：手动验算（auto_verify=False）下流内 eager **只过闸A（基因/平行度）
    #   不过闸B（sympy 验算/回炉）** —— 闸B 留给老师手动按需点。闸A 是纯代码三检（零 LLM、秒级），
    #   保留它让平行度徽章照常上卡；不剔题（pending 题进 solve_explain 挂 pending、绝不被剔）。
    _auto_verify = _auto_verify_on(config)
    sem = asyncio.Semaphore(GATE_CONCURRENCY)
    eager_raw: list[dict[str, Any]] = []  # 流内稳收的规整题（生成序；merge/兜底用）
    eager_tasks: list[asyncio.Task] = []
    spawned: dict[int, dict[str, Any]] = {}  # idx → 刚解析出的题（无 check/无 tier，partial 上卡）
    completed: dict[int, dict[str, Any]] = {}  # idx → 已过闸链的题（剔除题带 _dropped 哨兵）
    # 🔴 stale 标志（对抗审修复）：流重启（中转站中途熔断换站从头重流 / 空返回 retry 二次
    # 开流）或 shape 整组 retry 采纳后，已派发的 eager 闸链全部作废 —— 置位后：①不再派发
    # 新 task；②在跑的 task 不再记账/发帧（半发帧误导 FE）；③merge 不消费（use_eager=False）。
    stale = {"flag": False}

    def _cancel_eager() -> None:
        stale["flag"] = True
        for t in eager_tasks:
            t.cancel()

    def _emit_eager_frame() -> None:
        """🔴 P2b-BE 逐题上屏（PRD-C-013）：把「出卡」与「过闸」解耦。按生成序拼累计帧——
        已过闸的题用 completed[idx]（带 tier 定状态），未过闸但已解析的题用 spawned[idx]
        （无 check → _artifact_payload 自然 tier=None，FE 按「无tier帧」先上卡）。

        🔴 稳定 seq 不重压缩（对抗审①修复）：每个 cell 带 `_seq = 题原始生成序 k+1`，整生命周期
        不变（剔除题不让后题 index 前移 → FE 按 seq 原位 merge 不嫁接到别的卡）。剔除题
        （completed[k]._dropped）**显式入帧带 `_dropped:true`**（而非 `continue` 跳过）——
        让 FE 收到显式退场哨兵驱动退场过渡 + 从 mergedItems 移除，而不是靠压缩 index 隐式挤掉
        （后者会残留永删不掉的重复卡）。FE 契约：先收该题无 tier 帧 → 带 tier 帧 →（若被判废）_dropped 帧。"""
        shown: list[dict[str, Any]] = []
        for k in sorted(spawned):
            cell = dict(completed.get(k, spawned[k]))
            cell["_seq"] = k + 1  # 稳定 merge 键：题原始生成序（1-based），剔除题不重压缩
            shown.append(cell)  # 剔除题（_dropped）照样入帧 → _artifact_payload 透传哨兵
        _emit_artifact(
            dict(state, items=shown, knobs=knobs), partial=True, expected_total=total_n
        )

    async def _eager_chain(idx: int, item: dict[str, Any]) -> None:
        """单题闸链（闸A → 闸B）+ 完成即**原位重发**该题帧（同 seq，带 tier）。任何异常静默吞
        （G5：eager 是增强不是关卡，失败的题留给下游节点照旧串行补判）。"""
        try:
            async with sem:
                # B2·T2：闸A 内涵换纯代码三检（facts 直传，judge 配方对齐 facts 组装已退役）。
                judged = await _gene_one_item(item, facts, idx, total_n)
                # 🔴 BUG-09：手动模式只过闸A，闸B 留给老师手动点 → 挂 pending（不剔题、不回炉）。
                if _auto_verify:
                    kept, note = await _check_one_item(judged, facts, idx, total_n)
                else:
                    kept, note = dict(judged), None
                    if not kept.get("check"):
                        kept["check"] = {
                            "badge": "ok", "solved_answer": None,
                            "verify": VERIFY_PENDING, "tier": TIER_PENDING,
                        }
            if stale["flag"]:
                return  # 本轮 eager 已作废（流重启/整组 retry）→ 不记账、不发作废题的帧
            if kept is None:
                # 4d 方案A：剔除题转哨兵（带 gene 不带 check）→ 下游 solve_explain 收口
                # 进 dropped_notes；不出现在增量帧（本帧起退场）。
                kept = dict(judged)
                kept["_dropped"] = note or "1 道题程序验出标答错误（重生一次仍未过），已剔除"
            completed[idx] = kept
            # 思路条进度（已过闸计数）+ 原位重发该题帧（同 seq，现带 tier 上定状态）
            done_n = len([k for k in completed if not completed[k].get("_dropped")])
            _stage_detail = (
                f"第 {done_n}/{total_n} 道完成" if _auto_verify
                else f"第 {done_n}/{total_n} 道就绪（待手动验算）"
            )
            _emit_stage("verify", "程序验算", "running", _stage_detail)
            _emit_eager_frame()
        except Exception:  # noqa: BLE001 — eager 失败绝不炸 generate；该题留给下游节点补判
            pass

    _seen = {"n": 0}
    _acc_len = {"v": 0}

    def _gen_progress(acc: str) -> None:
        # 🔴 流重启检测（对抗审修复）：中转站首站吐若干 chunk 后熔断换下一站从头重流 /
        # _ainvoke_text 空返回 retry 二次开流时，acc 从头重积（len 回落）。此时 eager_raw
        # 里是死流的题、与新流（最终 text 的事实源）下标天然错位 —— 本轮 eager 全作废
        # （cancel + 静默 + use_eager=False），题目交还下游 gene_gate/solve_explain 节点
        # 照旧补判（宏观 DAG 不变，只损失 eager 增强）。
        if len(acc) < _acc_len["v"] and not stale["flag"]:
            _cancel_eager()
        _acc_len["v"] = len(acc)
        n = min(acc.count('"stem"'), total_n)
        if n > _seen["n"]:
            _seen["n"] = n
            m = re.findall(r'"stem"\s*:\s*"([^"]{0,24})', acc)
            peek = (m[-1].replace("\\n", " ").strip() + "…") if m and m[-1] else ""
            _emit_stage("generate", "生成题目", "running", f"正在写第 {n}/{total_n} 道 {peek}")
        if stale["flag"]:
            return  # eager 已作废 → 只保留进度叙事，不再派发新闸链
        # P2：增量解析已完整闭合的新题（半截题绝不派发）→ 立即起闸链 task
        # 🔴 P2b-BE 逐题上屏：一解析出完整题（stem/answer/solution 齐）就立即上卡——记 spawned
        #   并发**无 tier 的 partial 帧**（出卡与过闸解耦）；闸链 task 跑完再原位重发带 tier 帧。
        try:
            found = _iter_complete_items(acc)
            while len(eager_raw) < len(found):
                idx = len(eager_raw)
                item = _stamp_recipe(_normalize_generated_item(found[idx], facts), idx)
                eager_raw.append(item)
                spawned[idx] = dict(item)  # 无 check → 帧里该题 tier=None（上卡先行）
                _emit_eager_frame()
                eager_tasks.append(
                    asyncio.get_running_loop().create_task(_eager_chain(idx, dict(item)))
                )
        except Exception:  # noqa: BLE001 — 进度/派发是增强不是关卡（G5），绝不打断流
            pass

    try:
        text = await _ainvoke_text(
            [_gen_human],
            on_delta=_gen_progress,
            model=settings.variant_model("generate"),
            # 🔴 PRD-C-100 B3-perf：套总时长墙钟闸（默认 180s，与 §11「opus 带图 ≤180s」对齐）。
            #   此前不传 → 各站默认 150s × 多站熔断转移 × 空返重试可叠加拖到 ~11min（长尾根因）。
            #   超时由 relay_pool 的 asyncio.timeout 抛 TimeoutError，下面分两路有界收口。
            timeout=settings.VARIANT_TIMEOUT_GENERATE,
        )
    except (TimeoutError, asyncio.TimeoutError):
        # 🔴 B3-perf 有界降级（绝不拖到 11min / 绝不无界）：generate 超时——
        #   ① 已流内稳收完整题（eager_raw 非空）→ 用它有界收尾，「先出的先出」(D14)，
        #      不让长尾埋掉已产出的变式正文；
        #   ② 一道都没出 → 标可读「超时降级」文案（建议简化/重试），不裸 error 不卡死。
        _cancel_eager()
        if eager_tasks:
            await asyncio.gather(*eager_tasks, return_exceptions=True)
        if eager_raw:
            text = ""  # 走下方 len(items) < len(eager_raw) 兜底 → items=eager_raw
            _emit_stage("generate", "生成题目", "warn",
                        f"生成超时（>{int(settings.VARIANT_TIMEOUT_GENERATE)}s），已收已出的 {len(eager_raw)} 道")
        else:
            _emit_stage("generate", "生成题目", "warn",
                        f"生成超时（>{int(settings.VARIANT_TIMEOUT_GENERATE)}s）")
            return {
                "items": [],
                "knobs": knobs,
                "llm_call_budget": budget,
                "messages": [
                    AIMessage(
                        content="这道题过于复杂，变式生成超时了，建议把母题拆简单些或换一道再试。"
                    )
                ],
            }
    except BaseException:
        # 🔴 孤儿收口（对抗审修复）：全中转站熔断耗尽等按设计外抛时，已 spawn 的 eager
        # task 必须 cancel + 等待退场——否则每条链最多还有 4-5 次 LLM 调用在后台静默烧完，
        # 并继续向已 error 收尾的流发帧。收口后原样 re-raise，不改失败语义。
        _cancel_eager()
        if eager_tasks:
            await asyncio.gather(*eager_tasks, return_exceptions=True)
        raise
    items = _parse_generated_items(text, facts)
    if len(items) < len(eager_raw):
        # 整体解析比流内增量还少（尾部 JSON 破损等）→ 用流内稳收的题兜底
        items = [dict(it) for it in eager_raw]

    # 🔴 代码级配方校验（数量/题型分布/递增档位）：不符 → 带缺陷反馈整组 retry 1 次。
    #   含首稿解析为空（items=[] 时「要求N道实出0道」也是明确缺陷，值得一次重试）。
    #   🔴 shape 冲突处理（PRD-C-012）：数量类缺陷只能等流结束才判；整组 retry 产出
    #   新 items 时丢弃 eager 结果（接受罕见浪费），retry 题不做流内 eager —— 由下游
    #   gene_gate/solve_explain 节点（同一批 helper + gather 并发）一次性过闸。
    defects = shape_check(items, knobs, mother_d)
    retried = False
    if defects:
        feedback = (
            "\n\n[配方校验反馈] 你上一稿不满足老师指定配方："
            + "；".join(defects)
            + "。请整组重出，严格满足配方（数量/题型配比/难度计划逐项核对后再输出）。"
        )
        try:
            # 🔴 round4：整组 retry 同样喂母题图（出题二稿仍需看图写 figure_spec）；缺图退纯文本。
            if _mother_img_url:
                _retry_human = HumanMessage(content=[
                    {"type": "text", "text": prompt + feedback},
                    {"type": "image_url", "image_url": {"url": _mother_img_url}},
                ])
            else:
                _retry_human = HumanMessage(content=prompt + feedback)
            retry_text = await _ainvoke_text(
                [_retry_human],
                model=settings.variant_model("generate"),
                # 🔴 B3-perf：整组 retry 也套同一墙钟闸（这是第二次全量出题，不套则又是一条长尾）。
                timeout=settings.VARIANT_TIMEOUT_GENERATE,
            )
            retry_items = _parse_generated_items(retry_text, facts)
        except Exception:  # noqa: BLE001 — 重试失败/超时保留首稿（绝不卡死）
            retry_items = []
        if retry_items:
            items = retry_items
            defects = shape_check(items, knobs, mother_d)
            retried = True
            # 🔴 整组 retry 采纳 = 首稿题全部作废（对抗审修复）：cancel 仍在跑的 eager
            # 闸链（每条链最多还有 4-5 次 LLM 调用，gather 白等可达数十秒）+ 静默其
            # 后续发帧（作废题的 partial 帧只会误导 FE）。retry 失败回退首稿的分支
            # （retried=False）保持现状不取消——首稿仍是最终产物。
            _cancel_eager()

    # eager task 收口（绝不留孤儿任务；retry 整组重出/流重启时结果弃用）
    if eager_tasks:
        await asyncio.gather(*eager_tasks, return_exceptions=True)
    use_eager = bool(eager_tasks) and not retried and not stale["flag"]

    if not items:
        # 两稿皆空/解析失败 → 友好失败收尾（after_generate 走 done→END），绝不静默空轮
        _emit_stage("generate", "生成题目", "warn", "0 道（解析失败）")
        return {
            "items": [],
            "knobs": knobs,
            "shape_defects": defects,
            "llm_call_budget": budget,
            "messages": [
                AIMessage(
                    content="这一轮我没能产出可用的变式题（模型输出解析失败）。"
                    "请再发一次指令（可换种说法），或重贴题目图重试。"
                )
            ],
        }

    # 配方印记（_stamp_recipe 同一公式；eager 已在派发时落印，此处对 retry/未派发尾项
    # 落印 + 对已派发项幂等重写同值）
    if knobs:
        for i, it in enumerate(items):
            _stamp_recipe(it, i)

    if use_eager:
        # 已过闸链的题按生成序回填 + 🔴 同一性校验（对抗审修复）：completed[i] 派生自
        # eager_raw[i]（heal/补题换 stem 不影响——比的是派发时的原稿 stem），仅当它与
        # 全量解析第 i 道是同一道题（stem 一致）才采用。错位场景（模型偶发吐无 stem 的
        # 杂物对象：全量解析保留为 stem=None 而流内增量跳过，两套下标错一位）→ 保留
        # 原稿走下游节点补判（宏观 DAG 不变），绝不把验算徽章嫁接到另一道题上。
        def _same_item(i: int, it: dict[str, Any]) -> bool:
            if i >= len(eager_raw):
                return False
            key = _norm(eager_raw[i].get("stem"))
            return bool(key) and key == _norm(it.get("stem"))

        items = [
            completed[i] if (i in completed and _same_item(i, it)) else it
            for i, it in enumerate(items)
        ]

    _emit_stage("generate", "生成题目", "done", f"{len(items)} 道")
    return {
        "items": items,
        "knobs": knobs,
        "shape_defects": defects,
        # 🔴 新一组题 → 复位手排标记（对抗审③）：上一组的 manual_order 绝不泄漏到新母题/新出题轮。
        "manual_order": False,
        "llm_call_budget": budget,
        "messages": [],
    }
