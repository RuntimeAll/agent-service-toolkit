"""variant 引擎 · stage1_anchor 压缩闸（PRD-C-106 B2 新增节点）。

阶段一 → 阶段二的「压缩闸」：把阶段一对话/锚定产物**打包成一个不可变的母题核心参照
（MotherCoreRef）**塞进 state.mother_core_ref，阶段二（generate + per-variant fan-out）
**只读这份固化参照、不再各自 `_mother_facts(state)` 重算**（B0 坑2：并发期重算读脏）。

🔴 设计依据（PRD-C-106 §10 契约面板 + 编排图 STOP1 后/阶段二前）：
  - MotherCoreRef.dna：结构化 DNA（复用 `_mother_facts` 产出，不新造 schema）。
  - MotherCoreRef.summary：一句人话摘要（B2 唯一新增产物，纯模板拼，失败降级仍出）。
  - MotherCoreRef.stem/answer/solution：母题原文（变式生成需引用）。
  - MotherCoreRef.figure：母题配图信息（若有）。
  - 🔴 异常：阶段一失败/低置信 → incomplete=true（沿用 C-105 D 弹窗那条路，不产残缺参照硬塞）。

🔴 隔离（AC3/G3）：B0 已确认 generate 只吃 facts 不吃 messages → 隔离结构天然成立。本节点
   把「facts 的来源」从「每次 _mother_facts(state) 重算」收敛成「一次固化进 mother_core_ref」，
   为 B3 fan-out 子任务吃同一份不可变参照铺路。compress 自身**绝不读 state.messages**。

🔴 不引 interrupt：compress 是普通节点，放在 STOP1 resume（route_entry → compress）之后、
   generate 之前；→END+config 回传 resume 模型不动（C-104 frozen 行为）。

🔴 strangler：顶部 from agents.variant import 取依赖（运行期解析）。compress re-export 须置于
   mother_card / generate re-export 之后（本模块体内不依赖它们的装载期符号，仅 _mother_facts /
   _build_mother_card / mother_md_from_table，皆已先 re-export）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、mother_card/generate 之后导入）
    STAGE_DONE,
    VariantState,
    _build_mother_card,
    _emit_stage,
    _mother_facts,
    mother_md_from_table,
)

# MotherCoreRef schema 版本（契约演进留痕，下游可据此兼容）。
MOTHER_CORE_REF_VERSION = 1


def _ref_difficulty(state: VariantState, dna: dict[str, Any]) -> int | None:
    """母题难度走表驱动 grade_observed（与母题卡同源，铁律：不采信 LLM 自评）。

    降级（与 mother_card 同口径）：表驱动拿不到（库内母题无 DNA / 异常）→ 回退 dna.difficulty
    （opus 自评）→ mother_dna.difficulty；全无 → None。
    """
    difficulty: Any = None
    try:
        difficulty = mother_md_from_table(state)
    except Exception:  # noqa: BLE001 — 判档异常绝不炸压缩闸
        difficulty = None
    if isinstance(difficulty, int):
        return difficulty
    d = dna.get("difficulty")
    if isinstance(d, int):
        return d
    mdna = state.get("mother_dna") or {}
    md = mdna.get("difficulty")
    return md if isinstance(md, int) else None


def _build_summary(facts: dict[str, Any], difficulty: int | None, no_model: bool) -> str:
    """一句人话摘要（B2 唯一新增产物）·纯模板拼（零 LLM、零 IO、绝不抛）。

    形如「这是一道 <年级> 考「<考点>」、用「<模型>」、<难度档> 的<题型>题」。
    🔴 摘要**不承载机器关键字段**（kp/models/grade/difficulty 走结构化 dna，G3 反性自检），
       摘要只给老师/日志看，缺字段就省略对应短语，绝不留空白/报错。
    """
    dna = facts.get("dna") or {}
    grade = str(facts.get("grade") or "").strip()
    kp = str(facts.get("kp_name") or "").strip()
    qtype = str(facts.get("qtype") or "").strip() or "解答"

    parts: list[str] = ["这是一道"]
    if grade and grade != "未知年级":
        parts.append(grade)
    if kp and kp != "未知考点":
        parts.append(f"考「{kp}」")
    # 模型短语：诚实三态（有→点名套路；无→明说「无考模型」）。
    if no_model:
        parts.append("（无考模型）")
    else:
        names = [
            str(m.get("name") or "").strip()
            for m in (dna.get("models") or [])
            if isinstance(m, dict) and str(m.get("name") or "").strip()
        ]
        if names:
            parts.append("用「" + "、".join(names[:3]) + "」")
    if isinstance(difficulty, int):
        tier_word = {1: "送分", 2: "巩固", 3: "中档", 4: "压轴"}.get(difficulty, f"{difficulty} 档")
        parts.append(tier_word)
    parts.append(f"的{qtype}题")
    return "".join(parts)


def build_mother_core_ref(state: VariantState) -> dict[str, Any]:
    """🔴 PRD-C-106 §10：把阶段一产物打包成 frozen MotherCoreRef（纯函数·零 IO，可单测）。

    数据源 = `_mother_facts(state)`（结构化 DNA + 题面/答案/解析 + 锚定，复用不新造）
            + 母题难度表驱动 grade_observed + 一句人话摘要。

    🔴 异常路径：mother_dna/dna 缺失或无法定死（无年级/无主考点）→ incomplete=true，
       下游据此走 C-105 D 弹窗，不拿残缺参照硬出变式。
    """
    facts = _mother_facts(state)
    dna = facts.get("dna") or {}
    if not isinstance(dna, dict):
        dna = {}

    # 诚实三态：B1 已让无考模型时 dna.models=[] + model_flag="no_model"。compress 原样透传，
    # 并据此算 no_model（兼容旧线程无 flag：models 空亦视作无考模型态）。
    model_flag = dna.get("model_flag")
    models = [
        {
            "id": str(m.get("id") or "") or None,
            "name": str(m.get("name") or "") or None,
            "tier_int": m.get("tier_int") if isinstance(m.get("tier_int"), int) else None,
            "freq_int": m.get("freq_int") if isinstance(m.get("freq_int"), int) else None,
        }
        for m in (dna.get("models") or [])
        if isinstance(m, dict) and (m.get("id") or m.get("name"))
    ]
    if model_flag is None and not models:
        model_flag = "no_model"
    no_model = (model_flag == "no_model") or (not models)

    difficulty = _ref_difficulty(state, dna)

    # 结构化 DNA（机器可续；键名沿用现有 DNA schema，不新造）。
    ref_dna: dict[str, Any] = {
        "main_kp": dna.get("main_kp") or None,
        "secondary_kps": dna.get("secondary_kps") or [],
        "qtype": str(facts.get("qtype") or "") or None,
        "grade": str(facts.get("grade") or "") or None,
        "grade_code": facts.get("subject_id"),
        "main_kp_id": facts.get("dim1_kp_id"),
        "models": models,
        "model_flag": model_flag,
        "no_model": bool(no_model),
        "skeleton": dna.get("skeleton"),
        "scenario": dna.get("scene") or dna.get("scenario"),
        "difficulty": difficulty,
        "tags": [str(t) for t in (dna.get("tags") or []) if str(t).strip()],
        "exam_type": str(dna.get("exam_type") or "") or None,
        # 🔴 守恒/白名单/守门下游仍按「完整 dna」消费 → 原样保留整块（_mother_facts 的 dna）。
        "_raw": dna,
    }

    # 异常判据：无 DNA、或缺年级、或缺主考点（与 generate 入口防御断言同口径）。
    grade = str(facts.get("grade") or "").strip()
    kp = str(facts.get("kp_name") or "").strip()
    incomplete = (
        not dna
        or grade in ("", "未知年级")
        or kp in ("", "未知考点")
        or not facts.get("dim1_kp_id")
    )

    figure = {
        "image_url": str(state.get("image_url") or "") or None,
        "mother_figure_url": state.get("mother_figure_url"),
        "has_figure": bool(state.get("mother_has_figure")),
    }

    return {
        "version": MOTHER_CORE_REF_VERSION,
        "incomplete": bool(incomplete),
        "dna": ref_dna,
        "summary": _build_summary(facts, difficulty, no_model),
        "stem": facts.get("stem") or "",
        "answer": facts.get("mother_answer") or "",
        "solution": facts.get("mother_solution") or facts.get("skeleton") or "",
        "figure": figure,
        # 🔴 阶段二 fan-out 子任务直接吃这份 facts，不再 _mother_facts(state) 重算（B0 坑2）。
        #   facts 是 _mother_facts 的完整产物（含入库用 subject_id/dim1_kp_id/mother_* 等），
        #   generate/守恒/装配全套消费链零改即可从参照取。
        "facts": facts,
    }


def facts_from_ref(state: VariantState) -> dict[str, Any]:
    """🔴 隔离接缝：阶段二取 facts 的统一入口——**优先读 frozen mother_core_ref.facts**，
    缺（旧线程 / 库内母题直进 / 未过 compress 的旁路）→ 回退 `_mother_facts(state)`（旧行为）。

    这样 generate（及 B3 fan-out 子任务）不再各自 `_mother_facts(state)` 重算，吃的是
    compress 固化的同一份不可变参照（AC3/G3 隔离 + 并发不读脏）。
    """
    ref = state.get("mother_core_ref")
    if isinstance(ref, dict):
        f = ref.get("facts")
        if isinstance(f, dict) and f:
            return f
    return _mother_facts(state)


async def compress(state: VariantState, config: RunnableConfig) -> VariantState:
    """压缩闸节点（STOP1 后、阶段二 generate 前）：固化 MotherCoreRef 进 state。

    🔴 不读 state.messages（隔离铁律）；产物是 frozen 参照对象。异常路径标 incomplete=true。
    🔴 →END+resume 模型不动：本节点只写 state，图里 compress→generate 直连（见 graph.py）。
    """
    ref = build_mother_core_ref(state)
    if ref.get("incomplete"):
        # 阶段一未定死 → 不硬塞残缺参照进阶段二；发降级灯（沿用 C-105 D 弹窗那条路：
        # generate 入口防御断言会拦下并回确认态，此处只标灯不抢路由）。
        _emit_stage("compress", "固化母题参照", "warn", "母题未定死，参照标记为不完整")
    else:
        summary = str(ref.get("summary") or "").strip()
        _emit_stage("compress", "固化母题参照", STAGE_DONE, summary or "母题核心参照已固化")
    return {"mother_core_ref": ref}
