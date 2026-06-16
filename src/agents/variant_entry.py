# -*- coding: utf-8 -*-
r"""PRD-C-100 B1a · 塌缩入口：opus 一把（读图判年级册+章 + has_figure + 置信 + 解题 + 10维DNA打标）。

🔴 心智（PRD §1.1 + D3）：把旧入口三节点 `analyze(gpt5.4) + mother_precheck(nano) + classify(opus)`
   合并重写为单入口节点 `mother_opus_entry`：
   - **高置信（≥0.80 且无强歧义章）→ 1 次 opus 全 done**（读图判章+解题+10维DNA → 代码锚定 →
     母题卡先出 → 硬停 await_review）。
   - **低置信（<0.80 或 ≥2 强候选章）→ 弹窗确认年级册+章 → resume 走既有 `classify`（+1 次 opus
     按确认章重锚打标，池注入 build_mother_prompt）**。D3「高置信1次/低置信+1重锚」。
   - **带图不再打回**（反转 C-017 G13）：has_figure 仅如实记录，不 reject；母题卡照出，带图切图归 B3。

🔴 控制流重写边界（铁律）：本模块只动**入口段**；变式四节点（generate/gene_gate/solve_explain/
   assemble）+ 闸A基因/闸B sympy 判决**字节级不动**。母题闸A（validate_rich_text）/闸B（anchor_to_chapter）
   仍走 mother_opus.py 的**同一纯函数**（高置信路径在此调用，逻辑与 classify 一致）。

🔴 复用而非重造：opus 调用走 variant._ainvoke_text（自带 conv_trace 计费 + relay 熔断转移）；
   DNA 归一/闸A/闸B = mother_opus.opus_to_dna/validate_rich_text/anchor_to_chapter；模型锚 = model_anchor；
   母题卡帧/合并确认/事实冻结 = variant._emit_mother_card/build_mother_confirm（懒导入 variant 防循环）。

🔴 max_tokens 护栏（B0 H5 实测定 12288，见 settings.MOTHER_OPUS_MAX_TOKENS）：母题节点宽护栏防失控
   不截断（实测峰值 5158，零截断）。
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from agents import dna_extract, model_anchor, mother_opus

# D1：条件 confirm 触发阈值（置信 < 0.80 或 ≥2 强候选章 → 弹窗）。
CONF_CONFIRM_THRESHOLD = 0.80

# 6 个浙教版教材册（与 analyze 旧闭集对齐，防口径漂移）。
_GRADE_BOOKS = [
    "七年级上册", "七年级下册", "八年级上册", "八年级下册", "九年级上册", "九年级下册",
]


# ---------------------------------------------------------------------------
# opus 一把 schema（在 mother_opus.MOTHER_SCHEMA 上加「判年级册+章+置信+候选」头）
# ---------------------------------------------------------------------------
def _entry_schema() -> dict[str, Any]:
    base = mother_opus.MOTHER_SCHEMA
    props = dict(base["properties"])
    props["gradeBook"] = {"type": "string", "description": "6 册之一或空串"}
    props["chapter"] = {"type": "string", "description": "章名（如:第2章 一元二次方程），判不出留空"}
    props["gradeCandidates"] = {"type": "array", "items": {"type": "string"},
                                "description": "强候选年级册（拿不准时给 1~2 个）"}
    props["chapterCandidates"] = {"type": "array", "items": {"type": "string"},
                                  "description": "强候选章（≥2 个=章歧义，触发确认）"}
    props["confidence"] = {"type": "number", "description": "对年级册+章判定的整体置信 0~1"}
    return {
        "type": "object",
        "properties": props,
        "required": list(base["required"]) + ["gradeBook", "confidence"],
    }


ENTRY_SCHEMA = _entry_schema()
RESPONSE_FORMAT_ENTRY: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "mother_entry", "schema": ENTRY_SCHEMA, "strict": False},
}


# ---------------------------------------------------------------------------
# opus 一把 prompt（判章 + 解题 + 开集 10 维打标；无叶子池注入，kp 开集捕获，代码后锚）
# ---------------------------------------------------------------------------
def build_entry_prompt(*, utterance: str | None = None) -> str:
    """opus 一把合并 prompt：先判年级册+章+置信（含候选），再真解题，再据解答做 10 维 DNA 开集打标。

    🔴 与 mother_opus.build_mother_prompt 的区别：本 prompt **不注入叶子池**（入口阶段年级未定、
       池取不到，鸡生蛋）→ kp 开集（primaryKp.id 留空、name 如实写），由节点代码用年级叶子池**后锚**
       （_match_kp_in_pool + mother_opus.anchor_to_chapter）。低置信走确认后由 classify 池注入重锚。
    """
    exam_types = "/".join(dna_extract.EXAM_TYPES)
    books = "、".join(_GRADE_BOOKS)
    note = f"\n🔴 老师附带的出题要求（仅语境参考，**不要**在此抽出题配方）：{utterance}\n" if utterance else ""
    return f"""你是浙教版初中数学命题专家 + 题库打标师。看这张母题图，按顺序做三件事并一次输出：
① **判年级册 + 章 + 置信度**（判定母题属于哪个教材册、哪一章，给整体置信 0~1，拿不准给候选）；
② **真正把题解出来**（一步步算到最终答案，不许抄图、不许跳步）；
③ **据你的解答做 10 维 DNA 打标**（母题是所有变式的基准，解错则全变式跟着错——务必稳准）。{note}
================ 判年级册 + 章（闭集 + 候选） ================
- gradeBook **只能从这 6 册选一个**：{books}。判不出 → 留空串 ""、confidence 给低、gradeCandidates 给你最可能的 1~2 个。
- chapter：章名（如「第2章 一元二次方程」）；判不出留空。**章有歧义（说不清是哪一章）→ chapterCandidates 给 ≥2 个**。
- confidence：对「年级册+章」判定的整体把握 0~1（不是解题把握）。**没十足把握就给低（<0.8），别硬撑高分。**

================ 学段锁死红线 ================
🔴 解法不得超出你判定年级的进度（不许用更高年级才学的定理/方法绕过）；考点落在该年级/章范围内。

================ 10 维逐维规则（开集打标，kp 写真实考点名） ================
1. primaryKp 主考点：写**真实考点名**（id 留空串，由系统后锚到题库叶子）：{{"id":"","name":"一元二次方程的根的判别式"}}。
2. secondaryKps 副考点 0~3：与主不同体系才算，同样 {{"id":"","name":"..."}}；没有就空数组。
3. qtype 题型：选择/填空/解答 之一（闭集）。
4. assessmentType 考察类型：**闭集10选1** = {exam_types}。
5. solutionSkeleton 解法骨架：解题步骤序列；**最难的那一步用【】整步包住**（至多一处）。如实反映你解题真实路径，是变式守恒基因。
6. hardPointCount + breakthroughPoints 难点（克制·宁空不凑）：基础/纯套公式/直接计算/概念辨析/送分题 → breakthroughPoints **必空**、hardPointCount=0；🔴 hardPointCount **必须等于** breakthroughPoints 数组长度。
7. scenario 场景：一句话场景 或 "纯代数"。
8. difficulty 难度四档（按构造断言）：1★送分(无难点+概念辨析/单步)；2★★常规(无难点+{{直接计算·公式套用·性质判定}}+多步)；3★★★(1难点 或 {{证明推理·应用建模·探究归纳}} 或 骨架含【最难步】)；4★压轴(≥2难点 或 多突破口综合)。
9. tags 标签 3~6：检索标签（求什么/用什么定理/什么方法/什么场景）；禁近义增生。
10. modelCandidates 解题模型（克制）：真有可复用套路才给候选名（简单题空数组），只给名不给 M-id。

================ 富文本红线（题面/答案/解析） ================
🔴 数学式用行内 $...$；换行 \\n；禁裸 LaTeX 命令、禁 \\( \\) / \\[ \\] 定界；LaTeX 括号/命令参数配对完整（下游有机器闸逐项检）。
🔴 has_figure：题面真含图形/图表/几何图填 true；只是拍照的纯文本题填 false。

================ 输出（只输出一个 JSON，不要解释、不要 markdown fence） ================
{{
  "gradeBook": "六册之一 或 空串",
  "chapter": "章名 或 空串",
  "gradeCandidates": [],
  "chapterCandidates": [],
  "confidence": 0.0,
  "has_figure": true/false,
  "richText": {{"stem": "题干(Markdown+行内$LaTeX$)", "answer": "标准答案", "analysis": "解析(含解题过程)"}},
  "solvedAnswer": "你一步步解出的最终答案",
  "dna": {{
    "primaryKp": {{"id": "", "name": "真实考点名"}},
    "secondaryKps": [],
    "qtype": "选择/填空/解答",
    "assessmentType": "上述闭集10之一",
    "solutionSkeleton": ["步骤1", "步骤2(最难一步用【】整步包住)"],
    "hardPointCount": 0,
    "breakthroughPoints": [],
    "scenario": "一句话场景 或 纯代数",
    "difficulty": 1,
    "tags": ["3~6个检索标签"],
    "modelCandidates": []
  }}
}}"""


# ---------------------------------------------------------------------------
# D1 条件 confirm 判定（置信 < 0.80 或 ≥2 强候选章 → 弹窗）
# ---------------------------------------------------------------------------
def decide_confirm(entry: dict[str, Any]) -> dict[str, Any]:
    """据 opus 一把输出判是否需要弹窗确认。返回
       {needs_confirm:bool, reason:str, grade_book, chapter, grade_candidates, chapter_candidates, confidence}。
    D1：confidence < 0.80 → 确认；chapterCandidates 去重后 ≥2 → 章歧义 → 确认；gradeBook 空 → 确认。
    """
    conf = entry.get("confidence")
    try:
        conf_f = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        conf_f = 0.0
    grade_book = str(entry.get("gradeBook") or "").strip()
    chapter = str(entry.get("chapter") or "").strip()
    chap_cands = [str(c).strip() for c in (entry.get("chapterCandidates") or []) if str(c).strip()]
    chap_cands = list(dict.fromkeys(chap_cands))  # 去重保序
    grade_cands = [str(c).strip() for c in (entry.get("gradeCandidates") or []) if str(c).strip()]
    grade_cands = list(dict.fromkeys(grade_cands))

    reasons: list[str] = []
    if not grade_book:
        reasons.append("年级册判不出")
    if conf_f < CONF_CONFIRM_THRESHOLD:
        reasons.append(f"置信{conf_f:.2f}<{CONF_CONFIRM_THRESHOLD}")
    if len(chap_cands) >= 2:
        reasons.append(f"{len(chap_cands)}个强候选章歧义")
    return {
        "needs_confirm": bool(reasons),
        "reason": "；".join(reasons) or "高置信直过",
        "grade_book": grade_book,
        "chapter": chapter,
        "grade_candidates": grade_cands or ([grade_book] if grade_book else []),
        "chapter_candidates": chap_cands or ([chapter] if chapter else []),
        "confidence": conf_f,
    }


def _match_kp_in_pool(name: str, leaf_pool: list[tuple[str, str]]) -> str | None:
    """开集 kp 名 → 年级叶子池 id（高置信路径后锚）。精确名匹配优先，退包含匹配（最短名优先=最细叶子）。
    锚不到 → None（mother_opus.anchor_to_chapter 据此走「宁空不凑」need_anchor_review）。"""
    name = (name or "").strip()
    if not name:
        return None
    # 精确
    for pid, pname in leaf_pool:
        if str(pname).strip() == name:
            return str(pid)
    # 包含（叶子名 ⊆ opus名 或 opus名 ⊆ 叶子名）；多命中取叶子名最短（最具体）
    cands = [
        (str(pid), str(pname))
        for pid, pname in leaf_pool
        if name and (name in str(pname) or str(pname) in name)
    ]
    if cands:
        cands.sort(key=lambda x: len(x[1]))
        return cands[0][0]
    return None


# ---------------------------------------------------------------------------
# 入口节点（懒导入 variant 防循环：variant 模块加载期 import 本模块挂图，本节点运行期才反向用 variant 机具）
# ---------------------------------------------------------------------------
async def mother_opus_entry(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """B1a 塌缩入口节点：opus 一把（判章+解题+10维打标）→ 高置信直过/低置信弹确认。"""
    from agents import variant as V  # 懒导入防循环（运行期才用）

    url = V._extract_image_url(V._latest_human_text(state.get("messages", []))) or state.get(
        "image_url"
    )
    if not url:
        return {"_entry_finalized": False,
                "messages": [AIMessage(content="请先贴一张题目图的 OSS URL，我才能开始举一反三。")]}

    V._emit_stage("classify", "锚定考点", "running", "opus 读图判章 + 解题打标…")

    user_text = V._strip_urls(V._latest_human_text(state.get("messages", [])))
    prompt = build_entry_prompt(utterance=user_text or None)
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    opus_model = V.settings.variant_model("mother_solve_label")  # fail-fast 已锁 opus
    try:
        opus_text = await V._ainvoke_text(
            [msg], model=opus_model,
            max_tokens=V.settings.MOTHER_OPUS_MAX_TOKENS,
            temperature=mother_opus.MOTHER_OPUS_TEMPERATURE,
            response_format=RESPONSE_FORMAT_ENTRY,
            timeout=mother_opus.MOTHER_OPUS_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001 — opus 失败 → SSE error，绝不静默退 gpt-5.4（母题唯一安全网）
        V._emit_stage("classify", "锚定考点", "error", "母题读图解题失败（opus 超时/异常）")
        V._emit_error("mother_opus_failed", f"母题读图解题失败（{str(e)[:80]}），请重试或换更清晰的图。")
        return {
            "image_url": url, "_entry_finalized": False,
            "messages": [AIMessage(content="母题读图解题失败了（opus 超时或异常），请重试或换一张更清晰的题目图。")],
        }

    entry = V._parse_json(opus_text)
    if not isinstance(entry, dict):
        V._emit_stage("classify", "锚定考点", "error", "母题解题打标解析失败")
        V._emit_error("mother_opus_parse_fail", "母题解题打标结果解析失败，请重试。")
        return {
            "image_url": url, "_entry_finalized": False,
            "messages": [AIMessage(content="母题解题打标结果没解析出来，请重试。")],
        }

    has_figure = bool(entry.get("has_figure"))  # 🔴 仅记录，不 reject（反转 C-017 带图打回）
    decision = decide_confirm(entry)

    # 出题配方旋钮（与 analyze 同口径）：utterance 非空 → 独立纯文本抽取（数量词稳）；纯贴图不抽。
    knobs: dict[str, Any] = {}
    if user_text:
        try:
            knobs = await V._extract_knobs({**state, "image_url": url, "messages": state.get("messages", [])})
        except Exception:  # noqa: BLE001
            knobs = {}

    # 新母题轮基础 state（重置在途母题态 + 暂存 opus 一把输出供 confirm-resume 复用）
    base_out: dict[str, Any] = {
        "image_url": url,
        "images_count": 1,
        "questions_in_image": 1,
        "knobs": knobs,
        "shape_defects": [],
        "mother_precheck": None,
        "awaiting_mother_confirm": False,
        "mother_rejected": False,
        "confirmed_chapter_id": None,
        "confirmed_grade_book_id": None,
        # B1a：暂存 opus 一把判定（has_figure 给 B3 切图判定；entry_opus 给 confirm-resume 兜底）
        "mother_has_figure": has_figure,
        "entry_decision": decision,
        "_entry_finalized": True,  # after_mother_entry 路由信号（高置信/弹窗均算「本轮入口处理过」）
    }

    # ---- 低置信 / 章歧义 → 弹窗确认（resume 走既有 classify 池注入重锚 +1 次 opus） ----
    if decision["needs_confirm"]:
        payload = {
            "grade_book": {"id": "", "name": decision["grade_book"]},
            "chapter": {"id": "", "name": decision["chapter"]},
            "grade_candidates": [{"id": "", "name": n} for n in decision["grade_candidates"]],
            "chapter_candidates": [{"id": "", "name": n} for n in decision["chapter_candidates"]],
            "confidence": decision["confidence"],
        }
        V._emit_need_confirm(payload)
        V._emit_stage("classify", "锚定考点", V.STAGE_AWAIT, "请确认年级与章后继续")
        V._emit_stage("knobs", "解析配方", V.STAGE_AWAIT, "待确认年级章后定配方")
        grade_line = decision["grade_book"] or "（未判出，请手选）"
        chapter_line = decision["chapter"] or "（未判出，请手选）"
        body = (
            f"我读图判了一下母题范围（{decision['reason']}），**请确认年级册与章**再继续举一反三：\n\n"
            f"- 年级册：**{grade_line}**\n- 章：**{chapter_line}**\n\n"
            "确认无误请回复「确认」，需要修改请直接告诉我正确的年级/章。"
        )
        # 暂存 opus 一把的 richText/solve/dna 供 classify-resume 复用（避免 confirm 后白丢这次解题）
        prov_dna = dict(state.get("mother_dna") or {})
        rich = entry.get("richText") or {}
        if isinstance(rich, dict):
            if rich.get("stem"):
                prov_dna["stem"] = V._sanitize_rich_text(rich.get("stem"))
            if rich.get("answer"):
                prov_dna["answer"] = V._sanitize_rich_text(rich.get("answer"))
            if rich.get("analysis"):
                prov_dna["analysis"] = V._sanitize_rich_text(rich.get("analysis"))
        return {
            **base_out,
            "analysis": {
                "grade": {"value": decision["grade_book"], "confidence": decision["confidence"]},
                "subject": "数学",
                "kp": {"value": (entry.get("dna") or {}).get("primaryKp", {}).get("name")
                       if isinstance((entry.get("dna") or {}).get("primaryKp"), dict) else None,
                       "confidence": decision["confidence"]},
                "qtype": {"value": (entry.get("dna") or {}).get("qtype"), "confidence": decision["confidence"]},
            },
            "mother_dna": prov_dna,
            "awaiting_mother_confirm": True,
            "mother_confirmed": False,
            "messages": [AIMessage(content=body)],
        }

    # ---- 高置信 → 1 次 opus 全 done：代码后锚 + 母题卡先出 + 硬停 ----
    out = await _finalize_high_conf(state, config, entry, decision, base_out, V)
    return out


async def _finalize_high_conf(
    state: dict[str, Any], config: RunnableConfig, entry: dict[str, Any],
    decision: dict[str, Any], base_out: dict[str, Any], V: Any,
) -> dict[str, Any]:
    """高置信路径终结：grade_code → 年级叶子池 → 开集 kp 后锚（闸B）→ model_anchor → 母题卡 → await_review。

    🔴 闸A/闸B 走 mother_opus 同一纯函数（与 classify 一致逻辑，不另造判决）。
    🔴 不再调 opus（复用一把输出）—— 这是「高置信1次全done」的兑现。
    """
    analysis: dict[str, Any] = {
        "grade": {"value": decision["grade_book"], "confidence": max(decision["confidence"], V.CONF_GATE)},
        "subject": "数学",
        "kp": {"value": None, "confidence": 0},
        "qtype": {"value": None, "confidence": 0},
    }
    mother_dna = dict(state.get("mother_dna") or {})

    include_review_books = V._wants_review_books(V._latest_human_text(state.get("messages", [])))

    # 年级 code：opus 判的 gradeBook 归一 4 位 code（落空退粗考点反查）
    grade_code = V._grade_to_code(decision["grade_book"])
    if not grade_code:
        grade_code = await V._resolve_grade_code(analysis)
    if grade_code:
        analysis["grade"]["code"] = grade_code

    token = ((config or {}).get("configurable") or {}).get("ruoyi_token")
    client = V.RuoyiClient(token=token)
    leaf_pool: list[tuple[str, str]] = []
    try:
        leaf_pool = await V.leaf_pool_for_grade(
            grade_code, client, include_review_books=include_review_books
        )
    except Exception as e:  # noqa: BLE001 — 池故障 → 空池降级（与 classify 同口径，转 needs_confirm）
        analysis["_anchor_error"] = str(e)

    if not leaf_pool:
        await client.aclose()
        analysis.setdefault("_anchor_error", "知识点叶子池不可用（库未起/年级未识别）")
        V._emit_stage("classify", "锚定考点", "warn", "知识点池不可用，待老师确认")
        V._emit_stage("knobs", "解析配方", "warn", "待老师确认母题后再定配方")
        early: dict[str, Any] = {**base_out, "analysis": analysis, "mother_confirmed": False, "messages": []}
        early["mother_confirm"] = V.build_mother_confirm({**state, **early})
        return early

    # opus 富文本回填 mother_dna（题面/答案/解析）
    rich = entry.get("richText") or {}
    if isinstance(rich, dict):
        if rich.get("stem"):
            mother_dna["stem"] = V._sanitize_rich_text(rich.get("stem"))
        if rich.get("answer"):
            mother_dna["answer"] = V._sanitize_rich_text(rich.get("answer"))
        if rich.get("analysis"):
            mother_dna["analysis"] = V._sanitize_rich_text(rich.get("analysis"))

    # DNA 归一（mother_opus.opus_to_dna，与 classify 同函数）
    dna = mother_opus.opus_to_dna(entry)
    skeleton_lines = dna.get("skeleton") or []
    if skeleton_lines:
        mother_dna["solution_skeleton"] = "\n".join(str(s) for s in skeleton_lines)
    solved = entry.get("solvedAnswer")
    if solved:
        mother_dna["solved_answer"] = V._sanitize_rich_text(solved)
    mother_dna["mother_solve_source"] = "opus"

    # 🔴 开集 kp 名 → 年级叶子池后锚（高置信路径专属；id 落定后交闸B 校验前缀）
    main_kp_obj = dna.get("main_kp") or {}
    if not (main_kp_obj.get("id") or "").strip() and main_kp_obj.get("name"):
        matched = _match_kp_in_pool(main_kp_obj["name"], leaf_pool)
        if matched:
            dna["main_kp"] = {"id": matched, "name": main_kp_obj["name"]}
    for s in dna.get("secondary_kps") or []:
        if not (s.get("id") or "").strip() and s.get("name"):
            mid = _match_kp_in_pool(s["name"], leaf_pool)
            if mid:
                s["id"] = mid

    # 闸A 富文本机器验证（G10，同 classify）
    rt_check = mother_opus.validate_rich_text(
        rich if isinstance(rich, dict) else {}, has_table=bool(entry.get("has_table")),
    )
    if not rt_check["ok"]:
        analysis["_richtext_issues"] = rt_check["issues"]
        mother_dna["need_richtext_review"] = True
        V._emit_stage("classify", "锚定考点", "warn",
                      f"母题富文本机器检发现 {len(rt_check['issues'])} 处问题，待人工复核")

    # 闸B 锚定·宁空不凑（G11，同 classify）：高置信无确认章 → 以年级册 4 位 code 作前缀
    chapter_id = grade_code  # 高置信路径锚到年级（章为 opus 判定文本，记录不收窄前缀）
    dna = mother_opus.anchor_to_chapter(
        dna, chapter_id=chapter_id, leaf_pool=leaf_pool, include_review_books=include_review_books,
    )
    if dna.get("need_anchor_review"):
        mother_dna["need_anchor_review"] = True
    main_kp = dna.get("main_kp") if (dna.get("main_kp") or {}).get("id") else None

    await client.aclose()

    # 模型锚（双轴·同 classify）：M00 兜底，故障不空维
    try:
        m_ref = str((main_kp or {}).get("id") or "") or None
        m_res = await model_anchor.anchor_models(
            dna, stem=mother_dna.get("stem") or "",
            answer=mother_dna.get("answer") or mother_dna.get("solution_skeleton") or "",
            invoke=V._ainvoke_text, model=V.settings.variant_model("model_confirm"),
            record_overflow=lambda name, mm: model_anchor.record_overflow_candidate(
                name, mm, question_ref=m_ref),
        )
    except Exception as e:  # noqa: BLE001
        analysis.setdefault("_model_anchor_error", str(e))
        m_res = {"models": [dict(model_anchor.M00)], "model_overflow": [], "model_warn": True,
                 "model_flag": "lookup_unavailable"}
    dna["models"] = m_res.get("models") or [dict(model_anchor.M00)]
    dna["model_overflow"] = m_res.get("model_overflow") or []
    if m_res.get("model_warn"):
        dna["model_warn"] = True
    mother_dna["dna"] = dna

    # 锚到真叶子 → 抬三锚置信（同 classify）
    if main_kp and main_kp.get("id"):
        analysis["kp"] = {
            "value": main_kp.get("name"),
            "confidence": max(float(analysis["kp"].get("confidence", 0) or 0), V.CONF_GATE),
            "anchored": {"id": main_kp["id"], "code": str(main_kp["id"]), "name": main_kp.get("name")},
        }
        if grade_code:
            analysis["grade"]["code"] = grade_code
        analysis["grade"]["confidence"] = max(float(analysis["grade"].get("confidence", 0) or 0), V.CONF_GATE)
        if dna.get("qtype"):
            analysis["qtype"] = {"value": dna["qtype"],
                                 "confidence": max(float(analysis["qtype"].get("confidence", 0) or 0), V.CONF_GATE)}

    confirmed = V._conf_ok(analysis) and bool((analysis.get("kp") or {}).get("anchored"))
    kp_name = (analysis.get("kp") or {}).get("value") or "?"
    grade_name = (analysis.get("grade") or {}).get("value") or grade_code or "?"
    V._emit_stage("classify", "锚定考点", "done" if confirmed else "warn",
                  f"考点「{kp_name}」·年级「{grade_name}」")
    recipe = V.knobs_desc(base_out.get("knobs")) or "未指定，走默认配方（3 道 = 2 普通 + 1 难）"
    V._emit_stage("knobs", "解析配方", "done" if confirmed else "warn", recipe)

    out: dict[str, Any] = {
        **base_out,
        "analysis": analysis,
        "mother_dna": mother_dna,
        "mother_confirmed": bool(confirmed),
        "facts_locked": bool(confirmed),
        "awaiting_mother_confirm": False,
        "messages": [],
    }
    out["mother_confirm"] = V.build_mother_confirm({**state, **out})
    V._emit_mother_card({**state, **out})  # 母题卡先出（早于变式）
    return out
