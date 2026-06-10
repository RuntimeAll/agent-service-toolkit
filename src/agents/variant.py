"""PRD-C-009 · 图片举一反三 agent（块二·单题→N 道变式）。

事实源 = codeplace-C/claude-code-sign/14-举一反三-设计.md（vibe-coding：先改文档再改代码）。

主干 4 步流水线（§2）+ solve 自愈 + assemble 外显：
  analyze(读图) → classify(锚图谱+DNA置信闸) → [clarify] → generate(3=2普通+1难)
                → solve_explain(真解+自愈1次+守恒校验) → assemble(题组快照)

🔴 不变量（§3）：
  - 主考点 + 年级 = 硬守恒（贯穿 generate / 重生 / 补题 / patch）；
  - 凡进 items 的题一律过 solve_explain（无 check 不许进 assemble）；
  - generate 入口前置：mother_confirmed==true 或 三锚高置信（DNA 闸收口，多入口都过）。

LLM 层（§9）：toolkit 原生 get_model（COMPATIBLE=LangChain ChatOpenAI）→ model.ainvoke；
  多模态走 HumanMessage(content=[{type:text},{type:image_url,image_url:{url:OSS_URL}}])；
  思考型只取 content（先 reasoning_content 后 content）；max_tokens≥4096；
  LLM 外呼 lk888 走默认（别套 trust_env=False，那是治本地 localhost 的）。

checkpointer 不在此 compile（service lifespan 注入 saver；多轮 state 按 thread_id 持久）。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from agents.variant_support import anchor_subject, persist_items
from core import get_model, settings

# DNA 三锚高置信门槛
CONF_GATE = 0.75
# 自愈上限（设计 §5：1 次防死循环）
MAX_HEAL = 1
# 默认配方：3 道 = 2 普通 + 1 难（设计 §5）
DEFAULT_SHAPE = {"normal": 2, "hard": 1}


# ---------------------------------------------------------------------------
# State（设计 prompt 指定结构）
# ---------------------------------------------------------------------------
class VariantState(MessagesState, total=False):
    image_url: str | None
    images_count: int
    questions_in_image: int
    # analysis：年级/考点(kp)/题型(qtype) 各带置信
    analysis: dict[str, Any]
    mother_dna: dict[str, Any]
    mother_confirmed: bool
    # items[{stem, answer, solution, qtype, difficulty, level,
    #        injected_kp?, check:{badge:ok|warn, solved_answer}}]
    items: list[dict[str, Any]]
    history: list[dict[str, Any]]
    # 交互层：parse_instruction 的解析结果（intent/ops/knobs/...），路由后各分支消费并清空
    pending: dict[str, Any] | None


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _model():
    # COMPATIBLE provider → gemini-3-flash-preview（思考型+多模态）
    return get_model(settings.DEFAULT_MODEL)


def _content_text(resp: BaseMessage) -> str:
    """思考型只取 content（reasoning_content 不外放）。content 可能是 str 或 parts list。"""
    c = resp.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(c)


async def _ainvoke_text(messages: list[BaseMessage], retry: bool = True) -> str:
    """ainvoke + 取 content；偶发空返回重试一次。max_tokens≥4096 给思考型留头。"""
    model = _model().bind(max_tokens=settings.VARIANT_MAX_TOKENS)
    resp = await model.ainvoke(messages)
    text = _content_text(resp).strip()
    if not text and retry:
        resp = await model.ainvoke(messages)
        text = _content_text(resp).strip()
    return text


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> Any:
    """剥 markdown fence + JSON 容错（抄 ai-orchestrator llm/client 范式）。"""
    text = (text or "").strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        # 退而求其次：截第一个 { 到最后一个 }
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                pass
    return None


def _latest_human_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            c = msg.content
            return c if isinstance(c, str) else _content_text(msg)
    return ""


_URL_RE = re.compile(r"https?://[^\s)>'\"]+", re.IGNORECASE)


def _extract_image_url(text: str) -> str | None:
    """从用户消息抽 OSS 题图 URL（MVP 贴 URL，file_uploader future）。"""
    m = _URL_RE.search(text or "")
    return m.group(0) if m else None


def _conf_ok(analysis: dict[str, Any]) -> bool:
    """三锚（年级/考点/题型）任一低置信 → 闸不过。"""
    for k in ("grade", "kp", "qtype"):
        node = analysis.get(k) or {}
        if float(node.get("confidence", 0) or 0) < CONF_GATE:
            return False
    return True


# ---------------------------------------------------------------------------
# Router（入口分诊：有图? 在途母题? 库内母题跳 analyze/classify）
# ---------------------------------------------------------------------------
def route_entry(state: VariantState) -> Literal["analyze", "parse", "generate", "ask"]:
    url = _extract_image_url(_latest_human_text(state.get("messages", [])))
    # 跨轮新图 = 视作新母题（设计 §6：重走 analyze，覆盖在途状态）
    if url:
        return "analyze"
    # 老会话·纯文字（已出过题组）→ parse 分诊 5 意图（设计 §3 mermaid G0）
    if state.get("items"):
        return "parse"
    # 库内母题（已确认 DNA）→ 直接造（跳 analyze/classify）
    if state.get("mother_confirmed") and state.get("mother_dna"):
        return "generate"
    # 没图、无在途母题、无题组 → 催图（设计 §6 输入边界兜底）
    return "ask"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = """你是浙教版初中数学命题专家。看这张题目图，**流式**输出母题分析。

只输出一个 JSON（不要解释），结构：
{
  "is_question_image": true/false,   // 非题目图(风景/截图/空白)填 false
  "images_count": 1,                  // 检测到几张图
  "questions_in_image": 1,            // 这张图里有几道题
  "grade": {"value": "七年级上学期", "confidence": 0.0~1.0},
  "subject": "数学",
  "kp": {"value": "核心考点粗描述(如:一元二次方程求根)", "confidence": 0.0~1.0},
  "qtype": {"value": "选择/填空/解答", "confidence": 0.0~1.0},
  "stem": "题干(Markdown+LaTeX)",
  "answer": "标准答案(图里没有就先解母题得出)",
  "difficulty": 1~5,
  "structure": "结构指纹/解法骨架简述",
  "solution_skeleton": "解法骨架(步骤)"
}
🔴 无答案先解母题得答案/解法骨架(作 verify 基准)。各锚(年级/考点/题型)如实给 confidence。"""


async def analyze(state: VariantState, config: RunnableConfig) -> VariantState:
    """① 分析：multimodal 读图 → 年级/学科/粗考点/题型 + 题干/答案/难度/结构 + 几图几题 + 各锚置信。"""
    url = state.get("image_url") or _extract_image_url(
        _latest_human_text(state.get("messages", []))
    )
    if not url:
        return {
            "messages": [AIMessage(content="请先贴一张题目图的 OSS URL，我才能开始举一反三。")]
        }

    # 🔴 多模态走 LangChain HumanMessage(content=[text, image_url]) → model.ainvoke
    msg = HumanMessage(
        content=[
            {"type": "text", "text": ANALYZE_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    )
    text = await _ainvoke_text([msg])
    data = _parse_json(text) or {}

    if data.get("is_question_image") is False:
        return {
            "image_url": url,
            "messages": [
                AIMessage(content="这张图我没认出是题目（可能是风景/截图/空白）。请换一张清晰的题目图。")
            ],
        }

    analysis = {
        "grade": data.get("grade") or {"value": None, "confidence": 0},
        "subject": data.get("subject") or "数学",
        "kp": data.get("kp") or {"value": None, "confidence": 0},
        "qtype": data.get("qtype") or {"value": None, "confidence": 0},
    }
    mother_dna = {
        "stem": data.get("stem"),
        "answer": data.get("answer"),
        "difficulty": data.get("difficulty"),
        "structure": data.get("structure"),
        "solution_skeleton": data.get("solution_skeleton"),
    }
    return {
        "image_url": url,
        "images_count": int(data.get("images_count") or 1),
        "questions_in_image": int(data.get("questions_in_image") or 1),
        "analysis": analysis,
        "mother_dna": mother_dna,
        "messages": [],
    }


async def classify(state: VariantState, config: RunnableConfig) -> VariantState:
    """② 分类：粗考点 → 锚 biz_subject 真实节点 + 年级(编码前4位)。DNA 置信闸。"""
    analysis = dict(state.get("analysis") or {})
    kp_node = dict(analysis.get("kp") or {})
    coarse = kp_node.get("value")

    # 只读 SQL 锚定（同步，丢线程池）
    candidates: list[dict] = []
    if coarse:
        try:
            candidates = await asyncio.to_thread(anchor_subject, coarse)
        except Exception as e:  # 库未起/连接失败 → 降级，置信不抬，转 clarify
            candidates = []
            analysis["_anchor_error"] = str(e)

    if candidates:
        best = candidates[0]
        kp_node["anchored"] = {
            "id": best["id"],
            "code": best["code"],
            "name": best["name"],
        }
        kp_node["value"] = best["name"]  # 用标准考点名
        # 命中真实节点 → 抬考点置信
        kp_node["confidence"] = max(float(kp_node.get("confidence", 0) or 0), CONF_GATE)
        analysis["kp"] = kp_node
        # 年级 = 编码前4位反推（比 LLM 裸猜准）→ 抬年级置信
        if best.get("grade_code"):
            grade_node = dict(analysis.get("grade") or {})
            grade_node["code"] = best["grade_code"]
            grade_node["confidence"] = max(
                float(grade_node.get("confidence", 0) or 0), CONF_GATE
            )
            analysis["grade"] = grade_node

    confirmed = _conf_ok(analysis)
    return {
        "analysis": analysis,
        "mother_confirmed": bool(confirmed),
        "messages": [],
    }


def gate_after_classify(state: VariantState) -> Literal["generate", "clarify"]:
    """🔴 DNA 闸：三锚高置信 或 已确认 → 造题；否则先 clarify。"""
    if state.get("mother_confirmed"):
        return "generate"
    return "clarify"


async def clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """DNA 置信不足 → 回问老师确认（只问不造，进 WAIT 等下一句）。"""
    analysis = state.get("analysis") or {}
    asks = []
    g = analysis.get("grade") or {}
    k = analysis.get("kp") or {}
    q = analysis.get("qtype") or {}
    if float(g.get("confidence", 0) or 0) < CONF_GATE:
        asks.append(f"年级我拿不准（看着像「{g.get('value') or '?'}」），是几年级上/下学期？")
    if float(k.get("confidence", 0) or 0) < CONF_GATE:
        asks.append(f"核心考点我没锚准（粗看是「{k.get('value') or '?'}」），对吗？或请指正。")
    if float(q.get("confidence", 0) or 0) < CONF_GATE:
        asks.append(f"题型我读着像「{q.get('value') or '?'}」，对吗？")
    if not asks:
        asks.append("我对母题 DNA 还不够确定，请确认下年级/考点/题型再继续。")
    body = "我先确认母题 DNA，确认后再造变式：\n\n" + "\n".join(f"- {a}" for a in asks)
    return {"messages": [AIMessage(content=body)]}


GENERATE_PROMPT = """你是浙教版初中数学命题专家。基于母题 DNA，造 {n} 道举一反三变式。

母题 DNA：
- 主考点(硬守恒，不可改): {kp_name}
- 年级(硬守恒): {grade}
- 题型: {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}

配方(默认)：共 {n} 道 = {n_normal} 道普通(守难度) + {n_hard} 道难题(升一档)。
铁律：
- **主考点 + 年级 硬守恒**：每道题都必须仍考「{kp_name}」、仍在该年级范围内。
- 守{{解题结构, 难度(普通题)}}；只换{{数字, 场景}}。
- 难题 1 道升一档；可综合 1 个相邻知识点(主考点仍守，注入为副点)，填到 injected_kp。

只输出 JSON 数组(不要解释)，每个元素：
{{"stem":"题干(Markdown+LaTeX)","answer":"标准答案","solution":"完整解析(过程+答案)",
  "qtype":"选择/填空/解答","difficulty":1~5,"level":"normal/hard","injected_kp":"相邻kp名或null"}}"""


def _mother_facts(state: VariantState) -> dict:
    analysis = state.get("analysis") or {}
    dna = state.get("mother_dna") or {}
    kp = analysis.get("kp") or {}
    anchored = kp.get("anchored") or {}
    return {
        "kp_name": kp.get("value") or "未知考点",
        "grade": (analysis.get("grade") or {}).get("value") or "未知年级",
        "qtype": (analysis.get("qtype") or {}).get("value") or "解答",
        "stem": dna.get("stem") or "",
        "skeleton": dna.get("solution_skeleton") or dna.get("answer") or "",
        # 入库用：锚定到的真实节点编码 + 母题 id（图母题 MVP 无 id）
        "subject_id": anchored.get("code"),
        "mother_question_id": dna.get("mother_question_id"),
        # 🔴 图母题不在库 → 入库时先把母题(原题)也落库挂血缘，下面这几项给 build_mother_bo 用
        "mother_answer": dna.get("answer"),
        "mother_solution": dna.get("solution_skeleton") or dna.get("answer"),
        "mother_difficulty": dna.get("difficulty"),
        # 🔴 PRD-C-009 入库存 DNA/打标：结构指纹(dim5) + 锚定置信(labelConfidence)
        "mother_structure": dna.get("structure"),
        "kp_confidence": (kp.get("confidence") if isinstance(kp, dict) else None),
        "image_url": state.get("image_url"),
    }


async def generate(state: VariantState, config: RunnableConfig) -> VariantState:
    """③ 造题：默认 3 = 2 普通 + 1 难。🔴 入口断言 mother_confirmed 或三锚高置信（防裸奔）。"""
    if not (state.get("mother_confirmed") or _conf_ok(state.get("analysis") or {})):
        # DNA 闸未过却走到 generate（多入口兜底）→ 拒造，回 clarify 语义
        return {
            "messages": [
                AIMessage(content="母题 DNA 还没确认，我先不造题。请确认年级/考点/题型。")
            ]
        }

    facts = _mother_facts(state)
    n_normal, n_hard = DEFAULT_SHAPE["normal"], DEFAULT_SHAPE["hard"]
    n = n_normal + n_hard
    prompt = GENERATE_PROMPT.format(
        n=n, n_normal=n_normal, n_hard=n_hard, **facts
    )
    text = await _ainvoke_text([HumanMessage(content=prompt)])
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    items = []
    for it in data or []:
        if not isinstance(it, dict):
            continue
        items.append(
            {
                "stem": it.get("stem"),
                "answer": it.get("answer"),
                "solution": it.get("solution"),
                "qtype": it.get("qtype") or facts["qtype"],
                "difficulty": it.get("difficulty"),
                "level": it.get("level") or "normal",
                "injected_kp": it.get("injected_kp") or None,
                # check 待 solve_explain 填（无 check 不许进 assemble）
            }
        )
    return {"items": items, "messages": []}


SOLVE_PROMPT = """你是严谨的数学阅卷老师。真解下面这道题（不看给定答案，独立算一遍）。

题干：{stem}

只输出 JSON：
{{"solved_answer":"你独立算出的答案","solution":"完整解题过程(含答案)",
  "kp_name":"这道题实际考的主考点","grade":"这道题适配的年级"}}"""

REGEN_PROMPT = """下面这道变式题，独立解出的答案与题面标答不一致，请**重新出一道**等价变式重做。

主考点(硬守恒): {kp_name}
年级(硬守恒): {grade}
原题干: {stem}
要求：仍考「{kp_name}」、仍在「{grade}」、{level} 难度；换数字/场景使题面与答案自洽。

只输出 JSON：
{{"stem":"新题干","answer":"标准答案","solution":"完整解析","qtype":"{qtype}","difficulty":{difficulty},"level":"{level}","injected_kp":{injected_kp}}}"""


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip().lower()


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


async def _solve_one(stem: str) -> dict:
    text = await _ainvoke_text([HumanMessage(content=SOLVE_PROMPT.format(stem=stem or ""))])
    return _parse_json(text) or {}


async def solve_explain(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ solve 即验收 + 自愈1次。每题真解→产解析；解出≠标答→重生1次重解；

    🔴 重生版仍须过主考点+年级守恒（破则丢弃重生保留原版打⚠）；仍不过→check.badge=warn。
    🔴 凡进 items 的题一律过本节点，无 check 不许进 assemble。
    """
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    out: list[dict] = []

    for it in items:
        item = dict(it)
        if item.get("check"):  # 已定状态（如自愈过的补题再次流经）→ 不重复
            out.append(item)
            continue

        solved = await _solve_one(item.get("stem", ""))
        solved_answer = solved.get("solved_answer")
        # solve 产出解析（给老师当判题依据；优先用阅卷解析）
        if solved.get("solution"):
            item["solution"] = solved.get("solution")

        match = _norm(solved_answer) == _norm(item.get("answer"))
        if match:
            item["check"] = {"badge": "ok", "solved_answer": solved_answer}
            out.append(item)
            continue

        # 解出 ≠ 标答 → 自愈：重生 1 次 → 重解
        healed = None
        if MAX_HEAL >= 1:
            regen_text = await _ainvoke_text(
                [
                    HumanMessage(
                        content=REGEN_PROMPT.format(
                            kp_name=facts["kp_name"],
                            grade=facts["grade"],
                            stem=item.get("stem", ""),
                            level=item.get("level") or "normal",
                            qtype=item.get("qtype") or facts["qtype"],
                            difficulty=item.get("difficulty") or 3,
                            injected_kp=json.dumps(item.get("injected_kp"), ensure_ascii=False),
                        )
                    )
                ]
            )
            regen = _parse_json(regen_text)
            if isinstance(regen, dict) and regen.get("stem"):
                resolved = await _solve_one(regen.get("stem", ""))
                r_answer = resolved.get("solved_answer")
                r_match = _norm(r_answer) == _norm(regen.get("answer"))
                # 🔴 重生版仍须过守恒校验
                cons = _conservation_ok(
                    resolved.get("kp_name", ""), resolved.get("grade", ""), facts
                )
                if r_match and cons:
                    healed = {
                        "stem": regen.get("stem"),
                        "answer": regen.get("answer"),
                        "solution": resolved.get("solution") or regen.get("solution"),
                        "qtype": regen.get("qtype") or item.get("qtype"),
                        "difficulty": regen.get("difficulty") or item.get("difficulty"),
                        "level": regen.get("level") or item.get("level"),
                        "injected_kp": regen.get("injected_kp"),
                        "check": {"badge": "ok", "solved_answer": r_answer},
                    }

        if healed:
            out.append(healed)
        else:
            # 守恒破 或 重生仍不过 → 保留原版打 ⚠
            item["check"] = {
                "badge": "warn",
                "solved_answer": solved_answer,
            }
            out.append(item)

    return {"items": out, "messages": []}


def _fmt_item(idx: int, it: dict) -> str:
    badge = (it.get("check") or {}).get("badge", "warn")
    mark = "✓" if badge == "ok" else "⚠"
    lvl = "难题" if it.get("level") == "hard" else "普通"
    inj = it.get("injected_kp")
    inj_s = f"（综合相邻考点：{inj}）" if inj else ""
    warn_s = ""
    if badge != "ok":
        sa = (it.get("check") or {}).get("solved_answer")
        warn_s = f"\n> ⚠ 我没算准（独立解得「{sa}」与标答不一致），老师重点看。"
    return (
        f"### 第 {idx} 题 {mark}（{lvl}·{it.get('qtype', '')}·难度{it.get('difficulty', '?')}）{inj_s}\n\n"
        f"{it.get('stem', '')}\n\n"
        f"**答案**：{it.get('answer', '')}\n\n"
        f"**解析**：{it.get('solution', '')}{warn_s}\n"
    )


async def assemble(state: VariantState, config: RunnableConfig) -> VariantState:
    """题组快照：每题带 solution 解析 + check ✓/⚠（状态已定）+ 外显默认配方 + 变更摘要。"""
    items = state.get("items") or []
    facts = _mother_facts(state)
    n_ok = sum(1 for it in items if (it.get("check") or {}).get("badge") == "ok")
    n_warn = len(items) - n_ok

    head = (
        f"## 举一反三 · {len(items)} 道变式（配方：默认 3 = 2 普通 + 1 难）\n\n"
        f"**母题 DNA**：考点「{facts['kp_name']}」· 年级「{facts['grade']}」· 题型「{facts['qtype']}」（硬守恒）\n\n"
        f"**状态**：{n_ok} 道 ✓ 通过自检"
        + (f"，{n_warn} 道 ⚠ 需老师重点看" if n_warn else "")
        + "\n\n旋钮可拨：数量 / 数字 / 场景 / 难度 / 题型(可配比) / 解法。说「这组可以了」即入库。\n\n---\n"
    )
    body = "\n".join(_fmt_item(i + 1, it) for i, it in enumerate(items))
    return {"messages": [AIMessage(content=head + body)]}


# ===========================================================================
# 交互层（设计 §6）：多轮 WAIT → parse 判 5 意图 → 三层漏斗分诊
#   修正 / 编辑(remove·regenerate·add) / 确认 / 答疑 / clarify
# ===========================================================================
PARSE_PROMPT = """你是举一反三 agent 的指令解析器。老师正在看一组已出的变式题（共 {n} 道），下面是他的最新一句话。

母题 DNA（硬守恒，老师不能改这两项，撞它即 clarify 驳回）：
- 主考点: {kp_name}
- 年级: {grade}

老师最新一句话：
{utterance}

把它解析成一个 JSON（只输出 JSON，不要解释）：
{{
  "intent": "修正|编辑|确认|答疑|clarify",   // 5 选 1
  "ops": [                                  // intent=编辑 时的操作列表（其余为空数组）
    {{"action":"remove|regenerate|add", "index": 1, "count": 1, "note":"自由约束/旋钮说明"}}
  ],
  "knobs": {{"count":null, "number":null, "scene":null, "difficulty":null, "qtype":null, "method":null}},
  "comp": "可被旋钮吸收的软约束(超旋钮但 best-effort 能顺的)，没有填 null",
  "extra_constraints": ["其余自由约束句"],
  "mother_correction": {{"grade":null, "kp":null}},  // intent=修正 时老师纠正的年级/考点，否则全 null
  "confidence": 0.0~1.0
}}

判定规则：
- "为什么第N题…/这题怎么解/讲讲" = 答疑（只问不改题）。
- "第N题删掉/不要第N题" = 编辑 remove(index=N)。
- "第N题重出/换一道/改一下第N题" = 编辑 regenerate(index=N)。
- "再来2道/多出几道难的/加道选择题" = 编辑 add(count=N)。
- "这是八年级/考点应该是X/我说错了是…" = 修正（填 mother_correction）。
- "这组可以了/入库/就这些/保存" = 确认。
- 撞守恒(要改主考点为别的考点 / 超出该年级) 或 真说不清 = clarify。
- index 从 1 起；拿不准 index 时 ops 留空、intent 取 clarify。"""


def _items_brief(items: list[dict]) -> str:
    """给 parse / answer 当上下文：每题 index + 题干前 80 字。"""
    lines = []
    for i, it in enumerate(items):
        stem = (it.get("stem") or "").replace("\n", " ")[:80]
        lines.append(f"第{i + 1}题: {stem}")
    return "\n".join(lines)


async def parse_instruction(state: VariantState, config: RunnableConfig) -> VariantState:
    """WAIT 下一句 → 判 5 意图 + 三层漏斗参数。本节点只解析、不改 items（路由后各分支执行）。

    解析结果塞 state['pending']（含 intent/ops/knobs/comp/extra_constraints/mother_correction）。
    """
    utterance = _latest_human_text(state.get("messages", []))
    facts = _mother_facts(state)
    items = state.get("items") or []
    prompt = PARSE_PROMPT.format(
        n=len(items),
        kp_name=facts["kp_name"],
        grade=facts["grade"],
        utterance=utterance or "(空)",
    )
    text = await _ainvoke_text([HumanMessage(content=prompt)])
    parsed = _parse_json(text) or {}
    intent = parsed.get("intent")
    if intent not in {"修正", "编辑", "确认", "答疑", "clarify"}:
        intent = "clarify"
    pending = {
        "intent": intent,
        "ops": parsed.get("ops") or [],
        "knobs": parsed.get("knobs") or {},
        "comp": parsed.get("comp"),
        "extra_constraints": parsed.get("extra_constraints") or [],
        "mother_correction": parsed.get("mother_correction") or {},
        "confidence": parsed.get("confidence"),
        "utterance": utterance,
    }
    return {"pending": pending, "messages": []}


def route_after_parse(
    state: VariantState,
) -> Literal["patch", "dispatch", "answer", "save", "ask_clarify"]:
    """parse 后分诊（设计 §3 mermaid）：修正→patch / 编辑→dispatch / 确认→save / 答疑→answer / clarify。"""
    intent = (state.get("pending") or {}).get("intent")
    if intent == "修正":
        return "patch"
    if intent == "编辑":
        return "dispatch"
    if intent == "确认":
        return "save"
    if intent == "答疑":
        return "answer"
    return "ask_clarify"


def dispatch(state: VariantState) -> Literal["remove", "regenerate", "add", "ask_clarify"]:
    """三层漏斗收口：硬旋钮命中 → remove/regenerate/add；ops 空/说不清 → clarify。"""
    ops = (state.get("pending") or {}).get("ops") or []
    actions = {str(op.get("action")) for op in ops if isinstance(op, dict)}
    # 优先级：remove > regenerate > add（同句多操作时分步走，下游回 ASM 再 WAIT）
    if "remove" in actions:
        return "remove"
    if "regenerate" in actions:
        return "regenerate"
    if "add" in actions:
        return "add"
    return "ask_clarify"


# --- 答疑：只问不改（🔴 物理上 return 不含 items） -----------------------------
ANSWER_PROMPT = """你是数学老师，老师对下面这组变式题的某道有疑问，请耐心解惑（讲思路/为什么这么解）。

题组：
{brief}

各题答案/解析摘要：
{detail}

老师的问题：{question}

直接用人话回答（可含 LaTeX）。只解惑，不要改题、不要重出题。"""


async def answer_question(state: VariantState, config: RunnableConfig) -> VariantState:
    """答疑侧循环（设计 §6）：解老师对解析的疑问 → 回等待。

    🔴 代码层物理不写 state.items：return 的 update 不含 'items' 键，即便 parse 误判进此分支也改不了题。
    """
    items = state.get("items") or []
    question = (state.get("pending") or {}).get("utterance") or _latest_human_text(
        state.get("messages", [])
    )
    detail = "\n".join(
        f"第{i + 1}题 答案:{(it.get('answer') or '')[:60]} 解析:{(it.get('solution') or '')[:120]}"
        for i, it in enumerate(items)
    )
    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ANSWER_PROMPT.format(
                    brief=_items_brief(items), detail=detail, question=question or "(空)"
                )
            )
        ]
    )
    # 🔴 只返回 messages（绝不含 items），并清空 pending
    return {"messages": [AIMessage(content=text or "我没太理解你的疑问，可以再说具体点吗？")], "pending": None}


# --- 编辑·remove：删 + 重编号（删完的题不再过 solve，直接 ASM） -----------------
async def exec_remove(state: VariantState, config: RunnableConfig) -> VariantState:
    items = list(state.get("items") or [])
    ops = (state.get("pending") or {}).get("ops") or []
    drop = set()
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "remove":
            idx = op.get("index")
            try:
                drop.add(int(idx) - 1)  # 1-based → 0-based
            except (TypeError, ValueError):
                pass
    kept = [it for i, it in enumerate(items) if i not in drop]
    return {"items": kept, "pending": None, "messages": []}


# --- 编辑·regenerate：改造指定题 → 过 solve_explain（清 check 触发重判） ----------
async def exec_regenerate(state: VariantState, config: RunnableConfig) -> VariantState:
    """改造某道（按 note 软约束）→ 该题清 check 重入 solve_explain（每题状态须重新定）。"""
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    ops = (state.get("pending") or {}).get("ops") or []
    targets = []
    notes: dict[int, str] = {}
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "regenerate":
            try:
                t = int(op.get("index")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= t < len(items):
                targets.append(t)
                if op.get("note"):
                    notes[t] = str(op.get("note"))

    for t in targets:
        old = items[t]
        regen_text = await _ainvoke_text(
            [
                HumanMessage(
                    content=REGEN_PROMPT.format(
                        kp_name=facts["kp_name"],
                        grade=facts["grade"],
                        stem=(old.get("stem") or "")
                        + (f"\n额外要求：{notes[t]}" if t in notes else ""),
                        level=old.get("level") or "normal",
                        qtype=old.get("qtype") or facts["qtype"],
                        difficulty=old.get("difficulty") or 3,
                        injected_kp=json.dumps(old.get("injected_kp"), ensure_ascii=False),
                    )
                )
            ]
        )
        regen = _parse_json(regen_text)
        if isinstance(regen, dict) and regen.get("stem"):
            # 🔴 新题清 check → 必过 solve_explain 才能进 assemble（不变量）
            items[t] = {
                "stem": regen.get("stem"),
                "answer": regen.get("answer"),
                "solution": regen.get("solution"),
                "qtype": regen.get("qtype") or old.get("qtype") or facts["qtype"],
                "difficulty": regen.get("difficulty") or old.get("difficulty"),
                "level": regen.get("level") or old.get("level") or "normal",
                "injected_kp": regen.get("injected_kp"),
            }
    return {"items": items, "pending": None, "messages": []}


# --- 编辑·add：生成 N 道新题（吸收软约束/旋钮）→ 过 solve_explain ----------------
ADD_PROMPT = """你是浙教版初中数学命题专家。基于母题 DNA，**新增** {n} 道举一反三变式。

母题 DNA：
- 主考点(硬守恒): {kp_name}
- 年级(硬守恒): {grade}
- 题型(默认): {qtype}
- 母题题干: {stem}
- 母题答案/解法骨架: {skeleton}

老师的补充要求（best-effort 吸收，撞守恒的忽略）：{extra}

铁律：每道仍考「{kp_name}」、仍在「{grade}」；只换数字/场景（除非老师明确要改难度/题型）。

只输出 JSON 数组(不要解释)，每元素：
{{"stem":"题干","answer":"标准答案","solution":"完整解析","qtype":"选择/填空/解答","difficulty":1~5,"level":"normal/hard","injected_kp":"相邻kp名或null"}}"""


async def exec_add(state: VariantState, config: RunnableConfig) -> VariantState:
    """补题（设计 §5 三层漏斗②：超旋钮软约束 best-effort 吸收 + 外显）→ 追加 → 过 solve_explain。"""
    facts = _mother_facts(state)
    items = list(state.get("items") or [])
    pending = state.get("pending") or {}
    ops = pending.get("ops") or []
    n = 0
    notes = []
    for op in ops:
        if isinstance(op, dict) and op.get("action") == "add":
            try:
                n += int(op.get("count") or 1)
            except (TypeError, ValueError):
                n += 1
            if op.get("note"):
                notes.append(str(op.get("note")))
    if n <= 0:
        n = 1
    n = min(n, 5)  # 单轮补题上限，防失控

    extra_bits = notes + list(pending.get("extra_constraints") or [])
    if pending.get("comp"):
        extra_bits.append(str(pending["comp"]))
    extra = "；".join(extra_bits) or "无"

    text = await _ainvoke_text(
        [
            HumanMessage(
                content=ADD_PROMPT.format(
                    n=n,
                    kp_name=facts["kp_name"],
                    grade=facts["grade"],
                    qtype=facts["qtype"],
                    stem=facts["stem"],
                    skeleton=facts["skeleton"],
                    extra=extra,
                )
            )
        ]
    )
    data = _parse_json(text)
    if not isinstance(data, list):
        data = (data or {}).get("items") if isinstance(data, dict) else None
    for it in data or []:
        if not isinstance(it, dict):
            continue
        # 🔴 新题不带 check → 下游 solve_explain 必判
        items.append(
            {
                "stem": it.get("stem"),
                "answer": it.get("answer"),
                "solution": it.get("solution"),
                "qtype": it.get("qtype") or facts["qtype"],
                "difficulty": it.get("difficulty"),
                "level": it.get("level") or "normal",
                "injected_kp": it.get("injected_kp") or None,
            }
        )
    return {"items": items, "pending": None, "messages": []}


# --- 修正：patch 母题字段 → 只重算受影响下游（设计 §6 中途修正） ----------------
async def patch(state: VariantState, config: RunnableConfig) -> VariantState:
    """老师纠正年级/考点 → patch analysis；改年级/考点 → 清 items 触发重锚+重造（route_after_patch）。

    粒度 = 步边界：改年级/考点 = 重走 classify→generate（清 items + mother_confirmed）；
    其余（如只是补充场景偏好）= 当软约束，留待下次编辑指令，不在此重造。
    """
    analysis = dict(state.get("analysis") or {})
    corr = (state.get("pending") or {}).get("mother_correction") or {}
    changed = False

    if corr.get("grade"):
        g = dict(analysis.get("grade") or {})
        g["value"] = corr["grade"]
        g["confidence"] = 0.9  # 老师明示 → 高置信
        g.pop("code", None)  # 清旧编码，重锚时再填
        analysis["grade"] = g
        changed = True
    if corr.get("kp"):
        k = dict(analysis.get("kp") or {})
        k["value"] = corr["kp"]
        k["confidence"] = 0.9
        k.pop("anchored", None)  # 清旧锚定，classify 会重锚
        analysis["kp"] = k
        changed = True

    if not changed:
        # 没拿到可 patch 的字段 → 退化为 clarify 回问（不空转）
        return {
            "messages": [
                AIMessage(content="我没听准你要修正什么（年级还是考点？），可以再说一次吗？")
            ],
            "pending": None,
        }

    # 🔴 改了硬锚 → 清 items + mother_confirmed，触发重锚(classify)+重造(generate)
    return {
        "analysis": analysis,
        "items": [],
        "mother_confirmed": False,
        "pending": None,
        "messages": [AIMessage(content="收到修正，我按新的年级/考点重锚并重出这组变式。")],
    }


async def ask_clarify(state: VariantState, config: RunnableConfig) -> VariantState:
    """三层漏斗第③层 / 答非所问兜底：撞守恒 / 说不清 → 回问（设计 §6，不改 items）。"""
    pending = state.get("pending") or {}
    utterance = pending.get("utterance") or ""
    facts = _mother_facts(state)
    body = (
        "我没完全 get 到你的意思（也可能撞到了不能改的硬守恒）。\n\n"
        f"这组变式的硬守恒是：考点「{facts['kp_name']}」+ 年级「{facts['grade']}」——这两项不能改"
        "（要换考点/年级等于换一道母题，请重新贴图）。\n\n"
        "你可以这样说：\n"
        "- 删/重出/再加题（如「第 2 题重出」「再来 2 道难的」）\n"
        "- 拨旋钮（数字 / 场景 / 难度 / 题型配比）\n"
        "- 问解析（如「第 1 题为什么这么解」）\n"
        "- 「这组可以了」入库"
    )
    if utterance:
        body += f"\n\n（你刚说的是：「{utterance}」）"
    return {"messages": [AIMessage(content=body)], "pending": None}


# --- 确认入库（设计 §7）：变式+解析经 RuoYi 写老师个人题库，只写不判 -----------
async def persist_to_bank(state: VariantState, config: RunnableConfig) -> VariantState:
    """④ 入库：老师"这组可以了" → 逐题 POST /teacher/question/create（teacher token 定 owner）。

    🔴 只写不再判（质量门已在 solve_explain 闭合）。入库回执（入库 N 道 + 失败标注）。
    🔴 owner 由后端 LoginHelper 定，body 绝不传 createBy。
    """
    items = state.get("items") or []
    facts = _mother_facts(state)
    if not items:
        return {"messages": [AIMessage(content="当前没有可入库的变式题。先贴图举一反三吧。")]}

    # 🔴 身份透传：book-ui 经 agent_config 透传登录老师 access_token（config.configurable.ruoyi_token）
    # → 入库 owner = 该老师本人（后端 LoginHelper 取 token 身份），而非 .env 服务账号。
    token = (config.get("configurable") or {}).get("ruoyi_token") if config else None
    try:
        receipts = await persist_items(items, facts, token=token)
    except Exception as e:  # noqa: BLE001 — 登录/网络整体失败 → 友好兜底，不崩
        return {
            "messages": [
                AIMessage(content=f"入库时连不上题库服务（book-server :8090 是否在跑？）：{e}")
            ]
        }

    mother = next((r for r in receipts if r.get("role") == "mother"), None)
    var_receipts = [r for r in receipts if r.get("role") != "mother"]
    ok = [r for r in var_receipts if r.get("ok")]
    fail = [r for r in var_receipts if not r.get("ok")]

    lines = [f"## 入库完成 · 变式 {len(items)} 道，成功 {len(ok)} 道"]
    # 母题(原题)入库回执：图母题不在库 → 先落原题挂血缘
    if mother and mother.get("ok"):
        lines.append(f"📌 原题(母题)已一并入库，ID：{mother.get('id')}，变式都挂在它名下（血缘可追）。")
    elif mother and not mother.get("ok"):
        lines.append(f"⚠ 原题入库失败（变式仍已落，血缘暂缺）：{mother.get('error')}")
    if ok:
        ids = [str(r.get("id")) for r in ok if r.get("id") is not None]
        lines.append("已落入你的个人题库（来源标记「举一反三」）。" + (f"变式 ID：{', '.join(ids)}" if ids else ""))
    if fail:
        lines.append(f"\n⚠ {len(fail)} 道变式入库失败：")
        for i, r in enumerate(fail, 1):
            lines.append(f"  {i}. {r.get('error')}")
    lines.append("\n可回平台「我的题库」找题、组卷、导出 PDF。")
    return {"messages": [AIMessage(content="\n".join(lines))]}


# --- 输入边界兜底（设计 §6）：没图/无在途母题/无题组 → 催图 ------------------
async def ask_for_image(state: VariantState, config: RunnableConfig) -> VariantState:
    """route_entry 'ask' 分支落点：首轮无图无母题无题组，催老师贴题图。

    🔴 必须是真节点（不能直连 END）—— 否则首轮没有任何节点产消息，回复为空，
    '没图催' 提示从未触发（PRD-C-009 G15 红的 root cause）。
    """
    return {
        "messages": [
            AIMessage(
                content=(
                    "我还没看到题目图。请先贴一张题目图的 OSS URL，我才能开始举一反三。\n\n"
                    "（贴图后我会读图、锚定年级/考点/题型，再按你要的数量出变式题。）"
                )
            )
        ]
    }


# ---------------------------------------------------------------------------
# 图（StateGraph）
# ---------------------------------------------------------------------------
graph = StateGraph(VariantState)
graph.add_node("analyze", analyze)
graph.add_node("classify", classify)
graph.add_node("clarify", clarify)
graph.add_node("generate", generate)
graph.add_node("solve_explain", solve_explain)
graph.add_node("assemble", assemble)
# 交互层节点（多轮 WAIT 后的下一句）
graph.add_node("parse_instruction", parse_instruction)
graph.add_node("answer_question", answer_question)
graph.add_node("exec_remove", exec_remove)
graph.add_node("exec_regenerate", exec_regenerate)
graph.add_node("exec_add", exec_add)
graph.add_node("patch", patch)
graph.add_node("ask_clarify", ask_clarify)
graph.add_node("persist_to_bank", persist_to_bank)
graph.add_node("ask_for_image", ask_for_image)

graph.set_conditional_entry_point(
    route_entry,
    {
        "analyze": "analyze",
        "generate": "generate",
        "parse": "parse_instruction",
        # 🔴 'ask' 必落真节点（ask_for_image），不能直连 END —— 否则首轮无节点产消息，回复为空
        "ask": "ask_for_image",
    },
)
graph.add_edge("ask_for_image", END)

# analyze：非题目图/读图失败 → 直接 END（已吐友好报错）；成功 → classify
def after_analyze(state: VariantState) -> Literal["classify", "done"]:
    # analyze 友好报错时会塞 messages（且未产 analysis）→ 结束本轮等待
    if not state.get("analysis"):
        return "done"
    return "classify"


graph.add_conditional_edges("analyze", after_analyze, {"classify": "classify", "done": END})
graph.add_conditional_edges(
    "classify", gate_after_classify, {"generate": "generate", "clarify": "clarify"}
)
graph.add_edge("clarify", END)


# generate：裸奔兜底时只吐消息、无 items → 结束；正常 → solve_explain
def after_generate(state: VariantState) -> Literal["solve_explain", "done"]:
    if not state.get("items"):
        return "done"
    return "solve_explain"


graph.add_conditional_edges(
    "generate", after_generate, {"solve_explain": "solve_explain", "done": END}
)
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
        "ask_clarify": "ask_clarify",
    },
)


# 三层漏斗：编辑意图 → 选 remove/regenerate/add（route_after_parse 的 "dispatch" 实由本函数收口）
def route_dispatch(
    state: VariantState,
) -> Literal["exec_remove", "exec_regenerate", "exec_add", "ask_clarify"]:
    target = dispatch(state)
    return {
        "remove": "exec_remove",
        "regenerate": "exec_regenerate",
        "add": "exec_add",
        "ask_clarify": "ask_clarify",
    }[target]


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
        "ask_clarify": "ask_clarify",
    },
)

# remove/regenerate/add 三原语 → 过 solve_explain（凡进 items 的题一律重判）→ assemble
graph.add_edge("exec_remove", "solve_explain")
graph.add_edge("exec_regenerate", "solve_explain")
graph.add_edge("exec_add", "solve_explain")

# 答疑/clarify → END（不改 items，回等待下一句）
graph.add_edge("answer_question", END)
graph.add_edge("ask_clarify", END)
graph.add_edge("persist_to_bank", END)


# patch：改了硬锚（清 items + mother_confirmed=False）→ 重锚重造走 classify；
#        没改（仅回问消息，items 仍在）→ END 等下一句。
def after_patch(state: VariantState) -> Literal["classify", "done"]:
    if state.get("mother_confirmed") is False and not state.get("items"):
        return "classify"
    return "done"


graph.add_conditional_edges("patch", after_patch, {"classify": "classify", "done": END})

# 🔴 不在此 compile checkpointer：service lifespan 注入 saver（按 thread_id 持久 state）
variant = graph.compile()
variant.name = "variant"
