# -*- coding: utf-8 -*-
"""PRD-C-015 批2·W1' 模型锚定（双轴指纹的「怎么解」轴）。

与知识点锚定同构的第二个池（闭池 / 池内选 / 禁造词 / 允许空但**模型维永不为空**）：
  ① 反查：按母题锚到的 kp 叶子 code，从 `biz_solution_model_kp` 前缀命中候选模型（≤8，按 sort）。
  ② 确认：gpt-5.4-mini「先解题再选」在候选子集里确认 ≤3（禁造词；候选空/全不确认 → M00 保底）。
  ③ 处置（§3.2 穷举）：
       - 候选内命中 1~3      → 正常，写 DNA.models。
       - 候选空 / 全不确认   → M00 概念直用兜底（模型维非空拍板）。
       - LLM 给池外模型名    → 不入正式维，原文落 model_overflow + 待命名池 + 卡面 ⚠。
       - 反查库/表不可用     → 降级 M00 + ⚠（闸门必有降级路径，C-010 纪律）。

🔴 架构铁律（ARCHITECTURE.md §2）：
  - 反查 = **纯只读 ETL**（与 variant_support.anchor_subject 同精神，是架构允许的唯一直连例外）；
    绝不写业务数据；写 models 进 biz_question_ai 走 book-server HTTP（透传，见 variant_support._apply_labels）。
  - 不存 prompt / 不让 LLM 自评作判决——确认是「解法标注」不是「对错判据」（判决只读 sympy 不变）。

人读 SSOT = codeplace-C/claude-code-sign/26-解题模型词库-初版.md（改词库先改文档再出 V 号）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql

from core import settings

# 🔴 保底模型（V908 INSERT 进 biz_solution_model；不参与反查，是代码层兜底值）。
M00_ID = "M00"
M00_NAME = "概念直用"
M00: dict[str, str] = {"id": M00_ID, "name": M00_NAME}

# 确认上限（§1①：池内确认 ≤3）。
MODELS_MAX = 3
# 反查候选上限（26号 §0.5：LIMIT 8）。
CANDIDATE_LIMIT = 8


def _db_kwargs() -> dict:
    """复用 variant_support._db_kwargs 的同一只读连接配置（dev 库 miskt_data2 @ :3307）。"""
    pwd = settings.VARIANT_DB_PASSWORD
    return dict(
        host=settings.VARIANT_DB_HOST,
        port=settings.VARIANT_DB_PORT,
        user=settings.VARIANT_DB_USER,
        password=pwd.get_secret_value() if pwd else "",
        database=settings.VARIANT_DB_NAME,
        charset="utf8mb4",
    )


# 待命名池落点（§10.3）：toolkit 本地 JSONL，实战递增；judge 录入坑位预留 status="pending"。
_CANDIDATES_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model_candidates.jsonl"


def record_overflow_candidate(
    name: str, mother_models: list[str], *, question_ref: str | None = None
) -> bool:
    """把 LLM 给出的池外模型名落待命名池（§10.3）。返回是否落盘成功。

    字段 = {name, trigger_draft, question_ref, mother_models, status, ts}。
    🔴 status 固定 "pending"（judge 自动录入=留坑不上，消费者本期不存在）。
    🔴 落盘失败返回 False（由上层降级报错，G4：写失败不得静默成成功）。
    """
    name = str(name or "").strip()
    if not name:
        return False
    rec = {
        "name": name,
        "trigger_draft": "",
        "question_ref": question_ref or "",
        "mother_models": list(mother_models or []),
        "status": "pending",
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CANDIDATES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def _leaf_codes(dna: dict[str, Any] | None) -> list[str]:
    """从母题 DNA 取参与反查的 kp 叶子 code（主 kp + 副 kp，去重去空）。

    反查按主+副 kp 一并圈候选（H2 处置：守恒集合按"确认集∪反查候选池"放宽，防欠选）。
    """
    dna = dna or {}
    codes: list[str] = []
    mk = dna.get("main_kp") or {}
    if isinstance(mk, dict):
        mid = str(mk.get("id") or mk.get("code") or "").strip()
        if mid:
            codes.append(mid)
    for s in dna.get("secondary_kps") or []:
        if isinstance(s, dict):
            sid = str(s.get("id") or s.get("code") or "").strip()
        else:
            sid = str(s or "").strip()
        if sid:
            codes.append(sid)
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def lookup_candidates(
    leaf_codes: list[str], *, limit: int = CANDIDATE_LIMIT
) -> list[dict[str, Any]]:
    """反查（纯只读 ETL）：kp 叶子 code → 候选模型 [{id,name,trigger_feature,action_conclusion,sort}]。

    26号 §0.5 原样 SQL：`bind_type IN ('primary','native')` + `:leaf LIKE CONCAT(subject_id,'%')`
    + `ORDER BY sort` + `LIMIT`。多个 leaf_code 一并圈（DISTINCT 去重），合并后按 sort 取前 limit。

    🔴 前缀边界：`leaf LIKE subject_id%`（不是 subject_id LIKE leaf%）—— 绑定挂在节点(L3)上，
       叶子 code 是其子孙，故用「叶子 code 以 subject_id 开头」命中。兄弟节点 code 前缀不同 → 不误命中。
    🔴 取数失败（库未起/表不存在/网络）→ 抛 pymysql 异常，由上层 anchor_models 兜成 M00+⚠（不卡死）。
    """
    leaf_codes = [str(c).strip() for c in (leaf_codes or []) if str(c).strip()]
    if not leaf_codes:
        return []
    conn = pymysql.connect(**_db_kwargs())
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        rows_by_id: dict[str, dict[str, Any]] = {}
        for leaf in leaf_codes:
            cur.execute(
                """
                SELECT DISTINCT m.id, m.name, m.trigger_feature, m.action_conclusion, m.sort
                FROM biz_solution_model m
                JOIN biz_solution_model_kp k ON k.model_id = m.id
                WHERE k.bind_type IN ('primary','native')
                  AND m.status = '0'
                  AND %s LIKE CONCAT(k.subject_id, '%%')
                ORDER BY m.sort
                """,
                (leaf,),
            )
            for r in cur.fetchall():
                mid = str(r.get("id") or "").strip()
                if mid and mid not in rows_by_id:
                    rows_by_id[mid] = {
                        "id": mid,
                        "name": str(r.get("name") or "").strip(),
                        "trigger_feature": str(r.get("trigger_feature") or "").strip(),
                        "action_conclusion": str(r.get("action_conclusion") or "").strip(),
                        "sort": r.get("sort"),
                    }
        out = sorted(
            rows_by_id.values(),
            key=lambda x: (x.get("sort") if x.get("sort") is not None else 1 << 30),
        )
        return out[:limit]
    finally:
        conn.close()


# 确认 prompt（H2 定档 gpt-5.4-mini + 「先解题再选」风格：先想这题怎么解，再从候选里挑命中的）。
CONFIRM_PROMPT = """你是初中数学解题模型确认器。下面给你一道题和一份**候选解题模型**清单。

任务（先解题再选，禁造词）：
1. 先在心里把这道题解一遍，想清楚「这道题真正用到了哪些解题套路/模型」。
2. 再从【候选模型】里**只挑你第 1 步真正用到的**（最多 {max_n} 个）。
3. 🔴 只能从候选 id 里选，**严禁**编造候选外的 id/名称；这道题用不到候选里任何一个就返回空数组。
4. 基础题/无组合技巧的概念直用题：返回空数组即可（系统会兜底为「概念直用」）。

【题目】
{stem}

【标准答案/解析（辅助你判断用了哪些模型，可能为空）】
{answer}

【候选模型】（id｜名称｜触发特征 → 动作·结论）
{candidates}

只输出 JSON（不要解释、不要 markdown 围栏）：
{{"models": ["命中的候选id", ...]}}
"""


def _candidates_text(candidates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for c in candidates:
        trig = c.get("trigger_feature") or ""
        act = c.get("action_conclusion") or ""
        lines.append(f"{c.get('id')}｜{c.get('name')}｜{trig} → {act}")
    return "\n".join(lines) or "（候选为空）"


def _parse_models_json(text: str | None) -> list[str] | None:
    """解析 LLM 返回的 {"models":[id...]}；非 JSON/无 models 键 → None（上层兜底 M00）。"""
    if not text:
        return None
    s = text.strip()
    # 剥 markdown 围栏
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except Exception:
        # 容错：从文本里捞第一个 {...}
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    raw = obj.get("models")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return None
    return [str(x).strip() for x in raw if str(x).strip()]


async def confirm_models(
    stem: str,
    answer: str,
    candidates: list[dict[str, Any]],
    *,
    invoke: Any,
    model: str | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """池内确认（LLM 在候选子集里选 ≤3）。返回 (confirmed, overflow)。

    confirmed = [{id,name}]（id ∈ candidates；去重；截 ≤MODELS_MAX）。
    overflow  = [str]（LLM 给出的**池外**名/id，禁造词处置：不入正式维，落待命名池+⚠）。
    候选空 → 不调 LLM，直接 ([], [])（上层兜 M00）。
    LLM 失败/解析失败 → ([], [])（上层兜 M00，不卡死）。
    """
    from langchain_core.messages import HumanMessage  # 局部 import 避循环依赖

    candidates = candidates or []
    if not candidates:
        return [], []
    by_id = {str(c.get("id")): c for c in candidates}
    by_name = {str(c.get("name")): c for c in candidates}
    prompt = CONFIRM_PROMPT.format(
        max_n=MODELS_MAX,
        stem=stem or "（题面缺失）",
        answer=(answer or "（无）")[:1500],
        candidates=_candidates_text(candidates),
    )
    try:
        text = await invoke([HumanMessage(content=prompt)], model=model)
    except Exception:
        return [], []
    picked = _parse_models_json(text)
    if picked is None:
        return [], []
    confirmed: list[dict[str, str]] = []
    overflow: list[str] = []
    seen: set[str] = set()
    for p in picked:
        c = by_id.get(p) or by_name.get(p)
        if c is not None:
            cid = str(c.get("id"))
            if cid not in seen:
                seen.add(cid)
                confirmed.append({"id": cid, "name": str(c.get("name") or "")})
        else:
            # 池外名/id：禁造词 → 不入正式维，原文留 overflow（待命名池 + ⚠）
            if p not in overflow:
                overflow.append(p)
    return confirmed[:MODELS_MAX], overflow


async def anchor_models(
    dna: dict[str, Any] | None,
    *,
    stem: str,
    answer: str = "",
    invoke: Any,
    model: str | None = None,
    record_overflow: Any = None,
) -> dict[str, Any]:
    """W1' 模型锚定主入口（§3.2 处置穷举）。返回 {models, model_overflow, model_warn, model_flag}。

      models         : [{id,name}]，1~3 项，**非空**（无命中保底 M00）。
      model_overflow : [str]，LLM 给出的池外名（仅 ⚠/待命名池用，不入正式维，可空）。
      model_warn     : bool，是否卡面 ⚠（池外命中 / 反查库故障降级）。
      model_flag     : str|None，处置标记（"m00_fallback"/"overflow"/"lookup_unavailable"/None）。

    record_overflow(name, mother_models)：把池外名落待命名池的回调（None=不落，仅返回 overflow）。
    """
    leaf_codes = _leaf_codes(dna)

    # 反查（纯只读 ETL）；库/表不可用 → 降级 M00 + ⚠（C-010 闸门必有降级路径）。
    try:
        candidates = lookup_candidates(leaf_codes)
    except Exception:
        return {
            "models": [dict(M00)],
            "model_overflow": [],
            "model_warn": True,
            "model_flag": "lookup_unavailable",
        }

    confirmed, overflow = await confirm_models(
        stem, answer, candidates, invoke=invoke, model=model
    )

    # 池外名落待命名池（含题目指针由上层带；这里只回调名 + 母题 models 锚）。
    if overflow and record_overflow is not None:
        mother_models = [c["id"] for c in confirmed] or [M00_ID]
        for name in overflow:
            try:
                record_overflow(name, mother_models)
            except Exception:
                pass  # 待命名池落盘失败不拖垮主流程（G4：写失败由上层降级，不静默成成功）

    if not confirmed:
        # 候选空 / 全不确认 → M00 兜底（模型维非空拍板）。有池外名 → 一并 ⚠。
        return {
            "models": [dict(M00)],
            "model_overflow": overflow,
            "model_warn": bool(overflow),
            "model_flag": "overflow" if overflow else "m00_fallback",
        }

    return {
        "models": confirmed,
        "model_overflow": overflow,
        "model_warn": bool(overflow),
        "model_flag": "overflow" if overflow else None,
    }


# ===========================================================================
# 🔴 PRD-C-015 批3·W2' 难题注卡（模型卡片注入 GENERATE/REGEN prompt）
# 数据源 = 词库表 biz_solution_model（trigger_feature + action_conclusion 整行，G3 逐字一致）。
# 注卡条件矩阵（§3.2）：难度≥3 且命中**非 M00** 模型 → 注卡；难度<3 或仅 M00 → 不注。
# 注卡是 prompt 引导（生成侧），不是判决侧——绝不混进 pass/fail（铁律）。
# ===========================================================================

# 注卡难度阈值（§3.1 拍板，默认）：母题难度 ≥ 此值才注卡。
NOTE_CARD_DIFFICULTY_MIN = 3


def fetch_model_cards(model_ids: list[str]) -> dict[str, dict[str, str]]:
    """按 model id 取「模型卡片」整行（纯只读 ETL）。返回 {id: {id,name,trigger_feature,action_conclusion}}。

    🔴 与反查同一只读连接（架构允许的唯一直连例外）；M00 等无表条目/库故障 → 该 id 缺省（上层容缺）。
    模型卡片文本 = 词库表原行（G3：注入 prompt 的文本与词库表逐字一致），不在代码里改写。
    🔴 取数失败（库未起/表不存在）→ 抛 pymysql 异常，由上层（变式注卡）兜成「不注卡」降级（不卡死出题）。
    """
    ids = [str(i).strip() for i in (model_ids or []) if str(i).strip()]
    # M00 是代码兜底值（V908 有行但 trigger/action 为「保底」语义、不注卡），无需查表注卡。
    ids = [i for i in ids if i != M00_ID]
    if not ids:
        return {}
    conn = pymysql.connect(**_db_kwargs())
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        placeholders = ",".join(["%s"] * len(ids))
        cur.execute(
            f"""
            SELECT id, name, trigger_feature, action_conclusion
            FROM biz_solution_model
            WHERE id IN ({placeholders})
            """,
            tuple(ids),
        )
        out: dict[str, dict[str, str]] = {}
        for r in cur.fetchall():
            mid = str(r.get("id") or "").strip()
            if mid:
                out[mid] = {
                    "id": mid,
                    "name": str(r.get("name") or "").strip(),
                    "trigger_feature": str(r.get("trigger_feature") or "").strip(),
                    "action_conclusion": str(r.get("action_conclusion") or "").strip(),
                }
        return out
    finally:
        conn.close()


# 反退化 / 反表皮缩放约束（预研 H1 实测 B 组两副作用：换皮变薄 / 最优点端点退化）。
# 🔴 prompt 引导（生成侧），与 ⑦ 反退化代码闸（判决侧·纯代数）正交并存——这段是「让 LLM 别那样出」，
#    代码闸是「真出了那样就 REGEN」。两条都要，不互相替代。
ANTI_DEGEN_CLAUSE = (
    "🔴 反退化 / 反表皮缩放硬约束（违反 = 不合格变式）：\n"
    "① **不许换皮变薄**：变式必须保住上面模型卡片的**完整解法机制**（触发特征→动作→结论整条链），"
    "不许只换个数字/场景却把母题的关键构造步（隐圆/对称/旋转/相似等）简化掉或绕开——"
    "「换皮不换骨」，骨架最难步基因必须同类保留。\n"
    "② **不许最优点退化到区间端点**：若本题是最值/动点构型（动点在某线段/区间/弧上变化），"
    "设计数字时务必让**最优解（最大/最小值的取得点）落在动点定义区间的内部驻点**，"
    "**严禁**让最优点恰好落在区间端点（如动点 P 与定点 B 重合、PB=0、k 取临界值使机制失效），"
    "那是「答案碰巧对、解法机制其实失效」的退化废题，必须避免。\n"
    "③ 数字全换且设计成解恰好整洁；解不整洁宁可再换一组数。"
)

# 反退化代码闸（⑦）的载荷契约：仅最值/动点构型才产出 degen_payload，供程序纯代数判退化
# （最优点是否落动点区间端点）。非最值/动点题不产出此字段（缺省 = 不判退化）。
# 🔴 与 verify_payload（答案验算）正交：那个验答案对不对，这个验构型退不退化。
DEGEN_PAYLOAD_CONTRACT = (
    "🔴 反退化载荷（仅当本题是**最值 / 动点构型**时额外产出，供程序纯代数判「最优点是否落区间端点」）：\n"
    "在 JSON 里加一个 `degen_payload` 字段（不是最值/动点题就**不要**加这个字段）：\n"
    '  {"kind":"endpoint_extremum","objective":"<目标函数 f(t)，单变量，sympy 可解析纯 ASCII>",'
    '"var":"t","interval":["<动点区间下界>","<上界>"],"sense":"min"|"max"}\n'
    "其中 objective = 要最小化/最大化的目标量随动点参数 t 的表达式，interval = 动点 t 的定义区间，"
    "sense = 求最小(min)还是最大(max)。建模不出来就不要加 degen_payload。"
)


def build_degen_contract() -> str:
    """⑦ 反退化载荷契约段（纯函数·供注卡块拼接）。"""
    return DEGEN_PAYLOAD_CONTRACT


def build_model_cards_clause(
    cards: list[dict[str, str]], *, with_anti_degen: bool = True
) -> str:
    """把模型卡片拼成可注入 GENERATE/REGEN prompt 的「按模型卡片出变式」段（纯函数·可单测）。

    cards = [{id,name,trigger_feature,action_conclusion}]（非空、已过滤 M00）。空 → 返回 ""（上层不注）。
    🔴 卡片文本逐字取词库表（G3），不改写；附反退化/反表皮缩放约束（with_anti_degen）。
    """
    cards = [c for c in (cards or []) if isinstance(c, dict) and c.get("name")]
    if not cards:
        return ""
    lines: list[str] = [
        "🔴 解题模型卡片（这道母题命中以下解题模型，变式必须照模型卡片出——保住母题解法基因，「换皮不换骨」）："
    ]
    for c in cards:
        trig = c.get("trigger_feature") or ""
        act = c.get("action_conclusion") or ""
        lines.append(f"- 模型「{c.get('name')}」：触发特征 = {trig}；动作 → 结论 = {act}")
    clause = "\n".join(lines)
    if with_anti_degen:
        clause = clause + "\n\n" + ANTI_DEGEN_CLAUSE + "\n\n" + DEGEN_PAYLOAD_CONTRACT
    return clause


# ===========================================================================
# 🔴 PRD-C-015 批3·W3' 模型守恒软警（纯代码集合判·零 LLM·不打回）
# 变式 models ⊆ 母题 models ∪ {M00} ∪ 反查候选池 → 守恒；越界 → ⚠ + 待命名池（不打回、不阻断入库）。
# 🔴 守恒集合按「确认集 ∪ 反查候选池」放宽（H2 处置：防 mini 欠选伴生模型致软警误报）。
# ===========================================================================
def model_conservation_warn(
    variant_model_ids: list[str],
    mother_model_ids: list[str],
    *,
    candidate_pool_ids: list[str] | None = None,
) -> dict[str, Any]:
    """W3' 守恒软警（纯函数·零 LLM·可单测）。返回 {warn:bool, out_of_set:[id...]}。

    守恒集合 = 母题 models ∪ {M00} ∪ 反查候选池（H2 放宽，防欠选误报）。
    变式 models ⊄ 守恒集合 → warn=True + out_of_set 列越界 id（上层落待命名池 + ⚠，**不打回**）。
    🔴 变式无 models（未单独锚定，继承母题）→ 视同守恒（warn=False）；绝不因「没标」误报。
    """
    vids = [str(i).strip() for i in (variant_model_ids or []) if str(i).strip()]
    if not vids:
        return {"warn": False, "out_of_set": []}
    allowed: set[str] = {M00_ID}
    allowed.update(str(i).strip() for i in (mother_model_ids or []) if str(i).strip())
    allowed.update(str(i).strip() for i in (candidate_pool_ids or []) if str(i).strip())
    out_of_set = [i for i in vids if i not in allowed]
    # 保序去重
    seen: set[str] = set()
    oos: list[str] = []
    for i in out_of_set:
        if i not in seen:
            seen.add(i)
            oos.append(i)
    return {"warn": bool(oos), "out_of_set": oos}
