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
# 反查候选上限（26号 §0.5：LIMIT 8）—— 按章收窄路径用。
CANDIDATE_LIMIT = 8
# 🔴 PRD-C-105 B：按年级全量召回上限。单年级实测 distinct 模型 ~10-15（七上=15），放宽到 20
#   覆盖单年级全量（D2：全量塞 prompt 完胜向量，不上 RAG）。
GRADE_CANDIDATE_LIMIT = 20
# 🔴 年级根前缀长度（biz_subject level1 年级根 id = 3 位，如 100=七上 / 200=八下；
#   章 id = 6 位 100001…、叶子 = 9/12 位 100001002…，全部以年级根 3 位为前缀）。
#   反查的 biz_solution_model_kp.subject_id 均挂在该年级子树下 → 用 3 位根前缀命中本年级全量绑定。
GRADE_ROOT_PREFIX_LEN = 3


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
                SELECT DISTINCT m.id, m.name, m.trigger_feature, m.action_conclusion,
                       m.sort, m.model_kind, m.difficulty_tier, m.freq_band
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
                    # 🔴 PRD-C-103 WS1：带上控制面板列 difficulty_tier/freq_band（V911 已 apply、已填）。
                    #   这是「改表→反控变式难度」的电线根 —— 旧 SQL 漏选这两列，导致锚定模型不带难度信号、
                    #   grade_observed 永远退回基础阶/低频，改表完全不生效（C-103 ① 核实的真因）。
                    rows_by_id[mid] = {
                        "id": mid,
                        "name": str(r.get("name") or "").strip(),
                        "trigger_feature": str(r.get("trigger_feature") or "").strip(),
                        "action_conclusion": str(r.get("action_conclusion") or "").strip(),
                        "sort": r.get("sort"),
                        "model_kind": str(r.get("model_kind") or "").strip() or None,
                        "tier_int": r.get("difficulty_tier"),   # 1基础/2高阶（表真值，可空）
                        "freq_int": r.get("freq_band"),         # 1低频/2高频（表真值，可空）
                    }
        out = sorted(
            rows_by_id.values(),
            key=lambda x: (x.get("sort") if x.get("sort") is not None else 1 << 30),
        )
        return out[:limit]
    finally:
        conn.close()


def _grade_prefix_from_leaf_codes(leaf_codes: list[str]) -> str | None:
    """🔴 PRD-C-105 B：从已锚定的 kp 叶子 code 取「年级根 3 位前缀」（如 100=七上 / 200=八下）。

    为什么从叶子 code 取、不直接用上层传的 grade_code：上层 grade_code 有两条来源不同源——
      ① 预设章 `preset_chapter_id[:4]`（如七上章 100001 → '1000'）= DB 对齐前缀，可命中；
      ② opus 判年级走 `_grade_to_code`（七上 → **'3071'**）= **历史映射，与 live 库 100/200 根不一致**，
         拿它当前缀 LIKE 永远命中 0（这正是「模型没生效」的隐性二号坑）。
    锚定后的叶子 code（main_kp.id / secondary_kps.id）一定是 live 库真 id（100…），最可靠，
    故年级前缀以「叶子 code 的 3 位根」为准，grade_code 仅作旁证/override（见 _resolve_grade_prefix）。
    多个叶子若跨年级根（异常）→ 取第一个主 kp 的根（_leaf_codes 已把 main_kp 放首位）。
    无可用叶子 code → None（上层退回 grade_code override 或按章收窄兜底）。
    """
    for c in (leaf_codes or []):
        s = str(c or "").strip()
        if len(s) >= GRADE_ROOT_PREFIX_LEN and s[:GRADE_ROOT_PREFIX_LEN].isdigit():
            return s[:GRADE_ROOT_PREFIX_LEN]
    return None


def _resolve_grade_prefix(
    leaf_codes: list[str], grade_code: str | None
) -> str | None:
    """定本次年级全量召回用的「年级前缀」。优先叶子 code 的 3 位根（DB 对齐、最可靠）；
    仅当叶子取不到根、且 grade_code 看着是 live 库前缀（数字、与年级根同前 3 位）时才用 grade_code。

    🔴 grade_code='3071' 这类历史映射（非 100/200 根）= 不可信前缀，主动丢弃退 None（宁可让上层
    按章兜底，也不拿错前缀 LIKE 命中 0 制造「假空候选→M00」）。grade_code='1000' / '100001…'
    这类 DB 对齐前缀 → 取其 3 位根（'100'）放宽到整年级。
    """
    pfx = _grade_prefix_from_leaf_codes(leaf_codes)
    if pfx:
        return pfx
    gc = str(grade_code or "").strip()
    # grade_code 只有在「纯数字且前 3 位是合法年级根」时才采信。live 年级根 = 1xx/2xx…（3 位）。
    if gc.isdigit() and len(gc) >= GRADE_ROOT_PREFIX_LEN:
        return gc[:GRADE_ROOT_PREFIX_LEN]
    return None


def lookup_candidates_by_grade(
    grade_prefix: str, *, limit: int = GRADE_CANDIDATE_LIMIT
) -> list[dict[str, Any]]:
    """🔴 PRD-C-105 B·按年级全量反查（纯只读 ETL）：年级根前缀 → 该年级**所有**解题模型候选。

    与 lookup_candidates 同一 SELECT（含 id/name/trigger_feature/action_conclusion/sort/model_kind/
    difficulty_tier/freq_band、`bind_type IN('primary','native')`、`status='0'`、`ORDER BY sort`），
    **唯一区别 = WHERE 换成年级前缀**：`k.subject_id LIKE CONCAT(%s,'%%')`（grade_prefix 作前缀）。

    治什么（D1/AC-B1）：旧 lookup_candidates 按**章** `leaf LIKE subject_id%` 反查——跨章题（如折线
    数轴折叠题被标进方程章）拿方程章叶子找不到挂在数轴章的数轴大招 → 候选空 → M00 短路 → 模型没绑。
    年级全量召回拿该年级**所有**大招（七上实测 distinct 15 个）喂 CONFIRM「先解题再挑真正用到的」，
    跨章大招天然在池内 → 命中真实大招（DZ01 数轴折叠 / DZ08 距离 / DZ09 相遇）。

    🔴 取数失败（库未起/表不存在/网络）→ 抛 pymysql 异常，由上层 anchor_models 兜成 M00+⚠（不卡死）。
    """
    grade_prefix = str(grade_prefix or "").strip()
    if not grade_prefix:
        return []
    conn = pymysql.connect(**_db_kwargs())
    try:
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(
            """
            SELECT DISTINCT m.id, m.name, m.trigger_feature, m.action_conclusion,
                   m.sort, m.model_kind, m.difficulty_tier, m.freq_band
            FROM biz_solution_model m
            JOIN biz_solution_model_kp k ON k.model_id = m.id
            WHERE k.bind_type IN ('primary','native')
              AND m.status = '0'
              AND k.subject_id LIKE CONCAT(%s, '%%')
            ORDER BY m.sort
            """,
            (grade_prefix,),
        )
        rows_by_id: dict[str, dict[str, Any]] = {}
        for r in cur.fetchall():
            mid = str(r.get("id") or "").strip()
            if mid and mid not in rows_by_id:
                # 控制面板列 difficulty_tier/freq_band 随候选透传（WS1 电线根，同 lookup_candidates）。
                rows_by_id[mid] = {
                    "id": mid,
                    "name": str(r.get("name") or "").strip(),
                    "trigger_feature": str(r.get("trigger_feature") or "").strip(),
                    "action_conclusion": str(r.get("action_conclusion") or "").strip(),
                    "sort": r.get("sort"),
                    "model_kind": str(r.get("model_kind") or "").strip() or None,
                    "tier_int": r.get("difficulty_tier"),
                    "freq_int": r.get("freq_band"),
                }
        out = sorted(
            rows_by_id.values(),
            key=lambda x: (x.get("sort") if x.get("sort") is not None else 1 << 30),
        )
        return out[:limit]
    finally:
        conn.close()


# 确认 prompt（H2 定档 gpt-5.4-mini + 「先解题再选」风格：先想这题怎么解，再从候选里挑命中的）。
# 🔴 PRD-C-103 WS2·AC5（2026-06-26）：临时模型不跳过。候选里没有但确有【可跨题复用的解题套路】→
#   不再只丢个名字（旧 overflow 落 JSONL 后被丢，难度跑不出来），而是产「临时模型」对象（带 LLM 提的
#   临时 tier/freq），让它**参与 grade_observed 判档**。临时 tier 是待审草案，最终 authority = 人维护的
#   表（不违「难度不靠 LLM 自评」铁律：表才是 pass/fail 权威，临时 tier 只是没入库时的占位估计）。
CONFIRM_PROMPT = """你是初中数学解题模型确认器。下面给你一道题和一份**候选解题模型**清单。
{scope_note}
任务（先解题再选）：
1. 先在心里把这道题解一遍，想清楚「这道题真正用到了哪些解题套路/模型」。
2. 再从【候选模型】里**挑你第 1 步真正用到的**（最多 {max_n} 个），填进 hits（只填候选 id）。
3. 🔴 若这题真正用到了一个【候选里没有、但可跨题复用的解题套路】（不是基础运算技巧），
   不要硬塞进候选，也不要丢弃——把它作为**临时模型**填进 newModels（给 name + 触发特征 + 招式结论
   + 你判的难度阶 tier(基础/高阶) + 稀有度 freq(通法/一次性)）。临时模型最多 1 个，只在真有可复用
   套路时给；普通题不要给。
4. 🔴 基础运算技巧 ≠ 解题模型，**别塞 newModels**：提取公因数/凑整/分配律/通分/约分/去括号/移项/
   合并同类项/直接代入——这些是基本运算，不是模型。靠这些就能做的题：hits 和 newModels 都留空。
5. 概念直用/送分题：hits 和 newModels 都留空数组（系统兜底为「概念直用」）。

【题目】
{stem}

【标准答案/解析（辅助你判断用了哪些模型，可能为空）】
{answer}

【候选模型】（id｜名称｜触发特征 → 动作·结论）
{candidates}

只输出 JSON（不要解释、不要 markdown 围栏）：
{{"hits": ["命中的候选id", ...],
  "newModels": [{{"name": "套路名", "triggerFeature": "触发特征", "action": "招式结论",
                  "tier": "基础或高阶", "freq": "通法或一次性"}}]}}
"""


# 🔴 PRD-C-105 B·年级全量护栏（D1：宁缺毋滥）。年级全量分支下，候选 = 本年级**所有**大招（~10-15），
#   多数与本题无关 → 必须压住 mini 别乱勾一堆。按章收窄路径候选已贴题，不加此护栏（scope_note=""）。
GRADE_SCOPE_NOTE = (
    "\n🔴 注意：下面的【候选模型】是**本年级的全部解题模型**，"
    "其中**多数与本题无关**。只勾你第 1 步解题**确实动用**的那几个，"
    "拿不准、用不到的一律**不要勾**（hits 留空系统会兜底为「概念直用」），"
    "宁可少勾、漏勾，也绝不为了凑数把无关大招勾进来。\n"
)


def _candidates_text(candidates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for c in candidates:
        trig = c.get("trigger_feature") or ""
        act = c.get("action_conclusion") or ""
        lines.append(f"{c.get('id')}｜{c.get('name')}｜{trig} → {act}")
    return "\n".join(lines) or "（候选为空）"


def _loads_obj(text: str | None) -> dict | None:
    """剥 markdown 围栏 + JSON 容错 → dict（非 dict/失败 → None）。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _parse_confirm_json(text: str | None) -> tuple[list[str], list[dict[str, Any]]] | None:
    """解析确认 LLM 返回 → (hits[id...], newModels[临时模型对象...])；失败 → None（上层兜 M00）。

    🔴 WS2·AC5：兼容旧 schema {"models":[id...]}（无 newModels），新 schema {"hits":[...],"newModels":[...]}。
    newModels 每项归一为 {name, triggerFeature, action, tier_text(基础/高阶), freq_text(通法/一次性)}。
    """
    obj = _loads_obj(text)
    if obj is None:
        return None
    # hits：新 schema "hits" 优先，回退旧 schema "models"
    raw_hits = obj.get("hits")
    if raw_hits is None:
        raw_hits = obj.get("models")
    if raw_hits is None or not isinstance(raw_hits, list):
        # 没有 hits 键也没有 models 键 → 视为解析失败（上层兜 M00）
        if obj.get("newModels") is None:
            return None
        raw_hits = []
    hits = [str(x).strip() for x in raw_hits if str(x).strip()]
    # newModels：临时模型
    new_models: list[dict[str, Any]] = []
    for nm in (obj.get("newModels") or []):
        if not isinstance(nm, dict):
            continue
        name = str(nm.get("name") or "").strip()
        if not name:
            continue
        tier = str(nm.get("tier") or "").strip()
        freq = str(nm.get("freq") or nm.get("freqHint") or "").strip()
        new_models.append({
            "name": name,
            "triggerFeature": str(nm.get("triggerFeature") or "").strip(),
            "action": str(nm.get("action") or "").strip(),
            "tier_text": tier if tier in ("基础", "高阶") else "基础",
            "freq_text": freq if freq in ("通法", "一次性") else "一次性",
        })
    return hits, new_models


# 临时模型 tier/freq 文本 → 整数（与 core.difficulty 同口径，避免循环 import 故本地常量）。
_TIER_TEXT2INT = {"基础": 1, "高阶": 2}
_FREQ_TEXT2INT = {"一次性": 1, "通法": 2}


# ===========================================================================
# 🔴 PRD-C-106 B1·阶段一收口：带料解题 + 消重复解题 + 诚实三态（去 M00 兜底）
#   ① 带料解题：把「该年级全量模型名单（名称 + 触发特征）」做成可注入解题打标 prompt 的工具箱
#      文本（toolbox_for_grade / build_toolbox_clause），让 opus 带着工具箱解题、从中选真正用到的。
#   ② 消重复解题：opus 已从工具箱选了模型名（dna.modelCandidates）→ anchor_models_from_names 把名
#      纯代码映射成 M-id + tier/freq（不再调 confirm_models 第二次 LLM 解题）。
#   ③ 诚实三态：有考模型 → 返回真模型；真无 → models:[] + model_flag="no_model"（不再 M00 兜底）。
# ===========================================================================
# 模型展示无模型时的明示标记（替代旧 M00 兜底，FE/母题卡据此出「无考模型」态）。
NO_MODEL_FLAG = "no_model"


def toolbox_for_grade(grade_code: str | None) -> list[dict[str, Any]]:
    """🔴 B1①·带料解题工具箱：年级根前缀 → 该年级全量解题模型候选（纯只读 ETL）。

    复用 lookup_candidates_by_grade（同一 SELECT，带 tier_int/freq_int）。grade_code 归一成
    3 位年级根前缀（_resolve_grade_prefix 同口径：纯数字且前 3 位合法才采信）。取数失败/前缀不可用
    → 返回 []（上层注入空工具箱 → 降级标记，opus 仍可裸解；铁律④闸门必有降级路径，不卡死）。
    """
    pfx = _resolve_grade_prefix([], grade_code)
    if not pfx:
        return []
    try:
        return lookup_candidates_by_grade(pfx)
    except Exception:  # noqa: BLE001 — 库故障 → 空工具箱（上层降级裸解，不卡死）
        return []


def build_toolbox_clause(candidates: list[dict[str, Any]]) -> str:
    """🔴 B1①·把工具箱拼成可注入解题打标 prompt 的「可用解题大招工具箱」段（纯函数·可单测）。

    candidates = toolbox_for_grade 产出（[{id,name,trigger_feature,action_conclusion,...}]）。
    空 → 返回 ""（上层据此标降级、不注入空块）。文本含名称 + 触发特征 → 让 opus 解题时认得出该用哪个，
    并在 modelCandidates 里**只填它真正用到的模型名**（名称须与工具箱一致，便于下游纯代码映射 M-id）。
    """
    cands = [c for c in (candidates or []) if isinstance(c, dict) and c.get("name")]
    if not cands:
        return ""
    lines: list[str] = [
        "🔴 可用解题大招工具箱（本年级已沉淀的解题模型；解题时若用到其中某个套路，"
        "请在 modelCandidates 里**照下面的名称原样填**你真正用到的那几个，没用到就留空数组、绝不硬凑）："
    ]
    for c in cands:
        trig = c.get("trigger_feature") or ""
        act = c.get("action_conclusion") or ""
        lines.append(f"- 「{c.get('name')}」：触发特征 = {trig} → {act}")
    return "\n".join(lines)


def _match_name_to_candidate(
    name: str, by_name: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """opus 选的模型名 → 候选集里的条目（精确名优先，退包含匹配·最短候选名优先=最具体）。
    匹配不到 → None（上层按临时模型/丢弃处置，绝不硬塞）。"""
    name = str(name or "").strip()
    if not name:
        return None
    c = by_name.get(name)
    if c is not None:
        return c
    cands = [
        (pname, pc) for pname, pc in by_name.items()
        if pname and (name in pname or pname in name)
    ]
    if cands:
        cands.sort(key=lambda x: len(x[0]))
        return cands[0][1]
    return None


def anchor_models_from_names(
    model_names: list[str] | None,
    *,
    grade_code: str | None = None,
    chapter_scope: bool = False,
    leaf_codes: list[str] | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """🔴 B1②③·纯代码模型对齐（消重复解题 + 诚实三态）。

    opus 带料解题时已从工具箱选了模型名（dna.modelCandidates）→ 本函数**纯代码 SQL 映射**：
      模型名 → 在该年级（或章）候选集里匹配成 M-id + tier_int/freq_int。**不再调 confirm_models
      第二次 LLM 解题**（全流程解题 = 1 次 opus）。

    诚实三态（去 M00 兜底）：
      - 有考模型（名匹配到候选）→ models=[{id,name,tier_int,freq_int}]，model_flag=None。
      - opus 给了名但候选集里匹配不到 → 视作池外 overflow 留痕（不硬塞 M00、不入正式维）；
        若最终一个都没匹配到 → models=[] + model_flag="no_model"（明示「无考模型」，难度降级）。
      - opus 没给任何模型名（modelCandidates 空）→ models=[] + model_flag="no_model"（真无考模型）。

    candidates 给定 → 直接用（调用方已 toolbox_for_grade 备好，免二次查库，并保证与注入工具箱同集）；
    否则按 grade_code / leaf_codes 现查（chapter_scope=True 走按章 lookup_candidates）。
    取数失败 → model_flag="lookup_unavailable" + model_warn（不卡死、不 M00）。
    """
    names = [str(n).strip() for n in (model_names or []) if str(n).strip()]

    # 候选集：调用方给定优先（与注入工具箱同集，最稳）；否则现查。
    if candidates is None:
        try:
            if chapter_scope:
                candidates = lookup_candidates(leaf_codes or [])
            else:
                pfx = _resolve_grade_prefix(leaf_codes or [], grade_code)
                candidates = lookup_candidates_by_grade(pfx) if pfx else lookup_candidates(leaf_codes or [])
        except Exception:  # noqa: BLE001 — 库故障：无法对齐 → ⚠ 但不 M00、不卡死（诚实留空 + 标记）
            return {
                "models": [], "temp_models": [], "model_overflow": [],
                "model_warn": True, "model_flag": "lookup_unavailable",
            }
    candidates = candidates or []
    by_name = {str(c.get("name")): c for c in candidates if c.get("name")}

    confirmed: list[dict[str, Any]] = []
    overflow: list[str] = []
    seen: set[str] = set()
    for nm in names:
        c = _match_name_to_candidate(nm, by_name)
        if c is None:
            if nm not in overflow:
                overflow.append(nm)  # 池外名：留痕（不硬塞 M00、不参与判档）
            continue
        cid = str(c.get("id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        m_out: dict[str, Any] = {"id": cid, "name": str(c.get("name") or "")}
        if c.get("tier_int") is not None:
            m_out["tier_int"] = c.get("tier_int")
        if c.get("freq_int") is not None:
            m_out["freq_int"] = c.get("freq_int")
        if c.get("model_kind"):
            m_out["model_kind"] = c.get("model_kind")
        confirmed.append(m_out)
        if len(confirmed) >= MODELS_MAX:
            break

    if not confirmed:
        # 🔴 诚实三态：真无考模型（没给名 / 给了名全匹配不到）→ models:[] + no_model（绝不 M00）。
        return {
            "models": [], "temp_models": [], "model_overflow": overflow,
            "model_warn": bool(overflow), "model_flag": NO_MODEL_FLAG,
        }
    return {
        "models": confirmed, "temp_models": [], "model_overflow": overflow,
        "model_warn": bool(overflow), "model_flag": ("overflow" if overflow else None),
    }


async def confirm_models(
    stem: str,
    answer: str,
    candidates: list[dict[str, Any]],
    *,
    invoke: Any,
    model: str | None = None,
    scope_note: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """池内确认 + 临时模型识别。返回 (confirmed, temp_models, overflow)。

    confirmed   = [{id,name,(tier_int),(freq_int),(model_kind)}]（id ∈ candidates；去重；截 ≤MODELS_MAX）。
                  tier_int/freq_int = biz_solution_model 控制面板表真值（WS1，可空）。
    temp_models = [{id:None, name, model_kind:'derived', isNew:True, tier_int(LLM提·临时草案),
                  freq_int, tier_text, freq_text, triggerFeature, action}]（WS2·AC5：候选外但确有可
                  复用套路 → 临时模型，**参与 grade_observed 判档不跳过**；≤1 个）。
    overflow    = [str]（解析不出对象、只剩裸名的池外提议；仅 ⚠/待命名池留痕，不参与判档）。
    候选空 → 不调 LLM，直接 ([], [], [])（上层兜 M00）。
    LLM 失败/解析失败 → ([], [], [])（上层兜 M00，不卡死）。
    """
    from langchain_core.messages import HumanMessage  # 局部 import 避循环依赖

    candidates = candidates or []
    if not candidates:
        return [], [], []
    by_id = {str(c.get("id")): c for c in candidates}
    by_name = {str(c.get("name")): c for c in candidates}
    prompt = CONFIRM_PROMPT.format(
        max_n=MODELS_MAX,
        stem=stem or "（题面缺失）",
        answer=(answer or "（无）")[:1500],
        candidates=_candidates_text(candidates),
        scope_note=scope_note or "",
    )
    try:
        text = await invoke([HumanMessage(content=prompt)], model=model)
    except Exception:
        return [], [], []
    parsed = _parse_confirm_json(text)
    if parsed is None:
        return [], [], []
    hits, new_models = parsed
    confirmed: list[dict[str, Any]] = []
    overflow: list[str] = []
    seen: set[str] = set()
    for p in hits:
        c = by_id.get(p) or by_name.get(p)
        if c is not None:
            cid = str(c.get("id"))
            if cid not in seen:
                seen.add(cid)
                # 🔴 PRD-C-103 WS1：把表真值 tier_int/freq_int 随确认模型透传（控制面板电线）。
                #   grade_observed 优先吃整数 tier_int/freq_int；缺则回退文本 tier/freqHint 兼容旧路。
                m_out: dict[str, Any] = {"id": cid, "name": str(c.get("name") or "")}
                if c.get("tier_int") is not None:
                    m_out["tier_int"] = c.get("tier_int")
                if c.get("freq_int") is not None:
                    m_out["freq_int"] = c.get("freq_int")
                if c.get("model_kind"):
                    m_out["model_kind"] = c.get("model_kind")
                confirmed.append(m_out)
        else:
            # 池外裸 id/名（LLM 把不在候选里的 id 塞进 hits）→ overflow 留痕（不参与判档）
            if p not in overflow:
                overflow.append(p)
    # 🔴 WS2·AC5：临时模型（候选外可复用套路）→ 带 LLM 临时 tier/freq 参与判档，不跳过。
    temp_models: list[dict[str, Any]] = []
    cand_names = set(by_name.keys())
    for nm in new_models:
        nm_name = nm["name"]
        # 防重：LLM 把候选里已有的模型又当新模型提 → 归并到 confirmed（用候选表真值），不重复造临时
        if nm_name in cand_names:
            c = by_name[nm_name]
            cid = str(c.get("id"))
            if cid not in seen:
                seen.add(cid)
                m_out = {"id": cid, "name": nm_name}
                if c.get("tier_int") is not None:
                    m_out["tier_int"] = c.get("tier_int")
                if c.get("freq_int") is not None:
                    m_out["freq_int"] = c.get("freq_int")
                if c.get("model_kind"):
                    m_out["model_kind"] = c.get("model_kind")
                confirmed.append(m_out)
            continue
        temp_models.append({
            "id": None,
            "name": nm_name,
            "model_kind": "derived",
            "isNew": True,
            "tier_int": _TIER_TEXT2INT.get(nm["tier_text"], 1),
            "freq_int": _FREQ_TEXT2INT.get(nm["freq_text"], 1),
            "tier_text": nm["tier_text"],
            "freq_text": nm["freq_text"],
            "triggerFeature": nm["triggerFeature"],
            "action": nm["action"],
        })
    # 临时模型最多 1 个（prompt 已约束；防 LLM 多吐）
    return confirmed[:MODELS_MAX], temp_models[:1], overflow


async def anchor_models(
    dna: dict[str, Any] | None,
    *,
    stem: str,
    answer: str = "",
    invoke: Any,
    model: str | None = None,
    record_overflow: Any = None,
    grade_code: str | None = None,
    chapter_scope: bool = False,
) -> dict[str, Any]:
    """W1' 模型锚定主入口（§3.2 处置穷举 + WS2·AC5 临时模型不跳过 + PRD-C-105 B 年级全量召回）。

    🔴 PRD-C-105 B·召回粒度（D1）：
      - **默认（chapter_scope=False）= 按年级全量召回**：拿该年级**所有**解题模型（~10-15）喂 CONFIRM
        「先解题再挑真正用到的」（≤3、宁缺毋滥、可留空）。治「跨章题（折线数轴折叠题被标进方程章）
        按章反查找不到挂数轴章的数轴大招 → 候选空 → M00 短路 → 模型没绑」（AC-B1）。
      - **chapter_scope=True（仅用户明确指定章节时由上层传）= 按章收窄**：走原 lookup_candidates，
        拿 dna 锚定叶子按章反查（候选更贴题、更窄）（AC-B2，老路径不破）。
      - 年级前缀取法 = _resolve_grade_prefix（优先锚定叶子 code 的 3 位根，DB 对齐最可靠；grade_code
        仅作旁证，'3071' 这类历史映射会被主动丢弃，详见该函数）。年级前缀取不到 → 降级回按章召回。

    返回 {models, temp_models, model_overflow, model_warn, model_flag}。
      models         : [{id,name,(tier_int),(freq_int),(isNew)}]，1~N 项，**非空**（无命中保底 M00）。
                       🔴 WS2·AC5：含临时模型（isNew=True、id=None、带 LLM 临时 tier_int/freq_int），
                       直接进 models 喂 grade_observed → 临时模型**参与判档不跳过**。
      temp_models    : [{name,model_kind:'derived',isNew,tier_int,freq_int,...}]，本题待转正临时模型
                       （供 WS2 落库转正脚本消费；≤1 个，可空）。
      model_overflow : [str]，解析不出对象的裸池外名（仅 ⚠/待命名池用，不参与判档，可空）。
      model_warn     : bool，是否卡面 ⚠（有临时模型 / 裸池外名 / 反查库故障降级）。
      model_flag     : str|None（"m00_fallback"/"overflow"/"temp_model"/"lookup_unavailable"/None）。

    record_overflow(name, mother_models)：把裸池外名落待命名池的回调（None=不落，仅返回 overflow）。
    """
    leaf_codes = _leaf_codes(dna)

    # 🔴 PRD-C-105 B：定召回策略。默认年级全量；chapter_scope（上层指定章）→ 按章收窄。
    #   年级前缀取不到（无锚定叶子且 grade_code 不可用）→ 降级回按章（候选可能空，由 M00 兜底）。
    grade_prefix = None if chapter_scope else _resolve_grade_prefix(leaf_codes, grade_code)
    use_grade = grade_prefix is not None

    # 反查（纯只读 ETL）；库/表不可用 → 降级 M00 + ⚠（C-010 闸门必有降级路径）。
    try:
        if use_grade:
            candidates = lookup_candidates_by_grade(grade_prefix)
        else:
            candidates = lookup_candidates(leaf_codes)
    except Exception:
        return {
            "models": [dict(M00)],
            "temp_models": [],
            "model_overflow": [],
            "model_warn": True,
            "model_flag": "lookup_unavailable",
        }

    confirmed, temp_models, overflow = await confirm_models(
        stem, answer, candidates, invoke=invoke, model=model,
        scope_note=GRADE_SCOPE_NOTE if use_grade else "",
    )

    # 裸池外名落待命名池（含题目指针由上层带；这里只回调名 + 母题 models 锚）。
    if overflow and record_overflow is not None:
        mother_models = [c["id"] for c in confirmed] or [M00_ID]
        for name in overflow:
            try:
                record_overflow(name, mother_models)
            except Exception:
                pass  # 待命名池落盘失败不拖垮主流程（G4：写失败由上层降级，不静默成成功）

    # 🔴 WS2·AC5：临时模型并入 models（参与 grade_observed 判档），并随 temp_models 返回供转正。
    models_out: list[dict[str, Any]] = list(confirmed)
    for tm in temp_models:
        models_out.append({
            "id": None,
            "name": tm.get("name"),
            "model_kind": "derived",
            "isNew": True,
            "tier_int": tm.get("tier_int"),
            "freq_int": tm.get("freq_int"),
        })

    if not models_out:
        # 候选空 / 全不确认 且无临时模型 → M00 兜底。有裸池外名 → 一并 ⚠。
        return {
            "models": [dict(M00)],
            "temp_models": [],
            "model_overflow": overflow,
            "model_warn": bool(overflow),
            "model_flag": "overflow" if overflow else "m00_fallback",
        }

    flag = "temp_model" if temp_models else ("overflow" if overflow else None)
    return {
        "models": models_out,
        "temp_models": temp_models,
        "model_overflow": overflow,
        "model_warn": bool(overflow or temp_models),
        "model_flag": flag,
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
