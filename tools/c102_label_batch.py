# -*- coding: utf-8 -*-
r"""PRD-C-102 第4步·题目打标批处理 runner（只读库、落 JSON，不写库）。

用法（cwd = agent-service-toolkit，venv = .venv\Scripts\python.exe）：
  .venv\Scripts\python.exe tools\c102_label_batch.py --qids 2069819501599281153,2069819459874344961
  .venv\Scripts\python.exe tools\c102_label_batch.py --chapter 100005 --db ai_lesson_prep --limit 100
  .venv\Scripts\python.exe tools\c102_label_batch.py --qids ... --out tools\_label_out.json

干什么：
  1. pymysql 只读每题：题干(biz_question.stem_text)、答案/详解(biz_text_content A/E)、
     配图(biz_question_image 非装饰 oss_url)、候选模型(biz_solution_model 该书 gold + 通用)、
     知识点(biz_question_knowledge / dim1_kp_id + biz_subject 名)、年级(kp 顶层 subject)。
  2. 下配图 OSS → base64（urllib + ProxyHandler({}) 绕本机代理）。
  3. asyncio 限流并发跑 labeler.label_one + label_and_grade（中转 opus-4-8）。
  4. 累计 token usage（从 LLM response.usage），按 RELAY_PRICES（kiro opus ×0.28 折扣口径）估扣费。
  5. 打标结果 + 提议新模型 落 JSON（本阶段 A 不写库）。

DB 连接：默认读 workplace/.mcp.json mysql.env 的 host/port/user/pass；--db 覆盖库名
  （🔴 默认 ai_lesson_prep —— 新题库录入库，非 mcp 默认 miskt_data2）。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pymysql

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from langchain_core.messages import HumanMessage  # noqa: E402

from agents import labeler  # noqa: E402
from core import difficulty  # noqa: E402
from core.settings import settings  # noqa: E402

# ---------------------------------------------------------------------------
# 🔴 表驱动难度：tier/freq 现从 biz_solution_model.difficulty_tier/freq_band 列读（V911 已 apply
#   2026-06-26）。难度阶/考频不信 LLM 自评，只认【匹配中模型的表值】。维护这张表即控制难度。
#   下面 MODEL_TIER_MAP 仅留作 V911 列缺失时的离线兜底（生产路径已改读 DB 列，见 label_question）。
# ---------------------------------------------------------------------------
MODEL_TIER_MAP: dict[str, tuple[str, str]] = {f"DZ{i:02d}": ("高阶", "通法") for i in range(1, 16)}
MODEL_TIER_MAP.update({f"TY{i:02d}": ("高阶", "通法") for i in range(1, 13)})
MODEL_TIER_MAP["TY10"] = ("基础", "通法")  # 方程建模 = 基础通法，不抬档

# ---------------------------------------------------------------------------
# DB 配置（读 workplace/.mcp.json mysql.env）
# ---------------------------------------------------------------------------
_MCP_JSON = Path("D:/workplace/book-ai/workplace/.mcp.json")
DEFAULT_DB = "ai_lesson_prep"  # 🔴 新题库录入库（smoke qids 在此），非 mcp 默认 miskt_data2


def load_db_cfg(db_override: str | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {}
    try:
        mcp = json.loads(_MCP_JSON.read_text(encoding="utf-8"))
        env = (mcp.get("mcpServers", {}).get("mysql", {}) or {}).get("env", {}) or {}
    except Exception as e:
        print(f"[warn] 读 .mcp.json 失败，用兜底连接参数: {e}")
    return {
        "host": env.get("MYSQL_HOST", "127.0.0.1"),
        "port": int(env.get("MYSQL_PORT", 3307)),
        "user": env.get("MYSQL_USER", "root"),
        "password": env.get("MYSQL_PASS", "123456"),
        "database": db_override or DEFAULT_DB,
        "charset": "utf8mb4",
    }


# ---------------------------------------------------------------------------
# 只读取数（每题装配）
# ---------------------------------------------------------------------------
def fetch_candidate_models(cur, book_id: str | None) -> list[dict[str, Any]]:
    """候选解题模型：该书 gold + 通用（is_router=router 类）。本库 15 个 gold 同书，全带过去。"""
    cur.execute(
        "SELECT id, name, category, model_kind, trigger_feature, action_conclusion, "
        "difficulty_tier, freq_band "
        "FROM biz_solution_model WHERE status IN ('0','1') OR status IS NULL "
        "ORDER BY sort, id"
    )
    rows = cur.fetchall()
    # 当前库所有 gold 同书（BS7SZ），不再按 book_id 缩窄（避免漏候选）；未来多书可加 WHERE book_id=
    return [dict(r) for r in rows]


def _ancestor_names(cur, kp_id: str | None) -> list[str]:
    """kp 叶子 → 自身 + 各级祖先名（叶→根顺序）。用于年级(根)与按章召回(全链 blob)。"""
    names: list[str] = []
    cur_id = kp_id
    for _ in range(12):
        if not cur_id:
            break
        cur.execute("SELECT id, name, parent_id, level FROM biz_subject WHERE id=%s", (cur_id,))
        row = cur.fetchone()
        if not row:
            break
        if row.get("name"):
            names.append(row["name"])
        if str(row.get("level")) == "1" or not row.get("parent_id") or row["parent_id"] == "0":
            break
        cur_id = row["parent_id"]
    return names


def _grade_text_of(cur, kp_id: str | None) -> str:
    """kp 叶子 → 顶层 subject 名（level=1）= 年级。"""
    names = _ancestor_names(cur, kp_id)
    return names[-1] if names else ""


def scope_candidate_models(all_models: list[dict[str, Any]], chapter_blob: str) -> list[dict[str, Any]]:
    """按章召回候选模型（数据驱动，不硬编码章→类映射）：
      - 通用模型（category 含「通用」或为空）= 跨章工具，恒保留。
      - gold/专题大招：仅当其 category 关键词出现在本题知识点祖先链 blob 里才保留（数轴题不灌角/线段大招）。
    chapter_blob = 本题 kp 祖先链名 + 知识点名拼串。命中不了的专题大招过滤掉、减噪 + 防误匹配。"""
    blob = chapter_blob or ""
    out: list[dict[str, Any]] = []
    for m in all_models:
        cat = (m.get("category") or "").strip()
        if (not cat) or ("通用" in cat):
            out.append(m)
            continue
        # 专题大招：category（如 数轴/方程/动点/线段/角）须命中本章 blob
        if cat in blob or any(tok and tok in blob for tok in cat.split("/")):
            out.append(m)
    return out


def fetch_qtype_dict(cur) -> list[str]:
    """系统题型字典 label 闭集（sys_dict_data biz_question_type，按 sort）。"""
    try:
        cur.execute(
            "SELECT dict_label FROM sys_dict_data WHERE dict_type='biz_question_type' "
            "ORDER BY dict_sort, dict_value"
        )
        labels = [r["dict_label"] for r in cur.fetchall() if r.get("dict_label")]
        return labels or list(labeler.QTYPE_DICT_FALLBACK)
    except Exception as e:
        print(f"[warn] 读题型字典失败，用兜底集: {e}")
        return list(labeler.QTYPE_DICT_FALLBACK)


def fetch_question(cur, qid: int) -> dict[str, Any] | None:
    cur.execute(
        "SELECT id, stem_text, dim1_kp_id, dim2_qtype, question_type, source_raw, "
        "answer_text_content_id, analyze_text_content_id "
        "FROM biz_question WHERE id=%s", (qid,)
    )
    q = cur.fetchone()
    if not q:
        return None

    # 文本：A=答案 E=详解 S=题干（题干优先 stem_text，缺则取 S）
    cur.execute(
        "SELECT content_type, content FROM biz_text_content WHERE question_id=%s "
        "AND content_type IN ('A','E','S')", (qid,)
    )
    texts = {r["content_type"]: r["content"] for r in cur.fetchall()}
    stem = (q.get("stem_text") or texts.get("S") or "").strip()
    answer = (texts.get("A") or "").strip()
    analysis = (texts.get("E") or "").strip()

    # 配图（非装饰）
    cur.execute(
        "SELECT oss_url, role, seq FROM biz_question_image "
        "WHERE question_id=%s AND (is_decorative=0 OR is_decorative IS NULL) "
        "AND oss_url IS NOT NULL AND oss_url<>'' ORDER BY seq, id", (qid,)
    )
    images = [r["oss_url"] for r in cur.fetchall()]

    # 知识点（主 + 邻居名）
    cur.execute(
        "SELECT k.knowledge_id, s.name FROM biz_question_knowledge k "
        "LEFT JOIN biz_subject s ON s.id=k.knowledge_id WHERE k.question_id=%s "
        "ORDER BY k.is_primary DESC", (qid,)
    )
    kps = [{"id": r["knowledge_id"], "name": r.get("name") or ""} for r in cur.fetchall()]
    if not kps and q.get("dim1_kp_id"):
        cur.execute("SELECT name FROM biz_subject WHERE id=%s", (q["dim1_kp_id"],))
        nm = cur.fetchone()
        kps = [{"id": q["dim1_kp_id"], "name": (nm or {}).get("name", "")}]

    ancestors = _ancestor_names(cur, q.get("dim1_kp_id"))
    grade_text = ancestors[-1] if ancestors else ""
    # 按章召回用的章节 blob = 祖先链名 + 知识点名（含数轴/方程/角等主题词，供 scope 匹配 gold category）
    chapter_blob = " ".join(ancestors + [k.get("name") or "" for k in kps])

    return {
        "qid": qid,
        "stem": stem,
        "answer": answer,
        "analysis": analysis,
        "image_urls": images,
        "knowledge_points": kps,
        "grade_text": grade_text,
        "chapter_blob": chapter_blob,
        "source_raw": q.get("source_raw") or "",
        "dim2_qtype": q.get("dim2_qtype"),
    }


# ---------------------------------------------------------------------------
# 图 OSS → base64（绕本机代理）
# ---------------------------------------------------------------------------
def _opener_noproxy():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_image_b64(url: str, timeout: float = 30.0) -> tuple[str | None, str]:
    """下载 OSS 图 → base64（不含 data: 前缀）。返回 (b64|None, note)。"""
    try:
        opener = _opener_noproxy()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            return None, f"空响应 {url[:60]}"
        return base64.b64encode(data).decode("ascii"), f"ok {len(data)}B"
    except Exception as e:
        return None, f"下载失败 {url[:60]} : {e}"


# ---------------------------------------------------------------------------
# 直连中转 invoke（捕获 token usage）—— labeler.label_one 的 invoke 注入
# 不走 variant._ainvoke_text（那耦合 graph config/conv_trace）；批量用裸 AsyncOpenAI 干净计量。
# ---------------------------------------------------------------------------
class RelayInvoker:
    """OpenAI-compatible 直连中转（sui-xiang 主），把 HumanMessage 多模态转 chat 格式 + 累计 usage。"""

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        # 主站取 RELAY_POOL[0]（sui-xiang）；缺则回退 COMPATIBLE_*
        base_url = settings.COMPATIBLE_BASE_URL
        api_key = settings.COMPATIBLE_API_KEY.get_secret_value() if settings.COMPATIBLE_API_KEY else None
        relay_name = settings.RELAY_NAME
        if settings.RELAY_POOL:
            try:
                pool = json.loads(settings.RELAY_POOL)
                if pool:
                    base_url = pool[0].get("base_url", base_url)
                    api_key = pool[0].get("api_key", api_key)
                    relay_name = pool[0].get("name", relay_name)
            except Exception:
                pass
        self.relay_name = relay_name
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @staticmethod
    def _to_openai_content(msg: HumanMessage) -> Any:
        c = msg.content
        if isinstance(c, str):
            return c
        out: list[dict[str, Any]] = []
        for part in c:
            if not isinstance(part, dict):
                out.append({"type": "text", "text": str(part)})
            elif part.get("type") == "text":
                out.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                out.append({"type": "image_url", "image_url": part.get("image_url", {})})
        return out

    async def __call__(
        self,
        messages: list[HumanMessage],
        *,
        model: str,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        oai_msgs = [{"role": "user", "content": self._to_openai_content(m)} for m in messages]
        kw: dict[str, Any] = dict(model=model, messages=oai_msgs, temperature=temperature)
        if max_tokens and max_tokens > 0:
            kw["max_tokens"] = max_tokens
        if timeout and timeout > 0:
            kw["timeout"] = timeout
        if response_format:
            kw["response_format"] = response_format
        try:
            resp = await self.client.chat.completions.create(**kw)
        except Exception as e:
            # 逆向渠道偶发拒 response_format（json_schema 尾随）→ 退掉重试一次
            if response_format:
                kw.pop("response_format", None)
                resp = await self.client.chat.completions.create(**kw)
            else:
                raise
        self.calls += 1
        u = getattr(resp, "usage", None)
        if u:
            self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)
        return (resp.choices[0].message.content or "") if resp.choices else ""


# ---------------------------------------------------------------------------
# 计费（kiro opus-4-8 单价 输入 $1.40 / 输出 $7.00 每 1M token；×0.28 折扣口径）
# ---------------------------------------------------------------------------
USD_IN_PER_1M = 1.40
USD_OUT_PER_1M = 7.00
KIRO_DISCOUNT = 0.28
USD_TO_CNY = 7.2  # 估算汇率


def estimate_cost(pt: int, ct: int) -> dict[str, float]:
    usd = (pt / 1_000_000) * USD_IN_PER_1M + (ct / 1_000_000) * USD_OUT_PER_1M
    usd_disc = usd * KIRO_DISCOUNT
    return {
        "usd_list": round(usd, 6),
        "usd_kiro_x0.28": round(usd_disc, 6),
        "cny_kiro": round(usd_disc * USD_TO_CNY, 6),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
# dim2_qtype 数值 → biz_question_type 字典 value 对齐（1选择/2判断/3应用/4填空/5解答/6作图/7计算/8证明）
_DIM2_TO_LABEL = {1: "选择题", 2: "判断题", 3: "应用题", 4: "填空题",
                  5: "解答题", 6: "作图题", 7: "计算题", 8: "证明题"}


async def label_question(
    q: dict[str, Any], candidate_models: list[dict[str, Any]],
    invoker: RelayInvoker, model: str, sem: asyncio.Semaphore,
    img_cache: dict[str, str | None], qtype_dict: list[str],
) -> dict[str, Any]:
    qid = q["qid"]
    img_notes: list[str] = []
    b64_list: list[str] = []
    for url in q["image_urls"]:
        if url in img_cache:
            b64 = img_cache[url]
        else:
            b64, note = fetch_image_b64(url)
            img_cache[url] = b64
            img_notes.append(note)
        if b64:
            b64_list.append(b64)

    # 按章召回：通用模型恒留 + 本章 gold大招（防全库 27 个噪声 + 误匹配）
    scoped_models = scope_candidate_models(candidate_models, q.get("chapter_blob", ""))

    async with sem:
        t0 = time.monotonic()
        try:
            raw = await labeler.label_one(
                stem=q["stem"], analysis=q["analysis"], answer=q["answer"],
                image_b64_list=b64_list,
                candidate_models=scoped_models,
                knowledge_points=q["knowledge_points"],
                grade_text=q["grade_text"],
                source_marks=q["source_raw"],
                qtype_dict=qtype_dict,
                invoke=invoker, model=model,
                max_tokens=settings.MOTHER_OPUS_MAX_TOKENS,
            )
            dur = round(time.monotonic() - t0, 1)
            parsed = labeler.parse_label(raw, qtype_dict=qtype_dict)
            # 🔴 表驱动难度：tier/freq 从模型表 biz_solution_model 的 difficulty_tier/freq_band 读
            #   （V911 已 apply）。命中 id → 查表；裸策略词不抬档(high_strategies=[])；无匹配=基础题 K/R/D 定档。
            model_tf = {(m.get("id") or "").strip(): (
                "高阶" if (m.get("difficulty_tier") or 1) >= 2 else "基础",
                "通法" if (m.get("freq_band") or 1) >= 2 else "一次性",
            ) for m in candidate_models}
            table_hits = []
            for m in (parsed.get("models") or []):
                if m.get("isNew"):
                    continue
                tf = model_tf.get((m.get("id") or "").strip())
                if tf:
                    table_hits.append({"id": m.get("id"), "name": m.get("name"),
                                       "tier": tf[0], "freqHint": tf[1]})
            ff = parsed.get("factors") or {}
            graded = dict(parsed)
            graded["difficulty"] = difficulty.grade_observed(
                model_hits=table_hits, K=ff.get("K", 0), R=ff.get("R", 0),
                D=ff.get("D", 0), G=ff.get("G", 0), high_strategies=[])
            graded["_table_hits"] = table_hits
            # 🔴 题型 = dim2 权威（录入结构化字段；维护者裁定 2026-06-26）：
            #   有 dim2 → 用 dim2 当家；labeler 现判仅作 dim2 缺位时的补位。
            #   冲突（dim2≠labeler）时把 labeler 现判留痕 questionTypeLLM，供交叉表审计。
            dim2 = q.get("dim2_qtype")
            dim2_label = _DIM2_TO_LABEL.get(dim2) if dim2 else None
            llm_qtype = graded.get("questionType")
            if dim2_label:
                if llm_qtype and llm_qtype != dim2_label:
                    graded["questionTypeLLM"] = llm_qtype
                    graded["questionTypeConflict"] = True
                graded["questionType"] = dim2_label  # dim2 当家
            # （无 dim2：在线举一反三等场景 → 保留 labeler 现判，不动）
            return {
                "qid": qid,
                "ok": parsed.get("_parse_ok", False),
                "duration_s": dur,
                "has_images": bool(q["image_urls"]),
                "image_b64_ok": len(b64_list),
                "image_notes": img_notes,
                "grade_text": q["grade_text"],
                "label": graded,
                "raw_len": len(raw),
            }
        except Exception as e:
            return {"qid": qid, "ok": False, "error": str(e),
                    "has_images": bool(q["image_urls"]), "image_notes": img_notes}


async def run(qids: list[int], db_cfg: dict[str, Any], out_path: Path, concurrency: int) -> None:
    model = settings.variant_model("mother_solve_label") or "claude-opus-4-8"
    conn = pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **db_cfg)
    cur = conn.cursor()

    print(f"[cfg] db={db_cfg['database']}@{db_cfg['host']}:{db_cfg['port']} model={model} "
          f"relay={settings.RELAY_NAME} concurrency={concurrency} qids={len(qids)}")

    questions: list[dict[str, Any]] = []
    for qid in qids:
        q = fetch_question(cur, qid)
        if q is None:
            print(f"[warn] qid {qid} 不存在，跳过")
            continue
        questions.append(q)
        print(f"  · qid={qid} stem={len(q['stem'])}字 analysis={len(q['analysis'])}字 "
              f"图={len(q['image_urls'])} kp={[k['name'] for k in q['knowledge_points']]} "
              f"年级={q['grade_text']}")

    candidate_models = fetch_candidate_models(cur, None)
    qtype_dict = fetch_qtype_dict(cur)
    print(f"[cfg] 候选模型 {len(candidate_models)} 个（gold 大招 + 通用）；题型字典 {qtype_dict}")
    cur.close()
    conn.close()

    invoker = RelayInvoker()
    sem = asyncio.Semaphore(concurrency)
    img_cache: dict[str, str | None] = {}
    t0 = time.monotonic()
    results = await asyncio.gather(*[
        label_question(q, candidate_models, invoker, model, sem, img_cache, qtype_dict)
        for q in questions
    ])
    wall = round(time.monotonic() - t0, 1)

    cost = estimate_cost(invoker.prompt_tokens, invoker.completion_tokens)
    proposed_new = []
    for r in results:
        for m in (r.get("label", {}).get("models") or []):
            if m.get("isNew"):
                proposed_new.append({"qid": r["qid"], "name": m.get("name"),
                                     "triggerFeature": m.get("triggerFeature"), "action": m.get("action")})

    summary = {
        "model": model,
        "relay": invoker.relay_name,
        "questions": len(questions),
        "llm_calls": invoker.calls,
        "wall_s": wall,
        "tokens": {
            "prompt": invoker.prompt_tokens,
            "completion": invoker.completion_tokens,
            "total": invoker.prompt_tokens + invoker.completion_tokens,
        },
        "cost_estimate": cost,
        "proposed_new_models": proposed_new,
    }
    payload = {"summary": summary, "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n================ 汇总 ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[done] 结果落 {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", help="逗号分隔题 id")
    ap.add_argument("--chapter", help="按章前缀取题（dim1_kp_id LIKE 前缀%）")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"库名（默认 {DEFAULT_DB}）")
    ap.add_argument("--limit", type=int, default=100, help="--chapter 时取题上限")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", default="tools/_label_out.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    db_cfg = load_db_cfg(args.db)

    qids: list[int] = []
    if args.qids:
        qids = [int(x.strip()) for x in args.qids.split(",") if x.strip()]
    elif args.chapter:
        conn = pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **db_cfg)
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM biz_question WHERE dim1_kp_id LIKE %s ORDER BY id LIMIT %s",
            (args.chapter + "%", args.limit),
        )
        qids = [r["id"] for r in cur.fetchall()]
        cur.close()
        conn.close()
    else:
        print("需 --qids 或 --chapter")
        sys.exit(2)

    asyncio.run(run(qids, db_cfg, Path(args.out), args.concurrency))


if __name__ == "__main__":
    main()
