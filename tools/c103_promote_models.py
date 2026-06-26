# -*- coding: utf-8 -*-
r"""PRD-C-103 批2·WS2·AC6：临时模型转正 + 题↔模型落链（纯 pymysql，无 LLM）。

决策依据（PRD §3.1 D3/D4/D10/D11 + §10 契约）：
  - 临时模型（打标命中库里没有的可复用套路）由 LLM 提临时 tier/freq（待审草案），**脚本直改库转正**
    入 `biz_solution_model`（model_kind='derived'），**不设人工审核台 / 不加 book-server 接口**。
  - 难度最终 authority = 人维护的表；本脚本写表即把临时 tier 落成「表真值」，维护者随时可直改库覆盖。
  - 打标命中**已有模型**也落链（不止临时模型）→ `biz_question_model(question_id, model_id)`。
  - 幂等：同 name 已存在的 derived 模型 → 复用其 id，不重建；`biz_question_model` 有 uk(question_id,
    model_id) → INSERT IGNORE 防重；`biz_variation_trace` 有 uk(variant_question_id) → 防重。

数据来源（两种喂法，二选一）：
  ① --manifest <path.jsonl>：消费 toolkit persist_to_bank 落的「落链清单」（推荐，端到端真链）。
     每行 = {"question_id":int, "role":"mother|variant", "models":[{id?,name,isNew?,tier_int?,freq_int?,
              model_kind?,triggerFeature?,action?}...], "temp_models":[...同上...],
              "mother_question_id":int|null, "trace":{method,similarity,target_level,actual_level,retries}|null}
  ② --inline-json <json>：单条 dict 或 list[dict]（同上 schema），便于测试/手工补录。

跑（cwd = agent-service-toolkit）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_promote_models.py --manifest artifacts/c103_link_manifest.jsonl
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_promote_models.py --manifest ... --dry-run

🔴 纯 pymysql 直改 ai_lesson_prep（D11）；只 INSERT/UPDATE 这三表，绝不 DROP/DELETE（DDL/删走人工）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pymysql

ROOT = Path(__file__).resolve().parents[1]

# DB 连接（默认读 .env 的 VARIANT_DB_*；缺则 PRD 给的 dev 默认）。
_ENV: dict[str, str] = {}
_envf = ROOT / ".env"
if _envf.exists():
    for line in _envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        _ENV[k.strip()] = v.strip()


def _db():
    return pymysql.connect(
        host=_ENV.get("VARIANT_DB_HOST", os.environ.get("VARIANT_DB_HOST", "127.0.0.1")),
        port=int(_ENV.get("VARIANT_DB_PORT", os.environ.get("VARIANT_DB_PORT", "3307"))),
        user=_ENV.get("VARIANT_DB_USER", os.environ.get("VARIANT_DB_USER", "root")),
        password=_ENV.get("VARIANT_DB_PASSWORD", os.environ.get("VARIANT_DB_PASSWORD", "123456")),
        database=_ENV.get("VARIANT_DB_NAME", os.environ.get("VARIANT_DB_NAME", "ai_lesson_prep")),
        charset="utf8mb4",
        autocommit=False,
    )


# 临时模型 tier/freq 文本 → 整数（与 model_anchor / core.difficulty 同口径）。
_TIER_TEXT2INT = {"基础": 1, "高阶": 2}
_FREQ_TEXT2INT = {"一次性": 1, "通法": 2}
_DERIVED_PREFIX = "TY"


def _tier_int(m: dict) -> int:
    ti = m.get("tier_int")
    if ti in (1, 2):
        return int(ti)
    return _TIER_TEXT2INT.get(str(m.get("tier_text") or m.get("tier") or "").strip(), 1)


def _freq_int(m: dict) -> int:
    fi = m.get("freq_int")
    if fi in (1, 2):
        return int(fi)
    return _FREQ_TEXT2INT.get(str(m.get("freq_text") or m.get("freq") or m.get("freqHint") or "").strip(), 1)


def _next_derived_id(cur) -> str:
    """生成下一个 TY{nn} id（扫现有 TY* 取最大数字 +1）。"""
    cur.execute(
        "SELECT id FROM biz_solution_model WHERE id LIKE %s", (_DERIVED_PREFIX + "%",)
    )
    mx = 0
    for (mid,) in cur.fetchall():
        tail = str(mid)[len(_DERIVED_PREFIX):]
        if tail.isdigit():
            mx = max(mx, int(tail))
    return f"{_DERIVED_PREFIX}{mx + 1:02d}"


def _find_model_id_by_name(cur, name: str) -> str | None:
    """按 name 找已存在模型 id（幂等复用，避免同名重建）。"""
    cur.execute("SELECT id FROM biz_solution_model WHERE name=%s LIMIT 1", (name,))
    row = cur.fetchone()
    return str(row[0]) if row else None


def promote_temp_model(cur, tm: dict, *, dry_run: bool) -> tuple[str, bool]:
    """临时模型转正：返回 (model_id, created)。

    幂等：同 name 已存在 → 复用其 id（created=False）；否则建新 TY{nn}（created=True）。
    """
    name = str(tm.get("name") or "").strip()
    if not name:
        raise ValueError("临时模型缺 name，拒绝转正")
    existing = _find_model_id_by_name(cur, name)
    if existing:
        return existing, False
    new_id = _next_derived_id(cur)
    tier = _tier_int(tm)
    freq = _freq_int(tm)
    trigger = str(tm.get("triggerFeature") or tm.get("trigger_feature") or "").strip()[:500]
    action = str(tm.get("action") or tm.get("action_conclusion") or "").strip()[:500]
    if not dry_run:
        cur.execute(
            """INSERT INTO biz_solution_model
                 (id, name, model_kind, is_gold, trigger_feature, action_conclusion,
                  difficulty_tier, freq_band, sort, status, create_time)
               VALUES (%s, %s, 'derived', 0, %s, %s, %s, %s, 0, '0', NOW())""",
            (new_id, name[:100], trigger, action, tier, freq),
        )
    return new_id, True


def link_question_model(cur, question_id: int, model_id: str, *, is_primary: bool,
                        role: str | None, dry_run: bool) -> bool:
    """落 biz_question_model（INSERT IGNORE，uk(question_id,model_id) 防重）。返回是否新增。"""
    if not question_id or not model_id:
        return False
    if model_id == "M00":  # 概念直用兜底不落链（非真模型）
        return False
    if dry_run:
        cur.execute(
            "SELECT 1 FROM biz_question_model WHERE question_id=%s AND model_id=%s",
            (question_id, model_id),
        )
        return cur.fetchone() is None
    cur.execute(
        """INSERT IGNORE INTO biz_question_model
             (question_id, model_id, is_primary, source, role, create_time)
           VALUES (%s, %s, %s, 'AI', %s, NOW())""",
        (question_id, model_id, 1 if is_primary else 0, role),
    )
    return cur.rowcount > 0


def write_variation_trace(cur, rec: dict, model_id_hint: str | None, *, dry_run: bool) -> bool:
    """可选：落 biz_variation_trace（uk(variant_question_id) 防重）。返回是否新增。

    仅当 role=variant 且有 mother_question_id + trace 块时写。method 取 trace.method 或第一个算子。
    """
    if rec.get("role") != "variant":
        return False
    mother = rec.get("mother_question_id")
    variant = rec.get("question_id")
    trace = rec.get("trace") or {}
    if not mother or not variant:
        return False
    method = str(trace.get("method") or trace.get("operator") or "unknown").strip()[:32]
    detail = str(trace.get("method_detail") or "").strip()[:500] or None
    degree = trace.get("variation_degree")
    if degree is None:
        degree = trace.get("similarity")
    band = trace.get("similarity_band") or _degree_to_band(degree)
    if dry_run:
        cur.execute("SELECT 1 FROM biz_variation_trace WHERE variant_question_id=%s", (variant,))
        return cur.fetchone() is None
    cur.execute(
        """INSERT IGNORE INTO biz_variation_trace
             (mother_question_id, variant_question_id, method, method_detail,
              variation_degree, similarity_band, same_source, created_by, create_time)
           VALUES (%s, %s, %s, %s, %s, %s, 1, 'reverse-dna', NOW())""",
        (mother, variant, method, detail,
         round(float(degree), 2) if degree is not None else None, band),
    )
    return cur.rowcount > 0


def _degree_to_band(degree) -> str | None:
    try:
        d = float(degree)
    except (TypeError, ValueError):
        return None
    if d >= 0.75:
        return "高"
    if d >= 0.45:
        return "中"
    return "低"


def process_record(cur, rec: dict, *, dry_run: bool) -> dict:
    """处理一条落链记录：转正临时模型 → 落链所有模型 → 可选 trace。返回统计。"""
    qid = rec.get("question_id")
    role = rec.get("role")
    stat = {"promoted": [], "linked": 0, "trace": 0, "qid": qid}
    if not qid:
        stat["skip"] = "缺 question_id"
        return stat

    # 1) 临时模型转正（先转正拿到真 id，再落链）
    temp_ids: list[str] = []
    for tm in rec.get("temp_models") or []:
        mid, created = promote_temp_model(cur, tm, dry_run=dry_run)
        temp_ids.append(mid)
        if created:
            stat["promoted"].append({"id": mid, "name": tm.get("name")})

    # 2) 落链：命中已有模型（有 id）+ 转正后的临时模型
    linked_ids: list[tuple[str, bool]] = []  # (model_id, is_primary)
    first = True
    for m in rec.get("models") or []:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "").strip()
        if not mid or mid == "M00":
            # 无 id 的临时模型在 models 里（isNew）→ 按 name 找转正后 id
            if m.get("isNew") or m.get("model_kind") == "derived":
                mid = _find_model_id_by_name(cur, str(m.get("name") or "").strip()) or ""
            if not mid:
                continue
        linked_ids.append((mid, first))
        first = False
    # 转正的临时模型一并落链（若未在 models 里出现）
    seen = {mid for mid, _ in linked_ids}
    for mid in temp_ids:
        if mid not in seen:
            linked_ids.append((mid, not linked_ids))  # 全无其他模型时临时模型当主
            seen.add(mid)

    for mid, is_primary in linked_ids:
        if link_question_model(cur, qid, mid, is_primary=is_primary, role=role, dry_run=dry_run):
            stat["linked"] += 1

    # 3) 可选血缘 trace
    if write_variation_trace(cur, rec, linked_ids[0][0] if linked_ids else None, dry_run=dry_run):
        stat["trace"] += 1

    return stat


def _iter_records(args) -> list[dict]:
    recs: list[dict] = []
    if args.manifest:
        p = Path(args.manifest)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            print(f"[error] manifest 不存在: {p}", file=sys.stderr)
            sys.exit(2)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception as e:  # noqa: BLE001
                print(f"[warn] 跳过坏行: {e}", file=sys.stderr)
    if args.inline_json:
        obj = json.loads(args.inline_json)
        recs.extend(obj if isinstance(obj, list) else [obj])
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-C-103 临时模型转正 + 题↔模型落链")
    ap.add_argument("--manifest", help="落链清单 JSONL 路径（toolkit persist 落）")
    ap.add_argument("--inline-json", help="单条/列表 JSON（测试/手工补录）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写库")
    args = ap.parse_args()

    records = _iter_records(args)
    if not records:
        print("[error] 无记录可处理（给 --manifest 或 --inline-json）", file=sys.stderr)
        return 2

    conn = _db()
    total = {"promoted": 0, "linked": 0, "trace": 0, "records": 0}
    try:
        cur = conn.cursor()
        for rec in records:
            try:
                st = process_record(cur, rec, dry_run=args.dry_run)
            except Exception as e:  # noqa: BLE001 — 单条失败不拖垮整批，回滚该条
                conn.rollback()
                print(f"[fail] qid={rec.get('question_id')}: {e}", file=sys.stderr)
                continue
            if not args.dry_run:
                conn.commit()
            total["records"] += 1
            total["promoted"] += len(st["promoted"])
            total["linked"] += st["linked"]
            total["trace"] += st["trace"]
            tag = "[dry]" if args.dry_run else "[ok]"
            extra = f" 转正{st['promoted']}" if st["promoted"] else ""
            print(f"{tag} qid={st['qid']} role={rec.get('role')} 落链+{st['linked']} trace+{st['trace']}{extra}")
    finally:
        conn.close()

    print(f"\n==== 汇总 ====")
    print(f"记录 {total['records']} · 转正模型 {total['promoted']} · 落链 {total['linked']} "
          f"· trace {total['trace']}" + ("（dry-run，未写库）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
