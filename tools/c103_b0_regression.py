# -*- coding: utf-8 -*-
r"""PRD-C-103 批 0 · 举一反三「拆前回归网」基线采集器。

目的（§1 / AC1 / G0）：选 15 道真题母题→举一反三跑基线，把**当前行为**冻成回归网，
供后续难度接线(第 1 步)与引擎大重拆(PRD-C-104)比对「有没有回归」。本脚本**只建 harness +
跑基线，绝不改引擎业务代码**（variant.py / mother_opus.py 只读 import）。

每道母题落盘 artifacts/regression_baseline/<qid>.json，schema 照 PRD §10：
  {
    mother_id, source,
    frozen_assertions: {                         # 改难度时必须不回归（≥5 类）
      gateA:   [{idx, gate, flags}...],          # 闸A 基因闸判决（item.gene）
      gateB:   [{idx, verify, badge}...],        # 闸B 验算判决（item.check）
      surface_check: [{idx, verdict}...],        # 防撞母题（_surface_check 纯函数重算）
      conservation: {main_kp, grade, injected_kps:[...]},  # 守恒（主考点/年级/各变式注入副kp）
      stem_hash: [sha256...]                      # 每变式净化后题面字符级 hash
    },
    difficulty_snapshot: { variant_levels: [{idx, difficulty, level}...] }  # 旧 LLM 自评档，只存不 assert
    variant_count
  }

🔴 驱动方式 = **in-process 直驱 graph**（复用 c018_benchmark_replay_v2 范式）：注入 mother_confirmed=True
   + analysis(三锚高置信 + kp.anchored.code) + mother_dna → route_entry 直奔 generate→gene_gate→
   solve_explain→assemble，绕开读图/母题确认 SSE 流（那条流当前在 :8093 实例 0 变式，不适合做回归基线）。
   身份走 _probe_auth 真登录（graph 入口硬闸 require_login）。不入库（不调 persist）。

跑法（cwd = toolkit 根；book-server :8090/:8080 在跑，relay 可达，.venv 解释器）：
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/c103_b0_regression.py
    --limit N        只跑前 N 道（调试）
    --concurrency K  并发母题数（默认 3）
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from _probe_auth import real_token  # noqa: E402
from agents.variant import (  # noqa: E402  (只读 import，零业务改动)
    _sanitize_rich_text,
    _surface_check,
    graph,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "regression_baseline"

# ---------------------------------------------------------------------------
# 15 道系统库母题（ai_lesson_prep biz_question status=1，select_mothers 选定后内联固化 →
# 脚本自包含、可复现；覆盖 选择/填空/解答 多题型 + 8+ 章节，浙教版七年级上册）。
# 每道：qid / qtype(中文) / kp_id(主考点叶子 code, 喂 generate 入口 dim1_kp_id 闸) / kp(叶子名) /
#       grade(年级册, 七上) / stem / answer / skeleton(取 analysis 作解法骨架供 verify 基准)。
# stem/answer/skeleton 在运行时从 selected_mothers.json 注入（见 _load_mothers），此处只列元信息。
# ---------------------------------------------------------------------------
MOTHER_META: list[dict] = [
    {"qid": "2069819158257750018", "qtype": "选择", "kp_id": "100001001001001", "kp": "自然数的应用"},
    {"qid": "2069819158840758274", "qtype": "选择", "kp_id": "100001001001001", "kp": "自然数的应用"},
    {"qid": "2069819178373632002", "qtype": "选择", "kp_id": "100001002002", "kp": "有理数与数轴上点的关系"},
    {"qid": "2069819178897920001", "qtype": "选择", "kp_id": "100001002002", "kp": "有理数与数轴上点的关系"},
    {"qid": "2069819189333348354", "qtype": "填空", "kp_id": "100001003002", "kp": "绝对值的性质"},
    {"qid": "2069819189845053441", "qtype": "填空", "kp_id": "100001003003", "kp": "与绝对值有关的计算"},
    {"qid": "2069819199483564034", "qtype": "填空", "kp_id": "100001004002", "kp": "利用法则比较大小"},
    {"qid": "2069819201052233730", "qtype": "填空", "kp_id": "100001004002", "kp": "利用法则比较大小"},
    {"qid": "2069819220148899841", "qtype": "解答", "kp_id": "100001005", "kp": "全章综合训练"},
    {"qid": "2069819220731908098", "qtype": "解答", "kp_id": "100001005", "kp": "全章综合训练"},
    {"qid": "2069819227585400834", "qtype": "解答", "kp_id": "100002001001002", "kp": "有理数加法的应用"},
    {"qid": "2069819420108148737", "qtype": "解答", "kp_id": "100004004002", "kp": "合并同类项"},
    {"qid": "2069819563083583490", "qtype": "解答", "kp_id": "100006003002", "kp": "尺规作一条线段等于已知线段"},
    {"qid": "2069819225031069697", "qtype": "解答", "kp_id": "100002001001001", "kp": "有理数的加法法则"},
    {"qid": "2069819240654852097", "qtype": "解答", "kp_id": "100002002001001", "kp": "有理数的减法法则"},
]
GRADE = "七年级上册"

# 3 道测试文件（§「15 道怎么选」）—— 本批**降级跳过**，原因见报告/SKIPPED_TEST_FILES。
# root cause：(1) variant.py route_entry 的 _extract_image_url 只认 http(s) 图片 URL，本地文件/
#   data:base64 URI 不被识别为题图，无法进 mother_opus_entry 读图入口；(2) 即便上 OSS 拿到 URL，
#   实测当前 :8093 实例的「读图→母题确认 resume→变式」SSE 路径产 0 变式（probe 已验：round2
#   stages=3 但 items=0），到不了 assemble、抽不全 frozen 栏。故按 PRD 允许「图题过不了→跳过、
#   用系统题补到 15」，系统题已满 15，回归网完整。图路径回归留 PRD-C-104 或读图链修复后补。
SKIPPED_TEST_FILES = [
    {"file": "06f6d4ceb369bc6b7df66791374707b4.jpg", "reason": "本地图无 OSS URL，route_entry 图门只认 http(s) 图片 URL"},
    {"file": "f599a64dea8a2c954db53d6760152bc9.jpg", "reason": "同上；且 OSS 图确认 resume 路径当前 0 变式"},
    {"file": "奕辰错题.pdf", "reason": "PDF 需切图为题图，同样卡在本地文件→无 URL 的图门"},
]


def _load_mothers(limit: int = 0) -> list[dict]:
    """合并 MOTHER_META + selected_mothers.json 的 stem/answer/analysis。

    selected_mothers.json 由 scratchpad/select_mothers.py 产（pymysql 直读 ai_lesson_prep）。
    缺该文件时报错退出（基线必须吃真题面，不接受占位）。
    """
    sel_path = Path(
        r"C:\Users\25606\AppData\Local\Temp\claude"
        r"\d--workplace-book-ai-codeplace-C\0f1a43b2-a08e-4ae0-a358-f3bae01dd991"
        r"\scratchpad\selected_mothers.json"
    )
    if not sel_path.exists():
        sys.exit(f"[FATAL] 缺 {sel_path}，先跑 scratchpad/select_mothers.py")
    sel = {p["qid"]: p for p in json.loads(sel_path.read_text(encoding="utf-8"))}
    out: list[dict] = []
    for meta in MOTHER_META:
        s = sel.get(meta["qid"])
        if not s:
            sys.exit(f"[FATAL] selected_mothers.json 缺 qid={meta['qid']}")
        out.append({
            **meta,
            "grade": GRADE,
            "stem": s["stem"],
            "answer": s.get("answer") or "",
            # analysis 作解法骨架供闸B verify 基准；缺则退 answer。
            "skeleton": s.get("analysis") or s.get("answer") or "",
        })
    return out[:limit] if limit else out


def _inject_state(m: dict) -> dict:
    """母题 → 注入 state（c018 范式 + C-103 修正）：

    🔴 generate 入口 L4067 防御断言要 dim1_kp_id 非空（C-014 批2 新增，c018 老驱动已被它打断）→
       analysis.kp.anchored.code 必须填真叶子 code（我们有 DB 的 dim1_kp_id）。
    """
    return {
        "analysis": {
            "grade": {"value": m["grade"], "confidence": 0.95},
            "kp": {
                "value": m["kp"], "confidence": 0.95,
                "anchored": {"code": m["kp_id"], "name": m["kp"]},
            },
            "qtype": {"value": m["qtype"], "confidence": 0.95},
        },
        "mother_dna": {
            "stem": m["stem"],
            "answer": m["answer"],
            "solution_skeleton": m["skeleton"],
            "difficulty": 2,
            # dna 守恒白名单（主考点 = 母题主 kp；变式 injected_kp 须 ⊆ 此集）
            "dna": {
                "main_kp": {"id": m["kp_id"], "name": m["kp"]},
                "secondary_kps": [],
                "qtype": m["qtype"],
            },
        },
        "mother_confirmed": True,
        "knobs": {},
        "items": [],
    }


def _stem_hash(stem: str) -> str:
    """净化后题面字符级 hash（§10 stem_hash）：复用引擎 _sanitize_rich_text（与入库同口径），
    sha256 hex。冻结后改净化逻辑/题面一字不差才不回归。"""
    cleaned = _sanitize_rich_text(stem or "")
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def _capture(m: dict, items: list[dict]) -> dict:
    """从 assemble 后的 items 抽双栏。frozen 5 类全从真实 state 抽，零伪造。"""
    gateA, gateB, surface, stem_hashes, inj_kps, levels = [], [], [], [], [], []
    mother_stem = m["stem"]
    for idx, it in enumerate(items):
        gene = it.get("gene") or {}
        chk = it.get("check") or {}
        gateA.append({"idx": idx, "gate": gene.get("gate"), "flags": sorted(gene.get("flags") or [])})
        gateB.append({"idx": idx, "verify": chk.get("verify"), "badge": chk.get("badge")})
        # 防撞母题：纯函数重算（None=无撞，否则 defect 串）→ 冻成 verdict
        sdef = _surface_check(it.get("stem"), mother_stem, it.get("qtype"))
        surface.append({"idx": idx, "verdict": "clean" if sdef is None else sdef})
        stem_hashes.append(_stem_hash(it.get("stem") or ""))
        ik = it.get("injected_kp")
        if ik and str(ik).strip().lower() not in ("null", "none", ""):
            inj_kps.append(str(ik).strip())
        levels.append({
            "idx": idx,
            "difficulty": it.get("difficulty"),
            "level": it.get("level") or "normal",
        })
    return {
        "mother_id": m["qid"],
        "source": f"ai_lesson_prep:biz_question:{m['qid']}",
        "mother_meta": {"qtype": m["qtype"], "grade": m["grade"], "main_kp": m["kp"], "kp_id": m["kp_id"]},
        "variant_count": len(items),
        "frozen_assertions": {
            "gateA": gateA,
            "gateB": gateB,
            "surface_check": surface,
            "conservation": {
                "main_kp": m["kp"],
                "grade": m["grade"],
                "injected_kps": sorted(set(inj_kps)),
            },
            "stem_hash": stem_hashes,
        },
        "difficulty_snapshot": {"variant_levels": levels},
    }


async def run_one(app, m: dict, token: str) -> dict:
    tid = f"c103-b0-{m['qid']}"
    cfg = {"configurable": {"thread_id": tid, "ruoyi_token": token}}
    init = _inject_state(m)
    init["messages"] = [HumanMessage(content="请基于已确认的母题，出一组举一反三变式题。")]
    t0 = time.monotonic()
    try:
        await app.ainvoke(init, cfg)
    except Exception as e:  # noqa: BLE001
        return {"mother_id": m["qid"], "_error": f"{type(e).__name__}: {e}", "variant_count": 0}
    st = app.get_state(cfg).values
    items = st.get("items") or []
    cap = _capture(m, items)
    cap["_elapsed_s"] = round(time.monotonic() - t0, 1)
    return cap


def _compare_to_baseline(results: list[dict], out_dir: Path) -> int:
    """🔴 PRD-C-103 批4·AC12 `--compare` 模式：与冻结基线 `artifacts/regression_baseline/` 比对。

    - `frozen_assertions`（闸A/闸B/防撞/守恒/stem_hash）**逐道 assert 相等**——任一道破 = 回归（RED）。
    - `difficulty_snapshot` **只 diff 不 assert**（WS1 接线后难度档预期变，G-REG 允许）。
    本函数**只读基线、写临时目录（out_dir）**，绝不碰 `OUT_DIR`（基线）—— 复跑安全。
    """
    rc = 0  # 红计数（frozen 破）
    diff_diff = 0  # difficulty 出 diff 的道数（预期，不算红）
    miss = 0
    print("\n========== AC12 frozen 比对（--compare，不碰基线）==========", flush=True)
    for r in results:
        qid = r["mother_id"]
        base_p = OUT_DIR / f"{qid}.json"
        if not base_p.exists():
            print(f"  [MISS] {qid} 基线缺文件 → 无法比对", flush=True)
            miss += 1
            continue
        base = json.loads(base_p.read_text(encoding="utf-8"))
        cur_fa = r.get("frozen_assertions") or {}
        base_fa = base.get("frozen_assertions") or {}
        if r.get("_error"):
            print(f"  [RED ] {qid} 当前跑出错: {r.get('_error')}", flush=True)
            rc += 1
            continue
        # frozen 逐类 assert 相等
        broken = [k for k in ("gateA", "gateB", "surface_check", "conservation", "stem_hash")
                  if cur_fa.get(k) != base_fa.get(k)]
        if broken:
            rc += 1
            print(f"  [RED ] {qid} frozen 破: {broken}", flush=True)
            for k in broken:
                print(f"          基线[{k}]={json.dumps(base_fa.get(k), ensure_ascii=False)[:200]}", flush=True)
                print(f"          当前[{k}]={json.dumps(cur_fa.get(k), ensure_ascii=False)[:200]}", flush=True)
        else:
            # difficulty 只 diff 不 assert
            cur_d = (r.get("difficulty_snapshot") or {}).get("variant_levels")
            base_d = (base.get("difficulty_snapshot") or {}).get("variant_levels")
            if cur_d != base_d:
                diff_diff += 1
                print(f"  [OK·Δ难] {qid} frozen 全等；difficulty 出 diff（预期）", flush=True)
                print(f"          基线难度={json.dumps(base_d, ensure_ascii=False)}", flush=True)
                print(f"          当前难度={json.dumps(cur_d, ensure_ascii=False)}", flush=True)
            else:
                print(f"  [OK ] {qid} frozen 全等；difficulty 无变化", flush=True)
    ran = len([r for r in results if not r.get("_error")])
    green = (rc == 0) and (miss == 0)
    print(f"\nAC12 G-REG: 跑 {len(results)} 道 | frozen 破 {rc} 道 | 难度 diff {diff_diff} 道 | 基线缺 {miss} 道 "
          f"→ {'GREEN(frozen 不破)' if green else 'RED'}", flush=True)
    print(f"临时产物目录: {out_dir}（基线 {OUT_DIR} 未动）", flush=True)
    return 0 if green else 1


async def main() -> int:
    ap = argparse.ArgumentParser(description="PRD-C-103 批0 回归网基线采集 / 批4 --compare 比对")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--compare", action="store_true",
                    help="🔴 AC12 比对模式：写临时目录 + 与冻结基线 diff(frozen assert 相等 / "
                         "difficulty 只 diff)，绝不覆盖基线 artifacts/regression_baseline/。")
    ap.add_argument("--out-dir", type=str, default="",
                    help="--compare 时临时产物目录（默认 artifacts/regression_compare/）。")
    args = ap.parse_args()

    mothers = _load_mothers(args.limit)
    # 🔴 --compare 模式：写临时目录，绝不碰基线 OUT_DIR；非 compare：写基线（原行为）。
    if args.compare:
        write_dir = Path(args.out_dir) if args.out_dir else (
            Path(__file__).resolve().parent.parent / "artifacts" / "regression_compare")
    else:
        write_dir = OUT_DIR
    write_dir.mkdir(parents=True, exist_ok=True)
    token = await real_token()
    app = graph.compile(checkpointer=MemorySaver())
    sem = asyncio.Semaphore(max(1, args.concurrency))

    print(f"=== C-103 批0 回归网：{len(mothers)} 道系统库母题 并发={args.concurrency} ===", flush=True)

    async def _guarded(m):
        async with sem:
            print(f"--- [{m['qid']}] {m['qtype']}/{m['kp']} 开始 ...", flush=True)
            r = await run_one(app, m, token)
            n = r.get("variant_count", 0)
            err = r.get("_error")
            print(f"--- [{m['qid']}] 完成：{n} 变式 {r.get('_elapsed_s','?')}s"
                  f"{' ERROR='+err if err else ''}", flush=True)
            return r

    results = await asyncio.gather(*(_guarded(m) for m in mothers))

    # 落盘每道（compare 模式落 write_dir=临时目录；基线模式落 OUT_DIR）
    written = []
    for r in results:
        p = write_dir / f"{r['mother_id']}.json"
        p.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(p.name)

    # 🔴 --compare：与基线 diff（frozen assert 相等 / difficulty 只 diff），不跑 G0 采集自检。
    if args.compare:
        return _compare_to_baseline(results, write_dir)

    # 跳过台账（仅基线采集模式写）
    (OUT_DIR / "_SKIPPED_test_files.json").write_text(
        json.dumps({"skipped": SKIPPED_TEST_FILES, "backfilled_with": "15 system mothers"},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    # G0 自检
    print("\n========== G0 自检 ==========", flush=True)
    ok_files = 0
    for r in results:
        fa = r.get("frozen_assertions") or {}
        classes = [k for k in ("gateA", "gateB", "surface_check", "conservation", "stem_hash") if k in fa]
        has_variants = r.get("variant_count", 0) > 0
        both_cols = bool(fa) and bool(r.get("difficulty_snapshot"))
        ge5 = len(classes) >= 5
        ok = has_variants and both_cols and ge5 and not r.get("_error")
        ok_files += 1 if ok else 0
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {r['mother_id']} 变式={r.get('variant_count')} frozen类={len(classes)} "
              f"双栏={'齐' if both_cols else '缺'}", flush=True)
    g0_green = (ok_files == 15) and (len(written) == 15)
    print(f"\nG0: 落盘 {len(written)}/15 文件，每文件双栏齐 + frozen≥5 类 → "
          f"{ok_files}/15 OK → {'GREEN' if g0_green else 'RED'}", flush=True)
    print(f"产物目录: {OUT_DIR}", flush=True)
    return 0 if g0_green else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(asyncio.run(main()))
