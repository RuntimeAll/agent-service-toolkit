"""E2 · DNA 强约束出题模拟探针（20-举一反三-DNA维度裁定.md §5）。

吃 E1 的 DNA（e1_result.json），按「前置打标 → 强约束出题 → 代码三检 + 闸B sympy」
模拟未来编排：每道母题出 3 道变式（normal×2 + hard×1），出题 prompt 注入
主kp + 知识点守恒白名单 + 考察类型 + 解法骨架基因 + 题型契约 + 验算载荷契约，
然后**零回炉零裁判**直接验：
  ① 题型 lint（代码）② 表皮距离（代码，防抄母题）③ 闸B 同款 sympy 验算（载荷优先）。

指标 = 一次通过 sympy_pass 率，对照 c018 全管线（含回炉自愈）基线 55.6%。

跑法（cwd = toolkit 根；不需要 :8090/:8093，只调 LLM + 本地 sympy）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/e2_constrained_gen_probe.py --limit 2
  全量: .venv/Scripts/python.exe tools/e2_constrained_gen_probe.py
"""

import argparse
import asyncio
import difflib
import json
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
ASSET = TOOLS_DIR / "_e_probe"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(ROOT / "src"))

from c018_benchmark_replay import MOTHERS  # noqa: E402
from e1_dna_probe import _load_relay  # noqa: E402


def _parse_json(text: str):
    """E2 版：输出是数组优先（[{...},{...}]），对象兜底。"""
    t = re.sub(r"^```(json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    m = re.search(r"\[.*\]", t, flags=re.S) or re.search(r"\{.*\}", t, flags=re.S)
    return json.loads(m.group(0) if m else t)

from agents.variant import (  # noqa: E402
    _PAYLOAD_CONTRACT,
    _QTYPE_CONTRACT,
    _is_proof_like,
    _machine_verify,
    _proof_struct_ok,
)

E2_PROMPT = (
    """你是浙教版初中数学命题专家。下面给你一道母题和它的**题目 DNA**（已锚定知识图谱），
请出 {n} 道平行变式题（{level_plan}）。

【母题】年级：{grade}　题型：{qtype}
题干：{stem}
标准答案：{answer}

【题目 DNA（基因 = 必须守恒；表皮 = 必须更换）】
- 主知识点（必考）：{main_kp}
- 🔴 知识点守恒白名单：解题所需知识点必须 ⊆ {{{kp_whitelist}}}，禁止引入白名单外的知识点（超纲即废）。
- 考察类型（守恒）：{exam_type}
- 解法骨架（基因，变式必须按同一骨架可解；【】内是最难步，必须保留同类挑战）：
{skeleton}
- 场景/表皮（必换）：数字必须全换且解依然整洁；场景可换同类（{scene}）。
- hard 档变式：在同一骨架上**增加一个真实突破口**（如需一步构造/转化/分类讨论），不是把数字变丑。

每道题输出字段：stem / answer / solution / qtype（与母题同）/ level（normal|hard）/ verify_payload。

"""
    + _QTYPE_CONTRACT
    + "\n\n"
    + _PAYLOAD_CONTRACT
    + """

抽不成载荷（应用题难建模/几何证明等）→ verify_payload 填 {{"kind":"none","reason":"原因"}}。
只输出 JSON 数组（{n} 个对象），不要解释。格式硬规定：数学式一律 $...$ 包裹，换行用标准 \\n。"""
)


def _norm_stem(s: str) -> str:
    return re.sub(r"[\s$\\{}]+", "", str(s or "")).lower()


def _lint_qtype(item: dict, qtype: str) -> str | None:
    """题型 lint（代码，P11 同思路的探针版）。返回缺陷描述或 None。"""
    stem = str(item.get("stem") or "")
    ans = str(item.get("answer") or "").strip()
    if "选择" in qtype:
        opts = re.findall(r"[A-D][.、．]", stem)
        if len(set(opts)) != 4:
            return f"选择题选项数≠4（找到{sorted(set(opts))}）"
        if not re.fullmatch(r"[A-D]", re.sub(r"[$\s.。]", "", ans)):
            return f"选择题 answer 非字母（{ans[:20]}）"
        if re.search(r"[(（][123一二三][)）]|[①②③]", stem):
            return "选择题嵌多小问"
    if "填空" in qtype and not re.search(r"_{2,}|＿|[(（]\s*[)）]", stem):
        return "填空题无空位标记"
    return None


def _surface_check(item: dict, mother_stem: str) -> str | None:
    """表皮距离（代码）：题干过近 = 抄母题；数字完全相同也算。"""
    v, m = _norm_stem(item.get("stem")), _norm_stem(mother_stem)
    ratio = difflib.SequenceMatcher(None, v, m).ratio()
    if ratio > 0.85:
        return f"题干与母题相似度{ratio:.2f}>0.85（疑似抄题）"
    v_nums = re.findall(r"\d+(?:\.\d+)?", str(item.get("stem") or ""))
    m_nums = re.findall(r"\d+(?:\.\d+)?", mother_stem)
    if v_nums and v_nums == m_nums:
        return "数字与母题完全相同（表皮没换）"
    return None


async def _verify_item(item: dict, qtype: str) -> dict:
    """闸B 同款验算（零回炉）：证明类走结构软校验；其余 sympy（载荷优先）。"""
    if _is_proof_like(item.get("qtype") or qtype, item.get("stem")):
        ok = _proof_struct_ok(item.get("stem"))
        return {"route": "proof", "verdict": "struct_ok" if ok else "struct_bad"}
    res = await _machine_verify(item, solved_answer=None)
    return {"route": "sympy", "verdict": res.get("verdict"), "detail": (res.get("detail") or "")[:120]}


async def _one_mother(client, model: str, m: dict, dna_row: dict, sem) -> dict:
    dna = dna_row["dna"]
    wl = [(dna.get("main_kp") or {}).get("name")] + [s["name"] for s in dna.get("secondary_kps") or []]
    prompt = E2_PROMPT.format(
        n=3, level_plan="normal×2 + hard×1",
        grade=m["grade"], qtype=m["qtype"], stem=m["stem"], answer=m["answer"],
        main_kp=(dna.get("main_kp") or {}).get("name"),
        kp_whitelist="、".join(x for x in wl if x),
        exam_type=dna.get("exam_type"),
        skeleton="\n".join(f"  {i + 1}. {s}" for i, s in enumerate(dna.get("skeleton") or [])),
        scene=dna.get("scene") or "纯代数",
    )
    async with sem:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
    items = _parse_json(resp.choices[0].message.content)
    if not isinstance(items, list):
        items = [items]

    out = []
    for it in items:
        if not isinstance(it, dict) or not it.get("stem"):
            out.append({"defects": ["无效对象"], "verify": {"verdict": "invalid"}})
            continue
        defects = [d for d in (_lint_qtype(it, m["qtype"]), _surface_check(it, m["stem"])) if d]
        verify = await _verify_item(it, m["qtype"])
        out.append({
            "level": it.get("level"), "stem_head": str(it.get("stem"))[:50],
            "answer": str(it.get("answer"))[:40],
            "payload_kind": (it.get("verify_payload") or {}).get("kind"),
            "defects": defects, "verify": verify,
        })
    usage = getattr(resp, "usage", None)
    return {"name": m["name"], "qtype": m["qtype"], "items": out,
            "tokens": {"in": getattr(usage, "prompt_tokens", None), "out": getattr(usage, "completion_tokens", None)}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(MOTHERS))
    args = ap.parse_args()

    from openai import AsyncOpenAI

    e1 = json.loads((ASSET / "e1_result.json").read_text(encoding="utf-8"))
    dna_by_name = {r["name"]: r for r in e1["rows"]}
    relay = _load_relay()
    client = AsyncOpenAI(base_url=relay["base_url"], api_key=relay["api_key"])
    mothers = [m for m in MOTHERS[: max(1, args.limit)] if m["name"] in dna_by_name]
    sem = asyncio.Semaphore(3)

    results = await asyncio.gather(
        *(_one_mother(client, relay["model"], m, dna_by_name[m["name"]], sem) for m in mothers),
        return_exceptions=True,
    )

    rows, errs = [], []
    for m, r in zip(mothers, results):
        (errs if isinstance(r, Exception) else rows).append(
            {"name": m["name"], "error": repr(r)} if isinstance(r, Exception) else r
        )

    total = n_pass = n_fail = n_degrade = n_proof_ok = n_proof = n_defect = 0
    print(f"\n=== E2 强约束出题模拟（零回炉零LLM裁判）：{len(rows)} 母题成功 / {len(errs)} 失败 ===")
    for r in rows:
        print(f"\n[{r['name']}]（{r['qtype']}）")
        for it in r["items"]:
            total += 1
            v = it["verify"]
            if v.get("route") == "proof":
                n_proof += 1
                n_proof_ok += v["verdict"] == "struct_ok"
            else:
                n_pass += v.get("verdict") == "pass"
                n_fail += v.get("verdict") == "fail"
                n_degrade += v.get("verdict") == "degrade"
            if it["defects"]:
                n_defect += 1
            mark = {"pass": "✅", "fail": "❌", "degrade": "⚠", "struct_ok": "📐ok", "struct_bad": "📐bad"}.get(v.get("verdict"), "?")
            d = f"  缺陷:{it['defects']}" if it["defects"] else ""
            print(f"  {mark} [{it.get('level')}] payload={it.get('payload_kind')} {it.get('stem_head')}{d}")
            if v.get("verdict") in ("fail", "degrade"):
                print(f"      ↳ {v.get('detail')}")

    n_sympy = total - n_proof
    print(f"\n--- 汇总（对照 c018 全管线基线 55.6%=15/27，含回炉；本探针零回炉一次过）---")
    print(f"  总题数={total}（sympy 路由 {n_sympy} + 证明路由 {n_proof}）")
    if n_sympy:
        print(f"  sympy: pass={n_pass} fail={n_fail} degrade={n_degrade}  一次通过率={n_pass}/{n_sympy}={n_pass / n_sympy:.1%}")
    print(f"  全口径可验率（pass+proof_ok / total）={(n_pass + n_proof_ok)}/{total}={(n_pass + n_proof_ok) / max(total, 1):.1%}")
    print(f"  代码 lint 缺陷题数={n_defect}/{total}")

    out = ASSET / "e2_result.json"
    out.write_text(json.dumps({"rows": rows, "errors": errs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {out}")


if __name__ == "__main__":
    asyncio.run(main())
