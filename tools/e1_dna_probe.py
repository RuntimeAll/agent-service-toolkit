"""E1 · 前置打标模拟探针（20-举一反三-DNA维度裁定.md §5）。

模拟「analyze 即打标」：c018 的 10 道文字母题 → 两步锚定（年级→叶子池）+ 全维 DNA
抽取（主/副kp 池里选、考察类型闭集、骨架标最难步、难点克制、标签 top300 复用池）。
难度由**代码**从难点数派生（0→1/1→2/2→3/>2→4），不信 LLM 自报。

产出 tools/_e_probe/e1_result.json + 控制台人审表：
  主/副kp 越界率 / 难点克制度（基础题是否空） / 派生难度 vs c018 人工难度 / 标签复用率。

跑法（cwd = toolkit 根；只调 LLM，不需要 :8090/:8093）：
  $env:PYTHONIOENCODING='utf-8'; .venv/Scripts/python.exe tools/e1_dna_probe.py --limit 2
  全量: .venv/Scripts/python.exe tools/e1_dna_probe.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
ASSET = TOOLS_DIR / "_e_probe"
sys.path.insert(0, str(TOOLS_DIR))

from c018_benchmark_replay import MOTHERS  # noqa: E402

# ---------------------------------------------------------------------------
# LLM 出口：直读 .env 的 RELAY_POOL[0]（aigeek 主站），OpenAI 兼容
# ---------------------------------------------------------------------------
def _load_relay() -> dict:
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("RELAY_POOL="):
            pool = json.loads(line.split("=", 1)[1])
            return pool[0]
    raise RuntimeError("RELAY_POOL 不在 .env 里")


# ---------------------------------------------------------------------------
# 资产：年级叶子池 + 标签池
# ---------------------------------------------------------------------------
L1_BY_NAME = {
    "七年级上册": "3071", "七年级下册": "3072",
    "八年级上册": "3081", "八年级下册": "3082",
    "九年级上册": "3091", "九年级下册": "3092",
}


def _grade_to_l1(grade: str) -> str | None:
    g = grade.replace("学期", "册")
    return L1_BY_NAME.get(g)


def _load_leaves() -> dict[str, list[tuple[str, str]]]:
    """kp_leaves.tsv（id \t l1 \t name）→ {l1: [(id,name)]}"""
    pools: dict[str, list[tuple[str, str]]] = {}
    lines = (ASSET / "kp_leaves.tsv").read_text(encoding="utf-8").splitlines()[1:]
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) == 3:
            pools.setdefault(parts[1], []).append((parts[0], parts[2]))
    return pools


def _load_top_tags() -> list[str]:
    lines = (ASSET / "top_tags.tsv").read_text(encoding="utf-8").splitlines()[1:]
    return [ln.split("\t")[0] for ln in lines if ln.strip()]


def _load_all_tags() -> set[str]:
    lines = (ASSET / "all_tags.txt").read_text(encoding="utf-8").splitlines()[1:]
    return {ln.strip() for ln in lines if ln.strip()}


EXAM_TYPES = [
    "概念辨析", "直接计算", "公式套用", "性质判定", "证明推理",
    "应用建模", "作图", "探究归纳", "阅读理解迁移", "纠错",
]

E1_PROMPT = """你是浙教版初中数学命题专家。对下面这道母题做**打标式 DNA 抽取**，只输出一个 JSON。

【母题】年级：{grade}　题型：{qtype}
题干：{stem}
标准答案：{answer}

【知识点候选池】（该年级全部叶子知识点，主/副知识点**只能从池里选 id，禁止造词、禁止超纲**）：
{kp_pool}

【标签复用池】（线上高频标签，优先从中复用；确实没有贴切的才允许造新词，新词必须像池内词一样短）：
{tag_pool}

输出 JSON 结构：
{{
  "main_kp": {{"id": "池内id", "name": "池内名"}},
  "secondary_kps": [{{"id": "...", "name": "..."}}],   // 0~3 个，解这道题连带必须用到的其他知识点；没有就空数组
  "exam_type": "{exam_types}之一",
  "skeleton": ["步骤1", "步骤2", ...],                  // 解法骨架；把最难的那一步用【】整步包住，如 "【构造全等三角形】"
  "hard_points": ["..."],                               // 🔴 难点=让题目变质的突破口，**只有进阶题才有**。
                                                        // 基础知识考察/纯套公式/直接计算/概念辨析的题必须给空数组 []，宁空不凑。
  "tags": ["...", "..."],                               // 3~6 个检索标签（求什么/用什么定理/什么方法/什么场景）
  "scene": "纯代数 或 一句话场景描述"
}}
不要输出难度——难度由代码从难点个数派生，轮不到你评。不要任何解释文字。"""


def _parse_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(json)?\s*|\s*```$", "", t, flags=re.S)
    m = re.search(r"\{.*\}", t, flags=re.S)
    return json.loads(m.group(0) if m else t)


def derive_difficulty(n_hard: int) -> int:
    return {0: 1, 1: 2, 2: 3}.get(n_hard, 4)


async def _one(client, model: str, m: dict, pools, top_tags, sem) -> dict:
    l1 = _grade_to_l1(m["grade"])
    pool = pools.get(l1, [])
    pool_ids = {pid for pid, _ in pool}
    kp_pool_text = "\n".join(f"{pid} {name}" for pid, name in pool)
    prompt = E1_PROMPT.format(
        grade=m["grade"], qtype=m["qtype"], stem=m["stem"], answer=m["answer"],
        kp_pool=kp_pool_text, tag_pool="、".join(top_tags),
        exam_types="/".join(EXAM_TYPES),
    )
    async with sem:
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
    dna = _parse_json(resp.choices[0].message.content)

    # --- 代码校验闸（v3 铁律：id 绑定不跑偏） ---
    flags: list[str] = []
    mk = dna.get("main_kp") or {}
    if str(mk.get("id")) not in pool_ids:
        flags.append(f"主kp越界:{mk}")
    valid_sec = []
    for s in dna.get("secondary_kps") or []:
        if str(s.get("id")) in pool_ids:
            valid_sec.append(s)
        else:
            flags.append(f"副kp越界丢弃:{s}")
    dna["secondary_kps"] = valid_sec
    if dna.get("exam_type") not in EXAM_TYPES:
        flags.append(f"考察类型出闭集:{dna.get('exam_type')}")
    hard = dna.get("hard_points") or []
    dna["derived_difficulty"] = derive_difficulty(len(hard))
    usage = getattr(resp, "usage", None)
    return {
        "name": m["name"], "grade": m["grade"], "manual_difficulty": m["difficulty"],
        "dna": dna, "flags": flags,
        "tokens": {"in": getattr(usage, "prompt_tokens", None), "out": getattr(usage, "completion_tokens", None)},
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=len(MOTHERS))
    args = ap.parse_args()

    from openai import AsyncOpenAI

    relay = _load_relay()
    client = AsyncOpenAI(base_url=relay["base_url"], api_key=relay["api_key"])
    pools, top_tags, all_tags = _load_leaves(), _load_top_tags(), _load_all_tags()
    mothers = MOTHERS[: max(1, args.limit)]
    sem = asyncio.Semaphore(4)

    results = await asyncio.gather(
        *(_one(client, relay["model"], m, pools, top_tags, sem) for m in mothers),
        return_exceptions=True,
    )

    rows, errs = [], []
    for m, r in zip(mothers, results):
        if isinstance(r, Exception):
            errs.append({"name": m["name"], "error": repr(r)})
        else:
            rows.append(r)

    # --- 人审表 ---
    n_oob = sum(1 for r in rows if any("越界" in f for f in r["flags"]))
    tag_total = tag_reused = 0
    print(f"\n=== E1 前置打标模拟：{len(rows)} 道成功 / {len(errs)} 道失败 ===")
    for r in rows:
        d = r["dna"]
        tags = d.get("tags") or []
        tag_total += len(tags)
        tag_reused += sum(1 for t in tags if t in all_tags)
        sec = "、".join(s["name"] for s in d["secondary_kps"]) or "—"
        hard = d.get("hard_points") or []
        diff_mark = "✓" if d["derived_difficulty"] == r["manual_difficulty"] else f"≠人工{r['manual_difficulty']}"
        print(f"\n[{r['name']}]  主kp={(d.get('main_kp') or {}).get('name')}  副kp={sec}")
        print(f"  考察类型={d.get('exam_type')}  难点={len(hard)}个→派生难度{d['derived_difficulty']}({diff_mark})")
        if hard:
            print(f"  难点: {' | '.join(hard)}")
        print(f"  骨架: {' → '.join(d.get('skeleton') or [])[:120]}")
        print(f"  标签: {'、'.join(tags)}")
        if r["flags"]:
            print(f"  ⚠ {r['flags']}")
    for e in errs:
        print(f"\n[{e['name']}] ❌ {e['error']}")

    print(f"\n--- 汇总：kp越界题数={n_oob}/{len(rows)}  标签复用率={tag_reused}/{tag_total}"
          f"  难度命中={sum(1 for r in rows if r['dna']['derived_difficulty'] == r['manual_difficulty'])}/{len(rows)}（人工难度为参考非真值）---")

    out = ASSET / "e1_result.json"
    out.write_text(json.dumps({"rows": rows, "errors": errs}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {out}")


if __name__ == "__main__":
    asyncio.run(main())
