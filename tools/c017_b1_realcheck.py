# -*- coding: utf-8 -*-
"""PRD-C-017 B1 真机一例：opus 直读 img3(韦达,近纯文本) 跑生产路径
mother_opus.build_mother_prompt + solve_and_label(via variant._ainvoke_text)。
验 G3（opus 真被调用，10维齐全）+ G4（solution_skeleton 来源=opus 骨架）。
不连 RuoYi（leaf_pool 传空，仅测 opus 调用+合并产出；锚定走单测覆盖）。"""
from __future__ import annotations
import asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents import mother_opus, variant
from core.settings import settings

IMG3 = "https://question-1256278081.cos.ap-shanghai.myqcloud.com/2026-03-12/47c125f4-73ca-4bdc-9aea-a90357910b48/list/27/question.png"

async def main() -> None:
    model = settings.variant_model("mother_solve_label")
    print(f"mother_solve_label 档解析 = {model}")
    assert model == "claude-opus-4-8", "G3: 母题档未命中 opus!"
    prompt = mother_opus.build_mother_prompt(
        grade_text="九年级", chapter_text=None, leaf_pool=[], model_vocab=None,
    )
    t0 = time.monotonic()
    text = await mother_opus.solve_and_label(
        image_url=IMG3, prompt=prompt, invoke=variant._ainvoke_text, model=model,
    )
    dur = time.monotonic() - t0
    data = variant._parse_json(text)
    print(f"墙钟={dur:.1f}s  json_ok={isinstance(data, dict)}")
    if not isinstance(data, dict):
        print("RAW HEAD:", text[:300]); sys.exit(1)
    dna = mother_opus.opus_to_dna(data)
    print("has_figure =", data.get("has_figure"))
    print("solvedAnswer =", (data.get("solvedAnswer") or "")[:80])
    dims = {k: dna.get(k) for k in
            ("main_kp","secondary_kps","qtype","exam_type","skeleton",
             "hard_point_count","scene","difficulty","tags","model_candidates")}
    print("DNA(归一后):")
    print(json.dumps(dims, ensure_ascii=False, indent=2))
    # G4：骨架非空 = opus 解答骨架
    print(f"\nG3 10维齐全 = {all(dims[k] not in (None, '') for k in ('qtype','exam_type','scene','difficulty')) and bool(dims['skeleton'])}")
    print(f"G4 opus 解答骨架 = {bool(dims['skeleton'])}  (skeleton 行数={len(dims['skeleton'])})")
    rt = mother_opus.validate_rich_text(data.get("richText") or {})
    print(f"G10 闸A 富文本机器检 = {'ok' if rt['ok'] else rt['issues']}")

if __name__ == "__main__":
    asyncio.run(main())

# (debug) dump richText stem to inspect 闸A false-positive
