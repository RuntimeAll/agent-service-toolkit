"""把 data/llm_trace.jsonl 的 LLM 往返记录打成人读格式，供过 prompt 用。

用法（cwd=toolkit）：
  .venv\\Scripts\\python.exe tools\\read_llm_trace.py            # 全部
  .venv\\Scripts\\python.exe tools\\read_llm_trace.py -n 5       # 最近 5 条
  .venv\\Scripts\\python.exe tools\\read_llm_trace.py -l generate # 只看某个 prompt(analyze/generate/solve/regen/parse/answer/add)
  .venv\\Scripts\\python.exe tools\\read_llm_trace.py --raw      # 连模型原始返回(含 reasoning)一起打
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TRACE = Path(__file__).resolve().parents[1] / "data" / "llm_trace.jsonl"
BAR = "=" * 78
SUB = "-" * 78


def _fmt_request(req: list[dict]) -> str:
    lines = []
    for m in req:
        role = m.get("role", "?")
        if "text" in m:
            lines.append(f"[{role}]\n{m['text']}")
        elif "parts" in m:
            for p in m["parts"]:
                if "text" in p:
                    lines.append(f"[{role}·text]\n{p['text']}")
                elif "image_url" in p:
                    lines.append(f"[{role}·image_url] {p['image_url']}")
                else:
                    lines.append(f"[{role}·raw] {p.get('raw')}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=0, help="只看最近 N 条(0=全部)")
    ap.add_argument("-l", "--label", default="", help="只看某个 prompt 标签")
    ap.add_argument("--raw", action="store_true", help="连模型原始返回一起打")
    args = ap.parse_args()

    if not TRACE.exists():
        print(f"还没有往返记录：{TRACE}\n（先在 book-ui 跑一次举一反三，或调 /compose，会自动落盘）")
        return

    rows = []
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    if args.label:
        rows = [r for r in rows if r.get("label") == args.label]
    if args.n > 0:
        rows = rows[-args.n :]

    print(f"# LLM 往返记录  共 {len(rows)} 条  来源 {TRACE}\n")
    for r in rows:
        print(BAR)
        print(
            f"#{r.get('seq')}  [{r.get('label')}]  {r.get('ts')}  "
            f"{r.get('duration_ms')}ms"
            + ("  (retried)" if r.get("retried") else "")
            + (f"  ERROR={r.get('error')}" if r.get("error") else "")
        )
        print(BAR)
        print(">>> 发送给 LLM（填充后的完整 prompt）:")
        print(_fmt_request(r.get("request") or []))
        print(SUB)
        print("<<< 模型返回（提取的 content）:")
        print(r.get("response") or "(空)")
        if args.raw and r.get("response_raw"):
            print(SUB)
            print("<<< 原始返回(raw, 含 reasoning/usage 等):")
            print(json.dumps(r["response_raw"], ensure_ascii=False, indent=2, default=str))
        print()


if __name__ == "__main__":
    main()
