"""PRD-C-104：C-104 重构 4 道精测跑闸工具（B2-B5 复用）。

复用 `c103_b0_regression` 的 harness（零改 harness 逻辑）：把 `MOTHER_META` 过滤成 4 道
目标 qid，再以 `--compare` 模式（写临时 out-dir、不碰基线）调其 `main()`，与冻结基线
`artifacts/regression_baseline/` 做结构比对（守恒必等 + 闸无新 fail 类 + 防撞无新 defect 类 +
仍出题）。GREEN ⇒ 重构纯搬零改、frozen 结构不破。

跑法（cwd=agent-service-toolkit）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tools/c104_gate.py

返回码 0=GREEN / 非 0=RED（透传 c103 _compare_to_baseline 的判决）。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# PYTHONPATH=src 兜底（直跑也能 import agents.*）
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import c103_b0_regression as H  # noqa: E402  harness：零改复用

# 🔴 C-104 4 道精测目标 qid（PRD-C-104 §3：B1-B5 全程精测 ≤6，本批用这 4 道）。
#   选 4 类各一道：选择(自然数)/填空(绝对值)/解答(全章综合)/解答(尺规作图)，覆盖
#   route/solve/label/generate/gates/assemble 主链 + 富文本/几何分支。
TARGET_QIDS = [
    "2069819158257750018",  # 选择·自然数的应用
    "2069819189333348354",  # 填空·绝对值的性质
    "2069819220148899841",  # 解答·全章综合训练
    "2069819563083583490",  # 解答·尺规作一条线段等于已知线段
]


def main() -> int:
    # 过滤 harness 的 MOTHER_META → 仅 4 道目标（顺序按 TARGET_QIDS，落盘/比对都只走这 4 道）。
    by_qid = {m["qid"]: m for m in H.MOTHER_META}
    missing = [q for q in TARGET_QIDS if q not in by_qid]
    if missing:
        print(f"[FATAL] MOTHER_META 缺目标 qid: {missing}", flush=True)
        return 2
    H.MOTHER_META = [by_qid[q] for q in TARGET_QIDS]

    # 临时 out-dir（写产物不碰 baseline；--compare 只读基线、写这里）。
    out_dir = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "regression_compare_c104"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 复用 harness main()：以 --compare 模式跑（写临时目录，与冻结基线结构比对）。
    sys.argv = [
        "c104_gate",
        "--compare",
        "--out-dir",
        str(out_dir),
        "--concurrency",
        os.getenv("C104_CONCURRENCY", "2"),
    ]
    print(f"=== PRD-C-104 4 道精测跑闸（compare 不碰基线）：{TARGET_QIDS} ===", flush=True)
    return asyncio.run(H.main())


if __name__ == "__main__":
    raise SystemExit(main())
