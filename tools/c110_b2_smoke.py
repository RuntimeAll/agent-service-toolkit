# -*- coding: utf-8 -*-
"""PRD-C-110 B2·渲染切 smoke：验「端点产 DSL 过校验」+「DSL→exportPNG 出 PNG」两段机制。

不依赖在线 LLM / 不起服务（CI 友好）：用 mock invoke 注入一份合 schema 的 DSL，断言：
  1) compose_variant_dsl 返回 ok=True + dsl 过 dsl_schema.validate_dsl（端点产 DSL 这段机制）。
  2) 脏 DSL（type 越白名单）→ ok=False + needs_figure（降级闸生效，不让脏 DSL 进渲染端）。
  3) 纯文本/无几何关键词 + opus 不给 objects → needs_figure=False（不误催配图，承 BUG-B 精神）。
  4) DSL → PNG：客户端 exportPNG 走 JSXGraph（浏览器内），无头环境无法跑——故这里只静态校验
     DSL 结构可被 geo-dsl-render.js 的 switch 消费（type 全在白名单），PNG 出图由 book-ui
     e2e（GeoEngineTest 页 / 举一反三真机）覆盖，见回报⑤。

跑法（cwd=agent-service-toolkit，必 .venv）：
  .venv\\Scripts\\python.exe tools\\c110_b2_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.figure import compose, dsl_schema  # noqa: E402

# 一份合 schema 的中性 DSL（平面几何：三角形 + 外接圆 + 中点，含可拖红点）。
GOOD_DSL = {
    "bbox": [-5, 5, 6, -3],
    "objects": [
        {"id": "A", "type": "point", "coords": [-2, -1], "draggable": True, "label": "A"},
        {"id": "B", "type": "point", "coords": [3, -1.5], "draggable": True, "label": "B"},
        {"id": "C", "type": "point", "coords": [0.5, 3], "draggable": True, "label": "C"},
        {"id": "tri", "type": "polygon", "points": ["A", "B", "C"]},
        {"id": "M", "type": "midpoint", "points": ["A", "B"]},
        {"id": "oc", "type": "circumcircle", "points": ["A", "B", "C"]},
    ],
}
# 脏 DSL：type 越白名单（应被 schema 闸挡）。
BAD_DSL = {"objects": [{"id": "x", "type": "wormhole", "coords": [0, 0]}]}


def _mock_invoke_factory(payload: dict | None):
    async def _inv(messages, **kw):  # noqa: ANN001
        return json.dumps(payload) if payload is not None else "{}"
    return _inv


def _parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


async def main() -> int:
    fails: list[str] = []

    # 1) 好 DSL → ok + 过校验
    r1 = await compose.compose_variant_dsl(
        stem="如图，△ABC 内接于圆 O，求证……", invoke=_mock_invoke_factory(GOOD_DSL),
        parse_json=_parse_json, item_id="t1",
    )
    ok_schema, errs = dsl_schema.validate_dsl(r1.get("dsl") or {})
    if not (r1.get("ok") and r1.get("dsl") and ok_schema):
        fails.append(f"[1] 好 DSL 应 ok+过校验，实得 ok={r1.get('ok')} schema_ok={ok_schema} errs={errs}")
    else:
        print(f"[1] PASS 好 DSL ok + 过 schema（{len((r1['dsl'] or {}).get('objects', []))} objects）")

    # 2) 脏 DSL → 降级（ok=False + needs_figure）
    r2 = await compose.compose_variant_dsl(
        stem="如图三角形 ABC……", invoke=_mock_invoke_factory(BAD_DSL),
        parse_json=_parse_json, item_id="t2",
    )
    if r2.get("ok") or not r2.get("needs_figure"):
        fails.append(f"[2] 脏 DSL 应降级 ok=False+needs_figure，实得 {r2}")
    else:
        print(f"[2] PASS 脏 DSL 降级（needs_figure，schema_errs={r2.get('schema_errs')}）")

    # 3) 纯文本（无几何关键词）+ opus 不给 objects → 不误催配图
    r3 = await compose.compose_variant_dsl(
        stem="计算 3+5 的值。", invoke=_mock_invoke_factory({}),
        parse_json=_parse_json, item_id="t3",
    )
    if r3.get("ok") or r3.get("needs_figure"):
        fails.append(f"[3] 纯文本应 needs_figure=False（不误催），实得 {r3}")
    else:
        print("[3] PASS 纯文本不误催配图（needs_figure=False）")

    # 4) DSL 结构静态可消费性：好 DSL 每个 type 都在 geo-dsl-render switch 白名单内。
    bad_types = [
        o.get("type") for o in GOOD_DSL["objects"]
        if o.get("type") not in dsl_schema.WHITELIST
    ]
    if bad_types:
        fails.append(f"[4] 好 DSL 含白名单外 type（渲染端会 warn）：{bad_types}")
    else:
        print("[4] PASS 好 DSL 全 type 在白名单（geo-dsl-render switch 可消费 → 客户端可渲 SVG/exportPNG）")

    if fails:
        print("\n=== FAIL ===")
        for f in fails:
            print(" -", f)
        return 1
    print("\n=== ALL PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
