# -*- coding: utf-8 -*-
"""PRD-C-110 B1 smoke：验「配图链产中性 DSL 过 schema 校验」+「纯文本不触发配图（承 BUG-B）」。

两件事（与 B1 验收口径对齐）：
  ① 配图链产 DSL：几道几何题（平面几何/数轴/函数/统计），走 dsl_system.DSL_SYSTEM 让 LLM 直出
     中性 JSON DSL → 过 dsl_schema.validate_dsl 闸（断言 type 白名单 + 必填字段 + 引用 id 可解析 +
     functiongraph.expr 数学白名单）。这条 = compose_variant_dsl 的 system+闸 同一套（LLM 调用走
     spike 同款 OpenAI 直连，避免拉起整个 agents 包/torch；逻辑与产品函数同源）。
  ② 纯文本不触发配图（BUG-B）：用 compose._has_keyword + _FIGURE_KEYWORDS 代码闸断言——纯文字/
     纯代数题（无几何关键词）→ want_fig=False → compose_variant_dsl「objects 缺失」分支返
     needs_figure=False（= 不需配图，非「待补图」）。不调 LLM（纯代码闸断言，确定性）。

🔴 B1 不切渲染端：本 smoke 只验「产 DSL + 校验 + 触发闸」，**不渲染、不出 PNG**。
跑法（cwd = agent-service-toolkit）：
    set NO_PROXY=*
    .venv\\Scripts\\python.exe tools\\c110_b1_smoke.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIG = os.path.join(_ROOT, "src", "agents", "figure")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


# 产品模块（自包含、无重依赖，文件路径直载）。
dsl_schema = _load("dsl_schema", os.path.join(_FIG, "dsl_schema.py"))
dsl_system = _load("dsl_system", os.path.join(_FIG, "dsl_system.py"))

# compose 的关键词代码闸（_has_keyword + _FIGURE_KEYWORDS）——直接读常量/函数验 BUG-B，
# 不整包 import（compose.py 顶层 import 重依赖；这里只取两符号做闸断言，从源码切出执行）。
_COMPOSE_SRC = open(os.path.join(_FIG, "compose.py"), encoding="utf-8").read()


def _extract_figure_gate():
    """从 compose.py 源码切出 _FIGURE_KEYWORDS 元组 + _has_keyword 函数，独立执行（避免重依赖 import）。"""
    ns: dict = {}
    # _FIGURE_KEYWORDS = ( ... )
    m_kw = re.search(r"_FIGURE_KEYWORDS\s*=\s*\((.*?)\)", _COMPOSE_SRC, re.S)
    assert m_kw, "未在 compose.py 找到 _FIGURE_KEYWORDS"
    exec("_FIGURE_KEYWORDS = (" + m_kw.group(1) + ")", ns)
    # def _has_keyword(...): ...
    m_fn = re.search(r"def _has_keyword\(.*?\n(?=\ndef )", _COMPOSE_SRC, re.S)
    assert m_fn, "未在 compose.py 找到 _has_keyword"
    exec(m_fn.group(0), ns)
    return ns["_FIGURE_KEYWORDS"], ns["_has_keyword"]


_FIGURE_KEYWORDS, _has_keyword = _extract_figure_gate()


# --------------------------------------------------------------------------- #
# .env / LLM 客户端（spike 同款：COMPATIBLE_* / RELAY_POOL 兜底）
# --------------------------------------------------------------------------- #
def load_env(path: str) -> dict:
    out: dict = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


ENV = load_env(os.path.join(_ROOT, ".env"))
BASE_URL = ENV.get("COMPATIBLE_BASE_URL") or ENV.get("LLM_BASE_URL") or "https://sui-xiang.com/v1"
API_KEY = ENV.get("COMPATIBLE_API_KEY") or ENV.get("LLM_API_KEY") or ""
MODEL = ENV.get("LLM_MODEL_LIGHT") or ENV.get("COMPATIBLE_MODEL") or "claude-opus-4-8"
if not API_KEY or len(API_KEY) < 20:
    try:
        pool = json.loads(ENV.get("RELAY_POOL", "[]"))
        if pool:
            API_KEY = pool[0]["api_key"]
            BASE_URL = pool[0]["base_url"]
            MODEL = pool[0].get("model", MODEL)
    except Exception:
        pass


def extract_json(text: str):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < 0:
        return None
    frag = s[a:b + 1]
    try:
        return json.loads(frag)
    except Exception:
        try:
            return json.loads(re.sub(r"//.*", "", frag))
        except Exception:
            return None


GEOM_CASES = [
    ("平面几何·三角形外接圆", "画三角形 ABC（A(0,0)、B(6,0)、C(2,4)）三顶点 + 三边，再画过 A、B、C 的外接圆。"),
    ("数轴·表示有理数", "一条数轴（范围 -5 到 5），标出 -2、0、3 三个点。"),
    ("函数·二次函数图象", "坐标系中画 y=x^2-2x-3 的抛物线，标出与 x 轴交点 (-1,0)、(3,0) 和顶点 (1,-4)。"),
    ("统计·扇形统计图", "一个圆（扇形统计图外圈），分三块扇形：50%(0°~180°)、30%(180°~288°)、20%(288°~360°)。"),
]

PURE_TEXT_CASES = [
    ("纯文本·解方程", "解方程 2x + 3 = 11，并写出解题过程。"),
    ("纯文本·应用题", "小明有 12 个苹果，分给 3 个同学，每人分得几个？"),
    ("纯文本·因式分解", "把多项式 x² + 5x + 6 因式分解。"),
]


def test_dsl_chain() -> bool:
    """① 几何题走 DSL_SYSTEM 产 DSL → schema 闸校验，断言全过。"""
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)
    print(f"[smoke①] model={MODEL} base={BASE_URL} key=...{API_KEY[-6:]}")
    all_ok = True
    for name, spec in GEOM_CASES:
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": dsl_system.DSL_SYSTEM},
                    {"role": "user", "content":
                     f"【配图需求】\n{spec}\n\n按 schema 直出中性 JSON DSL（只输出 JSON）。"},
                ],
                temperature=0.2, max_tokens=2000,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ {name}: LLM 调用失败 {e}")
            all_ok = False
            continue
        data = extract_json(text)
        if data is None:
            print(f"  ❌ {name}: JSON 解析失败")
            all_ok = False
            continue
        ok, errs = dsl_schema.validate_dsl(data)
        types = [o.get("type") for o in data.get("objects", []) if isinstance(o, dict)]
        print(f"  {'✅' if ok else '❌'} {name} ⏱{time.time()-t0:.1f}s schema={ok} types={types}")
        if not ok:
            for e in errs:
                print(f"       - {e}")
            all_ok = False
        else:
            # 额外硬断言：所有 type 在白名单、functiongraph.expr 走数学白名单。
            for o in data.get("objects", []):
                assert o.get("type") in dsl_schema.WHITELIST, f"{name} type 越白名单"
    return all_ok


def test_bug_b_pure_text() -> bool:
    """② 纯文本（无几何关键词）→ 代码闸 want_fig=False → 不触发配图（承 BUG-B）。"""
    print("[smoke②] BUG-B 纯文本不触发配图（代码闸断言）")
    all_ok = True
    for name, stem in PURE_TEXT_CASES:
        want_fig = _has_keyword(stem, _FIGURE_KEYWORDS)
        # compose_variant_dsl「objects 缺失」分支：needs_figure = bool(want_fig)。
        # 纯文本 want_fig 必须为 False → needs_figure False（不触发配图，非「待补图」）。
        ok = (want_fig is False)
        print(f"  {'✅' if ok else '❌'} {name}: want_fig={want_fig} (期望 False)")
        all_ok = all_ok and ok
    # 反向自检：几何题关键词必须命中（否则闸太松，BUG-B 反向漏）。
    geo_hit = _has_keyword("画三角形 ABC 的外接圆", _FIGURE_KEYWORDS)
    print(f"  {'✅' if geo_hit else '❌'} 反向自检·几何题命中关键词: want_fig={geo_hit} (期望 True)")
    all_ok = all_ok and geo_hit
    return all_ok


def main() -> None:
    print("=" * 64)
    r2 = test_bug_b_pure_text()  # 先跑确定性代码闸（不依赖 LLM）
    print("=" * 64)
    r1 = test_dsl_chain()
    print("#" * 64)
    print(f"# smoke① DSL 产出过 schema: {'PASS' if r1 else 'FAIL'}")
    print(f"# smoke② BUG-B 纯文本不触发: {'PASS' if r2 else 'FAIL'}")
    sys.exit(0 if (r1 and r2) else 1)


if __name__ == "__main__":
    main()
