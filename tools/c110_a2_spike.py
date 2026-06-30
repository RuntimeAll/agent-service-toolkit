# -*- coding: utf-8 -*-
"""PRD-C-110 A2 预飞行 spike：验「AI 稳定产出合 geo-engine schema 的中性几何 JSON DSL」。

决定 C-110 命令格式走路 A（AI 直出 DSL）还是路 B（GeoGebra 命令→DSL 翻译器过渡）。
临时 spike 脚本，不动产品配图代码（figure/ 只读）。

跑法（cwd = agent-service-toolkit）：
    set NO_PROXY=*
    .venv\\Scripts\\python.exe tools\\c110_a2_spike.py

schema 真源 = geo-engine/geo-dsl-render.js 的 switch(o.type)（白名单 + 各 type 必填字段）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

# Windows 控制台 GBK → 强制 UTF-8 输出（spike 含大量中文 + ✅⏱ 符号）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 绕本机代理（中转直连）
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"

from openai import OpenAI  # noqa: E402

# ---------------------------------------------------------------------------
# .env 读取（COMPATIBLE_* / LLM_MODEL_LIGHT / RELAY_POOL 兜底）
# ---------------------------------------------------------------------------
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_env(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


ENV = load_env(ENV_PATH)
BASE_URL = ENV.get("COMPATIBLE_BASE_URL") or ENV.get("LLM_BASE_URL") or "https://sui-xiang.com/v1"
API_KEY = ENV.get("COMPATIBLE_API_KEY") or ENV.get("LLM_API_KEY") or ""
MODEL = ENV.get("LLM_MODEL_LIGHT") or ENV.get("COMPATIBLE_MODEL") or "claude-opus-4-8"
# RELAY_POOL 兜底取第一站 key（.env COMPATIBLE_API_KEY 可能是占位）
if not API_KEY or len(API_KEY) < 20:
    try:
        pool = json.loads(ENV.get("RELAY_POOL", "[]"))
        if pool:
            API_KEY = pool[0]["api_key"]
            BASE_URL = pool[0]["base_url"]
            MODEL = pool[0].get("model", MODEL)
    except Exception:
        pass

print(f"[spike] model={MODEL} base_url={BASE_URL} key=...{API_KEY[-6:]}")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

# ---------------------------------------------------------------------------
# geo-engine schema（真源 = geo-dsl-render.js switch）：type 白名单 + 必填字段校验
# ---------------------------------------------------------------------------
# 各 type 的必填字段（按 buildObjects switch 实际读取的字段；id 对引用型必需）
TYPE_REQUIRED: dict[str, list[str]] = {
    "point": ["coords"],
    "segment": ["points"],
    "line": ["points"],
    "ray": ["points"],
    "vector": ["points"],
    "polygon": ["points"],
    "midpoint": ["points"],
    "circle": [],          # center+through 或 center+r（特判）
    "circumcircle": ["points"],
    "perpendicular": ["line", "point"],
    "parallel": ["line", "point"],
    "anglebisector": ["points"],
    "intersection": ["of"],
    "angle": ["points"],
    "glider": ["on", "coords"],
    "tangent": ["at"],
    "functiongraph": ["expr"],
    "curve": ["xs", "ys"],
    "ellipse": ["cx", "cy", "rx", "ry"],
    "text": ["x", "y", "text"],
    "axisArrow": ["from", "to"],
    "bar": ["x0", "x1", "h"],
    "sector": ["cx", "cy", "r", "start", "end"],
    "numberline": ["xmin", "xmax"],
    "circleOutline": ["cx", "cy", "r"],
}
WHITELIST = set(TYPE_REQUIRED.keys())
# 引用型字段（值是别的 object 的 id，必须能在本 DSL 里解析到）
REF_FIELDS = {
    "points": "list", "center": "id", "through": "id", "line": "id",
    "point": "id", "of": "list", "on": "id", "at": "id",
}


def validate_dsl(spec: dict) -> tuple[bool, list[str]]:
    """过 geo-engine schema 校验：type 白名单 + 必填字段齐 + 引用 id 可解析 + functiongraph 白名单。
    返回 (ok, errors)。"""
    errs: list[str] = []
    if not isinstance(spec, dict):
        return False, ["顶层非 dict"]
    objs = spec.get("objects")
    if spec.get("solid3d") and not objs:
        objs = []  # 纯 3D spec 允许无 objects
    if not isinstance(objs, list) or (not objs and not spec.get("solid3d")):
        return False, ["objects 缺失或非数组"]
    ids = {o.get("id") for o in objs if isinstance(o, dict) and o.get("id")}
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            errs.append(f"objects[{i}] 非对象")
            continue
        t = o.get("type")
        tag = f"objects[{i}](id={o.get('id')},type={t})"
        if t not in WHITELIST:
            errs.append(f"{tag} type 不在白名单")
            continue
        # 必填字段
        for fld in TYPE_REQUIRED[t]:
            if fld not in o or o[fld] in (None, "", []):
                errs.append(f"{tag} 缺必填字段 {fld}")
        if t == "circle":
            has_through = "center" in o and "through" in o
            has_r = "center" in o and "r" in o
            if not (has_through or has_r):
                errs.append(f"{tag} circle 需 center+through 或 center+r")
        # 引用 id 可解析
        for fld, kind in REF_FIELDS.items():
            if fld not in o:
                continue
            val = o[fld]
            refs = val if isinstance(val, list) else [val]
            for r in refs:
                if isinstance(r, str) and r not in ids:
                    # of/which 第三项可能是数字；intersection.of 仅前两项是 id
                    errs.append(f"{tag} 字段 {fld} 引用未定义 id '{r}'")
        # functiongraph expr 安全白名单（防注入）
        if t == "functiongraph":
            expr = str(o.get("expr", ""))
            cleaned = re.sub(r"\b(sin|cos|tan|sqrt|abs|exp|log|pi)\b", "", expr, flags=re.I)
            if not re.fullmatch(r"[-+*/^().,0-9xeE \t]*", cleaned):
                errs.append(f"{tag} functiongraph.expr 含非白名单字符: {expr!r}")
    return (len(errs) == 0), errs


# ---------------------------------------------------------------------------
# 路 A few-shot prompt：让配图 LLM 直出中性 JSON DSL
# ---------------------------------------------------------------------------
DSL_SYSTEM = r"""你是几何配图引擎的「中性 DSL 生成器」：把一道初中数学题的配图需求，翻译成一份**中性几何 JSON DSL**（纯数据，非可执行代码），由确定性渲染器（JSXGraph）建图。

🔴 你只输出一个 JSON 对象（不要解释、不要 markdown ```fence```）。结构：
{
  "bbox": [xmin, ymax, xmax, ymin],   // 视窗，默认 [-5,5,6,-3]
  "axis": false,                       // 坐标系/函数题置 true（自动画 x/y 轴+网格）
  "grid": false,
  "keepAspect": true,                  // 数轴/直方图/扇形图等比无意义时置 false
  "objects": [ ... ]                   // 见下
}

每个 object = { "id":"唯一标识", "type":"...", ...该 type 的参数, "label"?, "draggable"?, "style"? }
🔴 引用别的对象一律用其 id 字符串（不要内联坐标），渲染器按 id 解析依赖、自动算派生坐标。
🔴 自由点直接给 coords:[x,y]；派生点（中点/外接圆/交点/垂足）用对应 type 让引擎算，别手填坐标。

================= 支持的 type 全集（白名单，只能用这些）=================
· point        {coords:[x,y]}                  自由点；draggable:true 渲成红色可拖
· segment      {points:[idA,idB]}              线段（line/ray/vector 同 points，分别=直线/射线/向量）
· polygon      {points:[id,id,id...]}          多边形（自动浅填充）
· midpoint     {points:[idA,idB]}              中点（派生，拖端点自动跟随）
· circle       {center:id, through:id} 或 {center:id, r:数}   圆
· circumcircle {points:[idA,idB,idC]}          三点外接圆
· perpendicular{line:id, point:id}             过点作某线的垂线
· parallel     {line:id, point:id}             过点作某线的平行线
· anglebisector{points:[idA,idB,idC]}          ∠ABC 角平分线
· intersection {of:[idA,idB], which?:0}        两对象交点
· angle        {points:[idA,idB,idC], right?:true}  角弧（B 是顶点）；right 画直角小方块
· glider       {on:id, coords:[x,y]}           约束在某对象上的可拖点
· tangent      {at:gliderId}                   过函数滑点的切线
· functiongraph{expr:"x^2-2*x-3", from?:数, to?:数}  函数图象（expr 走数学白名单，仅 x 单变量+ + - * / ^ () 数字与 sin/cos/tan/sqrt/abs/exp/log/pi）
· curve        {xs:[...], ys:[...], arrow?:true}     折线/参数曲线（s-t 行程图等）
· ellipse      {cx,cy,rx,ry, part?:"full|front|back"}  椭圆（立体底面，back=虚线）
· circleOutline{cx,cy,r}                        扇形图外圈
· sector       {cx,cy,r,start,end}             扇形（角度制，饼图分块）
· bar          {x0,x1,h}                        直方图柱
· numberline   {xmin,xmax,ticks:[...]}         数轴（带箭头+刻度，自动标数值）
· axisArrow    {from:[x,y], to:[x,y]}          带箭头坐标轴
· text         {x,y,text, anchorX?}            文字标注

🔴 公式/符号用 Unicode（√ ∠ ° ′ ² ³ ∥ ⊥ △ × ÷ ± π），不要 LaTeX 宏。
🔴 配图是给学生做的题：不标需要求解的角度数、不泄答案，只画必要构型。

================= few-shot 示例（照此范式产 DSL）=================
例1·三角形 + 外接圆：
{"bbox":[-2,6,8,-2],"objects":[
 {"id":"A","type":"point","coords":[0,0],"label":"A"},
 {"id":"B","type":"point","coords":[6,0],"label":"B"},
 {"id":"C","type":"point","coords":[2,4],"label":"C"},
 {"id":"tri","type":"polygon","points":["A","B","C"]},
 {"id":"oc","type":"circumcircle","points":["A","B","C"]}]}

例2·数轴标点（-3 与 2）：
{"bbox":[-5,2,5,-2],"keepAspect":false,"objects":[
 {"id":"nl","type":"numberline","xmin":-4,"xmax":4,"ticks":[-3,0,2]},
 {"id":"P","type":"point","coords":[2,0],"label":"P"},
 {"id":"Q","type":"point","coords":[-3,0],"label":"Q"}]}

例3·二次函数 y=x²-2x-3：
{"bbox":[-3,3,5,-5],"axis":true,"objects":[
 {"id":"f","type":"functiongraph","expr":"x^2-2*x-3","from":-2,"to":4},
 {"id":"V","type":"point","coords":[1,-4],"label":"顶点"}]}

例4·扇形统计图（两块）：
{"bbox":[-3,3,3,-3],"objects":[
 {"id":"ring","type":"circleOutline","cx":0,"cy":0,"r":2},
 {"id":"s1","type":"sector","cx":0,"cy":0,"r":2,"start":0,"end":120},
 {"id":"s2","type":"sector","cx":0,"cy":0,"r":2,"start":120,"end":360}]}
"""

DSL_USER_TMPL = "【题目】\n{stem}\n\n【配图需求】\n{spec}\n\n按上面的 schema 直出中性 JSON DSL（只输出 JSON）。"


# ---------------------------------------------------------------------------
# 测试题（6-8 道，覆盖：平面几何 / 数轴 / 函数图 / 统计图）
# ---------------------------------------------------------------------------
CASES = [
    {
        "name": "Q1 平面几何·三角形外接圆",
        "stem": "如图，△ABC 的三个顶点都在圆 O 上，已知 A、B、C 三点，画出△ABC 及其外接圆。",
        "spec": "画三角形 ABC（A(0,0)、B(6,0)、C(2,4)）三个顶点 + 三边，再画过 A、B、C 三点的外接圆。",
        "cat": "平面几何",
    },
    {
        "name": "Q2 平面几何·中点连线",
        "stem": "如图，在△ABC 中，D、E 分别是 AB、AC 的中点，连接 DE。",
        "spec": "三角形 ABC（A(0,4)、B(-3,0)、C(5,0)），D 为 AB 中点、E 为 AC 中点，连接 DE（中位线）。",
        "cat": "平面几何",
    },
    {
        "name": "Q3 平面几何·垂直平分/角平分",
        "stem": "如图，∠ABC 中 BD 平分∠ABC，画出射线 BD。",
        "spec": "画角 ∠ABC（顶点 B(0,0)，A(4,3)，C(5,0)），画 ∠ABC 的角平分线 BD。",
        "cat": "平面几何",
    },
    {
        "name": "Q4 数轴·表示有理数",
        "stem": "在数轴上表示出 -2、0、3 这三个数对应的点。",
        "spec": "一条数轴（范围 -5 到 5），标出 -2、0、3 三个点。",
        "cat": "数轴",
    },
    {
        "name": "Q5 函数·二次函数图象",
        "stem": "画出二次函数 y = x² - 2x - 3 的图象，并标出它与 x 轴的交点。",
        "spec": "坐标系中画 y=x^2-2x-3 的抛物线，标出与 x 轴交点 (-1,0)、(3,0) 和顶点 (1,-4)。",
        "cat": "函数图",
    },
    {
        "name": "Q6 函数·反比例函数",
        "stem": "画出反比例函数 y = 6/x 在第一象限的图象。",
        "spec": "坐标系中画 y=6/x，x 从 0.5 到 8（第一象限部分）。",
        "cat": "函数图",
    },
    {
        "name": "Q7 统计·条形统计图",
        "stem": "某班四个小组的人数分别为 8、12、10、6，画出条形统计图。",
        "spec": "条形统计图，四根柱高度分别 8、12、10、6，柱在 x=1、2、3、4 处，宽约 0.6。",
        "cat": "统计图",
    },
    {
        "name": "Q8 统计·扇形统计图",
        "stem": "某调查中 A、B、C 三类占比为 50%、30%、20%，画出扇形统计图。",
        "spec": "一个圆（扇形统计图外圈），分三块扇形：50%（0°~180°）、30%（180°~288°）、20%（288°~360°）。",
        "cat": "统计图",
    },
]


def extract_json(text: str) -> dict | None:
    """剥 markdown fence + 抓首个完整 JSON 对象。"""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # 抓第一个 { ... 最后一个 }
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < 0:
        return None
    frag = s[a:b + 1]
    try:
        return json.loads(frag)
    except Exception:
        # 容错：去掉行内 // 注释再试
        frag2 = re.sub(r"//.*", "", frag)
        try:
            return json.loads(frag2)
        except Exception:
            return None


def call_llm(stem: str, spec: str) -> tuple[str, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": DSL_SYSTEM},
            {"role": "user", "content": DSL_USER_TMPL.format(stem=stem, spec=spec)},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    return resp.choices[0].message.content or "", time.time() - t0


def main() -> None:
    results = []
    for c in CASES:
        print("\n" + "=" * 70)
        print(c["name"], f"[{c['cat']}]")
        try:
            text, dt = call_llm(c["stem"], c["spec"])
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ LLM 调用失败: {e}")
            results.append({**c, "schema_ok": False, "errs": [f"LLM调用失败:{e}"], "dsl": None})
            continue
        spec = extract_json(text)
        if spec is None:
            print("  ❌ JSON 解析失败，原文前 300:")
            print("    " + text[:300].replace("\n", " "))
            results.append({**c, "schema_ok": False, "errs": ["JSON解析失败"], "dsl": None, "raw": text[:500]})
            continue
        ok, errs = validate_dsl(spec)
        types = [o.get("type") for o in spec.get("objects", []) if isinstance(o, dict)]
        print(f"  ⏱ {dt:.1f}s  schema={'✅PASS' if ok else '❌FAIL'}  types={types}")
        if errs:
            for e in errs:
                print(f"     - {e}")
        results.append({**c, "schema_ok": ok, "errs": errs, "dsl": spec, "types": types})

    # 汇总
    print("\n" + "#" * 70)
    print("# 汇总")
    total = len(results)
    passed = sum(1 for r in results if r["schema_ok"])
    print(f"# schema 通过率: {passed}/{total} = {passed * 100 // total}%")
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["cat"], []).append(r["schema_ok"])
    for cat, oks in by_cat.items():
        print(f"#   {cat}: {sum(oks)}/{len(oks)}")
    # 落盘 DSL 供人眼核
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c110_a2_spike_out.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"name": r["name"], "cat": r["cat"], "schema_ok": r["schema_ok"],
              "errs": r["errs"], "dsl": r.get("dsl")} for r in results],
            f, ensure_ascii=False, indent=2,
        )
    print(f"# DSL 产物落盘: {out_path}")


if __name__ == "__main__":
    main()
