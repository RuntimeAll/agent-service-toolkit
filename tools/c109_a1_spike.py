"""PRD-C-109 预飞行 A1：母题编辑「工具选择器」准确率 spike（临时脚本，不动产品代码）。

验「LLM 把开放母题修改话术 映射到 对的工具 + 对的 effect」够不够准。
- 15 个母题编辑工具（§10），每工具声明 effect（即时生效/重写解析/重出本题）。
- ~36 条老师话术（每工具 2-3 条 + 易混对照）。
- 跑 claude-opus-4-8（sui-xiang 中转，与生产同 key/同站），低温受约束分类。
- 算 工具准确率 + 🔴 effect 跨类误判率（effect 错比工具错更隐蔽=错误重解）。

跑法（cwd=agent-service-toolkit，必走 .venv）：
  $env:NO_PROXY="*"; .venv\Scripts\python.exe tools\c109_a1_spike.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

# ---- 读 .env（不依赖 pydantic-settings，免拉整个 graph 进程）----
ROOT = Path(__file__).resolve().parents[1]
ENV = {}
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    ENV[k.strip()] = v.strip()

BASE_URL = ENV.get("COMPATIBLE_BASE_URL", "https://sui-xiang.com/v1")
API_KEY = ENV.get("COMPATIBLE_API_KEY", "")
MODEL = ENV.get("LLM_MODEL_LIGHT", "claude-opus-4-8")
# 绕本地代理（中转直连）
os.environ["NO_PROXY"] = "*"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

from openai import AsyncOpenAI  # noqa: E402

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# 15 工具注册表（§10）：tool -> effect。effect 由代码查表得出（LLM 不产 effect）。
#   即时生效 / 重写解析 / 重出本题 三类（难度不在工具集=只读）。
# ---------------------------------------------------------------------------
TOOL_EFFECT = {
    "set_主考点": "重出本题",
    "选副考点": "即时生效",
    "set_题型": "重出本题",
    "set_考察类型": "重出本题",
    "set_场景": "重出本题",
    "改解法骨架": "重写解析",
    "加模型": "重写解析",
    "删模型": "重写解析",
    "换模型": "重写解析",
    "set_难点": "即时生效",
    "加标签": "即时生效",
    "删标签": "即时生效",
    "改题面": "重出本题",
    "set_年级章": "重出本题",
    "重新解题": "重出本题",
    "开始出变式": "执行",
    # 特例：改难度不是母题工具（走变式难度旋钮 stage-2），期望工具选择器识别为 none/旋钮。
    "_难度旋钮": "旋钮",  # 期望 LLM 不选任何母题工具
}

TOOL_LIST_FOR_PROMPT = """\
1. set_主考点 —— 改这道母题的主考点（如「主考点改成根的判别式」）。【重出本题】
2. 选副考点 —— 加/设一个副考点（次要考点，如「副考点加个韦达定理」）。【即时生效】
3. set_题型 —— 改母题题型（选择/填空/解答，如「题型选填空」）。【重出本题】
4. set_考察类型 —— 改考察类型（如「改成计算题」「设成证明类」）。【重出本题】
5. set_场景 —— 改题目场景/情境（如「换成行程问题情境」「设个购物背景」）。【重出本题】
6. 改解法骨架 —— 改这道题的解法步骤骨架本身（如「解法骨架第二步改成移项」）。【重写解析】
7. 加模型 —— 给解法加一个解题模型（如「再加个韦达定理模型」「补一个数形结合」）。【重写解析】
8. 删模型 —— 从解法里删掉一个模型（如「把因式分解那个模型去掉」）。【重写解析】
9. 换模型 —— 把解法的模型换成另一个（如「换成判别式法模型」「这个模型换掉」）。【重写解析】
10. set_难点 —— 设/改这道题的难点描述（如「难点设成符号易错」）。【即时生效】
11. 加标签 —— 给母题加一个标签（如「加个易错标签」「打上中考高频标签」）。【即时生效】
12. 删标签 —— 删掉母题一个标签（如「把那个压轴标签删了」）。【即时生效】
13. 改题面 —— 改题面文字/排版（如「题面这里排版乱了重排」「题干第二句改一下」）。【重出本题】
14. set_年级章 —— 改母题年级/章（如「年级改成八下」「这是第二章的」）。【重出本题】
15. 重新解题 —— 让系统重新把这道母题解一遍（如「重新解一下」「再解一遍」）。【重出本题】
16. 开始出变式 —— 老师明确要开始生成变式（如「开始」「出3道」）。【执行】
"""

PROMPT = """你是举一反三工作流的「母题编辑·工具选择器」。老师正对一道已解出的母题做开放式修改。读【母题当前态】+【老师最新一句】，从下面 16 个工具里选**唯一一个**最匹配的工具（只选工具+目标值，不要执行、不要解释、不要自己判 effect）。

【16 个母题编辑工具】
{tool_list}

🔴 还有一种特殊情况：老师说的是【调整难度】（如「难度难一点」「来个简单的」「一颗星」「难度调高」）。难度是只读的表驱动维、**不是母题工具**——此时 tool 填 "_难度旋钮"（系统会路由到变式难度旋钮，不碰母题、不重解）。

🔴 关键易混区分（务必照此判）：
- 「换个模型 / 换成X法模型」= 换模型；「再加个X模型 / 补一个模型」= 加模型；「改解法骨架 / 第几步改成」= 改解法骨架；「重新解一遍 / 再解一次」= 重新解题。这四个都关解法，但工具不同，别混。
- 「主考点改成X」= set_主考点（重出本题）；「副考点加个X / 次要考点」= 选副考点（即时生效）。主≠副，effect 天差地别。
- 「难度X」= _难度旋钮（不选母题工具）；「题型X」= set_题型；「考察类型X」= set_考察类型。难度≠题型≠考察类型。
- 「加标签 / 删标签」只动标签元数据；「set_难点」改难点描述。别把"加标签"误判成"set_难点"。

只输出一个 JSON（不要 markdown fence、不要解释）：
{{
  "tool": "上面 16 个工具名之一 或 _难度旋钮",
  "value": "老师要改成的目标值（人话，没有就 null）",
  "confidence": 0.0
}}

【母题当前态】
一道已解出的初中数学母题：题面/答案/解法骨架/主考点/副考点[]/题型/考察类型/场景/模型[]/难点/标签[]/年级章 都在手，老师在逐维微调。

【老师最新一句话】
{utterance}"""

# ---------------------------------------------------------------------------
# 测试集：~36 条（每工具 2-3 条 + 易混对照）。expected = (工具, effect)。
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str]] = [
    # set_主考点（重出本题）
    ("主考点改成根的判别式", "set_主考点"),
    ("这道题主考点应该是一元二次方程根与系数关系", "set_主考点"),
    # 选副考点（即时生效）——易混：vs 主考点
    ("副考点加个韦达定理", "选副考点"),
    ("次要考点补一个数形结合", "选副考点"),
    # set_题型（重出本题）——易混：vs 难度/考察类型
    ("题型选填空", "set_题型"),
    ("把它改成选择题", "set_题型"),
    # set_考察类型（重出本题）
    ("改成计算题", "set_考察类型"),
    ("设成证明类的", "set_考察类型"),
    # set_场景（重出本题）
    ("换成行程问题的情境", "set_场景"),
    ("给它套个购物打折的背景", "set_场景"),
    # 改解法骨架（重写解析）——易混：vs 换模型/重新解题
    ("解法骨架第二步改成移项合并", "改解法骨架"),
    ("解题步骤里把配方那一步写细一点", "改解法骨架"),
    # 加模型（重写解析）——易混：vs 换模型
    ("再加个韦达定理模型", "加模型"),
    ("补一个数形结合的模型进去", "加模型"),
    # 删模型（重写解析）
    ("把因式分解那个模型去掉", "删模型"),
    ("删掉判别式法这个模型", "删模型"),
    # 换模型（重写解析）——易混对照核心：换模型 vs 改解法 vs 重新解题
    ("换成判别式法模型", "换模型"),
    ("这个解题模型换掉", "换模型"),
    ("把当前模型替换成配方法", "换模型"),
    # set_难点（即时生效）——易混：vs 加标签
    ("难点设成符号运算易错", "set_难点"),
    ("这题难点标成分类讨论", "set_难点"),
    # 加标签（即时生效）
    ("加个易错标签", "加标签"),
    ("打上中考高频这个标签", "加标签"),
    # 删标签（即时生效）
    ("把压轴这个标签删了", "删标签"),
    ("去掉那个送分标签", "删标签"),
    # 改题面（重出本题）
    ("题面这里排版乱了帮我重排一下", "改题面"),
    ("题干第二句话改通顺一点", "改题面"),
    # set_年级章（重出本题）——易混：vs 确认范围语气
    ("年级改成八年级下册", "set_年级章"),
    ("这道其实是第二章一元二次方程的", "set_年级章"),
    # 重新解题（重出本题）——易混核心：vs 换模型/改解法
    ("重新解一遍这道题", "重新解题"),
    ("再解一次看看", "重新解题"),
    # 开始出变式（执行）
    ("开始出3道", "开始出变式"),
    ("可以了，开始举一反三", "开始出变式"),
    # _难度旋钮（不选母题工具）——易混核心：难度 vs 题型/考察类型
    ("难度难一点", "_难度旋钮"),
    ("来个简单点的，一颗星", "_难度旋钮"),
    ("难度调高到压轴", "_难度旋钮"),
]


def _parse_json(text: str) -> dict | None:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def classify(utterance: str) -> dict:
    prompt = PROMPT.format(tool_list=TOOL_LIST_FOR_PROMPT, utterance=utterance)
    for attempt in range(2):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=400,
            )
            text = resp.choices[0].message.content or ""
            parsed = _parse_json(text)
            if isinstance(parsed, dict) and parsed.get("tool"):
                return parsed
        except Exception as e:  # noqa: BLE001
            if attempt == 0:
                await asyncio.sleep(1.5)
                continue
            return {"tool": f"<ERR:{str(e)[:60]}>", "value": None, "confidence": 0.0}
    return {"tool": "<PARSE_FAIL>", "value": None, "confidence": 0.0}


async def main() -> None:
    print(f"== A1 spike == model={MODEL} base={BASE_URL} cases={len(CASES)}")
    t0 = time.monotonic()
    sem = asyncio.Semaphore(6)

    async def run_one(utt: str, exp_tool: str):
        async with sem:
            got = await classify(utt)
        got_tool = got.get("tool")
        exp_eff = TOOL_EFFECT.get(exp_tool, "?")
        got_eff = TOOL_EFFECT.get(got_tool, "<未知工具>")
        tool_ok = got_tool == exp_tool
        eff_ok = got_eff == exp_eff
        return {
            "utt": utt, "exp_tool": exp_tool, "got_tool": got_tool,
            "exp_eff": exp_eff, "got_eff": got_eff,
            "tool_ok": tool_ok, "eff_ok": eff_ok, "conf": got.get("confidence"),
        }

    results = await asyncio.gather(*(run_one(u, t) for u, t in CASES))

    tool_hit = sum(1 for r in results if r["tool_ok"])
    eff_hit = sum(1 for r in results if r["eff_ok"])
    n = len(results)
    eff_cross_miss = [r for r in results if not r["eff_ok"]]  # effect 跨类误判（最危险）

    print(f"\n---- 逐条 ----")
    for r in results:
        mark = "OK " if r["tool_ok"] else "XX "
        emark = "" if r["eff_ok"] else "  <<EFFECT跨类误判!"
        print(f"{mark}[{r['exp_tool']}->{r['got_tool']}] eff[{r['exp_eff']}->{r['got_eff']}] c={r['conf']}  「{r['utt']}」{emark}")

    print(f"\n========== 汇总 ==========")
    print(f"工具准确率      : {tool_hit}/{n} = {tool_hit/n*100:.1f}%   (达标线 >=90%)")
    print(f"effect 正确率   : {eff_hit}/{n} = {eff_hit/n*100:.1f}%")
    print(f"effect 跨类误判 : {len(eff_cross_miss)}/{n} = {len(eff_cross_miss)/n*100:.1f}%   (达标线 <=5%)")
    if eff_cross_miss:
        print(f"\n!! effect 跨类误判明细（选错维度且 effect 跨类=隐蔽错误重解）：")
        for r in eff_cross_miss:
            print(f"   「{r['utt']}」 期望 {r['exp_tool']}({r['exp_eff']}) 得 {r['got_tool']}({r['got_eff']})")
    print(f"\n耗时 {time.monotonic()-t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
