# -*- coding: utf-8 -*-
"""PRD-C-109 B1 真机冒烟：母题编辑工具注册表 + effect 路由（≤4 道精测）。

验四件事（不堆量、精测覆盖路径）：
  ① resolve_tool 对 15 工具+旋钮返对的 {effect, regen_class}（断 UI effect→REGEN_CLASS 映射表）。
  ② mutator 薄包 edit_dna_state → 走进对的四分流（即时维只标注不脏 / 重出维标脏或重锚 /
     重写解析维标脏）；改该维 mother_dna/item 该字段变、别字段不变。
  ③ 加/删/换 模型与标签 list 操作正确（薄包不重写）。
  ④ 难度不在注册表（只读）；执行/旋钮类 fn=None 不经 apply_tool 改单维。

跑法（cwd = toolkit，必走 .venv）：
  $env:NO_PROXY="*"; .venv\\Scripts\\python.exe tools\\c109_b1_smoke.py
依赖：纯状态机逻辑（零 LLM、零 :8090）；与 batch4_smoke.py 互不干扰。
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agents.variant as V  # noqa: E402
from agents.variant import (  # noqa: E402
    EFFECT_TO_REGEN_CLASSES,
    TOOL_REGISTRY,
    apply_tool,
    resolve_tool,
)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# 母题基线（含已生成 1 道变式 item，验「改即时维别维不动」）。
_BASE = {
    "mother_confirmed": True,
    "facts_locked": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9, "code": "3071"},
        "kp": {"value": "一元一次方程", "confidence": 0.9,
               "anchored": {"id": "30710101", "code": "30710101", "name": "一元一次方程"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干", "answer": "x=1", "difficulty": 3,
        "dna": {
            "main_kp": {"id": "30710101", "name": "一元一次方程"},
            "secondary_kps": [{"id": "30710102", "name": "等式性质"}],
            "qtype": "解答", "exam_type": "直接计算",
            "skeleton": ["移项", "合并同类项"], "hard_points": ["符号"],
            "tags": ["解方程"], "scene": "纯代数",
            "models": [{"id": "M00", "name": "概念直用"}], "difficulty": 3, "flags": [],
        },
    },
    "facts_audit": [],
}


def st():
    s = copy.deepcopy(_BASE)
    s["items"] = [{"stem": "变式1题面", "qtype": "解答", "difficulty": 3,
                   "figure_url": "https://oss/fig1.png", "_seq": 1,
                   "models": [{"id": "M00", "name": "概念直用"}]}]
    return s


PASS = 0
FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {msg}")
    else:
        FAIL += 1
        print(f"  XX  {msg}")


# ---------------------------------------------------------------------------
# ① resolve_tool 返对的 {effect, regen_class}，15 工具+旋钮全覆盖 + 映射表自洽
# ---------------------------------------------------------------------------
def test_resolve():
    banner("① resolve_tool: 17 工具的 {effect, field, regen_class} 全覆盖 + effect→REGEN_CLASS 映射自洽")
    # 期望表（与 A1 spike TOOL_EFFECT + §10 一致）
    EXPECT = {
        "选副考点": ("即时生效", "secondary_kps", "meta"),
        "set_难点": ("即时生效", "hard_points", "meta"),
        "加标签": ("即时生效", "tags", "meta"),
        "删标签": ("即时生效", "tags", "meta"),
        "改解法骨架": ("重写解析", "skeleton", "rewrite_solve"),
        "加模型": ("重写解析", "models", "rewrite_solve"),
        "删模型": ("重写解析", "models", "rewrite_solve"),
        "换模型": ("重写解析", "models", "rewrite_solve"),
        "set_主考点": ("重出本题", "main_kp", "soft_regen"),
        "set_题型": ("重出本题", "qtype", "soft_regen"),
        "set_考察类型": ("重出本题", "exam_type", "soft_regen"),
        "set_场景": ("重出本题", "scene", "soft_regen"),
        "set_年级章": ("重出本题", "grade", "hard_anchor"),  # 🔴 A2 非 1:1：重出→hard_anchor
        "改题面": ("重出本题", None, None),      # exec（走 revise 流水线）
        "重新解题": ("执行", None, None),
        "开始出变式": ("执行", None, None),
        "_难度旋钮": ("旋钮", None, None),
    }
    ok(len(TOOL_REGISTRY) == len(EXPECT), f"注册表条数={len(TOOL_REGISTRY)} 期望 {len(EXPECT)}")
    for name, (eff, field, rc) in EXPECT.items():
        spec = resolve_tool(name)
        ok(spec is not None, f"{name}: resolve_tool 命中")
        if not spec:
            continue
        ok(spec["effect"] == eff, f"{name}: effect={spec['effect']} 期望 {eff}")
        ok(spec["dna_field"] == field, f"{name}: field={spec['dna_field']} 期望 {field}")
        ok(spec["regen_class"] == rc, f"{name}: regen_class={spec['regen_class']} 期望 {rc}")
        # 映射表自洽（有 field 的非 exec 工具）：effect 的允许 regen_class 集含本工具 rc
        allowed = EFFECT_TO_REGEN_CLASSES.get(eff)
        if allowed is not None and field is not None and not spec["exec"]:
            ok(rc in allowed, f"{name}: regen_class {rc} ∈ effect[{eff}] 允许集 {set(allowed)}")
    ok(resolve_tool("不存在的工具") is None, "未知工具 resolve_tool→None（B2 回退兜底）")


# ---------------------------------------------------------------------------
# ② mutator 薄包 edit_dna_state → 走对的四分流；改该维变、别维不变
# ---------------------------------------------------------------------------
def test_mutator_routing():
    banner("② mutator 薄包 edit_dna_state：即时维只标注不脏 / 重写解析维标脏 / 重出维标脏或解冻重锚")

    # 即时生效维（加标签）→ 不进 dirty；tags 变、figure_url 别维不动
    s = st()
    u, it, err = apply_tool("加标签", s, 1, "中考高频")
    ok(err is None, "加标签 无 error")
    ok("中考高频" in (u.get("mother_dna", {}).get("dna", {}).get("tags") or []), "加标签: tags 含新标签")
    ok("解方程" in (u.get("mother_dna", {}).get("dna", {}).get("tags") or []), "加标签: 旧标签保留")
    ok(not u["items"][0].get("dna_dirty"), "加标签(meta): item 不标 dirty")
    ok(u["items"][0].get("figure_url") == "https://oss/fig1.png", "加标签: figure_url 别维不动(G5①雏形)")
    ok((u.get("mother_dna", {}).get("dna", {}).get("scene")) == "纯代数", "加标签: scene 别维不变")

    # 重写解析维（改解法骨架）→ 标 dirty + dirty_dims 含 skeleton
    s = st()
    u, it, err = apply_tool("改解法骨架", s, 1, "第一步移项\n第二步配方")
    ok(err is None, "改解法骨架 无 error")
    ok(u["items"][0].get("dna_dirty") is True, "改解法骨架(rewrite_solve): item 标 dirty")
    ok("skeleton" in (u["items"][0].get("dirty_dims") or []), "改解法骨架: dirty_dims 含 skeleton")
    sk = (u.get("mother_dna", {}).get("dna", {}).get("skeleton") or [])
    ok("第一步移项" in sk and "第二步配方" in sk, "改解法骨架: 母题 skeleton 更新")

    # 重出本题·soft_regen（set_题型）→ 标 dirty；qtype 变、题面不变
    s = st()
    u, it, err = apply_tool("set_题型", s, 1, "填空")
    ok(err is None, "set_题型 无 error")
    ok(u["items"][0].get("dna_dirty") is True, "set_题型(soft_regen): item 标 dirty")
    ok(u["items"][0].get("stem") == "变式1题面", "set_题型: 题面不变(标脏非立即重出)")
    ok((u.get("mother_dna", {}).get("dna", {}).get("qtype")) == "填空", "set_题型: 母题 qtype 守恒维同步")

    # 重出本题·hard_anchor（set_年级章）→ 解冻重锚（清 items + mother_confirmed/facts_locked=False）
    s = st()
    u, it, err = apply_tool("set_年级章", s, 1, "八年级下学期")
    ok(err is None, "set_年级章 无 error")
    ok(u.get("items") == [], "set_年级章(hard_anchor): items 清空(解冻重锚)")
    ok(u.get("mother_confirmed") is False, "set_年级章: mother_confirmed=False(触发重锚)")
    ok(u.get("facts_locked") is False, "set_年级章: facts_locked=False")

    # 即时生效·set_难点（hard_points meta）→ 不进 dirty；hard_points 变
    s = st()
    u, it, err = apply_tool("set_难点", s, 1, "分类讨论")
    ok(err is None, "set_难点 无 error")
    ok(not u["items"][0].get("dna_dirty"), "set_难点(meta): 不标 dirty")
    ok("分类讨论" in (u["items"][0].get("hard_points") or []), "set_难点: item.hard_points 更新")


# ---------------------------------------------------------------------------
# ③ list 操作（加/删/换 模型 + 标签）薄包正确
# ---------------------------------------------------------------------------
def test_list_ops():
    banner("③ list 操作：加/删/换 模型 + 加/删标签（薄包不重写）")

    # 加模型（已有 M00，加 M25）
    s = st()
    u, it, err = apply_tool("加模型", s, 1, {"id": "M25", "name": "韦达定理"})
    ok(err is None, "加模型 无 error")
    ids = [m["id"] for m in (u["items"][0].get("models") or [])]
    ok("M00" in ids and "M25" in ids, f"加模型: models={ids} 含旧 M00+新 M25")
    ok(u["items"][0].get("dna_dirty") is True, "加模型(rewrite_solve): 标 dirty")

    # 加模型幂等（再加 M00 不重复）
    s = st()
    u, it, err = apply_tool("加模型", s, 1, {"id": "M00", "name": "概念直用"})
    ids = [m["id"] for m in (u["items"][0].get("models") or [])]
    ok(ids.count("M00") == 1, f"加模型幂等: M00 不重复（models={ids}）")

    # 删模型（删 M00）
    s = st()
    u, it, err = apply_tool("删模型", s, 1, {"id": "M00"})
    ok(err is None, "删模型 无 error")
    ids = [m["id"] for m in (u["items"][0].get("models") or [])]
    ok("M00" not in ids, f"删模型: M00 已删（models={ids}）")

    # 换模型（{old,new} 精确换）
    s = st()
    u, it, err = apply_tool("换模型", s, 1, {"old": {"id": "M00"}, "new": {"id": "M30", "name": "配方法"}})
    ok(err is None, "换模型 无 error")
    ids = [m["id"] for m in (u["items"][0].get("models") or [])]
    ok("M00" not in ids and "M30" in ids, f"换模型: M00→M30（models={ids}）")

    # 加标签 + 删标签
    s = st()
    u, it, err = apply_tool("加标签", s, 1, "易错")
    tags = u.get("mother_dna", {}).get("dna", {}).get("tags") or []
    ok("易错" in tags and "解方程" in tags, f"加标签: tags={tags}")
    s2 = copy.deepcopy(s); s2["mother_dna"]["dna"]["tags"] = ["解方程", "易错"]
    u2, it2, err2 = apply_tool("删标签", s2, 1, "解方程")
    tags2 = u2.get("mother_dna", {}).get("dna", {}).get("tags") or []
    ok("解方程" not in tags2 and "易错" in tags2, f"删标签: tags={tags2}")


# ---------------------------------------------------------------------------
# ④ 难度只读 + 执行/旋钮类 fn=None 不经 apply_tool 改单维
# ---------------------------------------------------------------------------
def test_difficulty_readonly_and_exec():
    banner("④ 难度只读(不在注册表) + 执行/旋钮类 fn=None 守门")
    # 注册表无任何 dna_field == difficulty 的工具
    has_diff = any(spec.get("dna_field") == "difficulty" for spec in TOOL_REGISTRY.values())
    ok(not has_diff, "注册表无「改难度」工具（难度只读·表驱动）")
    ok("_难度旋钮" in TOOL_REGISTRY and TOOL_REGISTRY["_难度旋钮"]["effect"] == "旋钮",
       "_难度旋钮: effect=旋钮（B2 路由到变式旋钮，不碰母题）")
    ok(TOOL_REGISTRY["_难度旋钮"]["fn"] is None, "_难度旋钮: fn=None（不改单维）")

    # 执行/旋钮类经 apply_tool → 拒绝改单维，返回提示串（不崩、不误改）
    s = st()
    u, it, err = apply_tool("重新解题", s, 1, None)
    ok(err is not None and not u, "重新解题(exec): apply_tool 拒改单维(返提示，路由走流水线)")
    u, it, err = apply_tool("_难度旋钮", s, 1, "难一点")
    ok(err is not None and not u, "_难度旋钮: apply_tool 拒改单维(路由走难度旋钮)")
    u, it, err = apply_tool("开始出变式", s, 1, None)
    ok(err is not None, "开始出变式(exec): apply_tool 拒改单维")


if __name__ == "__main__":
    test_resolve()
    test_mutator_routing()
    test_list_ops()
    test_difficulty_readonly_and_exec()
    banner(f"汇总: PASS={PASS}  FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
