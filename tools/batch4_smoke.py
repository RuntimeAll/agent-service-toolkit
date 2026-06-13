# -*- coding: utf-8 -*-
"""PRD-C-015 批4 真机冒烟（trace 取证）：DNA 改→重生状态机 8 项。

跑法（cwd = toolkit）：.venv\\Scripts\\python.exe tools\\batch4_smoke.py
依赖：纯状态机逻辑（重生用桩 LLM，避免网关依赖）；入库覆盖项用桩 RuoyiClient（不打 :8090）。
真机端到端入库/更新（打 :8090）由 verify 阶段单独走 service 端点，本脚本验状态机与回写正确性。
"""
import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agents.variant as V  # noqa: E402
from agents import variant_support as VS  # noqa: E402


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


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
            "skeleton": ["移项", "【合并同类项】"], "hard_points": ["符号"],
            "tags": ["解方程"], "scene": "纯代数",
            "models": [{"id": "M00", "name": "概念直用"}], "difficulty": 3, "flags": [],
        },
    },
    "facts_audit": [],
}


def st(items):
    s = copy.deepcopy(_BASE)
    s["items"] = items
    return s


def main():
    # ① 软重生维改 → 标 dirty 不重出
    banner("① 软重生维【题型】改 → 标 dirty + 角标（题面不变，不立即重出）")
    s = st([{"stem": "原题面", "qtype": "解答", "difficulty": 3}])
    u, it, err = V.edit_dna_state(s, 1, "qtype", "填空")
    assert err is None and u["items"][0]["dna_dirty"] is True
    assert "qtype" in u["items"][0]["dirty_dims"]
    assert u["items"][0]["stem"] == "原题面", "题面应不变"
    print("dna_dirty=", u["items"][0]["dna_dirty"], "dirty_dims=", u["items"][0]["dirty_dims"],
          "stem 不变=", u["items"][0]["stem"] == "原题面")
    print(">>> ① PASS：软重生维改 → 标 dirty，题面不变")

    # ② 点重生 → 待重生集合统一重出 + 清 dirty（桩 LLM）
    banner("② 点「重生」→ 对待重生集合一次性真重出 + 闸B + 清 dirty + 存快照")

    async def fake_regen(item, facts, feedback=None):
        return {"stem": "重出后题面（已变）", "answer": "x=2", "solution": "新解析",
                "qtype": item.get("qtype"), "difficulty": item.get("difficulty")}

    async def fake_check(item, facts, idx, total):
        out = dict(item); out["check"] = {"tier": "verified", "badge": "ok"}; return out, None

    V._regen_once = fake_regen
    V._check_one_item = fake_check
    s2 = st([
        {"stem": "原A", "qtype": "解答", "dna_dirty": True, "dirty_dims": ["qtype"]},
        {"stem": "原B", "dna_dirty": False},
    ])
    u2, r2, err = asyncio.run(V.regen_dirty_items(s2))
    assert err is None and r2["regenerated"] == [1]
    n1 = u2["items"][0]
    assert n1["stem"] == "重出后题面（已变）" and n1["dna_dirty"] is False
    assert n1["regen_snapshot"]["stem"] == "原A"
    assert u2["items"][1]["stem"] == "原B", "not dirty 题不动"
    print("regenerated=", r2["regenerated"], "新题面=", n1["stem"], "dirty 清=", not n1["dna_dirty"],
          "快照=", n1["regen_snapshot"]["stem"])
    print(">>> ② PASS：点重生 → 待重生集合重出 + 清 dirty + 存快照；not dirty 题不动")

    # ③ 硬锚【主考点】改 → 立即解冻重锚（清 items + facts_locked=False，不进 dirty）
    banner("③ 硬锚【主考点】改 → 立即解冻重锚（清 items + 解冻，不进 dirty 攒批）")
    s3 = st([{"stem": "q1"}, {"stem": "q2"}])
    u3, _i, err = V.edit_dna_state(s3, 1, "main_kp", {"code": "30710202", "name": "二元一次方程"})
    assert err is None and u3["items"] == [] and u3["mother_confirmed"] is False and u3["facts_locked"] is False
    print("items 清=", u3["items"] == [], "mother_confirmed=", u3["mother_confirmed"],
          "facts_locked=", u3["facts_locked"])
    print(">>> ③ PASS：硬锚改 → 立即解冻重锚（走既有 patch 路径），不进 dirty")

    # ④ 母题守恒维改 → 下游变式标 dirty 不自动重出 + 保留手改
    banner("④ 母题守恒维【考察类型】改 → 下游变式标 dirty 不自动重出 + 重生保留手改")
    s4 = st([{"stem": "v1", "qtype": "填空", "manual_edited": True}, {"stem": "v2"}])
    u4, _i, err = V.edit_dna_state(s4, 1, "exam_type", "证明推理")
    assert err is None and u4["mother_dna"]["dirty"] is True
    assert all(x["dna_dirty"] for x in u4["items"])
    assert all("exam_type" in (x.get("mother_dirty_dims") or []) for x in u4["items"])
    print("母题脏=", u4["mother_dna"]["dirty"], "下游全 dirty=", all(x["dna_dirty"] for x in u4["items"]),
          "题面不变(不自动重出)=", u4["items"][0]["stem"] == "v1")
    # 重生：手改的 qtype=填空 保留（母题脏维=exam_type，不含 qtype）
    s4r = copy.deepcopy(st(u4["items"]))
    s4r["mother_dna"]["dirty"] = True
    u4r, r4r, err = asyncio.run(V.regen_dirty_items(s4r))
    assert err is None
    assert u4r["items"][0]["qtype"] == "填空", "手改 qtype 应保留不被母题基准覆盖"
    assert u4r["mother_dna"]["dirty"] is False, "全重生完清母题脏"
    print("重生后手改 qtype 保留=", u4r["items"][0]["qtype"] == "填空", "母题脏清=", not u4r["mother_dna"]["dirty"])
    print(">>> ④ PASS：母题改 → 下游标 dirty 不自动重出；重生保留手改 + 清母题脏")

    # ⑤ 撤销重生 → 回上一版快照
    banner("⑤ 撤销重生 → item 回上一版（重生前快照）")
    snap = {"stem": "上一版题面", "qtype": "解答", "dna_dirty": True}
    s5 = st([{"stem": "重生后题面", "qtype": "填空", "regen_snapshot": snap, "dna_dirty": False}])
    u5, restored, err = V.undo_regen_item(s5, 1)
    assert err is None and u5["items"][0]["stem"] == "上一版题面" and u5["items"][0]["qtype"] == "解答"
    print("撤销后 stem=", u5["items"][0]["stem"], "qtype=", u5["items"][0]["qtype"])
    print(">>> ⑤ PASS：撤销重生 → 回上一版快照")

    # ⑥ dirty 题入库 → 被拒 + 提示
    banner("⑥ dirty 题入库 → 被拒 + 提示「第 N 题改了还没重生」（致命①）")
    s6 = st([{"stem": "干净", "dna_dirty": False}, {"stem": "脏题", "dna_dirty": True}])
    msg = V.persist_dirty_guard(s6)
    assert msg and "第 2 题" in msg
    print("拒绝提示:", msg)

    async def boom_persist(*a, **k):
        raise AssertionError("dirty 不该走到 persist_items")

    V.persist_items = boom_persist
    out6 = asyncio.run(V.persist_to_bank(s6, {}))
    assert "暂不能入库" in str(out6["messages"][-1].content)
    print("persist_to_bank 回执:", str(out6["messages"][-1].content)[:60])
    print(">>> ⑥ PASS：dirty 题入库被拒 + 精确到第 N 题")

    # ⑦ 重生后入库 → 覆盖原行 update by _persist_id（桩 RuoyiClient）
    banner("⑦ 重生后入库 → 覆盖原行 update by _persist_id（非新写）")
    calls = {"create": [], "update": []}

    class FakeClient:
        def __init__(self, token=None):
            pass
        async def create_question(self, body):
            calls["create"].append(body); return {"id": 11111}
        async def update_question(self, body):
            calls["update"].append(body); return {"id": body["id"]}
        async def aclose(self):
            pass

    VS.RuoyiClient = FakeClient
    facts = {"mother_question_id": 999, "stem": "母题", "kp_name": "x", "grade": "七上",
             "qtype": "解答", "dna": {}}
    items = [{"stem": "重生过(已入库)", "_persist_id": 555, "qtype": "解答"},
             {"stem": "全新", "qtype": "解答"}]
    receipts = asyncio.run(VS.persist_items(items, facts))
    assert len(calls["update"]) == 1 and calls["update"][0]["id"] == 555
    assert len(calls["create"]) == 1 and "id" not in calls["create"][0]
    print("update 调用 id=", calls["update"][0]["id"], "(覆盖原行)；create 调用数=", len(calls["create"]), "(全新)")
    print("回执:", [(r.get("role"), r.get("id"), r.get("updated")) for r in receipts])
    print(">>> ⑦ PASS：带 _persist_id → update 覆盖原行；无 → create 新写")

    # ⑧ skeleton/hard_points 老师改 → 冻结 setter 留痕（teacher 放行）+ 4/4 维齐
    banner("⑧ skeleton/hard_points 冻结 4/4 维（老师改放行+留痕；LLM 改忽略+audit）")
    assert "skeleton" in V._EDIT_DNA_FIELDS and "hard_points" in V._EDIT_DNA_FIELDS and "models" in V._EDIT_DNA_FIELDS
    s8 = st([{"stem": "q1"}])  # facts_locked=True
    u8, _i, err = V.edit_dna_state(s8, 1, "skeleton", ["老师改的骨架", "【新最难步】"])
    assert err is None and u8["mother_dna"]["dna"]["skeleton"] == ["老师改的骨架", "【新最难步】"]
    sk_aud = [a for a in u8["facts_audit"] if a["field"] == "skeleton"]
    assert sk_aud and not sk_aud[-1].get("ignored")
    # LLM 来源改 skeleton（locked）→ 忽略 + audit(ignored)
    md = {"dna": {"skeleton": ["旧"]}}; aud = []
    wrote = V._dna_fact_edit(md, "skeleton", ["LLM想改"], source="llm", locked=True, audit=aud)
    assert wrote is False and md["dna"]["skeleton"] == ["旧"] and aud[-1]["ignored"] is True
    print("老师改 skeleton 放行+留痕=", bool(sk_aud), "；LLM 改 skeleton 忽略+audit(ignored)=", aud[-1]["ignored"])
    print("_EDIT_DNA_FIELDS 含 skeleton/hard_points/models 4/4=",
          all(f in V._EDIT_DNA_FIELDS for f in ("skeleton", "hard_points", "models")))
    print(">>> ⑧ PASS：冻结 4/4 维齐；老师改放行留痕、LLM 改忽略+audit")

    banner("批4 八项全 PASS")


if __name__ == "__main__":
    main()
