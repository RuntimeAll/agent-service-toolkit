# -*- coding: utf-8 -*-
"""PRD-C-015 批2·双轴 models 锚定 + 落库 + 守恒维落库一致性单测（mock LLM/DB，零网络）。

覆盖 gate：
- G1（models 非空 + 池外名进正式维=0 + M00 兜底路径触发正确）
- G2（反查候选与 SQL 一致 + 前缀边界·兄弟节点不误命中）
- G5（落库：`模型:<name>` 走标签三轨；M00 也照落）
- G21（守恒维落库一致性：主 kp=「0」未分类时副考点/考察类型不落）

铁律对照（PRD §3.2 处置穷举 / §10.1(a) / §3.4 #3 缺口11）：
- 模型维永不为空：候选空/全不确认 → M00 保底。
- 禁造词：LLM 给池外名 → 不入正式维，落 model_overflow + 待命名池 + ⚠。
- 反查库故障 → 降级 M00 + ⚠（不卡死）。
- 落库走 book-server HTTP（透传 tags），toolkit 不直写库；模型名 `模型:` 前缀标签三轨，零 DDL。
"""

import asyncio

import agents.model_anchor as ma
from agents.model_anchor import (
    M00,
    M00_ID,
    M00_NAME,
    _leaf_codes,
    _parse_models_json,
    anchor_models,
    confirm_models,
)
from agents.variant_support import MODEL_TAG_PREFIX, build_create_bo, build_mother_bo


# ---------------------------------------------------------------------------
# 候选构造夹具（模拟 26号 §0.5 反查返回：[{id,name,trigger_feature,action_conclusion,sort}]）
# ---------------------------------------------------------------------------
def _cands(*ids):
    by = {
        "M25": ("定边对定角（隐圆主模型）", "定线段对定角的动顶点", "顶点轨迹=定弧→转圆问题", 250),
        "M26": ("四点共圆", "对角互补/同侧同弦等角", "判共圆借圆周角继续导", 260),
        "M32": ("隐圆最值", "动点在定圆/定弧上", "连心距±半径=最值", 320),
        "M29": ("将军饮马", "直线上动点到两定点和最小", "对称拉直", 290),
    }
    return [
        {"id": i, "name": by[i][0], "trigger_feature": by[i][1],
         "action_conclusion": by[i][2], "sort": by[i][3]}
        for i in ids
    ]


def _fake_invoke(ret):
    async def fake(messages, *, model=None, **kw):
        return ret
    return fake


# ===========================================================================
# G2·反查候选 + 前缀边界
# ===========================================================================
def test_leaf_codes_main_plus_secondary_dedup():
    dna = {
        "main_kp": {"id": "3091003014006", "name": "隐圆"},
        "secondary_kps": [{"id": "3091003008", "name": "圆周角"}, {"id": "3091003014006"}],
    }
    assert _leaf_codes(dna) == ["3091003014006", "3091003008"]


def test_leaf_codes_empty_dna():
    assert _leaf_codes(None) == []
    assert _leaf_codes({}) == []


def test_lookup_candidates_empty_codes_short_circuits(monkeypatch):
    """空 leaf_codes → 不连库、直接空候选（防无谓连库）。"""
    def boom(**kw):
        raise AssertionError("不该连库")
    monkeypatch.setattr(ma.pymysql, "connect", boom)
    assert ma.lookup_candidates([]) == []


def test_lookup_candidates_prefix_boundary_sql_shape(monkeypatch):
    """G2 前缀边界：反查 SQL = `:leaf LIKE CONCAT(subject_id,'%')`（叶子以绑定节点 code 开头）。
    用相邻 code（3081002001004003 两定一动 vs 3081002001005 折叠）当边界用例，验 SQL 形参传对、
    叶子 code 本身进 WHERE（兄弟节点前缀不同 → 库层自然不误命中，本测验形参传递正确性）。"""
    captured = []

    class _Cur:
        def execute(self, sql, params):
            captured.append((sql, params))
        def fetchall(self):
            return []

    class _Conn:
        def cursor(self, *a, **k):
            return _Cur()
        def close(self):
            pass

    monkeypatch.setattr(ma.pymysql, "connect", lambda **kw: _Conn())
    ma.lookup_candidates(["3081002001004003", "3081002001005"])
    assert len(captured) == 2  # 每个 leaf 一次反查
    sql0, params0 = captured[0]
    assert "LIKE CONCAT(k.subject_id" in sql0
    assert "bind_type IN ('primary','native')" in sql0
    assert params0 == ("3081002001004003",)
    assert captured[1][1] == ("3081002001005",)


def test_lookup_candidates_merge_dedup_sort(monkeypatch):
    """多 leaf 反查结果按 id 去重、按 sort 排序、截 limit。"""
    rows_seq = [_cands("M32", "M25"), _cands("M25", "M26")]  # M25 重复

    class _Cur:
        def __init__(self):
            self.i = 0
        def execute(self, sql, params):
            self._rows = rows_seq[self.i]
            self.i += 1
        def fetchall(self):
            return self._rows

    class _Conn:
        def cursor(self, *a, **k):
            return self._cur
        def close(self):
            pass

    conn = _Conn()
    conn._cur = _Cur()
    monkeypatch.setattr(ma.pymysql, "connect", lambda **kw: conn)
    out = ma.lookup_candidates(["a", "b"], limit=8)
    ids = [c["id"] for c in out]
    assert ids == ["M25", "M26", "M32"]  # 去重 + 按 sort（250/260/320）升序
    assert len(ids) == len(set(ids))


# ===========================================================================
# G1·池内确认（confirm_models）
# ===========================================================================
def test_confirm_in_pool_hits():
    confirmed, overflow = asyncio.run(confirm_models(
        "隐圆最值题", "解析", _cands("M25", "M32"),
        invoke=_fake_invoke('{"models":["M25","M32"]}'),
    ))
    assert [c["id"] for c in confirmed] == ["M25", "M32"]
    assert overflow == []


def test_confirm_empty_candidates_no_llm_call():
    """候选空 → 不调 LLM，直接 ([],[])（上层兜 M00）。"""
    async def boom(*a, **k):
        raise AssertionError("候选空不该调 LLM")
    confirmed, overflow = asyncio.run(confirm_models("s", "a", [], invoke=boom))
    assert confirmed == [] and overflow == []


def test_confirm_caps_at_three():
    confirmed, _ = asyncio.run(confirm_models(
        "s", "a", _cands("M25", "M26", "M32", "M29"),
        invoke=_fake_invoke('{"models":["M25","M26","M32","M29"]}'),
    ))
    assert len(confirmed) == 3  # ≤MODELS_MAX


def test_confirm_pool_outside_goes_overflow_not_main_dim():
    """禁造词：LLM 给候选外 id → 不入正式维，落 overflow。"""
    confirmed, overflow = asyncio.run(confirm_models(
        "s", "a", _cands("M25"),
        invoke=_fake_invoke('{"models":["M25","M99托勒密"]}'),
    ))
    assert [c["id"] for c in confirmed] == ["M25"]
    assert overflow == ["M99托勒密"]


def test_confirm_bad_json_returns_empty():
    confirmed, overflow = asyncio.run(confirm_models(
        "s", "a", _cands("M25"), invoke=_fake_invoke("我觉得用M25"),
    ))
    assert confirmed == [] and overflow == []


def test_parse_models_json_strips_fence():
    assert _parse_models_json('```json\n{"models":["M01"]}\n```') == ["M01"]
    assert _parse_models_json('{"models":[]}') == []
    assert _parse_models_json("not json") is None
    assert _parse_models_json('{"foo":1}') is None


# ===========================================================================
# G1·anchor_models 处置穷举（§3.2）
# ===========================================================================
def test_anchor_in_pool_normal(monkeypatch):
    monkeypatch.setattr(ma, "lookup_candidates", lambda codes, **k: _cands("M25", "M32"))
    res = asyncio.run(anchor_models(
        {"main_kp": {"id": "3091003014006"}}, stem="隐圆最值",
        invoke=_fake_invoke('{"models":["M25"]}'),
    ))
    assert [m["id"] for m in res["models"]] == ["M25"]
    assert res["model_warn"] is False and res["model_flag"] is None


def test_anchor_empty_candidates_m00(monkeypatch):
    """候选空 → M00 保底，模型维非空。"""
    monkeypatch.setattr(ma, "lookup_candidates", lambda codes, **k: [])
    res = asyncio.run(anchor_models(
        {"main_kp": {"id": "x"}}, stem="基础概念题", invoke=_fake_invoke('{"models":[]}'),
    ))
    assert res["models"] == [dict(M00)]
    assert res["model_flag"] == "m00_fallback" and res["model_warn"] is False


def test_anchor_all_not_confirmed_m00(monkeypatch):
    """有候选但 LLM 全不确认 → M00 保底。"""
    monkeypatch.setattr(ma, "lookup_candidates", lambda codes, **k: _cands("M25"))
    res = asyncio.run(anchor_models(
        {"main_kp": {"id": "x"}}, stem="s", invoke=_fake_invoke('{"models":[]}'),
    ))
    assert res["models"] == [dict(M00)] and res["model_flag"] == "m00_fallback"


def test_anchor_pool_outside_overflow_warn_and_records(monkeypatch):
    """池外名 → 不入正式维（confirmed 仍正常），overflow 落待命名池 + ⚠。"""
    monkeypatch.setattr(ma, "lookup_candidates", lambda codes, **k: _cands("M25"))
    recorded = []
    res = asyncio.run(anchor_models(
        {"main_kp": {"id": "x"}}, stem="s",
        invoke=_fake_invoke('{"models":["M25","梅涅劳斯"]}'),
        record_overflow=lambda name, mm: recorded.append((name, mm)),
    ))
    assert [m["id"] for m in res["models"]] == ["M25"]
    assert res["model_overflow"] == ["梅涅劳斯"] and res["model_warn"] is True
    assert res["model_flag"] == "overflow"
    assert recorded == [("梅涅劳斯", ["M25"])]


def test_anchor_lookup_unavailable_degrades_m00_warn(monkeypatch):
    """反查库/表不可用（连库抛异常）→ 降级 M00 + ⚠，不卡死。"""
    def boom(codes, **k):
        raise RuntimeError("库未起")
    monkeypatch.setattr(ma, "lookup_candidates", boom)
    res = asyncio.run(anchor_models(
        {"main_kp": {"id": "x"}}, stem="s", invoke=_fake_invoke('{"models":["M25"]}'),
    ))
    assert res["models"] == [dict(M00)]
    assert res["model_warn"] is True and res["model_flag"] == "lookup_unavailable"


def test_record_overflow_candidate_write(tmp_path, monkeypatch):
    """待命名池落盘成功（status=pending 坑位预留）。"""
    p = tmp_path / "model_candidates.jsonl"
    monkeypatch.setattr(ma, "_CANDIDATES_PATH", p)
    ok = ma.record_overflow_candidate("托勒密定理", ["M25"], question_ref="3091003014006")
    assert ok is True
    import json
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["name"] == "托勒密定理" and rec["status"] == "pending"
    assert rec["mother_models"] == ["M25"] and rec["question_ref"] == "3091003014006"


# ===========================================================================
# G5·落库（build_create_bo / build_mother_bo 把 models 走 `模型:` 标签三轨）
# ===========================================================================
def _facts_with_models(*model_dicts, **extra):
    dna = {
        "main_kp": {"id": "3091003014006", "name": "隐圆"},
        "secondary_kps": [{"id": "30910001", "name": "圆周角"}],
        "exam_type": "性质判定",
        "tags": ["隐圆", "最值"],
        "models": list(model_dicts),
    }
    facts = {"subject_id": "3091", "dim1_kp_id": "3091003014006", "dna": dna}
    facts.update(extra)
    return facts


def test_models_become_prefixed_tags_in_bo():
    facts = _facts_with_models({"id": "M25", "name": "定边对定角"}, {"id": "M32", "name": "隐圆最值"})
    bo = build_create_bo({"stem": "x", "difficulty": 3, "qtype": "解答"}, facts)
    assert f"{MODEL_TAG_PREFIX}定边对定角" in bo["tags"]
    assert f"{MODEL_TAG_PREFIX}隐圆最值" in bo["tags"]
    # 普通标签仍在
    assert "隐圆" in bo["tags"] and "最值" in bo["tags"]


def test_m00_also_落库_as_tag():
    """M00 概念直用 也照落（模型维非空展示）。"""
    facts = _facts_with_models({"id": M00_ID, "name": M00_NAME})
    bo = build_create_bo({"stem": "x", "difficulty": 1, "qtype": "选择"}, facts)
    assert f"{MODEL_TAG_PREFIX}{M00_NAME}" in bo["tags"]


def test_model_tag_dedup():
    facts = _facts_with_models({"id": "M25", "name": "定边对定角"}, {"id": "M25", "name": "定边对定角"})
    bo = build_create_bo({"stem": "x", "difficulty": 3, "qtype": "解答"}, facts)
    assert bo["tags"].count(f"{MODEL_TAG_PREFIX}定边对定角") == 1


def test_mother_bo_also_carries_model_tags():
    facts = _facts_with_models({"id": "M25", "name": "定边对定角"})
    facts["stem"] = "母题题面"
    bo = build_mother_bo(facts)
    assert f"{MODEL_TAG_PREFIX}定边对定角" in bo["tags"]


# ===========================================================================
# G21·守恒维落库一致性（缺口11）：主 kp 未分类时副考点/考察类型不落
# ===========================================================================
def test_unclassified_main_kp_drops_secondary_and_examtype():
    """主 kp 未分类（dim1_kp_id 缺）→ secondaryKpIds / examType 不落（防矛盾行）。"""
    dna = {
        "main_kp": None,
        "secondary_kps": [{"id": "30910001", "name": "圆周角"}],
        "exam_type": "性质判定",
        "tags": ["x"],
    }
    facts = {"subject_id": "0", "dna": dna}  # 无 dim1_kp_id
    bo = build_create_bo({"stem": "x", "difficulty": 2, "qtype": "解答"}, facts)
    assert "secondaryKpIds" not in bo
    assert "examType" not in bo
    # 标签/难度不依赖主 kp，照落
    assert bo["tags"] == ["x"] and bo["difficult"] == 2


def test_classified_main_kp_keeps_secondary_and_examtype():
    """主 kp 在库 → 守恒维正常落。"""
    facts = _facts_with_models()  # 含 dim1_kp_id
    bo = build_create_bo({"stem": "x", "difficulty": 3, "qtype": "解答"}, facts)
    assert bo.get("secondaryKpIds") == [30910001]
    assert bo.get("examType") == "性质判定"
