# -*- coding: utf-8 -*-
"""Unit tests for first-round recipe knobs (PRD-C-009 bug fix):

The user's text accompanying the image (e.g. "5 questions, increasing
difficulty, 2 choice + 2 blank + 1 answer") used to be dropped entirely.
Now it becomes a first-class citizen:

- normalize_knobs: pure clamp/normalization of the LLM extraction;
- recipe_from_knobs: knobs -> generate recipe (empty knobs == legacy default);
- shape_check: pure code-level recipe validation (count / qtype dist / monotonic);
- gene_judge_knobs_spec + _gene_judge_prompt: Gate-A criteria realignment so
  increasing-difficulty variants are not falsely reworked;
- generate node: knobs extraction wiring + group retry + legacy-path regression;
- assemble header: "按你的要求: ..." replaces the default recipe line.

Zero LLM / zero network: every LLM touchpoint is monkeypatched.
"""

import asyncio
import json

from langchain_core.messages import HumanMessage

import agents.variant as variant_mod
from agents.variant import (
    GENERATE_PROMPT,
    PLAN_INCREASING,
    _gene_judge_prompt,
    _is_proof_like,
    _mother_facts,
    analyze,
    assemble,
    exec_add,
    exec_remove,
    exec_regenerate,
    gene_gate,
    gene_judge_knobs_spec,
    generate,
    knobs_desc,
    normalize_knobs,
    recipe_from_knobs,
    shape_check,
)

# ---------------------------------------------------------------------------
# normalize_knobs: pure clamp of LLM extraction
# ---------------------------------------------------------------------------


def test_normalize_non_dict_falls_back_to_empty():
    assert normalize_knobs(None) == {}
    assert normalize_knobs("not json") == {}
    assert normalize_knobs([1, 2]) == {}


def test_normalize_count_clamped_to_1_8():
    assert normalize_knobs({"count": 12})["count"] == 8
    assert normalize_knobs({"count": 0})["count"] == 1
    assert normalize_knobs({"count": "5"})["count"] == 5
    assert "count" not in normalize_knobs({"count": "abc"})


def test_normalize_application_qtype_maps_to_jieda_and_keeps_scene_note():
    out = normalize_knobs({"qtype_dist": {"选择题": 2, "应用题": 1}})
    assert out["qtype_dist"] == {"选择": 2, "解答": 1}
    assert "应用场景" in out["note"]  # semantic kept after alias mapping
    assert out["count"] == 3  # count derived from dist


def test_normalize_alias_duplicates_merge_and_unknown_dropped():
    out = normalize_knobs({"qtype_dist": {"解答": 1, "计算题": 1, "看图说话": 9}})
    assert out["qtype_dist"] == {"解答": 2}
    out2 = normalize_knobs({"qtype_dist": {"看图说话": 2}})
    assert "qtype_dist" not in out2


def test_normalize_dist_total_wins_over_conflicting_count():
    out = normalize_knobs({"count": 5, "qtype_dist": {"选择": 2, "填空": 1}})
    assert out["count"] == 3  # dist total is authoritative


def test_normalize_all_empty_returns_empty_dict():
    assert normalize_knobs(
        {"count": None, "difficulty_plan": None, "qtype_dist": None, "note": ""}
    ) == {}


def test_normalize_difficulty_plan_words():
    assert normalize_knobs({"difficulty_plan": "递增"})["difficulty_plan"] == PLAN_INCREASING
    assert normalize_knobs({"difficulty_plan": "INCREASING"})["difficulty_plan"] == PLAN_INCREASING
    assert "difficulty_plan" not in normalize_knobs({"difficulty_plan": "default"})
    # free-form difficulty wish is kept verbatim
    assert normalize_knobs({"difficulty_plan": "都出难题"})["difficulty_plan"] == "都出难题"


def test_normalize_does_not_mutate_input():
    parsed = {"count": 12, "qtype_dist": {"应用题": 1}}
    snapshot = json.loads(json.dumps(parsed, ensure_ascii=False))
    normalize_knobs(parsed)
    assert parsed == snapshot


def test_normalize_dist_total_clamped_to_count_max_with_visible_note():
    """The 1~8 runaway guard must not be bypassed via qtype_dist (「出10道选择题」)."""
    out = normalize_knobs({"qtype_dist": {"选择": 10}})
    assert out["count"] == 8
    assert sum(out["qtype_dist"].values()) == 8
    assert "截到 8 道" in out["note"]
    # multi-type clipping: walk in declared order until the budget runs out
    out2 = normalize_knobs({"qtype_dist": {"选择": 4, "填空": 4, "解答": 4}})
    assert out2["count"] == 8
    assert sum(out2["qtype_dist"].values()) == 8


def test_normalize_count_dist_conflict_is_externalized_in_note():
    """「出5道, 2选择1填空」: dist total wins, but the adjustment must be visible."""
    out = normalize_knobs({"count": 5, "qtype_dist": {"选择": 2, "填空": 1}})
    assert out["count"] == 3
    assert "从 5 调整为 3" in out["note"]
    # no conflict -> no adjustment note
    out2 = normalize_knobs({"count": 3, "qtype_dist": {"选择": 2, "填空": 1}})
    assert "调整" not in out2.get("note", "")


# ---------------------------------------------------------------------------
# shape_check: pure code-level recipe validation
# ---------------------------------------------------------------------------


def _items(*specs):
    """specs = (qtype, difficulty) tuples."""
    return [{"qtype": q, "difficulty": d, "stem": f"s{i}"} for i, (q, d) in enumerate(specs)]


def test_shape_check_empty_knobs_never_flags():
    assert shape_check(_items(("选择", 1)), {}) == []
    assert shape_check([], None) == []


def test_shape_check_count_mismatch():
    defects = shape_check(_items(("解答", 3), ("解答", 3)), {"count": 3})
    assert len(defects) == 1 and "数量不符" in defects[0]


def test_shape_check_qtype_dist_mismatch():
    knobs = {"count": 2, "qtype_dist": {"选择": 1, "填空": 1}}
    defects = shape_check(_items(("选择", 3), ("解答", 3)), knobs)
    assert any("题型分布不符" in d for d in defects)


def test_shape_check_qtype_dist_normalizes_item_aliases():
    # items emitted with "解答题"/"应用" still count into the 解答 bucket
    knobs = {"count": 2, "qtype_dist": {"解答": 2}}
    assert shape_check(_items(("解答题", 3), ("应用", 3)), knobs) == []


def test_shape_check_increasing_requires_monotonic_non_decreasing():
    knobs = {"difficulty_plan": PLAN_INCREASING}
    defects = shape_check(_items(("解答", 3), ("解答", 5), ("解答", 4)), knobs)
    assert any("单调不减" in d for d in defects)
    # missing difficulty counts as violation
    defects2 = shape_check(_items(("解答", 3), ("解答", None)), knobs)
    assert any("单调不减" in d for d in defects2)


def test_shape_check_increasing_accepts_non_decreasing():
    knobs = {"difficulty_plan": PLAN_INCREASING}
    assert shape_check(_items(("解答", 3), ("解答", 3), ("解答", 4)), knobs) == []


def test_shape_check_full_recipe_all_good():
    knobs = {
        "count": 3,
        "difficulty_plan": PLAN_INCREASING,
        "qtype_dist": {"选择": 1, "填空": 1, "解答": 1},
    }
    items = _items(("选择", 3), ("填空", 4), ("解答", 5))
    assert shape_check(items, knobs) == []


def test_shape_check_increasing_with_mother_difficulty_uses_gate_a_ruler():
    """Same ruler as gate A: with mother difficulty known, [3,3,4] is a defect
    (expected exactly [3,4,5]) so the group retry gets a chance to fix it —
    instead of the code gate passing and gate A inevitably warning."""
    knobs = {"difficulty_plan": PLAN_INCREASING}
    defects = shape_check(_items(("解答", 3), ("解答", 3), ("解答", 4)), knobs, 3)
    assert any("难度档与计划不符" in d for d in defects)
    assert shape_check(_items(("解答", 3), ("解答", 4), ("解答", 5)), knobs, 3) == []
    # cap at 5 deep into the plan
    items = _items(("解答", 4), ("解答", 5), ("解答", 5))
    assert shape_check(items, knobs, 4) == []


# ---------------------------------------------------------------------------
# recipe_from_knobs: legacy-default equivalence + teacher spec section
# ---------------------------------------------------------------------------

_FACTS_STATE = {
    "mother_confirmed": True,
    "analysis": {
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.9, "anchored": {"code": "100200300"}},
        "qtype": {"value": "解答", "confidence": 0.9},
    },
    "mother_dna": {
        "stem": "母题题干",
        "answer": "x=1",
        "difficulty": 3,
        "solution_skeleton": "移项合并",
    },
}


def test_recipe_empty_knobs_is_legacy_default():
    recipe = recipe_from_knobs({}, 3)
    assert recipe == {
        "n": 3,
        "n_normal": 2,
        "n_hard": 1,
        "spec": "",
        "expected_difficulties": None,
    }
    assert recipe_from_knobs(None, None)["spec"] == ""


def test_recipe_increasing_caps_difficulty_at_5():
    knobs = {"count": 5, "difficulty_plan": PLAN_INCREASING}
    recipe = recipe_from_knobs(knobs, 4)
    assert recipe["n"] == 5
    assert recipe["expected_difficulties"] == [4, 5, 5, 5, 5]
    assert "老师指定配方" in recipe["spec"]
    assert "共 5 道" in recipe["spec"]


def test_recipe_spec_carries_dist_and_note():
    knobs = {"count": 3, "qtype_dist": {"选择": 2, "解答": 1}, "note": "贴近生活场景"}
    spec = recipe_from_knobs(knobs, 3)["spec"]
    assert "选择×2" in spec and "解答×1" in spec and "贴近生活场景" in spec


# ---------------------------------------------------------------------------
# Gate-A realignment: gene_judge_knobs_spec + prompt formatting never blows up
# ---------------------------------------------------------------------------


def test_gene_spec_empty_knobs_or_irrelevant_knobs_is_none():
    assert gene_judge_knobs_spec({}, 3) is None
    assert gene_judge_knobs_spec(None, 3) is None
    # count/note alone do not touch qtype/difficulty criteria
    assert gene_judge_knobs_spec({"count": 5, "note": "x"}, 3) is None


def test_gene_spec_increasing_uses_item_level_expected_difficulty():
    """Expected difficulty comes from the item-level stamp (set by generate), not
    from the live list index — so remove/add rounds never shift the bar."""
    knobs = {"difficulty_plan": PLAN_INCREASING, "count": 3}
    assert "预期难度档 = 3" in gene_judge_knobs_spec(knobs, 3)
    assert "预期难度档 = 5" in gene_judge_knobs_spec(knobs, 5)
    # increasing plan but no stamp (edit-round item) -> no difficulty section at all
    assert gene_judge_knobs_spec(knobs, None) is None


def test_gene_spec_qtype_set_judging():
    knobs = {"qtype_dist": {"选择": 2, "填空": 2, "解答": 1}}
    spec = gene_judge_knobs_spec(knobs, None)
    assert "qtype_match 改判" in spec and "选择/填空/解答" in spec


def test_gene_judge_prompt_formats_with_and_without_knobs_spec():
    facts = _mother_facts(_FACTS_STATE)
    item = {"stem": "变式题干", "qtype": "选择", "difficulty": 4, "level": "hard"}
    base = _gene_judge_prompt(item, facts)
    assert "平行题基因比对器" in base and "老师指定配方" not in base

    spec = gene_judge_knobs_spec(
        {"difficulty_plan": PLAN_INCREASING, "qtype_dist": {"选择": 2, "解答": 1}}, 4
    )
    with_spec = _gene_judge_prompt(item, dict(facts, knobs_spec=spec))
    assert with_spec.startswith(base)
    assert "老师指定配方" in with_spec and "预期难度档 = 4" in with_spec


def test_gene_gate_node_injects_per_item_spec(monkeypatch):
    """Wiring: gene_gate hands each from_recipe item a facts copy with its own spec,
    reading the item-level expected_difficulty stamp."""
    seen = []

    async def judge_spy(item, facts):
        seen.append(facts.get("knobs_spec"))
        return {
            "qtype_match": True,
            "difficulty_match": True,
            "structure_match": True,
            "surface_swapped": True,
        }

    async def solve_stub(stem):
        return {"kp_name": "一元一次方程", "grade": "七年级上学期"}

    monkeypatch.setattr(variant_mod, "_gene_judge_one", judge_spy)
    monkeypatch.setattr(variant_mod, "_solve_one", solve_stub)
    state = dict(
        _FACTS_STATE,
        items=[
            {"stem": "a", "from_recipe": True, "expected_difficulty": 3},
            {"stem": "b", "from_recipe": True, "expected_difficulty": 4},
        ],
        knobs={"difficulty_plan": PLAN_INCREASING, "count": 2},
    )
    out = asyncio.run(gene_gate(state, {}))
    assert out["messages"] == []
    assert len(seen) == 2
    assert "预期难度档 = 3" in seen[0]
    assert "预期难度档 = 4" in seen[1]


def test_gene_gate_edit_round_items_not_judged_by_stale_recipe(monkeypatch):
    """Finding fix: after「出5道递增」, a later「再来2道简单的」add-round item carries no
    from_recipe stamp -> gene_gate must NOT inject the old increasing plan's spec
    (the teacher's deliberately-easy items would be falsely reworked/warned)."""
    seen = []

    async def judge_spy(item, facts):
        seen.append(facts.get("knobs_spec"))
        return {
            "qtype_match": True,
            "difficulty_match": True,
            "structure_match": True,
            "surface_swapped": True,
        }

    monkeypatch.setattr(variant_mod, "_gene_judge_one", judge_spy)
    state = dict(
        _FACTS_STATE,
        # old items already judged (gene present) + one fresh add-round item (no stamp)
        items=[
            {"stem": "old", "gene": {"gate": "pass"}, "from_recipe": True, "expected_difficulty": 3},
            {"stem": "easy-new", "difficulty": 2},
        ],
        knobs={"difficulty_plan": PLAN_INCREASING, "count": 5},
    )
    out = asyncio.run(gene_gate(state, {}))
    assert len(seen) == 1  # only the new item is judged
    assert seen[0] is None  # and without the stale recipe spec
    assert out["items"][1]["gene"]["gate"] == "pass"


def test_gene_gate_node_without_knobs_keeps_plain_facts(monkeypatch):
    seen = []

    async def judge_spy(item, facts):
        seen.append(facts)
        return {
            "qtype_match": True,
            "difficulty_match": True,
            "structure_match": True,
            "surface_swapped": True,
        }

    monkeypatch.setattr(variant_mod, "_gene_judge_one", judge_spy)
    state = dict(_FACTS_STATE, items=[{"stem": "a"}])
    out = asyncio.run(gene_gate(state, {}))
    assert out["items"][0]["gene"]["gate"] == "pass"
    assert "knobs_spec" not in seen[0]


# ---------------------------------------------------------------------------
# generate node: legacy regression + knobs extraction + group shape retry
# ---------------------------------------------------------------------------

_ITEM_JSON = {
    "stem": "新题",
    "answer": "x=2",
    "solution": "略",
    "qtype": "解答",
    "difficulty": 3,
    "level": "normal",
    "injected_kp": None,
}


def _gen_state(text, knobs="absent"):
    state = dict(_FACTS_STATE, messages=[HumanMessage(content=text)])
    if knobs != "absent":
        state["knobs"] = knobs
    return state


def _items_json(n, **over):
    return json.dumps([dict(_ITEM_JSON, **over) for _ in range(n)], ensure_ascii=False)


def test_generate_without_user_text_keeps_legacy_prompt_and_no_knobs_call(monkeypatch):
    """knobs=None + URL-only first message -> no extraction call, prompt byte-identical to legacy."""
    prompts = []

    async def fake_llm(messages, retry=True, **kwargs):
        prompts.append(messages[0].content)
        return _items_json(3)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _gen_state("https://oss.example.com/q.png")
    out = asyncio.run(generate(state, {}))

    assert len(prompts) == 1  # no KNOBS extraction round-trip
    expected = GENERATE_PROMPT.format(n=3, n_normal=2, n_hard=1, **_mother_facts(state))
    assert prompts[0] == expected  # legacy behavior unchanged, no teacher section
    assert out["knobs"] == {}  # extracted-empty is persisted (never re-extract)
    assert out["shape_defects"] == []
    assert len(out["items"]) == 3 and "check" not in out["items"][0]


def test_generate_extracts_knobs_from_first_round_text(monkeypatch):
    calls = []

    async def fake_llm(messages, retry=True, **kwargs):
        prompt = messages[0].content
        calls.append(prompt)
        if "出题配方" in prompt:  # KNOBS_PROMPT round
            return json.dumps(
                {
                    "count": 5,
                    "difficulty_plan": "递增",
                    "qtype_dist": {"选择": 2, "填空": 2, "应用题": 1},
                    "note": "",
                },
                ensure_ascii=False,
            )
        # generate round: emit a recipe-conforming group
        # (increasing plan from mother difficulty 3 -> expected exactly [3,4,5,5,5])
        items = [
            dict(_ITEM_JSON, qtype="选择", difficulty=3),
            dict(_ITEM_JSON, qtype="选择", difficulty=4),
            dict(_ITEM_JSON, qtype="填空", difficulty=5, level="hard"),
            dict(_ITEM_JSON, qtype="填空", difficulty=5, level="hard"),
            dict(_ITEM_JSON, qtype="解答", difficulty=5, level="hard"),
        ]
        return json.dumps(items, ensure_ascii=False)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _gen_state("https://oss.example.com/q.png 出5道，难度递增，2道选择2道填空1道应用题")
    out = asyncio.run(generate(state, {}))

    assert out["knobs"]["count"] == 5
    assert out["knobs"]["difficulty_plan"] == PLAN_INCREASING
    assert out["knobs"]["qtype_dist"] == {"选择": 2, "填空": 2, "解答": 1}
    assert "应用场景" in out["knobs"]["note"]
    gen_prompt = calls[-1]
    assert "老师指定配方" in gen_prompt and "共 5 道" in gen_prompt
    assert out["shape_defects"] == []
    assert len(out["items"]) == 5
    # recipe stamps: gate-A reads these item-level fields, never the live list index
    assert all(it["from_recipe"] is True for it in out["items"])
    assert [it["expected_difficulty"] for it in out["items"]] == [3, 4, 5, 5, 5]


def test_generate_knobs_extraction_failure_falls_back_to_default(monkeypatch):
    async def fake_llm(messages, retry=True, **kwargs):
        if "出题配方" in messages[0].content:
            raise RuntimeError("relay down")  # extraction must never block generation
        return _items_json(3)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    out = asyncio.run(generate(_gen_state("https://o.ss/x.png 出5道难度递增的题"), {}))
    assert out["knobs"] == {}
    assert len(out["items"]) == 3  # legacy default shape


def test_generate_existing_knobs_skip_re_extraction(monkeypatch):
    prompts = []

    async def fake_llm(messages, retry=True, **kwargs):
        prompts.append(messages[0].content)
        return _items_json(2)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _gen_state("再出一组", knobs={"count": 2})
    out = asyncio.run(generate(state, {}))
    assert all("出题配方" not in p for p in prompts)  # no re-extraction across turns
    assert out["knobs"] == {"count": 2}
    assert out["shape_defects"] == []


def test_generate_shape_defect_triggers_one_group_retry(monkeypatch):
    calls = []

    async def fake_llm(messages, retry=True, **kwargs):
        prompt = messages[0].content
        calls.append(prompt)
        if "[配方校验反馈]" in prompt:
            return _items_json(2)  # retry conforms
        return _items_json(3)  # first draft: wrong count

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    out = asyncio.run(generate(_gen_state("url-free", knobs={"count": 2}), {}))
    assert len(calls) == 2 and "数量不符" in calls[1]
    assert len(out["items"]) == 2
    assert out["shape_defects"] == []


def test_generate_retry_still_failing_accepts_with_visible_defects(monkeypatch):
    calls = []

    async def fake_llm(messages, retry=True, **kwargs):
        calls.append(messages[0].content)
        return _items_json(3)  # both drafts: wrong count

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    out = asyncio.run(generate(_gen_state("url-free", knobs={"count": 2}), {}))
    assert len(calls) == 2  # exactly one retry, never loops
    assert len(out["items"]) == 3  # accepted anyway (non-blocking)
    assert any("数量不符" in d for d in out["shape_defects"])


# ---------------------------------------------------------------------------
# assemble header + knobs_desc
# ---------------------------------------------------------------------------


def test_knobs_desc_human_readable():
    desc = knobs_desc(
        {
            "count": 5,
            "difficulty_plan": PLAN_INCREASING,
            "qtype_dist": {"选择": 2, "填空": 2, "解答": 1},
        }
    )
    assert desc == "5 道·难度递增·2选择+2填空+1解答"
    assert knobs_desc({}) == ""
    assert knobs_desc(None) == ""


def _assembled(state):
    out = asyncio.run(assemble(state, {}))
    return out["messages"][0].content


def test_assemble_header_shows_teacher_recipe_and_defects():
    state = dict(
        _FACTS_STATE,
        items=[{"stem": "s", "answer": "a", "solution": "x", "check": {"badge": "ok"}}],
        knobs={"count": 5, "difficulty_plan": PLAN_INCREASING},
        shape_defects=["数量不符：要求 5 道，实出 1 道"],
    )
    text = _assembled(state)
    assert "按你的要求：5 道·难度递增" in text
    assert "⚠ 配方未完全满足：数量不符" in text
    assert "配方：默认" not in text


def test_assemble_header_without_knobs_keeps_legacy_text():
    state = dict(
        _FACTS_STATE,
        items=[{"stem": "s", "answer": "a", "solution": "x", "check": {"badge": "ok"}}],
    )
    text = _assembled(state)
    assert "配方：默认 3 = 2 普通 + 1 难" in text
    assert "按你的要求" not in text and "配方未完全满足" not in text


# ---------------------------------------------------------------------------
# qtype routing sanity: "解答(应用)" style items still take the sympy path
# ---------------------------------------------------------------------------


def test_application_style_jieda_is_not_proof_like():
    assert not _is_proof_like("解答", "某商场打折促销，一件衣服原价 200 元……求实际付款金额。")
    assert not _is_proof_like("解答(应用)", "甲乙两地相距 120 km，一辆汽车……")
    # existing proof routing unaffected
    assert _is_proof_like("解答", "求证：三角形 ABC 中……")
    assert _is_proof_like("证明", "任意题干")


# ---------------------------------------------------------------------------
# analyze: knobs reset/re-extraction across mothers in the same thread
# (fix: second image's accompanying text used to be dropped and the previous
#  mother's recipe wrongly re-applied)
# ---------------------------------------------------------------------------

_ANALYZE_JSON = json.dumps(
    {
        "is_question_image": True,
        "images_count": 1,
        "questions_in_image": 1,
        "grade": {"value": "七年级上学期", "confidence": 0.9},
        "kp": {"value": "一元一次方程", "confidence": 0.8},
        "qtype": {"value": "解答", "confidence": 0.9},
        "stem": "题干",
        "answer": "x=1",
        "difficulty": 3,
    },
    ensure_ascii=False,
)


def _patch_analyze_llm(monkeypatch):
    async def fake_llm(messages, retry=True, **kwargs):
        return _ANALYZE_JSON

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)


def test_analyze_new_mother_with_text_re_extracts_knobs(monkeypatch):
    """Same thread, second image + new text -> old knobs overwritten, new text honored."""
    _patch_analyze_llm(monkeypatch)

    async def fake_extract(state):
        return {"count": 2, "qtype_dist": {"填空": 2}}

    monkeypatch.setattr(variant_mod, "_extract_knobs", fake_extract)
    state = {
        "messages": [HumanMessage(content="https://oss/new.png 这题出2道填空")],
        "image_url": "https://oss/old.png",
        "knobs": {"count": 5, "difficulty_plan": PLAN_INCREASING},
        "shape_defects": ["数量不符：要求 5 道，实出 4 道"],
    }
    out = asyncio.run(analyze(state, {}))
    assert out["knobs"] == {"count": 2, "qtype_dist": {"填空": 2}}
    assert out["shape_defects"] == []


def test_analyze_new_mother_without_text_resets_knobs(monkeypatch):
    """New image with no accompanying text -> old mother's recipe must NOT leak."""
    _patch_analyze_llm(monkeypatch)
    called = []

    async def fake_extract(state):
        called.append(1)
        return {}

    monkeypatch.setattr(variant_mod, "_extract_knobs", fake_extract)
    state = {
        "messages": [HumanMessage(content="https://oss/new.png")],
        "image_url": "https://oss/old.png",
        "knobs": {"count": 5, "difficulty_plan": PLAN_INCREASING},
    }
    out = asyncio.run(analyze(state, {}))
    assert called == []  # empty user text -> no extraction round-trip
    assert out["knobs"] == {}  # reset, generate falls back to default recipe


def test_analyze_same_url_repaste_preserves_knobs(monkeypatch):
    """Clarify detour: re-pasting the SAME image with a clarify answer (no new recipe)
    must keep the first turn's extracted recipe."""
    _patch_analyze_llm(monkeypatch)

    async def fake_extract(state):
        return {}  # clarify answer carries no recipe words

    monkeypatch.setattr(variant_mod, "_extract_knobs", fake_extract)
    state = {
        "messages": [HumanMessage(content="https://oss/same.png 是七年级上")],
        "image_url": "https://oss/same.png",
        "knobs": {"count": 5, "difficulty_plan": PLAN_INCREASING},
    }
    out = asyncio.run(analyze(state, {}))
    assert out["knobs"] == {"count": 5, "difficulty_plan": PLAN_INCREASING}


def test_generate_re_extracts_after_analyze_reset(monkeypatch):
    """End-to-end of the leak fix: knobs reset to None (new mother) -> generate
    extracts fresh from the current human text."""
    calls = []

    async def fake_llm(messages, retry=True, **kwargs):
        prompt = messages[0].content
        calls.append(prompt)
        if "出题配方" in prompt:
            return json.dumps({"count": 2, "difficulty_plan": None, "qtype_dist": None, "note": ""})
        return _items_json(2)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    state = _gen_state("https://oss/new.png 出2道", knobs=None)
    out = asyncio.run(generate(state, {}))
    assert any("出题配方" in p for p in calls)
    assert out["knobs"] == {"count": 2}


# ---------------------------------------------------------------------------
# generate: empty-draft retry + friendly failure (never a silent empty turn)
# ---------------------------------------------------------------------------


def test_generate_empty_draft_with_knobs_retries_then_replies_friendly(monkeypatch):
    calls = []

    async def fake_llm(messages, retry=True, **kwargs):
        calls.append(messages[0].content)
        return "not json at all"  # both drafts unparseable

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    out = asyncio.run(generate(_gen_state("url-free", knobs={"count": 2}), {}))
    assert len(calls) == 2 and "[配方校验反馈]" in calls[1]  # empty draft still retried once
    assert out["items"] == []
    assert any("数量不符" in d for d in out["shape_defects"])
    assert out["messages"] and "没能产出" in out["messages"][0].content  # friendly, not silent


def test_generate_empty_draft_without_knobs_still_replies_friendly(monkeypatch):
    async def fake_llm(messages, retry=True, **kwargs):
        return "garbage"

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    out = asyncio.run(generate(_gen_state("https://o.ss/q.png"), {}))
    assert out["items"] == []
    assert out["messages"] and "没能产出" in out["messages"][0].content


# ---------------------------------------------------------------------------
# edit rounds: stale shape_defects cleared (teacher edits = manual takeover)
# ---------------------------------------------------------------------------


def _edit_state(ops, items=None):
    return dict(
        _FACTS_STATE,
        items=items if items is not None else [{"stem": "a", "check": {"badge": "ok"}}],
        pending={"ops": ops},
        shape_defects=["数量不符：要求 5 道，实出 4 道"],
    )


def test_exec_remove_clears_stale_shape_defects():
    out = asyncio.run(exec_remove(_edit_state([{"action": "remove", "index": 1}]), {}))
    assert out["shape_defects"] == []


def test_exec_add_clears_stale_shape_defects_and_adds_unstamped_items(monkeypatch):
    async def fake_llm(messages, retry=True, **kwargs):
        return _items_json(1)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    out = asyncio.run(exec_add(_edit_state([{"action": "add", "count": 1}]), {}))
    assert out["shape_defects"] == []
    assert "from_recipe" not in out["items"][-1]  # add-round item carries no recipe stamp


def test_exec_regenerate_clears_defects_and_carries_recipe_stamps(monkeypatch):
    async def fake_llm(messages, retry=True, **kwargs):
        return json.dumps(dict(_ITEM_JSON, stem="重出的题"), ensure_ascii=False)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    items = [
        {"stem": "v1", "qtype": "解答", "difficulty": 4, "level": "hard",
         "from_recipe": True, "expected_difficulty": 4, "check": {"badge": "ok"}}
    ]
    out = asyncio.run(
        exec_regenerate(_edit_state([{"action": "regenerate", "index": 1}], items=items), {})
    )
    assert out["shape_defects"] == []
    new_item = out["items"][0]
    assert new_item["stem"] == "重出的题" and "check" not in new_item
    # stamps follow the item (it still occupies the original plan slot)
    assert new_item["from_recipe"] is True and new_item["expected_difficulty"] == 4
