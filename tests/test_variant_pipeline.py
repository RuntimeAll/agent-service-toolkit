# -*- coding: utf-8 -*-
"""Unit tests for PRD-C-012 pipeline rework inside agents.variant.

Coverage map (all LLM calls monkeypatched -> zero network):
- 4a payload-first verification: _machine_verify consumes item.verify_payload
  when kind is legal (no extraction round-trip); kind none / illegal / missing
  falls back to the legacy _extract_payload path; verdict still reads ONLY
  math_verify.verify() (sympy) -- never an LLM self-grade
- verify_payload stays an item-internal field: carried by
  _parse_generated_items / _regen_once, NEVER leaked into artifact frames
  (explicit whitelist in _artifact_payload)
- P2 incremental parsing: _iter_complete_items returns only brace-balanced,
  json-parseable objects that carry a "stem" key (half items never returned;
  escaped quotes / braces inside strings / nested payload dicts / {"items":[...]}
  wrappers all handled)
- P2 per-item concurrency: gene_gate / solve_explain gather with
  Semaphore(GATE_CONCURRENCY), results backfilled in original index order
- P2 eager generate: stream callback discovers complete items, spawns the
  gate chain per item, emits cumulative partial artifact frames (fake writer
  captures frames emitted from create_task children -> frame order verified),
  dropped items become _dropped sentinels collected later by solve_explain
- shape-defect group retry: eager results discarded, retry items returned
  un-judged (downstream nodes re-gate them -- macro DAG unchanged)
- prompt re-layout (cache-friendly prefix): shared _PAYLOAD_CONTRACT pinned in
  GENERATE/REGEN/EXTRACT, fixed segments precede variable placeholders, all
  templates still .format() cleanly, trace markers still resolve
"""

import asyncio
import json

from langchain_core.messages import ChatMessage, HumanMessage

import agents.variant as variant_mod
from agents import math_verify
from agents.variant import (
    EXTRACT_PROMPT,
    GENE_JUDGE_PROMPT,
    GENERATE_PROMPT,
    PARSE_PROMPT,
    REGEN_PROMPT,
    SOLVE_PROMPT,
    _iter_complete_items,
    _machine_verify,
    _parse_generated_items,
    _regen_once,
    generate,
    gene_gate,
    solve_explain,
)

# ---------------------------------------------------------------------------
# shared fixtures
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

_FACTS = {"kp_name": "kp-x", "grade": "g7", "qtype": "解答"}

_GEN_ITEM = {
    "stem": "",
    "answer": "x=2",
    "solution": "略",
    "qtype": "解答",
    "difficulty": 3,
    "level": "normal",
    "injected_kp": None,
}


def _capture_frames(monkeypatch):
    captured: list = []
    monkeypatch.setattr(variant_mod, "get_stream_writer", lambda: captured.append)
    return captured


def _artifact_frames(captured):
    out = []
    for msg in captured:
        assert isinstance(msg, ChatMessage) and msg.role == "custom"
        if "artifact" in msg.content[0]:
            out.append(msg.content[0]["artifact"])
    return out


def _stage_frames(captured):
    return [m.content[0]["stage"] for m in captured if "stage" in m.content[0]]


# ---------------------------------------------------------------------------
# _iter_complete_items: incremental complete-item discovery (P2)
# ---------------------------------------------------------------------------


def test_iter_complete_items_half_item_never_returned():
    full = json.dumps([{"stem": "q1", "answer": "1"}, {"stem": "q2", "answer": "2"}], ensure_ascii=False)
    cut = full.index('"q2"')  # truncate inside the second item
    assert [it["stem"] for it in _iter_complete_items(full[:cut])] == ["q1"]
    assert [it["stem"] for it in _iter_complete_items(full)] == ["q1", "q2"]


def test_iter_complete_items_escaped_quotes_and_braces_inside_strings():
    item = {"stem": 'f(x) = {x | x > 0} 且含 "引号" 与 \\ 反斜杠 和 }', "answer": "1"}
    acc = "[" + json.dumps(item, ensure_ascii=False) + ","  # next item not started yet
    got = _iter_complete_items(acc)
    assert len(got) == 1
    assert got[0]["stem"] == item["stem"]


def test_iter_complete_items_nested_payload_and_wrapper_not_double_counted():
    payload = {"kind": "numeric", "expr": "1+1", "claimed": "2"}
    wrapper = {"items": [{"stem": "a", "verify_payload": payload}, {"stem": "b"}]}
    got = _iter_complete_items(json.dumps(wrapper, ensure_ascii=False))
    # nested verify_payload dict and the outer {"items":[...]} wrapper are not items
    assert [it["stem"] for it in got] == ["a", "b"]
    assert got[0]["verify_payload"] == payload


def test_iter_complete_items_garbage_and_no_stem():
    assert _iter_complete_items("") == []
    assert _iter_complete_items(None) == []
    assert _iter_complete_items('```json\n[ {"no_stem": 1} ]') == []
    assert _iter_complete_items("{ broken json }") == []


def test_iter_complete_items_monotone_over_growing_stream():
    full = json.dumps([{"stem": f"s{i}"} for i in range(4)], ensure_ascii=False)
    seen = 0
    for cut in range(len(full) + 1):
        n = len(_iter_complete_items(full[:cut]))
        assert n >= seen  # discovery is monotone: once complete, stays complete
        seen = n
    assert seen == 4


# ---------------------------------------------------------------------------
# 4a payload-first _machine_verify (+ fallback semantics unchanged)
# ---------------------------------------------------------------------------


def test_machine_verify_payload_first_skips_extraction(monkeypatch):
    async def must_not_extract(*a, **k):
        raise AssertionError("payload-first: _extract_payload must not be called")

    monkeypatch.setattr(variant_mod, "_extract_payload", must_not_extract)
    item = {
        "stem": "1+1=?",
        "answer": "2",
        "qtype": "解答",
        "verify_payload": {"kind": "numeric", "expr": "1+1", "claimed": "2"},
    }
    res = asyncio.run(_machine_verify(item, "2"))
    assert res["verdict"] == math_verify.PASS


def test_machine_verify_payload_fail_verdict_comes_from_sympy(monkeypatch):
    # verdict semantics unchanged: sympy says the claimed answer is wrong -> FAIL
    async def must_not_extract(*a, **k):
        raise AssertionError("must not extract")

    monkeypatch.setattr(variant_mod, "_extract_payload", must_not_extract)
    item = {
        "stem": "s",
        "answer": "3",
        "qtype": "解答",
        "verify_payload": {"kind": "numeric", "expr": "1+1", "claimed": "3"},
    }
    res = asyncio.run(_machine_verify(item, "3"))
    assert res["verdict"] == math_verify.FAIL


def test_machine_verify_falls_back_on_none_invalid_or_missing_payload(monkeypatch):
    extracted = []

    async def fake_extract(stem, answer, solved_answer, qtype):
        extracted.append(stem)
        return {"kind": "numeric", "expr": "2*2", "claimed": "4"}

    monkeypatch.setattr(variant_mod, "_extract_payload", fake_extract)
    payload_variants = [
        {"kind": "none", "reason": "应用题难建模"},  # explicit none
        {"kind": "bogus_kind"},  # illegal kind
        "not-a-dict",  # wrong type
        None,  # missing
    ]
    for i, payload in enumerate(payload_variants):
        item = {"stem": f"s{i}", "answer": "4", "qtype": "解答"}
        if payload is not None:
            item["verify_payload"] = payload
        res = asyncio.run(_machine_verify(item, "4"))
        assert res["verdict"] == math_verify.PASS
    assert extracted == ["s0", "s1", "s2", "s3"]  # every case fell back to extraction


def test_machine_verify_payload_first_timeout_still_degrades(monkeypatch):
    # G5 anti-hang budget applies to the payload-first path too
    import time as _time

    def stalled_verify(payload):
        _time.sleep(1)
        return {"verdict": "pass", "detail": "too late", "computed": "1"}

    monkeypatch.setattr(variant_mod.math_verify, "verify", stalled_verify)
    monkeypatch.setattr(variant_mod, "VERIFY_TIMEOUT_S", 0.2)
    item = {
        "stem": "s",
        "answer": "2",
        "qtype": "解答",
        "verify_payload": {"kind": "numeric", "expr": "1+1", "claimed": "2"},
    }
    res = asyncio.run(_machine_verify(item, "2"))
    assert res["verdict"] == math_verify.DEGRADE


# ---------------------------------------------------------------------------
# verify_payload flows inside items only (carry + whitelist no-leak)
# ---------------------------------------------------------------------------


def test_parse_generated_items_carries_dict_payload_drops_garbage():
    payload = {"kind": "equation_solve", "equations": ["x-1=0"], "unknowns": ["x"], "claimed": ["1"]}
    text = json.dumps(
        [
            dict(_GEN_ITEM, stem="a", verify_payload=payload),
            dict(_GEN_ITEM, stem="b", verify_payload="garbage-string"),
            dict(_GEN_ITEM, stem="c"),
        ],
        ensure_ascii=False,
    )
    items = _parse_generated_items(text, _FACTS)
    assert items[0]["verify_payload"] == payload
    assert "verify_payload" not in items[1]  # non-dict payload never carried
    assert "verify_payload" not in items[2]


def test_regen_once_carries_verify_payload(monkeypatch):
    payload = {"kind": "numeric", "expr": "3*3", "claimed": "9"}

    async def fake_llm(messages, retry=True, **kw):
        return json.dumps(
            {"stem": "new", "answer": "9", "solution": "s", "verify_payload": payload},
            ensure_ascii=False,
        )

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake_llm)
    draft = asyncio.run(_regen_once({"stem": "old", "qtype": "解答"}, _FACTS))
    assert draft["verify_payload"] == payload


def test_artifact_payload_never_leaks_internal_item_keys():
    state = {
        "items": [
            {
                "stem": "s",
                "answer": "1",
                "verify_payload": {"kind": "numeric", "expr": "1", "claimed": "1"},
                "check": {"badge": "ok", "verify": "sympy_pass"},
                "gene": {"gate": "pass"},
                "from_recipe": True,
                "expected_difficulty": 3,
            }
        ]
    }
    art = variant_mod._artifact_payload(state)
    it = art["items"][0]
    assert "verify_payload" not in it and "_dropped" not in it
    # explicit whitelist: exactly the FE contract keys, nothing internal
    assert set(it) == {
        "index", "stem", "answer", "solution", "qtype", "difficulty",
        "level", "verify", "tier", "gene", "persisted",
    }


# ---------------------------------------------------------------------------
# P2 per-item concurrency: order preserved + Semaphore(3) cap
# ---------------------------------------------------------------------------


def test_solve_explain_concurrent_results_keep_input_order(monkeypatch):
    finished = []

    async def fake_check(item, facts, idx, total):
        await asyncio.sleep(0.05 if idx == 0 else 0.005)  # first item finishes LAST
        finished.append(idx)
        item["check"] = {"badge": "ok"}
        return item, None

    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)
    items = [{"stem": f"s{i}"} for i in range(3)]
    out = asyncio.run(solve_explain(dict(_FACTS_STATE, items=items), {}))
    assert finished[0] != 0  # truly concurrent: item 0 was not first to finish
    # backfill by original index order, not completion order
    assert [it["stem"] for it in out["items"]] == ["s0", "s1", "s2"]
    assert out["dropped_notes"] == []


def test_solve_explain_concurrency_capped_at_gate_concurrency(monkeypatch):
    peak = {"now": 0, "max": 0}

    async def fake_check(item, facts, idx, total):
        peak["now"] += 1
        peak["max"] = max(peak["max"], peak["now"])
        await asyncio.sleep(0.02)
        peak["now"] -= 1
        item["check"] = {"badge": "ok"}
        return item, None

    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)
    items = [{"stem": f"s{i}"} for i in range(6)]
    out = asyncio.run(solve_explain(dict(_FACTS_STATE, items=items), {}))
    assert peak["max"] <= variant_mod.GATE_CONCURRENCY  # semaphore enforced
    assert peak["max"] >= 2  # and actually parallel
    assert len(out["items"]) == 6


def test_gene_gate_concurrent_results_keep_input_order(monkeypatch):
    async def fake_gene(item, facts, idx, total):
        await asyncio.sleep(0.04 if idx == 0 else 0.005)
        item["gene"] = {"gate": "pass"}
        return item

    monkeypatch.setattr(variant_mod, "_gene_one_item", fake_gene)
    items = [{"stem": f"s{i}"} for i in range(3)]
    out = asyncio.run(gene_gate(dict(_FACTS_STATE, items=items), {}))
    assert [it["stem"] for it in out["items"]] == ["s0", "s1", "s2"]
    assert all(it["gene"]["gate"] == "pass" for it in out["items"])


def test_solve_explain_sentinel_collected_without_recheck(monkeypatch):
    async def must_not_check(*a, **k):
        raise AssertionError("checked/sentinel items must not re-enter Gate-B")

    monkeypatch.setattr(variant_mod, "_check_one_item", must_not_check)
    items = [
        {"stem": "keep", "check": {"badge": "ok"}},
        {"stem": "bad", "gene": {"gate": "pass"}, "_dropped": "1 道题已剔除-叙事"},
    ]
    out = asyncio.run(solve_explain(dict(_FACTS_STATE, items=items), {}))
    assert [it["stem"] for it in out["items"]] == ["keep"]  # sentinel never shown
    assert out["dropped_notes"] == ["1 道题已剔除-叙事"]


# ---------------------------------------------------------------------------
# P2 eager generate: stream-discovered items go through the gate chain,
# cumulative partial frames emitted from create_task children (fake writer)
# ---------------------------------------------------------------------------


def _gen_state(text="https://o.ss/q.png", knobs=None):
    state = dict(_FACTS_STATE, messages=[HumanMessage(content=text)])
    state["knobs"] = {} if knobs is None else knobs
    return state


def _patch_gate_chain(monkeypatch, drop_idx=None):
    async def fake_gene(item, facts, idx, total):
        item["gene"] = {"gate": "pass"}
        return item

    async def fake_check(item, facts, idx, total):
        if drop_idx is not None and idx == drop_idx:
            return None, f"note-{idx}"
        item["check"] = {"badge": "ok", "tier": "verified"}
        return item, None

    monkeypatch.setattr(variant_mod, "_gene_one_item", fake_gene)
    monkeypatch.setattr(variant_mod, "_check_one_item", fake_check)


def _patch_streaming_llm(monkeypatch, n=3, retry_payloads=None):
    parts = [json.dumps(dict(_GEN_ITEM, stem=f"s{i}"), ensure_ascii=False) for i in range(n)]
    text = "[" + ",".join(parts) + "]"

    async def fake(messages, retry=True, *, on_delta=None, **kw):
        prompt = messages[0].content
        if retry_payloads is not None and "[配方校验反馈]" in prompt:
            return json.dumps(retry_payloads, ensure_ascii=False)
        if on_delta is not None:
            acc = "["
            for p in parts:  # feed item-by-item: each call closes exactly one more item
                acc += p + ","
                on_delta(acc)
        return text

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)


def test_generate_eager_judges_items_and_emits_partial_frames_in_order(monkeypatch):
    captured = _capture_frames(monkeypatch)
    _patch_gate_chain(monkeypatch)
    _patch_streaming_llm(monkeypatch, 3)

    out = asyncio.run(generate(_gen_state(), {}))
    # product = items already carrying gene+check, in generation order
    assert [it["stem"] for it in out["items"]] == ["s0", "s1", "s2"]
    assert all(it["gene"]["gate"] == "pass" for it in out["items"])
    assert all(it["check"]["badge"] == "ok" for it in out["items"])

    # partial frames: cumulative, generation order, expected_total stamped
    # (frames written from inside create_task children -> writer reachable there)
    partials = [a for a in _artifact_frames(captured) if a.get("partial")]
    assert [len(a["items"]) for a in partials] == [1, 2, 3]
    assert all(a["partial"] is True and a["expected_total"] == 3 for a in partials)
    assert [it["stem"] for it in partials[-1]["items"]] == ["s0", "s1", "s2"]
    for a in partials:
        for it in a["items"]:
            assert "verify_payload" not in it and "_dropped" not in it

    # thought-bar narration: per-item completion progress
    details = [s.get("detail") for s in _stage_frames(captured)]
    assert "第 1/3 道完成" in details and "第 3/3 道完成" in details


def test_generate_eager_drop_becomes_sentinel_then_solve_explain_collects(monkeypatch):
    captured = _capture_frames(monkeypatch)
    _patch_gate_chain(monkeypatch, drop_idx=1)
    _patch_streaming_llm(monkeypatch, 3)

    out = asyncio.run(generate(_gen_state(), {}))
    assert [it.get("stem") for it in out["items"]] == ["s0", "s1", "s2"]
    assert out["items"][1].get("_dropped") == "note-1"  # sentinel, not silently lost
    assert "check" not in out["items"][1]
    # partial frames never show the dropped item
    partials = [a for a in _artifact_frames(captured) if a.get("partial")]
    assert [len(a["items"]) for a in partials] == [1, 1, 2]

    # downstream solve_explain (macro DAG unchanged) removes the sentinel + keeps narration
    out2 = asyncio.run(solve_explain(dict(_FACTS_STATE, items=out["items"]), {}))
    assert [it["stem"] for it in out2["items"]] == ["s0", "s2"]
    assert out2["dropped_notes"] == ["note-1"]


def test_generate_shape_retry_discards_eager_results(monkeypatch):
    _capture_frames(monkeypatch)
    _patch_gate_chain(monkeypatch)
    retry_payloads = [dict(_GEN_ITEM, stem=f"r{i}") for i in range(2)]
    _patch_streaming_llm(monkeypatch, 3, retry_payloads=retry_payloads)

    # teacher asked for 2, first (streamed) draft has 3 -> defect -> group retry wins
    out = asyncio.run(generate(_gen_state(knobs={"count": 2}), {}))
    assert [it["stem"] for it in out["items"]] == ["r0", "r1"]
    # eager results discarded: retry items go out un-judged -> downstream gates re-run
    assert all("gene" not in it and "check" not in it and "_dropped" not in it for it in out["items"])
    assert out["shape_defects"] == []


def test_generate_eager_merge_rejects_misaligned_indices(monkeypatch):
    # adversarial fix: a stray no-stem dict is kept by the full parse (stem=None)
    # but skipped by the incremental parser -> indices shift by one; the merge must
    # not graft eager gene/check badges onto a different question.
    _capture_frames(monkeypatch)
    _patch_gate_chain(monkeypatch)
    parts = [
        json.dumps(dict(_GEN_ITEM, stem="s0"), ensure_ascii=False),
        json.dumps({"noise": 1}, ensure_ascii=False),  # full parse keeps, eager skips
        json.dumps(dict(_GEN_ITEM, stem="s1"), ensure_ascii=False),
    ]
    text = "[" + ",".join(parts) + "]"

    async def fake(messages, retry=True, *, on_delta=None, **kw):
        if on_delta is not None:
            acc = "["
            for p in parts:
                acc += p + ","
                on_delta(acc)
        return text

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
    out = asyncio.run(generate(_gen_state(), {}))
    stems = [it.get("stem") for it in out["items"]]
    assert stems[0] == "s0" and not stems[1] and stems[2] == "s1"
    assert "check" in out["items"][0]  # idx 0 aligned -> eager result adopted
    assert "check" not in out["items"][1]  # junk slot: badge NOT grafted onto it
    assert "check" not in out["items"][2]  # misaligned tail -> downstream re-judges


def test_generate_eager_discarded_on_stream_restart(monkeypatch):
    # adversarial fix: relay failover mid-stream / empty-return retry restreams the
    # accumulated text from scratch; eager results from the dead stream are stale
    # (cross-stream index mix) -> whole eager round discarded, downstream re-gates.
    _capture_frames(monkeypatch)
    _patch_gate_chain(monkeypatch)
    dead = [json.dumps(dict(_GEN_ITEM, stem=f"dead{i}"), ensure_ascii=False) for i in range(2)]
    live = [json.dumps(dict(_GEN_ITEM, stem=f"live{i}"), ensure_ascii=False) for i in range(3)]
    text = "[" + ",".join(live) + "]"

    async def fake(messages, retry=True, *, on_delta=None, **kw):
        if on_delta is not None:
            acc = "["
            for p in dead:  # first relay streams two complete items then dies
                acc += p + ","
                on_delta(acc)
            acc = "["  # failover: next relay restreams from scratch
            for p in live:
                acc += p + ","
                on_delta(acc)
        return text

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
    out = asyncio.run(generate(_gen_state(), {}))
    stems = [it["stem"] for it in out["items"]]
    assert stems == ["live0", "live1", "live2"]  # dead-stream items never surface
    # eager round voided wholesale: no dead-stream badge grafted anywhere
    assert all("check" not in it and "gene" not in it for it in out["items"])


def test_generate_orphan_eager_tasks_cancelled_on_llm_raise(monkeypatch):
    # adversarial fix: if the main generate stream raises (all relays exhausted),
    # spawned eager tasks must be cancelled/awaited -- no orphan background burn.
    _capture_frames(monkeypatch)

    async def slow_gene(item, facts, idx, total):
        await asyncio.sleep(30)  # would burn long after the node died
        return item

    monkeypatch.setattr(variant_mod, "_gene_one_item", slow_gene)
    part = json.dumps(dict(_GEN_ITEM, stem="s0"), ensure_ascii=False)

    async def fake(messages, retry=True, *, on_delta=None, **kw):
        if on_delta is not None:
            on_delta("[" + part + ",")
            await asyncio.sleep(0)  # let the eager task actually start
            raise RuntimeError("all relays down")
        return "[]"

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)

    async def main():
        try:
            await generate(_gen_state(), {})
        except RuntimeError:
            pass
        # after generate unwinds, no eager gate-chain task may still be pending
        return [
            t for t in asyncio.all_tasks()
            if getattr(t.get_coro(), "__name__", "") == "_eager_chain" and not t.done()
        ]

    assert asyncio.run(main()) == []


def test_machine_verify_rejects_inconsistent_claimed_payload(monkeypatch):
    # adversarial fix: payload-first only when claimed == the displayed answer;
    # a self-consistent but unrelated payload must fall back to extraction.
    extracted = []

    async def fake_extract(stem, answer, solved_answer, qtype):
        extracted.append(stem)
        return {"kind": "numeric", "expr": "2*2", "claimed": "4"}

    monkeypatch.setattr(variant_mod, "_extract_payload", fake_extract)
    item = {
        "stem": "s",
        "answer": "4",  # teacher-visible answer
        "qtype": "解答",
        # model wrote a payload that verifies a DIFFERENT value (9)
        "verify_payload": {"kind": "numeric", "expr": "3*3", "claimed": "9"},
    }
    res = asyncio.run(_machine_verify(item, "4"))
    assert res["verdict"] == math_verify.PASS  # via extraction fallback
    assert extracted == ["s"]  # inconsistent claimed -> payload rejected


def test_machine_verify_consistent_claimed_payload_still_first():
    # consistency gate is lenient containment: answer "x=2" with claimed "2" passes
    item = {
        "stem": "s",
        "answer": "$x_1=2, x_2=3$",
        "qtype": "解答",
        "verify_payload": {
            "kind": "equation_solve",
            "equations": ["x**2-5*x+6=0"],
            "unknowns": ["x"],
            "claimed": ["2", "3"],
        },
    }
    res = asyncio.run(_machine_verify(item, None))
    assert res["verdict"] == math_verify.PASS  # payload-first, no extraction needed


def test_add_prompt_carries_payload_contract():
    # adversarial fix: edit-round add items also self-carry verify_payload
    assert variant_mod._PAYLOAD_CONTRACT in variant_mod.ADD_PROMPT
    assert '"verify_payload"' in variant_mod.ADD_PROMPT
    # fixed contract segment precedes the variable mother-DNA segment (cache prefix)
    assert variant_mod.ADD_PROMPT.index("expr_equiv") < variant_mod.ADD_PROMPT.index("{skeleton}")
    # trace marker still resolves to "add" within the 120-char head
    assert variant_mod.ADD_PROMPT.index("**新增**") < 120
    variant_mod.ADD_PROMPT.format(
        n=2, kp_name="kp", grade="g", qtype="解答", stem="s", skeleton="k", extra="无"
    )


def test_generate_without_stream_callback_stays_legacy(monkeypatch):
    # fake LLM never calls on_delta -> no eager tasks -> items go out un-judged
    _capture_frames(monkeypatch)

    async def fake(messages, retry=True, **kw):
        return json.dumps([dict(_GEN_ITEM, stem=f"s{i}") for i in range(3)], ensure_ascii=False)

    monkeypatch.setattr(variant_mod, "_ainvoke_text", fake)
    out = asyncio.run(generate(_gen_state(), {}))
    assert len(out["items"]) == 3
    assert all("gene" not in it and "check" not in it for it in out["items"])


def test_assemble_final_frame_has_no_partial_key(monkeypatch):
    captured = _capture_frames(monkeypatch)
    items = [{"stem": "a", "answer": "1", "check": {"badge": "ok"}, "gene": {"gate": "pass"}}]
    asyncio.run(variant_mod.assemble(dict(_FACTS_STATE, items=items), {}))
    art = _artifact_frames(captured)[0]
    assert "partial" not in art and "expected_total" not in art  # FE 向后兼容定稿帧


# ---------------------------------------------------------------------------
# prompt re-layout (PRD-C-012 任务3): shared contract + fixed-prefix ordering
# ---------------------------------------------------------------------------


def test_payload_contract_shared_across_generate_regen_extract():
    for p in (GENERATE_PROMPT, REGEN_PROMPT, EXTRACT_PROMPT):
        assert variant_mod._PAYLOAD_CONTRACT in p  # single source, no drift
        assert "舍根" in p and "增根" in p  # root-rejection guidance pinned everywhere
        assert "equation_solve" in p and "禁止任何 Python 语法" in p
    assert '"verify_payload"' in GENERATE_PROMPT
    assert '"verify_payload"' in REGEN_PROMPT
    assert "verify_payload" not in SOLVE_PROMPT  # solve stays payload-free


def test_fixed_contract_segments_precede_variable_segments():
    # cache-friendly: fixed rule/contract blocks first, {placeholder} blocks last
    assert GENERATE_PROMPT.index("expr_equiv") < GENERATE_PROMPT.index("{skeleton}")
    assert GENERATE_PROMPT.index("格式硬规定") < GENERATE_PROMPT.index("母题 DNA：")
    assert EXTRACT_PROMPT.index("expr_equiv") < EXTRACT_PROMPT.index("题干: {stem}")
    assert SOLVE_PROMPT.index("solved_answer") < SOLVE_PROMPT.index("题干：{stem}")
    assert GENE_JUDGE_PROMPT.index("surface_swapped") < GENE_JUDGE_PROMPT.index("{mother_stem}")
    assert PARSE_PROMPT.index("分类标准") < PARSE_PROMPT.index("{utterance}")
    assert PARSE_PROMPT.index("硬约束") < PARSE_PROMPT.index("主考点: {kp_name}")


def test_reordered_prompts_format_cleanly():
    # brace-escape smoke: every reworked template must .format() without KeyError/IndexError
    GENERATE_PROMPT.format(
        n=3, n_normal=2, n_hard=1, kp_name="kp", grade="g", qtype="解答", stem="s", skeleton="k"
    )
    REGEN_PROMPT.format(
        kp_name="kp", grade="g", stem="s", level="normal", qtype="解答",
        difficulty=3, injected_kp="null",
    )
    EXTRACT_PROMPT.format(qtype="解答", stem="s", answer="1", solved_answer="1")
    SOLVE_PROMPT.format(stem="s")
    GENE_JUDGE_PROMPT.format(
        kp_name="kp", grade="g", qtype="解答", difficulty=3, skeleton="k",
        mother_stem="m", level="normal", v_qtype="解答", v_difficulty=3, variant_stem="v",
    )
    PARSE_PROMPT.format(n=3, kp_name="kp", grade="g", utterance="u")


def test_trace_markers_still_resolve_after_reorder():
    cases = [
        (SOLVE_PROMPT.format(stem="s"), "solve"),
        (EXTRACT_PROMPT.format(qtype="q", stem="s", answer="a", solved_answer="x"), "extract"),
        (PARSE_PROMPT.format(n=1, kp_name="k", grade="g", utterance="u"), "parse"),
        (
            GENERATE_PROMPT.format(
                n=3, n_normal=2, n_hard=1, kp_name="k", grade="g",
                qtype="q", stem="s", skeleton="k",
            ),
            "generate",
        ),
        (
            GENE_JUDGE_PROMPT.format(
                kp_name="k", grade="g", qtype="q", difficulty=3, skeleton="k",
                mother_stem="m", level="normal", v_qtype="q", v_difficulty=3, variant_stem="v",
            ),
            "gene_judge",
        ),
        (
            REGEN_PROMPT.format(
                kp_name="k", grade="g", stem="s", level="normal", qtype="q",
                difficulty=3, injected_kp="null",
            ),
            "regen",
        ),
    ]
    for prompt, label in cases:
        assert variant_mod._trace_label([HumanMessage(content=prompt)]) == label
