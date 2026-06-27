"""variant 引擎 · stage1_anchor 解题机器（PRD-C-104 B4 抽出，纯搬零改）。

stage1 解题/验算共用件：
- _solve_one：真解一道题（LLM 阅卷解析 + 净化出口）。
- _check_one_item：闸B per-item 协程（题型分流 + sympy 验算 + 回炉/自愈降级）。
- _gene_one_item：闸A per-item 协程（纯代码三检 + 越界候选落池）。

🔴 行为零改：判决只读 math_verify verdict / sympy 返回值（铁律），LLM 自评永不采信。
🔴 跨阶段共享：本仨被 stage2_variant/gates.py 经 facade `from agents.variant import ...`
   于 __init__ 末尾 re-export 后取用（gene_gate/solve_explain 节点 + generate eager 链共用）。
   故本模块 re-export 必须**先于** gates.py re-export（gates 顶部 import 这三件）→ 本模块
   只在顶部 import staying 符号（__init__ 早已定义）；对 gates.py 内符号（_regen_once /
   _anti_degen_gate / _conservation_ok / gene_gate_check）改为**函数体内延迟 import**（调用期
   解析，那时 gates 已 re-export 进 facade）→ 破除 stage1↔stage2 装载期循环、逻辑零改。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from agents import math_verify, model_anchor  # noqa: E402

from agents.variant import (  # noqa: E402  运行期解析（本模块在 __init__ 末尾、gates 之前导入）
    GENE_GATE_SKIPPED,
    MAX_HEAL,
    REVIEW_PROOF,
    SOLVE_PROMPT,
    VERIFY_FAIL_AFTER_REGEN,
    VERIFY_SYMPY_PASS,
    VERIFY_UNVERIFIED,
    _ainvoke_text,
    _apply_visibility,
    _budget_exhausted,
    _emit_stage,
    _is_proof_like,
    _machine_verify,
    _norm,
    _parse_json,
    _proof_struct_ok,
    _sanitize_rich_text,
    settings,
    structure_lint,
)


async def _solve_one(stem: str) -> dict:
    """真解一道题。🔴 LLM 调用异常吞掉返 {}（与 _extract_payload/_gene_judge_one 契约对齐）：
    瞬时网关抖动绝不外抛炸掉 solve_explain/gene_gate 节点（G5），调用方按"没解出来"降级。"""
    try:
        text = await _ainvoke_text(
            [HumanMessage(content=SOLVE_PROMPT.format(stem=stem or ""))],
            model=settings.variant_model("solve"),
        )
    except Exception:  # noqa: BLE001
        return {}
    solved = _parse_json(text) or {}
    # 🔴 阅卷解析会回写 item["solution"]（solve_explain 三处）—— 出口统一净化，
    # 否则 \( \) / 字面 \n 绕过 _parse_generated_items 的净化直达卡片/入库
    if isinstance(solved, dict) and solved.get("solution"):
        solved["solution"] = _sanitize_rich_text(solved["solution"])
    return solved


async def _check_one_item(
    item: dict[str, Any], facts: dict, idx: int, total: int
) -> tuple[dict[str, Any] | None, str | None]:
    """🔴 闸B per-item 协程（P2 单一事实源：solve_explain 节点与 generate 流内 eager 共用）。

    入参 item 视为本协程私有（调用方传副本）；idx/total 只用于思路条叙事编号。
    返回 (item, None)=保留（已带 check + 外显 tier）；(None, dropped 叙事)=4d 方案A 剔除。
    已带 check 的题原样通过（不重判、不发 stage）。判决/降级语义与重排前逐字一致：
    - 证明/开放/作图类 → 不进 sympy，软校验 + 人审标记；
    - sympy pass → sympy_pass；fail → REGEN 回炉 1 次（须 pass+守恒）→ 仍不过剔除；
    - degrade → LLM 独立解自检 fallback（match/mismatch + 自愈 1 次）。
    🔴 判决只读 verify() 的 verdict，永不采信 LLM 自评；任何环节失败降级继续，绝不抛（G5）。
    """
    # 🔴 PRD-C-104 B4：_regen_once / _anti_degen_gate / _conservation_ok 在 stage2_variant/gates.py，
    #   于 __init__ 末尾 re-export 进 facade（晚于本模块装载）→ 改函数体内延迟取（调用期已就绪）。
    from agents.variant import (  # noqa: E402
        _anti_degen_gate,
        _conservation_ok,
        _regen_once,
    )

    if item.get("check"):  # 已定状态（如自愈过的补题再次流经）→ 不重复
        return item, None

    # ── 题型结构 lint（P11.3）：先于数学验算，按 qtype 校验形态（选择题别长成多小问
    # 嵌合体/选项不足/标答非字母；填空缺空位）。命中缺陷 → 走既有题级 REGEN 回炉 1 次
    # （与下游 sympy 回炉同一通道，不是整组重试）；回炉清掉缺陷 → 顶位继续往下验算。
    # 🔴 降级（铁律④）：lint 返 []（解析不了/合规）不拦；回炉仍不过 → 标 ⚠ 注记并继续
    # 进 sympy（绝不卡死、不剔除，结构存疑交给老师人审）。
    struct_defects = structure_lint(item)
    if struct_defects:
        # 🔴 P13 预算闸：结构回炉是增强类调用，预算耗尽 → 跳过回炉，直接标 ⚠ 注记继续走
        #   sympy（既有降级路径，绝不卡死）。
        # 🔴 RC2「老师意志优先」补齐结构闸（题组编辑器 reverify 修复）：from_edit 题（老师手动
        #   编辑/点名改造）即便结构 lint 不过也**绝不回炉换题**——回炉会用模型重出覆盖老师改的
        #   内容（reverify 真机实锤：手改题干→结构不匹配旧 qtype→被静默换成另一道题）。只标 ⚠
        #   注记、保留老师原题继续验算，交人审。与下游 FAIL/degrade 的 from_edit 短路语义一致。
        if item.get("from_edit") or _budget_exhausted():
            item["structure_lint"] = {"badge": "warn", "defects": struct_defects}
        else:
            _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道结构不合题型，回炉重生中")
            feedback = (
                "本题结构不符合其题型契约：" + "；".join(struct_defects) + "。"
                "请严格按题型结构契约重新出一道同考点同年级的变式（选择题=单一设问+恰 4 选项+"
                "answer 为选项字母，禁止多小问；填空题题干须含空位 ____）。"
            )
            draft = await _regen_once(item, facts, feedback=feedback)
            if draft and not structure_lint(draft):
                # 回炉稿结构合规 → 顶位（保留闸A gene 印记，下游 sympy 照常验答案）
                draft["gene"] = item.get("gene")
                item = draft
            else:
                # 回炉失败/仍不合规 → 降级：item 保持原样继续往下走 sympy，记结构存疑注记。
                item["structure_lint"] = {"badge": "warn", "defects": struct_defects}
        # 断言：到此处 item 要么结构合规、要么已记 warn 注记（绝不卡死）。
    else:
        # 结构合规留痕（审计：本题进过结构闸且通过）
        item["structure_lint"] = {"badge": "ok", "defects": []}

    # 🔴 P14 叙事修正（RC3·PRD-C-013）：per-item 闸并发跑，旧「第 N/total 道」会让老师误读成
    #   顺序进度（实际三题同时验）。改为「第 N 题验算中」——只点本题号，不带误导性的「/总数」。
    #   编辑轮（from_edit 单题重验）明示「只重验第 N 题」——别让老师把题号读成「全部重验」。
    _emit_stage(
        "verify", "程序验算", "running",
        (f"只重验第 {idx + 1} 题" if item.get("from_edit") else f"第 {idx + 1} 题验算中"),
    )

    # ── 闸B·题型分流：证明/开放/作图 → 不进 sympy，软校验 + 人审标记 ──
    if _is_proof_like(item.get("qtype") or facts["qtype"], item.get("stem")):
        struct_ok = _proof_struct_ok(item.get("stem"))
        item["check"] = {
            "badge": "ok" if struct_ok else "warn",
            "solved_answer": None,
            "review": REVIEW_PROOF,
        }
        # 🔴 PRD-A-017 R2·真值细颗粒播报（并修编排 BUG「难题/证明类程序验算静默无提示」）：
        #   证明/作图/开放类不进 sympy，让老师可见「转人工复核」而非空白沉默。复用 verify key。
        _emit_stage(
            "verify", "程序验算", "running",
            f"第 {idx + 1} 题为证明/作图类，转人工复核",
        )
        _apply_visibility(item)
        return item, None

    solved = await _solve_one(item.get("stem", ""))
    solved_answer = solved.get("solved_answer")
    # solve 产出解析（给老师当判题依据；优先用阅卷解析）
    if solved.get("solution"):
        item["solution"] = solved.get("solution")

    # ── 闸B·程序验算：判决只读 verdict（G1/G2），不再用字符串比对自判 ──
    res = await _machine_verify(item, solved_answer)
    verdict = res.get("verdict")

    if verdict == math_verify.PASS:
        item["check"] = {
            "badge": "ok",
            "solved_answer": solved_answer,
            "verify": VERIFY_SYMPY_PASS,
            "verify_detail": res.get("detail"),
            "computed": res.get("computed"),
        }
        # 🔴 批3·⑦ 反退化闸：答案 sympy PASS ≠ 构型不退化（最优点落区间端点 = 答案碰巧对但机制失效）。
        #   退化 → REGEN（≤MAX_DEGEN_REGEN，超限弃）；非退化/不可判 → 原样放行。判决只读代数零 LLM。
        item, degen_dropped = await _anti_degen_gate(item, facts, idx, total)
        if degen_dropped:
            return None, str(item.get("_dropped") or "退化构型已剔除")
        # 🔴 PRD-A-017 R2·真值正向播报：本题 sympy 验算通过（带真算证据 computed）→ 让老师
        #   看见「第 N 题验算通过」而非只在收尾看到一个总「done」。复用 verify key、单题号不带/总数。
        _emit_stage(
            "verify", "程序验算", "running",
            f"第 {idx + 1} 题程序验算通过"
            + (f"（算得 {res.get('computed')}）" if res.get("computed") else ""),
        )
        _apply_visibility(item)
        return item, None

    if verdict == math_verify.FAIL:
        # 🔴 RC2「老师意志优先」补齐闸B（对抗审②修复）：编辑轮老师点名改造的题（item.from_edit，
        #   带 edit_note）即便 sympy 判 FAIL 也**不回炉/不换题/不剔除**——保留老师编辑的原题、
        #   标 ⚠ 注记（verify=fail_after_regen，走 4d both_low/silent 外显）交老师人审。否则
        #   回炉换题会用模型重出覆盖老师意志、剔除路径连 edit_note 一起丢 = 隐性数据丢失，与
        #   闸A from_edit 短路语义矛盾。降级不抛（G5）。
        if item.get("from_edit"):
            item["check"] = {
                "badge": "warn",
                "solved_answer": solved_answer,
                "verify": VERIFY_FAIL_AFTER_REGEN,
                "verify_detail": res.get("detail"),
                "computed": res.get("computed"),
            }
            _apply_visibility(item)
            return item, None
        # sympy 判定标答真错 → 既有回炉机制重生 1 次，computed/detail 注回 prompt
        # 🔴 P13 预算闸：heal/replenish 都是增强类调用，预算耗尽 → 跳过，直接走「剔除不外发」
        #   降级（4d 方案A，本组少一道 dropped 叙事）。绝不卡死、绝不抛。
        healed = None
        if MAX_HEAL >= 1 and not _budget_exhausted():
            _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道回炉重生中")
            feedback = (
                f"程序(sympy)验算判定该题题面标答错误：程序算得 computed={res.get('computed')}；"
                f"详情：{res.get('detail')}。请重新出一道题面与标答自洽、经得起程序验算的等价变式。"
            )
            draft = await _regen_once(item, facts, feedback=feedback)
            if draft:
                resolved = await _solve_one(draft.get("stem", ""))
                if resolved.get("solution"):
                    draft["solution"] = resolved.get("solution")
                # 🔴 重生版仍须过守恒校验 + 程序验算双闸
                cons = _conservation_ok(
                    resolved.get("kp_name", ""), resolved.get("grade", ""), facts
                )
                r_res = await _machine_verify(draft, resolved.get("solved_answer"))
                if cons and r_res.get("verdict") == math_verify.PASS:
                    draft["check"] = {
                        "badge": "ok",
                        "solved_answer": resolved.get("solved_answer"),
                        "verify": VERIFY_SYMPY_PASS,
                        "verify_detail": r_res.get("detail"),
                        "computed": r_res.get("computed"),
                    }
                    # 🔴 闸A 标记随愈合保留（healed 整体替换不丢 gene → 入库 auxTags.gene_gate
                    # 不断档 + 后续编辑轮 gene_gate 不重判/不静默换题）；原版无 gene（如持久化
                    # 旧线程存量题）→ 按既有 skipped 语义留痕（闸A 没判过，REGEN 锁同骨架）。
                    draft["gene"] = item.get("gene") or {
                        "gate": GENE_GATE_SKIPPED,
                        "reason": "healed-in-solve",
                    }
                    healed = draft
        if healed:
            _apply_visibility(healed)
            return healed, None
        # 🔴 整改4（2026-06-12·闸B 回炉松绑·维护者拍板「不能写得这么死」）：
        #   回炉 1 次仍 FAIL → **不再二次回炉、不再补题、不再剔除**——直接标 ⚠（verify=
        #   fail_after_regen，走 4d both_low/silent 外显矩阵）放行，交老师人审。
        #   （铁律「闸门必有降级路径、绝不卡死」本就在；旧 4d 剔除+补题路径单次回炉 95-134s
        #   且输出失控，松绑为「标 ⚠ 放行」减少一整条回炉链。判决仍只读 sympy，不采信 LLM 自评。）
        _emit_stage(
            "verify", "程序验算", "warn",
            f"第 {idx + 1} 道程序验算未过（回炉一次仍未过），已标注存疑交人审",
        )
        item["check"] = {
            "badge": "warn",
            "solved_answer": solved_answer,
            "verify": VERIFY_FAIL_AFTER_REGEN,
            "verify_detail": res.get("detail"),
            "computed": res.get("computed"),
        }
        _apply_visibility(item)
        return item, None

    # ── degrade：sympy 吃不下（载荷抽不成/超范围）→ 保留既有 LLM 自检 fallback ──
    match = _norm(solved_answer) == _norm(item.get("answer"))
    if match:
        item["check"] = {
            "badge": "ok",
            "solved_answer": solved_answer,
            "verify": VERIFY_UNVERIFIED,
            "verify_detail": res.get("detail"),
            "self_check": "match",  # 4d：独立复算一致 → 轻正面（不再打 ⚠ 未经程序验算）
        }
        # 🔴 PRD-A-017 R2·真值播报：sympy 吃不下载荷 → 独立复算与标答一致（轻正面）。
        _emit_stage(
            "verify", "程序验算", "running",
            f"第 {idx + 1} 题独立复算与标答一致",
        )
        _apply_visibility(item)
        return item, None

    # 独立解 ≠ 标答（LLM 自检）→ 既有自愈：重生 1 次 → 重解 + 守恒
    # 🔴 P13：degrade 自愈也是增强类调用，预算耗尽 → 跳过自愈，落下方 warn 保留（不抛）。
    # 🔴 RC2「老师意志优先」补齐 degrade 支（题组编辑器 reverify 修复）：from_edit 题不自愈换题，
    #   直接落下方 warn 保留老师原题（与结构闸/FAIL 支一致——reverify 只验不换）。
    healed = None
    if MAX_HEAL >= 1 and not _budget_exhausted() and not item.get("from_edit"):
        _emit_stage("verify", "程序验算", "warn", f"第 {idx + 1} 道回炉重生中")
        draft = await _regen_once(item, facts)
        if draft:
            resolved = await _solve_one(draft.get("stem", ""))
            r_answer = resolved.get("solved_answer")
            r_match = _norm(r_answer) == _norm(draft.get("answer"))
            # 🔴 重生版仍须过守恒校验
            cons = _conservation_ok(
                resolved.get("kp_name", ""), resolved.get("grade", ""), facts
            )
            if r_match and cons:
                draft["solution"] = resolved.get("solution") or draft.get("solution")
                draft["check"] = {
                    "badge": "ok",
                    "solved_answer": r_answer,
                    "verify": VERIFY_UNVERIFIED,
                    "self_check": "match",
                }
                # 🔴 闸A 标记随愈合保留（同 FAIL 自愈路径：不丢 gene、不被编辑轮重判）
                draft["gene"] = item.get("gene") or {
                    "gate": GENE_GATE_SKIPPED,
                    "reason": "healed-in-solve",
                }
                _apply_visibility(draft)
                healed = draft

    if healed:
        return healed, None
    # 守恒破 或 重生仍不过 → 保留（程序没证明它错，只是没把握）。
    # 4d：verify 侧低 → 单闸沉默 / gene 也低 → ⚠（_apply_visibility 矩阵裁决）
    item["check"] = {
        "badge": "warn",
        "solved_answer": solved_answer,
        "verify": VERIFY_UNVERIFIED,
        "verify_detail": res.get("detail"),
        "self_check": "mismatch",
    }
    _apply_visibility(item)
    return item, None


async def _gene_one_item(item: dict, facts_i: dict, idx: int, total: int) -> dict:
    """🔴 闸A per-item 协程（P2 单一事实源：gene_gate 节点与 generate 流内 eager 共用）。

    B2·T2 起内涵 = 纯代码三检（gene_gate_check），LLM judge 全链已删：
    - 三检全过 → item.gene={gate:"pass", flags:[]}；
    - 任一检命中 → item.gene={gate:"warn", flags, reason}（**只警示不硬拦、不回炉、不剔题**，
      闸门降级路径；真值随 item.gene 透传进 FE 4d 展示，铁律④）。
    已带 gene 的题原样通过（不重判、不发 stage）。
    🔴 保持 async 签名（节点 gather 并发 + generate eager 链 await 调用契约不变）。
    入参 item 视为本协程私有（调用方传副本）；idx/total 只用于思路条叙事编号。
    """
    # 🔴 PRD-C-104 B4：gene_gate_check 在 stage2_variant/gates.py（晚于本模块 re-export）→
    #   函数体内延迟取（调用期已就绪），破除 stage1↔stage2 装载期循环、逻辑零改。
    from agents.variant import gene_gate_check  # noqa: E402

    if item.get("gene"):  # 已判过 → 不重判（旧题预算保护）
        return item

    # 🔴 P14 叙事修正（RC3·PRD-C-013）：并发闸去掉误导性「/总数」；编辑轮单题明示「只重比第 N 题」。
    _emit_stage(
        "gene_gate", "平行度比对", "running",
        (f"只重比第 {idx + 1} 题" if item.get("from_edit") else f"第 {idx + 1} 题比对中"),
    )
    item["gene"] = gene_gate_check(item, facts_i)
    # 🔴 批3·W3' 越界落待命名池（含题目指针；软警不打回，仅记录可审，G4：写失败不静默）。
    oos = (item.get("gene") or {}).get("model_out_of_set")
    if oos:
        dna_i = facts_i.get("dna") or {}
        mother_ids = [
            str(m.get("id") or "").strip()
            for m in (dna_i.get("models") or [])
            if isinstance(m, dict) and str(m.get("id") or "").strip()
        ]
        ref = str(item.get("stem") or "")[:60] or str(facts_i.get("mother_question_id") or "")
        for name in oos:
            try:
                model_anchor.record_overflow_candidate(name, mother_ids, question_ref=ref)
            except Exception:  # noqa: BLE001 — 待命名池落盘失败不拖垮出题（软警是增强不是关卡）
                pass
    return item
