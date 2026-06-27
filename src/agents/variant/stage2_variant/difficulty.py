"""variant 引擎 · stage2_variant 难度判档层（PRD-C-104 B3a 抽出，纯搬零改）。

从 `variant/__init__.py` 原样剪出（桥到 core/difficulty，core 不动）：
  - _dna_factors_for_grade：母题 DNA → grade_observed 确定性因子 K/R/D/G + model_hits。
  - grade_variant_item：单道变式确定性判档（grade_observed），替 LLM 自评（WS1·AC1）。
  - _grade_difficulty：LLM 难度复核节点。

🔴 行为零改：函数体逐字搬。跨阶段/跨模块依赖（core.difficulty、_solution_steps、
   settings、_ainvoke_text、_parse_json、_to_int、_grade_difficulty_payload、
   _GRADE_DIFFICULTY_PROMPT、DIFFICULTY_CAP、HumanMessage）全部仍留在 __init__ /
   shared / core，本模块从 facade `agents.variant` 取（这些符号在 __init__
   末尾 re-import 本模块前已全部定义，无循环）。__init__.py 末尾 re-export 三函数 → 调用方零感。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from core import difficulty  # 桥到 core/difficulty（core 不动）

# 🔴 留在 __init__ 的依赖（运行期解析；本模块仅在 __init__ 末尾被导入，届时下列符号已定义）。
from agents.variant import (
    DIFFICULTY_CAP,
    _GRADE_DIFFICULTY_PROMPT,
    _ainvoke_text,
    _grade_difficulty_payload,
    _parse_json,
    _solution_steps,
    _to_int,
    settings,
)


def _dna_factors_for_grade(dna: dict | None, stem: str = "", analysis_text: str = "") -> dict:
    """从母题 DNA 抽 grade_observed 需要的确定性因子 K/R/D/G + model_hits。

    - model_hits = dna.models（锚定模型，已带表真值 tier_int/freq_int，见 model_anchor）。
    - R = 解法骨架步骤条数（dna.skeleton list 长度）。
    - K = KG 锚定知识点数（主+副 kp）；不足则由解析独立依据数兜（复用 difficulty.extract_K）。
    - D = 递进小问深度（题面小问 + 解析引用前问）。
    - G = 数形结合（题面/几何标识）。
    纯函数、零 LLM。缺则各因子 0（grade_observed 自带降级哨兵）。
    """
    dna = dna or {}
    skeleton = dna.get("skeleton") or []
    R = len(skeleton) if isinstance(skeleton, (list, tuple)) else 0
    # K：主 kp + 副 kp 锚定数（KG 计数）；difficulty.extract_K 内部会与解析依据数取较大值
    kp_count = (1 if (dna.get("main_kp") or {}).get("id") else 0) + len(
        [s for s in (dna.get("secondary_kps") or []) if isinstance(s, dict) and s.get("id")]
    )
    analysis_blob = analysis_text or (
        "\n".join(str(s) for s in skeleton) if isinstance(skeleton, (list, tuple)) else ""
    )
    K = difficulty.extract_K(analysis_blob, kp_count or None)
    D = difficulty.extract_D(stem or "", analysis_blob)
    G = difficulty.extract_G(stem or "", dna.get("verify_kind"), dna.get("dna_type"))
    return {"model_hits": dna.get("models") or [], "K": K, "R": R, "D": D, "G": G,
            "high_strategies": []}


def grade_variant_item(item: dict, mother_dna: dict | None) -> dict:
    """🔴 WS1·AC1：单道变式确定性判档（grade_observed），替代 generate 内嵌 rubric 的 LLM 自评。

    - model_hits = 继承母题锚定模型（mother_dna.dna.models，带表真值 tier_int/freq_int）；
      变式自带 models（edit-dna 改过）时优先用变式自己的。
    - K/R/D/G = 从该变式自己的 stem/solution 抽（R 走解析分步、D 走小问、K 走依据数、G 走几何）。
    - 难度永不取 LLM 自评：item 原 difficulty 不参与判档（被本函数覆盖）。
    返回 grade_observed 账单（含 level/modelHits/K/R/D/rule/...）。纯函数、零 LLM。
    """
    mother_dna = mother_dna or {}
    mdna_dna = mother_dna.get("dna") or {}
    # 变式 model_hits：变式自带优先，否则继承母题（变式当前主路径继承母题 models）
    item_models = item.get("models")
    model_hits = item_models if item_models else (mdna_dna.get("models") or [])
    stem = str(item.get("stem") or "")
    solution = str(item.get("solution") or "")
    steps = _solution_steps(solution)
    R = len(steps)
    # K：母题 KG 锚定数（变式守恒同知识点）+ 解析依据数兜底（取较大，见 extract_K）
    kp_count = (1 if (mdna_dna.get("main_kp") or {}).get("id") else 0) + len(
        [s for s in (mdna_dna.get("secondary_kps") or []) if isinstance(s, dict) and s.get("id")]
    )
    K = difficulty.extract_K(solution, kp_count or None)
    D = difficulty.extract_D(stem, solution)
    G = difficulty.extract_G(stem, mdna_dna.get("verify_kind"), mdna_dna.get("dna_type"))
    return difficulty.grade_observed(
        model_hits=model_hits, K=K, R=R, D=D, G=G, high_strategies=[],
    )


async def _grade_difficulty(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """P8 难度总评（S1.2）：一次 nano call 按绝对 rubric 复评全组，覆盖 item['difficulty']。

    ⚠️ 整改2（2026-06-12）已**退出主流程**：难度判定并入生题（GENERATE/REGEN/ADD 出题调用
       同步按嵌入的四档 rubric 产出 difficulty），assemble/revise whole 不再独立调用本函数。
       函数本体保留供单测 + 潜在按需复评用，rubric 与 _DIFFICULTY_RUBRIC 同口径（22-SSOT §2）。

    🔴 G5 降级：items 空 / LLM 异常 / 解析失败 / 个数对不上 → 保留各 item 原 difficulty 值，
    绝不抛、绝不卡死。逐项越界钳到 1-4（与 _clamp_difficult 入库口径一致）。
    返回新 list（不原地改入参）；用 LLM_MODEL_LIGHT 经 _ainvoke_text 的 per-call model 覆盖。
    """
    out = [dict(it) for it in items]
    if not out:
        return out
    try:
        prompt = _GRADE_DIFFICULTY_PROMPT.format(items=_grade_difficulty_payload(out))
        text = await _ainvoke_text(
            [HumanMessage(content=prompt)], model=settings.LLM_MODEL_LIGHT
        )
        parsed = _parse_json(text)
    except Exception:  # noqa: BLE001 — 难度总评是增强不是关卡，失败保留原值
        return out
    if not isinstance(parsed, list) or len(parsed) != len(out):
        return out  # 个数对不上 → 整体降级保留原值（不冒险错位覆盖）
    for it, raw in zip(out, parsed):
        d = _to_int(raw)
        if d is not None:
            it["difficulty"] = max(1, min(DIFFICULTY_CAP, d))  # 越界钳 1-4
        # d 解析不出 → 该题保留原 difficulty（不动）
    return out
