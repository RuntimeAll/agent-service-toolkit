"""PRD-C-109 B1 · 母题编辑工具注册表 + effect 路由（re-wire 不 rebuild）。

> 北极星（PRD-C-109 §10 / §11 铁律）：把「母题编辑」从代码死分支升级成
>   **LLM 编排 + 确定性工具**。LLM 只产 `{tool, 值}`；effect 由代码查
>   `TOOL_REGISTRY[tool].effect` 得出（LLM 不产 effect = 防漂移）；加工具 =
>   注册表加一行 + 薄 mutator，路由代码不变（守 langgraph-agent 铁律③）。

🔴 本模块**零重写重生逻辑**——所有 mutator 都是薄包一层现成 `edit_dna_state`
   （persist.py），后者据 `regen_class_of(field)`（→ `REGEN_CLASS`，__init__.py:372）
   做四分流（hard_anchor 解冻重锚 / soft_regen·rewrite_solve 标脏待重生 /
   meta 即时生效）。本卡只建「工具名 → field → edit_dna_state」的薄映射 +
   「UI effect 三类 → REGEN_CLASS 四分流」对齐表，**绝不改 REGEN_CLASS 本身**。

工具效果类（UI 徽章 = §10 三类 + 执行/旋钮特例）：
  - 即时生效（meta）        ：选副考点 / set_难点 / 加·删标签
  - 重写解析（rewrite_solve）：改解法骨架 / 加·删·换模型
  - 重出本题               ：set_主考点 / set_题型 / set_考察类型 / set_场景 /
                            改题面 / set_年级章
  - 执行类                 ：重新解题 / 开始出变式（不改单维，B2 路由到执行流水线）
  - 旋钮（只读）           ：_难度旋钮（难度表驱动只读，B2 路由到变式难度旋钮，不碰母题）

🔴 UI effect ≠ REGEN_CLASS 一一对应（A2 警告点）：「重出本题」按 field 落到
   soft_regen（主考点/题型/考察类型/场景）或 hard_anchor（年级=解冻重锚）。
   本表存 `dna_field`，`regen_class` **由 `regen_class_of(field)` 在建表时确定性
   派生**（事实源永远是 REGEN_CLASS），不在此处手抄、不在此处改。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents import dna_extract
from agents.variant import edit_dna_state, edit_mother_dna_meta, regen_class_of
from agents.variant.state import VariantState

# 🔴 PRD-C-109 fix·即时生效（meta）维白名单：母题卡就绪态（无题组）可「去 item 化」原地写
#   （edit_mother_dna_meta 只改 mother_dna.dna、不需 item、不重解/重锚/清组）。
#   与 persist._MOTHER_META_FIELDS 同口径——这三维是注册表里仅有的 EFFECT_IMMEDIATE 维的 field。
_NOITEM_META_FIELDS: frozenset[str] = frozenset({"tags", "secondary_kps", "hard_points"})

# ---------------------------------------------------------------------------
# UI effect 三类（+ 执行/旋钮特例）枚举常量。前端徽章 + 出口路由按 effect 走确定性边。
# ---------------------------------------------------------------------------
EFFECT_IMMEDIATE = "即时生效"   # 纯元数据，不重跑（meta）
EFFECT_REWRITE = "重写解析"     # 只重写解法/解析（rewrite_solve）
EFFECT_REGEN = "重出本题"       # 按新维重解 + 重打标该面（soft_regen / hard_anchor）
EFFECT_EXEC = "执行"            # 执行类（重新解题 / 开始出变式）= B2 路由到流水线
EFFECT_KNOB = "旋钮"            # 难度旋钮（只读·表驱动，不进母题工具）

# 🔴 「UI effect 三类 → REGEN_CLASS 四分流」对齐表（A2 注意点：语义一致、命名维度不同）。
#    只读校验用——证明每个有 field 的工具，其 UI effect 与 field 的 REGEN_CLASS 语义自洽；
#    **不改 REGEN_CLASS、不据本表派生 effect**，仅作断言锚（单测 G2 校它）。
EFFECT_TO_REGEN_CLASSES: dict[str, frozenset[str]] = {
    # 即时生效 = 元数据维（标签/副考点/难点）→ meta
    EFFECT_IMMEDIATE: frozenset({"meta"}),
    # 重写解析 = 解法骨架/模型 → rewrite_solve
    EFFECT_REWRITE: frozenset({"rewrite_solve"}),
    # 重出本题 = 主考点/题型/考察类型/场景（soft_regen） + 年级章（hard_anchor）
    EFFECT_REGEN: frozenset({"soft_regen", "hard_anchor"}),
}

# list-op 策略（加/删/换一项）。None = 标量维（直接整值写）。
LIST_ADD = "add"
LIST_DEL = "del"
LIST_REPLACE = "replace"


# ===========================================================================
# mutator 工厂 —— 薄包一层现成 edit_dna_state（零重写）。
#   标量维：(state, index, value) → edit_dna_state(state, index, field, value)
#   list 维：先读 state 现值 → 加/删/换一项 → 用**整列表**调 edit_dna_state（后者
#            对 tags/secondary_kps/models 收的就是整列表，list-op 只是组装入参）。
# 所有 mutator 签名统一：(state, index, value) -> (update, edited_item, error)
#   = edit_dna_state 的返回三元组（透传，端点/路由直接用）。
# ===========================================================================

Mutator = Callable[[VariantState, int, Any], "tuple[VariantState, dict[str, Any] | None, str | None]"]


def _scalar_mutator(field: str) -> Mutator:
    """标量维 mutator：直接把值写进 edit_dna_state（field 一一对应契约 key）。
    🔴 PRD-C-109 fix·index=None（母题卡就绪、无题组）+ meta 维 → 去 item 化原地写 mother_dna.dna
       （edit_mother_dna_meta），绝不重解/重锚。非 meta 维无题组无意义（重生须有题组），照旧报错。"""
    def _fn(state: VariantState, index: Any, value: Any):
        if index is None and field in _NOITEM_META_FIELDS:
            return edit_mother_dna_meta(state, field, value)
        return edit_dna_state(state, index, field, value)
    _fn.__name__ = f"mutate_{field}"
    return _fn


def _read_mother_list(state: VariantState, field: str) -> list:
    """读母题级守恒维当前列表值（tags/secondary_kps 住 mother_dna.dna；models 住
    item 级覆盖但 edit_dna_state 收整列表回写题级，读时优先 item.models 兜底母题）。"""
    if field == "models":
        # models 是题级覆盖维（edit_dna_state 写 item.models）；读时先看本道 item，
        # 缺了再退母题 dna.models（首改时题级未覆盖）。index 在 mutator 内传入校验。
        return []  # 由 _models_mutator 自己按 index 读 item，避免此处不知 index
    dna = (state.get("mother_dna") or {}).get("dna") or {}
    return list(dna.get(field) or [])


def _list_mutator(field: str, op: str) -> Mutator:
    """list 维 mutator：读现值 → 加/删/换一项 → 整列表交回 edit_dna_state。

    value 语义：
      - add     ：value = 要加的一项（标签字符串 / kp dict / model dict-or-id）
      - del     ：value = 要删的一项（按标签字面 / kp.id / model.id|name 匹配删）
      - replace ：value = {"old":旧项, "new":新项} 或直接 value=新项（整组换成单项）
    """
    def _fn(state: VariantState, index: Any, value: Any):
        # models 读题级 item（edit_dna_state 把 models 落 item.models），其余读母题 dna
        if field == "models":
            items = list(state.get("items") or [])
            if not isinstance(index, int) or index < 1 or index > len(items):
                return {}, None, f"index 越界（须 1..{len(items)}），收到 {index}"
            cur = list((items[index - 1] or {}).get("models") or [])
            if not cur:
                # 题级未覆盖 → 退母题 dna.models 作基线（首改时）
                cur = list(((state.get("mother_dna") or {}).get("dna") or {}).get("models") or [])
        else:
            cur = _read_mother_list(state, field)

        new_list, err = _apply_list_op(cur, op, value, field)
        if err:
            return {}, None, err
        # 🔴 PRD-C-109 fix·母题卡就绪态（无题组）+ meta list 维（tags/secondary_kps）→ 去 item 化
        #    原地写 mother_dna.dna（list 现值已在上面从母题 dna 读出、加/删/换好整列表）。
        if index is None and field in _NOITEM_META_FIELDS:
            return edit_mother_dna_meta(state, field, new_list)
        return edit_dna_state(state, index, field, new_list)

    _fn.__name__ = f"mutate_{field}_{op}"
    return _fn


def _item_key(field: str, item: Any) -> str:
    """list 项的去重/删除匹配键。tags=字面；kp/model=id（无 id 退 name）。"""
    if field == "tags":
        return str(item).strip()
    if isinstance(item, dict):
        return str(item.get("id") or item.get("code") or item.get("name") or "").strip()
    return str(item).strip()


def _apply_list_op(
    cur: list, op: str, value: Any, field: str
) -> "tuple[list, str | None]":
    """纯函数：在 cur 上加/删/换一项，返回新列表（不改原 list）。"""
    cur = list(cur)
    if op == LIST_ADD:
        if value is None or (isinstance(value, str) and not value.strip()):
            return cur, f"加{field}：值为空"
        key = _item_key(field, value)
        if any(_item_key(field, x) == key for x in cur):
            return cur, None  # 幂等：已有则不重复加（不报错）
        return cur + [value], None
    if op == LIST_DEL:
        if value is None:
            return cur, f"删{field}：未指定要删的项"
        key = _item_key(field, value)
        new = [x for x in cur if _item_key(field, x) != key]
        return new, None  # 删不存在的项 = 幂等 no-op
    if op == LIST_REPLACE:
        # value = {"old":..,"new":..}（精确换）或 直接新项（整组换成单项）
        if isinstance(value, dict) and ("new" in value or "old" in value):
            old, new = value.get("old"), value.get("new")
            if new is None:
                return cur, f"换{field}：缺 new 目标项"
            okey = _item_key(field, old) if old is not None else None
            if okey is not None and any(_item_key(field, x) == okey for x in cur):
                return [new if _item_key(field, x) == okey else x for x in cur], None
            # 找不到旧项 → 整组换成 [new]（"这个模型换掉"无明确旧项时的语义）
            return [new], None
        if value is None:
            return cur, f"换{field}：值为空"
        return [value], None  # 整组换成单项
    return cur, f"未知 list-op「{op}」"


# ===========================================================================
# 🔴 TOOL_REGISTRY —— 15 母题编辑工具 + 难度旋钮特例（共 16 条，与 A1 spike 同集）。
#   每条：{fn, effect, dna_field, list_op, exec, desc}
#     fn        = mutator（薄包 edit_dna_state）；执行/旋钮类 fn=None（B2 路由别处）
#     effect    = UI 徽章三类 + 执行/旋钮（前端徽章 + 出口确定性边）
#     dna_field = edit_dna_state 契约 field（无单维改的执行/旋钮类 = None）
#     list_op   = list 维加/删/换策略（标量维 = None）
#     exec      = 执行类标记（重新解题/开始出变式 = True；B2 不查 effect 路由、走流水线）
#     regen_class = 建表时由 regen_class_of(dna_field) 确定性派生（事实源=REGEN_CLASS）
# 🔴 难度只读：注册表**不含**「改母题难度」工具——"难度难一点"是变式旋钮（stage-2），
#    经 _难度旋钮 桩条目标记 effect=旋钮，B2 路由到难度旋钮，不进母题工具、不重解。
# ===========================================================================

# (tool_name, dna_field, list_op, effect, exec) —— regen_class 后由 field 派生
_TOOL_SPECS: list[tuple[str, str | None, str | None, str, bool]] = [
    # 即时生效（meta）
    ("选副考点", "secondary_kps", LIST_ADD, EFFECT_IMMEDIATE, False),
    ("set_难点", "hard_points", None, EFFECT_IMMEDIATE, False),
    ("加标签", "tags", LIST_ADD, EFFECT_IMMEDIATE, False),
    ("删标签", "tags", LIST_DEL, EFFECT_IMMEDIATE, False),
    # 重写解析（rewrite_solve）
    ("改解法骨架", "skeleton", None, EFFECT_REWRITE, False),
    ("加模型", "models", LIST_ADD, EFFECT_REWRITE, False),
    ("删模型", "models", LIST_DEL, EFFECT_REWRITE, False),
    ("换模型", "models", LIST_REPLACE, EFFECT_REWRITE, False),
    # 重出本题（soft_regen / hard_anchor）
    ("set_主考点", "main_kp", None, EFFECT_REGEN, False),
    ("set_题型", "qtype", None, EFFECT_REGEN, False),
    ("set_考察类型", "exam_type", None, EFFECT_REGEN, False),
    ("set_场景", "scene", None, EFFECT_REGEN, False),
    ("set_年级章", "grade", None, EFFECT_REGEN, False),
    # 重出本题·题面（改题面 = 重出/重排版；走执行流水线 revise_item·whole/排版，无单 field）
    ("改题面", None, None, EFFECT_REGEN, True),
    # 执行类（不改单维，B2 路由到流水线）
    ("重新解题", None, None, EFFECT_EXEC, True),
    ("开始出变式", None, None, EFFECT_EXEC, True),
    # 旋钮特例（难度只读·表驱动；B2 路由到变式难度旋钮，不碰母题）
    ("_难度旋钮", None, None, EFFECT_KNOB, False),
]


def _build_registry() -> dict[str, dict[str, Any]]:
    reg: dict[str, dict[str, Any]] = {}
    for name, field, list_op, effect, is_exec in _TOOL_SPECS:
        if field is not None and not is_exec:
            # 🔴 regen_class 事实源 = REGEN_CLASS（绝不手抄）。难度被排除在工具集外，
            #    故 field 一定是可编辑维；regen_class_of 返回 None 视为契约破裂、立即暴露。
            rclass = regen_class_of(field)
            if rclass is None:
                raise ValueError(
                    f"工具「{name}」的 field「{field}」不在 REGEN_CLASS 中——契约破裂"
                )
            # effect↔regen_class 自洽校验（A2 对齐表）：建表期硬断言，防 UI effect 与
            # 底层四分流语义漂移（如把 meta 维误标成「重出本题」）。
            allowed = EFFECT_TO_REGEN_CLASSES.get(effect)
            if allowed is not None and rclass not in allowed:
                raise ValueError(
                    f"工具「{name}」effect={effect} 与 field「{field}」的 "
                    f"regen_class={rclass} 不自洽（允许 {set(allowed)}）"
                )
            fn = _list_mutator(field, list_op) if list_op else _scalar_mutator(field)
        else:
            rclass = None
            fn = None
        reg[name] = {
            "fn": fn,
            "effect": effect,
            "dna_field": field,
            "list_op": list_op,
            "exec": is_exec,
            "regen_class": rclass,
        }
    return reg


TOOL_REGISTRY: dict[str, dict[str, Any]] = _build_registry()

# 🔴 难度只读护栏：注册表里绝不能出现「改母题难度」工具（difficulty 维不可经工具改）。
assert all(
    spec.get("dna_field") != "difficulty" for spec in TOOL_REGISTRY.values()
), "难度只读·表驱动：母题工具集禁含 difficulty 编辑工具（PRD-C-109 §11 铁律）"

# B2 意图层枚举用：可见工具名（不含旋钮特例的 16 工具名，含 _难度旋钮 供 LLM 输出口径）
TOOL_NAMES: tuple[str, ...] = tuple(TOOL_REGISTRY.keys())


def resolve_tool(tool_name: str) -> dict[str, Any] | None:
    """🔴 B2 意图层出口：LLM 产 `{tool, 值}` → 代码查表得 `{fn, effect, regen_class, ...}`。
    出口路由按 `effect` 走确定性三分支（即时生效/重写解析/重出本题）+ 执行/旋钮特例。
    未知工具 → None（B2 回退原 route 分诊，不卡死）。
    """
    return TOOL_REGISTRY.get(tool_name)


def apply_tool(
    tool_name: str, state: VariantState, index: int | None, value: Any
) -> "tuple[VariantState, dict[str, Any] | None, str | None]":
    """便捷执行：查表取 mutator 并执行。

    返回 edit_dna_state 三元组 (update, edited_item, error)。
    - 未知工具 → (空, None, 错误串)
    - 执行类/旋钮类（fn=None）→ (空, None, 提示串)：调用方应据 effect 路由到流水线/旋钮，
      不该调本函数改单维（防误用）。
    🔴 PRD-C-109 fix·index=None = 母题卡就绪态（无题组）：meta 维（tags/副考点/难点）走
       去 item 化原地写（edit_mother_dna_meta），edited_item 恒 None；非 meta 维 index=None
       会被 edit_dna_state 的 index 范围闸拒（重生本就需要题组），符合语义。
    """
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        return {}, None, f"未知工具「{tool_name}」"
    fn = spec.get("fn")
    if fn is None:
        return {}, None, (
            f"工具「{tool_name}」是{spec['effect']}类（fn=None）——"
            "应据 effect 路由到执行流水线/难度旋钮，不经 apply_tool 改单维"
        )
    return fn(state, index, value)
