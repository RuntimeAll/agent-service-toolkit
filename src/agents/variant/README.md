# variant 引擎 · skill 模块边界（PRD-C-106 B4① 轻收口）

> 本文 = 举一反三引擎（工作流 #1）的「能力 = 一个内聚 skill 模块」结构说明，结晶 B1/B2/B3
> 已经成形的模块边界，便于后续接工作流 #2/#3 时复用阶段骨架。**这是文档化现状、不是重构**
> （北极星红线：re-wire 不 rebuild，统一跨工作流接口留到工作流 #2 结晶）。
> 控制流仍是**固定 DAG**（`graph.py`），LLM 不自主路由 —— 这些 skill 模块是未来 agent 的积木，
> 本卡只搭积木、不交方向盘。

---

## 三个能力 = 三个 skill 模块（prompt + 它声明需要的注入上下文）

一个「能力」= 一段 prompt（怎么让 LLM 干这件事）+ 它**声明需要哪些注入上下文**（KG 知识点 /
解题模型表 / 旋钮）。注入照 Claude Code Agent Skills/attachment 机制：DAG 节点把上下文组装进
prompt，LLM 只填空。

### ① 解题 skill（stage1·带料解题）
- **prompt**：`stage1_anchor/solve.py`（`_solve_one` 独立验算）+ `prompts.py` `SOLVE_PROMPT`；
  母题主解题走 `agents/variant_entry.mother_opus_entry`（一次 opus 解题 + 10 维打标合一）。
- **声明注入的上下文**：
  - KG —— 年级/章/知识点锚定（`facts`）。
  - 解题模型表 —— `agents/model_anchor.toolbox_for_grade(grade)` 拉「该年级全量模型名单」，
    `build_toolbox_clause` 组装成「带料」注入块塞进解题 prompt（AC1：非裸解 stem）。
- **铁律**：全流程**只一次 LLM 解题**（B1 消重复解题）；模型名 → M-id+tier/freq 的库内对齐 =
  纯代码 SQL（`model_anchor.anchor_models_from_names`），**不再二次 LLM 解题**（AC2）。

### ② 打标 skill（stage1·10 维 DNA + 诚实三态模型）
- **prompt**：打标并入 `mother_opus_entry` 同一对话（与解题共结论，不另起 LLM）；
  `stage1_anchor/label.py` `_item_dna` 把结论组装成 DNA 帧。
- **声明注入的上下文**：解题结论（kp 叶子 / 模型命中）+ KG。
- **诚实三态（AC8/G8）**：去 M00 兜底 —— 有考模型 → 真模型 summary；真无 → `models:[]` +
  `model_flag:"no_model"`（FE 渲「无考模型」）；**绝不硬凑**。难度无模型仍出档（降级 K/R/D/G）。
- **难度**：表驱动 `grade_observed`（`stage2_variant/difficulty.py` + `core/difficulty`），
  **绝不 LLM 自评**（铁律）。

### ③ 变式 skill（stage2·每道独立子上下文）
- **prompt**：`stage2_variant/prompts.py` `GENERATE_ONE_PROMPT`（出**单道**变式）+ `REGEN_PROMPT`
  （回炉重出一道）。
- **声明注入的上下文（每道一份）**：母题核心参照（`compress.py` 固化的 frozen `MotherCoreRef`，
  经 `facts_from_ref` 取）+ 本道 PLAN 派工（`plan_variant_specs`：基准系数 + 带内浮动 + 轮换算子
  + 难度 md+i 递增）+ 组级共享前缀（守恒 / 上下文 / 图型闸）。
- **派工旋钮**：变式系数（`operator_band_from_similarity` + `_float_coeff` 带内浮动）→ 每道
  prompt 回填**真实系数数字**（AC5/G6：显示值与 prompt 指令一致）；算子轮换保证一组不撞车。

---

## 阶段骨架（固定 DAG，未来工作流可复用这层）

```
入口(mother_opus_entry: 解题①+打标②) → await_review【🛑STOP1】→ compress(固化 MotherCoreRef)
   → generate(变式③·节点内 asyncio.gather(Semaphore=3) fan-out) → gene_gate → solve_explain
   → assemble 【🛑STOP2 由 await_review/确认路径承载】
```

- **阶段边界 = compress 压缩闸**（`stage1_anchor/compress.py`）：阶段一对话蒸成 `MotherCoreRef`
  （结构化 DNA + 一句人话摘要 + 题面/答案/解析/配图），阶段二**只读它、不继承阶段一原始对话**
  （AC3/G3 隔离）。frozen 参照供 fan-out 子任务共享，**不各自重算 `_mother_facts`**（防并发读脏）。
- **fan-out 红线**：变式 per-item 走「generate 节点内 `gather(Semaphore=3)`」，**绝不用 LangGraph
  `Send`**（`merge_items` reducer 语义=「new=权威全集」，与 Send 的 fan-in 冲突；B0 钉死）。

---

## 两处硬停（AC6/G5，→END+config resume，不引 interrupt）

1. **STOP1 · 阶段一锚定完**：`await_review`（`graph.py`）置 `awaiting_mother_review` + END。
   载体 = 母题卡（`mother_card.py` `_build_mother_card`），呈现 **范围 + 模型 + 难度** 三行
   （模型走诚实三态）。老师点「开始举一反三」→ `route_entry` 经 config 回传 resume → compress → generate。
2. **STOP2 · 阶段二出题完**：出题落 assemble 后停，老师确认**验算 / 入库**（验算 = 单题
   `/variant/verify-one`、入库 = `/variant/persist`，确定性动作直连接口，FE 触发）。

resume 一律走「续聊回合 + agent_config 回传信号」（`route_entry` 分诊），**不引 `interrupt()`**
（固定 DAG·K-12 可复现，§4）。

---

## 加能力 = 加目录（拓展性根）

新工作流/新能力 = 加一个内聚模块（prompt + 声明注入上下文）+ 在 `graph.py` 接 DAG 节点，
`shared/` + `domain/` 不动。本期不提前做统一跨工作流接口（北极星：建第 2 条时顺手结晶）。
