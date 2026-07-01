# 图形 agent 切「构件 DSL」接线说明（待维护者 review + 真机端到端验后 flip）

> 背景：geometry-board 的**构件层**已落地 book-ui（`public/geo-engine/figure-builder.js` + 统一
> `geo-dsl-render.js`），前端已能自动展开高层 `build:[...]` 规格并渲染（真机验过）。本 agent 若改吐
> 构件 JSON，就能**不盲打坐标**（坐标/角度/共圆/内切由代码精确解，几何不变量 75 项 node 测保证）。
>
> 🔴 **为何没有直接 flip**：改 live 图形 prompt 会动生产举一反三管线，且必须**真机端到端验**
> （book-ui 登录 + toolkit :8093 + 真 LLM 调用 + 真图），无人值守跑无法完整验证 → 按 C 线 vibe 协议
> （prompt = 维护者 review 领域、"真实数据跑过才算完成"）**只做就绪基建，不擅自炸生产**。

## 已就绪（本轮已做，非破坏、dormant）
1. `dsl_schema.validate_dsl` 已**双格式**：spec 含 `build` 且无 `objects` → 走 `validate_construct`
   （结构浅校验：每项恰含一个 shape/add/mark/transform 键 + 值在白名单；几何深校验交前端 figure-builder）；
   否则按低层 DSL 深校验（**不回归**，7 项 pytest 绿：`tests/test_construct_validate.py`）。
2. `construct_system.CONSTRUCT_SYSTEM` = 构件生成器 prompt（就绪，含逃生口：罕见构型退低层 DSL）。
3. 前端 `geoEngine` 已能渲 `build` 规格（`GeoBoard` 三态 + `views/geo-board` 真机验过）。

## flip 三处改动（`src/agents/figure/compose.py`，维护者 review 后改）
在 `compose_variant_dsl`（约 505–580 行）：

1. **换 system 头**（约 523 行）：
   ```python
   from agents.figure import construct_system            # 顶部 import 处加
   ...
   SystemMessage(content=construct_system.CONSTRUCT_SYSTEM),   # 原 dsl_system.DSL_SYSTEM
   ```
2. **产出检查认 build**（约 536、549 行两处 `not data.get("objects")`）：
   ```python
   _has_payload = isinstance(data, dict) and (data.get("objects") or data.get("build"))
   if not _has_payload:   # 原 if not isinstance(data, dict) or not data.get("objects")
   ```
   （retry 提示文案顺带改成「请给 build 数组或 objects 数组」。）
3. **日志 n_objects**（约 574 行）容错：`len(data.get("objects") or data.get("build") or [])`。

> `validate_dsl` 不用改（已双格式）。前端不用改（已自动展开 build）。

## flip 后必验（真机端到端，过了才算完成）
- [ ] toolkit :8093 起；book-ui :8091 登录；举一反三跑一道**几何题**（含图母题）。
- [ ] agent 产出 `{"build":[...]}`（看日志/artifact），过 `validate_construct`，前端渲出图。
- [ ] 图**几何正确**（∠=90°真直角/等边真等/交点在对角线中点）——构件层保证，肉眼复核。
- [ ] 跑一道构件覆盖不到的**罕见构型**，确认 agent 走逃生口退低层 DSL、仍出图（能力不减）。
- [ ] 对照 flip 前后若干题，确认举一反三整体不回归（成图率、老师修正链路）。
- [ ] 🔴 nano/换模型复测（构件 prompt 在小模型上可能选错构件，参照 [[jiuyi-fansan-line-c107-c110]] 的 nano 雷）。

## 回滚
一处 import + 三处判断改回即恢复低层 DSL（生产原状）。`validate_construct` / `construct_system` 留着无害（dormant）。
