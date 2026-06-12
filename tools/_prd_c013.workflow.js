export const meta = {
  name: 'prd-c-013-batch',
  description: 'PRD-C-013 体验更迭二期：变式 agent 9 包多 agent 批次实施（serial variant.py spine + 并行 FE/sympy + 对抗 review）',
  phases: [
    { title: '实施', detail: 'variant.py 串行脊柱(4阶段) ‖ book-ui FE ‖ math_verify 4b' },
    { title: '收口', detail: 'nano bake-off + 全量 pytest' },
    { title: '对抗review', detail: '多 agent 挑刺(脊柱耦合逻辑/帧序/嫁接徽章)' },
  ],
}

// ============ 共享上下文（每个 agent prompt 都前置）============
const TK = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\_learn-langgraph\\\\agent-service-toolkit'
const UI = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\book-ui'
const VPY = `${TK}\\\\src\\\\agents\\\\variant.py`

const CTX = `
# 工作区与铁律（C 线·举一反三 agent 二期 PRD-C-013）
- toolkit (Python/FastAPI 变式 agent) 根目录: ${TK}，端口 :8093，git 分支 prd-c-009-variant。
- book-ui (Vue3) 根目录: ${UI}，端口 :8091，git 分支 master-ai。
- 主战场单文件: ${VPY}（~2800 行）。
- 🔴 铁律(CLAUDE.md §4/§5): ① 宏观 DAG 确定性，预算闸是"跳过增强调用"不是"LLM 决定流程"；② bug 先 root cause 再修，不黑盒；③ 真机/真数据跑过才算完成；④ 一切闸门必有降级路径，绝不卡死。
- 🔴 已知坑(全适用): 新建 biz_* 表必须加 application.yml tenant.excludes(本期应无新表)；graph 入口要 ruoyi_token 硬闸；\`_solve_one\` 回写 solution 必过 \`_sanitize_rich_text\`；\`_LITERAL_NL_RE\` = re.compile(r"\\\\n(?![a-z])")，\$m\\\\ne 0\$ 的 \\\\ne 是合法 LaTeX 别误伤；parse 无题组语境注记必须声明"最高优先级覆盖硬守恒段"(17号修复不可回退)；.env/data/* 永不入 git；别杀不是本 session 起的进程。
- LLM 出口: RELAY_POOL=[aigeek(主),lk888(备)]，默认思考模型 gpt-5.4，VARIANT_MAX_TOKENS≥4096 别动小；轻活模型 = gpt-5-nano(aigeek，无识图)。
- 测试命令(cwd=${TK}): .venv\\\\Scripts\\\\python.exe -m pytest <路径> -q  （一期基线 346 passed,2 skipped；忽略 tests/app 与 tests/service）。
- 行号会随编辑漂移：动手前用 Grep 按符号名重新定位，别盲信给定行号。
- 改完不要自行重启 :8093 服务、不要跑需要 live 服务的探针(主 session 统一驱动)；你只跑离线 pytest。
`.trim()

const REPORT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['summary', 'files_changed', 'tests_run', 'tests_pass', 'open_risks'],
  properties: {
    summary: { type: 'string', description: '改了什么、为什么(root cause 级)，3-8 行' },
    files_changed: { type: 'array', items: { type: 'string' }, description: '改动文件相对路径列表' },
    tests_run: { type: 'string', description: '跑的 pytest 命令 + 结果(passed/failed 数)' },
    tests_pass: { type: 'boolean', description: '本 agent 负责的离线测试是否全绿' },
    open_risks: { type: 'array', items: { type: 'string' }, description: '留给下游/review 的风险点或未决项' },
  },
}

// ============ 阶段 1：并行三轨 ============
phase('实施')

// ---- 轨 A：book-ui FE（独立 repo，按约定帧契约写，不等 BE）----
const FE_PROMPT = `${CTX}

你的任务 = book-ui 前端三项(P10 图进对话流 / P2b-FE 逐题上屏 / P8-FE 难度星按值渲染)。只动 ${UI} 下的文件，绝不碰 toolkit。

## 关键文件(已勘察, 行号需重新定位)
- src/views/variant/VariantCard.vue: props 在 ~21-25; verifyBadge computed ~31-43(按 item.tier: verified/self_ok/proof/both_low/silent); geneBadge ~45-48; 难度渲染当前在 line ~69 \`<span v-if="item.difficulty>0" class="meta-tag">难度 {{ item.difficulty }}</span>\`(纯文本, 无星, 无上限).
- src/views/variant/ArtifactPanel.vue: items computed ~31 = props.artifact?.items; 当前"整帧替换"语义; partial/expectedTotal 在 ~34-42(partial=true 时按 expectedTotal-items.length 渲染骨架占位卡).
- src/api/variant/index.ts: VariantArtifactItem 类型 ~74-111(difficulty:number, tier?, verify?, gene?, qtype?, seq?); streamVariant ~215-308(SSE 消费, artifact 帧处理).
- src/views/variant/index.vue: 消息类型 Bubble/RailItem ~46-59(无 images 字段); localStorage 注册表 key 'variant.sessions.v1' ~129-195(SessionMeta{id,title,at,img?}); 贴图/上传 uploadPastedImage ~85-124 + onPaste ~108-124; 母题图顶部锚位 .mother-bar/.mother-thumb ~554-577; 发送时文案构造 ~329-351.
- 类型门禁 = pnpm build (= vue-tsc -b && vite build)。**你不要跑 pnpm build(耗时且可能需依赖)**, 只做到代码改完、自检 import/类型自洽; 主 session 统一跑 build。

## P2b-FE(逐题上屏) — 按此帧契约实现(BE 同步在改, 契约已定死)
BE 的 artifact 增量帧语义二期变为: ① 每道题"刚解析出完整内容(stem/answer/solution齐)"即发一帧, 该题 item **无 tier 字段**(或 tier 缺省), partial=true; ② 该题闸链(闸A+闸B)完成后**原位重发**同 seq 的 item, 此时带 tier 字段。最终 assemble 发 partial=false 定稿帧。
要求:
- VariantCard 增 "checking" 过渡态: 当 item 缺 tier(或显式 tier==='checking')时, 徽章位显示呼吸/"验算中…"占位, 不显示 ✓/⚠; tier 到达后原位更新, 不闪烁、不换位。
- ArtifactPanel 改为**按 seq 原位 merge**而非整组重渲: 收到新 artifact 帧时, 对 items 按 seq 做 upsert(已存在的 seq 原位替换该卡数据, 新 seq 追加), 而不是整数组替换导致整列重渲。保留一期的骨架占位卡逻辑(partial 且 expectedTotal>已到达数 时补占位)。
- 保证: 首题内容可见后, 徽章后到只更新该卡; 剔除题(BE 后续帧里该 seq 消失/标记 _dropped)有可见退场过渡。

## P8-FE(难度星按值)
- 把 VariantCard 的"难度 N"文本改为按 item.difficulty 数值渲染星级(实心星 = difficulty, 空星补足)。**不要假设上限是 4 或 5**: 存量库题可能是 5, 新生成题是 1-4。用 Math.max(value, 5) 之类动态上限或直接画 value 颗实心星 + 一句 title="难度 N"。简洁即可, 不引新依赖。

## P10(母题图进对话流, 纯 FE)
- 消息模型(Bubble)加可选 images?: string[]。
- 贴图/上传后: 输入区先显示"待发附件"缩略图(可删除), 发送时图片随用户气泡进聊天流(~120px 缩略图, 点开看大图; 同气泡可带文字)。
- 会话恢复: checkpointer 回放的 messages 没有图 → 把每条消息的 imageUrl 存进 FE localStorage 会话注册表(sessions.v1 已有, 扩 per-message URL 记录), 恢复时贴回对应气泡。
- 顶部"母题"锚位保留为当前母题小徽章(守恒锚视觉提示), 不再是唯一展示位。
- BE 零改动(imageUrl 走独立字段、不进 messages)。

每项改完自检 TS 类型自洽。完成后按 schema 返回(tests_run 填 "FE 不跑 pytest, 类型门禁交主 session pnpm build"; tests_pass 填 true 表示你已自检类型自洽)。`

// ---- 轨 B：math_verify 4b 扩面 + 单测（独立文件）----
const SYMPY_PROMPT = `${CTX}

你的任务 = math_verify 三项 sympy 扩面(4b) + 各 ≥5 条单测。**只动这两个文件**: src/agents/math_verify.py 与 tests/test_math_verify.py。绝不碰 variant.py(其 _PAYLOAD_CONTRACT 由 BE 脊柱 agent 同步加新 kind, 与你无关)。

## 现状(已勘察)
- math_verify.py: 现有 kind = equation_solve / expr_equiv / numeric / choice; 分发表 _HANDLERS(~416); 公共入口 verify(payload)->{verdict,detail,computed}, 永不抛(异常→degrade)。verdict 词表: PASS="pass"/FAIL="fail"/DEGRADE="degrade"(常量 ~37-39)。解析器 _parse(~118)/\_parse_equation(~157)/\_equiv(~178)/\_solve_equations(~214)/\_match_solution_set(~244)。复杂度护栏 _complexity_guard(~81, _DegradeError)。函数白名单 _ALLOWED_FUNCS = sqrt/Abs/Min/Max(~56)。
- 单测约定: tests/test_math_verify.py, helper _v(payload) 包 verify 并断言返回形状; 按 G 规则分组(G1 对错、G2 干扰命中真值→fail、G5 垃圾→degrade 不抛 + 防爆 <5s)。

## 三项扩面(每项独立纯函数 + 注册进 _HANDLERS + ≥5 条单测)
1. **inequality_solve**(不等式/解集区间): 用 sympy solve_univariate_inequality(或 solveset 配 S.Reals)解单变量不等式, 把 claimed 解集与真解集比对(相等→pass, 不等→fail, 解析不了→degrade)。payload 形如 {kind:"inequality_solve", inequality:"2*x-3>5", unknown:"x", claimed:"x>4"}。注意比较解集需化成可比形式(Interval/Set)。≥5 测: 一次不等式 pass、错解集 fail、复合/取等边界、垃圾→degrade、等价不同写法→pass。
2. **舍根子集模式**(分式方程增根): 验"claimed 根 ⊆ 全解集 且 满足定义域(分母≠0)且非空"。做法: 解原方程候选根 → 对每个 claimed 根 checksol 代回原式 + 域校验(分母≠0)→ 剔增根 → claimed 应恰等于有效根集。payload 形如 {kind:"equation_solve" 扩展可选 domain/exclude 字段, 或新 kind "rational_roots"}。**自选最小侵入设计**(扩 equation_solve 加可选 domain 字段 或 新 kind 二选一, 在 summary 里说明你的选择与理由)。≥5 测: 正常无增根 pass、有增根但 claimed 正确剔除 pass、claimed 漏剔增根 fail、claimed 含使分母0的根 fail、垃圾 degrade。
3. **应用题建模验算**(复用 equation_solve): 不新增解题逻辑, 验"4a 载荷给的方程 + 申报答案"是否计算一致(方程解==答案)。本质是 equation_solve 的应用; 若现有 equation_solve 已能吃 {equations,unknowns,claimed} 就**只补单测**证明应用题载荷可走通, 不必新函数。≥5 测: 行程/工程/浓度类方程各 1 + 答案对 pass + 答案错 fail。在 summary 说明是否需新函数。

## 验收
- 跑: .venv\\\\Scripts\\\\python.exe -m pytest tests/test_math_verify.py -q  必须全绿。
- 不破坏现有 45 条 math_verify 测试。
- 完成后按 schema 返回。`

// ---- 轨 C：variant.py 串行脊柱（4 阶段，IIFE 内顺序 await，绝不并行同一文件）----
const SPINE_S1 = `${CTX}

你是 variant.py 串行脊柱**第1阶段(地基)**。本阶段为后续 P12/nano 铺路, 只做三件互相支撑的事, 改完跑 variant 相关离线测试。**只动 ${VPY} 与 .env.example**(不动 .env 真文件; 不动 FE/math_verify)。

## S1.1 per-call model 覆盖(nano 的前置, 当前不存在!)
现状: \`_ainvoke_text\`(~254) 经 relay_pool.ainvoke_failover() 调模型, **模型固定取 relay 配置(gpt-5.4), 无法 per-call 覆盖**。nano 降本需要按调用点传不同 model。
做法: 给 _ainvoke_text 增可选参 \`model: str|None=None\`; 透传到 relay_pool.ainvoke_failover(); 看 src/core/relay_pool.py 的 ainvoke_failover/\_relays(~53-84), 让其接受可选 model 覆盖该次请求的 model 参数(站点 base_url/api_key 不变, 只换 model 字段)。.env.example 增 \`LLM_MODEL_LIGHT=gpt-5-nano\`; settings 加对应字段(src/core/settings.py ~164-172 RELAY_POOL 附近)。保持向后兼容: model=None 时行为完全不变。

## S1.2 P8 难度总评 _grade_difficulty(items)
新增一次 nano call, assemble 前评全组成题(题干+答案+解析)按绝对 rubric 打 1-4:
\`1=送分概念直读 / 2=常规(单步套用~两三步常规综合) / 3=多步综合或需构造转化 / 4=压轴级\`, 锚浙教版初中。
- 用 LLM_MODEL_LIGHT(gpt-5-nano) 经 S1.1 的 model 覆盖调用。
- 返回每题难度值, 覆盖 item['difficulty']; 越界钳到 1-4; 解析失败/异常 → 降级保留各 item 原 difficulty 值, **绝不卡死**(G5 风格)。
- 在 assemble 节点(~2165)调用, 覆盖后再做后续(P9 排序会用到, 但 P9 在 S4 做; 本阶段先把 difficulty 覆盖落到 item 上 + 入库值跟随)。入库 difficult/dim4 来自 item.difficulty(variant_support._clamp_difficult 已钳 1-4, ~256)。

## S1.3 DIFFICULTY_CAP 5→4 + 算术钳同步
- DIFFICULTY_CAP(~849) 由 5 改 4。
- recipe_from_knobs(~929) 里 \`md = _to_int(mother_difficulty) or 3\`(~952) 与 \`expected=[min(md+i,DIFFICULTY_CAP) for ...]\`(~961) 及其它 md+1 算术(~1025/1239)同步钳到 4。
- 注意: FE 星显示按值渲染不假设上限, 故存量 5 不受影响; 这里只钳新生成题的配方申报上限。

## 验收
- 跑: .venv\\\\Scripts\\\\python.exe -m pytest tests/test_variant_knobs.py tests/test_variant_pipeline.py tests/test_relay_pool.py -q  应全绿(若 knobs 测试断言 DIFFICULTY_CAP=5 之类需同步改, 确认是语义升级再改, summary 说明)。
- _grade_difficulty 加一条单测(mock 越界钳 1-4 + 解析失败保留原值不抛)。
- 完成后按 schema 返回; open_risks 里务必交代: S1.1 的 model 覆盖签名(给 S3/S4 用)、_grade_difficulty 的位置与返回结构(给 S4 P9 排序用)。`

const SPINE_S2 = `${CTX}

你是 variant.py 串行脊柱**第2阶段(题型契约 P11 + payload 契约扩面)**。S1 已落地(per-call model 覆盖 + _grade_difficulty + DIFFICULTY_CAP=4)。**只动 ${VPY}**。先 Grep 重新定位行号(S1 已移动了行)。

## P11.1 新增共享常量 _QTYPE_CONTRACT(照 _PAYLOAD_CONTRACT 模式, ~720)
单一事实源, 定义各题型结构契约:
- 选择 = 单一设问 + 恰 4 个选项(A-D) + answer 为选项字母; **禁止 (1)(2) 小问嵌合体**。
- 填空 = 题干含空位(____) + answer 为值。
- 判断 = 单一陈述 + answer 为 对/错。
- 解答 = 允许 (1)(2) 多小问。
把固定段放前、变动段放后(吃中转 prefix cache)。

## P11.2 注入三个出题 prompt
GENERATE(~737) / REGEN(~1420) / ADD(~2553) 均插入 _QTYPE_CONTRACT 段(像现有 _PAYLOAD_CONTRACT 那样 .format 拼接)。

## P11.3 代码级结构 lint(纯函数, 进 shape_check 同位)
新增纯函数, 对每道生成题按其 qtype 校验结构:
- 选择题 stem 含 (1)(2)/①②等多小问标记 → 缺陷; 选项数 <3 → 缺陷; answer 非单个字母(A-D) → 缺陷。
- 填空: 题干无空位标记 → 缺陷(宽松, 缺空位才报)。
- 命中缺陷 → 产出题级缺陷反馈触发该题 retry(走既有 shape/题级重试通道, 不是整组重试)。通过后断言结构合规。
- 找到现有 shape_check / 题级缺陷处理位置(Grep "shape" / "_iter_complete_items" / 题级缺陷), lint 挂在同位; **降级**: lint 解析不了别卡死, 标 ⚠ 继续。

## P11.4 _PAYLOAD_CONTRACT 增 4b 新 kind(配合并行的 sympy 轨)
在 _PAYLOAD_CONTRACT(~720) 的 kind 枚举/说明里增: \`inequality_solve\`(不等式解集, 字段 inequality/unknown/claimed)、分式方程舍根(若 sympy 轨选"扩 equation_solve 加 domain 字段"则在 equation_solve 说明里补 domain/exclude; 若选新 kind 则加该 kind 名)、应用题(复用 equation_solve, 让 LLM 出题时同步给 equations+unknowns+claimed)。**与 math_verify 的实现保持 kind 名一致**(sympy 轨默认: inequality_solve + equation_solve 扩 domain + 应用题复用 equation_solve; 若不确定就都列上, 宁宽勿缺)。出题端只是"可选给载荷", 判决仍只读 sympy verdict(铁律不破)。

## 验收
- 跑: .venv\\\\Scripts\\\\python.exe -m pytest tests/test_variant_sanitize.py tests/test_variant_guardrails.py tests/test_variant_pipeline.py -q 应全绿。
- 给结构 lint 加 ≥3 条单测(选择题嵌合体被抓、选项不足被抓、合规通过)。放进 tests/test_variant_guardrails.py 或新 tests/test_qtype_contract.py。
- 完成后按 schema 返回; open_risks 交代 lint 函数名/位置 + _QTYPE_CONTRACT 注入点(给 S3 转题型 structure 判定用)。`

const SPINE_S3 = `${CTX}

你是 variant.py 串行脊柱**第3阶段(闸A 重构 P12 + nano 切换)**——本卡最核心、最易出错的一阶段。S1/S2 已落地。**动 ${VPY} + tests/test_gene_gate.py + tests/test_variant_heal.py**。先 Grep 重新定位。

## 根因(必须先理解再动手, 来自 conv_trace 实测)
闸A 70% 打回不是"对抗性强", 是**判错维度+回炉修不了刺**: 最近40次 judge 失败 difficulty_match 占28(judge 主观目测对赌 generate 算术申报值, 两个噪声源互比永远对不齐, 完美平行题被"感觉难一点"打回); rework 复用 REGEN_PROMPT 但①前提文案是验算自愈文案(前提就错)②prompt 里没有母题骨架却要求"与母题一致"(收敛率≈随机)③难度类失败无人告诉该出几档; judge 输入 _clip(300) 把多小问解答砍半。

## P12.1 difficulty_match 从闸A 删除
- GENE_JUDGE_PROMPT(~1957) 去掉 difficulty_match 判项; gene_judge_knobs_spec(~1062)里 difficulty 相关注入段删除/改为相对关系。
- 难度一致性改为**纯函数比较组内相对关系**(hard 题的总评难度 ≥ 同组 normal 题), 读 S1 的 _grade_difficulty 产出值, **零 LLM**。找到闸A 判决处(_gene_one_item ~2076 / gene_gate_decision), difficulty_match=False→rework 的逻辑改为不再因难度回炉。
- ⚠ test_gene_gate.py 会断言 difficulty_match=False→rework(~55-56)、_judge() mock 含 difficulty_match(~34): **改这些断言**(删 difficulty_match→rework 用例、_judge helper 去 difficulty_match)。改前确认是语义升级不是回归。

## P12.2 judge 输入不截断(_clip 300→1200)
- _clip 默认(~1981) 或 GENE_JUDGE 调用处(~2041-2046) 把 mother_stem/variant_stem 上限 300→1200(skeleton 可留 200 或提到 400)。

## P12.3 rework 重设计
- **难度类失败已不存在**(P12.1 移除)。
- **结构类失败**: 用**带母题骨架的专用 rework 反馈段**定向修(把母题 stem/skeleton 放进 REGEN feedback, 明示"符合目标题型规范结构 + 解法核心步骤同源(考点级)"); 或预算紧时直接 warn 不回炉。
- **编辑轮产物永不回炉**(RC2): exec_regenerate(~2487) 产出的带老师 note 的题进闸A 只判不 rework(判不过标 warn, 4d 下沉默, 老师意志优先)。
- **REGEN 透传老师 note**(RC2): 所有 REGEN 路径(_regen_once ~1670 / exec_regenerate ~2514)保住老师 note, feedback 不能只带基因失败原因把 note 丢了。
- **转题型 structure 判定**(RC1): knobs 含题型转换时, GENE_JUDGE 的 structure_match 改判为"符合**目标题型**规范结构 + 解法核心步骤同源", 不再对照母题原骨架(否则规范单问选择题被判"解法结构变了"→嵌合体)。用 S2 的 _QTYPE_CONTRACT 作目标题型规范依据。

## P12.4 nano 切换(parse 分类器)
- PARSE prompt 调用点改用 LLM_MODEL_LIGHT(gpt-5-nano)经 S1 的 model 覆盖(parse ~2210 的调用处, Grep _ainvoke_text 找 parse 调用)。语义不动, c017 探针(主 session 跑)把关。
- 闸A judge **不在本阶段切 nano** —— 留给主 session 的 bake-off(收口阶段)决定, 本阶段 judge 保持 gpt-5.4。

## 验收
- 跑: .venv\\\\Scripts\\\\python.exe -m pytest tests/test_gene_gate.py tests/test_variant_heal.py -q  全绿(已按语义升级改断言)。
- test_variant_heal 的 4d 可见性矩阵断言(~210-248: sympy_pass 即 verified、单闸低沉默、双闸低 warn)应仍成立——别破坏。
- 完成后按 schema 返回; open_risks 交代: 编辑轮"只判不 rework"的实现点(给 S4 P2b 帧序用)、structure_match 转题型判定的实现。`

const SPINE_S4 = `${CTX}

你是 variant.py 串行脊柱**第4阶段(预算闸 P13 + 叙事 P14 + 逐题上屏 P2b-BE + 排序 P9)**。S1/S2/S3 已落地。**动 ${VPY} + 必要的 tests**。先 Grep 重新定位。

## P13 预算闸(state 级 LLM 调用计数器)
- State(VariantState) 加 \`llm_call_budget\` 计数(per-round 重置)。
- 在 _ainvoke_text(~254) 处统一记账(每次成功调用 +1; 注意区分出题轮 vs 编辑轮上限)。
- 超限后**增强类调用**(闸A rework / 闸B heal / replenish 补题 / extract 兜底)直接**跳过走既有 G5 降级路径**(标 ⚠ / 保留原题, 不卡死); 核心链(parse/generate/grade)不跳。
- 上限进 .env.example: 出题轮 18 / 编辑轮 6(VARIANT_BUDGET_GENERATE=18, VARIANT_BUDGET_EDIT=6 之类); settings 加字段。
- 🔴 铁律: 是"跳过增强调用"不是"LLM 决定流程", 宏观 DAG 不破。
- 单测(G6): mock 调用计数超限后增强类调用被跳过且流程正常收尾。

## P14 叙事修正(RC3)
- stage 文案 \`第 {idx+1}/{total} 道\`(平行度比对 ~2092 / 程序验算 ~1276 等多处)改为 \`第 N 题验算中\`(去掉误导性的"/总数")。
- 编辑轮(单题重验)叙事明示 \`只重验第 N 题\`(别让老师把题号读成"全部重验")。
- 找全 Grep "第" + "道" / "{total}" / "idx+1" 的 stage 文案处。

## P2b-BE 逐题上屏(把"出卡"与"过闸"解耦)
现状(_eager_chain ~1260): 增量 artifact 帧在"单题过完整条闸链后"才发(~1276-1282), 三题闸链并发耗时相近→几乎同时完成→体感一次性蹦。
改为:
- _gen_progress(~1289) 流内一解析出完整题(stem/answer/solution 齐)**立即发增量帧**(该 item **无 tier 字段**, partial=true) 上卡。
- 闸链(_eager_chain)完成后**原位重发**该题帧(同 seq, 带 tier)。
- assemble(~2165) 发 partial=false 定稿帧。
- 剔除题(真fail): 后续帧里该 seq 标记退场, 配 stage 叙事"第 N 题程序验出标答错误, 已剔除、补一道中"。
- 🔴 必须保留一期语义: 中转熔断会从头重流→stale 流重启检测(_gen_progress 的 len 下降取消 stale eager 任务, ~1289-1318) + stem 同一性 merge 校验 全部保留, 别删。
- FE 已按"无tier帧→带tier帧→定稿帧"契约实现, 你只管 BE 按此发帧。
- 单测(G3): SSE 帧序 = 每题先出无 tier 帧、后出带 tier 帧。

## P9 排序
- **默认序**: assemble 前按 _grade_difficulty 总评难度**升序稳定排序**(同难度保持生成序)。seq 重编, persisted 簿记跟题走不错位。
- **指令排序**: PARSE 受约束枚举(~2210, R0-R7 护栏 ~2273-2286)新增 \`reorder\` op(与 remove/regenerate/add 同走指令通道; 越界 index/混类→clarify, 永不默认重排)。执行 = **纯代码 list 重排 + seq 重编**, 零 LLM 改题; artifact 整帧重发。
- 入库顺序 = 当前显示序。
- 单测(G5): reorder op 解析(含越界/混类→clarify) + 重排后 stem 集合不变断言。

## 验收
- 跑: .venv\\\\Scripts\\\\python.exe -m pytest tests/test_variant_stage.py tests/test_variant_pipeline.py tests/test_gene_gate.py tests/test_variant_heal.py tests/test_variant_route_entry.py tests/test_variant_knobs.py -q 全绿。
- 完成后按 schema 返回。`

const [feRes, sympyRes, spineRes] = await parallel([
  () => agent(FE_PROMPT, { phase: '实施', label: 'fe:P10+P2b+星级', schema: REPORT_SCHEMA }),
  () => agent(SYMPY_PROMPT, { phase: '实施', label: 'sympy:4b扩面', schema: REPORT_SCHEMA }),
  () => (async () => {
    const s1 = await agent(SPINE_S1, { phase: '实施', label: 'be-s1:地基(model覆盖+难度总评)', schema: REPORT_SCHEMA })
    const s2 = await agent(SPINE_S2 + `\n\n# 上游 S1 交代\n${s1 ? JSON.stringify(s1.open_risks) : '(S1 无返回)'}`, { phase: '实施', label: 'be-s2:题型契约', schema: REPORT_SCHEMA })
    const s3 = await agent(SPINE_S3 + `\n\n# 上游交代\nS1:${s1 ? JSON.stringify(s1.open_risks) : '-'}\nS2:${s2 ? JSON.stringify(s2.open_risks) : '-'}`, { phase: '实施', label: 'be-s3:闸A重构', schema: REPORT_SCHEMA })
    const s4 = await agent(SPINE_S4 + `\n\n# 上游交代\nS3:${s3 ? JSON.stringify(s3.open_risks) : '-'}`, { phase: '实施', label: 'be-s4:预算闸+P2b+排序', schema: REPORT_SCHEMA })
    return [s1, s2, s3, s4]
  })(),
])

log('阶段1完成: FE / sympy / variant.py 脊柱4阶段 均已落地, 进入收口')

// ============ 阶段 2：收口 = nano bake-off 脚本 + 全量 pytest ============
phase('收口')

const BAKEOFF_PROMPT = `${CTX}

阶段1已落地(per-call model 覆盖已存在于 _ainvoke_text)。你的任务 = 写闸A judge 的 nano vs gpt-5.4 bake-off 脚本(新文件 tools/c018_judge_bakeoff.py), 用于决定闸A judge 是否切 nano(一致率 ≥90% 才切)。

- 复用 tools/c018_benchmark_replay.py 的 10 母题数据 + tools/_probe_auth.real_token()(服务账号真登录)。
- 对同一批变式题, 分别用 gpt-5.4 与 gpt-5-nano 跑闸A judge(经 S1 的 _ainvoke_text model 覆盖, 看 variant.py 里 judge 调用点), 统计 judge 四项(qtype/structure/surface 等, difficulty_match 已删)结论的一致率。
- 输出一致率报告(总体 + 分项), 末尾打印 "一致率=X% → ≥90% 可切 nano / 否则保持 gpt-5.4"。
- ⚠ 这是 **live 脚本**(需 :8093 不依赖, 但需真 LLM + RuoYi token), **你只写脚本、不运行**(主 session 驱动 live 跑)。写完做语法自检(.venv\\\\Scripts\\\\python.exe -m py_compile tools/c018_judge_bakeoff.py)。
- 风格照 c017_clarify_probe.py / c018_benchmark_replay.py。
- 完成后按 schema 返回(tests_run 填 py_compile 结果)。`

const bakeRes = await agent(BAKEOFF_PROMPT, { phase: '收口', label: 'bake-off脚本', schema: REPORT_SCHEMA })

// 全量 pytest（barrier 后, 所有轨道改动都已落盘）
const PYTEST_PROMPT = `${CTX}

所有代码改动已落盘(variant.py 脊柱4阶段 + math_verify 4b + 各新单测)。你的任务 = 跑**全量离线 pytest** 并如实报结果, 修掉因二期语义升级而**误红**的既有断言(仅限确属语义升级的, 真回归要在 open_risks 标红别强行洗绿)。
- 跑: .venv\\\\Scripts\\\\python.exe -m pytest tests -q --ignore=tests/app --ignore=tests/service
- 一期基线 = 346 passed, 2 skipped; 二期新增单测后总数应上升。
- 若有 fail: 逐个判定是"语义升级需改断言"还是"真回归(代码 bug)"。前者改断言(说明理由); 后者**不要改测试掩盖**, 在 summary/open_risks 精确描述 root cause + 涉及文件行号, 留给主 session/review。
- 完成后按 schema 返回: tests_run 填最终 "N passed, M skipped, K failed"; tests_pass = 是否全绿(无真回归); open_risks 列所有未解 fail 的 root cause。`

const pytestRes = await agent(PYTEST_PROMPT, { phase: '收口', label: '全量pytest', schema: REPORT_SCHEMA })

log(`收口完成: pytest = ${pytestRes ? pytestRes.tests_run : 'n/a'}`)

// ============ 阶段 3：对抗 review（并行多 agent 挑刺）============
phase('对抗review')

const allChanged = [feRes, sympyRes, ...(spineRes || []), bakeRes, pytestRes]
  .filter(Boolean).flatMap(r => r.files_changed || [])
const changedList = [...new Set(allChanged)].join(', ')

const REVIEW_DIMS = [
  {
    key: '脊柱耦合逻辑',
    focus: `专挑 variant.py 二期改动的耦合 bug: ① 预算闸记账是否把 P8 难度总评/nano parse 这类核心调用误算进"增强类"被跳过(应只跳 rework/heal/replenish/extract); ② difficulty_match 删除后, 组内相对难度纯函数比较是否真的读到了 _grade_difficulty 的值、_grade_difficulty 失败降级时相对比较会不会 NaN/异常; ③ per-call model 覆盖 model=None 时是否真的 100% 等价旧行为(回归风险); ④ DIFFICULTY_CAP 5→4 是否漏改某处 md+1 算术导致越界。`,
  },
  {
    key: '帧序与嫁接徽章',
    focus: `专挑 P2b 逐题上屏的帧序 bug(一期对抗 review 抓出过2个 critical: eager 裸下标嫁接错徽章、补一道缺失): ① 无tier帧→带tier帧→定稿帧 原位重发是否按 seq 正确对位, 会不会把 A 题的 tier 嫁接到 B 题(裸下标/顺序假设); ② 中转熔断从头重流时 stale 检测 + stem 同一性 merge 是否真保留, 重流后会不会重复发帧/串题; ③ 剔除题(真fail)补一道后 seq/persisted 簿记是否错位; ④ P9 排序重排后 persisted 簿记是否跟错题。`,
  },
  {
    key: '契约与净化',
    focus: `专挑契约/净化/降级路径: ① _QTYPE_CONTRACT 结构 lint 的降级路径是否真不卡死(解析不了标⚠继续); ② _PAYLOAD_CONTRACT 新 kind 名是否与 math_verify 的 _HANDLERS 注册名完全一致(不一致→4b 载荷全 degrade); ③ _solve_one 回写 solution 是否仍过 _sanitize_rich_text(最易漏的净化路径); ④ reorder op 越界/混类是否真走 clarify 不默认重排; ⑤ 编辑轮"只判不 rework"是否真保住老师 note。`,
  },
]

const reviews = await parallel(REVIEW_DIMS.map(d => () =>
  agent(`${CTX}

你是对抗 review agent(维度: ${d.key})。**只读不改**, 专职挑刺——别说"看起来对", 默认假设有 bug 直到证明没有。
本批改动文件: ${changedList || '(见 git diff)'}。
用 git diff(cwd=${TK} 与 ${UI})看二期改动, 重点审查:
${d.focus}

对每个疑点: 给出 文件:行号 + root cause 级描述 + 是否 critical(会导致功能错/数据错/卡死) + 修复方向。没有真问题就明说"该维度未发现 critical"。`,
    {
      phase: '对抗review',
      label: `review:${d.key}`,
      schema: {
        type: 'object',
        additionalProperties: false,
        required: ['dimension', 'findings', 'has_critical'],
        properties: {
          dimension: { type: 'string' },
          has_critical: { type: 'boolean' },
          findings: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              required: ['where', 'issue', 'severity', 'fix'],
              properties: {
                where: { type: 'string' },
                issue: { type: 'string' },
                severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
                fix: { type: 'string' },
              },
            },
          },
        },
      },
    }))
)

return {
  implement: { fe: feRes, sympy: sympyRes, spine: spineRes, bakeoff: bakeRes },
  pytest: pytestRes,
  reviews: reviews.filter(Boolean),
  changed_files: changedList,
}
