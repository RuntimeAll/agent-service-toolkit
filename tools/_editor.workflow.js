export const meta = {
  name: 'qgroup-editor',
  description: '题组编辑器（插队批次）：拖动排序 + 内容编辑 + 重新验算按钮 + 规范排版。FE 为主(book-ui) + 3 个零/单题 BE 直连端点(toolkit)，各自 commit',
  phases: [{ title: '编辑器', detail: 'BE 端点 ‖ FE 编辑器，契约先定死' }],
}

const TK = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\_learn-langgraph\\\\agent-service-toolkit'
const UI = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\book-ui'

// 🔴 共享 API 契约（BE/FE 都按这个，先定死好并行）
const CONTRACT = `
# 题组编辑器 API 契约（3 个端点，照 /variant/persist 直连模式：aget_state→改→aupdate_state(as_node)→返回 _artifact_payload）
所有端点 cwd=service.py(@router.post)，body 带 thread_id（会话键），返回 {ok:true, artifact:<_artifact_payload>}；不入 RuoYi、无需 ruoyi_token（题组是 toolkit 会话状态，编辑不落库，"全部入库"仍走既有 /variant/persist）。

1. POST /variant/reorder  body {thread_id, order:int[]}
   - order = 1-based 全排列(长度=当前题数，每号恰一次)；非法(给不全/重复/越界) → 400 或 {ok:false,error}。
   - 纯代码重排 + seq 重编 + persisted/check/gene 等 item 字段跟题走不错位（复用/抽取 exec_reorder 的纯重排逻辑为公共 _reorder_items(items,order)，让 exec_reorder 与本端点共用）。零 LLM。

2. POST /variant/edit-item  body {thread_id, index:int(1-based), stem?:str, answer?:str, solution?:str}
   - 只 patch 传入的字段(其余不动)；写回前必过 _sanitize_rich_text(净化 \\( \\[ / 字面\\n，复用现有)。
   - 标该 item 为手动编辑：item['manual_edited']=True + item['from_edit']=True(复用既有"老师意志优先"语义)；check 徽章置中性——set item['check']={'tier':'manual'}(或等价)表示"手动编辑、验算待重跑"，清掉旧 verify/tier 误导。零 LLM。
   - manual_edited/from_edit 不外漏入库(字段白名单已挡，确认 build_create_bo/_artifact_payload 不透传新键；artifact 需要透传 tier='manual' 给 FE 渲染)。

3. POST /variant/reverify  body {thread_id, index:int(1-based)}
   - 对该 item 重跑闸B(_check_one_item 同款路径：_solve_one + _machine_verify → 回写 check{badge,solved_answer,verify,tier})，更新该题徽章。这是单题、用 LLM+sympy，on-demand(不是零 LLM，但只跑一题)。
   - 复用 solve_explain 节点里单题验算的同一函数，别另写一套判决（判决仍只读 sympy verdict，铁律不破）。_solve_one 回写 solution 必过 _sanitize_rich_text。
   - 跑完清掉 manual_edited 的"待验算"语义(tier 变成真实验算结果 verified/self_ok/both_low/silent 按 4d 矩阵)。

刷新恢复：FE 走现成 POST /variant/artifact 重建（已含全部 item 字段 + seq + tier）。
`.trim()

const REPORT = {
  type: 'object', additionalProperties: false,
  required: ['summary', 'files_changed', 'verify', 'committed', 'open_risks'],
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    verify: { type: 'string', description: '离线测试/构建结果' },
    committed: { type: 'string', description: 'commit hash + message 首行，或"未提交+原因"' },
    open_risks: { type: 'array', items: { type: 'string' } },
  },
}

phase('编辑器')

const BE_PROMPT = `你是题组编辑器的 **BE 端点** agent（toolkit）。只动 ${TK} 下文件（src/service/service.py + src/agents/variant.py + tests）。分支 prd-c-009-variant，当前已提交 ca96790。

${CONTRACT}

# 现状参考(已勘察, 行号需 Grep 重定位)
- /variant/persist 直连模板 = src/service/service.py ~424(aget_state→persist_to_bank→aupdate_state(cfg,update,as_node=...)→返回 _artifact_payload)。/variant/artifact ~458(无 token, 重建)。照这两个写新端点。
- exec_reorder = src/agents/variant.py ~3132(graph 节点, 读 state pending ops 的 order, 纯代码重排+seq重编, 不过 assemble)。抽出公共纯函数 _reorder_items(items, order)->items 让它和 /variant/reorder 共用。
- _check_one_item = variant.py 闸B 单题(~1993, _solve_one + _machine_verify → check{badge,verify,tier})。/variant/reverify 复用它(注意它要 facts/config 上下文, 照 solve_explain 节点里对单题的调用方式构造)。
- _artifact_payload / _sanitize_rich_text / 字段白名单都在 variant.py。

# 要求
- 3 个端点(reorder 零LLM / edit-item 零LLM / reverify 单题LLM+sympy)，输入校验(reorder 非法全排列→拒；index 越界→拒)。
- 入参 pydantic model 加在 service.py(照 VariantPersistInput)。
- 离线测试：给 _reorder_items 纯函数 + edit-item 的净化/标记 + reverify 的回写 加单测(reverify 可 mock LLM)。端点层可加轻量测试(TestClient)。跑 \`PYTHONUTF8=1 .venv\\\\Scripts\\\\python.exe -m pytest tests -q --ignore=tests/app --ignore=tests/service\`，不得低于 440 passed/2 skipped(基线已含 Phase0)。
- 🔴 别重启 :8093(主 session 统一重启验真机)；别 commit FE；别动 RuoYi。
- 跑绿后 **commit**(只 add src tests，中文 message)：\`git add src tests && git commit\`，message 首行 \`feat(变式): 题组编辑器 BE 端点(reorder/edit-item/reverify 直连)\`，body 列三端点契约 + 复用点，结尾 \`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\`。
- 按 schema 返回(committed 填 hash+首行)。open_risks 给 FE 交代最终端点路径/字段名（若与契约有出入）。`

const FE_PROMPT = `你是题组编辑器的 **FE** agent（book-ui，Vue3）。只动 ${UI} 下 src/views/variant/* + src/api/variant/index.ts。分支 master-ai，当前已提交 a1421f3。

${CONTRACT}

# 现状(已勘察)
- 依赖已有 sortablejs ^1.15.7(+@types) / element-plus / katex / markdown-it / 组件 MarkdownMath(渲染 stem/answer/solution 带 KaTeX)。**不引新依赖**。
- ArtifactPanel.vue：按 seq 原位 merge 渲染卡片(Phase0 已改)；VariantCard.vue：stem 经 MarkdownMath 渲染，**选项内嵌在 item.stem markdown 里(非独立字段)**；verifyBadge computed 按 item.tier 渲染(verified/self_ok/proof/both_low/silent/checking)。api/variant/index.ts：streamVariant + fetchVariantArtifact + persist 等，VariantArtifactItem 类型含 stem/answer/solution/qtype/difficulty/tier/seq。
- 题组顶部已有"全部入库"按钮(走 /variant/persist)，不要动它的语义。

# 要做的三件(傻瓜式可视化 + 持久化到 BE state)
1. **拖动排序**：ArtifactPanel 卡片用 sortablejs 可拖拽(拖手柄, 移动端友好)；drop 后取新 order(1-based 全排列)调 POST /variant/reorder → 用返回 artifact 刷新；拖动中乐观更新、失败回滚。
2. **内容编辑**：VariantCard 加"编辑"切换；编辑态把 stem/answer/solution 各显示为 textarea(或 el-input type=textarea) + **右侧/下方 MarkdownMath 实时预览**(KaTeX 立等可见)；"保存"→ POST /variant/edit-item(只传改过的字段)→ 用返回 artifact 刷新；"取消"还原。保存后该卡 tier='manual' → verifyBadge 显示中性"手动编辑"徽章(新增该分支)。
3. **重新验算按钮**：手动编辑过(tier='manual")的卡显示"重新验算"按钮 → POST /variant/reverify(带 loading 态, 单题可能几秒) → 返回 artifact 刷新, 徽章变真实验算结果。
4. **规范排版(格式)**：编辑态加"规范排版"按钮 —— FE 端对 stem 文本做行级 normalize(选择题每个选项独立成行/答案选项对齐; 填空题空位标准化为统一下划线; 判断题题干尾补"（  ）")；normalize 后填回编辑框(老师可再改)，保存仍走 edit-item。**纯 FE 文本处理, 解析不动选项语义**(只调换行/下划线/补括号), 解析不出就原样返回不报错(降级)。
- **持久化**：所有编辑都经 BE 端点落 toolkit 会话 state；刷新走现成 fetchVariantArtifact 重建即在。localStorage 存"未保存草稿"(可选增强, 编辑中途刷新不丢)。
- api/variant/index.ts 加 reorderVariant/editVariantItem/reverifyVariantItem 三个调用 + VariantArtifactItem 补 manual_edited?/tier 'manual' 类型。

# 要求
- 类型自洽；**自己不跑 pnpm build 也行但建议跑一次**(cwd=${UI} \`pnpm build\`, vue-tsc 门禁)确认过。
- 跑绿后 **commit**(只 add src，中文 message)：message 首行 \`feat(变式FE): 题组编辑器(拖动排序/内容编辑/重新验算/规范排版)\`，body 列四件 + 持久化走 BE 端点，结尾 \`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\`。
- 按 schema 返回(committed 填 hash+首行 或 "未提交+原因")。open_risks 列真机待验点(拖拽 drop 顺序/reverify loading/规范排版边角)。`

const [beRes, feRes] = await parallel([
  () => agent(BE_PROMPT, { phase: '编辑器', label: 'BE:3端点', schema: REPORT }),
  () => agent(FE_PROMPT, { phase: '编辑器', label: 'FE:编辑器', schema: REPORT }),
])

return { be: beRes, fe: feRes }
