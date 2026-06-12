export const meta = {
  name: 'qtype-formatter-editor',
  description: '题型模版自动规范(BE formatter) + 结构化编辑器(FE 字段+智能输入)。各自 commit',
  phases: [{ title: '模版编辑器', detail: 'BE formatter ‖ FE 结构化编辑' }],
}

const TK = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\_learn-langgraph\\\\agent-service-toolkit'
const UI = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\book-ui'

// 🔴 canonical 格式 SSOT = book-ui/src/views/variant/normalize.ts（已存在，编辑器批次产出）
const CONTRACT = `
# 题型规范格式 SSOT = ${UI}\\\\src\\\\views\\\\variant\\\\normalize.ts（normalizeChoice/normalizeBlanks/normalizeJudge）
两轨都以它为准，BE formatter 与 FE 结构化编辑器产出/解析的 canonical 题干格式必须与它一致（否则 BE 规范 ↔ FE 解析对不上）：
- 选择题：题干以「（ ）」收尾；选项 **每个独立成行** "A. xxx\\nB. xxx\\nC. xxx\\nD. xxx"（接在题干后空一行）；answer=选项字母。**存储一行一项**，2×2 是 FE 渲染层的事（不写进存储文本）。
- 填空题：空位统一为「____」（4 下划线）。
- 判断题：题干尾补「（  ）」。
- 解答题：不强排，保留多小问。
- 🔴 规范是**纯排版（cosmetic）**：只动换行/下划线/补括号/选项分行，**绝不改数学语义、绝不动 \$...\$ 内的 LaTeX**；解析不出 → 原样返回不报错（降级 G5）。
`.trim()

const REPORT = {
  type: 'object', additionalProperties: false,
  required: ['summary', 'files_changed', 'verify', 'committed', 'open_risks'],
  properties: {
    summary: { type: 'string' }, files_changed: { type: 'array', items: { type: 'string' } },
    verify: { type: 'string' }, committed: { type: 'string' },
    open_risks: { type: 'array', items: { type: 'string' } },
  },
}

phase('模版编辑器')

const BE_PROMPT = `你是「题型模版自动规范」的 BE agent（toolkit）。只动 ${TK} 下 src/ + tests。分支 prd-c-009-variant（编辑器批次已提交，最新 22a9745）。

${CONTRACT}

# 任务
1. 新增 Python 纯函数 \`format_by_qtype(stem: str, qtype) -> str\`（放 variant.py 或新 src/agents/qtype_format.py），**镜像 normalize.ts 的 normalizeChoice/normalizeBlanks/normalizeJudge 规则**（先 Read 那个文件，逐条对齐：选项一行一项+题干（ ）收尾 / 填空 ____ / 判断补（  ））。cosmetic-only、幂等、解析不出原样返回不抛。
2. **自动应用**（用户要"数据进来就自动规范"）：在题目内容定稿后、产出 artifact / 入库前，对 stem 跑一遍 format_by_qtype，规范文本落进 item（这样上屏 + 入库 + 会话恢复都吃规范文本）。
   - 🔴 落点要在 sympy/structure_lint **之后**（规范是排版、可能动选项行序/下划线，绝不能影响判决；判决先跑、规范后做）。建议在 _check_one_item 末尾（item 定稿、_apply_visibility 之后）或 assemble 收口处统一规范；编辑器 edit-item/reverify 端点回写后也顺手规范。选一个**覆盖所有路径又不重复打架**的点，在 summary 说明你的落点选择与理由。
   - 幂等：已规范的文本再跑一次结果不变（防多次应用漂移）。
3. 单测：format_by_qtype 各题型 ≥3 例（选择选项分行 / 填空下划线 / 判断补括号 / 幂等 / LaTeX 不被破坏 / 解析不出原样）。跑 \`PYTHONUTF8=1 .venv\\\\Scripts\\\\python.exe -m pytest tests -q --ignore=tests/app --ignore=tests/service\` 不低于 457 passed/2 skipped。
- 🔴 别重启 :8093、别 commit FE、别动 RuoYi。跑绿后 commit（git add src tests，中文 message 首行 \`feat(变式): 题型模版自动规范 formatter(BE 纯代码,生成/上屏/入库吃规范文本)\`，body 列落点+镜像 normalize.ts+幂等，结尾 Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>）。
- 按 schema 返回。open_risks 给 FE 交代最终 canonical 选择题格式（确认与 normalize.ts 一致）。`

const FE_PROMPT = `你是「结构化编辑器」的 FE agent（book-ui）。只动 ${UI} 下 src/views/variant/* + src/api/variant/*。分支 master-ai（编辑器批次已提交 07b5d9f）。

${CONTRACT}

# 背景：老师反馈编辑框是 raw LaTeX（A. \$\\frac{2}{5}\$）太难懂。改成**结构化字段 + 智能输入**。
现状：VariantCard.vue 编辑态是每字段一个 textarea（stem/answer/solution）+ MarkdownMath 预览 + normalize.ts 的「规范排版」按钮。选项内嵌在 stem markdown 里。

# 任务（按 qtype 拆字段，各带实时预览）
1. **选择题编辑态**：把 stem 解析成 题干 + 选项A-D（用 normalize.ts 的 canonical 格式解析，一行一项）+ 答案下拉(A/B/C/D)。渲染 = 题干 textarea + 4 个选项小输入框 + 答案 select，**每个框旁实时 MarkdownMath 预览**。保存时把字段**回拼成 canonical stem**（题干（ ）+ 空行 + A.\\nB.\\nC.\\nD.，与 normalize.ts/BE 一致）+ answer，走 edit-item。
2. **填空题**：题干 textarea（含 ____）+ 答案框。**判断题**：题干 + 答案下拉(对/错)。**解答题**：题干/答案/解析 textarea（维持现状，多小问不拆）。
3. **智能输入**（降 LaTeX 门槛，核心诉求）：字段内常见数学自动转 LaTeX —— "2/5"→\`\$\\frac{2}{5}\$\`、"x^2"→\`\$x^{2}\$\`、"sqrt(3)"/"sqrt3"→\`\$\\sqrt{3}\$\`、"<=>=!="→\`\\le \\ge \\ne\`。做成一个纯函数 smartMath(text)，老师在框里敲自然写法、失焦/输入时转成 LaTeX，预览立等可见。已是 \$...\$ 的不重复包。转不了的原样保留（不破坏已有 LaTeX）。
4. 选择题渲染一致性（顺手修用户反馈的"排版有点问题"）：存储一行一项，卡片展示按选项长度自适应（短则 2×2、长则每项一行），编辑框与预览不再不一致。
- 不引新依赖（用现有 element-plus/MarkdownMath/katex）。建议 cwd=${UI} 跑一次 \`pnpm build\`（vue-tsc 门禁）确认过。
- 跑绿后 commit（git add src，中文 message 首行 \`feat(变式FE): 结构化编辑器(题型拆字段+智能输入 raw LaTeX 免手写)\`，body 列四点，结尾 Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>）。
- 按 schema 返回。open_risks 列真机待验点（选项解析边角/智能输入误转/答案下拉）。`

const [beRes, feRes] = await parallel([
  () => agent(BE_PROMPT, { phase: '模版编辑器', label: 'BE:formatter', schema: REPORT }),
  () => agent(FE_PROMPT, { phase: '模版编辑器', label: 'FE:结构化编辑', schema: REPORT }),
])
return { be: beRes, fe: feRes }
