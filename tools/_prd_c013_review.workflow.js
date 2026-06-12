export const meta = {
  name: 'prd-c-013-review',
  description: 'PRD-C-013 Phase0 对抗 review：3 维并行挑刺(脊柱耦合/帧序嫁接徽章/契约净化) + critical 汇总',
  phases: [{ title: '对抗review', detail: '只读挑刺, 默认假设有 bug' }],
}

const TK = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\_learn-langgraph\\\\agent-service-toolkit'
const UI = 'D:\\\\workplace\\\\book-ai\\\\codeplace-C\\\\book-ui'

const CTX = `
# C 线·举一反三 agent · PRD-C-013 二期改动对抗 review
- toolkit 根: ${TK}(分支 prd-c-009-variant, :8093), book-ui 根: ${UI}(分支 master-ai, :8091)。改动均 uncommitted, 用 git diff 看。
- 主战场单文件 src/agents/variant.py(+705)。二期改动: P11 题型契约(_QTYPE_CONTRACT+structure_lint) / P12 闸A 重构(删 difficulty_match, judge 不截断, rework 带母题骨架, 编辑轮 from_edit 只判不回炉, 转题型按目标题型规范判) / P13 预算闸(state.llm_call_budget, 出题18/编辑6, 超限跳增强类调用) / P14 叙事 / P2b 逐题上屏(BE 无tier帧→带tier帧→定稿帧, FE 按 seq 原位 merge) / P8 难度总评(_grade_difficulty nano, DIFFICULTY_CAP 5→4) / P9 排序(_sort_by_difficulty 升序 + reorder op + exec_reorder) / nano(per-call model 覆盖, parse 切 nano) / 4b(math_verify inequality_solve/rational_roots/应用题) / P10(FE 图进对话流)。
- 🔴 铁律: 宏观 DAG 确定性(预算闸只跳增强调用非改流程); _solve_one 回写必过 _sanitize_rich_text; _LITERAL_NL_RE=r"\\\\n(?![a-z])"(\\\\ne 是合法 LaTeX); parse 无题组语境注记须压过硬守恒段(17号); reorder 越界/给不全→clarify。
- 离线全绿: 419 passed/2 skipped。你**只读不改**, 专职挑刺——别说"看起来对", 默认假设有 bug 直到证明没有。
`.trim()

const FIND_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'has_critical', 'findings'],
  properties: {
    dimension: { type: 'string' },
    has_critical: { type: 'boolean' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['where', 'issue', 'severity', 'fix'],
        properties: {
          where: { type: 'string', description: '文件:行号' },
          issue: { type: 'string', description: 'root cause 级描述' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          fix: { type: 'string' },
        },
      },
    },
  },
}

const DIMS = [
  { key: '脊柱耦合逻辑', focus: `专挑 variant.py 二期耦合 bug: ① 预算闸记账是否把 P8 难度总评/nano parse 这类核心调用误算进"增强类"被跳过(应只跳 rework/heal/replenish/extract); contextvar 与 state.llm_call_budget 双层记账是否会重复计数或漏算; ② difficulty_match 删除后, difficulty_consistency_defects 组内相对比较是否真读到 _grade_difficulty 覆盖后的值, _grade_difficulty 失败降级(保留原值)时相对比较会不会 NaN/KeyError; ③ per-call model 覆盖 model=None 时是否 100% 等价旧行为(relay_pool._chat_override 缓存键是否污染整站); ④ DIFFICULTY_CAP 5→4 是否漏改某处 md+1 算术致越界; ⑤ assemble 现含一次 nano call, 排序 _sort_by_difficulty 是否在 _grade_difficulty 之后消费新值。` },
  { key: '帧序与嫁接徽章', focus: `专挑 P2b 帧序(一期抓出过2 critical: eager 裸下标嫁接错徽章/补一道缺失): ① 无tier帧→带tier帧→定稿帧原位重发是否按 seq 正确对位, 会不会把 A 题 tier 嫁接到 B 题(裸下标/顺序假设); ② 中转熔断从头重流时 stale 检测(_gen_progress len 下降取消 stale eager)+stem 同一性 merge 是否真保留, 重流后会不会重复发帧/串题; ③ 剔除题(真fail)补一道后 seq/persisted 簿记是否错位; ④ P9 reorder/默认排序后 persisted/seq 簿记是否跟错题; ⑤ FE ArtifactPanel 按 seq upsert: BE 不带 seq 时 index 回退是否与一期等价、_dropped 退场过渡是否漏帧。` },
  { key: '契约与净化', focus: `专挑契约/净化/降级: ① structure_lint 降级路径是否真不卡死(解析不了标⚠继续, 回炉仍不过只标 warn 不剔除); ② _PAYLOAD_CONTRACT 新 kind 名(inequality_solve/rational_roots)是否与 math_verify._HANDLERS 注册名完全一致(不一致→4b 载荷全 degrade); ③ _solve_one 回写 solution 是否仍过 _sanitize_rich_text; from_edit/edit_note/seq/_dropped/from_recipe 等新 item 字段是否真不外漏入库/artifact(字段白名单); ④ reorder op R8 全排列护栏: 越界/重复/给不全是否真走 clarify 不默认重排; ⑤ 编辑轮"只判不 rework"是否真保住老师 edit_note 不被失败原因冲掉。` },
]

phase('对抗review')
const reviews = await parallel(DIMS.map(d => () =>
  agent(`${CTX}\n\n你是对抗 review agent(维度: ${d.key})。本批改动文件见两 repo git diff。重点审查:\n${d.focus}\n\n对每个疑点给 文件:行号 + root cause 级描述 + severity(critical=功能错/数据错/卡死, major, minor) + 修复方向。没有真问题就 has_critical=false 并说明该维度已审无 critical。`,
    { phase: '对抗review', label: `review:${d.key}`, schema: FIND_SCHEMA })))

const all = reviews.filter(Boolean)
const crits = all.flatMap(r => (r.findings || []).filter(f => f.severity === 'critical'))
return { reviews: all, critical_count: crits.length, criticals: crits }
