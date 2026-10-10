---
id: agent-search-components
title: "Search Agent从检索到证据回答的组件流程"
category: agent
difficulty: 进阶
tags: ["P1", "Agent", "Search Agent", "Retrieval", "Browser", "Evidence"]
updated: "2026-10-10"
summary: "区分search、fetch、scrape、crawl、browser和RAG，解释查询规划、证据存储、引用验证、恢复和成本。"
draft: false
---

## 1. 问题背景 找到网页不等于回答有依据

Search agent 的任务不是尽可能调用搜索，而是解决一个带约束的信息问题。例如“某工具在版本A、日期B是否支持保存运行中的解释器状态”，需要分别核对功能、版本、状态对象和日期；一个搜索摘要可能只写“支持持久化”，不足以判断是不是仅保存文件。

模型可以决定下一条 query、先读哪个来源、是否需要浏览器、哪里有冲突以及是否已能回答。工具负责返回候选和内容，证据层保存可追溯片段，验证层检查来源是否支持断言。这个过程可包含固定检索步骤，也可动态迭代；自主程度与工具数量没有直接等号。[Agent/workflow](#q=agent-workflow)。

贯穿例子是核查“某服务的快照在指定版本是否保留RAM、进程和外部连接”。这些对象必须拆开，不能把“保存状态”一个短语扩展成三条未经核验的结论。下面的 evidence 示例均为教学对象，真实产品差异按官方文档读取。

## 2. 前置知识 搜索正文解析浏览器与证据

**Query** 是查找条件；**search result** 是候选页面及相关性摘要；**fetch** 按URL取响应；**scrape/extract** 抽取某页内容或结构化字段；**crawl** 从起点按范围/规则遍历页面。一次 API 可以组合多种能力，但讲流程时仍要分清哪段产生了候选，哪段实际读到了全文。

**Browser** 执行页面脚本、维护DOM、cookie和交互；**CDP** 是与 Chromium 等浏览器调试能力交互的协议之一；**Playwright** 提供浏览器自动化抽象。远程 browser provider 提供会话，客户端库执行动作；endpoint 连接不等于某页内容已经提取。普通 fetch 的HTML壳可能没有实际正文，browser中的DOM文本和截图又是不同观察。

**Chunk** 是文档片段，**embedding** 把内容转成用于相似度等操作的向量，**index** 组织检索，**reranker** 对候选再排序，**RAG** 把检索内容接到生成。BM25 等词面方法适合明确名字/短语，向量方法可召回语义相近内容；都不直接给出真实性证明。

**Evidence** 是对某一断言有支持或反驳意义的来源片段；**citation** 是最终答案指向证据的引用。相关性 score、provider 的自动 answer、多个转载页面都不能代替独立的支持检查。首先明确“要证明什么”，再决定哪些工具值得调用。

## 3. 完整流程 Query到可追溯回答

![本文示意 Search Agent发现正文证据与回答验证](./assets/agent/agent-search-components/search-flow.svg)

**图解。** 左侧拆断言和时间条件，主线搜索候选→取正文→解析和去重→证据账本→综合→验证。脚本页面可绕到 browser 再回正文路径；缺口或冲突回到新 query。引用由 claim/evidence 关系生成，trace记录每段。图不把 search provider 返回的 answer 直接当作经过核验的最终答案。

### 先建立待核问题而不是无限搜索

把任务拆成条件明确的原子断言，例如“指定版本支持full snapshot”“full恢复包含进程内存”“外部连接可自动恢复”。每个断言记录目标对象、日期范围、优先来源及已知缺口。这样搜索失败可以定位到没有找到功能定义，而不是笼统地“还要多搜一点”。

query 可用全称、版本、准确错误或官方域名逐步收窄。第一次查询偏宽用于发现词汇和官方入口，后续查询针对缺口。需要排除同名项目时加入组织/仓库/产品域名。查询分解是任务规划，query rewrite只是改表达；不要把改变问题范围伪装成改写。

### 搜索找到入口然后读正文

搜索结果的 snippet 帮助排序入口，但可能截断限定条件或来自旧索引。fetch/scrape 读取当前页面，保存最终 URL、获取时间和可用内容；遇到重定向、404、登录或脚本壳，记录状态并选择适当替代来源。抓取失败意味着这次没有取得内容，不说明该功能不存在。

对版本问题，优先固定 release 文档、仓库 commit、原论文版本或有明确日期的官方说明。发布日期、更新时间、你获取的时间和搜索索引时间是不同字段；不能因为刚获取网页，就说其结论是最新发布。当前网页也不能自动证明过去某日期的状态。

### 解析去重与证据选择

删除导航/广告但保留标题、段落、表头、脚注、代码和版本说明；PDF/表格要确认列与单位，不能把相邻段的限定条件丢掉。parser 输出空文本时尝试适合的读取方式，不把空结果交给模型补事实。

URL 规范化、重定向、文本hash和来源关系帮助去重。两个不同URL若转载同一稿，只算同一证据链；同一域名的不同版本页面则不能随意合并。按问题保留相关片段及上下文，而不是只存 embedding，后者不能直接恢复可引用的原文。

## 4. 组件选择 同一层使用相同维度比较

| 组件/能力 | 接口角色 | 适合回答什么 | 容易误解什么 |
| --- | --- | --- | --- |
| Tavily Search | 查询与结果 | 候选URL、内容、域名/时间等条件 | score不是事实可信概率 |
| Exa search/contents | 查找与读取 | 找页面并取相应内容 | 使用“语义”不等于真实或完整 |
| Firecrawl scrape/crawl/search | 页面获取、遍历和搜索 | 把网页转成便于消费的内容 | crawl不是无限互联网检索 |
| HTTP client + parser | 按URL获取和解析 | 已知页面、静态正文 | 200可能只是脚本壳或错误页 |
| Playwright / remote browser | 有状态交互 | JavaScript、页面操作、会话 | 浏览器不是天然全文索引 |
| BM25 / vector index / reranker | 本地语料召回与排序 | 已保存文档、企业资料 | 高相似度不证明支持断言 |
| Evidence store + citation validator | 原文与结论关系 | 可追溯、冲突、覆盖和核查 | 有URL不等于引用有效 |

原始入口为[Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)、[Exa search](https://docs.exa.ai/reference/search)、[Firecrawl introduction](https://docs.firecrawl.dev/introduction)、[Playwright Python](https://playwright.dev/python/docs/intro)。API的组合能力会变化，选型要看所需结果、内容长度、过滤、失败状态、负载及预算；不依据产品名字断言哪个搜索质量最高。

### 一段真实API接线能说明什么

```python title="Tavily官方API字段的教学调用"
import os
from tavily import TavilyClient

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
candidates = []
response = client.search(
    query="product full snapshot process state official docs",
    max_results=5,
    include_raw_content=True,
)
for item in response["results"]:
    candidates.append({
        "url": item["url"],
        "snippet": item["content"],
        "body": item.get("raw_content"),
    })
```

此处 candidates 是调用者的候选集合；这是 API 使用示意，本次没有调用收费搜索服务。返回的 raw_content 可能为空，不能用 snippet 自动补成“已核正文”。include_answer若返回自动总结，也应该视为另一个生成结果；按原始支持片段核查，而不是给 provider 的总结加一个链接就结束。

### Browser session和下载对象

Browser session 持有页面、cookie、tab和登录状态；多个无关任务不应默认共用同一会话。页面动作需记录当前URL、定位对象、等待条件和结果；图像适合观察布局，DOM适合文本/元素定位，PDF或下载文件则进入独立解析流程。刷新、打开新tab和重建沙箱都可能改变状态。

当内容需要交互，browser可在E2B/ACS或远程服务中运行；这时要同时管理浏览器会话与底层 sandbox。重连一个旧 endpoint 不保证旧页面仍在，也不保证认证/session仍有效。浏览器集群排队和网页限流是工具侧耗时，不是模型推理时间。

## 5. 证据对象与引用怎样设计

```json title="教学数据结构 保存原文并绑定具体断言"
{
  "source_id": "S3",
  "requested_url": "https://docs.example.org/snapshots",
  "final_url": "https://docs.example.org/v2/snapshots",
  "retrieved_at": "2026-10-10T08:00:00Z",
  "document_version": "v2",
  "content_hash": "hash-of-normalized-document",
  "locators": ["section: Full snapshot", "paragraph: 2"],
  "passage": "The example full snapshot preserves process memory.",
  "supported_claim_ids": ["C1"],
  "contradicted_claim_ids": [],
  "limitations": ["Does not establish external connection recovery"]
}
```

example.org 是教学域名，passage 是教学文本，不冒充真实厂商引用。真实系统还可保存原始响应、解析版本、原始字节hash和normalized hash。哈希用于识别内容版本，不证明页面作者可信，也不表示解析没有丢脚注。

答案中的主张要指向 source_id 加片段定位，而不只保存整页URL。第一条证据说明RAM持久化，并不能顺带支持“外部连接不断”“所有SDK版本可用”。遇到反例/冲突，先比较产品、版本、条件和时间；确实无法消解时写出不确定范围，不通过多数投票选择一句更顺耳的话。

**Citation validity** 检查引用是否存在、页面是否可读、片段是否支持、条件是否一致；**coverage** 检查必要断言是否有足够支持；**correctness** 还需检查结论本身。若4个必要断言只有3个有证据，覆盖率是3/4；若3个支持都来自同一错误转载，覆盖高仍可能错。

```python title="教学伪代码 先暴露缺口再生成结论"
for claim in required_claims:
    matches = evidence_store.for_claim(claim.id)
    supported = check_passages(matches, claim.conditions)
    ledger[claim.id] = supported
unresolved = [key for key, value in ledger.items() if not value]
if unresolved and budget_available:
    plan_targeted_queries(unresolved)
else:
    render_answer_with_supported_claims(ledger, unresolved)
```

check_passages不能只看关键词，语义上还要核否定、范围、版本与日期。结构化程序能检查ID、hash和时间，但复杂语义需要人工或经校准的judge。本系列的CPU练习只演示预先标注支持关系的统计，不伪装自动证明任意网页真伪。

## 6. RAG记忆缓存和状态恢复

**Search** 可以访问当前外部索引；**RAG** 可以用已有语料库召回；两者可组合。外部API返回新页面后，保存内容供下一轮检索可减少重复获取；缓存键要含query、过滤条件、provider/版本与有效时间。缓存命中只是复用旧结果，不代表得到当前最新内容。

证据库与长期memory也有区别：证据库记录可核原文、版本和claim关系；memory可能存用户偏好或过去推断。把未经验证的摘要存为“事实”，下一轮再从memory取出来引用，会让错误循环自证。保存summary时保留source IDs与限定条件，必要时回查原文。

**Checkpoint** 记录待核断言、已搜索query、工具状态、证据引用与预算。恢复时先确认 evidence documents 可用，再找回browser/sandbox session；外部页面可能变化，应按任务时间要求决定复用或重新获取。删除环境后，已导出的证据可仍可用，cookie与运行中的browser却未必可恢复。

**输入材料是数据。** 网页中出现“忽略指令”或要求调用其他服务，不是用户任务授权。MCP resources、网页正文、PDF注释和搜索摘要都需要保持来源边界；控制层决定工具能做什么，内容层只提供证据。[MCP tools规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)的提示字段也不是执行隔离。

## 7. 成本停止条件与评测

总预算可按模型tokens/调用、搜索请求、取页量、浏览器时长和总wall time分别设限。假设一个问题search耗时1秒、3页fetch各2秒且可并行、模型两次各1秒，理想串行主链约1+2+2=5秒；不考虑调度、限流和重试。若逐页串行读，变成1+6+2=9秒。它说明独立取页的并行机会，不是任何产品实测。

只有独立请求适合并行：对不同公开静态页fetch可并行，同一browser先点击再读结果需有序。大量扩query可能制造重复证据并消耗上下文；工具返回过长还会提高模型处理成本。优化可看证据有效率、重复率、缺口解决率，而不是仅看“每秒搜多少次”。

停止规则结合必要断言支持、冲突、时效、覆盖与预算。模型说“资料足够”必须能在账本里对应证据；预算耗尽时给出已核部分与未解决项。对高动态问题，检索时间和文档条件是答案的一部分，不能用未来任务的缓存延伸当前结论。

参考[Agent评测文章](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)设计多层grader：

| 评分维度 | 要检查的对象 | 常见失败 |
| --- | --- | --- |
| 答案正确性 | 结论与任务真值/专家判断 | 文档理解错、遗漏否定 |
| 引用支持 | 每条断言与具体片段 | URL相关但不支持 |
| 覆盖和冲突 | 必要条件与相反证据 | 只回答容易部分 |
| 时效与版本 | 日期、产品、发布范围 | 当前页面回答历史问题 |
| 行动效率 | query/读取/模型/浏览器预算 | 重复搜索、无效crawl |
| 可恢复性 | checkpoint与证据/session | 消息已恢复但页面状态丢失 |

用固定问题集、明确日期和资源预算比较provider或harness；不要用不同任务的广告吞吐推导质量。搜索grader自身也需校准，特别是judge是否只检查“看起来像有引用”。

## 8. 面试问答

**Q01 [层次] Search scrape crawl browser怎么区别？** Search找候选，scrape读单页，crawl按范围遍历，browser执行交互和脚本；一个产品可组合这些能力，但数据流仍要分开。

**Q02 [证据] Snippet为什么不够？** 可能截断否定、版本和限制，且索引可能过时。用它发现来源，再读支持断言的正文片段。

**Q03 [时效] retrieved_at等于文档发布日期吗？** 不等于；分别记录发布/更新、获取时间和版本，当前页面不能直接证明历史状态。

**Q04 [质量] Rerank得分高证明可信度高吗？** 通常只说明排序相关性，事实支持还要核来源与片段，转载也可能重复高分。

**Q05 [状态] Search agent为什么可能需要sandbox？** 浏览器、解析文件、运行统计代码需要执行环境；纯search/fetch的最小流程未必需要自建沙箱。

**Q06 [完成] 有引用和覆盖率高为何仍可能错？** 链接可能不支持或来源共同错误，条件与结论也可能不一致。覆盖率不等于正确性。

**Q07 [恢复] 只保存最终summary有何不足？** 丢失来源片段、条件、冲突和待核项，难以验证及继续搜索；保留证据账本和状态引用。

**Q08 [流程] 怎么两分钟介绍search agent？** 拆断言、规划query、找候选、读正文、解析去重、建立claim-evidence关系、补缺口、生成带引用结论并验证；指出每步的工具与状态。

## 9. 课堂练习与来源

[CPU脚本](./assets/agent/agent-components-lab.py)使用固定教学证据，演示3/4覆盖、重复来源不能变成独立证据，以及版本条件不匹配的片段不得支持目标断言。[结果](./assets/agent/agent-components-lab-results.json)可下载；这是契约和计数练习，没有调用收费搜索或浏览器云服务。

[来源清单](./research/agent-search-components-sources.json)登记2026-10-10核读的官方接口和原始解释。回到[组件总览](#q=agent-components-overview)，将code/search两条流程按决策、工具、环境、状态、验证与观测六层对照。
