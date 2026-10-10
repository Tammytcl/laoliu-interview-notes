---
id: agent-components-overview
title: "Code 与 Search Agent 组件总览"
category: agent
difficulty: 基础
tags: ["P0", "Agent", "Harness", "Harbor", "E2B", "ACS"]
updated: "2026-10-10"
summary: "沿任务、模型、工具、环境、状态和评测解释常见组件，建立能讲清 code/search agent 流程的术语地图。"
draft: false
---

## 1. 问题背景 为什么名字越多越难讲清流程

介绍一个 agent 时，常会同时提到模型、LangGraph、Harbor、E2B、MCP、浏览器、向量库和 tracing。它们回答的问题不同：模型决定下一步；harness 组织循环；工具访问外部能力；sandbox 提供执行环境；状态层保存进展；评测层判断任务是否真正完成。只说“用了哪些框架”，听众仍不知道一次调用发生在哪里、输入输出是什么、失败由谁处理。

本系列以两个任务贯穿：code agent 修复仓库中的边界条件错误，并交付 patch 与测试证据；search agent 核查一个带时间条件的问题，并交付有来源的结论。两者共享模型—行动—观察闭环，外部环境和完成判据不同。先把组件放进流程，再理解产品如何实现其中一层，便能解释为什么换模型、换沙箱、换评测器是三种不同改动。

本文的 Harbor 指 Terminal-Bench 团队的 harbor-framework/harbor；不是同名容器镜像仓库 Harbor，也不是其他同名 agent 产品。**ACS 在这里指阿里云容器计算服务及其 Agent Sandbox 能力**，与 E2B 一起讨论执行环境。如果看到 Agent Control Standard 等同缩写协议，必须带全称区分，不能替换成这里的产品。ACP 则是 Agent Client Protocol，是另一个词。

![本文示意 Agent各层组件怎样进入一次任务](./assets/agent/agent-components-overview/component-map.svg)

**图解。** 顶部 task 与 budget 是目标和约束；中间 harness 调用 model，再通过 dispatcher 选择工具；工具分别进入代码环境或搜索/浏览环境；返回 observation 后更新状态，verifier 最后检查产物。底部 trace 观察全链路。Harbor 可组织整次评测，E2B/ACS 位于环境供给层。图中每个框是职责，不要求部署成独立服务；箭头表示信息或控制流，不表示所有工具可以任意并行。

## 2. 前置知识 行动观察状态和完成是四件事

**Action** 是模型提出的行动，如执行测试、读取网页；**tool call** 是经过具体接口编码的调用；**observation** 是工具返回的结果，如 stdout、网页正文或错误。模型提出行动不等于执行成功，执行成功也不等于用户任务完成。

**Context** 是这一次送给模型的输入，包含指令、历史和选取的观察；**state** 是程序维护的完整任务状态，可以包含没有塞进 prompt 的文件、任务表和证据账本；**memory** 是可跨步骤或任务检索的保存信息。上下文窗口有限，外部状态不必每轮全部发送。简化时删掉冗余文本，不能删掉尚未完成的副作用状态、待验证断言或关键来源。

**Tool schema** 规定工具名称、参数和结果结构；**transport** 规定调用如何跨进程传输；**authorization** 规定谁有权做什么；**isolation** 限制代码接触的文件、进程、网络和资源。JSON 参数合法不证明有权限，MCP 连接成功不证明沙箱隔离，运行在容器里也不说明允许把生产凭据交给模型。

**Termination** 是停止循环，如预算耗尽、模型输出 final、环境故障；**success** 是通过任务判据；**reward** 是评测或训练使用的反馈。终止原因至少应区分 verified_success、verified_failure、budget_exhausted、infrastructure_error。把所有停止写成“完成”，会污染线上指标和训练数据。

## 3. 术语地图 先定位再选组件

| 层次 | 常见术语或组件 | 负责什么 | 典型输入与输出 |
| --- | --- | --- | --- |
| 任务 | task、instruction、dataset、benchmark | 定义目标、初态、约束和成功条件 | 问题/仓库/环境 → 可运行任务 |
| 决策 | policy、model、planner、router | 根据当前上下文提出下一步 | context → action/final |
| 循环 | harness、runner、orchestrator、LangGraph、AgentScope | 管理步骤、工具结果、预算、状态转移 | task/state → 多轮执行 |
| 工具接口 | function calling、dispatcher、MCP | 暴露并执行外部能力 | name/args → structured result |
| 代码执行 | shell、PTY、file API、code interpreter、E2B | 操作环境里的文件和程序 | command/code → exit/output |
| 沙箱供给 | Docker、microVM、ACS、Daytona、Modal、OpenSandbox | 创建、连接、管理和回收环境 | template/resources → sandbox handle |
| 网页信息 | search、scrape、crawl、Tavily、Exa、Firecrawl | 找候选网址、读取正文、遍历站点 | query/URL → documents |
| 浏览器 | Playwright、browser session、CDP、Browserbase | 执行有页面状态的交互 | navigation/click → DOM/screenshot |
| 状态与记忆 | checkpoint、store、workspace、snapshot | 保存程序和环境的不同状态 | state/artifacts → 可恢复记录 |
| 评测 | harness、verifier、grader、Harbor | 运行任务并判断产物 | agent output/environment → score |
| 观察 | trace、span、event、ATIF、OpenTelemetry、Langfuse | 记录因果链、耗时、成本与失败 | events → 可分析轨迹 |

表中的品牌不是同层排行榜。LangGraph 可承担状态化循环；Harbor 也被称为 evaluation harness，但主要把 task、agent、environment、verifier 组合成实验；E2B 的 SDK 可以被任一 harness 调用。选择时先问缺的是“更好的决策”，还是“能运行的环境”，还是“可比较的任务和评分”。

**Code agent 与 code interpreter 不同。** 前者会读仓库、提出修改、运行测试、修复反馈并交付；后者通常提供执行一段 Python 等代码的接口，可能保持解释器状态，但本身不规定如何解决完整软件任务。一个 search agent 也可调用解释器做表格统计，它不会因此变成完整仓库修复系统。

**Agent 与 model 不同。** 模型换了但 tool schema、prompt、预算、上下文压缩和恢复规则不变，才比较接近单独测模型；若同时换整套 CLI，结果还包含 harness 差异。[Agent/workflow 基础](#q=agent-workflow)进一步讨论由谁决定下一步。

## 4. 一次任务怎样流经这些组件

### Code agent 从 issue 到经过验证的 patch

用户给出“空数组输入时函数应返回 0”。task loader 固定仓库 commit、依赖和测试条件；environment manager 创建 workspace；harness 把 issue、必要规范与工具描述发给模型。模型先提出 read/search，再 edit，随后 run tests。dispatcher 把工具调用送入正确 sandbox，stdout/stderr/exit code 返回为 observation，harness 保存工具 ID 和新的文件状态。

测试失败时，agent 基于真实错误继续修改；模型输出 final 后，外部 verifier 检查 patch 是否满足要求、独立测试是否通过、是否修改了不允许的范围。artifact collector 保存 diff 与日志；environment manager 依照保留策略回收。Harbor 可安排同一任务的多次 trial，也可把 agent 从本地 Docker 切到 E2B 适配器；模型 API 通常仍是另一个服务。

这条流程中，模型看见的是文本、结构化结果或图像；实际文件写入发生在执行环境。模型输出一段 diff 不说明已经修改文件。测试命令 exit 0 不说明跑了应有的测试：例如只运行了一个无关子集，也可能“绿色但未完成”。

### Search agent 从问题到可追溯结论

用户问“某功能在指定版本和日期是否支持”。harness 先拆成待核断言，search API 找候选页面，extract/scrape 读取正文；需要 JavaScript 或交互时再启用 browser。文档解析与去重把内容变成带 URL、获取时间、版本及片段定位的 evidence；模型比较冲突和覆盖缺口，决定追加搜索或结束。

答案生成器把每个主张绑定到具体证据，verifier 检查时间范围、原文是否支持和问题是否覆盖。search 成功只代表返回候选；fetch 成功只代表收到页面；引用存在只代表有链接。最终需要证明“这个来源支持这句话”。[Search 流程篇](#q=agent-search-components)给出证据对象与失败归因。

## 5. Harness框架SDK和协议分别是什么

**Harness** 是围绕模型执行任务的整套组织逻辑：上下文构造、调用工具、循环、停止、验证、状态与恢复。它是架构职责，不是某一家产品名。Anthropic 的[长期任务 harness 文章](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)说明跨上下文窗口的交接、进度与测试为何重要；这些文件辅助连续工作，但不能代替真正的运行环境恢复。

**Framework** 提供搭建 harness 的抽象，如图节点、路由、状态或消息；**SDK** 提供调用某项服务的客户端。LangGraph 的 checkpointer 保存 thread 内的图状态，store 保存应用定义的跨 thread 信息；这与 E2B 保存沙箱文件/RAM 是不同对象。[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)。

**MCP** 定义 host/client/server 组织及 tools、resources、prompts 等交换方式；本系列用[2025-11-25 固定规范](https://modelcontextprotocol.io/specification/2025-11-25/architecture)讲概念。模型的 tool call 可以由 host 转成 MCP tools/call，也可以调用普通 HTTP/Python 工具；两者都需要 host 调度结果回到模型。MCP server 提供工具，不自动替你建立任务循环。

**ACP** 连接编辑器等 client 与 coding agent，交换会话、消息、工具/权限相关交互；不是浏览器执行器。[ACP introduction](https://agentclientprotocol.com/get-started/introduction)。**A2A** 面向 agent 之间的能力发现与任务协作，不等于给模型加一个 shell 工具。[A2A overview](https://a2a-protocol.org/latest/topics/what-is-a2a/)。

**ATIF** 是 Harbor 的轨迹交换格式，**ASP** 是其文档介绍的远程 sandbox 协议入口。它们分别关心“怎么表示历史”和“怎么访问环境”，不能因为字母相似就当成 MCP/ACP。具体协议的版本和支持矩阵需按接入项目核查；会读格式不代表能恢复所有运行状态。

```text title="教学对照 同一行动在不同边界的表示"
模型输出:        tool=run_tests, args={"suite": "unit"}
host/dispatcher: 校验参数、查权限、确定sandbox/session
transport:       本地函数 / HTTP / MCP tools/call
runtime:         在任务workspace启动测试进程
observation:     tool_call_id、exit_code、stdout、stderr、artifacts
harness:         更新state，决定继续、验证或停止
```

## 6. 组件依赖和替换怎样解释

先用一张“谁调用谁、保存什么”的表描述系统。替换 model gateway 时核消息角色、tool call ID、多模态和采样参数；替换 sandbox 时核文件、cwd、进程、网络、超时、恢复和身份；替换 search provider 时核结果正文、域名限制、发布时间与 relevance score 含义；替换 verifier 时核测试集与评分对象。字段名相似不保证行为相同。

E2B SDK 与 ACS 的兼容接入说明“客户端可以指向相应 backend”，不表示所有 E2B 云功能、模板、snapshot mode 或异常语义都已在目标集群支持。Harbor 的 environment adapter 降低切换成本，但安装 agent、传递文件、资源限制和日志恢复仍有 provider 差异。核关键能力时从 task config 到 adapter 再到 provider 实际调用追踪。

选型可以按缺口推进：单个开发任务先有稳定 CLI/harness、本地可重置 workspace 和独立测试；并发或不可信执行再选择合适隔离与沙箱管理；需要系统对照再接 Harbor。search 原型先用 query+fetch+证据账本，网页交互确实必要时再加 browser。这里的顺序是根据职责作出的工程判断，不是产品性能排名。

**控制平面与数据平面。** 创建沙箱、领取预热实例、配置 TTL、记录所有权属于管理；进程实际执行、传输 stdout、读写文件属于运行。控制 API 200 后仍可能没等到工具完成；执行 RPC 失败时，命令可能已经启动。重试必须先判断跨过了哪一条边界。

## 7. 轨迹产物和状态怎样记录才方便复盘

task_id 标识任务定义，trial_id 标识一次尝试，session_id 标识会话，sandbox_id 标识环境，tool_call_id 标识调用，process_id 标识真正运行的进程。trace_id/span_id 则关联观察链路。不要只保存一个“请求 ID”，然后无法判断是重复模型提议、工具重试还是环境重建。

**Trajectory** 描述做过什么，**artifact** 是产生的 patch、报告、图片或数据；**checkpoint** 描述可以从哪里恢复，**snapshot** 是某种环境状态的保存。它们可能互相引用，却不能互相替代。日志保留“已编辑某文件”的事件不代表恢复后文件确实存在；报告附了 URL 不代表保存了当时读到的版本。

对 code agent，记录 base commit、最终 diff、独立测试结果、环境/template 版本和关键操作。对 search agent，记录 query、候选、最终页面、片段、日期范围和 claim-source 关系。训练时还要补模型版本、真实采样概率、mask 和奖励规则；可观看的 ATIF 轨迹并不自动是 PPO/OPD 训练样本。[工具设计](#q=agent-tool-design)、[上下文管理](#q=agent-context)和[评测基础](#q=agent-evaluation)各有独立主文。

## 8. 面试问答与讲课检查

**Q01 [定位] Harbor、E2B和ACS怎么用一句话区别？** Harbor 编排任务和评测，E2B 提供沙箱及客户端，ACS Agent Sandbox 提供云集群中的沙箱管理与执行能力；先说层次再说产品。

**Q02 [流程] 代码究竟由谁执行？** 模型提出代码/工具参数，dispatcher 调用 runtime，环境中的进程执行，结果作为 observation 回到 harness。模型生成代码不等于执行代码。

**Q03 [定义] Code interpreter就是code agent吗？** 前者提供代码执行能力，后者组织读仓库、修改、测试、反馈与交付闭环。执行器可以被不同类型 agent 复用。

**Q04 [协议] Function calling与MCP怎么衔接？** 前者是模型表达行动的接口，后者可承载 host 到工具服务的调用；host 把参数与结果在两者之间连接，仍需循环与权限。

**Q05 [恢复] LangGraph checkpoint能恢复E2B文件吗？** 通常不能独立做到；它保存图/会话状态，环境的文件/RAM需由独立策略保存，两边标识和状态要匹配。

**Q06 [完成] 模型final和任务成功有什么区别？** final 是停止提议；success 是产物满足 verifier。预算耗尽或环境异常应有独立终止状态。

**Q07 [观察] Trace和trajectory有什么不同？** Trace 用 span 表达调用关系、时间与错误；trajectory 用消息/行动/观察表示交互过程。二者可链接，缺完整内容时 trace 不能重建模型输入。

**Q08 [选择] Search agent一定要浏览器吗？** 若 search/fetch 已给可用正文则未必；页面脚本、表单和会话状态需要交互时才引入 browser，并承担 session 和环境成本。

**Q09 [组件] RAG与search agent什么关系？** RAG 是检索内容用于生成的模式；search agent 可决定多次检索、读页与核查，也可包含固定 RAG 步骤。向量库不是自主决策器。

**Q10 [重试] 工具超时后为什么不能直接重发？** 远端可能已产生副作用，先查操作状态、使用持久幂等键或进入不确定状态。transport timeout 不是执行未开始的证据。

**Q11 [评测] 换agent CLI后能说模型提升吗？** 不宜直接归因。CLI 包含 prompt、上下文、工具与预算策略；控制这些因素后才比较模型。

**Q12 [表达] 怎样两分钟介绍系统？** 用同一任务依次说目标与预算、决策循环、工具接口、执行环境、状态恢复、产物验证和观察，再说具体产品对应哪层。

## 9. 阅读路线与来源

继续读[Harbor E2B ACS与沙箱](#q=agent-sandbox-components) → [Code agent全流程](#q=agent-code-components) → [Search agent全流程](#q=agent-search-components)。课堂例子的[CPU脚本](./assets/agent/agent-components-lab.py)与[结果](./assets/agent/agent-components-lab-results.json)验证重试、恢复和证据计数；云执行参数及产品能力按各篇的来源条件解释。

核心资料为[Harbor concepts](https://docs.harborframework.com/core-concepts/index)、[E2B docs](https://docs.e2b.dev/)、[ACS Agent Sandbox概述](https://help.aliyun.com/zh/cs/user-guide/agent-sandbox/)、[ATIF](https://docs.harborframework.com/agents/atif)及上述协议/状态文档。[来源清单](./research/agent-components-overview-sources.json)登记版本与用途。
