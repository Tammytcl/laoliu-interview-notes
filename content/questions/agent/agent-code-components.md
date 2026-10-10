---
id: agent-code-components
title: "Code Agent从任务到Patch的组件流程"
category: agent
difficulty: 进阶
tags: ["P1", "Agent", "Code Agent", "Harness", "Workspace", "Verifier"]
updated: "2026-10-10"
summary: "用仓库修复任务解释agent loop、检索编辑、shell/PTY、环境状态、测试反馈、产物和评测接线。"
draft: false
---

## 1. 问题背景 为什么模型会写补丁仍不能稳定修仓库

软件任务往往没有固定行动序列：先读哪个文件、错误是否来自依赖、改哪一处、该跑哪些测试，都依赖上一步观察。Code agent 需要把模型的决策接到真实 workspace，再通过编译、测试、差异与独立验证收敛。模型一次输出正确代码只是其中一段，稳定交付还涉及环境版本、文件状态、工具契约、恢复和预算。

贯穿例子是“空数组输入时统计函数应该返回0”。目标不仅是最后回答一句修好了，还要在固定仓库上产生最小 patch，保持已有行为，证明边界条件通过。以下流程是教学设计；组件可合在同一服务，也可以拆成远程 API。产品位置先见[组件总览](#q=agent-components-overview)，环境供给见[Harbor E2B ACS](#q=agent-sandbox-components)。

## 2. 前置知识 仓库进程工具和验证

**Repository / base commit** 指版本化代码及起点；**workspace** 是实际可写的工作目录。可以用独立 checkout/worktree、容器或远程沙箱实现，但不能把所有任务都放进同一目录并假设相互不影响。git branch 只是版本引用，不隔离正在运行的进程、端口和数据库。

**Patch / diff** 描述相对起点的修改；**artifact** 包含交付补丁、构建输出与报告；**dependency lock** 约束依赖版本；**test fixture** 提供可控测试输入和资源。代码、依赖、运行时、测试条件共同影响结果，固定 model 不足以复现 trial。

**Shell** 解释命令字符串；**PTY** 模拟终端设备，适合交互式程序与连续会话；普通 subprocess 更适合结构化参数、明确 cwd 和收集结果。PTY 输出可能带颜色、控制字符和交互提示，不能直接等同干净日志。执行 API 采用 shell 字符串时，拼接未经检查参数会改变命令含义。

**Tool observation** 不只是 stdout。最少还应表达 return_code、stderr、是否超时、是否截断、实际 cwd 和产物引用；后台执行返回 process_id/handle，完成后再取最终状态。否则模型可能把安装尚未结束的进度当成成功，或者用被截断日志中的一行猜根因。

## 3. 完整流程 任务准备到独立验证

![本文示意 Code Agent修改验证与交付闭环](./assets/agent/agent-code-components/code-flow.svg)

**图解。** 主线是固定初态 → 读/检索 → 修改 → 运行检查 → 验证/交付。测试失败回到决策环；异常进入故障处理，预算耗尽也有自己的停止状态。独立 verifier 接受实际 patch/artifacts，trace 同时记录模型与工具。图中的“test pass”不是唯一完成判据，也没有默认授权部署或提交到外部系统。

### 准备任务和环境

记录 repo 来源与 base commit、issue、允许编辑范围、依赖/镜像、测试条件、tool capabilities 和预算。Task loader 应把这些作为可追溯配置；environment manager 创建实例；runner 等待 readiness，再运行初始化。warm pool 命中不能代替安装 agent 或检查业务依赖。

环境与模型服务通常分开：CPU sandbox 运行 shell、构建和测试，model gateway 调用远端模型或自建推理服务。测试用 GPU 时另行配置相应环境，不应默认每个代码沙箱都放一张模型推理 GPU。工具等待可能使模型推理暂时无请求，观察利用率时应分段分析。

### 读仓库和定位

read_file 读有限范围；search_code 找符号/文本；list_directory 给结构；语言服务/LSP 给定义、引用、诊断；repository map 提供压缩的组织信息。rg 的词面检索与 embedding 检索解决的问题不同：明确函数名先查符号/文本，模糊需求可先召回候选，再验证真实代码位置。

若路径匹配太多，应收窄目录、文件类型或符号；先看调用路径和测试，不把整个仓库逐轮灌进上下文。检索结果可能过时，编辑前仍核实际文件与 base 状态。代码中的注释、README 和工具输出是任务资料，不能自动越过上层约束成为任意操作授权。

### 编辑和执行

edit 工具可用精确替换、patch 或写文件，需反馈是否实际命中、修改对象与结果。匹配失败要重新读文件；不要在同名文件或旧内容上静默修改。Apply patch 成功只证明修改已应用，仍需 syntax、lint、unit 或 integration 检查。

测试反馈至少分成：测试断言失败、代码语法/编译错误、依赖缺失、网络限流、环境资源不足、测试本身异常。前两类经常需要改代码；后几类可能应该修环境或报告不可验证。反复改业务代码去“修复无法下载依赖”会把基础设施问题变成新 bug。

### 交付和完成

收集相对 base 的 diff、测试命令和结果、已知未验证条件。Verifier 使用稳定的预期行为和独立测试检查产物；agent 自己新写的测试可说明理解，但不是唯一真值。最终给用户的是改了什么、为什么、证据和剩余限制。模型说 final、所有工具调用正常、工作目录干净，三个条件都不单独保证任务正确。

## 4. Agent loop与工具契约

```python title="教学伪代码 一次工具循环的职责"
state = load_task_state(task)
while state.remaining_budget():
    reply = model(context_builder(state), tool_schemas)
    if reply.is_final:
        state.final_answer = reply.text
        break
    for call in dependency_order(reply.tool_calls):
        checked = validate_and_authorize(call, state)
        result = dispatcher.execute(checked, state.sandbox_id)
        state.record(call.id, result)
    save_checkpoint(state)
verdict = verifier.check(state.artifacts)
return package_result(state, verdict)
```

这是职责示意，不是可直接调用的框架函数。工具并行需要独立性：读两个不相关文件可并行；先编辑再测试、先安装再运行、两次覆盖同一文件必须有序。模型一次输出多个 tool calls 并不提供文件锁和事务保证。

```json title="教学工具契约 进程结果不能只有文本"
{
  "tool_call_id": "call-17",
  "sandbox_id": "sandbox-4",
  "process_id": "process-9",
  "status": "completed",
  "exit_code": 1,
  "stdout": "",
  "stderr": "AssertionError: empty input should return 0",
  "truncated": false,
  "artifact_refs": []
}
```

业务 schema 还要限制路径和参数范围，输出不可解析时保存原始诊断。MCP server 可以提供 read/edit/exec 工具，但 harness 仍需决定谁调用、怎样关联结果与历史。固定[MCP tools规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)中的 annotations 只是提示，不是系统执行的权限控制。

### 真实执行入口与shell语义

固定 E2B SDK 的[_start](https://github.com/e2b-dev/E2B/blob/df6c368152b067512702fc677952c0fc4e13c2f0/packages/python-sdk/e2b/sandbox_sync/commands/command.py#L316-L327)配置片段：

```python title="E2B真实源码 command字符串进入bash"
process=process_pb.ProcessConfig(
    cmd="/bin/bash",
    envs=envs,
    args=["-l", "-c", cmd],
    cwd=cwd,
),
```

此处 cmd 是传入的命令字符串，服务端执行 bash -l -c，不是一个自动安全的参数数组。引号、变量替换、管道和重定向都会影响结果。用 schema 限制工具参数与服务端策略，不让模型凭 tool 名直接得到任意宿主 shell；需要高自由度 shell 的任务也应在受限 workspace/环境中运行。

这份源码说明执行契约，不等于所有 E2B-compatible backend 都使用同样实现。Harbor adapter 可能对 cwd、user 和 env 做额外映射；root 是沙箱内用户概念，不应推成宿主 root，也不意味着共享目录和网络身份无风险。

## 5. Harness产品与同层选择

| 对象 | 更接近哪层 | 讲它时应回答 |
| --- | --- | --- |
| OpenHands、SWE-agent、mini-SWE-agent、coding CLI | agent/harness | 工具、上下文、编辑、循环与终止策略是什么 |
| LangGraph、AgentScope | 构建应用的框架 | 状态、路由、恢复和协作如何表达 |
| Harbor | 评测/试验组织 | task/trial、并发、环境、verifier、记录怎样固定 |
| E2B、ACS等 | 环境供给与执行 | 谁create、怎样exec、状态和回收如何处理 |
| MCP、ACP | 跨组件接口 | 连接哪两个角色、传什么对象、权限在哪里 |

这些是职责定位，不是全面支持矩阵；具体项目会覆盖多个层次。若仅需一个可靠 CLI，未必先建图框架；若应用有明确状态转换和恢复，图/工作流抽象可能更便于维护；做多模型、多环境比较则可用 Harbor 统一任务组织。框架替换不能省掉独立验证。

小型 agent 的代码阅读可从[mini-SWE-agent固定README](https://github.com/SWE-agent/mini-swe-agent/blob/04d809ceab9df28f9adaed044884180159172930/README.md)开始，按模型、环境与 agent 三种职责找入口；这只是阅读路线，不宣称核过完整实现。框架的复杂度是否值得，应由相同任务预算下的成功率、耗时与可诊断性证明。

## 6. 长任务状态恢复和幂等

只恢复 conversation 有两个常见问题：记录说“已改文件”但新环境仍是旧代码；记录说“测试正在运行”但旧进程已经死掉。恢复点应包含 task/base commit、checkpoint、environment/snapshot或产物标识、pending operations 与版本。恢复先 reconciling 对账，再让模型继续，不能直接把旧 stdout 当当前事实。

[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)的 checkpointer 管图状态；E2B snapshot 管环境文件/RAM；Git diff 管代码变化。一个可靠恢复协议要决定哪份状态为真、文件如何重放、测试是否重跑、外部副作用怎样查询。模型记忆不能把丢失的未保存文件变回来。

工具回复丢失时，远端命令可能已启动。理想调用包含可持久的 operation ID，服务端记录执行状态与结果；重试同 ID 应返回同操作状态，而不是新建进程。每一次新的模型决策可有新 tool_call_id，transport retry 则不应随意改变 operation ID。分布式环境还涉及去重记录持久化、并发互斥和结果保留，不能仅靠客户端计数。

```text title="教学恢复顺序 先对账再生成下一行动"
读取checkpoint → 验证task/base/template版本
              → 找回环境或重建并重放已保存artifact
              → 查询pending operation，区分running/completed/unknown
              → 重测必要事实
              → 构建当前context，继续agent loop
```

长任务可以使用 progress 文件、任务列表和 summary 跨上下文窗口交接。[长期harness文章](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)给出这类组织方式，但这些文件不是原子数据库；写到一半、摘要遗漏或环境 fork 时仍需一致性规则。

## 7. 评测与排障怎样按阶段归因

SWE-bench 类任务重点在仓库 issue 修复与独立测试；Terminal-Bench 类任务更广泛地检查终端环境中的完成能力。[SWE-bench官网](https://www.swebench.com/)、[Terminal-Bench官网](https://www.tbench.ai)。不能把 dataset、评测框架 Harbor 和某个 agent implementation 当成一个名字。

最少记录任务通过率、每成功任务成本、端到端延迟、步骤数和终止原因；诊断时分 create/setup、model、tool、verify、collect。工具返回慢可来自外部等待，模型 token 多可能来自重复读文件或不合理接口。增加 agent 并发既可能提高吞吐，也可能加剧 API 限流和环境排队。

失败样本按阶段拆解：

| 阶段 | 典型证据 | 优先检查 |
| --- | --- | --- |
| 初态/setup | clone/依赖失败、未ready | template、网络、版本、资源 |
| 定位 | 看错文件/调用链 | 检索、上下文选择、符号工具 |
| 修改 | patch未应用/逻辑错 | 编辑契约、文件版本、推理 |
| 执行 | timeout/exit/输出截断 | process状态、预算、结果结构 |
| 验证 | 测错对象/隐藏测试失败 | base/diff、artifact传递、判据 |
| 恢复 | 历史和当前文件不一致 | checkpoint与环境对账 |

评测不应只挑能展示成功的轨迹。固定预算、多次尝试分别报告 pass@1 与多次成功指标；降低 grader 质量不会使 agent 真正变好。[Agent评测原理](#q=agent-evaluation)有估计假设与过程诊断。

## 8. 面试问答

**Q01 [主线] 怎样介绍code agent完整流程？** 固定任务与仓库初态，建立环境，模型读/改/运行，根据观察迭代，独立验证，保存patch与日志，回收环境。说明每一段输入输出。

**Q02 [工具] Shell与PTY何时选？** 简单有界命令适合明确cwd和结果的执行API；交互或连续会话可用PTY，但要管理进程、控制字符、阻塞和session。

**Q03 [并行] 同一轮多个工具调用可以全部并行吗？** 只有数据与副作用独立时才适合。编辑后测试、共享文件写入和依赖安装要尊重顺序。

**Q04 [验证] 自己写的测试通过为什么仍可能错？** 测试可能复述同一错误假设或漏边界；外部判据与独立测试需要单独检查。

**Q05 [状态] 为什么记录trajectory仍不能恢复任务？** 日志说明发生过什么，不保证workspace、进程和外部服务状态还存在。恢复要对账实际环境。

**Q06 [重试] Tool call ID与operation ID为什么可能不同？** 前者关联模型消息，后者标识一次真实操作；同一操作的transport重试需保持幂等标识。

**Q07 [评测] 成功率下降如何先区分模型与infra？** 看终止原因和分段证据；环境未ready或grader失败不能直接归因模型推理。

**Q08 [训练] Harbor轨迹能直接做RL训练数据吗？** 不能仅凭格式；还需要模型/采样版本、token与action mask、行为概率、环境与reward契约，按算法核查。

## 9. CPU例子与来源

[课堂脚本](./assets/agent/agent-components-lab.py)在临时目录构建错误函数和独立测试，应用实际修改后重跑；另演示丢失回复的重复执行与恢复消息却丢 workspace。它不调用模型或云环境。[实际结果](./assets/agent/agent-components-lab-results.json)与[来源清单](./research/agent-code-components-sources.json)可下载。

介绍时先用脚本解释可观察结果，再将本地 execution object 替换成 Harbor environment adapter/E2B/ACS 的职责。不要把课堂例子的成功当成真实代码基准成绩。继续读[Search agent](#q=agent-search-components)，比较两类任务的环境与完成判据。
