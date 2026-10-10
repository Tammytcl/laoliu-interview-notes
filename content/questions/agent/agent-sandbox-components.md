---
id: agent-sandbox-components
title: "Harbor E2B ACS与Agent沙箱"
category: agent
difficulty: 进阶
tags: ["P1", "Agent", "Harbor", "E2B", "ACS", "Sandbox"]
updated: "2026-10-10"
summary: "详细区分评测编排、SDK、沙箱管理和隔离运行时，解释任务、预热池、会话、快照与真实源码。"
draft: false
---

## 1. 问题背景 把会写代码的模型变成能运行的系统

模型生成“安装依赖、改文件、运行测试”只是行动提议。真正执行需要环境：文件系统初态是什么、进程以谁的身份运行、依赖在哪里、任务之间是否隔离、断线后状态怎样找回、结束后怎么回收？当任务数量从一台机器上的几个变成成百上千时，还要解决启动、分配、容量、路由与生命周期。

Harbor、E2B 和 ACS Agent Sandbox 分别覆盖不同边界。Harbor 将任务、agent、环境和 verifier 组合成可记录的试验；E2B SDK 操作沙箱与进程；ACS 在云集群中管理沙箱并提供兼容接入。读本文时先画一次任务，再把创建 API、工具执行、结果验证放到相应位置。基础术语见[组件总览](#q=agent-components-overview)。

## 2. 前置知识 容器虚拟机SDK和管理器

**Container** 通常通过 namespace、cgroup 等机制隔离视图与资源，并共享宿主 kernel；**microVM** 用精简虚拟机提供 guest kernel 与硬件虚拟化边界。Docker 是容器构建/运行生态的一部分；Firecracker 是 microVM 技术；gVisor 通过用户态内核等机制改变系统调用的处理边界。不能把三个名字都翻译成“容器”，也不能只凭 microVM 字样就判断整个产品的网络和身份策略。

**Image** 定义起始文件系统和程序，**template** 是平台使用的环境定义或可创建环境标识，可能还带启动设置；**sandbox instance** 是某次创建的具体环境。十个请求用同一 template，不应该默认共享同一个正在运行的 workspace。template 名称也可能指可变标签，复现实验应记录固定版本或镜像 digest。

**Control plane** 管创建、分配、身份、配置和 TTL；**data plane** 执行命令、读写文件、传输输出。SDK 是客户端，server/manager 才处理远端请求；SDK 在你的 Python 进程里不等于生成代码也在本机执行。沙箱里常有 daemon/agent-runtime，接受文件和进程 RPC。

**Session affinity** 表示同一任务后续调用进入同一环境；**warm pool** 预先准备可分配实例；**readiness** 是环境已经达到可执行条件。CPU 已分配、Pod Running、daemon ready、依赖可用是不同状态。只把创建 API 的响应时间叫“任务启动时间”，可能漏掉后续等待。

## 3. Harbor 任务运行与评测编排

### Task Dataset Agent Verifier Trial Job

[Harbor concepts](https://docs.harborframework.com/core-concepts/index)定义 task 为指令、环境和验证逻辑；dataset 是 task 集合；agent 是执行任务的程序；verifier 判断产物并给 reward；trial 是一个 agent 对一个 task 的一次尝试；job 收集并运行多个 trial。模型名称只是 trial 配置的一部分，agent 还包含具体 harness。

```text title="Harbor任务目录 角色对照"
my-task/
  instruction.md        # agent要解决的目标
  task.toml             # 配置、资源和超时
  environment/          # 环境构建文件或定义
  tests/test.sh         # verifier入口
  solution/             # 可选参考解，不是默认给agent的上下文
```

task loader 读取目录，environment adapter 初始化环境，agent adapter 安装或连接 agent，runner 执行，verifier 检查，结果与轨迹保存。配置应同时固定 task 版本、agent/harness 版本、model、资源和预算；固定模型却给两个 agent 不同超时，并不是相同预算比较。

Harbor 能接不同 agent 与环境，但具体功能由 adapter 和 capabilities 决定。录制 ATIF、加载 ATIF、恢复 workspace 是三个能力：能输出标准 trajectory 不代表能重新建立全部运行态。Harbor 的[ATIF 文档](https://docs.harborframework.com/agents/atif)也区分写入和加载能力。

### 关键源码 先准备环境再运行最后收尾

固定源码基线 Harbor 09f5b964f8e5075f7e52fd08c9994dfc80ad5d24，包内版本 0.24.0。[Trial.run](https://github.com/harbor-framework/harbor/blob/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24/src/harbor/trial/trial.py#L492-L520)入口节选：

```python title="Harbor真实源码 运行入口节选"
try:
    await self._prepare()
    if not self.config.install_only:
        # Setup/install ran in _prepare(); skip the agent run + verification.
        await self._run()
```

这是 try 主体节选，后面还有取消/异常记录、输出恢复和 finally 收尾，不能复制这几行当完整函数。_prepare 设置环境、健康检查和 agent；具体 _run 由 trial 类型实现；finally 中 _finalize 回收环境并写 result。异常时仍要尽力取回日志，不能把“agent没答对”与“环境没建起来”混成同一种失败。

**配置版本以实际代码为准。** 此固定实现 TaskConfig 默认 schema_version=1.4，而核读时网页配置示例仍写 1.3。[固定TaskConfig](https://github.com/harbor-framework/harbor/blob/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24/src/harbor/models/task/config.py#L801-L820)与[网页配置](https://docs.harborframework.com/tasks/configuration)要一起看。已有任务应由所用 CLI 校验或初始化，不能把另一篇旧教程的字段无条件拼入新 task.toml。

### Verifier与reward为什么需要独立设计

Linux task 常用 tests/test.sh 作为入口，写 /logs/verifier/reward.txt 或 reward.json。[Harbor verifier](https://docs.harborframework.com/tasks/verifier)支持单个数值或标记数值指标，二者同时存在时优先 JSON。Reward 不一定只有 0/1，但名称和范围需要由 task 定义。

```bash title="教学实现 失败测试也写出reward"
#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
if pytest /tests/test_outputs.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

这是教学例子，不是 Harbor 原函数。在 set -e 下先裸跑失败命令、之后才读 $?，脚本可能早已退出；将预期失败的测试放进 if，才能区分明确失败与 reward 文件缺失。测试没有执行完、grader 崩溃、reward 格式错误应保留基础设施错误信息，不能假装是模型得了 0 分。

Separate verifier 环境可减少 agent 接触评分逻辑的机会，但需要明确产物传递。全新 verifier 不会自动继承 agent 的文件修改；task 必须声明需要保存和复制的 artifacts。保留参考解、隐藏测试和独立资源，是评测组织的一部分，不由 model prompt 自动保证。

## 4. E2B SDK与具体沙箱能力

[E2B 官方文档](https://docs.e2b.dev/)提供按需 Linux 沙箱、template 与文件/进程访问。普通 e2b Sandbox 适合 commands 与 files；e2b-code-interpreter 在对应环境中提供 run_code 等解释器能力。shell command 和 Python cell 不同：前者启动进程，后者还可能涉及解释器 kernel、变量与会话。浏览器/桌面类 template 又需要相应程序和接口。

```python title="E2B官方API的教学接线 不在本轮发起云调用"
from e2b import Sandbox

sandbox = Sandbox.create(timeout=300)
try:
    sandbox.files.write("/home/user/example.txt", "task input")
    result = sandbox.commands.run(
        "cat /home/user/example.txt", timeout=30
    )
    print(result.stdout)
finally:
    sandbox.kill()
```

这里显式使用沙箱绝对路径与 finally 回收。客户端的凭据放在运行配置中，不能硬编码进任务文件。create 的 timeout 与 command 的 timeout 不是同一个钟：前者管理沙箱存活，后者限制命令；API request timeout 又限制客户端等待。单位也需要核 SDK：本例 Python 以秒表示相关 timeout。

### 后台进程与工具完成

固定 E2B SDK df6c368152b067512702fc677952c0fc4e13c2f0 的[commands 实现](https://github.com/e2b-dev/E2B/blob/df6c368152b067512702fc677952c0fc4e13c2f0/packages/python-sdk/e2b/sandbox_sync/commands/command.py#L285-L302)：

```python title="E2B真实源码 返回handle或等待完成"
return (
    proc
    if background
    else proc.wait(
        on_stdout=on_stdout,
        on_stderr=on_stderr,
    )
)
```

proc 来自前面的 _start。background=True 返回进程 handle，不代表执行完成；通常还需查询、wait 或 terminate。HTTP/RPC 已返回不说明测试成功，stdout 也可能来自运行中的进程。需要连续会话的 shell/PTY 保留 cwd、环境和交互，普通独立命令则不能假设自动继承上次 cd；每次明确 cwd 更容易复盘。

### 休眠快照与fork保存的是什么

默认 full pause/snapshot 保存文件和内存，恢复时可恢复进程与内存状态；filesystem-only 保存磁盘，恢复时重新启动，变量与进程不能当作还在。[E2B persistence](https://docs.e2b.dev/sandbox/persistence)、[filesystem-only](https://docs.e2b.dev/sandbox/filesystem-only-snapshots)、[fork](https://docs.e2b.dev/sandbox/fork)。

| 操作/状态 | 能保留什么 | 调用者还需做什么 |
| --- | --- | --- |
| connect | 找回运行中或平台支持恢复的实例 | 核sandbox ID、权限和实际状态 |
| full pause/snapshot | 文件与RAM/进程状态 | 重新建立外部网络连接、匹配会话状态 |
| filesystem-only | 文件，不保留RAM进程 | 重新启动服务、解释器和依赖进程 |
| fork | 从某份snapshot创建新实例 | 分支ID独立、外部写入和身份重新约束 |
| kill | 终止并释放实例 | 先按策略导出需要的产物 |

官方 mode 参数文档标 JS/Python SDK 2.55.0 起可用；较早 keep_memory 等字段有不同兼容路径。本例只展示通用创建/执行 API，不把最新版 snapshot 参数强加给旧 SDK 或兼容 backend。外部网络连接在暂停时会断，保存 RAM 不能保证第三方服务连接和锁仍有效。快照也不是数据库的一致性备份协议。

## 5. ACS Agent Sandbox 集群里的管理链

[ACS概述](https://help.aliyun.com/zh/cs/user-guide/agent-sandbox/)介绍 MicroVM 级隔离、休眠/Checkpoint、预热与 E2B 兼容 SDK/声明式接入。这里只采用其架构和接口说明；文档宣传的启动/扩展倍率并不是同一任务、同一硬件下与 E2B 的测速结果。

![本文示意 ACS管理面与执行面](./assets/agent/agent-sandbox-components/sandbox-layers.svg)

**图解。** client 经 domain/auth/manager 请求环境，controller 根据 SandboxSet/Claim 管理分配，底层实例中的 runtime 处理文件/进程。预热池位于分配路径，任务状态和产物位于应用侧。控制箭头与执行箭头分别标示；图没有承诺每个厂商都实现相同快照和网络语义。

**Kubernetes** 提供 API 对象、调度与生命周期机制；**CRD** 扩展对象类型，**CR** 是某个对象实例；**controller** 持续把实际状态协调到声明的目标。ACS 的[创建文档](https://help.aliyun.com/zh/cs/user-guide/create-an-agent-sandbox)涉及 ack-agent-sandbox-controller、ack-sandbox-manager、Ingress 与相应集群组件。需要按目标集群版本安装，不把安装顺序简化成“import SDK就能用”。

**SandboxSet** 声明 template 和预热实例数量；**SandboxClaim** 请求分配；具体 **Sandbox** 承载某个实例。Warm pool 不是 agent 的历史样本池，replicas 也不必等于当前活跃任务数。多个副本共享定义而各有环境状态，服务端还需处理领取后的补充、故障与回收。

```yaml title="ACS官方字段的教学摘录 领取已有模板"
apiVersion: agents.kruise.io/v1alpha1
kind: SandboxClaim
metadata:
  name: code-interpreter-claim
  namespace: default
spec:
  templateName: code-interpreter
  replicas: 1
  claimTimeout: 5m
  ttlAfterCompleted: 15m
```

该片段调整了对象名，前提是已有 template/组件/权限。claimTimeout 限制领取等待；文档说明 ttlAfterCompleted 删除的是完成后的 Claim，**不是自动删除已领取 Sandbox**。清理策略仍要分别处理 claim、sandbox、snapshot 和外部产物；名字都带 TTL 也不能合成一个生命周期。

### E2B兼容入口改变了哪一部分

官方创建示例用 E2B_DOMAIN 指向管理服务、E2B_API_KEY 认证，证书链需被客户端信任，然后用 e2b_code_interpreter 的 Sandbox.create(template="code-interpreter") 获取实例。SDK 方法看起来相同，目标 backend、template 镜像、路由和授权已经改变。

Ingress/网关提供集群外入口，通配域名可参与实例路由，envd/agent-runtime 提供执行能力；这几层不是大模型本身。排查时按 DNS → TLS → auth → manager → claim/allocation → runtime 分层，避免遇到 500 就判断模型“不会调用工具”。

**自定义镜像是能力契约。** ACS 创建文档要求自定义镜像有基础命令与 /bin/bash，并注明自定义镜像时 run_code 暂不可用；官方 code-interpreter 示例镜像带解释器支持。能 commands.run 不代表能 run_code，也不保证文件、PTY、恢复和浏览器接口全部兼容。正式接入按所需接口做 capability test。

### Warm pool为什么既省等待又增加成本

预热把一部分启动工作移到任务到达前，命中时可以省掉该段等待；空闲实例仍占资源，未命中、镜像变化或依赖安装仍有延迟。假设到达率为每秒 2 个任务、平均服务时间 30 秒，稳定条件下 Little's law 给在服任务均值约 60；这不是最佳预热数，也不能描述突发峰值。池大小需结合等待目标、命中率、模板分布与资源限制确定。

统计应分开 cold create、warm allocate、ready wait、dependency setup、tool execution 和 cleanup；记录 p50/p95、成功创建比例与实际活跃并发。每分钟可建多少实例不等于每分钟能完成多少 agent 任务。扩并发还会遇到模型 API、网络、数据库和 verifier 的容量上限。

## 6. 同层比较与适配器的边界

| 对象 | 主要职责 | 需要比较的维度 |
| --- | --- | --- |
| Harbor vs 自建评测runner | 任务/trial/并发/评分/记录 | task格式、agent适配、产物、可复现预算 |
| E2B vs 其他托管sandbox | 环境创建与执行 | 接口、隔离、网络、状态、区域、生命周期 |
| ACS vs 自建集群sandbox管理 | 声明式与SDK环境供给 | 集群运维、warm pool、路由、资源和权限 |
| Docker / microVM / gVisor | 底层执行与隔离机制 | kernel边界、系统调用兼容、成本和资源约束 |
| OpenSandbox | 开源sandbox API/SDK及运行后端 | 所用部署方式、镜像、网络和能力矩阵 |

OpenSandbox 是独立开源项目，不等于 ACS 产品服务；其[固定README](https://github.com/alibaba/OpenSandbox/blob/9bc2967b78a9506a9282b0b28dc5db0cb8c50cc0/README.md)应按具体 backend 理解。Daytona/Modal 等可在 Harbor 环境列表中出现，选择时仍按部署需求核官方支持，不能把多种 provider 的 slogan 组成性能榜。

深层平台设计可连接[DSec原论文精读](#paper=paper-dsec)：它研究镜像/工作区供给、资源与隔离等系统问题，和这里的 task/harness/SDK 有联系；生产系统论文的规模结果不代表任一开源 SDK 自动具备同样吞吐。

## 7. 真实源码里的超时与重复执行

Harbor 固定[E2B adapter](https://github.com/harbor-framework/harbor/blob/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24/src/harbor/environments/e2b.py#L495-L507)在进程发出之后等待结果：

```python title="Harbor真实源码 非零退出仍是执行结果"
try:
    result = await handle.wait()
except CommandExitException as e:
    result = e

return ExecResult(
    stdout=result.stdout,
    stderr=result.stderr,
    return_code=result.exit_code,
)
```

前面的注释明确不因这里的传输失败重新分派已经运行的命令，避免双重副作用。非零 exit 转为包含 stderr/return_code 的结果，agent 可以依据它修复；连接异常则不能自动等同 exit 1。这条边界比“加一个retry装饰器”更关键。

日志中至少记录 sandbox_id、operation/tool_call_id、process_id、dispatch 状态和结果。已知只读请求可以有界重试；写入或进程启动需要服务端持久幂等键、状态查询或显式不确定状态。内存字典里的去重只能覆盖同一进程生命周期，不是分布式 exactly-once 保证。

## 8. 面试问答

**Q01 [定位] Harbor是不是沙箱厂商？** 它主要是任务/agent/环境/评测组织框架，可以调用沙箱 provider；底层隔离和资源供给由环境实现决定。

**Q02 [适配] E2B SDK指向ACS后什么需要重测？** template、commands/files/run_code、认证、域名/TLS、timeout、后台进程、pause/snapshot及异常；接口名相同不代表语义全等。

**Q03 [对象] Task Trial Job怎样区别？** Task 是定义，Trial 是某 agent/model 对它的一次尝试，Job 收集多个尝试。评测重跑不能覆盖旧 trial 的证据。

**Q04 [能力] commands.run与run_code为什么不能互换？** 前者执行进程命令，后者依赖解释器服务；自定义镜像可能只支持前者。

**Q05 [管理] SandboxSet和Claim是什么？** Set 定义模板/预热池，Claim 请求分配。Claim 的 TTL 与 Sandbox 存活分别管理。

**Q06 [状态] Full snapshot与filesystem-only差在哪？** 后者不保存RAM进程，恢复要重新启动；两者都不能保证外部连接自动续上。

**Q07 [测试] Verifier独立环境为何要声明artifacts？** 新环境不继承修改，需要把应评分的产物明确传递，否则可能测的是原始仓库。

**Q08 [故障] Exit code非零与RPC超时有什么不同？** 前者是程序的已知结果，后者可能不知道执行状态，不能据此重发有副作用操作。

**Q09 [资源] Warm pool大一点是否一定更好？** 可减少部分排队/创建等待，但增加空闲成本且不能解决模型或外部工具瓶颈，需测命中与等待目标。

**Q10 [版本] 为什么文档示例不能直接作为最新schema？** 文档与实现可能不同步，本轮 Harbor 默认1.4而网页示例1.3；结合固定实现和实际CLI校验。

**Q11 [安全] 用microVM是否就解决工具权限？** 它限制执行边界，应用工具的身份、网络、写入权限和第三方凭据仍需单独约束。

**Q12 [系统] 怎样向别人讲清沙箱接入？** 从create到ready、exec到wait、collect到pause/kill画三段，分别说明身份、状态、超时和错误，最后指出供给与模型服务是不同资源池。

## 9. 来源与课堂练习

日期2026-10-10；原始依据在各节附近链接，[来源清单](./research/agent-sandbox-components-sources.json)保存固定源码和文档基线。示例用于解释接口，本次没有创建收费云实例或部署真实 Harbor/ACS 服务。

[CPU练习](./assets/agent/agent-components-lab.py)和[结果](./assets/agent/agent-components-lab-results.json)检查“回复丢失后的重复副作用”“恢复消息却丢环境”“Claim与Sandbox独立生命周期”的教学模型，不声称验证厂商运行时。下一篇[Code agent](#q=agent-code-components)把环境接到仓库修改全过程。
