---
id: training-inference-frameworks
title: "训练框架怎么选、怎么用？以 verl 与 slime 为主线"
category: systems
difficulty: 进阶
tags: ["P0", "训练框架", "slime", "verl", "框架对比", "框架使用"]
updated: 2026-10-05
summary: "围绕训练框架的职责、区别和实际使用，重点比较 verl 与 slime，再说明 Megatron、DeepSpeed、FSDP 以及其他后训练工具的位置。"
draft: false
---

## 1. 先明确比较对象：这些框架分别负责什么？

本文围绕一个问题展开：**我有一个训练任务，为什么选择某个框架，选好后怎样开始使用？** 主线是 verl 与 slime 的对比。Megatron、DeepSpeed、FSDP 用来解释训练后端，OpenRLHF、AReaL、TRL 用来补充同类工具的选择。

### Q：为什么不能把 Megatron、verl、slime 放在一列直接排优劣？

因为它们处理的问题不完全相同。Megatron / DeepSpeed / FSDP 主要解决模型怎样在多卡上执行和更新；verl / slime 还要组织生成样本、计算奖励、更新策略和发布权重。一个后训练框架可以使用某个分布式训练后端。

| 对象 | 主要职责 | 使用时首先决定什么 |
|---|---|---|
| PyTorch FSDP / FSDP2 | 分片模型状态，执行分布式训练 | 怎样组织模型、分片和训练循环 |
| DeepSpeed | 提供 ZeRO、offload 等训练优化与执行能力 | 训练代码怎样接入 engine，怎样配置状态分片 |
| Megatron-LM / Megatron Core | 面向大规模 Transformer 的并行训练组件与训练入口 | 模型如何适配，并行配置与 checkpoint 如何组织 |
| verl | 用算法控制流组织训练角色，可结合不同训练与生成后端 | 使用哪条 recipe、怎样配置角色与后端 |
| slime | 主线围绕 Megatron 训练与 SGLang 生成，连接自定义 rollout 与训练 | 既有模型栈怎样接入，生成流程怎样返回训练样本 |

vLLM / SGLang 在本文只作为生成后端出现：它们负责给 RL 训练生成回答，训练参数更新由训练后端执行。理解这一层关系，就足以继续比较框架。[FSDP2 官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[DeepSpeed 文档](https://www.deepspeed.ai/)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[verl](https://github.com/verl-project/verl)、[slime](https://github.com/THUDM/slime)。

### Q：一个普通微调任务，是否也要用 verl 或 slime？

先看训练流程。若任务只是对固定数据做 SFT，通常可以从现有 trainer 配合 FSDP / DeepSpeed 开始。若任务需要当前策略在线生成回答，再用奖励驱动更新，就需要把生成、奖励和训练连接起来，verl / slime 的价值会更直接。它们也可能提供 SFT 等路径，但“支持这个任务”与“这个任务值得引入整套系统”是两件事。[TRL 文档](https://huggingface.co/docs/trl/index)、[slime 官方仓库](https://github.com/THUDM/slime)。

本文能力核对日期为 **2026-10-03**；固定版本见末尾来源清单。选型判断来自官方设计与接口，未执行 GPU 性能对比。基础推导、推理引擎优化和通用 Infra 面试题留到后续专题。

## 2. 训练后端：FSDP、DeepSpeed、Megatron 怎么选、怎么接？

这一节只解释选择框架时需要知道的区别。

### Q：FSDP 和 DeepSpeed 的使用方式有什么不同？

FSDP / FSDP2 是 PyTorch 内的分布式训练机制。自行编写训练代码时，使用者组织模型、loss、optimizer 和数据循环，再接入分片；通过上层 trainer 使用时，这些细节由 trainer 封装。DeepSpeed 通常让训练代码通过其 engine 执行 backward / step，并用配置描述 ZeRO、offload 等选项。两者都可以由上层工具接入，因此“用了 TRL / verl”之后仍然要问具体训练后端是什么。

可以把接入差别理解为：**FSDP 更贴近 PyTorch 模型和分片 API；DeepSpeed 更贴近 engine 与配置文件的整合。** 这不是功能完整度或速度排名。[FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[DeepSpeed 入门](https://www.deepspeed.ai/getting-started/)。

### Q：Megatron 与它们的关键区别是什么？

Megatron 强调 Transformer 的模型并行和对应高效实现；使用时经常需要检查模型结构映射、TP / PP / EP 等配置与 checkpoint 格式。FSDP 常从已有 PyTorch 模型接入参数分片，DeepSpeed 常从已有训练代码接入其 engine。这里的区别首先是**接入路径和需要承担的适配工作**，不是简单的“前两个只能小模型，Megatron 才能大模型”。[Megatron Core 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)。

| 维度 | FSDP / FSDP2 | DeepSpeed | Megatron |
|---|---|---|---|
| 常见接入点 | PyTorch 模型与训练循环，或上层 trainer | engine 与训练配置，或上层 trainer | 模型适配、并行组件与训练入口 |
| 首先关注 | 分片粒度、模型兼容、checkpoint | ZeRO / offload 配置、engine 生命周期 | 模型映射、并行布局、格式转换 |
| 更值得先评估的情况 | 已有 PyTorch / HF 模型和训练代码 | 已有 DeepSpeed 栈或需要其优化路径 | 已有 Megatron 栈，或模型并行需求明确 |
| 使用成本 | 分布式初始化、分片与恢复仍需理解 | engine、配置与版本组合要联调 | 模型和 checkpoint 适配通常更深入 |

选型表是结合职责作出的工程判断，具体支持和性能还需核对实际版本。现实中存在组合方案，不能把这些机制理解为完全互斥。[DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。

### Q：这和选择 verl / slime 有什么关系？

假设已有 HF 模型、希望先用熟悉的 FSDP 跑 RL，可以优先看 verl 的对应 recipe；假设已有经过验证的 Megatron 训练栈和 SGLang 服务，slime 的主线更贴近现有底座。verl 也有 Megatron 后端，所以这只是评估顺序，不能推出“用了 Megatron 就只能选 slime”。

## 3. verl：怎样理解它，又怎样开始使用？

### Q：verl 主要帮我组织什么？

verl 把 RL 算法的控制逻辑与各角色的分布式计算分开。上层决定生成、评分、算优势和更新的顺序；下层 worker 执行具体计算。WorkerGroup 表示逻辑计算角色，Resource Pool 表示这些角色使用的资源。这样改算法流程时，不必把每个后端的计算实现都重写一遍。[HybridFlow 论文](https://arxiv.org/abs/2409.19256)、[verl Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)。

![verl 官方文档引用的原图：Driver、WorkerGroup 与 Resource Pool](./assets/infra/training-inference-frameworks/verl-driver-workers.png)

**图源与解读：** 来自 verl HybridFlow Guide 引用的原图，保留作者归属。左侧 Driver 调用角色接口，中间 WorkerGroup 组织 worker，右侧 Resource Pool 指定资源。图的重点是“角色逻辑可以与物理放置分开”。展示的共置关系是示意，使用时要看所选 recipe；它并不表示所有数据都必须经 Driver 中转。

### Q：第一次使用 verl，从哪条路径开始？

先选最接近任务的官方 recipe，而不是把所有后端都装一遍。官方 Quickstart 用小模型和 GSM8K 展示 PPO 闭环；需要 GRPO、Megatron 或多轮任务时，再切到对应示例。**PPO 和 GRPO 是训练方法，FSDP / Megatron 是训练后端，vLLM / SGLang 是生成后端**，这三项要分别选择。[verl Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)。

实际准备分为五步：

1. 按该版本安装说明准备镜像或锁定依赖，保证 Driver 与 Ray worker 使用一致环境。
2. 将数据处理为 recipe 需要的格式，核对 prompt、答案或 reward 所需字段。
3. 指定模型与 tokenizer，检查所选训练后端是否支持这个模型。
4. 读启动配置，分别确认算法、训练角色、生成引擎和资源设置。
5. 跑小规模闭环，确认生成、奖励、策略更新和 checkpoint 都产生预期输出。

这里的终点是“训练流程已经正确连起来”，之后再做规模扩展。本文只核对示例与配置，未执行 GPU 任务。

### Q：verl 的配置应该怎么看？

以官方启动路径中常见的配置分组为例：

| 配置位置 | 使用者关心的问题 |
|---|---|
| `data.*` | 训练 / 验证文件、prompt 与 response 长度、数据 batch |
| `actor_rollout_ref.model.*` | 使用哪个模型与相关模型配置 |
| `actor_rollout_ref.actor.*` | 策略如何训练：optimizer、训练 batch、后端相关配置 |
| `actor_rollout_ref.rollout.*` | 用哪个生成后端、每个 prompt 生成几条、生成侧并行与资源 |
| `actor_rollout_ref.ref.*` / `critic.*` | 当前算法是否使用相应角色，以及它们的配置 |
| `algorithm.*` / `trainer.*` | 优势估计或算法选项、日志、保存、评估和作业资源 |

这张表帮助读配置，具体字段以固定版本的 schema 和 recipe 为准。一个目录名、脚本名或 entrypoint 不一定直接说明实际算法；例如 `main_ppo` 路径也可能承载 GRPO 配置。[verl 官方示例](https://github.com/verl-project/verl/tree/main/examples)。

### Q：一个实际示例能说明什么？

已核对的 [Qwen 数学 GRPO 示例](https://github.com/verl-project/verl/blob/8718ca30a3f002f93b7c4fd99b9b2506718681bc/examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh) 同时配置算法、Megatron 训练、vLLM 生成和数据路径。虽然文件名带有 `megatron_fsdp`，这里使用的是 Megatron 路径中的相关选项，不能只凭 `fsdp` 字样把它当成普通 PyTorch FSDP recipe。

阅读它时，按 **DATA → MODEL → ACTOR → ROLLOUT → ALGORITHM → TRAINER → LAUNCH** 追一遍：你会看到同一任务的训练侧并行与生成侧并行分别配置。直接改变 rollout 后端的名字，不能代替对应后端的安装、模型支持和配套配置检查。

## 4. slime：怎样理解它，又怎样开始使用？

### Q：slime 的主线是什么？

slime 主线连接 Megatron 训练与 SGLang 生成，用 Data Buffer 和自定义 rollout 接口组织 prompt 与训练样本。它的吸引力是贴近这两套底座的实际使用：训练侧保留 Megatron 配置，生成侧使用 SGLang 服务和参数，自定义任务接在生成接口上。[slime 作者设计博客](https://www.lmsys.org/blog/2025-07-09-slime/)、[slime 官方仓库](https://github.com/THUDM/slime)。

![slime 作者原图：Data Buffer、自定义 rollout、Megatron 与 SGLang 的连接](./assets/infra/training-inference-frameworks/slime-architecture.png)

**图源与解读：** 来自 slime 团队 2025-07-09 的 LMSYS 博客。Data Buffer 提供 prompt，custom rollout 调用 SGLang 并返回训练数据，Megatron 消费数据、更新策略，再把权重同步给生成服务。使用者主要要接好模型、数据与自定义生成逻辑。图中的 GPU 数量只是示意；Data Buffer 的存在也不意味着算法允许任意复用旧样本。

### Q：第一次使用 slime，应该准备什么？

同样先选官方支持的具体模型示例，再准备相匹配的环境、模型权重和数据。与熟悉 HF trainer 的用法相比，更需要关注 Megatron 模型配置和 checkpoint 的对应关系：生成侧与训练侧的格式需求可能不同，转换路径要按官方示例确认。[slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)。

| 准备项 | 在 slime 中首先看什么 |
|---|---|
| 环境 | 官方镜像、Megatron / SGLang 版本与运行目录 |
| 模型 | 模型参数定义、HF 权重、训练 checkpoint 与转换说明 |
| 数据 | prompt / label 字段、chat template 与 reward 定义 |
| 生成 | SGLang 配置、每 prompt 样本数、生成长度、任务接口 |
| 训练 | Megatron 并行参数、训练 batch、optimizer、loss 选项 |
| 启动 | Ray 作业入口、角色资源与是否共置 |

### Q：slime 的参数和 verl 有什么直观差别？

slime 的启动脚本常把参数分成几组 shell 数组，传给训练入口；Megatron 参数沿用其风格，SGLang 参数常带 `--sglang-` 前缀，另外有 slime 自己的 rollout 与作业参数。verl 常在配置树中按 data、actor、rollout、algorithm 等角色分组，再通过启动入口覆盖。**一个更贴近底座参数的直接拼装，一个更贴近角色配置的组织**，实际选择还取决于团队习惯和已有工具。[slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)、[verl 示例目录](https://github.com/verl-project/verl/tree/main/examples)。

### Q：一个实际示例应该怎样读？

已核对的 [Qwen3-4B 启动脚本](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/scripts/run-qwen3-4B.sh) 分别组织 checkpoint、rollout、评估、Megatron 并行、GRPO、optimizer 和 SGLang 参数。按数组分组阅读，比从最后一条很长的启动命令开始更容易理解。

例如 `--tensor-model-parallel-size` 描述训练侧配置，`--rollout-num-gpus-per-engine` 描述生成实例使用的 GPU 数；它们属于不同执行阶段，不能因为数值相同就认为语义相同。

**该固定版本脚本开头会清理 Python、Ray 和 SGLang 进程。** 按官方独占容器环境使用；改成自己的作业入口时，先处理这段清理逻辑，再核对路径与资源。这里提供的是源码阅读入口，不把整份上游脚本当成适用于任意共享环境的启动命令。

### Q：我要换一个奖励函数，或者接入自己的 Agent，主要改哪里？

按扩展范围选择接口：只改变答案评分，优先改 reward；需要多轮生成、工具交互或复杂数据采样，再实现自定义 generate / rollout。关键是按约定返回训练样本，让 token、reward、loss mask 和相关标识能够被训练侧正确消费；不是仅返回一段最终回答文本。[slime Agentic RL Guide](https://thudm.github.io/slime/get_started/agent.html)。

## 5. 横向对比与选型：slime 与 verl 到底差在哪里？

### Q：能用同一组维度直接比较吗？

| 比较维度 | verl | slime 主线 | 对使用者的实际影响 |
|---|---|---|---|
| 组织思路 | 算法控制流与角色计算分开，WorkerGroup / Resource Pool 组织执行 | 连接 Megatron、SGLang、Data Buffer 与自定义 rollout | 决定从角色 / recipe 还是既有底座 / rollout 入手 |
| 训练后端 | 官方提供 FSDP/FSDP2、Megatron 等路径 | 重点围绕 Megatron 集成 | 已有模型与训练栈决定适配工作量 |
| 生成后端 | 官方提供 vLLM / SGLang 等路径 | 重点围绕 SGLang 原生集成 | 比较实际任务的支持组合，而不只看引擎名称 |
| 配置组织 | 按 data、actor、rollout、algorithm、trainer 等分组 | Megatron、SGLang 和 slime 参数在脚本中组织 | 联调时需要找到真正控制该行为的配置层 |
| 典型扩展 | reward、算法控制流、Agent Loop、后端接口 | reward、自定义生成 / rollout、后端适配 | 扩展点与团队想改的部分是否匹配 |
| 模型与 checkpoint | 依训练后端选择相应适配和保存 / 合并路径 | 关注 Megatron 模型配置与训练 / 生成权重衔接 | 格式与模型映射可能比改启动参数更费工夫 |
| 异步与多轮 | 有 Agent Loop 与 fully async recipe | 有 agentic 接口与 fully_async 示例 | 要检查具体路径的限制，不能只背一个支持标签 |
| 优先评估情境 | 希望基于现有 recipe 做算法实验，或比较不同后端组合 | 已有 Megatron + SGLang，重点接自定义采样与训练闭环 | 是评估顺序，不是性能胜负结论 |

依据：[verl 官方仓库](https://github.com/verl-project/verl)、[slime 官方仓库](https://github.com/THUDM/slime)、[verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[verl Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html)。主仓库的路径与独立衍生项目要分开；多个功能分别支持，不等于所有模型 / 后端 / 模式组合都被验证。

### Q：哪些能力不能用来简单区分二者？

“支持 GRPO”“支持多轮”“支持异步”“能用 Megatron”都不足以独立完成选型。二者存在重叠能力，差别更多在**组织方式、接入接口、既有栈匹配和可用 recipe**。特别是“verl 只能同步”已不符合核对时的官方文档；slime 主线的 SGLang 定位也不应被扩展成对整个生态的永久能力断言。

### Q：能不能直接回答哪个更快？

必须比较同一个训练任务和实际组合。比如“verl + Megatron + SGLang”与“slime + Megatron + SGLang”，底座相近，才更容易讨论编排和接口差异；如果另一边换了模型、生成引擎、精度和资源布局，速度差不能全归因于顶层框架。

使用者首先比较能否正确跑通、达到目标效果所需成本、改动与维护量，再比较性能。本文没有同配置的 GPU 实测，因此不提供通用速度排名。

## 6. 同一个训练任务，换框架后具体要改什么？

以“用规则奖励训练一个数学模型”为例，任务本身固定：prompt 带标准答案，策略生成回答，评分函数检查答案，训练方法更新策略。换框架时，算法含义应保持一致，主要变化在接线方式。

| 项目 | 在 verl 中的常见落点 | 在 slime 中的常见落点 |
|---|---|---|
| 数据准备 | recipe 的预处理与数据配置，常见 parquet 路径 | 示例的数据加载与字段参数，格式按数据接口确定 |
| 模型 | model 配置，再选相应训练后端 recipe | 模型定义脚本、HF 权重和 Megatron checkpoint |
| 训练方法 | algorithm 配置、loss 与相关角色 | advantage / loss 参数与训练实现 |
| 生成 | rollout 配置与对应后端 | rollout 参数、SGLang 配置与 custom generate |
| 规则奖励 | RewardManager / reward 扩展路径，按该 recipe 接入 | reward 类型或自定义评分接口 |
| 多轮任务 | Agent Loop 或对应任务 recipe | 自定义生成接口与 agentic 示例 |
| 资源与启动 | trainer、worker / pool 与启动配置 | Ray job、actor / rollout 资源与启动脚本 |
| 保存与恢复 | 所选训练后端的 checkpoint 与合并路径 | Megatron / slime 的 checkpoint 与恢复路径 |

表中描述常见入口，不要求每个 recipe 使用同一种文件格式或 API。具体实现依据：[verl Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)、[slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)。

### Q：如何判断这次迁移真的完成了？

不要只看进程启动。至少确认四件事：同一 prompt 被正确编码；reward 对成功与失败样本给出预期结果；有效生成 token 进入训练 loss；更新后的策略能够用于下一轮生成并保存。再对齐生成参数、组内样本数、数据和评估方式，比较迁移前后的任务效果。

迁移顺序建议是：**单轮规则任务 → 模型与 reward 对齐 → checkpoint 闭环 → 多轮或异步扩展。** 这样每一步都对应明确的框架接口，出问题时也容易判断改错了哪一层。

### Q：我想先动手，第一份使用笔记应该记什么？

写一页就够：所选框架与版本、任务 recipe、模型 / 数据路径、训练 / 生成后端、主要配置分组、启动入口、reward 接口、checkpoint 位置和跑通判据。复杂性能参数先沿用该 recipe 的起点，随后再按自己的模型和硬件调整。把“我改了哪个框架接口”记录下来，比保存一条无法解释的长命令更有用。

## 7. 其他候选：OpenRLHF、AReaL、TRL 如何放进选择范围？

这一节用于补全候选范围，不逐个展开新的系统专题。

| 工具 | 常见使用入口与设计重点 | 什么时候值得先评估 | 应检查什么 |
|---|---|---|---|
| OpenRLHF | Ray、DeepSpeed、vLLM 与 HF 生态的 RLHF / Agentic RL 栈 | 团队已有这些组件，希望沿用相近的角色与训练路径 | 目标算法示例、模型支持、角色布局与实际扩展接口 |
| AReaL | 异步 RL 系统与 Agent 训练接口 | 任务需要生成与训练解耦，准备认真采用其训练流程 | 官方 workflow、模型支持、异步配置与现有任务的接入方法 |
| TRL | HF 生态的 SFT、DPO、GRPO 等 trainer | 已有 Transformers / PEFT 代码，先做标准训练方法 | trainer 的算法边界、分布式与生成集成、checkpoint |

这些评估顺序来自各项目设计，而不是性能排名。[OpenRLHF 官方仓库](https://github.com/OpenRLHF/OpenRLHF)、[AReaL 官方仓库](https://github.com/areal-project/AReaL)、[TRL 官方文档](https://huggingface.co/docs/trl/index)。

### Q：TRL 就只是单卡工具吗？

不是。不能仅凭熟悉的 `Trainer` 形式，推断它没有分布式或生成引擎集成。但它的入口与大型 RL 编排框架不同：选择时要看目标 trainer 能覆盖多少实际需求，扩展任务和资源管理是否仍然顺手。[TRL 文档](https://huggingface.co/docs/trl/index)。

### Q：OpenRLHF、AReaL 与 verl / slime 是直接替换吗？

可以在“完成同一个后训练任务”这一层比较，但接口、数据协议、配置和 checkpoint 不保证兼容。已有 reward 和 Agent 代码能复用多少，往往决定迁移成本。先找到对应官方例子，再估算改动范围，避免根据相似算法名称判断“只需改包名”。

## 8. 围绕框架选择与使用的问答

**Q01：已有 Megatron 训练栈，应该选 slime 吗？** slime 主线贴近 Megatron + SGLang，值得优先评估；verl 也支持 Megatron。继续比较模型、生成接口、任务 recipe 和迁移成本，不能只由训练后端决定。

**Q02：模型只有 HF 权重，直接用 slime 是否最省事？** 先核对该模型的 Megatron 适配与转换路径。若已有合适的 FSDP recipe，verl 或现有 HF trainer 也可能更顺手；这是接入成本判断，需要具体模型信息。

**Q03：我要修改 GRPO 的算法逻辑，应该看哪个框架？** 两者都需要找到优势与 loss 实现。verl 的控制流分离便于从算法循环追角色调用；slime 可从训练入口和 Megatron 适配追 loss。选择取决于要改算法顺序、loss 还是采样逻辑。

**Q04：我要把自己的代码 Agent 接上去，先比较什么？** 比较任务 loop 的接入点、生成 token / logprob 的捕获、reward 返回形式和样本协议。先验证最简单的一轮工具调用，再扩展完整任务。[verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[slime Agentic Guide](https://thudm.github.io/slime/get_started/agent.html)。

**Q05：切换 rollout 后端，改一个配置名字就行吗？** 名字只是入口，还要确认环境依赖、模型、logprob、权重更新与任务接口。使用对应后端的官方 recipe 作为起点，比混搭两份旧教程可靠。

**Q06：能用“是否支持异步”区分 slime 和 verl 吗？** 不能，核对时两者都有相关路径。比较目标模型、角色布局、任务接口和具体 recipe 的限制。[verl Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html)、[slime fully_async 示例](https://github.com/THUDM/slime/tree/main/examples/fully_async)。

**Q07：两个框架训练效果不一样，可以判断一个更好吗？** 先核对数据、reward、生成配置、训练目标、有效样本与评估。配置不等价时，结果不能直接归因于框架。代码能运行只是起点。

**Q08：面试中怎样说明自己会使用框架？** 用具体任务说明选了哪条 recipe、改了什么接口、为什么选该后端、怎样验证 reward / 参数更新 / checkpoint。能解释这些，比罗列启动参数更能体现理解。

以上是围绕本文框架对比重新组织的练习，不标注公司真题或高频次数。通用显存、通信、推理服务与系统设计题另行整理。

## 9. 来源与后续整理

| 来源 | 本文用途 |
|---|---|
| [verl 仓库](https://github.com/verl-project/verl)、[Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)、[示例目录](https://github.com/verl-project/verl/tree/main/examples) | 框架定位、使用流程与配置组织 |
| [HybridFlow](https://arxiv.org/abs/2409.19256)、[Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) | 控制流 / 计算流、角色与原图 |
| [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html) | 扩展接口与能力边界 |
| [slime 仓库](https://github.com/THUDM/slime)、[Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)、[Agentic Guide](https://thudm.github.io/slime/get_started/agent.html) | 底座关系、模型准备和任务接入 |
| [slime 作者博客，2025-07-09](https://www.lmsys.org/blog/2025-07-09-slime/) | 设计动机与架构原图；历史规划与当前支持分开 |
| [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[DeepSpeed 入门](https://www.deepspeed.ai/getting-started/)、[ZeRO](https://www.deepspeed.ai/tutorials/zero/)、[Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) | 训练后端的定位与接入区别 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[AReaL](https://github.com/areal-project/AReaL)、[TRL](https://huggingface.co/docs/trl/index) | 其他候选的使用入口与选型边界 |

两份脚本链接固定到本次核对 commit；完整记录见 [来源与版本清单](./research/framework-sources.json)。本文提供阅读与接入路线，GPU 运行结果需在实际使用后另行记录。

**本轮更新：** 文章收窄为训练框架的区别、选型与使用；仅保留两张直接解释框架架构的原图，以及八个框架相关问答。通用基础、推理优化、排障和原 60 题版本已移出正文，保存在仓库归档，留待后续分主题整理。

**2026-10-05 基础复习补充：** 已独立整理 [训练一步与显存](#q=infra-training-step)、[GPU性能与通信](#q=infra-gpu-performance)、[分布式并行](#q=infra-distributed-parallelism)、[推理引擎](#q=infra-inference-engine)、[RL/OPD训推闭环](#q=infra-rl-pipeline)。可以按此顺序补机制，再回来解释框架配置；本篇继续集中回答框架区别与使用。

后续整理沿用 [专题工作流](./docs/topic-workflow.md) 与 [模板](./templates/topic.md)，每一轮先确定一个主问题；基础知识按当前问题的需要补充，避免把相关内容全部塞进一篇。来源观察的候选变化见 [更新记录](./research/training-inference-frameworks-pending.md)。
