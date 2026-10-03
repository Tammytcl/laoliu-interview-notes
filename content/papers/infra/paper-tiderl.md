---
id: paper-tiderl
title: 'TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling'
paper_title: 'TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling'
authors:
- Yanyu Ren
- Xizheng Wang
- Xiao Liu
- Bowen Lv
- Hanchen Zhang
- Shudan Zhang
- Hanyu Lai
- Shuai Wang
- Li Chen
- Dan Li
- Jie Tang
affiliations:
- Tsinghua University
- Z.AI
- Zhongguancun Laboratory
author_affiliations:
- - 1
- - 3
- - 1
  - 2
- - 1
- - 1
- - 1
- - 1
- - 3
- - 3
- - 1
- - 1
venue: arXiv preprint (fixed v1)
year: 2026
direction: infra
areas:
- language
tasks:
- training-adaptation
- reasoning
- agents
published: '2026-08-11'
method_figure: ./assets/papers/paper-tiderl/method-source.png
method_caption: Figure 5 · TideRL architecture
paper_url: https://arxiv.org/abs/2608.10402v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- TideRL
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 以任务 token 占用准入和训练数据就绪状态驱动 Ref–Actor 执行与 GPU 角色弹性，区分生成吞吐与训练 goodput。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

TideRL 面向多轮 agentic RL 中不断变化的工作就绪状态。任务一会儿等待环境，一会儿生成很长上下文；同样数量的活跃 agent，可能对应完全不同的 GPU 工作量。与此同时，训练端必须为参考模型算 logprob，再为 actor 做 forward/backward。仅把 rollout 和 trainer 分到不同 GPU，并不能保证两个角色都持续忙碌。

**为什么异步之后仍然会空转？** 环境长尾会使 ready microbatch 的到达既稀疏又突发。稀疏时，专门保留的参考模型 GPU 没数据；突发时，参考模型与 actor 反复交换权重或排队，消费能力不足，rollout 又因缓冲积压或策略滞后被限流。固定 rollout/train 比例只能在某个平均状态上折中，难以适配冷启动、评测、正常训练等不同阶段。

论文把挑战拆成三部分：请求级并发忽视了多轮上下文导致的 KV 压力；Ref–Actor 执行策略不适配数据到达节奏；固定角色资源无法追随生产和消费速度变化。对应 CTB、RA2P 与 ERS 三个组件。它与 Heddle 都强调任务状态，但 Heddle 主要优化长轨迹的 rollout 完成时间，TideRL 把可训练数据与训练资源放进闭环，主指标也改为被 trainer 实际消费的 token/s。

**这里的 Ref 是否就是 OPD teacher？** 通常不是。RL 的 reference model 常用于 KL 约束，OPD teacher 则提供蒸馏目标，两者的分布、更新频率与 loss 依赖可能不同。TideRL 的队列和资源机制可借鉴，但不能直接把 Ref 模块换名为 teacher 就认为完成蒸馏流水线迁移。是否能推迟依赖到 backward，需要检查当前学生 logits、教师输出和梯度图的实际关系。

本文固定 [TideRL v1](https://arxiv.org/abs/2608.10402v1)，首发 2026-08-11。它同时覆盖文本和多模态环境，因此必须把“相对同步基线超过 5.6×”“相对某些异步基线超过 33%”分开阅读。系统 throughput、任务指标和等质量训练时间是三个相关但不同的证据。

## 2. 方法与实现机制

![Figure 5：TideRL 整体执行闭环](./assets/papers/paper-tiderl/method-source.png)

Figure 5 把 CTB、ready buffer、参考模型/actor 以及弹性协调器连接起来。数据从完整任务交互进入训练缓冲，控制信号再反馈到角色分配。读图时应区分任务池里“仍在运行”的对象与全局缓冲里“已经可训练”的对象：前者多不意味着后者多。ERS 针对后者的就绪量与到达间隔调整资源，CTB 针对前者的上下文占用限制并发。

**CTB 如何准入任务？** 每个 rollout DP rank 被看作 token 预算池，而不是固定数量的请求槽。先用一个轻量 probing task 估计该轨迹组首轮的 footprint；每个交互轮次开始时更新真实 token 占用。只有剩余预算能够容纳任务才准入。除此之外，还必须满足环境并发上限、策略过时约束和任务多样性约束，避免把同类慢环境集中到一个 batch。增加总 agent 数不能绕过这些约束。

**上下文继续增长怎么办？** 当 rank 超预算时暂停低优先级任务，保存可恢复状态；有余量时先恢复 paused task，再投放新任务。论文采用语义优先级：评测任务最高，因为它需要锚定权重并可能阻塞同步；接着考虑 GRPO 组完成比例、当前执行状态、上下文长度。原文组合形式为：

$$
P(t)=\omega_1 I_{\rm eval}+\omega_2 G_{\rm completion}
+\omega_3 I_{\rm active}+\omega_4 L_{\rm context}.
$$

各项不是任意“长任务优先”的同义词：高组完成率有利于尽快产出可算优势的完整组；不中断正在执行的任务避免损失已投入计算；保留大上下文减少重复 prefill。恢复时根据 group ID 保持 worker affinity，优先利用已有共享前缀。参数选择和优先级实现仍需按环境校准，不能从公式直接推导所有业务的最优权重。

**RA2P 为什么有两种模式？** RAS 是训练开始时已有的 ready backlog，TPRM 是后续 ready microbatch 的到达间隔。高 RAS 或短 TPRM 意味着专用 Ref 与 Actor GPU 能被持续喂饱，选择解耦 streaming；低 RAS 且长 TPRM 则倾向共置，把额外 GPU 释放给 rollout。

在解耦模式，Ref 只做 forward，一个 Ref rank 为多个 Actor rank 供给 reference logprob。Actor forward 不再等待参考分数；loss 的相关计算被移到 backward 前，需要 Ref 输出时再消费。这样 Ref forward 能与 Actor forward 重叠。依赖没有被删除，而是推迟到真正需要的位置；若参考分数仍未就绪，backward 依然要等。

在共置模式，Ref 与 Actor 使用相同 GPU，逐个 microbatch 交替切模型会产生频繁 I/O。RA2P 聚合当前 ready batches：一次加载 Ref 处理多个 microbatch，再切 Actor 处理这一组，摊薄交换成本；同节点通过 GPU 共享内存传递输入和 logprob，减少序列化与冗余拷贝。它不是凭空增加内存，模型和 KV 的总容量仍限制可行配置。

microbatch 的大小和 dispatch 也参与优化。前面的 batch 尽量凑足 token 以利用所有 DP ranks，最后一个刻意较小，减少 pipeline flush 尾巴。共置模式采用长度排序后的 zig-zag 分配来平衡总工作量；解耦模式让最终送入流水线的序列尽量短。这里调整的是 global batch 内的执行组织，是否改变 optimizer 更新样本数还需结合 ERS 的弹性 batching。

**ERS 如何决定迁移 GPU？** 每次 Actor 更新结束、权重广播开始前规划。Trainer 等数据过久意味着 generation deficit：ready volume 小于理想批量，选择共置并把 Ref rank 转成 rollout。Rollout 等消费或等新权重过久意味着 consumption deficit：最旧 microbatch 的队头等待超过阈值，或 RAS 超过解耦模式盈亏点，回收 rollout rank 建专用 Ref。

论文允许训练批量落在 $[B_{min},B_{ideal}]$，并在 rollout 缩容后适度降低 fetch size、保留 reservoir，避免下一轮完全饿死。这个 reservoir 是生产消费节奏控制，不自动代表重复训练历史样本的 replay。判断时应检查一个 sample ID 被消费几次，而不是看到 buffer 就推断为 experience replay。

资源切换借用权重同步边界：更新后旧 KV 不再适配新参数，迁移任务时丢弃旧 KV、在目标 rank 重新 prefill，避免物理 KV 搬运；Ref offload 与 Actor backward、新角色权重加载与广播重叠。因此 TideRL 的“cache-free migration”是借助已经要失效的缓存，不是一般情况下 KV 可以无成本迁移。与 Heddle 的 RDMA KV 迁移路线不同。

本次未发现可核验的作者官方代码，核读依据为 CTB 伪代码、RA2P 依赖重排、ERS 控制和附录实现说明。作者称约 16K 行 Python，支持 Megatron/FSDP、vLLM/SGLang；实验使用 vLLM + Megatron。报告没有将某个通用 verl recipe 当作 TideRL 实现。[方法原文](https://arxiv.org/pdf/2608.10402v1)

## 3. 实验设置与算力

| 项目 | 固定版披露 |
|---|---|
| 训练集群 | 4 节点，每节点 8×H100、64 CPU 核、1.5 TB RAM、NVLink，共 32 GPU |
| 存储 | JuiceFS-backed NFS，支持 checkpoint 与参数同步 |
| 环境集群 | 独立 bare-metal Kubernetes、1024 CPU 核，通过 API 管理环境生命周期 |
| 文本任务 | WebShop、AlfWorld；Qwen2.5 7B/14B |
| 多模态任务 | OSWorld、ScienceBoard；Qwen3-VL 4B、Qwen3.5 9B |
| 后端 | rollout vLLM，training Megatron |
| 对照 | 同步 VeRL、异步 AReaL、streaming 的 StreamRL |
| 过程 | 文本 100 steps，多模态 40 steps，每 20 steps 评估 |

多模态轨迹包含截图和 GUI 操作，上下文与环境等待会显著大于纯文本。Qwen3.5 使用不同注意力和预测机制，因此不宜只用参数量解释吞吐差异。环境 CPU 集群也是资源预算的一部分：GPU-hour 可比较训练成本，却不能忽略外部环境容量对到达速度的限制。

作者对固定分区异步基线扫描 rollout GPU 比例，报告对应任务的最好配置，TideRL 从同一候选比例开始后再动态调整。文本设置的最佳固定比例为 0.25。对照尽可能匹配 GPU 预算、任务流、checkpoint、过时上限、global batch 与后端，但“框架支持时匹配”仍是边界；不能默认所有软件能力完全相同。

主指标定义为最终被 Trainer 消费的生成 token/s，排除丢弃或过时而未用的数据，因此是操作意义上的 goodput。它比仅计推理发出的 token 更接近训练进展，但也不能直接证明每个 token 信息价值相同。任务指标采用 BoN reward 和 pass rate，另报等接近表现所需 wall-clock；这些应与 throughput 一起检查。

附加微基准包括单 H100 上 1312 个 WebShop tasks 的 CTB 对比；4 H100 处理 8 个 ready microbatch，每批 8192 token 的 RA2P 对比；8 H100 上前 100 WebShop steps 的等待时间对比。它们用于隔离组件，规模与端到端集群不同，不能把微基准的最大百分比叠乘成集群总收益。公开配置尚不足以声称完全复现全部系统；本地未启动训练或 GUI 环境。

原文表格还有值得复核的数值口径：某些 Improvement 列与相邻吞吐值直接相除并不精确一致。本文保留原表，不替作者更改结果；实际选型以同口径原始 throughput 和运行日志为准，复现实验需要进一步核对百分比的统计区间。

## 4. 结果与图表解读

![Figure 8：文本任务训练 goodput](./assets/papers/paper-tiderl/text-throughput-source.png)

Figure 8 横轴为文本模型规模，纵轴为训练吞吐，柱色区分四个框架。作者报告相对同步 VeRL 约 5.6×，相对 AReaL 约 1.8×；StreamRL 在文本任务中出现频繁模型切换，14B 运行六小时仍未完成，所以相对它的极大倍率不能作为所有 streaming 实现的普遍劣势。图支持的是当前资源、后端与微批组织下的 end-to-end goodput 差异。

文本任务到 100 steps 时，同步基线仍有最好的训练表现。TideRL 的 BoN reward 约差 0.01，pass rate 约差 0.5%，并以 48.9% 的同步 wall-clock 达到接近表现。把这些写成“准确率完全一致”不准确；把吞吐倍率直接写成达到质量目标的加速也不准确，两种分母不同。

![Figure 10：多模态任务训练 goodput](./assets/papers/paper-tiderl/multimodal-throughput-source.png)

Figure 10 比较多模态模型及框架，纵轴仍是被消费 token/s。作者报告相对 VeRL 6.02×、相对异步基线超过 1.33×。GUI 环境等待更长，streaming 更有机会隐藏计算，但较大的 trainer 资源组也减少了可迁移 rank 的粒度。ERS 不能无限细分 GPU，保守扩容使 trainer 最终可能成为瓶颈，数据自然进入一步 stale 的状态。

多模态 40 steps 的 wall-clock 是同步基线的 37.8%，即减少 62.2%；任务表现仍有轻微过时影响。因此 33% 指某些异步吞吐对照，62.2% 指相对同步的运行时间减少，不能放进一个无分母的“总加速”结论。

![Table 2：OSWorld、Qwen3-VL 4B 的累计消融](./assets/papers/paper-tiderl/table-2-pdf.png)

Table 2 使用 StreamRL baseline，统计前 10 steps，包含初始正确性评测。吞吐依次为 20.2、23.1、31.8、33.3 k token/s，分别逐步加入 RA2P、ERS、CTB。原表 Improvement 列写 10.4%、35.7%、4.8%；这些值与逐行简单相除有偏差，例如 23.1/20.2−1 约 14.4%。应保留这处披露差异，不能悄悄把表内数字改成自己的计算。

累计消融表的直观信息是 ERS 在这个场景贡献了较大的额外提升，而 CTB 的总体增益较小，但冷启动首步收益更明显。不能据此说 CTB 不重要：瓶颈位置、评测阶段和环境负载会改变组件价值。微基准中 CTB 的缓存命中率提高 1.58×、生成吞吐 1.15×；RA2P 的局部耗时最多减少 44.3%；ERS 等待总量减少约 68.6–77.6%。这些分别解释缓存、执行与生产消费平衡，范围不同。

**面试如何回答“为什么调参固定 GPU 比例还不够”？** 固定比例最优化的是某个平均到达分布，评测和长环境操作会改变 ready batch 数量及间隔。ERS 用队列信号在状态变化时切角色，并结合 Ref–Actor 模式选择。但必须同时说明切换成本、最小 rank 粒度、过时与批量变化，否则容易把弹性调度说成无成本增加算力。[实验原文](https://arxiv.org/pdf/2608.10402v1)

## 5. 局限、结论与后续阅读

TideRL 说明“活跃 agent 多”与“推理可运行工作多”“训练 ready 数据多”是不同状态。CTB 处理上下文和准入，RA2P 处理参考模型依赖与切换成本，ERS 处理角色容量；三层结合比单独扩大请求池更完整。

代价包括 probing 与优先级管理、ready buffer 观测、权重同步边界的角色切换，以及改变训练批量可能带来的算法影响。长环境任务能否暂停恢复也依赖环境接口，不是存 token 就足够。论文不提供普适 OPD 校正，Ref 的 KL 职责与蒸馏 teacher 仍需分开建模；原表比例口径差异和未发现官方实现也限制复现。

与 [Heddle](#paper=paper-heddle) 比较任务级信号，与 [AReaL](#paper=paper-areal) 比较有界积压，与 [AsyncOPD](#paper=paper-asyncopd) 比较旧采样数据的 loss 校正。建议实验先固定算法和总 GPU，记录 ready requests、ready train tokens、HoL 等待、RWT/TWT；再逐步打开 CTB、模式选择和 ERS。最终用相同 GPU-hour 的任务指标验证 goodput 是否转化为有效学习，而不是只展示更满的设备曲线。[固定版论文](https://arxiv.org/abs/2608.10402v1)
