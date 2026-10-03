---
id: paper-heddle
title: 'Heddle: A Distributed Orchestration System for Agentic RL Rollout'
paper_title: 'Heddle: A Distributed Orchestration System for Agentic RL Rollout'
authors:
- Zili Zhang
- Yinmin Zhong
- Chengxu Yang
- Chao Jin
- Bingyang Wu
- Xinming Wei
- Yuliang Liu
- Xin Jin
affiliations:
- School of Computer Science, Peking University
- Independent Researcher
author_affiliations:
- - 1
- - 1
- - 2
- - 1
- - 1
- - 1
- - 2
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
published: '2026-03-30'
method_figure: ./assets/papers/paper-heddle/method-source.png
method_caption: Figure 8 · Heddle architecture
paper_url: https://arxiv.org/abs/2603.28101v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- Heddle
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 通过剩余长度预测、长轨迹优先调度、KV 迁移和模型并行调整，减少多轮 agent rollout 的反复排队与批次长尾。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

Heddle 研究的是多轮 agent 的 rollout 编排：一条任务反复经历模型生成、工具执行、接收结果、继续生成，而现有服务端常只看到彼此独立的推理请求。任务越长，就越可能在每次工具返回后重新排队；当训练需要整批轨迹或完整 GRPO 组时，这些累积的等待最终决定 batch 何时完成。这里的关键问题不是模型单次 decode 太慢，而是系统没有保留“这些请求属于同一个长期任务”的调度信息。

**为什么请求吞吐很高，整批 rollout 仍然很慢？** 请求级调度可能不断完成短请求，却让少数多轮任务反复等待。同步训练直到这些任务结束才能开始，因此 GPU 的批次尾部会逐渐缺少可运行请求。离线 RL 关心的 batch makespan 是最后一条轨迹结束的时间；在线服务常关心平均请求延迟或短时间完成数量。优化目标不同，调度优先级也可以不同。Heddle 因此采用预计长任务优先，而不是把最短请求优先当作默认正确答案。

论文将瓶颈拆成三个因素：长轨迹的反复排队、同一个 worker 上长短轨迹的资源干扰、长轨迹的单 token 延迟。分别对应 progressive priority scheduling、trajectory-aware placement、trajectory-adaptive resource management。它基于 verl、SGLang 和 Ray 实现，属于 rollout 编排层；与 AReaL 解除 rollout/learner 屏障的研究互补，并不直接提供 OPD 蒸馏损失或异步旧数据校正。

**与简单增加并发的区别是什么？** 更多任务能填充工具等待造成的空洞，但也会增加 KV 占用、prefill 和排队。没有轨迹信息时，一个刚从工具返回的长任务与新到的短请求没有区别。Heddle 尝试让已有进度、预计剩余长度与资源分配共同参与决策。对“Generate 利用率 14.4%”这类现象，应该先拆开工具等待、就绪请求不足、缓存重算、排队延迟；14.4% 是用户提供的场景观察，本文没有复测该数值，也不能凭论文直接归因。

本文固定阅读 [Heddle v1](https://arxiv.org/abs/2603.28101v1)，首发 2026-03-30。以下加速数字均来自作者的 rollout 实验，不能转写成端到端训练加速或相同 GPU-hour 下的任务准确率收益。

## 2. 方法与实现机制

![Figure 8：Heddle 的数据面和控制面](./assets/papers/paper-heddle/method-source.png)

Figure 8 将控制面与实际执行分开：任务上下文进入预测器，预测结果影响优先级、路由和模型并行配置；数据面由推理 worker 和工具执行设施承担。阅读时沿着“轨迹更新 → 预测 → 调度/放置 → 下一次生成”追踪闭环，比只看模块名称更能解释为什么工具返回不应被当作无状态新请求。图中模型并行调整发生在 rollout 资源内部，不是为 OPD teacher 与 learner 自动分配 GPU。

**第一步：怎样知道谁会成为长尾？** 初始 prompt 的长度不能可靠预测多轮任务。相同 coding 问题可能一次通过测试，也可能多次修复。作者从历史轨迹构造 `(context, remaining_length)` 数据对，微调轻量 Qwen-0.6B 回归模型；每轮生成和工具反馈后更新估计。第一轮中的执行计划提供早期语义线索，后续真实执行信息逐步修正估计。预测器作为微服务与工具执行并行，目的是隐藏开销，但是否真正隐藏取决于工具等待时间，不能一概认为预测免费。

**第二步：预测怎样变成执行顺序？** PPS 把预计剩余长度作为优先级。工具返回时更新预测并插入 pending queue，按优先级重新排序；如果队头的优先级超过 active set 中最低者，则暂停后者、保留其 prefix cache，将高优先级任务放进活动槽位。这不只是队列重排，也包含 SGLang 内的抢占。抢占的对象是低优先级生成请求，任务本身没有被删除，后续仍可恢复。它是对 LPT 思路的动态近似，不是对所有含工具依赖、预测误差和缓存成本的调度问题证明全局最优。

**第三步：为什么还需要 placement？** 只改变顺序无法消除一个 worker 内的资源干扰。作者把一组轨迹的完成时间近似为“最大轨迹长度 × 单 token 基础时间 × 干扰系数”。用报告中的简化符号表示：

$$
C(g)=T\,F(g)\max_{\tau\in g}L(\tau),\qquad
\min_{g_1,\ldots,g_m}\max_i C(g_i).
$$

其中 $L$ 是预计轨迹长度，$T$ 是低并发情况下的 token 时间，$F$ 来自 batch 大小相关的 profiler 与模拟器。不能把 $F$ 理解成一个所有硬件通用的常数。作者在“同质 worker、长度已知、干扰只由组大小单调决定”的前提下证明，降序排列后的连续分组能够包含最优解，因此将搜索转成线性分区动态规划：

$$
D(i,j)=\min_{k<i}\max\{D(k,j-1),\ T F(i-k)L(\tau_{k+1})\}.
$$

$D(i,j)$ 表示前 $i$ 条轨迹分给 $j$ 个 worker 的最小预测 makespan。外层选切分点，内层取前面 worker 与最后一组的较慢者。复杂度为 $O(n^2m)$；短轨迹聚合减少实际问题规模。这些保证依赖简化模型，不应扩展成“运行时含任意异构 GPU 和预测误差也最优”。

**第四步：预测错了怎么办？** 路由器保留初始分区及排序信息。随着预测更新，轨迹的排序位置变化；各组的有效容量按剩余活跃轨迹数量缩放，路由器据此决定是否迁移，避免每轮重跑完整 DP。迁移利用工具执行间隙，使用 GPUDirect RDMA 传输 KV，降低重新 prefill 的成本。迁移请求按轨迹长度排序，并避免同一发送或接收端同时参与冲突传输。这里迁移的是仍在执行的轨迹及缓存，不是把历史样本重复放回训练 replay pool。

**第五步：模型并行为什么要动态变化？** 大模型并行度可降低长轨迹的逐 token 延迟，却减少独立副本数量，可能损害短任务总体吞吐。资源管理器在总 GPU 预算下联合考虑轨迹分组与 MP 配置，用模拟退火寻找近似方案。长轨迹倾向较高 MP，短轨迹倾向更多低 MP worker。收益取决于通信、重配置和工作量分布；不能把“扩大 TP”作为无需 profiler 的固定规则。

落地时有三个状态必须分开：轨迹身份与环境进度、推理 KV 的驻留位置、训练算法所需的组信息。前两项服务于调度效率，第三项决定数据何时真正可训练。即使 Heddle 提前完成某条轨迹，GRPO 仍可能要等它的兄弟样本才能算优势；而 token-level OPD 若不依赖组内奖励，其就绪边界可更细。这是对论文机制的场景推断，不是作者证明了 OPD 接线方式。

本次未发现可核验的作者官方仓库，因此实现核读限于原文 PPS 伪代码、Rust router 描述、DP 递推与资源管理流程。没有用同名开源工具代替官方实现，也没有声称运行过 15K 行系统。实际复现应先实现返回工具后的队列优先级和观测指标，再评估 KV 迁移与模型并行弹性；否则三个组件一起改动，会很难定位收益来源。[方法原文](https://arxiv.org/pdf/2603.28101v1)

## 3. 实验设置与算力

| 项目 | 原文披露 | 阅读边界 |
|---|---|---|
| 集群 | 8 台服务器、共 64 张 NVIDIA Hopper GPU；每节点 160 CPU 核、1.8 TB 内存 | Hopper 型号细节按原文，不自行补 H100 显存 |
| 网络 | 节点内 900 GB/s NVLink，跨节点 400 Gb/s InfiniBand、GPUDirect RDMA | KV 迁移收益依赖高带宽网络 |
| 软件 | PyTorch 2.8.0、CUDA 12.2、driver 535.161.08、Ray 2.49.0、ZMQ 4.3.5 | 论文环境，不是本地运行环境 |
| 模型 | instruction-tuned Qwen3 8B、14B、32B | 三档参数规模分别比较 |
| Coding | CodeForces、代码沙箱、测试与格式检查 | 不是 SWE-bench 仓库级修复 |
| Search / Math | HotpotQA 网页检索；DAPO-Math 计算器与 solver | 工具等待分布不同 |
| 采样 | 最大输出 40K token、每 prompt 16 个样本，温度 1.0、top-p 0.9 | GRPO 组规模与长尾相关 |
| 基线 | slime、verl、verl*，统一 SGLang 推理后端 | 比较的是实验时的调度方案 |

slime 基线采用 least-load 路由；verl 更强调缓存亲和性，将相关请求固定到 worker；verl* 在负载偏斜超过阈值时选择 least-load，否则保留亲和性。原文阈值采用最大/最小负载比超过 32。不能据此宣称当前所有版本的 verl 或 slime 都只有这一种路由策略，报告的对象是作者实验中的具体配置。

为公平比较，基线在 Qwen3 8B/14B/32B 上使用 MP 1/1/2、worker batch size 100，并匹配全局工作量。Heddle 额外利用动态配置优化相同预算下的执行。主指标为端到端 rollout token throughput：包括多轮任务生成过程，但不含完整 learner 的训练收益。控制面、工具面和推理面都在这个过程里起作用。若工具服务数量不同，或代码沙箱共享 CPU 饱和，即使 GPU 相同也不能认为对照等价。

原文单独测试预测准确性、调度、放置、资源管理与控制开销；组件实验保持其余设置相同。预测评价使用长尾识别 recall 与预测/实际长度的 Pearson 相关性。相关性高不等于每条轨迹剩余时长都准确，尤其应观察最末端任务是否被系统性低估。没有公开代码和完整任务清单时，这些结果应视为论文证据，不能视为本地可复现的完成状态。

对于自己的实验，建议额外记录每轮工具开始/结束时间、推理入队/出队时间、prefill token、KV 命中与迁移量、每 worker 活跃轨迹数。只有这些计时分解才能判断“并发不足”还是“已有可运行工作排队太久”。本报告完成的是原文和图表核读，没有申请或执行 64 GPU 训练。

## 4. 结果与图表解读

![Figure 12：不同模型和任务的 rollout 吞吐](./assets/papers/paper-heddle/throughput-source.png)

Figure 12 横向比较 coding、search、math 三类任务，每个任务覆盖三档 Qwen3 模型；纵轴是 rollout token throughput，柱子的颜色分别对应基线和 Heddle。应在同一任务、同一模型组内比较，不能把不同模型的绝对吞吐当成调度收益。作者汇总：相对 verl 提升 1.4–2.3×，相对 verl* 为 1.1–2.4×，相对 slime 为 1.2–2.5×。因此“最高 2.5×”具有明确分母，并不是相对任意异步训练系统的平均收益。

这张图支持轨迹级编排改善 rollout 效率，但不能单独回答训练多久达到相同准确率。它也没有说明 OPD 教师评分是否成为下一个瓶颈。若 rollout 快了而 teacher 的 ready queue 积压，新增吞吐就可能转成更大的策略滞后。迁移到三阶段 OPD 时，要把教师与 learner 的消费能力纳入全系统测量。

![Table 1：工具执行、长度预测与 KV 迁移的开销](./assets/papers/paper-heddle/table-1-pdf.png)

Table 1 的行是模型规模，列组是 coding/search/math 的工具时间、预测时间和迁移时间，单位是秒。以 14B 为例：工具分别约 0.41、1.41、0.046 秒；预测约 0.099、0.15、0.27 秒；迁移约 0.27、0.15、0.25 秒。search 工具等待较长，预测和迁移较容易重叠；math 工具只有几十毫秒，预测时间反而更长。因此“与工具并行”是一种隐藏成本的机会，不是每项工作负载都零额外延迟的实证结论。

表中 32B 的迁移时间约 0.35/0.27/0.33 秒，也提示模型大小和缓存规模会改变收益。对 KV 较短的任务，迁移的固定成本可能不划算；对长上下文任务，避免重算才可能更有价值。这里的原始表格裁剪保留了列组、单位和全部行，读者可以核对数字，而无需依赖本文重述。

单组件结果进一步限制了主图的解释：PPS 约改善 1.1–1.26×，placement 约 1.2–1.5×，资源调整约 1.1–1.3×。它们作用于相关的瓶颈，不能把各自最大值相乘推导系统总收益。控制面也不是零成本，placement 约几十毫秒，周期性的资源规划约数秒；应结合调用频率而不是只看单次耗时。

**面试如何解释长任务优先的反直觉？** 若目标是 batch 最后一条轨迹完成时间，长任务反复等待会留下最终尾巴，让其他 GPU 无事可做；提前推进长任务可能减少尾部空洞。若目标改成在线平均用户延迟，同一政策未必合适。正确答案需要先说清调度目标、任务是否可抢占、预测误差和饥饿风险，再讨论优先级。

**这是否证明慢轨迹应该永久优先？** 没有。PPS 持续更新预测，任务优先级会变化；实际系统还需评估长期公平性与估计误差。论文的主要证据是有限批次 rollout，不应把实验中的获益扩展成所有在线推理场景的普遍规律。[结果与开销原文](https://arxiv.org/pdf/2603.28101v1)

## 5. 局限、结论与后续阅读

Heddle 的贡献是把调度单位从零散请求提升为多轮轨迹，让剩余长度、缓存与模型并行资源共同参与决策。它尤其适合解释工具返回后排队累积的尾部延迟，但不解决所有训练侧屏障。GRPO 组内优势、teacher 评分、learner 数据就绪和权重广播仍是独立问题。

预测需要历史数据与额外微服务，任务分布变化可能降低可靠性；placement 的最优性依赖简化假设；KV 迁移依赖缓存布局、网络和推理后端能力；模型并行重配置也需要真实系统接口。论文 coding 场景是 CodeForces 沙箱，不等于保存文件系统、工具会话和外部副作用的长程仓库 agent。没有官方实现是当前复现限制，而不是方法无效的证据。

与 [TideRL](#paper=paper-tiderl) 对读，重点比较“长尾长度预测”与“可训练数据就绪”两个不同调度信号；与 [AReaL](#paper=paper-areal) 对读，区分 rollout 内部效率与 rollout/learner 解耦。实践中先测量工具返回后的队列延迟，再增加优先级 baseline；随后分别引入 placement 与资源弹性，记录相同 GPU-hour 下的任务结果。这样能判断调度收益是否真正穿过后续训练流水线。[固定版论文与作者信息](https://arxiv.org/abs/2603.28101v1)
