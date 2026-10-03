---
id: paper-flexmarl
title: Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning
paper_title: Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning
authors:
- Zhida Jiang
- Zhaolong Xing
- Jiawei Lu
- Yipei Niu
- Qingyuan Sang
- Liangxu Zhang
- Wenquan Dai
- Junhua Shu
- Jiaxing Wang
- Qiangyu Pei
- Qiong Chen
- Xinyu Liu
- Fangming Liu
- Ai Han
- Zhen Chen
- Ke Zhang
affiliations:
- JD.com
- Huawei
- Huazhong University of Science and Technology
author_affiliations:
- - 1
- - 1
- - 1
- - 2
- - 1
- - 1
- - 2
- - 1
- - 1
- - 2
- - 2
- - 2
- - 3
- - 1
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
published: '2026-02-10'
method_figure: ./assets/papers/paper-flexmarl/method-source.png
method_caption: Figure 2 · FlexMARL overall architecture
paper_url: https://arxiv.org/abs/2602.09578v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- FlexMARL
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 通过 agent 级体验存储、异步梯度累积和层次调度，让多智能体 rollout 与训练重叠；microbatch 提前 backward，global batch 齐备后统一更新。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

多 Agent 强化学习（MARL）让多个角色协作完成任务，并分别优化各自的策略。与“一个模型调用多个工具”相比，它还面临不同 Agent 之间的依赖、请求量偏斜、参数和训练配置差异。一次用户请求可能先进入主 Agent，再分流到库存、营销或售后角色，某些角色排队很长，其他角色却空闲。

直接复用单 Agent RL 框架有两种浪费。若 rollout 和 training 共置，两种阶段占用相同资源，只能频繁切换；若简单把两个资源池拆开，训练池又会一直等待整批多 Agent 轨迹。已有 [MARTI](https://github.com/TsinghuaC3I/MARTI)等工作将采样和分布式策略训练带入 MARL，异步 [AReaL](https://arxiv.org/abs/2505.24298v1)则允许消费旧版本数据。FlexMARL 关心能否在复杂角色交互中保留既定 on-policy 更新语义，同时提前让训练池做有用计算。

这里最重要的区分是 **backward 不等于 optimizer step**。一部分已完成样本可以在当前参数上计算梯度，先累计到缓存；等整个 global batch 的梯度齐了再更新。生成端仍使用同一个参数快照，不必因为提前训练就引入新的学生版本。它与 fully async learner 一边消费一边更新，是两个不同方案。

论文的正式标题是 Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning，FlexMARL 是系统名称。本文固定 2026-02-10 的 v1；主要实验是商业多 Agent 数据与 NPU 集群，不是通用开源 coding-agent 评测。对异步 OPD 的意义是建立一个保留更新边界的系统对照，不是宣称已经实现逐 turn 的 teacher / backward 交叠。

## 2. 方法与实现机制

### 联合编排与样本生命周期

![Figure 2 · Rollout、Joint Orchestrator 与 Training Engine](./assets/papers/paper-flexmarl/method-source.png)

**Figure 2 解读。** 左边根据 Agent 请求量并行采样、调整 inference instances；中间 Experience Store 存样本状态并驱动 microbatch 交付，global batch 完成后更新与同步；右侧让训练实例按 Agent 需要绑定硬件，并交换训练状态。底部两个独立资源池说明阶段解耦，但版本边界仍由中间统一管理。[图源：v1 系统总览](https://arxiv.org/pdf/2602.09578v1#page=4)。

Experience Store 按 Agent 分表，每条记录含 policy version、sample ID、已读未更新标记、用户定义的内容字段及字段 ready 状态。Sample ID 组合输入、turn 和 trajectory，使异步到达的数据仍可追踪；小元数据按值保存，大字符串 / tensor 按引用保存。因而 buffer 的含义不是“所有 token 放一个列表”，而是明确何时哪个 Agent 的样本已具备训练所需字段。

### Microbatch 提前计算，global batch 统一更新

![Figure 4 · one-step 异步与 microbatch 梯度累计](./assets/papers/paper-flexmarl/microbatch-source.png)

**Figure 4 解读。** 左面出现 stale parameters 的 rollout；右面绿色生成与橙色 Micro Batch GA 重叠，灰色竖条只在完整 batch 后进行参数更新 / 同步。右图说明提前做的是梯度计算，更新边界保留。它没有说明未完成轨迹中的每个 turn 都可以马上训练，也没有把需要完整组 reward 的数据依赖取消。[图源：v1 微批流水线](https://arxiv.org/pdf/2602.09578v1#page=6)。

令固定参数为 $\theta_k$，global batch 拆成 $M$ 个 microbatch，大小 $b_m$，总大小 $B$。在相同样本权重和损失定义下，下面是帮助理解作者 GA 思路的重述：

$$
g_k=\sum_{m=1}^{M}\frac{b_m}{B}\nabla_\theta L_m(\theta_k),\qquad
\theta_{k+1}=\operatorname{OptimizerStep}(\theta_k,g_k).
$$

例如 global 64、micro 16，可在第一批 16 个训练样本 ready 后先做 forward / backward；剩下三批仍在生成，所有梯度算完才更新。若不同 microbatch 有效 token 数不同，token-mean loss 不能机械地各取四分之一；需按实际 normalization 对齐。Global advantage normalization、GRPO 组内统计、跨样本辅助目标也必须在所需信息已齐后固定。上述条件是等价性成立的边界，不是“任何 loss 拆开都完全相同”的证明。

### 推理负载与训练硬件也一起调节

采样层使用 inter-query 与 intra-query 并行，让独立用户请求和无依赖的 Agent 分支重叠；有因果依赖的分支仍需等待。Hierarchical load balancing 分别处理 Agent 之间和同 Agent 实例之间的请求偏斜，以调整 inference instance 数目疏通热点角色。

训练层按 Agent 的实际训练需求绑定计算资源。Process group 负责一组分布式进程的 suspend / resume，训练状态在 device 与 host 间 swap，包括权重及 optimizer 状态，而非只迁移一份模型权重。状态越大，传输越贵，作者尝试把它藏在 rollout 的较长时间内；这与将 GPU 从 rollout rank 转给 trainer 的 TideRL 弹性路线有关，但管理单位不同。

### 源码状态与实现阅读

本次未发现论文作者提供的 FlexMARL 官方实现；原文引用的 [MARTI 仓库](https://github.com/TsinghuaC3I/MARTI)是前作，不能作为 FlexMARL 方法代码。这里依据固定 v1 的 §4.2 Experience Store、§4.3 Fine-Grained Asynchronous Pipeline、rollout / training engine 章节解释真实流程，不编造函数名。

若自行实现，关键状态应是每 Agent 的 policy version、完整 global batch 的训练成员、字段 ready 标记、梯度累计量和已读样本集合。重试要避免重复累计同一 microbatch；optimizer、scheduler、gradient clipping 与权重发布应留在统一更新边界。这些是从机制推导出的实现要求，并非已运行的复现。

## 3. 实验设置与算力

| 项目 | v1 公开配置 | 出处 / 复现限制 |
|---|---|---|
| 商业任务 MA | Merchant Assistant：经营分析、营销、售后等多角色协作 | 真实工业数据；详细内容因保密未公开 |
| 商业任务 CA | Category Assistant：订单、定价、库存建议 | 不能当作 SWE-Bench |
| MA 模型 | 各 Agent 独立 Qwen2.5-14B | 不共享参数，各自优化 |
| CA 模型 | Qwen2.5-14B / 32B | 角色参数规模不同 |
| 扩展测试 | 5×32B、3×32B+7×14B、15×14B 等 | Table 4，多策略模型部署 |
| 算法 | GRPO | 方法设计保持原目标和更新语义 |
| Batch | global 64，micro 16 | 早 backward、统一更新 |
| Optimizer / LR | Adam，1e-6 | §8.1；其他 optimizer 参数未完整披露 |
| 生成长度 | 最大 response 8,192 token | 实际长度分布不由上限推定 |
| 采样并行 | inter-query 4，intra-query 16 | 独立分支并行，非消除因果依赖 |
| 负载阈值 | disparity Δ=5 | 与本文 loss 无关的调度配置 |
| Seed | 2048 | 共享超参的系统对照 |
| 硬件 | 48 节点 × 16 commercial NPU，单设备 64GB | 型号未指明；不能填成 H100 |
| 节点互联 | HCCS，vendor SDK 经 PyTorch adapter | 厂商软件栈影响移植 |
| Baselines | MAS-RL、DistRL、MARTI | 同超参的多 Agent 系统，不与单 Agent 无条件比较 |
| 指标 | E2E latency、generated tokens/s、Agent request load、AI-core active time | 利用率是时间口径，不是 GPU MFU |
| 训练预算 | 主表记录 E2E 与 throughput；完整训练 steps / 总 device-hours 未披露 | 不能由单次延迟推总实验费用 |
| Code / data | 本次未发现官方系统代码；数据存在保密范围 | 静态核读，不声称公开可一键复现 |

这个账本解释了为什么不能把 7.3× 直接当成“同一 coding 任务在 slime 上也能快七倍”。主表的速度比基于较弱 MAS-RL 基线，多 Agent 商业任务本身有明显角色偏斜，硬件也有专用通信与进程资源控制。论文侧重系统效率，没有独立公开的质量 / GPU-hour 曲线证明在所有任务上保持相同收益。

## 4. 结果与图表解读

![Table 2 · MA / CA 的整体系统比较](./assets/papers/paper-flexmarl/table-2-pdf.png)

**Table 2 解读。** MA 中 MAS-RL 为 914.4s、MARTI 为 174.1s、FlexMARL 为 126.1s；7.3× 是相对 MAS-RL，相对 MARTI 只有约 1.38×（整理者以 174.1/126.1 计算）。CA 对应 438.6s、112.8s、78.8s，约 5.6× 与 1.43×。表中 throughput 为生成 token/s，MA 从 119.0 到 910.2 并不等同于延迟比；不同分母必须保留。[表源：v1 Table 2](https://arxiv.org/pdf/2602.09578v1#page=9)。

![Table 3 · 去除负载均衡和 microbatch 异步的消融](./assets/papers/paper-flexmarl/table-3-pdf.png)

**Table 3 解读。** MA 去掉 balancing 后 E2E 152.2s，去掉 async 为 256.2s，全系统 126.1s；CA 分别 95.9s、124.1s、78.8s。这比只读摘要更接近“提前 backward”实际贡献：与 w/o async 比，MA 的 E2E 减少约 50.8%，由 256.2/126.1 得到约 2.03×，而不是把所有 7.3× 都归给梯度累计。其余功能保持条件下的消融支持 microbatch 流水线，但仍不是逐 turn OPD 实验。[表源：v1 Table 3](https://arxiv.org/pdf/2602.09578v1#page=11)。

作者另报告 MA / CA 平均 active-core 利用率 32.4% / 19.8%，相较部分弱基线最高约 5.6×；这些绝对值说明收益来自缓解等待，并不表示硬件始终饱和。训练状态 swap 的 3B 到 32B 成本随规模增加，最大模型总交换约 11s，是否能隐藏取决于实际 rollout 时长。整体实验支持系统组合有效，仍需为新任务检查状态交换与尾部可重叠工作量。

**为何不能把消融当成严格数学等价证明？** Table 3 展示移除组件后的系统耗时，说明重叠确实回收等待，但没有展示逐样本梯度对比、全部 optimizer 状态或长期多 seed 收敛。提前计算的合理性还依赖固定参数与归一化条件；实际部署需同时测任务指标，避免只用延迟下降推断学习行为完全一致。

## 5. 局限、结论与后续阅读

FlexMARL 是“提前 backward、保留更新边界”最直接的系统近邻。可以把它作为严格版本一致的 baseline，与消费旧样本的 fully async OPD 分开。若自己的 loss 在单 turn 结束即可确定，仍需证明提前交付的字段、样本权重、global normalization 和环境依赖都成立；论文没有替 coding agent 的逐 turn 蒸馏完成这个证明。

工业数据未公开、设备型号及完整预算披露有限，公开源码本次也未发现，因此原文等价性描述与独立复现之间仍有距离。Gradient accumulation 的条件满足时有数学依据，但不同采样顺序、浮点归约、dropout 和 optimizer 实现可能改变数值；不能把“同一训练语义”写成 bitwise identical。

建议随后读 [RolloutPipe](https://arxiv.org/abs/2606.26997v1)，比较完整 GRPO group 提前交付与 microbatch GA；再读 [AsyncOPD](https://arxiv.org/abs/2606.24143v1)，分清提前评分 / 计算与真正引入版本滞后。本文核读 v1 系统、编排、实验与上述四张原图表，未部署商业 MARL。[固定原文](https://arxiv.org/abs/2602.09578v1)，整理日期 2026-10-03。
