---
id: paper-sao
title: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning
paper_title: Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning
authors:
- Zhenyu Hou
- Yujiang Li
- Jie Tang
- Yuxiao Dong
affiliations:
- Tsinghua University
author_affiliations:
- &id001
  - 1
- *id001
- *id001
- *id001
venue: arXiv preprint (fixed v1)
year: 2026
direction: infra
areas:
- language
tasks:
- training-adaptation
- reasoning
- agents
published: '2026-07-08'
method_figure: ./assets/papers/paper-sao/method-source.png
method_caption: Figure 2 · Single-rollout asynchronous readiness
paper_url: https://arxiv.org/abs/2607.07508v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- SAO
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 单 prompt 单轨迹与 value model 移除 GRPO 组内等待；直接行为 logprob 双侧 masking 稳定异步数学与 coding RL。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

SAO（Single-Rollout Asynchronous Optimization）把异步训练中的一个屏障定位到算法本身：GRPO 为每个 prompt 生成多个回答，再用组内相对奖励估计优势。因此系统即使可以连续接收完成轨迹，也可能因为同组最后一条回答未完成而无法训练。这种依赖与全局 rollout/learner 同步屏障不同，不能只通过线程或队列解除。

**每个 prompt 只生成一条会怎样？** 失去组内 reward baseline 后，梯度估计可能方差很大。SAO 使用 value model 估计优势，以额外 critic 训练换取单轨迹交付能力。单条轨迹完成即可进入训练队列，不表示 learner 的 optimizer 只用一条样本；训练仍组合一定批量。它的“single rollout”指每 prompt 的采样数，不是 global batch size 等于 1。

另一个问题是 policy lag。异步生成期间 learner 可以多次更新，一条长轨迹甚至包含多版本 token。用当前最新旧模型重新 forward，并不必然还原当时真实行为分布。SAO 直接利用 rollout 时记录的 token logprob，与当前策略比较，并对过度偏离的 token 屏蔽梯度。

论文同时优化 value model：每次 policy 更新配两次 critic 更新，冻结 value model 的 attention，agent 场景的 GAE 跳过外部 observation token，并扩展 critic 预训练数据。这些机制共同构成结果，不能把质量提升全部归给“单 rollout 更快”。value model 的训练与推理也消耗算力，若与无 critic 的 OPD 比较，应把这部分成本算入。

本文固定 [SAO v1](https://arxiv.org/abs/2607.07508v1)，2026-07-08 首发，主要实验包括 Python 工具数学推理和 SWE-Bench Verified。它给出了 coding agent 实证，但并不代表同一 masking 策略可以未经推导迁移到 token-level OPD。后者的 advantage 来自 teacher/student logprob，与 RL 的终端奖励和 critic 不同。

## 2. 方法与实现机制

![Figure 2：单轨迹就绪与完整组等待的区别](./assets/papers/paper-sao/method-source.png)

Figure 2 中编号表示轨迹生成顺序。SAO 的一条轨迹完成就可以交付，GRPO 则须收齐同 prompt 组。图的重点是 trainable readiness，而不是把任务中途片段直接训练。模型生成还没结束时，最终 reward 和完整 value targets 未必可得，因此“单轨迹无组等待”与“逐 turn 提前 backward”仍是不同机制。

**DIS 的分母是什么？** 真实生成端记录的行为 token logprob。记当前策略为 $\pi_\theta$，生成引擎分布为 $\pi_{rollout}$，则：

$$
r_t=\exp\big(\log\pi_\theta(a_t\mid s_t)-\log\pi_{rollout}(a_t\mid s_t)\big).
$$

这避免保存所有历史 checkpoint 或使用一个“最新旧模型”冒充整条 mixed-policy 轨迹的行为模型。但是它要求 rollout logprob 准确对应实际采样、token 位置与动作 mask。量化、temperature、截断采样或不同 attention kernel 的数值差异，都需要在实现中说明，不能以“记录了 logprob”代替语义核对。

**双侧 masking 与 PPO clipping 差在哪里？** 原文定义：

$$
f(r)=\begin{cases}r,&1-\epsilon_\ell<r<1+\epsilon_h,\\0,&\text{otherwise},\end{cases}
\qquad L=\hat{\mathbb E}_t[f(r_t)\hat A_t\log\pi_\theta(a_t\mid s_t)].
$$

超出区间的 token 不参加梯度，而不是只把 ratio 数值压到边界。标准 PPO 的有效截断还与 advantage 正负有关；SAO 的校准在两侧都屏蔽，目标是拒绝过度偏离的样本。它是有意引入 bias 的稳定化方式，不保证所有过时轨迹仍可得到完整更新。阅读实现时还要确认校准系数如何 stop-gradient；原文形式表达的是加权 policy-gradient 目标，不能在没有代码证据时猜测作者对 ratio 的 autograd 细节。

SAO 去掉独立 old-policy forward，直接用 current/rollout ratio。在 decoupled PPO 中，behavior 与 proximal policy 各有职责，不能因为 SAO 报告效果好，就说这两层区分对所有 RL 算法都没意义。算法目标、value baseline、masking 和训练设置同时变化，比较时应做独立消融。

**没有 GRPO 组，优势怎么稳定？** 使用 value model $V_\phi$。critic 跟不上 policy 时，优势噪声会产生破坏性更新。作者设 $K=2$，每个 policy update 做两次 critic update；critic 学习率也更高。冻结 value attention、主要训练 MoE projections，是针对作者观察到 attention 层梯度较大的经验设计，不意味着所有 dense critic 或模型结构都适合相同冻结方式。

agent 轨迹由 action 与 observation 交替组成。模型不会生成工具反馈，若把 observation token 当作普通动作，GAE 会跨过错误的概率边界。SAO 的 skip-observation GAE 在当前 action 最后一个 token 与下一 action 第一个 token 之间连接 value：

$$
\delta=r_t+\gamma V(a_{i+1,0})-V(a_{i,N}),\quad
\hat A(a_{i,N})=\delta+\gamma\lambda\hat A(a_{i+1,0}).
$$

这里跳过的是外部反馈 token 的优势递推位置，并不是工具返回的内容不再影响下一轮模型状态。Observation 仍进入上下文。value head 的 token 对齐、terminal value、工具错误与真正任务结束的 mask 都会影响实现正确性。

作者也测试 step-level value，但其结果不如 token-level。step 粗粒度可以减计算，却失去每个动作内部细致监督；token-level 则增加存储与批处理复杂度。哪种更合适取决于轨迹和反馈稀疏性，不能用论文中的局部结果否定全部 turn-level 蒸馏设计。

**系统需要什么数据记录？** 单样本 ID、prompt/task ID、动作 token mask、实际行为 logprob、奖励、终止原因、版本切换信息，以及 critic 所需 targets。队列按完成轨迹交付，不必等兄弟样本；训练端按 token 比率 mask，然后用 value 优势更新。需要统计被 mask 的 token 比例和它们来自哪些任务，避免困难任务因版本跨度大而系统性失去训练覆盖。

本次未发现可核验的作者官方仓库，方法核读限于固定版公式、GAE 与训练描述。没有把第三方同名复现当作官方 DIS 实现，也没有声称验证梯度代码。用于 OPD 时，必须独立推导 teacher reward、reverse-KL advantage 与 ratio masking 的关系，并设计保留/屏蔽不同版本 token 的对照。[方法原文](https://arxiv.org/pdf/2607.07508v1)

## 3. 实验设置与算力

| 项目 | 原文设置 |
|---|---|
| Backbone | Qwen3-30B-A3B-Thinking-2507 |
| 数学初始化 | GPT-OSS-120B 生成 TIR 数据，SFT 3 epochs，初始化 policy 与 value |
| Batch | SAO 128 prompts ×1；GRPO 16 prompts ×8，均 128 trajectories |
| 最大长度 | 128K token |
| Policy | LR 1e-6，数学 epsilon low/high=0.3/5.0 |
| Critic | LR 5e-6、lambda critic=1、10-step warmup、每 batch 两次更新 |
| Policy GAE | 长度自适应 lambda=1−1/(1.5l) |
| Coding | 直接用原 backbone，epsilon low/high=0.8/3.0 |
| SWE scaffold | OpenHands，最多 300 turns、128K context |
| 算力披露 | 本次核读原文未定位到完整 GPU 型号/数量与训练 GPU-hour |

“high epsilon=5.0”在公式中的上界是 $1+5=6$，不能把它误写成 ratio 上界 5。coding 的范围对应约 0.2–4，数学约 0.7–6；两种设置不相同。value model 的预训练规模与策略初始化也影响结果，单独替换采样数不能保证复现质量。

数学评测为 AIME2025、BeyondAIME、HMMT Nov 2025、IMOAnswerBench 的 pass@1，温度 1、top-p 1、max 128K、最多 50 turns。AIME/HMMT/IMO 使用 16 次评测均值，BeyondAIME 为 4 次。这里多次评测均值仍是 pass@1 口径，不等于 pass@16。coding 评测使用 300 次交互上限，应与更短 agent budget 的结果谨慎比较。

表格还有商业模型行，属于外部质量背景，不能用它们证明系统执行更快或成本更低。SFT 前后、是否允许 Python、不同 RL variants 也有不同初始化意义。最有针对性的对照是同 backbone、相同 batch trajectories 的 SAO 与 GRPO+DIS，再用 value 更新频率和冻结消融分析原因。

论文重点证明异步长训练的稳定性与任务表现，并未提供像 Heddle 那样详尽的 rollout 调度速度账本。GPU 配置和总 budget 的缺口意味着无法从质量表判断额外 critic 是否被充分摊薄。本文保存该缺口，未自行补硬件规格，未执行 Qwen MoE 训练或 OpenHands 环境；图表和公式核对不等于算力复现。

## 4. 结果与图表解读

![Figure 3：训练中三个数学基准的准确率](./assets/papers/paper-sao/training-curves-source.png)

Figure 3 的三个面板依次为 AIME2025、BeyondAIME、HMMT Nov 2025；横轴是训练 steps，纵轴为准确率，比较 SAO 与 GRPO+DIS。SAO 在大多数训练区间较高，支持单采样加 value model 的组合在这些设置中有效。横轴不是墙钟或 GPU-hour，因此不能仅凭曲线更高宣称每小时收益更好，也不能据图估算未披露硬件的吞吐。

![Table 1：数学基准与算法对照](./assets/papers/paper-sao/table-1-pdf.png)

Table 1 中 SAO 为 97.3、74.8、88.3、74.0；SAO DIS-only 对应 94.2、71.5、86.7、71.3；GRPO+DIS 为 93.5、70.8、84.0、70.0。表明完整方案优于只打开 DIS 的变体，不能把提升全部归因于概率 ratio。对 critic 快更新的消融，AIME 从 97.3 降至约 95.0；value 全参数训练降至约 90.6。数学不同列的变化也不完全一致，稳定化机制并非同幅改善所有任务。

原表某些 SFT/工具行的分数变化很大，读者应按原表保留并核对对应初始化，不宜用它们构建“工具调用必然降低能力”的泛化结论。商业模型与本文训练配置没有统一预算，所以列在背景对照中，不作为 DIS 的消融证据。

![Table 2：SWE-Bench Verified 的 coding 结果](./assets/papers/paper-sao/table-2-pdf.png)

Table 2 原 backbone 23.0%，GRPO+DIS 27.0%，SAO 29.8%。最相关增量是相对 GRPO+DIS 的 2.8 个百分点，而非“提升 2.8%”；相对比例约 10.4%，若引用必须明确基线。这个结果让 SAO 比纯数学系统更接近 coding 场景，但一张最终质量表没有证明所有异步调度成本、环境恢复与难例覆盖都已解决。

**面试如何解释组内屏障？** GRPO 的优势依赖同 prompt 多条奖励，单轨迹完成还不足以确定归一化基准；SAO 用 critic 代替组内 baseline，使单轨迹独立可交付，但引入额外 value 学习成本和误差。不能说把 group size 改成 1 就得到了同一算法。

**为什么 masking 不等于无损校正？** 被拒绝 token 没梯度，残留样本的训练分布发生变化。mask 稳定训练可以有效，但需查看失去的是随机 token 还是困难、长程、恢复型任务。对于 OPD，这些 token 可能仍有教师纠错价值。实验需要同时记录 ratio 分布、mask rate、版本差和任务难度。[结果原文](https://arxiv.org/pdf/2607.07508v1)

## 5. 局限、结论与后续阅读

SAO 将组内等待、行为概率记录与 value model 稳定化一起处理，为单轨迹异步 agentic RL 提供一条实际算法路线。贡献不只是调度，更包含取消组内 baseline 后必须承担的 critic 训练设计。

局限是额外 critic 成本与预训练依赖、mask 带来的选择偏差、冻结策略对 MoE 结构的适用范围，以及缺少完整算力和官方实现。token-level ratio 也无法恢复未访问状态的覆盖，单条完成即可训练不等于任务中途片段可立即使用。

与 [APRIL](#paper=paper-april) 比较完整组续跑，与 [RolloutPipe](#paper=paper-rolloutpipe) 比较保留 GRPO 组但提前交付，与 [AsyncOPD](#paper=paper-asyncopd) 比较蒸馏 advantage 的重算。对原本无组内奖励依赖的 OPD，优先保留独立就绪边界，无须继承 GRPO 的屏障；是否使用 SAO masking 则必须另做 loss 推导与任务实验。[固定版论文](https://arxiv.org/abs/2607.07508v1)
