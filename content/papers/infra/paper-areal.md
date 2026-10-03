---
id: paper-areal
title: 'AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning'
paper_title: 'AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning'
authors:
- Wei Fu
- Jiaxuan Gao
- Xujie Shen
- Chen Zhu
- Zhiyu Mei
- Chuyi He
- Shusheng Xu
- Guo Wei
- Jun Mei
- Jiashu Wang
- Tongkai Yang
- Binhang Yuan
- Yi Wu
affiliations:
- IIIS, Tsinghua University
- Ant Research
- HKUST
author_affiliations:
- - 1
  - 2
- - 1
  - 2
- - 2
- - 2
- - 1
  - 2
- - 2
- - 1
  - 2
- - 2
- - 2
- - 2
  - 3
- - 2
- - 3
- - 1
  - 2
venue: arXiv preprint (fixed v1)
year: 2025
direction: infra
areas:
- language
tasks:
- training-adaptation
- reasoning
published: '2025-05-30'
method_figure: ./assets/papers/paper-areal/method-source.png
method_caption: Figure 2 · 有界异步系统的数据与参数流
paper_url: https://arxiv.org/abs/2505.24298v1
github_url: https://github.com/areal-project/AReaL
code_note: Official repository, reviewed at a pinned current commit; this is not a claim of reproducing the paper experiments.
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- AReaL
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 以有界投放、可中断生成与 decoupled PPO 解耦 rollout / learner；重点区分 behavior policy、proximal policy 和单次消费 buffer。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

长推理 RL 的一轮通常是：策略生成回答，验证答案或运行代码，构造训练 batch，更新策略，发布新权重。同步组织要求整批回答完成后才训练。回答长度越不均匀，尾部剩下的少数请求越容易让大量推理卡和训练卡等待；把机器数加倍也不保证吞吐加倍，因为每卡 decode batch 可能更小，逐 token 读取权重的效率反而变差。

此前 one-step overlap 可以重叠相邻轮的生成与训练，却仍有完整生成批次屏障。[HybridFlow](https://arxiv.org/abs/2409.19256)提供 RL 角色编排，[DeepCoder](https://arxiv.org/abs/2504.01995)采用过相邻阶段重叠。AReaL 将问题推进到持续生产与消费：生成 worker 不必等同批的最慢请求，trainer 收到足够数据便更新。不过旧数据会与当前策略失配，一条尚未结束的回答还可能经历权重切换，因此系统加速必须与目标函数一起设计。

本文固定阅读 2025-05-30 的 v1，保留其 **最高 2.57×** 口径；后续 v5 摘要写 2.77×，属于另一固定版本，不能混在 v1 原图与配置中。这里讨论的是原论文系统及其算法；今天 AReaL 还支持更多训练后端与 Agent 接口，那些能力应另外核对当前文档。

最值得澄清的词是 replay buffer。原论文叫它 replay，但明确说明每条数据仅使用一次。它在此承担临时排队和组 batch，而不是无限回放旧经验。对异步 OPD，这意味着可以先搭建“生成一次、暂存、消费一次”的有界系统，之后再独立实验真正的 replay，避免同时改变系统和训练分布。

这种分层设计使它成为阅读后续异步系统时的基础参照：先辨认队列控制，再判断损失如何使用旧行为概率。

## 2. 方法与实现机制

### 从完整 batch 屏障改成持续数据流

![Figure 2 · AReaL 的角色、数据流与参数发布](./assets/papers/paper-areal/method-source.png)

**Figure 2 解读。** Rollout Controller 投放 prompt，把完成轨迹送给 Reward Service，带奖励的样本再经 buffer 进入 trainer；紫色箭头保存与加载参数，红色箭头触发生成中断与权重更新。生成和训练使用独立资源池，CPU 控制和验证服务也被画成独立角色。图中的“send full batch”是 learner 的消费单位，并不要求所有同时投放的请求一起完成。[图源：v1 Figure 2](https://arxiv.org/pdf/2505.24298v1#page=4)。

更新权重时，rollout worker 中断正在运行的生成，丢弃旧权重对应的 KV，使用新权重对已有 token 重算 prefix cache，然后继续 decode。Token 前缀保留，整条回答可能包含多个策略版本的片段。每个动作需要对应其真正生成时的行为概率，不能用最后一个版本的概率替换全序列。

### 两层控制：投放数量与 PPO 的信赖区域

原文令 $i$ 为 learner 版本，$B$ 为训练 batch，$N_r$ 为生成总量，以阈值 $\eta$ 控制投放：

$$
\left\lfloor N_r/B\right\rfloor\le i+\eta.
$$

Controller 拒绝会越界的新请求，并优先消费较旧轨迹。它限制产生多少尚未训练的工作；阈值过小会限流过度，阈值过大则让训练数据更旧。实现中的计数起点可能多一个当前 batch，阅读参数时要同时核对版本定义，不能仅凭变量名复制数值。

算法另外区分采样的 $\pi_{\rm beh}$、作为更新中心的近期 $\pi_{\rm prox}$ 和正在优化的 $\pi_\theta$：

$$
u_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm prox}(a_t\mid s_t)},\qquad
J=\mathbb E_{\pi_{\rm beh}}\sum_t
\frac{\pi_{\rm prox}}{\pi_{\rm beh}}
\min\!\left(u_t\hat A_t,\operatorname{clip}(u_t,1-\epsilon,1+\epsilon)\hat A_t\right).
$$

前面的比例处理“样本来自哪个策略”，clipping 中心处理“本次更新离近期策略多远”。普通 PPO 将两者都设成 old policy；旧样本落后较多时，会把当前学习拉回较旧策略。原实现不存昂贵 EMA 模型，而在 global batch 到达时，以更新前当前参数重算 proximal logprob。

例如行为概率 0.1、proximal 概率 0.2、当前概率 0.22，$u=1.1$，行为校正为 2。只把 2.2 塞进普通 PPO 的 clip，就会混淆这两个作用。对于跨版本回答，行为策略可按片段记录每个 token 的条件概率；这不意味着样本从当前策略整体生成，也不构成消除所有状态偏移的证明。

### 公开实现的源码伴读

核读当前仓库 commit `2fad2d0e308fe631e70e971b97188ad5c5cc03cb`，它已远多于 v1 的系统，以下是现代实现对照，不宣称恢复 2025 实验代码。[StalenessManager.get_capacity](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/areal/infra/staleness_manager.py#L79)同时计算并发空位和版本相关容量，再取较小值；运行数与已接受数一起计入，解释了为什么只限制 finished queue 长度还不够。

[grpo_loss_fn](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/areal/trainer/ppo/actor.py#L1031)读取 token 的 `old_logp`、`advantages`、`prox_logp` 与 `loss_mask`，解析 proximal 概率后进入具体 PPO loss；现代代码还包含 SAPO / CISPO / MOPD 分支，必须按配置追正确路径。[异步文档](https://github.com/areal-project/AReaL/blob/2fad2d0e308fe631e70e971b97188ad5c5cc03cb/docs/en/algorithms/async.md)要求 decoupled loss 配合概率重算，说明系统开关和算法开关应联合检查。本报告只做静态核读。

## 3. 实验设置与算力

| 项目 | v1 配置与证据 | 复现时要注意 |
|---|---|---|
| 模型 | DeepSeek-R1-Distilled-Qwen 系列，1.5B / 7B / 14B / 32B | 是已蒸馏 base checkpoint 再 RL |
| 数学训练 | DeepScaleR 开源训练集 | 不与评测 AIME 混同 |
| 代码训练 | DeepCoder 发布训练数据 | 同一任务的对照采用同数据 |
| 数学评测 | AIME24；另测 AIME25、AMC23、MATH500 | Table 1 / 2 和附录 |
| 代码评测 | LiveCodeBench，2024-08-01 至 2025-02-01 | 时间 split 影响对比 |
| Eval 生成 | 最长 32K；每题 32 个回答，平均 pass@1 | 不是 pass@32 |
| Batch | 512 prompts；每 prompt 16 answers；PPO minibatches 4 | prompt 数与轨迹数需区分 |
| 优化器 | Adam，LR 2e-5，WD 0.05，β=(0.9,0.95)，epsilon 1e-5 | 附录 Table 3 |
| 稳定参数 | PPO clip 0.2；grad clip 1；γ=λ=1；global advantage norm | critic 和 reference 关闭 |
| Reward | 最后 token 正确 +5、错误 −5 | 数学 / 代码可验证奖励 |
| 长度与解码 | prompt 1,024，最大生成 27,648；T=1、top-p=1 | 与系统 scaling 的 ctx 定义分别记录 |
| 默认过时量 | η=4，另扫描 0 至无界 | 原文按 learner 版本控制 |
| 精度 | 参数和 KV fp16；gradient / optimizer fp32 | 非一律 BF16 |
| 集群 | 64 节点，每节点 8 × H800；NVLink、3.2 Tbps RoCE | 不是每项都使用全部 512 卡 |
| 主结果用量 | 16 / 24 / 32 / 48 节点 | 对应四种规模，128–384 GPU |
| 分配 | inference 75%，training 25% | 固定启发式，不是自动最优比例 |
| Baseline 实现 | verl 2025-05-07 main；主要 SGLang 0.4.6 + FSDP，部分切 vLLM 0.8.4 | 后端差异也可能影响测量 |
| 时间 | 1.5B 1000 步 14.8h；14B 320 步 21.9h | 原文 Table 1，不是最低复现预算 |

系统 effective throughput 统计 PPO 实际消费的生成 token，经过 warmup。Scaling 实验的 context 是 prompt + response 总长度；不是把“32K context”和附录最大 response 长度视为同一个参数。作者也说明部分 baseline 的训练小时根据最新代码吞吐估计，而准确率标星引用前作已知结果；这种证据比“所有系统由作者同一次重跑”弱，应保留表注。

## 4. 结果与图表解读

![Table 1 · 数学与代码任务的最终效果及训练小时](./assets/papers/paper-areal/table-1-pdf.png)

**Table 1 解读。** 表上半比较 AIME24，下半比较 LiveCodeBench。1.5B 在同为 16 节点、1000 步时，Sync.AReaL 为 41.0h / 42.0，异步为 14.8h / 42.2；14B 代码任务为 48.8h / 56.7 对 21.9h / 58.1。它同时支持系统加速与相近最终效果，但不是每行准确率都提高：32B 为 61.2 对 61.0。星号准确率来自前作，7B / 32B verl 准确率空缺，不能补成零或直接宣布全任务效果排名。[表源：v1 Table 1](https://arxiv.org/pdf/2505.24298v1#page=7)。

![Figure 4 · 多模型、16K / 32K context 的 strong scaling](./assets/papers/paper-areal/scaling-source.png)

**Figure 4 解读。** 横轴 GPU 数，纵轴 effective tokens/s；上排 16K、下排 32K，列对应 1.5B / 7B / 32B。黑色虚线是理想线性 scaling，蓝色 AReaL 通常更接近它，橙色 verl 在部分规模扩展受限。左下大资源点体现原文最高 2.57× 的吞吐口径；32B / 32K 缺少 verl 点是 OOM，不能据此无限外推速度比。小资源 / 较短 context 条件也并非总有优势，生产与消费仍要平衡。[图源：v1 Figure 4](https://arxiv.org/pdf/2505.24298v1#page=8)。

算法消融的 Table 2 进一步解释“有界”和“校正”为什么都需要：η=4 时，AIME24 不用 decoupled objective 为 23.3，用后为 42.2，接近同步 42.0；η 无界即使用校正也只有 36.9。可中断生成单独带来约 12% / 17% throughput 增益，动态 microbatch 分配平均约 30%。这些是不同实验的局部收益，不能与 2.57× 相乘成整套系统加速。

**如何判断扩展收益是否有效？** 同时增加 GPU 会改变每个推理副本的并发与 trainer 更新时间，吞吐提高也可能增大版本差。Figure 4 因而要与 Table 1 的质量和消融一起读；最好报告相同 GPU-hour 的准确率，而不只选强扩展曲线上最大的比值。这里没有把论文测量转成自己的训练实测。

## 5. 局限、结论与后续阅读

AReaL 提供了有界连续生产、单次消费、token 行为概率和近期 proximal policy 的基础组合。最有用的设计是把控制积压放在新请求投放之前，而不是等 GPU 完成旧数据后再全部丢弃；但它的版本阈值仍是工程折衷，任务越长、更新越频繁，越需要检查真实版本分布和概率比。

v1 主要是数学与单轮代码推理，不是外部环境长程 coding agent 的完整验证。固定 75/25 分配也不能直接移到教师占卡明显的 OPD。其 buffer 单次消费，与经典 replay 不同；与 [GRPO / RLVR](#q=grpo-rlvr)一起读时，应记住本论文具体实现关闭 critic / reference，不能从 PPO 名字推断四模型俱全。

[AsyncOPD](https://arxiv.org/abs/2606.24143v1)的实验表明，AReaL 的 PPO 校正不能保证在蒸馏中最优；应先检查 OPD 的当前优势和缓存支持。这里核读 v1 正文、实施附录、三张原图表和现代源码接口，未执行训练。后续版本更新需连同配置和原图重审。[固定原文](https://arxiv.org/abs/2505.24298v1)、[官方仓库](https://github.com/areal-project/AReaL)，整理日期 2026-10-03。
