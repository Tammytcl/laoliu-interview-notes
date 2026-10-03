---
id: paper-april
title: 'APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation'
paper_title: 'APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation'
authors:
- Yuzhen Zhou
- Jiajun Li
- Yusheng Su
- Gowtham Ramesh
- Zilin Zhu
- Xiang Long
- Chenyang Zhao
- Jin Pan
- Xiaodong Yu
- Ze Wang
- Kangrui Du
- Jialian Wu
- Ximeng Sun
- Jiang Liu
- Qiaolin Yu
- Hao Chen
- Zicheng Liu
- Emad Barsoum
affiliations:
- Advanced Micro Devices, Inc. (AMD)
- Carnegie Mellon University (CMU)
- LMSYS Org
- University of California, Los Angeles (UCLA)
author_affiliations:
- - 1
  - 2
  - 3
- - 2
  - 3
- - 1
  - 3
- - 1
- - 3
- - 3
- - 3
  - 4
- - 3
- - 1
- - 1
- - 3
- - 1
- - 1
- - 1
- - 3
- - 1
- - 1
- - 1
venue: arXiv preprint (fixed v1)
year: 2025
direction: infra
areas:
- language
tasks:
- training-adaptation
- reasoning
published: '2025-09-23'
method_figure: ./assets/papers/paper-april/method-source.png
method_caption: Figure 2 · APRIL partial rollout workflow
paper_url: https://arxiv.org/abs/2509.18521v1
github_url: https://github.com/RLsys-Foundation/APRIL
code_note: Official repository, reviewed at a pinned current commit; this is not a claim of reproducing the paper experiments.
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- APRIL
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 超额投放、完成阈值与 continuation buffer：保留未完成前缀并跨轮续跑；区分 partial-resume、完整组交付与提前训练片段。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

APRIL 解决同步 RL rollout 的批次长尾：同一轮里有些回答很快结束，有些接近长度上限，前者结束后没有新工作补入，训练又必须等后者。因此推理 batch 逐渐变小、硬件效率下降。它的核心想法是超额投放请求，在收齐训练所需数据后暂停剩余生成，把前缀保留到下一轮，而不是等待所有最慢回答或丢弃全部前缀。

**partial rollout 是把半条回答拿去训练吗？** 在本工作主流程中不是。训练仍消费已经完成的回答；未完成轨迹进入 continuation buffer，之后继续生成，完成后才进入训练数据。保存片段、恢复生成和提前训练片段是三个不同动作。最后一项还需要提前可用的奖励、优势或蒸馏目标，APRIL 的 buffer 本身不提供这些信号。

**它与完全异步训练的关系是什么？** APRIL 保留按轮更新的结构：生成到阈值后中断，完成数据训练并更新权重，下一轮恢复旧前缀。它能减少全 batch 的尾部等待，但不自动表示 rollout 与 learner 在独立 GPU 上持续并行。AReaL 和 PipelineRL 则更强调持续生产消费和在途权重更新。APRIL 更适合作为容易解释的 partial-resume baseline。

GRPO 还需要同一个 prompt 的多个回答一起构造相对优势。作者实现按完整组交付：组中快回答可以保存，但仍要等同组慢回答结束。于是它消除跨 prompt 的全局长尾，未完全消除组内屏障。报告会将“数学推理中组内长度更接近”的经验，与“任意 coding agent 都不会有组内长尾”的推论分开。

本文固定 [APRIL v1](https://arxiv.org/abs/2509.18521v1)，首发 2025-09-23。作者讨论兼容 GRPO、DAPO、GSPO，主要展示的系统曲线是 GRPO 和 DAPO 的数学推理。最高 44% 是 rollout 吞吐改善，不是已经测得完整训练提升 44%，更不能据此断言 coding agent 的文件系统可恢复问题已经解决。

## 2. 方法与实现机制

![Figure 2：超额请求、阈值停止、前缀缓冲与恢复](./assets/papers/paper-april/method-source.png)

Figure 2 上半部分是常规同步流程：每轮等全部回答结束，再训练。下半部分在足够数量完成时切出，灰色箭头代表以前步骤保存的未完成前缀，蓝色代表本轮新增生成，buffer 把未完成对象接到下一轮。图中的 update weights 位于轮次边界，因此一条最终回答可以包含多个权重版本的 token；保留前缀不能让这些 token 自动变成当前策略采样。

完整状态转移是：投放 $N'>N$ 个实例；达到目标完成量时发 abort；已经完成的回答交付 learner；剩余对象记录前缀和状态；下一轮优先从 buffer 取出并恢复，再补充新实例。若按 GRPO 组交付，$N$ 的语义还必须分清是 prompt 数还是回答数。实验中 32 prompts × 8 samples 对应 256 trajectories，超额 64 prompts 对应 512 trajectories；代码的完整组边界比“最先完成任意 256 条”更严格。

为了说明版本问题，用本文记号写最终轨迹的行为分布：

$$
\mu(y\mid x)=\prod_t \pi_{v_t}(y_t\mid x,y_{<t}),\qquad
\rho_t=\exp\{\log\pi_\theta(y_t\mid h_t)-\log\pi_{v_t}(y_t\mid h_t)\}.
$$

$v_t$ 表示该 token 实际生成时的版本，可能随恢复而改变。这个分解用于解释记录行为概率的必要性，不是声称 APRIL 提出了新的无偏混合策略梯度定理。局部 ratio 只校正给定前缀上的动作分布，早期 token 导致的状态分布差异仍然存在。仅用回答结束时的版本重新计算所有旧 token，会失去真实采样分母。

**buffer 需要保存什么？** 单轮文本至少需要 prompt、已生成 token、结束状态、组 ID、奖励状态和剩余 token 预算。多轮 agent 还需要环境状态：文件改动、进程、工具会话、用户模拟状态、随机种子和外部操作记录。KV cache 可降低恢复 prefill 成本，但旧参数下的 KV 与更新后模型不完全一致，不能把“prefix 可恢复”与“旧 KV 精确可复用”混为一谈。

官方实现核读固定 [APRIL commit adbb017](https://github.com/RLsys-Foundation/APRIL/tree/adbb0175f010d66cdae215df8ffaca76aa142de6)。该仓库使用 slime 与 SGLang，以下是静态阅读，并未运行训练脚本。首先，[generate](https://github.com/RLsys-Foundation/APRIL/blob/adbb0175f010d66cdae215df8ffaca76aa142de6/slime/rollout/sglang_example.py#L71) 针对已有 response 继续生成，将前缀接到输入并调整剩余长度。它恢复的是生成上下文，不是一个任意环境的 snapshot。

其次，[generate_and_rm_group](https://github.com/RLsys-Foundation/APRIL/blob/adbb0175f010d66cdae215df8ffaca76aa142de6/slime/rollout/sglang_example.py#L155) 使用 gather 等整组样本，必要时再做 group reward。这处明确保留组完成依赖。第三，[abort](https://github.com/RLsys-Foundation/APRIL/blob/adbb0175f010d66cdae215df8ffaca76aa142de6/slime/rollout/sglang_example.py#L174) 向 worker 发送 abort，随后收集 pending tasks，把 partial group 放回 data buffer，并记录首次 rollout ID。中断返回前必须等待请求状态收敛，否则容易丢掉最后生成的 token 或重复入队。

最后，[generate_rollout_async](https://github.com/RLsys-Foundation/APRIL/blob/adbb0175f010d66cdae215df8ffaca76aa142de6/slime/rollout/sglang_example.py#L211) 从 buffer 取数据、投放任务、FIRST_COMPLETED 收取已完成组，并在目标达到后 abort 剩余请求。dynamic sampling 和 over-sampling filter 会改变有效样本计数，因此准入数量、已完成数量和被训练使用数量需要分别记账。代码还有未用完样本回收的 TODO，说明不能只看论文框图便认定所有边界都已完备。

**为什么这不等同于 replay？** continuation buffer 存的是仍待完成的任务，不意味着已训练数据再次抽样。一个轨迹可多次暂停，但最终消费一次，仍属于生成过程复用。只有同一完成样本重复参与多个更新，才进入经验重放问题；届时优先抽样会改变样本概率，还需单独考虑校正。APRIL 的前缀续跑不能当成经典 prioritized experience replay。

对 coding 场景，最小可审计实现应给轨迹固定 ID，明确 pending/paused/completed/consumed 四个状态，保存各段行为 logprob 和策略版本，并通过 sample ID 防止重复消费。先验证无环境的文本续跑，再验证可快照沙箱，最后才扩展到外部副作用。这里是迁移设计建议，不是作者实现已经验证的 coding 系统。

## 3. 实验设置与算力

![Table 1：论文原始超参数](./assets/papers/paper-april/table-1-pdf.png)

原表左侧列 rollout 和 partial 配置，右侧列优化器。`rollout_batch_size=32` 是输入 prompt 数，`n_samples_per_prompt=8` 才得到 256 条回答；`over_sampling_batch_size=64` 是两倍 prompt 超额投放。把 32 写成训练回答 batch 会把真实工作量缩小八倍。训练 response 上限 16384，而长尾分布诊断单独使用 32768，不应混用。

| 项目 | 固定版设置 |
|---|---|
| 模型 | Qwen3-4B、Qwen3-8B |
| 数据 | DAPO-Math-17k、DeepScaleR、DeepMath-103K |
| 主展示算法 | GRPO、DAPO；正文另讨论 GSPO 适配 |
| 算力 | 单节点 8×AMD MI300 或 8×NVIDIA H100，主图主要为 AMD 配置 |
| Batch | 32 prompts × 8，global batch 256；超额 64 prompts |
| Sampling | 温度 0.8，response max 16384 |
| 优化器 | Adam，LR 1e-6，weight decay 0.1，betas 0.9/0.98 |
| Loss | KL coefficient 0，clip 0.2，high clip 0.28 |
| 评估 | AIME2024，16 samples，max 16384，top-p 0.7 |

吞吐定义为每次 rollout iteration 的生成 token 数除以 wall-clock。它不直接扣除所有后续未消费 token，也不同于 TideRL 的最终训练 consumed token/s。因此比较不同论文时，不能只把数值放在同一列而省掉定义。APRIL 的超额投放还需要更多暂存容量，实际是否节省 GPU-hour 要结合训练计算、恢复 prefill 与完成数据质量。

文中跨平台说明存在标注差异：叙述提到 H100，而后续硬件图的 caption 写 H200。本报告主结论采用主要 AMD 曲线与原始配置表，不把硬件图统称 H100 复现实验。具体版本、驱动、全部训练时长与随机重复信息仍需结合官方脚本核对；本地只下载源码和资产，没有运行 AMD/NVIDIA 训练。

组内控制会缓存已完成回答直到其余同组回答完成，作者观察数学任务中组内长度方差比跨 prompt 小。这只是对应三类数学数据的经验结果。对于同一问题有的 coding agent 一次修复、有的反复重试，组内方差可能显著增加，GRPO 组屏障必须重新测量。

## 4. 结果与图表解读

![Figure 4：两档模型、两种算法与三组数据的吞吐曲线](./assets/papers/paper-april/throughput-source.png)

这组图保留原文 12 个面板：列依次为 DAPO-Math、DeepScaleR、DeepMath；行依次为 4B GRPO、4B DAPO、8B GRPO、8B DAPO。横轴是训练 step，纵轴为 rollout throughput；蓝线为 APRIL，橙线为非 partial baseline。各面板自身坐标范围不同，应在面板内比较，不应以视觉高度跨面板计算倍率。

作者汇总 4B GRPO 的提升约 24%、31%、35%，4B DAPO 为 8%、11%、10%；8B GRPO 为 26%、35%、44%，8B DAPO 为 8%、8%、10%。最高 44% 对应 8B、GRPO、DeepMath，不是所有配置的统一收益。DAPO 提升较小也说明原算法和采样流程已影响长尾空间，系统优化的收益并非固定常数。

曲线的局部尖峰不能替代长期平均。后期 response 长度增长会改变 token/s 的分子与批次时间，最好同时观察 completed trajectories/s、consumed tokens/s 和任务评估。作者的 AIME 曲线多数接近，有些配置末期高约 7–8 个百分点，但这些现象不能证明旧数据一定提高泛化，也不能将“未看到不稳定”当成任意 staleness 下的保证。

![Figure 6：混合策略轨迹的比例](./assets/papers/paper-april/off-policy-source.png)

Figure 6 横轴为 rollout step，纵轴是原日志中的 partial/off-policy ratio。曲线长期在约 0.3–0.4 区间波动，表明恢复的数据不是偶发边缘情况。原文 caption 称 hybrid-policy rollouts proportion，随后正文又用约 40% 的继承 token 描述；两种统计对象并不完全等价。因此本文将它作为混合策略数据占比可观的证据，不据该图推导精确 token-level staleness 分布。

作者未观察到完成一条回答需要超过五个连续策略版本，因而对更深版本跨度没有实证保证。如果 coding 轨迹跨数十次 learner 更新，风险可能不同。需要记录每段版本、最终训练时版本差与 ratio 分布，不能只报告“partial 占比 40%”来说明训练安全。

**面试追问：abort 后的任务是失败样本吗？** 不是，调度中断和任务失败是不同终止原因。abort 对象继续保存，下一轮优先恢复，只有达到真实任务终止或长度限制才成为最终轨迹。奖励计算必须知道终止原因，否则会把暂停误记成失败，改变训练分布。

**它能完全消除慢任务吗？** 不会，慢任务仍然消耗计算，只是跨轮推进，不再阻塞当前全局批次；完整组仍可能等待。应观察慢任务最后是否真正训练、是否跨多轮仍持续滞留、是否产生长度相关选择偏差。上述机制解释来自固定版原文与官方代码，性能数字是论文报告，本地没有复测。[结果原文](https://arxiv.org/pdf/2509.18521v1)

## 5. 局限、结论与后续阅读

APRIL 是清晰的超额投放与 continuation buffer baseline：提前交付完成组，保留慢轨迹，并在下一轮优先续跑。它证明数学推理场景下少量混合版本与实用吞吐改进可以共存，但没有解决任意旧数据的理论偏差，也没有证明 agent 环境状态可无损恢复。

主要边界包括超额工作占用、恢复 prefill、组内屏障和版本跨度。按最快完成顺序选数据可能改变短期训练分布；保留慢轨迹能缓解丢弃，却需要验证最后确实被消费。GSPO 的兼容性讨论不能替代全部 GRPO/DAPO 曲线之外的等规模实验，跨平台 caption 的差异也需要复现时核实。

与 [AReaL](#paper=paper-areal) 比较持续背压，与 [PipelineRL](#paper=paper-pipelinerl) 比较中途换权重，与 [SAO](#paper=paper-sao) 比较组内完成屏障。用于 OPD 时，应再结合 [AsyncOPD](#paper=paper-asyncopd) 检查 teacher cache 和当前学生概率；保存前缀只是系统条件，loss 的正确性还需独立验证。[论文](https://arxiv.org/abs/2509.18521v1)、[官方实现](https://github.com/RLsys-Foundation/APRIL)
