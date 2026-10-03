---
id: paper-rolloutpipe
title: 'RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning'
paper_title: 'RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning'
authors:
- Rongjian Chen
- Jianmin Hu
- Kejiang Ye
- Minxian Xu
affiliations:
- Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
- University of Chinese Academy of Sciences
- Southern University of Science and Technology
author_affiliations:
- - 1
  - 2
- - 1
  - 3
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
published: '2026-06-25'
method_figure: ./assets/papers/paper-rolloutpipe/method-source.png
method_caption: Figure 2 · RolloutPipe complete-group architecture
paper_url: https://arxiv.org/abs/2606.26997v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- RolloutPipe
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 完整 GRPO 组就绪即交付，frontier 准入改善连续供给；区分 U-group optimizer 更新与整轮末尾的推理权重发布。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

RolloutPipe 针对分离部署的 Slime/Megatron/SGLang 训练流程：推理和训练 GPU 已经不同，但串行控制仍等整轮 rollout 完成才交付数据。早完成的 GRPO 组已经有完整回答、reward 与组统计，却被挡在 trainer FIFO 之外。训练资源空闲不一定因为没有可训练数据，也可能只是控制流程没有提前交付。

**为什么交付单位是完整组？** GRPO 对同 prompt 的 K 条回答使用组内奖励均值和标准差构造优势。一条回答即使已经结束，没有其余奖励仍不能确定组优势。RolloutPipe 保留这一算法依赖，优化的是跨组交付；它不拿半个组、未完成回答或某个 turn 直接训练。

系统有两项机制：CGP 在组可训练时立即 materialize 并交给 FIFO，让训练与剩余 rollout 重叠；FGD 优先投放构成下一训练 batch 的 frontier groups，让就绪数据更早、更稳定出现。它同时改变交付时间和请求准入顺序，但不修改基本 GRPO 公式。

**与全面异步系统差在哪里？** 当前 rollout 的权重固定，完成的组提前进入 learner；全部 R 组消费完后才向 SGLang 发布新权重。AReaL 或 AsyncOPD 可以让新数据和 learner 跨更新持续流动，RolloutPipe 仍保持一轮的权重发布边界。它研究较保守的流水线重叠设计，降低新增系统复杂度，但还保留组内屏障和整轮发布。

本文固定 [RolloutPipe v1](https://arxiv.org/abs/2606.26997v1)，2026-06-25 首发。原文称保留 on-policy semantics，更精确的阅读是“保留串行基线的生成版本和既定 logical update 规则”。一轮包含多个 learner optimizer steps，后面的 batch 相对当前 learner 仍可能存在基线本来就有的差异，不能把措辞理解成每个更新的数据都严格来自更新前最新权重。

## 2. 方法与实现机制

![Figure 2：Rollout、Control、Training 三节点路径](./assets/papers/paper-rolloutpipe/method-source.png)

Figure 2 左侧 FGD admission 把暂不属于 frontier 的组留在 deferred，中间响应收集、reward verifier 和 Build Trainable Group 把 serving-complete 变成 trainable-complete，右侧梯度累积与 optimizer step 消费逻辑 batch。Weight Publisher 最后把参数传回 SGLang。这三个节点是职责划分，训练 GPU 可以跨多台机器，不能据“三节点”推导总共只有三张卡。

**先定义三个数量。** R 是一轮 rollout 的 prompt 组数，K 是每组回答数，B 是每个逻辑更新的样本数，因此：

$$
U=B/K,\qquad A_i=\frac{r_i-\mu_g}{\sigma_g+\epsilon_{std}}.
$$

U 是一次更新消费的完整组数。实验 K=8、B=16，所以 U=2。R=32/64/96 代表整轮分别有 256/512/768 条回答，而不是一次更新 batch 也跟着变成这些数量。分清这些计数，才能判断“梯度提前累积但何时 optimizer step”。

**CGP 的 readiness 何时成立？** 不是模型一输出 EOS 就立即可训练。组内所有回答必须完成，reward 和 verifier 返回，样本转换、dynamic filter 与组统计也完成，才成为 valid training unit。每个 trainable group 进入 ActorGroup FIFO；feasible batch selector 取符合 token budget 的完整组前缀，送到 ready queue。单组过大时仍保持组语义，由动态 microbatching 适配，而不是随意丢掉组内某条回答。

当 ready queue 足够 U 组，trainer 启动 forward/backward 并累积，累计到 U 后 optimizer step。这与串行基线相同的 logical update 边界不同于 FlexMARL 的“全部 global batch microbatch 都齐才一次更新”。learner 可以在同一 rollout 轮里做多个既定更新，推理权重却一直固定到所有 R 组消费完。因此提前 backward、optimizer step、weight publish 要记成三个事件。

以时间表示 CGP 的主要空间：

$$
t_{start}^{serial}=t_{complete},\qquad
t_{start}^{CGP}=t_{first}^{(U)},\qquad
\Delta t=t_{complete}-t_{first}^{(U)}.
$$

$t_{first}^{(U)}$ 是首 U 组完成 materialization 的时间，$t_{complete}$ 是整轮结束时间。这个差是能够回收的空闲窗口，不保证全部都变成实际 wall-clock 节省：网络、后续 FIFO 空洞、trainer 算力与数据变换仍会限制重叠。

**FGD 为什么优先某些组？** default FIFO admission 不知道哪些任务共同构成下一训练 batch。FGD 按提交次序保留至多 $F_w$ 个尚未 serving-complete 的组：

$$
\mathcal F=\operatorname{argmin}^{F_w}_{g\in O}\operatorname{order}(g),\qquad
\operatorname{admit}(q)\iff g(q)\in\mathcal F.
$$

不属于 frontier 的请求进入 deferred，某组 K 条回答完成后退出集合，再补下一组。实验 Fw=U=2。这里优先的是最早提交、构成下一 logical batch 的组，不是 Heddle 的预测长轨迹优先，也不是按 reward 高低挑训练数据。任务顺序、并发和缓存命中仍可能受到影响，需要复现时记录。

![Figure 3：CGP 与 CGP+FGD 的交付时间线](./assets/papers/paper-rolloutpipe/timeline-source.png)

Figure 3 上图在首 U 组完成时就训练，后续组仍生成；下图 FGD 让前两组更早形成 batch，再释放后面的组。颜色表示组身份，灰条表示 update。虚线是整轮 rollout 完成点，说明训练在此前已开始，但不是说组内生成尚未结束也拿去更新。时间线没有统计坐标，应作为机制图而不是性能证据。

**“不引入额外 staleness”的条件是什么？** 推理的版本集合、每次 learner 的样本集合、优势计算和更新顺序都应与基线对应。原文同时提到完成 FIFO 和提交顺序 frontier，真实完成顺序可能不同；没有官方代码和完整 trace 时，不能声称逐浮点梯度与任意串行实现完全相同。更稳妥的是保留同类 logical update/发布边界、避免新增跨轮积压；若改变组顺序，训练动态等价还需验证。

本文未发现可核验的作者官方仓库，实现核读依据原文 Algorithm 1 的 FGD_Admit、Rollout_Worker、CGP_Handoff、Train，以及 ActorGroup/Feasible Batch 的文字描述。论文基于开源 Slime 不等于专门改动已公开；没有把 Slime 上游代码当作 RolloutPipe 的官方实现。通信和控制开销作者认为可以被重叠，本地没有执行多卡验证。

迁移到 token-level OPD 时，若 teacher feedback 不依赖完整任务奖励，合法交付单位可能更细，但必须重新检查 loss 目标和归一化：每个 turn 可训练不代表全任务 loss 可提前确定。RolloutPipe 提供的是完整组流水线近邻，不能替代逐 turn 提前梯度的数学和执行证明。[方法原文](https://arxiv.org/pdf/2606.26997v1)

## 3. 实验设置与算力

| 项目 | 原文披露 |
|---|---|
| 模型 | Qwen3-1.7B，相同 base checkpoint |
| Training GPU | 8×RTX4090 24GB，TP4、DP2 |
| Rollout GPU | 2×A100 PCIe 40GB，TP2 |
| 后端 | Slime、Megatron-LM、SGLang、Ray |
| 任务 | LSAT-AR、Sci-XW、Sci-JL、OlyPhys |
| R / K | 每轮 R=32/64/96 prompts，各 K=8 responses |
| Update | B=16 samples，U=2 groups；frontier width=2 |
| 对照 | native Slime、CGP、CGP+FGD |
| 重复 | 每点4 rounds均值、error bar为sample standard deviation |
| 奖励 | exact-match accuracy reward/verifier，生成与训练配置匹配 |

四个任务覆盖逻辑推理、大学科学与竞赛数理问题；它们不是多轮 tool agent 的端到端 benchmark。论文选择小模型是为了在有限 GPU 预算下扫描工作量，而大模型推广主要是机制层推断，不能写成已经验证 30B/70B 的同幅收益。

主指标 `rollout-to-train-end` 是从 rollout 开始到本轮训练结束的时间；dispatch timing 是首个合法 U-group batch 从 FIFO 交付的时间；trainer compute time 与 waiting ratio 用来解释重叠。该主窗口不必包含 weight publish、checkpoint、evaluation 和环境准备，因此与其他论文全运行墙钟不能直接比较。

waiting ratio 在方法和实验描述中有两种表达：首训练启动前等待占主窗口的比例，以及训练侧全部 wait/(wait+compute)。前者不计流水线中间缺料，后者计入，因此即使都叫 waiting ratio 也应按图表对应定义阅读。自己的记录建议保存每个等待区间，从 trace 重新计算，避免只用一个百分比掩盖不同空洞。

所有配置保持相同样本总量、GRPO 和 optimizer 路径，作者另外比较 response length 与 trainer compute time。这个控制有助于证明主要来自重叠，而不是简单少做训练。但 FGD 会改变推理 admission 和组完成次序，若需要严格逐样本更新一致，还要公开 sample IDs、queue order 与随机状态。

GPU 型号、TP/DP 和逻辑批量足够解释主实验，全部超参数、完整运行日志及专门代码仍未可核验。本文完成公式、伪代码与原图表核读，没有运行十卡实验。正式复现应计入 4090 与 A100 的异构资源成本，不能把“十张卡”视为十张同型号 H100。

## 4. 结果与图表解读

![主结果：四任务与三档 rollout 组数的窗口耗时](./assets/papers/paper-rolloutpipe/training-time-source.png)

Figure 4 解读：主图四行对应 LSAT-AR、Sci-XW、Sci-JL、OlyPhys，横轴为 R=32/64/96，纵轴为秒。柱子为 Slime、CGP、CGP+FGD，标注为均值，误差线为四轮样本标准差。LSAT R=96 从约1098秒降至647秒，减少约41%；R=32从363到251，约31%。变化与更大整轮留下更多可重叠工作相符。

作者汇总主窗口减少 30.7–42.3%。这是耗时减少百分比，若换成速度倍率约为1/(1−reduction)，不能直接写“加速1.423倍”。CGP 贡献总减少量的71–96%，FGD 在CGP基础上进一步减少2.5–11.4%；这些是分解与额外改善，不能相加或相乘成新主结果。

![Table 1：相同任务和 R 的 trainer compute time](./assets/papers/paper-rolloutpipe/table-1-pdf.png)

Table 1 的上下两块涵盖四任务，三档 R 与三个配置。LSAT R=96 为587.4/589.0/586.4秒，OlyPhys R=96 为640.0/640.0/644.4秒，三种执行的训练计算接近；主窗口却明显缩短，支持收益主要来自等待减少和重叠。表格保留完整列头和全部任务，不能只挑最优单行给出结论。

训练 compute 相近不是收敛等价的证明。浮点累积顺序、样本顺序、异步 RPC 和 filtering 都可能变化；原文主要是短轮次系统实验，缺少长训练的多 seed 质量轨迹。对最终任务表现与GPU-hour的结论，应保留证据范围。小模型和异构硬件也可能放大某些队列瓶颈，扩大模型后需重新测量。

作者还报告 trainer waiting ratio 下降37–76%，与首交付提前一致。不同工作量下，首交付提前可能很大，但若后续供给稀疏，trainer 仍会中间等待；FGD 的价值就是改善这部分连续就绪。因此单独看首batch时间不够，最好同时展示ready queue深度和训练忙闲trace。

**面试如何回答与FlexMARL的区别？** FlexMARL核心是固定参数下提前计算microbatch梯度，global batch齐后统一更新；RolloutPipe以U完整组为每次既定逻辑更新，可在同一rollout轮做多个optimizer steps，并等R组全消费后统一发布推理权重。二者都有重叠，但更新粒度、版本滞后定义和可交付单位不同。

**与AsyncOPD的区别？** AsyncOPD有持续rollout、teacher、learner流水线和旧数据loss校正；RolloutPipe保留整轮发布边界，重点释放已有完整组。Teacher独立服务也不是RolloutPipe验证的内容。原始机制与数字见[固定版结果](https://arxiv.org/pdf/2606.26997v1)，本地没有重测其速度。

## 5. 局限、结论与后续阅读

RolloutPipe 说明可以先解除“完整组已就绪却不能交付”的控制屏障，再考虑更复杂的跨版本异步化。CGP改变交付时间，FGD改善组供给，optimizer与发布边界必须分别记录。

限制包括完整组内等待、整轮权重发布、小模型短轮次证据、waiting ratio定义和缺少官方改动代码。论文关于on-policy的措辞应理解为保持其串行更新结构，而非任意时刻样本都来自当前learner；样本顺序和浮点等价还需代码及trace证明。

与[FlexMARL](#paper=paper-flexmarl)对读更新边界，与[SAO](#paper=paper-sao)对读取消组屏障的算法代价，与[APRIL](#paper=paper-april)对读未完成组buffer。若验证逐turn OPD提前backward，先确定合法的loss就绪条件，再用完整组方案作保守系统baseline，避免将粒度更细误写成已有论文已经证明的无偏更新。[固定版论文](https://arxiv.org/abs/2606.26997v1)

复现还应记录每个组的 materialization、optimizer step 与 weight publish 时间，以核对实际更新边界。
