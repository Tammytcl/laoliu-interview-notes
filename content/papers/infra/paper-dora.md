---
id: paper-dora
title: 'DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training'
paper_title: 'DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training'
authors:
- Tianhao Hu
- Xiangcheng Liu
- Youshao Xiao
- Yang Zheng
- Xuan Huang
- Jinrui Ding
- Yufei Zhang
- Tao Liang
- Hongyu Zang
- Quan Chen
- Yueqing Sun
- Wenjie Shi
- Chao Zhang
- Wei Wang
- Qi Gu
- Yerui Sun
- Yucheng Xie
- Xunliang Cai
affiliations:
- Meituan, China
author_affiliations:
- &id001
  - 1
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
- *id001
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
published: '2026-04-29'
method_figure: ./assets/papers/paper-dora/method-source.png
method_caption: Figure 6 · DORA multi-version streaming timeline
paper_url: https://arxiv.org/abs/2604.26256v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- grpo-rlvr
tags:
- DORA
- RL
- Async Training
- Rollout
updated: '2026-10-04'
summary: 多版本并存保留慢轨迹，单轨迹固定原策略；窗口、动态 DP 分区和同版本 KV 迁移兼顾新鲜度与资源效率。
template_version: 5
depth_standard: ddpm
draft: true
---

## 1. 背景与已有工作

DORA 研究长序列 RL 的两类空洞：一个推理实例中快回答结束后只剩少数慢回答，以及多个实例间工作量不均导致部分实例空闲。解除同步屏障可以减少等待，但作者希望同时保持整条轨迹内部使用同一策略版本、不丢弃长轨迹和有界数据新鲜度。

**不在中途换权重，怎样让 learner 继续更新？** 同时保留多个 rollout 权重版本。不同 DP group 承载不同版本，旧任务在原版本上继续，完成的任务交付给 learner；新任务优先进入最新版本。这样不必等待每条长序列结束才开始下一次训练，但需要管理旧模型资源与迁移。

论文提出三个约束：C1 轨迹内策略一致、C2 保留轨迹完整性、C3 控制过时。它们不是完全严格 on-policy 的同义词。即使整条轨迹用固定旧版本，learner 消费时已经更新，仍是旧行为分布。DORA 保持单轨迹行为版本清楚并限制版本窗口，而不是证明旧数据完全无偏。

**与 partial rollout 的关键区别？** APRIL 保存旧前缀后用新权重续跑，PipelineRL 更频繁地在途更新；DORA 让旧前缀与后续 token 一直来自原模型。因此同版本实例之间可以迁移 KV 而无需因参数变化重算，但多版本会造成资源碎片：旧版本只剩一两条任务仍占一个模型副本。

作者用动态 DP 分区与请求迁移解决碎片，以 sliding window 控制滞后。当最旧版本仍有未完成任务时，窗口不能继续无限向前；不丢慢轨迹的约束仍可能形成局部背压，这是算法完整性与系统吞吐的折中，不应省略。

本文固定 [DORA v1](https://arxiv.org/abs/2604.26256v1)，首发 2026-04-29。原文只有图，没有可登记的原始表格；按当前仓库的原表证据要求，单篇暂保留完整草稿，专题可以阅读固定版论文。配置账本是本文转录，不冒充作者原表，所有速度数字按 dense、production rollout 和收敛实验分别列明。

## 2. 方法与实现机制

![Figure 6：多版本 streaming rollout 的执行时间线](./assets/papers/paper-dora/method-source.png)

Figure 6 顶部是 RolloutManager、TransferQueue 和 Trainer，下方 DP 行用不同颜色表示策略版本，编号表示请求。竖虚线标权重更新边界，斜箭头标请求迁移。跨越训练步骤的慢请求仍保留原颜色；新版本在释放出的 DP 资源上接受更多新任务。这张图最直接解释“整轨迹固定版本”与“整个系统停止更新”没有必然绑定关系。

**数据何时进入 learner？** 每轮超额投放 RBS>TBS 的生成请求，收齐 TBS 完成样本后训练。未完成任务不会 abort 或删除，继续在旧版本 group 上执行，完成后流入后续 train batch。TransferQueue 记录版本和 staleness；它是交付队列，不能据名称推断历史样本重复 replay。

**版本窗口怎样限制最旧工作？** 活跃版本集合为：

$$
W=\{w_j,\ldots,w_{j-K+1}\},\qquad |W|\le K.
$$

只有最旧版本的轨迹都已收集并交给训练，窗口才能前移。K 控制新鲜度与并发的折中。这里“已交给训练”不应随意改成“所有 optimizer 已更新完”的更强定义，版本数量与版本差也有计数边界；复现需要按原始事件规范记录。K=1 和 K=3 的比较不能直接照搬到另一框架不同计数方式的 staleness 参数。

**多版本怎样避免浪费副本？** orchestrator 观察每个版本 active/pending request 数、KV 利用率与生成进度。训练结束、KV 压力阈值或周期时间触发重分区。按原文近似分配：

$$
DP_w=\operatorname{Round}\left(DP_{total}\frac{R_w}{\sum_{w'}R_{w'}}\right).
$$

$R_w$ 是版本剩余工作量代理。比例分配后通过 P2P 权重传输重构 group，把请求和状态迁移到同版本的目标 group。实际实现还需要处理整数舍入、每个非空版本最低副本数、TP/PP/EP group 的可行粒度；原文简式没有覆盖所有边缘情况，不能据公式声称任意GPU数都可无碎片分配。

新数据先补给最新版本，达到 RBS 高水位后再机会式填补旧版本剩余空槽。因此“优先最新”不等于旧版本永不接新请求。旧版本补充策略会影响最旧版本何时清空，窗口前移协议应与补充一起验证，避免为了短期设备利用率而持续制造旧任务。

**为什么 KV 可以迁移？** C1 保证同一轨迹始终用同一权重。相同模型参数、输入、缓存格式与兼容执行语义下，目标实例可以接收原KV继续生成，避免完整prefix重算。数学兼容性是前提，物理传输仍耗带宽；不同TP布局、dtype、量化或后端实现还需转换或重新prefill。不能把“无需重算”写成“零传输成本”。

迁移分两阶段：轻量RPC传request ID、已decode长度、版本tag等元数据，大体积KV通过高性能通信传输。locality-aware调度尽可能保留原rank，只移动必须搬的任务；host memory offload缓解长上下文显存压力。缓存offload、请求迁移与策略版本管理应使用同一个轨迹ID，防止恢复错模型或重复生成。

**算法校正在哪里？** DORA的主要选择是保持轨迹版本一致和窗口限制，减少需要针对混合版本修改算法的程度。它没有给出AsyncOPD那样的教师缓存、current-student reverse-KL重算分析。多版本保留解决了行为模型可追溯性，并未消除behavior与learner之间的差异。对OPD仍需分别核对prefix distribution、teacher logprob和当前学生advantage。

**GRPO组的交付怎么办？** 论文示意采用trajectory-level streaming，但实验每prompt采16条，DAPO类优势依赖组统计。实际trainable边界必须保证组奖励等需要的数据齐备；仅凭框图中的单样本队列不足以证明任意一条完成就可更新。公开文字和伪代码没有完整代码接线，因此报告明确保留这一复现检查项，而不自行补出未披露算法。

本次未发现可核验的官方DORA实现。Meituan生产系统的RolloutManager、TransferQueue、Trainer与load-balancing流程是原文描述，不是已在开源仓库定位的类。代码核读限于Algorithm与RPC/缓存说明；未下载或执行内部训练脚本。对自己的系统，应先固定版本+有界队列baseline，再加同版本迁移，最后加入动态分区，独立测每项成本。[方法原文](https://arxiv.org/pdf/2604.26256v1)

## 3. 实验设置与算力

| 项目 | 原文披露 | 口径 |
|---|---|---|
| Dense集群 | 16节点，每节点8×H800，最多128GPU | 分别测试64/128 |
| 通信 | NVLink约400GB/s，跨节点8×400Gbps接口 | 强网络条件 |
| Dense模型 | Qwen2.5-32B | 中间checkpoint，均值response约2.4K、最大30K |
| 数据 | DAPO-Math-17k，input max2K、output max30K | 数学推理 |
| 算法 | DAPO类GRPO，每prompt16responses | 需组统计 |
| Batch | 512prompts、8192responses；每iteration16updates、microbatch512 | update/iteration勿混 |
| 软件 | CUDA12.4、PyTorch2.6、vLLM0.8.5、NCCL2.28、Megatron、torchRPC扩展 | 内部统一框架 |
| 系统对照 | all-colocated sync、one-step off-policy、partial rollout | 同硬件软件实现 |
| Production | LongCat-Flash 560B total MoE、4096非CUDA accelerators、max64K | 可用device memory约60GB/卡 |
| 收敛 | 72GPU、100trainingsteps、K=1/3 | 与效率实验规模不同 |

Dense吞吐是prompt与response总处理token/s，计入两种token。AsyncOPD常报告被learner消费的response token，APRIL计生成token，TideRL计最终训练消费token；这些口径不能直接排成一张速度排行榜。step time是每RL iteration墙钟，与16个内部update也不是同一数量。

效率数字在warm-up之后取五次RL iterations均值，主要证明稳态系统表现。收敛图另在72GPU上跑100steps，不能把128GPU最快吞吐自动绑定到这条质量曲线。production使用非CUDA卡，不能转写为4096H800，也不能只以总参数560B估计实际每token激活算力。

对照在同一内部框架中实现，有助于隔离执行范式，却与公开AReaL或APRIL原始实现不完全等价。partial rollout采用作者内部all-colocated架构；“比partial快”应说是该实现，而不是全面压过所有partial系统。大规模production因成本只比较同步基线，没有同规模全部异步对照。

本文配置账本由原文文字转录，是辅助阅读证据；作者未提供原表，不以自制表截图伪装原表。官方代码、完整任务实例、optimizer全部细节和总GPU-hour仍存在公开缺口。当前工作是固定原文、原图核读和复现项梳理，没有执行64–4096卡训练。

## 4. 结果与图表解读

![Figures 8、9、10：dense时间/吞吐与production rollout加速](./assets/papers/paper-dora/throughput-source.png)

三个面板来自原文三个独立图，按顺序组合，并非作者一个新编号的Figure。左图纵轴step time（分钟），64GPU同步22.91、DORA14.67，约1.56×；128GPU同步20.23、DORA10.46，约1.93×。柱内斜线区域标不能被重叠的rollout-only占比，从同步65%/73%降到DORA12%/24%，它不是整台GPU的利用率。

中图纵轴为总token throughput：64GPU DORA23327，对同步14143约1.65×，对partial19872约1.17×；128GPU DORA34135，对同步16070约2.12×，对partial约1.11×。time倍率与throughput倍率不同，因为response长度分布和prompt计数影响分子，不能认为二者必须精确互为倒数。

右图是4096卡LongCat-Flash生产配置的rollout speedup：math/TIR约3.6×，Tau2+Vita agentic约6.2×。这是rollout指标，不是dense的完整step，也不是6.2×端到端训练。生产部署另报告2–4×端到端速度，证据口径与样本公开程度都不同，应单独归为作者部署报告。

![Figure 11：不同执行范式与版本窗口的训练reward](./assets/papers/paper-dora/convergence-source.png)

Figure 11横轴trainingsteps，纵轴平均训练reward。K=1与K=3都稳定上升，K=3略慢于K=1；sync也有接近趋势。它支持有限窗口在作者设置中没有明显崩溃，但reward不是独立测试准确率，100steps也不能排除后期退化。横轴按step，无法单独读出墙钟收敛速度，必须结合对应运行耗时。

资源开销进一步说明多版本不是免费：load balancing在64/128GPU约占0.414%/1.519%，request transfer约3.627%/2.123%，free cache约0.019%/0.027%。这些占总时间百分比随规模变化，不应相加成某个通用固定税率。较高带宽和长任务使迁移成本更易摊薄，低带宽或短任务收益可能下降。

**面试如何解释“不丢慢轨迹”仍可能有等待？** 窗口前移需要最旧版本清空，否则背压限制新版本推进；系统减少全batch等待，但保留受控版本约束。好处是完整轨迹最终可使用，代价是旧副本碎片和最老任务的窗口阻塞。是否比中途换权重更好应按质量与GPU-hour验证。

本报告尚无作者原表截图，这是明确证据缺口。所有图来自固定版LaTeX资产，并核对轴、图例和版本；不能通过把这些柱状图改画为表就声称满足原表标准。[结果原文](https://arxiv.org/pdf/2604.26256v1)

## 5. 局限、结论与后续阅读

DORA 的核心是多版本并存，以轨迹内固定行为策略换取可追溯性和同版本KV复用，再通过动态分区缓解旧副本碎片。版本窗口限制积压，却没有让历史数据自动on-policy，也不等于teacher缓存正确。

主要局限是旧模型驻留成本、rank分配粒度、窗口最老任务阻塞、跨布局缓存迁移和内部实现不可核验。GRPO组统计与trajectory streaming的完整接线仍需代码证据；production巨大加速不能直接作为普通集群端到端收益。当前原文无表，单篇按仓库标准保留草稿，已整理全部方法、配置和原图。

与[PipelineRL](#paper=paper-pipelinerl)对照轨迹固定版本与在途更新，与[AReaL](#paper=paper-areal)对照背压和行为/近端策略分离，与[AsyncOPD](#paper=paper-asyncopd)对照蒸馏旧数据校正。实践优先比较“固定版本+消费一次队列”与“混合版本+真实token行为概率”，把版本管理、prefill和任务质量纳入同一GPU-hour账本。[固定版论文](https://arxiv.org/abs/2604.26256v1)
