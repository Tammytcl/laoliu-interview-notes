---
id: paper-asyncopd
title: 'AsyncOPD: How Stale Can On-Policy Distillation Be?'
paper_title: 'AsyncOPD: How Stale Can On-Policy Distillation Be?'
authors:
- Wonjun Kang
- Kevin Galim
- Seunghyuk Oh
- Minjun Kang
- Sanghyun Park
- Donghoon Kim
- Minjae Lee
- Minseo Kim
- Rishabh Tiwari
- Yuchen Zeng
- Hyung Il Koo
- Kangwook Lee
affiliations:
- FuriosaAI
- Ajou University
- UC Berkeley
- Microsoft Research
- KRAFTON
- Ludo Robotics
author_affiliations:
- - 1
- - 1
- - 1
- - 2
- - 2
- - 1
- - 1
- - 1
- - 3
- - 4
- - 1
  - 2
- - 5
  - 6
venue: arXiv preprint (fixed v1)
year: 2026
direction: infra
areas:
- language
tasks:
- training-adaptation
- reasoning
published: '2026-06-23'
method_figure: ./assets/papers/paper-asyncopd/method-source.png
method_caption: Figure 9 · 三阶段 streaming OPD 与同步 / step-off 对照
paper_url: https://arxiv.org/abs/2606.24143v1
github_url: https://github.com/furiosa-ai/async-opd
code_note: Official repository, reviewed at a pinned current commit; this is not a claim of reproducing the paper experiments.
evidence: 已核原文
note_ids:
- opd-vs-sft
tags:
- AsyncOPD
- OPD
- Async Training
- Rollout
updated: '2026-10-04'
summary: 研究异步 OPD 的动作与前缀过时，比较 KL 方向和有限教师缓存；以当前学生信号、IS、多动作 MC 连接持续三阶段流水线。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

On-policy distillation（OPD）让学生先按自己的策略生成回答，再让教师评价学生实际遇到的 token 前缀，并用这些信号更新学生。与照着教师现成答案训练的 SFT 相比，OPD 的监督发生在学生自己的访问分布上。对于长数学推理，代价也主要来自学生的自回归生成；若每轮必须等 rollout、教师评分、训练和权重同步依次完成，多个角色会轮流空闲。

异步系统让 rollout 在 learner 更新时继续运行，却使数据对应的学生版本落后。这里有两种不同的过时：旧学生访问了哪些前缀，是 **prefix-level staleness**；在同一个已访问前缀上，缓存的候选 token 和概率由哪个学生选出，是 **action-level staleness**。即使能修正第二种，第一种也不会自动消失。例如旧学生已经选错工具，后面再给 token 加重要性权重，不能把状态改回没有犯错的另一条轨迹。

[GKD](https://arxiv.org/abs/2306.13649)建立了学生分布上的广义蒸馏路径；[AReaL](https://arxiv.org/abs/2505.24298v1)说明有界异步和 PPO 校正可以改善 RL 利用率。AsyncOPD 的问题更具体：这些异步 RL 技巧能否直接稳定蒸馏？如果教师的全词表 logits 太大，只保存 top-k 或少量动作分数，旧缓存还能表达当前学生应优化的目标吗？论文通过 KL 方向、优势信号和缓存支持集三个维度分开实验，而不是先把一个异步 scheduler 的收益全部归到 loss 上。

本文核读固定 v1。作者的系统结果主要来自数学推理，教师固定；它不是长程 coding agent 或教师本身持续更新的完整验证。最值得迁移的内容是量的依赖关系：旧采样概率必须保留，当前学生概率可以重算，缺失的教师分数无法凭空恢复。

## 2. 方法与实现机制

### KL 方向与缓存到底改变了什么

令 $s$ 是生成前缀，$a$ 是下一 token，$q$ 是固定教师，$p_\theta$ 是当前学生，$p_{\rm old}$ 是采样时学生。两个局部目标为：

$$
D_F=\sum_a q(a\mid s)\log\frac{q(a\mid s)}{p_\theta(a\mid s)},\qquad
D_R=\sum_a p_\theta(a\mid s)\log\frac{p_\theta(a\mid s)}{q(a\mid s)}.
$$

Forward KL 的动作权重来自教师；reverse KL 的动作权重随当前学生改变。后一种若仍使用旧学生生成的动作和旧优势 $A_{\rm old}=\log q-\log p_{\rm old}$，训练时连局部方向都可能失配。论文建议在 learner 上重算 $A_\theta=\log q-\log p_\theta$，再用旧到当前的动作重要性比修正采样：

$$
\rho_\theta(a,s)=\frac{p_\theta(a\mid s)}{p_{\rm old}(a\mid s)},\qquad
\nabla D_R=-\mathbb E_{a\sim p_{\rm old}}
\left[\rho_\theta A_\theta\nabla\log p_\theta(a\mid s)\right].
$$

此式针对固定前缀，要求旧策略覆盖当前动作支持集。优势作为 stop-gradient 权重；不是对这三个乘数随意全部 detach。若旧概率为 0.2、当前为 0.4，重要性比为 2；教师若给该 token 概率 0.1，当前优势应为 $\log(0.1/0.4)$，而不是旧的 $\log(0.1/0.2)$。这个小例子解释了为何“已经用了 IS”仍不足以说明蒸馏信号正确。

Top-k 缓存还有另一层问题：旧学生 top-k 没选中的动作没有教师分数，集合内重加权不能补回缺失支持。Multi-sample MC 在每个实际访问前缀上额外抽多个下一 token，请教师评价后平均 IS 梯度；这些动作并不展开成多条完整轨迹。它保留可校正的抽样结构并降低方差，代价是更多局部评分与缓存。

### 三阶段持续执行与有界积压

![Figure 9 · 同步、固定 step-off 与 streaming AsyncOPD](./assets/papers/paper-asyncopd/method-source.png)

**Figure 9 解读。** 蓝色是 rollout，红色是教师评分，绿色是 learner，黑色是权重同步。中间路径虽有重叠，仍以完整 rollout batch 作为屏障；右侧把完成样本逐个交付，使三阶段持续消费。图说明屏障的位置，不表示 learner 在教师未评分时就能训练。[图源：v1 Figure 9](https://arxiv.org/pdf/2606.24143v1#page=9)。

调度器用 $(\tau+1)B$ 个 semaphore permits 限制未消费工作。提交 prompt 前获取 permit，learner 消费后释放；同步窗口另有 gate 阻止投放旧权重任务。队列 FIFO，不按年龄事后驱逐。Keep-mode 同步保留已生成 token，清掉旧 KV 后以新权重重建注意力状态，并记录 token 级版本切换点。因此这里允许一条回答跨版本，不是 DORA 的整轨迹固定版本。

### 核心源码伴读

官方仓库固定到 `e3f79e916aad9dba886256d202bab96f8504a447`，为核读时公开实现，不能视为原实验 commit 的证明。[streaming_stages.py](https://github.com/furiosa-ai/async-opd/blob/e3f79e916aad9dba886256d202bab96f8504a447/opd/streaming_stages.py)中的 `PromptFeeder.run` 先检查 sync gate 和容量，再向 worker 投放 prompt；`TrainDispatcher.run` 收集已评分样本，调用训练并等待完成，设置同步请求后释放容量。OPD 按单条样本积累，GRPO 分支则另等完整组，这解释了算法依赖仍会改变交付单位。

[multi_sample_policy_gradient_kl](https://github.com/furiosa-ai/async-opd/blob/e3f79e916aad9dba886256d202bab96f8504a447/opd/loss/kl.py#L1211)接收学生、教师与旧学生的多动作 logprob，布局为 batch × sequence × samples。`online_advantage` 分支用当前学生重算并 detach 优势；随后以最后一维为 sample axis 调用 loss。输入也可直接使用预 gather 的 logprob，或由 logits 和候选 token 索引计算。这里仅静态核读，未运行 GPU 蒸馏。

## 3. 实验设置与算力

论文把估计器过时扫描和最终 scheduler 比较分开。前者固定主要学生并改变 cache depth；后者在同一节点、相同 loss 下比较系统组织。不能把 200-iteration 通用配置套到所有 100-iteration scheduler 行。

| 项目 | 固定 v1 的配置 | 位置与解释 |
|---|---|---|
| 主要学生 | Qwen3-4B-Base | KL / 估计器分析 |
| Scheduler 学生 | Qwen3-{1.7B,4B,8B}-Base；另测禁用 thinking 的 Qwen3 | §7、附录 scheduler |
| 教师 | Qwen3-30B-A3B-Instruct-2507 | 固定外部教师 |
| 训练集 | DeepMath，难度至少 6，57,630 题 | 附录实验细节 |
| 评测 | AIME24 / AIME25 / AMC；最终 checkpoint Avg@32 | 每题 32 次，平均准确率 |
| 长度 | prompt 2,048，response 16,384 token | 通用配置，不是实际平均长度 |
| Batch | global 256，mini 64 | 每逻辑 batch 含 4 次 optimizer update |
| Optimizer | AdamW-style；LR 3e-6，常数 schedule，WD 0.01 | Table 3 |
| 生成 | temperature 1，top-p 1，不做 top-k 截断 | 动作概率与 IS 需对应 |
| Staleness 扫描 | 0、1、2、4、8、16、32、64、128 个逻辑 train-batch step | 不能当作同数量 mini update |
| Scheduler horizon | 100 training iterations；AsyncOPD τ=4 | §7 |
| 后端 | vLLM rollout / scoring；PyTorch FSDP learner | 附录 |
| GPU | 单节点 8 × B200 | 各 matched run 总资源一致 |
| 角色分配 | teacher 1；sync 在其余 7 卡分时；async 为 rollout 4、trainer 3 | 不能只比较 learner 卡数 |
| 时间 | 单项实验约 1–12 小时 | 没有给每行统一 GPU-hour 成本 |
| 复现状态 | 本报告未训练；代码固定当前 commit | 与作者实验区分 |

Train tok/s 是被 learner 实际使用的 response token 除以训练墙钟间隔，去掉最初五次 warmup；不是生成服务输出速率。Overlap 把三个阶段的 busy time 相加后除以墙钟，rollout 先按 worker 平均，理论最高 3，不是 GPU 利用率百分比。采用这两个指标能看出样本是否流到训练端，也避免把多生成但未消费的数据算成有效收益。

## 4. 结果与图表解读

![Figure 2 · KL 方向的过时扫描；从左至右为平均、AIME24、AIME25、AMC](./assets/papers/paper-asyncopd/kl-comparison-source.png)

**Figure 2 解读。** 横轴是逻辑 batch 过时量，纵轴是最终 Avg@32 准确率；四面板保留原顺序。红色 reverse-KL 基线在小过时量下较高，但到 64 / 128 时下降明显；绿色 forward KL 曲线较平。该比较使用具体 sparse / PPO-style 实现，支持“缓存约束下不同方向对旧动作依赖不同”，不证明所有 reverse-KL 估计器都比 forward KL 差；本文随后设计的当前优势与 MC 正是处理这一缺陷。[图源：v1 Figure 2](https://arxiv.org/pdf/2606.24143v1#page=5)。

![Table 2 · Qwen3-Base scheduler 的吞吐、重叠与准确率](./assets/papers/paper-asyncopd/table-2-pdf.png)

**Table 2 解读。** 同一学生内比较 strict sync、two-step-off 与 AsyncOPD，再分别读 MC64 和 MC1 三列。4B-Base / MC64 从 9.5k 提升至 15.8k train tok/s，1.66×，AIME24 同为 25.00；1.7B-Base / MC1 为 3.28×，但准确率 8.44 略低于 sync 的 8.65。系统吞吐普遍更高，准确率是“相近”，并非每个条件都严格提高。Overlap 约从 0.8 到 2.1–2.3，显示持续三阶段重叠。[表源：v1 Table 2](https://arxiv.org/pdf/2606.24143v1#page=9)。

![Table 7 · 禁用 thinking 的 Qwen3 补充比较](./assets/papers/paper-asyncopd/table-7-pdf.png)

**Table 7 解读。** 摘要中的最高约 3.8× 出现在此补充设置：4B / MC64，4.5k 到 17.0k，3.82×，准确率 54.90 对 54.69。它不是 Table 2 的 Base 模型结果，也不是 coding agent 加速。1.7B / MC64 的准确率 35.00 对 32.23 提醒读者：总体“comparable”描述不等于每行完全无损。多样本 MC 的额外缓存成本和 scheduler 收益，应在同一 MC 设置内比较，不能交叉挑最快行。[表源：v1 Table 7](https://arxiv.org/pdf/2606.24143v1#page=18)。

## 5. 局限、结论与后续阅读

证据支持的是：固定教师、有限评分缓存、数学任务上的异步 OPD 可以通过当前学生信号、动作 IS 和多样本 MC 取得较好吞吐 / 效果折衷。无 clipping 的结论来自这里的消融，不能据此取消任意 RL 或 coding distillation 的约束；局部 IS 更不能消除旧前缀访问分布偏移。

迁移时先盘点 cache 的 token ID、old logprob、teacher logprob、loss mask 和版本位置，再确认 learner 重算是否保留正确梯度。若 teacher 缓存仅是 deterministic old top-k，新增 IS 参数并没有使缺失动作得到监督。资源上还要计入多动作评分、缓存传输与同步重建 KV 的成本。

与 [OPD 基础笔记](#q=opd-vs-sft)和异步专题一起阅读；AReaL 的 decoupled PPO 在本文实验中没有胜过更简单的 OPD-specific surrogate，说明“系统可借鉴”与“目标可直接移植”应分开。报告核读正文、实验及 scheduler 附录、上述四张原图表和公开源码；其他附录消融按原文继续阅读，不声称全部复现。[固定论文](https://arxiv.org/abs/2606.24143v1)、[官方代码](https://github.com/furiosa-ai/async-opd)，整理日期 2026-10-03。
