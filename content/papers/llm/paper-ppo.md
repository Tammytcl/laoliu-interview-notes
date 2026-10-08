---
title: "Proximal Policy Optimization Algorithms"
authors: ["John Schulman", "Filip Wolski", "Prafulla Dhariwal", "Alec Radford", "Oleg Klimov"]
affiliations: ["OpenAI"]
venue: "arXiv preprint"
year: 2017
published: "2017-07-20"
summary: "从策略梯度、TRPO、GAE推导PPO的悲观裁剪目标，解释多轮更新、critic与参考策略的不同职责，并核对原始控制任务实验。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-08"
template_version: 5
depth_standard: "ddpm"
draft: false
id: "paper-ppo"
paper_title: "Proximal Policy Optimization Algorithms"
paper_url: "https://arxiv.org/abs/1707.06347v2"
method_figure: "./assets/papers/paper-ppo/figure-1-pdf.png"
method_caption: "Figure 1 正负优势下的裁剪目标"
github_url: "https://github.com/openai/baselines"
tags: ["PPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
author_affiliations: [[1], [1], [1], [1], [1]]
---

## 1. 背景与已有工作

PPO 要解决的是一个非常具体的矛盾：环境交互和模型采样很贵，我们想让同一批经验被训练多次；但经验来自更新前的策略，多次梯度更新以后，当前策略已经变了，继续把旧数据当成当前策略的数据会产生误差，甚至一次更新就破坏已有行为。论文用简单的一阶优化目标，在数据复用与策略变化之间建立实用约束。它的原始实验是机器人控制和 Atari，LLM 的 token、奖励模型、参考模型都属于后来的应用映射。

先建立三个基础概念。策略 $\pi_\theta(a\mid s)$ 是在状态 $s$ 下选择动作 $a$ 的概率；奖励描述某次行为或结果的反馈；价值 $V^\pi(s)$ 是从该状态开始、继续按策略行动的预期折扣回报。优势 $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$ 回答“这个动作比此状态下平均选择好多少”。奖励为正不等于优势为正：一个奖励为 5 的动作，如果同状态平均能拿 8，仍应相对减少其概率。

REINFORCE 利用 $\nabla\log\pi_\theta(a\mid s)$ 把不可微的环境反馈变成可微的策略更新：好动作更常出现，差动作更少出现。引入与动作无关的状态基线不会改变理想的期望策略梯度，却能减少方差。Actor-Critic 学习这个基线；GAE 用多步 TD 残差折中方差与偏差。它们回答“梯度朝哪里走”；PPO 进一步回答“这批数据还能复用多少次、每次走多远”。

[TRPO 原论文](https://arxiv.org/abs/1502.05477)通过 KL 约束限制新旧策略的变化，近似求解需要自然梯度、共轭梯度和线搜索。PPO 希望保留其经验上的稳健性，同时允许 Adam、minibatch、多 epoch 和参数共享。论文同时研究 PPO-Clip 与自适应 KL Penalty；今天通常说的 PPO 指前者，但算法家族并不只有裁剪版本。

| 方法 | 主要处理的问题 | 使用经验的方式 | 仍然需要注意 |
| --- | --- | --- | --- |
| REINFORCE / Vanilla PG | 通过回报学习随机策略 | 常规新采样与梯度估计 | 高方差、过大的策略变化 |
| Actor-Critic / GAE | 估计更稳定的优势 | 依赖 value 与 bootstrap | critic 误差和终止边界 |
| TRPO | 限制策略分布变化 | 有约束的 surrogate 优化 | 近似二阶求解复杂 |
| PPO-Clip | 多 epoch 复用同批经验 | 冻结旧策略，裁剪悲观目标 | 不是严格 KL 或性能保证 |
| PPO-KL | 用惩罚控制变化 | 根据观测 KL 调整惩罚系数 | 目标 KL 与系数需要调节 |

在面试中把 PPO 直接描述成“四个大模型”会混淆算法与 LLM RLHF 工程。2017 年的 PPO 可以使用小 MLP、CNN，actor 和 critic 也可以共享 backbone。LLM 中经常出现 actor、critic、reward、reference 四个逻辑角色，但它们不是原算法要求的四套独立同尺寸参数。旧策略又是另一种职责，可能只需保存采样时的 logprob。[原文 §2—§5](https://arxiv.org/pdf/1707.06347v2)。

## 2. 方法与实现机制

### 从策略梯度到冻结旧策略的 surrogate

一批经验由 $\pi_{\mathrm{old}}$ 生成，保存状态、动作、奖励、终止标记和旧 logprob。当前模型在相同状态和动作上重新计算 logprob，构造比值：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}
=\exp\bigl(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\bigr).
$$

分子参与梯度，分母和本轮优势固定。初始两策略相同时 $r_t=1$，但其对当前参数的导数仍非零；不能把“数值是 1”误读成“没有策略梯度”。比值衡量当前模型对这次已发生动作的相对倾向，允许用旧动作信息构建 surrogate。它并没有完整校正状态访问分布，也不意味着任意久远的 replay 都适合 PPO。

不加约束的 $\mathbb E[r_t\widehat A_t]$ 会不断推高正优势动作概率、压低负优势动作概率。对同批数据优化过久，有限样本的估计误差被放大，新策略远离采样分布。PPO-Clip 最大化：

$$
L^{\mathrm{CLIP}}(\theta)=\widehat{\mathbb E}_t
\left[\min\left(r_t\widehat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right].
$$

![Figure 1 · 正优势与负优势对应的两段裁剪目标](./assets/papers/paper-ppo/figure-1-pdf.png)

**Figure 1 解读。** 横轴是动作概率比，纵轴是单个样本的裁剪目标，红点 $r=1$ 表示更新起点。左侧 $A>0$ 时超过上界的继续增长不再提高目标；右侧 $A<0$ 时低于下界的继续下降不再提高目标。图并不是两端都水平的平台。它展示一个 surrogate 项的分段行为，不能单独推出总体 KL 被硬限制或整次参数更新必然提高真实回报。[图源：固定 v2 Figure 1，PDF 第 3 页](https://arxiv.org/pdf/1707.06347v2#page=3)。该版源码只包装编译好的整篇 PDF，没有独立图资产，因此紧裁原图。

### 裁剪究竟在哪个方向生效

令 $\epsilon=0.2$。正优势 $A=2$ 时，$r=1.3$ 对应两个候选值 $2.6$ 与 $2.4$，取较小的 $2.4$，该项不再鼓励增大比值；但 $r=0.7$ 对应 $1.4$ 与 $1.6$，取 $1.4$，仍有纠正错误方向的梯度。负优势 $A=-2$ 时，$r=0.7$ 对应 $-1.4$ 与 $-1.6$，取 $-1.6$，不再鼓励继续降低；$r=1.3$ 对应 $-2.6$ 与 $-2.4$，取 $-2.6$，仍会抑制正在变得更常见的坏动作。

因此，裁剪是“停止奖励已经过度有利的变化”，并非“越界样本一律删除”。实际参数共享使同一个 token 即使其本项梯度为零，概率仍可能受其他样本、value loss、entropy 或 KL 项影响。$\epsilon$ 不是对概率、参数距离或实际 KL 的硬上限；clip fraction 也不能被直接当作全部数据无效比例。

| 优势 | 当前比值 | 本项行为 |
| --- | --- | --- |
| 正 | 高于 $1+\epsilon$ | 上平台，停止继续推高 |
| 正 | 低于 $1-\epsilon$ | 保留梯度，修正概率下降 |
| 负 | 低于 $1-\epsilon$ | 下平台，停止继续压低 |
| 负 | 高于 $1+\epsilon$ | 保留梯度，修正坏动作上升 |

### GAE 与 critic 的计算路径

对真实终止用 $m_t=0$，对可以继续 bootstrap 的边界用 $m_t=1$，教学形式为：

$$
\delta_t=R_t+\gamma m_tV_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t),\qquad
\widehat A_t=\delta_t+\gamma\lambda m_t\widehat A_{t+1}.
$$

$\gamma$ 控制远期回报折扣；$\lambda$ 控制多步残差混合。较小 $\lambda$ 更依赖 critic，通常降低方差但可能引入更多偏差；较大 $\lambda$ 更接近长程回报，不能无条件保证更好。实现时倒序递推，一次计算整段优势，return target 常用 $\widehat A_t+V_{\mathrm{old}}(s_t)$。不要在 actor backward 时沿目标值回传到 critic。

例如两个动作后结束，奖励 $[0,1]$、旧价值 $[0.4,0.6]$，设 $\gamma=\lambda=1$。末步 $\delta_2=1-0.6=0.4$，首步 $\delta_1=0+0.6-0.4=0.2$，故优势 $[0.6,0.4]$。前面的动作没有立即奖励仍能学习，因为它影响了后续结果。若采样段仅因达到 horizon 被截断而环境未结束，就应使用末状态价值；真正终止、时间限制截断和 padding 需要不同处理。

共享 actor-critic 参数时，最大化的组合目标包含策略目标、负 value MSE 和正熵奖励：

$$
L=\widehat{\mathbb E}_t\left[L_t^{\mathrm{CLIP}}-c_1(V_\theta(s_t)-V_t^{\mathrm{targ}})^2+c_2H(\pi_\theta(\cdot\mid s_t))\right].
$$

训练代码通常最小化其相反数。entropy bonus 帮助保持探索；它不等于 reference KL，也不能替代 reward 质量。优势标准化会影响有限批次的尺度与学习率解释，value clipping、reward normalization、梯度裁剪和 KL early stop 是需要逐实现核对的细节，不能都当成论文主公式。

### 一轮训练和 LLM 映射

```python
# 本文教学重述，省略并行环境、critic优化和截断细节
batch = rollout(old_policy)  # old_logprob, rewards, values, masks
adv, target_return = gae(batch)  # 固定本轮目标
for epoch in range(K):
    for mini in shuffled_minibatches(batch):
        logp = policy.logprob(mini.states, mini.actions)
        ratio = exp(logp - mini.old_logprob)
        surrogate = minimum(ratio * mini.adv,
                            clamp(ratio, 1-eps, 1+eps) * mini.adv)
        loss = -masked_mean(surrogate) + value_loss - entropy_bonus
        optimizer_step(loss)
old_policy = snapshot(policy)
```

原文 $N$ 个 actor 每个收集 $T$ 步，共 $NT$ 个经验，再进行 $K$ 个 epoch；旧策略在这一批内部固定。不是每个 minibatch 更新后就把分母替换成当前模型，否则比值无法记录同批数据的累计偏离。更新后再采样，新一轮才获得新的 behavior policy。

映射到语言模型：状态是 prompt 与已生成前缀，动作是下一 token，终止通常是 EOS 或任务结束。终局 reward 可以由偏好 reward model、测试程序或环境提供。reference model 用于长期对齐约束，old policy 用于本轮更新比较。reference 可固定在初始 SFT 模型，old 则随 rollout 批次变化；这两个 KL 的方向、估计对象和更新周期必须说清楚。多轮 agent 还要让工具输出只进入条件上下文，把非模型生成的 observation 排除出 action loss。

### 历史源码与现代讲解

固定原始公开 PPO 提交 `da997060461e3cbf54ca4dc7a67081a731fb6b3b`，核读 [pposgd_simple.py](https://github.com/openai/baselines/blob/da997060461e3cbf54ca4dc7a67081a731fb6b3b/baselines/pposgd/pposgd_simple.py)。`traj_segment_generator` 连续收集长度为 horizon 的 observation、action、reward、旧 value 和新 episode 标记；通过 `nextvpred` 保留段末 bootstrap，而不是必须等 episode 完成。`add_vtarg_and_adv` 反向遍历奖励数组，根据 `new[t+1]` 判断是否接入下一状态的价值，返回 GAE 与 TD-lambda return。

`learn` 建立 `pi/oldpi`，由两者动作 logprob 差取 exp，构造 `minimum(surr1,surr2)` 并取负号；赋值操作冻结本轮旧模型，随后执行多个 minibatch epoch。此历史实现还包含 value clipping，它是实现事实，不应把“PPO-Clip”中的 clip 全部理解成同一个操作。输入主要是 `[NT,...]` 环境轨迹，非 `[B,L,V]` 的 LLM logits。

[OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)的正负优势讲解有助于理解平台为何只出现在一侧；[The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)提供了版本、GAE、minibatch、网络初始化和终止处理的逐项伴读。它研究的后续 ppo2 版本与这里的 2017 提交有差异，博客性能不能替代本论文原始实验。本文只静态核读源码。

## 3. 实验设置与算力

原论文的“效率”首先是环境交互次数与训练曲线，不是语言模型每秒生成 token。MuJoCo、Roboschool、Atari 分别代表连续控制、复杂人体控制和离散游戏；网络、动作分布、horizon 和超参也不同。把一种任务的 Adam 学习率复制到另一种任务不构成原论文复现。

| 项目 | 固定 v2 配置 | 出处与边界 |
| --- | --- | --- |
| 小规模连续控制 | 7 个 MuJoCo Gym v1 环境 | §6.1；各训练 1M timesteps |
| 网络 | 两层 64 单元 tanh MLP，Gaussian action distribution | actor/value 不共享参数，无 entropy bonus |
| 重复次数 | 每环境 3 seeds，共 21 runs / 设置 | 用末 100 episodes 平均回报汇总 |
| 优化器 | Adam，$3\times10^{-4}$ | Table 3 |
| rollout horizon | 2048 | 不要求完整 episode |
| 更新 | 10 epochs，minibatch 64 | epoch 与环境 timesteps 不同 |
| 折扣与优势 | $\gamma=0.99,\lambda=0.95$ | Table 3 |
| 裁剪 sweep | 0.1、0.2、0.3 | Table 1；0.2 为该 sweep 最优 |
| Roboschool | 3 类 humanoid，32 或 128 actors，horizon 512 | Table 4；15 epochs、minibatch 4096 |
| Atari | 49 games，3 seeds，8 actors，horizon 128 | §6.4 与附录 B |
| Atari 更新 | 3 epochs，minibatch $32\times8$ | clip 与 LR 随训练线性退火 |
| Atari 预算 | 40M frames，即 10M timesteps | frames 与决策 timestep 不同 |
| GPU / CPU 型号、数量 | 未完整披露逐任务设备账本 | 不能由 actor 数推定 GPU 数 |
| 精度、墙钟、GPU-hours | 未提供可横向复算的完整账本 | 不据此估计 LLM 训练卡数 |
| 代码版本 | 2017 原始公开提交，见方法模块 | 不等同今天 SB3 / TRL 的默认配置 |

![Table 3 · MuJoCo 1M timestep 实验超参](./assets/papers/paper-ppo/table-3-pdf.png)

**Table 3 解读。** 左列是环境段长、优化器、更新轮数和 GAE 超参，右列是本论文具体数值。2048 是每个 actor 收集的步长，不是 LLM 最大回答长度；10 epochs 意味着同批经验被多次使用，而不是模型训练了 10 次完整环境任务。该表没有硬件信息，也没有现代 LLM 的 precision、TP 或 KV cache 配置。[表源：v2 附录 A，第 10 页](https://arxiv.org/pdf/1707.06347v2#page=10)。

MuJoCo 汇总分数把随机策略移到 0，把该环境最佳结果归到 1，再平均 21 个 run。因此 0.82 是跨环境的归一化 score，不是成功率 82%。Atari 同时比较整个训练过程的平均 episode reward 与最后 100 episodes 的 reward，前者更看重学得快，后者更看重最终表现。论文给出的“赢了多少游戏”也不是一套全游戏累计总回报。

没有逐任务 GPU-hours 就不能从“采样更高效”推出“总计算减半”。现代 LLM 中 critic 的前反向、actor logprob 重算、rollout 解码、reward 服务和权重同步都要计入；原始小网络环境实验不覆盖这些成本。阅读复现实验还应核对 Gym 环境版本、frame skip、奖励裁剪、observation normalization 和 seed，后续实现博客正是在说明这些细节会改变结果。

## 4. 结果与图表解读

![Table 1 · 不同策略更新约束的连续控制对照](./assets/papers/paper-ppo/table-1-pdf.png)

**Table 1 解读。** 各行改变 surrogate 的约束或超参，第二列是 7 环境、3 seeds 的平均归一化 score。无 clip/penalty 为 -0.39，clip 0.1/0.2/0.3 为 0.76/0.82/0.70；最好的自适应 KL 设置为 0.74。负分主要受 HalfCheetah 严重退化影响，不能理解为所有任务都负回报。这支持“适度裁剪在此 sweep 中比无约束复用稳健”，也说明扩大 epsilon 并非单调改善。[表源：v2 Table 1，第 6 页](https://arxiv.org/pdf/1707.06347v2#page=6)。

0.82 对 0.74 的差是归一化得分差 0.08，不是准确率提高 8 个百分点。不同 penalty 值给出不同结果，说明比较必须包含调参，而不能拿一个未调过的 PPO-KL 默认值就认定该家族无效。该表没有误差条、显著性检验或每个超参的原始设备时间，也不能支撑任意新任务上的确定性排名。

![Figure 3 · 七个 MuJoCo 环境的训练曲线与完整图例](./assets/papers/paper-ppo/figure-3-pdf.png)

**Figure 3 解读。** 七个子图横轴为环境 timesteps，纵轴为该任务 reward；紫色为 PPO-Clip，蓝青色为 TRPO，其他颜色对应 A2C、trust-region A2C、CEM 和 adaptive vanilla PG。PPO 在 HalfCheetah、Reacher、Walker2d 等明显更好，但 Swimmer 上 TRPO 更优，部分任务曲线也接近或高方差。图支持作者“几乎所有连续控制环境”中的优势，不能改写成全任务、全时刻优胜。[图源：v2 Figure 3，第 7 页](https://arxiv.org/pdf/1707.06347v2#page=7)。

同一横坐标比较的是样本预算，而非相同墙钟或相同硬件开销。某方法需要更多内部优化、更多 value 更新或昂贵二阶操作，也可能在相同 timesteps 下计算成本不同。看阴影和训练中途的变化比只记最后一个点更有价值：早期快速增长、后期波动和不同 seed 稳定性分别回答不同问题。

Atari 的另一项结果是按游戏计算胜出个数：整个学习过程平均 reward 口径下 PPO/A2C/ACER 分别赢 30/1/18 个，末 100 episode 口径分别赢 19/1/28 个，另有 1 个平局。这说明 PPO 在该离散任务集具有较好的学习速度；最终胜出游戏数低于 ACER，但实现更简单。不要将“赢 30 个游戏”讲成“所有游戏 reward 平均提高 30%”。这些数字来自原文 Table 2；详细逐游戏数据在附录，本文重点展示核心约束 sweep 和连续控制曲线。

| 证据 | 对应问题 | 可以支持 | 不能推出 |
| --- | --- | --- | --- |
| Figure 1 | clip 如何按优势符号工作 | 分段 surrogate 机制 | KL 的硬上限 |
| Table 1 | 多 epoch 更新需何种约束 | 该 sweep 的 clip 稳健性 | 任意 epsilon 均有效 |
| Figure 3 | 相同环境样本数下如何学习 | 多数 MuJoCo 任务的样本效率 | 同 GPU-hour 的 LLM 性能 |
| Table 3 | 曲线具体用了什么配置 | 原始连续控制设置 | 今日框架默认等价 |

## 5. 局限、结论与后续阅读

PPO 的核心贡献是用容易实现的悲观 surrogate，使多 epoch 的经验复用在许多任务上有效。裁剪改善了经验稳定性，但真实状态分布仍变化，critic 仍可能错误，有限数据仍可能被过拟合。论文没有证明 PPO-Clip 对所有任务单调改善回报，也没有证明其 KL 永不越界。奖励错误、探索不足和 rollout 太旧都可能让训练失败。

### 面试问答与追问

以下为本文依据机制整理的题目，不声称某家公司真实题频。

**Q1：PPO 的 old policy 与 reference policy 是一回事吗？** old 负责描述本轮数据的生成分布，reference 负责长期对齐或行为锚定；前者通常每轮刷新，后者可能固定。追问：如果 actor 已更新三次才消费旧数据，分母应保留哪个版本？应保留实际 behavior logprob，并明确是否另有 proximal policy 与异步校正。

**Q2：为什么不能对 REINFORCE loss 无限做 epoch？** 同批动作来自旧策略，直接用当前 logprob 反复优化会放大样本误差和分布偏离。PPO 的 ratio 与 clip 缓解这种问题，但也没有使无限 epoch 安全。追问应谈 KL、clip fraction、advantage 质量和验证 reward，而不是只背 epsilon。

**Q3：超出 clip 区间是不是零梯度？** 要看优势符号及进入 min 的分支。正优势的上越界、负优势的下越界进入平台；反方向仍保留梯度。即使此 surrogate 项为零梯度，共享参数和其他 loss 仍可能改变该动作概率。

**Q4：为什么 reward 不可微也能训练？** 策略梯度通过已采样动作的 logprob 求导，reward 作为系数；不需要对环境、判题程序或工具执行反向传播。追问：可微 reward model 也通常不需要把梯度穿过它和采样 token。

**Q5：critic 的作用只是打分吗？** 它估计状态未来回报，用作优势基线和 bootstrap，不是学习人类偏好的 reward model。critic 预测可能参与策略方差降低，但策略不能只最大化 critic 的任意分数而忽略真实奖励。

**Q6：为什么 LLM 的 PPO 很贵？** 除 actor 反向还可能需要 critic 前反向、reference/reward 前向、解码及模型同步；长序列会增加激活与 rollout 开销。共享 backbone 或缓存能减少部分成本，不能按“四个模型 × 参数量”直接预测峰值显存。

**Q7：训练 reward 上升但验证下降查什么？** 查 reward hacking、过拟合、长度变化、value explained variance、KL、entropy、截断率、mask 和概率计算一致性。先分开反馈失真与优化不稳，再判断换 GRPO、GSPO 或 SAPO 能否解决其中某个环节。

**Q8：PPO通常称on-policy，为什么还用importance ratio？** 每一轮重新收集当前行为数据，但同一批的多epoch更新使learner逐渐偏离本轮behavior；ratio处理这段有限复用。这个称呼不意味着任意旧replay都安全，异步版本滞后要另设控制。

后续阅读：[DPO](#paper=paper-dpo)把离线偏好拟合重参数化；[DeepSeekMath / GRPO](#paper=paper-grpo)以组基线去掉单独 value model；[后训练优化专题](#report=survey-policy-optimization)统一比较概率比、优势与约束。原文版本固定为 1707.06347v2，首次公开日期是 2017-07-20。2026-10-08 完成正文、附录、原图表和历史源码核读；没有执行机器人、Atari 或 LLM 训练复现。
