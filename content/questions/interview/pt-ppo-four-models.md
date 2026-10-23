---
id: "pt-ppo-four-models"
title: "Q05 · PPO 的四个模型各做什么，完整训练一步怎么走？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "优化目标", "阿里", "美团"]
updated: "2026-10-23"
summary: "画出采样、奖励、价值估计与策略更新的数据流，并明确哪些模型冻结、哪些概率必须留存。"
draft: false
---

## 60–90 秒回答

典型 RLHF-PPO 有 actor、critic、reference 和 reward model 四种职责：actor 生成，reward model 或规则给反馈，critic 估计未来回报，冻结 reference 约束行为偏移。训练时先冻结一个 behavior 快照收集响应及其实际 log-prob，计算奖励、价值和优势，再对 actor 做 clipped 更新，对 critic 做价值回归。一个批次可更新多次，但旧概率与优势目标不能在中途随意改变。最后同步新 actor 继续采样。四种职责不意味着必须有四个同规模、完全独立的模型：奖励也可以是规则，部分模块可以共享，但它们的数学用途要讲清。

## 原理拆解

### 一次 rollout 批次怎样变成可训练数据

令 $s_t=(x,y_{<t})$、$a_t=y_t$。采样阶段保存真实 token、response mask、行为策略版本和 $\log\pi_b(a_t|s_t)$；reward/reference/critic 可以随后在这些相同历史上前向。典型 RLHF 将终局质量分与逐 token 的参考约束结合，例如对行为策略采样批次定义：

$$
r_t=-\beta[\log\pi_b(a_t|s_t)-\log\pi_{ref}(a_t|s_t)]+\mathbf1_{t=T}R(x,y).
$$

这里是一次批次固定的 KL-shaped reward 配方，不是所有 PPO 的唯一定义。用采样时的 value、奖励和边界计算优势与 return，随后作为当前批次的固定目标。RM 与 reference 冻结；critic 学的是 return，不是直接复制 RM 的最终分。

### 为什么采完以后还能更新多次

固定状态上的 clipped surrogate 用 $\rho_t=\exp(\log\pi_\theta-\log\pi_b)$ 比较当前与行为策略。刚同步且计算路径一致时 ratio 应接近 1；一个 rollout batch 切成多个 minibatch/epoch 后，当前参数逐渐偏离采样参数，ratio 与裁剪反映这种偏离。重复训练提高数据复用率，但也增加偏移和过拟合风险，不能认为 epoch 越多越省成本。[PPO 原论文](https://arxiv.org/abs/1707.06347v2)

一个常见最小化目标是 $L=L_{policy}+c_vL_V-c_eH$；系数、是否有 entropy bonus、value clipping 与 KL 的位置都依配方说明。若 KL 已进入 reward，又增加独立 KL loss，就要解释是否有意双重约束。更新结束再同步 actor 到 rollout，reference 通常仍保留初始锚点。

### 实现上哪些量能重算，哪些量不能改含义

Current log-prob 每次需要带梯度重算；reference 可缓存；旧 value、return 与 advantage 在这套固定目标配方中不随 minibatch 改写。Behavior log-prob 必须代表实际采样分布，不能用更新后的 actor 覆盖。优势标准化、有效 token 分母、分布式梯度缩放及 padding mask 都会影响实际步长。若生成用了 temperature/top-p，必须说明概率记录与训练 ratio 的定义；不能直接假定原始 softmax 概率就是变换后解码器的行为概率。

## 白板与例子

```text
prompt → behavior/rollout → response + behavior log-prob + version
                   ├→ reward model / verifier → reward
                   ├→ frozen reference       → reference log-prob
                   └→ critic                 → value → GAE → advantage/return
current actor + recorded history → current log-prob → ratio/clip → actor update
critic + fixed return target                                → value update
```

策略 ratio 的分母来自 behavior/proximal，KL 的另一个模型通常是 reference，两者不能换名混用。概念上可写：

$$
L_V=\frac12\,\mathbb E[(V_\phi(s_t)-\operatorname{sg}(\hat R_t))^2].
$$

若使用 value clipping、KL-shaped reward 或不同回报定义，需要说明具体版本。`sg` 表示停止梯度；actor 更新不应意外把固定奖励或优势的计算图一起反传。

## 追问怎么接

**Reference 能用旧 actor 替代吗？** 可以重新定义成约束最近策略，但这改变了锚点和目标。长期 reference 和每轮刷新 behavior 分别承担不同用途，不能说等价。

**为什么不直接训练 reward model 代替 critic？** Reward 衡量当前完成质量；critic 条件于状态，估计当前策略继续执行的未来回报。即使最终奖励二元，两者也不是同一预测任务。

**Actor、critic 可以共享 backbone 吗？** 可以，但有梯度干扰、内存与训练调度取舍。需要明确 value head、哪些参数共享、怎么加权和更新；PPO 定义没有要求共享或完全分离。

## 项目怎么联系

对你的 Agentic OPD，teacher 提供概率监督，在线 critic 可用于调整监督强度；这不自动等于“用了 PPO 四模型”。回答时先画自己的真实数据流，再说明借鉴了价值学习的哪一部分、没有采用哪类策略目标。

## 容易答错的点

- 把 reward model 的 pairwise loss 当成 critic 的 return 回归 loss。
- 把 reference log-prob 用作 PPO ratio 分母。
- 每次 optimizer step 都重算并替换旧 log-prob，导致更新约束失去固定锚点。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [美团算法一面面经（2025）](https://www.nowcoder.com/feed/main/detail/7197e4d2e8934c76a3bc65aedaf1afeb) | 经历自述，招聘真实性未独立核验；公开正文；仅作题型依据 |
| [阿里国际「国际一面」（2025）](https://www.nowcoder.com/feed/main/detail/166d1f3bb6b84624b912dcc3997f8081) | 经历自述，招聘真实性未独立核验；检索可见正文；直链读取返回简页；仅作题型依据 |
| [PPO 原论文](https://arxiv.org/abs/1707.06347v2) | clipped surrogate、旧策略与更新约束 |

**相关专题详解：** [Actor Critic从回报到策略梯度](#q=rl-actor-critic) · [PPO训练一步的数据梯度与实现](#q=rl-ppo-training)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-reward-preference-data) · [下一题](#q=pt-gae-credit)

[//]: # (interview-generated-end)
