---
id: "pt-grpo-objective"
title: "Q09 · 写出 GRPO 目标：不训练 critic，优势从哪里来？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "GRPO 与奖励", "腾讯", "美团"]
updated: "2026-10-23"
summary: "从同题多采样、组相对优势、token ratio 和 KL 四部分讲清 GRPO，而不只说“去掉 critic”。"
draft: false
---

## 60–90 秒回答

GRPO 对同一个 prompt 从旧策略采样一组回答，根据组内回报的相对位置构造优势，通常不额外训练独立 critic。随后用 current 对采样策略的 token 概率比构造 clipped surrogate，并根据具体配方加入相对 reference 的 KL。对只有终局奖励的版本，同一回答的 token 可以共享一个组优势，但具体 mask、长度归约和优化批次会改变实际梯度。它的节省来自省去常规 value 模型及相关前向与训练，不是免费生成奖励。对于长程任务、同组无差异或反馈噪声大的情况，优势质量和采样成本仍要验证。

## 原理拆解

### Group baseline 代替了什么，没有代替什么

对相同 prompt 的 $G$ 条回答用组内回报构造相对优势，省去了常规独立 critic 的学习与训练状态。它仍需要生成、评分和一个 baseline；只是 baseline 从组回报得到，而不是一个泛化到状态的 value 网络。奖励可以来自规则或 RM，reference KL 是否启用则由具体配方决定。[DeepSeekMath 的 GRPO 定义](https://arxiv.org/html/2402.03300v3)

只看中心化、不除标准差时，若样本条件独立且同分布，$\bar R$ 包含本条样本自身，会使期望策略梯度带有 $(G-1)/G$ 的缩放；leave-one-out baseline 不包含本条回报，可以避开这个自包含项。再除组标准差后，分母本身依赖采样，不能继续当作纯常数缩放。这说明组大小、baseline 和标准化都属于目标估计器的设计。

### 写出 token 级优化目标才算讲完整

一个带终局回报与参考约束的常见形式是：

$$
J=\mathbb E\left[\frac1G\sum_{i=1}^G\frac1{T_i}\sum_{t=1}^{T_i}\left(\min(\rho_{it}A_i,\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i)-\beta K_{it}\right)\right].
$$

$A_i$ 来自固定 rollout 的组回报；$\rho_{it}$ 比较 current 与 behavior；$K_{it}$ 是所声明的 reference KL 项或估计器。同一响应共享 $A_i$，不代表每个 token 有相同的因果贡献；它只是用序列结果给生成动作提供共同更新信号。若采用过程 reward，就要重新说明 token/step 优势如何累积。

### 长度归约怎样进入梯度

两条回答长度分别为 10 与 100，若每条先除自身长度再平均，则每条响应总系数相同，但短回答每个 token 的系数是长回答的 10 倍；若全批 token 平均，单 token 系数相同，长回答包含更多更新项。若用固定最大长度归约，又是另一种尺度。它们影响正负优势长短样本的累计作用，不能用“只是 batch mean 写法不同”略过。

实际训练还要记录原始 prompt 数、每题生成量、有效组占比、token 数与评分成本。省去 critic 是某类成本下降，并不保证比所有 PPO 配置更便宜；当 group 很大、响应很长或大量组同分时，rollout 可能成为主要成本。

## 白板与例子

给定同 prompt 的 $G$ 个回答，一种常见优势为：

$$
A_i=\frac{R_i-\bar R}{s_R+\varepsilon},\qquad
\rho_{i,t}=\frac{\pi_\theta(a_{i,t}|h_{i,t})}{\pi_b(a_{i,t}|h_{i,t})}.
$$

原始风格的 outcome 目标可概括为按响应平均其有效 token 的 clipped 项，再减 KL。要能说出“先每条平均、再组平均”和“全批有效 token 平均”的区别；不是写一个 `mean()` 就结束。

手算用奖励 $[0,0,1,1]$。若本例采用总体标准差，则均值 0.5、标准差 0.5，优势约为 $[-1,-1,1,1]$。实际代码要核对标准差是否采用无偏样本修正，数值会不同，不能据此误判 bug。

## 追问怎么接

**PPO 和 GRPO 谁信用分配更细？** 典型 token-level PPO 能通过 critic/GAE 引入状态依赖优势；仅终局组奖励的 GRPO 通常更粗。但 GRPO 也能设计过程反馈，不能说其定义永远只能整条打分。

**G 越大越好？** 更多候选可能改善组统计或发现成功，但增加 rollout 与评分成本，也可能高度相关。比较同总生成 token/卡时下的有效组和独立质量。

**多轮工具任务怎么算一个 group？** 先定义同一初始任务、环境版本与执行预算；不能把不同难度任务的回报随意放同组标准化。工具随机性也可能污染相对比较。

## 项目怎么联系

可用你熟悉的代码/数学训练讨论“同题多响应成本与有效反馈”的区别，再引出 OPD 的教师稠密监督。不要把 OPD 中的教师 log-prob 当作 GRPO 已训练的奖励模型，也别把自己没跑过的 GRPO group size 填成实际配置。

## 容易答错的点

- 把组均值直接当 advantage，遗漏中心化。
- 因为“没有 critic”而说 GRPO 完全没有 baseline。
- 不分原始目标、框架实现和后续变体的归约方式。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [腾讯技术研究 NLP 面经（2025）](https://www.nowcoder.com/feed/main/detail/c8ddab6cf8ea467c9e4003c9256616cb) | 经历自述，招聘真实性未独立核验；检索可见正文；直链读取返回简页；仅作题型依据 |
| [美团算法一面面经（2025）](https://www.nowcoder.com/feed/main/detail/7197e4d2e8934c76a3bc65aedaf1afeb) | 经历自述，招聘真实性未独立核验；公开正文；仅作题型依据 |
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300v3) | 组相对优势、策略目标及 outcome/process 监督 |

**相关专题详解：** [GRPO从组奖励到token梯度的完整流程](#q=grpo-rlvr) · [RL算法总览 从Actor Critic到PPO和组策略优化](#q=rl-policy-optimization-map)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-dpo-derivation) · [下一题](#q=pt-zero-variance-groups)

[//]: # (interview-generated-end)
