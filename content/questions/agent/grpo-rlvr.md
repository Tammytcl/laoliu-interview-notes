---
id: grpo-rlvr
title: "RLVR 与 GRPO 是一回事吗？组内奖励怎样变成训练信号？"
category: agent
difficulty: 深入
tags: ["P0", "GRPO", "RLVR"]
updated: 2026-09-30
summary: "区分奖励来源与优化算法，理解组内优势、采样成本及零方差问题。"
draft: false
---

## 从问题出发

“用可验证奖励训练”说明反馈来源；“用 GRPO”说明策略怎么更新。两者不是同一概念。

先修：[SFT/DPO/RL](#q=sft-dpo-rl)、[多轮 loss mask](#q=agent-gradient)。

## 一个常见 outcome GRPO 流程

对同一任务由采样策略 pi_old 生成 G 条回答，获得结果奖励 R_i，用组内标准化作为优势估计：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
r_it = exp(log pi_theta(a_it|s_it) - log pi_old(a_it|s_it))
clip-surrogate = min(r_it*A_i, clip(r_it,1-e,1+e)*A_i)
```

最大化有效 action token 上的 surrogate，通常还包含与参考策略相关的 KL 项。此处仅示意原始常见形式，不涵盖全部后续 GRPO 变体。pi_old 用于采样和概率比，pi_ref 用于约束偏移，不能把它们混为一个对象。

策略梯度不要求奖励或工具可微；梯度经过 action 的 logprob。采样、优势与旧 logprob 在训练更新中通常作为已固定的数据。

## 为什么不需要单独 critic，为什么不免费？

组内结果提供相对基线，省掉单独 value 模型的训练和存储。但每题需要多次 rollout，长链推理和工具环境可能比更新更贵。G 增大不保证按比例改善收益。

若同组全成功或全失败，标准化优势常为零，不能提供相对任务奖励的区分信号；若保留 KL 等项，整体仍可能有梯度。不能说“零优势就所有训练完全停止”。

## RLVR 的关键边界

可验证不等于无漏洞。答案匹配器、测试覆盖不足或环境 bug 可能被利用；长输出、反复试探也能增加成本。记录任务成功率、奖励一致性、输出长度与 [有效吞吐](#q=training-inference-frameworks)。

## 小实验与自检

手算三组奖励 [0,0,1,1]、[1,1,1,1]、[0,0,0,0] 的优势；再改变组大小。说明组内相对信号为何不能替代覆盖不同难度任务。

## 面试表达

先区分奖励可验证性与优化算法，接着画采样、评分、优势、重算 logprob、更新，最后谈全同奖励与系统成本。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：读第 4.1 节 GRPO；本篇明确采用 outcome supervision 简化说明。
