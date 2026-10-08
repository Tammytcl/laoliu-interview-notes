---
id: sft-dpo-rl
title: "SFT、DPO 和在线 RL 分别在优化什么？"
category: llm
difficulty: 进阶
tags: ["P0", "后训练", "DPO"]
updated: 2026-10-08
summary: "按数据来源、目标与反馈形式比较方法，避免把偏好优化等同于推理能力。"
draft: false
---

## 从问题出发

模型能够续写，不意味着会按要求完成任务。后训练用示范、偏好或结果反馈改变输出分布，三种信号解决的问题不同。

## 方法对比

| 方法 | 数据与反馈 | 直接优化量 | 主要风险 |
|---|---|---|---|
| SFT | 示范回答 | 示范 token 的负 logprob | 示范覆盖不足 |
| 标准离线 DPO | chosen/rejected 偏好对与参考策略 | 相对 logprob 的偏好损失 | 数据偏好偏差与分布错位 |
| 在线 RL | 当前/近期策略 rollout 与奖励 | 奖励导出的策略目标 | 奖励投机、方差和采样成本 |

固定离线数据和在线采样是常见设置，不是绝对边界；DPO 也有在线变体，SFT 也可持续更新数据。

## DPO 的一个标准公式

同一 prompt x 下，偏好回答 y_w 与 y_l，固定参考策略 pi_ref：

```text
delta = [log pi_theta(y_w|x)-log pi_ref(y_w|x)]
      - [log pi_theta(y_l|x)-log pi_ref(y_l|x)]
L_DPO = -log sigmoid(beta*delta)
```

标准序列 logprob 是有效回答 token logprob 的和；长度归一化等改动会改变目标。beta 不是学习率。DPO 在特定 KL 正则奖励建模假设下推导，不需要先显式训练奖励模型，但并不意味着任何偏好数据都对应真实任务奖励。

## 为什么不能只比 loss？

SFT 的低 loss 可能来自短且重复的回答；DPO 的偏好准确率可能提高而事实正确性下降；RL 奖励可能被漏洞绕过。要用同一独立任务集、相同推理预算评测，并拆分能力、风格和长度变化。

## 小实验与自检

给同一个知识问答构造“简短正确”和“冗长错误”的偏好对，讨论标注者会不会误偏好后者。手算四个 logprob 的 delta，判断一次更新会鼓励什么相对变化。

自检：DPO 是否保证 chosen 的绝对概率一定升高？为什么相对 margin 变大与绝对概率变化不能混为一谈？

## 面试表达

先说反馈从哪里来，再说优化目标和独立评测。在线 RL 的优势构造继续看 [GRPO](#q=grpo-rlvr)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [DPO](https://arxiv.org/abs/2305.18290)：核对偏好目标与参考策略的作用。
- [InstructGPT](https://arxiv.org/abs/2203.02155)：理解示范、奖励模型与 RLHF 的阶段关系。


## 原论文精读与机制比较

[进入完整原论文报告](#paper=paper-dpo)，继续核对公式推导、原图表、实验配置、源码和问答。横向复习见[PPO DPO GRPO及其变体](#report=survey-policy-optimization)：按优势、概率比、更新约束和奖励组织比较，避免把离线偏好拟合与在线组训练混为一谈。
