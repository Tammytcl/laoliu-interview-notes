---
id: agent-gradient
title: "多轮 Agent 调用工具后，梯度如何回传？"
category: agent
difficulty: 深入
tags: ["Agent","梯度","Loss mask"]
updated: 2026-09-30
summary: "区分 rollout 的离散交互和训练阶段重新计算的可微 loss。"
draft: false
---

## 一句话回答

通常不对工具执行过程反向传播，而是保存各轮真实上下文和生成 token，训练时重新 forward，对 assistant action token 计算 loss。

## 核心思路

```text
任务 → thinking + command → tool output → thinking + command
mask     1（生成部分）          0             1（生成部分）
```

提示词和工具输出一般不作为预测目标，但仍是模型的条件上下文。不能把 loss mask=0 理解为模型看不到它们。

## 面试追问

### 删除历史 thinking 还能训练吗？

可以，只要保存每轮实际使用的 prompt 和生成 action。历史被重写后，不能随意把多轮记录拼成一条假想的连续序列；应按真实条件分段处理。

### 为何需要保存原始 token？

文本 decode 后再 encode 可能改变分词，工具调用的解析与重序列化也可能改变格式。训练 logprob 应对应实际采样的 token 及其前缀。

## 常见误区

不需要保留整个 sandbox 交互的计算图。策略梯度也不要求 shell、文件系统或测试程序可微。


## 从真实轨迹到训练样本

多轮记录至少保存每轮实际输入 token、输出 action token、工具结果、采样策略版本，以及算法需要的旧 logprob/reward。工具执行通常不可微；策略梯度通过“在这个真实状态下选择这些 action 的概率”优化。

如果每轮都重写上下文，就按真实轮次构造样本。对已经执行的 action 重新序列化，或拿最终摘要当旧轮前缀，都会改变条件概率。loss mask 只控制监督位置，不控制 attention 可见性。

对一个 trajectory 赋予最终 reward 时，如何分配到各轮、各 token，是否按长度归一化，必须随具体算法说明。并非所有训练都使用同一种 advantage。

## 小实验与自检

做两轮玩具交互：第一轮读取文件，第二轮根据结果作答。打印两轮 prompt、action 和 mask，再人为改写第一轮历史，观察第二轮 action 的条件 logprob 是否改变。此处是练习设计，尚未执行。

自检：工具输出 mask=0 时，参数梯度是否仍可能受它影响？可以，因为它参与后续 action 的条件计算。采样图是否必须一直保留？通常不需要，训练可以重算 forward。

## 原始资料与边界

调研日期：2026-09-30。[ReAct](https://arxiv.org/abs/2210.03629) 提供交互轨迹的背景；[DeepSeekMath](https://arxiv.org/abs/2402.03300) 用于理解策略概率与优势目标。本文多轮记录规范是工程整理，不声称这两篇使用同一种多轮训练实现。继续阅读 [GRPO](#q=grpo-rlvr)。
