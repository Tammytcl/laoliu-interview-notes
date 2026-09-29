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
