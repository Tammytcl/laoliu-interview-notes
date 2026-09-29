---
id: attention-complexity
title: "Self-Attention 的计算复杂度为什么是平方级？"
category: llm
difficulty: 基础
tags: ["Transformer","Attention"]
updated: 2026-09-30
summary: "从矩阵形状出发，区分序列长度、隐藏维度和中间结果的开销。"
draft: false
---

## 一句话回答

标准稠密 Self-Attention 需要计算每个 query 与所有 key 的相关性，产生一个 n × n 的分数矩阵，因此包含随序列长度 n 平方增长的计算项。

## 核心思路

设单头 Q、K、V 的形状均为 n × d：

1. QKᵀ 生成 n × n 的分数，计算量为 O(n²d)。
2. 注意力权重乘 V，同样为 O(n²d)。
3. 线性投影还会引入与模型维度相关的计算，不能把整个层的开销只写成 O(n²)。

公式为：Attention(Q, K, V) = softmax(QKᵀ / √d)V。

## 面试追问

### FlashAttention 是否把计算复杂度变成线性？

不是。它主要通过分块和在线 softmax，避免完整注意力矩阵在显存中反复读写。标准稠密 attention 的算术复杂度仍是平方级，但内存访问和中间存储得到优化。

### Prefill 和 decode 的区别？

Prefill 处理一段输入；使用 KV cache 的逐 token decode 只计算新 query 与历史 key 的注意力，两者的负载特点不同。

## 常见误区

- 不区分计算复杂度和显存复杂度。
- 忽略线性投影开销和具体的 n、d 大小关系。
- 把 FlashAttention 说成稀疏 attention。
