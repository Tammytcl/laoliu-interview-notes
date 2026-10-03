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


## 把形状与完整层的开销连起来

多头输入 [B,N,D] 投影后通常拆为 [B,H,N,d_head]，其中 D=H*d_head。attention 的主项约 O(B*N²*D)，QKV/输出投影约 O(B*N*D²)，MLP 也有随 N 线性、随模型宽度增长的项。因此当 N 较短、D 很大时，attention 的平方项未必是整层最大的成本。

除以 sqrt(d_head) 用于控制点积的尺度：在独立、零均值、单位方差的简化假设下，d_head 项之和的方差随 d_head 增长。实际训练分布更复杂，不能把这一假设当成所有层的精确统计。

## 小实验与自检

固定 D 比 N=128/512/2048 的 attention 项与线性层项；然后固定 N 增大 D。画张量形状，说明哪种场景更受序列长度影响。实验为建议练习，未提供硬件实测数值。

自检：FlashAttention 不保存 N×N 矩阵，为何仍计算稠密的 token 对？读 [FlashAttention 推导](https://arxiv.org/abs/2205.14135) 或练习 [手撕 causal attention](#q=causal-attention-coding)。

## 原始资料

调研日期：2026-09-30。[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 用于结构与注意力表达式；[FlashAttention](https://arxiv.org/abs/2205.14135) 用于 IO 与中间存储。复杂度算例是本文教学整理。
