---
id: causal-attention-coding
title: "手撕单头 causal attention：形状、mask 和稳定 softmax"
category: coding
difficulty: 基础
tags: ["P1", "手撕代码", "Attention"]
updated: 2026-09-30
summary: "只保留与模型面试直接相关的一个编程练习，检查维度和因果边界。"
draft: false
---

## 为什么选这道练习

它能同时检查矩阵乘法、广播、mask 和数值稳定性，比暂时扩充大量算法题更贴近当前四方向学习。

先修：[Attention](#q=attention-complexity)、[自回归 mask](#q=causal-lm-training)。

## 题目约定

输入 Q/K/V 均为 [batch, seq, head_dim]，同长度 self-attention，允许关注自己和过去，不带 padding、dropout、KV cache。返回同形状输出。下面是教学 NumPy 实现，不是生产高性能 kernel。

```python
import numpy as np

def causal_attention(q, k, v):
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise ValueError('expected equal [batch, seq, head_dim] shapes')
    _, n, d = q.shape
    if n == 0 or d == 0:
        raise ValueError('seq and head_dim must be positive')
    # 教学版先转 FP32；实用实现需要明确 dtype 策略。
    q, k, v = [x.astype(np.float32) for x in (q, k, v)]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    future = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v
```

## 每一步为什么这样写

QKᵀ 的形状是 [B,N,N]，softmax 沿最后一维 key 归一化。k=1 只屏蔽严格未来，自身保留，避免第一行全部被 mask。先减行最大值防止指数溢出；全 masked 行的情况需在扩展 padding 时额外处理。

## 有意义的边界检查

N=1 时输出应等于 V；修改未来位置的 K/V 不应影响前面输出；零 Q/K 时，第 i 行输出应为前 i+1 个 V 的均值。反转 mask 或错在 query 轴 softmax，通常会破坏这些性质。

CPU 检查记录（2026-09-30）：本篇代码已通过 N=1、未来 K/V 不影响前缀、零 Q/K 输出前缀均值三项 NumPy 性质检查；未做 GPU 性能测试。

小实验：不用标准答案比对，先用上述性质验证；再解释 O(B*N²*d) 时间和 O(B*N²) 中间存储。此教学实现未宣称在仓库完成性能测试。

## 追问到哪里就够

多头要增加 head 维；decode 的 query 长度可为 1、KV 长度更大，不能直接套用这里的同长度 mask；[FlashAttention](https://arxiv.org/abs/2205.14135) 避免实体化大矩阵，但数学目标相同。

面试时先写形状，再写 mask 和 softmax，最后讲复杂度与边界。其余算法暂只保留已有 [LRU](#q=lru-cache)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：核对 scaled dot-product attention；代码是本文教学实现。
