---
id: kv-cache
title: "KV Cache 缓存了什么，为什么能加速生成？"
category: llm
difficulty: 基础
tags: ["推理","KV Cache"]
updated: 2026-09-30
summary: "理解自回归生成中的重复计算，以及速度与显存之间的取舍。"
draft: false
---

## 一句话回答

KV Cache 保存历史 token 在各层的 key 和 value，使后续 decode 不必重复计算历史 K/V；代价是缓存随上下文长度和并发数增长。

## 核心思路

生成新 token 时，只需计算它自己的 Q/K/V，将新 K/V 追加到缓存，再让新 Q 关注历史及当前的 K/V。

粗略缓存容量：

```text
2 × 层数 × KV头数 × 头维度 × 缓存token总数 × 每元素字节数
```

其中 2 表示 K 与 V。并行切分、分页管理、量化及额外元数据会影响实际设备占用。

## 面试追问

- 为什么通常不缓存历史 Q？
- MHA、GQA、MQA 的 KV cache 大小有什么不同？
- 长上下文和高并发为什么容易相互制约？

## 常见误区

KV cache 减少重复计算，但不会让注意力访问历史的成本消失。缓存容量占满也不意味着 GPU 算力利用率很高。


## 为什么不保存历史 Q？

标准因果生成的新输出依赖当前 query 对历史 key 的匹配，以及对应 value 的加权。过去位置的 query 已经完成其输出计算，新 token 不需要它们，因此常规实现缓存 K/V。非因果模型或更特殊算法要另说。

在因果注意力且既有前缀不改写的条件下，历史 token 的隐藏状态不会被未来 token 反向影响，才能安全复用缓存。改写先前 token 或前缀条件后，不能继续无条件用旧缓存。

## 容量与访问成本算例

32 层、8 KV 头、头维 128、2 bytes/元素，则每 token 128 KiB，8192 个 token 约 1 GiB。并发请求按各自有效 token 数累加；前缀共享、分页与张量并行会影响每卡实际占用。

一次 decode 不再重算整段历史的 K/V 和前向，但仍要访问历史 KV 并计算新 query 的 attention。上下文增长可能增加单步耗时，不能说用了 cache 后 decode 成本恒定。

## 小实验与自检

用小模型对同一前缀比较“每步完整重算”和“增量 KV”得到的 logits，固定 eval 模式与输入格式，允许浮点误差；再测随上下文增长的缓存容量。这里是练习方案，未声称完成实测。

继续读 [GQA](#q=mha-gqa-mqa)、[显存账本](#q=training-inference-frameworks) 与 [prefix caching](#q=training-inference-frameworks)，分别理解结构、容量和跨请求复用。

## 原始资料

调研日期：2026-09-30。[GQA](https://arxiv.org/abs/2305.13245) 提供 KV 头共享背景；[PagedAttention](https://arxiv.org/abs/2309.06180) 提供服务中的 KV 管理。容量算例是本文推算，非测量值。
