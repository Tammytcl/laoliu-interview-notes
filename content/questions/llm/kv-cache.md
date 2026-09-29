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
