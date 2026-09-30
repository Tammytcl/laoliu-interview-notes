---
id: "paper-pagedattention"
title: "PagedAttention：把 KV 缓存管理变成块映射"
paper_title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
authors: ["Woosuk Kwon", "Zhuohan Li", "Siyuan Zhuang", "Ying Sheng", "Lianmin Zheng", "Cody Hao Yu", "Joseph E. Gonzalez", "Hao Zhang", "Ion Stoica"]
year: 2023
direction: "infra"
paper_url: "https://arxiv.org/abs/2309.06180v1"
evidence: "资料整理"
note_ids: ["paged-attention-serving", "prefix-cache"]
tags: ["基础论文", "infra"]
updated: "2026-09-30"
summary: "从逻辑块、物理块和共享读 KV，理解内存管理怎样影响服务吞吐。"
template_version: 1
draft: false
---

## 1. 收录动机与阅读目标

把 Infra 学习从单个 kernel 延伸到服务。读前问题：“模型数学不变，改变 KV 放置方式为何能提高系统吞吐？”

## 2. 原文信息与核验范围

[原文 v1 / SOSP 2023](https://arxiv.org/abs/2309.06180v1)。核对第 4 节 KV 管理、Figure 6 与第 7 节评估/消融；未对当前 vLLM 版本做复现，也未独立审核全部服务负载。

## 3. 研究问题与先修知识

理解 [KV cache](#q=kv-cache) 与 [服务指标](#q=paged-attention-serving)。可变输入输出长度让预留连续内存浪费，共享前缀的重复 KV 也影响可容纳的并发。

## 4. 一句话核心贡献

用逻辑到物理块映射管理 KV，并在服务系统中利用动态分配与共享减少内存浪费。

## 5. 方法与关键推导

逻辑 token 顺序保持连续，物理块不必相邻。块表让 attention 根据请求映射读取 K/V；共享与写时复制需要额外生命周期管理。

教学例子：块大小为 4，长度 7 的前缀占两个逻辑块；再增加一个 token 能用末块剩余位置，继续增加才需要新块。这个例子帮助手算分配，不是硬件结果。

## 6. 关键图表与证据

| 位置 | 阅读问题 | 要核对的条件 | 边界 |
|---|---|---|---|
| Figure 6、第 4.3 节 | token 追加时映射怎么变？ | 块编号与填充位置 | 分配节省不代表数学 FLOPs 减少 |
| 第 7 节端到端评估 | 吞吐与延迟怎样共同比较？ | 模型、请求长度、到达与解码设置 | 不保证自己的负载同倍提速 |
| 块大小/kernel 开销消融 | 碎片与访问开销如何权衡？ | 读取模式和块大小 | kernel 更快不是唯一系统目标 |

## 7. 作者结论与我的判断

作者的服务评估展示内存管理收益。整理者的判断：先区分容量、调度和算子成本，再选指标；更大有效 batch 是一种可能的收益路径，不能忽略排队延迟。

## 8. 局限、反例与失败条件

块映射有开销，仍有末块余量；内存压力会影响共享缓存生命周期。没有足够请求、非 KV 瓶颈或严格延迟预算时，收益需重新测量。

## 9. 和已有知识的连接

[Prefix caching](#q=prefix-cache) 讨论跨请求复用；[GQA 精读](#paper=paper-gqa) 从模型结构减少 KV。二者可互补，但一个不是另一个的替代品。

最小练习：模拟 3 个不同长度请求的块分配/回收，再比较静态预留和按需分配的容量；尚未执行。

## 10. 不看报告的复述问题

先在个人阅读记录中写一个读前问题；读后收起正文，用自己的话回答：本文改变了哪个环节？为什么合理？最关键的证据是什么？哪里仍不确定？这里不替你填“我的理解”。

## 11. 待验证与下一步

本人模型实验尚未执行。先核读正文标出的机制与图表，再完成一项有明确对照的最小练习；把真实配置、结果和失败样本写回记录。个人阅读进度从“待精读”开始，不由报告生成自动推进。

## 12. 更新记录与来源

2026-09-30：作为论文收录组件的基础精读示例整理，原文重点位置列于第 2/6 节；尚未进行本人复现。正文中的阅读建议和学习推演不当作原论文实验结论。
