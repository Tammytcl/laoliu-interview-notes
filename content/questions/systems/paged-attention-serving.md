---
id: paged-attention-serving
title: "PagedAttention 和 continuous batching 各解决什么？为什么高吞吐可能让首 token 更慢？"
category: systems
difficulty: 进阶
tags: ["P0", "PagedAttention", "Serving"]
updated: 2026-09-30
summary: "区分 KV 内存管理、请求调度和服务延迟指标。"
draft: false
---

## 从问题出发

用户输入与输出长度不同，静态 batch 经常等最慢请求，连续显存预留又造成浪费。服务系统需要同时管理内存和调度。

## 两种机制分开讲

PagedAttention 用逻辑块到物理块的映射管理 KV，避免每个请求必须预留一块很大的连续空间。块可以按需分配、回收；共享前缀时可引用已有块，写入分叉处需要正确处理共享和复制。

Continuous batching 在迭代间加入新请求、移出完成请求，不必一直维持同一组请求。它是调度机制，不等于 KV 分页算法。两者经常组合，但各自负责不同问题。

## Prefill 与 decode

Prefill 一次处理输入的多个 token；decode 通常每个请求每轮生成一个新 token并读取历史 KV。在许多实际负载中前者更易产生较大的计算，后者受权重/KV 读取与并发影响；不能无条件认定所有 prefill 都 compute-bound、所有 decode 都 memory-bound。

大段 prefill 可能占用调度时间，让正在 decode 的请求等待。Chunked prefill 将长 prompt 分片，可改善两阶段的混合调度，但 chunk 大小与服务负载需要测量。

## 指标及 trade-off

| 指标 | 含义 | 主要影响因素 |
|---|---|---|
| TTFT | 提交到首 token | 排队、prefill |
| ITL | 相邻输出 token 间隔 | decode、混合调度 |
| E2E latency | 完整请求耗时 | 输入输出长度、排队 |
| tokens/s | token 吞吐 | batching、请求组合 |

吞吐提高可能伴随更长队列，TTFT 反而恶化。要在固定延迟目标下比较 goodput，而非只报告最大 tokens/s。

## 小实验与自检

构造短 prompt/长 prompt 和短输出/长输出的混合请求，控制到达速率。记录 p50/p95 TTFT、ITL、吞吐、显存与错误率，再改变 batch token 预算。

自检：KV 内存碎片减少为何不意味着模型计算减少？长尾输出为何影响静态 batch？

## 面试表达

先用“内存管理”和“调度”区分两种机制，再把优化收益落在服务质量与延迟预算上。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：查看块式 KV 管理与服务背景；指标拆解是本文实践整理。
