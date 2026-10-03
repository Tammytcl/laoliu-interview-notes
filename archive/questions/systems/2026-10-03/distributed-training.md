---
id: distributed-training
title: "DDP、ZeRO/FSDP、TP 和 PP 分别切什么？"
category: systems
difficulty: 进阶
tags: ["P0", "分布式训练", "FSDP"]
updated: 2026-09-30
summary: "按数据、模型状态、层内算子和层间划分理解通信与资源取舍。"
draft: false
---

## 从问题出发

“多卡训练”不是一种方法。要先问模型能否单卡放下、哪类状态最大、互联能支持多少通信。

## 四类切分

| 方式 | 切分对象 | 主要通信或代价 |
|---|---|---|
| DDP | 不同数据，各卡复制完整模型 | 同步梯度，通常 all-reduce |
| ZeRO/FSDP | 优化器、梯度、参数等状态 | 状态分片与参数聚合 |
| TP | 层内矩阵/张量 | 层间或算子间频繁通信 |
| PP | 不同层到不同 stage | 传激活、反传梯度与流水空泡 |

ZeRO-1 分优化器状态，ZeRO-2 再分梯度，ZeRO-3 再分参数。FSDP 也通过分片参数等状态减少每卡常驻内存，但具体 API、参数聚合时机和状态语义应按实现版本理解，不等于所有细节都与 ZeRO 完全相同。

## 常见通信关系

DDP 在 backward 中把梯度桶通信与计算重叠。完整参数分片时通常在计算前 all-gather 所需参数、反向后 reduce-scatter 梯度；峰值还受预取、层粒度和重分片策略影响。

TP 常希望放在互联快的卡之间；PP 对 stage 负载均衡敏感；DP 可以在组合并行外层分数据。这是常见工程出发点，不是对所有网络拓扑的固定最佳布局。

## 最容易漏的数学问题

全局 batch = 每设备 microbatch × 梯度累积步数 × 数据并行副本数。TP/PP 卡数不都算新的独立数据副本。跨 rank 不同有效 token 数时，局部平均后再平均可能不等于全局 token 平均，需明确 loss 归一化方式。

## 小实验与自检

给“单卡权重放得下但 Adam 状态放不下”和“单层矩阵太大”两种情境分别选方案，说明为什么。画两卡 DDP 与四卡 TP×DP 的数据分配，计算全局 batch。

自检：FSDP 参数分片为何仍可能出现参数聚合峰值？PP stage 耗时不均衡会造成什么等待？

## 面试表达

按切分对象回答，随后连接内存收益与通信代价。不要只背框架名字。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [ZeRO](https://arxiv.org/abs/1910.02054)：读三个模型状态分片阶段。
- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：官方实现入口，具体 API 以环境版本为准。
- [Megatron-LM](https://arxiv.org/abs/2104.04473)：理解 tensor、pipeline 与 data parallel 的组合。
