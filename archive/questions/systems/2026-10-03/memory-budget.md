---
id: memory-budget
title: "训练和推理显存怎么估？为什么参数量不能直接换算可运行规模？"
category: systems
difficulty: 基础
tags: ["P0", "显存", "OOM"]
updated: 2026-09-30
summary: "把权重、梯度、优化器、激活、KV 和运行余量分开记账。"
draft: false
---

## 从问题出发

“7B 用 FP16 只有 14 GB，所以 16 GB 卡能训练”忽略了大部分训练状态。先做账本再谈并行和优化。

## 训练与推理的账本

| 项目 | 训练 | 推理 |
|---|---|---|
| 权重 | 基础权重，可能另有 master copy | 基础或量化权重 |
| 梯度与优化器 | 随可训练参数增长 | 通常无 |
| 激活与 workspace | 反向保存，受 batch/序列影响 | 临时计算，也可能有大峰值 |
| KV cache | 标准全序列训练通常不持续缓存 | 自回归服务随 token/并发增长 |
| 框架与分配器 | reserved、碎片、通信 buffer | 同样需要余量 |

一种示意混合精度 Adam 配置，每参数有 2-byte 权重、2-byte 梯度、4-byte master 权重、两个 4-byte moment，共约 16 bytes。7B 参数对应约 112 GB（十进制），仍不包含激活和额外缓冲。这不是任何框架的固定数字；梯度 dtype、优化器和分片方式会改变估算。

仅 FP16 权重为 14 GB ≈13.0 GiB。单位要统一，不能把 GB 与 GiB 当作相同。

## 影响各项的技术

[LoRA](#q=lora-finetuning) 主要减少可训练状态；[ZeRO/FSDP](#q=distributed-training) 分片状态；[checkpointing](#q=activation-checkpointing) 降低保存激活；量化减少特定张量的位宽。技术作用在不同项上，不能用一个“省显存倍数”包办所有来源。

量化也有 scale、zero-point 或未量化层等开销；“4-bit”并不表示整个运行时每参数严格只占半字节。

## 推理例子

32 层、8 KV 头、128 头维、FP16，单请求 8192 token 约 1 GiB KV。16 个这样的请求仅 KV 就约 16 GiB，还需权重和运行余量；容量不足时增并发会更糟。

## 小实验与自检

用一个小模型分别测初始化、第一次 forward、backward 和 optimizer step 的 allocated/reserved 峰值；首次更新时优化器状态可能才建立。估算值与测量值逐项对照。

自检：为什么 optimizer step 前后显存可能增加？为什么 reserved 高不表示所有内存都在保存活跃张量？

## 面试表达

先按张量用途拆分账本，再根据最大项选优化方案。OOM 发生阶段通常比模型参数量更能提示问题。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [ZeRO](https://arxiv.org/abs/1910.02054)：核对模型状态组成和分片动机。
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：查看内存与算子观测；容量算例是正文教学估算。
