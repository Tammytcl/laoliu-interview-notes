---
id: gpu-utilization
title: "显存占满，为什么 GPU 利用率仍然很低？"
category: systems
difficulty: 进阶
tags: ["GPU","性能分析"]
updated: 2026-09-30
summary: "从数据供给、同步等待和设备忙碌率三个层次定位问题。"
draft: false
---

## 一句话回答

显存占用描述容量，GPU 利用率描述计算活动。权重、KV cache 常驻显存时，设备仍可能在等待数据、工具或同步屏障。

## 核心思路

先分清指标，再分阶段定位：

| 观察 | 可能原因 | 下一步 |
| --- | --- | --- |
| 有请求时忙，整体很闲 | 请求供给不连续 | 看队列和阶段时间线 |
| 推理等待工具 | CPU、I/O 或环境延迟 | 分解工具耗时 |
| 大部分 worker 已结束 | 长尾和 batch barrier | 看逐样本完成时间 |
| 始终有任务仍较慢 | 带宽、通信或 kernel 开销 | 做 profiler 分析 |

## 面试追问

- 如何区分系统调度瓶颈与单个 kernel 瓶颈？
- 为什么不能只看平均 GPU utilization？
- 应该优先加卡还是提高并发？

## 常见误区

利用率不是模型 FLOPs 利用率。应同时报告端到端吞吐、GPU-hour、延迟分布和任务质量，避免单一指标优化。


## 从症状走到证据

第一步统一指标：显存容量、设备忙碌率、SM 活动、内存带宽与 MFU 不是同一量。然后按端到端时间线拆开 CPU 数据读取、H2D、forward/backward、通信、工具等待与队列。

若设备空闲，先查供给和屏障；若一直忙却慢，再查小 kernel 启动、内存带宽和通信。看到 all-reduce 时间长，也要判断是通信本身还是某个 rank 较晚进入同步。

## 测量的常见陷阱

GPU 执行异步，直接用 CPU 时钟包围调用可能只测到提交时间。测单段 GPU 计算可用 CUDA event，测完整区间可在正确边界同步；不要在每个小算子后同步导致负载被测量方式改变。

先 warmup，固定形状与 dtype，再多次测量。profiler 有额外开销，应采样有代表性的时间窗口，不能把带重型 profile 的数字直接当生产吞吐。

## 小实验与自检

对同一小训练脚本分别人为增加 DataLoader 延迟、增大 batch 和增加同步等待。画阶段占比，观察“显存高但空闲”和“设备忙但吞吐低”的不同。这里是练习设计，未在站点宣称任何硬件实测结果。

自检：CPU 用时接近零为什么不说明 GPU 算子很快？提高 microbatch 后吞吐更好、p95 延迟更差是否矛盾？

## 原始资料

调研日期：2026-09-30。[PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 提供 CPU/CUDA 活动、内存和算子观测方法。诊断表为本文工程整理；接着读 [rollout 系统](#q=rollout-systems)。
