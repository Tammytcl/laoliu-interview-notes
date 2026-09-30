---
id: consistency-distillation
title: "少步生成如何蒸馏？Consistency 与减少采样步数有什么区别？"
category: diffusion
difficulty: 深入
tags: ["P1", "Consistency", "蒸馏"]
updated: 2026-09-30
summary: "比较推理求解器和训练出的少步模型，建立公平质量与延迟评估。"
draft: false
---

## 从问题出发

直接把 50 步改成 1 步往往不行：原网络学习局部方向，一大步的积分误差可能很大。蒸馏要改变学生的学习问题。

## Consistency 的直觉

在同一条 probability-flow ODE 轨迹上，不同噪声时刻应能映射到同一个低噪声端点。Consistency 模型学习这种映射，支持一次或少次网络评估生成。端点边界条件防止把所有输入映射到无意义常量。

Consistency distillation 利用已有模型提供轨迹上的关联状态；consistency training 则可不依赖预训练 teacher，具体训练构造要看论文版本。不是所有少步蒸馏都属于 consistency，也不是所有 teacher-student 目标都相同。

## 三种提速分别改了什么

| 方法 | 主要改变 | 要核对的条件 |
|---|---|---|
| 减少采样节点 | 数值积分预算 | 原模型在大步长下误差 |
| 换求解器 | 更新方法 | NFE、输出与日程兼容 |
| 少步蒸馏 | 训练目标与学生权重 | 特定步数、guidance 与条件接口 |

蒸馏过程可能引入额外训练成本。学生的 CFG 设置、适用步数和 scheduler 不能照搬 teacher；生成偏差也可能从 teacher 继承。

## 小实验与自检

做一张评估表：teacher 常规步数、teacher 极少步数、学生推荐步数。对每组固定分辨率和 prompt 集，报告 NFE、端到端延迟、多样性与任务质量。论文中的一次生成能力不保证所有新条件上都稳定。

自检：为什么只拟合 teacher 在一个时间点的噪声预测，不必然得到一步模型？边界条件解决什么退化？

## 面试表达

先区分“更便宜地解原来的过程”和“训练一个能直接跨越更大时间跨度的模型”，再说明质量、泛化范围与训练成本。

关联：[DDIM](#q=ddim-sampling)、[Flow Matching](#q=flow-matching)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Consistency Models](https://arxiv.org/abs/2303.01469)：区分 consistency distillation 与直接训练，关注边界条件。
