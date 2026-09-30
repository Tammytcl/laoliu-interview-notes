---
id: flow-matching
title: "Flow Matching 与 Rectified Flow 怎么理解？直线路径为何不保证一步生成？"
category: diffusion
difficulty: 进阶
tags: ["P0", "Flow Matching", "Rectified Flow"]
updated: 2026-09-30
summary: "由噪声到数据的路径推导速度回归，区分条件路径与实际 ODE 轨迹。"
draft: false
---

## 从问题出发

去噪可以理解为学习运输分布的方向。Flow Matching 直接拟合概率路径的速度场，采样时解 ODE。

## 固定约定再推导

本篇 x_0 为噪声，x_1 为数据，时间从 0 走到 1；与 [DDPM](#q=ddpm-denoising) 的命名方向不同。简单线性条件路径：

```text
x_t = (1-t)*x_0 + t*x_1
条件速度 u_t = x_1 - x_0
L = E[||v_theta(x_t,t) - (x_1-x_0)||²]
采样：dx/dt = v_theta(x,t)，x(0) ~ 噪声分布
Euler：x_next = x + delta_t*v_theta(x,t)
```

训练时采样端点和 t，直接构造中间点与速度监督，不需要在每次训练中求解完整 ODE。此处是简单条件路径示例；Flow Matching 也支持别的概率路径。

## 关键区别：每条监督线段与边缘速度场

同一个 x_t 可能对应不同端点配对。平方损失学到的是这些条件速度在该位置的条件均值。沿这个边缘场积分的实际轨迹不必等于某一条训练线段，因此“监督路径直”不推出“一次 Euler 就精确”。

Rectified Flow 以直线插值回归为核心，并可通过 reflow 改善耦合、使轨迹更直。Flow Matching 是更广的框架；二者联系紧密，但所有 FM 不都等于同一种 RF 配方。

## 与扩散的联系与边界

高斯扩散路径也可以纳入速度场描述。具体关系取决于路径、时间方向和输出参数化。仅把 epsilon 网络改名为 v 网络，通常无法完成迁移。少步表现还受向量场误差和求解步长影响。

## 小实验与自检

在二维双峰分布上画训练配对线段，再画学到场的积分轨迹；用 1/4/16 次 Euler 比较分布覆盖。只画好看的单条轨迹不足以验证整个生成分布。

自检：两个相交线段给出的监督方向相反，模型会学到什么？为什么训练不用积分，推理却要积分？

## 面试表达

用“路径 → 速度监督 → 边缘场 → ODE 采样”四步回答，再主动说明直线路径与实际轨迹的差别。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Flow Matching](https://arxiv.org/abs/2210.02747)：看条件路径与边缘速度场的对应关系。
- [Rectified Flow](https://arxiv.org/abs/2209.03003)：看 rectification、耦合和 reflow；不要默认一次回归就得到直轨迹。
