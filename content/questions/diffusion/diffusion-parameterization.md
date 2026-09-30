---
id: diffusion-parameterization
title: "epsilon、x0、v 和 score 预测，到底在预测什么？"
category: diffusion
difficulty: 进阶
tags: ["P0", "参数化", "Score"]
updated: 2026-09-30
summary: "固定一套时间和系数约定，推导不同预测目标的转换与训练权重差异。"
draft: false
---

## 从问题出发

换 checkpoint 后图像崩坏，可能不是模型坏了，而是 scheduler 把输出当成了另一种物理量。先修：[DDPM](#q=ddpm-denoising)。

## 同一个扰动状态，不同学习目标

用 x_t = a_t*x_0 + s_t*epsilon，其中 a_t²+s_t²=1。这里 s_t 是噪声幅度，不能与别的文献中的“时间”或“方差”混用。

```text
epsilon-prediction: 直接估计 epsilon
x0-prediction:      直接估计干净数据 x_0
v-prediction:       v = a_t*epsilon - s_t*x_0
由 v 转换：         x0_hat = a_t*x_t - s_t*v_hat
                   epsilon_hat = s_t*x_t + a_t*v_hat
score(x_t,t) = grad_x log p_t(x_t)
score_hat = -epsilon_hat / s_t    （s_t > 0，高斯扰动条件下）
```

v 的两个转换式来自一个二维正交变换，代回原式即可验证。score 的关系则通过高斯扰动下的条件噪声均值得到；网络输出并非每张图真实抽样噪声的精确值。

## 为什么换目标会影响优化？

即使目标可以互相转换，“各自使用同样权重的 MSE”也不是同一个优化问题。epsilon 误差折算为 x0 误差时会乘上 s_t/a_t；高噪声端 a_t 很小时更明显。时间采样分布、SNR 权重、端点处理共同决定模型重视哪个区域。

这里的 v-prediction 与 [Flow Matching](#q=flow-matching) 中的速度场名字相近，但不能直接代换；路径和时间约定决定含义。

## 工程检查

检查 checkpoint 配置、scheduler 的 prediction_type、训练噪声日程、推理时间网格及输入缩放。转换公式正确也不意味着训练日程随意可换。端点 a_t=0 或 s_t=0 时，含除法的表达要单独处理。

## 小实验与自检

只做代数实验就够：随机生成 x0、epsilon 和非端点 a_t，用上述公式构造 v，再恢复两者；记录最大误差。随后给预测量加同样幅度扰动，画出不同 t 的 x0 恢复误差。

CPU 代数检查记录（2026-09-30）：用 a_t=0.6、s_t=0.8 的随机小向量验证了 v 到 x0/epsilon 的逆变换；这只验证公式，不表示已完成扩散模型训练。

自检：为什么低噪声端 score 公式需要谨慎？为什么相同 MSE 数值不能跨参数化比较？

## 面试表达

先给定义和时间约定，再讲可转换但优化权重不同，最后用 scheduler 配置不匹配的故障说明工程意义。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [DDPM](https://arxiv.org/abs/2006.11239)：用于核对噪声参数化；推导由正文固定约定展开。
- [Diffusers schedulers](https://huggingface.co/docs/diffusers/using-diffusers/schedulers)：查看 prediction_type 与训练、推理配置的配合。
