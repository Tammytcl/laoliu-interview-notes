---
id: classifier-free-guidance
title: "CFG 如何增强条件控制？为什么 guidance 太大反而变差？"
category: diffusion
difficulty: 基础
tags: ["P0", "CFG", "条件生成"]
updated: 2026-09-30
summary: "用有条件与无条件预测的组合理解控制强度、分布偏移与成本。"
draft: false
---

## 从问题出发

同样提示词，调高 guidance 后更符合文字，却可能过饱和、失真。这是方向强化带来的取舍。

## 训练和推理各做什么

训练时随机丢弃条件，使一个网络同时学会有条件和无条件预测。推理时采用常见约定：

```text
epsilon_guided = epsilon_uncond + g*(epsilon_cond - epsilon_uncond)
g=0：无条件分支
g=1：普通条件分支
g>1：沿条件差异方向外推
```

有些论文写成 (1+w)*cond-w*uncond，这时 g=1+w。先说明约定再比较数值。

在理想 score 解释下，这一线性组合强化条件相关方向；网络估计误差和有限步求解会影响实际结果。训练时需要见过空条件；推理时凭空用一个没有训练过的占位字符串并不等于无条件分支。

## 质量与成本

较大 g 会放大两分支差值中的误差，可能牺牲多样性或造成极端纹理。标准双分支 CFG 每步通常需要两个样本的模型计算，可以 batch 合并以减少调用开销，但不能把算力成本算成免费。带内置 guidance 或蒸馏的模型需按其训练接口处理。

“negative prompt”作为负分支条件时已经不是严格的无条件预测，含义要单独说清。

## 小实验与自检

固定 prompt 和 seed，比 g=1/3/7，记录条件一致性、饱和度和吞吐。再用多个 seed 看多样性。观察是否为了某个词牺牲整体构图。

自检：g=1 为什么不需要差值外推？合并成一次 batch forward 后计算量为何仍增加？

## 面试表达与追问

30 秒：CFG 在推理时用有条件和无条件预测的差强化控制，训练依赖条件 dropout。追问时说明尺度约定、成本和强度过大时的误差放大。

关联：[Latent Diffusion](#q=latent-diffusion)、[Flow Matching](#q=flow-matching)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)：查看条件 dropout 和 guidance 的原始定义。
