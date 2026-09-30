---
id: dit-architecture
title: "DiT 怎样把 Transformer 用于扩散？它和 LLM 相同在哪里？"
category: diffusion
difficulty: 进阶
tags: ["P1", "DiT", "Transformer"]
updated: 2026-09-30
summary: "梳理 latent patch、时间条件、adaLN 与分辨率造成的 token 成本。"
draft: false
---

## 从问题出发

Transformer 能预测文本，也能去噪图像，但相同骨干不代表相同训练目标。DiT 将图像 latent 切成 patch token，对整段带噪 latent 做条件预测。

先修：[Latent Diffusion](#q=latent-diffusion)、[Attention](#q=attention-complexity)。

## 结构拆解

```text
noisy latent → patch embedding + 位置编码 → Transformer blocks → unpatchify
                                     ↑
                           时间 t 与条件（如类别）
```

原始 DiT 重点研究类别条件图像生成。其 adaLN-Zero 用条件产生归一化调制与残差门控参数，以特定零初始化方式改善训练。不能直接把“原始 DiT”讲成默认带文本 cross-attention；后续文本图像架构有不同条件设计。

LLM 的 causal decoder 通常预测下一个离散 token；常见 DiT 对当前带噪图像 token 同时预测连续目标，并使用双向空间注意力。二者都使用 attention/MLP/残差，但 mask、目标、位置表示与采样流程不同。

## 分辨率为什么敏感？

latent 为 H×W，patch 边长 p，token 数 N=(H/p)*(W/p)。若两轴都扩大 2 倍且 p 不变，N 变为 4 倍，标准稠密 attention 的 N² 项变为 16 倍。总延迟不一定 16 倍，因为投影、MLP、通信和实现也占时间。

增加 patch 大小会减少 token，但可能改变细节表达。增加模型 FLOPs 是论文中的扩展变量，不能据此保证任何任务上“越大越好”。

## 小实验与自检

手算两种分辨率与 patch 大小的 token 数，列出 attention 项和线性层项。若有现成模型，测三档分辨率的时间与显存，解释实测与理论的差异。

自检：把 causal mask 加到图像 token 上会改变什么？为何时间 t 必须作为条件？

## 面试表达

先讲输入如何 token 化，再讲条件进入 block 的位置，最后连接分辨率与算力；这条链能自然串起生成模型与 infra。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)：核对 patchify、条件设计及 adaLN-Zero。
