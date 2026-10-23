---
id: "pt-lora-budget"
title: "Q03 · 全参、LoRA、QLoRA 怎么选，量化后为什么还会 OOM？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "SFT 与数据", "字节", "美团"]
updated: "2026-10-23"
summary: "先做显存账本，再讨论低秩容量和量化误差；参数高效不等于所有训练成本都同比下降。"
draft: false
---

## 60–90 秒回答

全参微调更新整个模型，容量和适配范围更大，但梯度与优化器状态开销高。LoRA 冻结基础权重，训练低秩增量，主要减少可训练参数及对应状态；基础模型和激活仍然占内存。QLoRA 进一步量化冻结基础权重，用更高精度的计算与 adapter 学习，但量化常数、临时反量化、激活和长上下文并不会都变成 4-bit。我会按能力迁移幅度、卡数、上下文长度、部署合并需求与实测结果选方案，而不是只按模型参数大小。OOM 时先定位发生阶段，再用峰值内存证据决定是减 batch、重算激活、分片还是改变 rollout 配置。

## 原理拆解

### 低秩更新怎样开始学习

对一个线性层，LoRA 将增量限制为 $\Delta W=(\alpha/r)BA$。常见初始化是 $A$ 随机、$B=0$，使初始输出等于基础模型；第一次反向时 $B$ 可以获得梯度，而 $A$ 的梯度受零 $B$ 影响，随后两者共同学习。若两者都初始化为零，乘积结构会让双方梯度都为零。这是低秩参数化的直接结果，而不是调大学习率能解决的问题。

Rank 决定单个线性增量的秩上限；target modules 决定允许在哪些计算路径改变行为。只调 Q/V 与同时调 attention、MLP 的可训练范围不同。增加 rank、扩大模块覆盖、增加训练数据是三个不同方向，应该在相同预算和能力评测下比较，不能单凭 adapter 参数越多就判断越好。

### QLoRA 省掉的是哪一部分

QLoRA 将冻结的基础权重以低位格式存储，在计算时结合反量化与较高精度计算，把梯度传向 adapter。NF4 利用权重分布设计编码，double quantization 压缩量化常数，paged optimizer 用分页机制管理内存压力；它们分别改变存储和内存管理，不代表激活、adapter、优化器与通信全部变成四位。[QLoRA 原论文](https://arxiv.org/abs/2305.14314)

冻结基础权重可免其参数梯度与优化器状态，但为了求 adapter 的梯度，网络反向仍需相应输入和中间激活。长上下文、完整词表 logits、attention 临时量及多轮拼接可能主导峰值；因此“可训练参数很少”与“backward 显存很高”可以同时成立。

### 从 OOM 阶段推到改法

加载即失败，先看基础权重精度、量化与设备映射；forward 峰值，查长度、padding、attention 与 logits；backward 峰值，查激活保留、重算和实际解冻模块；optimizer step 才失败，查状态延迟初始化；rollout 时失败，则另查 KV 和并发。每次调整同时记录峰值显存、有效 tokens/s 与任务质量。部署前还要验证 adapter 合并或热加载后的输出、量化误差、模板和权重版本，特别是在线训练侧与生成侧能否准确同步。

## 白板与例子

对于 $W\in\mathbb R^{d_o\times d_i}$，LoRA 使用：

$$
W'=W+\frac{\alpha}{r}BA,\quad
A\in\mathbb R^{r\times d_i},\quad B\in\mathbb R^{d_o\times r}.
$$

新增参数为 $r(d_i+d_o)$。假设方阵维度 4096、rank 为 16，新增 131,072 个参数，而原矩阵有 16,777,216 个参数；这是单层参数算例，不是总训练显存缩小 128 倍的结论。

调 rank 与 target modules 时固定数据、预算和起点，至少比较任务效果、遗忘、峰值显存和训练时间。LoRA 合并通常在较高精度权重上进行；直接修改低位量化存储、再量化会引入额外误差，需重新评测。

## 追问怎么接

**LoRA 为什么还能学到复杂变化？** 多层低秩增量经网络非线性组合，不等于整网只有一个低秩输出；但不保证所有能力变化都可被低 rank 捕获。

**QLoRA forward 不 OOM，backward OOM？** 查激活保留、梯度峰值、重算设置、attention 临时张量及实际可训练模块，而非继续压基础权重。

**能和在线 RL 搭配吗？** 可以设计组合，但还要确认 rollout 引擎加载 adapter、参数更新/同步以及保存与恢复都与训练侧一致。不能只证明 SFT 能跑，就声称 RL 全链路兼容。

## 项目怎么联系

用你实际训练过的模型说明为何选全参或 adapter，只报告有记录的 rank、模块、精度与卡型。对 27B 任务合版可以讲容量与部署约束；没有做过 adapter 对照，就把它作为后续实验，不说成已验证结论。

## 容易答错的点

- 将 NF4 讲成普通整数 INT4，或说训练全程都是 4-bit。
- 用 LoRA 参数占比直接推导总 GPU 显存和速度收益。
- 忽略多 adapter 组合后模板、量化和合并误差的验证。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法面试记录（2024）](https://www.nowcoder.com/feed/main/detail/30c5b57a0acb4a0f9d2250cafadf94b1) | 经历自述，招聘真实性未独立核验；公开正文；仅作题型依据 |
| [美团算法一面面经（2025）](https://www.nowcoder.com/feed/main/detail/7197e4d2e8934c76a3bc65aedaf1afeb) | 经历自述，招聘真实性未独立核验；公开正文；仅作题型依据 |
| [LoRA 原论文](https://arxiv.org/abs/2106.09685v2) | 低秩增量、训练参数及合并 |
| [QLoRA 原论文](https://arxiv.org/abs/2305.14314) | 4-bit 基础权重、量化常数及 adapter 训练 |

**相关专题详解：** [LoRA 到底省了什么？为什么用了 LoRA 仍会 OOM？](#q=lora-finetuning) · [01 · 训练一步在做什么？从梯度、精度到显存账本](#q=infra-training-step)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-sft-data) · [下一题](#q=pt-reward-preference-data)

[//]: # (interview-generated-end)
