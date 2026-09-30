---
id: opd-vs-sft
title: "On-policy distillation 和 SFT 有什么区别？"
category: agent
difficulty: 进阶
tags: ["OPD","后训练"]
updated: 2026-09-30
summary: "先说清轨迹由谁生成，再说明监督信号从哪里来。"
draft: false
---

## 一句话回答

SFT 常在固定示范数据上训练；on-policy distillation 让学生按自身策略生成，再由 teacher 提供蒸馏信号，使训练更多覆盖学生实际会访问的前缀。

## 核心思路

一种 sampled-token OPD 流程：

1. 学生根据任务生成输出。
2. teacher 对学生生成的 token 计算条件 logprob。
3. 用 teacher 与学生的概率差构造训练信号。
4. 更新学生并继续采样。

OPD 不只有一种 loss，实现时应说明 sampled-token、全词表 KL 或其他具体变体。

## 面试追问

### 失败轨迹也可以用于 OPD 吗？

可以。失败轨迹中的 token 可以获得正或负的蒸馏优势，并不等于对所有错误文本做正向 SFT。但 teacher 的条件概率也不等价于最终任务成功率。

### 和 RLVR 有什么区别？

RLVR 的关键监督来自可验证的任务结果；纯 OPD 可以不使用任务 reward。二者可以组合，但必须说明各项权重与优势构造。

## 常见误区

- 把“teacher 给 logprob”说成“teacher 重新生成标准答案”。
- 把训练 loss 下降直接等同于任务得分提升。


## 更精确地理解“学生分布”

on-policy 的关键是前缀来自当前或近期学生策略，而不是只让 teacher 再生成一套答案。teacher 可在学生实际走到的状态上给分布信息；是否使用全词表 KL、何种 KL 方向、如何混合示范数据，需按实现定义。

例如固定前缀上的 reverse KL 是 sum_v pi_student(v)*log(pi_student(v)/pi_teacher(v))。对学生采样 token 计算 logprob 差可用作 Monte Carlo 估计的一部分；但样本来自随参数改变的分布，对采样项做何种 stop-gradient、用何种梯度估计会改变更新。不能随意把一个 logprob 差当作完整、普适的 OPD loss。

## 数据与验证检查

检查 tokenizer 和词表是否相同，teacher/student 是否看到同一段真实上下文，温度缩放与概率归一化是否一致。词表不同的模型不能直接按 token ID 比较全词表概率；须另定义对齐或使用文本级方案。

teacher 在失败前缀上给出的高概率不代表最终成功，因此同时看任务结果、分布差异和学生探索范围，不能只看蒸馏 loss。

## 小实验与自检

在只有三个 token 的玩具词表中，给出学生与 teacher 的概率，手算 forward/reverse KL；再只采样一个 token，观察估计波动。这是建议练习，未声称站内已复现。

## 原始资料

调研日期：2026-09-30。[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) 展示学生生成样本与可选分布匹配目标。本文不把某一个 KL 或 sampled-token 实现定义为全部 OPD。关联：[后训练比较](#q=sft-dpo-rl)。
