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
