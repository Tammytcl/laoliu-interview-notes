---
id: "pt-sft-data"
title: "Q02 · SFT 数据怎么筛、怎么配比，loss 应该算在哪些 token？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "SFT 与数据", "阿里", "美团"]
updated: "2026-10-23"
summary: "将数据质量、任务配比、训练目标和模板实现串起来，避免只回答“清洗、去重、增加多样性”。"
draft: false
---

## 60–90 秒回答

我会先定义能力目标和评测切分，再按来源、任务、难度、长度和正确性审计数据。训练与验证至少按题目、代码仓库或相关来源分组去重，防止近重复泄漏。配比不能只看条数，长回答在 token-mean 训练中可能贡献更多梯度，需要同时看有效 token 和分域效果。实现上用原生 chat template，明确监督全部 assistant 轮还是末轮：用户输入和工具观察通常作为上下文可见，但不参与生成 loss。最后做逐样本可视化、少量过拟合检查和分域验证，确认数据、mask、EOS 与目标一致。

## 白板与例子

设有效目标 mask 为 $m_t$，因果监督是：

$$
L_{\mathrm{SFT}}=-\frac{\sum_t m_t\log\pi_\theta(x_{t+1}\mid x_{\leq t})}{\sum_t m_t}.
$$

这是 token 平均的一种写法。序列平均会改变长短样本权重，不能当作同一训练目标。

| 消息或 token | 可被后续 attention 看到 | 典型生成 loss |
| --- | --- | --- |
| 用户需求 | 是 | 0 |
| assistant 生成的工具名和参数 | 是 | 1 |
| 外部工具返回 | 是 | 0 |
| assistant 基于反馈给出的修复 | 是 | 1 |
| padding | 按 padding mask 处理 | 0 |

实际取值还取决于控制 token 是否由模型生成，以及模板能否正确标注 assistant 区间。Packing 也必须明确跨样本 attention、position IDs、EOS 与 loss 边界。

## 追问怎么接

**只保留 teacher 做对的数据好吗？** 正确性筛选适合示范质量控制，但会改变难度和领域分布；看过滤保留率，重新审计覆盖。对 OPD 而言，失败访问状态还可能承载恢复监督。

**Math:Code:IF 按 1:1:1 就均衡了吗？** 不一定。还要考虑 token 长度、损失尺度、难度、采样重复率与关键域的退化约束。先给基线配比，再用分域验证调整。

**训练 loss 降、任务效果不升？** 对比独立验证、模板和解码一致性；检查标签错误、泄漏、格式记忆、停止行为及高频域压过长尾域。不要先宣布“学习率不对”。

## 项目怎么联系

你的 SciDeriv 是 prompt-only 数据，不能拿空标签直接当作常规有答案的 SFT 数据。介绍时应分开说明 prompt 来源、谁生成监督、如何筛选，以及 PHYRD-40 评测参考从未进入生成输入。合版项目可补充真实的各域采样和有效 token 统计。

## 容易答错的点

- 将 loss mask 与 attention mask 混为一谈。
- 在重建多轮样本时删掉报错观察，却保留原修复答案。
- 认为随机按行切分天然消除了同源或同仓库污染。

## 闭卷自检

拿三轮工具交互逐个标出模型生成区间、监督区间、终止标记和第一个 label 的预测位置，再解释一个长响应为何可能比多个短响应贡献更多梯度。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [美团算法一面面经（2025）](https://www.nowcoder.com/feed/main/detail/7197e4d2e8934c76a3bc65aedaf1afeb) | 经历自述，招聘真实性未独立核验；公开正文；仅作题型依据 |
| [阿里国际 AI 算法公开问答](https://www.nowcoder.com/feed/main/detail/21fe1ff2b5de4c6fa2a56a27b54f3d3c) | 带参考答案的公开整理，非技术权威；公开正文；页面仅显示月日，不推断年份；仅作题型依据 |
| [TRL SFT 官方文档](https://huggingface.co/docs/trl/en/sft_trainer) | 数据格式、assistant-only loss、packing 与配置边界 |

**原理只在主文深入维护：** [LLM 训练能并行，为什么生成还要逐 token？](#q=causal-lm-training) · [SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-method-selection) · [下一题](#q=pt-lora-budget)

[//]: # (interview-generated-end)
