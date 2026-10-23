---
id: "pt-dpo-derivation"
title: "Q08 · DPO 的四个概率怎么推导，β 和输出变长有什么关系？"
category: "interview"
difficulty: "深入"
tags: ["后训练面试", "优化目标", "字节", "百度"]
updated: "2026-10-23"
summary: "从 KL 正则最优策略到偏好 loss，解释四个序列概率、β 的边界和长度偏差的排查路径。"
draft: false
---

## 60–90 秒回答

DPO 在相应的 KL 正则奖励最大化和偏好概率假设下，把显式 reward 差改写成策略相对 reference 的 log-ratio 差。每个同 prompt 的 chosen/rejected 各需要 current 与 reference 两个序列概率，共四个量；通常只对 response token 求 log-prob 的和。β 同时出现在理论关系和训练 loss 的尺度里，实际效果要结合学习率、margin 和饱和情况分析。若输出变长，先检查偏好数据中的长度关联、EOS 与解码、mask 和 reference 设置，再做长度受控对照，而不是直接认定模型推理增强。

## 白板与例子

KL 正则的最优策略关系可写成 $r(x,y)=\beta\log[\pi^*(y|x)/\pi_{ref}(y|x)]+C(x)$。将其代入同 prompt 的 Bradley–Terry 差值，$C(x)$ 抵消。定义：

$$
\Delta=\log\frac{\pi_\theta(y^+|x)}{\pi_{ref}(y^+|x)}-\log\frac{\pi_\theta(y^-|x)}{\pi_{ref}(y^-|x)},
\qquad L=-\log\sigma(\beta\Delta).
$$

白板检查点：当 current=reference 时，$\Delta=0$、loss 为 $\log2$，但梯度并非 0。对 margin 的导数为 $-\beta\sigma(-\beta\Delta)$，说明 β、饱和和学习率一起决定更新，不能把固定数据上的动态简单说成单调更保守。

## 追问怎么接

**Reference 需要跟着更新吗？** 标准离线 DPO 通常固定参考；迭代或同步变体是另一个具体算法设定，必须说明刷新规则和目标变化。

**把序列 log-prob 的和改平均可以吗？** 可以构造长度归一变体，但目标发生变化。别为了“避免长答案低概率”悄悄改实现后仍声称是原始 DPO。

**Chosen 的绝对概率一定升吗？** 不一定。目标提升相对 pair margin，共享参数下两者都可能下降、rejected 下降更多；可同时监控 chosen NLL、偏好准确率和独立质量。

**变长怎么区分好坏？** 固定解码预算，分桶比较长度、正确率、重复率和正常停止比例；对比长度平衡的 pair，观察是否仍有收益。

## 项目怎么联系

若面试官追问你为何用 OPD 而不是 DPO，可以对比学生状态覆盖、教师概率监督与离线 pair 的差异。没有做过同预算 DPO 实验时，就明确这是方法分析，不把未跑过的基线结果说成亲测。

## 容易答错的点

- 把 prompt 的 log-prob 混入 response 序列分数，或 chosen/rejected 条件不同。
- 认为没有显式 reward model 就没有奖励/偏好假设。
- 说 DPO 保证 chosen 概率上升、保证不遗忘或必然更稳定。

## 闭卷自检

写出一对样本的四个 log-prob，算 margin 和 loss；解释“loss 初值 0.693”为什么不说明数据没有训练信号。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [百度算法岗公开题型汇总](https://www.nowcoder.com/discuss/927018114900922368) | 二次汇总，不按公司认证真题处理；页面公开内容与检索摘要；未读取付费部分；仅作题型依据 |
| [DPO 原论文](https://arxiv.org/abs/2305.18290v3) | 偏好模型、KL 正则最优策略与 DPO loss |
| [TRL DPO 官方文档](https://huggingface.co/docs/trl/en/dpo_trainer) | 偏好格式、reference log-prob 和训练实现 |

**原理只在主文深入维护：** [SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl) · [后训练面试：40个问题驱动SFT、偏好优化、RL与蒸馏学习](#q=post-training-interview)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-clip-kl) · [下一题](#q=pt-grpo-objective)

[//]: # (interview-generated-end)
