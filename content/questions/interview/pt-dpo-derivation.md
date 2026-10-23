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

## 原理拆解

### 第一步：把带 KL 的最优策略解出来

固定 prompt $x$，对分布 $\pi$ 最大化 $\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log[\pi(y)/\pi_{ref}(y)]$，并加上概率和为 1 的拉格朗日约束。对 $\pi(y)$ 求导并置零，得到 $r(y)-\beta(\log[\pi(y)/\pi_{ref}(y)]+1)+\eta=0$，故：

$$
\pi^*(y|x)=\frac{\pi_{ref}(y|x)e^{r(x,y)/\beta}}{Z(x)}.
$$

这一步假设在相应支持集上能优化分布。实际参数化网络、有限数据和优化步数并不保证达到这个最优解。

### 第二步：用偏好差消掉配分函数

重排可得 $r(x,y)=\beta\log[\pi^*(y|x)/\pi_{ref}(y|x)]+\beta\log Z(x)$。在同 prompt 的偏好对中，$\beta\log Z(x)$ 抵消，代入 Bradley–Terry 概率，再用 $\pi_\theta$ 参数化，得到前述 DPO loss。Reference 的两个序列分数通常冻结；current 两个分数带梯度。每个序列 log-prob 是其 response token 条件 log-prob 的和，通常还要明确 EOS 和截断。[DPO 原论文及附录推导](https://arxiv.org/html/2305.18290v3)

设 $z=\beta\Delta$，则

$$
\nabla L=-\beta\sigma(-z)\,[\nabla\log\pi_\theta(y^+|x)-\nabla\log\pi_\theta(y^-|x)].
$$

它增大相对 margin；参数共享时 chosen 的绝对概率仍可能下降。已明显偏向 chosen 的 pair 权重较低，矛盾标签与难判 pair 则可能长期贡献梯度。

### β 为什么不能只背成“越大越保守”

对固定 reward 的理论最优策略，大 $\beta$ 表示更强 KL 约束；而在固定离线偏好数据的训练中，$\beta$ 同时进入梯度系数与 sigmoid 饱和。初始化 current=reference 时 $z=0$，梯度系数是 $\beta/2$；所以不控制学习率、训练步数和数据 margin，不能直接用理论最优关系预测每一步的实际变化。

输出变长时分别检查：偏好标签是否奖励长度、长短负样本是否不平衡、EOS 是否被监督、截断是否切掉答案以及 chosen/rejected 是否按同条件评分。缓存 reference log-prob 可节省算力，但仅对固定模板、分词、截断和参考权重有效；任一项改变均应重新计算。实现接口见 [TRL DPO Trainer](https://huggingface.co/docs/trl/en/dpo_trainer)。

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

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [百度算法岗公开题型汇总](https://www.nowcoder.com/discuss/927018114900922368) | 二次汇总，不按公司认证真题处理；页面公开内容与检索摘要；未读取付费部分；仅作题型依据 |
| [DPO 原论文](https://arxiv.org/abs/2305.18290v3) | 偏好模型、KL 正则最优策略与 DPO loss |
| [TRL DPO 官方文档](https://huggingface.co/docs/trl/en/dpo_trainer) | 偏好格式、reference log-prob 和训练实现 |

**相关专题详解：** [SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl) · [后训练面试：40个问题驱动SFT、偏好优化、RL与蒸馏学习](#q=post-training-interview)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-clip-kl) · [下一题](#q=pt-grpo-objective)

[//]: # (interview-generated-end)
