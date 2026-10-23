---
id: "pt-grpo-variants"
title: "Q11 · DAPO、Dr. GRPO、GSPO 分别改了什么，怎样比较才公平？"
category: "interview"
difficulty: "深入"
tags: ["后训练面试", "GRPO 与奖励", "字节"]
updated: "2026-10-23"
summary: "用优势、ratio、surrogate 和归约四个轴比较变体，指出每个改动试图解决的具体问题。"
draft: false
---

## 60–90 秒回答

我会先固定基础 GRPO，再逐项比较。DAPO 包含非对称裁剪、有效组动态采样、token 级损失聚合及超长响应处理等配方；Dr. GRPO 分析逐响应长度与组标准差归一带来的优化偏差，并调整这些缩放；GSPO 将长度归一的序列 likelihood ratio 用于序列级裁剪与优化。它们不只是同一个 loss 换名字，也不在同一层修改。选型时要控制模型、prompt、奖励、解码、KL、生成预算和停止规则，再通过逐项消融找原因；不能一次换完整配方后把全部收益归因到某一个公式。

## 原理拆解

### DAPO：分别处理探索、无效组与长响应

非对称 clip 将上界和下界分开，让提高低概率有利动作的空间可以不同于压低不利动作的空间；这并不保证 entropy 一定上升。动态采样补足有奖励差异的组，但带来额外生成和分布筛选。Token 级聚合改变长短响应的梯度份额；超长处理避免把长度截断一概当作可靠负反馈，同时仍需约束无效延长。这几项分别作用于 surrogate、数据选择、归约和奖励，比较时应独立开关。[DAPO 原论文](https://arxiv.org/html/2503.14476v2)

### Dr. GRPO：区分长度分母与奖励标准差

对相同 $A_i$，原始每响应除以 $T_i$ 会让长短样本的每个 token 系数不同；组标准差则让任务的回报离散程度进一步缩放梯度。Dr. GRPO 针对这些因素，去除所分析的标准差和响应长度归一，采用固定因子归约。这里的无偏性针对其声明的优化目标与假设，不能扩展成“所有任务都不应长度归一”。尤其把分母换成当前 batch 的随机总长度，与换成固定常数不是同一步改动。[Dr. GRPO 分析与实现](https://arxiv.org/html/2503.20783v2)

### GSPO：把裁剪单元改成响应

GSPO 先算长度归一序列 ratio $s_i$，再优化 $\min(s_iA_i,\operatorname{clip}(s_i)A_i)$。未裁剪分支的导数满足：

$$
\nabla s_i=\frac{s_i}{T_i}\sum_t\nabla\log\pi_\theta(a_{it}|h_{it}).
$$

因此一条响应共享同一个 ratio 系数与裁剪判定；GRPO 则可对不同 token 分别裁剪。例子：两个 token ratio 为 2 和 0.5，几何平均是 1，序列级判定可能不裁剪，而 token 级两项都有大幅偏离。这个例子说明行为差异，也说明序列平均可能掩盖局部尖峰，不能把它当作所有 token 都接近旧策略的证明。其 MoE 稳定性收益是论文设定下的实证结果。[GSPO 原论文](https://arxiv.org/html/2507.18071v2)

### 同预算比较先固定哪几个变量

先固定起点、奖励、数据顺序、采样温度、上下文预算、reference 设置与训练 token 口径，再逐轴换 advantage、ratio、clip、归约。动态采样另报全部候选生成量；长响应方法另画成功率—推理 token 曲线；系统吞吐另报全部资源。一次替换整套 recipe 的结果只能证明该组合更好，无法独立证明某个目标项的贡献。

## 白板与例子

| 方法 | 优势 / 归一 | ratio 或 surrogate | 额外关注 |
| --- | --- | --- | --- |
| 基础 GRPO | 组相对回报，常用 std 缩放 | token ratio + clip | 响应与 token 归约 |
| DAPO | 组反馈与动态有效组采样 | 上下界可非对称 | token 聚合、超长处理、采样成本 |
| Dr. GRPO | 去除所分析的组 std / 响应长度偏置 | 不等于自动换序列 ratio | 固定归约因子与目标差异 |
| GSPO | 组相对回报 | 长度归一序列 ratio，再做序列级裁剪 | 对长响应及 MoE 稳定性的设定 |

GSPO 的关键量可以写成：

$$
s_i=\exp\left(\frac1{T_i}\sum_t\log\frac{\pi_\theta(a_{i,t}|h_{i,t})}{\pi_b(a_{i,t}|h_{i,t})}\right).
$$

它是 token ratio 的几何平均形式，不是未经归一的完整轨迹重要性权重。两条响应长度为 2 与 8 时，序列平均可能让每条总权重相同；全 token 平均则让后者拥有更多 token 项，二者梯度不相同。

## 追问怎么接

**DAPO 只是把上界调大吗？** 不止。要逐项说优化问题、实现变化和成本；其原论文具体配方还涉及移除 KL 等设置，不应泛化成所有任务的必需条件。

**GSPO 就解决了所有 off-policy 吗？** 没有。长度归一和裁剪改善特定 surrogate 的行为，不恢复任意旧状态访问分布；完整轨迹和局部动作校正仍要区分。

**怎么设计最小消融？** 先只改一个轴，如固定采样与其他设置，仅改 token/sequence reduction；另设动态采样实验，按全部原始生成量核算成本。

## 项目怎么联系

你做跨词表蒸馏时同样应分开方法目标、概率近似与 Infra 调度。面试官问“为什么比 baseline 好”，用同目标不同实现、同实现不同目标的对照矩阵，比罗列算法名更容易建立证据。

## 容易答错的点

- 把 GSPO 的长度归一 ratio 当作严格 joint-trajectory IS。
- 把 Dr. GRPO 的分析结论扩展成所有长度归一都错误。
- 把汇总帖里的改进题说成每家公司都问过所有变体；此处 Dr. GRPO/GSPO 为论文延伸比较。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [DAPO 原论文](https://arxiv.org/abs/2503.14476v2) | 非对称裁剪、动态采样、token 归约与超长样本处理 |
| [Dr. GRPO 分析论文](https://arxiv.org/abs/2503.20783v2) | 组标准差、长度归一与优化偏差 |
| [GSPO 原论文](https://arxiv.org/abs/2507.18071v2) | 长度归一的序列 ratio 与序列裁剪 |

**相关专题详解：** [RL算法总览 从Actor Critic到PPO和组策略优化](#q=rl-policy-optimization-map) · [GRPO从组奖励到token梯度的完整流程](#q=grpo-rlvr)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-zero-variance-groups) · [下一题](#q=pt-rlvr-hacking)

[//]: # (interview-generated-end)
