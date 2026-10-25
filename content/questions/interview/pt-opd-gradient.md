---
id: "pt-opd-gradient"
title: "Q17 · OPD 与教师 SFT、普通 KD、RL 有什么区别，sampled KL 怎么求梯度？"
category: "interview"
difficulty: "深入"
tags: ["后训练面试", "OPD 与项目"]
updated: "2026-10-23"
summary: "以学生访问状态上的监督解释 OPD，并能区分 KL 值估计、局部梯度和整条轨迹分布的导数。"
draft: false
---

## 60–90 秒回答

OPD 的关键是学生自己生成轨迹，教师在学生实际访问的前缀上提供监督，更新后再采样。教师 SFT 主要学习教师示范出现的历史，普通离线 KD 也未必覆盖学生犯错后的状态；在线 RL 则围绕任务反馈优化策略，监督来源和目标与教师概率蒸馏不同。OPD 的稠密反馈可能有利于学习和恢复，但依赖教师在学生状态上的可靠性与接口能力，也增加生成和教师评分成本。实现时要明确用 full-vocab KL、采样 surrogate 还是别的目标；不能因为输出了一个叫 KL 的标量就认为反向传播正确。

## 原理拆解

### On-policy 描述状态来自谁，KL 方向描述监督怎样学

离线教师示范让学生主要在教师前缀上学习；学生生成让教师看到真实会访问的偏差与恢复状态，缓解访问分布不匹配。On-policy 不自动规定 KL 方向：在相同词表、固定前缀下，可以用教师到学生的 forward KL，也可以用学生到教师的 reverse KL，或者其他散度。[GKD 原论文](https://arxiv.org/html/2306.13649v3)讨论学生轨迹上的蒸馏；[Thinking Machines 的 OPD 实践](https://thinkingmachines.ai/blog/on-policy-distillation/)给出相应训练思路。

固定前缀的 forward KL 梯度为 $-\sum_aq(a)\nabla\log p_\theta(a)$，可通过教师分布软标签直接反传；reverse KL 则对学生概率取期望，梯度含采样分布导数。二者常被解释为覆盖教师质量与避开教师低概率区域的不同倾向，但容量、温度和教师可靠性都会影响实际结果，不能把这种倾向说成保证。

### Sampled reverse KL 为什么需要 score-function 项

令 $f_\theta(a)=\log p_\theta(a)-\log q(a)$。对 $\mathbb E_{a\sim p_\theta}f_\theta(a)$ 求导有两项：$\mathbb E[f_\theta\nabla\log p_\theta]+\mathbb E[\nabla f_\theta]$。后项的期望为零，因此留下带 log-ratio 权重的 score-function 梯度。把采样 token 当固定 index、只对 $f_\theta(a)$ 直接求导，恰好只保留了期望为零的那一项。

可在当前参数点 $\theta_0$ 构造局部 surrogate：$\mathbb E_{a\sim p_{\theta_0}}[\frac{p_\theta(a)}{p_{\theta_0}(a)}\operatorname{sg}(\log p_{\theta_0}(a)-\log q(a))]$，在 $\theta=\theta_0$ 求导匹配该局部梯度。多次复用、旧 advantage、clip 或温度不一致后，必须重新说明是在优化哪个 surrogate，不能沿用“当前分布精确梯度”的结论。

### 两动作算例与完整轨迹的区别

取学生 $p=(0.5,0.5)$、教师 $q=(0.8,0.2)$，用 $p_1=\sigma(z)$ 参数化。局部 reverse KL 的 $\partial L/\partial z=p_1p_2\log[(p_1/q_1)/(p_2/q_2)]\approx-0.347$，梯度下降会提高动作 1 概率。若直接对采样 log-ratio 反传，其期望为 $0.5(1-0.5)+0.5(0-0.5)=0$。数值例说明错误来自梯度估计，而非 KL 标量估计本身。

若完整目标还对学生访问状态 $h\sim d_\theta$ 取期望，则严格求导还包含状态分布随策略变化的影响。将已采样前缀固定、逐位置蒸馏是一种训练选择，不能声称自动求出了整个长程目标的所有导数。稠密教师监督也不等于任务价值：教师可能在坏状态不可靠，最终仍需独立任务反馈检验。

## 白板与例子

固定前缀 $h$、冻结教师 $q$，学生为 $p_\theta$。局部 reverse KL 梯度为：

$$
\nabla D_{KL}(p_\theta\|q)=\mathbb E_{a\sim p_\theta}[(\log p_\theta(a)-\log q(a))\nabla\log p_\theta(a)].
$$

从对 $\sum_a p(a)(\log p(a)-\log q(a))$ 求导出发，额外常数项的期望为零。若采样 index 固定后仅对 $\log p(a)-\log q(a)$ 直接反传，梯度期望反而为零，漏掉了采样分布的导数。

一个只表达**固定前缀、当前分布局部目标**的教学 surrogate：

```python
# Samples must match the stated behavior distribution.
adv = (logq - logp).detach()
ratio = (logp - logp_behavior.detach()).exp()
loss = -(ratio * adv * response_mask).sum() / response_mask.sum().clamp_min(1)
```

需要解释行为分母、支持集、优势是否随当前参数重算、token 归约及使用的实际采样温度。裁剪或复用旧优势会改变 surrogate。上式不能自动代表对完整未来状态访问分布求导的精确轨迹目标。

## 追问怎么接

**只有 teacher 文本 API，能做吗？** 可以设计学生状态上的教师示范/评价等学习，但不能直接声称已计算原定义 full-vocab KL。先确认接口是否支持给定学生 token 的 log-prob、top-k 或完整分布。

**失败轨迹要丢吗？** 区分任务失败、无效状态与系统故障。恢复监督正可能发生在失败前缀；教师也可能在坏状态不可靠，需条件化分析与筛选对照。

**是不是教师更大就一定有效？** 参数多不等目标任务强；还要看状态适配、反馈校准、学生容量和成本。用强 SFT 与同预算实验检验。

## 项目怎么联系

把你的 Cross-Tokenizer、MOPD 与 Agentic OPD 放在这一共同问题下，再分别讲公共监督事件、多任务能力迁移和长程交互组织。先回答共同数据流，再进入自己的实际贡献，不需要把三项工作的细节全塞在开场。

## 容易答错的点

- 直接对 sampled log-ratio 反传，称其为无偏 reverse-KL 梯度。
- 把 teacher、reference、critic、behavior 的职责混用。
- 用局部动作 IS 声称同时纠正全部旧状态分布。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：项目自拟。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [GKD / On-Policy Distillation](https://arxiv.org/abs/2306.13649v3) | 学生生成分布与教师反馈的蒸馏目标 |
| [Thinking Machines：On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 学生轨迹、教师 token 评分与反向 KL 蒸馏 |

**相关专题详解：** [OPD系统整理：概率基础、公式推导、训练实现与面试问答](#q=opd-interview-foundations) · [On-policy distillation 和 SFT 有什么区别？](#q=opd-vs-sft)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-agent-credit-mask) · [下一题](#q=pt-escd-events)

[//]: # (interview-generated-end)
