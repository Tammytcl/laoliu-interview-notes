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

## 闭卷自检

拿两动作 softmax 手推 full-vocab 梯度，与“固定采样 index 的直接反传”做比较；说出代码中的哪个 detach 正在表达你的数学假设。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：项目自拟。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [GKD / On-Policy Distillation](https://arxiv.org/abs/2306.13649v3) | 学生生成分布与教师反馈的蒸馏目标 |
| [Thinking Machines：On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 学生轨迹、教师 token 评分与反向 KL 蒸馏 |

**原理只在主文深入维护：** [OPD面试基础：目标、梯度、训练代码与强基线](#q=opd-interview-foundations) · [On-policy distillation 和 SFT 有什么区别？](#q=opd-vs-sft)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-agent-credit-mask) · [下一题](#q=pt-escd-events)

[//]: # (interview-generated-end)
