---
id: "pt-agent-credit-mask"
title: "Q16 · 多轮 Agent 的 reward 怎么分到 turn 和 token，工具返回要不要训练？"
category: "interview"
difficulty: "深入"
tags: ["后训练面试", "系统与 Agent", "字节", "快手"]
updated: "2026-10-23"
summary: "先定义状态、动作、环境观察和目标，再讨论 turn/token 权重及长程信用分配，避免把 mask 当成全部答案。"
draft: false
---

## 60–90 秒回答

多轮 Agent 的状态是实际可见的交互历史，动作可以是一段推理、工具调用或最终回复，工具结果是环境给出的观察。通常只在模型生成的动作 token 上优化策略，观察不算 actor loss 但参与后续条件。终局任务回报可以通过 return、critic/GAE 或组比较形成优势，也能设计过程反馈；它们不能不加区分直接广播成“每轮因果价值”。若使用 turn 权重，还要说清长度归约与监督就绪条件。观察、padding、人工控制 token、终止和跨 turn 递推的 mask 负责不同事情，应逐项核对。

## 白板与例子

教学轨迹：查看文件 → 收到缺依赖错误 → 安装/调整方案 → 通过测试。第二轮的报错是条件，不意味着第一轮必然应被惩罚；模型利用错误反馈恢复正是目标能力之一。

若某个 OPD 设计采用非负 turn 权重，可写：

$$
L=\frac{\sum_j w_j\sum_{t\in\mathrm{response}_j}m_{jt}\,\ell_{jt}}{\sum_j\sum_t m_{jt}}.
$$

这是有效 token 分母的一种明确定义；改成除以加权 token 总数或每 turn 先平均，会改变监督整体尺度或长短 turn 权重。它也不是普通带正负 advantage 的 policy gradient；两种目标需分别解释。

## 追问怎么接

**终局成功就给所有轮相同正优势可以吗？** 可构造 trajectory-level 估计，但信用分配粗，包含的无效尝试也会一起获得信号。要对照更细反馈，不声称天然最优。

**上下文压缩为什么影响训练？** 压缩改变模型的条件历史和状态信息。需记录生成时真正看到的上下文；不能在压缩前后混用同一 log-prob，评测也应使用对应机制。

**怎么防止 critic 权重越学越偏？** 检查有效回报、分布变化、校准、不同任务的预测行为和监督强度边界。独立验证“价值差是否对应真实进展”，并做固定/随机/等权消融。

## 项目怎么联系

你的 Infra 与打分协同设计可拆成两条证据：轮级流式异步是否减少等待；在线 critic 的价值变化加权是否改善学习。先分别对照，再组合，避免把最终收益全归因于某一项或将实验假设当作已证定理。

## 容易答错的点

- 说“tool mask=0，所以模型看不到工具结果”。
- 将 turn-level 加权、GAE 和 PRM 当作同一个数学对象。
- 认为终局回报能精确标注每一步贡献，或认为价值差就是因果效应。

## 闭卷自检

给一个 assistant/tool/assistant 的 token 示意，分别画 loss mask、attention 可见性和 terminal/continuation mask；说明改成每 turn 平均会怎样改变梯度权重。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [快手大模型算法二面公开问答](https://www.nowcoder.com/discuss/876505573997432832) | 公开问答整理，不按认证真题处理；仅公开片段；未读取付费部分；仅作题型依据 |
| [GAE 原论文](https://arxiv.org/abs/1506.02438v6) | TD residual、优势估计与偏差方差 |

**原理只在主文深入维护：** [Agentic RL多轮环境信用分配与训练工作](#q=agentic-rl-training) · [Agentic Code OPD turn框架：从请求到梯度、窗口更新与权重发布](#q=agentic-turn-framework) · [Monte Carlo TD与GAE怎样分配回报](#q=rl-returns-gae)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-rollout-staleness) · [下一题](#q=pt-opd-gradient)

[//]: # (interview-generated-end)
