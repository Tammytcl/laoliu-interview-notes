---
id: "pt-gae-credit"
title: "Q06 · Critic 已经估值了，为什么还要 GAE？终止和超时怎么处理？"
category: "interview"
difficulty: "深入"
tags: ["后训练面试", "优化目标", "字节"]
updated: "2026-10-23"
summary: "区分状态价值、优势估计和边界处理，能用一个两步例子解释 critic 不是“每轮好坏标签器”。"
draft: false
---

## 60–90 秒回答

Critic 给出从当前状态继续执行的预期回报，而策略梯度需要评价某次动作相对该状态平均水平的好坏。GAE 把多步 TD residual 组合成优势，利用 critic 降低方差，通过 λ 调节估计的偏差与方差。实现前必须定义 reward、折扣、状态边界和截断含义：真实终止不再 bootstrap，时间限制如果只是截断而任务仍可继续，可能要 bootstrap，但不能把新任务的 value 接上旧任务。不能用“失败轨迹所有轮都负分”代替这个推导。

## 原理拆解

### 从 policy gradient 到 baseline

策略梯度需要 $\mathbb E[\nabla\log\pi(a|s)Q^\pi(s,a)]$。减去只依赖状态的 baseline 不改变其期望，因为 $\sum_a\pi(a|s)\nabla\log\pi(a|s)=0$；选 $V^\pi(s)$ 可把动作相对状态平均水平的好坏写成 advantage。Critic 估出 $V$，并不直接给出某个已执行动作的真实 $Q$，还需要奖励和后续状态来形成样本估计。

### GAE 为什么是 TD residual 的加权和

一步估计是 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$；两步、三步 return 减去 $V(s_t)$ 可依次写成 $\delta_t+\gamma\delta_{t+1}$ 等。对这些多步估计做几何加权，得到 $\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$，实际在轨迹边界截断。较小 $\lambda$ 更依赖短程 bootstrap，较大 $\lambda$ 让更远的实际回报进入估计；value 不准时前者更易受估值误差影响，后者往往有更高采样方差。[GAE 原论文](https://arxiv.org/abs/1506.02438v6)

这里的“偏差与方差”需要给出针对哪个回报定义、critic 和边界的条件，不应承诺某个 $\lambda$ 对所有任务最优。训练 critic 常以 $\hat R_t=\operatorname{sg}(\hat A_t+V_{old}(s_t))$ 回归；若先把 advantage 标准化再加回 value，当作原尺度 return，就会破坏这个目标。

### 终止、截断与模型上下文长度不是同一个边界

任务真实成功/失败终止，后续价值为零；仅因采样窗口结束而截断，若定义的任务仍可继续，可用截断状态的 value bootstrap，但不把下一条独立 episode 接上递推。若产品本身规定最大步数即任务失败，那么到预算就是该有限时域任务的终止，不能套“所有超时都 bootstrap”。状态最好包含剩余预算，否则相同历史在不同剩余步数下的价值可能混在一起。

工具执行耗时还涉及折扣单位：按 token、turn 或墙钟折扣是不同目标。多轮 Agent 把一个响应视作一项宏动作时，可以在 turn 边界定义 return；若要和 token 层传播结合，应明确中间工具观察是否消耗决策步、如何处理可变响应长度，不能仅把所有 mask 乘在同一处。

## 白板与例子

使用两个不同的边界 mask：

$$
\delta_t=r_t+\gamma b_tV(s_{t+1})-V(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

$b_t$ 控制是否 bootstrap，$c_t$ 控制优势递推是否跨过当前边界。环境语义不同，二者不一定相同。

手算例：两步、$\gamma=\lambda=1$，价值为 $V(s_0)=0.4,V(s_1)=0.6$，第一步奖励 0、第二步真实终止奖励 1。则 $\delta_0=0.2,\delta_1=0.4$，所以 $A_1=0.4,A_0=0.6$，相应 target $A_t+V_t$ 都为 1。它不是把总分 1 原封不动当作每步 advantage。

## 追问怎么接

**λ=0 和 λ=1 是什么？** 前者只使用一步 TD residual；后者累积多步 residual。在合适边界和 bootstrap 下接近相应 Monte Carlo return 减 value；截断、value 误差和折扣条件要说明。

**Critic 足够准，还需要 GAE 吗？** Value 只给平均预期，不包含这次动作实际后果。还要借助 reward 和后继价值估计 advantage；精确 value 也不等于已知精确 action value。

**长程代码任务失败，前面操作是否都不值得学？** 不一定。好操作可以出现在失败轨迹中；终局回报只能监督后续结果价值，不能直接提供每轮因果贡献。过程反馈、反事实比较或更细状态建模要另外设计。

## 项目怎么联系

可联系你在线 critic 的训练目标：有效终局结果用于估计学生从状态继续执行的未来任务价值。如果用行动前后 value 差调节 OPD 权重，应解释这是进展代理及其待验证假设，不称为精确因果贡献或直接套用 GAE。

## 容易答错的点

- 不说明折扣、value target 和 terminal mask 就背 GAE 公式。
- 将超时、模型输出超长、沙箱故障和任务真实失败全部混为终止。
- 认为 GRPO 必须像 PPO 一样训练 critic、计算 GAE。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [GAE 原论文](https://arxiv.org/abs/1506.02438v6) | TD residual、优势估计与偏差方差 |

**相关专题详解：** [Monte Carlo TD与GAE怎样分配回报](#q=rl-returns-gae) · [Actor Critic从回报到策略梯度](#q=rl-actor-critic)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-ppo-four-models) · [下一题](#q=pt-clip-kl)

[//]: # (interview-generated-end)
