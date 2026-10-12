---
id: rl-actor-critic
title: "Actor Critic从回报到策略梯度"
category: agent
difficulty: 基础
tags: ["P0", "RL", "Actor Critic", "Policy Gradient", "REINFORCE"]
updated: "2026-10-12"
summary: "从状态动作回报推导REINFORCE与baseline，解释actor/critic各自学习什么、TD误差、共享参数和实际源码。"
draft: false
---

## 1. 问题背景 有反馈却没有正确动作标签怎样学习

监督学习通常知道输入对应什么标签；强化学习面对的是“选了一个动作，最后任务得到了多少反馈”。例如 agent 先读文件、改代码、运行测试，十轮后才知道修复是否成功。环境没有给每一轮标准操作，也未告诉第一次读文件究竟贡献了多少。需要一种方法，把整次经历的结果转成“今后哪些动作更值得选择”的更新。

Actor–critic 是一类职责分工：actor 学行动策略，critic 估计在当前状态下能期待多少未来回报，用来判断动作相对好坏并辅助信用分配。它不是必须部署两个独立大模型的产品方案，也不是只有一个固定 loss 的算法。PPO、A2C、A3C 等采用这种思想，但优势估计、更新约束和数据组织不同。

本篇先建立数学对象，再推导为什么不可微的环境反馈也能训练策略，最后把这些对象接到代码与 LLM agent。下一篇[回报 TD与GAE](#q=rl-returns-gae)解决“这个动作应拿多少优势”，随后读[PPO训练一步](#q=rl-ppo-training)。问题背景讲学习困难；以下前置知识只定义需要使用的概念。

## 2. 前置知识 MDP轨迹奖励与折扣

有限 episodic MDP 可写为 $(\mathcal S,\mathcal A,P,R,\rho_0)$。状态 $s_t$ 描述决策时信息，策略 $\pi_\theta(a_t\mid s_t)$ 给动作概率，环境转移 $P(s_{t+1}\mid s_t,a_t)$ 产生新状态，$r_t$ 是这一步反馈。初始分布 $\rho_0$ 和环境规则本篇假设不随 actor 参数变化。

轨迹 $\tau=(s_0,a_0,r_0,\ldots,s_T)$，目标是期望累计回报：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\qquad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

$\gamma\in[0,1]$ 控制远期反馈权重。它不只是“为了稳定而默认0.99”：按 token 折扣和按工具 turn 折扣，表达不同的时间单位。有限任务中取 $\gamma=1$ 很常见，但要说明终止、长度成本和预算。持续任务的存在性和收敛需要另外的条件。

奖励是单步反馈，回报包含后续；分数大小依赖任务定义，未必是概率。若终局奖励1而此前全0，早期动作仍可能影响成功。环境中的 test runner、搜索结果或判题器不必能反向传播；本篇通过动作概率建立梯度，而不是对 shell 的输出调用 backward。

**Fully observed 与 partially observed。** 机器人状态可能近似观测齐全；agent 的文件、网页和外部服务状态往往只被部分观察。实际 policy 可条件于历史 $h_t$，或经过上下文选择后的输入 $c_t$。如果丢掉影响转移的信息，不能仅把模型输入叫 state 就宣称满足 Markov 假设。后续公式的 $s_t$ 也可理解为足够的历史状态表示。

![Actor / critic: one transition](./assets/learning/rl-actor-critic/actor-critic.svg)

本文教学示意。上方沿一次决策的真实信息流阅读，下方列出训练所需的职责和边界。箭头表示数据传递，不表示环境反馈可以直接反向传播；具体公式、mask和梯度在后续章节逐项展开。

## 3. 从REINFORCE推导不可微环境下的更新

轨迹概率分解为：

$$
p_\theta(\tau)=\rho_0(s_0)\prod_t\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

把求导放进期望，利用 likelihood-ratio identity：

$$
\nabla_\theta J
=\mathbb E_\tau\left[\left(\sum_t\gamma^t r_t\right)
\nabla_\theta\log p_\theta(\tau)\right],
\qquad
\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

环境和初始分布不依赖 $\theta$，所以其 log 项导数为0。即使不知道转移概率的具体表达，也能在已访问状态上求动作 logprob 梯度。这说明为什么真实工具不需要可微。若环境中还含可训练、与 actor 共享参数的模块，则上述假设需要重审。

用因果性去掉某动作发生前的奖励，可得到 reward-to-go 形式：

$$
\nabla_\theta J
=\mathbb E_\tau\left[\sum_t\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t\right].
$$

动作不能改变已经发生的奖励，那些过去奖励项只给估计增加噪声。有限长度、$\gamma=1$ 时，该式是每个动作的 score function 乘后续回报。折扣场景若按普通时间点等权采样而省掉 $\gamma^t$，需说明是在使用相应状态分布或一种实践近似；不能悄悄把不同目标写成完全相同的式子。

**REINFORCE** 使用 Monte Carlo 回报估计策略梯度，通常等轨迹结束后可得完整回报。[Williams1992](https://link.springer.com/article/10.1007/BF00992696)是相关原始工作；现代有限轨迹的上述推导用于本课，不声称1992原文的每一种延迟反馈设置都与这里完全一致。[Sutton等1999策略梯度论文](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)进一步阐明策略函数近似与价值辅助的关系。

## 4. Baseline为什么不改变理想期望梯度

取与当前动作无关的状态函数 $b(s)$。在同一状态：

$$
\mathbb E_{a\sim\pi}\left[b(s)\nabla_\theta\log\pi(a\mid s)\right]
=b(s)\sum_a\nabla_\theta\pi(a\mid s)
=b(s)\nabla_\theta1=0.
$$

因此可用 $G_t-b(s_t)$ 替代 $G_t$，在假设成立时保留期望梯度，减少噪声。这个结论依赖 baseline 对当前动作的条件独立性；随意把本样本奖励含在组均值里，并不自动保留同样的有限样本估计。Leave-one-out 会在[算法地图](#q=rl-policy-optimization-map)专门推导。

### 两动作的完整手算

设一个状态，动作1概率 $p=\sigma(\theta)=0.25$，动作0概率0.75；奖励分别2和0。$J=2p=0.5$，直接求导：

$$
\frac{dJ}{d\theta}=2p(1-p)=0.375.
$$

动作1的 $\nabla\log\pi=1-p=0.75$，动作0为 $-p=-0.25$。用 baseline 0.5，两个可能样本的梯度值分别：

| 样本 | 概率 | reward-baseline | score | 单样本梯度 |
| --- | --- | --- | --- | --- |
| 动作0 | 0.75 | -0.5 | -0.25 | 0.125 |
| 动作1 | 0.25 | 1.5 | 0.75 | 1.125 |

加权平均 $0.75\times0.125+0.25\times1.125=0.375$，与直接导数相同。无 baseline 的梯度方差为0.421875，baseline0.5时为0.1875。这是可枚举算例，不是某任务训练结果。

**V也不必是最小方差的baseline。** score 的平方会影响最优常数。本例 baseline1.5使两个单样本梯度都为0.375，方差为0；它却不等于 $V=0.5$。实际 actor–critic 用价值估计有清晰Bellman结构和通用性，不应把“预测平均回报”说成对所有参数方向都达到最小梯度方差。

## 5. V Q A与critic究竟学什么

$$
V^\pi(s)=\mathbb E[G_t\mid s_t=s],\quad
Q^\pi(s,a)=\mathbb E[G_t\mid s_t=s,a_t=a],\quad
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

V 固定当前状态后对策略后续行动取平均；Q 额外固定当前动作；A 是相对当前策略通常表现的差异。reward为5，若该状态通常能拿8，优势可能为负；reward0也可能比其他动作更好。策略优化需要相对信息，不是见到正分就无条件强化。

Critic 常拟合 V，也可拟合 Q 或优势。它预测“当前策略继续下去能取得的回报”，不是环境真值，也不是自动判断代码正确性的 verifier。训练策略持续变，价值函数也要追踪；critic 较准确可减少方差，较差的 bootstrapping 会引入偏差。

**Reward model 与 critic 的差别。** Reward model 可以对给定回答打偏好分，用来定义反馈；critic 估计从当前状态出发的未来累计反馈。Teacher 则可提供另一个模型的动作分布；reference policy 用于KL锚定；old/behavior policy 描述数据从哪来。这五种职责不能因为都输出一个数就互换。

## 6. 一步Actor Critic和TD误差

Bellman关系为：

$$
V^\pi(s_t)=\mathbb E[r_t+\gamma V^\pi(s_{t+1})\mid s_t].
$$

一次观察产生 TD residual：

$$
\delta_t=r_t+\gamma m_tV_\phi(s_{t+1})-V_\phi(s_t).
$$

真实终止时 $m_t=0$；仍可继续的边界需按任务定义 bootstrap。若 value 准确，条件于 $(s_t,a_t)$ 的 $\delta_t$ 期望等于优势；一次样本有噪声，value 不准确时也有偏差。一步 TD 不需要等完整最终回报，代价是更依赖 critic。

```python title="教学实现 一步actor critic的梯度边界"
logprob, value = actor_critic(actual_state, sampled_action)
with torch.no_grad():
    next_value = critic(actual_next_state)
    target = reward + gamma * bootstrap_mask * next_value
    advantage = target - value.detach()
actor_loss = -(logprob * advantage).mean()
critic_loss = 0.5 * ((value - target) ** 2).mean()
loss = actor_loss + value_coef * critic_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这是 PyTorch 教学接线，不是完整训练脚本。reward、target和advantage应作为固定反馈；actor_loss通过logprob更新policy，critic_loss通过当前value更新critic。若 actor advantage 不 detach，actor目标可能沿value分支优化，已经不是原本使用固定优势的策略更新。下一状态target若与当前critic共同反传，也会从semi-gradient TD变成另一个目标。

共享encoder时两个loss都能更新共享参数，但value head与policy head仍有不同监督。也可以完全分离参数和optimizer。共享有资源优势和梯度干扰，分离有额外成本；“actor–critic”只规定职责，不强制某一种部署。

## 7. 真实源码 Actor和Value怎样出现在同一次forward

历史 OpenAI Baselines 固定commit da997060461e3cbf54ca4dc7a67081a731fb6b3b 的[MlpPolicy](https://github.com/openai/baselines/blob/da997060461e3cbf54ca4dc7a67081a731fb6b3b/baselines/pposgd/mlp_policy.py#L27-L41)分别使用 value 和 policy MLP。关键行：

```python title="Baselines真实节选 critic输出每个状态一个数"
self.vpred = U.dense(last_out, 1, "vffinal",
                    weight_init=U.normc_initializer(1.0))[:,0]
```

这里 last_out 来自上面的value分支；输出先是[B,1]，切片为[B]。policy分支另产生分布参数；这份旧连续控制实现并不是LLM的token分类头，不能套它的Gaussian参数布局到语言模型。

[act路径](https://github.com/openai/baselines/blob/da997060461e3cbf54ca4dc7a67081a731fb6b3b/baselines/pposgd/mlp_policy.py#L46-L52)节选：

```python title="Baselines真实节选 行动与价值是不同返回值"
stochastic = tf.placeholder(dtype=tf.bool, shape=())
ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
self._act = U.function([stochastic, ob], [ac, self.vpred])
```

sample/mode决定实际行动方式，vpred是价值预测。Actor给动作或分布，critic给预期回报，不能把value最大的数当成生成token。本文仅静态核历史源码；现代框架通常还有向量环境、截断标记、AMP和分布式处理。

## 8. A2C A3C与PPO是怎样接在这条链上

A2C常指同步的多环境优势actor–critic，A3C采用多个worker异步更新，是[2016原论文](https://arxiv.org/abs/1602.01783v2)的重要路线。多环境增加经验来源，n-step帮助利用较远反馈；异步更新又带来参数版本与数据来源问题。这个“A3C异步”不等于今天多轮LLM中的任意异步队列设计。

PPO增加old-policy比值、裁剪或KL约束，便于在同批数据上做多轮minibatch更新；critic和GAE解决优势估计，PPOsurrogate解决更新幅度，两层各有职责。GRPO/RLOO可用组或leave-one-out统计替代学习V，减少critic成本，但长程信用分配不会因此消失。

| 名称 | 主要改哪一层 | 保留的共同对象 |
| --- | --- | --- |
| REINFORCE | 回报直接估梯度 | policy、采样、return |
| Actor–Critic | 学价值辅助优势 | policy、critic、反馈 |
| n-step / GAE | 回报/优势估计 | TD、bootstrap、边界 |
| A2C / A3C | 多环境与更新组织 | actor–critic、版本 |
| TRPO / PPO | 控制策略变化与复用 | surrogate、采样分布 |
| RLOO / GRPO | 更换baseline/优势估计 | 组采样、策略梯度 |

## 9. LLM和Agent里的状态动作粒度

token级状态是实际prefix，动作是下一个token，V可对prefix输出标量；turn级状态是完成一次工具返回后的历史，动作是整个模型响应/工具调用，其logprob通常是生成token logprob之和。把turn内token平均会改变权重，不再是完整动作概率的log。

工具观察并非policy生成动作，通常不给它actor loss，但会影响后续生成条件。value target与折扣需要按实际决策单位定义；不能先把工具文本插进token序列，再把所有位置都当policy时间步。Context压缩会改变下一轮实际输入，所以保存每轮真实条件很重要。

未来更强的teacher、可验证reward、更稳定的surrogate能帮助训练，但并不能替代正确的状态/动作定义。后续[Agentic RL](#q=agentic-rl-training)和[Agentic OPD](#q=agentic-opd-training)会将这条基础链连接到具体研究。

## 10. 面试问答与讲课检查

**Q01 [定义] Actor–critic究竟是哪两个对象？** Actor是策略分布，critic估计价值/优势。可以共享encoder，也可分离，不必是两个独立大模型。

**Q02 [推导] 环境不可微为何仍能训练？** 对轨迹概率求导，固定环境项没有actor梯度，留下动作logprob的score function乘回报。

**Q03 [因果] 为什么用reward-to-go？** 当前动作不能影响过去奖励，去掉过去项保留理想期望并可减少噪声；折扣和状态采样口径要一致。

**Q04 [baseline] 减一个状态价值为什么不改期望？** 条件于状态，score的期望为0。要求baseline不依赖当前采样动作，有限批统计还需单独分析。

**Q05 [相对性] Reward为正会被抑制吗？** 会。若低于该状态策略通常回报，advantage可为负。

**Q06 [对象] Critic就是reward model吗？** 不是。前者预测未来累计反馈，后者可定义某个输出的即时评分。

**Q07 [TD] 一步TD residual何时是优势估计？** 当value准确时，条件于状态动作的期望对应优势；实际critic误差和转移噪声会影响估计。

**Q08 [梯度] Advantage为什么detach？** Actor用固定反馈更新动作概率；沿advantage/value反传会改变目标，并可能混入不希望的梯度路径。

**Q09 [终止] Done字段够用吗？** 不一定。真实终止、时间/采样截断和跨episode边界要分开，下一篇给两个mask。

**Q10 [结构] 共享encoder会不会让critic影响actor？** 会通过共享参数产生影响，即使actor优势detach；这属于联合优化的取舍。

**Q11 [计数] Agent动作是一token还是一turn？** 两种都可定义，但logprob、折扣、reward和mask必须匹配，不能混用时间单位。

**Q12 [方差] V一定是最佳baseline吗？** 不保证。score大小影响最低方差baseline；两动作例子说明期望价值与最小方差常数可不同。

## 11. 来源与CPU练习

[CPU课堂脚本](./assets/learning/agent-training-lab.py)和[结果](./assets/learning/agent-training-lab-results.json)枚举上述两动作期望与方差，并用autograd核梯度；不运行环境训练。[来源清单](./research/rl-actor-critic-sources.json)保留原始工作、固定代码和教学依据。

阅读辅助可用[SpinningUp策略梯度推导](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)，它用于因果性与baseline的讲解；公式和本例数字分别由原始理论与课堂计算支撑。接着读[回报 TD与GAE](#q=rl-returns-gae)。
