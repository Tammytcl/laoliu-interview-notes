---
id: rl-ppo-training
title: "PPO训练一步的数据梯度与实现"
category: agent
difficulty: 进阶
tags: ["P0", "PPO", "Actor Critic", "KL", "Rollout", "Implementation"]
updated: "2026-10-12"
summary: "从采样到optimizer逐项追踪PPO，解释冻结量、四种裁剪、actor/critic/reference职责、KL、mask和多epoch复用。"
draft: false
---

## 1. 问题背景 样本昂贵时怎样复用而不过度更新

RL经历由一个策略采样，训练后参数变化，同一动作在当前策略下的概率就不同。若把高分轨迹反复当作正向标签训练，可能过度强化有限样本、放大reward误差并偏离采样分布。PPO让同一批样本可以经过多轮minibatch优化，用概率比与悲观裁剪抑制部分过度有利的变化。

原始PPO验证控制任务，后来LLM/agent训练加入语言模型、reward/verifier、reference KL和工具轨迹。本篇沿训练步讲接线；原2017论文的图表与实验见[PPO精读](#paper=paper-ppo)。不会把今天RLHF常见组件误写成原论文必需部分。

## 2. 前置知识 先修和一批数据的对象

先掌握[Actor Critic](#q=rl-actor-critic)和[TD GAE](#q=rl-returns-gae)：策略梯度为什么乘优势，哪些量是固定估计，终止/截断怎样处理。PPO是更新目标，GAE是优势估计，二者可以组合但不相互定义。

本轮数据 $D_k$ 含真实输入/状态、动作token、old logprob、reward、value、mask、版本、终止和产物。常见LLM张量logprob/advantage/response mask为[B,L]；prompt和工具观察参与条件，但不自动成为actor监督位置。Variable length、packed episode和跨turn重写需要明确布局。

**Actor current** 是参与当前梯度的策略；**behavior** 是生成数据的真实采样规则；**old/proximal** 是本轮更新比较的基线；**critic** 估计value；**reference** 锚定预训练/初始行为；**teacher** 为蒸馏提供分布。同步简单场景behavior与old可重合，异步或采样处理后未必如此。

## 3. 一次iteration的完整顺序

![本文示意 PPO采样冻结更新与发布](./assets/learning/rl-ppo-training/ppo-cycle.svg)

**图解。** 采样时确定行为与版本，反馈和value构造固定advantages/returns；训练对同一真实prefix重算current logprob，计算ratio与loss，做多epoch更新；完成一轮后再发布新策略。图中缓存量不会每个minibatch跟着当前参数悄悄变。Reference和teacher是可选不同角色，工具环境无需保留可微计算图。

1. 固定本轮采样策略及任务/环境初态，生成实际action与观察。
2. 得到任务reward或reward model反馈，若方案使用KL reward则按明确版本计算。
3. 估计GAE或另一种优势，构造critic target，保存为固定量。
4. 对若干epoch/minibatch重新forward current policy和value。
5. 用old logprob、固定advantage、mask/reduction计算loss并更新。
6. 统计KL、clip、value、reward、长度、任务成功及成本，结束本轮并同步下一策略。

这张表不是所有系统都同步执行的保证。异步流水线会让轨迹、评分、learner和权重发布重叠，仍需要标出各数据量的版本。反向传播提前完成、optimizer更新和推理权重发布是三个不同事件。

## 4. 比值从哪来 为什么初始为1仍有梯度

对同一真实状态动作：

$$
r_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)
-\operatorname{sg}[\log\pi_{\mathrm{old}}(a_t\mid s_t)]\right).
$$

sg表示stop-gradient。新旧策略相同时数值为1，但分子仍可微，因此$\nabla_\theta r_t=r_t\nabla_\theta\log\pi_\theta$不为零。若把分子分母都用同一可微tensor，梯度互相抵消；若把两者都detach，也没有策略梯度。

概率比只对当前观察到的动作条件概率作比较，不能把旧策略访问的状态分布完整变成当前策略状态分布。PPO适合近期数据的局部更新，不是任意历史replay的无损校正机制。

**采样变换必须另核。** Temperature/top-p/grammar等可将raw actor分布变成behavior $\mu$。原PPO的current/old actor比值、rollout-to-train采样校正和reference KL是不同量。若目标是raw policy，不能随意把processed behavior logprob塞进old栏而仍宣称起点ratio必为1；若目标就是processed policy，则当前分布也要按同一契约重算。现代框架可能使用近似或额外correction，应如实记录。

## 5. 四种clip分支和真实梯度

PPO-Clip最大化：

$$
L^{\mathrm{clip}}
=\mathbb E\left[\min(rA,\operatorname{clip}(r,1-\epsilon,1+\epsilon)A)\right].
$$

取epsilon0.2、A=±2，结果如下；训练最小化负号：

| A | r | rA | clipped rA | 选取U | 对log r的loss导数 |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.7 | 1.4 | 1.6 | 1.4 | -1.4 |
| 2 | 1.3 | 2.6 | 2.4 | 2.4 | 0 |
| -2 | 0.7 | -1.4 | -1.6 | -1.6 | 0 |
| -2 | 1.3 | -2.6 | -2.4 | -2.6 | 2.6 |

正优势只在过度上升方向进入平台，负优势只在过度下降方向进入平台；另一侧仍纠正坏方向。四样本mean时表中导数再除4。参数共享、entropy/value/KL等其他项仍可能改变平台样本的概率，因此epsilon不构成真实概率或KL的硬界。

```python title="教学实现 最小化clip loss"
ratio = (current_logprob - old_logprob.detach()).exp()
fixed_adv = advantages.detach()
u = ratio * fixed_adv
clipped_u = ratio.clamp(1 - eps, 1 + eps) * fixed_adv
per_token_loss = -torch.minimum(u, clipped_u)
actor_loss = (per_token_loss * action_mask).sum() / action_mask.sum()
```

这里分母需保证大于0，分布式时需明确全局有效token计数。不要改成对ratio先clamp再始终使用clipped_u，否则错误方向的纠正梯度也被截掉；也不要将越界token全部mask，因为这不等于min的符号分支。

### 与原始实现对照

历史Baselines固定commit的[策略目标](https://github.com/openai/baselines/blob/da997060461e3cbf54ca4dc7a67081a731fb6b3b/baselines/pposgd/pposgd_simple.py#L110-L113)：

```python title="Baselines真实源码 四行surrogate"
ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
surr1 = ratio * atarg
surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
```

pi/oldpi是当前与旧分布，ac是已采样动作，atarg是优势。旧模型变量和更新顺序由周围代码保证，不能仅见没有detach就说有bug：TensorFlow的变量组与optimizer选取也决定哪些参数更新。这是控制任务的旧实现，不带LLM的token mask、teacher和reference角色。

## 6. Critic loss与联合目标

raw GAE加old value构造 $V^{\mathrm{targ}}$，固定target后拟合当前value：

$$
L_V=\frac12\mathbb E[(V_\phi(s)-\operatorname{sg}[V^{\mathrm{targ}}])^2],\qquad
L_{\mathrm{total}}=L_{\mathrm{actor}}+c_VL_V-c_HH+c_KL_{\mathrm{KL}}.
$$

此式是常见组合的教学写法，最后一项可选，不意味着原PPO一定包含LLM reference。Entropy提高策略分布的熵；reference KL约束与某锚点的差；value loss改善回报预测。三个对象不可互换，数值尺度与系数也不同。

Value clipping是另一项实现选择：可用old value限制预测变化，再用最大误差等形式构造loss；它不由policy clip自动导出。Critic target不沿当前value反传，actor advantage也不沿critic反传；共享encoder仍接收两个loss的梯度。

**单位检查。** Reward若整体乘10，value target与raw advantage也放大；actorwhitening可能去掉部分尺度，criticMSE却约放大100倍。只调整actor学习率，不一定恢复原训练。检查reward normalization、value范围、entropy/KL系数和global reduction。

## 7. Reference KL究竟放在哪

给定状态的reverse KL：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=\sum_a\pi_\theta(a\mid s)
\left[\log\pi_\theta(a\mid s)-\log\pi_{\mathrm{ref}}(a\mid s)\right].
$$

整项非负，但某个采样动作的logprob差可以为负。用样本估计、旧策略数据、不同KL估计器和stop-gradient，会改变有限样本loss与梯度路径。只看到代码里减两个logprob，不足以判断它是完整current KL的无偏梯度。

**放进reward。** 本轮采样后计算固定KL penalty，再进入GAE/return，是reward shaping与credit的一部分。**直接放loss。** Learner计算current/reference分数并构造单独可微项，作用路径不同。二者可以组合，但应分别记录版本、方向、mask、分母和系数，不能重复扣分而不知情。

PPO的old/proximal用于控制“这一轮更新离比较策略多远”，reference用于“长期偏离哪份锚点”。若每个minibatch把reference换成current，它就不再是固定reference；若始终把old换成初始SFT，它就不再是通常近期PPO旧策略。

## 8. 多epoch缓存和不可变的数据契约

同一iteration内old logprob、优势和critic target通常固定，current forward每个minibatch重算。多个epoch增加样本利用率，但会使ratio偏离1、clip增加，旧优势也可能不再适合远离采样策略的更新。KL early stop是实用控制，不能当成TRPO理论硬保证。

Cache old logprob可免去每次完整old-model forward，不表示整个old actor从系统消失：rollout时仍需要生成它对应的分布；新iteration或策略版本切换后要重新记录。也不能拿最新checkpoint重算并覆盖历史behavior概率。

```python title="教学伪代码 一轮PPO的更新边界"
batch = collect_with_version(policy_version)
fixed = build_rewards_advantages_returns(batch)
for epoch in range(update_epochs):
    for microbatch in split(batch, fixed):
        current_lp, current_v = learner_forward(microbatch.actual_inputs)
        loss = build_losses(current_lp, current_v, microbatch, fixed)
        accumulate_or_step(loss, global_normalization)
publish_updated_policy()
```

accumulate_or_step由预定global batch控制，不是每个turn都有loss就随意step。若完整轨迹的终局reward尚未确定，早期token优势可能仍会变化；若GRPO组统计未齐，也不能提前把最终目标当已知。异步系统要分别安排就绪、backward、step和publish。

## 9. 长度mask与同预算比较

两条轨迹长度2和4，若每tokenloss分别1和3，sequence平均是2，global token平均是7/3。都是合法目标，但给予轨迹的权重不同。Clip和advantage估计不自动决定reduction；DAPO、Dr.GRPO等正是围绕部分尺度/长度问题改动。

LLM多轮中工具观察mask0仍参与attention；将模型真实响应压缩后重新拼接会改变条件概率。Reward验证错误、环境恢复错误、logprob对象错误，都可能表现为训练loss正常但任务成绩下降。诊断应先验证样本契约，再调clip超参。

常看reward/任务成功、KL、entropy、clip fraction、ratio尾部、value误差/解释方差、长度、有效样本数和GPU-hour。Clip fraction高可能来自大更新或不匹配概率，不自动意味着全部样本应丢弃；reward升也可能来自reward hacking。

## 10. 面试问答

**Q01 [顺序] 一轮PPO里哪些量固定？** 本轮old logprob、reward/advantages/returns固定，current logprob/value更新；下一轮重新采样并登记版本。

**Q02 [梯度] Ratio起点1为何有梯度？** 分母固定、分子可微。把二者同图相除或全部detach会得到错误梯度。

**Q03 [clip] 越界就无梯度吗？** 不。正优势只上侧平台，负优势只下侧平台，另一侧仍纠错。

**Q04 [理论] Epsilon限制真实KL吗？** 不保证，单项平台不约束所有共享参数的变化；需监控KL和其他loss。

**Q05 [角色] Old reference critic有什么差别？** Old比较本轮更新，reference锚定长期行为，critic估未来回报。

**Q06 [单位] 优势白化后能直接训value吗？** Return要先在原回报单位构造，actor白化系数不应直接与old value相加。

**Q07 [概率] Top-p后logprob可以直接当old吗？** 先明确训练策略与behavior契约；PPO更新比与采样校正不同，不能混名。

**Q08 [复用] 多epoch为什么不等于无限replay？** 策略偏离、状态分布和优势陈旧会累积；clip不提供任意历史数据的完整校正。

**Q09 [数据] 为什么保存actual inputs而非最终摘要？** 摘要可能改变过去action的条件概率；学习应重现真实采样条件。

**Q10 [目标] KL reward与KL loss等价吗？** 不自动等价，版本、return传播、采样与梯度路径不同。

**Q11 [工程] 同一turn就绪能立即step吗？** 还要目标是否确定、归一化、组依赖、global batch与权重发布规则。

**Q12 [排障] Loss下降而任务分下降先查什么？** Reward/环境、mask/边界、actual logprob、版本与reduction，然后再调优化器。

## 11. 参考讲解与CPU验证

原始[PPO v2](https://arxiv.org/abs/1707.06347v2)、[TRPO v5](https://arxiv.org/abs/1502.05477v5)与[SpinningUp PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)提供方法和教学入口；[实现细节博客](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)辅助检查工程边界。[来源清单](./research/rl-ppo-training-sources.json)固定代码。

[CPU课堂结果](./assets/learning/agent-training-lab-results.json)核四分支梯度、ratio1的非零导数、mask/reduction与detach边界，不做模型训练。接下来用[算法地图](#q=rl-policy-optimization-map)把RLOO、GRPO和后续变体放回这条链。
