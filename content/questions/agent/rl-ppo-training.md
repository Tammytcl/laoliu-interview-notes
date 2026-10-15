---
id: rl-ppo-training
title: "PPO训练一步的数据梯度与实现"
category: agent
difficulty: 进阶
tags: ["P0", "PPO", "Actor Critic", "KL", "Rollout", "Implementation"]
updated: "2026-10-15"
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

### 先解释为什么需要old分母

固定一个已经访问的状态$s$，希望评估新策略对固定优势$A_{old}(s,a)$的加权效果。如果动作来自old分布，可用概率比改写：

$$
\sum_a\pi_\theta(a\mid s)A_{old}(s,a)
=\mathbb E_{a\sim\pi_{old}}
\left[\frac{\pi_\theta(a\mid s)}{\pi_{old}(a\mid s)}
A_{old}(s,a)\right].
$$

这是给定状态、支持覆盖下的action换测度。真实策略目标还依赖策略访问哪些状态；PPO局部surrogate沿old状态分布做近似，再用clip抑制部分过大更新。先从这个恒等式理解ratio，再区分完整目标和局部近似，比把ratio只当“稳定训练的技巧”更清楚。

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
surr1 = ratio * atarg # surrogate from conservative policy iteration
surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
```

pi/oldpi是当前与旧分布，ac是已采样动作，atarg是优势。`U.clip`和`U.mean`是这份实现的封装调用；上面按固定源码保留原句，与前面的PyTorch教学重述分开。旧模型变量和更新顺序由周围代码保证，不能仅见没有detach就说有bug：TensorFlow的变量组与optimizer选取也决定哪些参数更新。这是控制任务的旧实现，不带LLM的token mask、teacher和reference角色。

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

### 从模型logits提取实际动作分数

对LLM，policy loss前还要完成一次因果移位。假设`action_mask`标记的是`input_ids`中实际动作token的位置，完整序列logits为`[B,S,V]`：

```python title="教学实现 当前logprob来自同一真实prefix"
prediction = logits[:, :-1, :].log_softmax(dim=-1)  # 位置j预测j+1
targets = input_ids[:, 1:]                        # 实际下一个token
logp = prediction.gather(-1, targets[..., None]).squeeze(-1)
mask = action_mask[:, 1:]                         # 和targets同一布局
```

第一个响应token由其前一个位置预测，不能先截response logits再错误地整体右移。Prompt与工具观察通常不承担actor loss，但应保留在模型输入中。真实高效实现常用融合cross entropy或词表并行logprob，避免物化完整`[B,S,V]`的log-softmax；上面用完整张量只是为了把位置与梯度讲清。

### 一个真正执行forward backward和step的最小例子

沿[Actor Critic的两bit任务](#q=rl-actor-critic)：首步选bit，次状态按首bit分成两种，只有[1,1]成功。初始actor均匀；枚举四条路径就能得到其精确采样期望。故这里没有随机collector误差，也不加载语言模型。

为观察critic如何学习，故意初始化value为[0.4,0.1,0.6]，区别于真实value[0.25,0,0.5]。更新使用SGD学习率0.1、value系数0.5，actor与value均按这批有效决策平均。取$\gamma=\lambda=1$，GAE target就是各路径的终局reward，优势与target如下：

| 两步动作 | 两步old value | 两步优势 | critic target |
| --- | --- | --- | --- |
| [0,0] | [0.4,0.1] | [-0.4,-0.1] | [0,0] |
| [0,1] | [0.4,0.1] | [-0.4,-0.1] | [0,0] |
| [1,0] | [0.4,0.6] | [-0.4,-0.6] | [0,0] |
| [1,1] | [0.4,0.6] | [0.6,0.4] | [1,1] |

表中成功轨迹首步优势0.6并非真实$A^\pi$的精确值，而是这条样本的MC return减不完美baseline；四条路径平均后仍给正确的策略梯度方向。Value target是监督，不等于每个单样本上的真实状态value。

```python title="统一课堂脚本的训练核心 两个epoch共用冻结标签"
for epoch in range(2):
    logp = actor[states].log_softmax(-1).gather(
        -1, actions[..., None]
    ).squeeze(-1)
    actor_loss = policy_loss(
        logp, old_logp, adv, torch.ones_like(logp), "ppo", "token"
    )
    value_loss = .5 * (critic[states] - returns).square().mean()
    optimizer.zero_grad()
    (actor_loss + .5 * value_loss).backward()
    optimizer.step()
```

`actor`是`[3,2]`可训练logits表，`critic`是`[3]`value表；查表是这个小MDP的forward。`states/actions`为`[4,2]`，表示四条路径、每条两步。`gather`后得到每条路径实际动作的logprob，再进入和LLM相同结构的PPO目标。

代码同时更新actor和critic，但`adv/returns/old_logp`在进入循环前由no-grad计算并固定。首轮actor在首状态与“首bit为1”的次状态上，两个logit梯度均为[0.0625,-0.0625]，梯度下降提高选择1的概率；critic通过MSE纠正不完美value。第二轮重新forward current logits，ratio已经能偏离1，却继续使用同一old分母和标签。

[完整可运行脚本](./assets/learning/rl-algorithms-unified-lab.py)断言了梯度方向、缓存不变和两次step后的精确期望reward提高；[结果JSON](./assets/learning/rl-algorithms-unified-lab-results.json)包含每轮梯度。它说明数据到更新能闭环，不代表真实LLM训练的收敛或效果。

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
