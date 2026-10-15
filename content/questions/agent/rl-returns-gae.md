---
id: rl-returns-gae
title: "Monte Carlo TD与GAE怎样分配回报"
category: agent
difficulty: 进阶
tags: ["P0", "RL", "GAE", "TD", "Bootstrap", "Credit Assignment"]
updated: "2026-10-15"
summary: "从MC和n-step推导GAE，手算优势与value target，区分真正终止、截断、padding、工具观察和两个mask。"
draft: false
---

## 1. 问题背景 最终成功怎样反馈给早期动作

一个agent先决定搜索关键词，随后读网页、再搜索、综合答案，十轮后才得到最终评分。若每个动作只看即时reward，早期全0就没有有用信号；若每一步都直接使用完整终局回报，估计又可能噪声很大。需要将“已经观察到的反馈”与“critic对未观察未来的估计”按明确规则组合。

Monte Carlo、TD、n-step和GAE主要回答回报/优势怎样估计。PPOclip主要回答已有优势下怎样限制策略更新，它们是不同环节。先理解[actor–critic](#q=rl-actor-critic)的V、Q、A，再看本篇的时间轴，之后读[PPO训练一步](#q=rl-ppo-training)。

## 2. 前置知识 Bootstrap和采样边界

**Monte Carlo return** 用实际后续reward；**bootstrapping** 在尚未观察完整未来时接上value估计；**TD residual** 是一步观察与value预测的差；**eligibility/trace** 将多个时间点的误差信号向早期传播。Bootstrap不是用teacher答案替代未来，也不是统计学里的有放回抽样。

**Terminal** 指所定义任务真正结束；**truncation** 指采样/时间窗口停止，但底层过程按定义仍可继续；**padding** 只是张量对齐，不是环境转移。时间限制是否定义为任务失败，也取决MDP：若“必须在30步内成功”是目标的一部分，预算结束可以是真终止；若只是采样器切块，不能自动把未来价值清零。

**Value target** 训练critic，**advantage** 训练actor。二者关系常为 return_target=raw_advantage+old_value，但actor优势随后可whiten，value target不能改成whitened_advantage+old_value，否则改变回报单位。

每个训练记录至少保存真实reward、old value、实际next state/value、episode/segment标识以及终止语义。只有一个done字段，很难同时描述“还能bootstrap”和“能不能把下一条记录的优势串进来”。

![Two boundaries in GAE](./assets/learning/rl-returns-gae/gae-masks.svg)

本文教学示意。上方沿一次决策的真实信息流阅读，下方列出训练所需的职责和边界。箭头表示数据传递，不表示环境反馈可以直接反向传播；具体公式、mask和梯度在后续章节逐项展开。

## 3. MC TD和n-step的统一表达

完整有限episode的MC target为：

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
$$

若在T处只是采样截断，需依任务定义补 $\gamma^{T-t}V_{\mathrm{old}}(s_T)$。真实terminal该项为0。MC靠实际远期反馈，可能高方差；没有随机性也不代表必然无方差，因为初态与策略动作仍可随机。

一步TD target：

$$
G_t^{(1)}=r_t+\gamma V_{\mathrm{old}}(s_{t+1}).
$$

n-step target：

$$
G_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^nV_{\mathrm{old}}(s_{t+n}).
$$

遇到真终止时截住并移除终止后的value。n较小更依赖critic；n较大使用更多实际反馈但可能有更大方差。“TD低方差、MC无偏”是常见取舍方向，不是对所有环境、value误差和采样组织的无条件保证。

例子：两步reward[0,1]，value[0.4,0.6,0]，$\gamma=0.9$。首步TDtarget=0.54、优势0.14；MCtarget=0.9、优势0.5。一个看近一步的估计与完整终局观察，给出的信号不同；critic若过高或过低，也可能改变优势符号。

## 4. GAE从TD残差到倒序递推

### 先从多个n-step target的混合看lambda

设当前片段还剩$N$个真实transition，每种$n$都有一个$n$步reward加末value的target $G_t^{(n)}$。有限片段的lambda-return可写成：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{N-1}
\lambda^{n-1}G_t^{(n)}+\lambda^{N-1}G_t^{(N)}.
$$

最后一项接收剩余权重，全部权重和为1。$G_t^\lambda-V_{old}(s_t)$可以展开成下文TD残差的加权和，这就是GAE与多步回报的连接。$\lambda$较小更依赖短步value，较大更使用远期实际reward；它不改变已经观察到的reward本身。

两步例子中，首步TD target为0.54，完整MC target为0.9。$\lambda=0.8$时混合target为$0.2\times0.54+0.8\times0.9=0.828$，减value0.4得到优势0.428。下面的倒序算法会给出完全相同的数字。先理解混合的对象，再看递推，不必从一行反向for循环猜算法含义。

### 用TD残差得到线性时间递推

对同一真实episode：

$$
\delta_t=r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t),\qquad
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda$控制保留多少更远TD残差，而不是新reward权重。倒序计算避免对每个t重复遍历整段：

$$
\widehat A_t=\delta_t+\gamma\lambda\widehat A_{t+1}.
$$

[GAE固定原论文v6](https://arxiv.org/abs/1506.02438v6)讨论偏差/方差折中及其策略优化应用。$\lambda=0$退化一步TD；$\lambda=1$在边界正确、固定value的有限片段上可望远镜消去中间value，得到带正确末端bootstrap的return减当前value。改变$\gamma$同时可能改变优化目标，不能只称其为方差参数。

### 一步一步手算

沿上述两步例子，末步terminal，因此$\delta_1=1-0.6=0.4$；首步$\delta_0=0+0.9\times0.6-0.4=0.14$。取$\lambda=0.8$：

$$
\widehat A_1=0.4,\qquad
\widehat A_0=0.14+0.9\times0.8\times0.4=0.428.
$$

critic target为[0.828,1.0]。改成$\lambda=1$，优势变[0.5,0.4]，target为[0.9,1]；改成$\gamma=\lambda=1$，则优势[0.6,0.4]。每一组数字有明确参数，不能把不同设置的算例拼在一起。

为什么critic target在lambda0.8时不同于实际MC0.9？因为它使用了短/长估计的混合，不是把真实最终reward改成0.828。实际reward不变，value训练目标的估计发生变化。

## 5. 两个mask 终止和切断trace不是一回事

通用教学形式分开两个开关：

$$
\delta_t=r_t+\gamma b_tV_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t),\qquad
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
$$

$b_t$控制能否使用下一状态value；$c_t$控制能否连接下一条样本的GAE。若下条记录来自reset后的新episode，$c_t=0$，即使当前因时间截断可bootstrap也不能接上新episode优势。只用一个nonterminal字段需要数据组织保证两个语义一致。

| 边界 | bootstrap b | trace c | 应使用的next state |
| --- | --- | --- | --- |
| 同episode连续transition | 1 | 1 | 实际后继状态 |
| 真正terminal | 0 | 0 | 不使用value |
| 采样窗口/时间截断，过程可继续 | 1 | 0 | 截断前真实末状态 |
| 下条记录来自另一个episode | 按当前边界定义 | 0 | 不能拿reset新初态替代末状态 |
| padding | 不参与transition估计 | 不建立新连接 | 没有环境状态 |

单步reward0、当前value2、末状态value3、gamma0.9：真终止优势是-2；可继续截断优势是0.7。给二者相同done处理会改变方向。向量环境自动reset时，返回observation可能是新episode初态，须获取真正final observation再算bootstrap。

### 错接episode会翻转优势

两个独立单步episode，reward[0.5,0]、value[0.2,0.9]，二者都真终止。正确优势为[0.3,-0.9]。若把第一条trace接到第二条，在gamma0.9、lambda0.8下，第一条变$0.3+0.72(-0.9)=-0.348$，本该强化的动作反而被抑制。

这不是PPOclip能修复的错误：优势已经错了，更新限制只限制错信号的幅度。应在数据层先验证episode和segment边界。

## 6. 教学代码和历史源码的差别

### 输入输出与梯度各归谁

`rewards`、`old_values`、`next_values`、`bootstrap`和`trace`均为`[B,T]`，这里的T是实际决策数。返回两个对象：`advantages`给actor，`returns`给critic。函数在no-grad中计算，因此即使value来自模型forward，target也不保留训练图。

| 量 | 后续用途 | 更新时是否重新算 |
| --- | --- | --- |
| old value | 生成优势与target | 这轮固定 |
| raw advantage | actor反馈、构造target | 通常这轮固定 |
| returns | critic的监督标签 | 这轮固定 |
| current value | 与returns比较求MSE | 每次learner forward重算 |

区别在于“用旧估计构造标签”和“拟合标签的当前预测”。把二者都替换成同一current tensor，可能让目标随预测一起移动；若还允许target反传，就改变了semi-gradient TD的更新规则。

```python title="教学实现 分开bootstrap与trace"
adv = torch.zeros_like(rewards)
carry = torch.zeros_like(rewards[:, 0])
for t in reversed(range(rewards.shape[1])):
    delta = (rewards[:, t]
             + gamma * bootstrap[:, t] * next_values[:, t]
             - old_values[:, t])
    carry = delta + gamma * lam * trace[:, t] * carry
    adv[:, t] = carry
returns = adv + old_values
```

输入[B,T]包含每条真实decision的reward/value及对应next value；如果含padding，还要显式跳过而不是将其当r=0的真实动作。函数调用应在no_grad或对输入detach后执行，避免target梯度泄露。运行版本和边界例子在[CPU脚本](./assets/learning/agent-training-lab.py)。

历史Baselines[GAE节选](https://github.com/openai/baselines/blob/da997060461e3cbf54ca4dc7a67081a731fb6b3b/baselines/pposgd/pposgd_simple.py#L68-L78)：

```python title="Baselines真实源码 单个nonterminal的历史递推"
for t in reversed(range(T)):
    nonterminal = 1-new[t+1]
    delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
    gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
seg["tdlamret"] = seg["adv"] + seg["vpred"]
```

new表示下一位置是否新episode，vpred附加nextvpred。源码显示怎样倒序累积，但不能仅改变量名就视为支持所有现代terminated/truncated语义；其collector和nextvpred处理也属于算法的一部分。课堂两mask是职责展开，不声称是这份旧代码的原接口。

## 7. LLM工具观察和padding的mask

LLM的action mask通常在模型生成token位置为1，prompt、工具观察和padding为0。0表示该位置不承担actor预测目标，不表示后续token看不到这些内容。GAE在模型decision上的链也需要与这个mask一致，不必让每个工具返回token消耗一个折扣步。

固定verl[GAE实现](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L244-L263)使用response_mask跳过观察：

```python title="verl真实节选 观察位置不推进nextvalue和carry"
nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
```

这两行在计算候选delta/lastgaelam_后选择是否更新carry，不能用“mask=0就把carry清零”替代。此版本从nextvalues=0起步，支持的终止/response组织仍需由数据契约保证；不要仅凭response_mask字段宣称任意截断都已正确bootstrap。

连续语言轨迹与每轮重新构造prompt的训练布局不同。若第二轮使用摘要而不是原始第一轮文本，训练第二轮logprob必须使用实际摘要上下文；value估计也要对应实际条件。不能在最终拼接文本上随意重新生成所有位置，再声称是同一条行为轨迹。

## 8. Whitening reward shaping和value loss

Reward normalization改变反馈尺度，advantage whitening改变actor系数尺度，value normalization改变critic预测空间。顺序会影响单位与训练意义；将actor白化优势反加old value作为return target，会把两个不同尺度的量相加。

verl同一函数先构造returns=advantages+values，再masked_whiten advantages。它说明返回两个对象供不同loss使用；并不说明所有算法都使用同样的mean/std或跨卡统计。

Potential-based shaping常写$r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t)$。把有限轨迹的折扣回报展开，中间势函数项会抵消：$\sum_{t=0}^{T-1}\gamma^t r'_t=\sum_{t=0}^{T-1}\gamma^t r_t-\Phi(s_0)+\gamma^T\Phi(s_T)$。若terminal势函数设为零，且初始状态分布固定，新增项不依赖所选动作，才能得到相应策略保持结论；无限时域则需要有界势函数等条件使尾项消失。任意“每次工具调用减分”“每个中间思考加分”不是自动potential shaping。工具成本可以是目标的一部分，改变目标就应明确报任务成功与成本的取舍。[原始证明：Ng、Harada与Russell，1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。

value MSE小也不自动说明actor优势可靠：小误差可能集中在少数关键分支，或value只学到任务难度平均。诊断可看explained variance、target范围、优势符号、终止/截断比例与价值在关键决策点的误差；不要凭一个均方误差排除credit问题。

## 9. 在Agent任务中怎样选择估计器

若反馈只有终局，critic能提供对早期决策的相对估计，但价值学习也可能困难；若环境提供可信中间reward，n-step/GAE可更快利用它，评分错误也可能更快被优化。组优势省critic，却通常将同一轨迹级系数广播到许多token，需要更细credit时可看GiGPO等工作。

先固定state/action/reward和预算，比较MC/GAE/组baseline，而不是先讨论某个缩写。对每条样本记录真实终止语义和采用的估计规则；同样长度的两条序列也可能包含不同turn数、工具等待和决策粒度。

## 10. 面试问答

**Q01 [估计] MC和TD最核心区别？** MC用实际后续reward，TD接value bootstrap；后者更依赖critic，前者通常有更长等待和较高方差。

**Q02 [推导] GAE是什么的加权和？** TD residual的几何加权和，权重gamma×lambda，不是直接把reward随意平滑。

**Q03 [参数] Lambda=1是否总是原始MC？** 还看gamma和末端bootstrap；真终止时有限段可化为折扣MC减value，截断时保留末value。

**Q04 [边界] 两个mask分别控制什么？** b控制bootstrap下一状态价值，c控制trace是否接下一条样本；截断后reset常需要b1/c0。

**Q05 [例子] 同一个r0/V2/nextV3为何可得-2或0.7？** Gamma0.9下前者真终止移除bootstrap，后者可继续截断保留它。

**Q06 [数据] 为什么不能用自动reset后的obs？** 它可能是另一个episode初态，不是当前动作真实后继，造成错误value target。

**Q07 [mask] 工具观察位置mask0为何不一定清carry？** 它是条件信息但不作为policy decision，GAE可跳过它并继续连接前后有效动作。

**Q08 [目标] Whitened advantage加old value能训critic吗？** 通常单位不一致，应先用raw advantage构造return，再仅白化actor系数。

**Q09 [折扣] Token gamma和turn gamma一样吗？** 不一样，时间单位不同；同一turn越长会改变token折扣影响。

**Q10 [诊断] Clip为何不能修错误的GAE？** 它约束使用既有优势的更新，不知道优势来自错误episode连接或bootstrap。

**Q11 [奖励] 加工具成本会保持原最优策略吗？** 不保证；如果直接改reward就改变目标，需说明效率/成功率取舍或证明特定shaping条件。

**Q12 [讲课] 怎样快速证明自己理解GAE？** 手算两步、列terminal/truncation两mask，再说明工具观察与padding处理，最后核return和actor优势的梯度边界。

## 11. 来源与CPU例子

[统一课堂脚本](./assets/learning/rl-algorithms-unified-lab.py)的`gae`采用显式真实transition接口；其中两步任务取$\gamma=\lambda=1$，故每条路径的target为实际终局reward。Token布局中的工具观察跳过逻辑留在上面的verl伴读，不能把两种输入布局混用。[统一结果](./assets/learning/rl-algorithms-unified-lab-results.json)还核对了terminal优势-2与可继续截断优势0.7。

[CPU结果](./assets/learning/agent-training-lab-results.json)记录两步GAE、lambda变化、terminal/truncation和错误跨episode连接；验证固定小张量，不是训练收敛或框架完整复现。[来源清单](./research/rl-returns-gae-sources.json)登记原文、源码和[作者实现细节博客](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)。继续读[PPO训练一步](#q=rl-ppo-training)。
