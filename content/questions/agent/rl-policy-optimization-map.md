---
id: rl-policy-optimization-map
title: "RL算法总览 从Actor Critic到PPO和组策略优化"
category: agent
difficulty: 进阶
tags: ["P0", "RL", "PPO", "GRPO", "DAPO", "GSPO", "GDPO", "SAPO", "DPO"]
updated: "2026-10-15"
summary: "以一条训练数据流整合基础专题和七篇论文，逐层比较优势、概率比、更新目标与归一化，并用统一接口和CPU梯度例子讲解代码。"
draft: false
---

## 1. 问题背景 为什么越读论文越容易混淆

后训练算法名称很多，但它们经常修改不同的环节。PPO讨论如何限制策略更新；GAE讨论如何估计动作的好坏；GRPO用同题多次采样替代价值基线；GSPO改变概率比的粒度；GDPO处理多个奖励的归一化。把这些名字排成“后一种替代前一种”，会丢掉它们真正的关系。

理解一套配方，应能沿一条样本回答：**谁产生动作，谁给反馈，怎样得到优势，当前策略与谁比较，哪些位置反向传播，最后按什么分母更新。** 本篇是统一入口。理论证明在基础专题中展开，原图、实验与逐行真实源码保留在论文精读中；下面用相同符号和接口把它们连接起来。

### 阅读顺序与每一步的学习目标

| 顺序 | 主文 | 读完应能完成 |
| --- | --- | --- |
| 1 | [Actor Critic与策略梯度](#q=rl-actor-critic) | 推导reward乘logprob梯度，解释baseline和V/Q/A |
| 2 | [回报 TD与GAE](#q=rl-returns-gae) | 手算两步优势，正确处理terminal和truncation |
| 3 | [PPO完整训练一步](#q=rl-ppo-training) | 从采样张量追到actor/critic的backward与step |
| 4 | [GRPO与RLVR](#q=grpo-rlvr) | 按prompt分组，算优势并广播到token |
| 5 | 本篇第5至9节 | 指出各变体替换哪一层，并预测梯度差异 |
| 6 | [七篇论文与真实源码对照](#report=survey-policy-optimization) | 用原图、配置、实验和代码检验机制解释 |
| 7 | [Agentic RL](#q=agentic-rl-training) | 把状态、动作、反馈和版本落实到工具交互 |

[DPO与SFT的入口](#q=sft-dpo-rl)是一条反馈方式不同的支线，可在第3步后阅读；OPD继续看[Agentic OPD](#q=agentic-opd-training)。不需要先读完所有缩写，再回头补策略梯度。

## 2. 前置知识 统一符号与模型职责

### 先定义我们优化的对象

设prompt为$x$，第$i$条回答为$y_i=(y_{i,1},\ldots,y_{i,L_i})$，生成第$t$个动作token时的真实前缀为$h_{i,t}$。自回归分解给出：

$$
\log\pi_\theta(y_i\mid x)=\sum_{t=1}^{L_i}
\log\pi_\theta(y_{i,t}\mid h_{i,t}).
$$

每个token的logprob是负数或0，完整回答的logprob是求和。只看总和大小会受长度影响；把它除长度得到另一个量，可能适合某种surrogate，但不能继续当作原回答概率。

| 符号或代码变量 | 含义 | 常见形状 |
| --- | --- | --- |
| $R_i$ / `rewards` | 完整回答或轨迹的反馈 | `[B]` |
| $V_\phi(s_t)$ / `values` | 从状态出发的预期未来回报 | `[B,T]` |
| $\widehat A_{i,t}$ / `advantages` | 该动作相对基线的估计好坏 | `[B,L]` |
| $\ell_{i,t}$ / `logp` | 当前模型对已采样token的logprob | `[B,L]`，可微 |
| $\ell^{old}_{i,t}$ / `old_logp` | 冻结比较策略的同token logprob | `[B,L]`，固定 |
| $m_{i,t}$ / `mask` | 该位置是否承担actor loss | `[B,L]` |
| $G$ | 同一prompt的回答数 | 标量，与生成长度无关 |

$B$是展平后的回答数。例如两个prompt、每题3条回答，$B=6$。组统计还要保留prompt UID；不能展平后直接对所有6个reward求一个mean来冒充GRPO。

### 五种职责为什么要分开

Actor决定动作；critic估计未来回报；reward model或verifier定义反馈；reference锚定某份行为分布；old/behavior记录比较策略与数据来源。Teacher为蒸馏提供动作分布，是另外一种反馈角色。它们可共享某些模型或缓存，但数学用途不同。

同步的原始简化场景中old与behavior相同。使用temperature、top-p、grammar或异步生成时，实际采样分布可能与训练侧raw softmax不同；要先说清概率契约，再解释ratio。Reference也不是“更旧的old”：old通常随轮次刷新，reference可长期固定。

## 3. 从理论走到代码 一轮在线训练只有一条主链

![本文示意 策略优化六层与不同算法的修改位置](./assets/learning/rl-policy-optimization-map/algorithm-map.svg)

**读图方法。** 先沿箭头读采样、评分、优势、概率比、目标与归约，再看算法标签落在哪一层。图中GDPO不替换采样器，GSPO不负责训练critic；DAPO跨越多层，所以无法用一个loss开关表达全部配方。

```python title="教学伪代码 所有在线变体共享的外层"
batch = collect(policy_version)             # 实际输入、动作、版本、边界
feedback = score(batch)                     # verifier / RM / environment
advantages, value_targets = estimate(batch, feedback)
for epoch in range(update_epochs):
    logp, values = learner_forward(batch.actual_inputs)
    actor_loss = objective(
        logp, batch.old_logp, advantages, batch.action_mask
    )
    loss = actor_loss + value_coef * value_loss(values, value_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
publish_new_policy()
```

这是逻辑骨架。GRPO没有学习value时应删除相应forward和loss；真实训练还要按global batch做minibatch、梯度累积与分布式归一化。这里的`estimate`输出通常在一轮内固定，`learner_forward`每次更新都重新计算。**改变估计器与改变更新目标是两件事。**

### 哪些代码可以真的运行

新的[统一CPU课堂脚本](./assets/learning/rl-algorithms-unified-lab.py)提供`gae`、`group_advantage`、`policy_loss`和`dpo_loss`四个明确接口，并执行一个两步MDP中的两次PPO更新。再用同一批六条回答比较GRPO、GSPO、DAPO局部改动、SAPO和CISPO的梯度。

```bash title="运行公式到梯度的课堂演示"
python assets/learning/rl-algorithms-unified-lab.py
```

[完整运行结果](./assets/learning/rl-algorithms-unified-lab-results.json)包含输入、优势、value target、两轮梯度和共享样例。它使用CPU float64与人工小状态，不加载LLM；不同目标的loss可带不同常数偏移，应比较梯度与机制，不能按脚本loss数值给算法排名。

## 4. 第一层 优势从哪里来

### REINFORCE 为什么乘回报

目标是让高回报轨迹更常发生。对轨迹概率使用$\nabla p=p\nabla\log p$，固定环境转移没有actor梯度，得到回报乘动作score function。用reward-to-go并取有限任务$\gamma=1$：

$$
\nabla J(\theta)=
\mathbb E\left[\sum_t G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
$$

动作是采样得到的，不能直接对离散采样或工具执行求导；训练时重新计算这个已采样动作的logprob。减去不依赖当前动作的状态baseline不会改变理想期望，因为$\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)]=0$。这就是actor–critic中V能辅助策略梯度的出发点。[推导伴读](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

### PPO常配的GAE

Critic给出状态value，用TD残差和倒序累积形成优势：

$$
\delta_t=r_t+\gamma b_tV_{old}(s_{t+1})-V_{old}(s_t),\qquad
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
$$

$b_t$控制下一状态bootstrap；$c_t$控制是否接下一条记录的trace。真终止二者为0，截断但任务可继续时常用$b=1,c=0$。$\lambda=0$只用一步TD，$\lambda=1$在正确边界下得到带末端bootstrap的return减value。

例：reward为[0,1]、value为[0.4,0.6,0]、$\gamma=0.9,\lambda=0.8$，TD残差为[0.14,0.4]，优势为[0.428,0.4]，critic target为[0.828,1]。Raw advantage加old value构造target之后，actor优势才可以单独白化。详细推导与两个mask的代码见[GAE主文](#q=rl-returns-gae)。

### RLOO 排除自身的组基线

同prompt采$G\ge2$条独立回答，RLOO使用其他回答reward的均值：

$$
A_i^{LOO}=R_i-\frac{\sum_{j\ne i}R_j}{G-1}
=\frac{G}{G-1}(R_i-\bar R).
$$

[Back to Basics](https://arxiv.org/abs/2402.14740v2)研究这种REINFORCE式RLHF。给定prompt和冻结策略，其他独立样本不依赖当前动作，符合baseline的条件。若reward为[0,1,3]，组centered为[-4/3,-1/3,5/3]，LOO为[-2,-0.5,2.5]，相差1.5倍。

### GRPO 用组统计代替学习V

Outcome GRPO再将centered reward除同组std：

$$
A_i^{GRPO}=\frac{R_i-\bar R}{\sigma_R+\epsilon}.
$$

这省去学习V的模型，却需要同题多次生成。它主要比较“这条回答比同题其他回答如何”，不像状态value那样直接估计每个中间状态未来能得多少分。Outcome形式将一个回答系数广播到所有有效动作token；原DeepSeekMath也讨论process supervision，不能把所有GRPO都限定为终局奖励。

```python title="教学实现 先按prompt分组 再广播"
# rewards: [Q,G]，这里显式使用样本std，G至少2
centered = rewards - rewards.mean(dim=1, keepdim=True)
group_std = rewards.std(dim=1, keepdim=True, correction=1)
answer_adv = centered / (group_std + 1e-6)
advantages = answer_adv.reshape(-1, 1).expand_as(action_mask)
```

Reward全同则centered为0，通常得到零组优势；epsilon不会创造区分信号。Torch样本std与population std数值不同：[0,0,1,1]分别得到约±0.866和±1。真正训练使用哪一种，应以固定实现为准。[DeepSeekMath原文](https://arxiv.org/abs/2402.03300v3)

### Dr GRPO与REINFORCE++怎样理解

[Dr.GRPO](https://arxiv.org/abs/2503.20783v2)分析组std与长度归一化带来的权重偏置，配方移除对应归一化。[REINFORCE++](https://arxiv.org/abs/2501.03262v9)需区分基础版和w/ Baseline版：前者可单题单次采样、做全局归一化；后者先减prompt组均值，再做global normalization。

去掉local std后，centered reward仍与字面LOO差$G/(G-1)$。还需核对全局统计、固定分母、KL和采样复用；“无critic”不是完整算法定义。

## 5. 第二层 PPO和GRPO怎样用概率比更新

先固定优势，再比较新旧策略对同一动作的概率：

$$
r_{i,t}=\exp(\ell_{i,t}-\operatorname{sg}[\ell^{old}_{i,t}]),\qquad
U_{i,t}=\min(r_{i,t}A_{i,t},
\operatorname{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_{i,t}).
$$

训练最小化$-U$。Ratio初始为1仍可更新，因为分母固定、分子可微。PPO常用GAE优势，GRPO换成组优势，随后仍可进入同一种token clip目标；**GRPO不是把PPO全部训练流程删除后重写。**

```python title="教学实现 固定优势进入悲观裁剪"
ratio = (logp - old_logp.detach()).exp()
plain = ratio * advantages.detach()
bounded = ratio.clamp(1 - low, 1 + high) * advantages.detach()
loss_items = -torch.minimum(plain, bounded)
actor_loss = reduce_tokens(loss_items, action_mask)
```

### 用四格理解裁剪 而不是背越界即无梯度

取对称epsilon=0.2、$|A|=2$，列出单项loss对当前logprob的导数：

| 优势与ratio | 行为含义 | 导数 | 作用 |
| --- | --- | --- | --- |
| A=2，r=0.7 | 好动作反而变少 | -1.4 | 继续提高概率 |
| A=2，r=1.3 | 好动作已明显变多 | 0 | 本项进入平台 |
| A=-2，r=0.7 | 坏动作已明显变少 | 0 | 本项进入平台 |
| A=-2，r=1.3 | 坏动作反而变多 | 2.6 | 继续降低概率 |

平台只抑制有利方向的额外收益。共享参数、KL或value loss还可能改变同一概率；PPO clip不是对真实KL的硬约束。PPO原图与历史四行源码见[PPO精读](#paper=paper-ppo)，完整张量到step的解释见[PPO主文](#q=rl-ppo-training)。

## 6. 第三层 GSPO SAPO与CISPO各改了什么

### GSPO 让一条回答共享概率比

Token ratio各自波动时，一条回答可能只有局部token进入平台。GSPO先对有效token的log-ratio求平均：

$$
s_i=\exp\left(\frac1{L_i}\sum_t(\ell_{i,t}-\ell^{old}_{i,t})\right).
$$

再用$s_i$替换clip中的token ratio，原sequence形式还在回答层面平均。它是token ratio的几何平均：ratio为[2,2]，$s=2$，完整序列概率比却为4；[2,0.5]的几何平均为1，算术平均是1.25。长度归一化改善尺度，也改变了权重，所以通常不能称为完整序列的无偏importance sampling。[Qwen作者讲解](https://qwenlm.github.io/blog/gspo/)

GSPO-token用下面接线，前向共享$s_i$，后向保留token自己的logprob路径：

```python title="教学实现 GSPO的前向共享与后向路径"
delta = logp - old_logp.detach()
seq_delta = (delta * mask).sum(-1) / mask.sum(-1).clamp(min=1)
linked_delta = seq_delta.detach()[:, None] + logp - logp.detach()
ratio = linked_delta.exp()
```

`logp-logp.detach()`前向为0，但导数是1。若直接把`seq_delta.detach()`广播并exp，actor梯度会断开。若保留sequence梯度又额外加token路径，则可能重复求导。

**何时和直接sequence式等价？** 同一回答内优势相同，clip设置相同，且使用sequence mean；此时脚本验证两者loss和梯度一致。Token优势不同时，GSPO-token可以表达局部监督，而直接拿平均优势优化sequence ratio一般不是同一个梯度。详细原式和维护者源码见[GSPO完整稿](./paper-drafts/paper-gspo.md)。

### SAPO 以平滑函数代替硬平台

定义按优势符号选择的$\tau$：

$$
f_\tau(r)=\frac4\tau\sigma(\tau(r-1)),\quad
U=A f_\tau(r).
$$

其对ratio的导数为$f'_\tau(r)=4p(1-p)$，其中$p=\sigma(\tau(r-1))$；对logprob的导数还要乘$r$，因为$\partial r/\partial\ell=r$。因此目标$f$、gate $f'$、实际logprob权重$rf'$是三个对象。

```python title="教学实现 SAPO让autograd求正确的gate"
tau = torch.where(advantages > 0, tau_pos, tau_neg)
ratio = (logp - old_logp.detach()).exp()
soft_surrogate = (4 / tau) * torch.sigmoid(tau * (ratio - 1))
loss_items = -advantages.detach() * soft_surrogate
```

在$r=1$处$f'=1$，局部梯度与未裁剪目标对齐；离开中心后平滑衰减。$\tau$乘在sigmoid输入上，越大衰减越快。不要手写`gate(r) * ratio`作为可微目标，否则多出gate自身的导数。原实验、正负温度及现代源码见[SAPO完整稿](./paper-drafts/paper-sapo.md)。

### CISPO 裁系数但继续训练logprob

CISPO的教学通式是：

$$
L=-\mathbb E[\operatorname{sg}(\operatorname{clip}(r,l,u))A\log\pi_\theta(a\mid s)].
$$

权重停止梯度，logprob继续可微，所以越界token不必像PPO有利方向那样进入零梯度平台。相应裁剪仍引入偏差，优势为0或mask为0也仍没有这项更新。[MiniMax-M1](https://arxiv.org/abs/2506.13585v1)

```python title="教学对照 三种surrogate的局部差异"
ppo_items = -torch.minimum(ratio * adv, ratio.clamp(low, high) * adv)
cispo_items = -(ratio.clamp(low, high).detach() * adv * logp)
sapo_items = -(4 / tau) * torch.sigmoid(tau * (ratio - 1)) * adv
```

三行的`adv`均视为固定，第一和第二行的`low/high`是直接ratio边界，例如0.8/1.2。这里比较局部目标，完整论文的采样、优势、阈值和其他正则仍需分别配置。

## 7. 第四层 DAPO改整套数据与更新配方 GDPO改多奖励优势

### DAPO 为什么不能只写一个loss函数

DAPO针对长推理中探索下降、无区分组增加和长度处理等问题。四个机制作用位置不同：

| 机制 | 修改代码的位置 | 需要讲清的效果 |
| --- | --- | --- |
| Clip Higher | policy loss上下界 | 上界放宽，给正优势低概率动作更多增长空间 |
| Dynamic Sampling | collector与过滤补采循环 | 排除无区分组后继续补足有效组 |
| Token level loss | 最后reduction | 按全部有效token平均，改变长短回答权重 |
| Overlong处理 | reward与样本有效性 | 区分长度截断、长度软惩罚和真实错误 |

```python title="教学伪代码 Dynamic Sampling改变外层控制流"
accepted = []
while len(accepted) < target_prompt_groups:
    groups = collect_complete_prompt_groups()
    informative = [g for g in groups if has_reward_variation(g)]
    accepted.extend(informative)
batch = pack_groups(accepted[:target_prompt_groups])
# 另外记录本轮为补采实际生成的所有tokens、环境调用和耗时
```

纯binary reward下，“有变化”对应组正确率严格在0与1之间；多reward或连续分数时不能机械复用这个判断。真实系统还要有补采上限和不足batch的处理，避免所有题都无法区分时无限循环。源码控制流和超长reward细节见[DAPO精读](#paper=paper-dapo)，[作者项目](https://dapo-sia.github.io/)提供整体动机。

统一脚本的`method="dapo"`只展示上clip与token mean，特意把filter/refill放在外层解释。运行这个局部函数不等于实现完整DAPO。

### GDPO 为什么先分开标准化 再加权合并

如果正确性、格式、长度的尺度不同，先加总再组标准化会隐藏reward的组成。GDPO保留reward轴：

$$
z_{i,k}=\frac{R_{i,k}-\mu_{q,k}}{\sigma_{q,k}+\epsilon},\quad
S_i=\sum_k w_kz_{i,k},\quad
A_i=\operatorname{Whiten}_{batch}(S_i).
$$

先将某维乘正权重再z-score，这个倍率可能被该维std消掉；把权重放在归一化之后，才能表达对已标准化信号的相对重视。若各reward等权互补，例如[1,1,0]和[0,0,1]，合并后仍可完全抵消，所以GDPO并不保证所有同总分回答都能区分。[作者项目](https://nvlabs.github.io/GDPO/)

```python title="教学实现 保留reward维度"
# reward_vectors: [Q,G,K]
z = reward_vectors - reward_vectors.mean(dim=1, keepdim=True)
z = z / (reward_vectors.std(dim=1, keepdim=True, correction=1) + 1e-6)
merged = (z * reward_weights).sum(dim=-1)   # [Q,G]
advantages = whiten_over_responses(merged)
```

这是等权回答统计的教学版本；[作者GDPO分支](#paper=paper-gdpo)将优势广播后按token mask whiten，长度不同会改变统计权重。GDPO主要替换优势，不自动更换ratio、surrogate或采样器；可组合也不保证收益相加。

## 8. 第五层 mask与reduction也是目标的一部分

Attention mask决定可看什么；action/loss mask决定在哪些位置承担监督。工具观察通常mask为0，却仍是后续生成的条件。Padding没有真实动作，不能算入概率比平均、优势统计或loss分母。

设两条回答有效长度分别2和4，每token loss分别为1与3：

$$
L_{seq}=\frac12(1+3)=2,\qquad
L_{token}=\frac{2\times1+4\times3}{2+4}=\frac73.
$$

Sequence mean让每条回答总权重相同；token mean让每个有效token权重相同。统一脚本的`reduce_tokens`将此差别显式化：

```python title="教学实现 相同loss矩阵的两种归约"
lengths = mask.sum(-1)
sums = (loss_items * mask).sum(-1)
token_mean = sums.sum() / lengths.sum()
valid = lengths > 0
sequence_mean = (sums / lengths.clamp(min=1))[valid].mean()
```

全batch没有有效action应中止或跳过；给分母clamp为1不能把无样本训练变为有效训练。分布式global token mean要按全局有效token数归一化，不是把各卡局部mean等权平均。若DDP最终对$D$个rank梯度取平均，各rank使用本地loss sum时通常还需乘$D/N_{global}$；具体后端的求和/平均约定要查清。

梯度累积同理：每个microbatch的token sum除同一个global分母，才等价于整batch token mean。分别除自己的长度再等权累积，会改变不同microbatch权重。脚本验证了两种拆分在global分母下得到同一梯度；这属于目标实现正确性，与选择PPO还是SAPO无关。

## 9. DPO和OPD 为什么放在旁边而不是硬排进替代链

### DPO 从偏好对得到分类目标

原始离线DPO使用同prompt的chosen和rejected，四个完整回答logprob构成margin：

$$
m=\beta[(\ell_{\theta,w}-\ell_{ref,w})
-(\ell_{\theta,l}-\ell_{ref,l})],\qquad L=-\log\sigma(m).
$$

它来自KL正则奖励最优化与Bradley–Terry偏好模型的重参数化；原训练不需要每一步在线rollout、显式critic或PPO clip。这里reference是偏好对比较的锚点，不是采样行为策略。

```python title="教学实现 四个回答分数进入DPO"
relative_chosen = chosen_logp - ref_chosen_logp.detach()
relative_rejected = rejected_logp - ref_rejected_logp.detach()
loss = -F.logsigmoid(beta * (relative_chosen - relative_rejected)).mean()
```

当前chosen=-2、rejected=-3，reference二者均=-2.5，beta=0.1时margin=0.1、loss约0.6444；两个回答分数的loss梯度方向相反。它强化相对margin，但共享模型更新后不保证chosen绝对概率每次都上升。[DPO完整推导与官方实现](#paper=paper-dpo)

### OPD 从学生访问的状态获得教师分布

OPD让student采样轨迹，teacher在这些prefix上提供概率信息。其loss可使用不同KL方向或采样近似，不需要天然有任务reward。与上述reward-RL组合时要分别说明：状态来自谁、教师提供什么、优化哪个分布距离，以及行为采样是否校正。

同tokenizer时可比较词表分布；cross-tokenizer时还要定义共同事件，不能把不同ID的概率直接相减。该问题继续看[OPD原理](#q=opd-vs-sft)、[Agentic OPD](#q=agentic-opd-training)与[ESCD项目](#q=escd-project-interview)，不在RL变体表里把它压成一个clip修改。

## 10. 一张表复述所有差异

| 路线 | 反馈或优势 | ratio或surrogate | 数据与归约重点 |
| --- | --- | --- | --- |
| Actor Critic | 学V/Q辅助估计A | 不指定唯一更新目标 | MC/TD、状态与边界 |
| PPO | 常配critic与GAE | current/old比、悲观clip | 同批复用、KL与版本 |
| RLOO | 排除自身的组baseline | REINFORCE式或明确复用配方 | G≥2、独立性、组大小 |
| GRPO | 同prompt组center/std | 常用token PPO clip | 同题多采样、组屏障 |
| Dr.GRPO | 去local reward std | 检查具体配方 | 去长度偏置、固定分母 |
| REINFORCE++ | global normalization，baseline版先减组mean | 按其版本配置 | 全局统计与KL |
| GSPO | 通常继承组优势 | 几何平均sequence比与共享clip | sequence mean、token梯度路径 |
| SAPO | 可接组优势 | 平滑sigmoid代理目标 | 正负tau与gate饱和 |
| CISPO | 固定优势 | clipped stop-grad权重乘logprob | 权重与梯度路径分开 |
| DAPO | 组反馈 | 非对称clip | 过滤补采、token mean、超长处理 |
| GDPO | 逐reward组归一化后加权 | 可接GRPO式更新 | batch whitening的统计单位 |
| DPO | 离线偏好pair与reference | logitsigmoid分类 | 序列logprob求和、偏好覆盖 |
| OPD | teacher在学生状态上的分布 | 条件蒸馏 | KL方向、事件对齐与状态覆盖 |

面试讲两算法时，先说共同骨架，再说修改位置，随后手算一个数或指明一行代码，最后解释适用条件。论文的实验数字与算力口径在[证据对照](#report=survey-policy-optimization)中，不在此拼成排行榜。

## 11. 面试问答 用问题检查理论和实现

**Q01 [分类] PPO和GRPO最先比较什么？** PPO是近策略更新框架；常见PPO用critic/GAE，GRPO以同题组统计替代学习V，仍可用token clip。追问：原GRPO的reference KL与old ratio各做什么？

**Q02 [推导] Reward不可微为什么能训练模型？** 梯度经过已采样动作的logprob，环境不需要可微。追问：reward若显式依赖可训练参数，目标还要哪些导数？

**Q03 [baseline] 减V为什么不改变理想期望梯度？** 条件于状态的score期望为0，baseline须不依赖当前动作。追问：含本样本的组均值是否直接满足条件？

**Q04 [RLOO] 去掉GRPO的std就是LOO吗？** 固定G时还差G/(G-1)；变G不能吸收成同一常数。追问：G=1怎么办？

**Q05 [统计] 组reward全同会爆炸吗？** 正确center后通常为0，缺区分信号；不是epsilon自动制造巨大优势。追问：独立KL项是否仍会更新？

**Q06 [GAE] 它把什么沿时间累积？** TD残差，递推系数gamma×lambda。追问：bootstrap与trace为什么需要两个mask？

**Q07 [概率] Ratio为1是否意味着梯度0？** 不，冻结分母时分子仍可微。追问：分子分母用同一个未detach张量会怎样？

**Q08 [clip] 越界就不训练吗？** 看优势符号，正优势上侧、负优势下侧进入平台，反方向仍纠错。追问：KL是否受到硬约束？

**Q09 [GSPO] 为什么几何平均不是完整IS比？** 完整比是乘积，GSPO取长度次方根；[2,2]给4与2。追问：平均是否能隐藏单token异常？

**Q10 [梯度] GSPO-token的前向0项有什么用？** logp−sg(logp)保留token梯度。追问：何时与直接sequence式等价？

**Q11 [SAPO] f、f′、r f′分别是什么？** 代理目标、ratio导数gate、logprob梯度因子。追问：把gate×ratio再求导多出哪一项？

**Q12 [CISPO] 为什么裁剪了还可以更新？** 系数detach，logprob继续可微。追问：这是否保留严格无偏IS？

**Q13 [DAPO] 一个loss函数能表达完整算法吗？** 不能，过滤补采是外层控制流，超长处理在reward/有效性层。追问：没有可区分组时如何退出？

**Q14 [GDPO] reward weight放在std之前会怎样？** 整维正倍率可能被z-score抵消。追问：逐维后互补reward还可能抵消吗？

**Q15 [归约] 长度2与4的loss为1和3怎么算？** Sequence mean是2，token mean是7/3。追问：按卡局部mean再平均是否正确？

**Q16 [mask] 工具文本mask0会让模型看不到吗？** Loss mask控制监督，attention条件仍保留。追问：GAE是否应让每个工具字符推进时间？

**Q17 [DPO] reference是behavior吗？** 原DPO是偏好锚点，不承担PPO采样校正。追问：四个序列分数如何得到？

**Q18 [target] 白化优势加old value能训critic吗？** 原回报单位不一致，先构造raw return target。追问：reward乘10后value MSE怎样变化？

**Q19 [采样] Top-p logprob能直接塞old吗？** 要匹配目标策略和behavior契约，采样校正与proximal ratio分开。追问：何时起点ratio不为1？

**Q20 [复用] Soft gate能解决任意staleness吗？** 不能，状态分布漂移、行为概率与优势陈旧仍存在。追问：队列里到底存什么版本？

**Q21 [实验] 七篇结果能排一个总榜吗？** 模型、reward、长度和预算不同，应回同预算控制实验。追问：过滤补采的浪费算在哪里？

**Q22 [实现] Loss正常下降但成功率下降先查什么？** Reward/环境、真实输入、mask、边界、版本、统计分母，再调超参。追问：如何构造一个可枚举反例？

**Q23 [Agent] 换成GRPO能自动解决长程credit吗？** 组优势主要提供回答相对反馈，中间决策仍可能粗粒度。追问：细粒度状态baseline改变哪一层？

**Q24 [教学] 怎样证明自己会讲而非背名字？** 推导一段、手算一组、读一处真实分支、运行一次梯度核对，并解释一个失败条件。

## 12. 来源与下一步

原论文、作者项目和固定实现见[来源清单](./research/rl-policy-optimization-map-sources.json)，本轮直接核读的公开页面与代码另记[字节账本](./assets/learning/rl-algorithms-source-evidence.json)；[论文对照报告](#report=survey-policy-optimization)保留每篇实验条件、真实源码摘录和原图表入口。GSPO与SAPO的完整稿可直接下载，继续沿用其单篇草稿状态。

课堂材料分工：本篇[统一脚本](./assets/learning/rl-algorithms-unified-lab.py)连接训练数据和optimizer；原[策略优化局部演示](./assets/learning/policy-optimization-lab.py)专门检查各变体手算；[基础训练演示](./assets/learning/agent-training-lab.py)覆盖baseline方差、GAE边界与组统计。先运行统一样例，再按疑问选择局部练习，无需逐个重复运行。
