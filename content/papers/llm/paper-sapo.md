---
title: "Soft Adaptive Policy Optimization"
authors: ["Chang Gao", "Chujie Zheng", "Xiong-Hui Chen", "Kai Dang", "Shixuan Liu", "Bowen Yu", "An Yang", "Shuai Bai", "Jingren Zhou", "Junyang Lin"]
affiliations: ["Qwen Team, Alibaba Inc."]
venue: "arXiv preprint"
year: 2025
published: "2025-11-25"
summary: "推导SAPO的sigmoid目标与4p(1-p)门控，解释正负优势不对称温度、序列一致性近似条件，以及延迟训练崩溃而非稳定性保证。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-09"
template_version: 5
depth_standard: "ddpm"
draft: true
id: "paper-sapo"
paper_title: "Soft Adaptive Policy Optimization"
paper_url: "https://arxiv.org/abs/2511.20347v2"
method_figure: "./assets/papers/paper-sapo/figure-1-source.png"
method_caption: "Figure 1 sigmoid代理目标与平滑梯度门控"
github_url: "https://github.com/verl-project/verl"
tags: ["SAPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
author_affiliations: [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
---

## 1. 背景与已有工作

### 问题背景

硬clip存在稳定性与信号利用的取舍：范围宽时异常token可能造成噪声，范围窄时有用更新又突然进入平台。对同一个长回答，少量off-policy token可能使sequence clip连带抑制其他仍有用的token。SAPO希望以连续衰减替代突然的平台切换。

这里指Qwen的Soft Adaptive Policy Optimization，而非其他同缩写论文。它保留token ratio与组优势，主要改变surrogate和正负优势的门控宽度。没有因为soft gate而自动解决reward错误、组完成等待或无限stale数据的问题。

### 前置知识

**代理目标与导数。** 训练最大化一个surrogate函数，实际更新取决于其导数；函数值大不等于梯度大。若用$f(r)A$作目标，logprob梯度系数是$f'(r)rA$，不能把$f$、$f'$或$f'r$三个量混在一起。

**sigmoid与tau约定。** $\sigma(z)=1/(1+e^{-z})$，导数为$\sigma(z)(1-\sigma(z))$。SAPO将tau乘在输入上，tau越大有效区域越窄；这不同于softmax用logits除temperature的约定。4/tau因子用于保持ratio=1附近导数尺度。

**advantage正负与softmax竞争。** 正优势强化采样token，负优势降低它并把概率分散到其他词。LLM大词表中的未采样词未必正确，这解释为何作者对off-policy负更新更保守；不能据此说负反馈没有作用。

**token与sequence尺度。** GRPO逐token硬clip，GSPO按几何平均ratio共享sequence clip，SAPO仍逐token平滑抑制。sequence coherence是特定条件下平均门控的近似性质，不是训练中真正只计算一个sequence ratio。

**近似条件与误差范围。** ratio接近1时$r-1\approx\log r$；同sequence log-ratio方差较小时，平均gate可接近sequence gate。标量gate误差界不自动成为完整向量gradient界，还需考虑梯度与权重的相关性。

### 已有工作与本文位置

PPO/GRPO以有利方向硬平台控制更新；GSPO改变ratio与clip单元。SAPO用sigmoid目标产生连续gate，并用$\tau_->\tau_+$更快衰减负优势更新。[固定v2 §2—§4](https://arxiv.org/pdf/2511.20347v2)。

| 方法 | 权重与平台 | SAPO要改变的点 |
| --- | --- | --- |
| GRPO | token ratio与单侧硬平台 | 让越界附近信号连续衰减 |
| GSPO | sequence共享ratio与平台 | 避免局部异常连带抑制全部token |
| SAPO | token软目标与符号不同tau | 需控制饱和、温度和近似条件 |

## 2. 方法与实现机制

### sigmoid代理目标而非手工乘一个权重

对同题组回答$y_i$、有效长度$L_i$和组相对优势$\widehat A_{i,t}$，令tokenratio$r_{i,t}=\exp(\ell_\theta-\ell_{\mathrm{old}})$。SAPO最大化：

$$
J_{\mathrm{SAPO}}=\mathbb E\left[\frac1G\sum_i\frac1{L_i}\sum_t
f_{i,t}(r_{i,t})\widehat A_{i,t}\right],\qquad
f_{i,t}(r)=\frac4{\tau_{i,t}}\sigma\bigl(\tau_{i,t}(r-1)\bigr).
$$

温度取决于优势符号：$\tau_{i,t}=\tau_+$当$\widehat A>0$，否则$\tau_-$。所谓temperature在这个公式中是sigmoid输入的乘数，越大曲线越陡、衰减越快，不能套用“softmaxtemperature越大越平”的常见直觉。

![Figure 1 · SAPO代理目标与门控导数的原始比较](./assets/papers/paper-sapo/figure-1-source.png)

**Figure1解读。** 左图纵轴objectivevalue，右图纵轴梯度gate $w$，横轴均为tokenratio；绿色/蓝色/紫色是不同tau，橙色是positiveadvantage下的硬clip示意，黑虚线为未裁剪。左侧不同温度有不同常数偏移，不能从loss绝对值判优；右侧都在ratio=1达到1，偏离后平滑衰减。示意hardclip使用epsilon1，不是GSPO实际3e-4/4e-4阈值。[图源：v2 Figure1，独立原资产](https://arxiv.org/src/2511.20347v2)。

### 真实梯度为什么是4p乘1减p

定义$p=\sigma(\tau(r-1))$，则：

$$
f'(r)=4p(1-p)=w(r),\qquad
\nabla_\theta[f(r)\widehat A]
=w(r)\,r\,\widehat A\nabla_\theta\log\pi_\theta(a\mid s).
$$

$w(r)\in(0,1]$，ratio=1时$p=0.5,w=1$。**门控是$w$，最终乘在logpolicygradient前的系数是$wr\widehat A$**，二者不能混淆。若直接优化`w(r)*r*A`并让梯度经过w，会多出$w'$项，得到另一个目标；若用detach人工加权，则需要证明与$f$导数相符。

因子4/tau使$f'(1)=1$，保留on-policy附近的未裁剪梯度尺度。$f(1)=2/\tau$并不统一，但它只是每个固定advantage项的参数无关常数偏移，比较不同温度时看gradient与quality比看rawloss更有意义。由于advantage需固定，不能让这个常数沿reward/advantage生成管线回传。

手算tau=1：ratio1时w=1；ratio2时p约0.7311、w约0.7864，logprob梯度系数wr约1.5729；ratio5时w约0.0707、wr约0.3533。它不是ratio稍超过1就几乎没梯度，也不是w有界就代表整体梯度范数严格有界。极端ratio下sigmoid饱和，有限精度还会实际出现0；“soft”不保证所有旧数据都有可用梯度。

一个易错实现是把ratio替换成log-ratio输入sigmoid。原算法是$\sigma(\tau(r-1))$，论文后面用log-ratio建立近似理论联系，但训练核心目标并非$\sigma(\tau\log r)$。两者在r接近1时相近，远离1时衰减明显不同。

### 正负优势为何使用不同tau

对采样token$a$，softmaxlogit$z_v$的logprob梯度为：

$$
\frac{\partial[\widehat A\log\pi(a)]}{\partial z_v}
=\widehat A\bigl(\mathbb1\{v=a\}-\pi(v)\bigr).
$$

正优势提高采样token的logit并降低其余；负优势降低采样token，概率质量分散给未采样token。LLM词表很大，当前状态下合适token只是其中少数，负更新可能把概率推向许多未验证的选择。作者用$\tau_->\tau_+$更快抑制off-policy负更新，并通过温度消融支持该设计。

这是一种关于分布与实测的稳定性解释，不能推导成“负优势没有用”或“越大tau越安全越好”。负反馈对避免错误与探索重要，过度抑制会让模型只学强化而不学纠错。不同reward、词表、temperature、G和长度设置下，最优tau差值需要实测。

与DAPO非对称clip相比：DAPO保留两侧不同硬边界；SAPO仍以r=1居中，但按优势符号使用不同平滑宽度。与GSPO相比：SAPO按token调节，GSPO共享sequence权重；不能把“sequencecoherence”译成“sequencehardclip完全一样”。

### 序列一致性的条件与误差界

原文假设A1：多数tokenratio接近1，使$r-1\approx\log r$；A2：同序列log-ratio离散度较低。令$z_t=\log r_t$、$\mu=\frac1L\sum_tz_t=\log s$，近似gate为$g_\tau(z)=\operatorname{sech}^2(\tau z/2)$。对均值Taylor展开，一阶偏差项平均消掉，得到：

$$
\left|\frac1L\sum_tg_\tau(z_t)-g_\tau(\mu)\right|
\le\frac{\tau^2}{4}\operatorname{Var}(z).
$$

这个界约束**平均标量gate**，不是完整策略gradienterror界；gradient中还有每token的logprob梯度向量、ratio与advantage，乘积平均要考虑相关性。作者进一步将其解释为GSPO-like平滑sequence更新，本文保留“近似、低离散度、小步”条件，不称为普遍严格等价。

论文在Qwen3-30B-A3B与Qwen3-4B上统计超过$10^5$sequence、$10^9$token，观察ratio集中1附近、多数sequencevariance小于0.02。这个诊断支持多数样本中的条件，但MoE尾部更宽，且仍有异常样本。长期异步旧数据、环境分布变化或跨turn换权重可能让A1/A2失效；SAPO的门控不能替代staleness控制。

### 实现接线与日志解释

```python title="SAPO 教学重述 · 流程示意"
# 原始SAPO目标的教学重述
ratio = exp(logp - old_logp.detach())
tau = where(adv > 0, tau_pos, tau_neg)
f = 4 / tau * sigmoid(tau * (ratio - 1))
loss = mean_of_sequence_means(-f * adv.detach(), action_mask)
# 不直接把w=4*p*(1-p)当loss；w来自上述目标的自动求导。
```

原文没有可复查全部Qwen3-VL训练的完整专属代码。这里静态核读维护者verl实现，固定 `75879f7f475fd6b64c779f7d9212e45503f58b8f`。[core_algos.py](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py)。

`compute_policy_loss_sapo`输入 `[B,L]` logprob、oldlogprob、advantage和mask，在差值上做[-20,20]数值clamp后exp，按advantage符号选择tau，并通过内部`gate_function`返回4/tau乘sigmoid；最小化`-gates*advantages`。这个变量gates实际上存$f$，不是导数$w$；阅读名字不能替代公式。

`agg_loss`在SAPO入口明确使用`seq-mean-token-mean`，即回答内平均后sequence平均。组优势仍由GRPO估计入口提供，非actiontooloutput不参与mask。可选rolloutimportanceweights是额外校正，不包含在原SAPO核心式中。tau必须为正，否则4/tau不定义；放大tau虽然更快衰减，也增加标量近似界中的tau平方。

该实现为了统一接口把clipfraction日志返回0，因为没有硬clip；**日志0不意味着没有抑制任何token**。要观察gate分布、有效权重、ratio、KL和validationquality。`ppo_kl`也是采样logprob差的近似诊断，不能自动视为完整分布KL。本文核读函数与reduction，未训练复现。

### 课堂源码拆解：目标值与梯度权重要分两列

在白板上先写$f(r)$，旁边另写$f'(r)$，第三列写对logprob的导数$f'(r)rA$。把三者混在一起最容易写出“看着像SAPO，实际另一个算法”的代码。

#### 训练实际优化的函数

**真实源码节选：[SAPO 现代verl · 可微代理函数](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L1657-L1659)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="SAPO 现代verl · 可微代理函数"
def gate_function(x, tau):
    """The gating function used in SAPO"""
    return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)
```



变量命名gate_function，但返回的是$f=(4/\tau)\sigma(\tau(r-1))$，不是$w=4p(1-p)$。自动求导才产生w；不需要手工再乘一次w。若手工把w当loss并让其也求导，会多出w′项。

#### 选择温度并保留actor梯度

**真实源码节选：[SAPO 现代verl · 分符号温度与目标](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L1664-L1681)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="SAPO 现代verl · 分符号温度与目标"
negative_approx_kl = log_prob - old_log_prob
# Clamp negative_approx_kl for stability
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
# finally exp() to remove log and get r_{i,t}(θ)
ratio = torch.exp(negative_approx_kl)

# tau_{i,t} is tau_pos if adv > 0 else tau_neg
taus = torch.where(
    condition=advantages > 0,
    input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
    other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
)

# compute the gates f_{i,t}(r_{i,t}(θ)) at token level
gates = gate_function(ratio, taus)

# compute policy gradient loss
pg_losses = -gates * advantages
```



`negative_approx_kl`是current−old logprob，先数值clamp、exp得到ratio。`advantages>0`选taupos，否则tauneg；这些fixed advantages来自组估计器。`pg_losses=-gates*advantages`保留f的梯度，原函数后续按sequence均值reduce。额外rollout correction接口与此核心式分开。

输入 `[B,L]`无需full-vocabulary KL。ratio1时f=2/tau，但f′=1，因此同一on-policy样本不同tau的rawloss不同而初始策略梯度尺度一致。ratio2,tau1时w约0.7864，但对logprob系数是1.5729×A；这个数值能检验学生是否漏乘ratio。

对tau1/2/3画f与w，不以rawloss高低排名；增大tauneg会更快衰减负优势，不表示负例可以全部删除。最后比较同一回答中的outlier：SAPO弱化局部token，GSPO则共享sequence权重；只有近on-policy、低离散度等条件下才建立sequence近似联系。


## 3. 实验设置与算力

SAPO有两类主实验：受控文本数学训练比较优化算法和temperature，实际多任务Qwen3-VL训练比较大型MoE的reward与验证。固定版本没有实验/结果原表，公开证据主要是完整曲线；缺少LR、batch和硬件的详细账本使精确复现仍需作者或框架配方支持。

| 项目 | v2披露 | 口径与限制 |
| --- | --- | --- |
| 文本controlled模型 | Qwen3-30B-A3B-Base的cold-startcheckpoint | 初始化不是直接未SFTbase |
| controlled任务 | mathematicalreasoningqueries | 具体数据快照与全量配比未给出 |
| baseline | GSPO与GRPO-R2 | R2表示routingreplay，不是新reward |
| 更新组织 | rolloutbatch分成4个mini-batch | 与同批off-policy更新有关 |
| 主temperature | taupos1.0、tauneg1.05 | 高tau更快衰减负优势 |
| temperature消融 | negative1.05/1.0/0.95，positive均1.0 | 有限三点，不是完整sweep |
| controlledeval | AIME25、HMMT25、BeyondAIME | 平均Pass@1，16samples |
| 理论诊断 | 30B-A3BMoE与4Bdense | 超10万sequence、10亿token统计 |
| VL主比较 | cold-startQwen3-VL-30B-A3B | 原文也声明应用于不同规模，但主图是此模型 |
| VL任务 | text与multimodal的math/code/logicalreasoning | batch内固定taskratio |
| VL更新 | 大batch分2个mini-batch | 不与文本4个mini-batch混写 |
| VL验证 | AIME25 32samples、LCBv6 8samples、ZebraLogic、MathVision | 报aggregatevalidation，不是逐任务统一胜率 |
| optimizer/LR/G/maxlength/seeds | 未给完整独立可执行配置表 | 引用GSPO配方也不足以自动补齐全部缺项 |
| GPU型号/数量/precision | 未披露此比较完整设备账本 | 不由30B总参数推算原训练卡数 |
| 时间/GPU-hours | 没有可复算绝对成本 | 相同步数不自动相同end-to-end预算 |
| 原表/代码状态 | 无原始结果表；现代verl实现伴读 | 不制造实验表或声称作者完整代码公开 |

文本数学是AIME2025，GSPO原论文是AIME2024；本文不把二者分数混成一个增长列表。16/32samples用于估计averagePass@1，不是Pass@16/32。VL图中的aggregatevalidation还混合数学、代码、逻辑与图像数学，需要任务配比和聚合定义才能解释绝对分值，不能说“所有任务提高某固定百分点”。

各baseline使用同初始化、预算和配方的controlled比较，比跨论文榜单更可信；但没有多seedconfidence和完整硬件数据仍限制其泛化。主temperature仅相差0.05，证明这组设置有明显影响，不代表所有模型都应固定1.05/1.0。

“continuous trust region”是软抑制的设计描述，没有像TRPO约束一样逐状态硬KL限制。训练与推理数值差异、routing、长序列mask和采样行为仍需要系统验证。对自己的异步OPD项目，SAPO是policy-gradientRL的近邻，不可不经推导就把sigmoidgate搬到teacherweightedKL蒸馏上。

## 4. 结果与图表解读

![Figure 4 · 文本数学的三算法训练与验证曲线](./assets/papers/paper-sapo/figure-4-source.png)

**Figure4解读。** 四面板为trainingreward、HMMT25、BeyondAIME、AIME25，横轴gradientsteps；蓝SAPO、橙GSPO、绿GRPO-R2。绿线在约后段突然下降，橙线也有退化；蓝线在展示预算内保持更久的有效提升，AIME约0.81、HMMT约0.64只是视觉近似，不报成精确表格成绩。图支持延迟不稳定与提升可用训练quality，不证明SAPO永不崩溃。[图源：v2 Figure4](https://arxiv.org/src/2511.20347v2)。

作者导言明确承认这些方法最终都可能表现出不稳定迹象。将“extendsstabletraining”译成“保证训练稳定”会夸大结论。图中各曲线长短不同，应比较共同steps范围与各自最好/最后性能，不把最后一个点当成同预算、同选择规则的精确summary。

![Figure 5 · 正负temperature关系的三配置消融](./assets/papers/paper-sapo/figure-5-source.png)

**Figure5解读。** 四面板与Figure4指标相同；蓝色tauneg>taupos、橙相等、绿tauneg<taupos。绿色更早崩溃，橙色能先达到较高reward却后期突然下降，蓝色维持更长提升。这说明softgate本身还需合适的符号不对称；早期reward高不能保证后期quality。三组tau只是1.05/1.0/0.95的有限设置，不支持tauneg越大越好。[图源：v2 Figure5](https://arxiv.org/src/2511.20347v2)。

![Figure 6 · Qwen3VL多任务训练的reward与聚合验证](./assets/papers/paper-sapo/figure-6-source.png)

**Figure6解读。** 左为trainingreward，右为aggregatevalidationscore，横轴gradientsteps；蓝SAPO在同一曲线范围内高于橙GSPO、绿GRPO-R2。右端约0.758对约0.749是图读近似，而且是聚合指标，不可据此声称每个任务都提升0.9个百分点。实验把softgate推广到multimodal场景，但主图只有一个30B-A3B模型，没有分别展示所有dense/MoE规模的逐项对照。[图源：v2 Figure6](https://arxiv.org/src/2511.20347v2)。

三组证据分别回答“与baseline比”“是否真需要temperature不对称”“是否能进入实际多任务VL配方”。它们来自同作者团队，不是三个独立复现。模型、采样次数、benchmark年份和曲线选择都应该随数字保留；训练曲线高于baseline也不能证明降低部署延迟或消除group-completionbarrier。

## 5. 局限、结论与后续阅读

SAPO用平滑代理目标与正负不同temperature，改善硬clip带来的突然信号损失，在作者文本和VL实验中保持更久的有效学习。它仍可能sigmoid饱和、受reward偏差与分布漂移影响；sequencecoherence只在小步、低离散度等条件下近似成立，没有严格普遍KL或回报保证。

### 面试问答与追问

**Q1：SAPO的loss与gate有什么区别？** loss使用$f=(4/\tau)\sigma(\tau(r-1))$乘优势；求导后gate是$w=4p(1-p)$，logpolicy梯度还乘ratio。直接把w放进loss不保持相同导数。

**Q2：为什么需要4/tau？** 它让$f'(1)=1$，保持on-policy附近梯度尺度。$f(1)=2/\tau$只是参数无关偏移，不能拿不同tau的rawloss大小排名。

**Q3：tau越大越平滑吗？** 此处tau乘在sigmoid输入上，越大衰减越快、有效区域越窄；仍连续但不是更宽容。与softmax除temperature的约定不同。

**Q4：softgate是否所有样本都有梯度？** 数学上有限r与正tau的导数非零，但可能极小，浮点饱和还会变零；有限gate也不意味着整个gradientnorm有硬界。

**Q5：为什么负优势更强抑制？** 负更新把概率分散给许多未采样词，大词表off-policy环境中可能更不稳；作者用tauneg更大并做消融。负反馈仍重要，不能直接删掉所有negativeadvantage。

**Q6：与GSPO严格等价吗？** 不等价。小ratio变化、低logratio离散度时平均gate可近似sequencegate；标量gate误差界不是完整梯度误差界。离散度高时token适应性是它的关键区别。

**Q7：如何看日志clipfraction=0？** 无硬clip接口常回零，并不代表没有抑制。应监控gate分布、ratio、有效更新权重、KL与held-outquality，不以日志名称替代机制。

**Q8：能处理任意旧数据或异步OPD吗？** 它针对RLpolicygradient权重，不能修正所有状态分布变化，也不自动解决teacher缓存。异步需要behavior版本、背压与算法校正，OPD需独立推导loss依赖。

参考[固定v2原文](https://arxiv.org/pdf/2511.20347v2)、现代verl固定源码、GSPO作者讲解与[统一专题](#report=survey-policy-optimization)。2026-10-08核读正文、原图与函数，补充gate/目标区别、近似条件和结果边界；未执行个人训练复现。原文无表，按当前仓库规则保留草稿，等待本轮关于原曲线替代表格的处理选择。




**2026-10-09更新。** 背景拆为问题背景、前置知识、已有工作；补固定源码节选、逐段形状/梯度讲解与课堂检查。源码节选不是完整可运行训练程序；课堂张量练习见[CPU演示脚本](./assets/learning/policy-optimization-lab.py)，不下载模型且不执行真实RL训练。
