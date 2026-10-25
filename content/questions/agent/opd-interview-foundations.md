---
id: opd-interview-foundations
title: "OPD系统整理：概率基础、公式推导、训练实现与面试问答"
category: agent
difficulty: 进阶
tags: ["P0", "OPD", "Interview", "KL", "Policy Gradient", "Distillation", "系统教材"]
updated: "2026-10-25"
summary: "面向初学者的OPD完整学习链：统一符号，推导FKL/RKL；用同一算例、梯度与可运行PyTorch代码对照sampled-token/top-k/全词表，连接异步、多教师、复读/截断/reward hacking，配80道问答与两套CPU验算。"
draft: false
---

## 1. 问题背景：OPD要解决什么，本文怎样阅读

On-Policy Distillation，简称OPD，中文可译为“同策略蒸馏”或“在线学生轨迹蒸馏”。它的核心是：**学生按照自己的策略生成，教师在学生实际访问到的上下文中提供监督，学生更新后重新生成。**“在线”在这里指训练数据随学生更新而重新产生，不表示必须实时接收真实用户请求。

理解OPD有一条连续的因果链：离线示范不能覆盖所有学生错误前缀，因此改用学生轨迹；既然教师在学生前缀上给出概率，就要选择概率匹配目标；词表太大时需要监督近似；近似、状态覆盖与教师可靠性共同决定纠偏能力，也共同解释复读、截断和目标偏离。

本文先讲基础，再给推导和实现，最后提供面试问答。第一次阅读建议按第2—7节理解概率、前缀、角色和目标；第二次读第8—16节掌握梯度、近似与工程；第三次读第17—24节理解异常、评估与扩展，最后闭卷回答题库。只准备简答时，可先看第27节的分层讲稿，再回到正文查公式。

**如果最困惑的是“采样token、top-k、全词表有什么区别”，先读第10节开头的对照，再看10.6的同一个四token算例，最后对照第14节代码。**三种方法通常使用同一条学生生成的轨迹；区别首先是每个位置比较多少个“可能的下一个token”、采用什么数学目标，以及怎样获得梯度，不是生成三条不同的答案。

本篇覆盖标准自回归语言模型OPD的基础，以及异步、多教师、跨tokenizer和Agent场景的核心接口与边界。复杂跨词表源码详解读[Cross-tokenizer专题](#q=opd-interview-cross-tokenizer)，真实工具交互和系统接线读[Agentic OPD](#q=agentic-opd-training)与[Pipeline](#q=agentic-opd-pipeline)，项目材料组织读[项目答辩](#q=opd-project-defense)。这些链接是深入阅读，不是初读本文的先修门槛。

以下公式默认使用自然对数，单位为nat；所有算例均为本文构造的有限概率例子。第28节给出实际执行的CPU验算及范围，没有把公式验算写成GPU训练、论文复现或个人项目成绩。题库按已登记公开题型与本文整理区分，不承诺覆盖所有可能的面试，也不宣称题目具有某个公司认证或出现频率。

## 2. 前置知识：从一个token的概率开始

### 2.1 Token、词表、logit和概率分别是什么

Token是分词器产生的离散符号，可能是一个字、一个词的一部分、标点或特殊标记。词表$\mathcal V$是模型可以输出的token集合，$V=|\mathcal V|$是词表大小。Token ID只是该集合中的编号，不能直接当成不同模型之间共享的语义。

模型在每个位置输出一个长度为$V$的实数向量，称为logits。Logit $z_v$不是概率；它可以为负，也不要求所有logit之和等于1。Softmax把它转换成概率：

$$
p(v\mid h)=\frac{\exp(z_v(h)/\tau)}{\sum_{u\in\mathcal V}\exp(z_u(h)/\tau)},\qquad \tau>0.
$$

这里$h$是当前上下文，$z_v(h)$是token $v$对应的logit，$\tau$是温度。概率满足$p(v\mid h)\geq0$且$\sum_vp(v\mid h)=1$。较高温度通常使分布更平缓，较低温度通常使分布更集中；温度只改变概率分布，不自动提供新知识。

实际计算常使用$\log p$，称为logprob。它不大于0，越接近0代表概率越高。例如概率0.8的自然对数约为$-0.223$，概率0.1约为$-2.303$。应使用稳定的`log_softmax`，避免先在低精度中得到0再取对数。

### 2.2 条件概率和完整回答概率

设问题或输入为$x$，回答为$y=(y_1,\ldots,y_T)$，其中$T$是回答包含的token数。自回归模型把完整回答概率分解为：

$$
\pi(y\mid x)=\prod_{t=1}^{T}\pi(y_t\mid x,y_{<t}),\qquad
\log\pi(y\mid x)=\sum_{t=1}^{T}\log\pi(y_t\mid x,y_{<t}).
$$

$y_{<t}$表示第$t$个token之前已经生成的内容。$h_t=(x,y_{<t})$称为前缀、历史或状态。对普通文本生成，这几个词常指同一段可见上下文；对Agent，真实环境状态还可能包含上下文中没有完整表达的信息。

EOS表示自然结束的特殊token或明确建模的终止动作。讨论完整可变长序列分布时，应包含它的概率；达到长度预算后强行停止，与模型自然选择EOS是两种事件。

### 2.3 期望、采样和梯度的直观意义

对于有限词表上的函数$f(v)$，期望为：

$$
\mathbb E_{a\sim p}[f(a)]=\sum_{v\in\mathcal V}p(v)f(v).
$$

这表示“每个值乘以发生概率，再相加”。从$p$随机抽取token $a$，用$f(a)$近似期望，叫Monte Carlo采样估计。单次结果可以偏高或偏低；无偏指在给定条件下对所有可能抽样取平均后等于目标，不指单个样本准确，也不保证训练一定有效。

梯度是损失对可训练参数的变化率。最小化损失时使用梯度下降$\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L$，$\eta>0$为学习率。本文中“提高/降低某token的倾向”首先指固定前缀下的局部梯度；真实模型参数跨任务共享，最终生成变化还取决于优化过程和其他样本。

## 3. 全文符号、条件与可微对象

为与站内既有OPD笔记一致，**$p$始终表示学生，$q$始终表示教师**。不要把“forward/reverse”由字母顺序猜出来，应看教师、学生的身份。

| 符号 | 定义与取值 | 本文中的作用 |
| --- | --- | --- |
| $x\sim\mathcal D$ | 从任务输入分布$\mathcal D$抽取问题$x$ | 决定学习哪些任务 |
| $y=(y_1,\ldots,y_T)$ | 学生生成的一条回答，$T$为实际长度 | 提供训练轨迹 |
| $h_t=(x,y_{<t})$ | 预测第$t$个回答token前可见的历史 | 教师和学生比较的条件 |
| $\mathcal V,V$ | 公共token集合及大小 | 定义当前离散样本空间 |
| $\theta$ | 学生当前可训练参数 | 唯一默认更新的模型参数 |
| $\bar\theta$ | 本批轨迹采样时的学生参数快照 | 区分采样版本与更新版本 |
| $\pi_\theta(v\mid h),p_v$ | 当前学生条件概率；固定$h$时简写为$p_v$ | KL中的学生分布 |
| $\pi_b(v\mid h),b_v$ | 实际行为策略及其概率 | 轨迹来自谁、IS分母是什么 |
| $\pi_T(v\mid h),q_v$ | 教师条件概率；默认教师冻结 | 提供蒸馏目标 |
| $\pi_{\mathrm{ref}}$ | 可选冻结参考策略 | 限制学生漂移，不等于教师身份 |
| $z_v$ | 学生当前位置token $v$的logit | 推导logit梯度 |
| $\tau_r,\tau_s,\tau_T$ | rollout、学生损失、教师评分的温度 | 三者不能默认相同 |
| $m_{it}\in\{0,1\}$ | 第$i$条轨迹位置$t$的有效训练mask | 区分模型动作、padding、外部观察 |
| $B,T_i,N$ | 批次轨迹数、第$i$条长度、$N=\sum_{i,t}m_{it}$ | 定义聚合与分母 |
| $S\subseteq\mathcal V,k$ | 候选token集合及目标大小 | top-k近似 |
| $P_S=\sum_{v\in S}p_v$ | 学生在集合$S$内的概率质量 | 不能与教师质量混用 |
| $Q_S=\sum_{v\in S}q_v$ | 教师在集合$S$内的概率质量 | 定义尾部质量$1-Q_S$ |
| $\operatorname{sg}(u)$ | 前向值等于$u$、反向梯度为0的停止梯度算子 | 构造局部代理损失 |
| $\rho_\theta(a,h)=\pi_\theta(a\mid h)/\pi_b(a\mid h)$ | 当前动作概率与行为动作概率之比 | 局部重要性修正 |
| $r_t,A_t,G_t$ | 即时信号、更新权重、未来累计回报 | 不把三者都叫“奖励” |

基础推导还假设：教师和学生使用相同样本空间；给定$h$时教师概率不随学生参数变化；默认温度为1；支持集满足相关KL和重要性采样的要求。跨tokenizer、上下文不同或教师随训练变化时，应重新定义目标，不能机械套公式。

训练中可微的是学生logits、学生概率以及学生参数。固定教师的logits、已采到的token ID、记录的行为概率和本轮采样历史不直接反传。具体代理损失是否还停止某个学生概率的梯度，必须由推导决定，不能凭“它属于target”就随意detach。

## 4. 离线蒸馏、SFT与OPD的关系

### 4.1 SFT怎样使用一条示范

监督微调SFT在固定示范$(x,y^*)$上最小化负对数似然：

$$
\mathcal L_{\mathrm{SFT}}(\theta)=
-\mathbb E_{(x,y^*)\sim\mathcal D_{\mathrm{demo}}}
\left[\sum_t\log\pi_\theta(y_t^*\mid x,y_{<t}^*)\right].
$$

$\mathcal D_{\mathrm{demo}}$是示范数据分布。公式目标是提高示范token在示范前缀上的概率。示范可以来自人工、教师生成、成功筛选或其他记录；SFT本身不规定数据一定由谁产生，也不保证示范一定正确。

硬标签给定一个目标token；软标签给出一个概率分布。软标签可以表达多个合理选择及相对偏好。教师生成完整答案后对答案做SFT，常称sequence-level KD；在每个固定前缀上匹配教师分布，常称logit/soft-label KD。二者不应混叫成相同实现。[经典KD原论文](https://arxiv.org/abs/1503.02531v1)说明软目标的动机。

### 4.2 为什么学生自己的前缀很重要

训练只看到“教师正确完成第1—3步”的上下文，而学生部署时第2步已经出错，之后的恢复状态可能不在示范分布中。前缀分布差异就是exposure bias讨论的主要内容之一，不等于所有泛化误差都由此产生。

OPD让学生产生这些真实前缀，再请求教师监督。它与DAgger的“在学习者访问的状态上询问专家”共享思路，但标准DAgger的算法与保证有自身条件，不能直接作为任意语言模型OPD的性能保证。[DAgger原论文](https://arxiv.org/abs/1011.0686v3)提供这一模仿学习背景。

### 4.3 三个独立设计轴

| 设计轴 | 要明确的选择 | 不能推出什么 |
| --- | --- | --- |
| 前缀来源 | 教师/固定数据、当前学生、旧学生、混合历史 | 学生生成不自动规定KL方向 |
| 匹配目标 | FKL、RKL、JSD、教师动作/偏好等 | 采用RKL不自动等于PPO |
| 监督表示 | 全词表、top-k、采样token、公共文本事件 | 只返回logprob不等于无需教师前向 |

本文的标准OPD指学生轨迹上的教师概率监督；广义学生状态上的文本示范、评价或偏好也可以研究，但需要按实际反馈定义，不能声称其已计算原定义的全词表KL。

## 5. 一次OPD训练究竟发生了什么

![本文示意：学生轨迹、教师评分与学生更新构成闭环](./assets/learning/opd-interview-foundations/opd-training-loop.svg)

图中实线表示数据流，回到学生的箭头表示更新后重新生成；教师只评分，不被这个基础流程更新。Teacher-forcing读入学生的整段序列，因果注意力保证第$t$个目标的概率只依赖$h_t$。教师可以并行计算多个位置的条件分布，不需要自回归地重新写一条标准答案。图不表达GPU配比、实际耗时或多轮环境的完整状态。

一次迭代至少包含：采样任务；学生生成并记录token、真实输入和行为概率；教师在对应前缀上评分；learner重算当前学生概率；构造损失、mask与全局分母；只更新学生；同步采样权重；继续下一轮。

Teacher冻结表示参数不更新，不表示它对所有前缀给出相同分数。只缓存prompt ID而不缓存学生后续历史，会把不同条件的教师概率错误复用。

建议每条轨迹记录以下信息：

| 字段 | 为什么需要 |
| --- | --- |
| 原始任务与模板版本 | 确保概率是在同样条件下比较 |
| 输入token、回答token、角色与可见历史 | 重建真实前缀、检查未来信息泄漏 |
| 行为模型版本、实际采样配置、行为logprob | 区分当前策略与数据生成策略 |
| 教师模型/词表/温度版本及payload | 确认监督对象、候选集和概率含义 |
| 模型动作mask、padding mask、终止原因 | 防止把工具观察或人工截断当采样动作 |
| 任务结果、工具失败与预算信息 | 区分模型失败、环境失败与硬截断 |

## 6. OPD目标：训练状态和匹配目标分开定义

令$d_b(h)$表示行为策略产生并按所选权重采集的前缀分布。本轮固定这些前缀，定义：

$$
\mathcal L_D(\theta;b)=\mathbb E_{h\sim d_b}\big[D_h(\pi_T,\pi_\theta)\big].
$$

$D_h$可以选择FKL、RKL或其他散度。这个目标是在学生访问的前缀上，减小教师与学生的条件分布差异；它不是自动最大化准确率，也不是默认对$d_b$求导。采样策略与当前学生一致时称严格的当前on-policy；使用近期版本时需要说明“近期学生数据”而非假装完全一致。

对于一个固定批次，常见token平均经验损失是：

$$
\widehat{\mathcal L}_D(\theta)=
\frac{\sum_{i=1}^{B}\sum_{t=1}^{T_i}m_{it}D_{it}(\pi_T,\pi_\theta)}{N},
\qquad N=\sum_{i,t}m_{it}>0.
$$

这里$i$为轨迹索引，$t$为其回答位置；$m_{it}$规定哪些模型动作参与训练。损失对$\theta$求导，token ID、教师参数和本轮mask固定。$N=0$时应跳过该批次或明确处理，不应以分母夹成1掩盖系统性空监督。

GKD可进一步混合离线示范与学生轨迹：

$$
\mathcal L_{\mathrm{mix}}=(1-\lambda)\mathcal L_{\mathrm{offline}}+\lambda\mathcal L_{\mathrm{on}},\qquad 0\leq\lambda\leq1.
$$

$\lambda$控制学生轨迹监督比例；两项各自采用什么散度和归一化仍需说明。它不是“教师信任度”的通用定义。[GKD方法与算法](https://arxiv.org/html/2306.13649v3)给出数据分布和散度可分别选择的框架。

## 7. Forward KL：目标、推导、梯度与边界

### 7.1 公式具体在优化什么

固定前缀$h$，教师为$q$、学生为$p$：

$$
\boxed{\ell_F=D_{\mathrm{KL}}(q\|p)=\sum_{v\in\mathcal V}q_v\log\frac{q_v}{p_v}.}
$$

该目标按教师概率加权。教师给某个token较大概率、学生给得很小，会产生较大惩罚；它推动学生覆盖教师认为重要的选择。相同条件下KL不小于0，且在两个分布相同时为0；当$q_v>0,p_v=0$时FKL无穷。某一个求和项可以为负，这不违反完整KL非负。

把式子拆开：

$$
\ell_F=\underbrace{-\sum_vq_v\log p_v}_{H(q,p):\,\text{交叉熵}}
-\underbrace{\left(-\sum_vq_v\log q_v\right)}_{H(q):\,\text{教师熵}}.
$$

$H(q,p)$度量学生对教师分布的编码代价，$H(q)$在固定教师、固定前缀下不依赖$\theta$。因此最小化FKL与最小化教师软标签交叉熵具有相同学生梯度；二者数值并不相等。

### 7.2 为什么logit梯度是学生减教师

温度为1时，$\partial\log p_v/\partial z_u=\mathbf1[v=u]-p_u$，其中$\mathbf1[\cdot]$为指示函数。代入交叉熵：

$$
\frac{\partial\ell_F}{\partial z_u}
=-\sum_vq_v(\mathbf1[v=u]-p_u)=p_u-q_u.
$$

**目标含义：学生分配过多概率的方向被压低，教师认可但学生遗漏的方向被提高。**这个结论针对完整归一化的条件分布，依赖温度和目标定义；top-k重归一后的梯度需要重新推导。

### 7.3 教师采样为什么能估计这个梯度

从教师分布抽取目标$a\sim q$，最小化$-\log p_a$，其期望梯度为$-\sum_aq_a\nabla_\theta\log p_a=\nabla_\theta\ell_F$。这里前缀仍可来自学生；“学生采前缀”和“教师采监督token”是两层不同的采样。

若只保留成功教师答案，目标分布通常已变成条件化后的教师数据分布，不再是原始$q$；这种筛选可能有价值，但需要如实定义。

## 8. Reverse KL：目标、推导与策略梯度联系

### 8.1 为什么按学生概率加权

$$
\boxed{\ell_R=D_{\mathrm{KL}}(p\|q)=\sum_vp_v\log\frac{p_v}{q_v}.}
$$

学生当前经常选择、教师概率很低的方向，在该目标中权重较大。教师有合理token但学生概率极小时，该token对当前求和的直接贡献也很小。这是mode-seeking直觉的重要来源；不等于学生永远学不到低概率token，也不等于所选模式一定正确。

有限容量、任务约束和优化误差会导致FKL与RKL的实际行为不同；当学生能完整匹配教师时，两者全局最优解都是$p=q$。教师概率为0而学生为正时，RKL无穷；把top-k外教师概率直接当0会人为制造这个问题。

### 8.2 一步一步推导参数梯度

教师冻结，固定前缀：

$$
\begin{aligned}
\nabla_\theta\ell_R
&=\sum_v\nabla_\theta p_v\left(\log\frac{p_v}{q_v}+1\right)\\
&=\sum_vp_v\left(\log\frac{p_v}{q_v}+1\right)\nabla_\theta\log p_v.
\end{aligned}
$$

利用$\sum_vp_v\nabla\log p_v=\sum_v\nabla p_v=\nabla1=0$，得到：

$$
\boxed{\nabla_\theta\ell_R=\mathbb E_{a\sim p}\left[\log\frac{p_a}{q_a}\nabla_\theta\log p_a\right].}
$$

式中$\nabla\log p_a$叫score-function项；它表示改变采样动作概率的方向，不是任务评分器的“score”。log-ratio决定更新强弱。这正是用学生采样动作估计局部RKL梯度的基础。

对应的学生logit梯度为：

$$
\frac{\partial\ell_R}{\partial z_u}
=p_u\left(\log\frac{p_u}{q_u}-\ell_R\right).
$$

因此RKL的完整期望梯度并非简单逐token比较$p_u-q_u$，也不能从某个样本权重的正负直接推断所有logit最终都怎样变化。

### 8.3 与教师奖励和熵的关系

$$
-\ell_R=\mathbb E_{a\sim p}[\log q_a]+H(p),\qquad H(p)=-\sum_vp_v\log p_v.
$$

最小化RKL等价于在当前状态最大化教师logprob加学生熵。教师概率提供偏好，学生熵避免目标退化成只找教师最高概率动作。若额外增加或删除熵项、改变系数，就要重新说明最优目标，不能仍称完全相同的RKL。

这也是OPD可借用策略梯度实现的原因，但OPD不要求必须采用PPO、GRPO或critic。常见的即时token权重不是通过价值网络估计出来的任务优势。[MiniLLM](https://arxiv.org/html/2306.08543v6)和[作者OPD实现说明](https://thinkingmachines.ai/blog/on-policy-distillation/)从不同配方讨论这种联系。

### 8.4 JSD是另一类可选散度

令$\alpha\in(0,1)$，混合分布$u_v=\alpha q_v+(1-\alpha)p_v$，定义：

$$
\operatorname{JSD}_{\alpha}(q,p)=\alpha D_{\mathrm{KL}}(q\|u)+(1-\alpha)D_{\mathrm{KL}}(p\|u).
$$

$u$是教师和学生的概率混合，不是一个新训练模型；$\alpha$控制这个定义中的混合比例。两侧都和混合分布比较，因而在原分布部分不重叠时也可以避免直接互比造成的某些无穷项。它表达另一种匹配折中，不意味着具有任意任务上的最优稳定性。

JSD原式在$\alpha=0$或1时退化为0，不能直接把端点写成FKL/RKL。在支持和极限条件允许时，$\operatorname{JSD}_{\alpha}/\alpha$随$\alpha\to0$趋于FKL，$\operatorname{JSD}_{\alpha}/(1-\alpha)$随$\alpha\to1$趋于RKL。有些训练库为端点单独分派KL实现，参数名也未必采用本文方向；配置的beta值应回到实际公式核对。

## 9. Sampled-token：数值估计、梯度估计与surrogate

### 9.1 采到一个token后可以估计什么

若$a\sim p$，则$k_1(a)=\log p_a-\log q_a$的期望等于$\ell_R$。单样本可以小于0，因为某个token可能满足$p_a<q_a$；只有完整KL或其期望保证非负。若不随机抽样而只取argmax，这个无偏性不再成立。

已经从$p$抽样后，期望本身已包含$p$权重，不能再无理由乘一次$p_a$。否则期望变成$\sum_ap_a^2\log(p_a/q_a)$，通常是另一个量。

### 9.2 为什么直接backward可能错

采样ID作为固定索引，直接对$k_1(a)$反传，只得到$\nabla\log p_a$；其期望为0。这遗漏了“采样概率也依赖参数”的score-function贡献。**一个标量是无偏KL值估计，并不保证对这个标量自动求导后还是无偏KL梯度。**

一般地，当被平均的函数$f_\theta(a)$和采样分布都依赖参数时，在可交换求导与求和等条件下：

$$
\nabla_\theta\mathbb E_{a\sim p_\theta}[f_\theta(a)]
=\mathbb E_{a\sim p_\theta}\left[\nabla_\theta f_\theta(a)+f_\theta(a)\nabla_\theta\log p_\theta(a)\right].
$$

第一项是“这个动作的函数值怎样变化”，第二项是“这个动作被抽到的概率怎样变化”。普通索引取值的自动求导只保留第一项。代入$f_\theta=\log p_\theta-\log q$后，第一项的期望为0，第二项恰好给出第8节的RKL梯度；这解释了为什么需要PG代理或经过推导的其他路径，而不是简单对采样log-ratio调用`backward()`。

当前采样参数为$\bar\theta$时，可用局部代理损失：

$$
A_b(a,h)=\log\pi_T(a\mid h)-\log\pi_{\bar\theta}(a\mid h),
\qquad
\ell_{\mathrm{PG}}=-\operatorname{sg}(A_b)\log\pi_\theta(a\mid h).
$$

在$\theta=\bar\theta$、动作按同一学生分布采样、前缀固定时，其梯度期望匹配局部RKL梯度。$A_b>0$时单样本更新鼓励该动作，$A_b<0$时抑制；不意味着正确token全部为正或错误token全部为负。

### 9.3 K1、K2、K3必须同时看前向值和反向路径

下面只采用本文定义。令$r(a)=\log p_a-\log q_a$，动作从当前$p$抽取，自动微分时token索引固定：

| 名称 | 本文采用的标量 | 期望与自动微分的区别 |
| --- | --- | --- |
| K1 | $r$ | 期望是RKL；裸反传的期望梯度为0，需正确PG等处理 |
| K2 | $r^2/2$ | 数值一般不是精确RKL；在这里裸反传的期望梯度却匹配局部RKL |
| K3 | $\exp(-r)+r-1$ | 在完整支持下期望是RKL；这里裸反传的期望梯度匹配FKL，而非RKL |

K2结论来自$\nabla(r^2/2)=r\nabla\log p$；K3来自$\mathbb E_p[(1-q/p)\nabla\log p]=-\mathbb E_q\nabla\log p$。这些是给定公式、给定停止梯度和采样条件下的代数结论；若库里使用相反log-ratio、额外detach、IS、PPO clip或不同提案分布，结果要重新核对。不能只看配置名就判断目标。本文CPU脚本同时检查这些值和梯度。

### 9.4 学生采样如何用于FKL

动作$a\sim b$，且教师支持被$b$覆盖时：

$$
\nabla_\theta\ell_F
=-\mathbb E_{a\sim b}\left[\frac{q_a}{b_a}\nabla_\theta\log p_a\right].
$$

对应教学代理为$-\operatorname{sg}(q_a/b_a)\log p_a$。$q_a/b_a$是重要性权重，不是再乘一次学生概率。教师偏好某动作而行为几乎不采它时，权重可能很大，产生高方差；行为top-k令$b_a=0$后，普通IS也无法恢复这个未采动作的贡献。

### 9.5 一个采样token不等于只更新一个logit

固定当前前缀，在$p=b$的参数点，记$r(a)=\log p_a-\log q_a$，一次正确sampled RKL更新对任意学生logit $z_u$的梯度估计为：

$$
\widehat g_u(a)=\operatorname{sg}(r(a))\big(\mathbf1[u=a]-p_u\big).
$$

即使只查询了一个教师概率$q_a$，`log_softmax`的分母也包含其他学生token，所以通常整个logit向量都有梯度。**缺少的是其他token各自的教师概率信息，不是其他logit一定没有梯度。**不同未采token在这次更新中通过归一化受到耦合影响，无法因此分别知道教师更喜欢哪一个。

例：学生采到EOS且$p_{\mathrm{EOS}}>q_{\mathrm{EOS}}$时，$r(\mathrm{EOS})>0$，会压低EOS并相对抬高其他方向；但一次更新不能仅凭$q_{\mathrm{EOS}}$区分其他方向中谁正确。第10.6节给出具体梯度数值。

### 9.6 多采几次、全词表和方差是什么关系

对**同一个固定前缀**独立采$R$次动作，令$\widehat g^{(j)}$为第$j$次的无偏局部梯度估计，则$R^{-1}\sum_{j=1}^R\widehat g^{(j)}$仍无偏，有限方差时各分量的方差是单次的$1/R$。独立性与同条件是这里的前提；一条回答的相邻位置有不同前缀，不能当成同一前缀的独立重复抽样。

全词表直接求和相当于把这一层动作期望精确算完，去掉的是给定前缀下的动作采样方差。任务抽样、前缀覆盖、随机优化和教师误差仍存在。因而“sampled也能无偏”不等于有限样本与full的每步更新相同；“full的信息更完整”也不能直接推出任意预算下最终准确率必然最高。

## 10. 全词表、top-k与尾部质量：三种不同近似

先固定一个学生已经访问到的前缀$h$。假设这一步实际生成的是token B，而词表还允许A、C和EOS。所谓“全词表”是在这个位置比较A/B/C/EOS的所有**可能性**，不是要求学生把四个token都生成出来；“top-k”选其中$k$个候选；“sampled-token”通常只保留实际采到的B的师生分数。

| 对照维度 | Sampled-token | Top-k | 全词表 |
| --- | --- | --- | --- |
| 每个位置向教师索取什么 | 实际采样动作的教师logprob | 公共候选ID及其教师logprobs；tail还需总质量语义 | 全部V个token的分布 |
| 常见教师payload形状 | `[B,L]` | IDs与logprobs为`[B,L,k]` | `[B,L,V]` |
| 损失使用哪些动作 | 一个或若干随机样本 | 候选集，可能另加tail/随机校正项 | 对所有动作求和 |
| 与原完整局部KL的关系 | 正确估计器、支持和采样条件下，梯度可无偏 | 普通确定性截断/重归一通常改变目标；特定随机校正可恢复梯度 | 直接计算所选完整条件KL |
| 主要代价或风险 | 有采样方差；低概率有效动作难观察 | 候选偏差、质量丢失、EOS与尾部盲区 | logits存储/传输、损失计算与反向成本 |

这里“监督多少个token”说的是**每个位置沿词表这一维**有多少概率项；不是只训练回答中的一个位置，也不是只训练top-k个回答位置。只要mask允许，sampled方法同样可以为回答的每个生成位置提供信号。FKL/RKL是另外一条设计轴：不要把sampled固定等同RKL、top-k固定等同FKL；它们只是存在常见组合。

### 10.1 全词表完整在哪里

固定前缀上全词表KL对$V$个token精确求和，提供相应条件分布的完整梯度。它没有消除轨迹采样误差、未访问状态、教师偏差或学生容量限制。

朴素logits张量大小为$[B,L,V]$，其中$L$为批次补齐后的序列长度。仅一个BF16 logits张量约占$2BLV$字节。例如$B=1,L=4096,V=150000$时约1.229 GB（约1.144 GiB），尚未计算另一模型、梯度、激活与缓冲。分块、融合、分布式词表和按需传输可以减少峰值，因此这只是朴素物化张量的大小，不是所有实现必须满足的峰值显存下界。

### 10.2 Top-k集合怎样选择

教师top-k保留教师偏好的方向；学生top-k覆盖学生当前主要选择，包含潜在错误；并集保留两边，交集可能进一步丢掉分歧。集合大小$k$应与概率覆盖一起报告；同一个$k$在高熵和低熵位置提供的信息不同。Top-k candidate selection与生成时的top-k sampling是两个独立设置。

两边必须按同一个token ID集合取概率。分别取各自第1到第$k$名再按排名相减，会把不同token错配。教师API只返回自己的top-k时，不能假设自动得到了学生top-k上所有教师分数。

### 10.3 三种处理方法不能混称“top-k KL”

**方法A：直接保留部分项。**记$\ell_{F,S}^{\mathrm{partial}}=\sum_{v\in S}q_v\log(q_v/p_v)$，它是完整FKL的一部分，不一定非负。把负值截成0、再按token裁剪，又会改变目标与梯度。

**方法B：子集内重新归一化。**定义$\tilde p_v=p_v/P_S$、$\tilde q_v=q_v/Q_S$后计算$\ell_{F,S}^{\mathrm{renorm}}=D_{\mathrm{KL}}(\tilde q\|\tilde p)$，得到候选集条件分布的合法KL，但忽略$P_S,Q_S$的不同。RKL则交换两侧身份。该方法可有很小的损失而学生把绝大多数概率放在集合外。

**方法C：保留一个tail类别。**将集合外所有token合并为一个事件，FKL为：

$$
\ell_{F,S+\mathrm{tail}}
=\sum_{v\in S}q_v\log\frac{q_v}{p_v}
+(1-Q_S)\log\frac{1-Q_S}{1-P_S}.
$$

RKL则交换学生与教师的加权身份：

$$
\ell_{R,S+\mathrm{tail}}
=\sum_{v\in S}p_v\log\frac{p_v}{q_v}
+(1-P_S)\log\frac{1-P_S}{1-Q_S}.
$$

方法C约束了尾部总质量，但仍不知道尾部内部各token的分配，不等于全词表目标。假设两侧尾部质量均为正，令$p^{\mathrm{tail}}_v=p_v/(1-P_S)$、$q^{\mathrm{tail}}_v=q_v/(1-Q_S)$，对$v\notin S$定义，则：

$$
\ell_F=\ell_{F,S+\mathrm{tail}}+(1-Q_S)D_{\mathrm{KL}}(q^{\mathrm{tail}}\|p^{\mathrm{tail}}).
$$

这条式子明确写出了合并尾部后遗漏的非负项。零尾部质量时应按极限和支持条件处理，不直接除以0。

还可以把方法C写成两个容易理解的任务。令$\operatorname{Ber}(a)$表示“在$S$内”的概率为$a$、在集合外的概率为$1-a$的二元分布，则在头部和尾部质量均非零时：

$$
\ell_{F,S+\mathrm{tail}}
=D_{\mathrm{KL}}(\operatorname{Ber}(Q_S)\|\operatorname{Ber}(P_S))
+Q_S D_{\mathrm{KL}}(\tilde q\|\tilde p).
$$

第一项比较“多少概率放进候选集”，第二项比较“放进来之后怎样分配”，并按教师头部质量加权。RKL对应交换两侧，头部条件KL按$P_S$加权。只做方法B时，恰好没有第一项，也没有这里的质量加权；这比笼统说“top-k丢了些token”更精确。

### 10.4 用提前EOS反例理解为什么尾部重要

| 下一个token | 教师$q$ | 学生$p$ |
| --- | --- | --- |
| 继续A | 0.45 | 0.05 |
| 继续B | 0.45 | 0.05 |
| EOS | 0.10 | 0.90 |

仅取教师top-2的A、B，双方归一化后都是$(0.5,0.5)$，子集KL为0；完整FKL为$0.8\log9\approx1.757780$。学生仍有90%的概率过早结束。若EOS可能决定回答完整性，宜单独保留它；tail可以约束集合外质量，但并不总能区分“结束”和其他尾部动作。

![本文示意：子集归一化隐藏EOS质量差异](./assets/learning/opd-interview-foundations/opd-topk-tail.svg)

图用同一总宽表示总概率1。上两行完整分布有明显差异，下两行只保留A/B并归一化后相同。箭头表示近似时被丢掉的信息，不表示真实训练一定会提前EOS。$k=1$时，若两边都只把同一个候选归一化成1且没有tail或其他项，KL恒为0；真实库的“top-1”可能采用采样代理或二元tail目标，应读具体实现。

这个三token例子中，集合外**恰好只有EOS一个token**，因此加上tail就恢复了完整分布；这是一种特例。真实词表的tail通常包含很多token，第10.6节用两个尾部token展示仍然遗漏的信息。

### 10.5 无偏尾部校正与工程成本

一种数学思路是对$S$内求和精确计算，对尾部用保留完整支持的提案分布$r$采样。对任意求和项$f_\theta(v)$，

$$
\sum_{v\in\mathcal V}f_\theta(v)
=\sum_{v\in S}f_\theta(v)
+\mathbb E_{a\sim r}\left[\frac{\mathbf1[a\notin S]f_\theta(a)}{r(a)}\right].
$$

这是数值分解；要构造无偏梯度，还必须正确处理$r$、$f_\theta$的可微路径和停止梯度，不能随意自动微分后沿用值估计结论。TT-OPD研究了利用学生采样token校正top-k梯度的具体构造；tail bucket则是另一个粗粒度目标。[无偏Top-k论文](https://arxiv.org/html/2609.34447v2)、[Tail-Aware论文](https://arxiv.org/html/2608.14728v1)分别支撑这两类问题。

Sampled/top-k主要减少需要保留、传输或聚合的概率项。教师通常仍需前向，学生通常仍需得到归一化概率。不能从$O(k)$与$O(V)$的监督项计数直接推出端到端加速$V/k$倍。

### 10.6 同一个四token算例：每种方法到底算出了什么

在同一个前缀下，令词表为$\{A,B,C,\mathrm{EOS}\}$，教师top-2为$S=\{A,B\}$：

| Token | 教师q | 学生p | 是否在候选集S |
| --- | --- | --- | --- |
| A | 0.30 | 0.10 | 是 |
| B | 0.40 | 0.15 | 是 |
| C | 0.20 | 0.25 | 否 |
| EOS | 0.10 | 0.50 | 否 |

教师集合内质量$Q_S=0.70$，学生$P_S=0.25$；尾部质量分别为0.30和0.75。以下均为自然对数，四舍五入到6位：

| 方法 | FKL或其部分/投影量 | RKL或其部分/投影量 |
| --- | --- | --- |
| 全词表精确求和 | 0.516343 | 0.603519 |
| 只加A/B原始项，不重新归一 | 0.721915 | -0.256986 |
| A/B各自重归一后做KL | 0.001688 | 0.001677 |
| A/B与一个tail类别做KL | 0.447028 | 0.430232 |

这组数值回答四个问题：第一，部分求和可为负，也可比完整KL大，不是完整KL的通用上下界；第二，候选内比例几乎一致，并不代表集合总质量一致；第三，合法tail粗粒度KL不超过对应full KL；第四，C和EOS在尾部的相对比例仍不同，所以这里tail没有恢复full的全部信息。不能把这几种不同目标的loss大小当作训练质量排名。

Sampled不应再填一个“必然等于full”的固定数。若这次从学生采到EOS，K1值是$\log(0.50/0.10)=\log5\approx1.609438$；若采到A，则为$\log(0.10/0.30)\approx-1.098612$。按学生概率对所有抽样结果取期望才得到0.603519。

采到EOS时，正确局部sampled RKL梯度估计按A/B/C/EOS顺序为：

$$
\widehat g(\mathrm{EOS})
=\log5\,(-0.10,-0.15,-0.25,0.50)
\approx(-0.160944,-0.241416,-0.402359,0.804719).
$$

四个分量都非零，但只查询了$q_{\mathrm{EOS}}$。把每种可能动作的梯度按$p$平均，会恢复full RKL梯度；单次EOS更新不等于full更新。第14节的可运行脚本逐项枚举，并另做固定随机种子的10万次采样对照。

### 10.7 候选外的token有没有梯度：必须看目标

固定候选集$S$，温度为1，对FKL三种处理分别求学生logit梯度。令$\tilde p_u=p_u/P_S$、$\tilde q_u=q_u/Q_S$仅在$u\in S$定义：

$$
\frac{\partial\ell_{F,S}^{\mathrm{partial}}}{\partial z_u}
=Q_Sp_u-\mathbf1[u\in S]q_u.
$$

$$
\frac{\partial\ell_{F,S}^{\mathrm{renorm}}}{\partial z_u}
=\begin{cases}\tilde p_u-\tilde q_u,&u\in S,\\0,&u\notin S.\end{cases}
$$

$$
\frac{\partial\ell_{F,S+\mathrm{tail}}}{\partial z_u}
=\begin{cases}p_u-q_u,&u\in S,\\
p_u\left(1-\dfrac{1-Q_S}{1-P_S}\right),&u\notin S.\end{cases}
$$

最后一式的外部项假设$1-P_S>0$；空尾部单独回到full。只做候选内归一时，学生全词表softmax分母会在$p_u/P_S$中抵消，实际只剩候选logits的softmax，故候选外logit直接梯度为0。Partial使用原始归一化概率，分母没有抵消，因此外部logit仍有耦合梯度。Tail进一步比较外部总质量，但不使用外部每个token自己的$q_u$。

在上例中，EOS的FKL logit梯度：full为0.40，partial为0.35，renormalized为0，tail为0.30。**“不在top-k，所以没有梯度”只适用于特定目标，不能泛指所有top-k实现。**这里说的是固定前缀的独立logit梯度；真实网络共享参数，即使某个logit的这一直接导数为0，更新其他位置/方向仍可能使它改变。若由学生选$S$，常见实现将离散top-k索引视为固定，本式不对排名切换求导。

RKL也要区分直接部分求和与截断score代理。若$\ell_{R,S}^{\mathrm{partial}}=\sum_{v\in S}p_v\log(p_v/q_v)$，则：

$$
\frac{\partial\ell_{R,S}^{\mathrm{partial}}}{\partial z_u}
=\mathbf1[u\in S]p_u\left(\log\frac{p_u}{q_u}+1\right)
-p_u\left(\ell_{R,S}^{\mathrm{partial}}+P_S\right).
$$

把完整score期望$\sum_vp_v\log(p_v/q_v)\nabla\log p_v$直接截到$S$，不再自动等于上面部分标量的导数；原先可消去的$\sum_v\nabla p_v=0$只对全词表成立。因此“top-k RKL”尤其需要同时给出目标和detach路径。

### 10.8 从教师payload到实际开销，怎样选择

完整概率满足$\log q_v=z^T_v-\log\sum_u\exp z^T_u$，其中$z^T$为教师logits。只要一个token的准确归一化logprob，常见dense LM也仍需计算全词表归一化；API可以通过融合/分块只返回少量结果，而不是完全不计算其余词表。若API只给top-k的原始logits而不给全词表归一化常数，就不能从这些数字恢复真实$Q_S$和tail；若只给候选内归一化概率，也同样丢失总质量。

工程上先问接口，再选目标：能取完整分布且预算允许，可建立full对照；只能取指定学生动作logprob，可选推导正确的sampled路径；能取教师top-k的全局归一化logprob，可定义partial、renormalized或tail，并报告$P_S,Q_S$；需要学生top-k或并集时，还得让教师能评分这些指定ID。候选top-k的$k$与生成时的top-k/top-p分别记录。

学生侧即使最终只返回$[B,L]$采样logprob，仍可能通过完整输出层、softmax和Transformer反传。因此要分开报告教师前向、返回字节、学生峰值显存、loss归约和端到端耗时；教学代码为了易读先构造完整logits，也不能据此声称实现了生产系统的稀疏通信或省显存优化。

## 11. 局部条件KL、完整序列KL与信用分配

### 11.1 全词表不等于整条轨迹的所有梯度

令$P_\theta(y\mid x)$为学生完整序列分布，$Q(y\mid x)$为教师完整序列分布。两者使用一致的序列样本空间、终止处理和可积条件时：

$$
D_{\mathrm{KL}}(P_\theta\|Q)
=\mathbb E_{y\sim P_\theta}\left[\sum_t\log\frac{p_t(y_t)}{q_t(y_t)}\right]
=\mathbb E_{y\sim P_\theta}\left[\sum_tD_{\mathrm{KL}}(p_t\|q_t)\right].
$$

这是一条目标数值的链式分解。对学生参数求导时，$P_\theta$决定未来访问哪些前缀；把已采历史固定、只优化每个当前条件KL，会省略这部分状态分布影响。GKD的固定轨迹训练明确采用这种选择；不能因此声称它对任意完整序列目标给出了全部导数。

### 11.2 一个没有复杂模型的反例

学生第一步用概率$\alpha=\sigma(\theta)$进入分支A，其中$\sigma(\theta)=1/(1+e^{-\theta})$是sigmoid函数，另一概率进入分支B；教师第一步概率为$(0.5,0.5)$。后续策略固定：A分支上的条件RKL为常数$C>0$，B分支为0。

完整序列目标是：

$$
\mathcal J(\theta)=D_{\mathrm{KL}}((\alpha,1-\alpha)\|(0.5,0.5))+\alpha C.
$$

当$\theta=0$、$\alpha=0.5$时，根节点局部KL梯度为0，但完整目标梯度包含$\alpha(1-\alpha)C=C/4$。原因是改变第一步会改变以后访问高差异分支的机会。本文CPU脚本枚举了这个两步例子。

### 11.3 即时token权重与return-to-go

令$r_t=\log q_t(y_t)-\log p_t(y_t)$。只用当前$r_t$更新当前动作，是局部即时反馈。若希望考虑动作对未来的影响，可以构造$G_t=\sum_{u=t}^{T}\gamma^{u-t}r_u$，其中$\gamma\in[0,1]$是折扣系数；$\gamma=1$是未折扣累计反馈。严格完整序列目标的推导还需检查熵项、参数依赖、终止和采样假设。

局部方法通常更简单、反馈更直接；序列回报更明确地考虑未来，但方差和归因难度可能更大。不能把有teacher token信号等同于已经解决长程信用分配。[MiniLLM的单步与长程分解](https://arxiv.org/html/2306.08543v6)提供方法背景。

### 11.4 如果追问完整序列梯度，可以写到哪一步

令$c_t=\log p_t(y_t)-\log q_t(y_t)$，$s_t=\nabla_\theta\log p_t(y_t)$。在教师冻结、完整支持、自然终止一致、可交换求导与期望等条件下，对第11.1节未归一化的序列RKL，有：

$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(P_\theta\|Q)
&=\mathbb E_{y\sim P_\theta}
\left[\left(\sum_uc_u\right)\left(\sum_ts_t\right)\right]\\
&=\mathbb E_{y\sim P_\theta}
\left[\sum_ts_t\left(\sum_{u=t}^{T}c_u\right)\right].
\end{aligned}
$$

第一行中的直接导数$\mathbb E\nabla\log P_\theta=0$已消去。第二行利用因果性：当前动作的score乘过去已确定的代价，其条件期望为0；保留下当前和未来代价。以第11.3节$r_t=-c_t$表示奖励，就对应$\gamma=1$的return-to-go，而只使用$c_t$会去掉未来信用。任意折扣、长度平均、筛选或人工终止都会要求重新定义目标，不能无条件沿用这一等式。

Forward序列KL的前缀权重又不同：

$$
D_{\mathrm{KL}}(Q\|P_\theta)
=\mathbb E_{y\sim Q}\left[\sum_tD_{\mathrm{KL}}(q_t\|p_t)\right].
$$

这里前缀由教师$Q$访问。OPD在学生前缀上计算FKL依然是一个合理的条件匹配目标，但一般不等于这个教师前缀加权的完整序列FKL。**“每个位置用全词表”描述词表维度的完整性，不决定前缀权重，也不自动给出完整序列目标的导数。**

## 12. 温度、熵、KL系数和归一化分别控制什么

Rollout温度$\tau_r$影响学生探索和实际行为分布；教师温度$\tau_T$改变希望学生匹配的分布；学生损失温度$\tau_s$改变损失中的学生概率。这三者可以不同，但必须在行为分母和推导中如实体现。教师温度改变了目标，不只是“数值稳定技巧”。

经典温度蒸馏有时给软目标损失乘$\tau^2$，用于补偿特定设置下梯度随温度的变化；不能给每一种OPD损失、混合温度和策略梯度方案机械添加该系数。

若损失为$\beta\ell_R$，$\beta>0$控制蒸馏梯度强度，**不自动等于教师可信概率**。另加$\lambda_{\mathrm{ref}}D(\pi_\theta\|\pi_{\mathrm{ref}})$限制策略漂移；教师KL与reference KL的职责不同。单独最大化教师logprob而删除RKL中的学生熵，会变成不同目标，可能更集中在少数教师高概率动作上。

三种归约方式如下，其中$N_i=\sum_tm_{it}>0$：

$$
\mathcal L_{\mathrm{token}}=\frac{\sum_{i,t}m_{it}\ell_{it}}{\sum_iN_i},\quad
\mathcal L_{\mathrm{seqmean}}=\frac1B\sum_i\frac{\sum_tm_{it}\ell_{it}}{N_i},\quad
\mathcal L_{\mathrm{seqsum}}=\frac1B\sum_{i,t}m_{it}\ell_{it}.
$$

Token-mean使长回答提供更多位置的监督；seq-mean先在每条内部平均，给予每条相同权重；seq-sum保留每条累计损失的尺度。它们一般不是相同目标。长度、mask依赖生成时，不应忽略这种设计对训练行为的影响。

梯度累积、多卡DDP、CP/TP分片中应保留全局有效token分母。对长度不同的microbatch先各取mean、再简单平均，通常不能复现完整批次的token平均。若后端默认跨卡平均梯度，还需结合后端缩放约定核对，不能只在每卡分母上做直觉修改。

关于“无偏”的进一步条件：第9节针对固定前缀、固定且不依赖待抽动作的权重成立。若抽到EOS会改变长度，按完成结果筛选会改变保留率，或随机分母$N$与动作相关，就不能只凭单个位置的推导，声称聚合后的sampled梯度与另一套full目标必然严格等价。固定已记录批次求导，与对数据生成、长度和筛选的总体期望求导，是两种不同操作。

## 13. 教师接口：白盒、top-k API和纯文本API

| 教师能力 | 可以提供的监督 | 需要检查 |
| --- | --- | --- |
| 可访问模型参数和完整分布 | 全词表或自行选择支持集 | 词表、模板、位置与显存 |
| 对给定学生token返回logprob | sampled-token目标、指定事件评分 | 是否真在学生前缀上评分、EOS能否评分 |
| 返回全分布归一化后的top-k logprob | 教师top-k近似，可计算残余总质量 | 候选缺失、概率是否已再次归一化 |
| 只返回教师自生成文本及其logprob | 对那些文本的监督 | 不能直接当成任意学生轨迹的评分 |
| 只返回文本 | 学生状态上的教师示范、评价或偏好设计 | 不能宣称拿到了原定义KL |

如果教师API不支持评分指定学生token，而该token又不在返回top-k中，不能把它随意设为0概率或固定极小值；这会引入人为极端奖励。可以扩大payload、追加评分请求、改变估计器或承认目标为部分监督。

全文重编码时必须核对causal shift。对token序列`[PROMPT, A, B, EOS]`，prompt最后一个位置预测A，A所在位置预测B，B所在位置预测EOS；拿A已输入后的logits去评分A，就是错位或泄漏。不同模型模板可以各自合法使用，但对应历史和待比较事件的语义必须一致。

Teacher cache的key至少应能区分教师/词表/模板版本、实际输入前缀、温度和候选事件。教师冻结允许相同条件复用，学生更新后优势中的学生项、学生top-k集合却可能已经变化。缓存命中率不能替代概率等价性检查。

## 14. 最小训练实现：公式、张量与代码逐项对应

以下为本文独立编写的教学实现，不是某框架源码复制。完整可运行文件见[PyTorch监督对照实验](./assets/learning/opd-supervision-lab.py)，实际执行结果见[CPU结果JSON](./assets/learning/opd-supervision-lab-results.json)。只用CPU上的有限维logits演示，不下载模型、不调用教师服务。所有函数共用$p$学生、$q$教师的约定，默认温度1、公共词表和有限logits；实际零支持、跨词表与分布式归约需另外实现。

### 14.1 全词表：先取有效位置，再匹配两个分布

`student_logits`与`teacher_logits`均为`[B,L,V]`，每个row必须已对应相同预测前缀；`action_mask`为布尔`[B,L]`。先保留有效位置后得到`[N,V]`，$N$是本批有效模型动作数。它不是把词表缩成$N$类。第14.5节展示如何从模型原始输出shift成这组输入。

```python
import torch
```

```python
def valid_logprobs(student_logits, teacher_logits, action_mask):
    # [B,L,V], [B,L,V], bool[B,L] -> two [N,V] tensors, N valid actions.
    if student_logits.ndim != 3 or teacher_logits.shape != student_logits.shape:
        raise ValueError("expected matching [B,L,V] logits")
    if action_mask.dtype != torch.bool or action_mask.shape != student_logits.shape[:2]:
        raise ValueError("expected bool[B,L] action_mask")
    if not action_mask.any().item():
        raise ValueError("batch has no valid model action")
    # Select before log_softmax: NaN * 0 is still NaN.
    s = student_logits[action_mask]
    t = teacher_logits.detach()[action_mask]
    if not torch.isfinite(s).all().item() or not torch.isfinite(t).all().item():
        raise ValueError("valid rows require finite, full-support logits")
    dtype = torch.float64 if s.dtype == torch.float64 else torch.float32
    return s.to(dtype).log_softmax(-1), t.to(dtype).log_softmax(-1)


def kl_per_row(logp, logq, direction="reverse"):
    logq = logq.detach()
    if direction == "forward":
        return (logq.exp() * (logq - logp)).sum(-1)
    if direction == "reverse":
        return (logp.exp() * (logp - logq)).sum(-1)
    raise ValueError("direction must be forward or reverse")


def full_vocab_opd(student_logits, teacher_logits, action_mask,
                   direction="reverse"):
    logp, logq = valid_logprobs(student_logits, teacher_logits, action_mask)
    return kl_per_row(logp, logq, direction).mean()
```

`kl_per_row`先沿最后一维词表求和，得到每个有效位置一个值；外层`mean()`沿$N$个位置平均，正好对应第6节token-mean。Full RKL中的`logp.exp()`保持可微，因为概率权重本身依赖学生。Teacher detach不等于学生概率也detach。

这里先选择有效行，再做`log_softmax`；不采用“先在全部位置算完，再乘零”的写法。无效行中的NaN或全负无穷不应进入loss运算图；有效行出现非有限值则报错核查。这个选择只能避免loss阶段的无效运算，不能自动修复模型前向内部已有的NaN。空mask明确报错，由训练循环决定跳批，不能默默制造一个零损失批次。

函数为了梯度验算保留FP64输入；常见BF16/FP16/FP32输入在FP32中计算概率。教师前向仍应在`no_grad()`中执行。该实现未做词表分片、融合内核、packed attention或多卡分母，因此是正确性参考，不是省显存实现。

### 14.2 Sampled：相同动作上的三个logprob与停止梯度

输入都是`[B,L]`：`current_logp`来自当前学生且可微；`behavior_logp`来自生成这批动作的真实行为分布并冻结；`teacher_logq`是教师对同一动作、同一前缀的评分并冻结。

```python
def sampled_opd_pg(current_logp, behavior_logp, teacher_logq, action_mask,
                   signal="behavior"):
    # Inputs: [B,L] on the SAME sampled actions; behavior records actual sampling.
    if (current_logp.shape != behavior_logp.shape
            or current_logp.shape != teacher_logq.shape
            or current_logp.shape != action_mask.shape
            or current_logp.ndim != 2 or action_mask.dtype != torch.bool):
        raise ValueError("expected matching [B,L] tensors and a boolean mask")
    if not action_mask.any().item():
        raise ValueError("batch has no valid model action")
    lp = current_logp[action_mask]
    lb = behavior_logp.detach()[action_mask]
    lq = teacher_logq.detach()[action_mask]
    dtype = torch.float64 if lp.dtype == torch.float64 else torch.float32
    lp, lb, lq = lp.to(dtype), lb.to(dtype), lq.to(dtype)
    if not all(torch.isfinite(v).all().item() for v in (lp, lb, lq)):
        raise ValueError("valid sampled logprobs must be finite")
    if signal == "behavior":
        advantage = (lq - lb).detach()  # Local RKL match only at p == b.
    elif signal == "current":
        advantage = (lq - lp).detach()  # Current local RKL, with full-support IS.
    else:
        raise ValueError("signal must be behavior or current")
    ratio = (lp - lb).exp()  # Keep the numerator differentiable.
    if not torch.isfinite(ratio).all().item():
        raise ValueError("importance ratio overflow; inspect policy mismatch")
    return (-ratio * advantage).mean()
```

`signal="behavior"`复用$A_b=\log q-\log b$，在$p=b$的参数点匹配局部RKL梯度，随后是旧信号surrogate；`signal="current"`使用$A_\theta=\operatorname{sg}(\log q-\log p_\theta)$，在行为支持完整、IS正确且权重固定等条件下修正同一旧前缀上的动作采样。它仍不修正前缀访问分布。两种模式都应让ratio分子的学生路径可微。

前向返回值是代理标量，可能为负，也不是应当作为精确KL报告的数。不能通过比较它和`full_vocab_opd`的loss大小判断训练优劣；应比较预期梯度与独立任务效果。

PPO clipping是额外的近端更新设计，不是所有OPD的定义。Group-relative归一化也不是必需步骤；每题减去由同组采样动作估计的均值或按标准差缩放，可能改变尺度和估计性质。只有给定状态、与当前动作独立的baseline等满足条件时，才能直接使用经典“baseline不改变期望梯度”的结论。

### 14.3 真实代码应怎样对照教学公式

本次核读的[Tinker固定版本](https://github.com/thinking-machines-lab/tinker-cookbook/blob/54b3cb293038ccf2e49952e46ed5abac7119fdf8/tinker_cookbook/distillation/train_on_policy.py)使用学生与教师logprob差构造token信号，并提供未来累计选项；它包含完整训练系统，本文两段教学代码只解释局部loss。

本次核读的[verl固定OPD文档](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/docs/algo/opd.md)区分teacher-top-k FKL、带tail的粗粒度FKL和sampled估计器/PG路径。文档中的配置名、clamp与调用方式属于该commit，不能推广成所有版本或所有框架的约定。静态核读不等于已执行多GPU框架训练。

### 14.4 Top-k：在相同ID上实现三种不同目标

这里复用第14.1节返回的`logp, logq`，形状均为`[N,V]`。为了能对照所有模式，教学函数拥有完整教师分布；生产API只返回top-k时，要用真实payload接入，不能先假定能访问教师遗漏的每个token。

```python
def topk_opd(logp, logq, k, direction="forward", mode="tail"):
    # Already normalized logprobs [N,V], common vocabulary, finite full support.
    # This reference has full teacher logits; a real API may only supply q on S.
    if logp.ndim != 2 or logq.shape != logp.shape:
        raise ValueError("expected matching [N,V] logprobs")
    if not 1 <= k <= logp.shape[-1]:
        raise ValueError("k must be between 1 and V")
    logq = logq.detach()
    ids = logq.topk(k, dim=-1).indices  # SAME token IDs for both models.
    lp = logp.gather(-1, ids)
    lq = logq.gather(-1, ids)
    if mode == "renormalized":
        lp = lp - lp.logsumexp(-1, keepdim=True)
        lq = lq - lq.logsumexp(-1, keepdim=True)
    elif mode == "tail":
        if k < logp.shape[-1]:
            selected = torch.zeros_like(logp, dtype=torch.bool)
            selected.scatter_(-1, ids, True)
            # Stable log tail; avoids subtracting an almost-one head mass.
            pt = logp.masked_fill(selected, -torch.inf).logsumexp(-1, keepdim=True)
            qt = logq.masked_fill(selected, -torch.inf).logsumexp(-1, keepdim=True)
            lp, lq = torch.cat((lp, pt), -1), torch.cat((lq, qt), -1)
    elif mode != "partial":
        raise ValueError("mode must be partial, renormalized or tail")
    # In partial mode these are incomplete sums, not normalized KLs.
    return kl_per_row(lp, lq, direction).mean()
```

三种调用分别为`mode="partial"`、`"renormalized"`、`"tail"`，并可选择FKL/RKL。`gather`按教师选出的同一批token ID取学生分数，避免按排名错配。Tail模式在log空间对集合外求和，减少$1-P_S$在$P_S$接近1时的相消；$k=V$直接使用全词表，不构造一个零概率的假tail。

若教师只返回全局归一化后的top-k logprob，则其tail由$\log(1-Q_S)$确定，宜用经过验证的`log1mexp`等稳定实现；只有候选内归一化分数时不能恢复它。上面的代码显式访问全词表尾部，便于验证公式，不宣称已经实现top-k API的通信收益。

还需特别区分一种错误接法：先算full/top-k分布KL，把**整个位置的一个KL标量**detach成负奖励，再只乘该位置已采动作的`logp`。若这个标量只依赖固定前缀而不依赖抽到的动作，它相当于状态常数，当前策略下score期望为0，并不能恢复该分布KL的梯度。分布损失应直接反传，或按逐动作贡献推导合法估计器；不是所有叫“KL loss”的张量都能互换进PG接口。

### 14.5 训练循环伪代码：从rollout接到shift与backward

下面是一轮一更新的同步伪代码；`sync_rollout_weights`和`collect_rollouts`是由实际框架实现的抽象接口，不是可直接调用的库函数。假设教师/学生输入模板与词表相同、右padding、普通文本轨迹，生成温度1且没有top-k/top-p截断或其他logit处理。三条loss分支共用同一批学生前缀，**每轮只选择一条分支**。

Collector返回的`input_ids/attention_mask/response_mask/behavior_logp`均为`[B,M]`。`response_mask`标注真实生成的模型动作，包含自然EOS，不包括prompt、padding和人工补EOS。`behavior_logp[:,j]`记录`input_ids[:,j]`这个动作在其生成前缀下的实际采样logprob；非动作位置可放NaN，并在取值前排除。

```python
mode = "full"  # Choose "full", "topk_tail" or "sampled".
teacher.eval().requires_grad_(False)
student.eval()  # eval disables dropout, NOT autograd.

for prompts in prompt_loader:
    sync_rollout_weights(student)
    with torch.no_grad():
        batch = collect_rollouts(prompts, temperature=1.0,
                                 top_p=1.0, top_k=None)

    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    labels = ids[:, 1:]  # logits at j-1 predict token j.
    mask = batch["response_mask"][:, 1:] & attn[:, 1:].bool()

    optimizer.zero_grad()
    with torch.no_grad():
        teacher_logits = teacher(ids, attention_mask=attn).logits[:, :-1, :]
    student_logits = student(ids, attention_mask=attn).logits[:, :-1, :]
    logp, logq = valid_logprobs(student_logits, teacher_logits, mask)

    if mode == "full":
        loss = kl_per_row(logp, logq, direction="reverse").mean()
    elif mode == "topk_tail":
        loss = topk_opd(logp, logq, k=2,
                        direction="reverse", mode="tail")
    elif mode == "sampled":
        actions = labels[mask].unsqueeze(-1)  # [N,1], actual sampled IDs.
        current = logp.gather(-1, actions).squeeze(-1)[None, :]  # [1,N]
        teacher_action = logq.gather(-1, actions).squeeze(-1)[None, :]
        behavior = batch["behavior_logp"][:, 1:][mask][None, :]
        loss = sampled_opd_pg(current, behavior, teacher_action,
                             torch.ones_like(current, dtype=torch.bool),
                             signal="current")
    else:
        raise ValueError("unknown supervision mode")

    loss.backward()
    optimizer.step()
    # Next iteration resamples after synchronizing the updated student.
```

`[:, :-1]`去掉最后一个没有后继目标的位置，`[:, 1:]`去掉第一个没有本序列内预测行的目标；prompt最后一个位置因而可以预测回答首token。动作mask跟着**目标token**移动。即使EOS与padding恰好共用ID，也不能只凭`ids != pad_id`决定动作mask，应保留真实长度、角色与停止原因。

这份伪代码为了对照三种方法统一算全词表，sampled分支并未因此省下教师payload；真实系统应在教师接口处只请求所需动作分数，并在学生侧按内核能力只保留所需中间量。多轮Agent、不同模板、TP/CP/DDP、混合任务奖励与多次复用旧batch，需要另加相应条件和归约，而不是无条件套这段循环。

### 14.6 使用PyTorch KL接口，方向和mean怎样核对

对已选出的`[N,V]`有效位置，使用`log_target=True`时：

```python
import torch.nn.functional as F

# Both tensors are logprobs. Freeze the teacher, keep the student differentiable.
fkl_per_row = F.kl_div(logp, logq.detach(),
                       reduction="none", log_target=True).sum(-1)
rkl_per_row = F.kl_div(logq.detach(), logp,
                       reduction="none", log_target=True).sum(-1)
fkl = fkl_per_row.mean()
rkl = rkl_per_row.mean()
```

这个接口计算`target.exp() * (target - input)`，所以`input=logp, target=logq`是FKL，交换后才是RKL；RKL中的student处在target参数也必须保留梯度。对于`[N,V]`，直接`reduction="mean"`会额外除以$V$；`batchmean`在这里除以$N$，但如果仍传`[B,L,V]`，它除以的是$B$，不是有效token数。显式`none → 词表sum → 有效位置mean`更容易核查。接口语义见[PyTorch 2.11官方文档](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html)，本轮在实际安装的PyTorch 2.11.0中核对方向、数值与梯度。

### 14.7 读代码时的逐项自检

先打印一个有效位置的token ID、真实前缀、教师/当前/行为logprob；再检查概率归一化、候选ID对齐和head/tail质量；然后在有限词表上对照loss值与学生梯度，确认teacher与behavior不收梯度。最后检查mask/shift、空批、旧策略修正、全局分母，以及训练后真实任务指标。

尤其不要把“loss是一个标量”“张量形状能相减”“backward没有报错”当成目标正确的证据。本节下载脚本用独立解析梯度、有限差分、枚举与采样分别验证这些问题；EOS演示只优化三个独立logit，不表示已经训练语言模型修好了截断。

## 15. 异步、旧轨迹和重要性采样

### 15.1 旧动作分布和旧状态分布是两件事

对一个固定旧前缀$h$，动作$a\sim b(\cdot\mid h)$，若$b$覆盖当前学生$p_\theta$的支持：

$$
\nabla_\theta D_{\mathrm{KL}}(p_\theta\|q)
=\mathbb E_{a\sim b}\left[
\underbrace{\frac{p_\theta(a\mid h)}{b(a\mid h)}}_{\rho_\theta}
\log\frac{p_\theta(a\mid h)}{q(a\mid h)}
\nabla_\theta\log p_\theta(a\mid h)\right].
$$

这可以修正**同一前缀内的动作抽样**。若这些前缀本身来自$d_b$而目标希望是$d_{p_\theta}$，还存在状态访问分布差异；局部动作ratio不会自动把旧历史变成新历史。多轮工具执行、跨版本上下文构造会进一步扩大这个区别。

### 15.2 为什么只加ratio仍可能不够

如果继续复用旧$A_b=\log q-\log b$，而非当前$\log q-\log p_\theta$，则不再是上式中的当前log-ratio。再加clip、截断、sample mask或分数clamp后，更新可能成为有意设计的近似。稳定性约束可以有价值，但需要说清偏差与收益。

版本落后几步不是可靠的概率距离；一步大更新可能比多步小更新差异更大。应监控ratio分布、有效样本量、clip比例、数据年龄、长任务保留率及真实质量。对非负IS权重$w_j$，常用诊断$\mathrm{ESS}=(\sum_jw_j)^2/\sum_jw_j^2$；它衡量权重集中程度，不是整条训练无偏的证明。

[AsyncOPD](https://arxiv.org/abs/2606.24143v1)把旧策略、当前信号与有限教师缓存作为独立问题研究。本文只概括这些问题，不把其特定实验中哪种方案更好外推到任意系统。

## 16. Mask、数据质量与数值稳定

### 16.1 三种mask与一次工具交互

| 可见内容 | Attention可见性 | 通常是否作为学生生成动作计算loss |
| --- | --- | --- |
| 用户问题与系统指令 | 对后续模型位置可见 | 通常不计算动作loss |
| 学生生成的推理、回答、工具参数 | 按因果规则可见 | 按目标计算 |
| 工具返回的搜索结果或执行输出 | 后续模型可见 | 外部观察通常不计算学生动作loss |
| 自然采样的EOS | 可见终止事件 | 若目标包含终止，应正确监督 |
| padding、人工补齐与重建标记 | 按实际attention规则处理 | 不能自动当作真实采样动作 |

Attention mask决定能看什么，loss mask决定什么位置训练，终止/回报mask决定episode如何结束或是否bootstrap。这些对象不等价；把工具观察的loss设为0，也不会自动修复错误的packed attention跨样本可见性。

### 16.2 失败数据不等于全部有害

模型推理错误、格式错误、工具服务超时、上下文损坏、违规输出和预算截断应分别标注。可恢复错误前缀可能正是OPD的训练价值；全删失败会改变访问分布。教师完全不可靠或系统数据损坏的样本则可能需要gate、降权或丢弃。过滤后应报告保留率、任务难度与失败类型覆盖，不能把过滤带来的数据变化归因于loss本身。

### 16.3 NaN、Inf与极端权重

使用FP32的`log_softmax`和归约通常更便于调试；教师冻结和eval模式避免无意更新或不必要的随机性。先检查合法支持、位置对应、空mask与logprob条件，再考虑学习率、梯度裁剪和数值clamp。

`NaN * 0`仍为NaN；已经进入计算图的非法除法/对数，也未必能靠末尾mask消除坏梯度。第14节因而在loss运算前先选有效行，并检查有效行有限。词表中有合法零概率时，应按明确支持和极限专门处理，不能把含负无穷的logprob直接送入所有通用乘法。相关机制见[PyTorch autograd官方说明](https://docs.pytorch.org/docs/2.11/notes/autograd.html)。`eval()`改变dropout等模块行为，不会关闭autograd；`no_grad()`才关闭相应前向的梯度记录。

给0概率加$\epsilon$、截断ratio或裁剪token损失都会改变精确目标。可以作为明确的数值或优化设计，但不应在实现后仍宣称所有原始数学性质原封不动成立。对top-k聚合质量，宜在稳定精度下计算残余概率，区分舍入保护与真实概率重定义。

## 17. 学生复读：什么时候能纠正，什么时候会恶化

如果学生访问重复前缀$h$，教师在该状态下给继续重复较低概率、给恢复或结束较高概率，OPD存在纠偏信号。例如学生$p_{\mathrm{repeat}}=0.8$、教师$q_{\mathrm{repeat}}=0.1$，完整FKL对重复logit的梯度为$0.7$，梯度下降会压低它。

不过，教师从头生成时不复读，不保证在学生已经重复多轮的前缀上也不顺着继续。教师可能更倾向补全既有模式；top-k可能漏掉恢复动作；学生支持过窄、更新过大、坏数据被持续强化，也可能令循环加重。

排查应定位第一次进入重复的前缀，而不只截取末尾长重复段。对同一前缀比较重复token、恢复方向、EOS的师生概率；分别检查完整分布、候选集和tail质量；观察是否有足够正常轨迹以及损失正负/极端权重。然后用小学习率、短更新间隔、SFT起点、可靠前缀监督或明确的异常gate做受控对照。

Repetition penalty、no-repeat n-gram、最小长度等推理设置也会改变实际行为分布。它们可能改善部署表现，但不是OPD学习到恢复的证据；如用于rollout，应记录其对行为概率的影响。

## 18. 学生截断：EOS、长度上限与配置分别处理

### 18.1 提前自然结束

如果学生EOS概率偏高，而教师认为还应该继续，则FKL提供$\partial\ell_F/\partial z_{\mathrm{EOS}}=p_{\mathrm{EOS}}-q_{\mathrm{EOS}}>0$的信号。Sampled RKL也可能通过采到EOS后的负权重抑制它，但受采样、支持与估计方差影响。EOS若被过滤或mask遗漏，监督可能看不见问题。

### 18.2 达到max_new_tokens或上下文上限

这是外部预算终止。OPD不能直接修改上限，也没有见到未生成部分的on-policy状态。它可能减少无效重复，使答案更容易在预算内完成；也可能学到更长推理，反而增加截断。应按任务合理长度调整预算、分段或续写，并分别评价完整性、正确率与成本。

不能一概认为所有部分rollout都没有训练价值：局部OPD本来可以用部分轨迹。但如果混合SFT/CE或终止奖励，不能把人工补EOS、强制截断标记当成自然完成的正标签。应保留真实`finish_reason`与相应mask。

### 18.3 Stop字符串、模板和服务问题

误配stop、EOS ID、思考/回答结束标记、工具协议或服务端超时，会导致看似“学生不会写完”。先检查原始token与停止原因，再决定是否训练。学习方法不能修复推理系统把合法内容切掉的配置错误。

## 19. OPD与reward hacking：理想目标和实际代理分开

Reward hacking指优化的反馈改善，但真实期望的任务质量没有改善甚至恶化。OPD通常没有独立的标量reward model，却仍可能出现教师隐式信号与任务目标偏离。

**理想层面。**对完整、规范化且终止一致的序列分布，$D_{\mathrm{KL}}(P_\theta\|Q)=0$意味着学生分布等于教师分布。不能一边声称匹配已精确完成，一边声称学生仍在同一分布意义上任意偏离教师。教师自身质量则是另一个问题，匹配教师不等于教师永远正确。

**实际层面。**局部固定前缀更新、token平均、top-k投影、有限采样、有限训练数据、clip/mask和终止处理，都可能令监控标量不等于完整序列KL。教师对学生异常前缀的高概率也不等于正确性或长程价值。只最大化教师logprob而丢掉相应学生熵，更是改变了目标。

[2026年10月的OPD坍塌研究](https://arxiv.org/html/2610.03185v1)在其sampled-token、策略梯度实现和数学任务设置中观察到过长与重复行为被教师隐式偏好强化，并研究了异常轨迹mask和SFT初始化。该观察说明具体OPD实现可能出现目标偏离，不证明所有KL方向、估计器、教师和任务都必然发生相同现象。

处理时同时检查独立任务指标和监督诊断：正确率是否下降，重复/长度上限比例是否上升，异常响应是否得到更优权重，top-k/tail与EOS是否失真。筛掉异常轨迹可能缓解强化，也会改变数据分布并可能删掉有价值的恢复前缀，应做保留率与覆盖消融。不能只因KL下降就宣布训练成功。

## 20. 与RL、DPO、SFT怎样选择和组合

| 方法 | 典型数据/状态 | 典型反馈 | 首先需要的能力 |
| --- | --- | --- | --- |
| SFT/教师答案蒸馏 | 固定示范前缀 | 目标token | 高质量示范与充分调参 |
| 离线soft KD | 固定前缀 | 教师条件分布 | 可获得soft target |
| OPD | 学生或近期学生前缀 | 教师分布/对应token信号 | 学生可生成、教师可评价这些前缀 |
| 可验证RL | 学生交互轨迹 | 测试/数学判定等任务奖励 | 可靠验证器与探索预算 |
| DPO | 通常固定chosen/rejected对 | 成对偏好 | 可靠偏好数据与明确参考模型 |

这些是典型用法，不是穷尽定义。RL也可以有稠密过程反馈；DPO也有在线数据构造；SFT数据也可以在学生状态上查询专家后不断更新。

一种可能的组合目标为：

$$
\mathcal L=\lambda_D\mathcal L_{\mathrm{OPD}}
+\lambda_S\mathcal L_{\mathrm{SFT}}
-\lambda_R\mathbb E[R(y,x)]
+\lambda_{\mathrm{ref}}\mathcal L_{\mathrm{reference}},
\qquad \lambda_D,\lambda_S,\lambda_R,\lambda_{\mathrm{ref}}\geq0.
$$

$R$是独立任务回报，负号把最大化回报写成最小化损失。各项的采样来源、梯度估计和归约应分别定义。教师提供局部指导，验证器提供结果约束，两者可能互补，也可能在某些位置产生冲突，不能仅靠把loss加起来保证收益。

更强教师不一定是参数更多的教师：应看目标任务能力、前缀兼容性、概率校准、模板/词表兼容和成本。[Trust Region OPD](https://arxiv.org/html/2606.01249v1)研究了师生分布差异大时不可靠监督与离线引导。离线成功教师SFT与soft KD也应充分调参，不能故意用弱基线突出在线方法。

## 21. 跨tokenizer、多教师、自蒸馏与Agent扩展

### 21.1 跨tokenizer先定义公共事件

不同模型的token ID与分词边界不同，不能直接把两个词表向量按编号做KL。例如同一字符串可能在学生里是一个token，在教师里是三个；应定义共同的文本/byte事件或其他映射，再说明位置、质量守恒、EOS和残余事件。比较公共事件上的投影分布，不等于已经计算原始两个token词表之间不存在的“直接KL”。完整实现见[跨tokenizer专题](#q=opd-interview-cross-tokenizer)。

### 21.2 多教师的算术平均和几何平均不是同一个目标

设有$M$个同样本空间、同前缀的冻结教师$q_j$，权重$w_j\geq0$且$\sum_jw_j=1$，所有教师概率为正。在不受模型表达限制的单个状态上：

$$
\min_p\sum_{j=1}^Mw_jD_{\mathrm{KL}}(q_j\|p)
\quad\Rightarrow\quad p_v^*=\sum_jw_jq_{j,v};
$$

$$
\min_p\sum_{j=1}^Mw_jD_{\mathrm{KL}}(p\|q_j)
\quad\Rightarrow\quad
p_v^*=\frac{\exp(\sum_jw_j\log q_{j,v})}{\sum_u\exp(\sum_jw_j\log q_{j,u})}.
$$

前者是概率算术平均，后者是归一化几何平均。它们可能产生不同的冲突处理：某教师极不认可的动作会受到几何组合较强影响。按任务选择一个教师、随机路由、平均概率、平均logits与给多个loss，是不同设计；跨域配比与遗忘也要单独评价。以上是有限分布的理论最优，不保证共享参数模型达到它。

### 21.3 自蒸馏和特权上下文

教师可以来自同一基础模型、EMA快照，或有额外解题信息的上下文。此时教师不必参数更大。若教师使用学生部署时不可得的解答、检索或工具信息，目标是把这些信息产生的监督迁移给学生；学生输入中不能泄漏这些答案来伪造学习效果。EMA教师会随训练变化，不再满足本文“教师永久固定”的基础假设，需要记录更新规则。

### 21.4 Agentic OPD的三层on-policy

一段响应的token由学生采样，不代表完整环境历史由学生执行。应分别问：谁提出动作，谁真正执行动作推动环境，谁的参数被训练。Teacher-history上的学生局部采样、由fast policy推进环境的长响应监督、真实学生多轮交互，会产生不同的状态分布。

工具输出通常只作为观察而不做模型动作loss；上下文总结、重试、沙箱状态恢复都会改变后续条件。对Code/Search任务还需保留环境版本和独立测试结果。具体流程读[Agentic训练](#q=agentic-opd-training)与[系统Pipeline](#q=agentic-opd-pipeline)。

## 22. 成本、并行与怎样证明方法有效

### 22.1 教师大，为什么学生rollout仍可能最慢

教师通常对已给定文本做批量评分，学生rollout却逐token串行生成；多轮还包含工具等待、重试和重复prefill。因此不能只按参数量估计瓶颈。需要分别统计学生生成、教师评分、learner前反向、参数同步、工具环境、排队与通信的时间和预留GPU成本。

短rollout可降低生成成本，但会改变状态覆盖、终止学习和长程任务代表性。Teacher API缓存、分块logits、异步流水线、TP/CP/EP等是不同层的优化，性能收益须在真实模型、长度、并发和硬件条件下测量。

### 22.2 建立有竞争力的基线

对照至少包括同初始化的无训练学生、充分调参的教师答案SFT、soft KD和所提出OPD配方。固定任务划分、教师、可用数据和评价协议；单独比较同生成token、同教师调用与同GPU-hour预算。成功筛选、SFT学习率和训练步数应独立调优，而非让所有方法机械共享一个不适合各自的超参。

近期多教师蒸馏审计强调了强离线基线与成本比较的重要性；它在特定同源教师/学生设置下的结论不能泛化成“OPD全部无效”。详见[多教师审计原文](https://arxiv.org/abs/2610.04272v1)与[站内精读](#paper=paper-rethink-mopd)。

### 22.3 分开观察任务质量、行为和训练信号

| 组别 | 建议观测 | 能回答的问题 |
| --- | --- | --- |
| 任务结果 | 独立准确率、代码测试、工具任务成功率、OOD | 是否真正解决任务 |
| 采样能力 | 固定温度/预算下pass@1与pass@k、问题覆盖 | 是提高采样效率还是改变可解范围 |
| 行为健康 | 复读率、自然EOS率、截断率、长度分位数 | 是否发生生成退化 |
| 分布诊断 | 完整或明确近似的KL、熵、EOS、head/tail质量 | 学到了哪些概率变化 |
| 更新稳定 | 梯度范数、ratio、clip fraction、ESS、数据年龄 | 近似和旧数据是否失稳 |
| 系统成本 | 总预留GPU-hour、teacher tokens、有效tokens/s、环境耗时 | 性能收益来自哪里 |

pass@1提高不自动证明原来不能解决的问题现在可解；大$k$采样也不能证明一个模型绝对没有某项能力。基于有限测试应报告“在该协议下观察到什么”，不能把具体研究的观察变成所有OPD的能力定理。

消融宜一次隔离一条假设：固定前缀对比FKL/RKL；固定目标对比full/top-k/sampled；固定方法改变tail/EOS；固定训练检查不同起点、teacher与staleness；系统优化单独比较正确性和总成本。不同定义loss的标量不可直接按大小排名，最终回到相同任务指标。

## 23. 从异常现象到诊断动作

| 现象 | 第一轮核对 | 第二轮对照 |
| --- | --- | --- |
| KL一开始异常大或Inf | tokenizer/shift、教师0概率、候选缺失、模板 | 在固定三token分布上检查目标与梯度 |
| loss很小但没有学习 | 裸K1反传、top-1归一化、错误detach、空mask | 检查学生梯度与微小手算例 |
| top-k KL下降但质量降 | head/tail概率、EOS是否遗漏、clamp | full或tail-aware小规模对照 |
| 复读越来越严重 | 首次进入循环的前缀及教师条件概率 | SFT起点、更新幅度、可靠状态gate |
| 回答过早结束 | finish_reason、EOS目标与mask、stop配置 | 显式EOS监督与相同预算评估 |
| 总撞长度上限 | 任务长度、重复、思考预算与模板 | 合理扩预算、长度/质量联合分析 |
| 异步后突然退化 | 行为版本、实际输入、current/old信号与ratio | 限制数据年龄、当前信号重算、同步基线 |
| 多卡或累积后梯度变化 | 全局有效token分母、DDP缩放、mask | 同一合成批次拆分前后梯度一致性 |
| 教师更强但OPD变差 | 学生状态兼容、概率目标与任务质量 | 中间教师、离线冷启动、任务分层评价 |

这些是排查顺序，不是保证有效的超参处方。特别是看到“异常样本很多”时，要先区分数据损坏与可恢复错误，不把全部失败删除后仍称相同访问分布实验。

## 24. 研究脉络、方法选择与学习边界

经典KD解释软目标；DAgger提供学习者状态上询问专家的背景；GKD把学生轨迹与可选散度统一起来；MiniLLM讨论序列RKL与优化；作者OPD实践展示可直接接入token反馈的训练闭环；后续top-k、尾部校正、异步与可信区域研究分别处理信息、估计和系统约束。该脉络用于理解问题，不是完整论文排行榜。

选择方法时，先确认高质量教师示范能否以更低成本满足目标，再判断学生错误前缀上的在线监督是否有额外价值。需要明确任务结果的场景，可以加入可靠验证器；学生初始支持太窄时，可以先做SFT或混合前缀；教师对异常状态不可靠时，需要改监督、加可靠性约束或承认适用范围。

本文的基础结论不依赖“教师一定更大”“RKL一定更好”“在线一定胜过离线”等口号。真正需要说清的是：**前缀由谁访问，教师监督哪个公共事件，优化哪个目标，使用哪个梯度，怎样处理终止与近似，以及如何用独立结果验证。**下面的分层问答沿这一契约组织。


## 25. 面试问答：既有24题与题源账本

### 25.1 前置知识与历史题源


| 来源 | 能确认什么 | 本篇怎样使用 |
| --- | --- | --- |
| [牛客AI Infra面经题解](https://www.nowcoder.com/discuss/933731201871806464?sourceSSR=post) | 页面讨论OPD动机、错误轨迹筛选与项目表达；系公开整理，无法独立核公司真实性 | Q01/Q18的问题方向参考，答案独立推导 |
| [九坤Agent题解章](https://ricardokevins.github.io/notes/llm-interview-question-bank/chapters/091.html) | 整理OPD流程、与SFT区别、API概率可得性 | Q02/Q12题型参考；不宣称公司原题认证 |
| [个人Interview知识页](https://lpq29743.github.io/wiki/interview/) | 公开复习页面含后训练/OPD | 阅读路线补充，不用作技术事实或频次证据 |
| [Thinking Machines作者博客](https://thinkingmachines.ai/blog/on-policy-distillation/) | 介绍学生rollout、教师sampled评分与训练流程 | 教学思路参考，不是面经 |
| [GKD](https://arxiv.org/abs/2306.13649)、[MiniLLM](https://arxiv.org/abs/2306.08543) | 目标与蒸馏原始方法 | 数学与机制依据 |
| [新MOPD审计](https://arxiv.org/abs/2610.04272v1) | 强SFT配方、独立调参与成本边界 | Q22–24为围绕新论文自拟答辩问题 |

以上网页于2026-10-12核对。Q01/Q02/Q12/Q18是从公开题型改写，其余为根据原文、代码和你的项目方向自拟，包含推导题、白板题、排障题；不是照抄一份不明来源的“真题大全”。

约定：$p_\theta$学生、$q$冻结教师、$p_b$行为策略；$h$为固定当前prefix。局部词表KL和完整轨迹目标分开讨论，避免把访问分布导数偷偷丢掉又称严格总梯度。

### 25.2 定义与方法选择

**Q01 [公开题型·动机] 什么是OPD，为什么不直接做SFT？**

先说学生生成自己的轨迹，教师在学生实际prefix上提供监督，learner更新，循环再采样。Teacher SFT在教师记录历史上拟合输出；OPD覆盖学生自己犯错后的前缀，适合分析暴露偏差与恢复。随后限定条件：教师在学生状态上要可用，学生需具有足够起始能力，在线生成和环境成本也要值得。不能回答“OPD必然比SFT强”，新MOPD审计正要求认真筛选与调优SFT。

追问：你项目的增益发生在初始正确率、错误恢复、OOD还是训练效率？哪组消融能定位？

**Q02 [公开题型·比较] OPD、RL、DPO和普通KD的区别？**

至少拆数据来源、反馈、目标三轴。普通离线KD用固定上下文；OPD用学生状态。RL可用环境终局reward，也可有稠密过程反馈，不能定义成永远稀疏；OPD通常用教师概率/动作指导。DPO通常训练固定chosen/rejected偏好pair与reference相对概率，不等于在线学生rollout。OPD可以用PG surrogate实现，也可直接词表KL，所以“用PG代码”不自动证明优化的是任务回报。

**Q03 [自拟·命名] On-policy究竟指哪些分布？**

说明生成token由谁采样、历史状态由谁执行、版本是否落后。Teacher-prefix局部学生生成，token内前缀来自学生但跨turn历史来自教师。ActFirst的full响应来自学生，环境却由fast策略推进；异步数据来自旧学生。不能只说“student生成了”就把所有层都称严格当前部署on-policy。

**Q04 [自拟·教师] Teacher一定要参数更多吗？**

重要的是目标任务能力、学生状态兼容性和反馈质量，不是参数量标签。小任务RL专家可能胜过大未适配基础模型。MoE总参数与active参数影响内存/计算不同。多教师要说明领域路由、概率校准、冲突与各域训练权重；同一缩写MOPD也可能指multi-rollout而非multi-teacher。

### 25.3 概率、梯度与白板题

**Q05 [自拟·公式] 写出FKL/RKL，并说明哪个分布加权。**

$$
D_F(q\|p)=\sum_aq(a)\log\frac{q(a)}{p(a)},\quad
D_R(p\|q)=\sum_ap(a)\log\frac{p(a)}{q(a)}.
$$

FKL按教师概率加权，RKL按学生概率加权；它们都需要同一条件和公共样本空间。Mode-covering/seeking是有用直觉，不是“RKL保证学生只学一个正确答案”的定理。$q(a)=0,p(a)>0$使RKL无穷；模型softmax理论正概率，实际top-k截断/量化/下溢仍会造支持问题。

**Q06 [自拟·梯度] FKL对student logits的梯度是什么？**

固定$q$且词表完整归一时为$p-q$。教师熵与学生参数无关，FKL与教师soft-label cross-entropy差常数。教师token按$q$采样的SFT梯度条件期望也为$p-q$；单样本有方差，成功过滤后采样分布变为$q_{\rm selected}$，不能再说无条件仍等于原$q$监督。

**Q07 [自拟·推导] RKL对参数的梯度怎样写？**

固定prefix与教师：

$$
\nabla D_R=\sum_ap(a)(\log p(a)-\log q(a)+1)\nabla\log p(a)
=\mathbb E_{a\sim p}[(\log p-\log q)\nabla\log p].
$$

最后一步用$\mathbb E_p\nabla\log p=0$去掉常数。它是当前prefix局部梯度；若目标对未来状态占用也求导，还需相应score/return项，不把二者混用。

**Q08 [自拟·陷阱] Sampled KL是无偏值估计，直接backward就对吗？**

不对。采样$a\sim p$后将$a$当固定索引，对$\log p(a)-\log q(a)$直接autograd，梯度期望是$\mathbb E_p\nabla\log p=0$，遗漏采样分布导数。Full-vocab RKL中$p$权重必须有梯度；PG实现可用detach的$(\log q-\log p)$作为优势，配正确score梯度。值估计无偏与自动求导梯度无偏是不同命题。

**Q09 [自拟·代码] 写一个能自动求导的固定prefix full-vocab loss。**

```python
# Reuse the validated teaching function from section 14.1.
# Logits: [B,L,V]; action_mask: bool[B,L], targets already shifted.
loss = full_vocab_opd(student_logits, teacher_logits,
                      action_mask, direction="reverse")
loss.backward()  # teacher is detached; student probabilities remain differentiable.
```

应解释形状、teacher no_grad、稳定计算、shift和有效token分母；第14.1节先选有效行，并对空mask报错，不依赖末尾乘零或分母夹成1隐藏无效批次。这个例子是token-mean，若论文用seq-mean或group-mean要改聚合，不能只说“都取mean一样”。拿两/三token词表检查数值与梯度，再检teacher.grad为空。

**Q10 [自拟·PG] Sampled OPD的一步如何写？**

保存行为logprob$\log p_b$；current forward得$\log p_\theta$；教师同条件评分$\log q$。固定prefix的当前RKL示意采用$A_\theta=\mathrm{sg}(\log q-\log p_\theta)$与$\rho=p_\theta/p_b$，最小化$-\rho A_\theta$。若复用$A_b=\log q-\log p_b$、PPO clip或其他约束，则是具体surrogate，需说明何时近似或改变梯度。不要把所有配置叫同一个严格目标。

**Q11 [自拟·温度] Teacher/student temperature和top-k是一个参数吗？**

不是。Rollout温度/截断控制实际behavior分布，teacher soft distribution温度控制监督，候选top-k控制存储/近似，投影top-k控制映射。$p_{\rm raw}$与top-p重归一后的$p_{\rm sample}$不同。若用raw概率当实际采样denominator要解释偏差/修正；截断支持缺失时普通IS无法恢复未采动作。温度KD的$T^2$缩放与特定目标相关，不能给所有OPD默认机械乘。

### 25.4 教师接口与训练数据

**Q12 [公开题型·API] Teacher API不返回完整logits，还能做OPD吗？**

要问能返回哪些信息。支持给定学生token的logprob，可实现对应sampled目标；只返回top-k，需处理采样token缺失与残余质量；只返回文本，无法直接计算原定义full-vocab KL，但可设计教师在学生状态上生成动作进行SFT、偏好或评价。黑盒反馈可能仍是广义学生状态监督，不能把它说成标准词表KL已被计算。

**Q13 [自拟·shape] Logits和labels为什么要shift？**

因果LM位置$t$的hidden预测下一个token，不是已经看到的当前位置。Teacher重编码后长度不同，不能直接使用student位置索引teacher row。Prompt最后一个hidden预测首个response token；padding/EOS与人工插入token还需mask。最小例子用可打印三token序列检查每个target和条件prefix，防止future token泄漏。

**Q14 [自拟·角色] OPD teacher就是PPO reference吗？**

Teacher提供希望学生学习的分布，reference限制策略变化；critic估值，behavior记录采样来源，proximal是更新约束锚点。可以复用模型或把某个KL正则替成蒸馏目标，但需要重新解释目标职责。说“都有logprob所以一样”会混淆KL方向和ratio分母。

**Q15 [自拟·缓存] Teacher frozen意味着所有缓存可永久复用吗？**

只有相同模型、tokenizer/template、actual prefix、候选/事件与温度条件的教师分数可以复用。Advantage含学生项、候选支持来自旧学生top-k，更新后都可能变。Cache key只用prompt ID会把不同student continuation错合并。缓存命中率应和概率等价性、bytes与前向成本同时报告。

**Q16 [自拟·mask] 工具文本、思考和动作谁有loss？**

取决目标，通常模型生成thought/action可训练、外部观察不训练但可见。手动插入标签不属于采样动作，自生成标签则可能属于。Loss mask、attention mask、padding与GAE边界mask职责不同。应给一次真实工具调用的逐token表，而不是回答“assistant都乘1”后忽略重建方式。

**Q17 [自拟·数据] 为什么不能用修改后的prompt重算旧logprob？**

行为概率条件是当时真实prefix。换summary、删除观察、修改工具schema或重排role会换条件，即使权重不变ratio也可能≠1。重算可用于current/prox模型，但必须在同样已记录的输入上；旧分母要保留实际behavior概率或经严格同版本同输入验证的重算值。

### 25.5 失败轨迹、异步与性能

**Q18 [公开题型·过滤] 错误学生轨迹为什么不全部丢？**

OPD正希望在学生错误状态提供恢复监督，全部删失败会变成只学成功访问分布。系统故障、无合法输出、数据污染和任务失败分开；可恢复错误与正确前缀可能有价值，教师完全不可靠的状态则需gate/降权/截断。筛选改变分布，必须报告保留率和任务/难度覆盖，而不是“清洗越严格越好”。

**Q19 [自拟·异步] 数据落后两版本就安全吗？**

版本次数不等概率距离，跨turn混合版本还需每段标注。控制在途工作、FIFO/就绪消费、版本限制与局部IS只是不同层的处理；[AsyncOPD](https://arxiv.org/abs/2606.24143v1)明确旧动作与旧prefix不同。要看ratio、ESS、clip fraction、年龄、长任务覆盖和独立质量，不把bounded staleness当无偏保证。

**Q20 [自拟·性能] Teacher更大为什么rollout仍可能最慢？**

Teacher可一次prefill评分，student生成串行多token、多turn，工具等待和重复prefill进一步延迟。Full-vocab输出、跨词表对齐和teacher KV也可能形成新瓶颈，需真实profile。GPU utilization高不等MFU高，卡时按全部预留GPU计；只算learner卡会漏rollout与teacher成本。

**Q21 [自拟·早算] 每turn有teacher信号就能边执行边训练吗？**

要先区分score提前、backward提前累积、optimizer提前更新。Local token OPD信号或已知，终局RL回报/GRPO组优势可能未就绪。提前backward还需完整loss分母、同参数版本与collective顺序；更新提前会改变未来行为分布。对照实验应把三种改动分别隔离，而非统称pipeline。

### 25.6 强基线、证据与展示

**Q22 [自拟·新论文] 新MOPD论文否定了OPD吗？**

没有。它在四个1.7B–8B模型、两种专家组成与相同初始化/词表条件下，独立调优与成功教师语料使离线方法接近MOPD的ID恢复。它还指出OOD保留和成本不同；不能外推跨tokenizer、所有长程coding或任意更大teacher。读[精读](#paper=paper-rethink-mopd)时说明主实验与旧论文复比较的验证口径不同。

**Q23 [自拟·实验] 你的对照至少有哪些？**

相同起点、prompt、teacher与独立测试，充分调LR的teacher-success SFT、soft KD、局部teacher-history蒸馏、真实学生rollout OPD。另按同生成token、同teacher调用、同GPU-hour各比较；报告工具、重试、评价和语料生成的成本，保留OOD与长度。方法组合需拆loss-only、scheduler-only/必要校正、联合处理，不能一个混合方案就称单因果结论。

**Q24 [自拟·结论] 怎样回答“你的贡献是不是只是工程实现”？**

先明确自己实现了什么，不以“加pool”自动称研究创新。问题定义、现有机制不足、可证伪假设、区别于近邻的设计和对应消融形成研究证据；系统工程贡献可以用正确性契约、可靠性、效率与可复用性证明。只跑短训练不能证明收敛或统计非劣，公开论文成绩也不能冒充个人复现结果。


## 26. 面试问答：从基本概念到估计器、异常和系统的56道延伸

Q25—Q80均为本文整理的教学与面试追问；不作为具体公司面试记录。建议先独立回答，再用正文公式检查采样分布、可微量、方向和条件。Q01—Q24保留既有题号，便于旧阅读路线继续使用。

### 26.1 基本概念与方法选择

**Q25 [自拟·软标签] 教师答案和教师分布传递的信息有什么不同？**

硬标签说明某次示范选了哪个token；软标签还表达其他选择及相对概率。例如教师对两个同义续写都给较高概率，完整分布监督可以同时保留它们。教师采样硬标签在适当条件下可以估计软标签梯度，但有限样本具有方差；经过成功过滤、贪心或重采样后，目标分布也可能改变。

**Q26 [自拟·价值网络] OPD一定需要critic吗？**

不需要。直接计算固定前缀的词表KL可以普通反传；即时sampled-token权重也可以不训练价值网络。若引入长程任务回报、GAE或其他方差控制，才需要按具体算法判断是否有critic。把一个token权重叫advantage，不等于它已经是经价值函数估计的真实任务优势。

**Q27 [自拟·自训练] OPD和直接用学生自己的答案做SFT一样吗？**

不一样。学生答案在OPD中首先提供访问状态，教师提供分布或对应评分；错误动作可能得到抑制信号。直接把学生每个token当正标签做SFT，则是在提高这些已采动作的概率。两者都使用学生生成文本，但监督来源和梯度不同。

**Q28 [自拟·评分] 教师为什么不必逐token重新生成答案？**

学生文本已经给定，教师可一次因果前向计算多个位置的条件分布。它读取整段张量不代表能在当前位置偷看未来，因为因果attention会限制可见性。真正应核对的是shift、模板和mask；如果教师评分实现泄漏未来，得到的就不再是部署可用的同条件监督。

**Q29 [自拟·恢复] OPD是否能回头修改已经生成的错误？**

本轮离散轨迹本身不会被反向传播改写。更新改变学生下一轮的生成概率，既可能减少进入错误前缀的机会，也可能改善错误后的恢复。若要给早期动作分配其未来后果的信用，还需区分局部token监督与序列回报；“当前能评价”不等于“所有早期责任已精确归因”。

**Q30 [自拟·公共空间] Teacher和student来自同一家模型就一定能按token ID做KL吗？**

不能只凭家族名称判断。需核对词表映射、扩展token、特殊标记、tokenizer版本和前缀编码。即使词表相同，模板或上下文内容不同也会改变条件概率。跨tokenizer时应定义公共事件和映射，不能拿相同编号当相同文本含义。

**Q31 [自拟·数据] OPD是否需要每道题的标准答案？**

纯教师概率蒸馏可以只用输入问题和学生轨迹，不要求每题有人工标准答案。但教师仍需提供有效监督。若用任务结果筛选、验证器奖励或报告准确率，就需要相应标签、测试或独立判断机制。没有人工答案不等于没有监督，也不等于结果无需验证。

**Q32 [自拟·探索] 学生完全不会某种格式或知识，OPD能凭空学会吗？**

完整教师分布为未采token提供各自的教师信息；sampled的归一化梯度也会影响未采logit，但要取得某动作自己的教师评分，仍需抽到它或另外查询。如果有价值路径在学生中概率极低，单纯rollout可能效率很差。可使用示范冷启动、混合前缀或适配教师，但是否学到新能力需通过独立协议验证，不能仅由OPD名称推断。

**Q33 [自拟·GKD] 混合离线和在线数据的参数控制什么？**

在$\mathcal L_{\mathrm{mix}}=(1-\lambda)\mathcal L_{\mathrm{offline}}+\lambda\mathcal L_{\mathrm{on}}$中，$\lambda$控制两种数据/损失来源的相对贡献，前提是两项的归约定义清楚。它与KL系数、教师温度、成功筛选比例不是同一个参数。改变$\lambda$还会改变训练访问状态，宜与起点和成本一起消融。

**Q34 [自拟·选择] 什么情况下优先考虑离线教师SFT？**

已有可靠、覆盖充分的教师示范，学生任务相对固定，在线生成或环境交互成本高时，应先建立有竞争力的离线基线。OPD的价值需要通过学生错误状态上的增益或同预算表现证明。成功教师数据筛选和学习率应充分调优，不能把“在线”当成天然优越性。

### 26.2 目标、分布和梯度

**Q35 [自拟·交叉熵] FKL和交叉熵数值相同吗？**

不相同。$D_{\mathrm{KL}}(q\|p)=H(q,p)-H(q)$。教师分布和前缀固定时，$H(q)$对学生参数是常数，因此二者学生梯度相同。跨样本比较数值或教师温度变化时，不能忽略教师熵的变化；交叉熵未下降也不能只凭另一种KL日志判断原因。

**Q36 [自拟·方向] 写KL时怎样避免把forward和reverse说反？**

先写教师$q$、学生$p$，再写FKL为$D(q\|p)$、RKL为$D(p\|q)$。不要记“p放前面就是forward”，因为论文的p/q命名不同。用“求和权重是谁”再核对：FKL教师加权，RKL学生加权。

**Q37 [自拟·logit梯度] RKL的logit梯度也是学生减教师吗？**

不是。固定前缀、完整归一化、温度为1时，FKL梯度为$p_u-q_u$；RKL为$p_u[\log(p_u/q_u)-D_{\mathrm{KL}}(p\|q)]$。它含相对log-ratio和分布平均项。把FKL的简洁梯度用于RKL，会导致符号解释和实现核查错误。

**Q38 [自拟·模式] Reverse KL的mode-seeking是否保证只保留正确答案？**

不保证。所谓模式指教师概率分布的高密度区域，不自动等于正确解法。学生容量、温度、支持集和优化决定实际选择。无容量约束且目标可达到时，FKL/RKL都以完整匹配教师为最优，并非RKL数学上必然只剩一个token。

**Q39 [自拟·JSD] JSD怎样连接FKL和RKL？**

令$\alpha\in(0,1)$、$m=\alpha q+(1-\alpha)p$，广义JSD为$\alpha D(q\|m)+(1-\alpha)D(p\|m)$。它通过混合分布缓和直接支持不匹配。原式在端点退化为0；在适当支持条件下，除以$\alpha$后取$\alpha\to0$才得到FKL极限，除以$1-\alpha$后取$\alpha\to1$得到RKL。框架可能对端点单独实现，不能只凭beta名称猜公式。

**Q40 [自拟·温度] 生成温度、教师温度和蒸馏温度怎样分别解释？**

生成温度决定实际行为采样，教师温度改变监督分布，学生损失温度改变参与匹配的学生概率。它们可以不同，但应明确行为logprob来自哪个分布。Top-p/top-k还会截断并重归一，不能仅用raw softmax概率当作实际采样概率。

**Q41 [自拟·正则] 教师KL系数就是reference KL系数吗？**

不是。教师KL推动学生学习目标能力；reference KL限制偏离参考策略；二者可对应不同模型、不同方向和权重。即使复用同一个checkpoint，也应按损失职责区分。改变蒸馏系数还会改变它与任务奖励/SFT的梯度比例。

**Q42 [自拟·负权重] 一个batch的OPD权重几乎都为负，还能学习吗？**

可以。完整当前分布下$\mathbb E_p[\log q-\log p]=-D_{\mathrm{KL}}(p\|q)\leq0$，所以平均非正本来就正常。负权重以不同强度抑制相应动作，概率归一化使其他方向相对获得质量。不能仅以负号判断“所有概率都在下降”；也不能把平均负值直接推成必然短回答。

**Q43 [自拟·负KL] 训练日志中的KL为负，必然是bug吗？**

不必然。单token log-ratio、有限样本均值和未归一化部分求和都可能为负。完整合法分布的KL则非负。先确认日志到底记录哪个量、采样来自谁、是否截断或减baseline；若声称计算的是完整归一化KL却显著为负，才需要检查实现或数值问题。

**Q44 [自拟·相对信号] 教师给某token高概率，它就一定被鼓励吗？**

不一定。常见sampled更新看$\log q-\log b$的相对差异，学生已经比教师更偏向它时权重仍可能为负；完整KL还包含概率归一化的耦合。更重要的是条件概率高不等于最终任务正确，尤其在学生已有错误前缀时。

**Q45 [自拟·零概率] 能把teacher top-1当one-hot分布直接做RKL吗？**

若学生给其他token正概率，而教师one-hot在这些token上为0，RKL无穷。对one-hot目标做交叉熵是另一个可行目标；平滑、截断或只在子集比较会改变定义。应先确定真正要优化的散度，不能把所有硬标签操作都叫相同RKL。

### 26.3 采样估计与top-k

**Q46 [自拟·自动求导] 为什么KL数值无偏还不能直接backward？**

因为期望中的采样分布也依赖参数。把采样ID固定后只对log-ratio求导，会漏掉score-function项。K1是典型反例：其期望等于RKL，裸梯度期望却为0。核查时必须同时写采样分布、标量公式和反向路径。

**Q47 [自拟·K2] 平方log-ratio是不是精确的KL损失？**

通常不是。按本文$r=\log p-\log q$定义，$r^2/2$的期望一般不等于RKL，但固定索引自动求导给$r\nabla\log p$，在当前$p$采样等条件下其期望匹配局部RKL梯度。这个例子说明目标数值和更新方向是两层命题；旧样本或不同停止梯度会改变结论。

**Q48 [自拟·K3] 非负低方差KL估计器是否自动得到RKL梯度？**

不自动。本文K3为$q/p-\log(q/p)-1$，在$p$采样、完整支持下期望是RKL；固定采样索引直接求导的期望却是FKL梯度。其他实现可能有detach、IS或PG路径，必须逐式核对。标量非负和梯度正确没有逻辑等价关系。

**Q49 [自拟·接口] 有teacher top-k，是否就能计算student top-k上的RKL？**

不一定。学生候选可能不在教师返回集合中，所需教师概率未知。需支持指定token评分、扩大候选payload或使用明确的部分监督/尾部目标。把未知概率设成极小常数会注入强烈人为信号，不等于补全了真实教师分布。

**Q50 [自拟·top-1] 为什么某些top-1 KL恒为0，实际库却还能训练？**

仅对同一个候选分别归一化后，两边都是概率1，KL当然为0。真实实现可能加tail形成二元分布、使用原始质量、采用采样梯度代理或硬标签交叉熵，因此仍有梯度。应检查数学定义和代码路径，不能只看`top_k=1`。

**Q51 [自拟·两层采样] 从教师采监督token，为什么仍可以叫on-policy？**

因为on-policy讨论的是训练前缀怎样访问。可以先由学生生成$h$，再在这个固定$h$下从教师$q$抽一个目标token估计FKL梯度；该目标token不必用来推进下一步学生轨迹。若真的由教师动作推进历史，就需要重新说明后续状态分布。

**Q52 [自拟·重归一] Top-k集合内KL很小，能说明整体对齐吗？**

不能。子集重归一只比较相对比例，学生可以在子集里只放很小总质量。A/B比例一致而EOS概率0.9的例子说明这种盲区。应报告head覆盖和tail总质量，并考虑EOS、恢复token及并集支持。

**Q53 [自拟·tail] 加tail以后是否已经是全词表KL？**

不是，除非尾部为空或尾部内部条件分布也相同等特例。它是合并事件后的合法粗粒度KL。完整FKL还包含$(1-Q_S)D(q^{\mathrm{tail}}\|p^{\mathrm{tail}})$，RKL则按学生尾部质量加权。Tail能约束质量，却不能自动识别尾部每个具体错误动作。

**Q54 [自拟·无偏] Top-k加一个学生采样token就一定无偏吗？**

不一定。要看是否采用正确的尾部校正、动作提案是否覆盖尾部、是否匹配当前策略，以及是否正确处理概率权重和detach。把sampled token简单加入归一化集合，与构造无偏完整梯度不是同一件事。无偏也不保证低方差或有限预算下任务效果最好。

**Q55 [自拟·支持集] Top-p抽样后还能用普通IS恢复完整目标吗？**

如果被截断token在目标分布中仍有正概率，提案支持缺失，普通动作IS无法恢复那些从不被采到的项。温度只重加权、没有造零支持时也需用真实提案概率。可以改变目标支持、保留探索或用其他估计器，但必须明确偏差和条件。

### 26.4 复读、截断和目标偏离

**Q56 [自拟·复读] 学生已经严重复读，OPD是否还能救回来？**

存在机会，但需教师在重复前缀上提供可靠的恢复方向，监督覆盖该方向，且学生有足够探索与稳定更新。只看教师从头生成不复读不够。先定位首次重复状态，比较重复/恢复/EOS概率，再做SFT起点、tail、更新幅度和可靠性处理的对照。

**Q57 [自拟·条件评分] 教师自己不复读，为何仍可能强化学生复读？**

因为教师自己的历史分布与学生循环历史不同。条件在一段已重复文本上，继续重复可能成为教师较高概率的补全；它并未按最终任务价值打分。局部概率监督、有限近似和访问分布更新可以共同强化这种行为，需检查教师在实际坏前缀上的可靠性。

**Q58 [自拟·EOS] 提前EOS怎样通过公式判断能否纠正？**

完整FKL在温度1时，EOS logit梯度为$p_{\mathrm{EOS}}-q_{\mathrm{EOS}}$。学生高于教师就有压低的局部梯度。还需检查真实EOS是否参与loss、是否与padding共ID导致误mask，以及EOS是否被top-k监督遗漏。这个公式不替代整体训练效果验证。

**Q59 [自拟·预算截断] Max tokens截断为什么不能靠OPD直接修好？**

它是外部停止规则，模型即使想继续也无法超过该上限。OPD可改变内容效率或推理长度，但未生成的后续状态没有被采样监督。需按任务调整预算、续写和上下文策略，并分别报告自然结束率与预算截断率。

**Q60 [自拟·错误终止] 学生答案错误，但teacher对最后token概率很高，是否矛盾？**

不矛盾。教师条件在此前整段错误推理上，最后一个词可能很容易预测；条件语言概率不是完整答案正确性。可以检查更早的分叉位置、引入任务验证或未来信用，但不能要求每个错误答案token都得到负更新权重。

**Q61 [自拟·hacking] OPD没有独立reward model，为什么仍可能reward hack？**

教师概率与更新代理形成隐式优化信号，仍可能和真实质量不一致。局部条件、top-k、有限轨迹、长度与mask等设计可进一步造成盲区。理想完整KL匹配与实际近似应分开讨论；不能把“没有单独RM”推成“不可能出现优化信号改善但质量变坏”。

**Q62 [自拟·日志] KL下降但正确率下降，该怎样定位？**

先确认日志是完整KL、采样值还是裁剪后的部分损失；再分任务、长度和成功/失败响应看教师信号、EOS、熵及head/tail。检查教师在坏前缀上的偏好是否和正确性一致。若同时改数据筛选、loss和系统调度，应拆开消融，不能直接认定一种散度“天然无效”。

**Q63 [自拟·筛选] 全部失败轨迹都过滤掉是不是更稳？**

可能减少某些异常强化，也可能删掉最需要学习恢复的状态，改变任务难度和访问分布。应区分服务失败、上下文损坏、不可恢复状态与有价值错误前缀；报告保留比例和覆盖。保留正确前缀、只屏蔽不可靠片段等也需按目标验证，不是一条通用最优规则。

**Q64 [自拟·裁剪] 把所有负KL裁成0有什么问题？**

如果原量是单样本log-ratio或部分KL，负值可能有正常含义。裁成0会删除这些区域的梯度，改变原目标。优化约束可以有价值，但不能为“让图更好看”擅自处理，也不能在裁剪后继续直接套原来无偏推导。

**Q65 [自拟·序列目标] 全词表KL已经精确，为什么仍可能不是精确序列优化？**

因为词表求和只解决固定前缀内的动作维度。学生参数还影响未来访问状态，固定旧前缀的局部更新没有自动包含这部分导数。两步分支例子在根局部梯度为0时，完整序列梯度仍可为$C/4$。逐token平均与序列累计也会进一步改变目标。

### 26.5 实现、系统与成本

**Q66 [自拟·batch] 面试让你画一个训练batch，至少有哪些字段？**

任务/模板、输入与回答token、角色和动作mask、实际行为logprob及版本、教师概率payload及版本、候选支持、终止原因、任务和工具结果。还要说明learner从哪里得到当前logprob、各张量的形状与归约分母。只画prompt/response不足以检查概率条件和异步正确性。

**Q67 [自拟·缓存] 教师不变，为什么advantage不能永久缓存？**

教师分数只在同教师、同前缀、同温度和同事件下可复用。Advantage还包含学生logprob，学生top-k候选也会变化。可缓存教师项，但更新时应按目标决定是否重算学生项。只用prompt ID作key会错误合并不同学生回答。

**Q68 [自拟·归约] 为什么梯度累积后结果可能与大batch不同？**

若各microbatch有效token数不同，先各取mean再平均，与全局分子除以全局token数不同。DDP的默认梯度平均还会影响缩放。应在同一合成批次上检查拆分前后目标与梯度，并解释空mask、跨卡分母与累积步数。

**Q69 [自拟·瓶颈] 教师更大，为什么profile显示学生rollout最慢？**

教师对给定文本可并行评分，而学生自回归生成串行；Agent还可能等待工具、重试和重复prefill。大词表payload、teacher KV与通信也可能成为瓶颈，需实测拆分。不能只根据参数量、GPU utilization或某个内核吞吐判定端到端成本。

**Q70 [自拟·公平成本] 报告OPD快多少，应该算哪些资源？**

包括学生生成、教师评分、learner训练、参数同步、环境交互和所有预留设备。报告GPU-hour时需说明硬件、精度、长度、负载、模型与基线；只统计learner卡或忽略教师数据生成会失真。同token预算与同卡时预算回答的问题不同，应分别比较。

**Q71 [自拟·历史] 旧轨迹换成新的summary再算ratio为什么有问题？**

Ratio要求在同一个实际条件下比较动作概率。改变summary、工具schema、角色顺序或观察内容就改变了条件；即使模型权重相同，概率也可能不同。应保存当时真实输入，区分“重建同一历史”与“创造一个新训练状态”。

**Q72 [自拟·部分轨迹] 回答没结束就评分训练，可以吗？**

局部OPD可以从部分rollout取得有效前缀监督，不要求每个局部目标等到终局结果。若混合结果奖励、组内优势、序列归约或提前optimizer更新，则需重新检查何时信号和分母就绪，以及是否改变后续行为版本。部分监督不等于自然终止标签。

**Q73 [自拟·LoRA与精度] OPD可以用LoRA、BF16或量化模型吗？**

可以作为实现选择，但要分别验证。LoRA限制可更新参数空间，不改变所定义的教师目标；BF16和量化会影响数值、概率尾部与内核兼容。概率和KL归约宜先在稳定精度下验证。低训练显存、低教师推理成本和蒸馏质量是不同指标，不能互相替代。

### 26.6 扩展、结果解释与项目答辩

**Q74 [自拟·多教师] 为什么平均多个教师的FKL和RKL不等价？**

同前缀、同样本空间和正支持下，最小化加权FKL的自由分布最优解是教师概率的算术平均；加权RKL则是归一化几何平均。按域路由一个教师、合并分布和加多个loss也不同。概率校准、任务权重与遗忘应单独验证，不能只说“取平均就融合能力”。

**Q75 [自拟·身份] Behavior、teacher、reference、current分别是什么？**

Behavior生成已记录动作，current是正在训练的学生，teacher提供目标监督，reference限制漂移。同步时behavior与current在开始更新前可相同，更新后不再相同；teacher与reference也可以恰巧复用参数。身份按数学职责定义，不能由服务名称或是否冻结来猜。

**Q76 [自拟·跨词表] 跨tokenizer蒸馏最先要定义什么？**

先定义可比较的公共文本/byte事件、事件概率和位置对应，再讨论KL。Token到文本可能一对多或多对一，需处理质量守恒、尾部、EOS、截断与非法事件。公共事件投影可能丢失信息，不自动等于原词表的完整监督；生产实现还需检查并行分片和接口契约。

**Q77 [自拟·Agent] 学生采样的token为什么可能不是完整环境on-policy？**

因为环境可能由教师、fast policy或其他版本动作推进，学生长响应只在这些历史上被查询。需分别记录proposal、execute和train策略。工具观察、上下文压缩与沙箱恢复也会改变状态分布，不能只看一个student sampling调用就判断部署访问分布一致。

**Q78 [自拟·自蒸馏] 相同模型给自己做老师，为什么可能有学习信号？**

教师可使用不同上下文、额外证据、EMA参数、不同采样或筛选，从而与学生条件分布不同。若同参数、同上下文、同温度和完全相同分布，KL为0，没有凭空产生信息。使用特权上下文时要防止训练/评测泄漏，并说明部署时学生实际能获得什么。

**Q79 [自拟·能力] OPD后的学生能超过教师吗，能否断言只能重排已有能力？**

具体指标上可能出现超过教师的结果，但需区分解码、预算、任务分布、容量约束与外部奖励。理想分布匹配目标本身不是寻找比教师更优策略的保证；某些有限实验观察到pass@1提高而问题覆盖变化不大，也不能推出所有OPD永远无法获得新能力。应明确评价协议和证据范围。

**Q80 [自拟·项目] 怎样把OPD项目讲得有技术含量而不夸大？**

按“具体问题—原机制为何不足—可检验假设—目标与实现变化—正确性检查—受控实验—局限”讲。数学目标、数据分布、调度系统和评测协议分别交代。公式验算、静态源码核读、真实训练和个人贡献分层举证；没有执行过的实验说成方案，不把论文结果或目录里的代码存在当作个人成绩。


## 27. 面试表达：30秒、3分钟与10分钟的连续主线

### 27.1 30秒：定义、价值与边界

OPD让学生自己生成轨迹，教师在学生实际访问的前缀上提供监督，学生更新后再采样。它希望缓解离线教师示范与学生推理状态的分布偏移。On-policy规定前缀来源，FKL/RKL规定匹配目标，sampled-token/top-k/全词表规定监督近似，三者分开。它有机会纠正错误和复读，但依赖教师在这些状态上的可靠性，最终仍要看独立任务结果。

### 27.2 3分钟：从离线问题串到异常行为

传统教师SFT主要在教师示范前缀上训练；学生实际生成会犯自己的错误，进入教师示范里少见的状态，因此可能累积偏差。OPD让学生先生成，再由教师对同一前缀计算下一token概率，使学习发生在学生真实遇到的状态上。

具体目标可以用教师到学生的FKL，也可以用学生到教师的RKL。前者按教师概率加权，直接推动学生覆盖教师认可的选择；后者按学生概率加权，重点约束学生正在做而教师不认可的方向。RKL可以展开成教师logprob加学生熵，也可以用“教师减学生logprob”的采样更新权重实现，但要核对停止梯度、行为分母与近似条件。

监督可用全词表、top-k或sampled-token。全词表在当前前缀上最完整；sampled方式数据量少但有方差；top-k需要特别处理尾部和EOS，否则子集KL很小也可能仍有提前结束问题。这些开销主要涉及logits保留、传输和损失计算，不意味着教师前向消失。

复读能否纠正，要看教师在重复前缀上是否提高恢复方向；提前EOS能否纠正，要看教师的继续信号和EOS监督；达到max tokens则先处理外部预算。OPD也不保证没有目标偏离，因为教师条件概率不是任务正确性，实际训练又有局部与采样近似。落地时同时检查正确率、复读、终止原因、长度、head/tail以及更新稳定性，并用强SFT和同预算对照证明价值。

### 27.3 10分钟：白板与追问的展开顺序

1. 用错误前缀说明离线与学生访问分布不同，画学生rollout—教师评分—learner更新闭环。
2. 定义$x,y,h,\theta,p,q,b,m$，写固定批次的OPD目标，说明哪些量冻结。
3. 写FKL/RKL，推导$p-q$与RKL score-function梯度，解释模式倾向的条件。
4. 写sampled代理，解释“KL值无偏但裸backward梯度不对”，说明温度、IS与旧信号。
5. 用A/B/EOS例子说明top-k重归一的盲区，再区分tail bucket与无偏尾部校正。
6. 区分当前条件KL、完整序列KL、即时信号与未来信用；说明OPD可用PG但不必须用PPO。
7. 分复读、提前EOS、预算截断、配置停止讨论能否纠正，接reward hacking的理想/实际边界。
8. 沿batch字段讲teacher API、shift、mask、归约、缓存、异步与参数同步。
9. 用强基线、质量/行为/系统三类指标及同预算消融收尾；自己的具体贡献只引用真实记录。

这是一条可选择展开深度的路线，不要求一次开场背完全部章节。遇到追问，先重述所采用的目标和条件，再给公式、算例与验证证据。

## 28. 小实验、自检与本轮验证范围

### 28.1 不装PyTorch也能运行的基础验算

下载[OPD基础CPU验算脚本](./assets/learning/opd-foundations-check.py)与[本轮执行结果](./assets/learning/opd-foundations-check-results.json)。脚本只依赖Python标准库，枚举有限分布并使用有限差分，不加载模型、不访问教师API、不运行GPU训练。

```bash
python assets/learning/opd-foundations-check.py
```

本轮实际通过15项检查：完整FKL/RKL的logit梯度；FKL与交叉熵关系；K1值与梯度；K2/K3的前向与反向差别；学生提案估计FKL梯度；截断支持缺失；top-2/EOS反例；tail粗粒度分解；部分KL负值；当前/旧信号IS差异；局部/序列梯度反例；全局token分母；多教师两类最优分布；重复乘学生概率的错误；因果shift与真实EOS对应。

### 28.2 实际backward的PyTorch监督对照实验

下载[完整教学代码](./assets/learning/opd-supervision-lab.py)与[实际运行结果](./assets/learning/opd-supervision-lab-results.json)。在已安装PyTorch的环境、从仓库根目录执行：

```bash
python assets/learning/opd-supervision-lab.py
```

本轮使用Python 3.12与PyTorch 2.11.0，在CPU上通过47项检查。脚本先用FP64核对解析梯度和有限差分，再检查第14节函数的实际反传：full FKL/RKL、三种top-k的两个方向、$k=V$、教师/行为冻结、K1/K2/K3、sampled期望、current/old信号IS、候选外梯度、tail分解、极小尾部的log空间计算，以及无效位置和空批处理。另以固定随机种子抽10万个动作，对照采样平均梯度与完整梯度；它验证的是同一有限前缀上的估计，不是10万条语言模型回答。

结果JSON中的`top2_comparison`保存第10.6节每种top-k的loss和logit梯度，`three_logit_EOS_traces`保存20步梯度下降的EOS概率。三logit例子的学生起始EOS为0.90、教师为0.10；候选内归一的FKL无纠偏梯度，full与tail会降低EOS，且因集合外只有EOS一个事件，两者轨迹一致。这个可运行反例用于理解目标盲区，不是声称某个真实模型已修复提前结束。

### 28.3 怎样把练习推进到真实训练

建议按三个层次复习：先手算三/四token的FKL/RKL和$p-q$；再运行两套脚本，对照值、采样分布、停止梯度与实际backward；最后在自己的小模型训练中单独验证teacher输入、mask、EOS、长度与任务质量。第三层是后续实验计划，本文没有把它登记为已完成。

第一轮内容审阅核对了角色、概率空间、KL方向、可微路径、温度、尾部、终止和归约；第二轮通过可枚举反例与数值梯度复核关键结论，并在站点构建中检查公式、来源、链接和图表。发布审阅另见[本轮审阅记录](./assets/learning/opd-interview-foundations/review-2026-10-25.md)。数学正确性、静态代码核读与实际大模型实验是三种不同证据。

## 29. 原始资料、版本与继续阅读

下表说明资料具体用于什么。数学推导和教学算例由本文独立组织，论文的实验观察只在其原设置范围内引用，不转写成个人复现。动态代码固定到本轮核读commit；正文不承诺上游之后的配置名称仍相同。

| 资料 | 核读基线 | 支撑范围 |
| --- | --- | --- |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531v1) | v1，2015 | 软目标与温度蒸馏背景 |
| [DAgger](https://arxiv.org/abs/1011.0686v3) | v3，2011 | 学习者访问状态上的专家监督；不移植任意保证 |
| [GKD](https://arxiv.org/html/2306.13649v3) | v3，2024 | 学生轨迹、固定采样历史、可选散度与离线混合 |
| [MiniLLM](https://arxiv.org/html/2306.08543v6) | v6，2026；初稿2023 | RKL、序列目标、单步/长程分解 |
| [Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/) | 2026-10-25核读 | Sampled-token训练思路与teacher-forcing评分 |
| [Unbiased Top-k Estimation](https://arxiv.org/html/2609.34447v2) | v2，2026 | 全词表/采样/top-k梯度和具体尾部校正 |
| [Tail-Aware Top-k](https://arxiv.org/html/2608.14728v1) | v1，2026 | 子集归一化与尾部质量问题 |
| [Trust Region OPD](https://arxiv.org/html/2606.01249v1) | v1，2026 | 师生状态不匹配、不可靠监督与引导 |
| [AsyncOPD](https://arxiv.org/abs/2606.24143v1) | v1，2026 | 旧动作、当前信号、有限缓存和异步问题 |
| [Gains and Collapse in OPD](https://arxiv.org/html/2610.03185v1) | v1，2026，预印本 | 特定sampled实现中的重复/过长与隐式信号偏离 |
| [多教师离线基线审计](https://arxiv.org/abs/2610.04272v1) | v1，2026，预印本 | 强SFT、独立调参与公平成本对照 |
| [Tinker源码](https://github.com/thinking-machines-lab/tinker-cookbook/blob/54b3cb293038ccf2e49952e46ed5abac7119fdf8/tinker_cookbook/distillation/train_on_policy.py) | `54b3cb293038ccf2e49952e46ed5abac7119fdf8` | token信号和未来累计选项的静态核读 |
| [verl OPD文档](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/docs/algo/opd.md) | `3c6d9b43b67c74ec343d77874cd8fa6e841523da` | top-k、tail、sampled与PG路径的版本化接口 |
| [PyTorch kl_div](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html) | 2.11文档；本机2.11.0实测 | 参数方向、log_target、归约；学生target侧梯度已核对 |
| [PyTorch autograd](https://docs.pytorch.org/docs/2.11/notes/autograd.html) | 2.11文档 | 非法运算与mask、eval/no_grad及冻结的不同作用 |

第25节保留的社区题源只支持问题方向，不用于替代原始技术依据。首次登记日期仍为2026-10-12，不把历史登记冒充本轮新抓取。来源清单可下载[JSON账本](./research/opd-interview-foundations-sources.json)；源码SHA与文档hash用于核对，不等于已安装或运行对应框架。

进一步阅读：[OPD快速入口](#q=opd-vs-sft)、[后训练逐题练习Q17](#q=pt-opd-gradient)、[Agentic OPD训练](#q=agentic-opd-training)、[训练Pipeline](#q=agentic-opd-pipeline)、[跨tokenizer实现](#q=opd-interview-cross-tokenizer)、[项目答辩](#q=opd-project-defense)、[后训练方法比较](#q=sft-dpo-rl)。资料与正文核对日：2026-10-25。
