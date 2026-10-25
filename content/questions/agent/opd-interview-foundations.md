---
id: opd-interview-foundations
title: "OPD系统整理：概率基础、公式推导、训练实现与面试问答"
category: agent
difficulty: 进阶
tags: ["P0", "OPD", "Interview", "KL", "Policy Gradient", "Distillation", "系统教材"]
updated: "2026-10-25"
summary: "从学生轨迹讲清FKL/RKL、sampled/top-k/full的目标与梯度，配四token算例、关键PyTorch代码、异常排查与80道速答。"
draft: false
---

## 1. 问题背景：为什么在学生轨迹上蒸馏

**OPD让学生自己生成，再让教师评价这些生成位置，更新学生后重新采样**。教师提供的是当前前缀下的token概率，学生学习怎样调整下一步选择。

离线教师示范主要覆盖教师会走的路径。学生一旦在中途出错，就可能进入示范中没有的前缀；OPD直接在这些前缀上训练恢复能力。

全文围绕三个问题展开：前缀由谁生成；用哪种KL匹配师生分布；每个位置用采样token、top-k还是全词表。先读第2—10节掌握公式，再读第14节代码。复读、截断和reward hacking分别见第17—19节。

## 2. 前置知识：token、概率与梯度

### 2.1 从logit得到概率

Token是分词器输出的离散符号。词表$\mathcal V$包含全部可输出token，大小为$V$。在前缀$h$下，模型输出$V$个实数logits，softmax将其转换为概率：

$$
p(v\mid h)=\frac{\exp(z_v(h)/\tau)}{\sum_{u\in\mathcal V}\exp(z_u(h)/\tau)},\qquad \tau>0.
$$

$z_v(h)$是token $v$的logit，$\tau>0$是温度。概率之和为1；提高温度会使分布更平缓。`logprob`就是概率的自然对数：$\log0.8\approx-0.223$，$\log0.1\approx-2.303$。代码用`log_softmax`直接计算，避免先得到极小概率再取对数。

### 2.2 一个位置的概率与整条回答的概率

输入为$x$，回答为$y=(y_1,\ldots,y_T)$。模型逐步生成，完整回答概率为：

$$
\pi(y\mid x)=\prod_{t=1}^{T}\pi(y_t\mid x,y_{<t}),\qquad
\log\pi(y\mid x)=\sum_{t=1}^{T}\log\pi(y_t\mid x,y_{<t}).
$$

$y_{<t}$是第$t$个token之前的内容，$h_t=(x,y_{<t})$称为前缀。EOS是模型选择结束的特殊token；达到`max_new_tokens`则是外部强制停止。

### 2.3 期望、采样与梯度

$$
\mathbb E_{a\sim p}[f(a)]=\sum_{v\in\mathcal V}p(v)f(v).
$$

期望是“每个值乘以发生概率，再相加”。从$p$抽一个动作$a$，用$f(a)$近似这个和，就是Monte Carlo估计。无偏指估计量的期望等于目标；有限样本仍有误差。

梯度描述参数变化怎样影响损失。梯度下降更新为$\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L$，$\eta>0$是学习率。后文的logit梯度用于判断：当前前缀下，哪些输出倾向会被提高或压低。

## 3. 全文符号、条件与可微对象

统一用$p$表示学生、$q$表示教师、$b$表示采样这批数据的行为策略。所有对数为自然对数，KL单位为nat。

| 符号 | 含义 |
| --- | --- |
| $x\sim\mathcal D$ | 从输入分布抽取的问题 |
| $y=(y_1,\ldots,y_T)$ | 学生回答及其长度$T$ |
| $h_t=(x,y_{<t})$ | 预测第$t$个token前的可见历史 |
| $\mathcal V,V$ | 公共词表及大小 |
| $\theta,\bar\theta$ | 当前学生参数、生成本批轨迹时的参数快照 |
| $p_v=\pi_\theta(v\mid h)$ | 当前学生在前缀$h$下的概率 |
| $q_v=\pi_T(v\mid h)$ | 冻结教师在同一前缀下的概率 |
| $b_v=\pi_b(v\mid h)$ | 行为策略实际采样概率 |
| $z_v$ | 学生logit |
| $\pi_{\mathrm{ref}}$ | 可选参考模型，用于限制策略漂移 |
| $\tau_r,\tau_s,\tau_T$ | 生成、学生损失、教师评分的温度 |
| $m_{it}\in\{0,1\}$ | 第$i$条回答第$t$个位置是否计算loss |
| $B,T_i,N$ | 回答条数、各条长度、有效位置总数$N=\sum_{i,t}m_{it}$ |
| $S,k$ | top-k候选集合及大小 |
| $P_S=\sum_{v\in S}p_v$ | 学生在候选集内的概率总量 |
| $Q_S=\sum_{v\in S}q_v$ | 教师在候选集内的概率总量 |
| $\operatorname{sg}(u)$ | stop-gradient：保留数值，反向梯度为0 |
| $\rho=p_\theta(a\mid h)/b(a\mid h)$ | 动作重要性权重 |
| $r_t,A_t,G_t$ | 即时反馈、更新权重、未来累计反馈 |

第7—10节默认：固定前缀、冻结教师、同一词表、温度1、相关概率为正。只对学生概率和参数求导；采到的token ID、历史、教师分数和行为概率固定。跨词表与不同上下文的处理见第21节。

## 4. 离线蒸馏、SFT与OPD的关系

### 4.1 SFT与两种蒸馏

SFT提高示范token在示范前缀上的概率：

$$
\mathcal L_{\mathrm{SFT}}(\theta)=
-\mathbb E_{(x,y^*)\sim\mathcal D_{\mathrm{demo}}}
\left[\sum_t\log\pi_\theta(y_t^*\mid x,y_{<t}^*)\right].
$$

$\mathcal D_{\mathrm{demo}}$是示范数据分布，$y^*$是示范答案。教师生成答案后做SFT，称为sequence-level KD；保留教师在各位置的概率分布并做匹配，称为soft-label KD。[经典KD](https://arxiv.org/abs/1503.02531v1)讨论了软目标与温度。

硬标签只告诉学生“这次选A”；软标签还能告诉学生“A占60%，B占35%，其余占5%”。

### 4.2 OPD改变的是训练前缀

例如教师示范始终是“先乘除，再加减”，学生却先算了加法。继续训练教师示范看不到这个错误前缀；OPD让教师直接评价学生走到这里以后该怎样继续。这与[DAgger](https://arxiv.org/abs/1011.0686v3)在学习者状态上查询专家的思路相同。

### 4.3 三项设计分别决定什么

| 设计 | 决定的内容 | 常见选择 |
| --- | --- | --- |
| 前缀来源 | 在哪里训练 | 离线示范、当前学生、旧学生、混合 |
| 匹配目标 | 学什么分布 | FKL、RKL、JSD |
| 监督表示 | 每个位置比较哪些动作 | sampled-token、top-k、全词表 |

OPD首先规定使用学生轨迹。KL方向和监督表示仍可分别选择。

## 5. 一次OPD训练究竟发生了什么

![本文示意：学生轨迹、教师评分与学生更新构成闭环](./assets/learning/opd-interview-foundations/opd-training-loop.svg)

1. 采样问题，学生生成回答，记录token与行为logprob。
2. 教师读入学生轨迹，给每个前缀的下一token打分。
3. Learner重算当前学生概率，计算蒸馏loss并反传。
4. 更新学生，将新权重同步给生成端，再采下一批。

教师评分采用teacher forcing：整段学生文本已经给定，因果attention让每个位置只看此前内容，因此多个位置可以并行评分。图中的回环对应第4步，教师参数始终冻结。

一个batch至少保存：实际输入和回答token、行为版本与logprob、教师分数及候选ID、动作mask、真实长度与停止原因。工具轨迹还要保存角色和外部观察。

## 6. OPD目标：在学生前缀上匹配教师分布

令$d_b(h)$为行为策略产生的前缀分布。固定本批前缀，优化：

$$
\mathcal L_D(\theta;b)=\mathbb E_{h\sim d_b}\big[D_h(\pi_T,\pi_\theta)\big].
$$

$D_h$选FKL、RKL或JSD。目标是减小这些前缀下的师生分布差异。$b=p_\theta$时数据来自当前学生；异步生成或复用旧batch时，$b$会落后于当前学生。

批次中常用token平均：

$$
\widehat{\mathcal L}_D(\theta)=
\frac{\sum_{i=1}^{B}\sum_{t=1}^{T_i}m_{it}D_{it}(\pi_T,\pi_\theta)}{N},
\qquad N=\sum_{i,t}m_{it}>0.
$$

$m_{it}$选择参与训练的模型动作，$N$是其总数。$N=0$时跳过或报错。本轮将历史与mask视为固定量；第11节讨论对完整序列分布求导时增加的项。

[GKD](https://arxiv.org/html/2306.13649v3)还可混合离线与学生轨迹监督：

$$
\mathcal L_{\mathrm{mix}}=(1-\lambda)\mathcal L_{\mathrm{offline}}+\lambda\mathcal L_{\mathrm{on}},\qquad 0\leq\lambda\leq1.
$$

$\lambda$是在线项的权重。两项应分别明确散度和归约方式。

## 7. Forward KL：按教师概率覆盖合理动作

### 7.1 优化目标

$$
\boxed{\ell_F=D_{\mathrm{KL}}(q\|p)=\sum_{v\in\mathcal V}q_v\log\frac{q_v}{p_v}.}
$$

教师给某token较大概率而学生给得很小时，FKL惩罚较大。例如$q_A=0.4,p_A=0.01$，训练会提高学生选择A的倾向。

展开后：

$$
\ell_F=\underbrace{-\sum_vq_v\log p_v}_{H(q,p):\,\text{交叉熵}}
-\underbrace{\left(-\sum_vq_v\log q_v\right)}_{H(q):\,\text{教师熵}}.
$$

$H(q,p)$为交叉熵，$H(q)$为教师熵。固定教师和前缀时，$H(q)$对学生参数是常数，所以FKL与教师软标签交叉熵的学生梯度相同。

### 7.2 Logit梯度

Softmax满足$\partial\log p_v/\partial z_u=\mathbf1[v=u]-p_u$，代入得：

$$
\frac{\partial\ell_F}{\partial z_u}
=-\sum_vq_v(\mathbf1[v=u]-p_u)=p_u-q_u.
$$

梯度下降压低$p_u>q_u$的方向，抬高$p_u<q_u$的方向。完整KL非负；个别求和项可以为负。若$q_u>0,p_u=0$，FKL为无穷。

### 7.3 用一个教师采样token估计

在固定学生前缀上抽$a\sim q$，对$-\log p_a$求导，其期望就是FKL梯度。因此，学生可以负责生成前缀，教师另采一个监督token；这个监督token无需用于推进轨迹。

## 8. Reverse KL：约束学生常选而教师不认可的动作

### 8.1 优化目标

$$
\boxed{\ell_R=D_{\mathrm{KL}}(p\|q)=\sum_vp_v\log\frac{p_v}{q_v}.}
$$

RKL按学生概率加权。学生经常选择、教师却给低概率的动作，对损失贡献大；学生几乎不选的动作贡献小。这解释了mode-seeking倾向。模型容量足够且可达到最优时，FKL与RKL都以$p=q$为解。

$q_v=0,p_v>0$会使RKL无穷，所以教师top-k外的未知概率不能直接填0。

### 8.2 从概率求和推导采样梯度

$$
\begin{aligned}
\nabla_\theta\ell_R
&=\sum_v\nabla_\theta p_v\left(\log\frac{p_v}{q_v}+1\right)\\
&=\sum_vp_v\left(\log\frac{p_v}{q_v}+1\right)\nabla_\theta\log p_v.
\end{aligned}
$$

利用$\sum_vp_v\nabla\log p_v=\nabla\sum_vp_v=0$，消掉常数1：

$$
\boxed{\nabla_\theta\ell_R=\mathbb E_{a\sim p}\left[\log\frac{p_a}{q_a}\nabla_\theta\log p_a\right].}
$$

这就是sampled RKL的依据：从学生抽动作，用log-ratio加权该动作的score梯度$\nabla\log p_a$。

对应的logit梯度为：

$$
\frac{\partial\ell_R}{\partial z_u}
=p_u\left(\log\frac{p_u}{q_u}-\ell_R\right).
$$

它含全分布平均项$\ell_R$，与FKL的$p-q$不同。

### 8.3 教师logprob与学生熵

$$
-\ell_R=\mathbb E_{a\sim p}[\log q_a]+H(p),\qquad H(p)=-\sum_vp_v\log p_v.
$$

最小化RKL等价于最大化教师logprob与学生熵之和。删除熵项后，目标会变成寻找教师最高概率动作。OPD可以直接反传完整KL，也可以借用PG；PPO、GRPO和critic不是必需组件。[MiniLLM](https://arxiv.org/html/2306.08543v6)与[作者实践](https://thinkingmachines.ai/blog/on-policy-distillation/)分别给出序列和token更新方案。

### 8.4 JSD：与混合分布比较

令$u=\alpha q+(1-\alpha)p$，$0<\alpha<1$：

$$
\operatorname{JSD}_{\alpha}(q,p)=\alpha D_{\mathrm{KL}}(q\|u)+(1-\alpha)D_{\mathrm{KL}}(p\|u).
$$

两侧都与混合分布比较，因此即使原支持不重叠，JSD也有界。原式在$\alpha=0,1$退化为0；适当支持条件下，$\operatorname{JSD}_\alpha/\alpha$在$\alpha\to0$趋于FKL，$\operatorname{JSD}_\alpha/(1-\alpha)$在$\alpha\to1$趋于RKL。框架若在端点分派KL，应按其实际定义读取参数。

## 9. Sampled-token：怎样从一个动作得到KL梯度

### 9.1 数值估计

学生采样$a\sim p$，令$r(a)=\log p_a-\log q_a$，则$\mathbb E_p[r(a)]=\ell_R$。一次采样可得到负值；抽样平均才逼近非负KL。采样已包含$p$权重，再乘$p_a$会错误地引入$p_a^2$。

### 9.2 为什么裸log-ratio反传不对

动作索引固定后，`(logp[a] - logq[a]).backward()`只得到$\nabla\log p_a$，期望为0。完整期望的导数包含两项：

$$
\nabla_\theta\mathbb E_{a\sim p_\theta}[f_\theta(a)]
=\mathbb E_{a\sim p_\theta}\left[\nabla_\theta f_\theta(a)+f_\theta(a)\nabla_\theta\log p_\theta(a)\right].
$$

第一项来自函数值变化，第二项来自采样概率变化。普通索引自动求导漏掉了第二项。

在采样参数$\bar\theta$处，可构造代理loss：

$$
A_b(a,h)=\log\pi_T(a\mid h)-\log\pi_{\bar\theta}(a\mid h),
\qquad
\ell_{\mathrm{PG}}=-\operatorname{sg}(A_b)\log\pi_\theta(a\mid h).
$$

前缀固定、按$\pi_{\bar\theta}$抽动作且$\theta=\bar\theta$时，代理的期望梯度等于局部RKL梯度。$A_b>0$鼓励该动作，$A_b<0$抑制它。复用旧batch后的当前信号与IS见第15节。

### 9.3 K1、K2、K3：值和梯度分别检查

以下动作从当前$p$抽取，索引固定求导，教师冻结，支持完整：

| 名称 | 标量，r为学生减教师logprob | 数值期望 | 裸反传的期望梯度 |
| --- | --- | --- | --- |
| K1 | `r` | RKL | 0 |
| K2 | `r*r/2` | 通常不等于RKL | 局部RKL梯度 |
| K3 | `exp(-r)+r-1` | RKL | FKL梯度 |

K2求导得到$r\nabla\log p$；K3得到$(1-q/p)\nabla\log p$，按$p$平均后为$-\mathbb E_q\nabla\log p$。换log-ratio方向、detach或提案分布后，应重新推导。

### 9.4 学生样本估计FKL

若动作来自$b$且$b$覆盖教师支持，用重要性采样：

$$
\nabla_\theta\ell_F
=-\mathbb E_{a\sim b}\left[\frac{q_a}{b_a}\nabla_\theta\log p_a\right].
$$

代理loss为$-\operatorname{sg}(q_a/b_a)\log p_a$。$b_a$很小会产生大权重；若$b_a=0,q_a>0$，该动作永远采不到，普通IS无法恢复其贡献。

### 9.5 一个token怎样影响整个logit向量

在$p=b$处，一次sampled RKL更新的logit梯度为：

$$
\widehat g_u(a)=\operatorname{sg}(r(a))\big(\mathbf1[u=a]-p_u\big).
$$

$r(a)$只用一个教师概率，但softmax分母使所有学生logits相互耦合。未采动作缺少的是**各自的教师分数**，而不是必然没有梯度。第10.6节给出EOS样本的完整梯度向量。

### 9.6 方差

同一前缀独立采$R$次并取梯度平均，有限方差下各分量方差降为单次的$1/R$。回答中的相邻位置有不同前缀，不属于这种独立重复抽样。全词表精确完成动作期望，消除这一层采样方差；前缀采样误差仍在。

## 10. 全词表、top-k与采样token：监督信息的区别

**三者通常使用同一条学生轨迹，区别在于每个位置比较多少个候选动作**。假设学生这一步生成B：sampled只查询B的教师分数；top-k查询一组候选；全词表比较全部可能的下一token。

| 方法 | 每个位置拿到的教师信息 | 主要特点 |
| --- | --- | --- |
| Sampled-token | 已采动作的logprob；形状`[B,L]` | 数据量小，有采样方差 |
| Top-k | k个ID及logprobs；形状`[B,L,k]` | 信息更多，目标取决于截断和归一化方式 |
| 全词表 | V个token的分布；形状`[B,L,V]` | 当前前缀上的精确条件KL，存储与通信开销大 |

这里“一个token”指每个位置只观察一个动作，并非整条回答只训练一个位置。FKL/RKL与这三种表示可分别选择。

### 10.1 全词表的成本

全词表对$V$项求和。仅一个BF16 logits张量占$2BLV$字节：$B=1,L=4096,V=150000$时约1.229 GB，即1.144 GiB。分块、融合和词表分片可以降低物化开销。

### 10.2 候选集怎么选

教师top-k覆盖教师偏好，学生top-k覆盖学生常选动作，并集覆盖双方分歧。记录$k$的同时记录$P_S,Q_S$，才能知道保留了多少概率质量。取分数必须对齐同一组token ID，不能把双方第1名、第2名直接按排名相减。

生成时的top-k/top-p控制行为分布；loss里的top-k控制监督集合，二者独立。

### 10.3 三种top-k目标

**A. 保留原始项。**$\ell_{F,S}^{\mathrm{partial}}=\sum_{v\in S}q_v\log(q_v/p_v)$。它是部分求和，可以为负，也可能大于完整FKL。

**B. 集合内归一化**。定义$\tilde p_v=p_v/P_S$、$\tilde q_v=q_v/Q_S$，计算$\ell_{F,S}^{\mathrm{renorm}}=D_{\mathrm{KL}}(\tilde q\|\tilde p)$。它只约束候选间比例，丢掉两侧候选总质量的差异。

**C. 加一个tail类别**。把集合外所有动作合并为“其他”，得到：

$$
\ell_{F,S+\mathrm{tail}}
=\sum_{v\in S}q_v\log\frac{q_v}{p_v}
+(1-Q_S)\log\frac{1-Q_S}{1-P_S}.
$$

$$
\ell_{R,S+\mathrm{tail}}
=\sum_{v\in S}p_v\log\frac{p_v}{q_v}
+(1-P_S)\log\frac{1-P_S}{1-Q_S}.
$$

Tail约束外部总质量，但丢掉其内部比例。定义尾部条件分布$p_v^{\mathrm{tail}}=p_v/(1-P_S)$、$q_v^{\mathrm{tail}}=q_v/(1-Q_S)$，$v\notin S$，有：

$$
\ell_F=\ell_{F,S+\mathrm{tail}}+(1-Q_S)D_{\mathrm{KL}}(q^{\mathrm{tail}}\|p^{\mathrm{tail}}).
$$

遗漏项非负，因此粗粒度KL不超过对应full KL。若尾部只有一个token，或双方尾部条件分布相同，遗漏项为0。公式中的条件分布要求相应质量非零。

进一步令$\operatorname{Ber}(a)=(a,1-a)$：

$$
\ell_{F,S+\mathrm{tail}}
=D_{\mathrm{KL}}(\operatorname{Ber}(Q_S)\|\operatorname{Ber}(P_S))
+Q_S D_{\mathrm{KL}}(\tilde q\|\tilde p).
$$

两项分别约束“多少质量放进候选集”和“进去后怎样分配”。RKL对应的头部权重为$P_S$。

### 10.4 EOS反例：候选内相同，整体仍可很差

| Token | 教师q | 学生p |
| --- | --- | --- |
| A | 0.45 | 0.05 |
| B | 0.45 | 0.05 |
| EOS | 0.10 | 0.90 |

只保留A/B并归一化，两边都是$(0.5,0.5)$，KL为0；完整FKL却为$0.8\log9\approx1.757780$。学生仍有90%概率提前结束。

![本文示意：子集归一化隐藏EOS质量差异](./assets/learning/opd-interview-foundations/opd-topk-tail.svg)

上两行展示完整质量，下两行展示归一化后的候选比例；箭头指出EOS质量被丢掉。在这个例子中，tail只有EOS，加入tail即恢复完整目标。单候选归一化更极端：两边都是1，KL恒为0。

### 10.5 随机尾部校正

另一种做法是在候选内精确求和，尾部随机估计。若提案分布$r$覆盖尾部：

$$
\sum_{v\in\mathcal V}f_\theta(v)
=\sum_{v\in S}f_\theta(v)
+\mathbb E_{a\sim r}\left[\frac{\mathbf1[a\notin S]f_\theta(a)}{r(a)}\right].
$$

要恢复梯度，还需正确处理提案、权重与stop-gradient。[TT-OPD](https://arxiv.org/html/2609.34447v2)使用学生采样token校正top-k梯度；[Tail-Aware](https://arxiv.org/html/2608.14728v1)优化合并tail后的目标。前者恢复被省略梯度的期望，后者保留外部总质量。

### 10.6 同一个四token算例

词表为$\{A,B,C,\mathrm{EOS}\}$，教师top-2集合$S=\{A,B\}$：

| Token | 教师q | 学生p |
| --- | --- | --- |
| A | 0.30 | 0.10 |
| B | 0.40 | 0.15 |
| C | 0.20 | 0.25 |
| EOS | 0.10 | 0.50 |

$Q_S=0.70,P_S=0.25$。各目标计算结果为：

| 目标 | FKL方向 | RKL方向 |
| --- | --- | --- |
| 全词表 | 0.516343 | 0.603519 |
| A/B部分求和 | 0.721915 | -0.256986 |
| A/B重归一 | 0.001688 | 0.001677 |
| A/B加tail | 0.447028 | 0.430232 |

重归一损失很小，是因为A/B相对比例接近；它漏掉了0.70与0.25的总质量差异。Tail仍小于full，是因为C/EOS内部比例不同。

Sampled结果随抽样变化：抽到EOS，K1为$\log5\approx1.609438$；抽到A，为$\log(1/3)\approx-1.098612$。按学生概率平均后才得到0.603519。

EOS样本的RKL梯度估计为：

$$
\widehat g(\mathrm{EOS})
=\log5\,(-0.10,-0.15,-0.25,0.50)
\approx(-0.160944,-0.241416,-0.402359,0.804719).
$$

四个logit都有梯度，但只使用一个教师概率$q_{\mathrm{EOS}}$。对全部可能样本的梯度取期望，才恢复full RKL梯度。

### 10.7 候选外的token有没有梯度

固定$S$、温度1，三种FKL的logit梯度分别为：

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

重归一时全词表softmax分母被抵消，外部logit的直接梯度为0；partial保留原始分母，外部仍有耦合梯度；tail根据总质量差调整外部。在上例中，EOS梯度分别为0.35、0、0.30，而full为0.40。空尾部直接回到full；tail外部公式要求$P_S<1$。

令$\ell_{R,S}^{\mathrm{partial}}=\sum_{v\in S}p_v\log(p_v/q_v)$，其梯度还多出未消掉的质量项：

$$
\frac{\partial\ell_{R,S}^{\mathrm{partial}}}{\partial z_u}
=\mathbf1[u\in S]p_u\left(\log\frac{p_u}{q_u}+1\right)
-p_u\left(\ell_{R,S}^{\mathrm{partial}}+P_S\right).
$$

把完整score求和直接截到$S$，与对部分RKL标量求导是两种更新：$\sum_v\nabla p_v=0$只对全词表成立。学生选择top-k时，常见实现固定离散索引，不对排名切换求导。共享网络参数也会让这些局部logit变化相互影响。

### 10.8 接口与开销

准确logprob需要全词表归一化：$\log q_v=z_v^T-\log\sum_u\exp z_u^T$。教师返回一个分数，通常仍要执行模型前向和归一化；省下的主要是结果保留、传输与loss计算。

只给top-k原始logits或候选内概率，无法恢复真实$Q_S$。有全局归一化logprobs才能计算tail；要做学生top-k或并集，还需教师支持指定ID评分。监督项从$V$降到$k$，端到端加速取决于实际瓶颈。

## 11. 局部条件KL、完整序列KL与信用分配

### 11.1 全词表只保证当前前缀的动作求和完整

令$P_\theta(y\mid x)$、$Q(y\mid x)$为学生、教师完整序列分布，终止规则和支持一致，则：

$$
D_{\mathrm{KL}}(P_\theta\|Q)
=\mathbb E_{y\sim P_\theta}\left[\sum_t\log\frac{p_t(y_t)}{q_t(y_t)}\right]
=\mathbb E_{y\sim P_\theta}\left[\sum_tD_{\mathrm{KL}}(p_t\|q_t)\right].
$$

数值可分解为逐token KL；求导时还要计算学生怎样改变未来前缀的访问概率。把采到的历史固定后逐位置反传，省略了这部分导数。

### 11.2 两步反例

第一步以$\alpha=\sigma(\theta)=1/(1+e^{-\theta})$进入A，否则进入B。教师根节点为$(0.5,0.5)$；后续策略固定，A分支条件RKL为$C>0$，B为0：

$$
\mathcal J(\theta)=D_{\mathrm{KL}}((\alpha,1-\alpha)\|(0.5,0.5))+\alpha C.
$$

$\theta=0$时根节点局部KL梯度为0，但完整目标有$C\alpha(1-\alpha)=C/4$的梯度：减少进入A的机会，也会降低后续差异。

### 11.3 即时反馈与未来反馈

令$r_t=\log q_t(y_t)-\log p_t(y_t)$。只用$r_t$更新动作是即时反馈；用$G_t=\sum_{u=t}^T\gamma^{u-t}r_u$则纳入未来，$\gamma\in[0,1]$为折扣。$\gamma=1$对应未折扣累计反馈。[MiniLLM](https://arxiv.org/html/2306.08543v6)讨论了单步与长程优化。

### 11.4 完整序列梯度

令$c_t=-r_t$、$s_t=\nabla_\theta\log p_t(y_t)$。教师冻结、自然终止一致、支持完整且可交换求导与期望时：

$$
\begin{aligned}
\nabla_\theta D_{\mathrm{KL}}(P_\theta\|Q)
&=\mathbb E_{y\sim P_\theta}
\left[\left(\sum_uc_u\right)\left(\sum_ts_t\right)\right]\\
&=\mathbb E_{y\sim P_\theta}
\left[\sum_ts_t\left(\sum_{u=t}^{T}c_u\right)\right].
\end{aligned}
$$

第一行消去了期望为0的直接score项；第二行消去当前score乘过去代价的项。剩下的$\sum_{u=t}^Tc_u$就是未来信用。长度平均、折扣和筛选会改变目标权重。

序列FKL则由教师前缀加权：

$$
D_{\mathrm{KL}}(Q\|P_\theta)
=\mathbb E_{y\sim Q}\left[\sum_tD_{\mathrm{KL}}(q_t\|p_t)\right].
$$

因此，在学生前缀上优化条件FKL，与完整序列FKL通常是不同目标。

## 12. 温度、熵、KL系数与归约

- **生成温度**$\tau_r$：决定实际探索和行为概率。
- **教师温度**$\tau_T$：改变目标分布。
- **学生损失温度**$\tau_s$：改变参与匹配的学生概率。
- **蒸馏系数**$\beta$：缩放蒸馏梯度；reference KL另负责限制漂移。

传统温度KD的$\tau^2$补偿有其梯度推导；混合温度和PG配方应按实际目标计算。RKL已包含学生熵，额外增删熵项会改变目标。

令每条有效位置数$N_i=\sum_tm_{it}>0$，三种归约为：

$$
\mathcal L_{\mathrm{token}}=\frac{\sum_{i,t}m_{it}\ell_{it}}{\sum_iN_i},\quad
\mathcal L_{\mathrm{seqmean}}=\frac1B\sum_i\frac{\sum_tm_{it}\ell_{it}}{N_i},\quad
\mathcal L_{\mathrm{seqsum}}=\frac1B\sum_{i,t}m_{it}\ell_{it}.
$$

Token-mean按位置加权；seq-mean给每条回答相同权重；seq-sum保留累计损失尺度。梯度累积和多卡计算token-mean时，应使用全局分子、分母，并结合DDP的梯度平均规则缩放。

单位置无偏推导假设权重与待抽动作独立。若EOS、成功筛选或随机长度分母与动作相关，聚合后的估计性质需要单独分析。

## 13. 教师接口：能返回什么，就能训练什么

| 接口能力 | 可用监督 |
| --- | --- |
| 完整词表分布 | full，或自行选候选 |
| 指定学生token的logprob | sampled-token |
| 教师top-k的全局logprobs | partial、重归一或tail |
| 仅教师生成文本 | 教师示范SFT；额外设计评价或偏好 |

学生动作不在教师返回的top-k内时，其概率是未知值，不能随意填0或极小常数。应追加评分、扩大返回集合或改用信息足够的目标。

对`[PROMPT,A,B,EOS]`，prompt最后一行预测A，A行预测B，B行预测EOS。教师评分要对齐这个shift，缓存key要包含实际前缀、模型/模板版本、温度与候选事件。

## 14. 最小训练实现：从分布到loss

代码统一使用$p$学生、$q$教师、温度1和公共词表。完整输入检查与可运行实验在[下载脚本](./assets/learning/opd-supervision-lab.py)；下面先展示决定梯度的计算。训练循环调用的`valid_logprobs`、`sampled_opd_pg`也定义在该文件中。

### 14.1 准备概率：有效位置N，词表V

输入logits为`[B,L,V]`，动作mask为布尔`[B,L]`。先取有效行，再做softmax；空mask报错，有效行应为有限logits。

```python
import torch

# Teacher forward runs under no_grad(); only student has gradients.
logp, logq = valid_logprobs(student_logits, teacher_logits, action_mask)
# Both outputs: [N,V]. Teacher logq is detached.
```

`valid_logprobs`内部先按mask取行，再用稳定精度计算`log_softmax`。这可避开无效行上的NaN；末尾乘0无法消除已产生的NaN。

### 14.2 全词表：对词表求和，再对位置平均

```python
def kl_per_row(logp, logq, direction="reverse"):
    logq = logq.detach()
    if direction == "forward":
        return (logq.exp() * (logq - logp)).sum(-1)
    if direction == "reverse":
        return (logp.exp() * (logp - logq)).sum(-1)
    raise ValueError("direction must be forward or reverse")

loss = kl_per_row(logp, logq, direction="reverse").mean()
```

RKL中的`logp.exp()`保持可微。教师detach，学生概率权重与logprob两条路径都反传。

### 14.3 Sampled：取实际动作，构造PG代理

`actions`为`[N]`的学生采样ID，`behavior_logp`已对齐这N个动作：

```python
lp = logp.gather(-1, actions[:, None]).squeeze(-1)
lq = logq.gather(-1, actions[:, None]).squeeze(-1).detach()
lb = behavior_logp.detach()

advantage = (lq - lp).detach()  # Current signal: teacher minus current student.
ratio = (lp - lb).exp()       # Numerator remains differentiable.
loss = -(ratio * advantage).mean()
```

这对应第15节当前信号IS。若改为旧信号`lq-lb`，只在$p=b$处匹配局部RKL；PPO clip与组内归一化是额外设计。这个代理的标量值不是精确KL日志。

### 14.4 Top-k：同一ID上的三种目标

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

`partial`保留原始概率项；`renormalized`在集合内归一；`tail`把外部合并。函数用完整分布计算log-tail，作为正确性参考；真实top-k API应根据返回的全局logprobs计算残余质量。

将整个位置的分布KL detach成常数，再乘单个已采动作的`logp`，不能代替上述分布梯度：状态常数乘score的条件期望为0。

### 14.5 训练循环伪代码：shift与更新

假设教师与学生模板一致、右padding、生成温度1，无top-k/top-p截断。`collect_rollouts`和权重同步由实际框架提供；其余loss函数见下载脚本。`behavior_logp[:,j]`是生成`input_ids[:,j]`时记录的概率。

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

`[:, :-1]`是预测行，`[:, 1:]`是目标token。动作mask随目标移动，保留自然EOS，排除prompt、padding和人工补EOS。这里统一生成完整teacher logits以便对照；生产sampled路径可只返回已采动作分数。

### 14.6 PyTorch KL接口与mean

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

接口计算`target.exp() * (target - input)`，所以FKL的input是学生、RKL的target是学生。对于`[N,V]`，`mean`额外除以V；对`[B,L,V]`用`batchmean`只除以B。显式词表求和、有效位置平均最容易核查。[PyTorch 2.11文档](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html)。

### 14.7 源码对照

[Tinker固定版本](https://github.com/thinking-machines-lab/tinker-cookbook/blob/54b3cb293038ccf2e49952e46ed5abac7119fdf8/tinker_cookbook/distillation/train_on_policy.py)构造teacher/student logprob差，并提供未来累计选项。[verl固定文档](https://github.com/verl-project/verl/blob/3c6d9b43b67c74ec343d77874cd8fa6e841523da/docs/algo/opd.md)区分teacher-top-k FKL、tail与sampled/PG路径。阅读时对照目标、动作ID、detach和归约，不只看配置名称。

## 15. 异步、旧轨迹与重要性采样

### 15.1 修正旧动作采样

固定旧前缀$h$，动作来自$b$。若$b$覆盖当前学生支持，当前局部RKL梯度为：

$$
\nabla_\theta D_{\mathrm{KL}}(p_\theta\|q)
=\mathbb E_{a\sim b}\left[
\underbrace{\frac{p_\theta(a\mid h)}{b(a\mid h)}}_{\rho_\theta}
\log\frac{p_\theta(a\mid h)}{q(a\mid h)}
\nabla_\theta\log p_\theta(a\mid h)\right].
$$

两个学生项都使用当前$p_\theta$：ratio分子和log-ratio信号。它修正同一前缀内动作的抽样，不修正前缀本身的分布$d_b$。[AsyncOPD](https://arxiv.org/abs/2606.24143v1)分别处理了旧动作、当前信号与教师缓存。

### 15.2 旧advantage与监控

复用$A_b=\log q-\log b$，配ratio或PPO clip，得到旧信号surrogate；它与上式当前信号不同。

监控ratio分布、clip比例、数据年龄与任务质量。有效样本量$\mathrm{ESS}=(\sum_jw_j)^2/\sum_jw_j^2$可衡量非负IS权重的集中度。版本落后几步不直接代表概率距离。

## 16. Mask、失败数据与数值稳定

### 16.1 哪些位置训练

| 内容 | 后续可见 | 通常计算动作loss |
| --- | --- | --- |
| 系统指令、用户输入 | 是 | 否 |
| 学生推理、回答、工具参数 | 是 | 是 |
| 工具观察 | 是 | 否 |
| 自然EOS | 终止事件 | 目标包含终止时保留 |
| padding、人工补EOS | 按attention规则 | 否 |

Attention mask管可见性，loss mask管训练位置，终止mask管episode边界。Packed样本还要隔开跨样本attention。EOS与pad共ID时，按真实长度与生成记录做mask，不能只比较token ID。

### 16.2 哪些失败应保留

可恢复的模型错误前缀应保留，教师在这些状态上提供纠偏。工具超时、输入损坏、模板错位属于系统故障，应修复或过滤；教师也无法评价的前缀可降权或丢弃。记录各类保留率，避免筛选后只剩简单题。

### 16.3 数值检查

先选有效行，再算概率与loss；有效行若有NaN/Inf，立即定位来源。`NaN*0`仍为NaN，末尾mask不能修复上游非法运算。[Autograd文档](https://docs.pytorch.org/docs/2.11/notes/autograd.html)解释了这一行为。

概率与归约先用FP32检查。`eval()`关闭dropout等训练行为，`no_grad()`关闭梯度记录。加epsilon、裁剪ratio和loss都会改变原目标；先排除支持、shift、mask和分母错误。

## 17. 学生复读：纠偏信号在哪里

**可以纠正，前提是教师在重复前缀上更支持恢复或结束**。若$p_{\mathrm{repeat}}=0.8,q_{\mathrm{repeat}}=0.1$，FKL对重复logit的梯度为0.7，梯度下降会压低它。

教师自己从头生成不复读，却可能顺着已重复的前缀继续补全；这时OPD可能强化循环。排查第一次进入重复的位置，比较重复、恢复、EOS的师生概率，再检查候选覆盖、采样支持与更新幅度。

可对照SFT冷启动、full/tail监督、小学习率与短更新间隔。生成端的repetition penalty能改变输出，也会改变行为概率，rollout必须如实记录。

## 18. 学生截断：先看停止原因

### 18.1 模型提前选EOS

若$p_{\mathrm{EOS}}>q_{\mathrm{EOS}}$，完整FKL的梯度$p_{\mathrm{EOS}}-q_{\mathrm{EOS}}>0$，会压低EOS。Sampled RKL采到EOS时也可能抑制它。先确认自然EOS参与loss、概率评分正确、监督未漏掉相关质量。

### 18.2 达到长度或上下文上限

这是外部预算终止。增加预算、续写或压缩上下文解决硬限制；OPD可通过减少重复节约长度，也可能学出更长推理。评价时分开记录自然结束率、预算截断率、正确率与长度。

部分rollout仍可提供局部监督；人工补EOS不应当作自然完成的正标签。

### 18.3 Stop或服务配置

错误的stop字符串、EOS ID、模板结束标记和服务超时也会切断输出。检查原始token与`finish_reason`，先修配置再判断模型是否需要训练。

## 19. OPD与reward hacking

**会发生优化信号改善、任务质量变坏的情况**。教师条件概率衡量“这段前缀后常接什么”，不等于答案正确性；错误推理后接一个错误结论，也可能很好预测。

完整、终止一致的序列KL为0意味着学生分布等于教师。但实际loss可能只覆盖部分前缀、top-k或经过clip/mask，KL日志下降未必代表完整分布更接近教师。

[OPD坍塌研究](https://arxiv.org/html/2610.03185v1)在数学任务的sampled/PPO实现中观察到重复和过长行为被强化。排查时同时看独立正确率、重复率、截断率、EOS/head/tail质量，并比较异常轨迹得到的权重。处理可包括SFT起点、更新约束和教师可靠性筛选。

## 20. OPD、SFT、DPO与RL怎样选择

| 方法 | 主要反馈 | 适合先解决的问题 |
| --- | --- | --- |
| SFT/教师答案蒸馏 | 示范token | 已有可靠示范，先建立行为起点 |
| 离线soft KD | 固定前缀的教师分布 | 保留教师多个候选偏好 |
| OPD | 学生前缀上的教师信号 | 学生错误前缀与恢复 |
| 可验证RL | 测试、数学判定等任务结果 | 用可靠结果反馈探索 |
| DPO | chosen/rejected偏好对 | 根据成对偏好调整输出 |

也可组合：

$$
\mathcal L=\lambda_D\mathcal L_{\mathrm{OPD}}
+\lambda_S\mathcal L_{\mathrm{SFT}}
-\lambda_R\mathbb E[R(y,x)]
+\lambda_{\mathrm{ref}}\mathcal L_{\mathrm{reference}},
\qquad \lambda_D,\lambda_S,\lambda_R,\lambda_{\mathrm{ref}}\geq0.
$$

$R$为任务回报，$\lambda$们为非负权重。负号表示最大化回报；教师项提供局部概率指导，任务奖励约束最终结果。每项分别定义采样来源、梯度估计与归约。

教师选择看任务能力、学生前缀可靠性与成本。[Trust Region OPD](https://arxiv.org/html/2606.01249v1)研究了师生分布差异大时的引导问题。

## 21. 跨tokenizer、多教师、自蒸馏与Agent

### 21.1 跨tokenizer

不同token ID和分词边界不能直接对应。先定义公共文本/byte事件及概率，再处理位置对齐、质量守恒、尾部与EOS。见[跨tokenizer实现](#q=opd-interview-cross-tokenizer)。

### 21.2 多教师

同前缀、公共词表，教师$q_j$概率为正，权重$w_j\geq0,\sum_jw_j=1$。自由分布的单状态最优解为：

$$
\min_p\sum_{j=1}^Mw_jD_{\mathrm{KL}}(q_j\|p)
\quad\Rightarrow\quad p_v^*=\sum_jw_jq_{j,v};
$$

$$
\min_p\sum_{j=1}^Mw_jD_{\mathrm{KL}}(p\|q_j)
\quad\Rightarrow\quad
p_v^*=\frac{\exp(\sum_jw_j\log q_{j,v})}{\sum_u\exp(\sum_jw_j\log q_{j,u})}.
$$

FKL对应概率算术平均；RKL对应归一化几何平均。后者更强地抑制某教师极不认可的动作。共享参数学生受容量限制，未必达到该自由分布解。按域路由单教师与合并教师分布是不同方案。

### 21.3 自蒸馏

同模型可以用额外证据、答案上下文或EMA快照当教师。监督来自条件或参数差异；同参数、同输入、同温度的完整分布KL为0。特权信息只供教师使用，学生部署输入仍按真实条件设置。

### 21.4 Agentic OPD

分别记录谁提出动作、谁执行动作推动环境、训练谁。学生token若采在教师或fast-policy推进的历史上，环境访问分布仍来自后者。工具观察不做动作loss；总结、重试和状态恢复要保留实际输入。见[Agentic训练](#q=agentic-opd-training)与[Pipeline](#q=agentic-opd-pipeline)。

## 22. 成本与评估：怎样判断训练有效

### 22.1 成本拆分

学生rollout逐token生成，教师对给定文本可批量评分；因此学生生成可能比大教师评分更慢。分别统计生成、教师前向、learner前反向、同步、工具与排队，GPU-hour计入全部预留设备。

### 22.2 对照方法

使用同一起点、教师和任务划分，对照无训练学生、充分调参的教师SFT、soft KD与OPD。分别按生成token、教师调用和GPU-hour做预算匹配。[多教师基线审计](https://arxiv.org/abs/2610.04272v1)展示了同源模型设置下强离线基线的重要性。

### 22.3 指标与消融

| 问题 | 指标 |
| --- | --- |
| 是否更会做题 | 独立正确率、代码测试、工具成功率、OOD |
| 是否只提高采样效率 | 固定温度/预算下pass@1、pass@k、问题覆盖 |
| 是否生成退化 | 重复率、EOS率、截断率、长度分位数 |
| 概率学到了什么 | 明确定义的KL、熵、EOS、head/tail质量 |
| 更新是否稳定 | 梯度范数、ratio、clip比例、ESS、数据年龄 |
| 是否值得成本 | 总GPU-hour、teacher tokens、有效tokens/s |

消融一次改一个因素：固定前缀比较散度；固定目标比较full/top-k/sampled；固定方法比较tail/EOS、起点、教师与异步程度。

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

## 24. 方法选择：从约束到训练方案

| 当前主要问题 | 优先尝试 |
| --- | --- |
| 学生缺少基本格式或解题起点 | 教师示范SFT，再接OPD |
| 学生常进错误前缀，教师能纠正 | 学生rollout上的OPD |
| Teacher API只支持指定动作评分 | sampled，并检查提案支持与梯度 |
| 候选内loss小，EOS或tail异常 | full小规模对照、显式EOS或tail目标 |
| 结果可可靠验证，教师局部偏好不够 | 任务RL，或与OPD组合 |
| 异步后退化 | 同步对照，再拆旧动作与旧前缀问题 |

## 25. 面试问答：既有24题与题源账本

### 25.1 题源

Q01/Q18的问题方向参考[牛客题解](https://www.nowcoder.com/discuss/933731201871806464?sourceSSR=post)，Q02/Q12参考[Agent题解](https://ricardokevins.github.io/notes/llm-interview-question-bank/chapters/091.html)；[复习知识页](https://lpq29743.github.io/wiki/interview/)为补充。来源于2026-10-12登记，归类为公开题型，其余为自拟追问；技术依据集中在第29节。

### 25.2 定义与方法选择

**Q01 [公开题型·动机] 什么是OPD，为什么不直接做SFT？**

学生生成轨迹，教师评价学生访问的前缀，再更新学生。与教师答案SFT相比，OPD直接覆盖学生自己的错误前缀，训练恢复行为。

**Q02 [公开题型·比较] OPD、RL、DPO和普通KD的区别？**

SFT模仿示范token；soft KD匹配教师概率；OPD将蒸馏放到学生轨迹；RL优化任务回报；DPO利用成对偏好。比较时先说数据来自谁，再说反馈是什么。

**Q03 [自拟·命名] On-policy究竟指哪些分布？**

首先指前缀由当前学生访问。复用旧学生轨迹是近期数据；教师生成历史、学生只在其中采一个局部token，则历史仍来自教师。

**Q04 [自拟·教师] Teacher一定要参数更多吗？**

看目标任务能力和学生前缀上的评分质量。任务专家、额外上下文或EMA快照都可以作为教师；参数规模只是成本和容量因素。

### 25.3 概率、梯度与白板题

**Q05 [自拟·公式] 写出FKL/RKL，并说明哪个分布加权。**

FKL为$\sum_vq_v\log(q_v/p_v)$，按教师加权；RKL为$\sum_vp_v\log(p_v/q_v)$，按学生加权。双方比较同一前缀、同一事件。

**Q06 [自拟·梯度] FKL对student logits的梯度是什么？**

温度1、教师冻结、全词表归一化时为$p-q$。因为FKL等于交叉熵减去固定教师熵。

**Q07 [自拟·推导] RKL对参数的梯度怎样写？**

$\nabla\ell_R=\mathbb E_p[\log(p_a/q_a)\nabla\log p_a]$。推导中出现的常数1由$\mathbb E_p\nabla\log p=0$消去；这是固定前缀的局部梯度。

**Q08 [自拟·陷阱] Sampled KL是无偏值估计，直接backward就对吗？**

裸K1反传只得到$\nabla\log p_a$，期望为0。应加入采样分布的score项，例如用detach的teacher-minus-student logprob作PG权重。

**Q09 [自拟·代码] 写一个能自动求导的固定prefix full-vocab loss。**

沿用第14节函数，mask与预测目标已对齐：

```python
logp, logq = valid_logprobs(student_logits, teacher_logits, action_mask)
loss = kl_per_row(logp, logq, direction="reverse").mean()
loss.backward()
```

**Q10 [自拟·PG] Sampled OPD的一步如何写？**

保存行为$\log b$，重算当前$\log p$，教师给$\log q$。以$\rho=p/b$和$A=\operatorname{sg}(\log q-\log p)$构造$-\rho A$；前提是同前缀和完整行为支持。

**Q11 [自拟·温度] Teacher/student temperature和top-k是一个参数吗？**

生成温度控制探索，教师温度控制目标，学生损失温度控制匹配分布，top-k控制截断或候选。IS分母要用经过真实生成处理后的行为概率。

### 25.4 教师接口与训练数据

**Q12 [公开题型·API] Teacher API不返回完整logits，还能做OPD吗？**

支持指定学生动作logprob就可做sampled；返回全局top-k可做相应候选目标；仅文本可做教师示范。计算full KL需要完整可比较分布。

**Q13 [自拟·shape] Logits和labels为什么要shift？**

因果LM当前行预测下一token。Prompt末行预测回答首token，B行预测其后的EOS；mask跟目标位置移动。

**Q14 [自拟·角色] OPD teacher就是PPO reference吗？**

Teacher提供学习目标，reference限制漂移。两者可恰巧使用同一checkpoint，但损失职责与方向要分别定义。

**Q15 [自拟·缓存] Teacher frozen意味着所有缓存可永久复用吗？**

只可复用相同教师、前缀、模板、温度与事件的分数。Advantage含学生项，学生更新后需按目标决定是否重算。

**Q16 [自拟·mask] 工具文本、思考和动作谁有loss？**

通常学生生成的推理、回答、工具参数参与；工具观察只作为后续可见输入。自然EOS保留，padding与人工插入标记按来源处理。

**Q17 [自拟·数据] 为什么不能用修改后的prompt重算旧logprob？**

改prompt、summary或工具观察就改了条件概率。Ratio比较的是同一历史下当前与行为概率，分母必须保留真实生成条件。

### 25.5 失败轨迹、异步与性能

**Q18 [公开题型·过滤] 错误学生轨迹为什么不全部丢？**

可恢复错误正是OPD的训练对象。把模型失败与工具超时、输入损坏分开；保留有教师纠偏信号的前缀，修复或过滤系统故障。

**Q19 [自拟·异步] 数据落后两版本就安全吗？**

看ratio、ESS、clip比例和任务质量。版本数只计更新次数，无法衡量一次更新导致的概率变化。

**Q20 [自拟·性能] Teacher更大为什么rollout仍可能最慢？**

教师对给定文本批量评分，学生逐token串行生成；多轮还要等待工具和重复prefill。用分阶段耗时定位瓶颈。

**Q21 [自拟·早算] 每turn有teacher信号就能边执行边训练吗？**

局部教师信号可提前得到；终局奖励、组优势或最终分母可能未就绪。区分提前评分、提前backward和提前optimizer更新三种动作。

### 25.6 强基线、证据与展示

**Q22 [自拟·新论文] 新MOPD论文否定了OPD吗？**

它说明特定同源教师/学生设置下，成功教师SFT和独立调参能形成强基线。外推到跨词表或其他任务仍需对应对照。

**Q23 [自拟·实验] 你的对照至少有哪些？**

同起点的未训练学生、充分调参的教师SFT、soft KD、OPD；固定教师和测试集，分别按同生成token与同GPU-hour比较。

**Q24 [自拟·结论] 怎样回答“你的贡献是不是只是工程实现”？**

说清实现解决的具体问题，再给正确性、吞吐、可靠性或可复用性证据。若主张研究创新，还要有新假设和隔离因素的消融。

## 26. 面试问答：从基本概念到估计器、异常和系统的56道延伸

Q25—Q80为按主题排列的速答；完整推导见对应正文。

### 26.1 基本概念与方法选择

**Q25 [自拟·软标签] 教师答案和教师分布传递的信息有什么不同？**

答案提供一次选择，分布提供其他候选及相对偏好。教师采样硬标签可在条件合适时估计软标签梯度，但有限样本有方差。

**Q26 [自拟·价值网络] OPD一定需要critic吗？**

直接KL和即时token PG都不要求critic。使用长程回报、GAE等估计时，再按算法决定价值网络。

**Q27 [自拟·自训练] OPD和直接用学生自己的答案做SFT一样吗？**

学生轨迹提供前缀，教师提供监督；错误动作可被抑制。把学生原答案直接做SFT，则会提高所有已采动作的概率。

**Q28 [自拟·评分] 教师为什么不必逐token重新生成答案？**

学生文本已知，可一次因果前向并行评分所有位置。因果mask与正确shift保证每个概率只条件在过去。

**Q29 [自拟·恢复] OPD是否能回头修改已经生成的错误？**

离散轨迹不会被backward改写。更新改变下一轮生成：少走错误分支，或在错误前缀后恢复；未来后果的归因见第11节。

**Q30 [自拟·公共空间] Teacher和student来自同一家模型就一定能按token ID做KL吗？**

必须核对词表映射、特殊token和版本；同家族也可能扩展词表。模板差异还会改变前缀条件。

**Q31 [自拟·数据] OPD是否需要每道题的标准答案？**

纯概率蒸馏只需问题、学生轨迹和教师评分。成功筛选、任务奖励与准确率评价另外需要标签或验证器。

**Q32 [自拟·探索] 学生完全不会某种格式或知识，OPD能凭空学会吗？**

若有效路径在学生中概率极低，sampled监督很难碰到它。Full可给未采动作自身的教师信息；示范SFT或混合前缀可改善起点。

**Q33 [自拟·GKD] 混合离线和在线数据的参数控制什么？**

$\lambda$控制在线项相对权重。它与KL系数、教师温度、成功样本比例分别调节不同对象；先统一两项归约再比较。

**Q34 [自拟·选择] 什么情况下优先考虑离线教师SFT？**

示范可靠且覆盖任务、生成或环境交互成本高时，先建立教师SFT基线。再用同预算对照判断学生错误前缀监督的增益。

### 26.2 目标、分布和梯度

**Q35 [自拟·交叉熵] FKL和交叉熵数值相同吗？**

$D(q\|p)=H(q,p)-H(q)$。固定教师、固定前缀时梯度相同，数值相差教师熵。

**Q36 [自拟·方向] 写KL时怎样避免把forward和reverse说反？**

先写教师q、学生p，再写FKL为$D(q\|p)$、RKL为$D(p\|q)$。用求和权重的身份检查，避免受论文换字母影响。

**Q37 [自拟·logit梯度] RKL的logit梯度也是学生减教师吗？**

FKL为$p_u-q_u$；RKL为$p_u[\log(p_u/q_u)-\ell_R]$。后者含学生权重和平均log-ratio。

**Q38 [自拟·模式] Reverse KL的mode-seeking是否保证只保留正确答案？**

Mode是教师分布中的高概率区域，可能包含错误答案。学生容量有限时出现模式倾向；容量足够时两类KL的最优解均为$p=q$。

**Q39 [自拟·JSD] JSD怎样连接FKL和RKL？**

与$\alpha q+(1-\alpha)p$比较。原式端点为0；除以$\alpha$后令$\alpha\to0$得FKL极限，除以$1-\alpha$后令$\alpha\to1$得RKL，见8.4。

**Q40 [自拟·温度] 生成温度、教师温度和蒸馏温度怎样分别解释？**

分别改变探索分布、教师目标和学生损失分布。Top-k/top-p还会截断支持，因此行为logprob必须按实际采样规则记录。

**Q41 [自拟·正则] 教师KL系数就是reference KL系数吗？**

蒸馏系数调学习教师的强度；reference系数调允许的策略漂移。它们可对应不同模型与散度方向。

**Q42 [自拟·负权重] 一个batch的OPD权重几乎都为负，还能学习吗？**

可以，平均权重本来就是$\mathbb E_p[\log q-\log p]=-\ell_R\leq0$。不同动作被不同强度抑制，归一化会把概率质量重新分配。

**Q43 [自拟·负KL] 训练日志中的KL为负，必然是bug吗？**

单样本log-ratio、有限样本均值和partial求和可为负。完整归一化KL显著为负时，再检查实现和精度。

**Q44 [自拟·相对信号] 教师给某token高概率，它就一定被鼓励吗？**

看教师相对学生的概率。即使q高，只要学生更高，teacher-minus-student权重仍为负。

**Q45 [自拟·零概率] 能把teacher top-1当one-hot分布直接做RKL吗？**

One-hot在其他动作上为0，而学生通常为正，导致RKL无穷。硬标签交叉熵、平滑或子集目标有各自定义。

### 26.3 采样估计与top-k

**Q46 [自拟·自动求导] 为什么KL数值无偏还不能直接backward？**

采样分布也依赖参数。固定ID自动求导只覆盖函数值导数，漏掉score项；K1的裸梯度期望为0。

**Q47 [自拟·K2] 平方log-ratio是不是精确的KL损失？**

按9.3的定义，K2数值一般不是精确RKL，但当前p采样、固定索引求导的期望梯度是局部RKL梯度。

**Q48 [自拟·K3] 非负低方差KL估计器是否自动得到RKL梯度？**

按9.3的K3定义，值期望是RKL，裸反传期望却是FKL梯度。结论依赖log-ratio方向和detach路径。

**Q49 [自拟·接口] 有teacher top-k，是否就能计算student top-k上的RKL？**

学生候选可能不在教师返回集合，所需q未知。需要指定ID评分、扩大集合或改目标。

**Q50 [自拟·top-1] 为什么某些top-1 KL恒为0，实际库却还能训练？**

单候选双方归一化成1，KL恒为0。实际库可能加tail、用原始质量、硬标签或采样代理，所以仍可训练。

**Q51 [自拟·两层采样] 从教师采监督token，为什么仍可以叫on-policy？**

前缀先由学生生成，教师只另采该前缀下的监督token。若用教师token推进后续历史，后面的状态来源就改变了。

**Q52 [自拟·重归一] Top-k集合内KL很小，能说明整体对齐吗？**

只说明集合内比例接近。第10.4节中A/B归一化完全相同，学生EOS却高达0.9；还要比较$P_S$与$Q_S$。

**Q53 [自拟·tail] 加tail以后是否已经是全词表KL？**

Tail只保留外部总质量。完整KL还含尾部内部条件KL；尾部只有一项或两侧内部比例相同时，这个遗漏项才为0。

**Q54 [自拟·无偏] Top-k加一个学生采样token就一定无偏吗？**

还需正确的随机校正权重、完整提案支持和stop-gradient。简单把样本ID并入候选再归一化，只是在改变候选集合。

**Q55 [自拟·支持集] Top-p抽样后还能用普通IS恢复完整目标吗？**

目标有正概率、提案却为0的动作永远采不到，普通IS不能恢复。可扩大探索支持，或明确改为截断目标。

### 26.4 复读、截断和目标偏离

**Q56 [自拟·复读] 学生已经严重复读，OPD是否还能救回来？**

教师必须在重复前缀上更支持恢复，监督也要覆盖该信号。先查首次进入重复的位置，再对照SFT起点、tail和更新幅度。

**Q57 [自拟·条件评分] 教师自己不复读，为何仍可能强化学生复读？**

教师条件在已经重复的历史上，也可能把继续重复视为高概率补全。要检查实际坏前缀上的分布，而非只看教师从头生成。

**Q58 [自拟·EOS] 提前EOS怎样通过公式判断能否纠正？**

FKL的EOS梯度为$p_{EOS}-q_{EOS}$，为正就压低EOS。再查EOS是否真实采样、参与mask及对应评分。

**Q59 [自拟·预算截断] Max tokens截断为什么不能靠OPD直接修好？**

模型无法突破服务端上限。OPD能改变长度与内容效率，硬限制仍需预算、续写或上下文策略处理。

**Q60 [自拟·错误终止] 学生答案错误，但teacher对最后token概率很高，是否矛盾？**

错误推理后的错误结论可能很容易预测。局部条件概率与最终正确性不同，应检查更早分叉或增加结果验证。

**Q61 [自拟·hacking] OPD没有独立reward model，为什么仍可能reward hack？**

教师概率仍是优化信号。错误前缀补全、top-k盲区、长度权重和有限覆盖，都可能使代理改善而任务质量下降。

**Q62 [自拟·日志] KL下降但正确率下降，该怎样定位？**

先分清日志记录的是哪种KL；再查EOS、tail、重复与截断、动作权重和旧策略偏差。用full或同步小对照定位。

**Q63 [自拟·筛选] 全部失败轨迹都过滤掉是不是更稳？**

会删掉纠偏最需要的错误前缀，也会改变任务难度分布。区分可恢复失败、系统坏数据与教师无法评价的状态。

**Q64 [自拟·裁剪] 把所有负KL裁成0有什么问题？**

完整KL非负，但单样本与partial值可负。将它们裁0会删掉对应梯度，使更新目标变化。

**Q65 [自拟·序列目标] 全词表KL已经精确，为什么仍可能不是精确序列优化？**

Full只精确求当前动作维度的和。若固定采样前缀，仍漏掉改变早期动作所导致的未来状态访问导数。

### 26.5 实现、系统与成本

**Q66 [自拟·batch] 面试让你画一个训练batch，至少有哪些字段？**

实际输入/回答、动作mask、行为版本与logprob、教师分数和候选ID、停止原因；Agent另保存角色、观察与工具结果。

**Q67 [自拟·缓存] 教师不变，为什么advantage不能永久缓存？**

Teacher项可按相同条件缓存；advantage还有当前或旧学生项。学生更新后，它的含义随所选代理变化。

**Q68 [自拟·归约] 为什么梯度累积后结果可能与大batch不同？**

不同长度microbatch各取mean再平均，会改变token权重。先汇总有效loss分子与token分母，再结合DDP缩放。

**Q69 [自拟·瓶颈] 教师更大，为什么profile显示学生rollout最慢？**

参数量不决定流水线耗时。教师批量评分，学生串行生成；工具等待、重复prefill和payload也应分别计时。

**Q70 [自拟·公平成本] 报告OPD快多少，应该算哪些资源？**

计入生成、教师、learner、参数同步、工具和全部预留卡时，并固定硬件、精度、长度、负载和任务协议。

**Q71 [自拟·历史] 旧轨迹换成新的summary再算ratio为什么有问题？**

新summary产生了新条件。旧行为概率只对应当时真实输入；保存实际历史，才能构造同条件ratio。

**Q72 [自拟·部分轨迹] 回答没结束就评分训练，可以吗？**

即时局部OPD可用部分轨迹。混合终局奖励、组优势或提前更新时，应等相关信号和分母就绪；保留真实停止原因。

**Q73 [自拟·LoRA与精度] OPD可以用LoRA、BF16或量化模型吗？**

LoRA限制更新参数，低精度改变数值与内核路径。先核查概率尾部、梯度和兼容性，再评估显存与质量。

### 26.6 扩展、结果解释与项目答辩

**Q74 [自拟·多教师] 为什么平均多个教师的FKL和RKL不等价？**

同状态自由分布最优解分别是概率算术平均与归一化几何平均。后者更强地抑制某教师极不认可的动作，见21.2。

**Q75 [自拟·身份] Behavior、teacher、reference、current分别是什么？**

Behavior生成数据，current接受更新，teacher给目标，reference限制漂移。同步初始时behavior可等于current，更新后便分开。

**Q76 [自拟·跨词表] 跨tokenizer蒸馏最先要定义什么？**

公共文本或byte事件、概率与位置映射。随后检查质量守恒、tail、EOS与截断，详见跨tokenizer专题。

**Q77 [自拟·Agent] 学生采样的token为什么可能不是完整环境on-policy？**

推动环境的可能是教师、fast policy或旧版本。记录proposal、execute、train三种身份，才能判断真实状态分布。

**Q78 [自拟·自蒸馏] 相同模型给自己做老师，为什么可能有学习信号？**

额外上下文、证据或EMA参数可产生分布差异。同参数、同输入、同温度的完整分布则KL为0。

**Q79 [自拟·能力] OPD后的学生能超过教师吗，能否断言只能重排已有能力？**

某个指标可能超过教师，但要固定解码、预算与任务协议。理想分布匹配本身不保证超越，也不证明只能重排已有能力。

**Q80 [自拟·项目] 怎样把OPD项目讲得有技术含量而不夸大？**

讲具体问题、目标与实现变化，再给正确性检查、受控实验和个人分工。源码、公式验算与真实runmanifest分别提供对应证据。

## 27. 面试表达：怎样把主线讲清楚

### 27.1 30秒

OPD让学生生成，教师在学生实际访问的前缀上提供概率监督，学生更新后再采样。它解决教师示范与学生错误前缀之间的分布偏移。前缀来源、KL方向和sampled/top-k/full是三项独立选择。

### 27.2 3分钟

离线教师SFT主要覆盖教师自己的前缀，学生中途出错后缺少恢复示范。OPD让学生先生成，再用教师在这些位置上的概率训练学生。

FKL按教师概率加权，推动覆盖教师认可的动作；RKL按学生概率加权，约束学生常做而教师不认可的动作。全词表直接反传条件KL；sampled需要正确的PG或其他梯度构造；top-k要明确部分求和、重归一还是tail。

实现上保存行为概率，教师同前缀评分，learner重算当前概率；对齐shift、动作mask和全局分母。复读看教师在循环前缀上的恢复信号，提前EOS看结束概率，预算截断先改外部限制。最后用正确率、重复/截断、成本和强SFT基线评估。

### 27.3 10分钟白板顺序

1. 错误前缀例子与生成—评分—更新闭环。
2. 定义p/q/b与固定批次目标。
3. 推导FKL的p−q与RKL的score梯度。
4. 说明K1裸反传问题、PG代理与旧动作IS。
5. 用EOS例子区分三种top-k，再讲tail遗漏项。
6. 连接序列信用、shift/mask、异步和实际异常。
7. 用独立质量、同预算基线和个人runmanifest回答项目追问。

## 28. 小实验、自检与验证范围

### 28.1 标准库：15项公式验算

[脚本](./assets/learning/opd-foundations-check.py) · [结果](./assets/learning/opd-foundations-check-results.json)

```bash
python assets/learning/opd-foundations-check.py
```

检查FKL/RKL梯度、K1/K2/K3、IS支持、EOS反例、tail分解、局部/序列梯度、归约、多教师和shift。

### 28.2 PyTorch：47项反传检查

[完整教学代码](./assets/learning/opd-supervision-lab.py) · [运行结果](./assets/learning/opd-supervision-lab-results.json)

```bash
python assets/learning/opd-supervision-lab.py
```

已在Python 3.12.3、PyTorch 2.11.0的CPU环境运行。包括解析梯度与gradcheck、top-k三模式、teacher/behavior冻结、旧信号IS、极小tail、mask和10万次固定前缀抽样。JSON里的`top2_comparison`保存第10节数值，`three_logit_EOS_traces`保存三logit的20步EOS优化。

### 28.3 真实训练前的检查

先检查一个位置的前缀、动作ID、师生/行为概率；再检查shift、mask、EOS和拆批归约；最后用独立任务评价训练结果。以上实验针对有限分布和合成接线，未执行真实LLM或多GPU训练。[审阅记录](./assets/learning/opd-interview-foundations/review-2026-10-25.md)。

## 29. 原始资料、版本与继续阅读

技术来源与使用的固定版本如下。

| 资料 | 核读基线 | 支撑范围 |
| --- | --- | --- |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531v1) | v1，2015 | 软目标与温度蒸馏背景 |
| [DAgger](https://arxiv.org/abs/1011.0686v3) | v3，2011 | 学习者访问状态上的专家监督 |
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

来源与版本记录见[JSON账本](./research/opd-interview-foundations-sources.json)。

进一步阅读：[OPD快速入口](#q=opd-vs-sft)、[后训练逐题练习Q17](#q=pt-opd-gradient)、[Agentic OPD训练](#q=agentic-opd-training)、[训练Pipeline](#q=agentic-opd-pipeline)、[跨tokenizer实现](#q=opd-interview-cross-tokenizer)、[项目答辩](#q=opd-project-defense)、[后训练方法比较](#q=sft-dpo-rl)。资料与正文核对日：2026-10-25。
