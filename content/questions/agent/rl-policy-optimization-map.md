---
id: rl-policy-optimization-map
title: "从PPO到RLOO GRPO及后续变体的算法地图"
category: agent
difficulty: 进阶
tags: ["P1", "PPO", "RLOO", "GRPO", "DAPO", "GSPO", "GDPO", "SAPO", "CISPO"]
updated: "2026-10-12"
summary: "按反馈、优势、概率比、更新目标、采样和归一化六层比较RL算法，推导RLOO与组均值关系并核关键源码。"
draft: false
---

## 1. 问题背景 新缩写究竟替换了哪一层

“把PPO换成GRPO”“用DAPO”“加GSPO”常把多个改动合成一句话。实际一轮训练包含谁采样、如何给reward、怎样估计优势、用什么概率比、采用什么surrogate、哪些位置训练以及怎样归一化。两份配置都写GRPO，若组统计、std、长度分母、KL或动态过滤不同，梯度也可能明显不同。

本篇接[Actor Critic](#q=rl-actor-critic)、[TD GAE](#q=rl-returns-gae)与[PPO训练一步](#q=rl-ppo-training)，把旧七篇精读和RLOO、Dr.GRPO、REINFORCE++、CISPO放回同一条计算链。单篇原图和实验仍在各自精读；本篇比较机制，不将异构benchmark分数当算法排名。

## 2. 前置知识 六个轴和两个粒度

六个轴分别是**反馈来源、采样方式、优势估计、概率比、更新目标、mask/reduction**。Actor/critic/reference/teacher是角色；token、response、turn、trajectory是粒度。不同论文可能在两个以上轴同时改动，不能只按有没有critic分成所有算法。

完整response的logprob是有效生成token logprob之和；turn归一化/sequence几何平均可能用于稳定和权重选择，但不再是同一个完整动作概率。把原始动作比、长度归一化比和重要性权重叫成同一个ratio，会掩盖目标差异。

离线偏好pair的DPO、可验证reward的在线RL和教师分布蒸馏的OPD反馈不同。一个方法可组合多种信号，必须分别说明目标和系数，不能凭都“更新policy”就判等价。

![本文示意 策略优化六层与不同算法的修改位置](./assets/learning/rl-policy-optimization-map/algorithm-map.svg)

**图解。** 从采样和反馈到优势，再到ratio/surrogate，最后mask与reduction。RLOO/GRPO主要标在baseline，GSPO标在ratio粒度，DAPO跨采样/clip/长度，GDPO处理多reward，DPO/OPD改变反馈与数据。箭头表示依赖，不意味着任意组合都保持原论文理论。

## 3. REINFORCE Actor Critic TRPO PPO的基础链

REINFORCE用return乘score-function梯度；actor–critic用学到的value辅助估计优势；n-step/GAE改变估计；TRPO/PPO改变更新约束与采样复用。这些名字对应不同层，不能说“PPO的全部定义就是有critic”。

原PPO可与GAE常见组合，但也可以用另一种优势；组优势可以配PPO式tokenclip。所谓critic-free通常指没有学习V的网络，不表示没有baseline、没有分数计算或不用多次采样。

[原PPO](#paper=paper-ppo)说明clip四分支与控制实验；[TRPO v5](https://arxiv.org/abs/1502.05477v5)的KL约束不等于PPOepsilon；A2C/A3C进一步改变并行经验和更新组织。复杂系统中参数发布和行为采样版本仍需单独管理。

## 4. RLOO 组均值与GRPO的统计关系

同一prompt采G个独立响应，reward为$R_i$。RLOO的leave-one-out baseline排除当前样本：

$$
b_i^{\mathrm{LOO}}=\frac{1}{G-1}\sum_{j\ne i}R_j,\qquad
A_i^{\mathrm{LOO}}=R_i-b_i^{\mathrm{LOO}}.
$$

给定prompt、冻结策略且独立采样时，其他样本的reward与本动作score条件独立，baseline项期望为0。[Back to Basics v2](https://arxiv.org/abs/2402.14740v2)研究这类REINFORCE式RLHF优化；其结果在相应配置内成立，不证明所有长程agent都应去critic。

组均值$\bar R$包含当前样本，两者关系：

$$
A_i^{\mathrm{LOO}}=\frac{G}{G-1}(R_i-\bar R).
$$

rewards[0,1,3]的mean为4/3，centered=[-4/3,-1/3,5/3]，RLOO=[-2,-0.5,2.5]，恰好放大1.5倍。理想i.i.d条件下，组均值版本的期望score估计含$(G-1)/G$缩放。G固定时有些实现把它吸收进学习率/目标系数；G可变时不能当统一常数。

GRPO还通常除组std：

$$
A_i^{\mathrm{GRPO}}=\frac{R_i-\bar R}{\operatorname{std}(R)+\epsilon}.
$$

分母依赖本组样本，不能继续用“只是固定缩放”的证明。它改变prompt之间尺度，还受到组大小、成功概率、reward组成和std定义影响。Torch默认样本std与population std不同，rewards[0,0,1,1]分别得到约±0.866和±1。

### 全同reward并不是数值爆炸

全组reward完全相同时，centered为0，加epsilon后优势通常仍为0。此时没有组相对信号；不是必然得到无限大优势。接近零variance、稀有成功、不同reward尺度或错误epsilon策略可造成不稳定，具体分开解释。独立KL/entropy等其他loss仍可更新policy。

RLOO要求G≥2才能做这份LOO公式；G1需另一种baseline/估计。某框架在singleton组使用mean0/std1，不等于自动实现了LOO。

### 真实verl接线

固定[GRPO分支](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L314-L329)：

```python title="verl真实节选 std选项与广播mask"
if norm_adv_by_std_in_grpo:
    scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
else:
    scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask
```

scores先由每条response的token reward求和，index区分prompt组，最终[B]系数广播成[B,L]。这不是每token独立judge得分。关闭std仍使用本组mean；若要字面LOO，还应考虑G/(G-1)尺度以及组大小变化。

## 5. Dr.GRPO与REINFORCE++ 归一化怎样改变权重

[Dr.GRPO v2](https://arxiv.org/abs/2503.20783v2)分析response长度分母与reward std的偏置，提出移除相应归一化的配方。要看具体实现是否用固定最大长度/统一常数、每sequence mean或global token mean；“去长度归一化”不等于把所有loss分母彻底删掉。

每条response先除自己的长度，会让长/短样本有相同总权重；global token mean则让有效token多的response贡献更多。哪种更合适取决目标，不能把“同样的训练token”当作梯度相同。概率比中的长度归一化又是另一层。

[REINFORCE++ v9](https://arxiv.org/abs/2501.03262v9)区分基础版与w/ Baseline版：基础版可以每prompt一次采样，使用全局batch归一化；baseline版先减prompt组mean，再采用global normalization，避免把local std当所有任务共同尺度。它们仍要明确KL/reward、PPO式复用、mask和具体配置，不能只叫“把PPO的critic删除”。

全局统计本身也依赖有限样本；white mean/std若来自同批数据，并不保证任意估计都严格无偏。跨GPU局部mean再平均、长短microbatch平均，也未必等于全局有效token统计。数学上去掉某种偏置与系统上正确实现归一化，需要分别验证。

## 6. GSPO DAPO SAPO CISPO 改的是ratio与surrogate

### GSPO 几何平均比不是完整序列importance ratio

GSPO使用：

$$
s_i=\exp\left(\frac{1}{L_i}\sum_t
[\log\pi_\theta(y_{i,t}\mid h_{i,t})-\log\pi_{\mathrm{old}}(y_{i,t}\mid h_{i,t})]\right).
$$

tokenratio[2,0.5]的几何平均为1，算术平均1.25，完整response比为1；若[2,2]，几何平均2而完整比4。长度归一化用于稳定与权重，不能因此称其是完整序列的无偏density-ratio校正。GSPO-token的前向恒等项仍有梯度，原推导和源码见[完整草稿](./paper-drafts/paper-gspo.md)。

### DAPO 不是一个clip参数

[DAPO精读](#paper=paper-dapo)同时包含非对称Clip-Higher、动态过滤补采、token-level loss以及超长reward处理。过滤全同reward组会改变有效prompt分布与采样成本；补采不意味着同样数量的环境任务。Token mean与sequence mean也改变轨迹权重，不能只比较optimizer steps。

### SAPO 平滑目标与梯度gate要分开

[SAPO完整草稿](./paper-drafts/paper-sapo.md)使用平滑sigmoid型surrogate，对正负优势选择不同温度。目标函数f、其导数gate w和最终对logpolicy的导数因子wr是不同量；将w(r)×r当作目标再求导，会引入额外w'项。

平滑不等于任意旧数据都可用，也不保证没有饱和或高方差。原论文缺原表的发布状态继续按既有证据规则处理，读者可下载完整稿学习，正文不以占位图补门槛。

### CISPO 裁importance系数而保留logprob梯度

[MiniMax-M1 v1](https://arxiv.org/abs/2506.13585v1)的CISPO采用clipped IS-weight思路。用教学通式表示：

$$
L_{\mathrm{CISPO}}=-\mathbb E[\operatorname{sg}(\operatorname{clip}(r,l,u))A\log\pi_\theta(a\mid s)].
$$

sg使importance系数不沿自身反传，logpolicy仍可微。与PPOmin平台相比，r越界不必让该token的policy项导数变0；但A=0、mask=0或参数路径无梯度时仍无更新。Clipping仍改变原importance estimator，引入相应偏差，不能说“保留token梯度就严格无偏”。

```python title="教学对照 相同ratio的两种更新"
ppo_loss = -torch.minimum(ratio * adv,
                          ratio.clamp(lower, upper) * adv)
cispo_loss = -(ratio.clamp(lower, upper).detach()
               * adv.detach() * current_logprob)
```

本例只比较局部目标，不填原论文超参或宣称完整MiniMax配方。当前probability、旧数据及采样处理的契约依然必要。

## 7. GDPO DPO与OPD 反馈和目标不是同一轴

[GDPO精读](#paper=paper-gdpo)处理多个reward：先逐维组标准化，再按权重合并，可进一步batch whitening。若先给某一reward乘正数再在该维z-score，权重可能被抵消；正确性/格式等reward冲突时，逐维归一化也不保证永不抵消。等权互补[1,1,0]与[0,0,1]仍可合为0。

[DPO精读](#paper=paper-dpo)从偏好pair与reference logprob构造分类目标，原训练无需在线工具rollout、显式critic或PPOclip。它是另一条学习反馈路线；后续on-policy pair或agent轨迹偏好方法不能直接套原始离线结论。

OPD让teacher在学生访问的prefix上提供概率信息，具体可用不同KL/采样估计；纯OPD可以无任务reward。它与reward-RL可组合，但必须分别核状态来源、KL方向、teacher/learner缓存和mask。[Agentic OPD](#q=agentic-opd-training)详述多轮干预、状态漂移与计算复用。

## 8. 一张统一表怎样使用

| 路线 | 优势/反馈 | 更新或归一化主要变化 | 特别要检查 |
| --- | --- | --- | --- |
| PPO | critic/GAE，任意合适reward | current/old token或action比，clip/KL | bootstrap、版本、critic误差 |
| RLOO | 同prompt其他样本baseline | REINFORCE式，亦可有近策略复用 | G、独立性、LOO尺度 |
| GRPO | prompt组center/std | critic-free组优势广播 | std定义、零方差、组屏障 |
| Dr.GRPO | 组mean，不用local std | 长度/统计尺度改动 | 固定分母与真实loss实现 |
| REINFORCE++ | global normalization，baseline版有组mean | 组/全局统计拆开 | 版本、变体、跨卡统计 |
| GSPO | 常继承组优势 | 长度归一化sequence ratio | 非完整密度比、token实现 |
| DAPO | 组reward | clip/过滤补采/token mean/长度 | 总rollout成本与覆盖 |
| GDPO | 多reward分开标准化 | 权重合并、whitening | 冲突、条件reward、分母 |
| SAPO | 常继承组优势 | 平滑surrogate与导数gate | f/w/wr、温度方向、饱和 |
| CISPO | reward/优势 | clipped权重stop-grad×logpolicy | 权重偏差与梯度路径 |
| DPO | 偏好pair/reference | pair分类目标 | 数据覆盖和reference分数 |
| OPD | teacher分布 | 条件蒸馏与学生访问状态 | tokenizer、KL、梯度估计 |

先把具体配方填进六个轴，再比较另一份配方。相同算法名不保证相同训练；可组合不表示每一组合都保留所有原始性质。

## 9. Agentic场景怎样选对照

带终局任务reward的多轮环境可先用正确状态/边界的PPO或组baseline；若critic成本显著，用RLOO/GRPO等对照，同时记录采样G与组等待。若失败主要来自中间决策credit，增加细粒度估计比替换clip名称更直接；若reward本身不可靠，先改善判据。

比较需固定模型、环境/任务初态、采样温度、turn/token预算、verifier和训练资源，报告task success与每成功任务成本。不同教师、不同prompt、不同测试集或最大长度不构成算法单变量对照。再记录组过滤率、ratio尾部、ESS、mask比例与GPU-hour，追踪性能改善来自哪层。

## 10. 面试问答

**Q01 [分类] 怎么不用缩写比较两算法？** 填反馈、采样、优势、ratio、surrogate、mask/reduction六栏，指出哪些变化及条件。

**Q02 [baseline] LOO与组mean差多少？** G/(G-1)倍；排除自身帮助理想独立样本baseline证明，变量G不能当统一常数。

**Q03 [统计] 去std就等于完整RLOO吗？** 不一定，还看LOO尺度、采样独立性、组大小及更新目标。

**Q04 [零方差] 全组同reward一定爆炸吗？** 正确center/epsilon通常得到零优势；近零variance与其他loss分开处理。

**Q05 [归一化] Global whitening能证明完全无偏吗？** 不能自动，有限批次依赖和跨卡实现仍需核查。

**Q06 [粒度] GSPO的ratio是完整序列概率比吗？** 是几何平均型归一化比，不是完整乘积ratio；用[2,2]可手算区别。

**Q07 [DAPO] 只提高upper clip够吗？** 不等于完整DAPO，过滤/补采、reduction和长度reward也是配方。

**Q08 [SAPO] Gate能直接当loss乘ratio吗？** 不可以随意替换，目标与导数不同，会引入额外导数项。

**Q09 [CISPO] 为什么clipped token仍能有梯度？** 权重stop-grad，logpolicy继续反传；zero advantage或mask仍可无梯度。

**Q10 [GDPO] 多reward为什么先归一化再加权？** 先乘权重再z-score可能把权重抵消，但正确顺序也不保证冲突全消除。

**Q11 [DPO] 它是在线RL的一个clip变体吗？** 原始是离线偏好分类目标，数据与反馈路径不同。

**Q12 [Agent] 有没有一个算法天然解决长程工具训练？** 仍需状态、mask、反馈、版本、环境恢复与预算正确；选对照按实际瓶颈。

## 11. 来源与练习

[七篇算法精读对照](#report=survey-policy-optimization)保留原图、源码与实验条件。[CPU课堂脚本](./assets/learning/agent-training-lab.py)和[结果](./assets/learning/agent-training-lab-results.json)验证LOO缩放、std定义、clip与CISPO局部梯度、全同reward和长度reduction，均为小张量例子。

[来源清单](./research/rl-policy-optimization-map-sources.json)登记本轮新增原始论文；算法名的当前框架支持以固定实现为准，不以二手题库定义。继续读[Agentic RL](#q=agentic-rl-training)，把机制连接到具体环境研究。
