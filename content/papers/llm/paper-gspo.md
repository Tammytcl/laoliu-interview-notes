---
title: "Group Sequence Policy Optimization"
authors: ["Chujie Zheng", "Shixuan Liu", "Mingze Li", "Xiong-Hui Chen", "Bowen Yu", "Chang Gao", "Kai Dang", "Yuqiong Liu", "Rui Men", "An Yang", "Jingren Zhou", "Junyang Lin"]
affiliations: ["Qwen Team, Alibaba Inc."]
venue: "arXiv preprint"
year: 2025
published: "2025-07-24"
summary: "解释GSPO的几何平均概率比、序列级裁剪及GSPO-token梯度接线，辨析MoE routing replay、clip比例和重要性采样无偏性边界。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-08"
template_version: 5
depth_standard: "ddpm"
draft: true
id: "paper-gspo"
paper_title: "Group Sequence Policy Optimization"
paper_url: "https://arxiv.org/abs/2507.18071v2"
method_figure: "./assets/papers/paper-gspo/method-equations-pdf.png"
method_caption: "原文公式5至7 序列目标和长度归一化概率比"
github_url: "https://github.com/verl-project/verl"
tags: ["GSPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
author_affiliations: [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
---

## 1. 背景与已有工作

GSPO 的核心问题是：同一个完整回答获得一个奖励，为什么更新时却让每个 token 使用差别很大的 current/old 概率比？长回答、MoE routing 变化和训练推理数值差异，会使少量异常 token 的权重影响长期训练。Qwen 团队提出以完整回答的长度归一化 likelihood ratio 统一加权和裁剪，使更新单元与序列级 reward 更一致。

它继承 GRPO 的同题组优势，没有重新训练 critic，也没有把 reward 换成人类偏好 pair。变化主要在 **概率比与裁剪粒度**：原始 GRPO 用 token ratio，各 token 独立进入 clip 分支；GSPO 用 sequence ratio，每条回答共享一个权重和裁剪条件。组大小、reward质量、优势是否除std和长度reduction仍是独立配置。

大型 MoE 让问题更显著。给定一个前缀，router 选择部分专家参与运算；参数更新或不同后端会改变专家选择，进而改变 token logprob。旧策略生成的token与新策略重算的token即使相同，其激活路径也可能不同。Routing Replay缓存旧expert选择并在训练侧复用，是GRPO的工程稳定措施；GSPO试图通过序列级权重降低对单token波动的敏感性，减少这类依赖。

但重要性采样的理论表述需要审慎。标准 importance sampling 对相同随机变量使用 target/behavior完整密度比，在支持覆盖条件下具有期望恒等式；单样本估计可能高方差，但“只有一个样本”本身并不证明比值无效。GSPO作者对GRPO提出的批评是一种算法设计论证和经验诊断，不应改写成已经证明所有token级PPO目标都非法。

| 设计轴 | GRPO | GSPO |
| --- | --- | --- |
| reward / baseline | 同题组相对评分 | 同题组相对评分 |
| current/old ratio | 每个token单独计算 | 完整序列比开长度次方根 |
| clip决策 | token独立进入对应分支 | 整回答共享分支 |
| 去掉critic | 是 | 是，非本论文新贡献 |
| token级credit调整 | process等变体可不同 | GSPO-token允许不同优势 |
| MoE稳定措施 | 主对照使用routing replay | 本文设置不依赖routing replay |

这里的sequence ratio是**几何平均token ratio**，不是完整密度比本身，更不是token ratio的算术平均。它降低数值随长度爆炸/消失的风险，同时也改变标准importance sampling估计器。理解这一点才能同时说明它为何实用、哪些“无偏校正”说法不能直接继承。[固定v2 §3—§4](https://arxiv.org/pdf/2507.18071v2)。

## 2. 方法与实现机制

### 从完整回答概率到长度归一化比值

自回归回答概率为 $\pi(y_i\mid x)=\prod_{t=1}^{L_i}\pi(y_{i,t}\mid x,y_{i,<t})$。若直接计算完整序列的current/old比：

$$
\rho_i=\frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}
=\exp\left(\sum_t\Delta\ell_{i,t}\right),\qquad
\Delta\ell_{i,t}=\log\pi_\theta(y_{i,t}\mid h_{i,t})-\log\pi_{\mathrm{old}}(y_{i,t}\mid h_{i,t}).
$$

即使每token都只稍微变化，长序列的连乘也可能产生巨大ratio。例如所有token ratio为1.01，长度1000时完整比约为20959；普通PPO的1.2阈值会与长度严重纠缠。GSPO定义：

$$
s_i=\rho_i^{1/L_i}=\exp\left(\frac{1}{L_i}\sum_t\Delta\ell_{i,t}\right)
=\left(\prod_t r_{i,t}\right)^{1/L_i}.
$$

这个变换统一长度尺度，1000个1.01仍得到1.01。但$\rho^{1/L}$不再是$\rho$，所以一般没有 $\mathbb E_{\mathrm{old}}[s_i f]=\mathbb E_{\theta}[f]$ 的标准无偏换测度恒等式。它是策略优化用的长度归一化surrogate权重，应与严格trajectory importance sampling分开。

![原文公式5至7 · GSPO序列目标 优势与几何平均概率比](./assets/papers/paper-gspo/method-equations-pdf.png)

**原式解读。** 上式给出序列级裁剪目标，中式仍是GRPO同题组标准化优势，下式将完整回答likelihood ratio取长度次方根并转成平均log-ratio。原文没有方法流程图，这里展示其真实算法原式，不能称为作者绘制的架构图。它直接支持“共享序列权重”的解释；三张实验曲线另在结果模块，不冒充方法。[原式来源：v2 §4.1，第3页](https://arxiv.org/pdf/2507.18071v2#page=3)。

### 序列裁剪与梯度

用组优势 $\widehat A_i=(R_i-\mu_q)/\sigma_q$，GSPO最大化：

$$
J_{\mathrm{GSPO}}=\mathbb E\left[\frac1G\sum_i
\min\left(s_i\widehat A_i,
\operatorname{clip}(s_i,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\widehat A_i\right)\right].
$$

忽略clip，单序列梯度为：

$$
\nabla_\theta J_i=s_i\widehat A_i\frac1{L_i}\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}\mid h_{i,t}).
$$

所有token共享$s_i\widehat A_i$，并通过$1/L_i$平均。GRPO则给每token乘自己的$r_{i,t}\widehat A_i$。GSPO不是把token完全合并成一个不可微的字符串；梯度仍经过所有有效action token的logprob，只是权重和clip决策相同。

clip依旧是正优势上平台、负优势下平台的悲观目标。不是“sequence ratio越界就整条回答无梯度”：正优势下如果比值过低，仍有纠正梯度；负优势下比值过高也仍有抑制梯度。在有利方向进入平台时，整条sequence的该surrogate项共享停止条件，才会一起失去这部分梯度。

手算一条2-token回答，两个ratio为2与0.5。几何平均$s=1$，算术平均1.25，完整比也是1。对正优势1、clip0.2，GRPO对第一个token进入上平台，第二个仍贡献0.5；GSPO两token均共享$s=1$的权重并保留梯度。相互抵消让序列平均看起来接近on-policy，但单token变化仍可能很大；这解释了GSPO稳健性与掩盖局部异常的同时存在。

若长序列只一个token ratio很大，其log变化可能被长度平均稀释；若很多token偏离，它们会累积到sequence ratio。对长程agent，回答长度、工具observation mask和把多turn视为一条sequence还是多条segment，会改变实际裁剪单位。这是目标定义问题，不能仅凭框架标签GSPO决定。

### GSPO-token与detach为何不能随意替换

原文§4.3引入允许token优势不同的变体：

$$
s_{i,t}=\operatorname{sg}(s_i)\frac{\pi_\theta(y_{i,t}\mid h_{i,t})}{\operatorname{sg}(\pi_\theta(y_{i,t}\mid h_{i,t}))}.
$$

`sg`是停止梯度。前向数值上分数为1，因此$s_{i,t}=s_i$；后向却通过分子的token probability传递梯度。在相同advantage和相同reduction下，该变体与序列GSPO的理论梯度一致；若每tokenadvantage不同，就能表达process/turn级监督。

log域接线常写成：

```python
# 教学重述；有效action token参与平均
log_ratio = logp - old_logp.detach()
seq_log_ratio = masked_sum(log_ratio) / valid_length
token_surrogate_log_ratio = (
    seq_log_ratio.detach()[:, None] + logp - logp.detach()
)
ratio = exp(token_surrogate_log_ratio)
```

这里第二个差前向为零，梯度为当前token的logprob梯度。若只写`exp(seq_log_ratio.detach())`，会完全切断actor梯度；若保留sequence ratio的梯度再额外乘token surrogate，又可能重复路径。复现时应对前向数值、梯度系数、优势是否broadcast和sequence均值逐项核对。

### MoE routing与训练推理差异

原文routing replay保存旧策略activated experts，当前策略计算importance ratio时复用其选择，使激活网络更一致。它增加状态、通信与内存，也可能限制新router探索。GSPO在论文MoE实验中无需这一策略，是重要工程收益，但不是宣称MoE routing问题从此完全消失。

推理引擎返回的logprob还受temperature、top-p、精度、kernel、tokenization和权重版本影响。平均log-ratio可以降低独立局部噪声，却无法消除每token同方向的系统性偏差。如果每token都有常数偏差$c$，平均后仍是$c$。因此“更容忍差异”与“可以无条件取消训练侧重算”不是同一句话；原文将后者主要作为潜在基础设施简化方向。

### 维护者实现伴读

论文未提供一套可核对全部Qwen3实验的完整专属训练仓库；正文伴读公开维护者verl实现，固定commit `75879f7f475fd6b64c779f7d9212e45503f58b8f`。[core_algos.py](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py)。

`compute_policy_loss_gspo`输入 `[B,L]`新旧logprob、优势和mask，先按有效长度平均log-ratio，再用detach接线广播回token矩阵，取exp构造两支clip loss。实现还对log ratio上界进行数值clamp，这不是论文定义的理论clip范围，也不能与$\epsilon$混淆。其可选`rollout_is_weights`是额外校正接口，不是原GSPO几何平均式的一部分。

`agg_loss`的`seq-mean-token-mean`先回答内平均，再在全局sequence上平均，对应原定义；换`token-mean`会按长度改变权重。`compute_grpo_outcome_advantage`仍负责同题组优势，因此GSPO的loss函数不独立决定reward与baseline。本文静态核读两个主要函数和优势入口，不声称复现作者私有训练系统。

[Qwen作者博客](https://qwenlm.github.io/blog/gspo/)提供“优化单位与reward单位一致”的讲解顺序，本文吸收其直观入口，同时把严格密度比、几何平均surrogate和实现容忍度分开。作者博客与论文都描述相同实验，不当作两份独立复现证据。

## 3. 实验设置与算力

GSPO短论文主要用曲线说明大规模训练行为，没有给出完整实验配置表或逐模型GPU-hour。阅读时应同时记录已公开参数与缺失参数；不能把Qwen3模型技术报告的整个资源预算当作这次controlled对照。

| 项目 | v2披露 | 解释与限制 |
| --- | --- | --- |
| 模型 | Qwen3-30B-A3B-Base的cold-start微调checkpoint | 30B总参数、约3B激活；不是3B dense |
| baseline | GRPO，配备routing replay | 主结果不是未稳定化的GRPO |
| 优势 | 同题组reward标准化 | 原文未完整披露G及所有scorer实现 |
| rollout组织 | 每批分成4个mini-batch更新 | 会产生同批数据内off-policy偏离 |
| GSPO clip | lower $3\times10^{-4}$，upper $4\times10^{-4}$ | 对sequence ratio；原文非对称范围 |
| GRPO clip | lower0.2、upper0.27 | token ratio，不能直接复用为GSPO |
| AIME24 | 平均Pass@1，32 samplings | 不是至少成功一次的Pass@32 |
| LiveCodeBench | 202410—202502范围，8 samplings平均Pass@1 | 区间与后续LCB版本不能混用 |
| CodeForces | Elo Rating | 不是百分比accuracy |
| 扩展过程 | 增加训练compute、刷新query、延长generation | 并非所有阶段数据和长度固定 |
| LR / batch / optimizer / seeds | 未给出完整可执行账本 | 不能自行补入常用AdamW参数 |
| GPU / 数量 / precision | 未披露此对照完整配置 | 不能由MoE规模推定训练卡数 |
| 墙钟 / GPU-hours | 未标注可绝对换算的compute横轴 | 曲线支持相对观察，无法复算加速倍数 |
| 完整配置表 | 原文无实验/结果表 | 只展示真实曲线，不制造作者表格 |
| 代码 | 现代verl固定commit伴读 | 不等于作者全部历史实验实现 |

clip量级相差约三数量级，原因是度量的随机变量不同：token比值可以大幅波动，平均log-ratio的sequence比值更集中。不能把同一epsilon作为“公平统一参数”强塞两方法；应按各自度量调节并报告KL、clip fraction和任务收益。

AIME每题多次采样取平均用于降低单次评测噪声，LiveCodeBench同理。CodeForces的Elo随评测题集合和评分规则变化，不能与AIME百分比求一个无定义的“平均分”。MoE还要分清总参数存储与每token激活计算，省routing replay状态不意味着权重显存变成只有3B。

原文结果曲线在不同阶段刷新queries和增加length。公平评估自己的实现时，应固定相同初始化、queries、reward、长度阶段、采样预算、updates-per-rollout和是否routing replay，再比较相同GPU-hour下的quality。仅凭作者没有标数字的compute轴，不应估计最低复现算力或宣称某固定倍率。

## 4. 结果与图表解读

![Figure 1 · Training reward与数学代码评测的compute曲线](./assets/papers/paper-gspo/figure-1-source.png)

**Figure 1解读。** 顶部是training reward，下方三个面板是AIME24、LiveCodeBench和CodeForces，横轴均为training compute。红色GSPO在较少compute达到蓝色GRPO加routing replay的相近水平，支持本文设置中的训练效率和稳定性收益。横轴没有可换算的绝对GPU-hour刻度，曲线端点也不同，因此不从图中报“快了某倍”；约80分AIME的视觉读数也不伪装成作者精确表格值。[图源：v2 Figure1，四面板原资产](https://arxiv.org/src/2507.18071v2)。

这组结果不是只比较GRPO崩溃与GSPO不崩溃。主对照已使用routing replay，使GRPO可以正常训练，GSPO仍展示较好的compute收益。新query、长generation和延长训练共同参与scale-up，不能把整条曲线收益都归因于任何单个超参。

![Figure 2 · 被裁剪token比例的反直觉对照](./assets/papers/paper-gspo/figure-2-source.png)

**Figure 2解读。** 横轴为clipping fraction，GSPO0.15、GRPO0.0013，相差约115倍，是两数量级。GSPO按sequence一起裁剪，主实验仍更有效；这说明高clip fraction不自动等于样本利用失败。该比例不是“15%的GPU算力浪费”，也不是所有越界token最终总梯度为零，其他loss与clip分支方向仍需区分。[图源：v2 Figure2](https://arxiv.org/src/2507.18071v2)。

它并不单独证明所有被保留的GRPOtoken都是噪声，也不支持把clip比例调到越高越好。算法ratio尺度不同，比例的统计对象和曲线quality必须一起解释；单监控clip fraction会错过reward hacking和局部token异常。

![Figure 3 · GRPO在MoE上是否使用routing replay的消融](./assets/papers/paper-gspo/figure-3-source.png)

**Figure 3解读。** 横轴training compute，纵轴training reward；紫色GRPO配routing replay持续改善，橙色无routing replay后期下降。这验证作者MoE设置中的routing重要性，是GRPO内部工程消融，不是同一图中GSPO对GRPO的直接对照。结合Figure1才可说GSPO省去这一特定依赖且保持有效训练，不能说所有MoE都必须routing replay或GSPO永不失败。[图源：v2 Figure3](https://arxiv.org/src/2507.18071v2)。

原文三张图构成“质量/compute、clip统计、routing机制”三类证据。缺少完整超参表、seed不确定性和绝对成本意味着结论强度应停留在作者设置中的经验优势。原文全部实验图已解读，没有拿其他论文表格替代此处缺失的原表。

## 5. 局限、结论与后续阅读

GSPO把概率比和clip移到sequence层，在特定MoE大规模RL上更稳定，也便于减少routing replay工程依赖。长度归一化避免完整比值爆炸，但改变严格换测度含义；平均可能稀释局部异常，共享clip可能丢掉仍然有用的token信号。它仍依赖组奖励、行为概率与训练数据覆盖。

### 面试问答与追问

**Q1：GSPO与GRPO的最小区别是什么？** 组优势基本不变，token ratio改为几何平均sequence ratio，clip单元改为回答。追问应推导两个无clip梯度的权重，而不是只说一个group、一个sequence。

**Q2：sequence ratio为什么开长度次方根？** 完整回答概率连乘会随长度强烈放大变化；取根相当于平均log-ratio，使尺度更稳定。追问：开根后是否仍是无偏importance sampling？一般不能继承完整密度比恒等式。

**Q3：GSPO-token里的detach只是省内存吗？** 不是。它保留sequence ratio的前向值，同时构造token级梯度路径。完全detach会没有梯度，不detach可能改变估计器；必须同时检查值和导数。

**Q4：更高clip fraction为何还更有效？** token权重噪声和每条样本有效信息量影响训练，不能只按保留token个数衡量sample efficiency。图中GSPO与GRPOratio定义不同，比例高低不是统一质量判据。

**Q5：GSPO能完全解决MoE的不稳定吗？** 原文解决该实验对routing replay的依赖，但不能保证所有routing、precision和长程agent设置稳定。应核新旧模型激活路径、实际行为logprob、长度阶段与reward。

**Q6：能否直接用推理logprob？** 可以作为需验证的工程路线；sequence平均更容忍局部差异，不消除系统性偏差或错误behavior分母。要分别比较reference、old、rollout后处理和当前模型的概率定义。

**Q7：它适合process reward吗？** 原始GSPO为同回答优势，GSPO-token允许token优势不同；实际多turn要定义action mask、segment长度与reduction。只是打开GSPO开关并不自动完成细粒度credit设计。

**Q8：sequence ratio接近1能证明所有token都on-policy吗？** 不能。ratio2与0.5几何平均为1，局部变化仍很大；相互抵消和长序列平均都可能隐藏outlier，应额外看tokenratio分布与logratio方差。

参考[作者博客](https://qwenlm.github.io/blog/gspo/)、[固定v2全文](https://arxiv.org/pdf/2507.18071v2)、现代verl固定源码。后续读SAPO对soft gate的改动及[统一专题](#report=survey-policy-optimization)。2026-10-08完成方法原式、全部结果原图与实现伴读；未执行训练复现。原文无表，按现有仓库证据规则保留草稿，等待本轮关于原始曲线替代表格的处理选择。
