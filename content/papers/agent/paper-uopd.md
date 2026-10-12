---
id: paper-uopd
title: "UOPD：先由学生提案，按不确定性决定教师接管"
paper_title: "UOPD: Uncertainty-Aware Intervention for On-Policy Distillation of Multi-Turn Agents"
authors: ["Wenbo Zhang", "Pengcheng Xu", "Weizhi Du", "Jing Zhang", "Hengrui Cai"]
affiliations: ["University of California, Irvine", "University of Michigan, Ann Arbor"]
author_affiliations: [[1], [1], [2], [1], [1]]
venue: "arXiv preprint (fixed v1)"
year: 2026
direction: agent
areas: [language]
tasks: [agents, retrieval, reasoning, training-adaptation]
published: "2026-09-27"
paper_url: "https://arxiv.org/abs/2609.34036v1"
github_url: "https://github.com/onepounchman/UOPD"
code_note: "静态核读61f90c5c3f2c75b0f4be2742682baab63b8a8672；源码数值log-ratio限幅与PPO clip分开。"
method_figure: "./assets/papers/paper-uopd/method-source.png"
method_caption: "Figure 1 · 候选评分、教师介入与双分支更新"
evidence: 已核原文
note_ids: [agentic-opd-training, rl-ppo-training]
tags: [UOPD, OPD, Agentic OPD, Teacher Intervention, Uncertainty, SFT]
updated: "2026-10-12"
summary: "按学生候选的不确定性在执行前换教师动作，并分开student PG与teacher SFT；解释门控分布、理论条件、源码和干预率消融。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：不必每一步都让教师执行

多轮OPD中学生错误动作会进入后续状态，造成重复尝试、回退和更长任务。始终让教师替学生执行又失去学生自主状态覆盖，增加调用成本。UOPD先让学生提案，教师给当前候选评分，只在不确定性较高的turn生成教师替代响应并执行，希望把有限干预集中在可能改变长期结果的地方。

这里的不确定性是用于选择时机的信号，不是已验证错误标签。教师低概率的学生动作可能是另一种正确路径；置信差也可能反映模型校准或长度差。因此论文需要比较定向干预与随机干预、是否执行teacher、SFT权重与干预日程，不能只展示一个成功案例。

最容易混淆的是“被提出”“被评分”“被执行”“被训练”四个动作身份。触发时学生提案已经耗费生成，却不进入环境；教师替代动作才推进环境且成为SFT目标。未触发时学生动作执行，并使用OPD更新。轨迹因此是协助执行分布，而非纯学生部署分布。

### 前置知识：序列置信、阈值和混合策略

教师可对学生响应每token给logprob，聚合为confidence；也可比较student与teacher对候选的分数。阈值由最近历史分数的分位数计算，目标干预率随训练调整。同一episode开始时确定阈值并保持，不是每个token都重新算分位数。

目标率不是实际率：分数并列、窗口有限、模型和任务变化会导致偏离。先采学生候选再决定是否换动作的gate依赖候选，因此执行分布一般不是一个独立Bernoulli固定混合；这与Guided-OPD随机选执行者不同。

### 已有工作：OPD与教师辅助

[GKD](https://arxiv.org/abs/2306.13649)与[MiniLLM](https://arxiv.org/abs/2306.08543)提供学生上下文蒸馏基础。TCOD和teacher-prefix路线控制哪些阶段由教师帮助；[Guided-OPD](#paper=paper-guided-opd)按turn随机混合并衰减。UOPD将介入位置与学生候选评分绑定，增加针对性，同时引入选择偏差与门控统计的实现问题。

## 2. 方法与实现机制

### 执行前评分，触发后换掉学生动作

![Figure 1 · 学生提案、教师评分、介入执行与分支loss](./assets/papers/paper-uopd/method-source.png)

**Figure 1解读。** 学生从真实历史生成候选，教师评分，gate决定是否介入。未触发执行学生、走OPD；触发再生成teacher action并执行、走teacher SFT。虚线/分支不要读成两个动作都执行，也不能把被换掉的学生候选当作后续环境历史。图源figures/uai_opd_pipeline.pdf，sections/introduction.tex标签fig:opd_comparison，对应固定v1方法示意。

Teacher-confidence可用负平均teacher logprob作为越大越不确定的$U(h,a_S)$；confidence-gap度量学生与教师对同一候选的差异。符号方向按具体定义核对，不能两个量都叫“confidence”却用同样greater-than逻辑。最近score buffer提供阈值$\tau$，例如目标率$r$取$1-r$分位数。

### 为什么候选门控改变执行分布

记$I(h,a)$为候选是否触发，学生proposal分布$p_S$、teacher替代分布$q_T$。给定历史实际执行动作$a$的分布为：

$$
\mu_G(a|h)=p_S(a|h)[1-I(h,a)]
+\left(\sum_b p_S(b|h)I(h,b)\right)q_T(a|h).
$$

第一项保留未触发学生候选，第二项是所有触发候选的总质量被教师响应重新分配。若$p=(0.2,0.5,0.3)$、$q=(0.5,0.3,0.2)$，只有后两个学生动作触发，则实际率0.8，执行分布$(0.6,0.24,0.16)$；独立0.8概率混合却为$(0.44,0.34,0.22)$。二者不同，不能用目标率直接写固定mixture likelihood。

未触发学生样本也受到gate选择，分布是被过滤的学生proposal；对其做普通PG并不自动是完整未过滤学生分布下RKL的无偏梯度。方法使用这一训练配方，不意味着理论消除了所有分布偏差。评测学生独立执行，才能验证学到的能力而非教师在线救场。

### 理论条件不等于不确定性自动保证

论文讨论平均可恢复性：在介入位置，若teacher action相对student proposal在学生后续策略下的$Q$改善累计平均为正，可得到协助执行的回报改善。用$\eta>0$表示该margin，可概括为：

$$
\frac{\mathbb E\sum_t I_t[Q^{\pi_S}(h_t,a_t^T)-Q^{\pi_S}(h_t,a_t^S)]}
{\mathbb E N_{\rm int}}\ge\eta>0.
$$

这是需要成立的条件，不是“teacher低置信→该动作一定坏”的定理，也不是每次SGD更新学生最终任务回报单调提高。若教师未在该环境学好，或候选低概率只是不同有效行动，条件可能失败。项目应记录介入前后恢复、成功、额外调用和实际率来验证。

### 双分支loss与固定源码

未触发turn保留student sampled tokens，教师在同一prompt/前缀给logprob，形成detach的$A=\lambda(\log q-\log p_{\rm old})$。触发turn以teacher tokens做plain SFT，权重$\beta_{\rm SFT}$。环境观察作为条件，不参与action loss。

官方commit61f90c5c3f2c75b0f4be2742682baab63b8a8672的[uopd_policy_loss.py](https://github.com/onepounchman/UOPD/blob/61f90c5c3f2c75b0f4be2742682baab63b8a8672/trinity/algorithm/policy_loss_fn/uopd_policy_loss.py#L35)直接按行flag路由：

```python
# Official excerpt; omitted comments and blank lines.
log_ratio = torch.clamp(
    logprob - old_logprob, min=-self.clip_log_ratio, max=self.clip_log_ratio
)
ratio = torch.exp(log_ratio)
pg_tok = -advantages * ratio
sft_tok = self.sft_coef * (-logprob)
is_trig = is_triggered.bool().view(-1, 1)
per_tok = torch.where(is_trig, sft_tok, pg_tok)
loss = aggregate_loss(per_tok, action_mask, loss_agg_mode=self.loss_agg_mode)
```

logprob、old_logprob、advantages、action_mask为$[B,L]$，触发flag为$[B]$扩到$[B,1]$。教师SFT不应要求教师采样response有学生behavior denominator；对应pg_tok即使先构造，也在该行被where替代。默认loss聚合token-mean，SFT系数1、log-ratio限幅20。**±20是exp前数值限幅，不是PPO的$[1-\epsilon,1+\epsilon]$ surrogate clip**，不能看到clamp就称标准PPO。

[on_policy_distill_advantage.py](https://github.com/onepounchman/UOPD/blob/61f90c5c3f2c75b0f4be2742682baab63b8a8672/trinity/algorithm/advantage_fn/on_policy_distill_advantage.py#L24)消费old_log_probs与teacher_logprobs，标准分支用response mask，特殊长度对齐分支还支持teacher_valid_mask。原样保存旧采样概率是接线关键，不能用更新后的学生概率覆盖它。

环境workflow则管理学生候选、阈值、teacher replacement与实际执行历史；[ALFWorld workflow](https://github.com/onepounchman/UOPD/blob/61f90c5c3f2c75b0f4be2742682baab63b8a8672/trinity/algorithm/envs/UOPD/alfworld/UOPD_workflow.py)可沿episode入口与生成/执行路径阅读。版本固定用于静态核读，不执行仓库训练。字段名不能替代逐行核执行者，尤其要防止把学生被拒提案写进history。

### Buffer存的是什么

Score buffer只用于近期分数分位数；rollout/experience buffer用于训练样本。二者不能混为replay pool。阈值episode内固定避免执行中突然改判定准则，完成后追加score并给下一episode更新。真实异步还有策略版本控制，与干预阈值完全不同。

保存被拒提案有诊断价值，但它不是实际执行轨迹；保存teacher action用于训练，需要对应真实前缀、模型/模板版本和token mask。若迁移coding，teacher替代操作也要真的修改同一sandbox，再继续后续状态，不能只替文本。

## 3. 实验设置与算力

ALFWorld/WebShop教师为任务GiGPO训练的Qwen2.5-7B-Instruct，学生3B/1.5B-Instruct；Search教师为SearchR1-Qwen2.5-7B-em-ppo，学生1.5B Base。这不是Guided的未做任务RL教师，跨论文不能只按参数大小比较。

| 配置项 | ALFWorld / WebShop / Search |
| --- | --- |
| 训练step | 250 / 150 / 300 |
| 任务batch | 16 / 16 / 64，group1 |
| 优化 | AdamW beta(0.9,0.999)，WD0.01，LR1e-6，constant、无warmup |
| epoch/minibatch | 更新1epoch，minibatch64 |
| grad clip/KL | 1 / 蒸馏系数1 |
| prompt | 2048 / 4096 / 4096 |
| response | 512 |
| turn上限 | 50 / 15 / 8 |
| 历史 | ALF/Web最近两步，Search完整历史 |
| score | ALF/Web teacher confidence，Search confidence gap |
| 3B介入率 | 30%→5%，ALF120/Web75step完成 |
| 1.5B介入率 | ALF/Web 50%→30%，Search固定30% |
| 生成/评测 | 训练T1，ALF/Web评测T0.4，Search greedy |
| 版本 | 至多落后2个版本；具体消费规则以实现为准 |
| 动态token预算 | 16384 / 16384 / 9728每GPU |
| 主计时GPU | 一节点8 RTX PRO 6000 Blackwell |
| GPU分配 | 4 student rollout、2 teacher、2 learner，推理TP2，训练FSDP/Ulysses2 |

附录还记录四卡H100 80GB或RTX A6000 48GB配置，但所有报告训练时间比较使用八卡RTX PRO 6000 Blackwell，不能把计时说成八H100。评测ALF140seen/134unseen，WebShop128test，Search22,523多跳题。Search在8turn用尽且没答案时额外强制一轮回答，仅用于评测；不是训练上限9。

表1为三个**评测seed**的均值±std，不能自动解释成三次独立训练。所有评测使用最终训练step checkpoint，不与其他论文验证最优checkpoint混用。SFT与OPD分支同训练但采样来源不同，token-mean还会改变长度/干预分支的有效权重。

本轮未训练模型，CPU只检查门控分布和局部梯度。论文硬件与任务划分固定，不能根据几个SR差异推导cross-tokenizer适配已验证；共享词表的teacher logprob字段在异词表场景需要新的公共事件或对齐定义。

## 4. 结果与图表解读

![Table 1 · ALFWorld和WebShop主比较](./assets/papers/paper-uopd/table-1-pdf.png)

**Table 1解读。** 3B OPD ALF seen88.1/unseen85.3，UOPD90.0/88.8；Web score77.5→82.4，SR66.9→69.3，turn7.4→6.5。1.5B Web score66.3→76.8即10.5点、约15.8%相对，SR52.3→58.6即6.3个百分点。Teacher行是任务RL教师独立结果，不能把其在线介入质量当学生评测值。±对应评测seed，未形成完整训练方差估计。[表源第9页](https://arxiv.org/pdf/2609.34036v1#page=9)。

![Table 2 · 介入信号、SFT与执行消融](./assets/papers/paper-uopd/table-2-pdf.png)

**Table 2解读。** 默认3B teacher confidence、SFT系数1、30→5。随机介入ALF unseen83.8/Web score77.3低于默认88.8/82.4；去teacher execution Web77.6，说明仅学teacher文本与真实改变训练状态不同。默认并非每列最优：固定15% unseen89.6，50→30 seen90.5，但后者Web score69.9、turn8.2。介入更多并不等于学习更好，SFT系数0/0.5/2也不能简单推单调关系。[表源第11页](https://arxiv.org/pdf/2609.34036v1#page=11)。

![Figure 2 · 固定模型下的一次介入诊断](./assets/papers/paper-uopd/motivation-source.png)

**Figure 2解读。** 前两面板是置信随turn与介入后的变化，后两面板在93个相同ALF任务比较student/random/intervention：SR13.98/16.13/25.81，含回退轨迹93.55/91.40/80.65。这里没有更新学生参数，证明的是定向协助执行改善某组rollout，不是训练后的学生提升。图源四个fig_motivation子资产按原顺序合并，坐标与图例保留，不把它当model training消融。

![Figure 3 · Search多跳问答EM](./assets/papers/paper-uopd/search-source.png)

**Figure 3解读。** HotpotQA/2Wiki/MuSiQue/Bamboogle OPD27.1/23.6/5.2/12.8，UOPD29.8/27.7/9.1/20.8。它扩大任务类型，但student Base、teacher Search-R1、score gap和介入日程与ALF设置不同，不能用一个因素解释所有增益。平均turn另为5.19→3.90，需要结合成功率而非仅少轮数判断效率。图源fig_search_15b_multihop_compact.pdf、fig:search-15b-multihop。

这些结果支持定向介入与学习配方有效，仍没有保证teacher confidence是真错误标签。项目应报告实际干预率、teacher调用量、同teacher数据SFT强基线、相同GPU-hour的独立学生表现，避免只比较固定step。

## 5. 局限、结论与后续阅读

UOPD把教师算力用在候选依赖的少数执行turn，双分支分别监督学生和teacher动作。它主动改变轨迹分布，减少错误累积但引入选择偏差；理论条件需要验证，teacher信号不等于oracle。共享词表、任务RL教师与有限环境使跨tokenizer/coding迁移仍需要新实验。

对照[Guided-OPD](#paper=paper-guided-opd)的独立role选择，[ActFirst](#paper=paper-actfirst-opd)的快动作与完整训练响应，以及[新MOPD基线审计](#paper=paper-rethink-mopd)对SFT和总成本的要求。

**Q01 触发时执行谁？** 教师替代响应，学生候选未执行。

**Q02 gate是独立随机混合吗？** 候选依赖，一般不是。

**Q03 target rate一定等实际rate吗？** 不一定，ties/窗口/分布变化都会影响。

**Q04 阈值每token变吗？** episode开始设定保持，之后更新score buffer。

**Q05 teacher低置信一定错吗？** 不是，校准和多路径都可能影响。

**Q06 ±20 clip是PPO clip吗？** 是log-ratio数值限幅，非PPO ε surrogate。

**Q07 理论保证SGD单调提升吗？** 条件性协助执行分析，不是该保证。

**Q08 主要计时硬件是什么？** 八RTX PRO 6000 Blackwell，另有配置不混用。

**来源。** [固定v1全文与附录](https://arxiv.org/abs/2609.34036v1)、原图表、上述官方固定源码；未模型复现，原图版权归作者。
