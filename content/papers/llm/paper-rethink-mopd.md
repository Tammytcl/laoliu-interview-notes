---
id: paper-rethink-mopd
title: "Rethinking Self-Distillation：多教师OPD与强SFT基线的受控比较"
paper_title: "Rethinking Self-Distillation for Multi-Teacher Capability Merging"
authors: ["Roy Xie", "Dan Friedman", "Feng Nan", "Yukun Huang", "Zhichao Xu", "Chengjiu Zhang", "Jun Xu", "Manaal Faruqui", "Vivek Rathod", "Bhuwan Dhingra"]
affiliations: ["Apple", "Duke University"]
author_affiliations: [[1, 2], [1], [1], [1], [1], [1], [1], [1], [1], [1, 2]]
venue: "arXiv preprint (fixed v1)"
year: 2026
direction: llm
areas: [language]
tasks: [training-adaptation, reasoning, agents]
published: "2026-10-03"
paper_url: "https://arxiv.org/abs/2610.04272v1"
github_url: null
code_note: "v1列出Apple官方仓库，注明仍在legal review；2026-10-12核查公开入口返回404，无法固定作者实现。"
method_figure: "./assets/papers/paper-rethink-mopd/method-pdf.png"
method_caption: "Figure 2 · 共享初始化、专家训练和四种能力整合方式"
evidence: 已核原文
note_ids: [opd-vs-sft, agentic-opd-training, rl-policy-optimization-map]
tags: [OPD, MOPD, Self Distillation, SFT, Model Merging, Fair Comparison]
updated: "2026-10-12"
summary: "用相同提示集、成功教师轨迹和独立超参调优比较四种蒸馏，分清能力恢复、OOD保留与训练成本，并复查四项SFT–OPD比较。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：性能差距来自目标，还是来自配方

一个基础模型分别做数学、指令遵循、工具交互训练，可以得到多个领域专家。部署时希望一个模型同时保留这些能力，于是要把专家能力重新合并。多教师OPD的自然方案是学生自己生成，各领域教师在学生实际到达的上下文上给监督，然后更新学生。多轮agent每轮还要真实执行工具，获得下一轮观察。这样的状态更接近学生部署分布，但会增加持续生成、环境交互与评分成本。

论文问的是：**当成功教师数据和学习率都认真处理后，多教师on-policy蒸馏是否仍然比离线方法更准确？差距需要多少算力？** 它没有提出新OPD loss，而是把配方与证据拆开做比较。主要对象是从同一初始化训练出的专家与学生，即能力整合型self-distillation。“自”不代表教师和学生当前权重相同；教师已经过各自领域RL，学生再从基础checkpoint学习专家能力。

如果SFT包含失败教师样本、只监督最后一轮、学习率未单独调优，再与精调OPD比较，差距不能全部归因于on-policy。反过来，SFT在一组实验接近OPD，也不能证明学生状态监督永远没有价值。本文提供的证据要求同时限定模型、数据、状态访问和预算，而不能把“OPD无用”当作无条件结论。

### 前置知识：提示、轨迹、概率目标和预算

同一道任务可以由教师或学生生成响应；在agent中，不同动作产生不同观察，因此同初始prompt不等于同后续状态。Hard-label SFT监督采样的教师token，soft KD监督教师分布；学生在教师保存的历史上生成一轮新响应，又与学生真实执行整条轨迹不同。token内前缀与跨turn环境状态是两层分布。

Recovery不是绝对准确率，而是相对基础模型恢复了多少专家增益；OOD retention是非专家领域相对基础模型的分数变化。恢复99%增益不等于任务成功率99%。训练GPU-hours、一次性教师数据生成与全部超参搜索成本也不是同一个账本。把这些对象分开，才能正确解释“差不多准确、贵很多”的实验描述。

### 已有工作：蒸馏与参数合并

[GKD](https://arxiv.org/abs/2306.13649)研究学生生成上下文上的监督，[MiniLLM](https://arxiv.org/abs/2306.08543)研究reverse KL蒸馏。本文继承相应目标，重点审计能力整合。作者复查Open-MOPD、SKD、TCOD、ReOPD，但四者的模型与任务不同，不能把四行当统一排行榜。另一条路线[Task Arithmetic](https://arxiv.org/abs/2212.04089)直接合并专家相对共同初始化的参数变化，要求参数兼容，与跨tokenizer概率对齐是不同问题。

## 2. 方法与实现机制

### 一个固定语料，四条训练路径

记基础模型$\pi_0$，领域$d$的专家$\pi_d$，学生$\pi_\theta$。先用专家生成教师轨迹，保留任务验证reward=1的样本，每prompt至多一条，形成固定语料$D_d$。四方法使用完全相同的prompt集合，作者以hash核对集合。SFT/Soft-KD用记录响应；Offline-MOPD用记录历史但新采样学生响应；MOPD只取初始任务，然后真实执行学生轨迹。

![Figure 2 · 多教师能力整合与四种蒸馏的数据来源](./assets/papers/paper-rethink-mopd/method-pdf.png)

**Figure 2解读。** 左到右是基础模型、独立RL专家、成功轨迹筛选和统一学生。底部虚线表示学生也从基础模型初始化，并非接着某一个专家训练。下方四行分别列prompt/prefix、响应来源、是否真实多轮交互：Offline-MOPD响应来自学生，但后续历史来自教师记录；MOPD才执行学生动作并接收新观察。原图由源码内联TikZ绘制，没有独立方法资产，故采用[固定PDF第4页局部裁图](https://arxiv.org/pdf/2610.04272v1#page=4)，保留图注与全部行列。

主实验按已知任务领域选择教师，没有训练自动router。它测的是正确路由条件下的能力整合，未检验领域识别错误；部署项目要额外评估路由、教师冲突和故障回退。

### 三种局部目标与两种学生状态分布

在token上下文$s$上记学生$p=\pi_\theta(\cdot|s)$、教师$q=\pi_d(\cdot|s)$，词表共享：

$$
\ell_{\rm SFT}=-\log p(a),\qquad
\ell_{\rm FKL}=\sum_a q(a)\log\frac{q(a)}{p(a)},\qquad
\ell_{\rm RKL}=\sum_a p(a)\log\frac{p(a)}{q(a)}.
$$

SFT监督记录的教师token；Soft-KD在记录的教师上下文上做forward KL。Offline-MOPD与MOPD均做reverse KL，主要区别是状态形成。单轮任务没有环境历史可回放，两者学生采样与目标一致；不能在单轮任务中解释成完全不同的算法。

MOPD的Eq.4为：

$$
L(\theta;\bar\theta)=\sum_{d=1}^{K}w_d
\mathbb E_{x\sim\widehat P_d,\tau\sim p_d^{\bar\theta}(\cdot|x)}
\left[\frac1{|\tau|}\sum_{t=1}^{|\tau|}\ell_{\rm RKL}(\theta;d,s_t)\right],
\quad \sum_dw_d=1.
$$

$\bar\theta$是rollout时学生，$\theta$是learner当前参数；$|\tau|$只数assistant tokens。环境观察可见但不作为学生输出训练。作者明确固定采样上下文，只对局部loss求导，下一批使用更新后的学生。不能看到期望分布含学生参数，就宣称实现包含完整状态访问分布的梯度；这是一种semi-gradient组织，并非可微环境总导数。

例如教师记录“找杯子→观察在桌上→拿杯子”。Offline-MOPD在教师历史上生成学生响应，即使学生说“去厨房”，下一轮仍读教师记录的“在桌上”；MOPD执行“去厨房”，下一轮观察来自厨房。局部学生token前缀监督和真实错误状态覆盖被分开了。前者节省环境成本，后者可能覆盖部署时错误与恢复，但价值需实测。

### 关键代码：先固定上下文，再计算概率目标

公开作者实现尚不可取得。以下按公式编写的教学代码，**不是作者源码**；其目的是说明张量、梯度与归一化，不替代完整训练器：

```python
# Teaching code: fixed contexts; shared vocabulary.
logp = student_logits.float().log_softmax(-1)  # [B,L,V], has grad
with torch.no_grad():
    logq = teacher_logits.float().log_softmax(-1)
p, q = logp.exp(), logq.exp()
fkl = (q * (logq - logp)).sum(-1)              # [B,L]
rkl = (p * (logp - logq)).sum(-1)
token_loss = fkl if mode == "soft_kd" else rkl
counts = assistant_mask.sum(-1).clamp_min(1)
loss = ((token_loss * assistant_mask).sum(-1) / counts).mean()
```

教师forward无需梯度，softmax用稳定的log空间计算，loss mask为$[B,L]$。最后两行先对每轨迹平均，再对batch平均，通常不同于全batch token平均。这里假定每行是一条完整轨迹、领域采样已完成；若真实训练拆为turn级行，需要按trajectory ID重组才能兑现Eq.4。没有公开源码，不能从公式认定packing、动态batch和全局归一化都一致。

若$p=(0.8,0.2)$、$q=(0.5,0.5)$，RKL约0.192745，FKL约0.223144；两者非负却不相等。教师采样第一个token时SFT为$-\log0.8$，在教师采样条件期望下其梯度对应forward cross-entropy，有限样本不是完整分布监督。学生采样token上的$\log p-\log q$则不能直接autograd后就当作正确RKL梯度；采样分布求导和stop-gradient位置需要明确，详见[Agentic OPD](#q=agentic-opd-training)的小张量推导。

### 参数合并：没有梯度训练，仍有验证选择

Task Arithmetic取$\Delta\theta_d=\theta_d-\theta_0$，再合并：

$$
\theta_{\rm merge}=\theta_0+\sum_dc_d\Delta\theta_d.
$$

参数名、形状和共同初始化必须兼容，不能直接把不同词表或架构相加。TIES等还做符号冲突处理和稀疏化。作者比较11种merge方法，验证集选择混合系数与稀疏度；“训练自由”不意味着零GPU或零超参搜索。

### 指标：恢复专家增益，而不是绝对成功率

定义$\Delta_b(\pi)=S_b(\pi)-S_b(\pi_0)$，每列取最佳教师增益$\Delta_b^\star=\max_d\Delta_b(\pi_d)$：

$$
\mathrm{Rec}=100\frac{\sum_{b\in\mathcal B}\Delta_b(\pi)}
{\sum_{b\in\mathcal B}\Delta_b^\star},\qquad
\mathrm{Ret}=\frac1{|\mathcal B|}\sum_b\Delta_b(\pi).
$$

假设两任务基础分数40/60，教师60/70，学生58/69，则Rec为$100(18+9)/(20+10)=90\%$，并非学生准确率90%。Teacher行各列可能来自不同专家，是组合参考，不是一个真实可部署模型。换教师规模改变分母，因此跨规模recovery不能直接作绝对能力比较。分母接近零时也需检查稳定性。

**代码状态。** v1给出[Apple仓库入口](https://github.com/apple-aiml-research/ml-rethink-mopd)，脚注注明legal review后将发布；2026-10-12公开API返回404，本轮未取得可固定作者代码。这里只用原文§3、附录D/G解释接线，不虚构官方函数。

## 3. 实验设置与算力

MIA组成是数学、指令遵循、InterCode-SQL三专家；Agent×3组成是ALFWorld、WebShop、InterCode-SQL。主设置教师和学生共享初始化、tokenizer和chat template。固定1.7B学生蒸馏1.7B/8B专家是另一组规模对照，不能与主表混用。

| 配置项 | 固定v1披露 |
| --- | --- |
| 模型 | Qwen3-1.7B、Qwen3-8B、Qwen3.5-4B-Base、Ministral-3-3B-Instruct |
| 数学专家 | DAPO，DAPO-Math-17k |
| 指令专家 | GRPO，IF-RLVR multi-constraint |
| Agent专家 | GiGPO，各环境训练数据 |
| 语料 | 每prompt最多一条reward=1教师轨迹，四方法prompt集合相同 |
| 框架 | SFT/Soft-KD用TRL；MOPD两变体与RL用verl，agent用verl-agent；vLLM负责生成/评分 |
| 预算 | 每方法独立调优，五epoch，MOPD每prompt一条rollout |
| LR搜索 | 三值网格重定中心，端点最佳则扩展；每方法/模型/设置至多五值，主要搜索至多160配置，另有seed/补充搜索 |
| 选择 | 各领域未加权平均验证准确率最高checkpoint |
| 随机性 | 蒸馏/Mix-RL三训练seed；基础/教师/merge三评测seed |
| GPU | 每run八张B200；GPU-hours统计八卡占用时间 |
| 8B MIA LR | SFT 5e-6，Soft-KD 6.3e-6，Offline-MOPD 1e-6，MOPD 3e-6 |
| 8B MIA batch | SFT/Soft-KD 32，Offline-MOPD 24，MOPD 96 prompts |
| 8B Agent×3 | SFT/Soft-KD LR 5e-6、batch32；两个MOPD LR 3e-6、batch40 |
| 复现缺口 | 未用相邻论文默认值补齐全套优化器、长度/精度与KL实现细节；代码公开后需核具体接线 |

评测八个单轮benchmark加三个多轮benchmark，ALFWorld再分seen/unseen，因此表内12个测试列、论文称11个benchmark。MATH-500/GSM8K为答案准确率，IFEval/IFBench为prompt-strict，agent环境为任务成功率；AIME24/25是温度0.6的avg@16，**不是pass@16或best-of-16**。

主研究训练、验证、测试独立，而且排除教师RL训练覆盖。InterCode-SQL按数据库拆分：124个训练DB、16验证DB（815条），测试Spider dev 1,034条；ALFWorld验证200，测试seen140/unseen134；WebShop训练目标编号≥1500、验证500–1499、测试0–499。只在蒸馏数据中随机留出prompt不一定能排除教师训练泄漏；作者发现一次指令验证子集与教师训练重合，使教师上限抬高超过40点，影响恢复分母。

训练GPU-hours包含在线学生rollout与教师评分，排除一次性语料生成、验证/测试及其他搜索run。附录另报8B教师语料生成MIA13.55、Agent×3 45.27 B200 GPU-hours，可在$N$次复用run中按$C_{\rm gen}/N$摊销。项目端到端账本还要计教师RL、环境、失败重试、全部搜索和模型选择。正文14.8–23.1倍并非项目总成本比例。

## 4. 结果与图表解读

![Table 1 · Qwen3-8B逐任务能力与OOD保留](./assets/papers/paper-rethink-mopd/table-1-pdf.png)

**Table 1解读。** 上半MIA、下半Agent×3，†/‡标对应设置的ID/OOD。MIA四种蒸馏恢复98.2/97.3/96.9/97.9%；Agent×3为99.0/97.6/97.8/99.3%，没有一个目标稳定全面领先。OOD另读：MIA SFT−3.0、MOPD−0.8点，Agent×3为−2.0/+0.6点；相近ID恢复可以伴随不同能力保留。Teacher逐列取最大，粗体不意味着显著性检验通过。[原表第6页](https://arxiv.org/pdf/2610.04272v1#page=6)。

![Table 2 · 四模型两组组成的恢复率](./assets/papers/paper-rethink-mopd/table-2-pdf.png)

**Table 2解读。** 八个模型–设置组合中，同一模型四种蒸馏最大跨度MIA3.5、Agent×3 4.6个百分点。右侧OOD是八组等权平均，SFT−3.6、MOPD−2.5点，但不表示每一组都更好。Merge在8B两组接近99%，1.7B则86.2/88.3，显示模型规模与任务干扰影响合并。恢复率可超过100，因为学生可超过专家参考增益，并非成功概率超过1。[原表第7页](https://arxiv.org/pdf/2610.04272v1#page=7)。

![Figure 3 · 准确率对训练进度与GPU-hours](./assets/papers/paper-rethink-mopd/cost-source.png)

**Figure 3解读。** 各设置分别画归一化训练进度和GPU-hours，纵轴是领域平均验证准确率，成本横轴为对数，标记为最佳验证checkpoint。最终分数相近，SFT达到该区域所需GPU-hours更少；这不是同step同batch比较。原资产是实际引用的fig_cost.pdf、标签fig:cost；源码包中另有未引用的fig_curves.pdf，其指标和坐标与本图不同，不能误用。图源为[固定源码](https://arxiv.org/src/2610.04272v1)，对应PDF第7页。

![Table 19 · 跑满预算和选中checkpoint成本](./assets/papers/paper-rethink-mopd/table-19-pdf.png)

**Table 19解读。** MIA跑满SFT25.90/MOPD383.32 GPU-hours，比例14.8；Agent×3 18.00/415.80，比例23.1。选中checkpoint成本另为MIA14.57/276.84、Agent×3 4.50/270.27，不能混用这组比例。OPD累计成本按恒定step时间估算$8kT/3600$，复现应核这一计时假设与八卡占用。383.32 GPU-hours对应此八卡配置47.915小时，仍不是全部研究工时。[原表第28页](https://arxiv.org/pdf/2610.04272v1#page=28)。

![Figure 4 · 权重、长度、学习率和成功样本筛选](./assets/papers/paper-rethink-mopd/diagnostics-source.png)

**Figure 4解读。** (a)attention权重相对变化小于0.7%，不意味着全网络行为变化小。(c)仅学习率就使SFT验证分数跨度MIA8.5/Agent×3 11.3点；(d)成功轨迹过滤使SFT恢复提升6.5/7.0点，高于MOPD两变体0.5–1.6点，支持认真调基线。**一致性备注：** (b)Soft-KD等MIA柱有正值，但正文和图注概括“蒸馏缩短MIA响应”，并不完全一致；本报告保留原图，不据概括句断言每种方法都缩短23–29%。(d)纵轴90起、含断轴，视觉柱高不等于绝对增益。[原图第9页](https://arxiv.org/pdf/2610.04272v1#page=9)。

![Table 4 · 四项既有SFT–OPD比较](./assets/papers/paper-rethink-mopd/table-4-pdf.png)

**Table 4解读。** 四行分别是Open-MOPD、SKD、TCOD、ReOPD，各自有published/reproduced列，应在同一复现口径内比RS SFT与OPD，残余差约0.5/2.8/2.8/0.1点。SKD在tuned SFT上做RS反降0.8点，筛选减少样本和prompt覆盖，不总有效。TCOD复现初始SFT已从published28.8变成65.7，其中配方改变包括监督每个assistant turn而非只末轮；不能把全部变化归因于某一个loss。Open-MOPD、TCOD、ReOPD使用reporting benchmarks调参，只有主实验用独立验证；本表是配方诊断，不能当无泄漏新排行榜。[原表第9页与附录A](https://arxiv.org/pdf/2610.04272v1#page=9)。

固定1.7B学生换8B教师的补充实验中，Agent×3 SFT恢复99.3、MOPD73.3，但SFT OOD损失14.8点、MOPD3.3点。更大教师不自动放大OPD优势，ID与OOD有取舍；不同教师规模改变恢复分母，也仍未涉及跨家族或跨tokenizer。

## 5. 局限、结论与后续阅读

本文是强基线审计与成本比较。在1.7B–8B、两组专家组成、共享初始化和tokenizer的范围内，成功教师数据与独立调优使离线方法获得接近MOPD的ID恢复。它没有证明OPD无用、merge能替代一切；较远教师、超大模型、跨词表、其他环境与错误状态覆盖仍需独立验证。平均恢复也可能掩盖单领域退化，预算不包括全部前置与搜索成本。

自己的Agentic OPD至少应比较成功教师SFT、教师历史上的局部学生蒸馏、真实环境轨迹OPD，同时匹配prompt、分别调LR、记录总预算与错误恢复表现。Cross-tokenizer OPD还要分离对齐误差与状态覆盖收益，本文主设置没有该误差源，不能直接为自己的对齐方案背书。

### 八道讲解与项目答辩问题

**Q01 self-distillation为什么不是教师等于学生？** 专家从共同初始化经过领域RL，学生再从基础checkpoint整合能力。

**Q02 Offline-MOPD为什么仍生成学生token？** 响应来自学生，跨turn状态来自记录；采样与执行状态分别定义。

**Q03 SFT等于FKL吗？** 教师采样下条件期望梯度对应forward cross-entropy，有限样本不等于完整soft监督。

**Q04 同prompt为什么还不够公平？** 还要核筛选、mask、模板、LR搜索、长度、验证泄漏与预算。

**Q05 14.8–23.1倍是总项目费用吗？** 是跑满配置的单run训练GPU-hours比例，排除语料生成、评测及其他调参。

**Q06 Teacher行是一个可部署模型吗？** 不是；逐benchmark选择最佳专家形成组合参考。

**Q07 Merge是否零GPU？** 参数构造用CPU，但评测与配置选择仍使用GPU。

**Q08 怎样证明自己OPD增益来自方法？** 强SFT/Soft-KD基线、独立测试和总预算，然后定位状态覆盖、监督或系统效率。

**来源与更新。** 核读[固定v1全文和附录A–H](https://arxiv.org/abs/2610.04272v1)、LaTeX与原图表，核官方入口状态；未执行作者训练或登记个人复现。图表版权归作者。继续读[OPD综述](#report=survey-opd-llm-2026)、[Agentic OPD教学](#q=agentic-opd-training)与[Actor–critic基础](#q=rl-actor-critic)。
