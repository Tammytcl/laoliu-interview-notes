---
title: "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization"
authors: ["Shih-Yang Liu", "Xin Dong", "Ximing Lu", "Shizhe Diao", "Peter Belcak", "Mingjie Liu", "Min-Hung Chen", "Hongxu Yin", "Yu-Chiang Frank Wang", "Kwang-Ting Cheng", "Yejin Choi", "Jan Kautz", "Pavlo Molchanov"]
affiliations: ["NVIDIA", "The Hong Kong University of Science and Technology"]
author_affiliations: [[1, 2], [1], [1], [1], [1], [1], [1], [1], [1], [2], [1], [1], [1]]
venue: "arXiv preprint"
year: 2026
published: "2026-01-08"
summary: "用多奖励手算解释先求和再标准化的信息压缩，核读GDPO的逐维标准化、batch whitening与条件奖励，比较正确率和长度约束的真实取舍。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-08"
template_version: 5
depth_standard: "ddpm"
draft: false
id: "paper-gdpo"
paper_title: "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization"
paper_url: "https://arxiv.org/abs/2601.05242v1"
method_figure: "./assets/papers/paper-gdpo/figure-1a-source.png"
method_caption: "Figure 1a 逐奖励组内标准化与batch归一化"
github_url: "https://github.com/NVlabs/GDPO"
tags: ["GDPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
---

## 1. 背景与已有工作

这篇GDPO是NVIDIA的多奖励RL工作，完整题名为Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization；它不是同名的GFlowNet多样性对齐或Group Direct Preference Optimization。核心问题是：训练同时考虑正确率、格式、长度、代码运行错误时，若先把所有reward相加再做GRPO组内标准化，部分有意义的奖励组合差异会被压缩。GDPO把标准化提前到每个reward维度，再合并优势并做batch归一化。

单奖励推理训练主要关心答案正确，实际产品还有正确的JSON、工具参数、长度限制和不抛异常等要求。常见办法是$R=\sum_kw_kR_k$，然后照常使用GRPO。这看起来只是奖励设计，但std归一化对正比例缩放近似不敏感，且求和丢掉每个维度的组成；“两个目标同时变好”可能和“一个目标变好”获得相同优势。

以两个rollout为例，总奖励$(0,1)$与$(0,2)$的差不同，但各自在组内减均值、除样本std后都得到约$(-0.7071,0.7071)$。这不是计算bug，而是z-score本身消除幅度的性质。单目标中这种尺度不变性可能有利；多目标中它可能削弱区分满足多少要求的信号，是否值得保留由训练目标决定。

[GRPO](#paper=paper-grpo)解决去critic的baseline问题，[DAPO](#paper=paper-dapo)解决探索、有效组、长token权重与截断；GDPO主要改变**多reward优势构造**。它可以沿用DAPO的dynamic sampling、clip-higher和token-mean loss，论文数学/代码实验也确实这样做。因此比较结果不是GDPO与完整DAPO之间的互斥算法排名。

| 做法 | 归一化顺序 | 能保留的信息 | 可能的问题 |
| --- | --- | --- | --- |
| GRPO多reward | 先加权求和，再组内z-score | 当前组总分相对排序 | 维度组成与差值幅度压缩 |
| GRPO去std | 求和，减组均值 | 保留总reward差值幅度 | 仍先聚合维度、尺度与主导reward |
| GDPO | 每reward组内归一化，合并后batch归一化 | 增加不同reward组合的优势分辨率 | 更改尺度权衡，仍可能冲突或抵消 |
| 条件reward | 次目标依赖主目标满足门槛 | 体现任务优先级 | 使信号更稀疏，门槛需设计 |

“保留更多distinct advantage”不等于恢复全部奖励向量或自动求Pareto最优。最终优势仍是一个标量，完全相反的维度变化可能抵消；标量化总需要某种偏好选择。论文的经验收益应放在这个边界内理解。[v1 §3—§4](https://arxiv.org/pdf/2601.05242v1)。

## 2. 方法与实现机制

### 三步优势构造

设$i$是prompt、$j$是同题rollout、$k$是reward维度。传统方法先计算$R^{i,j}=\sum_kw_kR_k^{i,j}$，再对$R$归一化。GDPO改为：

$$
A_k^{i,j}=\frac{R_k^{i,j}-\mu_{i,k}}{\sigma_{i,k}+\epsilon},\qquad
S^{i,j}=\sum_kw_kA_k^{i,j},\qquad
\widehat A^{i,j}=\frac{S^{i,j}-\mu_B}{\sigma_B+\epsilon}.
$$

每个reward的统计只在同题组内计算，合并后统计在batch计算。数值epsilon用于零方差稳定。若某维同组没有变化，其优势为零；增加epsilon不赋予额外信息。权重在归一化之后乘，否则先把某reward整体乘正数再z-score会基本抵消你想表达的权重。

![Figure 1a · 逐reward归一化再合并的原始方法图](./assets/papers/paper-gdpo/figure-1a-source.png)

**Figure1a解读。** 上方GRPO先对总reward减均值除std；下方GDPO把每个reward拆开，先在prompt组内标准化，再求和并用batch-normalization稳定数值范围。右侧$i,j,n,G,B$对应问题、rollout、目标数、组大小和batch。这里的batch normalization是优势whitening，不是给Transformer加入BatchNorm层，也不是学习状态价值。[图源：v1 Figure1a，源码teaser.png](https://arxiv.org/src/2601.05242v1)。

多reward维度增多会改变$S$的方差，取决于各维相关性；单纯求和不保证稳定尺度。batchwhitening帮助稳定步长，但不能修复reward语义错误。归一化统计是否按sequence、validtoken或各rank局部计算，也会决定实际目标；这正是源码伴读需要核对的地方。

### 两个rollout的手算

使用样本std并省略epsilon，组1两回答reward向量为$(0,0)$与$(1,0)$，只有维1发生差异，GDPO合并前优势为$(-0.7071,0.7071)$。组2是$(0,0)$与$(1,1)$，两个维度都发生差异，各贡献同一方向，合并前得到$(-1.4142,1.4142)$。总reward先标准化则两组都压成相同的$(-0.7071,0.7071)$。

![Figure 2 · 原作者枚举的reward组合与优势分组](./assets/papers/paper-gdpo/figure-2-source.png)

**Figure2解读。** 每个格子表示两个rollout的总reward，箭头把组合映射到优势；左侧GRPO只有零组和非零组两类，右侧GDPO区分同时满足两reward的更强信号。图省略最终batchwhitening且是两个binaryreward的特定构造，不能仅看总分就复原任意reward向量。样本std产生0.7071；换populationstd数值会不同。[图源：v1 Figure2](https://arxiv.org/src/2601.05242v1)。

有一个关键边界：若只有一个prompt两条回答，再把这两个$S$做最终whitening，幅度仍可能再次变成同样的正负值。图中的区分对应先合并优势阶段；跨prompt共同做batch归一化才有机会保留组间相对幅度。因此不能说“逐维归一化后任何batch里所有原差异都永远保留”。

再看相同总分的三回答：reward维1为$[1,1,0]$，维2为$[0,0,1]$，总分全是1，GRPO优势为零。两维样本std相同，GDPO等权相加也完全抵消。**这个具体互补例子并不产生非零优势。** 原文关于异质reward组合的一般直觉，需要结合尺度、权重或更丰富组合判断，不能把它转述成所有相同总分都能恢复信号。此处是本文对公式的可复算边界分析，不是新增实验成绩。

### 优势之外的policy loss

GDPO本身不必改tokenratio或clip，通常将新优势代回GRPO/DAPO式surrogate：

$$
J=\mathbb E\left[\operatorname{Reduce}_{i,j,t}
\min\left(r_{i,j,t}\widehat A^{i,j},
\operatorname{clip}(r_{i,j,t},1-\epsilon_l,1+\epsilon_h)\widehat A^{i,j}\right)\right].
$$

这里Reduce需按实验指定：tool与math/code可能不同。GDPO可与sequence-levelratio或softgate组合，但组合后的收益必须重新验证；不能因为修改不同环节就假定可加性或无限堆叠更好。它去critic沿用组方法属性，不是本论文另一个省显存创新。

### 权重调整与条件reward

多目标常有难易差异：让回答短很容易，让答案对很难。即使正确率权重大，模型也可能先利用容易改善的长度分数。GDPO改善归一化信息，不会自动确定“正确性优先”的产品约束。作者比较不同lengthweights，还研究条件长度reward：只有正确性满足门槛，短回答才拿到长度正反馈。

$$
\widetilde R_{\mathrm{len}}=R_{\mathrm{len}}\,\mathbb 1\{R_{\mathrm{correct}}\ge c\}.
$$

这改变reward函数语义，不是单纯换GDPO。对应coding，代码passrate、无runtime/compilationerror与整体任务正确并不相同：程序可以跑而所有测试都错，也可以部分测试成功但任务未完全解决。应记录每维分数、方差、相关性和梯度贡献，不只看求和reward。

### 官方实现的真实限制

固定[NVlabs/GDPO](https://github.com/NVlabs/GDPO)commit `4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115`。论文给出TRL、verl、NeMo-RL三个实现入口，这里沿verl路径核读，不声称三个后端结果都复现过。

[ray_trainer.py的compute_advantage](https://github.com/NVlabs/GDPO/blob/4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115/verl-GDPO/verl/trainer/ppo/ray_trainer.py)在`gdpo`分支分别读取correctness和format的 `[B,L]` reward；对二者调用同一个GRPO归一化函数，合并后`masked_whiten`，返回advantages和returns。这个具体分支注明目前处理两reward，不能把论文任意$n$公式直接说成该文件已完整支持所有维数。

[compute_grpo_outcome_advantage](https://github.com/NVlabs/GDPO/blob/4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115/verl-GDPO/verl/trainer/ppo/core_algos.py)先沿token维求reward，按UID计算groupmean/std，再将sequenceadvantage广播到有效token。reward维在此之前必须仍分别保存；如果只传入已求和score，就无法在函数里重新拆开原始目标。

[masked_whiten](https://github.com/NVlabs/GDPO/blob/4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115/verl-GDPO/verl/utils/torch_functional.py)用mask计算均值与方差，并通过rsqrt归一化。实际统计对象是展开后的有效token；不同长度回答重复其优势不同次数，因此是token加权统计，与简式逐回答batch均值并不在所有长度分布下等价。分布式是否gather完整组和全局batch亦需核配置。本文指出这个实现边界，不将“有源码”当作公式无差异保证。

[作者项目讲解](https://nvlabs.github.io/GDPO/)以两个reward的枚举入手，本文采用这一可手算入口，同时补上batchwhitening和等权抵消的条件。博客与论文同作者同结果，不算独立外部验证。本文只静态核读，未执行代码或训练。

## 3. 实验设置与算力

论文覆盖toolcalling、数学reasoning、codingreasoning三类多reward任务；不同任务的模型、steps、reward范围和长度不同。相比之下，算法对照在每类任务内保持相同配方，因而主要考察优势归一化改动。

| 项目 | v1配置 | 证据与口径 |
| --- | --- | --- |
| Tool模型 | Qwen2.5-Instruct 1.5B/3B | 每配置5runs，汇总mean与曲线median/IQR |
| Tool训练集 | ToolACE2k、Hammer1k、xLAM1k | 共约4k样本，ToolRL配方 |
| Toolreward | format0/1、correctness[-3,3] | 工具名、参数名、参数内容匹配 |
| Tool训练 | 100steps，G4，promptbatch512，response1024 | 正文§4.1 |
| Tool配置 | LR$10^{-6}$，mini128，prompt2048，KL0.001 | AppendixD；epoch15字段与正文100steps分列 |
| 数学模型 | DeepSeek-R1 1.5B/7B系列与Qwen3-4B-Instruct | 原文简写R1不等于671B模型 |
| 数学数据 | DeepScaleR-Preview约40k | 500steps，G16，batch512 |
| 数学/代码长度 | rollouthard8000，lengthreward阈值4000 | 不等于evalhard32k |
| 数学/代码附录 | mini64，ppoepochs1，LR$10^{-6}$，clip0.2/0.28 | DAPO式filter/token-mean，KLloss0.0005 |
| 数学eval | AIME24、AMC22/23、MATH500、Minerva、Olympiad | 每题16samples平均Pass@1 |
| coding | Eurus-2-RL约24k，DeepSeek-R1-7B，400steps | test-casepassrate、条件length、bugreward |
| codingeval | PRIME验证的Apps、CodeContests、Codeforces、Taco | 16rollouts；passrate不是完整题Pass@1 |
| 数学/代码解码 | temperature0.6、top-p0.95、maxresponse32k | 报Exceed阈值仍为4000 |
| GPU型号数量 / precision | 原文未给完整逐实验设备账本 | 不补写A100/H100卡数 |
| 时间 / GPU-hours | 未完整披露 | 图中trainingsteps不是总算力 |
| 代码 | NVlabs固定commit的verl分支 | 静态核读，未训练复现 |

![Table 6 · ToolRL超参原表](./assets/papers/paper-gdpo/table-6-pdf.png)

**Table6解读。** 原表披露batch512、mini128、promptlength2048、G4、LR1e-6与KL0.001。`trainer.total_epochs=15`是公开配置字段，而正文报告100训练steps；二者不应擅自换算或覆盖。表中不包含GPU数量、reward方差定义和所有模型revision，只能作为配置账本的一部分。[表源：v1附录D第21页](https://arxiv.org/pdf/2601.05242v1#page=21)。

BFCL-v3的格式正确率不等于工具任务执行成功率。数学的Exceed表示输出超过4000token的比例；评测允许生成32k，恰好用于观察模型是否遵守软目标，不能把超长看成评测已硬截断。coding的Pass是测试用例通过比例，Bug是runtime或compileerror比例，二者可以不一致。

论文工具曲线给出5runsmedian与IQR，主表是5runsmean，不把IQR当标准差或95%置信区间。数学与代码没有给同样完整的多seed统计，不能把tool稳定性证据无条件推广到全部模型和任务。GPU与总采样预算缺失也限制对“更高算力效率”的定量判断。

## 4. 结果与图表解读

![Table 1 · BFCL正确率与格式正确率的双目标对照](./assets/papers/paper-gdpo/table-1-pdf.png)

**Table1解读。** 1.5B的averageaccuracy从GRPO30.18%到GDPO32.81%，增加2.63个百分点；format76.33%到80.66%，增加4.33点。3B对应39.20%→40.87%和81.64%→82.23%，改善较小。Multi-turn整体accuracy仍仅2.50%/4.59%，不能用更好的格式包装成复杂工具agent能力已解决。[表源：v1第8页](https://arxiv.org/pdf/2601.05242v1#page=8)。

![Figure 4 · 五次run的median与IQR训练曲线](./assets/papers/paper-gdpo/figure-4-source.png)

**Figure4解读。** 左右是format与correctnessreward，横轴100steps，绿GDPO、灰GRPO、蓝GRPO去std。蓝线correctness可以提升，但format接近零；绿线在两目标上更均衡。左图早期IQR很宽，说明format收敛时刻仍有明显run间波动；不能只取绿色median就说完全无方差。[图源：v1 Figure4](https://arxiv.org/src/2601.05242v1)。

![Table 3 · 三个模型的数学accuracy与长度超限率](./assets/papers/paper-gdpo/table-3-pdf.png)

**Table3解读。** 每benchmark分Acc↑/Exceed↓两行；7B在AIME的GRPO50.2%/2.1%，GDPO53.1%/0.2%，quality与长度均改善。但MATH7B accuracy94.1%→93.9%，部分AMC/Minerva也不是GDPO更高。相比原模型AIME55.4%，GDPO53.1%仍有质量损失，换取85.6%→0.2%的超限率下降。它是多目标trade-off，不能概括为所有accuracy都提高。[表源：v1第10页](https://arxiv.org/pdf/2601.05242v1#page=10)。

原表含一个Minerva基线数字的排版错误，本文不擅自重写原图；关键比较使用原文清晰标注的GRPO/GDPO值。正文还使用“up to80% reduction”措辞，具体哪个baseline与百分点/相对比例必须回到行列核对，不将该短语作为跨所有模型平均节省量。

![Figure 5 · 长度reward接近饱和后仍可能出现退化](./assets/papers/paper-gdpo/figure-5-source.png)

**Figure5解读。** 三面板分别为correctness、lengthreward和batch最大response长度；两算法早期都迅速满足容易的长度目标，correctness先降再恢复。约400steps以后灰GRPO正确率下降、最长回答回升，绿GDPO继续改善。中间近满分并不能排除尾部超长，第三图解释为何平均约束分数不足以诊断稳定性。[图源：v1 Figure5](https://arxiv.org/src/2601.05242v1)。

![Table 5 · 两reward与三reward的coding对照](./assets/papers/paper-gdpo/table-5-pdf.png)

**Table5解读。** 各任务同时列Pass、Exceed、Bug；两rewardCodeContests pass63.2%→65.8%，Exceed14.2%→14.3%，并非全部目标都单调改善。三rewardpass同为65.6%，GDPO将Exceed19.3%→15.8%、Bug3.9%→2.5%；Apps三rewardpass68.1%→67.8%也有小幅损失。证据支持更均衡trade-off，不支持“GDPO每一格必胜”或把testcasepassrate说成题目成功率。[表源：v1第12页](https://arxiv.org/pdf/2601.05242v1#page=12)。

## 5. 局限、结论与后续阅读

GDPO让多reward优化显式保留各维统计，然后控制合并优势尺度，在作者任务中往往得到更好的约束与质量平衡。它没有自动辨认reward语义、恢复全部向量信息或解决rewardhacking；低方差维可能被放大、目标冲突可能抵消、容易目标仍可能被优先优化。条件reward和权重是独立且必要的设计。

### 面试问答与追问

**Q1：GDPO与DPO有直接关系吗？** 这里GDPO是GRPO式在线多reward优势改动，不是DPO偏好分类的group版本。先说完整题名才能避免同名论文误读。

**Q2：先求和再标准化为什么会丢信息？** 求和消掉维度组成，组z-score又消掉幅度；两个样本$(0,1)$和$(0,2)$会得到同样正负优势。逐维先标准化能在部分构造下保留同时满足目标的更强信号。

**Q3：归一化前乘rewardweight为什么未必有效？** 对每维整体正比例缩放会被该维z-score抵消；GDPO应对归一化后的优势乘权重。原始reward的门槛或条件化则改变语义，不能通过权重完全替代。

**Q4：batchwhitening是否总保留幅度差？** 它仍会缩放batch，单prompt两样本时可再次抹平幅度；跨prompt才有相对幅度空间。源码按tokenmask统计还会受长度影响，不能照抄图中省略步骤的结论。

**Q5：相同总reward一定能产生GDPO非零优势吗？** 不一定。互补等权binaryreward可能在逐维归一化后完全抵消；标量优势不能无损恢复多维向量。这是很适合手算验证的追问。

**Q6：如何表达正确性优先于长度？** 用正确性门槛条件化长度reward，或明确多目标权衡再调优势权重。只是归一化每维并不会自动知道产品优先级。

**Q7：能与GSPO/SAPO组合吗？** 概念上作用于不同轴，可以做组合实验；但归一化尺度、ratio门控、长度reduction会交互，必须固定其余变量比较。不能仅凭模块不同声称增益相加。

**Q8：实际监控哪些量？** 各维rewardmean/std、零方差组比例、rewardcorrelation、每维优势贡献、正确率与约束的独立验证、长度尾部和GPU-hour。sumreward上升不代表所有约束都改善。

参考[作者项目解释](https://nvlabs.github.io/GDPO/)、[v1全文](https://arxiv.org/pdf/2601.05242v1)、NVlabs固定源码与[统一专题](#report=survey-policy-optimization)。2026-10-08核读方法、任务配方、原图表和verl实现，补充归一化抵消反例及源码长度加权边界；没有执行个人训练复现。
