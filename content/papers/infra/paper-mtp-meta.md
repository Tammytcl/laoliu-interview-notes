---
id: "paper-mtp-meta"
title: "Better & Faster Large Language Models via Multi-token Prediction"
paper_title: "Better & Faster Large Language Models via Multi-token Prediction"
authors: ["Fabian Gloeckle", "Badr Youbi Idrissi", "Baptiste Rozière", "David Lopez-Paz", "Gabriel Synnaeve"]
affiliations: ["FAIR at Meta", "CERMICS, Ecole des Ponts ParisTech", "LISN, Université Paris-Saclay"]
author_affiliations: [[1, 2], [1, 3], [1], [1], [1]]
venue: "ICML 2024"
year: 2024
direction: "infra"
areas: ["language"]
tasks: ["training-adaptation", "text-generation"]
published: "2024-04-30"
method_figure: "./assets/papers/paper-mtp-meta/figure-1-source.png"
method_caption: "Figure 1 并行多头训练及模型规模下的MBPP结果"
paper_url: "https://arxiv.org/abs/2404.19737v1"
github_url: null
code_note: "官方研究模型card可读取，模型源码需Hugging Face访问授权，本轮未核原模型代码；原文反传调度与本文CPU教学实现分开说明。"
evidence: "已核原文"
note_ids: ["infra-mtp-overview", "infra-mtp-training", "infra-mtp-speculative", "infra-mtp-serving"]
tags: ["MTP", "Multi-Token Prediction", "Self-Speculative Decoding", "Auxiliary Training"]
updated: "2026-10-09"
summary: "从共享主干的独立未来头解释MTP的训练信号和逐头反传，再区分参数匹配质量结果、greedy自投机加速与实际训练开销。"
template_version: 5
depth_standard: "ddpm"
draft: false
---

## 1. 背景与已有工作

### 问题背景

普通自回归语言模型通过每位置的下一token交叉熵学习。这个目标简单、可并行训练，但很多局部过渡很容易，模型可能没有被直接要求在当前前缀表示中保留更远的决策信息。论文提出：同样的训练文本和参数预算，若每个位置同时预测多个未来token，能否更高效地学到代码、规划与生成能力？辅助预测还能否用于自投机解码，摊薄昂贵模型调用？这两种收益分别需要质量与速度证据。

### 前置知识

teacher forcing在训练时使用真实前缀，让全序列位置并行计算，因果mask仍限制位置t只能读$x_{\le t}$。NTP目标是$x_{t+1}$；MTP第j个head目标是$x_{t+j}$。训练并行处理位置与推理一次提交多个正确token是不同概念。

主干hidden通常是[B,T,H]，输出logits是[B,T,V]。V大于H意味着多head词表输出很占内存；多个loss对同一hidden的梯度可求和，但detach后必须明确怎样接回trunk。独立未来分布不等于joint序列概率：即便各head边际校准，它们的候选仍可能不一致。

推理中target的第一个head规定普通下一token行为，其余head提供draft。自投机验证只接受与target规则相符的连续前缀，不能把所有head输出直接拼接。token/byte、head数/草稿长度、接受率/产出长度、样本效率/训练时间也需要分开。

### 已有工作与本文位置

未来预测目标此前已在ProphetNet等工作中出现；本文并非第一个想到预测多个位置，而是探索decoder模型从头训练的简单多head目标、参数匹配的大规模质量结果和自投机执行。[speculative decoding](https://arxiv.org/abs/2211.17192v2)提供draft/target校正框架，[Medusa](https://arxiv.org/abs/2401.10774v3)侧重附加解码头与树验证。本文的关键差别在于把未来预测作为预训练目标而非只做后置解码器微调。

| 路线 | 主要问题 | 本文比较时的注意 |
| --- | --- | --- |
| NTP | 最邻近未来token监督 | 强baseline，训练所有位置已有并行 |
| 未来辅助目标 | 改变表示学到的信息 | 结构、粒度、权重与条件信息不同 |
| Medusa式heads | 提供便宜解码候选 | 冻结或联合微调，接受策略另核 |
| Meta MTP | 从头训练多个未来heads | 主模型质量与自投机速度分别验证 |
| 后来的DeepSeek MTP | 串行条件链与共享层 | 不可当成本文独立head架构的原配置 |

## 2. 方法与实现机制

### 一个共享前缀怎样对应多个目标

trunk $f_s$产生$h_t=f_s(x_{\le t})$，第j个独立Transformer head $f_j$处理这个表示，共享unembedding $W_u$输出词表分布：

$$
q_j(x_{t+j}\mid x_{\le t})=\operatorname{softmax}(W_u f_j(h_t)),\qquad
L_n=-\sum_t\sum_{j=1}^{n}\log q_j(x_{t+j}\mid x_{\le t}).
$$

n=1还原下一token训练，n=4包括下一token和三个更远目标。这里“独立”指给定同一prefix后各head分别建模未来边际，而非直接输入此前head预测token形成autoregressive条件链。head可以共享学习信号和参数统计，但未显式处理候选之间的条件依赖。

![Figure 1 · 上部多头结构 下部MBPP规模结果](./assets/papers/paper-mtp-meta/figure-1-source.png)

**Figure1解读。** 上半图同一trunk表示进入head1—4，各位置监督后续四个token；粉色未来head可在普通推理时丢弃。下半图是MBPP Pass@1增益随规模变化，不是结构组成：0.3/0.6B有下降，更大模型改善。保留作者完整复合图并分别说明用途，不能把下方结果当作新方法模块，也不能把“up to3times”视为每次调用保证。[图源：v1 Figure1，源码main_fig_col.png](https://arxiv.org/src/2404.19737v1)。

假设下一步A/B各一半，第二步由其决定C/D；独立head分别学[A,B]与[C,D]，仍可能提出A+D。验证成本与接受率正来自这种不一致。多未来loss要求trunk保留更远信息，但不能恢复缺少的显式条件关系。论文附录也讨论串行结构变体；后来DeepSeek采用的链式模块应单独读原配置。

### 标签对齐与有限序列边界

对长度T训练片段，j-step目标仅在t+j<T有效。实现可截断对应logits/labels，或用mask忽略尾部；packed文档还应屏蔽跨文档未来目标。SFT场景是否预测prompt、response、工具observation属于数据规则，原预训练目标不能不经设计直接迁移。

```python title="教学重述 · 一位置多未来标签"
hidden = trunk(tokens)       # [B,T,H]，因果mask
for offset, head in enumerate(heads, start=1):
    logits = unembedding(head(hidden))  # [B,T,V]
    labels = future_labels(tokens, document_ids, offset)
    # labels中-100排除尾部、padding与跨文档目标
    loss += masked_cross_entropy(logits, labels)
```

这段是本文重述，不是作者完整代码；原head是Transformer层，课堂Linear toy只检查梯度。还需固定reduction：不同offset可训练位置不同，各headmean与全局有效tokenmean产生不同尺度；不能声称改reduction只是无意义的数值优化。

### 逐头forward/backward为何能减少峰值

如果同时保存n个head的logits和局部图，词表相关暂存近似O(nBTV)。论文安排trunk先forward，然后head各自forward/backward，累加到detached hidden leaf，最后向trunk反传一次。

![Figure 2 · 内存友好的计算图调度及作者图内示例代码](./assets/papers/paper-mtp-meta/figure-2-source.png)

**Figure2解读。** 左边green为forward、orange为backward、灰圆为shared hidden；head1贡献后先释放其输出，再计算head2，编号表示顺序。右边d.detach/requires_grad建立leaf，head losses累加d.grad，末行z.backward接回trunk。它减少同时驻留的词表输出，不是去掉所有head参数、optimizer、trunk激活与通信。[图源：v1 Figure2，原独立资产backward_order.png](https://arxiv.org/src/2404.19737v1)。

```python title="教学重述 · 不切断主干贡献的分阶段反传"
z = shared_trunk(tokens)
d = z.detach().requires_grad_(True)
for head, target in zip(heads, future_targets):
    logits = shared_unembedding(head(d))
    cross_entropy(logits, target).backward()
z.backward(gradient=d.grad)
# 先完成所有head贡献，再optimizer.step()
```

head独立图之间没有必要保留彼此激活，shared unembedding参数的grad仍要相加。若每headstep、清掉共享grad、或忘记z.backward，已不与整体目标等价。AMP与分布式同步也不能机械套用这段；例如已scale的d.grad不能再重复scale。

本轮[CPU课堂脚本](./assets/infra/mtp-lab.py)用相同reduction对比整体loss与逐头方案，最大参数梯度差约$2.78\times10^{-17}$。它检查小张量构造而非GPU峰值或论文训练复现。logit容量例1×4096×128000×4×2bytes约3.906GiB；一头约0.977GiB，均只算logit单张量，不代表整机所需显存。

### 参数匹配与普通推理

论文为了总参数匹配，增加n−1个head层时减少n−1个trunk层。这样质量比较控制总参数，但main推理只用trunk加一个head，和另一个完整NTP模型的主路径深度可能不同。不能说除了loss所有网络结构完全不变，也不能把“训练head可丢弃”理解成推理模型参数仍等于训练总参数。

速度实验将同一个4-token模型仅用1head的普通解码当baseline，然后增用额外heads做greedy self-speculation。这个速度对照不是直接拿从头训练的另一个NTP模型作唯一baseline。一次昂贵调用能验证并产出多个token，实际倍率还受接受前缀与新增算子成本约束。

### 理论解释与不能保证的部分

作者用consequential token、induction能力和算法任务解释为何未来监督有帮助。一个决定性token的后续多个关联目标会使当前表示受到更多学习压力；但同一future仍可能本身高熵，额外目标也可能增加噪声。正文直觉不构成对任意数据、规模、window或任务的稳定收益保证。

自然语言多选评测的n=4存在退化，GSM8K在200B/500B训练量下排序变化，足以说明收益依赖目标与预算。去掉heads后质量改善可以体现trunk表示变化，保留head加速则要接受/verify证据，两种论证应分别陈述。

### 公开实现状态与源码阅读边界

[作者官方研究模型card](https://huggingface.co/facebook/multi-token-prediction)列出NTP与4-head研究checkpoint、model.py与generation.py等入口，固定model-card revision为`6580f46982ea783120bfe10796717e5c139144b3`。本轮可读取公开card，但未能读取需要访问授权的官方模型代码，未取得原始训练loop；不把第三方PyTorch复现当作官方实现。

因此本报告的代码为原文图内调度和明确标注的课堂重述。现代DeepSeek MTP实际接线在[训练笔记](#q=infra-mtp-training)用固定vLLM源码伴读，它是另一种串行结构，也不能据此声称核过本文原训练实现。

## 3. 实验设置与算力

质量实验包含代码规模、不同prediction horizon、byte-level、多epoch、CodeContests微调、自然语言与合成算法任务。token、byte、模型训练总参数与主路径参数各有口径；本账本聚焦核心代码与self-speculation，不混合各实验预算。

| 项目 | 固定v1配置 | 出处与边界 |
| --- | --- | --- |
| 规模实验 | 0.3/0.6/1.3/3/6.7/13B，从头训练 | 6.7B简称7B；总参数匹配 |
| 数据量 | 小模型约91B code tokens；7/13B约209.7B | TableS13；正文约数200B |
| 网络层数/宽度 | 7B H4096、32总层、32attentionheads；13B H5120、40层、40heads | TableS14，MTPhead数不是attentionhead数 |
| MTP horizon | token实验n1/2/4/6/8等，byte实验可更长 | n不是runtime独立draft步数 |
| 代码训练优化 | Adam系，beta1=0.9、beta2=0.95，weightdecay.1，clipnorm1 | 实際数值0.9/0.95；TableS13 |
| LR/length/batch | peak3e-4，context4096，常用batch8×2^20 tokens | warmup1000/2000与cosine decay按实验分开 |
| 代码评测 | MBPP、HumanEval、APPS Intro | 每题200samples估计Pass@k |
| temperature | 用test scores选oracle温度 | Table1说明；不是统一固定解码公平验证 |
| 自投机code | 7B、1T训练tokens，4200条未见512-token prompt，生成512token | greedy；maxbatch42 |
| 自投机text | 7B、500B训练tokens，Wikipedia/books | 不混作code模型同数据 |
| byte实验 | 约314B bytes，byte vocabulary | Table1行标签313B；配置314.6B约数 |
| 自然语言 | 7B，200B/500B，n1/2/4 | 多选、摘要、GSM8K分任务 |
| hardware | 全部报告实验累计约500K GPU-hours，A100-80GB与H100 | carbon声明；不是单模型最低复现成本 |
| 单组GPU数量/推理具体硬件 | 未给可统一拆分的完整账本 | 不反推batch42等于某固定卡数 |
| official code | 研究card可见，原模型源码本轮访问受限 | 教学与源码身份明确分开 |

batch8×2^20是约8.39M训练tokens，不是8个prompt。head数变化时标签数量与有效尾部不同，valid-token reduction需按配方核对。Pass@100估计是从多样本中至少一次成功的量，不是单次生成正确率；oracle温度选择又会影响不同模型/任务的比较。

同原文语料token数不一定意味着所有算子、通信和模型主路径都一样。对论文所谓“same computational budget”，应结合参数匹配设计和实际时间附录读，而不直接转换成某现代GPU上严格相同GPU-hour。碳足迹段只披露聚合约500K，不能分配成每个7B实验具体时长。

原文并未给全部data snapshot、seed、精度和分布式设置的现代可执行账本。本轮没有下载研究checkpoint或GPU训练；CPU只测toy梯度与概率，结果不进入论文质量表。

## 4. 结果与图表解读

![Figure 3 · 不同规模与Pass at k的代码质量增益](./assets/papers/paper-mtp-meta/figure-3-source.png)

**Figure3解读。** 左右为MBPP/HumanEval，三行Pass@1/10/100，横轴模型规模；柱表示4-token相对NTP增益，数字给baseline，误差条为对评测样本bootstrap的90%区间。小0.3/0.6B下降，大模型多数改善，不能把几个大模型点推广成MTP对所有规模都更好。区间来自评测样本重采，不等于多个training seeds的方差。[图源：v1 Figure3](https://arxiv.org/src/2404.19737v1)。

13B的MBPP Pass@1约增加4.5个百分点，HumanEval约增加1.7点。摘要中12%/17%是相对问题解决量改善，不是12/17个准确率百分点；图中的baseline与误差条共同限制了结论强度。

![Table 1 · Token和byte两种粒度及horizon的完整比较](./assets/papers/paper-mtp-meta/table-1-pdf.png)

**Table1解读。** 200B token部分，MBPP n1=30.0、n4=33.8，增加3.8点；HumanEval22.8→24.0，增加1.2点；APPS/Intro n6的3.5更好，n4却1.6低于n1的2.8。byte部分8最好不表示BPE也应选8；1T多epoch下HumanEval Pass@1也并非提升。该表清楚说明window与任务、数据量交互，而非越多heads越好。[表源：v1第4页](https://arxiv.org/pdf/2404.19737v1#page=4)。

![Table S2 · 同一4head模型的greedy自投机速度](./assets/papers/paper-mtp-meta/table-s2-pdf.png)

**TableS2解读。** 每域两列分别为相对速度与tokens/forward；k4时Wikipedia2.74×/3.12、books2.67×/3.09、code3.05×/3.50。3.50包含本轮最终验证/预测产出，不应说草稿接受率350%；正文code约2.5个额外建议被接受，加一个基础推进token。速度少于4的理想上界，说明接受与执行cost都起作用。[表源：v1附录A第13页](https://arxiv.org/pdf/2404.19737v1#page=13)。

![Figure S10 · Batch变化下的相对throughput与latency](./assets/papers/paper-mtp-meta/figure-s10-source.png)

**FigureS10解读。** 两面板横轴batchsize，左throughput越大越好，右relative latency越小越好，k1为同模型baseline。k4在该code实验和batch范围保持约3×收益，不能推到所有大并发、多GPU或长context服务。latency与throughput都已做相对归一化，不能从曲线读出绝对tok/s或GPU-hour。[图源：v1附录FigureS10](https://arxiv.org/src/2404.19737v1)。

![Table S5 · 实际训练时间不能只复述正文无开销](./assets/papers/paper-mtp-meta/table-s5-pdf.png)

**TableS5解读。** n4相对NTP时间从1.07到1.22，13B为1.09，即多约9%训练时间。表注将开销归因于分头反传破坏FSDP通信计算重叠，并认为可通过更合适实现改善；该判断不是已经提供零开销测量。它支持内存调度需要同时评估通信，并限制摘要“no overhead”的工程泛化。[表源：v1附录C第15页](https://arxiv.org/pdf/2404.19737v1#page=15)。

本轮解读架构、反传、主要规模结果、horizon、速度与真实训练时间。自然语言多选n4退化、GSM8K排序随200B/500B变化等补充分析在正文/附录，不能只挑代码改善忽略适用范围。其他合成任务图保留为后续深入阅读，不称为本轮重新验证的结果。

## 5. 局限、结论与后续阅读

本文证明的是一种简单辅助目标在若干大规模代码/生成任务中的有效性，以及其heads可提供较好的自投机候选。它不保证未来边际joint一致、任意horizon有益、主路径质量与速度共同单调增加；参数匹配缩短trunk、训练时间、oracle温度与未拆分硬件预算都会影响解释。

### 面试与讲课问答

**Q01：MTP与训练全位置并行有什么区别？** NTP原本就teacher forcing并行位置；MTP增加每个位置预测的未来距离，不取消生成串行依赖。

**Q02：独立heads是否建模真正joint future？** 条件在同prefix的边际分布分别学习，仍可能组合不一致；需要额外结构或验证，不把乘积近似视为真实联合关系。

**Q03：为什么共享unembedding仍需累加梯度？** 同一参数被多个head使用，各loss的路径贡献都必须保留；每head清零或step会改变目标。

**Q04：detach省内存为什么不会冻结trunk？** 因为最后把汇总leaf.grad传回原z；没有这一步就是另一种训练方式。CPU等价检查只验证toy构造。

**Q05：参数匹配是否说明main结构不变？** 不说明，增加head层时减少trunk层，质量比较控制训练总参数；速度baseline又是同MTP模型只用1head。

**Q06：3.05倍是否所有模型都能得到？** 是特定7B code模型、greedy prompt/output、batch42等条件结果，不能替代新后端实际测速。

**Q07：MTP训练是不是免费？** 附录实测有额外时间；logit峰值优化不免除参数、激活与通信。需要完整训练/显存账本。

**Q08：哪些证据说明收益有条件？** 小模型退化、APPS最佳n不同、n4自然语言多选下降，以及多epoch/math变化；先确定任务与budget，再选horizon。

从[Infra系列入口](#q=infra-mtp-overview)接[DeepSeek训练结构](#q=infra-mtp-training)、[采样校正](#q=infra-mtp-speculative)和[后端落地](#q=infra-mtp-serving)。[MTP专题对照](#report=survey-mtp-infra)提供分层阅读路线，不将不同硬件结果排成榜单。

**来源与更新。** 2026-10-09核读固定2404.19737v1正文、附录、源码图表与公开研究card；model/source访问状态和CPU教学范围分别说明。原图归原作者，source/PDF hash及裁剪框在figures.json。未执行真实训练或GPU benchmark。
