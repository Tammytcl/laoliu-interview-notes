---
id: paper-gqa
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
paper_title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
authors: ["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebrón", "Sumit Sanghai"]
affiliations: ["Google Research", "University of Southern California"]
author_affiliations: [[1], [1], [1, 2], [1], [1], [1]]
venue: "EMNLP 2023"
year: 2023
areas: [language]
tasks: [text-generation]
published: 2023-05-22
method_figure: "./assets/papers/paper-gqa/figure-2-pdf.png"
method_caption: "Figure 2 · Query heads 与共享 KV groups"
direction: llm
openalex_id: W4389518760
paper_url: "https://arxiv.org/abs/2305.13245v3"
github_url: "https://github.com/google/flaxformer"
code_note: "Paper-linked implementation framework; not a standalone reproduction package."
evidence: 已核原文
note_ids: [mha-gqa-mqa, kv-cache]
tags: [LLM, Attention, KV Cache, Uptraining, Inference, T5]
updated: 2026-09-30
summary: "通过分组共享 K/V 与已有 checkpoint 的继续预训练，在生成质量、KV 容量和解码延迟之间建立可调折中。"
template_version: 4
draft: false
---

## 1. 背景与已有工作

先从语言模型的一次生成理解问题。模型看到提示词后，要预测下一个 token；新 token 出现后，它再预测下一个。Attention 让当前位置根据相关性读取之前的位置：query 表示“现在要找什么”，key 用来计算与各位置的匹配程度，value 是匹配后要取出的信息。一个 head 是一套这样的投影与读取规则，多个 head 可以学习不同的关联。

对于带因果约束的解码器，已出现 token 在各层的 K/V 可以保存起来，下次继续使用，这就是 KV cache。缓存避免反复计算过去的位置，但并不消除每一步读取历史 K/V 的开销。训练通常并行处理许多 token；decode 则往往每条序列每步只增加一个 token，算术工作较少、历史读取越来越多。因此“减少 K/V 头”主要改变缓存与带宽需求，并不等于把 query 的表达能力或所有 attention 运算等比例删掉。GQA 要研究的正是这个结构选择以及从已有模型过渡的成本。

自回归生成每一步只产生少量新 token，却需要读取模型权重以及过去 token 的 key/value。序列越来越长时，KV cache 随之增长；解码的瓶颈常常是把数据从显存搬到计算单元，而不是矩阵乘法能力不足。标准 multi-head attention（MHA）给每个 query head 配置独立的 key/value head，表达能力充分，但缓存与读取成本也随头数增长。

这个问题并非 GQA 首次发现。[Shazeer 的 MQA 论文（2019）](https://arxiv.org/abs/1911.02150)已经提出：保留多个 query head，让它们共用一组 key/value。这样可以显著减少 KV 存储和读取，但更强的共享约束可能损害质量；已有 MHA checkpoint 也不能直接无损变成 MQA。GQA 论文还讨论了 [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) 对推理带宽和并行的分析，以及 [FiDO](https://arxiv.org/abs/2212.08153) 对 encoder-decoder 推理的优化。

本文面对的是两个相互关联的问题：**能否利用已经训练好的 MHA 模型，付出较少的额外训练成本得到较快的模型？能否不把全部 K/V 压到一个头，而是在速度和质量之间保留可调空间？** 前者对应 checkpoint conversion + uptraining，后者对应 grouped-query attention。论文也承认 Markus Rabe 独立实现了 GQA；阅读时不应将“首次发现 KV 瓶颈”“首次提出所有分组思想”和本文的实验证明混为一谈。

| 路线 | 改动对象 | 解决的成本 | 与 GQA 的关系 |
| --- | --- | --- | --- |
| MQA | 所有 query 共用一组 K/V | KV 缓存与带宽 | GQA 的单组极端 |
| FlashAttention | 注意力计算与内存访问方式 | 中间注意力矩阵的 IO | 可与 GQA 组合，不等同于减少 KV 头 |
| KV / 权重量化 | 数值表示的位宽 | 每个元素的存储和读取 | 可叠加，但另有精度误差 |
| PagedAttention | KV 的分配、索引与共享 | 碎片和重复占用 | 不修改模型的 query/KV 头结构 |

必要的先修是 [MHA/GQA/MQA 的维度关系](#q=mha-gqa-mqa) 和 [KV cache 的容量计算](#q=kv-cache)。这里的关键区分是：GQA 减少的是 **K/V 头数**，并不是将全部 query 头合并，也没有把 token 间的稠密注意力改为稀疏注意力。

## 2. 方法与实现机制

### 分组共享的结构

设 query 头数为 $H$，KV 组数为 $G$，每组包含 $H/G$ 个 query head，单头维度为 $d$。同组 query 使用共同的 $K_g,V_g$，但每个 query 仍然计算独立的注意力权重。对第 $h$ 个 query head，可以写成：

$$
g(h)=\left\lfloor\frac{h}{H/G}\right\rfloor,\qquad O_h=\operatorname{softmax}\left(\frac{Q_hK_{g(h)}^{\mathsf T}}{\sqrt d}+M\right)V_{g(h)}.
$$

其中 $M$ 是相应任务的 mask；decoder self-attention 要遵守因果关系，cross-attention 的 K/V 来自 encoder 输出。$G=1$ 是 MQA，$G=H$ 是 MHA，中间取值才是通常讨论的 GQA。本文将改动用于 **decoder self-attention 和 cross-attention**，不改变 encoder self-attention：encoder 表示并行计算，不是同一种逐 token 带宽瓶颈。

![Figure 2 · MHA、MQA 与 GQA 的 query / key / value 共享关系](./assets/papers/paper-gqa/figure-2-pdf.png)

**Figure 2 解读。** 三幅结构图比较的是“有多少套 K/V 对应这些 query”。MHA 中每个头独立，MQA 中所有 query 汇聚到同一组 K/V，GQA 中各组内部共享。应沿连线追踪一个 query 读到哪套 K/V，而不是把图中的多个 query 当作同一个注意力分布。这张图说明结构与容量关系，本身不证明质量恢复，也不提供实测加速比。[图源：原文 Figure 2](https://arxiv.org/pdf/2305.13245v3#page=2)。

### 从 MHA checkpoint 转换并继续训练

转换时，不随机新建一套 K/V，也不是保留组内第一个头。作者把同组原有投影做 mean pooling。若组内原 key 投影为 $W^K_h$，则新的组投影为：

$$
\widetilde W^K_g=\frac{1}{|\mathcal H_g|}\sum_{h\in\mathcal H_g}W^K_h,\qquad \widetilde W^V_g=\frac{1}{|\mathcal H_g|}\sum_{h\in\mathcal H_g}W^V_h.
$$

Q 投影保留；模型需要新的 K/V 参数布局。均值池化尽量保留原 checkpoint 中的信息，但它改变了模型函数，因此作者还按照原预训练配方进行 uptraining，让其他参数与新的共享结构共同适应。这里的 $\alpha=0.05$ 表示额外预训练步数约为原预训练的 5%，不是每个用户都能用“原模型总成本的 5%”完成任何结构转换。

![Figure 1 · 原 MHA 的 K/V 投影均值池化，得到共享投影](./assets/papers/paper-gqa/figure-1-pdf.png)

**Figure 1 解读。** 左右比较转换前后投影矩阵；mean pooling 发生在权重的头维度，不是对当前请求的 token 做池化。图画的是 MHA→MQA 的单组示例，推广到 GQA 时在每个组内部执行相同操作。结构转换只是第一步，后面的继续预训练不能省略为一个 reshape。[图源：原文 Figure 1](https://arxiv.org/pdf/2305.13245v3#page=1)。

下面的伪代码解释投影的分组逻辑，**不是可直接载入 Flaxformer checkpoint 的完整转换脚本**；真实布局、分片轴和优化器状态都要单独处理。

```python
# 教学假设：W_k / W_v 的轴为 [d_model, H, head_dim]
assert H % G == 0
W_k_grouped = W_k.reshape(d_model, G, H // G, head_dim).mean(axis=2)
W_v_grouped = W_v.reshape(d_model, G, H // G, head_dim).mean(axis=2)
# W_q 保留；加载新的 attention 结构，继续原预训练任务。
```

对于普通 decoder-only KV cache，若 batch 为 $B$、层数为 $L$、缓存长度为 $S$、每元素占 $b$ bytes，容量的教学推导为：

$$
\operatorname{KVBytes}=2BLSGdb.
$$

固定其他项时，MHA→GQA 的 KV 容量比为 $G/H$。例如 $L=32,H=32,G=8,d=128,b=2,B=1$，每 token 的 KV 从 512 KiB 降到 128 KiB。这是**给定假设下的算式**，不属于本文 T5 实验，也不包含权重、激活、分片副本或 allocator 开销。真实执行应避免为方便计算而把共享 K/V 物理复制成 $H$ 份，否则可能抵消容量收益。

### 核心思想：共享的是记忆，不是所有查询

假设有 8 个 query head、2 个 KV head，则每 4 个 query head 共用一个 K/V 组。它们仍然有不同的 query 投影，因此面对相同的历史 keys 可以得到不同的匹配权重；共享的是可被查阅的 key/value 表示，而不是最终 attention 输出。把 K/V 头数记作 $G$，当 $G=H$ 时回到 MHA，当 $G=1$ 时就是 MQA。这个连续的结构选择允许研究质量与缓存成本的折中。

但从已有 checkpoint 平均合并 K/V 投影，会改变模型已经学到的表示。原来各 query 配合独立 keys 的关系不一定在合并后成立，因此必须继续预训练来适应。论文的重要证据包括结构与 uptraining recipe 两部分；仅修改头数并观察显存下降，还不足以证明论文声称的质量恢复。后面的实验应沿“如何合并、继续训练多少、最终质量和延迟如何变化”来读。

### 源码对照：先读 MHA 与 MQA 两个端点

论文链接 Flaxformer 作为实现框架，但不是一个完整发布的 GQA checkpoint 转换与 uptraining 复现包。本报告核读固定 commit 的 `dense_attention.py`：`dot_product_attention` 处理独立 K/V 头，`dot_product_attention_multiquery` 展示所有 query heads 共用 K/V 的端点。这里没有将框架里的 MQA 函数冒充成论文全部 GQA recipe。

多查询函数中，query 形状为 `[batch...,q_length,H,d]`，key/value 则没有独立 head 轴，形如 `[batch...,kv_length,d]`。计算权重的 einsum `...qhd,...kd->...hqk` 保留 query 的 `h`，对同一份 keys 计算每个 head 的权重，再沿历史位置 softmax 并读取 values。输出仍保留 $H$ 个 query head。沿这些轴看代码，比把 K/V 简单复制 $H$ 次更能看清共享本质：复制可以用于教学广播，但物化成完整缓存会抵消节省。

`MultiQueryDotProductAttention` 负责把输入投影成这些 Q/K/V 张量并组织 decode cache；底层函数主要负责 attention 运算，两者不能混为一层。源码还说明 T5 的缩放可以折进投影初始化，因此通用公式中的 $1/\sqrt d$ 不一定以一行显式除法出现。对于 GQA 的中间情况，需要按组连接 query 与相应 K/V；本报告已有的分组伪代码用于说明这一映射，均值合并公式来自原论文，而没有声称已在此框架中定位或运行完整转换流水线。

## 3. 实验设置与算力

本文使用 T5 encoder-decoder 模型：encoder 处理输入文本，decoder 自回归生成输出，实验并不是今天常见的任意 decoder-only 模型。摘要任务要求保留输入的关键信息，通常以 ROUGE 比较生成与参考摘要；翻译任务用 BLEU 评价与参考译文的匹配；TriviaQA 关注事实问答的答案匹配。作者将这些不同任务的指标作汇总，用来观察转换后的整体质量变化，但平均分不是一种天然统一的能力刻度。质量表和延迟实验必须结合各自长度、batch、模型和硬件设置阅读。

以下设置按原文 §3.1 与附录 A 整理，训练与测速分开登记。

| 项目 | 原文设置 / 披露范围 |
| --- | --- |
| 基础模型 | 公共 T5.1.1 Large 与 XXL checkpoint；主比较包括 MHA-Large、MHA-XXL、MQA-XXL、GQA-8-XXL |
| 模型范式 | encoder-decoder；GQA/MQA 用于 decoder self-attention 与 cross-attention |
| 实现框架 | JAX、Flax、Flaxformer；页首 GitHub 是论文引用的框架，不是完整、锁定版本的复现包 |
| Uptraining 数据 | 继续原 T5 预训练设置与数据；T5.1.1 公共配方使用 C4，不能把下游摘要数据说成此次继续预训练语料 |
| 优化器与 schedule | Adafactor，沿用 T5 超参与学习率 schedule；GQA 正文没有完整重印所有预训练配置 |
| 主实验转换 | K/V mean pooling，uptraining 比例 $\alpha=0.05$ |
| 微调超参 | 恒定学习率 0.001，batch size 128，dropout 0.1 |
| 模型选择 | 训练至收敛，选择 dev 表现最高的 checkpoint |
| 生成 | greedy decoding；不能与 beam search 或多采样服务直接比较 |

[T5 checkpoint 官方说明](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md)是核查模型与预训练配方的补充来源。GQA 本身没有逐条披露本次语料快照、数据过滤版本、随机种子和精确预训练 token 总数；这些项复现时必须再补，不能用常见 T5 参数值填充成“本文设置”。

| 下游任务 / benchmark | 输入长度 | 输出长度 | Table 1 指标 |
| --- | --- | --- | --- |
| CNN/Daily Mail 摘要 | 512 | 256 | ROUGE-1 |
| arXiv、PubMed、MediaSum、Multi-News 摘要 | 2048 | 512 | ROUGE-1 |
| WMT 2014 English→German | 512 | 256 | BLEU |
| TriviaQA | 2048 | 32 | F1 |

这些 benchmark 是下游微调与评价任务，**不是都拿来做原始预训练**。消融只用 CNN/Daily Mail、Multi-News 和 TriviaQA 的代表性子集；不能把三任务消融平均分误认成 Table 1 七任务平均分。结果表报告 dev 表现，未提供面向今天通用对话、编码或长上下文检索的统一结论。

| 算力环节 | 原文披露 | 正确理解 |
| --- | --- | --- |
| 5% uptraining | 约 600 TPUv3 chip-days | 累计芯片时间；正文未给出可直接还原本次运行的卡数与墙钟时间组合 |
| 推理测速 | 8 个 TPU，报告每 sample、每 TPUv4 chip 的时间，用 xprof 测量 | 不等于单张 GPU 端到端服务延迟 |
| 测速 batch | 能放入的最大 batch，每 TPU 最多 32 | 各模型分别优化并行设置，非统一固定 batch 的纯算子比较 |
| GPU 数量 / GPU-hours | 原文未提供 | 不能将 TPUv3 chip-days 换写成 A100 卡数或训练时长 |

若只做教学验证，可用小模型比较共享头形状、KV 理论容量与输出差异；若要复现 Table 1，则必须有兼容的 T5 checkpoint、训练数据与分片配置，且需重新建立自己的硬件测速口径。本文没有给出“最低几张消费级 GPU 就能复现”的证据。

## 4. 结果与图表解读

### 主结果：质量接近 MHA，速度接近 MQA

![Table 1 · T5 模型的原始推理时间与七个任务 dev 结果](./assets/papers/paper-gqa/table-1-pdf.png)

Table 1 使用秒作为时间单位。MHA-XXL 的时间为 1.51，平均分 47.2；GQA-8-XXL 为 0.28、47.1；MQA-XXL 为 0.24、46.6。按表中取整后的数值计算，GQA 相对 MHA-XXL 的该项时间约缩短 **5.39 倍**，平均分相差 **0.1**。这两个数是本报告的算术推导；它们描述论文规定的 TPU 测量，不是今日任意服务的速度保证。[表源：原文 Table 1](https://arxiv.org/pdf/2305.13245v3#page=3)。

逐任务看比只看 Average 更有意义：GQA 的 PubMed 与 Multi-News 分数甚至高于表中 MHA-XXL，而 CNN、arXiv 与 TriviaQA 略低；MQA 的 WMT 分数又略高。结果支持“折中较好”，不支持“每一个任务无损”。Average 把不同指标的数字放在一起汇总，不是一个具有统一量纲的泛化能力测量。

![Figure 3 · 七任务平均分与推理耗时的折中](./assets/papers/paper-gqa/figure-3-pdf.png)

**Figure 3 解读。** 横轴是每样本推理时间，越左越快；纵轴是汇总任务表现，越上越好。GQA-XXL 接近 MHA-XXL 的高度，却靠近 MQA-XXL 的水平位置，因此显示出有利折中。应注意原始图横轴写了 “ms”，但 Table 1 与对应数值写的是秒，源码也保留了这一不一致；本报告以 Table 1 的 **s** 口径登记，不把图中的 0.28 改称 0.28 ms。[图源：原文 Figure 3](https://arxiv.org/pdf/2305.13245v3#page=3)。

### 消融：收益来自哪些设计？

![Figure 4 · Mean、First、Random 三种 checkpoint 转换方式](./assets/papers/paper-gqa/figure-4-pdf.png)

**Figure 4 解读。** 这里比较 T5-Large→MQA，并固定 5% uptraining；横轴是转换策略，纵轴是三个代表任务的汇总表现。mean pooling 最好，选第一个头次之，随机初始化较差。对照的意义是控制继续训练条件，观察初始 K/V 权重处理的影响；它不是“任何模型里均值池化一定最佳”的证明，也不是 GQA-8-XXL 的全部主结果。[图源：原文 Figure 4](https://arxiv.org/pdf/2305.13245v3#page=4)。

![Figure 5 · 额外预训练比例与 MQA / GQA 的任务表现](./assets/papers/paper-gqa/figure-5-pdf.png)

**Figure 5 解读。** 横轴是继续预训练占原训练的比例，纵轴是代表性任务表现；比较 MQA 与 GQA-8 的恢复轨迹。GQA 在刚转换后就保留较多质量，MQA 更依赖 uptraining；从 0 增加到 5% 有明显收益，继续到 10% 的边际收益较小。它支持作者选 5% 的经验取舍，但没有覆盖所有训练规模、数据或目标组数。[图源：原文 Figure 5](https://arxiv.org/pdf/2305.13245v3#page=4)。

![Figure 6 · KV 组数增加时的 GQA-XXL 每样本推理耗时](./assets/papers/paper-gqa/figure-6-pdf.png)

**Figure 6 解读。** 输入长度固定 2048、输出长度固定 512，横轴是组数，纵轴是每样本时间。由 1 组增加到 8 组的额外耗时较小，接近 MHA 时成本增加更明显；8 是该实验中的较好中间点。此图只画速度，不单独回答质量，必须和前面任务结果一起读；也不能把 8 当作所有层、所有硬件的最优常数。[图源：原文 Figure 6](https://arxiv.org/pdf/2305.13245v3#page=4)。

本报告覆盖原文 **6 张主图与 1 张结果表**。结构图解释机制，转换/步数/组数消融解释选择，主结果建立特定任务与硬件上的质量—速度证据；它们共同支撑结论，而不是每张图都用来重复一句“更快”。

## 5. 局限、结论与后续阅读

本文最可靠的结论是：对测试的 T5 encoder-decoder 模型，K/V 组内共享配合已有 checkpoint 的继续预训练，可以用少量额外预训练获得比 MQA 更好的质量折中。它同时给出一种可迁移的工程分析方式：**先明确减少了什么状态，再分别验证模型质量和执行效率。**

作者没有比较 GQA-XXL uptraining 与同规模从零训练的完整对照；ROUGE 也不足以全面评价长文本生成。附录 A 记录 MQA 的预训练 loss spike 与长输入微调不稳定，部分不稳定任务的 MQA 结果取三次微调平均；GQA 显示较稳定，但论文并未完整追溯根因。因此“质量接近”应限定到原文的任务、指标与训练方式。

迁移到 decoder-only GPU 服务时，应重新检查 KV 分片副本、kernel 的原生 GQA 支持、prefill 与 decode 比例、上下文长度和 batch 负载。减少 cache 容量不必然改善权重占主导、排队占主导或工具等待占主导的系统。与 [PagedAttention 报告](#paper=paper-pagedattention) 一起读，可以把“模型需要多少 KV”与“KV 如何分配”两层分开；[显存账本](#q=memory-budget)用于补足权重、激活和临时工作区。

**整理范围与版本。** 核对 arXiv v3（2023-12-23）的正文、附录 A、LaTeX 图表数据及 Table 1；图表为原论文 HTML 截图，版权与学术贡献归原作者。2026-09-30 更新为五模块报告，增加元数据、全图解释与实验口径。本报告未进行训练复现；页首“已核原文”表示来源核对，不表示完成了硬件复现。后续更新应补实际 checkpoint/代码 commit、语料快照和自己的性能日志，不能将待做实验登记为实测。

### 参考讲解与源码版本

本报告参考 [Sebastian Raschka 的 A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants)，吸收把不同结构沿共享维度并排比较的解释角度；本文实验结论仍只取 GQA 原论文。解释已融入问题与方法部分；数字、图表和实验口径回到固定版本原文核对。

源码静态核读固定于 `399ea3a85e9807ada653fd0de1a9de627eb0acde`。核心文件：[flaxformer/components/attention/dense_attention.py](https://github.com/google/flaxformer/blob/399ea3a85e9807ada653fd0de1a9de627eb0acde/flaxformer/components/attention/dense_attention.py)。没有执行代码或重新训练。

**图表来源。** 本报告使用固定版本原论文 PDF 的核对裁剪图，不重新排版原表；对应 PDF 页码、裁剪区域和文件校验值记录在 assets/papers/paper-gqa/figures.json。LaTeX 源码保留在本地缓存，用于核查图表及上下文。
