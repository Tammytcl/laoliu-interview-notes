---
id: paper-pagedattention
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
paper_title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
authors: ["Woosuk Kwon", "Zhuohan Li", "Siyuan Zhuang", "Ying Sheng", "Lianmin Zheng", "Cody Hao Yu", "Joseph E. Gonzalez", "Hao Zhang", "Ion Stoica"]
affiliations: ["UC Berkeley", "Stanford University", "Independent Researcher", "UC San Diego"]
author_affiliations: [[1], [1], [1], [1, 2], [1], [3], [1], [4], [1]]
venue: "SOSP 2023"
year: 2023
areas: [language]
tasks: [text-generation]
published: 2023-09-12
method_figure: "./assets/papers/paper-pagedattention/figure-6-source.png"
method_caption: "Figure 6 · 逻辑块与物理块映射"
direction: infra
openalex_id: W4387321091
paper_url: "https://arxiv.org/abs/2309.06180v1"
github_url: "https://github.com/vllm-project/vllm"
code_note: "Official vLLM repository; current main is not the 2023 experimental revision."
evidence: 已核原文
note_ids: [kv-cache]
tags: [LLM, LLM Serving, PagedAttention, KV Cache, Memory Management, vLLM]
updated: 2026-09-30
summary: "将请求的逻辑 KV 序列映射到非连续物理块，通过按需分配、共享和写时复制提高可批处理容量。"
template_version: 4
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

先明确系统任务：许多用户同时请求语言模型，每个请求的提示词长度、生成长度和结束时间都不同。服务端既要装下模型权重，又要为活跃请求保存 KV cache。KV 是每层 attention 中历史 token 的中间结果，它会随着生成增长；不像模型参数那样大小固定。显存留给某个请求却没有被使用时，其他请求就可能进不了 batch，即使 GPU 仍有计算余量。

这里有两种容易混淆的浪费。为未知的未来输出预留一大段空间，会产生“已经保留、还没用上”的内部浪费；不同请求释放不同大小的连续区域，又可能留下零散空洞，空闲总量足够却找不到所需的大连续块。本文将序列的逻辑连续性和显存的物理连续性分开：attention 仍然读取有序历史，但这些历史可以放在许多分散的小块中。其目标是提高服务系统能同时容纳的有效请求数，并不是减少每个 token 本来必须保存的 K/V 信息。

LLM 服务的输入长度差异很大，输出长度又事先未知。若给每个请求预留一段连续显存，按最大输出长度预约会造成尚未使用的空间；动态分配又可能留下无法利用的小空洞。同一 prompt 的多个采样分支还会复制相同 KV。这些浪费直接限制可以同时放入 GPU 的请求数，形成系统吞吐瓶颈。

这一问题来自服务系统，而不是模型不理解任务。FasterTransformer 已优化推理 kernel，但原文评估中的服务调度由作者额外实现；Orca 已提出 iteration-level scheduling，允许不同请求按生成迭代进入和退出 batch。然而批处理调度不能自动消除 KV 的过量预留、内外部碎片与重复存储。PagedAttention 借鉴操作系统分页，把逻辑连续性与物理连续性解耦，再在 vLLM 中组织 allocator、scheduler 和 GPU worker。

| 已有做法 | 已解决的部分 | 剩余问题 |
| --- | --- | --- |
| 低延迟推理 kernel | 加速单批次模型执行 | batch 容量仍受 KV 管理影响 |
| iteration-level batching | 请求可按迭代进入 / 退出 | 请求状态怎么分配、共享仍需处理 |
| 连续 KV 预留 | 地址计算直接 | 未知输出长度带来保留浪费与碎片 |
| 为分支复制 prompt KV | 易于实现独立序列 | 多采样、beam search 重复占用 |

本文要证明的不是“注意力公式更准确”，而是：**保持模型计算语义，通过更高效地管理请求状态，让更多请求同批执行，在相近延迟下承受更高负载。** [KV cache](#q=kv-cache) 提供先修。

## 2. 方法与实现机制

请求仍有按 token 顺序排列的逻辑 KV blocks，但每个逻辑块通过 block table 指向一个空闲物理块。物理块不要求相邻；只有新 token 到来、当前块填满时才分配新块。若每块容量为 $B$ 个 token，token 位置 $t$ 的寻址可表示为：

$$
\operatorname{logical}(t)=\lfloor t/B\rfloor,\quad \operatorname{offset}(t)=t\bmod B,\quad \operatorname{physical}(t)=\operatorname{BlockTable}[\operatorname{logical}(t)].
$$

这是一种索引推导，不是论文给出的逐字伪代码。PagedAttention kernel 读取 block table 后，在对应物理块中计算 attention；它仍然要读需要的历史 K/V。不能把“Paged”理解成稀疏注意力或只读当前页。

![Figure 6 · 逻辑 token 块到物理 KV 块的映射](./assets/papers/paper-pagedattention/figure-6-source.png)

图源：[arXiv 2309.06180v1 LaTeX 源码](https://arxiv.org/src/2309.06180v1)，`figures/multi-sequence-block-mapping.pdf`。

**Figure 6 解读。** 图左侧是请求的逻辑序列，右侧是可以散布在不同位置的物理块，中间表格负责翻译。沿一个逻辑块的指针追到物理块，可以看到 token 顺序并未因存储不连续而改变。示例块大小用于画图，不能直接替代实验 block size。[图源：原文 Figure 6](https://arxiv.org/pdf/2309.06180v1#page=6)。

多个采样分支可以共享完整 prompt blocks，并用 reference count 追踪使用者。如果分支要写入仍被别人使用的最后一个块，就先复制该块再追加，这就是 copy-on-write。共享并不意味所有生成后缀永久共用；不同 token 导致的分支仍需要各自状态。beam search 的状态继承也可以用映射与引用更新表示。

![Figure 4 · vLLM 的调度器、块管理器与 GPU 执行器](./assets/papers/paper-pagedattention/figure-4-source.png)

图源：[arXiv 2309.06180v1 LaTeX 源码](https://arxiv.org/src/2309.06180v1)，`figures/pagedattention.pdf`。

**Figure 4 解读。** 这是一张系统边界图：中心调度器管理请求与 block tables，GPU worker 根据 token 与映射执行模型。多 GPU 情况下，worker 处理自己的 attention heads，管理器维持一致的逻辑映射。图里的控制流不等于每张 GPU 保存完整 KV；张量并行下实际数据是分片的。[图源：原文 Figure 4](https://arxiv.org/pdf/2309.06180v1#page=5)。

内存不足时，原文采用 FCFS 调度、优先抢占后到请求，并比较把 KV 换到 CPU 与重新计算。共享的多序列请求作为 sequence group 一起调度。重计算能够把已生成 token 拼到 prompt，利用一次 prefill 重建缓存；是否比 swapping 更好取决于 CPU-GPU 带宽、模型和序列长度，不是分页机制本身保证的常数。

### 核心思想：把分配单位改成块，再让 attention 理解映射

假设一块容纳 16 个 token，一个长度为 35 的序列需要 3 块：前两块满、最后一块只用 3 个位置。随着长度增长，只在跨过块边界时增加物理块；请求结束后这些块可以归还给池。逻辑上第 0、1、2 块仍按 token 顺序排列，物理块编号却可以分别是 7、2、19。块表记录这个映射，让序列无需得到一整段连续空间。

这不是只修改 Python 内存分配器就能完成的优化。Attention kernel 必须利用块表读取正确的历史 K/V，按所有块上的 token 做同一个 attention 归一化和加权求和。分页不意味着每块独立 softmax 再直接相加，也不意味着丢弃远处 token。系统节省的主要是预留与碎片，以及多输出间可共享的重复 KV；由此能增加有效 batch，吞吐才可能提升。

### 核心源码：分配、共享与写时复制

为避免用今天的系统解释 2023 年论文，本报告对照 vLLM `v0.2.0` 的固定 commit。`BlockSpaceManager.allocate` 为 prompt 的逻辑块取得物理块，构建 `block_table`，并让同组序列持有自己的块表列表、共享底层物理块；引用计数反映有多少序列需要该块。方法操作的是 KV 存储槽位，token ID 的逻辑块是索引依据，两者内容不能混淆。

`append_slot` 检查生成后的逻辑块数量是否增长：跨界时分配新块；仍在最后一块内且引用计数为一时，直接使用现有槽位。如果最后一块被共享，则分配新块、替换当前序列的末尾映射并释放旧块的一次引用，返回 `(旧块号, 新块号)` 给后续执行器做实际复制。返回复制任务而不是在该函数里搬运 CUDA 张量，是控制逻辑与 GPU 数据操作的分工。

`fork` 复制块表并增加各物理块的引用计数，不立即复制完整 KV。比如同一 prompt 生成两个回答，前缀可以共用；只有写入共享的部分填满块时才需要分离。这也解释了为什么“可以共享”不等于所有分叉都零成本：新分配、copy-on-write 和逐步分歧仍有代价。此历史实现还包含 sliding-window 分支，不应把该分支当作本文所有实验都启用的设定。源码阅读未执行吞吐复现。

## 3. 实验设置与算力

这里的 benchmark 是服务负载而不是训练数据：ShareGPT 提供会话长度特征，Alpaca 提供另一种请求分布，作者据这些特征构造输入输出长度和到达过程。系统要在给定模型与 GPU 上同时处理这些请求。吞吐回答一段时间完成多少工作，延迟回答单个请求等待多久；提高 batch 往往改善前者却可能恶化后者。因此论文比较在延迟约束下能承受的请求率，而不是只看一个孤立 tokens/s。改变输出长度、并行采样数或到达密度都会改变 KV 压力，同一系统不保证在所有场景得到相同倍数提升。

这是推理系统论文，**没有为验证 PagedAttention 重新预训练 LLM**。ShareGPT/Alpaca 用于构造服务负载；将它们说成本文“训练数据”会错置实验目的。

| 项目 | 原文设置 |
| --- | --- |
| 模型 | OPT-13B、OPT-66B、OPT-175B；另测 LLaMA-13B |
| 服务器 | Google Cloud A2 instances，NVIDIA A100 |
| 实现 | Python 调度与块管理；PyTorch / Transformers 模型执行；C++/CUDA kernel；NCCL 张量并行通信 |
| 负载数据 | ShareGPT 和 Alpaca 文本的 tokenized 输入 / 输出长度 |
| 到达过程 | 原数据无时间戳，使用 Poisson 到达、扫描请求率 |
| 测试时长 | 大多数实验 1 小时 traces；OPT-175B 因成本限制使用 15 分钟 |
| 评价指标 | 请求率—normalized latency 曲线、batch 请求数、KV 内存节省；不是 MMLU 等模型能力分数 |
| Baselines | FasterTransformer + 作者实现的 dynamic batch scheduler；作者重实现的 Orca Oracle / Pow2 / Max variants |
| 解码场景 | 单样本生成、parallel sampling、beam search 与共享前缀；不同图使用各自配置 |

![Table 1 · 论文模型规模与 GPU 显存配置](./assets/papers/paper-pagedattention/table-1-pdf.png)

| 模型规模 | GPU 配置 | 总显存 | 表中 KV 预算 |
| --- | --- | --- | --- |
| 13B | 1 × A100 40 GB | 40 GB | 12 GB |
| 66B | 4 × A100 40 GB | 160 GB | 21 GB |
| 175B | 8 × A100 80 GB | 640 GB | 264 GB |

这些数字来自原文 Table 1，属于原模型与执行配置，不是今日所有 vLLM 模型的最低显存要求。[表源](https://arxiv.org/pdf/2309.06180v1#page=9)。精度、权重体积和运行时开销要随实际 checkpoint 重新核对；也不能直接用参数数乘以 2 bytes 就把余下显存全部分配给 KV。

Orca 原实现未公开，作者自行实现三种预留策略：Oracle 预先知道最终输出长度，Pow2 按 2 的幂扩容，Max 按上限预留。这是对照实验的重要限制。不同策略不是三个独立公开服务产品，更不能把某个最差预留基线的最高加速比当作对所有系统的收益。

## 4. 结果与图表解读

![Figure 12 · 单序列生成的请求率与延迟曲线](./assets/papers/paper-pagedattention/figure-12-source.png)

图源：[arXiv 2309.06180v1 LaTeX 源码](https://arxiv.org/src/2309.06180v1)，`figures/experiments/n1-sharegpt.pdf`、`figures/experiments/n1-alpaca.pdf`；按原图顺序排列源码子图，不重新绘制数据。

**Figure 12 解读。** 横轴是提供给服务的 request rate，纵轴是 normalized latency。曲线在低负载下缓慢上升，接近系统处理能力后突然抬升，表示排队积累，而不是某个 attention kernel 突然变慢。比较时应在近似延迟水平看哪条曲线能承受更高请求率，并核对模型、GPU 数量和上 / 下两行的负载数据。[图源](https://arxiv.org/pdf/2309.06180v1#page=10)。

在 ShareGPT 基础单样本负载上，原文报告 vLLM 相对 Orca Oracle 支持约 1.7–2.7 倍请求率，相对 Orca Max 约 2.7–8 倍；这些范围比“普遍快若干倍”更准确。FasterTransformer 对照还同时缺少细粒度调度，因此其较大差距不能全归因于 block table。OPT-175B + 较短 Alpaca 序列时，Orca 某些策略也能容纳较大 batch，系统更接近 compute-bound，vLLM 优势缩小。这个反例正好验证：内存管理收益依赖真实瓶颈。

![Figure 13 · 同时批处理的请求数](./assets/papers/paper-pagedattention/figure-13-source.png)

图源：[arXiv 2309.06180v1 LaTeX 源码](https://arxiv.org/src/2309.06180v1)，`figures/experiments/batched_requests_sharegpt.pdf`、`figures/experiments/batched_requests_alpaca.pdf`；按原图顺序排列源码子图，不重新绘制数据。

**Figure 13 解读。** 横轴是运行时间，纵轴是 batch 中的请求数；ShareGPT 子图对应 OPT-13B、2 requests/s，Alpaca 对应 30 requests/s。它补足 Figure 12 的因果链：更少 KV 浪费让更多请求共存，批处理容量提高，负载拐点向右移动。单看 batch 大小不能保证延迟低，还要与排队和请求长度一起判断。[图源](https://arxiv.org/pdf/2309.06180v1#page=10)。

![Figure 15 · 并行采样和 beam search 的 KV 共享节省](./assets/papers/paper-pagedattention/figure-15-source.png)

图源：[arXiv 2309.06180v1 LaTeX 源码](https://arxiv.org/src/2309.06180v1)，`figures/experiments/mem_saving_parallel_gen.pdf`、`figures/experiments/mem_saving_beam.pdf`；按原图顺序排列源码子图，不重新绘制数据。

**Figure 15 解读。** 两幅图分别看 parallel sampling 与 beam search，改变输出分支数量，比较共享 KV 带来的内存节省。prompt 越能复用，复制同一前缀越浪费；beam search 还可共享部分生成路径。图里的收益与前缀长度、分支数和分歧位置有关，并不代表不同内容的任意请求都能共享缓存。[图源](https://arxiv.org/pdf/2309.06180v1#page=11)。

图表核读范围本轮包括系统图 Figure 4、映射图 Figure 6、硬件 Table 1、主吞吐 Figure 12/13 与共享 Figure 15。其余图表已放入资产清单并列出待核项，后续可补充 block size、kernel 开销、swapping/recomputation 与混合负载的更细证据；本报告不把尚未解释的图标为已完成。

## 5. 局限、结论与后续阅读

PagedAttention 改进的是 **KV 内存管理和可批处理容量**，不是语言模型的任务质量。它与 [GQA](#paper=paper-gqa)互补：GQA 决定需要多少组 K/V，PagedAttention 决定这些状态如何分配与共享。新 kernel 的索引与读块开销必须由系统级收益覆盖；短序列、权重占主导、低并发或通信受限场景都可能减少收益。

论文中的 Orca 为作者重实现，工作负载使用合成时间戳，实验版本也不是今天 vLLM main。计划复现时应记录代码 commit、模型 revision、tokenizer、精度、block size、batch token 预算、抢占策略、到达分布与并行拓扑，并重新测 TTFT、每 token 延迟及吞吐，避免与原文 normalized latency 混用。

**来源与更新。** 根据 [arXiv v1 正文与实验](https://arxiv.org/html/2309.06180v1)、Table 1 和原图整理；单位按发表时列示。2026-09-30 更新为五模块报告，新增英文元数据、硬件账本与原图。图片保留原作者归属；本报告为原文核读与分析，未登记个人运行结果。[前缀缓存笔记](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)和 [推理显存专题](#report=survey-inference-memory)可继续连接模型层与服务层。

### 参考讲解与源码版本

本报告参考 [Woosuk Kwon 与 Zhuohan Li 的 vLLM 作者讲解](https://vllm.ai/blog/2023-06-20-vllm)，吸收用按需分配和共享解释显存节省；博客的 HF/TGI 测速与正式论文的 FT/Orca 实验没有混用。解释已融入问题与方法部分；数字、图表和实验口径回到固定版本原文核对。

源码静态核读固定于 `e2fb71ec9f2c3168ba8614408fa807a5f65707c5`（vLLM v0.2.0）。核心文件：[vllm/core/block_manager.py](https://github.com/vllm-project/vllm/blob/e2fb71ec9f2c3168ba8614408fa807a5f65707c5/vllm/core/block_manager.py#L102-L169)。没有执行代码或重新训练。

**图表来源。** 本报告优先提取固定版本 LaTeX 源码中的原始图片；表格及没有独立图片的 TeX 绘图使用局部 PDF 裁图，不截整页。源码文件、版本、校验值或 PDF 裁剪区域记录在图表证据清单 figures.json。
