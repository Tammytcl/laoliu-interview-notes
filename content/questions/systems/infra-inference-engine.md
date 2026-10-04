---
id: infra-inference-engine
title: "04 · 推理引擎如何提速？从 prefill/decode 到 KV、批处理与缓存"
category: systems
difficulty: 进阶
tags: ["P0", "Infra基础", "推理", "vLLM", "SGLang", "KV Cache"]
updated: 2026-10-05
summary: "区分prefill与decode负载，推导GQA KV容量，比较FlashAttention、PagedAttention、continuous batching、prefix caching与chunked prefill，建立调优与验收流程。"
draft: false
---

## 1. 主问题：推理引擎优化的是哪一段？

本篇从一次LLM请求开始：输入prompt，得到多个新token。目标是解释为什么vLLM/SGLang需要调度器、KV管理与高效kernel，而不是给项目排速度名次。先修 [自回归训练与生成](#q=causal-lm-training)、[Attention](#q=attention-complexity)、[GPU性能模型](#q=infra-gpu-performance)。与verl/slime的连接见最后两节。

核对日2026-10-05。容量数字与队列例子是教学假设；未运行GPU serving benchmark。默认参数与后端支持会变化，本篇说明机制，并将文档与引用来源登记在清单中。

## 2. 请求生命周期：prefill与decode有什么不同？

**Prefill**计算已有prompt各位置的隐状态和K/V，生成下一token所需分布。prompt已知，位置可并行，较大的矩阵运算更易复用权重。**Decode**得到前一个新token后，才能输入下一个位置；每步通常每条活跃序列推进一个token，并读取此前缓存的K/V。

![本文示意：请求排队、prefill、逐步decode与完成释放KV](./assets/infra/infra-inference-engine/request-lifecycle.svg)

**图解：** 上排是请求生命线，TTFT覆盖排队及产生第一个token的过程；后续两个token之间的ITL还可能包含调度等待。下排显示prefix随decode增长，KV占用随缓存token增长。它是概念图，实际引擎可能把prefill拆块、交错不同请求，并批量发送流式输出。

训练拥有完整序列，可以一次并行计算各位置条件概率；生成不能知道未来token。KV cache减少重复计算已知prefix的K/V，但不消除自回归依赖。低并发decode常接近权重/KV访存瓶颈；prefill更容易变成大矩阵负载。长上下文、大batch、MoE和通信会改变这一概括。[Transformer inference 教程](https://jax-ml.github.io/scaling-book/inference/)。

指标首先统一定义：

| 指标 | 本篇约定 | 易误读的地方 |
|---|---|---|
| TTFT | 请求提交到客户端收到首token | 包含队列、网络；服务端计时可能范围不同 |
| ITL | 相邻可见token之间间隔 | 流式chunk不等于每token独立发送 |
| TPOT | 首token之后耗时除以后续token数 | 是平均量，单token输出需另定义 |
| E2E latency | 请求提交到最后token完成 | 受输出长度、排队与decode影响 |
| output token/s | 时间窗内产生新token数 | 不应混入prompt token后仍称生成速度 |
| goodput | 满足给定质量/时延条件的有效产出 | 条件必须先写清，不能只挑快请求 |

p50和p95刻画分布，不只是平均值。在线聊天可能重视TTFT/ITL，训练rollout通常更重视完成样本吞吐，但长尾导致整批等待时仍需关注完成时间。

## 3. KV容量：用模型结构与实际长度算，而不是只数请求

### Q：KV cache存什么，为什么不是模型参数？

每层把各历史位置投影为key和value，新位置用query对它们做attention。参数跨请求共享，KV依赖具体输入prefix、模型版本和计算路径，各请求通常不同。标准完整attention、等长缓存、无额外复制的容量是：

$$
M_{KV}=2LBSH_{KV}D_hb
$$

$L$层数、$B$缓存序列数、$S$缓存长度、$H_{KV}$ KV heads、$D_h$ head维度、$b$每元素字节。2表示K和V。[KV基础](#q=kv-cache)、[GQA原理](#q=mha-gqa-mqa)。

例子：L32、B8、S8192、KV heads8、head dim128、BF16每元素2字节，得到8 GiB，即每序列1 GiB。如果KV heads32则同配置32 GiB。这是GQA节省KV容量的一条直接来源，不表示总服务显存也恰好除4。

真实多请求应按长度求和，用 $\sum_i S_i$ 替换BS。还要容纳权重、workspace、临时激活、CUDA Graph池和allocator。滑窗、MLA、KV量化、共享prefix、TP复制/分片都需要重算。TP数大于可切KV heads时，某些实现复制KV，不能无条件再除TP。

**准入例子：** 若剩余KV预算24 GiB、每条最终8k长度需1 GiB，24条只是完整长度下的理论上限。当前64条都很短也许可以运行，但后来同时增长就可能触发抢占/重算。只看当前“活跃请求数”无法保证后续容量。

### Q：KV量化与权重量化会省同一项吗？

权重量化主要改变模型权重字节，KV量化主要改变历史状态字节；不同kernel与反量化成本、误差也不同。decode可以同时受两者流量影响，不能用“模型权重已INT8”推断KV也减半。比较前固定精度、输出质量与长上下文分布。

## 4. Attention kernel与KV管理：Flash和Paged解决不同问题

### Q：FlashAttention如何少搬数据而保持attention语义？

朴素实现常物化 $S\times S$ 分数或概率到HBM。FlashAttention将attention分块，在片上处理局部块，并维护online softmax统计，减少大中间矩阵的HBM读写。它不是默认删掉部分token的稀疏attention；dense情况下算量仍随token对数增长。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135v2)。

为理解online softmax，维护已见score最大值m、指数和l、未归一化加权value和u。新块score为s，value为v，设 $m'=\max(m,\max s)$，则：

$$
l'=e^{m-m'}l+\sum_j e^{s_j-m'},\qquad u'=e^{m-m'}u+\sum_j e^{s_j-m'}v_j
$$

最后输出 $u'/l'$。旧部分按新最大值重新缩放，避免一次保留全部score。这个式子是教学解释，GPU实现还涉及并行、tile、mask、反向与数值误差，不是几行Python即可复现全部性能。

### Q：PagedAttention为何使用逻辑块与物理块？

假设为每条请求预留最长8k连续KV，而很多输出很短，会浪费空间；连续分配还容易受不同请求生命周期影响。PagedAttention让逻辑token块映射到分散物理块，按增长申请，减少大段预留并支持共享与copy-on-write。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180v1)。

![PagedAttention原论文Figure6：两条请求的逻辑KV块映射到物理显存块](./assets/infra/infra-inference-engine/pagedattention-source.png)

**原图解读与归属：** 图来自Kwon等作者的固定v1原图，本库此前已从LaTeX独立资产提取，本篇复核后引用。两侧分别是请求A/B的逻辑块，中间是物理KV块；箭头表达映射，A的逻辑0/1/2分别落到物理7/1/3，B的逻辑0/1落到物理5/2。不同颜色区分请求及已有/新增token，空白表示尚未占用；逻辑上连续不要求物理上相邻。block table是实现映射的数据结构，本图用箭头表示它的作用。图说明内存组织，不是速度曲线；末块内部未满、块表元数据和分配成本仍存在。分页不自动改变attention算术语义，也不自动实现批处理调度。

可把两者区别说成：Flash优化运算过程中的中间数据搬运，Paged优化持续存在的KV空间管理。现代引擎可组合相应kernel/布局；兼容性需看具体版本，不能把论文机制名称当可随意拼接的API。

## 5. 调度与缓存：同一块GPU怎样服务不同长度请求？

### Q：continuous batching为什么不等于“大batch一次跑完”？

静态batch常等整批结束再接新请求。迭代级调度每轮决定哪些请求参与下一次模型执行：短请求结束即可退出，空位让新请求加入。这样减少已完成请求占位和整批末尾空转。[Orca 原论文与分享](https://www.usenix.org/conference/osdi22/presentation/yu)。

教学例子：A还需2token，B需6token。静态batch只放A/B时，A完成后位置闲到B结束；迭代调度可以让C在下一轮加入。但C需要prefill，不是“只需给空位写一个名字”；引擎还要安排算量、token预算和KV容量。continuous batching提高机会，不保证零等待。

### Q：chunked prefill怎样影响TTFT与ITL？

长prompt如果一次prefill占用很久，会推迟正在decode的请求。拆成多个chunk可以将prefill与decode交错，在单轮token预算内选择工作。当前vLLM V1文档描述优先安排decode，剩余预算给prefill；小token预算通常有利于ITL，大预算可能提高prefill吞吐，但代价取决于负载。[vLLM 当前调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)。

`max_num_seqs`约束同时调度的序列数，`max_num_batched_tokens`约束单轮token量，`max_model_len`约束单序列上下文，KV容量约束常驻总token。它们不是同一个batch size，更不等于训练global batch。增加一个上限可能不产生效果，因为另一个已成为瓶颈。

### Q：prefix caching与KV cache的区别？

KV cache一般指同一请求生成时复用其已计算prefix；prefix caching进一步在不同请求间复用完全相同prefix的KV。共享system prompt、相同历史对话或树状采样可受益。缓存命中不是“语义相似”，tokenization、模板、模型权重、adapter与影响状态的设置必须兼容。

SGLang原论文将RadixAttention用于组织共享prefix，用树结构表达分叉的token前缀。查找/淘汰/调度可以关注缓存亲和性，但缓存策略和任务公平性需要一起考虑。[SGLang 原论文](https://arxiv.org/abs/2312.07104v2)。

缓存减少重复prefill，不必然减少新token decode次数。在RL中每轮更新权重后，旧KV通常不能被当作新模型的精确KV直接复用；部分研究允许近似保留，必须标明mixed-version与数值假设。模板改变也可能让命中骤降。

### Q：CUDA Graph与speculative decoding为什么要单独理解？

CUDA Graph可重放已捕获的GPU执行图，减少CPU提交开销，常需要相容shape、内存地址和执行路径；捕获多个batch档位也会占内存。图缓存与KV缓存对象不同，动态图shape和后端限制要核对。[PyTorch CUDA Graph说明](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。

Speculative decoding先用较廉价draft提议多个token，再由target批量验证。在匹配定义的采样分布与正确接受/补偿算法下，可保留target的输出分布；它不是任意地接受“看起来相近”的draft。性能取决于接受率、draft成本、验证并行、负载与长度。高并发decode已充分batch时，额外draft未必值得。[Speculative Decoding 原论文](https://arxiv.org/abs/2211.17192v2)。

## 6. 连接vLLM/SGLang与verl/slime：参数应该怎样调？

引擎选择先看模型、硬件、接口与依赖组合，然后按同一任务做对照。官方架构博客可帮助理解engine、scheduler与KV manager的分工；本文采用这些模块概念，不直接照搬其历史默认值。[vLLM 官方架构博客](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)。

建议调优顺序：

1. **正确性基线**：固定模型/tokenizer/template，检查stop、最大长度与输出结构。
2. **负载画像**：记录prompt/output长度分布、共享prefix比例、到达率、工具等待与任务成功率。
3. **容量检查**：估权重与KV预算，记录抢占/recompute、峰值和OOM。
4. **调度预算**：依次比较序列上限、每轮token预算、prefill chunk与缓存策略。
5. **并行组织**：在能放下的前提下，对比较低TP+多个DP副本与较高TP；看通信和每副本KV。
6. **全链路验收**：相同完成任务/GPU-hour和质量，再看token/s、TTFT、ITL与p95。

不要直接复制某博客的显存比例。vLLM与SGLang的内存参数名称、统计基准和外部常驻内存不同，共置训练时optimizer和图池也会占空间。[SGLang 调优文档](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)、[verl Performance Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html)。

在verl/slime rollout中，返回文本还不够：往往需要token IDs、实际behavior logprob、mask与采样设置。推理引擎能提供OpenAI式服务接口，不表示该接口自动携带训练所需全部信息，需核对框架adapter。

多轮agent的工具等待不执行decode，64个活跃agent可能只有4个ready请求。增加总agent数仍要受环境容量和KV/token预算约束；引擎内调度与任务级调度见 [RL流水线](#q=infra-rl-pipeline)。

## 7. 面试问答：把优化对象说清楚

均为本文整理题；引用指向机制来源，无公司真题或频率声明。

**Q01 [基础] prefill与decode为什么瓶颈不同？** 已知prefix可并行形成大矩阵，新token逐步依赖历史；低并发decode更易搬权重/KV。追问：长上下文一定如此？还需分析attention、KV与通信字节量。

**Q02 [推导] 8条8k请求KV多大？** 先列层数、KV heads、head dim和dtype，按本篇设置8 GiB；GQA与MHA分别算。追问：TP后一定除TP吗？确认KV切分还是复制。

**Q03 [机制] FlashAttention与PagedAttention区别？** 前者减少attention中间量HBM IO，后者管理KV块。追问：Flash是否将dense计算变线性？不是；少存矩阵不等于少算所有token对。

**Q04 [机制] continuous batching如何解决长短输出？** 每轮退已完成、接新工作，减空位；仍受prefill与KV预算。追问：GPU很忙为何p95恶化？排队、长prefill与高到达率可能提升尾延迟。

**Q05 [设计] 缩小prefill token预算有什么收益和代价？** 常改善decode间隔，但prefill被拆得更碎，TTFT与吞吐可能受影响。追问：测什么？固定长度/到达率，报告TTFT、ITL、产出和抢占。

**Q06 [排障] 相同system prompt缓存为何没命中？** 比较实际token prefix、模板、模型/adapter版本、路由到的副本及淘汰。追问：参数更新后KV可继续用吗？通常需失效，保留是需验证的近似路线。

**Q07 [设计] speculative decoding一定加速？** 接受率不足或draft/验证成本高就未必；不能只报draft便宜。追问：怎样保留target分布？必须明确采样分布与接受/补偿逻辑。

**Q08 [排障] rollout有很多agent却低busy？** 区分active与ready：工具等待、环境限流、排队、prefill和KV抢占。追问：提高并发如何验收？完成任务/GPU-hour、长尾和质量，不能只计token。

## 8. 复习练习与来源

**手算练习：** 把8条等长请求改成长度[1k,2k,4k,8k]，按长度和求KV；画A2token/B6token/C新加入的调度；用三个score分两块求online softmax，验证与整块结果一致。本轮CPU公式检查有独立脚本，GPU调优尚未实施。

| 来源 | 类型 | 阅读用途 |
|---|---|---|
| [Transformer inference](https://jax-ml.github.io/scaling-book/inference/) | 作者教程 | prefill/decode与容量 |
| [FlashAttention](https://arxiv.org/abs/2205.14135v2) | 原论文 | 分块与online softmax |
| [PagedAttention v1](https://arxiv.org/abs/2309.06180v1) | 原论文/原图 | 逻辑/物理KV块 |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | 原论文与作者分享 | 迭代级调度 |
| [SGLang](https://arxiv.org/abs/2312.07104v2) | 原论文 | RadixAttention |
| [vLLM anatomy](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | 官方博客 | 引擎模块与工作流程 |
| [vLLM tuning](https://docs.vllm.ai/en/latest/configuration/optimization/) / [SGLang tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning) | 官方教程 | 当前调参入口 |
| [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) | 官方API，2.14 | 图捕获与内存约束 |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192v2) | 原论文 | draft/target与接受 |
| [verl tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) | 官方教程 | rollout与训练共置 |

[本篇来源清单](./research/infra-inference-engine-sources.json)。最后一篇将这些阶段组织成可训练、可恢复、可测量的闭环。

**数值练习下载：** [CPU验算脚本（仅Python标准库）](./assets/infra/infra-foundations-check.py) · [本次验算输出](./assets/infra/infra-foundations-check-results.json)。覆盖梯度有限差分、状态/KV字节、Roofline、TP与分块softmax；不包含GPU训练或测速。
