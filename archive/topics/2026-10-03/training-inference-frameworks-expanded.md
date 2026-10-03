---
id: training-inference-frameworks
title: "训推框架梳理：从 Megatron、FSDP 到 vLLM、SGLang、verl 与 slime"
category: systems
difficulty: 进阶
tags: ["P0", "训练框架", "推理框架", "RL Infra", "slime", "verl", "面试问答"]
updated: 2026-10-03
summary: "先理解训练、推理与 RL 的资源和数据流，再比较框架与选型；配合原图、计算例子、60 道面试练习和可复用调研流程。"
draft: false
---

## 1. 阅读路线与范围

遇到 slime、verl、Megatron、DeepSpeed、vLLM、SGLang 这些名字时，先回答三个问题：**它在哪一层？处理什么输入输出？解决什么瓶颈？** 把这三个问题搞清楚，框架就不再是一串需要背诵的名字。

本文面向刚开始准备 AI Infra、LLM 训练或 RL 系统面试的读者。建议分三次读：第一次读第 2—5 节，建立训练与生成的基础；第二次读第 6—9 节，理解框架取舍和 RL 数据流；第三次用第 10—12 节做口述、排障和设计练习。已有基础可直接从 slime / verl 对比开始。

核对日期为 **2026-10-03**。框架“支持”指官方仓库、文档或明确 recipe 存在对应路径，不等于任意模型、精度、GPU、训练后端和 rollout 后端的组合都能运行。本文没有执行 GPU 训练性能复现；计算例子和资源分配例子是教学估算。源码入口以文末来源记录的版本为准，`main`、`latest` 文档可能继续变化。

面试题来源采用两种标记：**[公开题库主题]** 表示公开社区题库能找到相同知识主题；**[本文延伸]** 表示基于官方技术资料设计的练习。前者也不是已验证的某公司原题，更不代表题目出现频率统计。题目表述与答案为本文重新组织；外部题库只用于确认覆盖方向。

## 2. 框架地图：哪些名字处于同一层？

### Q：能先用一张表区分它们吗？

| 层次 | 代表工具 | 主要职责 | 典型输入 → 输出 |
|---|---|---|---|
| 张量与自动微分 | PyTorch | 算子、autograd、设备执行 | 张量与计算图 → 数值结果、梯度 |
| 模型与微调接口 | Transformers、PEFT、TRL | 模型定义、adapter、训练 recipe | 模型配置、数据、目标 → 可训练模型与训练任务 |
| 分布式训练执行 | DDP、FSDP/FSDP2、DeepSpeed、Megatron Core | 分片、并行计算、通信、优化器状态 | tokens、loss、参数分片 → 更新后的权重 |
| 推理与生成服务 | vLLM、SGLang、TensorRT-LLM | KV 管理、请求调度、生成 kernel、服务接口 | prompt、采样参数 → token、可选 logprob |
| RL / 后训练编排 | verl、slime、OpenRLHF、AReaL | rollout、reward、advantage、训练、权重同步 | prompt、环境、奖励定义 → 新策略与样本账本 |
| 资源与作业管理 | Ray、Kubernetes、Slurm | worker 放置、资源分配、作业生命周期 | 资源请求、任务 → 运行中的进程与服务 |
| 底层通信与算子 | NCCL、CUDA、Triton、FlashAttention | collective、GPU kernel、IO 优化 | 张量与执行请求 → 通信或算子结果 |

这是学习用的职责分类，现实项目可能跨层。FSDP 是 PyTorch 的分布式机制，不是独立服务；FlashAttention 是 attention 算子方法，不是完整训练框架；TensorRT-LLM 与编写 GPU kernel 的 Triton、NVIDIA Triton Inference Server 也不是同一个东西。[PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[TRL](https://huggingface.co/docs/trl/index) 提供对应实现入口。

### Q：slime、verl 和 vLLM 是竞争关系吗？

slime 与 verl 可以在 RL 系统这一层比较；vLLM 与 SGLang 可以在生成引擎这一层比较。verl 可以调用 vLLM 或 SGLang 做 rollout；slime 的主线围绕 SGLang 与 Megatron 组织。这意味着“用 verl 还是 vLLM”通常不是二选一，而是要决定编排层与生成层怎样组合。[verl 官方仓库](https://github.com/verl-project/verl)、[slime 官方仓库](https://github.com/THUDM/slime)。

### Q：框架地图怎样连接到一次实际任务？

```text
预训练 / SFT：固定数据 → tokenizer / packing → trainer → loss → backward → optimizer
在线推理：   请求 → 排队 → prefill → KV cache → 多次 decode → 返回结果
在线 RL：    prompt → rollout / 环境 → reward → advantage → trainer → 新策略
                        ↑                                  │
                        └──────── 权重同步与版本发布 ──────┘
```

图为本文概念示意。RL 循环里的 trainer 可以采用 Megatron 或 FSDP；rollout 可以采用 SGLang 或 vLLM。训练与 rollout 不必共用相同的 GPU 数、batch 或并行布局。因此 RL Infra 的难点还包括数据重分布、权重重分片与一致性。

## 3. 基础知识：训练、推理和 rollout 为什么不同？

### Q：预训练、SFT、DPO、PPO、GRPO 各需要什么系统能力？

预训练与 SFT 都可用 teacher forcing：一次把输入序列送进模型，同时计算多个位置的 next-token loss。区别主要在数据、loss mask 与训练目标的组织。标准离线 DPO 在固定偏好对上计算策略与参考概率，不要求每个训练 step 在线生成；在线变体另说。PPO / GRPO 的在线训练则先从策略采样，再根据 reward 得到更新信号。GRPO 常用同一 prompt 的多个回答构造组内相对优势，通常不需要独立的 learned critic；reference 与 KL 项是否存在要看具体算法和配置。算法名不是框架名。[DPO 论文](https://arxiv.org/abs/2305.18290)、[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)。

例如 SFT 的 512 条样本可从磁盘直接取；GRPO 的 128 条 prompt × 每条 8 个回答，要先生成 1024 条 response。生成长度、工具调用和评分速度都可能不一致。trainer 有再好的 backward kernel，也要等数据可用。

### Q：为什么训练时不能像生成时一样，一个 token 一个 token 地跑？

自回归描述的是概率分解：

$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
$$

训练时目标序列已知，用 causal mask 可以并行计算各位置的条件概率。生成时下一个 token 取决于刚采样出来的 token，通常必须逐步 decode。这里的“并行”是多个位置的数值计算可同时执行，不是把未来 token 暴露给过去位置。可以继续阅读 [因果语言模型训练](#q=causal-lm-training)。

### Q：prefill 和 decode 分别做什么？

prefill 处理已有输入，计算它们的中间表示并建立 KV；decode 使用已有 KV，为每条活跃序列继续生成 token。长 prompt 的 prefill 常具有较大矩阵乘，单步 decode 的 query 较短，低 batch 时经常受权重和 KV 读取带宽限制。但“prefill 一定计算受限、decode 一定带宽受限”并非定律：batch、上下文、模型结构、量化、硬件都会改变瓶颈。[vLLM 优化文档](https://docs.vllm.ai/en/latest/configuration/optimization/)。

### Q：KV cache 缓存什么？为什么不缓存 Q？

它保存历史 token 每层 attention 的 K、V。当前 token 的 Q 会与历史 K 做匹配，再用权重聚合历史 V；历史 Q 通常不再用于后续 token 的 attention，因此常规 decoder 生成缓存 K/V 就够。KV cache 省下了历史位置重复计算，但历史越长，当前 query 仍可能需要读更多 KV。训练时一般不沿用这套跨请求生成缓存，因为要处理梯度和全序列计算。[KV Cache 基础](#q=kv-cache)。

### Q：怎么估算训练显存？7B 到底需要多少？

先列账本：

$$
M_{train}=M_{weights}+M_{gradients}+M_{optimizer}+M_{master}+M_{activation}+M_{temporary}.
$$

假设全参数 Adam 训练，BF16 权重 2 字节、BF16 梯度 2 字节、FP32 一阶二阶状态共 8 字节，另保留 FP32 master 权重 4 字节，则**模型状态**约 $16P$ 字节。7B 对应 112 GB，即约 104.3 GiB，尚未计激活、通信 bucket 和 workspace。此处是明确配置下的估算：并非所有实现都保留 master 权重，梯度 dtype 也可能不同。不能把“7B BF16 权重约 14 GB”当作“全参训练约 14 GB”。ZeRO / FSDP 的分片对象与这张账本直接对应。[DeepSpeed ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)。

### Q：推理显存怎样估？给一个可手算的例子。

对普通 dense attention / GQA、忽略 KV 量化、滑窗和跨卡分片时：

$$
M_{KV}=2LBSH_{KV}D_hb.
$$

$L$ 为层数，$B$ 为并发序列数，$S$ 为每条缓存长度，$H_{KV}$ 为 KV heads 数，$D_h$ 为 head 维度，$b$ 为每元素字节数；前面的 2 表示 K 和 V。设 $L=32,B=8,S=8192,H_{KV}=8,D_h=128,b=2$，得到 **8 GiB KV**。每条约 1 GiB；若换成 32 个 KV heads，同样配置变为 32 GiB。还要另外容纳权重、激活、CUDA Graph 和临时内存。MLA、滑窗、KV 压缩或复制分片需要重新推导，不能直接套这个式子。[GQA 基础](#q=mha-gqa-mqa)、[PagedAttention 论文](https://arxiv.org/abs/2309.06180)。

### Q：LoRA、量化、activation checkpointing 能互相替代吗？

LoRA 减少可训练参数相关梯度与优化器状态，基础权重和多数前向计算仍存在；权重量化减少特定权重存储与部分执行成本，KV 量化则是另一项；activation checkpointing 保存部分边界激活，在 backward 时重算内部激活，交换计算与内存。它们作用于不同账目。冻结基础参数也不意味着完全不需要反向传播，因为梯度要到达各层 adapter。[PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)、[量化概览](https://huggingface.co/docs/transformers/quantization/overview)、[PyTorch checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html)。

## 4. 训练框架：DDP、FSDP、DeepSpeed、Megatron 怎样比较？

### Q：数据并行与模型并行，究竟“并行”了什么？

| 机制 | 切分对象 | 常见通信 | 收益与代价 |
|---|---|---|---|
| DDP | 输入数据；模型状态每个副本完整保留 | 梯度 all-reduce | 简单高效，但每个副本要放下完整状态 |
| ZeRO-1 | 优化器状态 | 分片状态所需同步 | 降低 optimizer 常驻内存 |
| ZeRO-2 | 再切梯度 | reduce-scatter 等 | 更省状态内存，通信与实现更复杂 |
| ZeRO-3 / fully sharded FSDP | 再切参数 | 参数 all-gather、梯度 reduce-scatter | 支持更大模型，但计算前会有聚合峰值 |
| TP | 层内张量与矩阵运算 | all-reduce / all-gather / reduce-scatter，依布局而定 | 降低单卡层内负担，频繁通信依赖互联 |
| PP | 模型层分给不同 stage | 激活与梯度点对点传输 | 可跨节点分层；有流水空泡与负载不均 |
| CP | 上下文 token 维 | K/V 交换等，依实现而定 | 分摊长上下文压力，新增 attention 通信 |
| EP | MoE experts | token dispatch / combine，常见 all-to-all | 分布专家，受路由偏斜和网络影响 |

FSDP 与 ZeRO-3 的内存目标相近，但参数组织、API、预取、重分片和 checkpoint 行为并非逐项相同。Sequence Parallel 常在 TP 组内减少部分激活副本，不能直接等同于完整的 Context Parallel。[FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[Megatron 并行文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.html)、[ZeRO 论文](https://arxiv.org/abs/1910.02054)。

### Q：FSDP 已经分片了参数，为什么还会 OOM？

参数常驻分片小，不代表计算瞬间也一样小。每个 FSDP 单元计算前要聚合需要的参数；预取下一个单元、当前激活和通信 buffer 可以同时存活。wrap 粒度过粗会抬高聚合峰值，过细又可能增加小 collective 开销。先看峰值发生在 forward、backward 还是 optimizer step，再检查 wrapping、prefetch、reshard、microbatch 和序列长度。[FSDP2 官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

### Q：DeepSpeed 与 Megatron 是二选一吗？

DeepSpeed 是分布式优化库，ZeRO、offload 和训练执行是常见入口；Megatron-LM / Megatron Core 重点是大规模 Transformer 的模型并行与高效实现。历史上也有 Megatron-DeepSpeed 集成。选择应比较**实际代码栈、模型支持、并行方案和维护成本**，不能把“ZeRO”和“TP”当成必然互斥。[DeepSpeed 文档](https://www.deepspeed.ai/)、[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。

### Q：Megatron-LM、Megatron Core 与训练脚本是什么关系？

Megatron Core 提供可复用的并行模型、通信和训练组件；Megatron-LM 仓库包含相关组件及训练示例。实际项目可能嵌入 Core，沿用自己的数据、loss、checkpoint 和作业启动逻辑。因此“我们用 Megatron”之后要问：用哪个 commit、哪种模型映射、哪些并行组件、是否使用自定义 fork。slime 与 verl 的 Megatron 后端也有自己的适配层。[Megatron Core 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)。

### Q：8 张卡，TP=2、PP=2、DP=2，全局 batch 怎么算？

这是一种不包含 CP/EP 的简化组合：$TP\times PP\times DP=8$。每个 DP 副本包含 4 张协同处理同一批样本的卡，独立数据副本数是 2。microbatch 为 4、梯度累积 8 步时：

$$
B_{global}=4\times8\times2=64.
$$

不能再把 TP 和 PP 乘进去得到 256。变长样本还应记录有效 tokens；局部先取平均再跨 rank 平均，未必等于全局有效 token 平均。MoE 的 expert-data-parallel group 与 dense 层可能不同，应根据实际 rank group 检查，而非机械乘所有并行度。[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。

### Q：为什么 TP 常放节点内，PP / DP 可以跨节点？

TP 的通信紧贴层内计算、频率高，通常更依赖 NVLink/NVSwitch 这样的快互联。PP 更多传 stage 边界激活，DP 更多同步梯度或分片状态，通信形态不同。这个经验不等于拓扑定律：跨节点链路、模型规模、CP/EP 布局、通信重叠都可能改变最优解。选型要测实际消息大小和阶段时间。[NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

## 5. 推理框架：vLLM、SGLang 与性能基础

### Q：vLLM 的 PagedAttention 解决什么？continuous batching 又解决什么？

PagedAttention 用逻辑块到物理块的映射管理 KV，减少预留大块连续空间带来的浪费，并支持共享等机制；continuous batching 在迭代过程中允许完成请求退出、新请求加入，减少整批等待。前者偏内存管理，后者偏调度。二者可以配合，但不应混为一个算法。分页也有尾块内部浪费和元数据成本，不是“显存零碎片”。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)、[vLLM 官方仓库](https://github.com/vllm-project/vllm)。

### Q：FlashAttention 与 PagedAttention 能一起用吗？

能。FlashAttention 通过分块和在线 softmax 避免将完整 attention score / probability 矩阵写入 HBM，减少中间量读写；PagedAttention 管理持久 KV 的物理布局。一个关心计算过程的 IO，一个关心请求状态的存放；具体兼容情况取决于 attention backend、设备、dtype 和模型结构。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)。

### Q：FlashAttention 为什么不需要存完整 softmax 矩阵？

对一行的分块 scores，维护最大值 $m$、指数和 $\ell$ 和未归一化加权和 $u$。合并新块时：

$$
m'=\max(m,\max_j s_j),\quad
\ell'=e^{m-m'}\ell+\sum_j e^{s_j-m'},\quad
u'=e^{m-m'}u+\sum_j e^{s_j-m'}v_j.
$$

最终输出为 $u/\ell$。旧块统计量可以根据新最大值重缩放，因此不必保存全部概率。它仍计算稠密 attention 所需 token 对；“不存 $S^2$ 矩阵”不意味着计算量自动变成线性，也不意味着浮点逐位相同。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。

### Q：SGLang 的 RadixAttention 与 prefix caching 是什么关系？

前缀缓存是复用相同 token 前缀的 KV；RadixAttention 是 SGLang 早期提出的一种自动管理复用的方式，把 token 序列前缀组织成 radix tree，查找共享路径，并配合缓存淘汰和调度。vLLM 也有自动前缀缓存，不能据此说“只有 SGLang 能复用 prefix”。缓存主要省重复 prefill；新 token 的 decode 仍然要做。[SGLang 作者博客](https://www.lmsys.org/blog/2024-01-17-sglang/)、[vLLM APC 文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)。

![SGLang 作者原图：radix tree 中的缓存复用与淘汰](./assets/infra/training-inference-frameworks/radix-attention.jpg)

**图源与解读：** 图来自上述 2024-01-17 作者博客，保留原作者归属。按 (1)—(9) 看，同一系统提示、对话历史和 few-shot 例子形成共享路径；新请求在共享节点后分叉；橙色虚线表示淘汰。图强调“哪些 KV 可以复用”，没有证明某引擎在今天所有场景都更快。判断命中要看 token 前缀、权重 / adapter 和引擎的 cache-key 语义，而不仅是肉眼看到的文字。

### Q：如何公平比较 vLLM 和 SGLang？

两者都是持续迭代的推理系统。SGLang 有结构化生成与 radix cache 的设计背景，vLLM 有 PagedAttention 与通用 serving 的设计背景；今天各自拥有更多相互重叠的能力。选型应固定模型、版本、精度、GPU、TP/DP、attention backend、prompt / response 长度分布、到达率和前缀命中率，再分别测 TTFT、TPOT、P95/P99、输出吞吐与失败率。对 RL 还要检查 logprob、在线权重更新、暂停恢复、工具交互接口和训练适配。本文不提供跨版本的单一速度排名。[vLLM 优化指南](https://docs.vllm.ai/en/latest/configuration/optimization/)、[SGLang 官方文档](https://docs.sglang.io/)。

### Q：TTFT、TPOT、吞吐与 goodput 怎么区分？

TTFT 是请求到首 token 的时间，包含排队与 prefill 等；TPOT 通常描述首 token 后的平均 token 间隔，流式场景也会观察每段 inter-token latency。吞吐需说明是请求数、输入 tokens 还是输出 tokens / s。goodput 是满足约定质量或延迟约束的有效工作量；应明确计算口径。批越大，吞吐可能越高，但排队和 TPOT 也可能恶化。只报“tokens/s”不足以判断服务是否更好。

### Q：chunked prefill 与 prefill/decode 分离分别有什么代价？

chunked prefill 将长输入分块，让调度器有机会穿插 decode；可能改善长 prompt 对生成请求的干扰，但 chunk 太小会增加调度或 kernel 开销。PD disaggregation 把 prefill 与 decode 放在不同资源池，分别配置设备和 batch，代价是 KV 传输、路由、负载配比与故障处理。只有算上网络搬运和排队后收益仍然成立，才值得采用。它和 RL 的 trainer/rollout 分离是两个不同维度。[vLLM 调优文档](https://docs.vllm.ai/en/latest/configuration/optimization/)、[SGLang PD 文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)。

### Q：TensorRT-LLM、llama.cpp 需要一起了解吗？

可以作为边界补充：TensorRT-LLM 面向 NVIDIA 推理优化栈，llama.cpp 常见于 GGUF、CPU / 本地 / 边缘运行。若目标是 NVIDIA GPU 服务，要检查 TensorRT-LLM 的模型、精度与部署路径；若目标是本地离线运行，llama.cpp 可能更贴近约束。它们的定位帮助你把“最适合当前场景”与“榜单最快”分开。[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)、[llama.cpp](https://github.com/ggml-org/llama.cpp)。

## 6. RL Infra：为什么训练里还需要推理引擎？

### Q：Actor、Rollout、Reference、Critic、Reward 是五个模型吗？

它们首先是**职责角色**，并不一定对应五份独立参数。

| 角色 | 做什么 | 是否训练 | 资源特点 |
|---|---|---|---|
| Actor / policy | 当前策略，计算可微 logprob 并更新 | 是 | 梯度、优化器、激活 |
| Rollout | 用策略生成 response 或 action | 通常负责生成，不在引擎内优化 | KV、调度、采样、工具等待 |
| Reference | 提供参考策略 logprob 等 | 通常冻结 | 前向；可能与 adapter 基座复用 |
| Critic | 估计 value，构造部分算法的优势 | PPO 等通常训练 | 另有训练状态；GRPO 常省略 |
| Reward / verifier | 给轨迹评分、检验答案或运行测试 | RL 循环中通常固定 | 可以是 CPU 规则、GPU 模型或外部环境 |

Actor 与 rollout 可以表示同一份策略在不同执行引擎中的副本。GRPO 没有 critic 不等于没有 reward；规则 reward 没有 GPU 模型也不等于没有评分成本。[verl HybridFlow Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)、[OpenRLHF 论文](https://arxiv.org/abs/2405.11143)。

### Q：一条完整 RL 样本应该带什么信息？

除了 prompt / response，至少要考虑 token IDs、attention / loss mask、终止原因、reward、group ID、sample ID、采样 logprob、policy version、采样配置、模型 / tokenizer / adapter 标识；多轮 Agent 还要记录工具输入输出、时间、环境状态与可重放边界。哪些字段必需取决于算法，但**行为策略是谁**与**哪些 token 接收 loss**不能含糊。[verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[slime Agentic RL Guide](https://thudm.github.io/slime/get_started/agent.html)。

例如工具返回一段网页，它是观察 token，不是 policy 采样出来的 action，通常不直接作为 policy-gradient 的动作训练；它仍可参与后续 action 的上下文。分段、多模态输入和重新渲染 chat template 时，要检查原始 token 与训练序列对齐。

### Q：rollout 的 logprob 与 trainer 重算的 logprob 为什么不一致？

要分开三类问题：输入不一致，如 template、token IDs、position IDs、mask；策略不一致，如权重版本、adapter、MoE 路由；执行数值不一致，如精度、kernel、batch shape、并行归约顺序。还要确认采样记录的概率究竟是温度 / top-p 等处理前还是处理后的分布，不能把不同分布的 logprob 直接混用。固定相同 tokens 和版本，先比较 teacher-forced per-token logprob，再逐项控制变量。[SGLang 确定性博客](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)、[verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)。

### Q：采样策略、旧策略、当前策略、参考策略怎么区分？

行为策略 $\mu$ 是实际生成数据的分布；$\pi_{old}$ 常表示 PPO 更新所固定的旧策略；$\pi_\theta$ 是正在更新的策略；$\pi_{ref}$ 常用于 KL 约束。理想同步场景下行为与旧策略可以对齐，但异步、采样变换或训推数值差异会让它们不同。概念上，行为校正会涉及：

$$
w_t=\frac{\pi_{old}(y_t\mid x,y_{<t})}{\mu(y_t\mid x,y_{<t})}
=\exp(\log\pi_{old,t}-\log\mu_t),
$$

而 PPO 更新比值是 $\pi_\theta/\pi_{old}$；reference KL 又是另一个对象。token / sequence 粒度、截断、拒绝采样等要按具体算法实现，不能看见一个比值就说问题已解决。[verl Rollout Correction 官方文档](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)。

### Q：同步、异步、共置、分离是什么关系？

这是两个独立轴：**同步 / 异步描述时间与数据依赖，共置 / 分离描述资源布局**。

| 模式 | 优点 | 代价 | 适合先检查的指标 |
|---|---|---|---|
| 共置 + 同步 | 同一批卡轮流生成和训练，资源复用、版本清晰 | phase 切换、offload、长尾屏障 | 切换时间、峰值内存、各 phase 占比 |
| 分离 + 同步 | 两边独立配置并行布局 | 保留同步屏障，两资源池可能轮流闲置 | 等待时间、权重同步、成本 |
| 分离 + 异步 | 生成与更新重叠，缓解长尾 | 策略陈旧、队列、校正、恢复更复杂 | queue age、policy lag、有效样本率 |
| 共置 + 受控重叠 | 可能更充分用硬件 | GPU 内存、SM 和通信竞争；实现依赖具体方案 | 争用与端到端收益 |

“异步调用 API”可能只表示并发生成请求，不代表训练已经消费旧策略数据；“训推分离”也不自动等于异步 RL。verl 当前有 fully async recipe，slime 也有相应示例，不能以旧教程把任何一方写死为同步。[verl Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html)、[slime fully_async 示例](https://github.com/THUDM/slime/tree/main/examples/fully_async)。

### Q：异步的理论收益怎样估？为什么队列必须有上限？

在各 stage 用独立资源的教学模型中，设采样 60s、评分 10s、更新 25s、串行同步 5s，同步循环约 100s；流水充分填满且每个 stage 可持续处理同样批量时，周期下界接近最慢 stage 的 60s。理想上限约 $100/60=1.67$ 倍，不是“用了异步就翻倍”。若评分依赖 CPU 长尾、同步阻塞生成或阶段共享 GPU，这个模型还要加争用。

生产率 $\lambda$ 长期大于消费率 $\mu$ 时，队列会积压；即使内存够，样本也越来越旧。需要 bounded queue、背压、按 policy lag 限制或算法允许的校正。丢弃超长 / 过慢轨迹会改变采样分布，必须检查任务偏差。AReaL 论文提供异步 RL 的系统与算法背景，但本文的 60/10/25/5 例子不是其测量数据。[AReaL 论文](https://arxiv.org/abs/2505.24298)。

### Q：训练与推理的权重同步，为什么不是直接 copy？

trainer 可能用 FSDP 或 TP×PP×EP 分片，rollout 用另一种 TP/EP；同名参数的分片边界、融合 QKV 排列、expert 归属、dtype 都可能不同。必须建立 logical tensor 到双方 shard 的映射，再传输、重排和校验。避免把完整大模型先聚到 rank 0 / CPU，能减少峰值内存与瓶颈；但具体传输方案要按后端和拓扑选择。

发布还需要明确切换边界：新请求能否拿到完整新版本？旧请求继续旧版本、被暂停还是截断？KV 属于生成它的权重版本，不能在版本变化后直接假设它有效。双 buffer、版本 ACK、暂停恢复或 per-request 策略各有内存和语义代价。[slime 权重同步实现入口](https://github.com/THUDM/slime/tree/main/slime/backends/megatron_utils/update_weight)、[verl Fully Async 文档](https://verl.readthedocs.io/en/latest/advance/fully_async.html)。

## 7. slime：SGLang 原生的后训练框架

### Q：slime 的核心架构是什么？

slime 主线把 Megatron 训练、SGLang rollout 和 Data Buffer 连接起来，Ray 用于资源 / worker 管理与异步执行，自定义生成接口用于接入任务、奖励与 Agent 环境。重点是复用训练和推理底座，并处理二者之间的运行与权重更新，而非重写一个模型训练引擎。[slime 作者博客](https://www.lmsys.org/blog/2025-07-09-slime/)、[官方 README](https://github.com/THUDM/slime)。

![slime 作者原图：Data Buffer、custom rollout、Megatron 与 SGLang 的连接](./assets/infra/training-inference-frameworks/slime-architecture.png)

**图源与解读：** 来自 slime 团队 2025-07-09 的 LMSYS 博客，保留原作者归属。按箭头看：Data Buffer 给 custom rollout 提供 prompt，后者通过 router 调 SGLang server，返回训练数据；Megatron 消费数据并把更新后的权重传给生成服务。图中的 4 个训练 GPU 与 2×2 个推理 GPU 是示意，并非推荐比例，也不表示所有组件永远独占卡。Data Buffer 是数据组织桥梁，不能仅凭名称将它理解为任意旧样本都能重复使用的 off-policy replay buffer。

### Q：“SGLang-native” 对使用者意味着什么？

主线直接利用 SGLang 的服务与优化接口；训练侧沿用 Megatron 的能力，并提供配置传递和独立调试路径。对已经有 Megatron / SGLang 栈的团队，这能降低重新接线的成本。但“原生”并不保证所有上游参数组合都经过联合测试，也不消除模型格式、checkpoint 与数值一致性的核对工作。2025 年博客中的规划不应直接视为 2026 年的能力清单，当前能力以固定版本仓库为准。[作者设计说明](https://www.lmsys.org/blog/2025-07-09-slime/)、[slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)。

### Q：什么情况下优先评估 slime？

已经在使用 Megatron，尤其需要大模型 / MoE 的模型并行，并且准备采用 SGLang 做 rollout；需要自定义生成、代码执行或长尾 Agent 轨迹时，slime 是值得验证的候选。这里是根据架构作出的选型判断，不是性能优越性的实测结论。代价包括 Megatron 的模型适配、并行配置、格式转换，以及训练和 serving 版本联调。第一次试验应先用官方支持的小模型验证闭环，再加多轮工具、MoE 与异步。[slime Agentic RL Guide](https://thudm.github.io/slime/get_started/agent.html)。

### Q：源码怎么读才不会陷进所有配置？

从 `train.py` 追一次 batch：资源初始化 → rollout manager → sample / reward → training dispatch → loss → weight update。当前 README 推荐的入口包括 `slime/ray/placement_group.py`、`slime/ray/rollout.py`、`slime/rollout/sglang_rollout.py`、`slime/ray/actor_group.py` 和 `slime/backends/megatron_utils/`。先写清输入输出和版本边界，再读传输优化；路径可能随 commit 调整。[slime 官方代码阅读路线](https://github.com/THUDM/slime#code-reading-path)。

## 8. verl：用 HybridFlow 把 RL 控制流与计算流分开

### Q：HybridFlow 的 “hybrid” 在讲什么？

算法级控制流描述“先生成、再评分、算优势、更新策略”；计算流描述一个角色内部的多卡 forward / backward / optimizer。HybridFlow 让上层控制逻辑以接近单进程的方式编写，底层 Worker / WorkerGroup 组织分布式计算与数据派发。这种设计兼顾算法可修改性与计算后端复用，不是简单把“混合精度”叫成 HybridFlow。[HybridFlow 论文](https://arxiv.org/abs/2409.19256)、[verl Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)。

![verl 官方文档引用的原图：Driver、WorkerGroup 与 Resource Pool](./assets/infra/training-inference-frameworks/verl-driver-workers.png)

**图源与解读：** 来自 verl HybridFlow Guide 引用的 `verl-community/docs/driver_worker.png`，保留原作者归属。左侧 Driver 调角色 API、接收 Future；中间每个 WorkerGroup 管理一组计算 worker；右侧 Resource Pool 指定使用哪些 GPU。图中 Actor 与 Critic、Reference 与 Reward 各共享一个资源池，是可配置 placement 的示意，并非所有任务的默认配置。它说明逻辑角色与物理 GPU 布局可以分开；不能据此推出所有大 tensor 必须经过 Driver，也不能把 Future 直接解读为完全异步 RL。

### Q：verl 是否只能用 FSDP？是否只支持单轮同步训练？

都不是。核对时官方仓库 / 文档有 FSDP/FSDP2、Megatron 等训练路径，vLLM / SGLang rollout 路径，也有 Agent Loop 和 Fully Async Policy Trainer recipe。是否能同时组合指定模型、后端、异步模式和特性，需要核对该版本的 recipe 与限制。不能把“多个后端各自支持”写成“所有组合任意切换”。[verl 仓库](https://github.com/verl-project/verl)、[Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)、[Fully Async recipe](https://verl.readthedocs.io/en/latest/advance/fully_async.html)。

### Q：什么时候优先评估 verl？

需要比较多种训练 / rollout 组合、修改 RL 算法控制流、研究已有 recipe 或接入 Agent Loop 时，verl 是值得评估的候选。代价在于理解角色数据协议、资源池、engine 生命周期和具体后端配置；灵活也意味着更多组合需要自己验收。先读最接近目标任务的 recipe，不必同时搞懂每个 backend。[verl 官方文档](https://verl.readthedocs.io/en/latest/)。

### Q：源码阅读从哪里开始？

可从 `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py` 看角色构建和训练循环，再追 `verl/protocol.py` 的数据协议、`verl/workers/engine/` 的训练执行与 `verl/workers/rollout/` 的生成执行；异步 recipe 另追其专用入口。旧博客可能还引用 legacy worker 文件，按来源快照查路径。先在纸上列出一个 batch 的字段，再逐次标记在哪个角色产生、在哪个角色消费。[verl HybridFlow Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)。

## 9. 横向对比与选型：怎样回答“slime 和 verl 有什么区别”？

### Q：可以用同一组维度比较吗？

| 维度 | slime 主线 | verl | 面试时应补充 |
|---|---|---|---|
| 组织思路 | Megatron + SGLang + Data Buffer，自定义 rollout | 控制流 / 计算流分离，WorkerGroup / Resource Pool | 二者都要管理采样、更新与同步 |
| 训练侧 | 重点围绕 Megatron 集成 | 官方提供 FSDP/FSDP2、Megatron 等路径 | 同模型支持与版本组合需验证 |
| 生成侧 | 重点围绕 SGLang 原生集成 | 官方提供 vLLM / SGLang 等路径 | logprob、更新接口与多轮能力同样重要 |
| 自定义入口 | rollout / reward / 数据接口和训练后端适配 | 算法控制流、reward、Agent Loop、backend 接口 | 接口存在不等于环境自动接好 |
| 异步 | 有 fully_async 示例等路径 | 有 fully_async recipe | 比较具体队列、陈旧度与校正规则 |
| 更适合先验证的情境 | 已有 Megatron / SGLang，强调大模型、MoE 与自定义采样 | 后端组合与算法实验，已有 verl recipe | 这是选型推断，不是通用性能排名 |
| 主要联调风险 | 模型映射、Megatron 成本、训推格式与数值 | 组合兼容、数据协议、角色与 engine 配置 | 看端到端质量与成本，不只看生成速度 |

该表只总结已核对的主线定位；独立衍生项目、实验后端和社区扩展不自动算作主仓库的稳定能力。依据：[slime](https://github.com/THUDM/slime)、[verl](https://github.com/verl-project/verl)。

### Q：30 秒的回答怎么说？

“slime 和 verl 都处在 RL 后训练编排层，连接 rollout、奖励、策略更新与权重同步。slime 主线偏向 Megatron 与 SGLang 的原生集成，用 Data Buffer 和自定义生成接口组织数据；verl 用 HybridFlow 分离算法控制流和分布式计算，提供多个训练和 rollout 后端。选型看既有模型栈、并行布局、任务接口、异步一致性和可复现的成本质量结果。不能只凭框架名字断言谁更快。”

### Q：OpenRLHF、AReaL、TRL 在什么位置？

| 候选 | 官方定位 / 常见入口 | 值得先看什么 | 不应直接推断什么 |
|---|---|---|---|
| OpenRLHF | Ray、DeepSpeed、vLLM 与 HF 生态的 RLHF / Agentic RL 栈 | 角色布局、算法 recipe、异步与 agent 接口 | 不能只按早期 PPO 教程描述今天全部能力 |
| AReaL | 异步 RL 系统与 Agent 训练接口 | rollout / train 解耦、陈旧度和算法匹配 | 异步收益不是所有任务固定倍率 |
| TRL | HF 生态中的 SFT、DPO、GRPO 等 trainer | 与 Transformers / PEFT 配合、生成集成 | 不能绝对说它只适合单卡或完全没有 vLLM |

这些也值得关注，但本文重点是建立框架地图，避免扩成无边界项目目录。[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)、[AReaL](https://github.com/areal-project/AReaL)、[TRL](https://huggingface.co/docs/trl/index)。

### Q：如何把选型变成能验证的决定？

先填写任务约束：dense / MoE、参数规模、最长上下文、每 prompt 回答数、多轮工具、奖励来源、GPU / 网络、预算与评估集。随后检查官方支持组合，完成最小闭环，再分别跑 trainer-only、rollout-only 与端到端。最终比较每 GPU-hour 的**有效训练 tokens / 样本**、达到目标质量的成本、失败率与维护成本。具体框架 recommendation 是上述约束下的判断，应随版本更新重新检查。

例如只有 8 张卡，先做 dense 7B 数学 RL：重点是兼容的小模型 recipe、奖励正确性和可观察的同步闭环。若已有大型 MoE Megatron checkpoint，改用 SGLang rollout：重点变成 EP/TP 映射、权重同步、专家路由一致性和内存峰值。若任务大量调用慢工具：环境延迟、并发限制、队列与 lag 可能比 GPU kernel 更先决定收益。

## 10. 排障与系统设计：用现象组织知识

### 场景 A：GPU 显存占满，训练利用率很低

先把“占了多少内存”与“单位时间做了多少有效工作”分开。采集 CPU / GPU timeline，标出数据读取、生成、工具、reward、参数传输、forward / backward、optimizer 与 checkpoint。若 GPU 等数据，先修供给；若 NCCL 长时间占用，看通信与拓扑；若小 kernel 密集，查 shape、融合和 launch；若更新高效但训练卡长期等 rollout，回到 RL 流水。`nvidia-smi` utilization 不能等同于 MFU。[PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

### 场景 B：rollout tokens/s 提升了，训练质量却下降

检查 tokens/s 的增量是否来自更长输出、低质量样本或不同温度。对齐 prompt、奖励、有效 mask、生成长度、总训练 tokens 与评估预算。再比较 policy lag、行为 / trainer logprob 偏差、group 完整性、超时丢弃比例和 reward hacking。更快的生成可以生产更多无效工作；只有相同质量目标下的总成本下降，才是端到端改善。[verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)、[SGLang 确定性说明](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)。

### 场景 C：长 response 下，共置模式 OOM

在阶段切换处检查训练 optimizer、激活、rollout KV、重复权重、图捕获内存是否同时驻留。减少最大长度 / 并发只是在限制峰值，后续应确认 offload / sleep 生命周期真的完成，以及取消请求是否释放资源。记录 max allocated / reserved 并区分 allocator 缓存和实际 live tensors。切成分离模式可能降低竞争，也可能增加独立权重副本和空闲成本；要重新算账。

### 场景 D：NCCL hang，最后报错的 rank 是根因吗？

不一定。某 rank 先 OOM 或异常退出，其余 rank 才在 collective 超时。按时间和 collective 序号找**第一个分歧**，检查所有参与 rank 的调用顺序、shape、dtype、group membership，再检查设备映射与网络。先单机 / 两节点最小复现，`nccl-tests` 正常只能说明该测试模式正常，不能排除真实作业调用错误或并发通信争用。[NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)。

### 场景 E：8 卡 RL 系统怎样分配训练与生成资源？

先确定模型能否在每个候选布局装下；分别测训练每批有效 tokens/s、生成每条轨迹 tokens/s、评分延迟及权重同步成本。假设只为教学，两个 2 卡生成实例生产速度合计 100 条有效轨迹 / min，4 卡 trainer 消费 120 条 / min，则 trainer 经常饥饿；增加训练卡未必有用。若任务变长，让生成降到 50 条 / min，就需要调整资源配比或处理长尾。**4:4 是示例而非推荐值**，正式选择应固定总卡时成本做 A/B。

### 场景 F：异步队列和恢复边界如何设计？

每条记录至少有 sample / group ID、policy version、状态和可审计 reward。组内相对算法先保证 group 语义，再按算法允许的 lag 消费。队列设容量、age / lag 上限、背压与拒绝计数；不要只设 max length。sample retry 要幂等，工具是否能重放需单独定义。训练 step 失败后，model、optimizer、scheduler、RNG、数据位置、队列消费状态与已发布策略版本要匹配，避免重复计入或跳过数据。[PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)、[slime Fault Tolerance](https://thudm.github.io/slime/advanced/fault-tolerance.html)。

## 11. 面试练习：60 道问题、回答要点与追问

以下问题重新组织了训练、生成和 RL 系统需要掌握的主题。**公开题库主题只证明有公开练习线索，不证明公司、年份或高频次数。** 主要线索来自 [LLM Training Infra 公开题库](https://github.com/XFWang522/llm-rl-infra-interview) 和 [AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide)；技术答案依据各节官方资料。为了便于检索，题号保持稳定；题目分组内的答案是口述提纲，详细推导回到正文。

### 11.1 基础、显存与训练目标

**Q01 [公开题库主题] 只有权重大小，能判断模型能否训练吗？** 不能；列权重、梯度、优化器、master、激活和临时 buffer，再检查分片与峰值。**追问：** 为什么 LoRA 的基础权重冻结了仍会 OOM？依据第 3 节。

**Q02 [本文延伸] 为什么 SFT 能并行处理位置，rollout 却通常逐 token？** teacher forcing 已知目标，causal mask 隔离未来；生成依赖上一步的采样结果。**追问：** speculative decoding 是否消除了概率上的因果依赖？没有，只改变验证执行。

**Q03 [公开题库主题] 梯度累积改变了哪些量？** 固定 microbatch 时可增大 global batch，减少 optimizer 更新频率；不会把每步激活峰值按累积步数复制保存。**追问：** DDP `no_sync` 怎么影响同步？需在合适累积边界恢复通信。[DDP 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

**Q04 [公开题库主题] BF16、FP16、FP8 怎么比较？** 区分指数范围、有效精度、累积 dtype 与硬件支持；FP16 常涉及 loss scaling，BF16 不表示绝不会 NaN。**追问：** 训练与 rollout 不同精度如何验证？固定 token 比较 logprob 与任务质量。[Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/introduction/introduction.html)。

**Q05 [公开题库主题] activation checkpointing 为什么省显存？** 保存边界，在 backward 重算内部；代价是额外计算，正确性关心 RNG、状态副作用和重算路径。**追问：** 重算与 checkpoint 文件存盘有何区别？前者是计算机制，后者是恢复资产。[checkpoint 文档](https://docs.pytorch.org/docs/2.14/checkpoint.html)。

**Q06 [本文延伸] DPO 为什么一般不需要在线 rollout？** 标准离线 DPO 直接在固定偏好对计算 loss；数据收集可以另行生成。**追问：** 在线 DPO 的系统需求会怎样变？需要增加采样与版本管理。[DPO](https://arxiv.org/abs/2305.18290)。

**Q07 [公开题库主题] MFU 与 GPU utilization 相同吗？** MFU 衡量模型有效 FLOPs 相对约定峰值的比例；utilization 只反映设备忙碌情况等观测，通信忙不等于模型计算高效。**追问：** 重算 FLOPs 是否计入？明确 MFU / HFU 和分母口径。[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)。

**Q08 [公开题库主题] 训练突然 NaN，先查什么？** 定位第一步异常、数据 / loss / 梯度分布，核对 dtype、scale、clip、kernel 和 rank 一致性。**追问：** 能否直接降低学习率？可做验证，但先留证据，避免掩盖数据或实现错误。

### 11.2 分布式并行与通信

**Q09 [公开题库主题] DDP 为什么不能自动容纳更大的模型？** 每个副本仍保留完整模型状态，主要分数据与同步梯度。**追问：** DDP bucket 怎样与 backward 重叠？按梯度就绪启动通信。[DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

**Q10 [公开题库主题] ZeRO 三阶段分别省哪一项？** 依次分优化器、梯度、参数；逐项代入显存账本。**追问：** ZeRO-3 能否把总峰值直接除以卡数？不能，激活、聚合与临时量另算。

**Q11 [公开题库主题] FSDP wrapping 太粗或太细会怎样？** 太粗抬高参数聚合峰值，太细增加小通信与调度；还影响预取重叠。**追问：** 自动 wrap 应参考哪些 profile？参数、激活、执行时间和消息大小。

**Q12 [公开题库主题] TP 的 column / row parallel linear 怎样连起来？** 列切输出维后，可让下一层按输入维 row 分片计算局部部分，再做求和；实际 layout / SP 会改变 collective。**追问：** 为什么中间 activation 不必每次聚成完整？下游可消费相容分片。[Megatron TP 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.tensor_parallel.html)。

**Q13 [公开题库主题] PP 的 bubble 从哪里来？** 流水填充、排空和 stage 不均衡造成等待；更多 microbatch 有助摊薄，但影响 batch、调度与内存。**追问：** 1F1B 相比 GPipe 为什么能降低激活驻留？前后向更早交错。[PyTorch Pipeline](https://docs.pytorch.org/docs/2.14/distributed.pipelining.html)。

**Q14 [公开题库主题] TP / PP / DP 怎样决定 global batch？** 只有独立数据副本数乘进去，协同处理同一批的模型并行卡不算新样本。**追问：** tokens 不等长时 loss 如何归一化？按目标语义加权。

**Q15 [公开题库主题] CP 与 SP 为什么不是一个概念？** CP 面向上下文维切分与 attention 交互；Megatron 常见 SP 在 TP 组内分摊部分序列激活，具体边界依实现。**追问：** 每层通信模式不同在哪里？查实际 attention 路径。[Megatron CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)。

**Q16 [公开题库主题] MoE EP 的瓶颈在哪里？** dispatch / combine 通信、expert 负载偏斜与不等长计算；激活参数少不意味着所有权重都不用存。**追问：** dropless 是否消除了长尾？没有。[Megatron MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。

**Q17 [公开题库主题] all-reduce 与 reduce-scatter + all-gather 什么关系？** 相同归约目标下可分解为两阶段，具体数值顺序与算法实现另论。**追问：** ring 大消息的每 rank 传输量？简化均匀块模型约 $2(N-1)M/N$。[NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

**Q18 [公开题库主题] 调用异步 collective 就能重叠吗？** 还要有独立可执行计算、正确 stream 依赖、资源余量与适合 bucket；过早 wait 会串行化。**追问：** NCCL 占 SM 会有什么影响？可能争用计算资源。

**Q19 [公开题库主题] 为什么跨节点比单机慢很多？** 测实际消息大小、NIC / NUMA 亲和、带宽、拥塞、拓扑和 collective；不要只看理论链路峰值。**追问：** nccl-tests 正常说明什么？只说明测试覆盖的配置正常。

**Q20 [公开题库主题] 分布式作业 hang 怎样定位？** 找最早 rank 异常和 collective 分歧，查顺序、shape、group 与进程退出，再查网络。**追问：** 什么日志必须留？rank、step、collective 序号、时间与异常。[NCCL 故障文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)。

### 11.3 推理内存与服务调度

**Q21 [本文延伸] KV 的容量为什么与 KV heads 而非 Q heads 直接相关？** 缓存的是 K/V；GQA 多个 Q heads 共享 KV heads。**追问：** TP 分片会不会复制 KV？取决于 head 数和实现。

**Q22 [本文延伸] PagedAttention 与操作系统虚拟内存类比有哪些边界？** 都有逻辑到物理映射；这里管理的是 GPU KV blocks，不代表拥有一般 CPU 虚拟内存所有机制。**追问：** 尾块浪费怎样估？根据 block size 与序列长度分布。

**Q23 [本文延伸] continuous batching 能自动减少 TTFT 吗？** 不一定，接纳策略、长 prefill 干扰、负载和 KV 容量决定首 token 延迟。**追问：** 调大并发怎样影响 TPOT？测同一到达率下的分位数。

**Q24 [本文延伸] prefix cache 为什么看着一样却没命中？** token 前缀、template、adapter、权重版本或 cache-key 规则可能不同。**追问：** cache hit 会免除 decode 吗？不会。[vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)。

**Q25 [公开题库主题] FlashAttention 是近似 attention 吗？** 目标是 exact dense attention 的 IO 优化，浮点执行顺序不同可有数值差异。**追问：** online softmax 如何合并块？写第 5 节的 $m,\ell,u$ 更新。

**Q26 [本文延伸] chunked prefill 的 chunk 越小越好吗？** 不是，调度公平与 kernel 效率要权衡。**追问：** 该看哪些指标？长短请求 TTFT / TPOT、吞吐与排队。

**Q27 [本文延伸] PD 分离什么时候收益可能消失？** KV 搬运、网络排队、pool 失衡超过隔离收益时。**追问：** RL 训推分离和 PD 分离能共存吗？能，它们切分不同阶段。

**Q28 [本文延伸] vLLM 与 SGLang 怎么做基准？** 固定版本、硬件、模型、精度、长度、到达率、cache 与并行布局；包含失败与尾延迟。**追问：** 为什么不能只比较一个离线大 batch？在线负载存在排队与 SLO。

**Q29 [公开题库主题] 权重量化一定让推理更快吗？** 取决于支持 kernel、shape、反量化、瓶颈和 batch；容量下降不等于时延等比例下降。**追问：** KV 量化与权重量化影响哪条带宽路径？分别算账。[量化官方概览](https://huggingface.co/docs/transformers/quantization/overview)。

**Q30 [本文延伸] CUDA Graph 为什么对 decode 常有帮助？** 复用捕获执行可减少 launch 开销；动态 shape、图内地址和预留内存限制适用范围。**追问：** graph 为什么会增加内存预算？检查捕获与缓存配置。[CUDA Graph](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

### 11.4 RL 数据流、正确性与权重同步

**Q31 [公开题库主题] 为什么不能用纯 trainer 完成在线 RL？** 还需要生成、环境 / reward、样本整理与策略发布；teacher-forced logprob 不是新轨迹采样。**追问：** 哪些阶段不需要 GPU？看 reward 与工具类型。

**Q32 [本文延伸] GRPO group 的完整性为什么重要？** 相对优势依赖同 prompt 的组内结果；缺样本或按长短筛掉会改变统计与分布。**追问：** 一条超时怎么办？按定义处理失败或重新生成，并记录组语义。[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer)。

**Q33 [公开题库主题] Actor 和 rollout 为什么使用不同并行度？** 训练有 optimizer / gradient / activation，生成有 KV 与逐步 decode，最优布局不同。**追问：** 怎样从 trainer shard 映射到 serving shard？先定义 logical tensor。

**Q34 [本文延伸] 参考模型与采样旧模型相同吗？** reference 用于约束等目标，行为 / old policy 用于采样与更新；职责不同。**追问：** 三种 logprob 混用会怎样？错误比值或 KL 导致训练信号变形。

**Q35 [公开题库主题] 权重更新频繁时怎样保证版本完整？** 记录版本、校验映射、定义切换边界与 ACK；不让新请求用到半新半旧权重。**追问：** 老请求 KV 怎么处理？按版本隔离、暂停或重建。

**Q36 [公开题库主题] rank 0 聚合完整权重有什么问题？** 参数与缓冲峰值、串行瓶颈和 CPU 复制；大模型应评估 shard-to-shard 传输。**追问：** 小模型一定不能用吗？可作为简单基线，先量化成本。

**Q37 [本文延伸] rollout / trainer logprob 不同就一定是 off-policy 吗？** 先排输入、版本、采样变换和数值差异；off-policy 是分布问题，不能替代故障归因。**追问：** 怎样设计最小比对？同 tokens、版本、mask、dtype。

**Q38 [公开题库主题] 共置为什么可能节省卡时却更易 OOM？** 角色复用卡，phase 状态切换不完全或缓存未释放时内存叠加。**追问：** 各阶段都能单独跑，为何组合失败？组合峰值与生命周期不同。

**Q39 [公开题库主题] 异步队列为什么不能无限大？** 长期积压增加内存、queue age 与策略 lag，降低样本有效性。**追问：** 背压与丢弃怎么取舍？同时关注速率稳定和采样偏差。

**Q40 [公开题库主题] partial rollout 中途换策略要记录什么？** token 段的行为版本 / logprob、KV 来源、截断与恢复规则，明确混合版本轨迹如何用于算法。**追问：** 是否只存一个 sequence version 就够？不一定。

**Q41 [公开题库主题] 生成速度提升，端到端提升为何很小？** Amdahl 约束，reward、训练或同步成为瓶颈；画各阶段时间线。**追问：** 快路径为什么可能只是输出更长？同时比较有效样本与质量。

**Q42 [本文延伸] token-level 与 sequence-level importance weight 有何不同？** 后者组合整段比值，可能方差更大；截断与归一化改变估计。**追问：** correction 能弥补错 tokenizer 吗？不能，先修输入。[verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)。

**Q43 [公开题库主题] reward 成为瓶颈时先加 rollout 卡吗？** 先测评分服务和队列；可做评分 batch、缓存或并发，但保持奖励语义、超时与幂等。**追问：** 失败当 0 reward 合理吗？取决于任务定义，不应悄悄更改。

**Q44 [公开题库主题] 多轮 Agent 的工具 token 要不要训练？** 观察通常不作为 policy action 直接算 policy-gradient，但是后续 action 的条件；按 loss mask 区分。**追问：** 把轨迹转回文本再 tokenize 有何风险？模板与 token 对齐可能改变。

### 11.5 框架比较与源码阅读

**Q45 [本文延伸] 30 秒解释 slime vs verl。** 同层后训练编排；slime 主线强调 Megatron / SGLang，verl 强调控制流分离和多后端；选型用兼容性、接口、质量成本。**追问：** 你是否真的跑过？未运行就说明是资料判断。

**Q46 [公开题库主题] WorkerGroup 与 Resource Pool 解耦解决什么？** 逻辑计算角色可以映射到不同物理资源，不必把算法写死在每 rank 中。**追问：** Driver 是否该持有所有大 tensor？避免按图误读实际传输路径。

**Q47 [本文延伸] slime 的 Data Buffer 是传统 replay buffer 吗？** 它是 prompt 与生成数据组织的桥梁；是否复用旧数据、允许多大 lag 由训练流程决定。**追问：** 持久化队列自动保证恢复吗？还要对应消费与模型版本。

**Q48 [本文延伸] 为什么“verl 只能同步”是错误表述？** 当前官方有 fully async recipe；按具体版本与路径说能力。**追问：** 异步 serving 与异步学习分别指什么？看 request 并发和采样更新的依赖。

**Q49 [本文延伸] 怎样给一个新 reward 接口做验收？** 固定输入输出、mask 对齐、单位、失败处理、确定性与批处理等价性；用成功 / 失败 / 超时样本验证。**追问：** 奖励高却任务失败怎么办？检查验证器覆盖与 hacking。

**Q50 [本文延伸] 需要扩展新的 rollout backend，会改哪几层？** 生成请求 / 响应协议、token / logprob 捕获、暂停恢复与权重加载、调度和模型兼容。**追问：** 只接 OpenAI-style HTTP API 为什么可能不够？训练需要版本与 token 级信息。

### 11.6 容错、排障与现场设计

**Q51 [公开题库主题] 能恢复训练的 checkpoint 需要什么？** 模型、optimizer、scheduler、RNG、训练步与数据位置；RL 再关联已发布版本与队列消费状态。**追问：** 改并行度恢复是否总能成功？检查分片格式与 reshard 支持。

**Q52 [公开题库主题] 异步存 checkpoint 的一致性风险是什么？** 写盘时参数还在更新，快照须一致；额外 staging 内存、后台失败和完成标记也要处理。**追问：** 文件存在就可恢复吗？先看完整性与提交标记。[PyTorch DCP](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)。

**Q53 [公开题库主题] Ray placement group 能替代 NCCL 吗？** 不能，前者管理逻辑资源预留与放置，后者负责 GPU collective；二者协作。**追问：** 逻辑 GPU 数是否能强制限制进程显存？不是显存隔离机制。[Ray Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html)。

**Q54 [公开题库主题] token throughput 逐渐下降，先看哪些趋势？** 输出 / 序列长度、有效 mask、cache / 重试、queue age、数据供给、热降频、通信与 checkpoint 周期。**追问：** 固定 token 数下比较是否更合理？明确成本口径。

**Q55 [公开题库主题] rollout worker、reward worker、trainer rank 失败分别怎么恢复？** request / group、评分幂等与训练 step 是不同边界；先定义何时可以重放或必须回滚。**追问：** 只重启失败 rank 会怎样？collective 与参数状态可能不一致。

**Q56 [公开题库主题] 怎样实现 bounded blocking queue？** 定义容量、阻塞 / 超时、关闭唤醒、入队出队原子性和取消；消费者退出不能让生产者永远等。**追问：** policy version 淘汰在哪做？明确入队和消费边界。

**Q57 [公开题库主题] 给多 rank 日志，怎样找第一次 collective 分歧？** 按 job、group、sequence 对齐调用，比较 shape / dtype / 时间，追先前异常。**追问：** 日志时钟不完全一致怎么办？结合序号与因果关系。

**Q58 [本文延伸] 设计一个 8 卡、慢工具 Agent RL 实验。** 先做小模型同步闭环，测工具 P95/P99、生成和训练速率；再加入有界异步与 lag 上限。**追问：** 如何保持基准公平？固定总 GPU-hour、数据、奖励和评估预算。

**Q59 [公开题库主题] 怎样做训练性能回归检查？** 固定模型、版本、shape、精度、硬件与 warmup，多次测量后设噪声容忍；先验证正确性再比速度。**追问：** kernel 快 20% 为什么 step 没变化？该 kernel 占比或新瓶颈。

**Q60 [本文延伸] 如何证明自己理解一个框架，而非只会启动脚本？** 画数据流与资源布局，指出 logprob / mask / 版本在哪里产生，解释一次 OOM 或等待，并提出可证伪的实验。**追问：** 哪些是本人测量、作者测量和教学推演？分别说明。

## 12. 最小验证路线与复习清单

### Q：没有大集群，怎样开始？

第一步做不用 GPU 的纸面或 CPU 验证：手算 7B 状态与 KV；画 8 卡 TP×PP×DP；实现 bounded queue，给 80% 轨迹 0.1s、20% 轨迹 2s 的模拟延迟，观察同步长尾与 queue age。模拟只验证调度概念，不能证明 RL 收敛。

第二步有 GPU 时选官方支持的小模型，固定 checkpoint、镜像、tokenizer 和数据，先分别验证 generation、reward、teacher-forced logprob、loss mask 和单次 optimizer update。第三步验证权重发布后新请求的版本与输出概率；第四步才加多 GPU、分离、异步、多轮工具和精度优化。每次只增加一个变量，保存失败样本和配置。本文未执行这些 GPU 实验，后续运行结果应单独登记。

### Q：一份合格的实验账本至少写什么？

| 类别 | 必填项 |
|---|---|
| 可追溯版本 | 编排框架、训练 / 推理引擎 commit，镜像 digest，模型与 tokenizer revision |
| 硬件与布局 | GPU 型号 / 数量 / 显存，节点与网络，TP / PP / DP / CP / EP，角色放置 |
| 数据与算法 | prompt 来源，长度分布，group size，loss / reward / mask，采样参数，policy lag 限制 |
| 执行成本 | 各阶段时间，峰值内存，通信与同步，成功 / 失败 / 重试，GPU-hour |
| 质量 | 固定评估集与 seed，reward 和外部成功率，有效样本，长度与分布变化 |
| 结论边界 | 作者数字、本次实测或估算；哪些组合没测，是否存在偏差 |

复习时可尝试不看正文完成六件事：画层次地图、解释训练 / 生成不同、手算内存、区分并行机制、说出 slime / verl 的差别、按症状定位瓶颈。每个答案都加一句“我要收集什么数据来验证”。

## 13. 来源、原图与更新记录

### 技术原始资料

核对日期统一为 2026-10-03；本文按职责和问题组织资料，没有宣称复现作者 benchmark。仓库维护者 / 文档中的性能数字只在明确实验配置下成立。以下入口用于继续阅读，代码 commit 与图像校验记录见仓库的 `content/research/framework-sources.json` 和 `assets/infra/training-inference-frameworks/figures.json`。

| 来源 | 类型 | 支持的内容 / 阅读重点 |
|---|---|---|
| [slime 官方仓库](https://github.com/THUDM/slime) | 官方代码 | 架构、配置、源码路线与当前示例 |
| [slime 设计博客，2025-07-09](https://www.lmsys.org/blog/2025-07-09-slime/) | 作者博客 | SGLang-native 设计动机；架构原图；历史规划需区分 |
| [slime Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) | 官方文档 | 环境、数据与训练 / rollout 参数入口 |
| [slime Agentic RL Guide](https://thudm.github.io/slime/get_started/agent.html) | 官方文档 | 多轮生成、样本与任务接口 |
| [verl 官方仓库](https://github.com/verl-project/verl) | 官方代码 | 多后端、算法 recipe 与版本支持 |
| [HybridFlow](https://arxiv.org/abs/2409.19256) | 原论文 | 控制流 / 计算流与分布式 RL 系统设计 |
| [verl HybridFlow Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) | 官方文档 | Driver、WorkerGroup、Resource Pool 原图与源码 |
| [verl Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html) | 官方 recipe | 异步、streaming、partial rollout 和限制 |
| [verl Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html) | 官方文档 | Agent rollout 接口与数据组织 |
| [verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) | 官方文档 | 行为分布、训推差异与校正方法 |
| [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) / [ZeRO 论文](https://arxiv.org/abs/1910.02054) | 官方文档 / 论文 | 模型状态分片与资源账本 |
| [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 官方教程 | 分片、聚合、reshard 与预取 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) / [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) | 官方代码 / 文档 | TP / PP / CP / EP 与实现入口 |
| [vLLM 仓库](https://github.com/vllm-project/vllm) / [PagedAttention](https://arxiv.org/abs/2309.06180) | 官方代码 / 论文 | 生成与分页 KV 管理 |
| [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) / [调优](https://docs.vllm.ai/en/latest/configuration/optimization/) | 官方文档 | prefix caching 与 serving 取舍 |
| [SGLang 仓库](https://github.com/sgl-project/sglang) / [RadixAttention 博客，2024-01-17](https://www.lmsys.org/blog/2024-01-17-sglang/) | 官方代码 / 作者博客 | 前缀复用与原图；历史 benchmark 不作今天排名 |
| [SGLang Deterministic，2025-09-22](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) | 作者博客 | 确定性、数值差异和 RL 复现 |
| [SGLang PD](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) | 官方文档 | prefill / decode 分离 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 原论文 | 分块、IO-awareness 与 online softmax |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) / [论文](https://arxiv.org/abs/2405.11143) | 官方代码 / 论文 | Ray、DeepSpeed、生成服务与角色布局 |
| [AReaL](https://github.com/areal-project/AReaL) / [论文](https://arxiv.org/abs/2505.24298) | 官方代码 / 论文 | 异步 RL 系统与算法配合 |
| [TRL](https://huggingface.co/docs/trl/index) / [GRPO](https://huggingface.co/docs/trl/grpo_trainer) | 官方文档 | SFT / DPO / GRPO trainer 与生成集成 |
| [DPO](https://arxiv.org/abs/2305.18290) / [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 原论文 | 离线偏好目标、组内优势背景 |
| [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) / [troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) | 官方文档 | 通信语义和 hang 诊断 |
| [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) / [DCP](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) | 官方文档 | 时间线、内存观测和分布式恢复 |
| [Ray Placement Groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) | 官方文档 | 逻辑资源预留与 worker 放置 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) / [llama.cpp](https://github.com/ggml-org/llama.cpp) | 官方代码 | 推理工具的场景边界 |

### 面试线索与使用边界

[LLM / RL Training Infra 题库](https://github.com/XFWang522/llm-rl-infra-interview) 用于核对训练显存、并行通信、RL 编排和恢复等主题；[AIInfraGuide](https://github.com/caomaolufei/AIInfraGuide) 用于补充训练 / 推理 / 算子方向。两者属于社区整理。本文没有逐条核实其底层面经、公司归属或出现频率，因此统一保留“公开题库主题”的低强度证据标记；具体技术事实回到官方资料核对。

### 更新记录

- **2026-10-03：** 将原 Infra 8 篇短笔记移入仓库归档，合并为本篇总览；加入框架分层、三张来源图、资源估算、slime / verl 对比、60 道练习和实验路线。原 URL 保留跳转，其他专题的关联入口改为本篇。
- 后续更新须区分“发现来源变化”与“正文完成核对”。自动检测只产生候选变更记录，不能把未核对的上游变化直接写成知识结论。整个整理流程见仓库 `docs/topic-workflow.md`。

### 可下载的整理资料

- [完整来源与固定版本清单](./research/framework-sources.json)
- [最新来源观察状态](./research/training-inference-frameworks-observations.json)
- [待核对变化记录](./research/training-inference-frameworks-pending.md)
- [专题整理工作流](./docs/topic-workflow.md)
- [下一篇专题模板](./templates/topic.md)
