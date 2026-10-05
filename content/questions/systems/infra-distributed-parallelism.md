---
id: infra-distributed-parallelism
title: "03 · 多卡到底怎么切？理解 DDP、ZeRO/FSDP、TP、PP 与长上下文并行"
category: systems
difficulty: 进阶
tags: ["P0", "Infra基础", "分布式训练", "FSDP", "ZeRO", "Megatron"]
updated: 2026-10-05
summary: "按切分对象、通信位置与更新语义比较并行机制，用显存账本、16卡布局与pipeline算例解释选型，连接到verl/slime的训练后端。"
draft: false
---

## 1. 主问题与读法：先问切什么，再问用哪套框架

“多卡训练”至少包含三种诉求：模型或状态单卡放不下；一个样本的长上下文放不下；希望单位时间处理更多数据。它们没有同一个默认答案。本篇先读 [模型状态与激活账本](#q=infra-training-step)、[通信与性能下界](#q=infra-gpu-performance)，最后回到 [框架选型](#q=training-inference-frameworks)。

一套并行方案用四问来描述：**每卡存什么、一次算什么、何时通信、怎样组成一次更新**。名称相同的机制在不同版本里也可能有不同组织；下文以普通dense Transformer为推导基线，MoE单独说明。核对日2026-10-05，布局与数字为教学推导，未执行多GPU训练。

![本文示意：DP切数据，TP切矩阵，PP切层，状态分片切存储](./assets/infra/infra-distributed-parallelism/parallel-layout.svg)

**图解：** 四个小图分别表示：DP的完整模型副本处理不同数据；TP的多个rank合算同一层；PP把不同层放在不同stage；状态分片让副本组各存部分状态、需要时通信。最后一项改变存储生命周期，并不天然改变训练目标。这是概念图，实际组合会同时应用多个维度。

## 2. DDP：复制完整模型、拆分训练数据

### Q：DDP的每个rank各自训练，会不会学成不同模型？

标准同步DDP从相同参数开始，每个rank处理不同数据；backward时通过collective聚合梯度；各rank在同样梯度、optimizer状态和配置下执行相同更新，因此参数保持一致。DDP通常按bucket把梯度同步与后续反向重叠。数据分发不是DDP自动全包的任务，仍需检查sampler是否分片、shuffle和epoch是否一致。[PyTorch DDP API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

例子：两个rank各用4个样本，本地loss为各自均值；若DDP平均两边梯度，等价于8个样本整体均值。若两边有效token数不同，各自token平均再平均不再等于全局token平均，见第一篇的归一化例子。

DDP每卡仍持有完整参数、梯度和optimizer。把同一模型放到8卡不代表每卡容量除8；它主要扩展数据吞吐。每个rank中的batch变小，会让激活小一些，但模型状态副本并未变小。

### Q：为什么rank挂了，有时别的rank报NCCL timeout？

collective需要组内成员调用相容操作。一个rank先OOM或Python异常退出，其他rank可能只是等不到它。看到timeout先找**所有rank中最早的异常**，再查设备映射、通信序列和网络；后面的超时未必是网络根因。对不等量输入，还要看是否正确使用join等机制，不能随意让某些rank少执行backward。

## 3. ZeRO与FSDP：把模型状态分开存

### Q：ZeRO的三个stage怎么推导，而不是怎么背？

在全参训练模型状态里，参数、梯度、optimizer统计都是大数组，而且DP副本会重复存储。ZeRO依次消除这些副本：stage1分optimizer，stage2再分梯度，stage3再分参数。[ZeRO 原论文](https://arxiv.org/abs/1910.02054v3)、[官方 ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)。

沿用第一篇的特定16字节布局：参数2、梯度2、optimizer相关12（含FP32 master4和两份统计8），DP分片数D。忽略聚合峰值、激活与buffer：

| 方案 | 每rank模型状态近似，字节/参数 | 7B、D=8的十进制GB |
|---|---|---:|
| DDP | 16 | 112 |
| ZeRO-1 | $4+12/D$ | 38.5 |
| ZeRO-2 | $2+14/D$ | 26.25 |
| ZeRO-3 | $16/D$ | 14 |

如果实际master管理方式、梯度dtype或optimizer不同，必须重算。最右列是**分片常驻状态**而非显存峰值，不能据此保证14GB卡可训练7B。

### Q：参数都分走了，一层的forward怎么计算？

典型全分片执行按模块聚合当前计算所需参数，计算后重新分片；反向需要参数时也聚合，再reduce-scatter梯度。一次更新的optimizer可只更新本rank的参数分片。于是容量省下来了，代价是参数通信与生命周期管理。

FSDP2用`fully_shard`组织参数组和DTensor分片，自动注册计算前后的hook。分层包裹比只在最外层包整模更有机会限制一次聚合的大小，也更容易预取下一层。但过细粒度增加小通信与管理成本；预取更多层提高重叠机会，同时增加在途完整参数。[FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

FSDP和ZeRO-3容量目标接近，不表示API、参数表示、预取、梯度累积和checkpoint可互换。FSDP1的`no_sync`、forward/backward prefetch设置，也不能不看版本就套在FSDP2上。第一次接入应从后端官方示例和上层recipe开始，再做单变量优化。

### Q：分片是不是一定比DDP快？

不是。放得下的模型可能用DDP具有较少参数聚合开销；分片也可能释放容量，让更大的microbatch提高计算效率。应固定global batch、数据和精度，比峰值、有效token/s与质量。跨机链路慢时，全分片的参数流量可能暴露得更明显，hybrid sharding可在某些拓扑中改变跨机通信结构，但不是无条件优胜。

## 4. TP：多个rank共同计算同一层

### Q：切矩阵如何保持数学结果？

以MLP两层为例，输入 $X[n,H]$，第一层 $W_1[H,4H]$，第二层 $W_2[4H,H]$。把 $W_1$ 沿输出列切为T份：每rank计算 $XW_{1,j}$，得到 $[n,4H/T]$；逐元素激活可在各自分片上做。$W_2$沿输入行切，每rank用对应输入计算一份 $[n,H]$ 部分结果；这些结果求和得到完整第二层输出。

这种成对column/row切法让第一层后不必立即收集整个中间激活，第二层后需要归约。若后续算子可以继续消费分片，还可调整通信位置。这是张量并行的基础，不是把两层训练成不同模型。[Megatron-LM 大规模训练论文](https://arxiv.org/abs/2104.04473v5)。

TP可以降低单rank参数与某些计算量，但每层附近都有通信；因此常优先利用节点内高速互联。TP增大后每个本地矩阵更小，效率可能下降，不能假定算力时间严格除T。维度/heads是否可整除、KV heads是否复制、词表是否分片也都需要核对具体实现。

### Q：TP=4的4卡能同时放4个独立样本吗？

它们共同服务一个TP副本，其数据是同一批样本的不同计算部分，并非4个DP副本。可在这个副本上用batch=4，但那是另一个维度。global batch公式应乘DP数，而不是乘TP数。

训练TP与推理TP也不一定相同。训练含反向、状态和激活；推理含KV与小batchdecode。发布权重时要检查切分布局和重分片，不能把训练shard原样塞给任意推理rank。

## 5. PP：把层放到不同stage，用microbatch填流水线

### Q：一个样本必须经过全部层，切层为什么还能提速？

单个microbatch依次经过stage0→stage1→…，相邻stage传激活，反向再传梯度。不同microbatch可同时处于不同stage，因此吞吐提升来自流水线。首批的填充与末批的排空产生bubble，层间负载不均还会让较快stage等待。

对P个等耗时stage、m个microbatch的简化fill-drain模型，bubble占比约：

$$
f_{bubble}\approx\frac{P-1}{m+P-1}
$$

P=4、m=4约42.9%，m=16约15.8%。这是均衡阶段与理想调度的演示；实际前后向耗时、1F1B/interleaving、虚拟stage、通信和显存约束都会改变它。增加m也可能通过减小microbatch让kernel更低效，因此不是“越多越好”。

GPipe式统一更新与PipeDream式流水执行存在不同权重版本语义。1F1B只是前后向安排，不能仅凭名字判断是否严格保持相同参数。看参数何时更新、在途microbatch使用哪个版本，再谈等价性。[GPipe](https://arxiv.org/abs/1811.06965v5)、[PipeDream](https://arxiv.org/abs/1806.03377v1)。

PP在容量不足、层数多时很有价值，但首尾stage可能承担embedding/loss，MoE层耗时也可能不同；按层数均分不一定按时间均分。profiling后再调整stage布局。

## 6. SP、CP、EP：不要把所有并行度相乘

**Sequence Parallel（Megatron常见语义）**通常与TP配合，把一些原本在TP rank重复的激活沿序列切开，特别是layernorm/dropout附近，以减少副本。**Context Parallel**则让一条长序列的不同位置分布到rank上，并组织attention所需的跨分片K/V访问；每个query仍必须按mask得到它应该看到的上下文。CP并不会天然把模型权重分片。[Megatron 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)。

各项目也可能把Ulysses等长上下文方案统称sequence parallel。verl的`ulysses_sequence_parallel_size`应按其具体实现理解，不能仅凭“SP”二字推断它与Megatron SP完全相同。长上下文OOM首先检查激活/attention/KV是哪项，再评估切序列，而不只提高DP。

**Expert Parallel**将MoE不同expert放到不同rank；路由token发给对应expert，执行后再combine。需要关注dispatch/combine通信、每expert收到的token不均与容量策略。激活参数少描述每token执行量，不代表全部expert权重都不占存储。

对于一种常见dense布局，可写 $W=DP\times TP\times PP\times CP$，SP往往复用TP组。EP在MoE里可在数据维度内再划组或采用parallel folding；具体关系要看框架，不能把EP和SP不加说明地都乘进去。Megatron的rank group API可以帮助确认真实组，而不是根据启动参数猜。[Megatron parallel_state](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html)。

## 7. 一套16卡布局怎样推到框架配置？

**教学任务：** 两节点各8卡；dense模型；TP=2、PP=2、CP=1、DP=4。每个模型副本使用4卡，独立数据副本4个。如果每副本microbatch=2、累积4次，则一次更新32个样本。TP组优先放在节点内，PP可按拓扑放置；具体rank映射仍由实现决定。

按顺序作决定：

1. **确认目标与容量**：哪个状态/激活单卡放不下？是否可先用重算与合适microbatch？
2. **确定模型支持**：HF/PyTorch路径或Megatron模型映射是否成熟？不要先定并行度再发现结构不支持。
3. **确定通信结构**：TP每层通信、FSDP聚合、PP跨stage、CP上下文流量分别放在哪条链路。
4. **确定更新语义**：DP数、global batch、token归一化与累积边界是否一致。
5. **确定保存与推理交接**：sharded checkpoint能否恢复？训练与rollout布局不同如何重分片？

| 实际需求 | 优先评估方向 | 必须验收 |
|---|---|---|
| 模型能放下，主要要更多吞吐 | DDP或轻量状态分片 | 数据确实分开，扩展效率合理 |
| 模型状态放不下 | ZeRO/FSDP，必要时结合TP/PP | 聚合峰值与每rank状态dtype |
| 单层太大或需模型并行kernel | TP、模型适配 | 本地shape与每层通信 |
| 模型深、需切层容量 | PP | bubble、stage负载、版本语义 |
| 单条上下文太长 | 重算、合适attention实现、CP等 | 完整上下文与mask数值一致 |
| MoE expert数量/存储大 | EP与其他维度组合 | token负载、all-to-all、全部权重账本 |

这是约束下的评估顺序，不是框架速度排名。在verl中按recipe选择FSDP/Megatron等后端，再核实配置分组；slime主线围绕Megatron训练。下面只给阅读入口，不提供未经安装验证的万能launch命令。

- [verl固定代码树](https://github.com/verl-project/verl/tree/8718ca30a3f002f93b7c4fd99b9b2506718681bc)：追训练worker、batch拆分与后端初始化。
- [slime固定代码树](https://github.com/THUDM/slime/tree/8c17b676cb57af1d17ee4402e91e9209af84b60b)：追Megatron并行参数与权重同步。
- [Megatron固定代码树](https://github.com/NVIDIA/Megatron-LM/tree/ef8cd10b8e34a94c9ad7452e7851f58337f353a0)：追模型parallel groups与checkpoint布局。

这些是本次观察到的commit，不是论文实验版本。分布式恢复应先做短运行“保存→重启→继续”，检查模型、optimizer和数据游标，而不是直到长任务结束才测试恢复。[PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)。

## 8. 面试问答：切分机制、边界与追问

均为本文整理题，依据本篇论文/官方教程；没有将公开材料推导成公司真题。

**Q01 [基础] DDP会把模型除以卡数吗？** 不会，常规DDP每卡完整副本，切的是数据；举7B状态112GB仍各卡存在的例子。追问：怎么让optimizer分片？ZeRO-1等。

**Q02 [推导] ZeRO-3峰值能直接16P/D吗？** 这只是特定布局的常驻状态；加当前聚合层、激活、预取与workspace。追问：预取两层改了什么？增加在途完整参数，可能改善重叠。

**Q03 [机制] TP的column/row切法怎么减少通信？** 第一层输出各自分片，激活本地，第二层本地产生部分和后归约。追问：非逐元素跨分片操作怎么办？重新安排通信，不能直接独立做。

**Q04 [推导] 16卡TP2 PP2，DP是多少？** CP1、无其他占用的dense布局下DP4；micro2累积4得到global32。追问：EP能直接再乘吗？需看组定义，不能自动乘。

**Q05 [基础] SP和CP是不是同一个东西？** Megatron常见语义不同；SP处理一些TP复制激活，CP切完整上下文。追问：不同项目叫法一样怎么办？看张量布局与collective实现。

**Q06 [设计] PP=8总会比PP=4快吗？** 更细stage可省容量，但bubble、通信和阶段不均会增加。追问：怎么估第一版？等stage近似，再用trace测真实布局。

**Q07 [排障] NCCL timeout是不是网络坏？** 先看最早rank异常、调用次数/顺序与shape，再查网络。追问：某rank少一个batch怎么办？明确join或统一输入策略，不能让collective序列失配。

**Q08 [设计] FSDP换Megatron该比较什么？** 模型适配、并行需求、格式转换、数值与恢复，再比较同任务吞吐。追问：只比step时间够么？不同batch/更新语义需统一到同质量和GPU-hour。

## 9. 复习练习与来源

纸上画16个rank，标TP/PP/DP组；将本篇状态账本改成FP32梯度重算；画P4、m4/m16流水线。GPU练习待做：同一模型DDP与分片短运行，记录每rank峰值、有效token/s和恢复误差，再放大到跨机。

| 来源 | 类型 | 阅读用途 |
|---|---|---|
| [DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html) | 官方API，2.14 | 同步、输入与平均语义 |
| [ZeRO](https://arxiv.org/abs/1910.02054v3) / [教程](https://www.deepspeed.ai/tutorials/zero/) | 论文/官方 | 三阶段状态账本 |
| [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 官方教程 | 分组、聚合、预取 |
| [Megatron-LM](https://arxiv.org/abs/2104.04473v5) | 原论文 | TP与PP组合 |
| [GPipe](https://arxiv.org/abs/1811.06965v5) / [PipeDream](https://arxiv.org/abs/1806.03377v1) | 原论文 | pipeline调度与版本语义 |
| [Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) / [parallel_state](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html) | 官方教程/API | CP、SP、EP与组关系 |
| [Distributed Checkpoint](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) | 官方API，2.14 | 分片恢复 |

[本篇来源清单](./research/infra-distributed-parallelism-sources.json)还记录了三个固定代码入口。下一篇从推理角度重算容量和调度，解释为什么训练布局不能直接作为推理最优布局。

**数值练习下载：** [CPU验算脚本（仅Python标准库）](./assets/infra/infra-foundations-check.py) · [本次验算输出](./assets/infra/infra-foundations-check-results.json)。覆盖梯度有限差分、状态/KV字节、Roofline、TP与分块softmax；不包含GPU训练或测速。

进一步阅读：[ZeRO三阶段完整机制](#q=infra-zero-deepspeed) · [DeepSpeed/Lightning/FSDP怎样接入](#q=infra-training-backends) · [分片格式、恢复与1T容量](#q=infra-checkpoint-formats)。
