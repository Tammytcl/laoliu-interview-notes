---
id: "pt-memory-parallel"
title: "Q14 · 7B 后训练要多少显存，DDP、FSDP、ZeRO 分别省什么？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "系统与 Agent", "字节", "百度"]
updated: "2026-10-23"
summary: "用精度假设明确的显存账本回答卡数问题，再区分训练状态分片、激活与推理 KV。"
draft: false
---

## 60–90 秒回答

我不会只按参数乘两个字节估算训练。要分参数、梯度、优化器状态、可能的高精度 master copy，以及激活、通信/临时 buffer；RL 还包括 rollout、reference、critic、奖励或教师服务和 KV cache。DDP 通常复制模型状态、分摊数据；ZeRO 逐级分片优化器状态、梯度和参数；FSDP 也以参数等状态分片与按需聚合为核心，但具体策略需看实现。最后根据上下文、microbatch、并行方式和重算配置估峰值，并用实际测量校准。减少状态内存不保证减少激活或提高吞吐，通信也可能变成瓶颈。

## 原理拆解

### 先算静态状态，再算运行峰值

设可训练参数量为 $P$，按前述 BF16 参数/梯度与 FP32 master、Adam moments 的假设，每参数 16 字节。$D$ 卡数据并行时，简化的每卡模型状态为：

| 方案 | 每卡状态字节的简化表达 |
| --- | --- |
| DDP | $16P$ |
| ZeRO-1 | $4P+12P/D$ |
| ZeRO-2 | $2P+14P/D$ |
| ZeRO-3 | $16P/D$ |

表中没有激活、通信桶、参数临时 all-gather、碎片与保存峰值，精度假设改变则所有系数重算。ZeRO-3/FSDP 在模块计算前聚合参数，粒度越细可能省峰值却增加通信；prefetch 改善重叠也可能抬高同时驻留量，不能只用表里最后一行决定是否能跑。[ZeRO 原论文](https://arxiv.org/html/1910.02054v3)

### 长上下文为什么会改变显存主项

训练激活与 batch、序列长度、层数、隐藏维度及是否重算有关；朴素 attention 显式矩阵有平方长度项，内存高效 attention 可以改变其存储形态，但 MLP 激活和输出 logits 仍需考虑。仅一个 $B\times T\times|V|$ logits 张量就可能很大，teacher 全词表蒸馏还会加重这一项，需要分块或融合实现。

Rollout 的 KV 账本可近似写成 $2LBTn_{kv}d_hb$ 字节，2 对应 K/V，$L$ 是层数、$n_{kv}$ 是 KV heads、$d_h$ 是 head dimension、$b$ 是单元素字节。GQA 的 KV heads 可能少于 query heads。这个式子不含分页浪费、前缀共享、并行分片和动态并发；它解释了为什么提升同时在途的长任务会吃掉生成服务显存。

### 并行策略解决的维度不同

DDP 分数据、通常复制状态；ZeRO/FSDP 在数据并行组内分片状态；TP 切单层张量计算；PP 切层但有流水线气泡；长上下文并行切序列或 attention 计算；MoE expert parallel 分专家并引入 token 派发。能否组合取决于框架和拓扑。多卡后仍 OOM，先问峰值在哪个角色、哪个阶段，再决定切哪一维。

后训练系统还需总账：actor 训练、rollout KV、teacher/reference/critic、CPU/offload 与权重同步分别占多少资源。评估优化既看单卡峰值，也看总 GPU-hour、通信与有效吞吐；将 actor 卡借给 rollout 可能提升利用率，但切换与同步开销也要进入端到端时间线。

## 白板与例子

**明确假设的教学下界例子：** 7B 个可训练参数，参数和梯度均为 BF16，每参数另有 FP32 master 权重以及 Adam 的两个 FP32 moment。模型状态约为：

$$
7\times10^9\times(2+2+4+4+4)=112\times10^9\ \mathrm{bytes},
$$

即 112 GB 十进制，约 104.3 GiB。若具体优化器没有独立 master copy 或采用其他梯度精度，这个数会变。它还没有包含激活、KV、通信缓存和 RL 的其他模型，更不能直接推出要几张卡。

在上述假设、$D$ 路数据并行的简化账本中，ZeRO-1 主要把每参数 12 字节优化器相关状态分摊；ZeRO-2 再分摊梯度；ZeRO-3 再分摊参数。运行时聚合和 buffer 仍会产生峰值。

## 追问怎么接

**Activation checkpointing 怎么提效？** 主要以重算换激活内存；不保证单步计算更快。若因内存释放能提升 batch 或避免 offload，端到端吞吐才可能更好。

**为什么 FSDP 后长上下文还 OOM？** 长度相关激活、attention 临时量、输出 logits 和推理 KV 不会因参数分片自动全部解决。先查峰值发生在哪一阶段。

**MoE 用 active 参数估显存行吗？** 算力和存储要分开；未激活专家也需要存放在某处。总参数、专家并行、通信、路由负载与实际驻留共同决定成本。

## 项目怎么联系

你的公开小规模 ESCD 指学生参数规模，并不意味着单卡训练；已发布示例有多机资源配比。面试时分别报模型大小、实际 GPU 分工、长度、精度和总预算，避免用“小模型实验”让人误解为低资源结论。

## 容易答错的点

- 把 GB 与 GiB 混用，或者只算 actor 忽略 rollout/teacher。
- 认为 ZeRO-3 除以卡数后的参数状态就是全部峰值显存。
- 把更多卡必然更快作为答案，忽略小 batch 和通信开销。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [百度算法岗公开题型汇总](https://www.nowcoder.com/discuss/927018114900922368) | 二次汇总，不按公司认证真题处理；页面公开内容与检索摘要；未读取付费部分；仅作题型依据 |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [ZeRO 原论文](https://arxiv.org/abs/1910.02054v3) | 优化器、梯度和参数分片的区别 |
| [verl 性能调优文档](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) | rollout、batch、packing 和训练/推理内存取舍；参数支持随版本核对 |
| [ESCD 公开实现与评测说明](https://github.com/Tammytcl/ESCD) | 事件补全、公开 baseline、数据和独立 PHYRD-40 评测的证据边界 |

**相关专题详解：** [01 · 训练一步在做什么？从梯度、精度到显存账本](#q=infra-training-step) · [07 · ZeRO究竟省了什么？从三阶段、通信到offload与checkpoint](#q=infra-zero-deepspeed) · [03 · 多卡到底怎么切？理解 DDP、ZeRO/FSDP、TP、PP 与长上下文并行](#q=infra-distributed-parallelism)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-rl-collapse) · [下一题](#q=pt-rollout-staleness)

[//]: # (interview-generated-end)
