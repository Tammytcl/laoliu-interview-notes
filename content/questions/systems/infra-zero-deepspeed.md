---
id: infra-zero-deepspeed
title: "07 · ZeRO究竟省了什么？从三阶段、通信到offload与checkpoint"
category: systems
difficulty: 进阶
tags: ["P0", "Infra基础", "ZeRO", "DeepSpeed", "Offload", "显存"]
updated: 2026-10-05
summary: "逐对象推导ZeRO-1/2/3的状态所有权、参数生命周期和通信，解释bucket/预取/持久参数、CPU/NVMe offload、配置与保存恢复的常见问题。"
draft: false
---

## 1. 主问题：同一模型为什么每张卡都存了重复状态？

本篇在 [多卡并行总览](#q=infra-distributed-parallelism)基础上深入ZeRO。先看 [训练状态账本](#q=infra-training-step)，再看 [权重与分片格式](#q=infra-checkpoint-formats)。目标是自己推导容量、解释一次更新的数据流、读懂配置，并知道容量省下来后付出了什么。

ZeRO是Zero Redundancy Optimizer：在数据并行副本组里减少模型状态的重复存储。它是机制，不是独立于DeepSpeed/FSDP等实现的“万能launch工具”；不同实现、精度与配置决定实际生命周期。核对日2026-10-05，本文配置用于讲解，未运行DeepSpeed多GPU训练。[ZeRO v3论文](https://arxiv.org/abs/1910.02054v3)。

## 2. 从DDP到状态所有权：每个rank仍能处理不同数据

普通同步DDP里，每rank存完整参数、梯度、optimizer统计，处理不同batch；聚合梯度后各自得到一致参数。ZeRO保留这种数据并行目标，但把部分状态的**持久所有权**分给不同rank：某rank只负责某段状态，计算需要时通过通信得到所需量。

![本文示意：DDP与ZeRO各阶段的参数、梯度、优化器状态所有权](./assets/infra/infra-zero-deepspeed/zero-stages.svg)

**图解：** 三列是参数P、梯度G、optimizer O；完整色块表示每rank保留完整副本，窄分块表示各rank只持有一部分。stage1只分O，stage2再分G，stage3再分P。图画的是主要持久状态，不代表计算时永远没有完整参数，也没有画激活、workspace或offload位置。

### 一个4-rank例子

把全模型状态按逻辑参数范围切成A/B/C/D四段。所有rank有不同训练数据，但rank0负责A的optimizer，rank1负责B，依此类推。参数段的所有权不是“这个rank的样本只经过A层”；它的样本仍需要经过完整网络。这个区别解释了为何ZeRO仍需要聚合或同步，而不是四张卡各自只算四分之一模型。

Stage3可按flat groups/模块组织，真实切点不一定与层边界重合；状态还可能有padding。概念A/B/C/D不能代替checkpoint里的shape与分区metadata。

## 3. 三阶段怎样执行一次训练更新？

### ZeRO-1：只分optimizer

各rank前后向仍可使用完整模型，梯度按数据并行目标聚合；每rank只更新自己负责的参数段及optimizer状态，再将已更新参数段传播给组内其他rank，以重建下一步一致的完整参数。具体实现可把归约/分片更新组织成更高效流水线。

省下的是大份Adam状态和相关master权重；完整工作参数、梯度仍需考虑。若OOM来自长序列激活，stage1未必能明显帮助。若optimizer占了大头，stage1可在不引入逐层参数聚合的情况下减轻常驻压力。

### ZeRO-2：再分梯度

梯度在产生后规约到拥有该段的rank，通常采用reduce-scatter类组织，减少持续保留完整梯度。各rank对自己的状态和梯度执行更新，再all-gather/同步参数段，让下一步保持完整工作参数。

“梯度分片”不等于每个反向算子在生成瞬间就不需要任何临时梯度；梯度bucket、准备规约的量和累积实现会影响峰值。多个microbatch累积时，还要明确哪些状态跨块保留、何时通信和step。

### ZeRO-3：再分参数

每rank持久保留参数分区；计算当前模块前，聚合它需要的完整参数；计算后通常重新分片；反向若参数已释放则再次聚合，再将梯度规约到分区所有者。每rank只更新自己的分区。[DeepSpeed ZeRO教程](https://www.deepspeed.ai/tutorials/zero/)。

这让单卡常驻模型状态随DP分片数下降，但当前执行模块仍有必要的参数/激活。若一个巨大矩阵的完整计算需求本身放不下，不能指望stage3把当前算子自动改成TP矩阵乘。还需更细粒度组织、memory-centric tiling、TP等适合该结构的路径，验证模型支持。

## 4. 状态容量推导：16P/D只是特定布局的一项

定义P参数量、D为ZeRO分片组大小；参数2字节、梯度2字节、FP32 master4字节、Adam统计8字节。把master与统计归入optimizer相关12字节，得到：

$$
M_{DDP}=16P,\quad M_{Z1}=(4+12/D)P,\quad M_{Z2}=(2+14/D)P,\quad M_{Z3}=16P/D
$$

| P=7B、D=8 | 每rank常驻模型状态 | 主要仍完整保存 |
|---|---:|---|
| DDP | 112GB≈104.31GiB | P/G/O |
| stage1 | 38.5GB≈35.86GiB | 工作参数与梯度 |
| stage2 | 26.25GB≈24.45GiB | 工作参数 |
| stage3 | 14GB≈13.04GiB | 依实际持久参数策略有额外量 |

这里是十进制GB与二进制GiB的转换，**未包括激活、通信bucket、聚合层、预取与workspace**。使用FP32梯度、低位optimizer、没有master或混合冻结参数时，都要重新列账本。

1T在该布局下模型状态合计16TB；D=256时，理想stage3每rank状态62.5GB。看起来接近80GB卡容量，并不能因此保证可运行：剩余空间要容纳当前聚合、激活和buffer。磁盘checkpoint往往不含梯度，故也不能直接写“1T checkpoint就是16TB”。见 [1T磁盘与推理预算](#q=infra-checkpoint-formats)。

### 为什么step0能跑，step1才OOM？

optimizer统计可能首个更新才分配；offload/预取也可能在特定阶段初始化。应分别记录模型初始化、第一轮forward、backward、optimizer.step、保存与rollout切换的峰值。用nvidia-smi的一次采样或只看初始化值不能证明预算足够。

## 5. 通信代价：更少副本不是更少所有流量

沿用**参数与梯度都以2字节通信**、ring类分片/聚合、每步只按所列阶段通信的教学模型，设S=2P字节：

- DDP一次梯度all-reduce，每rank发送约 $2(D-1)S/D$。
- stage2的梯度reduce-scatter加更新参数all-gather，各约 $(D-1)S/D$，加总量与上述DDP近似相同。
- stage3若forward和backward各聚合一次参数，再规约一次梯度，总发送约 $3(D-1)S/D$，约为上述基线1.5倍。

这个推导解释论文中常见的容量/通信取舍，不是测得NCCL时长。梯度dtype、参数是否在forward后保留、累积次数、重算、bucket、网络拓扑与实际collective算法都会改变流量和暴露时间。[通信基础与ring模型](#q=infra-gpu-performance)。

### Q：为什么stage3有时反而更快？

容量释放可能允许更大microbatch、更少offload或更好的kernel效率，通信也可能与计算重叠。反过来，小batch、慢跨机网络、过碎模块和高频参数聚合可能让它更慢。应在相同global batch、数据长度、目标精度和模型质量下比较，不把“卡数增加”或“训练batch变了”算作单一机制收益。

## 6. bucket、预取与持久参数：从配置名推到内存对象

| 配置/机制 | 主要对象 | 取舍 |
|---|---|---|
| `reduce_bucket_size` | 规约梯度的一批元素 | 大bucket减少启动次数，但缓冲更大、ready更晚 |
| `allgather_bucket_size` | stage1/2等路径同步参数的分组 | 通信粒度与内存峰值 |
| `stage3_prefetch_bucket_size` | 预取未来参数 | 可能覆盖等待，也增加在途参数 |
| `stage3_param_persistence_threshold` | 小参数的持久完整副本 | 减少小消息，牺牲一点重复容量 |
| `stage3_max_live_parameters` | stage3活跃参数管理 | 限制同时活跃量，不能小于实际执行需求就保证成功 |
| `overlap_comm` | 通信与计算重叠 | 需依赖ready和资源余量，可能额外占缓冲 |
| `contiguous_gradients` | 梯度连续缓冲组织 | 减少碎片/改善布局，不是“梯度不占内存” |

这些大小通常按**元素个数**，不是字节。5千万元素的BF16 buffer约100MB，FP32约200MB；还可能有多个在途buffer。调bucket前先看dtype和峰值，而不是把5e7理解成50MB。[DeepSpeed配置schema](https://deepspeed.readthedocs.io/en/latest/zero3.html)、[配置说明](https://www.deepspeed.ai/docs/config-json/)。

为什么持久小参数有意义？如果一个几百元素的bias也每层都单独聚合，固定通信延迟可能远大于它的计算。保留小副本可以减少小消息，但很多模块/很大threshold会累计明显容量。默认值与别名随版本检查，不建议一次把所有阈值调大。

## 7. Offload：从GPU换到CPU/NVMe，状态没有消失

**分片**回答“哪些rank负责哪部分状态”；**offload**回答“状态放在哪一层存储”。stage2可以把optimizer等卸到CPU，stage3还可以卸载参数。ZeRO-Infinity把GPU、CPU、NVMe作为不同容量/带宽层，组织取回与计算。[ZeRO-Offload教程](https://www.deepspeed.ai/tutorials/zero-offload/)、[ZeRO-Infinity v1](https://arxiv.org/abs/2104.07857v1)。

| 放置 | 可缓解 | 新瓶颈 |
|---|---|---|
| GPU状态分片 | 每GPU常驻模型状态 | GPU间collective |
| CPU optimizer offload | GPU optimizer容量/部分计算 | 主机RAM、CPU更新、PCIe/NUMA |
| CPU parameter offload | GPU参数常驻容量 | forward/backward前取回、传输窗口 |
| NVMe offload | CPU RAM容量压力 | 存储IO、buffer staging、排队与带宽 |

主机pinned memory有利于传输，但会占主机资源；每rank的buffer乘以同机进程数，会放大RAM需求。使用优化CPU optimizer可能帮助更新吞吐，却不能消除CPU/GPU搬运。

**教学下界：** 如果一段权重100GB必须通过有效25GB/s链路取回，纯搬运至少4秒；若1T BF16的2TB全部需要逐步经过同等链路，累计纯传输至少80秒。缓存/分层/多链路和预取可改变暴露时间，这不是offload系统实测。decode反复访问权重时，常驻容量与响应时延尤其需要共同评估。

activation checkpointing是少存并重算激活；parameter offload是搬参数；optimizer offload是搬状态/计算。三者可以组合但不能互相替代，OOM定位必须先知道缺的是哪项。

## 8. DeepSpeed配置与训练循环：谁负责累积与step？

下面是**教学配置**，用来定位对象，不是任何模型/硬件的推荐最优值：

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "train_batch_size": 64,
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "overlap_comm": true,
    "reduce_bucket_size": 50000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 100000
  }
}
```

batch64假设D=8个DP副本，$2\times4\times8=64$；若8卡实际还含TP/PP，不能仍按D=8。BF16需目标设备与kernel支持。若接入上层trainer，它可能已生成或覆盖某些配置，先打印最终有效配置。

```python
# 示意：模型、数据、optimizer已准备；未运行GPU任务
engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model, optimizer=optimizer, config="ds_config.json"
)
for batch in loader:
    loss = model_loss(engine, batch)
    engine.backward(loss)
    engine.step()   # 每个microbatch调用；engine按累积边界决定实际更新
```

不要在外部又除一次loss、又自行累积/step，却假定engine不知道。归一化要按实际loss reduction和框架行为确认；动态token长度更要检查全局分母。grad clipping也应走匹配的引擎/框架路径，不能只对某rank不完整分片计算“全局范数”。[DeepSpeed入门](https://www.deepspeed.ai/getting-started/)。

CPU offload示意是在stage配置中增加`offload_optimizer: {device: "cpu", pin_memory: true}`；参数offload只适用于相应stage3路径。NVMe还涉及路径、异步IO与buffer等，不能只把cpu改成nvme就假定环境完成联调。

## 9. 保存与恢复：checkpoint分区不能当部署权重

`engine.save_checkpoint`是保存训练状态的一条路径，要求所有参与进程按协议调用、tag一致；不是只让rank0调用就完成。其他rank不进入某些collective时可能挂起。ZeRO-3保存后直接在同一个已partitioned模型上加载也有特定限制，官方说明要求重新初始化相应engine/模型状态再走恢复路径。[固定engine保存说明](https://github.com/deepspeedai/DeepSpeed/blob/a77aeb676507beb9fe0604bc31c2bf075daf7c46/deepspeed/runtime/engine.py)。

导出16-bit部署权重是一条不同路径，stage3可能需启用聚合权重的保存选项，或者从FP32 optimizer分区离线恢复逻辑权重。参数聚合和导出会引入CPU/GPU/磁盘峰值，1T更需预算。[checkpoint转换文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。

ZeRO改变每rank保存位置，不必然降低全局纯权重字节。恢复卡数、TP/PP布局或框架变化，应查该版本的支持路径；不能仅凭分片文件存在就保证可在任意拓扑恢复。格式与1T例子见 [权重与checkpoint](#q=infra-checkpoint-formats)。

## 10. 排障与验收：一次只改变一个对象

| 现象 | 优先确认 | 有针对性的动作 |
|---|---|---|
| stage1仍OOM | 是否激活/完整参数才是大项 | 缩microbatch、重算或评估更高stage |
| stage3 forward聚合OOM | 当前模块、预取、持久参数与bucket | 调低在途量、细化组织；必要时TP/tiling |
| 首个step OOM | optimizer/master是否此时创建 | 查实际dtype与状态放置 |
| offload后GPU省了但step极慢 | CPU更新/PCIe/NVMe时间线 | 核对NUMA、CPU optimizer、预取和buffer |
| 保存卡住 | 是否所有rank调用、tag一致 | 先找最早异常，按框架协议保存 |
| 数值/收敛变化 | batch、loss分母、精度、clip与累积 | 固定小任务，比较梯度/更新和质量 |

基线应是小模型、固定数据与global batch，记录每rank阶段峰值、实际通信、CPU/RAM/IO、有效token/s与恢复后一步误差。本文只提供验收方案，CPU容量验算不是stage3 GPU训练的替代证据。

## 11. 面试问答与来源

**Q01 [基础] ZeRO和DP矛盾吗？** 不矛盾，ZeRO减少DP重复状态，rank仍处理不同数据。追问：stage3是否让每rank只跑一部分层？不是PP，执行时仍需获得完整计算所需参数。

**Q02 [推导] 三阶段怎么推到容量公式？** 分别把12字节optimizer、再2字节梯度、再2字节参数除D。追问：哪项不是这个布局？实际dtype/master不同就重列账本。

**Q03 [机制] stage2为什么不要求每层都聚合参数？** 工作参数通常完整，分的是梯度与optimizer；更新后同步参数段。追问：何时瞬时仍有梯度buffer？产生/规约/累积时，峰值要量。

**Q04 [机制] stage3为什么需要参数all-gather？** 持久只存分区，当前算子需要其他分区。追问：开stage3就能解决单个巨大矩阵容量吗？未必，考虑TP/tiling与执行峰值。

**Q05 [推导] stage3通信约1.5倍成立在哪些条件？** 相同2字节参数/梯度、每步两次聚合一次规约的简化模型。追问：重算/FP32梯度会怎样？改变次数/字节，不能继续套倍数。

**Q06 [使用] bucket50000000是多少显存？** 先看元素dtype；BF16约100MB、FP32约200MB，多个buffer另加。追问：为什么调大不一定快？ready延后、峰值和资源竞争。

**Q07 [设计] ZeRO offload是不是磁盘训练无代价？** 状态搬到另一存储层，容量省下但付IO/CPU/PCIe；用100GB/25GB/s下界说明。追问：哪些传输可被覆盖？已有独立计算窗口且缓存/预取满足依赖时。

**Q08 [排障] stage3保存只让rank0调用为何挂？** 保存包含分布式状态和collective协议。追问：所有rank都调用就一定可恢复任意卡数吗？还需支持该布局转换与optimizer映射。

| 来源 | 类型 | 阅读用途 |
|---|---|---|
| [ZeRO v3](https://arxiv.org/abs/1910.02054v3) / [DeepSpeed教程](https://www.deepspeed.ai/tutorials/zero/) | 原论文/官方 | 状态分片与数据流 |
| [schema](https://deepspeed.readthedocs.io/en/latest/zero3.html) / [配置](https://www.deepspeed.ai/docs/config-json/) | 官方文档 | 元素单位、预取、持久参数与offload |
| [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/) / [ZeRO-Infinity v1](https://arxiv.org/abs/2104.07857v1) | 官方教程/原论文 | CPU/NVMe层级 |
| [入门](https://www.deepspeed.ai/getting-started/) | 官方教程 | engine生命周期 |
| [固定engine](https://github.com/deepspeedai/DeepSpeed/blob/a77aeb676507beb9fe0604bc31c2bf075daf7c46/deepspeed/runtime/engine.py) / [checkpoint文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) | 固定官方代码/文档 | 保存、恢复与导出 |

[来源清单](./research/infra-zero-deepspeed-sources.json)。复习时先画状态所有权与一次step数据流，再填实际模型/卡数的容量，最后解释每个配置控制哪个对象；不先背一套巨大配置模板。
