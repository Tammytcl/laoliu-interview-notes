---
id: infra-training-step
title: "01 · 训练一步在做什么？从梯度、精度到显存账本"
category: systems
difficulty: 基础
tags: ["P0", "Infra基础", "训练", "显存", "梯度累积", "混合精度"]
updated: 2026-10-05
summary: "从一次 forward/backward/step 建立训练系统模型，推导模型状态和峰值显存，解释 batch、Adam、BF16、重算与 OOM 排查。"
draft: false
---

## 1. 主问题与先修：先看懂一张 GPU 上的训练

读完应能完成三件事：画出训练一步的生命周期；根据实际 dtype 估算全参训练的常驻状态；解释为什么“7B 权重能放进显存”不等于“7B 能训练”。先修只需张量形状、链式法则和平均数。next-token 标签、causal mask、loss mask 见 [自回归训练基础](#q=causal-lm-training)，本文从它们进入训练循环的地方接着讲。

五篇的顺序是：**本篇 → [GPU 性能与通信](#q=infra-gpu-performance) → [分布式并行](#q=infra-distributed-parallelism) → [推理引擎](#q=infra-inference-engine) → [RL 训推流水线](#q=infra-rl-pipeline)**。[verl/slime 框架选型](#q=training-inference-frameworks)是总入口，那里讲用谁，这里讲配置为什么会产生这些行为。

调研核对日为 2026-10-05。本文的数字是按假设手算；附带 CPU 练习有独立验算记录，GPU 片段用于教学，未运行大模型训练。

## 2. 训练一步：forward、backward、optimizer step 分别改变什么？

### Q：训练究竟在更新什么？

模型参数 $\theta$ 是被学习的权重；训练数据给出输入与目标；loss 将预测差异变成标量；autograd 按计算图求导。最简单的 SGD 是 $\theta_{t+1}=\theta_t-\eta g_t$，其中 $g_t=\nabla_\theta L$、$\eta$ 为学习率。梯度说明“参数朝哪个方向改变会影响目标”，不是模型输出，也不是参数本身。

线性层 $Y=XW$ 中，若 $X$ 形状为 $[n,d_{in}]$、$W$ 为 $[d_{in},d_{out}]$，前向输出为 $[n,d_{out}]$。反向既要计算 $\nabla_W L=X^\top\nabla_Y L$，也可能需要向前一层传递 $\nabla_X L=\nabla_Y L W^\top$。这解释了为什么前向只见到权重和输入，反向却需要保存输入等中间量。[PyTorch Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)。

![本文示意：一个 optimizer step 内数据、激活、梯度和参数的生命周期](./assets/infra/infra-training-step/training-lifecycle.svg)

**图解：** 上排从数据进入 forward，再由 loss 触发 backward，最后 optimizer 更新参数。下排表示对象的存活范围：参数贯穿整步；激活主要从 forward 保留到对应 backward；梯度在 backward 写入，并跨 microbatch 累积；Adam 状态跨 step 保留。箭头表示依赖，不代表阶段耗时比例，也不代表激活必须一次性全部保存。

| 调用 | 主要产物 | 是否通常改变参数 |
|---|---|---|
| `optimizer.zero_grad()` | 清理上一更新的梯度 | 否 |
| `model(batch)` | logits 与反向需要的中间量 | 否；但某些有状态模块可能更新 buffer |
| `loss.backward()` | 将梯度累加到 `.grad` | 否 |
| `optimizer.step()` | 使用梯度与状态更新参数 | 是 |
| 保存 checkpoint | 写出可恢复的训练状态 | 本身不是学习更新 |

`model.train()` 改变 dropout 等模块行为，不能代替 `backward()`；`model.eval()` 也不会自动关闭 autograd。评估时常同时使用 eval 和 no-grad/inference mode。训练 dropout 会让相同输入两次前向不同，这和随机采样导致的变化也要分开。

### Q：为什么 Adam 比 SGD 多占显存？

Adam 还维护梯度的一阶、二阶移动统计 $m,v$。以忽略权重衰减的版本为例：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

校正初始化偏差后，用 $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ 调整更新。两份状态的形状通常与对应参数相同，因而参数越多，状态越大。它们保留的是优化历史，不能在每一步随意清零；梯度通常可以在更新后清理。[Adam 原论文](https://arxiv.org/abs/1412.6980v9)。

“存了权重就可无损继续训练”由此不成立。恢复训练还要考虑 optimizer、学习率 scheduler、随机数状态、混合精度 scaler（若使用）与数据游标。只加载权重可能适合启动新训练，但不是恢复原来的优化轨迹。

## 3. Batch 与梯度累积：样本数、token 数、更新频率

### Q：microbatch、global batch、gradient accumulation 有什么区别？

microbatch 是一次 forward/backward 处理的数据；梯度累积是多个 microbatch 先求梯度，暂不 step；global batch 是一次参数更新聚合的数据。对于普通数据并行、每个副本拥有不同样本且各步等量的情况：

$$
B_{global}=B_{micro}\times A\times D
$$

$A$ 为累积次数，$D$ 为**数据并行副本数**。如每卡 microbatch=2，累积4次，8个 DP 副本，则 global batch=64。8卡若被组织成 TP=4、DP=2，则只有2个独立样本副本，同样配置 global batch=16。不能把总 GPU 数直接填成 $D$。

梯度累积不会把64条数据的激活同时保留，因此可降低单次激活峰值；但模型、optimizer 状态并没有变小，且小 microbatch 可能降低矩阵运算效率。先跑64个 microbatch再step和每个microbatch都step，也不会产生同一条优化轨迹：后一种早已改变参数。[PyTorch AMP 累积示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)。

### Q：把每个 microbatch 的平均 loss 除以累积次数就一定正确吗？

只有等权目标与分组方式匹配时才正确。设两段各有10和90个有效 token，平均 loss 分别为2与1。平均两个平均数得到1.5；整个更新按 token 平均得到 $(10\times2+90\times1)/100=1.1$。它们优化不同目标。

如果目标是整个更新的有效 token 平均，应该累加 loss sum，并使用整个更新的有效 token 总数归一化。分布式还要纳入 DDP 默认的梯度平均约定：若每个 rank 对本地 loss sum 除以全局 token 数，再由 DDP 平均，梯度会额外小 $D$ 倍；可以在该约定下将本地分子乘 $D$，或使用等价实现。分片后端和框架可能已处理这一点，不能盲目再乘一次。[DDP 官方 API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

这不是要求所有任务都做 token 平均。样本平均、序列平均、组平均各有意义；关键是数据切成不同 microbatch 后，是否保持选定目标不变。对 GRPO，prompt 数、回答数、训练序列数、有效 token 数更不能互换。

### 可运行的 CPU 自检：同一个目标，切成两块后梯度是否一致？

```python
import numpy as np
x = np.array([1., 2., 3., 4.])
y = np.array([2., 4., 6., 8.])
w = 0.5
# L = mean((w*x-y)^2)；3个样本和1个样本分组
whole = np.mean(2 * (w*x-y) * x)
parts = [slice(0, 3), slice(3, 4)]
weighted = sum(np.sum(2*(w*x[s]-y[s])*x[s])/len(x) for s in parts)
wrong = np.mean([np.mean(2*(w*x[s]-y[s])*x[s]) for s in parts])
print(whole, weighted, wrong)  # -22.5, -22.5, -31.0
```

这是本文构造的代数练习。改变分组后正确梯度不变，是验收归一化的好办法。不要用“loss 看着差不多”代替梯度比较。

## 4. 显存账本：从参数字节数到实际峰值

### Q：7B 全参训练为什么远不止14 GB？

设 $P=7\times10^9$ 个参数。采用一个**特定教学布局**：BF16参数2字节、BF16梯度2字节、FP32 master 参数4字节、Adam的两份FP32状态共8字节。则常驻模型状态是 $16P=112$ GB，约104.3 GiB。

| 对象 | 此布局每参数字节 | 7B 对应十进制 GB |
|---|---:|---:|
| 工作参数 | 2 | 14 |
| 梯度 | 2 | 14 |
| master 参数 | 4 | 28 |
| Adam 一阶+二阶 | 8 | 56 |
| 总计 | 16 | 112 |

这里 $1\,GB=10^9$ 字节，$1\,GiB=2^{30}$ 字节。16并非所有PyTorch配置的常数：FP32参数配autocast通常不另存一份完整BF16工作参数；梯度dtype可能随参数或后端变化；某些实现不保留master，或使用低位optimizer。必须检查实际张量而不是套广告数字。[HF GPU memory anatomy](https://huggingface.co/docs/transformers/en/model_memory_anatomy)、[ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)。

实际峰值还要加上：

$$
M_{peak}\approx M_{state}+M_{activation}+M_{workspace}+M_{communication}+M_{other}
$$

相加表达的是同一时刻活跃对象。各阶段各自的最大值不一定同时发生，严格账本需看生命周期。初始化、首个step创建optimizer状态、反向、参数all-gather或保存checkpoint，都可能成为峰值。

### Q：激活显存随什么增长？

一个形状 $[B,S,H]$ 的BF16张量有 $2BSH$ 字节。取 $B=2,S=4096,H=4096$，单个张量64 MiB。32层如果每层只保留这样一份张量，也已经2 GiB；真实层会保存多类输入与中间量，这只是“单张量×层数”的示例，不是完整模型估算。

朴素 attention 若显式存 $[B,h,S,S]$ 分数，序列长度翻倍会让这一项约四倍。采用 FlashAttention 后不能继续把整张 $S^2$ 矩阵当作常驻项，但长序列的attention算量与其他激活仍需考虑。推理KV缓存是另一种对象，见 [推理引擎](#q=infra-inference-engine)。

输出头同样值得检查。若完整 logits 是 $[2,4096,128000]$、dtype FP32，单张量约3.91 GiB。是否真的物化、分块计算、只求目标token概率，取决于实现；“参数很小”不能保证末层临时内存小。

### Q：gradient checkpointing、offload、LoRA 各省什么？

| 方法 | 主要改变 | 代价与边界 |
|---|---|---|
| activation checkpointing / 重算 | 少存部分反向中间量，需要时再算 | 多做计算；随机数与状态行为需保持语义 |
| CPU offload | 部分状态或激活暂放主机内存 | 主机容量、PCIe传输、同步与流水线成本 |
| 状态分片 | 各rank保存部分参数/梯度/optimizer | 计算前聚合、同步、分片恢复成本 |
| LoRA | 主要训练小矩阵，减少可训练梯度和optimizer | 基座权重仍需容纳；反向所需激活仍存在 |
| 减 microbatch / 长度 | 缩小一次参与运算的激活与临时量 | 可能影响吞吐或任务覆盖 |

重算是一种保存策略，文件checkpoint是一种恢复策略，同名不代表同一种操作。PyTorch官方博客还讨论选择性重算：把廉价操作重算、较贵的算子保留，是一种更细的显存/计算取舍；是否比整层重算快需测量。[PyTorch 重算技术分享](https://pytorch.org/blog/activation-checkpointing-techniques/)、[LoRA 基础](#q=lora-finetuning)。

## 5. 混合精度：dtype 是逐对象、逐算子决定的

FP32共32位；FP16和BF16共16位，但分配不同。FP16指数5位、尾数10位；BF16指数8位、尾数7位，因而BF16有更广范围、较粗的尾数精度。这解释了为什么两者同样省字节，却可能在溢出和误差上表现不同。

autocast是按算子选计算dtype，不是“一键把所有状态砍半”。归约、softmax、累积或optimizer可能采用更高精度。FP16常使用loss scaling缓解梯度下溢：先放大loss，再在更新前还原梯度；若要clip，先unscale再clip。BF16常不需要同样的缩放，但仍可能NaN，数据、除零、不稳定loss和过大学习率都要查。[AMP 官方示例](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)。

FP8进一步缩小数值编码，需要缩放策略、硬件支持及数值验证；仅把权重存储dtype变小，不能证明每个算子都使用低精度kernel。训练端与rollout端精度不同，还会影响同一prefix/token的logprob一致性，后文再讲。

## 6. 使用与排障：一个最小训练循环怎样看？

```python
# 教学片段：batch、model已准备；使用支持BF16的CUDA设备
optimizer.zero_grad(set_to_none=True)
for batch in microbatches:       # 此处假设每块等量、loss为样本平均
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model_loss(model, batch) / len(microbatches)
    loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

不要把这段直接替代verl/slime的训练worker：它们还组织并行同步、动态batch、算法loss和状态切换。读源码时追四个位置：loss reduction、microbatch拆分、backward累积、step条件。动态长度需改成与全局目标相容的权重。DDP `no_sync()` 可减少中间累积的梯度同步，但上下文应覆盖forward与backward；最后一块仍同步。FSDP有不同存储代价，不能照搬。[DDP API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。

OOM按发生阶段处理，比轮流打开开关更有信息：

1. **加载模型就OOM**：先核参数dtype、模型副本、可见设备和预期分片。
2. **首个optimizer step才OOM**：检查状态是否此时才分配，是否存在master副本。
3. **长样本forward/backward OOM**：检查长度尾部、完整logits、激活和packing策略。
4. **多步持续增长**：检查列表是否保存带计算图的loss/输出；日志通常应detach或转标量。
5. **切换到rollout时OOM**：检查训练状态是否仍驻留、KV池与图捕获内存，不能只按一个角色估算。

`memory_allocated`近似框架活跃张量；`memory_reserved`是allocator保留池。`nvidia-smi`还包含上下文和框架外分配。reserved大于allocated不是自动证明泄漏。`empty_cache()`不能释放仍被活跃张量持有的内存，也不是减少模型容量需求的方案。用各阶段峰值或memory snapshot区分根因。[PyTorch CUDA 内存](https://docs.pytorch.org/docs/2.14/notes/cuda.html)、[CUDA memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory)。

## 7. 面试问答：从30秒回答走到可验证追问

以下均为**本文整理题**。设计习惯参考Chip Huyen的作者面试材料：先问约束、数据和评测，再解释选择；未声称为特定公司的真题或统计高频。[作者的系统面试讨论](https://huyenchip.com/machine-learning-systems-design/research-vs-production.html)。

**Q01 [基础] backward和step有什么区别？** 直答：前者计算并累加梯度，后者使用梯度更新参数。机制：Adam还更新长期状态。例子：累积4次backward后只step一次。追问：什么时候清梯度？回答应明确更新边界，不能每块都清。

**Q02 [推导] 7B需要多少显存？** 先确认训练还是推理、参数/梯度/状态dtype、并行方式。按本篇16字节布局是104.3 GiB模型状态，再加激活/临时量；不能给一个无条件常数。追问：换FP32梯度后增加什么？增加约14 GB。

**Q03 [基础] 增加累积次数能减少optimizer状态吗？** 不能，主要改变一次激活占用和更新频率。例子：相同模型、相同可训练参数的Adam状态不随microbatch变小。追问：怎样真正分掉状态？转到ZeRO/FSDP。

**Q04 [推导] 不等长microbatch平均为什么错？** 分母变了；用10/90 token例子说明1.5与1.1。追问：DDP为何还要看默认平均？因为归一化可能被做两次。

**Q05 [排障] 开BF16后为什么显存没减半？** 检查参数、optimizer和临时量是否仍FP32；autocast只处理选定计算。追问：如何证明？统计实际dtype/字节及各阶段峰值，比较相同任务配置。

**Q06 [排障] loss正常但更新后NaN怎么查？** 区分loss、梯度、参数的首次非有限位置；查grad norm、scaler是否跳step、分母、学习率。追问：clip能修所有NaN吗？不能，NaN梯度不是单纯范数过大。

**Q07 [使用] activation checkpoint为什么可更慢却提高吞吐？** 每块更慢，但释放容量后可能允许更大microbatch，减少小kernel/同步开销。这是可能性，需比较有效token/s、峰值和质量。追问：应固定哪些量？global batch、训练目标与数据分布。

**Q08 [设计] 用LoRA就不用估激活了吗？** 仍需反向传播到adapter并经过基座计算，很多中间量仍要保留。追问：冻结层在什么条件下可以no-grad？若更早的可训练参数需要经该层反向传播梯度，就不能随意截断。

## 8. 复习练习、来源与下一步

**先独立回答：** 画训练生命周期；按实际dtype填写5行显存账本；算global batch；用不等长分组证明梯度归一化。**再设计GPU练习：** 相同数据、模型和global batch，逐一比较microbatch、BF16与重算；报告有效token/s、峰值allocated/reserved、最大梯度误差。GPU结果本轮没有测得。

| 来源 | 类型 | 阅读用途 |
|---|---|---|
| [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html) | 官方教程，2.14固定路径 | 计算图、保存张量 |
| [Adam](https://arxiv.org/abs/1412.6980v9) | 原论文 | optimizer统计与更新 |
| [AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html) | 官方教程，2.14 | 缩放、累积与clip顺序 |
| [DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html) | 官方API，2.14 | 平均语义与no_sync |
| [Memory anatomy](https://huggingface.co/docs/transformers/en/model_memory_anatomy) | 官方教程 | 状态、临时量与dtype账本 |
| [ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) | 官方教程 | 分片对象与容量 |
| [Activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/) | 官方技术博客/分享 | 重算策略的取舍 |
| [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) / [Memory snapshot](https://docs.pytorch.org/docs/stable/torch_cuda_memory) | 官方API与工具 | allocator与OOM调查 |
| [Chip Huyen 系统面试讨论](https://huyenchip.com/machine-learning-systems-design/research-vs-production.html) | 作者面试材料 | 约束、部署与评测的问法 |

来源清单可下载：[本篇来源](./research/infra-training-step-sources.json)。动态文档是本次核对快照，不是对任意版本的兼容承诺。下一篇用性能模型解释“能跑”和“跑得快”的差别。

**数值练习下载：** [CPU验算脚本（仅Python标准库）](./assets/infra/infra-foundations-check.py) · [本次验算输出](./assets/infra/infra-foundations-check-results.json)。覆盖梯度有限差分、状态/KV字节、Roofline、TP与分块softmax；不包含GPU训练或测速。
