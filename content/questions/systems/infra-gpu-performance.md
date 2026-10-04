---
id: infra-gpu-performance
title: "02 · GPU 为什么没跑满？从 Roofline、通信到正确 profiling"
category: systems
difficulty: 基础
tags: ["P0", "Infra基础", "GPU", "Roofline", "NCCL", "Profiling"]
updated: 2026-10-05
summary: "拆开算力、显存带宽、启动开销与通信等待，用矩阵算例建立性能下界，再学习计时、collective、重叠与端到端排障。"
draft: false
---

## 1. 主问题：利用率低，首先要知道谁在等谁

本篇解释“模型能放下但很慢”。先读 [训练一步与显存](#q=infra-training-step)，再看这里的时间账本。GPU擅长大量并行的规则运算，实际任务还要经过CPU准备、kernel启动、HBM读写、GPU互联、跨机网络和阶段屏障。一个指标不能把这些区别全部表达出来。

先定义四个单位：**FLOP**是一项浮点操作，**FLOP/s**是算术速度，**byte/s**是搬运速度，**second**是完成任务的时间。一次乘加通常按2 FLOPs计算。TFLOP/s用十进制 $10^{12}$，不与显存GiB的二进制单位混用。

调研核对日为2026-10-05。下文硬件能力采用自设数值，只用于算例；不对应某款GPU的测量结果。性能推断依据官方性能文档与教程，不能用一个下界预测实际benchmark。

## 2. 硬件与运算：显存带宽、Tensor Core、SM是什么？

HBM是GPU外部高带宽显存；寄存器和片上共享内存容量小但更靠近计算；缓存帮助复用数据。SM是调度线程与执行运算的处理单元，Tensor Core适合支持格式的矩阵运算。全局显存里有空闲容量，并不意味着读写速度足够；“放得下”和“供得上”是不同约束。

一个Transformer含矩阵乘、attention、归一化、激活、索引与采样。大矩阵乘可以高效复用数据，小矩阵或零散算子则可能没有足够并行工作，还会付出启动成本。FP16/BF16张量存在，并不能独自证明每条路径都命中Tensor Core；要看形状、布局、kernel与实际指令路径。[NVIDIA GEMM 性能背景](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。

`nvidia-smi`的GPU busy指标反映采样窗口内是否有工作在执行，不能当作“Tensor Core达到多少峰值”。一个访存瓶颈kernel也可以长时间占用GPU；相反，小kernel之间频繁CPU空隙可能令busy降低。SM occupancy表示驻留warp等资源占用，也不等于有效运算吞吐。

## 3. Roofline：怎样用算量与字节量定位瓶颈？

设一个操作有 $F$ FLOPs、需从HBM传输 $Q$ 字节，硬件可提供算力 $C$ FLOP/s、带宽 $W$ byte/s。在理想重叠和无额外成本下：

$$
t\geq\max(F/C,Q/W),\qquad I=F/Q
$$

$I$叫算术强度，单位FLOP/byte。可达算术吞吐的上界为 $\min(C,WI)$。转折点 $I^*=C/W$：低于它主要受带宽上界限制，高于它更可能受算力上界限制。[How To Scale Your Model · Rooflines](https://jax-ml.github.io/scaling-book/roofline/)。

![本文示意：Roofline的带宽斜线、算力平台与转折点](./assets/infra/infra-gpu-performance/roofline.svg)

**图解：** 横轴是算术强度，纵轴是可达吞吐。左侧线性增长代表每字节只做少量运算时，HBM限制性能；右侧平台代表算力上限。示意没有刻度与实测点。真实kernel会位于上界下方；网络通信还要使用对应网络带宽，不能把HBM带宽代入跨机传输。

### 手算：同样4096×4096权重，为什么batch变化很大？

考虑 $X[B,4096]W[4096,4096]$，输入/权重/输出BF16，忽略cache、额外读写与融合，权重只读取一次。则：

$$
F=2B\times4096^2,\qquad Q=2(4096^2+2B\times4096)
$$

- $B=1$：约33.55 MFLOPs、33.57 MB，$I\approx1$ FLOP/byte。
- $B=128$：约4.29 GFLOPs、35.65 MB，$I\approx120.47$。

以**假想设备** $C=100$ TFLOP/s、$W=1$ TB/s计算，转折点是100 FLOP/byte。B=1的算力时间下界约0.34微秒，访存下界约33.57微秒；B=128对应约42.95和35.65微秒。batch增大128倍，权重流量并没有增大128倍，复用使瓶颈发生变化。

这不表示B=128一定运行42.95微秒。真实算子还受tile形状、可用并行度、读写模式与启动成本影响；缓存可能减少HBM读量，bias/activation等可能增加它。算例的作用是提出可测试假设：低batch是否主要在重复搬权重？增加batch是否有效复用？

这也解释低并发decode常受访存影响，prefill将多个已知token组织成较大矩阵更容易利用算力。但长上下文KV读写、MoE路由与多卡通信会改变整体瓶颈，不能把所有decode都归为一个类别。[Transformer inference 教程](https://jax-ml.github.io/scaling-book/inference/)。

## 4. 通信基础：collective是在搬什么？

rank是参与分布式任务的进程编号，process group是一起通信的集合。collective要求成员按相容顺序调用，通常不是“发送给一个中心服务器”。以两个rank的向量为例：

| 操作 | 输入示例 | 结果 |
|---|---|---|
| AllReduce(sum) | rank0=[1,2]，rank1=[3,4] | 每个rank得到[4,6] |
| AllGather | rank0=[1,2]，rank1=[3,4] | 每个rank得到[1,2,3,4] |
| ReduceScatter(sum) | 两边都持有2段输入 | 对应元素求和后，每个rank只得到一段 |
| Broadcast | 一个rank有参数 | 其他成员得到同一内容 |
| AllToAll | 每rank有发给各rank的不同段 | 按目的rank交换，常用于expert dispatch |

DDP通常在backward期间同步梯度；全分片训练会在计算前聚合参数，计算后分片梯度。TP常交换中间结果，MoE会交换路由token。先问对象、大小、频率、通信组，再谈NCCL慢不慢。[NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)。

### Q：为什么小消息与大消息要分开看？

教学模型 $t_{comm}\approx\alpha+Q/W_{eff}$ 中，$\alpha$是启动/握手等固定延迟，$W_{eff}$是该负载有效带宽。小消息可能主要付延迟；大消息主要付字节搬运。collective会经历多个步骤，不能把一次点对点公式直接当完整成本。

以ring all-reduce、D个rank、每rank持有S字节完整输入为例，reduce-scatter加all-gather总发送量约 $2(D-1)S/D$，步骤数约 $2(D-1)$。近似：

$$
t_{ring}\approx2(D-1)\alpha+\frac{2(D-1)S}{D\,W_{link}}
$$

这是环与均衡链路假设，不是NCCL永远采用的算法。树、分层、多rail、交换网络与实现可改变结果。区分算法带宽和bus bandwidth也很关键：它们可能采用不同的传输量换算，不宜把benchmark显示值与网卡标称值直接相除。

假设D=8、S=1 GiB，每rank发送约1.75 GiB。若链路有效25 GB/s，纯传输下界约75.16 ms，另有延迟。25 GB/s是本例假设，不是“某网卡25 Gbps”；Gbps换GB/s要除8。

### Q：卡数多了为什么不快？

数据量固定时，每卡算量下降，但同步、启动与跨机开销未必同比下降，即strong scaling。保持每卡负载不变、增加总数据量属于weak scaling。比较8与16卡应先明确是哪种。

吞吐加速比 $S_D=T_1/T_D$，效率 $E_D=S_D/D$。若8卡耗时20s、16卡15s，扩容2倍只快1.33倍；总GPU-second从160升到240。更快交付与更低训练成本是两个目标。Amdahl定律 $S\leq1/(f+(1-f)/D)$ 说明不可并行部分会限制扩展，但现实通信还可能随D增长，使效果更差。

## 5. 重叠与计时：异步API不是性能证据

CUDA kernel通常异步提交：CPU调用返回，不等于GPU完成。用Python计时包住一次matmul，可能主要测到提交时间。基准应warmup，使用CUDA events或在明确边界同步；不要把每个算子后都synchronize加进正常训练，它会破坏重叠。[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。

```python
# CUDA教学片段；x、w已驻留GPU，本轮未执行GPU测速
for _ in range(10):
    y = x @ w
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    y = x @ w
end.record()
end.synchronize()
print(start.elapsed_time(end) / 100)   # 毫秒/次，不是秒
```

events记录所在stream的事件；多stream或分布式端到端任务，需要包含所有依赖完成，而不只测主stream。一次cold start可能含加载、编译、图捕获、缓存初始化；steady-state另报，不用warmup隐藏真实启动需求。

### Q：通信如何与反向重叠？

已完成的梯度bucket可以开始同步，同时后面的层继续反向。必须同时满足：当前通信的输入已经ready；存在不依赖其结果的计算；stream与事件正确；硬件资源允许并发。过早调用wait会串行化，两个活动争用SM或链路也可能让重叠收益变小。只写 `async_op=True` 并不能证明这些条件成立。

bucket太小会增加通信次数；太大要等更多梯度ready，暴露尾部等待。最优大小取决于层结构与网络，不宜背一个固定阈值。FSDP预取通过提前聚合下一层参数来重叠，但会同时保留更多参数，容量与速度又产生取舍。[FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。

## 6. Profiling工作流：从端到端时间线逐层缩小

先记录端到端每step耗时，再拆data wait、forward、backward、optimizer、通信、保存与评估。PyTorch profiler适合把CPU调用与CUDA活动放到时间线，`record_shapes`、stack和memory信息有额外成本，使用短窗口，不能把重度profile运行直接当生产吞吐。[Profiler 官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。

| 时间线现象 | 首先验证 | 有针对性的下一步 |
|---|---|---|
| GPU活动之间大空洞 | dataloader、CPU调度、tokenization、同步 | 分阶段计时；查看CPU栈与等待点 |
| 很多小kernel | shape小、碎片算子、Python提交 | 适当batch、融合或图捕获；验证数值 |
| 大部分时间在collective | 通信组、字节量、拓扑、尾部rank | 同拓扑通信微基准；查看ready时刻 |
| matmul有活动但吞吐低 | 强度、tile、dtype、layout | 固定shape sweep，核对kernel |
| 只有某些step突然慢 | 长样本、checkpoint、编译重触发 | 按长度与事件关联，报告p95 |
| 工具等待时GPU闲 | 没有ready推理请求 | 任务级调度和准入；不是kernel问题 |

**案例（教学假设）：** step=10s，其中rollout=7s、训练=2s、同步/其他=1s。训练kernel快2倍后，step只变9s，收益约1.11倍。若把两阶段完全理想重叠，不能简单宣称得到7s；还要算数据依赖、评分、同步与长尾。优化预算应优先分配给真实关键路径。

建议维护一个固定负载表：模型与版本、dtype、GPU/互联、batch/长度分布、并行度、是否warmup、tokens定义、计时范围、峰值与质量。改变两项后变快，无法确定是哪一项贡献；按单变量对照，再组合。

在verl/slime中，先看角色与阶段，再看worker中的kernel。rollout低利用率可能来自工具等待，learner等数据可能来自评分服务，不能都归因于“NCCL慢”。具体闭环见 [RL流水线](#q=infra-rl-pipeline)。

## 7. 面试问答：回答瓶颈时给出观察和反证

题目均为本文根据官方机制整理的练习；系统设计问法参考 [Chip Huyen 作者面试材料](https://huyenchip.com/machine-learning-systems-design/research-vs-production.html)，没有公司频率统计。

**Q01 [基础] 算力高的GPU为什么未必快？** 直答：任务可能受带宽、启动、通信或外部等待限制。例子：B=1读取整个权重矩阵，强度约1。追问：你怎么证明？events计时加shape sweep，结合profile和字节模型，不只看busy百分比。

**Q02 [推导] Roofline转折点单位是什么？** $C/W$为FLOP/byte。带宽用byte/s而不是bit/s。追问：加入网络通信如何做？给网络流量单独算下界，不用HBM带宽。

**Q03 [基础] AllReduce和AllGather区别？** 一个规约合并同位置值，另一个拼接各自分片。例子：梯度求和与参数聚合。追问：AllReduce一定平均吗？通信可求sum，框架另决定是否除以组大小。

**Q04 [推导] 8卡梯度all-reduce发送多少？** 在ring假设下每rank约1.75S，不是S/8。追问：这个能精确预测NCCL吗？不能，算法、拓扑、延迟与协议仍需测。

**Q05 [排障] Python测matmul只用几微秒可信么？** 先确认是否只测提交，是否包含warmup与同步。追问：每次同步更准确吗？测单次latency可以，但会改变正常流水线，不能直接当steady-state吞吐。

**Q06 [排障] async_op开启却没有重叠？** 查通信ready时刻、wait位置、独立计算与stream依赖。追问：重叠多了为什么还能更慢？资源争用或更小计算粒度的损失可能超过收益。

**Q07 [设计] 16卡比8卡快25%值得扩容吗？** 先问目标是时延还是GPU-hour，再固定global/per-rank工作口径计算。追问：收敛步数也变化了呢？必须比较到同质量目标的总成本。

**Q08 [设计] 性能优化后的验收指标是什么？** 同负载端到端有效吞吐/时延、峰值和质量；阶段指标帮助归因。追问：把失败任务丢掉得到更高token/s算成功吗？需要计入完成率和任务收益。

## 8. 复习练习与来源

**CPU可验算：** 写出B=1/128的F/Q/I，计算假想设备下界；换成FP32字节，观察转折；计算8→16卡的GPU-hour。**GPU待做：** 以固定matmul sweep batch，分别报告events时间与CPU提交时间，再与profile时间线对照。

| 来源 | 类型 | 本篇用途 |
|---|---|---|
| [NVIDIA GEMM 性能背景](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) | 官方性能教程 | 形状、强度、tile与对齐 |
| [Rooflines](https://jax-ml.github.io/scaling-book/roofline/) | 作者系统教程 | 算量/带宽下界 |
| [Transformer inference](https://jax-ml.github.io/scaling-book/inference/) | 作者系统教程 | prefill/decode负载区别 |
| [NCCL collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) | 官方文档 | 通信语义 |
| [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html) | 官方API，2.14 | stream、同步与events |
| [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 官方教程 | 聚合与预取的时间/容量取舍 |
| [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | 官方教程 | 时间线与采样窗口 |
| [Chip Huyen 面试讨论](https://huyenchip.com/machine-learning-systems-design/research-vs-production.html) | 作者材料 | 约束与评测追问 |

[本篇来源清单](./research/infra-gpu-performance-sources.json)。下一篇把“存什么、搬什么”用于解释各类并行机制，最后再回到框架配置。

**数值练习下载：** [CPU验算脚本（仅Python标准库）](./assets/infra/infra-foundations-check.py) · [本次验算输出](./assets/infra/infra-foundations-check-results.json)。覆盖梯度有限差分、状态/KV字节、Roofline、TP与分块softmax；不包含GPU训练或测速。
