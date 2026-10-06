---
id: infra-ai-accelerators
title: "09 · A卡、H卡、B卡怎样训练模型？GPU架构、显存、互联与其他AI算力卡"
category: systems
difficulty: 进阶
tags: ["P0", "Infra基础", "GPU架构", "A100", "H100", "Blackwell", "AMD", "算力卡"]
updated: 2026-10-06
summary: "以A100→H100/H200→B200/GB200为主线，比较架构、精度、HBM、互联与软件栈，补充B300、AMD、Gaudi、TPU、Trainium和昇腾，用训练/推理算例解释选型。"
draft: false
---

## 1. 主问题与命名：先确认在比较哪一种产品

这里将训练场景中的“A/H/B卡”理解为 **NVIDIA A100、H100/H200、B200等系列**；“A卡”口语中也可能指AMD，AMD Instinct在第7节独立介绍。字母不是完整规格：同架构可包含不同产品，同型号也有不同显存、封装、功耗与互联形态。

读完应能回答：这些代际改变了什么；为什么更高TFLOPS不总是更快；全参训练、长上下文、LoRA与RL rollout分别应先看哪些硬件条件；从CUDA换到另一家加速器要验证什么。先修 [训练显存账本](#q=infra-training-step)、[GPU性能模型](#q=infra-gpu-performance)、[并行](#q=infra-distributed-parallelism)与 [1T容量](#q=infra-checkpoint-formats)。

核对日 **2026-10-06**。本文对比的是官方规格与机制；计算为标明假设的CPU算例，未执行跨显卡GPU benchmark。正文保留来源与SKU，不把厂商宣传的最高倍数当统一训练结果。

### GPU、板卡、Superchip、基板和机架怎样分层？

| 名称 | 所在层 | 比较时不能混用 |
|---|---|---|
| A100/H100/B200 | GPU产品，仍需SKU/形态 | 某GPU峰值和整机峰值 |
| PCIe卡、SXM模块、AMD OAM | 封装/板卡形态 | SXM互联能力不能套给任意PCIe机器 |
| HGX | 多GPU基板/平台 | HGX8卡总量不能当单卡量 |
| DGX | NVIDIA整机/系统产品 | CPU、网卡、内存、存储也影响训练 |
| GB200 Superchip | Grace CPU与Blackwell GPU组合 | 不是“一张普通B200显卡” |
| GB200 NVL72 | 72GPU、36Grace CPU的机架级系统 | 不是72个可当单卡共享内存使用的槽位 |

[GB200官方产品页](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)区分Superchip的1CPU+2GPU与NVL72的36CPU+72GPU。Grace CPU的LPDDR内存也不是GPU HBM；一致性互联不代表二者具有相同带宽和访问成本。

## 2. 读规格的方法：容量、带宽、精度、通信、软件

评价训练硬件至少需要五个维度：

1. **容量**：参数、梯度、optimizer、激活、临时量是否放得下。
2. **HBM/GDDR带宽**：单位时间能把多少数据供给计算，而非能存多少。
3. **目标精度算力**：BF16/FP16、FP8、FP4对应的kernel和数值策略是否能使用。
4. **互联**：TP、分片聚合、CP与MoE的数据经哪条链路、以何种拓扑交换。
5. **软件路径**：框架、算子、编译目标、通信库、checkpoint和部署是否支持。

![本文示意：算力、显存容量、带宽、互联与软件共同决定训练结果](./assets/infra/infra-ai-accelerators/training-limits.svg)

**图解：** 中心是同一个任务，外围五块分别给出约束。任何一项成为瓶颈，都可能阻止更高峰值算力转化为端到端收益。没有画“谁最快”的排名或实测刻度；比较前应先明确任务与目标。

### 为什么必须区分dense、sparse和不同精度？

NVIDIA规格表经常把带结构化稀疏的Tensor Core峰值加星号。特定2:4等结构满足硬件/算子要求时才可利用；普通dense Transformer或只含不少零值，不会自动获得该倍数。把H100的FP8 sparse峰值与A100的BF16 dense峰值相除，还同时混入精度和稀疏因素。

TF32是某些FP32输入矩阵运算的加速数值路径，不是新的4字节checkpoint格式。FP8/FP4则需要缩放/量化、累积与稳定性策略。能执行某个低精度算子，不等于整个训练的参数、梯度、master与Adam都采用该精度。见 [精度与状态](#q=infra-training-step)。

### 为什么同一数字也要确认单位与方向？

HBM带宽通常按TB/s或GB/s；网卡常用Gb/s，400Gb/s的字节率是50GB/s。NVLink、PCIe常给双向总量，拿它直接当单向有效all-reduce带宽会高估。规格是接口/硬件峰值，实际通信还涉及collective算法、并发、拓扑和消息大小。

## 3. NVIDIA A/H/B主线：按明确SKU核对的规格表

以下**显存容量按厂商GB标签列出**；算例按十进制换算，实际部署应读取驱动报告的byte与可用容量。HBM与互联均为标称值，NVLink列按官方双向总带宽口径；BF16/FP16列统一采用dense Tensor Core理论峰值，不列普通CUDA core FP32指标。

| 对象与形态 | 架构 / Tensor Core代际 | 显存 | HBM带宽 | BF16/FP16 dense约 | NVLink标称双向 |
|---|---|---:|---:|---:|---:|
| A100 80GB SXM | Ampere / 第3代 | 80GB HBM2e | 2.039TB/s | 312TFLOP/s | 600GB/s |
| H100 SXM | Hopper / 第4代 | 80GB HBM3 | 3.35TB/s | 989TFLOP/s级 | 900GB/s |
| H200 SXM | Hopper / 第4代 | 141GB HBM3E | 4.8TB/s | 989TFLOP/s级 | 900GB/s |
| HGX B200中的单GPU | Blackwell / 第5代 | 180GB HBM3E | 7.7TB/s | 2.25PFLOP/s | 1.8TB/s |
| GB200中的单GPU | Blackwell / 第5代 | 186GB HBM3E | 8TB/s | 2.5PFLOP/s | 1.8TB/s |
| HGX B300中的单GPU | Blackwell Ultra | 270GB HBM3E | 7.7TB/s | 2.25PFLOP/s | 1.8TB/s |

来源：[A100](https://www.nvidia.com/en-us/data-center/a100/)、[H100](https://www.nvidia.com/en-us/data-center/h100/)、[H200](https://www.nvidia.com/en-us/data-center/h200/)、[Blackwell固定PDF第8页](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf)、[Blackwell Ultra固定PDF第5页](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf)。H100/H200网页1979TFLOPS带稀疏脚注，除2约989.5；B200的4.5PFLOPS带稀疏脚注，除2得到2.25。不能把它们直接当dense。

**资料口径记录：** 本篇B200/B300使用此次下载PDF中具体HGX列的180/270GB与7.7TB/s，GB200列为186GB与8TB/s；其他发布期或产品资料中的192/288等数字不替代这张表。不同资料的SKU、修订与容量口径应核对，本文不凭数字差异推断原因。两份PDF的SHA256记录在来源清单，避免以后上游同URL更换文件时丢失本次依据。

### 同型号的形态差别如何影响训练？

A100 80GB PCIe官方带宽为1.935TB/s，低于表中的SXM2.039；PCIe bridge连接能力与HGX/NVSwitch的多卡布局也不同。H100 NVL官方列每GPU94GB、3.9TB/s，不能和SXM80GB/3.35混为同一对象。H200也有NVL，必须查桥接规模、功耗及服务器配置，不能仅按“H卡”套参数。

SXM常服务密集多GPU平台，PCIe更容易接入常规服务器槽位；但GPU产品与实际服务器决定互联，SXM本身不是“保证快”的算法。价格/库存/可用镜像取决于供应商，本篇不给脱离实际服务条件的采购结论。

## 4. 架构差异：A100→Hopper→Blackwell改变了哪条训练路径？

### Ampere/A100：BF16、TF32与成熟多GPU基线

A100的第3代Tensor Cores支持BF16等路径，Ampere引入TF32与结构化稀疏等能力。对已有BF16/FP16训练代码，它可作为稳定能力基线：数值目标、优化器和mask仍由框架定义，GPU主要改变执行与容量约束。[Ampere作者技术介绍](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。

它适合被理解为“我能否在现有成熟软件栈下跑通目标任务”，而不是“老卡只能推理”。A100也可全参训练、SFT、LoRA、Diffusion或RL learner，具体规模看状态/激活与并行布局。是否值得使用取决于同质量下实际成本和训练时长。

### Hopper/H100：FP8与更高效的数据移动

Hopper的第4代Tensor Cores加入FP8路径，Transformer Engine组织低精度与缩放；TMA等机制帮助块级数据传输和异步执行。收益需要算子/编译器使用相应机制，不是换卡后每个Python操作都自动得到同样倍数。[Hopper作者技术介绍](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)。

![NVIDIA原图：GH100完整芯片结构示意，来源为2022年Hopper技术文章](./assets/infra/infra-ai-accelerators/h100-full-chip-original.png)

**图源与逐层解读：** NVIDIA作者原图。中央重复单元是GPC/TPC/SM的组织，Tensor Core等计算资源位于SM内；中间蓝色L2缓存减少重复HBM访问；两侧HBM和memory controllers供给数据；底部NVLink用于GPU间通信；顶部PCIe是主机接口。图中144SM是完整GH100结构，文章另区分H100 SXM/PCIe的启用SM配置，不能把144当每张出厂H100的数量。该历史架构图不表达当前SKU全部规格，也不是性能测量。

### H200：Hopper计算代际中提升显存与带宽

H200仍属Hopper，表中BF16 dense峰值与H100同量级，关键变化是141GB HBM3E与4.8TB/s。因此，纯GEMM瓶颈未必有与容量比相同的加速；如果原任务受显存、KV、offload或访存限制，额外容量/带宽更可能有价值。这个判断来自规格与性能模型，需同负载验证。

### Blackwell/B200：计算、内存与系统互联一起扩大

数据中心Blackwell的第5代Tensor Cores与新的低精度路径、双die GPU设计和更大NVLink带宽共同影响模型执行。两个计算die以片上互联组成一个GPU，不应直接算成两个独立DP副本。GB200进一步把Grace CPU与GPU组成Superchip，再扩展成机架级NVLink域。[Blackwell架构介绍](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)。

Blackwell支持更低位浮点运算，不等于可把任意BF16训练脚本直接改成全FP4；训练中的哪些GEMM、缩放、累积、梯度和状态采用什么dtype，仍由方法与实现决定。[Transformer Engine低精度指南](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/index.html)。

### Blackwell Ultra/B300：为什么“新卡”也不保证所有BF16算子更快？

本篇固定PDF的HGX B300与B200列BF16 dense峰值同为约2.25PFLOP/s；B300提高容量并扩展某些低精度/attention能力。若任务保持BF16且没有使用改变的路径，不能仅用新型号给出统一训练加速倍数。B300/GB300形态及系统配置应独立看。

架构谱系仍在更新；[当前HGX页面](https://www.nvidia.com/en-us/data-center/hgx/)也列出Rubin。本文保持A/H/B主线，不把Blackwell称为永远最新的一代，也不据产品页面预测你的云端供给。

## 5. 训练模型的区别：容量、吞吐、精度与收敛同时看

### Q：换H卡/B卡要改变模型算法吗？

若同模型、同精度、同数据和同更新目标，只换支持的硬件/算子路径，算法不必变化；但并行度、microbatch、kernel选择、精度和随机顺序可能实际改变。先比较小任务的loss/梯度/更新，再比较训练收敛与端到端时间。

采用FP8/FP4训练时，数值路线已改变，需要同时核对质量与速度。更大显存允许更大microbatch而保持global batch，但若也改变global batch、学习率或序列长度，就不能把所有差异归因于硬件。

### 手算1：70B全参训练为何80GB与180GB差别很大？

使用之前的特定16字节/参数布局：BF16参数/梯度、FP32 master与两份Adam统计，70B合计1120GB模型状态。ZeRO-3/FSDP在D8副本理想均分，每GPU常驻状态140GB，尚未计激活/当前聚合层/buffer：

- 80GB A100/H100不能只靠这一均分状态预算放下。
- 141GB H200即使名义接近，也几乎没有其他空间，不能据此保证能跑。
- 180GB B200留约40GB名义余量，仍需检查聚合、激活与任务长度；更高stage/卡数/重算/offload也可能改变预算。

若D16，状态约70GB/卡，80GB卡仍很紧。此例只比较常驻状态，不是“70B训练必需某卡”的结论。LoRA可减少可训练状态，但基座和反向激活仍要核算。详见 [ZeRO](#q=infra-zero-deepspeed)。

### 手算2：H100与H200为什么同算力也可有不同访存收益？

假定模型某路径有F=100TFLOPs，需搬Q=400GB，两者可理想重叠。使用约989.5TFLOP/s计算、H1003.35TB/s与H2004.8TB/s：

$$
t\geq\max(F/C,Q/W)
$$

计算下界约101.1ms；H100的访存下界119.4ms，H200为83.3ms。理想最大值约119.4→101.1ms，只是约1.18倍，而不是带宽比1.43倍，因为改善访存后算力成为限制。真实任务还有cache、kernel、通信等开销。这是CPU算例，不是两卡测速。[Roofline复习](#q=infra-gpu-performance)。

### 手算3：更大显存是否能直接拼成一块内存？

8×80GB的名义总容量640GB；在普通DP复制下，每副本仍只有80GB，并没有单模型640GB空间。TP/PP/FSDP/CP分别分配不同对象，需要执行相应通信。GB200 NVL72的13.4TB HBM也不是普通单卡allocator可直接当同等带宽使用的一块内存。

对上一轮1T BF16纯权重2TB，用每卡80%容量作**均匀分片教学下界**：80GB卡需要至少32个设备；180GB卡需要至少14个；270GB卡需要至少10个。没有计KV/激活、实际dtype与可切分形状，不能当可部署机器清单。单位与公式见 [1T容量详解](#q=infra-checkpoint-formats)。

## 6. 互联与系统形态：NVLink不能替代所有网络

### 单GPU、节点内scale-up与跨节点scale-out

HBM是GPU内部供给路径；NVLink/Infinity Fabric等用于设备之间；PCIe连接主机/设备；InfiniBand或Ethernet/RoCE等承担跨节点通信。具体机型可能扩展设备互联域，例如NVL72，但不能据NVLink名字假设任意服务器之间都有同带宽连接。

TP高频交换每层结果，通常更依赖低延迟高速互联；PP传stage激活；FSDP/ZeRO聚合参数与梯度；CP交换上下文，MoE EP进行token dispatch/combine。节点内很快不代表跨机也快，网卡数量、路由、oversubscription、CPU/PCIe亲和性都要记录。[并行与通信](#q=infra-distributed-parallelism)。

600→900→1800GB/s标称NVLink升级提供更多通信能力，但NCCL带宽不能直接填成该数字。不同统计方向、路径、collective与SM竞争会改变实际结果。400Gb/s链路的理论字节率50GB/s，远不是400GB/s；微基准也要在真实拓扑运行。

### 功耗、散热和CPU架构为何是训练问题？

功耗限制、频率、散热和同机其他任务会影响可持续吞吐，不能用GPU名字代替机器信息。GB200中的Grace是Arm CPU，镜像和原生扩展也需匹配CPU架构；即使Python源码相同，x86二进制轮子/工具不一定可用。

MIG可将支持的GPU切为隔离实例，适合某些多租户任务；租到的是MIG slice时，显存、计算和通信能力不是完整GPU。比较训练结果前核对是不是共享/MIG/vGPU，不能只记录父卡型号。

## 7. 其他AI算力卡：同一模型还需要哪个软件栈？

本节是定位与迁移差异，不是把所有加速器做跨平台速度排名。容量/带宽有相同物理意义，但执行单元、精度和互联的口径不能只凭一列TOPS直接比较。

| 家族 / 示例 | 架构与规格线索 | 软件/训练接入重点 |
|---|---|---|
| AMD MI300X | CDNA3；192GB HBM3、5.3TB/s | ROCm/HIP、RCCL；框架和自定义kernel适配 |
| AMD MI325X | CDNA3；256GB HBM3E、6TB/s | 更大容量/带宽；不是换了另一代CDNA |
| AMD MI355X | CDNA4；288GB HBM3E、8TB/s | 新dtype/算子与版本支持；OAM/服务器拓扑 |
| Intel Gaudi3 | AI专用加速器；128GB HBM、3.7TB/s | SynapseAI/Habana、HPU与分布式路径 |
| Google TPU v6e | 每chip32GB HBM、1638GB/s，2D torus ICI | JAX/XLA或PyTorch/XLA；按slice拓扑规划 |
| Google TPU7x / Ironwood | 每chip192GB HBM，约7.37TB/s | 官方BF16/FP8训练recipe，compiler/sharding栈 |
| AWS Trainium2 | Neuron文档每chip96GiB；Trn2实例含多个chip | Neuron SDK/compiler、NeuronLink，按实例配置 |
| 昇腾 / Atlas | AI加速芯片与系统系列，具体SKU另查 | CANN、HCCL、MindSpore/Ascend for PyTorch适配 |

来源：[MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)、[MI325X](https://www.amd.com/en/partner/articles/instinct-mi325x-accelerators.html)、[MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)、[Gaudi3官方介绍](https://www.intel.com/content/www/us/en/newsroom/news/vision-2024-gaudi-3-ai-accelerator.html)、[TPU v6e](https://docs.cloud.google.com/tpu/docs/v6e)、[TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x)、[Trainium2架构](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html)、[昇腾官方入口](https://www.hiascend.com/en/)。本篇不对未核到具体官方板卡规格的昇腾型号填写传闻TFLOPS数字。

### AMD：显存大意味着哪些机会，哪些仍需测试？

CDNA计算GPU以CU/Matrix Core执行标量、向量和矩阵路径；MI300系列采用XCD计算chiplet与I/O die、HBM及Infinity Fabric组织系统，不能把一个XCD当一张独立显卡。[MI300微架构](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html)。CDNA4的Matrix Cores扩展MXFP8/MXFP6/MXFP4等格式；它们与NVIDIA NVFP4的命名、缩放与kernel路径不应自动当作可互换。[MI350微架构](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html)。

更大HBM可给全参训练状态、长序列激活或推理KV更多空间，可能减少模型分片/CPU offload；但CUDA专用扩展、attention/quantization kernel、通信和框架recipe需要核查。不能因底层都有矩阵单元就把CUDA镜像直接当ROCm镜像。

PyTorch ROCm路径仍复用很多`torch.cuda`接口名称；实际应看`torch.version.hip`、安装包和设备信息，而不是仅凭API里有cuda认定NVIDIA。[PyTorch HIP语义](https://docs.pytorch.org/docs/2.14/notes/hip.html)。AMD集合通信路径可用RCCL，具体支持组合按[ROCm兼容矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)与[RCCL文档](https://rocm.docs.amd.com/projects/rccl/en/latest/)核对。

### Gaudi、TPU、Trainium、昇腾：迁移不只是改device字符串

Gaudi3把矩阵乘引擎MME、可编程TPC和集成网络接口一起组织，体现了计算与通信路径的不同；不是把NVIDIA Tensor Core改一个商标。[Gaudi3架构说明](https://www.intel.com/content/www/us/en/newsroom/news/vision-2024-gaudi-3-ai-accelerator.html)。TPU以MXU的systolic array推进矩阵数据流，另配向量等执行路径；需要编译器/分片计划安排计算、片上数据复用与ICI通信。[TPU架构](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)。

Gaudi文档提供HPU训练路径与软件组件，[官方PyTorch训练文档](https://docs.habana.ai/en/latest/PyTorch/index.html)是接入起点。TPU/Trainium更多依赖各自编译器与分片执行栈；需要验证shape、算子、数据输入、分布式计划和保存恢复，而不是把NCCL配置原样套过去。

TPU7x官方文档已经给出BF16/FP8训练资源，不能仅按早期“面向推理”宣传就断言无法训练。Trainium2包含多个NeuronCore-v3及数据移动/collective执行组件；Neuron文档将每chip容量标为96GiB，EC2页面将16chip实例容量标为1.5TB。这是不同层次与单位标签，部署应核对真实byte，不把系统总量当每chip容量。[Trn2实例规格](https://aws.amazon.com/ec2/instance-types/trn2/)。

昇腾的CANN包含算子/运行时/通信等相关组件，PyTorch适配和MindSpore是不同接入路径；支持矩阵应按对应Atlas产品与版本核查。[CANN官方组件说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/index/index.html)。模型能跑forward，不表示backward、optimizer、多卡和权重发布都已通过。

## 8. L40S、RTX与其他GPU：为什么也能训练，又为何不能按A/H/B替代？

[L40S官方规格](https://www.nvidia.com/en-us/data-center/l40s/)为48GB GDDR6，官方注明不支持NVLink；它使用Ada架构，具有AI算力，也能承担训练/微调等任务，但多卡通信和容量结构与HGX H100不同。不能把L40S的宣传AI峰值直接当相同BF16训练速度。

[RTX PRO6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)有96GB GDDR7；[RTX5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)有32GB GDDR7。它们的容量、功耗、系统设计和目标软件路径不同于数据中心B200，即使都叫Blackwell。

### 同为Blackwell，为什么kernel也不能随意互换？

官方CUDA compute capability表中，A100为8.0，H100/H200为9.0，B200/GB200为10.0，B300/GB300为10.3，RTX5090/RTX PRO6000 Blackwell为12.0。[官方CC表](https://developer.nvidia.com/cuda/gpus)。CC表示指令/硬件特性，不是CUDA Toolkit版本，也不是“12.0 GPU必然比10.0快”。

编译的CUDA扩展、cubin/PTX与JIT目标需要适配GPU架构；旧镜像可能缺新GPU可用的kernel image。CUDA/PTX前向兼容是有条件的，不能简单保证任意旧二进制能跑；某些架构专用特性还需要专门处理。[Blackwell兼容指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)。

对于本地LoRA、算法验证或较小模型，已有RTX设备可以是合理起点；如果要大规模全参、多机通信或高并发长上下文服务，需重新估容量、拓扑、稳定运行条件与维护成本。判断来自任务约束，不基于“消费卡不能训练”的绝对说法。

## 9. 同一张卡用于预训练、SFT、RL learner与rollout有何区别？

| 工作负载 | 主要硬件压力 | 判断入口 |
|---|---|---|
| dense预训练 / 全参SFT | 大矩阵、状态、激活与梯度通信 | BF16/FP8路径、HBM、DP/TP/PP、收敛 |
| LoRA/QLoRA | 基座+部分可训练状态+反向激活 | 目标长度、量化kernel、单卡峰值 |
| 长上下文训练 | attention算量、激活、CP流量 | kernel、重算、KV相关结构与切序列 |
| MoE训练 | expert权重、路由偏斜与all-to-all | EP/拓扑、expert负载、计算/通信重叠 |
| Diffusion/DiT | attention、空间/视频token、反向激活 | 分辨率/帧数、tensor形状、kernel与batch |
| RL learner | 训练状态、概率计算与更新 | 算法目标、packing、后端、梯度归一化 |
| rollout / serving | 权重、KV、prefill/decode、调度 | 每轮token预算、并发与实际ready请求 |
| coding agent rollout | 推理+工具/环境等待+长尾 | ready/active区分，不能只看模型峰值 |

H卡/B卡更高算力不能修复工具等待或teacher服务成为瓶颈的整个RL流程。换卡后Generate占用仍低，可能是没有ready请求，而不是GPU不够强。任务闭环见 [RL/OPD流水线](#q=infra-rl-pipeline)。

### 对verl/slime怎样接到配置？

先核框架版本与后端支持目标设备，再选择训练/生成布局；确认FSDP/Megatron、vLLM/SGLang、自定义reward/agent插件所需的kernel、通信与CPU环境。不同角色可有不同TP/DP，但权重同步与样本概率必须正确。

迁移路线：单卡forward/logprob → backward/optimizer → 小规模多卡collective → 保存/恢复 → rollout权重同步 → 完整短训练。这比“服务接口返回文字了，所以RL训练已支持”更能定位缺口。连接 [框架主文](#q=training-inference-frameworks)与 [训练后端](#q=infra-training-backends)。

## 10. 性能对比与选型：先证明训练目标一致

硬件对照至少记录：GPU准确SKU/形态、设备数与拓扑、可用显存、功耗限制、CPU/网卡/存储、驱动/框架/算子版本；模型、精度、dense/sparse、batch/长度、并行、重算/offload；数据、loss、收敛目标与计时边界。

先跑相同小任务检查数值，再测steady-state step、有效token/s、每rank峰值与通信；最后比较达到相同质量的总wall time、GPU-hour、能耗或实际费用。不同服务商定价与可用性需在真实报价中核对，不依据这篇静态规格表推导当前性价比。

[MLPerf Training](https://mlcommons.org/benchmarks/training/)衡量达到指定质量目标的训练时间。阅读时选同版本/benchmark与适合的division，并核设备数量、软件和系统配置；它是系统结果，不是抽离软件的单GPU恒定速度。厂商2x/30x宣传可能改变精度、batch、软件、系统规模和负载，不宜当跨任务排名。

### 最小验收表

| 要回答的问题 | 必须收集的证据 |
|---|---|
| 能否放下？ | 参数/状态/激活/KV与临时量的byte账本、阶段峰值 |
| 单设备是否更快？ | 同shape/dtype kernel时间与端到端数据供给 |
| 多设备能否扩展？ | 同global工作口径、通信/拓扑与扩展效率 |
| 低精度是否有效？ | 相关算子确实使用目标路径、loss/质量验收 |
| 换平台是否正确？ | forward/梯度/更新/恢复与完整角色交接 |
| 对RL是否有收益？ | 相同GPU-hour任务质量、消费比例与版本偏移 |

本篇的结论是评估步骤和机制解释，不给未经同负载实测的总榜单。

## 11. 面试问答：型号、架构与训练场景

以下为本文按官方机制整理题，不声称为公司真题或招聘频率统计。

**Q01 [基础] A/H/B字母分别是什么？** 本篇按Ampere A100、Hopper H100/H200、Blackwell B200理解；先确认完整SKU/形态。“A卡”也可能在口语中指AMD，不能仅凭字母判断。

**Q02 [机制] H200为什么不是简单H100算力翻倍？** 同属Hopper，表中BF16峰值同量级，主要增加容量和HBM带宽。追问：谁受益？访存/KV/容量瓶颈任务，仍需相同负载测试。

**Q03 [机制] HGX B200与GB200是什么关系？** 一个是多GPU平台中的GPU产品，后者还组合Grace CPU与Blackwell GPU；NVL72再是机架级。追问：CPU内存能当同带宽HBM吗？不能。

**Q04 [推导] sparse峰值为什么不能直接比较dense训练？** 特定稀疏结构与kernel是前提，精度也要一致。追问：H1001979BF16应怎样看？带稀疏脚注，dense约一半，不是所有BF16训练实测。

**Q05 [容量] 70B全参训练D8状态有多少？** 特定16P布局下1120GB总状态、每卡140GB理想分片，其他峰值另加。追问：H200141GB够吗？不能据名义1GB余量保证。

**Q06 [性能] NVLink1800GB/s就是NCCL1800GB/s吗？** 不是，统计方向、实际拓扑和collective不同。追问：400Gb/s网卡多少GB/s？理论50GB/s，实际还需扣协议等。

**Q07 [设计] 有8张80GB卡就有单模型640GB吗？** DP复制下不是；状态/模型分片才改变驻留方式，且要通信。追问：哪些对象能分？区分ZeRO/FSDP、TP、PP、CP。

**Q08 [数值] 有FP8/FP4就能把optimizer全部低位吗？** 算子精度、存储和更新状态是不同对象；按方法/实现核对。追问：换精度如何验收？质量、稳定性与实际kernel路径同时看。

**Q09 [迁移] AMD PyTorch代码里为什么也有torch.cuda？** HIP后端复用接口命名，看torch.version.hip和安装路径。追问：CUDA自定义扩展怎么办？确认HIP/ROCm适配，不能保证无需修改。

**Q10 [迁移] TPU/Gaudi/昇腾只是改device吗？** 编译、算子、通信、分片与保存栈不同。追问：最小验收？单设备数值→多卡→恢复→完整训练角色闭环。

**Q11 [兼容] RTX5090与B200都Blackwell，kernel为什么可能不通用？** 官方CC12.0和10.0不同，所需指令/编译目标并非一项。追问：CC更高必然训练更快？不是性能排名。

**Q12 [排障] 换B卡RL利用率仍低先看什么？** ready请求、工具/环境等待、teacher和learner依赖；更高峰值不消除外部等待。追问：验收指标？同GPU-hour的质量/有效产出，不只busy。

## 12. 来源、复习与本次验证

| 来源 | 用途 |
|---|---|
| [A100](https://www.nvidia.com/en-us/data-center/a100/) / [H100](https://www.nvidia.com/en-us/data-center/h100/) / [H200](https://www.nvidia.com/en-us/data-center/h200/) | 具体形态、显存、带宽与稀疏脚注 |
| [Blackwell PDF](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf) / [Ultra PDF](https://dam-cdn.nvd.orangelogic.com/AssetLink/1k0p832eq8r5ca0u5383ie5o4tp3bst1.pdf) / [GB200](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) | GPU/平台/机架与本文数值口径 |
| [Ampere技术介绍](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) / [Hopper技术介绍](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) / [Blackwell架构](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) | 架构机制与历史GH100原图 |
| [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/index.html) / [CC](https://developer.nvidia.com/cuda/gpus) / [兼容指南](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/) | 精度路径、指令与编译兼容 |
| [AMD MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) / [MI325X](https://www.amd.com/en/partner/articles/instinct-mi325x-accelerators.html) / [MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html) / [ROCm兼容](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) | CDNA产品与迁移版本 |
| [Gaudi训练](https://docs.habana.ai/en/latest/PyTorch/index.html) / [TPU7x](https://docs.cloud.google.com/tpu/docs/tpu7x) / [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) / [CANN](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/index/index.html) | 其他加速器的训练软件入口 |
| [L40S](https://www.nvidia.com/en-us/data-center/l40s/) / [RTX PRO6000](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/) / [RTX5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | GDDR产品与数据中心GPU边界 |
| [MLPerf Training](https://mlcommons.org/benchmarks/training/) | 质量目标与系统性能口径 |

[完整来源清单](./research/infra-ai-accelerators-sources.json)。复习练习：不看表说出每代最关键变化；为70B训练填写状态/激活与互联账本；为一个rollout任务区分算力、KV和工具等待；设计同负载跨设备对照，明确自己的代码/精度/恢复验证边界。

**本次CPU验算：** 70B的16P状态合计1120GB，D8/D16分别140/70GB；H100/H200假想负载的理想时间下界约119.4/101.1ms；1T BF16按80%名义显存均匀分片，80/180/270GB设备的纯权重下界32/14/10。结果未包含真实kernel效率与峰值，也没有执行GPU benchmark。下载：[标准库计算脚本](./assets/infra/infra-accelerator-check.py) · [实际验算结果](./assets/infra/infra-accelerator-check-results.json)。
