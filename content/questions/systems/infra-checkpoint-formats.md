---
id: infra-checkpoint-formats
title: "06 · 权重怎样保存与分片？ckpt、safetensors、恢复训练与 1T 容量计算"
category: systems
difficulty: 基础
tags: ["P0", "Infra基础", "Checkpoint", "safetensors", "模型容量", "存储"]
updated: 2026-10-05
summary: "区分文件格式、逻辑状态、HF文件分卷与分布式切分；推导参数量到磁盘/显存的关系，给出1T不同精度算例、转换流程和恢复验收。"
draft: false
---

## 1. 主问题：一个权重文件到底包含什么？

看到`model.ckpt`、`pytorch_model.bin`、`model.safetensors`或几十个rank文件，先问四件事：**里面是什么状态、怎样序列化、怎样分片、要用哪个加载器**。后缀只提示惯例，不能独自回答这四问。本文也回答“1T模型占多大磁盘、推理要多大显存”，所有容量都会标精度和保存对象。

先修 [训练状态与显存](#q=infra-training-step)、[并行切分](#q=infra-distributed-parallelism)。后续接 [ZeRO详解](#q=infra-zero-deepspeed)和 [DeepSpeed/Lightning/FSDP](#q=infra-training-backends)。核对日2026-10-05；本文附CPU小模型实验与数值脚本，未创建真实1T模型，未做GPU大模型部署。

## 2. 四个维度：后缀、内容、编码与分片不是同一件事

![本文示意：逻辑状态、序列化格式、文件分卷与rank切分四层](./assets/infra/infra-checkpoint-formats/checkpoint-layers.svg)

**图解：** 第一层区分模型张量与训练恢复状态；第二层描述如何编码；第三层区分普通文件分卷与分布式张量切分；第四层是匹配的加载路径。比如“训练时用了ZeRO-3，最后导出HF safetensors”同时涉及不同层，不存在冲突。图中箭头是可选关系，不表示每种任意组合均受框架支持。

| 看到的名称 | 能推断的线索 | 不能直接推断 |
|---|---|---|
| `.pt` / `.pth` / `.bin` | 常用于PyTorch保存 | 一定只有权重、一定非分片 |
| `.ckpt` | 常用于某个训练系统的checkpoint | 存在统一ckpt二进制规范 |
| `.safetensors` | 通常按该格式存张量与字符串metadata | 自动包含训练循环、optimizer与tokenizer |
| `model-00001-of-00008.safetensors` | 有文件分卷 | 每文件对应一个TP rank |
| `.distcp` + `.metadata` | 常见DCP文件系统后端产物 | 可以直接`from_pretrained` |
| `mp_rank...` / `zero...optim_states.pt` | 分布式训练保存布局线索 | 文件数就是DP数、每个单独可恢复 |

文件目录示例只表达常见布局。具体命名随版本和checkpoint后端变化，要核对索引/metadata和保存代码。

### Q：model.state_dict与“完整模型”有什么不同？

`state_dict`通常是参数与persistent buffers的名字→值映射；buffer可能是运行统计，不一定是可训练参数。它通常不包含Python模型定义、forward逻辑或完整的架构配置。加载时要先创建匹配模型，检查keys、shape和dtype，再`load_state_dict`。

完整训练checkpoint可以是字典，内含model state、optimizer、scheduler、step、随机数及数据游标等。保存整个`nn.Module`对象则依赖类路径与Python对象反序列化，并不是通用部署格式。PyTorch现在常见`torch.save`使用ZIP容器，张量storage与对象描述分开保存；旧版序列化也存在。[PyTorch serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html)。

PyTorch 2.6起，在未显式传入pickle_module的条件下，`torch.load`默认采用`weights_only=True`；这里的名字指**限制反序列化对象类型**，不是“从optimizer文件中自动提取模型权重”。自己的格式检查示例仍显式写出该参数，避免依赖默认值。[官方load说明](https://docs.pytorch.org/docs/2.14/generated/torch.load.html)。

## 3. safetensors：张量容器、header与共享权重

safetensors是张量序列化格式，不是一个模型架构。固定源码说明其结构：先8字节小端无符号整数N，再N字节UTF-8 JSON header，之后是数据区。每个张量记录dtype、shape、`data_offsets`，偏移相对数据区起点；结束偏移不包含在区间内。header可有空格padding。[固定格式说明](https://github.com/safetensors/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/README.md)。

```text
8-byte header length | JSON header (+ padding) | tensor byte buffer
weight: {dtype: F32, shape: [2,3], data_offsets: [0,24]}
```

这张2×3 FP32张量的数据区24字节，文件仍要加header。`__metadata__`用于字符串metadata，不是把任意Python类或optimizer对象直接pickle进去。理论上可自行编码optimizer张量，但“写成safetensors”并不自动提供optimizer参数关联、训练游标或恢复协议。

格式便于按header定位张量、按需读取和内存映射；mmap不等于磁盘读取时间为0，更不等于GPU无需拷贝。加载还受page fault、存储带宽、CPU内存、dtype转换与设备传输影响。[Safetensors API](https://huggingface.co/docs/safetensors/api/torch)。

### Q：为什么tied embedding导出可能报共享storage错误？

语言模型的embedding和输出头可能共用同一权重。`state_dict`里能出现两个名字，但底层storage只有一份。逐key的`numel`相加可能重复计算；转换时对每项clone也可能把共享权重复制两份，增加磁盘，并丢失alias关系。

Safetensors的普通`save_file`不能按PyTorch任意共享view的语义保存；可使用官方`save_model/load_model`或模型库的`save_pretrained`路径，由加载器恢复相应共享关系。某些名字会被省略，不能将这种预期省略与漏保存混为一谈。[共享张量说明](https://huggingface.co/docs/safetensors/torch_shared_tensors)。

还有反方向的坑：PyTorch保存一个小view时，可能保存其更大的底层storage，文件远大于view的`numel×bytes`。因此模型参数量、state_dict逻辑元素数、唯一storage字节、实际文件大小应分开统计。

## 4. 分片格式：HF文件分卷、ZeRO分区、TP切片与DCP

### Q：HF的8个safetensors文件是否就能分给8张GPU？

HF常见分卷按文件大小把完整张量分配到不同文件，用`model.safetensors.index.json`中的`weight_map`记录“张量名→文件名”。这通常是**完整张量的文件分组**，不是按TP把一个矩阵切成8块；加载器再按目标设备/布局放置张量。一个张量大于文件阈值时，也不能假定它被任意切断到多个文件。[HF大模型加载教程](https://huggingface.co/docs/transformers/main/big_models)。

```json
{
  "metadata": {"total_size": 32},
  "weight_map": {
    "linear.weight": "model-00001-of-00002.safetensors",
    "linear.bias": "model-00002-of-00002.safetensors"
  }
}
```

这里total_size是教学小模型的**逻辑张量字节**，不是两个文件的总文件长度，因为文件还各有header。

### Q：ZeRO与TP分片怎样不同？

ZeRO在DP组内分模型状态，例如optimizer与参数的flat partitions，需要形状、分区顺序和padding信息重建。TP按模型算子的维度切张量；PP按层/模块分配状态；EP涉及expert归属。一套checkpoint可叠加这些切分，不能简单把所有rank文件concat后宣布恢复。

比如一个4×4矩阵，TP按列切两块：每rank本地是4×2；若按文件字节顺序拼成8×2就错了。ZeRO flat partition可能还跨越多个参数边界，拼完也要按原形状切回。完整张量语义与存储排列须由metadata/框架解释。

### Q：DCP为何可支持保存2卡、恢复1卡？

PyTorch Distributed Checkpoint登记逻辑张量及分块metadata，加载规划器可以将保存时的块映射到当前目标张量布局。通常先按目标并行方案创建模型/状态，再用DCP加载；模型与optimizer关联可借助`get_state_dict/set_state_dict`等API组织。[DCP教程](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)、[DCP API](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)。

“支持重分片”不表示任意PyTorch版本、任意旧checkpoint都兼容；要核对planner、状态key、shape、parallelism和版本。weights重新分布后，optimizer、数据游标及rank随机数也要按新拓扑正确解释。本篇CPU实验只验证一个DTensor从2个Gloo rank保存、1个rank加载的张量重分片，不将其推广为所有训练状态都已验证。

## 5. 参数量与磁盘：1T模型怎么算？

先定义 **1T参数=$10^{12}$个参数**；T不是一TB文件。未压缩、统一dtype、无重复存储的纯权重数据近似：

$$
M_{weights}=P\times b=P\times\frac{q}{8}
$$

P是唯一存储参数量，b是每参数字节，q是位数。混合dtype时逐类求和 $\sum_kP_kb_k$；文件总大小还加buffers、metadata、padding、重复storage等。

| 1T参数的存储精度 | 每参数理论字节 | 纯数据十进制容量 | 二进制容量约 |
|---|---:|---:|---:|
| FP32 | 4 | 4 TB | 3.638 TiB |
| BF16 / FP16 | 2 | 2 TB | 1.819 TiB |
| INT8 / FP8编码 | 1 | 1 TB | 0.909 TiB |
| packed 4-bit | 0.5 | 0.5 TB，即500 GB | 0.455 TiB |

$1\,TB=10^{12}$字节，$1\,TiB=2^{40}$字节；2TB约1862.65 GiB。FP8/INT8只是此行的编码字节假设，实际格式可能需要scales、不同层精度及转换；不意味着把BF16文件改后缀就完成量化。

### Q：4-bit为什么经常大于P/2字节？

分组量化可能每g个参数附scale/zero point。**教学假设**：所有权重都packed INT4，g=128，每组scale2字节、zero2字节，则：

$$
b_{effective}=0.5+\frac{2+2}{128}=0.53125
$$

1T对应531.25 GB，而不是500 GB。这是示例，具体量化可能只存scale、压缩zeros、采用多级量化或保留部分BF16模块，还要考虑每个张量末尾的packing对齐。[量化概念教程](https://huggingface.co/docs/transformers/main/quantization/concept_guide)。

文件分卷本身不减少这些总字节；state分片减少每rank的一部分，不自动减少系统总数据。压缩文件、文件系统实际块分配、稀疏文件与逻辑文件长度也有不同统计口径，报告时注明是在量哪一种。

### Q：只知道磁盘大小，能倒推参数量吗？

可以粗估，但要先确认权重占比与dtype：2TB纯BF16权重对应约1T；2TB训练checkpoint可能混有FP32 optimizer和状态，不能这样反推。MoE包含很多expert，若“1T总参数、100B激活参数”，常规完整驻留仍需存全部1T权重；激活参数描述每token经过多少参数，不是文件只需存100B。共享/重复、量化metadata也都会影响估计。

## 6. 训练checkpoint磁盘：为何和训练显存账本不同？

训练显存会有梯度和激活；标准更新边界保存通常不保存整套激活，也往往不保存梯度。沿用第一篇的教学布局，若保存BF16参数2P、FP32 master4P、Adam两份状态8P，逻辑持久数据约 **14P字节**；1T约14TB。若只保存FP32参数4P与FP32两份Adam状态8P，约12TB。二者均不是所有框架默认值。

ZeRO-3 checkpoint可能依靠optimizer中的FP32分区重建权重，而不另存完整BF16副本；冻结参数、buffer、重复rank文件或其他状态也可能改变大小。必须列出实际文件内容，而不是把内存16P直接当磁盘16P。[DeepSpeed checkpoint文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。

| 要保留的产物 | 容量的计算口径 |
|---|---|
| 部署权重 | 纯张量+header/config/tokenizer等 |
| 一个恢复checkpoint | 实际保存的model/optimizer/loop/data状态 |
| 多个历史checkpoint | 各份实际字节之和，不能假定自动去重 |
| 转换临时产物 | 源目录+目标导出+中间文件可能同时存在 |
| 保存中的快照 | staging、临时文件及最终提交的瞬间峰值 |

例如保留3份12TB训练状态，再保留2TB BF16导出，静态已38TB；还没算转换和正在写入的临时量。若有效写带宽10GB/s，2TB至少约200秒、12TB约1200秒；这是理想字节下界，不含metadata、通信、CPU staging、共享存储竞争和文件提交。

## 7. 1T推理显存：权重是下界，KV不能从参数量直接推得

完整GPU驻留推理的容量框架是：

$$
M_{inference,peak}=M_{weights}+M_{KV}+M_{workspace}+M_{activations}+M_{runtime}
$$

推理不必加载训练的Adam状态与梯度，因此不能套16P；但2TB BF16权重也不是完整峰值。KV要从模型结构、batch、上下文和KV dtype单独算：

$$
M_{KV}=2LB S H_{KV}D_hb_{KV}
$$

**独立教学配置**：L128、B8、S32768、KV heads8、head dim128、BF16 KV，得到128 GiB，约137.44 GB。这个配置只用于演示KV，不能由“1T”反推出它；同参数量可有不同层数、heads、GQA/MLA或滑窗设计。[KV与推理预算详解](#q=infra-inference-engine)。

假设某1T BF16模型恰采用此KV配置，则纯权重+KV已约2.137TB，另需workspace/激活/图池。量化了权重但没量化KV时，这128GiB也不会随权重位数自动缩小。

### Q：要多少张80GB卡？

下面**明确按每卡80GB十进制**做均匀分片的权重容量下界，不是硬件采购建议；真实设备应以可用byte数计算。设完全用于权重，或只允许80%用于权重：

| 1T精度 | 理论纯权重 | 每卡80GB全部用于权重 | 每卡64GB用于权重 |
|---|---:|---:|---:|
| FP32 | 4TB | 至少50卡 | 至少63卡 |
| BF16 / FP16 | 2TB | 至少25卡 | 至少32卡 |
| 8-bit编码 | 1TB | 至少13卡 | 至少16卡 |
| 理想packed 4-bit | 500GB | 至少7卡 | 至少8卡 |

计算是 $\lceil M_{weights}/M_{usable,perGPU}\rceil$。这里64GB并非实际KV/临时内存预算，只是20%预留的假设。对2.137TB权重+上述KV，若每卡可用64GB，两项均匀分片的容量下界变成34卡；还没加入其他项，也没有验证层/heads可切性。

若80是GiB而非GB，结果会变化：2TB/80GiB约23.28，需要至少24卡存纯权重。TP/PP布局、某层峰值、KV复制、通信与负载不均还会增加限制。DP复制模型不能把2TB除DP；需要模型切分或其他驻留方案。

CPU/NVMe offload可以降低GPU常驻权重量，但权重仍占主机/磁盘，并需传输；能加载不等于满足decode延迟。ZeRO-Inference是具体研究/实现路线，不能把训练ZeRO三个stage直接当成通用serving策略。[ZeRO-Inference官方介绍](https://www.deepspeed.ai/2022/09/09/zero-inference.html)。

## 8. 转换与加载：先恢复逻辑权重，再改容器

推荐顺序：确认源框架/version与分片metadata → 重建逻辑张量或使用支持的重分片 → 映射模型keys/shape → 选择dtype与共享权重策略 → 导出部署目录 → 数值与生成验收。

```python
# 自己创建的普通PyTorch模型权重；非ZeRO/TP分区转换器
import torch
from safetensors.torch import save_file
state = torch.load("weights.pt", map_location="cpu", weights_only=True)
assert isinstance(state, dict)
assert all(isinstance(v, torch.Tensor) for v in state.values())
# 限定无共享storage；不能对真实tied模型盲目这样保存
save_file({k: v.contiguous() for k, v in state.items()}, "weights.safetensors")
```

若源是Lightning完整checkpoint，模型权重通常位于`state_dict`键；`model.`等前缀要按目标module映射，不能无条件删掉。恢复训练用`trainer.fit(..., ckpt_path=...)`，只构造模型权重的加载接口不等于恢复整个Trainer。[Lightning固定checkpoint教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/common/checkpointing_basic.rst)。

ZeRO checkpoint应走官方`get_fp32_state_dict_from_zero_checkpoint`或对应离线转换工具，再处理模型适配与目标精度。1T FP32导出仅张量就4TB，CPU聚合/clone可能出现更大RAM峰值；支持lazy/sharded输出也要检查其实际实现，不把“输出分卷”当“整个转换过程恒定低内存”。[DeepSpeed转换文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。

HF部署目录通常还需要架构config、tokenizer、模板或adapter/量化配置。正确导出目录不等于“几个safetensors文件改名”。对verl/slime，checkpoint导出与rollout在线权重同步也是不同路径：后者还涉及训练/生成布局和版本。

## 9. 恢复与导出验收：到底证明了什么？

部署验收：张量keys/shape/dtype一致，转换前后固定输入的输出或logprob一致到合理容差，必要时检查tied关系与真实采样。精度转换/量化应另外测任务质量，不能要求所有bit完全相同。

恢复验收：相同模型、optimizer、数据、RNG和调度，在保存点后继续一步，与不中断基线比较参数、optimizer统计、学习率和step。只看load没报错，不能证明从相同优化位置继续。

本篇CPU脚本会实际执行PyTorch/safetensors round trip、读取header/文件分卷索引、比较恢复Adam之后的下一步，并使用Gloo验证DTensor的2→1重分片。它不会验证Lightning或DeepSpeed多GPU恢复，也没有真实1T权重。

**本次CPU实测（PyTorch 2.11.0+cu129、safetensors 0.7.0；全程CPU）：** 8参数FP32小模型的张量数据32字节，safetensors文件160字节（8字节长度+120字节header+32字节数据），普通权重pt文件1893字节，含Adam/RNG等状态的教学ckpt文件9048字节。这里小文件metadata占比大，不能将大小倍数推广到1T模型。权重往返一致；恢复后下一步参数最大误差0、optimizer统计一致；DCP两rank→一rank恢复4×4张量最大误差0。

下载并运行：[CPU格式/恢复/重分片实验](./assets/infra/infra-checkpoint-check.py) · [本次结果](./assets/infra/infra-checkpoint-check-results.json)。脚本使用临时小文件并清理，容量算式不实际分配1T张量。

## 10. 面试问答：格式、分片、容量与恢复

均为本文按官方机制整理的练习题，不标成公司真题。

**Q01 [基础] ckpt和safetensors的区别？** ckpt通常是保存产物后缀/约定，safetensors是具体张量容器。追问：safetensors能恢复训练吗？要看是否另外编码全部状态和协议，容器不自动替你做。

**Q02 [机制] state_dict包含模型结构吗？** 通常含参数和persistent buffer，不自动含forward定义。追问：为什么load shape不匹配？架构、词表、层结构或模型映射不同。

**Q03 [机制] HF8个文件就是TP8吗？** 通常只是完整张量分卷与index映射，TP还要切矩阵和通信。追问：怎样区分？看weight_map指向完整tensor，还是带slice/offset与rank布局的metadata。

**Q04 [推导] 1T BF16权重多大？** $10^{12}\times2=2TB\approx1.819TiB$；纯权重，不含KV/optimizer。追问：为什么文件更大？metadata、buffer、重复、dtype或保存内容不同。

**Q05 [推导] 1T INT4是否就是500GB？** 这是packed纯数据下界；本篇group128、scale+zero4字节例子是531.25GB。追问：哪些层保留高精度？按实际量化格式逐层计。

**Q06 [设计] 1T BF16推理25张80GB够不够？** 25卡只是总纯权重除容量的下界，未容纳KV/峰值及切分约束。追问：KV怎样算？用实际L/B/S/KV heads/dim/dtype，不用P倒推。

**Q07 [设计] MoE激活100B就只存100B吗？** 全expert驻留时仍存全部权重；激活量影响每token执行。追问：offload少存能保证低延迟吗？还要看选expert、传输与缓存命中。

**Q08 [排障] 改后缀为何不能转换模型？** 字节编码、逻辑张量和分片metadata都没变化。追问：正确转换第一步是什么？匹配源加载器恢复张量语义。

**Q09 [排障] load成功为什么继续训练不同？** 可能缺optimizer/scheduler/RNG/数据游标，或更新边界不同。追问：如何证明？相同下一批做恢复后一步对照。

**Q10 [排障] 张量numel很少，pt文件为什么很大？** view可能保留大storage，或者ckpt包含状态/重复权重。追问：怎么查？逐键shape/dtype、唯一storage与实际文件字节分别统计。

## 11. 来源与练习

| 来源 | 类型 | 用途 |
|---|---|---|
| [PyTorch序列化](https://docs.pytorch.org/docs/2.14/notes/serialization.html) / [torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html) | 官方API，2.14 | 容器、view与加载限制 |
| [Safetensors固定README](https://github.com/safetensors/safetensors/blob/e246a2560645b7525f5775669ed816eb57c5bcc8/README.md) / [API](https://huggingface.co/docs/safetensors/api/torch) / [共享张量](https://huggingface.co/docs/safetensors/torch_shared_tensors) | 固定源码/官方 | header、数据偏移、alias |
| [HF大模型](https://huggingface.co/docs/transformers/main/big_models) / [量化概念](https://huggingface.co/docs/transformers/main/quantization/concept_guide) | 官方教程 | 文件分卷与量化额外量 |
| [DCP教程](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) / [API](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) | 官方教程/API | 保存与目标布局重分片 |
| [DeepSpeed checkpoint](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) | 官方文档 | ZeRO保存/转换 |
| [Lightning固定教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/common/checkpointing_basic.rst) | 官方源码文档 | 完整状态与Trainer恢复 |
| [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html) | 官方研究介绍 | offload推理的对象与代价 |

[来源清单](./research/infra-checkpoint-formats-sources.json)。练习：先按真实dtype列1T容量，再把三份checkpoint与导出/临时量加入磁盘预算；运行小模型转换与恢复检查，最后为目标大模型制定分片和KV账本。
