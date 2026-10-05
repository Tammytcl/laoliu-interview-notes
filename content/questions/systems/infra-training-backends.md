---
id: infra-training-backends
title: "08 · DeepSpeed、Lightning与FSDP怎样用？从职责、训练循环到保存恢复"
category: systems
difficulty: 进阶
tags: ["P0", "Infra基础", "DeepSpeed", "Lightning", "FSDP", "框架使用"]
updated: 2026-10-05
summary: "区分训练流程框架与分片后端，详细比较原生PyTorch/FSDP2、DeepSpeed engine与Lightning strategy，说明接入、配置、优化器所有权、checkpoint和选型。"
draft: false
---

## 1. 主问题：它们是替代关系，还是可以组合？

本篇延伸 [verl/slime选型](#q=training-inference-frameworks)：现在详细看单个训练角色内部怎样执行。先读 [训练一步](#q=infra-training-step)、[ZeRO机制](#q=infra-zero-deepspeed)、[checkpoint格式与1T容量](#q=infra-checkpoint-formats)。主问题是“已有模型和训练目标，谁组织循环、谁执行分片、怎样接入与恢复？”

核对日2026-10-05。Lightning官网页面在本次抓取中主要返回JavaScript壳，故能力结论回到其固定官方源码与RST教程核查；不把搜索摘要当完整文档。以下GPU配置为教学入口，没有实测吞吐排名。

## 2. 职责层次：Lightning不是另一种ZeRO stage

![本文示意：训练任务、Lightning循环、strategy与计算后端的层次关系](./assets/infra/infra-training-backends/backend-layers.svg)

**图解：** 顶层是模型、数据和目标；中间的训练流程可以自己写，也可交给Lightning；下层用strategy/engine组织DDP、FSDP或DeepSpeed。verl/slime还包含rollout与算法角色，是另一套上层控制流程，不能因Lightning能训练模型就假定它替代完整RL系统。箭头表达接入关系，不表示一个进程里应该同时套全部包装器。

| 对象 | 主要职责 | 使用者首先实现/选择 |
|---|---|---|
| 原生PyTorch循环 | forward、loss、backward、step等基础控制 | 显式组织数据、优化器、分布式与恢复 |
| FSDP/FSDP2 | 模型状态分片与计算前后通信 | 分片粒度、device mesh、精度、状态保存 |
| DeepSpeed | engine与优化能力，如ZeRO/offload | initialize、配置、模型支持、engine循环 |
| Lightning Trainer/LightningModule | 训练/验证循环、hook、日志、callback、策略组织 | training_step、optimizer、数据与Trainer配置 |
| Lightning Fabric | 提供较轻的分布式/精度/保存工具，保留自写循环 | setup、backward与自身控制流 |
| verl/slime | 在线rollout、评分、更新及权重交接 | 算法recipe、各角色后端和任务接口 |

因此可以“Lightning + DeepSpeedStrategy”或“Lightning + FSDPStrategy”；也可以不使用Lightning，直接接原生FSDP或DeepSpeed engine。不能把这些组合写成彼此完全互斥的速度排行榜。

## 3. 原生PyTorch与FSDP：保留训练循环，改变状态生命周期

### FSDP1与FSDP2怎样不同？

FSDP1常见入口是`FullyShardedDataParallel`包装模型/模块，按组管理flat parameter、聚合与重分片。FSDP2入口是`fully_shard`，以per-parameter DTensor等表示分片参数，并通过模块的计算hook组织通信。功能目标相近，API、状态表示、分片组织和checkpoint路径不逐项相同。[FSDP2教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[fully_shard API](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)。

不要看到后端标“fsdp”就认定FSDP2，也不要把FSDP1的flat参数/`StateDictType`配置照搬。保存与optimizer映射常要走当前后端支持的state_dict API/DCP。

### 怎样接入FSDP2？

一个常见思路是：初始化进程组与设备 → 创建模型 → 从重复block向外逐级fully_shard → 创建optimizer → 保持普通forward/backward/step形式。下面只展示机制，模型/data定义须按实际任务补齐：

```python
# 教学片段：torchrun启动，目标PyTorch版本支持FSDP2
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl")
mesh = init_device_mesh("cuda", (dist.get_world_size(),))
model = MyTransformer()          # 省略模型定义；不能直接运行这行
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for batch in loader:             # 省略数据分发；各DP rank应处理不同样本
    loss = model_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

分片会影响参数表示，应在与后端一致的初始化阶段创建optimizer。不要先拿一个旧parameter列表构造optimizer，再任意替换参数包装，假定引用必然正确。

如果只在最外层分片，一次可能聚合大量参数；按Transformer block分组可以缩小聚合峰值并预取。但过细又可能产生许多小collective。FSDP2默认/显式预取的支持和行为以该版本文档为准；预取是重叠机会，也增加在途参数，并非免费提升。

### 使用者仍负责什么？

数据sampler与loss归一化、global batch、累积边界、gradient norm、保存协议与恢复后的数据位置仍需组织；FSDP不会自动把你的算法改正确。混合精度的参数、归约与输出dtype也应按policy核对，不把“bf16”当全部状态同一dtype。

保存sharded训练状态与导出完整部署权重是两条需求：前者可避免某rank聚合整个模型，后者通常需要重建张量或支持转换。1T纯FP32参数就4TB，选择full state_dict前必须估算RAM/磁盘和阶段峰值。

## 4. DeepSpeed：engine包住计算、配置描述策略

### 怎样理解initialize/backward/step？

DeepSpeed接入通常通过`deepspeed.initialize`得到engine，engine承接模型执行、backward和step，配置控制ZeRO、精度、累积与offload。上层trainer也可以替你接入，但有效配置与生命周期仍需核对。[DeepSpeed入门](https://www.deepspeed.ai/getting-started/)。

```python
engine, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json"
)
for batch in loader:
    loss = model_loss(engine, batch)
    engine.backward(loss)
    engine.step()
```

若optimizer由配置创建，就不要再假定外部另一个optimizer应手动step；若传入自定义optimizer，要检查该ZeRO/offload路径是否支持。`engine.step()`可按microbatch调用，实际更新由累积边界决定，详见 [ZeRO配置例子](#q=infra-zero-deepspeed)。

### 要重点看哪些配置？

| 组 | 核对问题 |
|---|---|
| batch/accumulation | 全局、每DP副本、token目标分别是什么 |
| precision | FP16/BF16/其他，scaling与累积/状态dtype如何处理 |
| zero_optimization | stage、bucket、预取、持久参数与offload |
| optimizer/scheduler | 谁创建、何时初始化、clip与step由谁执行 |
| checkpoint/export | 所有rank参与、恢复布局、16-bit导出/FP32重建 |

DeepSpeed的范围不止ZeRO，还可能涉及推理、pipeline、MoE等能力；本文深入的是训练engine与状态优化路径。判断是否支持目标模型与组合，仍需查看具体代码/recipe，不能用能力列表代替接入验证。[配置文档](https://www.deepspeed.ai/docs/config-json/)。

CPU offload不是“把所有PyTorch Adam自然搬到CPU就完成”；CPU optimizer实现、build环境、RAM/NUMA与传输需联调。NVMe路径又多一层IO。优先先证明无offload小闭环正确，再加容量策略。

## 5. Lightning：把训练组织交给Trainer，后端通过strategy选择

### LightningModule、Trainer、strategy各管什么？

`LightningModule`组织模型及任务hook，例如`training_step`返回loss、`configure_optimizers`给出optimizer/scheduler。`Trainer`调度训练、验证、日志、callback与保存。`strategy`处理分布式执行、设备与对应后端交接。[Lightning固定官方代码树](https://github.com/Lightning-AI/pytorch-lightning/tree/84df182f50ab34301aabb3c0eb4031815bfb413d)。

```python
# 教学骨架：模型/数据由任务提供；本轮未运行Lightning GPU训练
import lightning as L
import torch

class LitTask(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        return task_loss(self.model, batch)   # 实际任务实现

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

trainer = L.Trainer(accelerator="gpu", devices=8, strategy="ddp",
                    precision="bf16-mixed", accumulate_grad_batches=4)
trainer.fit(LitTask(model), train_dataloaders=loader)
```

默认automatic optimization时，Trainer/strategy组织backward/step；不要在training_step里再做原生step，然后仍假定它只更新一次。复杂多optimizer或RL控制流可以采用manual optimization，但责任也回到使用者，包括累积、clip、scheduler等，需要按文档/代码核对。

Lightning Fabric更偏“我保留自写循环，但希望用工具完成setup/backward/保存”，不需要所有任务都迁移为Trainer循环。它也不是FSDP或ZeRO的一种stage。[Fabric固定代码入口](https://github.com/Lightning-AI/pytorch-lightning/tree/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/fabric)。

### Lightning使用DeepSpeed是什么组合？

```python
from lightning.pytorch.strategies import DeepSpeedStrategy
trainer = L.Trainer(
    accelerator="gpu", devices=8, precision="bf16-mixed",
    strategy=DeepSpeedStrategy(stage=2),
    accumulate_grad_batches=4
)
```

这里外层训练流程是Lightning，计算/状态路径由DeepSpeed strategy接入。不能同时再手动给同一模型套一个独立DeepSpeed engine，并让两个层级各做step。offload等配置从支持的strategy参数或完整配置进入，打印最终结果。[Lightning固定DeepSpeed教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/advanced/model_parallel/deepspeed.rst)。

### Lightning的FSDPStrategy是否就是FSDP2？

本次固定源码中的`FSDPStrategy`调用`torch.distributed.fsdp.FullyShardedDataParallel`，对应FSDP1路径；它不是仅凭类名就能当作FSDP2。[固定strategy源码](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/strategies/fsdp.py)。

```python
from lightning.pytorch.strategies import FSDPStrategy
strategy = FSDPStrategy(
    auto_wrap_policy={TransformerBlock},       # 实际模型的block类型
    activation_checkpointing_policy={TransformerBlock},
    state_dict_type="sharded"
)
trainer = L.Trainer(accelerator="gpu", devices=8,
                    strategy=strategy, precision="bf16-mixed")
```

FSDP2/2D并行另有`ModelParallelStrategy`配合`configure_model`内`fully_shard`与TP计划的官方示例。这需要声明mesh与模型切分，不是把FSDPStrategy名字改一下。该固定示例还使用过composable入口，迁移到当前PyTorch时应核对import与API兼容。[Lightning固定TP/FSDP教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/advanced/model_parallel/tp_fsdp.rst)。

## 6. 同一个任务，换后端要改什么？

假设已有HF/PyTorch模型与SFT目标，首先固定tokenizer/template、loss mask、global batch、精度与评测；再换实现路径。不要同时换模型定义与训练语义，又把差异都归因于框架。

| 维度 | 原生FSDP2 | DeepSpeed engine | Lightning + strategy |
|---|---|---|---|
| 模型接入 | 模块逐级fully_shard与mesh | initialize和配置兼容 | LightningModule + strategy |
| 循环控制 | 自己组织 | 自己/上层循环调用engine | Trainer；manual模式另明确责任 |
| backward/step | 原生调用配分片hook | engine调用与累积 | automatic交给Trainer/strategy |
| 精度配置 | policy/autocast等 | engine配置与路径 | Trainer precision与strategy |
| batch归一化 | 使用者/算法负责 | 使用者+engine语义需确认 | Trainer累积也不消除token归一化责任 |
| 保存/恢复 | DCP/支持的state_dict路径 | engine checkpoint与转换 | 对应strategy的checkpoint路径 |
| 部署导出 | 重建/导出逻辑张量 | 16-bit导出或FP32重建 | 根据策略转换，不由.ckpt后缀保证 |

### 选择时的约束顺序

1. **已有代码栈**：团队是否已有验证过的Lightning、DeepSpeed或PyTorch训练循环？
2. **模型兼容**：冻结/tied参数、自定义层、动态执行路径、LoRA/量化是否受所选路径支持？
3. **容量来源**：是optimizer、完整参数、长序列激活，还是rollout共置？对症选分片/重算/offload。
4. **并行与拓扑**：DP分片组、TP/PP/CP、节点内/跨机流量分别怎样组织？
5. **保存与交接**：恢复卡数变化、checkpoint导出、rollout权重同步怎样验收？
6. **维护和观测**：版本锁定、故障定位、日志/trace与升级成本是否可承受？

由此可以推断：已有高度自定义训练流程且希望直接控制分片，先评估原生FSDP2；已有DeepSpeed栈或offload需求，先评估engine；已有标准训练/验证/日志流程，希望统一组织，可评估Lightning并选择后端。它们是评估方向，具体性能仍需同任务测量。

## 7. 保存恢复与1T模型：上层封装不能消除物理容量

Lightning完整checkpoint可保存optimizer、scheduler、loops、callback、hyperparameters等；只通过`load_from_checkpoint`构造模型，不等于让Trainer继续原step。恢复训练用支持的`trainer.fit(..., ckpt_path=...)`路径，并确认strategy与源产物匹配。[固定checkpoint教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/common/checkpointing_basic.rst)。

FSDP分片checkpoint、DeepSpeed checkpoint、普通Lightning单文件或HF部署分卷，都有各自加载协议。不要将任意rank文件交给普通`torch.load`后，因没抛异常就宣称恢复了完整模型。[DCP恢复与重分片](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)、[DeepSpeed保存](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)。

任何上层框架都不会使1T BF16纯权重的约2TB消失；ZeRO改变每rank状态分布，offload改变存储层，量化改变表示。推理峰值还加KV等。使用框架前先填 [容量与格式账本](#q=infra-checkpoint-formats)，避免把“策略支持stage3”误当“任意硬件能跑”。

## 8. 回到verl/slime：为什么不直接用Lightning替代？

verl/slime还要组织在线采样、reward/teacher、优势或蒸馏loss、learner更新、rollout权重发布和策略版本。Lightning的训练循环可以构建/承接一部分任务，但不会仅凭Trainer启动就自动成为同等的RL orchestration。

同理，选择了verl仍需问actor worker用FSDP还是Megatron，选择slime仍需理解其训练栈。若真的要替换某角色后端，需要满足样本契约、概率/版本、checkpoint、资源放置和更新语义。见 [RL/OPD闭环](#q=infra-rl-pipeline)，不是简单替换一个import。

## 9. 面试问答：机制、接入与恢复都要说清楚

题目为本文整理，依据官方机制/固定源码，不声称任何公司真题。

**Q01 [基础] DeepSpeed、Lightning、FSDP为什么不是同一列替代品？** 分别强调engine/优化、训练流程、分片执行，可组合。追问：Lightning+DeepSpeed谁step？automatic模式由Trainer/strategy连接engine，不能双重更新。

**Q02 [机制] FSDP1/FSDP2区别在哪里？** 包装/flat参数与fully_shard/per-parameter DTensor等路径不同，不能照抄状态保存和policy。追问：怎么判断Lightning用了哪个？读实际strategy源码与版本。

**Q03 [使用] 接FSDP2最关键的顺序？** 进程/mesh、模型、逐级分片、optimizer、循环。追问：为什么不能任意替换参数后保留旧optimizer？参数引用与状态关联可能不匹配。

**Q04 [使用] DeepSpeed每microbatch都step是不是错？** engine按累积边界决定真实更新；调用次数与更新次数不同。追问：loss是否还手工除累积次数？看engine/上层语义，避免重复缩放。

**Q05 [设计] Lightning为什么适合标准SFT流程？** 可统一训练/验证hook、日志/callback/策略；不是自动优化所有kernel。追问：复杂在线RL呢？需另组织rollout与依赖，或明确manual控制责任。

**Q06 [排障] 加了strategy后OOM为何仍存在？** 分片没消除激活/聚合层/预取/图池。追问：首先做什么？各阶段峰值与实际对象账本，确认最终有效配置。

**Q07 [恢复] load_from_checkpoint后step为何从0开始？** 模型加载与Trainer恢复不同；用匹配的fit ckpt_path并检查optimizer/loop。追问：变卡数呢？需要支持的重分片与其他状态映射。

**Q08 [设计] 什么情况下优先自写PyTorch/FSDP循环？** 需要深入控制算法/分片或已有成熟自写栈时先评估，接受更多维护责任。追问：不使用Lightning就一定更快？封装层不是充分性能证据。

**Q09 [评审] 怎样证明换框架正确？** 固定同任务、精度、batch与数据；对比小规模梯度/更新/恢复，再比峰值、吞吐和质量。追问：配置名字一致就目标一致吗？动态batch和归一化可能不同。

**Q10 [容量] 框架支持1T模型是否意味着80GB卡够？** 支持说明机制/模型路径，容量仍按精度和放置计算。追问：offload如何解释？移到CPU/NVMe并付传输，不是消掉数据。

## 10. 来源与使用练习

| 来源 | 类型 | 阅读用途 |
|---|---|---|
| [FSDP2教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) / [API](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html) | 官方教程/API | 分片、mesh与初始化 |
| [DeepSpeed入门](https://www.deepspeed.ai/getting-started/) / [配置](https://www.deepspeed.ai/docs/config-json/) | 官方教程 | engine与控制责任 |
| [Lightning固定代码树](https://github.com/Lightning-AI/pytorch-lightning/tree/84df182f50ab34301aabb3c0eb4031815bfb413d) / [Fabric](https://github.com/Lightning-AI/pytorch-lightning/tree/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/fabric) | 官方代码 | Trainer/Fabric组织层次 |
| [DeepSpeed strategy教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/advanced/model_parallel/deepspeed.rst) | 固定官方RST | 组合路径 |
| [FSDPStrategy源码](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/src/lightning/pytorch/strategies/fsdp.py) / [TP/FSDP教程](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/advanced/model_parallel/tp_fsdp.rst) | 固定官方代码/教程 | FSDP1/2边界与model hook |
| [Lightning checkpoint](https://github.com/Lightning-AI/pytorch-lightning/blob/84df182f50ab34301aabb3c0eb4031815bfb413d/docs/source-pytorch/common/checkpointing_basic.rst) / [DCP](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html) / [DeepSpeed](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) | 官方文档 | 模型加载与训练恢复 |

[来源清单](./research/infra-training-backends-sources.json)。练习：为同一个小任务填写三条接入路径的责任表、有效batch/精度配置和保存协议；先跑数值/恢复对照再测GPU性能。本轮完成资料与接口核查，没有跑三框架GPU对比。
