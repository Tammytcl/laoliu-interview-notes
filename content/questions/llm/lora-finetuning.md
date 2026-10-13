---
id: lora-finetuning
title: "LoRA 到底省了什么？为什么用了 LoRA 仍会 OOM？"
category: llm
difficulty: 进阶
tags: ["P0", "LoRA", "微调"]
updated: 2026-10-13
summary: "合并LoRA/QLoRA清单，补初始化梯度、量化、合并、多adapter、手写层与16道问答；训练状态和激活成本分别算。"
draft: false
---

## 问题背景：LoRA省了什么

LoRA 降低微调的可训练参数量，但没有让基础模型消失。

2026-10-13合并[外部清单](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)41–50及149，继续使用本页原入口。下文从基础分解进入实现、QLoRA和部署，不把清单中示例配置当你的项目实际参数。

## 前置知识：冻结与训练状态

Frozen parameter不进入optimizer更新，不意味着所有前向/反向中间量都可删除。Adapter依赖底座产生的hidden时，相关链式求导仍需计算；权重、gradient、Adam状态、master weights与activation分别记账。Rank约束的是增量，不是底座矩阵。

## 低秩更新

若 W 的形状为 d_out×d_in：

```text
W_eff = W + (alpha/r)*B*A
A: r×d_in，B: d_out×r
可训练参数量：r*(d_in+d_out)
原矩阵参数量：d_out*d_in
```

冻结 W，只训练 A/B。对 4096×4096 矩阵，r=16 时增加 131072 个参数，而原矩阵有 16777216 个参数。具体模型通常在多个投影上添加 adapter，因此要按实际 target modules 加总。

标准 LoRA 常把其中一侧初始化为零，使初始增量为零；这时最开始两侧梯度行为并不完全对称。初始化方式有变体，以实现为准。

## 省与不省

训练梯度和优化器状态主要随可训练参数减少。基础权重仍需存储，模型前向和对 adapter 的反向还需中间激活；长序列、大 batch、attention 中间量仍可能导致 OOM。基础权重量化是另一项技术，不能把“LoRA”自动等同于“量化 LoRA”。

合并 adapter 可消除额外分支，但量化底座、多个动态 adapter 和精度要求会改变是否适合合并。推理速度也不只由可训练参数数目决定。

## 怎么选择 rank 与目标层？

按任务做小规模对照：固定数据划分和训练预算，改变 rank、目标层与学习率；看验证集泛化，不只看训练 loss。更高 rank 增加更新容量，也可能更容易过拟合。

## 小实验与自检

对小线性层冻结 W，只训练 A/B，检查 W 的 grad；对比合并前后同一输入输出差异。训练时分别记录权重、优化器状态和峰值显存，不把一个总数解释成所有来源。

自检：冻结底座后为什么仍需反向经过底座算子？低秩更新是否表示原始 W 本身必须低秩？

## 面试表达

先写增量分解与参数公式，再明确节省的是训练状态的一部分；OOM 要回到 显存账本。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [LoRA](https://arxiv.org/abs/2106.09685)：看冻结底座、低秩增量与合并推理的设定。

## 初始化、梯度与Scaling的深入推导

写$y=W_0x+sBAx$，常见原始$s=\alpha/r$。令$g=\partial L/\partial y$，则

$$
\frac{\partial L}{\partial B}=s\,g(Ax)^\top,\qquad
\frac{\partial L}{\partial A}=s\,B^\top gx^\top.
$$

B初始化0、A随机时首步A gradient0、B一般非零；后续B更新后A可学。A/B都零则可能失去有效信号，不能把“任意一边置零”理解成“两边全零无妨”。这与Critic零head冷启动有相似梯度结构，但不是同一训练目标。

4096×4096、r16新增131072，占原参数0.78125%，不是所有模型都这个比例。多个target modules、可训bias/head/embedding要加总。原W可full-rank，LoRA只把更新限制在当前低秩参数化。Scaling改变输出与梯度；增rank时还应调α/LR，不据单个变量猜泛化。

其他变体可使用rank-stabilized等缩放，原始α/r不是所有库默认。先核config与实现。Attention/FFN target选择需按真实层名、fused projection和任务消融，不照抄另一架构的字符串后声称已经训练所有QKV。

## QLoRA：量化底座与Adapter训练分开

[QLoRA原论文](https://arxiv.org/abs/2305.14314)强调NF4、Double Quantization和Paged Optimizers；以LoRA训练框架加这三个组件可按四层解释，但不能为了清单的“四技术”凭空再编一项。

NF4针对近似正态权重分布设计4bit表示，不是所有激活、乘法和gradient都4bit，也不证明任意分布上优于INT4。Frozen基础读取时按compute dtype反量化，adapter用自己的参数与优化状态；参数存储精度和算子compute精度分开。

Double quantization进一步量化quantization constants/scales等信息，压metadata，不是把4bits简单降2bits或再训一次模型。收益随格式/group size变化，不将论文某个容量节省比例套所有实现。

Paged optimizer使用分页/统一内存等机制应对状态峰值，可能产生CPU/GPU迁移成本；与ZeRO分片不是一个层次，不能保证长上下文不OOM。真正峰值还包括基础权重、激活、临时dequant、attention/head、通信buffer。

粗略账本$M=M_{\rm frozen\ base}+M_{\rm adapter\ params/grads/optimizer}+M_{\rm activation}+M_{\rm temporary}$。冻结BF16底座约$2P$bytes，Adam moments常FP32；是否有master副本依backend，不给所有冻结参数都按可训16bytes计算。Forward FLOPs也不按trainable百分比减少。

排查先按加载、forward、backward、step、保存定位，再看长度/batch、checkpoint、分片、quant/offload。数据不覆盖或template错的质量问题不能只靠增rank解决。参考[训练显存](#q=infra-training-step)与[后端](#q=infra-training-backends)。

## Adapter、Prefix、Prompt与部署合并

Adapter增加模块，Prefix/Prompt tuning学习prefix/输入等参数，LoRA修改权重增量；可训容量、序列/KV成本、是否可合并和部署约束不同。名字都含PEFT不表示可以相同方式merge。

同一未量化底座、相同dtype理想算术、关闭dropout条件下，$W_{\rm merge}=W_0+sBA$与已训LoRA双支路forward等价；它不等于另外全参微调学出的权重。Quantized merge还可能反量化/再量化，需要比较误差与任务quality。

多个adapter动态切换可支持多任务/租户，但必须绑定base revision、adapter ID/version、rank、target和请求。Static merge减少runtime支路，dynamic serving需要额外资源与调度。产品支持组合依版本，不用旧博客推断当前所有上限。

同prefix不同adapter改变hidden/KV，因此cache key需adapter身份，不能跨adapter直接reuse。请求中换adapter会混策略，发布与并发需明确。MoE式adapter路由需额外路由/训练设计，不是加载多个文件就自动学好组合。

## 手写一个低秩层

```python
# Teaching example: no bias, dropout, quantization or adapter loader.
class LowRankLinear(torch.nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        dout, din = weight.shape
        self.register_buffer("base", weight.detach().clone())
        self.A = torch.nn.Parameter(torch.randn(rank, din) * .02)
        self.B = torch.nn.Parameter(torch.zeros(dout, rank))
        self.scale = alpha / rank
    def forward(self, x):
        return (torch.nn.functional.linear(x, self.base)
                + self.scale * torch.nn.functional.linear(
                    torch.nn.functional.linear(x, self.A), self.B))
```

输入可为[B,L,din]，两次linear为din→r→dout。Base buffer不被optimizer更新，真实库也可能用requires_grad=False的Parameter；这是教学层，不是完整QLoRA/PEFT实现。应核初始与base输出一致、W0不动、A/B首步梯度及merge输出，不能只检查loss下降。

## 面试问答16题

### 对照实验和常见训练误判

最小四组可设全参、attention-only LoRA、attention+FFN LoRA、量化底座LoRA，但起点、数据、模板、总训练tokens和评价协议要对齐，LR/rank按各方法合理调优。报告trainable数量、加载/forward/backward峰值、步时和独立质量，不能用一个方法参数默认值硬约束所有对照后宣布胜者。

若loss无变化，先核adapter真进入forward和optimizer、基础是否正确冻结、A/B初始化、labels/mask与LR；若B有grad而A首步0，这可能是正确零B初始化，不要直接改成“两边随机”掩盖原因。若只有少数target modules生效，检查fused linear层名与实际shape，不用config截图替参数审计。

若merge前后不一致，先切eval关闭dropout、固定dtype和相同base，核是否重复merge、scale、转置布局、bias和量化；容差由dtype与用途定义，再看真实生成。理想代数差很小不证明不同kernel/量化后完全相同。

Serving多adapter可在同prompt分别取结果，确认请求实际resolved adapter version、cache隔离和rank兼容。用户session中途切adapter需明确策略与状态；“热切”支持不等零影响、零显存或零重载成本。与你turn项目的resident weights类似，pointer detach也不能自动成为可并发更新的独立snapshot。

可运行[课堂CPU脚本](./assets/learning/post-training-project-lab.py)核第一步A/B梯度与理想merge，结果[JSON](./assets/learning/post-training-project-lab-results.json)只说明小线性层，不代表QLoRA内核或真实多adapter部署benchmark。

**Q01 低秩的是W吗？** 增量BA受约束，基础W可full-rank。

**Q02 A/B形状和参数数？** A=r×din，B=dout×r，r(din+dout)。

**Q03 B0谁首步有梯度？** B通常有，A可为0，后续A也学习。

**Q04 α/r有什么用？** 缩放增量与gradient，rank/LR变化一起核。

**Q05 Target选哪些？** 真架构层名、任务、资源消融，不按模板字符串猜。

**Q06 LoRA同比省activation吗？** 不自动，基础计算与链式反向仍在。

**Q07 QLoRA全程4bit吗？** Frozen存储、compute、adapter与optimizer分别定义。

**Q08 NF4普遍最优？** 有分布/格式前提，需实测。

**Q09 双量化压什么？** 量化常数和metadata，不等再蒸馏模型。

**Q10 Paged optimizer等ZeRO吗？** 分页迁移与分片不同。

**Q11 合并等全参微调吗？** 只与该LoRA支路理想等价，不等另一次训练。

**Q12 量化merge核什么？** dtype/再量化误差、推理行为和真实quality。

**Q13 Prefix/Adapter差在哪？** 作用位置、序列和部署成本分别解释。

**Q14 多adapter为何隔离cache？** Hidden/KV依参数身份。

**Q15 Rank越高越好？** 容量/数据/优化取舍，用独立验证。

**Q16 项目参数怎么答？** Base/adaptor版本、r/α、target、quant/compute、data/budget与实测，不借用别人的具体数值。

## Checklist合并与复习

本页原公式、参数算例与OOM边界保留，本轮在同入口补机制与问题。更详细的量化kernel、多adapter服务当前支持组合回官方版本资料，不把教学层当性能benchmark。清单关联的156/159/163等部署题读[模型服务化](#q=model-service-interview)，原始题号归属见[合并地图](#q=post-training-checklist-map)。
