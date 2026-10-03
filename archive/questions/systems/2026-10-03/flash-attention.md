---
id: flash-attention
title: "FlashAttention 为什么快？在线 softmax 如何避免保存完整矩阵？"
category: systems
difficulty: 进阶
tags: ["P0", "FlashAttention", "IO"]
updated: 2026-09-30
summary: "用分块、在线归一化与内存访问解释精确注意力的加速。"
draft: false
---

## 从问题出发

标准 attention 计算 S=QKᵀ，再 softmax、乘 V。若反复把 N×N 中间矩阵写入显存，数据移动可能很贵。

先修：[Attention 复杂度](#q=attention-complexity)。

## 分块后的难点：全局归一化

softmax 需要整行的分母，不能把各块分别 softmax 后直接相加。维护最大值 m、归一化和 l，以及未归一化的加权值向量 z：

```text
一块分数 s_b、对应 V_b：
m_new = max(m, max(s_b))
l_new = exp(m-m_new)*l + sum(exp(s_b-m_new))
z_new = exp(m-m_new)*z + sum_j exp(s_b[j]-m_new)*V_b[j]
最终输出 = z/l
```

初始 m=-infinity、l=0、z=0；实际实现还需处理完全 masked 的行等边界。更改最大值时必须同时缩放旧 l 和 z，否则不同块不在同一数值尺度。

## 为什么节省内存访问？

将 Q/K/V 分块加载到片上存储，在块内计算、累积结果，不需要将完整注意力矩阵在高带宽显存里实体化。训练 backward 可结合重算，以额外计算减少保存量。

标准稠密 attention 的算术项仍然 O(N²d)，不是线性注意力，也不是自动稀疏化。“exact”指算法计算同一注意力表达式，浮点运算顺序不同仍可产生细小数值差异。

## 什么时候收益有限？

极短序列、很小 batch、非 attention 瓶颈、dtype/形状不满足高效 kernel 条件时，端到端收益可能有限。库调用也可能按硬件选择不同后端，不能看到函数名就默认已用了某个 kernel。

## 小实验与自检

先实现一维分块 softmax，与整行版本比较极大 logits 下的误差；再带 V 比较输出。真实 GPU 实验要固定 dtype、mask、shape，经过 warmup，并分别看算子时间和完整 step 时间。

CPU 数值检查记录（2026-09-30）：按正文公式在随机分块 logits 与 V 上累计，输出与整行稳定 softmax 的加权输出一致（NumPy allclose）；不表示已实现或测量 GPU FlashAttention kernel。

自检：为什么各块独立 softmax 后平均不正确？“不保存完整矩阵”与“没有计算每对 token”有什么区别？

## 面试表达

先明确复杂度仍为平方，再用在线 softmax 的缩放关系解释 IO 节省；最后说明实测条件。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [FlashAttention](https://arxiv.org/abs/2205.14135)：阅读 IO-aware 分块与在线 softmax；公式是学习性展开。
