---
id: causal-lm-training
title: "LLM 训练能并行，为什么生成还要逐 token？"
category: llm
difficulty: 基础
tags: ["P0", "Causal LM", "Loss mask"]
updated: 2026-09-30
summary: "从标签右移、因果 mask 和 teacher forcing 理解训练与推理差异。"
draft: false
---

## 从问题出发

“自回归”描述概率分解，不表示训练必须逐 token 调用网络。

先修：[Attention](#q=attention-complexity)。

## 标签与条件

```text
序列：      [BOS, 我, 喜欢, 学习, EOS]
模型输入：  [BOS, 我, 喜欢, 学习]
预测目标：  [我, 喜欢, 学习, EOS]
位置 i 的分布：p(token_i | token_<i)
L = -sum_i mask_i*log p(token_i|token_<i) / sum_i mask_i
```

训练时正确前缀已知，causal mask 阻止访问未来，所以所有位置可以在一个 forward 中并行计算。推理时新 token 尚未知，采样后才能确定下一个位置的输入；[KV cache](#q=kv-cache) 减少重复计算，但不消除这个依赖。

有些训练框架在 loss 内部自动 shift，数据预处理再 shift 一次会让输入与目标错位。

## 三种 mask 不要混用

| 类型 | 控制什么 | 例子 |
|---|---|---|
| causal mask | 哪些位置能互相关注 | 不允许读取未来 |
| padding mask | 哪些位置不是有效输入 | 填充 token 不参与注意力 |
| loss mask | 哪些预测位置有监督 | SFT 只监督 assistant 输出 |

prompt 的 loss=0 不等于 prompt 不进入模型；它仍决定后续输出的条件。工具输出同理，见 [多轮轨迹训练](#q=agent-gradient)。全零 loss mask 要跳过或明确定义，避免除零。

## 长短样本怎样加权？

把所有有效 token 平均会让长输出贡献更多项；先按样本平均再按 batch 平均会给予每个样本相近权重。两者不是无条件优劣，需要对应任务和数据分布。sequence packing 时还要定义不同样本间是否隔离注意力，不能只拼接并忽略边界。

## 小实验与自检

手画 4×4 causal mask。构造 prompt+answer，分别打印输入、labels、监督位置；检查每个 label 对应的前缀。用两条长度不同的样本比较 token 平均与样本平均。

自检：没有 causal mask 但标签正确，会发生什么信息泄漏？训练 loss 下降为什么不保证自由生成质量同步提高？

## 面试表达

训练拥有完整正确序列，能并行求各条件分布；推理要等待前一步结果。再区分 attention mask、loss mask 与标签移位。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：核对 decoder 的 masked attention；本文的标签示例是教学展开。
- [InstructGPT](https://arxiv.org/abs/2203.02155)：了解示范监督在指令模型中的作用。
