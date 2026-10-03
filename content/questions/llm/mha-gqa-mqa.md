---
id: mha-gqa-mqa
title: "MHA、GQA、MQA 如何在质量与 KV 显存之间取舍？"
category: llm
difficulty: 基础
tags: ["P0", "GQA", "KV Cache"]
updated: 2026-09-30
summary: "比较 query 与 KV 头的数量、共享方式及推理内存收益。"
draft: false
---

## 从问题出发

长上下文推理不只受权重显存限制，历史 K/V 也很贵。减少 KV 头是从模型结构上压缩这部分开销。

## 头数与共享

| 机制 | Query 头数 | KV 头数 | 共享方式 |
|---|---|---|---|
| MHA | H_q | H_q | 每个 Q 头有对应 K/V |
| GQA | H_q | H_kv，介于 1 与 H_q | 多个 Q 头共享一组 K/V |
| MQA | H_q | 1 | 所有 Q 头共享 K/V |

常见均匀分组要求 H_q 能被 H_kv 整除。共享 KV 不等于 query 头也减少，各头可以根据自己的 Q 得到不同注意力结果。

## 容量例子

忽略元数据、分页余量和并行切分，32 层、8 KV 头、头维 128、每元素 2 字节：

```text
每 token KV = 2*32*8*128*2 = 131072 bytes = 128 KiB
8192 token 的单请求 KV ≈ 1 GiB
改成 32 KV 头，同样长度约 4 GiB
```

这只是容量估算，不是实测；真实服务还要放权重、激活、workspace 和多个请求。见 显存预算。

## 速度与模型能力

较少 KV 头减少缓存存储和 decode 的 KV 读写压力，收益依赖 kernel、batch 和硬件。它也改变表达容量。原论文包含从多头 checkpoint 进行 uptraining 的方法，不能把训练好权重简单平均后就认为质量无损。

## 小实验与自检

实现概念层的 repeat/broadcast，让两组 query 共享 K/V，打印每头输出。再估算 32 层模型在三种 KV 头数和两个上下文长度下的容量。

自检：GQA 是否把标准 attention 对序列长度的平方计算项消除？为什么显存减少 4 倍不代表总吞吐增加 4 倍？

## 面试表达

从 KV cache 的线性容量公式讲起，再说明结构共享与质量、带宽之间的取舍。

关联：[PagedAttention](#paper=paper-pagedattention)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [GQA](https://arxiv.org/abs/2305.13245)：阅读分组 query 与 uptraining；质量比较限于论文设置。
