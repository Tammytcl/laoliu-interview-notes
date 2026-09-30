---
id: "paper-gqa"
title: "GQA：用多少 KV 头换取质量与速度？"
paper_title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
authors: ["Joshua Ainslie", "James Lee-Thorp", "Michiel de Jong", "Yury Zemlyanskiy", "Federico Lebrón", "Sumit Sanghai"]
year: 2023
direction: "llm"
paper_url: "https://arxiv.org/abs/2305.13245v3"
evidence: "资料整理"
note_ids: ["mha-gqa-mqa", "kv-cache"]
tags: ["基础论文", "llm"]
updated: "2026-09-30"
summary: "读清模型结构、checkpoint 转换与 uptraining，避免把结构节省当成质量无损。"
template_version: 1
draft: false
---

## 1. 收录动机与阅读目标

连接模型结构和推理容量。主问题：“减少 K/V 头后省了什么，质量为什么需要再训练来验证？”

## 2. 原文信息与核验范围

[原文 v3](https://arxiv.org/abs/2305.13245v3)。核对第 2 节方法、Figure 2、Table 1 与第 3.3 节消融；未做本人 uptraining 或硬件复现。这是资料整理，不把 T5/TPU 实验直接外推到任意 decoder/GPU 服务。

## 3. 研究问题与先修知识

先会 [KV 容量公式](#q=kv-cache) 与 [MHA/GQA/MQA](#q=mha-gqa-mqa)。瓶颈是自回归服务中 KV 的存储/读取；减少 query 头与共享 KV 头不是同一种结构改变。

## 4. 一句话核心贡献

在 MHA 与 MQA 之间以分组共享 KV 做折中，并研究已有多头 checkpoint 的转换及继续预训练。

## 5. 方法与关键推导

同组 query 使用共同 K/V，各 query 仍产生自己的注意力权重。论文的转换把组内原 K/V 投影做均值池化，再 uptrain。不能只改 reshape 就视为训练完成。

下面是教学容量算例，并非论文实测：

```text
32 层、head_dim=128、2 bytes/元素
KV 每 token = 2*32*H_kv*128*2 bytes
H_kv=32：512 KiB；H_kv=8：128 KiB
```

固定上述假设时 KV 容量缩至 1/4；总延迟、权重显存和任务质量都不能据此换算。

## 6. 关键图表与证据

| 位置 | 比较问题 | 设置与观察 | 边界 |
|---|---|---|---|
| Figure 2 | GQA 处于哪两个极端之间？ | query 与 KV 共享结构 | 不消除稠密 token 对计算 |
| Table 1 | 质量与推理耗时怎样折中？ | T5 变体、指定任务与 TPU 测量 | 不直接当作现代 GPU 服务速度 |
| 第 3.3 节 | 转换、继续训练和组数各有什么影响？ | 读对应消融曲线 | 不能归因给一个单独参数 |

## 7. 作者结论与我的判断

作者在所测配置中报告 GQA 的折中收益。整理者的判断：设计容量时先看 KV 头，选型时再看实际任务和 kernel；这是有条件的工程推演，不是本人验证。

## 8. 局限、反例与失败条件

多头 checkpoint 的转换需要兼容结构和训练。减少 KV 容量不一定解决权重、工具等待或队列瓶颈；硬件并行与内核也会影响收益。

## 9. 和已有知识的连接

[KV cache](#q=kv-cache)、[显存账本](#q=memory-budget)，以及 [PagedAttention 精读](#paper=paper-pagedattention)：前者改模型所需 KV，后者改这些 KV 如何放置和共享。

最小练习：不先训练大模型，先打印 grouped attention 的头维布局，并手算不同 H_kv 的容量；尚未执行。

## 10. 不看报告的复述问题

先在个人阅读记录中写一个读前问题；读后收起正文，用自己的话回答：本文改变了哪个环节？为什么合理？最关键的证据是什么？哪里仍不确定？这里不替你填“我的理解”。

## 11. 待验证与下一步

本人模型实验尚未执行。先核读正文标出的机制与图表，再完成一项有明确对照的最小练习；把真实配置、结果和失败样本写回记录。个人阅读进度从“待精读”开始，不由报告生成自动推进。

## 12. 更新记录与来源

2026-09-30：作为论文收录组件的基础精读示例整理，原文重点位置列于第 2/6 节；尚未进行本人复现。正文中的阅读建议和学习推演不当作原论文实验结论。
