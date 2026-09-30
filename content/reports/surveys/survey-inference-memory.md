---
id: "survey-inference-memory"
title: "专题调研：LLM 推理显存，模型结构与缓存管理如何分工？"
type: "survey"
date: "2026-09-30"
directions: ["llm", "infra"]
paper_ids: ["paper-gqa", "paper-pagedattention"]
tags: ["专题调研", "KV Cache", "推理"]
updated: "2026-09-30"
summary: "以 GQA 和 PagedAttention 为起点，比较减少 KV 需求与改善 KV 放置，并标出 FlashAttention 等下一轮范围。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

问题：长上下文、高并发服务被显存限制时，应该改模型结构、改缓存管理还是改 kernel？本轮先分清机制与指标，不替未指定硬件/模型的项目直接选型。

## 2. 范围、检索与覆盖边界

调研日期 2026-09-30；以 GQA 与 PagedAttention 原论文及已有笔记为范围，核对核心机制和评估章节。纳入标准是直接影响 KV 需求或放置；不做完整新论文综述。

FlashAttention 用作注意力中间量/IO 的边界参照，其单篇精读仍待补；KV 量化、缓存驱逐、分离式服务等尚未纳入比较。没有把未覆盖路线当作已调查。

## 3. 论文集合与方法谱系

- [GQA](#paper=paper-gqa)：模型结构层，减少每 token 所需 KV 头。
- [PagedAttention](#paper=paper-pagedattention)：服务内存层，管理逻辑/物理块与共享。
- [FlashAttention 笔记](#q=flash-attention)：算子 IO 层，减少 attention 中间量的实体化和读写；不直接等于跨请求 KV 管理。

先按瓶颈分层，再决定下一篇要补哪条路线。

## 4. 统一比较表

| 路线 | 改变环节 | 主要作用 | 适用假设/代价 | 证据边界 |
|---|---|---|---|---|
| GQA | 模型 K/V 共享 | 降低 KV 存储/读取需求 | 要求兼容模型；转换/训练与质量取舍 | 原文 T5/TPU 设置，不直接外推 |
| PagedAttention | KV 放置/共享 | 降低碎片与重复占用 | 块映射和生命周期有开销 | 原文服务负载与延迟设置 |
| FlashAttention（待补精读） | 稠密 attention 的执行 | 降低中间量 IO | dtype/形状/硬件和非 attention 瓶颈 | 本轮仅用作机制边界参照 |

不在不兼容任务、模型和预算之间用单一加速倍数排名。

## 5. 共识、分歧与证据强度

学习性推演：三条路线作用在不同开销上，因此存在组合空间。它们的原实验不构成对同一现代服务负载的头对头比较；“组合后是否更好”仍需要对照实验。

先拆 [显存账本](#q=memory-budget)，看权重、KV、激活/workspace 的占比；再用阶段时间线找计算、带宽和等待。本文没有提供本人硬件实测。

## 6. 场景化结论与选型

模型结构可选时先核对 KV 头与任务质量；模型固定而缓存浪费明显时调查分页/共享；prefill 算子 IO 成本高时调查 attention 内核。结论均需符合实际瓶颈，不能用“都能提速”作为选择理由。

## 7. 空白与下一轮验证

待补 KV 量化的误差、前缀复用的命中与驱逐、不同到达速率的调度影响。最小对照可固定模型与请求集，只改一个缓存策略，报告 TTFT/ITL、goodput 与峰值内存；尚未执行。

## 8. 阅读顺序与个人理解检查

先读 GQA 的结构图，再读 PagedAttention Figure 6，再回看 FlashAttention 的在线归一化。合上报告回答：减少需求、减少浪费、减少 IO 各是什么？为什么显存占用变少不保证尾延迟更好？

## 9. 持续更新记录与来源

2026-09-30：建立两篇精读的分层对照，并明确未覆盖路线。未来每次更新注明新增证据改变哪条判断。

原始来源：[GQA v3](https://arxiv.org/abs/2305.13245v3)、[PagedAttention v1](https://arxiv.org/abs/2309.06180v1)、[FlashAttention](https://arxiv.org/abs/2205.14135)。证据位置见单篇记录；场景建议是整理者推演。
