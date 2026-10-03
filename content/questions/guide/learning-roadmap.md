---
id: learning-roadmap
title: "从哪里开始？Diffusion、LLM、Agent、Infra 四条学习路线"
category: guide
difficulty: 基础
tags: ["P0", "学习路线", "开始这里"]
updated: 2026-10-03
summary: "按先修关系安排学习，每个问题都落到解释、推导或可验证练习。"
draft: false
---

## 这份知识库怎样用

定位是广方向 AI 学习与面试准备：先通过问题理解机制，再用短表达检验理解。四个方向都维护；算法编程仅少量手撕练习，项目论文与综合面试暂保留原入口。

P0 表示第一轮应掌握的主干，P1 表示主干后再深入。这是本笔记的学习优先级，不是统计出的面试频率，不声称为任何公司的真实面经。

每篇按“问题 → 机制/例子 → 误区 → 小实验 → 表达 → 原始资料”阅读。除已标注的 CPU 数值检查外，模型实验尚未复现；公式算例与经验判断不冒充测量结果。

## 共同先修与两条桥梁

先确认矩阵乘法形状、概率与期望、softmax/logprob、链式法则，以及 PyTorch 张量/autograd 的基本概念。遇到空白可先用小向量手算，不必一次补完全部数学。

两条跨方向连接：

- **生成模型 → Infra**：Attention/DiT 的 token 数 → 激活和 KV 账本 → kernel 与调度 → 端到端性能。
- **LLM → Agent → 训练系统**：token 概率和 mask → 工具轨迹 → 结果评测 → 策略更新 → rollout 长尾与版本。

## Diffusion：从加噪到连续流

1. [DDPM：噪声监督为何能生成](#q=ddpm-denoising)。产出：写出 x_t 和 x0_hat。
2. [epsilon/x0/v/score](#q=diffusion-parameterization)。产出：验证参数化转换。
3. [DDIM 与采样器](#q=ddim-sampling)。产出：区分训练目标、时间网格、NFE。
4. [CFG](#q=classifier-free-guidance)。产出：解释尺度约定和过强引导。
5. [Latent Diffusion](#q=latent-diffusion)。产出：画编码、条件、去噪、解码的数据流。
6. [Flow Matching/RF](#q=flow-matching)。产出：推导直线路径速度和 Euler 更新。
7. P1：[DiT](#q=dit-architecture)、[Consistency](#q=consistency-distillation)。产出：说明分辨率成本与少步模型的训练差异。

## LLM：从概率分解到后训练

1. [Attention](#q=attention-complexity) → [自回归训练与 mask](#q=causal-lm-training)。产出：画张量形状和标签移位。
2. [RoPE](#q=rope-position)。产出：证明旋转点积依赖相对距离。
3. [KV cache](#q=kv-cache) → [GQA/MQA](#q=mha-gqa-mqa)。产出：手算一个并发服务的 KV 容量。
4. [LoRA](#q=lora-finetuning)。产出：区分更新参数、权重和激活。
5. [SFT/DPO/RL](#q=sft-dpo-rl)。产出：按数据、目标、评测比较。
6. [RAG](#q=rag-vs-finetuning)。产出：用人工证据对照拆开检索与生成失败。

## Agent：从交互闭环到学习

1. [Agent/workflow](#q=agent-workflow) → [工具接口](#q=agent-tool-design)。产出：画执行与验证闭环。
2. [上下文管理](#q=agent-context)。产出：区分模型输入、外部状态和记忆。
3. [评测](#q=agent-evaluation)。产出：定义完成判据并区分单次可靠性、多次探索。
4. [多轮梯度与 mask](#q=agent-gradient)。产出：保存每轮真实训练条件。
5. [GRPO/RLVR](#q=grpo-rlvr) → [OPD](#q=opd-vs-sft)。产出：说清采样者、监督来源与概率目标。

## Infra：先比较训练框架，再逐轮补充系统知识

本轮先读 [训练框架怎么选、怎么用？以 verl 与 slime 为主线](#q=training-inference-frameworks)。

1. 第 1—2 节：区分训练后端与后训练框架，理解 FSDP、DeepSpeed、Megatron 的接入方式。
2. 第 3—6 节：读 verl / slime 的架构、配置与官方示例，说明同一任务换框架需要改哪里。
3. 第 7—8 节：比较其他候选，练习回答框架选择和实际使用的问题。

产出：一张框架职责表、一份针对具体任务的选型说明、一页可解释的使用配置。显存、并行通信、推理优化和系统排障后续分别整理。

## 可重复的复习循环

每次选一个主问题：先自己解释两分钟；读机制并做一个小推导；完成或设计小实验；写下失败条件；最后不看笔记回答 30 秒。能解释假设、边界和例子，再标记“已掌握”。

第一轮可用 12 次学习时段：每个方向三次，依次覆盖机制、对比、实践。每次只取 1–2 篇，不把阅读数当掌握程度。路线不限定天数，可按自己时间循环。

轻量手撕：[单头 causal attention](#q=causal-attention-coding) 和已有 [LRU](#q=lru-cache)。项目表达仍保留 [三分钟讲项目](#q=project-story)，真实经历由本人填写。

进一步看 [适合四方向的项目与知识架构](#q=learning-architecture)，以及 [资料索引](#q=research-sources)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：作为生成与系统的共同基础；路线排序是本笔记的整理建议。
