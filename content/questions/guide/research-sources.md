---
id: research-sources
title: "资料索引：每个知识点该读哪篇论文或官方文档？"
category: guide
difficulty: 基础
tags: ["P0", "资料索引", "调研方法"]
updated: 2026-09-30
summary: "集中记录原始来源、对应知识点与核验边界，方便继续深挖。"
draft: false
---

## 调研范围与使用原则

本轮面向 diffusion、LLM、agent、infra 的共同主干：公开检索后优先访问原始论文、官方工程文章和官方文档。不是面经真实性统计，也不声称覆盖所有近期进展。

来源用于支撑具体机制；正文中的算例、练习与架构建议为学习性整理。关键目标和架构选取了原文重点段落核对，不把访问论文摘要表述为全文复现。除个别文章明确记录的 CPU 代数/代码性质检查外，模型训练、生成质量与 GPU 性能实验尚未执行。阅读时间按文本长度估计，不含练习。

每篇末尾说明原文该看什么。不要只收藏论文；带着一个问题读相关公式或实现约束。官方文档可更新，运行代码时应核对环境版本；prefix cache 链接固定 vLLM v0.13.0，避免误用版本配置。

## 去哪里查某个机制

| 原始来源 | 类型 | 对应学习笔记 |
|---|---|---|
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | 官方工程文章 | [Agent 和 workflow 的边界在哪里？什么时候值得让模型自己决策？](#q=agent-workflow)、[怎样让项目与知识架构适合同时学习四个方向？](#q=learning-architecture) |
| [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | 官方工程文章 | [Agent 到底怎么评测？为什么 pass@k 高仍可能不好用？](#q=agent-evaluation) |
| [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | 官方工程文章 | [多轮 Agent 上下文越来越长，应该删、压缩还是检索？](#q=agent-context) |
| [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | 官方工程文章 | [工具调用为什么经常失败？怎样设计模型容易正确使用的接口？](#q=agent-tool-design) |
| [PyTorch checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html) | 官方文档 | [Activation checkpointing 为什么能省显存？重算有哪些正确性陷阱？](#q=training-inference-frameworks) |
| [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | 官方文档 | [DDP、ZeRO/FSDP、TP 和 PP 分别切什么？](#q=training-inference-frameworks) |
| [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) | 官方文档 | [训练和推理显存怎么估？为什么参数量不能直接换算可运行规模？](#q=training-inference-frameworks)、[Agent/RL rollout 为什么拖慢训练？同步与异步怎样权衡？](#q=training-inference-frameworks)、[显存占满，为什么 GPU 利用率仍然很低？](#q=training-inference-frameworks) |
| [vLLM APC v0.13.0](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/) | 官方文档 | [Prefix caching 缓存什么？为什么看起来相同的 prompt 没命中？](#q=training-inference-frameworks) |
| [Diffusers schedulers](https://huggingface.co/docs/diffusers/using-diffusers/schedulers) | 官方文档 | [epsilon、x0、v 和 score 预测，到底在预测什么？](#q=diffusion-parameterization) |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 论文 | [手撕单头 causal attention：形状、mask 和稳定 softmax](#q=causal-attention-coding)、[从哪里开始？Diffusion、LLM、Agent、Infra 四条学习路线](#q=learning-roadmap)、[Self-Attention 的计算复杂度为什么是平方级？](#q=attention-complexity)、[LLM 训练能并行，为什么生成还要逐 token？](#q=causal-lm-training) |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 论文 | [DDP、ZeRO/FSDP、TP 和 PP 分别切什么？](#q=training-inference-frameworks)、[训练和推理显存怎么估？为什么参数量不能直接换算可运行规模？](#q=training-inference-frameworks) |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 论文 | [知识问答该用 RAG 还是微调？怎样判断失败在检索还是生成？](#q=rag-vs-finetuning) |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 | [为什么预测噪声就能生成图像？从 DDPM 的训练目标讲起](#q=ddpm-denoising)、[epsilon、x0、v 和 score 预测，到底在预测什么？](#q=diffusion-parameterization) |
| [DDIM](https://arxiv.org/abs/2010.02502) | 论文 | [DDIM 为什么能少步采样？采样器和训练目标怎样分工？](#q=ddim-sampling) |
| [Megatron-LM](https://arxiv.org/abs/2104.04473) | 论文 | [DDP、ZeRO/FSDP、TP 和 PP 分别切什么？](#q=training-inference-frameworks) |
| [RoFormer](https://arxiv.org/abs/2104.09864) | 论文 | [RoPE 为什么能表达相对位置？长上下文外推卡在哪里？](#q=rope-position) |
| [LoRA](https://arxiv.org/abs/2106.09685) | 论文 | [LoRA 到底省了什么？为什么用了 LoRA 仍会 OOM？](#q=lora-finetuning) |
| [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 论文 | [为什么在 latent 里做扩散？VAE、文本编码器和去噪器怎样协作？](#q=latent-diffusion) |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 论文 | [LLM 训练能并行，为什么生成还要逐 token？](#q=causal-lm-training)、[SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl) |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 论文 | [Self-Attention 的计算复杂度为什么是平方级？](#q=attention-complexity)、[FlashAttention 为什么快？在线 softmax 如何避免保存完整矩阵？](#q=training-inference-frameworks) |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 论文 | [CFG 如何增强条件控制？为什么 guidance 太大反而变差？](#q=classifier-free-guidance) |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | 论文 | [Flow Matching 与 Rectified Flow 怎么理解？直线路径为何不保证一步生成？](#q=flow-matching) |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 论文 | [Flow Matching 与 Rectified Flow 怎么理解？直线路径为何不保证一步生成？](#q=flow-matching) |
| [ReAct](https://arxiv.org/abs/2210.03629) | 论文 | [多轮 Agent 调用工具后，梯度如何回传？](#q=agent-gradient) |
| [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | 论文 | [DiT 怎样把 Transformer 用于扩散？它和 LLM 相同在哪里？](#q=dit-architecture) |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 论文 | [少步生成如何蒸馏？Consistency 与减少采样步数有什么区别？](#q=consistency-distillation) |
| [GQA](https://arxiv.org/abs/2305.13245) | 论文 | [KV Cache 缓存了什么，为什么能加速生成？](#q=kv-cache)、[MHA、GQA、MQA 如何在质量与 KV 显存之间取舍？](#q=mha-gqa-mqa) |
| [DPO](https://arxiv.org/abs/2305.18290) | 论文 | [SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl) |
| [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) | 论文 | [On-policy distillation 和 SFT 有什么区别？](#q=opd-vs-sft) |
| [PagedAttention](https://arxiv.org/abs/2309.06180) | 论文 | [KV Cache 缓存了什么，为什么能加速生成？](#q=kv-cache)、[PagedAttention 和 continuous batching 各解决什么？为什么高吞吐可能让首 token 更慢？](#q=training-inference-frameworks) |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 论文 | [RLVR 与 GRPO 是一回事吗？组内奖励怎样变成训练信号？](#q=grpo-rlvr)、[多轮 Agent 调用工具后，梯度如何回传？](#q=agent-gradient)、[Agent/RL rollout 为什么拖慢训练？同步与异步怎样权衡？](#q=training-inference-frameworks) |

## 如何继续维护

优先补已经学到但解释不清的机制或实验失败，不按“题目数量”扩充。新笔记保留问题、假设、机制、误区、练习和原始链接；涉及性能数字注明硬件、版本、负载与测量边界。

将自己的复现记录与资料整理分开写，明确真实结果和待验证猜想。领域有新方法时先说明它解决哪一篇的哪个局限，再补入对应主线。算法与项目面经暂保持轻量。

回到 [学习路线](#q=learning-roadmap)，或查看 [知识与实践架构](#q=learning-architecture)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：性能学习的官方实践入口；上表中的外链是本轮资料索引。
