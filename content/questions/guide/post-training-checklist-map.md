---
id: post-training-checklist-map
title: "后训练清单专题合并：222项知识地图与复习路线"
category: guide
difficulty: 基础
tags: ["P0", "Checklist", "Post Training", "Interview", "Study Map"]
updated: "2026-10-13"
summary: "按用户提供页面实际222条建立逐编号映射，优先复用原笔记，补基础分词、PEFT、评分、RAG与服务化；不把源作者个人项目参数移植给用户。"
draft: false
---

## 1. 问题背景 从清单找到知识缺口，而非再造同题笔记

用户给出的[来源页面](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)网址保留100问，导航仍写152问，2026-10-13正文编号实际1–222，共29组。本文按实际编号完整建立知识映射；来源文字不整页复制，下方自测提示由本文重述，技术答案回原始论文/文档与本站正文。源作者医疗项目的模型、数据数量与指标不是你的经历。

合并原则：Attention/RoPE/手撕、PPO/DPO/GRPO/OPD、推理/显存/并行已有正文就直接复用；LoRA和RAG在原入口扩写；缺口新增基础分词、Reward/Judge、模型服务化；个人项目换成ESCD与turn框架。不是222篇单独文章，也不声称完成了222个真实项目或独立实验。

## 2. 前置知识与回答结构

每题先判断对象：数据/策略/teacher/reference/critic、目标/估计器/runtime、理论/CPU/实际GPU/任务结果。先30秒回答核心，再给公式或一条数据流，最后说边界和项目证据。答案不清时从对象与输入输出补课，不从缩写的年份猜选型。

本文链接表示知识入口，不表示读完即熟练，也不要求UI新增背诵状态。源编号用于回查，一些条目应做实际代码练习或填写个人运行证据，不能从作者的示例数值直接补齐。

## 3. 这轮具体合并了什么

| 知识层 | 正文入口 | 本轮处理 |
| --- | --- | --- |
| LLM/Transformer/Tokenizer | [基础专题](#q=llm-foundations-interview)及既有Attention/RoPE | 补对象、归一化、分词工具/算法、模板与特殊符号 |
| SFT/偏好/RL/OPD | [40道后训练问答](#q=post-training-interview)、既有论文和RL六篇 | 复用深推导，补问题驱动直答/知识/追问 |
| PEFT | [原LoRA页](#q=lora-finetuning) | 原入口扩写QLoRA、梯度、merge、多adapter与代码 |
| RM/过程评价/Judge | [评分专题](#q=reward-judge-interview) | BT、标签、校准、评委偏差与统计 |
| RAG质量与工程 | [原RAG页](#q=rag-vs-finetuning) | 保留定位矩阵，合并chunk/hybrid/rerank、并行、缓存和路由 |
| Infra | 既有训练/显存/并行/引擎、[服务化](#q=model-service-interview) | 核心机制复用，服务/观测/发布新增 |
| 个人项目 | [ESCD](#q=escd-project-interview)、[turn实现](#q=agentic-turn-framework)、[创新论证](#q=agentic-turn-innovation) | 按真实论文与工作树替换通用模板，不借源作者成绩 |

## 4. 题面需要修正的前提

QLoRA通常按LoRA框架加NF4、双量化与分页优化器解释，不为“四项”编一项；BF16常免FP16式loss scaling，不保证数值永不溢出；中文token效率依语料/词表和计数单位，不普遍必差；DAPO是多个配方，不能只写长尾token preference；OPD和DPO没有脱离任务的固定提升倍率；医学安全“三原则”不当统一算法定理。

所有模型/数据量/指标类项目题用自己的runmanifest填写。示例“下降百分比”应区分百分点、相对变化和统计不确定性；公开模型未披露的内部配置不当八股标准答案。

## 5. 逐编号专题地图

下表覆盖来源全部222个编号。Knowledge prompts是本文的简短自测方向；回来源看原题，去学习入口看答案/机制/追问。涉及代码的条目还要真正手写、运行和检查边界，不把导航命中当作个人已完成练习。

### 模型基础与能力（来源 1–10）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 1 | 规模与计算预算 | [基础与分词](#q=llm-foundations-interview) |
| 2 | 公开代际演化 | [基础与分词](#q=llm-foundations-interview) |
| 3 | 三类网络任务 | [基础与分词](#q=llm-foundations-interview) |
| 4 | 因果生成取舍 | [基础与分词](#q=llm-foundations-interview) |
| 5 | 能力阈值与指标 | [基础与分词](#q=llm-foundations-interview) |
| 6 | 幻觉归因 | [基础与分词](#q=llm-foundations-interview) |
| 7 | 训练阶段职责 | [基础与分词](#q=llm-foundations-interview) |
| 8 | 上下文示例学习 | [基础与分词](#q=llm-foundations-interview) |
| 9 | 推理路径与预算 | [基础与分词](#q=llm-foundations-interview) |
| 10 | 代理目标退化 | [基础与分词](#q=llm-foundations-interview) |

### Transformer机制（来源 11–20）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 11 | 层级数据形状 | [基础与分词](#q=llm-foundations-interview) |
| 12 | 匹配与内容投影 | [基础与分词](#q=llm-foundations-interview) |
| 13 | 多头子空间 | [基础与分词](#q=llm-foundations-interview) |
| 14 | 点积量级缩放 | [基础与分词](#q=llm-foundations-interview) |
| 15 | 归一化与残差 | [基础与分词](#q=llm-foundations-interview) |
| 16 | 跨序列信息融合 | [基础与分词](#q=llm-foundations-interview) |
| 17 | 未来可见性 | [Attention手写](#q=causal-attention-coding) |
| 18 | 位置表示 | [基础与分词](#q=llm-foundations-interview) |
| 19 | 门控前馈 | [基础与分词](#q=llm-foundations-interview) |
| 20 | 注意力IO优化 | [基础与分词](#q=llm-foundations-interview) |

### 分词和输入协议（来源 21–28）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 21 | 合并式词表构造 | [基础与分词](#q=llm-foundations-interview) |
| 22 | 切分算法差别 | [基础与分词](#q=llm-foundations-interview) |
| 23 | 分词工具与算法 | [基础与分词](#q=llm-foundations-interview) |
| 24 | 控制标记 | [基础与分词](#q=llm-foundations-interview) |
| 25 | 词表与长度成本 | [基础与分词](#q=llm-foundations-interview) |
| 26 | 跨语言单位统计 | [基础与分词](#q=llm-foundations-interview) |
| 27 | 消息模板 | [基础与分词](#q=llm-foundations-interview) |
| 28 | ID与参数兼容 | [跨分词问答](#q=opd-interview-cross-tokenizer) |

### SFT与数据（来源 29–40）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 29 | 监督目标 | [后训练总问答](#q=post-training-interview) |
| 30 | 消息与示范格式 | [后训练总问答](#q=post-training-interview) |
| 31 | 输入输出mask | [后训练总问答](#q=post-training-interview) |
| 32 | 重复训练与泛化 | [后训练总问答](#q=post-training-interview) |
| 33 | 有效数据质量 | [后训练总问答](#q=post-training-interview) |
| 34 | 任务覆盖 | [后训练总问答](#q=post-training-interview) |
| 35 | 学习率与批量 | [后训练总问答](#q=post-training-interview) |
| 36 | 拟合与真实质量 | [后训练总问答](#q=post-training-interview) |
| 37 | 多轮响应选择 | [后训练总问答](#q=post-training-interview) |
| 38 | 示范筛选 | [后训练总问答](#q=post-training-interview) |
| 39 | 能力迁移与遗忘 | [后训练总问答](#q=post-training-interview) |
| 40 | 个人运行配置 | [后训练总问答](#q=post-training-interview) |

### 参数高效训练（来源 41–50）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 41 | 低秩层形状 | [LoRA与QLoRA](#q=lora-finetuning) |
| 42 | 更新缩放 | [LoRA与QLoRA](#q=lora-finetuning) |
| 43 | 投影层选择 | [LoRA与QLoRA](#q=lora-finetuning) |
| 44 | 量化适配组件 | [LoRA与QLoRA](#q=lora-finetuning) |
| 45 | 权重量化分布 | [LoRA与QLoRA](#q=lora-finetuning) |
| 46 | 元数据压缩 | [LoRA与QLoRA](#q=lora-finetuning) |
| 47 | 适配位置对照 | [LoRA与QLoRA](#q=lora-finetuning) |
| 48 | 支路合并等价 | [LoRA与QLoRA](#q=lora-finetuning) |
| 49 | 多adapter路由 | [LoRA与QLoRA](#q=lora-finetuning) |
| 50 | 个人量化配置 | [LoRA与QLoRA](#q=lora-finetuning) |

### 偏好优化（来源 51–62）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 51 | 四分数margin | [后训练总问答](#q=post-training-interview) |
| 52 | 偏好概率假设 | [后训练总问答](#q=post-training-interview) |
| 53 | 温度与约束尺度 | [后训练总问答](#q=post-training-interview) |
| 54 | 饱和梯度 | [后训练总问答](#q=post-training-interview) |
| 55 | 偏好泛化 | [后训练总问答](#q=post-training-interview) |
| 56 | 成对数据质量 | [后训练总问答](#q=post-training-interview) |
| 57 | 相对基线模型 | [后训练总问答](#q=post-training-interview) |
| 58 | 在线与离线优化 | [后训练总问答](#q=post-training-interview) |
| 59 | 偏好与真实性 | [后训练总问答](#q=post-training-interview) |
| 60 | 迭代采样 | [后训练总问答](#q=post-training-interview) |
| 61 | 个人偏好配置 | [后训练总问答](#q=post-training-interview) |
| 62 | 排序分数与生成质量 | [后训练总问答](#q=post-training-interview) |

### 在线策略优化（来源 63–74）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 63 | 偏好训练闭环 | [RL算法对照](#q=rl-policy-optimization-map) |
| 64 | 角色分工 | [RL算法对照](#q=rl-policy-optimization-map) |
| 65 | 悲观更新约束 | [PPO训练一步](#q=rl-ppo-training) |
| 66 | 策略正则 | [PPO训练一步](#q=rl-ppo-training) |
| 67 | 时序优势 | [回报与GAE](#q=rl-returns-gae) |
| 68 | 组相对估计 | [RL算法对照](#q=rl-policy-optimization-map) |
| 69 | 留一baseline | [RL算法对照](#q=rl-policy-optimization-map) |
| 70 | 回报归一变体 | [RL算法对照](#q=rl-policy-optimization-map) |
| 71 | 探索与有效组配方 | [RL算法对照](#q=rl-policy-optimization-map) |
| 72 | 可验证反馈 | [RL算法对照](#q=rl-policy-optimization-map) |
| 73 | 目标套利 | [RL算法对照](#q=rl-policy-optimization-map) |
| 74 | 行为分布偏移 | [RL算法对照](#q=rl-policy-optimization-map) |

### 在线蒸馏（来源 75–79）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 75 | 学生状态监督 | [OPD基础问答](#q=opd-interview-foundations) |
| 76 | 蒸馏与偏好目标 | [OPD基础问答](#q=opd-interview-foundations) |
| 77 | 循环生成评分 | [OPD基础问答](#q=opd-interview-foundations) |
| 78 | 同预算强基线 | [OPD基础问答](#q=opd-interview-foundations) |
| 79 | 迭代数据与目标区别 | [OPD基础问答](#q=opd-interview-foundations) |

### 奖励与过程评价（来源 80–87）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 80 | 偏好标量训练 | [奖励与评委](#q=reward-judge-interview) |
| 81 | 过程和结果单位 | [奖励与评委](#q=reward-judge-interview) |
| 82 | 过程标签来源 | [奖励与评委](#q=reward-judge-interview) |
| 83 | 奖励捷径 | [奖励与评委](#q=reward-judge-interview) |
| 84 | 评分分布漂移 | [奖励与评委](#q=reward-judge-interview) |
| 85 | 排序与校准 | [奖励与评委](#q=reward-judge-interview) |
| 86 | 隐式奖励关系 | [奖励与评委](#q=reward-judge-interview) |
| 87 | 规则反馈边界 | [奖励与评委](#q=reward-judge-interview) |

### 检索增强质量（来源 88–99）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 88 | 证据处理链 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 89 | 检索增强层次 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 90 | 表示与召回评测 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 91 | 结构切块 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 92 | 混合召回 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 93 | 候选重排 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 94 | 证据位置影响 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 95 | 检索故障定位 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 96 | 外部知识与参数更新 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 97 | 分层任务指标 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 98 | 领域语料来源 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 99 | 风险验证模块 | [RAG质量与工程](#q=rag-vs-finetuning) |

### 生成式评委（来源 100–107）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 100 | Rubric与偏差 | [奖励与评委](#q=reward-judge-interview) |
| 101 | 回答顺序控制 | [奖励与评委](#q=reward-judge-interview) |
| 102 | 绝对与成对评分 | [奖励与评委](#q=reward-judge-interview) |
| 103 | 人审一致性 | [奖励与评委](#q=reward-judge-interview) |
| 104 | 个人评价协议 | [奖励与评委](#q=reward-judge-interview) |
| 105 | 风险评分补偿 | [奖励与评委](#q=reward-judge-interview) |
| 106 | 评测任务口径 | [奖励与评委](#q=reward-judge-interview) |
| 107 | 评委与训练评分器 | [奖励与评委](#q=reward-judge-interview) |

### 推理引擎（来源 108–113）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 108 | 推理资源管理 | [推理引擎](#q=infra-inference-engine) |
| 109 | KV物理布局 | [推理引擎](#q=infra-inference-engine) |
| 110 | 请求持续批处理 | [推理引擎](#q=infra-inference-engine) |
| 111 | 后端条件比较 | [推理引擎](#q=infra-inference-engine) |
| 112 | 服务容量约束 | [推理引擎](#q=infra-inference-engine) |
| 113 | 个人压测配置 | [推理引擎](#q=infra-inference-engine) |

### 训练资源与数值（来源 114–121）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 114 | 浮点范围 | [训练与显存](#q=infra-training-step) |
| 115 | 批量梯度累计 | [训练与显存](#q=infra-training-step) |
| 116 | 激活重算 | [训练与显存](#q=infra-training-step) |
| 117 | 状态分片阶段 | [ZeRO与DeepSpeed](#q=infra-zero-deepspeed) |
| 118 | 分片后端 | [训练后端](#q=infra-training-backends) |
| 119 | 非有限值诊断 | [训练与显存](#q=infra-training-step) |
| 120 | 显存组成 | [训练与显存](#q=infra-training-step) |
| 121 | 个人资源账本 | [训练与显存](#q=infra-training-step) |

### 应用风险（来源 122–129）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 122 | 风险目标与责任 | [模型服务化](#q=model-service-interview) |
| 123 | 误拒与漏检 | [模型服务化](#q=model-service-interview) |
| 124 | 输入侧行为约束 | [模型服务化](#q=model-service-interview) |
| 125 | 输出语义审计 | [模型服务化](#q=model-service-interview) |
| 126 | 领域风险测试 | [模型服务化](#q=model-service-interview) |
| 127 | 证据适用范围冲突 | [模型服务化](#q=model-service-interview) |
| 128 | 安全反馈数据 | [模型服务化](#q=model-service-interview) |
| 129 | 用途边界 | [模型服务化](#q=model-service-interview) |

### 个人项目证据（来源 130–144）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 130 | 任务到产物链 | [项目展示答辩](#q=opd-project-defense) |
| 131 | 模型选择依据 | [项目展示答辩](#q=opd-project-defense) |
| 132 | 真实示范来源 | [项目展示答辩](#q=opd-project-defense) |
| 133 | 真实偏好来源 | [项目展示答辩](#q=opd-project-defense) |
| 134 | 评价维度理由 | [项目展示答辩](#q=opd-project-defense) |
| 135 | 故障和幻觉复盘 | [项目展示答辩](#q=opd-project-defense) |
| 136 | 检索验证设计 | [项目展示答辩](#q=opd-project-defense) |
| 137 | 相对与绝对变化 | [项目展示答辩](#q=opd-project-defense) |
| 138 | 下一轮实验 | [项目展示答辩](#q=opd-project-defense) |
| 139 | 工程与研究差别 | [项目展示答辩](#q=opd-project-defense) |
| 140 | 个人算法贡献 | [项目展示答辩](#q=opd-project-defense) |
| 141 | 模型练习收获 | [项目展示答辩](#q=opd-project-defense) |
| 142 | 方法比较 | [项目展示答辩](#q=opd-project-defense) |
| 143 | 岗位职责 | [项目展示答辩](#q=opd-project-defense) |
| 144 | 研究方向 | [项目展示答辩](#q=opd-project-defense) |

### 代码白板（来源 145–152）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 145 | 稳定注意力 | [Attention手写](#q=causal-attention-coding) |
| 146 | 分头张量变换 | [Attention手写](#q=causal-attention-coding) |
| 147 | Shift与忽略标签 | [因果LM训练](#q=causal-lm-training) |
| 148 | 偏好margin代码 | [DPO精读](#paper=paper-dpo) |
| 149 | 低秩支路 | [LoRA与QLoRA](#q=lora-finetuning) |
| 150 | 完整更新窗口 | [训练与显存](#q=infra-training-step) |
| 151 | 相邻合并 | [Attention手写](#q=causal-attention-coding) |
| 152 | 旋转或距离偏置 | [RoPE原理](#q=rope-position) |

### 服务与模型产物（来源 153–160）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 153 | 产物发布链 | [模型服务化](#q=model-service-interview) |
| 154 | 调用契约 | [模型服务化](#q=model-service-interview) |
| 155 | 协议选择 | [模型服务化](#q=model-service-interview) |
| 156 | 版本与回滚 | [模型服务化](#q=model-service-interview) |
| 157 | 知识和权重兼容 | [模型服务化](#q=model-service-interview) |
| 158 | 服务故障语义 | [模型服务化](#q=model-service-interview) |
| 159 | Adapter身份 | [模型服务化](#q=model-service-interview) |
| 160 | 上线回归 | [模型服务化](#q=model-service-interview) |

### Serving配置与优化（来源 161–168）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 161 | 运行配置影响 | [推理引擎](#q=infra-inference-engine) |
| 162 | 多维压测 | [推理引擎](#q=infra-inference-engine) |
| 163 | 静态与动态adapter | [推理引擎](#q=infra-inference-engine) |
| 164 | KV block取舍 | [推理引擎](#q=infra-inference-engine) |
| 165 | 部署容量估算 | [推理引擎](#q=infra-inference-engine) |
| 166 | Prefill与Decode调度 | [推理引擎](#q=infra-inference-engine) |
| 167 | 前缀重用 | [推理引擎](#q=infra-inference-engine) |
| 168 | 引擎条件对照 | [推理引擎](#q=infra-inference-engine) |

### 请求性能（来源 169–174）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 169 | 首响应时间 | [模型服务化](#q=model-service-interview) |
| 170 | 后续token间隔 | [模型服务化](#q=model-service-interview) |
| 171 | 有效生成吞吐 | [模型服务化](#q=model-service-interview) |
| 172 | 全请求分位数 | [模型服务化](#q=model-service-interview) |
| 173 | 到达负载与饱和 | [模型服务化](#q=model-service-interview) |
| 174 | 流式感知 | [模型服务化](#q=model-service-interview) |

### RAG关键路径（来源 175–182）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 175 | 阶段时序拆解 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 176 | 向量索引瓶颈 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 177 | 重排成本控制 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 178 | 生成负载 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 179 | 慢依赖与尾延迟 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 180 | 召回预算 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 181 | 并行分支 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 182 | Trace定位 | [RAG质量与工程](#q=rag-vs-finetuning) |

### RAG并行和批量（来源 183–188）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 183 | 独立召回并发 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 184 | 依赖与推测执行 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 185 | 部分失败处理 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 186 | 离线批量与在线等待 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 187 | 请求合批 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 188 | 流水线就绪 | [RAG质量与工程](#q=rag-vs-finetuning) |

### RAG缓存（来源 189–193）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 189 | 缓存对象分层 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 190 | 语义答案复用 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 191 | 内容表示缓存 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 192 | 版本主动失效 | [RAG质量与工程](#q=rag-vs-finetuning) |
| 193 | 权限与个性化隔离 | [RAG质量与工程](#q=rag-vs-finetuning) |

### 路由与优先级（来源 194–198）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 194 | Fast与Full路径 | [模型服务化](#q=model-service-interview) |
| 195 | 复杂度和风险估计 | [模型服务化](#q=model-service-interview) |
| 196 | 高风险人工升级 | [模型服务化](#q=model-service-interview) |
| 197 | 高风险验证流程 | [模型服务化](#q=model-service-interview) |
| 198 | 路由可审计性 | [模型服务化](#q=model-service-interview) |

### 分布式训练（来源 199–203）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 199 | 梯度同步时机 | [多卡并行](#q=infra-distributed-parallelism) |
| 200 | 状态生命周期 | [训练后端](#q=infra-training-backends) |
| 201 | 分片通信 | [ZeRO与DeepSpeed](#q=infra-zero-deepspeed) |
| 202 | 后端参数契约 | [ZeRO与DeepSpeed](#q=infra-zero-deepspeed) |
| 203 | Node与Rank启动 | [多卡并行](#q=infra-distributed-parallelism) |

### 分布式Serving（来源 204–207）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 204 | 模型并行与副本 | [模型服务化](#q=model-service-interview) |
| 205 | 同请求协作 | [模型服务化](#q=model-service-interview) |
| 206 | 缓存和负载取舍 | [模型服务化](#q=model-service-interview) |
| 207 | Ready与健康探针 | [模型服务化](#q=model-service-interview) |

### 观测与告警（来源 208–212）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 208 | 质量与资源metrics | [模型服务化](#q=model-service-interview) |
| 209 | 请求版本追踪 | [模型服务化](#q=model-service-interview) |
| 210 | 输出证据日志 | [模型服务化](#q=model-service-interview) |
| 211 | 采集与基数 | [模型服务化](#q=model-service-interview) |
| 212 | SLO和风险告警 | [模型服务化](#q=model-service-interview) |

### 发布策略（来源 213–215）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 213 | 分阶段切流 | [模型服务化](#q=model-service-interview) |
| 214 | 双环境与灰度 | [模型服务化](#q=model-service-interview) |
| 215 | 版本组合回滚 | [模型服务化](#q=model-service-interview) |

### 故障与风险控制（来源 216–218）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 216 | 语义风险检测 | [模型服务化](#q=model-service-interview) |
| 217 | 依赖熔断 | [模型服务化](#q=model-service-interview) |
| 218 | 明确降级 | [模型服务化](#q=model-service-interview) |

### 敏感应用服务（来源 219–222）

| 来源编号 | 本文自测知识点 | 合并后的学习入口 |
| --- | --- | --- |
| 219 | 敏感信息最小化 | [模型服务化](#q=model-service-interview) |
| 220 | 用户缓存隔离 | [模型服务化](#q=model-service-interview) |
| 221 | 风险请求调度 | [模型服务化](#q=model-service-interview) |
| 222 | 反馈到数据版本 | [模型服务化](#q=model-service-interview) |

## 6. 学习顺序与项目串联

第一轮基础/模板→SFT/LoRA→偏好与RL→OPD，第二轮评分/评测→RAG→Infra/服务，第三轮用ESCD和turn项目回答方法/实现/证据问题。每轮遮住答案讲一条流程、写一个公式或例子，再看源码。无需按222条线性背完，优先解决与你方向有关的机制。

ESCD专题强调公共事件、root/child、集合梯度与原报告指标；turn专题强调sealed response、真实反馈、same-version critic pair、窗口分母与request-boundary发布。医学项目例子转成权限、证据、风险路由和治理练习，技术设计不能当临床决策经验。

## 7. 面试问答与使用自检

**Q01 为什么实际222而不是100？** 按本次正文编号核对，网址/导航标题保留历史值。

**Q02 是否重复制了222个答案？** 逐编号映射到合并后的专题，原理/代码和项目讲解在正文展开。

**Q03 原有内容怎么合并？** 保留原ID与入口，LoRA/RAG扩写；算法与Infra用现有深文，不同版本互相链接。

**Q04 链接命中等于会了吗？** 还要直答、手算、代码与边界，项目配置需真实证据。

**Q05 源作者数据数量能用吗？** 不能当自己的，替换为实际模型、数据、训练/评测记录。

**Q06 原题中的前提都对吗？** 题面也要审查，条件性现象和普遍结论分开。

**Q07 重点怎么排？** 你的后训练/ESCD/Agentic主线优先，服务与风险按岗位补课。

**Q08 之后如何迭代？** 保留source hash与编号，源更新先查新增/重复/错误，再补对应正文，不复制新大杂烩。

## 8. 来源、映射下载与边界

[原清单](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)作为题型范围，不作为技术事实唯一依据；未找到可确认开放内容license，因此不转载完整问题正文。本文自写知识标签、解释和答案，各知识文引用primary来源。[机器可读映射](./assets/learning/post-training-checklist-map/mapping.json)含源网页hash、222个编号和本站入口，方便下轮合并检查。

本轮不自动收集用户学习进度，不把模板配置/纸面成绩当个人实验。参考来源、固定源码、图表和CPU验证仍沿既有专题工作流。
