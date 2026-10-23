---
id: "pt-sft-data"
title: "Q02 · SFT 数据怎么筛、怎么配比，loss 应该算在哪些 token？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "SFT 与数据", "阿里", "美团"]
updated: "2026-10-23"
summary: "将数据质量、任务配比、训练目标和模板实现串起来，避免只回答“清洗、去重、增加多样性”。"
draft: false
---

## 60–90 秒回答

我会先定义能力目标和评测切分，再按来源、任务、难度、长度和正确性审计数据。训练与验证至少按题目、代码仓库或相关来源分组去重，防止近重复泄漏。配比不能只看条数，长回答在 token-mean 训练中可能贡献更多梯度，需要同时看有效 token 和分域效果。实现上用原生 chat template，明确监督全部 assistant 轮还是末轮：用户输入和工具观察通常作为上下文可见，但不参与生成 loss。最后做逐样本可视化、少量过拟合检查和分域验证，确认数据、mask、EOS 与目标一致。

## 原理拆解

### 从一条多轮样本到训练张量

先用与部署一致的 chat template 序列化 system/user/assistant/tool，再分词成 `input_ids`。构造 labels 时只保留目标生成区间，其余位置设为忽略值；causal shift 必须在模型或数据管线中的一处完成，避免错移或移两次。比如序列是 `用户 … assistant标记 答案 EOS`，预测第一个答案 token 的 logits 来自它前一个位置。若结束标记是部署时由模型预测的动作，也应按目标设置纳入监督，否则容易学会内容却不会停。

Loss mask 决定哪个预测参与优化，attention mask 决定当前位置能读什么，两者用途不同。Tool 返回不计算生成 loss，但后续回答通常必须读到它。用逐 token 打印的方式检查一个真实样本，比只看平均 loss 更容易发现 assistant 边界错位。TRL 的 assistant-only 设置还依赖模板能返回正确的生成区间，不能只开配置就默认正确。[SFT Trainer 文档](https://huggingface.co/docs/trl/en/sft_trainer)

### 数据比例为何会变成另一种梯度比例

若任务域 $k$ 的抽样概率为 $p_k$、平均有效长度为 $\bar T_k$，在全局 token-mean 且其他尺度相近时，该域的 token 占比近似为 $p_k\bar T_k/\sum_jp_j\bar T_j$。因此 Math 与 IF 即使样本数相等，较长的 Math 也可能占据大部分监督。若改成每样本先平均，单条长短样本的总系数更接近，但每个 token 的权重随长度改变。两种归约都能被明确定义，关键是与能力目标一致。

梯度累积也要保留这个含义：两个 microbatch 分别有 100 与 900 个有效 token，分别算均值再各乘一半，并不等于 1000 token 的统一均值。应累计 loss 的分子与有效 token 数，结合分布式梯度平均方式计算正确缩放；不同框架的默认行为需实际核对。

### 清洗应当围绕可验证的失真

重复问题按规范化题面、来源和语义近邻一起排查；代码任务按仓库/版本分组，避免相邻修复进入训练与测试。正确性过滤记录每域保留率，防止难例被几乎清空；指令跟随检查约束是否真的可满足；多轮轨迹保留导致修复的错误反馈。Packing 时说明独立样本是否使用分块因果 attention，不能假定插入 EOS 就隔离了可见上下文。最后用少量样本过拟合确认数据链路，再看独立分域验证，分别定位实现失败与泛化失败。

## 白板与例子

设有效目标 mask 为 $m_t$，因果监督是：

$$
L_{\mathrm{SFT}}=-\frac{\sum_t m_t\log\pi_\theta(x_{t+1}\mid x_{\leq t})}{\sum_t m_t}.
$$

这是 token 平均的一种写法。序列平均会改变长短样本权重，不能当作同一训练目标。

| 消息或 token | 可被后续 attention 看到 | 典型生成 loss |
| --- | --- | --- |
| 用户需求 | 是 | 0 |
| assistant 生成的工具名和参数 | 是 | 1 |
| 外部工具返回 | 是 | 0 |
| assistant 基于反馈给出的修复 | 是 | 1 |
| padding | 按 padding mask 处理 | 0 |

实际取值还取决于控制 token 是否由模型生成，以及模板能否正确标注 assistant 区间。Packing 也必须明确跨样本 attention、position IDs、EOS 与 loss 边界。

## 追问怎么接

**只保留 teacher 做对的数据好吗？** 正确性筛选适合示范质量控制，但会改变难度和领域分布；看过滤保留率，重新审计覆盖。对 OPD 而言，失败访问状态还可能承载恢复监督。

**Math:Code:IF 按 1:1:1 就均衡了吗？** 不一定。还要考虑 token 长度、损失尺度、难度、采样重复率与关键域的退化约束。先给基线配比，再用分域验证调整。

**训练 loss 降、任务效果不升？** 对比独立验证、模板和解码一致性；检查标签错误、泄漏、格式记忆、停止行为及高频域压过长尾域。不要先宣布“学习率不对”。

## 项目怎么联系

你的 SciDeriv 是 prompt-only 数据，不能拿空标签直接当作常规有答案的 SFT 数据。介绍时应分开说明 prompt 来源、谁生成监督、如何筛选，以及 PHYRD-40 评测参考从未进入生成输入。合版项目可补充真实的各域采样和有效 token 统计。

## 容易答错的点

- 将 loss mask 与 attention mask 混为一谈。
- 在重建多轮样本时删掉报错观察，却保留原修复答案。
- 认为随机按行切分天然消除了同源或同仓库污染。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [美团算法一面面经（2025）](https://www.nowcoder.com/feed/main/detail/7197e4d2e8934c76a3bc65aedaf1afeb) | 经历自述，招聘真实性未独立核验；公开正文；仅作题型依据 |
| [阿里国际 AI 算法公开问答](https://www.nowcoder.com/feed/main/detail/21fe1ff2b5de4c6fa2a56a27b54f3d3c) | 带参考答案的公开整理，非技术权威；公开正文；页面仅显示月日，不推断年份；仅作题型依据 |
| [TRL SFT 官方文档](https://huggingface.co/docs/trl/en/sft_trainer) | 数据格式、assistant-only loss、packing 与配置边界 |

**相关专题详解：** [LLM 训练能并行，为什么生成还要逐 token？](#q=causal-lm-training) · [SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-method-selection) · [下一题](#q=pt-lora-budget)

[//]: # (interview-generated-end)
