---
id: post-training-interview-practice
title: "后训练面试题库：公司题型、回答与项目追问"
category: guide
difficulty: 进阶
tags: ["P0", "后训练面试", "SFT", "PPO", "DPO", "GRPO", "OPD"]
updated: "2026-10-23"
summary: "首批 20 道主问题：五组学习结构，逐题口述、白板、追问、易错点和原理拆解；按公司查题源，按概念维护，链接既有原理笔记。"
draft: false
---

## 问题背景：从会看笔记，到能在面试里讲清楚

本轮面向大模型后训练岗位，先准备 20 道主问题。覆盖数据、目标、梯度、训练故障、系统与个人项目；一个主问题下的追问不另算新题。不是以题量替代理解，而是让每个回答都能继续讲到公式、代码、日志和边界。

已有的[后训练知识提纲](#q=post-training-interview)、[RL 算法讲义](#q=rl-policy-optimization-map)和[OPD 详细题库](#q=opd-interview-foundations)继续作为原理层。这里按主问题组织回答：先给可以口述的结论，再展开推导、数据流、实现条件和故障分析。每题包含独立理解所需的关键原理；更完整的专题背景通过关联笔记继续查阅。

## 统一回答框架：先短答，再接受追问

每题有六个固定模块，再附题源、关联笔记和前后题导航。

| 模块 | 练到什么程度 |
| --- | --- |
| 60–90 秒回答 | 先给结论，接机制与条件，最后说怎么验证；不用逐字背诵 |
| 原理拆解 | 从目标推到梯度与数据流，解释方法为何有效、在哪些条件下失效 |
| 白板与例子 | 写对对象、条件、分布、mask 和归约；用小例子检查 |
| 追问怎么接 | 面试官换一个前提后，仍能解释自己的选择 |
| 项目怎么联系 | 将方法映射到真实负责的模块或实验，缺证据时明确说是假设 |
| 容易答错的点 | 排除看似熟悉但关键条件错误的口号 |

练习时可用“任务与反馈 → 数据和模型角色 → 目标/数据流 → 工程约束 → 对照证据”的顺序。不是每题都要讲完五层，而是被追问时能继续向下走。

## 面试题目录

[//]: # (interview-generated-start)

**当前收录：20 道主问题。** 追问属于同一主问题，不重复计数。

### 01 / SFT、数据与偏好基础

| 题号 | 主问题 | 题源类型 |
| --- | --- | --- |
| 01 | [SFT、DPO、PPO、GRPO 怎么选，什么时候进入 RL？](#q=pt-method-selection) | 美团 / 阿里 · 改写 |
| 02 | [SFT 数据怎么筛、怎么配比，loss 应该算在哪些 token？](#q=pt-sft-data) | 美团 / 阿里 · 改写 |
| 03 | [全参、LoRA、QLoRA 怎么选，量化后为什么还会 OOM？](#q=pt-lora-budget) | 字节 / 美团 · 改写 |
| 04 | [奖励模型和偏好数据怎么做，如何验证标注真的可靠？](#q=pt-reward-preference-data) | 阿里 · 改写 |

### 02 / PPO、GAE、KL 与 DPO

| 题号 | 主问题 | 题源类型 |
| --- | --- | --- |
| 05 | [PPO 的四个模型各做什么，完整训练一步怎么走？](#q=pt-ppo-four-models) | 美团 / 阿里 · 改写 |
| 06 | [Critic 已经估值了，为什么还要 GAE？终止和超时怎么处理？](#q=pt-gae-credit) | 字节 · 改写 |
| 07 | [PPO clip 和 KL 是一回事吗，KL 到底怎么估计？](#q=pt-clip-kl) | 字节 / 百度 · 改写 |
| 08 | [DPO 的四个概率怎么推导，β 和输出变长有什么关系？](#q=pt-dpo-derivation) | 字节 / 百度 · 改写 |

### 03 / GRPO、变体与奖励

| 题号 | 主问题 | 题源类型 |
| --- | --- | --- |
| 09 | [写出 GRPO 目标：不训练 critic，优势从哪里来？](#q=pt-grpo-objective) | 腾讯 / 美团 · 改写 |
| 10 | [一组回答奖励全 0 或全 1 怎么办，采样越多越好吗？](#q=pt-zero-variance-groups) | 快手 · 改写 |
| 11 | [DAPO、Dr. GRPO、GSPO 分别改了什么，怎样比较才公平？](#q=pt-grpo-variants) | 字节 · 改写 |
| 12 | [RLVR 奖励怎么设计，reward 上涨为什么任务可能更差？](#q=pt-rlvr-hacking) | 阿里 · 改写 |

### 04 / 训练系统与多轮 Agent

| 题号 | 主问题 | 题源类型 |
| --- | --- | --- |
| 13 | [RL 训练突然复读、熵坍缩或 KL 飙升，按什么顺序排查？](#q=pt-rl-collapse) | 阿里 / 百度 · 改写 |
| 14 | [7B 后训练要多少显存，DDP、FSDP、ZeRO 分别省什么？](#q=pt-memory-parallel) | 百度 / 字节 · 改写 |
| 15 | [Rollout 为什么拖慢训练，异步之后如何控制策略滞后？](#q=pt-rollout-staleness) | 字节 / 阿里 · 改写 |
| 16 | [多轮 Agent 的 reward 怎么分到 turn 和 token，工具返回要不要训练？](#q=pt-agent-credit-mask) | 字节 / 快手 · 改写 |

### 05 / OPD 与项目答辩

| 题号 | 主问题 | 题源类型 |
| --- | --- | --- |
| 17 | [OPD 与教师 SFT、普通 KD、RL 有什么区别，sampled KL 怎么求梯度？](#q=pt-opd-gradient) | 项目自拟 |
| 18 | [Cross-Tokenizer OPD 为什么不能直接对齐 logits，ESCD 解决哪一步？](#q=pt-escd-events) | 项目自拟 |
| 19 | [Math、Code、IF 任务合版怎么讲，如何发现负迁移和能力遗忘？](#q=pt-multi-task-distillation) | 项目自拟 |
| 20 | [怎样用三分钟讲清后训练项目，并证明收益来自你的设计？](#q=pt-project-evidence) | 字节 / 腾讯 · 改写 |

## 公司题型怎么查

以下是所选公开帖中的题型映射，不代表某公司的完整题库或出现频率。原帖中的错误技术表述没有沿用。

| 帖子标注的公司 | 公开记录涉及的方向 | 对应练习 | 证据范围 |
| --- | --- | --- | --- |
| [字节 · 字节算法面试记录（2024）](https://www.nowcoder.com/feed/main/detail/30c5b57a0acb4a0f9d2250cafadf94b1) | LoRA、PPO、优化效果与项目分工 | [Q03](#q=pt-lora-budget)、[Q20](#q=pt-project-evidence) | 经历自述，招聘真实性未独立核验；公开正文 |
| [字节 · 字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 优势与 KL、DAPO、显存并行、长尾 rollout、过程反馈及消融 | [Q06](#q=pt-gae-credit)、[Q07](#q=pt-clip-kl)、[Q08](#q=pt-dpo-derivation)、[Q11](#q=pt-grpo-variants)、[Q14](#q=pt-memory-parallel)、[Q15](#q=pt-rollout-staleness)、[Q16](#q=pt-agent-credit-mask)、[Q20](#q=pt-project-evidence) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月 |
| [阿里 · 阿里国际「国际一面」（2025）](https://www.nowcoder.com/feed/main/detail/166d1f3bb6b84624b912dcc3997f8081) | 方法选型、PPO 模型角色和 critic/reward 的损失 | [Q01](#q=pt-method-selection)、[Q04](#q=pt-reward-preference-data)、[Q05](#q=pt-ppo-four-models) | 经历自述，招聘真实性未独立核验；检索可见正文；直链读取返回简页 |
| [阿里 · 阿里算法岗公开题型汇总](https://www.nowcoder.com/discuss/923309531445071872) | SFT 数据、GRPO、奖励设计、质量达标和策略版本滞后 | [Q01](#q=pt-method-selection)、[Q04](#q=pt-reward-preference-data)、[Q12](#q=pt-rlvr-hacking)、[Q15](#q=pt-rollout-staleness) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 7 月 |
| [阿里 · 阿里国际 AI 算法公开问答](https://www.nowcoder.com/feed/main/detail/21fe1ff2b5de4c6fa2a56a27b54f3d3c) | 数据筛选、奖励作弊、训练质量与策略退化 | [Q02](#q=pt-sft-data)、[Q12](#q=pt-rlvr-hacking)、[Q13](#q=pt-rl-collapse) | 带参考答案的公开整理，非技术权威；公开正文；页面仅显示月日，不推断年份 |
| [腾讯 · 腾讯技术研究 NLP 面经（2025）](https://www.nowcoder.com/feed/main/detail/c8ddab6cf8ea467c9e4003c9256616cb) | GRPO 公式、方法比较、论文讲解及项目参数 | [Q09](#q=pt-grpo-objective)、[Q20](#q=pt-project-evidence) | 经历自述，招聘真实性未独立核验；检索可见正文；直链读取返回简页 |
| [美团 · 美团算法一面面经（2025）](https://www.nowcoder.com/feed/main/detail/7197e4d2e8934c76a3bc65aedaf1afeb) | SFT 后为何 RL、PPO 四模型、QLoRA、数据多样性 | [Q01](#q=pt-method-selection)、[Q02](#q=pt-sft-data)、[Q03](#q=pt-lora-budget)、[Q05](#q=pt-ppo-four-models)、[Q09](#q=pt-grpo-objective) | 经历自述，招聘真实性未独立核验；公开正文 |
| [快手 · 快手大模型推荐算法一面分享](https://www.nowcoder.com/feed/main/detail/327c7fee6b5b4a8e92f318b317eff650) | SFT 质量、奖励设计、全零或全一组反馈 | [Q10](#q=pt-zero-variance-groups) | 公开分享，未独立核验经历；公开正文；页面仅显示月日，不推断年份 |
| [快手 · 快手大模型算法二面公开问答](https://www.nowcoder.com/discuss/876505573997432832) | Agent 训练流程、多轮上下文及工具观察 mask | [Q16](#q=pt-agent-credit-mask) | 公开问答整理，不按认证真题处理；仅公开片段；未读取付费部分 |
| [百度 · 百度算法岗公开题型汇总](https://www.nowcoder.com/discuss/927018114900922368) | DPO、KL、输出长度及 7B 显存和 ZeRO 分片 | [Q07](#q=pt-clip-kl)、[Q08](#q=pt-dpo-derivation)、[Q13](#q=pt-rl-collapse)、[Q14](#q=pt-memory-parallel) | 二次汇总，不按公司认证真题处理；页面公开内容与检索摘要；未读取付费部分 |

## 来源账本

本轮核对：2026-10-23。面经只支持题型归类；技术解释依据原论文、作者资料与独立推导。未确认年份的日期不自行补全年份。

| 资料 | 用途 |
| --- | --- |
| [PPO 原论文](https://arxiv.org/abs/1707.06347v2) | clipped surrogate、旧策略与更新约束 |
| [GAE 原论文](https://arxiv.org/abs/1506.02438v6) | TD residual、优势估计与偏差方差 |
| [DPO 原论文](https://arxiv.org/abs/2305.18290v3) | 偏好模型、KL 正则最优策略与 DPO loss |
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300v3) | 组相对优势、策略目标及 outcome/process 监督 |
| [DAPO 原论文](https://arxiv.org/abs/2503.14476v2) | 非对称裁剪、动态采样、token 归约与超长样本处理 |
| [Dr. GRPO 分析论文](https://arxiv.org/abs/2503.20783v2) | 组标准差、长度归一与优化偏差 |
| [GSPO 原论文](https://arxiv.org/abs/2507.18071v2) | 长度归一的序列 ratio 与序列裁剪 |
| [LoRA 原论文](https://arxiv.org/abs/2106.09685v2) | 低秩增量、训练参数及合并 |
| [QLoRA 原论文](https://arxiv.org/abs/2305.14314) | 4-bit 基础权重、量化常数及 adapter 训练 |
| [ZeRO 原论文](https://arxiv.org/abs/1910.02054v3) | 优化器、梯度和参数分片的区别 |
| [GKD / On-Policy Distillation](https://arxiv.org/abs/2306.13649v3) | 学生生成分布与教师反馈的蒸馏目标 |
| [LLM-as-a-Judge / MT-Bench](https://arxiv.org/abs/2306.05685v4) | 位置、长度及自我偏好等裁判偏差 |
| [IMPALA / V-trace](https://arxiv.org/abs/1802.01561v3) | actor–learner 解耦和 off-policy 校正的经典依据 |
| [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782v4) | 任务梯度干扰的分析；不是本人已使用方法的声明 |
| [TIES-Merging](https://arxiv.org/abs/2306.01708v2) | 区分权重合并与联合训练；不是本人已使用方法的声明 |
| [TRL SFT 官方文档](https://huggingface.co/docs/trl/en/sft_trainer) | 数据格式、assistant-only loss、packing 与配置边界 |
| [TRL DPO 官方文档](https://huggingface.co/docs/trl/en/dpo_trainer) | 偏好格式、reference log-prob 和训练实现 |
| [verl 性能调优文档](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) | rollout、batch、packing 和训练/推理内存取舍；参数支持随版本核对 |
| [Thinking Machines：On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | 学生轨迹、教师 token 评分与反向 KL 蒸馏 |
| [ESCD 公开实现与评测说明](https://github.com/Tammytcl/ESCD) | 事件补全、公开 baseline、数据和独立 PHYRD-40 评测的证据边界 |

[//]: # (interview-generated-end)

## 项目证据：公开页面不替你编造实验经历

准备 Cross-Tokenizer OPD/ESCD、任务能力合版与 Agentic OPD 三张卡即可。卡上记录真实任务、角色、个人贡献、基线、指标、预算、失败样例与公开范围。这里的口述是练习模板，具体数值由你的记录补充；教学算例、作者论文结果、模拟 API、CPU 检查和真实训练结果分别对待。

对 ESCD，能公开核查的小规模实现、数据和评测入口，与未发布的大规模训练 recipe 分开说明。对 Agentic OPD，把异步 Infra 与在线打分的实验依据分别讲清，不把方法假设当成已经证明的收益。
