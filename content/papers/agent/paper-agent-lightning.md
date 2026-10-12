---
id: paper-agent-lightning
title: "Agent Lightning：把Agent执行trace适配为RL训练数据"
paper_title: "Agent Lightning: Train ANY AI Agents with Reinforcement Learning"
authors: ["Xufang Luo", "Yuge Zhang", "Zhiyuan He", "Zilong Wang", "Siyun Zhao", "Dongsheng Li", "Luna K. Qiu", "Yuqing Yang"]
affiliations: ["Microsoft Research"]
author_affiliations: [[1], [1], [1], [1], [1], [1], [1], [1]]
venue: "arXiv preprint (fixed v1)"
year: 2025
direction: agent
areas: [language]
tasks: [agents, retrieval, reasoning, training-adaptation]
published: "2025-08-05"
paper_url: "https://arxiv.org/abs/2508.03680v1"
github_url: "https://github.com/microsoft/agent-lightning"
code_note: "论文为2025 v1；源码静态核读2026 commit 69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9，不混为原实验实现。"
method_figure: "./assets/papers/paper-agent-lightning/method-source.png"
method_caption: "Figure 3 · Agent执行trace到动作级RL样本"
evidence: 已核原文
note_ids: [agentic-rl-training, agent-components-overview, rl-policy-optimization-map]
tags: [Agent Lightning, Trace, Agentic RL, Credit Assignment, Training Agent Disaggregation]
updated: "2026-10-12"
summary: "详解训练与Agent解耦、每次LLM调用形成transition、终局奖励分配及当前rollout级统计/聚合源码；不把2026实现当2025实验。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：真实Agent程序怎样接入训练

真实agent可能由LangChain、AutoGen或自写程序编排，不只是单一“prompt→answer”。一次任务里可能多次调用模型、处理工具异常、总结记忆、分支、调用多个角色。若训练框架要求重写整个agent执行逻辑，已有产品接入成本很高；如果只把最终聊天文本拼成一行，又会丢失每次调用实际看到的上下文、角色和局部决策。

Agent Lightning提出Training–Agent Disaggregation：agent保持自己的控制流程，通过标准接口收集可训练trace，训练服务把trace适配为RL样本，再把更新模型通过API供agent使用。它主要解决执行与训练的接口，以及复杂trace的动作粒度转换。标题ANY是设计愿景，不表示任意程序无需改动就能安全、正确地训练，也不表示所有环境都有实验验证。

一个SQL agent可以有writer、checker、rewriter。作者想只优化writer与rewriter的模型调用，checker保留在执行流程。这里“只训练两角色”指选择其调用形成loss；若三角色共享同一个底层模型权重，被排除角色的行为仍可能随共享参数更新而变化，不能据角色名宣称物理冻结了独立模型。

### 前置知识：程序状态、模型输入和动作不完全相同

Agent程序完整状态包含变量、工具结果与分支；模型某次调用的输入是对这些状态的可观察表示。一次LLM响应可作为一个高层动作，内部token又可按自回归策略分解。工具执行负责状态转移，terminal reward来自最终任务评价。转换器应保留当时真实input/output，而不能训练时重新拼一份“更完整上下文”再假定行为概率一样。

Trace记录与RL数据契约也不同。可观察性系统能记录文本、latency与span，但训练还可能要求原始token、采样配置、版本、logprob与mask。本文标准接口提供桥梁；实际某个logging SDK是否包含这些字段，仍需单独核验。上下文压缩改善内存，可能同时丢失决策信息或改变状态表示，不能只描述为无代价压缩。

### 已有工作：框架、工具RL与分层动作

[PPO](https://arxiv.org/abs/1707.06347)和GRPO是底层可替换优化器；Search-R1/RAGEN等建立工具或环境训练。Agent Lightning的核心位置是把不同编排框架trace统一成可被这些算法消费的对象。它称LightningRL为hierarchical credit assignment，原始v1的基本分配规则仍是把最终结果给相关调用，再使用已有token算法；不能据“分层”一词推导出论文一定训练了独立turn价值网络或反事实过程reward。

## 2. 方法与实现机制

### 从程序trace到每调用一条transition

![Figure 3 · 标准transition与分层训练接口](./assets/papers/paper-agent-lightning/method-source.png)

**Figure 3解读。** Agent执行形成trace，转换为每次LLM调用的input、output与反馈，再进入RL算法。高层动作是一次调用响应，低层是该响应的token。拆分的意义是让每个样本对应模型当时收到的实际上下文，支持非线性程序与多角色调用。图不是“工具可反向传播”的证明，也未证明信用自动细到每个真实因果动作。图源固定v1的figures/algorithm.png、sections/2.2_alg.tex中的fig:alg。

接口记录调用metadata、input、output、reward以及相互关系。为同一任务选择可训练调用，将其转成$(o_t,a_t,R)$；$o_t$是模型实际输入，$a_t$是完整响应，$R$是任务最终结果。原v1基本规则可以表为：

$$
R_t^{\rm assigned}=R_{\rm episode},\qquad
\log\pi_\theta(a_t|o_t)=\sum_j\log\pi_\theta(a_{t,j}|o_t,a_{t,<j}).
$$

这是把终局监督落到调用，不是证明每个调用的因果贡献都相等。后续可用PPO/GRPO构造优势与clip，但“终局reward分配”“baseline统计”“token loss归一化”是三个不同步骤。需要逐步说明哪一步有组依赖，哪一步可以在turn结束时准备，哪一步仍需任务终止。

### 为什么按调用拆分有用，以及它的代价

SQL writer的prompt可能只包含题目和schema，rewriter收到错误SQL与执行报错，checker看到另一份结构化输入。若把所有内容拼成一个连续token串，训练器可能不存在一个与原模型调用条件相同的概率分解。按调用保存实际输入可以避开这一误配，并直接接入已有工具。代价是同一历史可能在多个样本中重复prefill，长轨迹产生更多训练行，需处理长度权重。

论文比较不依赖复杂“整条拼接trace上选择动作段”的接线，但不能将其解读为模型不需要任何mask：padding、prompt/response、角色筛选仍要正确处理。动作loss mask也不等于删除观察或改变position IDs；不能把某一种错误拼接实现的问题概括为所有loss masking都会破坏RoPE。

![Figure 4 · 执行client、训练server与模型API](./assets/papers/paper-agent-lightning/architecture-source.png)

**Figure 4解读。** 左侧训练后端维护trainer/LLM engine，右侧agent由既有框架执行，中间server/client桥接。上方箭头是更新模型API，下方是带reward的trace回流。两条链解耦有利于环境和GPU独立扩缩，但不自动定义异步算法校正：推送权重时机、旧数据版本、partial resume仍需具体训练器实现。原图含框架举例，不代表这些框架的任意版本都无需适配。图源figures/agent_lightning_architecture_v3.png、fig:agent_lightning_architecture。

### 原论文与当前代码必须按时间分开

下面静态核读微软官方仓库commit69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9的2026实现，帮助理解现代接线。它不是固定2025 v1实验版本，不能据此声称论文当时用了CISPO或新增聚合模式。另有2026 Agent Lightning v1.0论文，也与这里2025 arXiv v1是不同论文。

[rollout_level_advantage.py](https://github.com/microsoft/agent-lightning/blob/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9/agentlightning/verl/rollout_level_advantage.py#L42)先按rollout ID分组，每个episode挑一个代表行，并验证同任务uid和相同reward：

```python
# Official excerpt; relevant representative-row construction.
reward_sums = token_level_rewards.sum(dim=-1).detach().float()
representative_indices: list[int] = []
for rollout_id, row_indices in rollout_to_indices.items():
    representative_indices.append(row_indices[0])
    _validate_same_uid(rollout_id, row_indices, uid_values)
    _validate_same_reward(rollout_id, row_indices, reward_sums)
```

再在代表行上计算优势，将每rollout标量广播给对应turn的response mask。这个设计避免长episode有更多turn、在组reward统计中被重复计权。调用结果只有返回值“看起来相同”还不够，代码明确验证reward一致，接口错接时应该报错而不是悄悄训练。

另一方面，组统计去重不自动保证loss也按episode等权。[per_rollout_loss.py](https://github.com/microsoft/agent-lightning/blob/69f5a40b36d9bff646b8e7a5532e8b78bd41eaf9/agentlightning/verl/per_rollout_loss.py#L16)的normalize_advantages_by_rollout对每rollout汇总所有有效token，构造行除数：

```python
# Official excerpt after counting tokens per rollout.
row_divisors = torch.tensor(
    [rollout_token_counts[rollout_id] * num_trained_rows for rollout_id in rollout_ids],
    dtype=advantages.dtype,
    device=advantages.device,
).clamp_min(1.0)
return advantages / row_divisors.unsqueeze(-1)
```

其loss路径再按mask求和，并处理DP/global batch缩放。要沿完整代码理解num_trained_rows与最终reduce的关系，不能只抓“除token数”就叫每episode均值。这里的优势归一与loss归一是两个独立接口，这正是面试追问“为什么长轨迹权重变大”的好例子。

### 一次训练接入最少要核哪些条件

确定任务ID/rollout ID、可训练角色、调用时模型版本、实际prompt、生成token、工具状态与终止反馈；转换器不能把评分时新上下文当rollout原输入。若只留下自然语言日志，没有采样token与行为概率，就不能直接做需要ratio的更新。共享模型角色选择只是mask选择，不是独立权重隔离。

同一个最终reward赋给10次调用，会产生10条训练行。如果逐行平均，长episode可能比短episode更重；若按token平均，长响应也更重。应该结合训练目标选择turn、token或episode权重，并报告选择。框架抽象让接入容易，不替算法和实验设计作决定。

## 3. 实验设置与算力

原始v1验证三种不同agent框架与工具，模型均为Llama3.2-3B-Instruct。实验展示训练趋势，未给统一的跨框架SOTA排序或全面算法消融。数据库、retriever和calculator对应不同环境，不能只报一个平均reward把三任务混合。

![Table 1 · 三任务的框架、工具与训练角色](./assets/papers/paper-agent-lightning/table-1-pdf.png)

**Table 1解读。** SQL用LangChain/Spider/SQL executor，三个agent角色、训练两个；QA用OpenAI Agents SDK/MuSiQue/Wikipedia retriever，一角色；Math用AutoGen/Calc-X/calculator，一角色。表是实验设置账本，不是性能表，也不是“任意框架”兼容性穷举。原表第11页局部裁图保留全部配置行。

| 项目 | 固定2025 v1披露 |
| --- | --- |
| 基础模型 | Llama3.2-3B-Instruct |
| SQL数据/工具 | Spider，SQL executor |
| SQL角色 | writer、checker、rewriter，优化writer与rewriter调用 |
| SQL reward | 最终执行正确性 |
| QA数据/工具 | MuSiQue，Wikipedia retrieval |
| QA索引 | 论文称约21M documents，BGE cosine retrieval；不解释为21M完整文章 |
| QA reward | 0.9 word F1 + 0.1 format |
| Math数据/工具 | Calc-X，calculator |
| Math reward | 最终答案正确性 |
| 高层信用 | 基本分配终局任务reward至LLM调用，再接token RL |
| 框架 | 训练与agent解耦，标准trace转换，示例接verl |
| GPU预算缺口 | 固定v1未提供足以对齐三任务的完整GPU SKU/卡时、LR与batch账本；不以2026代码默认值补齐 |

Spider的数据集概览约10k问题、200DB、138领域，不等于本次训练实际使用每一条的数量。训练/测试曲线横轴是steps而非统一GPU-hours，SQL、QA、Math的执行和模型调用量也不同。v1没有足够信息给出今天的最低复现卡数或全任务同预算比较，需公开配置或实验日志补充。

可观察性方案包含OpenTelemetry/AgentOps或API日志接入，便利数据采集；具体工具日志是否含token-level behavior字段需自己验证。这里只静态核读官方代码，不运行SDK、不执行agent模型训练。共享CPU实验检查聚合差异，但不是原论文成功率复现。

## 4. 结果与图表解读

![Figure 5 · Spider训练与测试reward曲线](./assets/papers/paper-agent-lightning/sql-source.png)

**Figure 5解读。** 左训练、右测试，横轴step、纵轴reward。训练逐步升高但波动明显，测试由约0.15进入约0.5–0.57范围。这里只按轴读趋势和近似区间，不把图片最后一个点写成带置信区间的精确benchmark。曲线无强基线或多seed阴影，支持该接入任务可改善，不证明优于所有text-to-SQL系统。图源train_spider.png与test_spider.png，按原figure顺序合并。

![Figure 6 · MuSiQue训练与测试reward曲线](./assets/papers/paper-agent-lightning/rag-source.png)

**Figure 6解读。** 测试reward从约0.04升到约0.23，训练也总体改善。这是混合F1/format奖励，不是EM成功率；不得乘100就标为23%全部任务成功。检索环境与输入框架在该实验固定，不能推导所有search agent同样收益。原图左/右分别训练与测试，纵轴范围不同，视觉高度不能跨图比绝对改进。图源train_rag_llama.png/test_rag_llama.png。

![Figure 7 · Calculator训练与测试reward曲线](./assets/papers/paper-agent-lightning/calc-source.png)

**Figure 7解读。** 测试reward在前期快速提高，后期约0.75–0.77附近，训练曲线更噪。工具准确计算不意味着模型一定选对表达式或解析正确，reward仍约束整个agent。图无统一硬件时间轴，不能把400多step和QA200多step直接比收敛快慢。图源train_calc.png/test_calc.png，组合只排版，不改数据或坐标。

三图与Table1共同说明不同编排程序能转换数据并获得训练改进。它们没有隔离“trace接口”“信用分配”“底层RL算法”的各自因果贡献，也没有全面比较长上下文重算、环境吞吐和异步staleness。框架性能、算法准确率与开发接入便利应分别评估。

复现实验最好增加不训练的相同agent、保持执行流程只换优化器的对照，并报告训练seed与测试采样seed。三种任务reward定义不同，单条曲线没有误差带，不能从相邻点波动推导统计显著或直接估计训练方差；模型选择是否使用测试反馈也需从完整实验日志核验。

## 5. 局限、结论与后续阅读

Agent Lightning提供执行程序到训练样本的桥梁，核心是保留每次调用真实输入输出，并明确可训练角色。它的基本终局reward分配不自动解决复杂agent信用；抽象接口也不保证行为概率、状态恢复和权重同步正确。论文实验是三个配置的可行性展示，不是ANY任务充分验证。

继续对照[GiGPO](#paper=paper-gigpo)的step相对优势和[Agentic RL](#q=agentic-rl-training)的完整轨迹契约。阅读当前代码时把2025论文、2026实现与另篇v1.0论文分开；按rollout去重、按token/episode归一这两件事尤其需要独立检查。

**Q01 主要解决什么？** Agent执行与训练接口、trace到transition适配。

**Q02 一次高层action是什么？** 一次LLM调用的响应，内部token为低层动作。

**Q03 hierarchical代表一定有turn critic吗？** 不代表，原始基本规则分配终局reward。

**Q04 只训练writer会冻结checker吗？** 共享权重时不会自动物理冻结，只排除对应调用loss。

**Q05 有trace就能算PPO ratio吗？** 还需真实token、原输入、行为概率和版本。

**Q06 rollout优势去重解决长度权重了吗？** 只解决统计一层，loss聚合还要另查。

**Q07 2026 CISPO代码是2025实验依据吗？** 不是，必须按版本分开。

**Q08 三任务曲线证明了SOTA吗？** 展示接入和改善，未提供全面强基线同预算比较。

**来源。** [固定2025 v1论文](https://arxiv.org/abs/2508.03680v1)、原图表和上述官方commit静态源码。未执行模型训练，原图版权归作者。
