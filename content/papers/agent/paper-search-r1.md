---
id: paper-search-r1
title: "Search-R1：通过真实检索交互训练推理与搜索"
paper_title: "Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning"
authors: ["Bowen Jin", "Hansi Zeng", "Zhenrui Yue", "Jinsung Yoon", "Sercan Ö. Arık", "Dong Wang", "Hamed Zamani", "Jiawei Han"]
affiliations: ["Department of Computer Science, University of Illinois at Urbana-Champaign", "Center for Intelligent Information Retrieval, University of Massachusetts Amherst", "Google Cloud AI Research"]
author_affiliations: [[1], [2], [1], [3], [3], [1], [2], [1]]
venue: "COLM 2025"
year: 2025
direction: agent
areas: [language]
tasks: [agents, retrieval, reasoning, training-adaptation]
published: "2025-03-12"
paper_url: "https://arxiv.org/abs/2503.09516v5"
github_url: "https://github.com/PeterGriffinJin/Search-R1"
code_note: "静态核读598e61bd1d36895726d28a8d06b3a15bed19f5d3；当前仓库实现与固定v5实验分别标注。"
method_figure: "./assets/papers/paper-search-r1/method-source.png"
method_caption: "Figure 1 · 推理、搜索、观察与RL更新"
evidence: 已核原文
note_ids: [agentic-rl-training, rl-ppo-training, rl-returns-gae]
tags: [Search-R1, Agentic RL, PPO, GRPO, Retrieval, Loss Mask]
updated: "2026-10-12"
summary: "把检索作为可执行动作，使用答案EM反馈训练搜索与推理；重点解释工具观察mask、PPO/GRPO、数据与GPU配置及固定源码。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：知道怎样回答，还需要知道何时搜索

参数中记住的知识可能缺失，复杂问答还需要多步证据。把一次检索结果放进prompt的RAG，提供了外部信息，却没有直接训练模型怎样连续查询、读证据、改变搜索方向和决定结束。Search-R1把检索变成agent动作，在训练时真的调用检索器，让最终答案反馈影响查询与推理的选择。这里R1表示RL推理路线，主模型是Qwen2.5，并非直接训练DeepSeek-R1权重。

最重要的训练边界是：学生生成查询和答案，搜索引擎返回观察。若把检索器吐出的文字当成学生动作计算策略loss，目标便误把外部内容算为模型输出；但如果把观察从注意力上下文里也删掉，学生下一次查询又失去证据。论文因此把“观察可见”和“观察不参与actor loss”同时保留。

任务仍是有标准答案的离线知识问答，不是开放互联网浏览或长程coding修复。训练回报是最终答案的规则exact match，不能说它需要一个神经reward model，也不能据此宣称验证了引用真实性、网页安全或实时新闻能力。研究目标是让模型学会用给定检索环境提高答案准确率。

### 前置知识：动作、观察和策略比值

LLM生成的一串token可包含思考、搜索标签与最终答案；动作解析器发现搜索请求后暂停生成，把query发给检索服务，再将观察拼回上下文。PPO里旧策略是收集这些学生token的behavior policy，当前策略是更新中的actor；固定reference用作KL约束。三者职责不同。搜索返回的文字没有由actor采样，不需要学生行为概率，但其位置会影响后续student token的条件概率。

PPO需要critic与优势估计，GRPO对同prompt多条结果做组内比较，免去独立critic；两者都可以使用这条交互执行链。对于没有中间任务奖励的搜索过程，前几轮query的信用仍来自最终结果与相应优势估计，不会因为有搜索标签就自动获得正确的逐轮reward。

### 已有工作：提示工程、检索和RL推理

[ReAct](https://arxiv.org/abs/2210.03629)把推理与动作交替，IRCoT探索交错思考与检索；Search-o1等使用强模型提示流程。Search-R1关注通过RL训练模型本身的交互策略，而不是只换一个提示模板。它采用[PPO](https://arxiv.org/abs/1707.06347)和DeepSeekMath中的GRPO作为可替换优化器。最终应同时比较非RL检索、SFT与rejection sampling，才能定位完整训练配方的收益。

## 2. 方法与实现机制

### 从初始问题到真正执行的一条轨迹

![Figure 1 · 模型生成搜索请求、获得结果并继续推理](./assets/papers/paper-search-r1/method-source.png)

**Figure 1解读。** 模型在推理中生成查询，搜索模块返回内容，再作为下一段生成的条件。最终答案进入规则验证，训练更新actor；PPO还涉及critic，GRPO改为组内优势。原图箭头中的搜索调用是真环境交互，回传的文字是观察，不是另一段学生动作。图源为固定v5源码的figures/main.pdf，对应3.methodology.tex中的fig:main；图展示闭环，不单独证明某个优化器更优。[原文](https://arxiv.org/abs/2503.09516v5)。

论文使用搜索、观察和答案标签，生成遇到搜索动作后提取query；检索器返回top-k段落，随后继续生成。完整轨迹交替包含学生片段和外部观察。查询可以重新措辞或做多跳补充；预算耗尽或生成答案后结束。无效格式也占执行预算，并返回格式纠正信息，不等于凭空跳过该轮。

一个简单例子是“作者A出生地所在国家是什么”：先搜索作者出生地，观察城市名，再搜索该城市所在国家，最后回答。若首个搜索返回错误文档，第二次动作的条件上下文已经改变；学习者需要保留这条真实观察，不得在评分时换成理想教师证据。训练初始问题相同，也不意味着不同rollout后续前缀相同。

### Reward、mask与PPO目标

回报是最终答案与gold的EM，成功1、否则0。检索内容不参与actor优化，可将学生token集合记为$\mathcal A(\tau)$。PPO局部比值为：

$$
r_t(\theta)=\exp\{\log\pi_\theta(a_t|h_t)-\log\pi_{\rm old}(a_t|h_t)\},
\quad
J_{\rm clip}=\mathbb E\left[
\min(r_tA_t,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\right].
$$

只在$t\in\mathcal A(\tau)$累计该项，并按具体实现加入reference KL。论文的方法叙述有一处采样策略称谓容易混淆，读Eq.3与实现应把实际rollout归于$\pi_{\rm old}$，不能把reference当behavior。对于GRPO，同prompt采多条轨迹，以最终回报组统计构造优势，数据的完成依赖来自组，而不是搜索器。

这里有至少三种mask：attention让问题、学生历史、检索观察可见；action/loss mask只保留学生生成部分；padding mask处理对齐空位。mask观察loss不应该改写原始位置编码或删除观察。工具观察之间相邻的学生决策还可能需要正确的GAE索引，详见[两个mask与GAE](#q=rl-returns-gae)。不能把一个乘零操作泛化为所有mask语义。

### 固定源码一：查询怎样推进环境

源码固定commit为598e61bd1d36895726d28a8d06b3a15bed19f5d3。[generation.py](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/search_r1/llm_agent/generation.py#L220)的run_llm_loop维护active_mask、turn计数、左侧输入与右侧累计响应；只给活跃样本新生成，并经execute_predictions判定搜索、答案或无效动作。

搜索分支中的核心语句如下，摘自同文件约390–394行，省略其他分支：

```python
# Official excerpt: a retrieved observation, not a sampled actor action.
next_obs.append(
    f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n'
)
dones.append(0)
valid_action.append(1)
is_search.append(1)
```

next_obs对应环境返回，done=0表示仍可继续；答案分支done=1。valid_action用于执行统计，不等于搜索正确、证据相关或有中间reward。对batch中的样本，结束状态独立维护，避免把某个任务的下一轮观察接到另一任务。

### 固定源码二：完整上下文和loss mask如何同时存在

同文件_compose_final_output把prompt与累计responses拼起来，attention_mask来自完整上下文；另行构造info_mask，在检索位置使用pad占位，区分外部信息与学生输出。position_ids由attention_mask计算，因此检索内容仍影响后续位置。[最终组合函数](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/search_r1/llm_agent/generation.py#L321)。

ray_trainer中的_create_loss_mask选取response对应尾部：

```python
# Official excerpt: ray_trainer.py, relevant lines only.
response_length = batch.batch['responses'].shape[-1]
loss_mask = batch.batch['info_mask'][:, -response_length:]
batch.batch['loss_mask'] = loss_mask
```

见[固定源码854–867行](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/trainer/ppo/ray_trainer.py#L854)。输出是$[B,L_{\rm response}]$，并非把整段prompt拿去训练。这里静态说明当前固定仓库的接线，不能用2026仓库HEAD替代v5历史实验配置；没有执行模型训练。学生输入若被长度截断，也需保持评分时的相同条件，而不能只靠文本“看起来一样”。

### 课堂检查：一条轨迹至少要保存什么

应保留原始prompt、学生token、每个观察插入位置、action mask、行为logprob、策略版本、解析动作、真实检索query和结果、终止原因与最终reward。只保存一篇拼接后的自然语言记录，会失去采样概率与token边界，无法可靠重算ratio。搜索引擎的top-k、索引版本或retrieved text改变也会改变状态；训练重放时不要 silently 重新搜索再把新结果当旧轨迹。该要求是从方法推导出的工程检查，不是论文声称已经提供了完整容错系统。

## 3. 实验设置与算力

配置固定于v5正文与附录，主要表为Qwen2.5 3B/7B Base与Instruct；14B在补充实验，不能混为同一主设置。训练NQ与HotpotQA，评测NQ、TriviaQA、PopQA、HotpotQA、2Wiki、MuSiQue、Bamboogle七组，EM为[0,1]比例。

| 配置 | 固定v5披露 |
| --- | --- |
| 主模型 | Qwen2.5-3B/7B，Base与Instruct分别训练 |
| 训练数据 | NQ + HotpotQA |
| 检索 | 2018 Wikipedia，E5，top3 passages |
| 动作预算 | 最大4次交互轮次，包含无效动作 |
| PPO actor LR | 1e-6 |
| PPO critic LR | 1e-5 |
| 训练步数 | 500 |
| warmup | actor0.285，critic0.015 |
| GAE | gamma=1，lambda=1 |
| batch | 512，minibatch256，microbatch64 |
| 长度 | 总序列4096，response500，retrieved content500 |
| KL/clip | 系数0.001，clip0.2 |
| 生成 | temperature1、top-p1 |
| 后端 | FSDP、CPU offload、gradient checkpointing |
| vLLM | TP1，显存利用比例0.6 |
| GPU | 一节点8张H100 |
| GRPO | actor LR1e-6，group5，500steps；组采样成本需另计 |

论文使用规则EM，不额外加格式reward；标签与解析器依然约束可执行动作空间。检索语料和top-k固定后，结果只适用于该环境，不代表互联网实时搜索。单个response长度与整条多轮上下文长度是两种限制，前者500不能理解为完整任务只生成500token。

每100step保存模型；若训练发散，依据训练reward选择最近稳定checkpoint，否则用最终checkpoint。这种选择策略会影响不同优化器的比较；并不是所有行都一定用最后500step。正文给出了八H100配置，但没有以统一GPU-hour详表证明PPO与GRPO同预算。GRPO每prompt五rollout，在同step表上更好并不意味着同生成token或同GPU-hour更好。

复现应记录prompt模板、搜索标签、检索索引、EM标准化与停止策略。答案准确率不直接评价查询次数、证据覆盖或幻觉引用；如果面试项目的目标是search agent质量，还要报告搜索成本和证据正确性。把原始EM基准扩展到新环境时，这些新指标要明确为项目评测设计。

## 4. 结果与图表解读

![Table 2 · 七组问答任务的主结果](./assets/papers/paper-search-r1/table-2-pdf.png)

**Table 2解读。** 上半7B、下半3B，列为七benchmark及等权平均。7B Base Search-R1平均EM0.431，Rejection Sampling为0.348，差0.083即8.3个百分点，相对提升约23.9%；不是提高24个百分点。7B Instruct为0.385，3B Base/Instruct为0.303/0.325，说明初始化和规模都影响结果。不同baseline采用各自生成/检索配方，表支持完整方法效果，不单独证明某一个标签或loss贡献。原表裁图来自[PDF第8页](https://arxiv.org/pdf/2503.09516v5#page=8)，保留全部行与表注。

![Table 3 · PPO与GRPO在不同模型上的对照](./assets/papers/paper-search-r1/table-3-pdf.png)

**Table 3解读。** 7B Base PPO0.431对GRPO0.350，但7B Instruct GRPO0.396对PPO0.385；3B Base GRPO0.312对PPO0.303，3B Instruct0.336对0.325。因此不能概括“PPO普遍比GRPO强”。这张表回答的是具体配置下的最终EM，优化器的稳定性、checkpoint选择和rollout group成本也应一起读。单纯比较模型是否有critic，不足以解释全部差异。[原表第8页](https://arxiv.org/pdf/2503.09516v5#page=8)。

![Table 4 · 检索观察loss mask消融](./assets/papers/paper-search-r1/table-4-pdf.png)

**Table 4解读。** 固定Qwen2.5-7B Base/PPO，with mask平均0.431，without0.343，提升8.8个百分点，七列都改善。它支撑“不把检索器返回内容当学生动作训练”的重要性，却不意味着观察从attention删除；源码恰好把注意力可见与loss排除分开。该消融在一个模型/优化器设置下，不自动证明所有外部信息都应以同样实现mask。[原表第9页](https://arxiv.org/pdf/2503.09516v5#page=9)。

三张表一起读：Table2说明完整闭环有效，Table3定位算法结果有模型依赖，Table4检验关键训练语义。它们没有报告coding任务、并发sandbox可靠性或最新网页时效，也没有证明更长trajectory必然更高EM。若从搜索迁移到代码agent，需要替换动作执行与verifier，再独立验证reward、观察mask、episode边界和成本。

## 5. 局限、结论与后续阅读

Search-R1把真实搜索纳入训练闭环，最值得复用的是正确区分动作与观察，以及把同一轨迹的真实上下文保留下来评分。规则EM简单可验证，但不等同证据忠实度、可解释性或搜索成本；知识源固定、任务相对短，也限制了对长程agent的外推。稳定checkpoint选择与组采样数量使算法比较不能仅看step。

继续读[GiGPO](#paper=paper-gigpo)怎样增加step信用，[Agentic RL流程](#q=agentic-rl-training)怎样统一轨迹契约，以及[PPO训练一步](#q=rl-ppo-training)中的概率比和缓存。以下问题可用于课堂或面试，不标成某公司的原题。

**Q01 Search-R1和普通RAG差在哪？** 训练模型何时搜索、怎样迭代query，而非仅固定检索后回答。

**Q02 R1意味着DeepSeek权重吗？** 主实验用Qwen2.5，名称不能替代模型配置。

**Q03 检索观察为什么不能actor loss？** 未由actor采样，不能当行为动作；仍应进入上下文。

**Q04 有神经reward model吗？** 主设置用答案EM规则反馈。

**Q05 reference是采样策略吗？** behavior old负责rollout，reference负责约束，需区分。

**Q06 PPO一定优于GRPO吗？** 原表在不同初始化/规模上并不一致。

**Q07 平均0.431比0.348提高多少？** 8.3个百分点，约23.9%相对。

**Q08 迁移到coding首先查什么？** 环境执行、观察边界、真实verifier、回报归属和同预算对照。

**来源。** [固定v5全文与附录](https://arxiv.org/abs/2503.09516v5)、原始LaTeX资产与上述固定commit静态源码。本报告未重新训练，没有个人复现性能；图表版权归作者。
