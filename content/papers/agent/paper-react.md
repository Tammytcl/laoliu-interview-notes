---
id: paper-react
title: "ReAct: Synergizing Reasoning and Acting in Language Models"
paper_title: "ReAct: Synergizing Reasoning and Acting in Language Models"
authors: ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran", "Karthik Narasimhan", "Yuan Cao"]
affiliations: ["Department of Computer Science, Princeton University", "Google Research, Brain team"]
author_affiliations: [[1], [2], [2], [2], [2], [1], [2]]
venue: "ICLR 2023"
year: 2023
research_categories: [reasoning-decision]
method_figure: "./assets/papers/paper-react/figure-1.png"
method_caption: "Figure 1 · 推理、行动与环境反馈"
direction: agent
paper_url: "https://arxiv.org/abs/2210.03629v3"
github_url: "https://github.com/ysymyth/ReAct"
code_note: "Author-maintained code and prompts; PaLM model access and training resources are separate."
evidence: 已核原文
note_ids: [agent-tool-design, agent-evaluation]
tags: [Agent, Tool Use, Reasoning, Prompting, ReAct]
updated: 2026-09-30
summary: "把语言推理与环境动作交错组织，以外部观察修正后续行动；通过问答和交互任务检验这种闭环。"
template_version: 3
draft: false
---

## 1. 背景与已有工作

可以先想一个需要两次查证的问题：“某部电影导演的出生地在哪里？”直接回答要求模型同时记住导演和出生地；只搜索整句话又可能找不到准确证据。一个可靠的过程应先确认导演是谁，再查这个人的出生信息，如果查到的是同名人物还要调整检索。任务的答案与下一步该做什么，都依赖刚获得的观察，这就是本文研究的交互式决策。

在这里，语言模型负责提出推理和动作，执行器负责真正调用环境，环境返回观察。推理文字并不是已经验证的事实，也不是工具执行结果。例如“我应该查导演”只是计划，`search[...]` 才是动作，返回的网页片段才是观察。ReAct 将这些角色写进连续上下文，使模型下一轮能根据实际结果改计划；读懂这个边界，比记住一个提示词模板更有用。

仅靠内部知识进行多步推理，模型可能把未核实的事实继续传递到后续结论；只调用工具又可能缺乏明确的检索目标、任务分解和进度判断。ReAct 要解决的是：**如何把模型内部的语言推理与真实环境观察放在同一条可更新的轨迹中？**

这个问题已有两条相关路线。Chain-of-Thought 将中间推理写出来，但未必接触外部证据；WebGPT、交互式决策与模仿学习等路线让模型执行动作，却可能把计划和状态维护隐含在策略中。ReAct 将 thought、action、observation 显式交错，用推理决定下一个动作，再用动作返回的信息修正推理。它不只是一个后来框架里的 API 名称，也不要求先训练一个新的通用 agent 模型。

原文最早预印本发表于 2022 年，正式收录于 ICLR 2023；本报告固定 arXiv v3。作者单位按原论文列示，Shunyu Yao 的工作在 Google internship 期间完成，不应据此把所有作者单位简化成同一家机构。

## 2. 方法与实现机制

设已有上下文包含问题、示例与之前的交互轨迹。模型根据上下文生成语言 thought 或领域 action；只有 action 被环境执行，observation 才由工具或环境返回。语言 thought 用于分解任务、摘取证据、发现异常或决定切换目标，不直接改变外部环境。

```text
输入问题与 few-shot 轨迹示例
  → 生成 Thought：当前需要查证什么 / 下一步怎样推进
  → 生成 Action：search、lookup 或环境操作
  → 执行器验证并执行 Action
  → 将真实 Observation 追加到上下文
  → 再次决策，直到 finish 或达到预算
```

这段流程是整理后的实现说明。对知识型问答，原文采用密集 thought-action-observation；对长交互任务，thought 可以稀疏出现，由模型在重要位置生成。把它实现成“每一步强制一个长 thought”会改变原文设置与成本。

![Figure 1 · 推理、动作与环境观察的组织方式](./assets/papers/paper-react/figure-1.png)

**Figure 1 解读。** 图中将 Standard、CoT、Act 与 ReAct 放在同一任务下比较：Standard 直接给答案，CoT 展开内部推理，Act 有工具动作但没有语言推理，ReAct 同时利用两者。另一部分展示交互环境中稀疏 thought 如何辅助计划。这里最需要分辨的是：Observation 来自环境，不应由语言模型自由续写成“假装工具成功”。图是行为示例，不是总体成功率统计。[图源](https://arxiv.org/html/2210.03629v3#S1.F1)。

知识检索环境包含三个动作：`search[entity]` 返回页面前五句或相近实体；`lookup[string]` 在当前页面查找下一个含该字符串的句子；`finish[answer]` 返回答案。它比现代全文检索器弱，作者刻意用这种环境观察语言推理如何指导检索。因此检索效果、模型推理和工具设计共同影响成绩。

ReAct 与 CoT-SC 还可互补：ReAct 超过步数预算时退回 CoT-SC；CoT-SC 多数答案不足半数时转去查证。它表达的是内部知识与外部证据的条件切换，而不是“ReAct 在所有问题上比 CoT 强”。工程落地需要另外加入动作 schema、异常处理、超时、成本预算和执行日志；这些是实现设计，不是本文已完成的全部系统能力。

### 核心思想：让推理随证据更新

把先前动作和实际观察保留在上下文，模型就有机会发现“原计划的前提不成立”。例如第一次查询没有找到导演，下一轮应换实体名或先查电影，而不是继续在错误导演上推理。这个反馈闭环使 thought 既能规划动作，也能整理新证据；其收益依赖模型确实利用观察，并不由三种字段名自动保证。

还要区分推理可见与推理可靠。读者能够检查轨迹，定位在哪一步取错证据或选错动作，但一段流畅 thought 仍可能包含错误。环境工具也可能返回不完整结果。ReAct 的实验比较的是特定模型、prompt、工具和预算下的任务完成效果，不能把可读轨迹直接视为全部推理经过验证。

### 核心源码：谁生成文字，谁改变环境

官方 `hotpotqa.ipynb` 的 `webthink` 先拼接 instruction、六个示例和当前问题，然后在最多七轮中生成 thought/action。模型调用设置 stop 到 `Observation i:`，防止把工具返回内容继续编造出来；解析失败时再请求 action。之后代码调用环境 `step`，把真正的 `obs` 与本轮 thought/action 一起追加到 prompt。下一轮重新输入这条扩展后的文本历史，本文的工作记忆主要体现在上下文里，而不是一个新引入的外部记忆网络。

`WikiEnv.step` 接受一个动作字符串，解析 `search[...]`、`lookup[...]`、`finish[...]`。Search 更新当前页面；Lookup 按关键词建立匹配句子列表、维护游标，连续调用会返回后续结果；Finish 写入答案并把 `done` 设为真。其返回接口是 `(observation, reward, done, info)`，这使语言策略和真实环境状态分离。无效字符串返回无效动作观察，也会消耗一步；失败并不是能靠多写一段 thought 自动消除的。

`webthink` 在环境结束时提前退出，预算用尽则提交空 finish，并记录调用次数、格式失败数和最终轨迹。公开 notebook 是基于 API 的演示环境，不能据此声称拿到了论文主实验 PaLM 权重，也不能把 notebook 默认模型直接写成所有原文实验的模型。这里静态解读的价值是定位反馈机制和失败分支，原文实验配置仍以下节论文为准。

## 3. 实验设置与算力

HotpotQA 是多跳问答，答案往往需要连接两个页面中的事实；question-only 表示不提前送入正确支持段落，检索本身也是任务的一部分。FEVER 是事实核验，要判断一条陈述是否受到证据支持、反驳或缺少信息。ALFWorld 通过文字描述和动作接口完成家庭环境任务，WebShop 则要求根据需求搜索并选择商品。这四类任务分别考察证据连接、核验、长程动作和带约束选择，不能把同一条成功率理解为统一推理能力。HotpotQA 的 exact match 比较最终答案是否匹配，环境 success 衡量任务是否完成，可读 thought 并不单独计分。

| 实验环节 | 模型 / 数据 / 配置 |
| --- | --- |
| 主 prompting 模型 | 冻结的 PaLM-540B；附录另给 GPT-3 结果 |
| QA / verification | HotpotQA 与 FEVER，question-only：不提前提供 supporting paragraphs |
| Few-shot 例子 | HotpotQA 6 个、FEVER 3 个训练集实例，人工写 thought/action/observation 轨迹 |
| CoT-SC 对照 | temperature 0.7，采样 21 条轨迹，用多数答案 |
| 回退预算 | HotpotQA 7 步、FEVER 5 步，用于原文的 ReAct→CoT-SC 切换 |
| 交互任务 | ALFWorld 与 WebShop；少量人写交互示例，任务 action space 不同 |
| 额外微调 | 用 3,000 条答对的模型生成轨迹微调 PaLM-8B / 62B，不是重新预训练 540B |
| 微调 batch | 64；ReAct / Act 在 8B 与 62B 上均为 4,000 steps |
| 其他微调对照 | Standard / CoT：8B 为 2,000 steps，62B 为 1,000 steps；不能称训练预算完全相同 |
| GPU / TPU 数量与总成本 | 原文没有给出足以复现的完整硬件清单与 accelerator-hours；不填推测卡数 |

模型原始预训练语料不是 ReAct 在本文构造的训练数据。需要分开记录：PaLM 已有预训练、few-shot 示例、生成轨迹微调集与评测集。原论文没有逐项重印 PaLM 的预训练配方，也没有给出 PaLM-540B 的本地 GPU 复现方案；作者仓库的 prompt 与 environment code 不等于模型权重、服务访问和完整硬件资源。

HotpotQA 用 exact match，FEVER 看事实判断准确率；ALFWorld 与 WebShop 看任务成功率，WebShop 另给 score。这些分数分别来自知识检索和交互行为，不能算成一个统一“agent 智力分”。原文各场景的提示例数和工具不同，自己的实验应记录模型版本、prompt、环境快照、temperature、步数上限、随机种子和调用费用。

## 4. 结果与图表解读

![Table 1 · PaLM-540B 在 HotpotQA 与 FEVER 上的 prompting 结果](./assets/papers/paper-react/table-1.png)

ReAct 在 HotpotQA 的 EM 为 27.4，CoT 为 29.4；在 FEVER 为 60.9，CoT 为 56.3。它在两项任务上优于 Act，但并非在 HotpotQA 上单独胜过 CoT。原文混合策略在 HotpotQA 上达到 35.1（ReAct→CoT-SC），在 FEVER 上另一切换方向达到 64.6。比较时还应记住 CoT-SC 的多次采样成本，不把更高分直接当成单次调用更高效。[表源](https://arxiv.org/html/2210.03629v3#S3.T1)。

![Figure 3 · prompting 与轨迹微调在不同 PaLM 规模上的结果](./assets/papers/paper-react/figure-3.png)

**Figure 3 解读。** 对照模型规模及 prompting / finetuning 设置，观察加入 reasoning+action 轨迹后小模型是否获益。图支持轨迹训练能提升结果，但微调数据来自答对轨迹筛选，训练步数也随方法变化，不能把图中的差异全归因于 thought 字段本身。它与主表的 frozen-540B prompting 属于不同实验。[图源](https://arxiv.org/html/2210.03629v3#S3.F3)。

![Table 3/4 · ALFWorld 与 WebShop 的交互成绩](./assets/papers/paper-react/table-4.png)

ALFWorld 的 best ReAct trial 达到 71%，best Act 为 45%，BUTLER 为 37%；这些是 best-trial 比较，并不等于所有 prompt 都达到 71%。WebShop 上 ReAct 的成功率为 40.0，Act 为 30.1，专家人类为 59.6。读表时要区分 WebShop 的 score 与 success rate；“提升约 10 个百分点”指成功率口径，不是所有任务平均提升 10%。[原文交互实验](https://arxiv.org/html/2210.03629v3#S4)。

![Figure 5 · 人对 ReAct 轨迹的中途行为修正](./assets/papers/paper-react/figure-5.png)

**Figure 5 解读。** 这是一条 ALFWorld 轨迹修正示例：人调整中间语言状态后，后续行动能随之变化。它说明显式轨迹提供干预点，并不证明模型产生的 thought 是其内部真实因果解释，也不等于所有失败都能通过改一句话解决。[图源](https://arxiv.org/html/2210.03629v3#A1.F5)。

原文还人工分析 200 条成功 / 失败轨迹，发现 CoT 的事实幻觉与 ReAct 的检索失败、错误推理有不同分布。ReAct 的外部证据减少了一类问题，却引入了工具和环境依赖。此轮正文重点解释 Figure 1/3/5 与主结果表，其余图与附录轨迹在图表清单中保留待补状态。

## 5. 局限、结论与后续阅读

本文建立的是一种把推理与行动共同放进上下文的范式，以及若干具体任务上的实验支持。关键机制是外部观察能够改变接下来的决策；观察缺失、检索返回空结果、上下文越来越长、错误动作反复发生时，闭环仍会失败。Thought 可读不等于可靠，日志可见也不等于执行安全。

对自己的 agent 工作，应将“模型生成了合理计划”“工具真的成功执行”“环境任务完成”分开评价。[工具调用笔记](#q=agent-tool-design)和 [Agent 评测](#q=agent-evaluation)可进一步拆出成功率、行动预算与错误恢复。换成今日模型时，重新跑固定环境与 prompt 对照，不能直接继承 PaLM-540B 的数值。

**来源与更新。** 核对 [arXiv v3 正文](https://arxiv.org/html/2210.03629v3)、附录 B.1 的微调设置及原始图表；代码来自作者维护的 [ReAct 仓库](https://github.com/ysymyth/ReAct)。2026-09-30 更新为五模块精读，补充单位、实验数据流与原图解读。没有登记本人 GPU 训练或 API 跑分；算力未披露项保留明确缺口。图片与论文成果归原作者。

### 参考讲解与源码版本

本报告参考 [Shunyu Yao 与 Yuan Cao 的作者讲解](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/)，吸收推理驱动行动、观察反过来修改计划的双向解释；保留原文中密集与稀疏 thought 的区别。解释已融入问题与方法部分；数字、图表和实验口径回到固定版本原文核对。

源码静态核读固定于 `6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9`。核心文件：[hotpotqa.ipynb](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/hotpotqa.ipynb)；[wikienv.py](https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/wikienv.py)。没有执行代码或重新训练。
