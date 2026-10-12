---
id: agentic-rl-training
title: "Agentic RL多轮环境信用分配与训练工作"
category: agent
difficulty: 深入
tags: ["P1", "Agentic RL", "Search-R1", "GiGPO", "Agent Lightning", "Credit Assignment"]
updated: "2026-10-12"
summary: "从真实工具轨迹建立训练契约，按搜索、信用分配、探索、训练解耦、coding与异步系统比较代表工作。"
draft: false
---

## 1. 问题背景 单条回答优化为何不够

单轮数学RL通常生成一条回答再评分；agent执行动作会改变文件、网页、购物车、机器人或外部数据库，再决定后续输入。错误不仅是文本错了，还可能把环境带到不同状态。最终reward稀疏，任务长度不一，工具等待与环境故障也会影响采样。Agentic RL研究怎样在这样的交互闭环中更新模型的决策策略。

先修[Actor Critic](#q=rl-actor-critic)、[回报GAE](#q=rl-returns-gae)、[PPO训练一步](#q=rl-ppo-training)和[算法地图](#q=rl-policy-optimization-map)。本篇按真正的训练瓶颈组织论文，不把agent组件、全部RL缩写和系统速度堆成一个排行榜。

## 2. 前置知识 历史动作反馈与粒度

令真实历史$h_t$包含任务、已执行动作和环境观察；模型实际输入是经格式化/选取后的$c_t$。模型产生response $y_t$，parser提取执行动作$a_t$，环境返回$o_{t+1}$。必须记录y和a的关系，不能把被parser拒绝的proposal当成已执行行为。

token级策略以prefix为条件，turn级策略以整个response为动作。完整response logprob是生成token之和；turn和trajectory的长度归一化改变权重。工具返回通常不属于policy action，但会影响下一轮概率。终止、采样截断和预算失败按MDP定义区分。

**Credit assignment** 把反馈归到影响结果的决策；**exploration** 决定采哪些不同决策；**curriculum** 改变任务/难度/参与训练范围；**systems scheduling** 决定何时在哪些资源上执行。它们相互影响，但增加并发并不能自动提供更细credit，换advantage也不能解决工具服务故障。

## 3. 一条可训练轨迹的契约

![本文示意 Agentic RL采样环境反馈与learner](./assets/learning/agentic-rl-training/agent-rl-loop.svg)

**图解。** 模型生成proposal，经解析执行形成真实transition；reward/value/组统计产生优势，learner重算对应logprob和loss。环境状态与模型文本状态分别保存；最后同步策略并开始新采样。工具观察是条件，不是自动训练目标。图中的trace转成训练样本还需要版本、token和语义对齐。

```json title="教学数据结构 记录proposal与实际执行"
{
  "task_id": "task-7",
  "trajectory_id": "traj-3",
  "turn_id": 2,
  "policy_version": 11,
  "actual_input_token_ids": [12, 20, 31],
  "generated_token_ids": [41, 42],
  "behavior_logprobs": [-0.8, -0.3],
  "executed_action": "run_tests",
  "observation_ref": "tool-result-2",
  "action_mask": [1, 1],
  "reward": 0,
  "terminated": false,
  "truncated": false
}
```

这是最小职责示意，不是任何框架的完整schema。真实系统还要采样温度/截断规则、model/tokenizer/template版本、tool_call/process ID、环境版本、最终reward、缓存与恢复信息。若上下文摘要改变了下一轮prompt，保存实际输入；不要用最终拼接文本替代过去条件。

推理阶段不必保存整个工具计算图。训练重新forward已发生token，在正确mask下优化logprob。工具程序、网页响应和文件系统不需要可微。Actor loss mask0也不阻止工具观察影响后续token表示。

## 4. 搜索与工具交互 Search-R1 RAGEN与Coding

### Search-R1 给搜索动作真实反馈

[Search-R1精读](#paper=paper-search-r1)把生成和检索交错：search标签触发query，检索文本插入information区域，再继续生成。终局答案EM提供reward，PPO/GRPO只对模型生成token训练，检索token被mask。模型既学习回答，也学习什么时候和如何检索。

Figure里的“Reward Model”是角色框，原实验使用规则式EM，没有额外训练神经reward模型。Reference用于KL，rollout由行为policy生成；不能把两者混成同一个模型。正文与代码都要核，不能只背架构图。

论文v5用NQ+HotpotQA训练，七个QA集评测，主要是Wikipedia2018/E5检索环境。它不是直接证明当前互联网浏览、网页交互或citation核查都已学会；这类能力需要相应环境与评分。原始mask消融和PPO/GRPO差异在单篇保留。

### RAGEN 轨迹学习与行为崩溃

[RAGEN v2](https://arxiv.org/abs/2504.20073v2)以StarPO组织state、thinking、action和reward，研究多轮环境中策略演化与崩溃问题。它提醒我们：训练reward曲线可能与推理质量、探索和长程行为不同步。复杂环境应看轨迹而非仅最终loss，简单可控环境也可用于诊断。

环境复杂度、reward稀疏、策略先验、模型温度和轨迹过滤会共同影响学习。不要把某个训练配方在小型可控环境的改善直接外推到长程coding sandbox。

### Coding RL 真值与环境成本

[DeepSWE作者说明](https://www.together.ai/blog/deepswe)展示Qwen3-32B在软件任务中用RL学习的路线，训练系统为rLLM。这里的重要位置是“真实仓库/工具交互加结果反馈”，不是借一个展示分数证明全部算法选择。

代码任务要固定base commit、依赖、测试入口与独立验证，区分测试失败、环境失败、未执行测试和真正未修复。推理多次尝试、best-of选择或test-time scaling与训练收益也分开。场景接线见[Code Agent组件](#q=agent-code-components)，环境尺度见[DSec](#paper=paper-dsec)。

## 5. 信用分配 从critic到GiGPO

轨迹级GRPO对同一任务多条完整trajectory比较总reward，再把一个系数广播到各turn/token。高分轨迹也可能包含无用或有害操作，低分轨迹也可能包含正确早期动作。广播提供宏观方向，未必细分每个决策贡献。

### GiGPO 同状态下比较后续回报

[GiGPO精读](#paper=paper-gigpo)保留episode相对优势，并把已经采到、遇到相同anchor state的动作聚成step组，不额外从每个状态重新rollout。对每个动作计算真实后续折扣return，在state组内做相对比较，再组合：

$$
A_{i,t}=A_i^{E}+\omega A_{i,t}^{S}.
$$

例如两条trajectory都经过同一个搜索页面，一条选择有效商品，另一条反复返回。Episode信号评价总体成功，step信号比较从这个状态出发的选择。相同页面URL未必代表相同完整环境状态，购物车、任务和可见信息都需要纳入组定义。

固定源码[组合入口](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py#L160-L171)节选：

```python title="GiGPO真实节选 两级优势组合"
step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)
step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)
scores = episode_advantages + step_advantage_w * step_advantages
```

episode_advantages在前面计算，step_rewards应对应后续return而非任意即时文本分数。当前仓库有exact/相似anchor选项与统计默认值，单篇区分论文式与现代实现。Singleton、相似度、跨任务误分组及长trajectory重复计权都会影响估计。

**不额外采样不等于已经解决所有因果归因。** 需要足够重复state与有效对比，相关轨迹不是严格独立反事实；不同隐藏历史也可能改变行为。后续return未确定时，step advantage可能还不就绪，因此不能仅因一个turn完成就提前使用最终目标。

### Critic与分层价值路线

[ArCHer v1](https://arxiv.org/abs/2402.19446v1)提供分层多轮RL的对照：turn级与token级的动作/学习对象可以不同，价值估计用于长程决策。它与critic-free组方法的取舍，应比较value误差、数据使用、环境长度和资源，不能用“无需critic”直接证明更好。

## 6. 探索和环境reward 先改变哪里

[ARPO v1](https://arxiv.org/abs/2507.19849v1)观察工具返回附近的高不确定性，并用熵相关的adaptive rollout/step sampling和优势归因调配探索。它与GiGPO的关键区别是是否主动增加/改变分支采样：后者主要复用已有轨迹中的重复state，前者把更多探索放到选定决策附近。

新分支要从相同环境初态或正确snapshot恢复；只复制token prefix，不足以重建文件改动、工具会话和网页状态。分支多也增加环境、生成与评分成本，需按相同task/GPU-hour预算比较。

[Practitioner’s Guide v2](https://arxiv.org/abs/2510.01132v2)按environment、reward、policy三柱研究多轮学习，比较任务复杂度、稀疏/密集reward、策略先验以及SFT/RL预算。密集反馈可加速某些配置，但错误中间评分、与算法不匹配或改变目标也会伤害稳定性。

中间reward必须解释来源：环境自带子目标、经过验证的进度、模型judge或潜势函数各有条件。工具调用成功不等于任务进步；代码通过agent自己新写的宽松测试也不能当独立真值。

## 7. Agent Lightning Trace怎样接到训练

[Agent Lightning精读](#paper=paper-agent-lightning)将agent执行与learner分开，记录真实input/output/reward，把被优化模型的调用转换成transition。可以选择多角色系统中参与学习的调用，沿用已有agent逻辑。

2025原论文的基础LightningRL将同一episode最终return分给每个action，再用token级算法更新。**它不等于已经学到了细粒度中间critic**；更复杂credit是可扩展模块。与GiGPO比较时应说明原始return广播和state组对比的不同。

每轮独立transition可以保留真实summary/结构化prompt，无需假装所有调用来自一条连续文本。代价是可能重复计算长prompt；trace还需token/采样/版本信息，不能把普通可观测日志自动当完全对齐训练样本。

[Agent Lightning v1.0原文](https://arxiv.org/abs/2608.17528v1)是后续harnessed训练工作，应与2025原论文区分。当前公开repo的adapter/rollout-level优势路径也不自动等于2025实验实现。框架统一接口与算法信用分配需要分别评价。

## 8. 多任务和异步系统 资源收益与算法风险

[THUDM AgentRL v1](https://arxiv.org/abs/2510.04206v1)把多任务环境接口、异步生成/训练、cross-policy探索和任务优势标准化放在一起。它不是其他同名AgentRL项目；不同任务的reward尺度、长度、难度和采样比例，会影响共享策略更新。

异步可减少等待，但数据新鲜度、behavior/proximal/current角色需要分开。[AReaL](#paper=paper-areal)讨论有界异步与算法校正；[SAO](#paper=paper-sao)提供单rollout加value的对照；[Heddle](#paper=paper-heddle)、[TideRL](#paper=paper-tiderl)关注工具等待、轨迹调度和ready数据/资源。

系统表里的rollout吞吐、训练吞吐、goodput、端到端时间与相同GPU-hour任务收益不同。高GPU busy也可能在重复prefill、无效分支或过时数据上忙碌。异步importance校正的动作比不能自动修复所有旧state访问分布。

## 9. 统一对照怎样填

| 工作/路线 | 直接瓶颈 | 数据/credit对象 | 不能自动推出 |
| --- | --- | --- | --- |
| PPO/GAE | 稀疏反馈与局部更新 | value、真实transition | 任意截断已正确处理 |
| Search-R1 | 多轮检索怎样学 | 生成tokenmask、终局EM | 网上任何事实检索都可靠 |
| GiGPO | episode广播太粗 | 同anchor后续return组 | 完整因果反事实、立即turn训练 |
| ArCHer | 长程与token层级 | 分层价值/策略 | 所有critic-free方法劣于它 |
| ARPO | 工具边界探索 | 选择分支与归因 | 额外探索没有成本 |
| RAGEN/实践指南 | 学习稳定与配方 | 环境/reward/prior/轨迹 | 单个因素普遍最优 |
| Agent Lightning | 任意harness接训练 | actual call transition | 日志天然有正确训练概率 |
| AgentRL | 多任务/异步扩展 | 跨policy/任务归一化 | 多任务采样自然公平 |
| DeepSWE | coding真实任务 | patch、环境和验证 | pass@k等同单次稳定性 |
| AReaL/SAO/Heddle/TideRL | 版本/队列/长尾资源 | 完成样本、ready请求、调度 | 吞吐等同任务学习收益 |

阅读顺序可先Search-R1看mask，再GiGPO看credit，接Lightning看接线，最后按实际瓶颈读探索和系统。单篇配置与出处保持独立。

## 10. 最小实验和两分钟讲法

固定任务、环境版本、模型初始checkpoint、温度、turn/token预算与verifier。至少对照：正确PPO/GAE；同预算组baseline；细粒度credit；必要的系统变体。每次只改明确轴，分别报告成功率、reward、长度、工具数、wall time/GPU-hour和基础设施失败。

先验证token/role/mask、episode边界、真实prefix重算、reward统计与state恢复。框架能跑不等于样本正确；reward上升也可能只是环境/评分漏洞。对coding和search，先用独立测试/证据支持建立结果真值。

两分钟介绍沿“任务与环境→policy action→真实观察→反馈与credit→learner update→版本/验证”展开，之后说所选论文改的是哪一段。这样既能讲基础，又能解释新工作。

## 11. 面试问答

**Q01 [定义] Agentic RL比单轮RL多了什么？** 动作改变环境和后续状态，多轮反馈、信用、恢复、工具时延与版本组织都变重要。

**Q02 [数据] Proposal与executed action为何分别存？** Parser或干预可替换/拒绝proposal，训练条件与后续reward应对应实际行为。

**Q03 [mask] Search-R1为什么mask检索token？** 它们来自环境，不是policy动作；仍作为后续条件可见。

**Q04 [credit] GiGPO怎么省额外采样？** 在已有trajectory里找同anchor的决策比较后续return，额外做分组统计，不每state重新rollout。

**Q05 [状态] URL相同能作anchor吗？** 需要确认任务、环境及相关隐藏状态；字符串相同不保证完整状态等价。

**Q06 [时间] Step完成可立即知道最终优势吗？** 若用future return/组统计，可能要等后续或组齐；局部TD估计是另一个目标选择。

**Q07 [框架] LightningRL已经解决细粒度credit了吗？** 原论文基础实现广播终局return，统一接口允许扩展，但不等于已学习中间value。

**Q08 [探索] ARPO与GiGPO区别？** 一个调整选定位置的采样探索，一个主要复用重复state的已有采样；都需算总成本。

**Q09 [策略] Dense reward一定更好？** 取决真值、目标和算法；错误反馈会被更快优化。

**Q10 [系统] Async比同步快能说明学得更好吗？** 需相同预算task收益、新鲜度/ESS、有效样本与失败归因，吞吐只是一个指标。

**Q11 [评测] Code与search如何分别验收？** Code验实际patch与独立测试，search验断言、原文支持、版本时间和覆盖，不能只信final。

**Q12 [讲课] 最先画什么？** 画一条真实transition和模型/工具token边界，再画反馈怎样成为优势，最后落到论文修改点。

## 12. 原始来源与课堂练习

[Agentic RL专题](#report=survey-agentic-rl-2026)按方法、任务和证据层次比较；上述六份核心/相关原文与既有异步精读分别保留条件。[来源清单](./research/agentic-rl-training-sources.json)记录核读范围，摘要/作者说明定位不当作全面实验复测。

[CPU脚本](./assets/learning/agent-training-lab.py)和[结果](./assets/learning/agent-training-lab-results.json)检查mask、episode归一化和GiGPO玩具分组，不调用真实agent或训练模型。下一篇[Agentic OPD](#q=agentic-opd-training)比较教师信号与任务reward的另一条学习路线。
