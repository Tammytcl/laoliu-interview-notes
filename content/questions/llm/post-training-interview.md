---
id: post-training-interview
title: "后训练面试：40个问题驱动SFT、偏好优化、RL与蒸馏学习"
category: llm
difficulty: 进阶
tags: ["P0", "Post Training", "Interview", "SFT", "DPO", "RLVR", "OPD"]
updated: "2026-10-13"
summary: "从数据和目标到梯度、训练故障、评测与项目论证；每题给直答、要补的知识和追问，公开题型与自拟题区分，不靠背缩写选算法。"
draft: false
---

逐题口述练习请进入[后训练面试题库](#q=post-training-interview-practice)。本篇保留为知识点定位与查漏提纲，完整口述稿、白板与项目追问在主问题页维护，避免同一答案多处扩写。

## 1. 问题背景 面试要把方法选择讲成可检验的决策

后训练面试常从“PPO与GRPO区别”开始，再问critic怎么训练、KL放哪里、数据怎么构造、为什么变长、显存为什么炸以及如何证明收益。准备时应从问题暴露的知识缺口进入原理，而不是先背一串loss。回答先明确任务和反馈，再给公式或数据流，再限定条件，最后用自己的证据说明选择。

本篇是总入口，40题覆盖SFT、偏好、在线RL、蒸馏、Infra和评测；更深入的推导已在既有专题，避免重写同一套公式。问题背景描述学习目标，以下前置知识建立对象；每题的“学习”给出应复习的内容，“追问”用于检验理解。

## 2. 前置知识与题源

数据对象有prompt、teacher示范、偏好pair、学生trajectory、环境反馈；模型角色有actor、behavior、reference、critic、teacher。Objective定义优化什么，estimator定义怎样用有限样本算梯度，runtime定义何时生成、消费和更新，三层不能混用。

| 公开来源 | 本次确认的题型 | 使用范围 |
| --- | --- | --- |
| [个人暑期实习记录](https://www.nowcoder.com/feed/main/detail/0a36b958ef864c8db004bd45694ec9a6) | reward设计、PPO/GRPO、critic训练、KL与项目追问 | 公开经历自述，未独立核招聘真实性 |
| [字节算法岗题型汇总](https://www.nowcoder.com/discuss/922308546966847488?sourceSSR=post) | DPO公式、优势、KL、DAPO、分片/OOM、Agentic过程反馈 | 汇总题型，不统计频率、不当公司认证真题 |
| [百度算法岗汇总](https://www.nowcoder.com/discuss/927018114900922368) | PPO/DPO/GRPO、输出变长、大batch与off-policy | 按公开可见内容改写，未读取付费部分 |
| [TRL DPO官方文档](https://huggingface.co/docs/trl/en/dpo_trainer) | preference格式与trainer机制 | 技术依据，不是面经 |

以上页面2026-10-13核对。标“公开题型”的条目是改写，其余是结合原论文与项目自拟。技术答案依据[PPO](https://arxiv.org/abs/1707.06347v2)、[DPO](https://arxiv.org/abs/2305.18290v2)、[DeepSeekMath](https://arxiv.org/abs/2402.03300v3)、[DAPO](https://arxiv.org/abs/2503.14476v2)、[GKD](https://arxiv.org/abs/2306.13649v3)与既有精读。网络检索不是穷尽全部题目，也不赋予题解技术权威。

![教学示意 · 问题到对象、目标、运行和证据](./assets/learning/post-training-interview/study-flow.svg)

本文学习流程图。箭头表示回答时逐层定位问题，非模型训练框架原图；先定task/feedback，再看数据支持与梯度，最后核runtime和实际证据。

## 3. 面试问答一 数据与SFT

**Q01 [自拟] 后训练解决什么，预训练不能直接替代吗？**

回答：预训练提供通用预测能力，后训练围绕指令、偏好、推理和工具行为改变模型表现。不是每种能力都只由某阶段产生，也不保证后训练创造完全新知识。学习：任务目标、训练分布、可验证结果。追问：你的提升是知识获取、行为调整，还是解码预算变化？

**Q02 [自拟] 一次SFT究竟监督哪些token？**

回答：因果LM对目标response预测下一token；prompt、工具观察和padding通常不参与目标，但作为条件可见。多轮需明确监督全部assistant turn还是末轮。学习：shift、attention/loss mask、模板。追问：用户代码修正后重拼prompt，会不会改变原条件？

**Q03 [自拟] SFT loss下降为什么测试不一定更好？**

回答：可能记住格式/答案、数据分布偏、标签错误、长度权重不当或泛化退化；训练CE与任务正确率不是同一指标。学习：独立验证、任务/仓库去重、错误分组。追问：怎样区分训练不足、过拟合和评测模板错配？

**Q04 [自拟] Teacher生成数据要全部保留吗？**

回答：检查任务有效性、答案/测试、格式、重复、难度和覆盖；成功过滤可以增强SFT，但会改变prompt和路径分布。学习：rejection sampling、有效样本数、选择偏差。追问：只保留成功会丢掉哪些恢复过程，过滤后是否需要重平衡？

**Q05 [自拟] LoRA与全参怎样选？**

回答：看任务变化范围、可训练容量、显存、部署和合并约束。LoRA减少训练参数及相应梯度/优化器状态，不消除基础权重与激活。学习：[状态与训练后端](#q=infra-training-backends)、低秩更新。追问：低rank能力不足还是学习率/数据造成失败，怎样对照？

**Q06 [自拟] QLoRA为什么仍会OOM？**

回答：基础权重量化不等于激活、长上下文、临时dequant与所有优化器都同样压缩；推理teacher和KV也可能占内存。学习：[显存账本](#q=infra-training-step)。追问：OOM发生加载、forward、backward、同步还是保存阶段？

**Q07 [自拟] Packing与padding有什么训练区别？**

回答：packing可减少空位，但不同样本间attention边界、位置编码、label与loss计数必须正确。不是把文本串起来即可；动态shape也影响kernel编译。学习：causal/block mask、position IDs、global denominator。追问：跨样本注意力泄漏怎么检查？

**Q08 [自拟] 为什么多轮训练要保留错误工具反馈？**

回答：失败操作可能帮助模型定位错误、学恢复；系统故障和真实任务失败应分开。观察仍是条件，通常不做生成loss。学习：[Agentic RL](#q=agentic-rl-training)。追问：删掉报错后这段修复动作的概率还对应原采样吗？

## 4. 面试问答二 偏好与DPO

**Q09 [公开题型] DPO loss怎么写，四个概率是什么？**

$$
L_{\rm DPO}=-\log\sigma(\beta\Delta),\quad
\Delta=(\log p_\theta(y^+|x)-\log p_{\rm ref}(y^+|x))
-(\log p_\theta(y^-|x)-\log p_{\rm ref}(y^-|x)).
$$

回答：chosen/rejected各有current与reference两个序列logprob，通常response token求和。学习：[DPO精读](#paper=paper-dpo)。追问：把序列和改平均，是否还同原目标？

**Q10 [自拟] DPO为什么不用显式reward model？**

回答：在相应KL正则最优策略关系和偏好概率模型下，把reward差转成相对reference的策略logratio，直接拟合偏好。不是“没有任何reward假设”。学习：Bradley–Terry、隐式reward、训练支持集。追问：离线pair覆盖不到新行为时怎么办？

**Q11 [自拟] β大是不是必然更保守？**

回答：理论约束意义与固定pair上的loss尺度要分开。固定$\Delta$时梯度含$\beta\sigma(-\beta\Delta)$，饱和、学习率和数据共同影响更新；不能只背单调口号。学习：sigmoid导数与KL约束推导。追问：初始current=reference时loss和梯度是多少？

**Q12 [公开题型] 什么是高质量DPO数据？**

回答：同prompt、可辨别且可信的偏好、控制难度/领域、排除长度与风格伪相关、去重并验证标签一致性。Pair较“难”不自动更好，噪声会误导。学习：偏好标注、margin、judge校准。追问：teacher分数当偏好时如何防止同模型偏爱？

**Q13 [公开题型] DPO后输出变长或整体变差怎么排查？**

回答：先比较模板/解码/EOS，再看偏好长度偏差、reference起点、β/LR、监督mask、数据噪声和遗忘；做长度受控pair及独立质量评测。学习：序列概率与长度、样本选择。追问：变长是更多正确推理还是重复与停止失准？

**Q14 [自拟] DPO会自动提高chosen绝对概率吗？**

回答：目标约束相对pair margin，策略共享参数可使两者概率都下降但rejected降更多；不保证chosen绝对概率上升。学习：相对logratio及chosen NLL诊断。追问：为何可能加入SFT辅助项，加入后目标怎样变？

**Q15 [自拟] DPO与SFT能互相替代吗？**

回答：SFT示范提供可模仿行为，DPO用pair比较偏好，各自数据与目标不同。冷启动和格式可先SFT，是否再DPO由偏好任务验证。学习：数据支持与起点。追问：只有rule verifier时怎样构造pair，是否优于直接RL？

## 5. 面试问答三 PPO、GRPO与RLVR

**Q16 [公开题型] PPO与GRPO核心区别是什么？**

回答：典型PPO用critic和GAE，GRPO用同prompt组回报相对统计，免去常规独立value模型；ratio、clip、KL与reduction另列。PPO并不限单轮。学习：[算法对照](#q=rl-policy-optimization-map)。追问：同group全对/全错时有多少学习信号？

**Q17 [公开题型] Critic怎么训练，预测reward还是return？**

回答：估计给定状态下未来return，用MC或固定TD/GAE target回归；二元终局、gamma1时才可解释为成功率。学习：[Actor–critic](#q=rl-actor-critic)。追问：任务失败标签复制每个前缀是否说明每轮都坏？

**Q18 [公开题型] Advantage怎样算？**

回答：先定义reward、折扣和边界。PPO可用$\delta_t=r_t+\gamma b_tV(s_{t+1})-V(s_t)$、GAE递推；GRPO用组reward减均值并可除std。学习：[两mask与GAE](#q=rl-returns-gae)。追问：timeout、reset和真正terminal是否一样？

**Q19 [公开题型] PPO、GRPO的KL是两种固定散度吗？**

回答：不能按算法名字决定KL估计。都可能约束current与reference，放reward或独立loss，实现选择不同；PPO clip比较current/proximal，不等reference KL。学习：[PPO接线](#q=rl-ppo-training)。追问：你的代码到底以哪个分布采样、哪个方向、哪个mask算KL？

**Q20 [自拟] Importance ratio初值1，梯度为什么不为0？**

回答：数值1不等常数。$\rho=\exp(\log p_\theta-\mathrm{sg}(\log p_0))$，其导数仍含$\nabla\log p_\theta$；若分母不detach，才可能错误抵消。学习：autograd图与固定锚点。追问：single-forward可以省哪个forward、不能声称省掉什么？

**Q21 [自拟] PPO clip为什么分正负优势？**

回答：悲观min允许增大好动作/减小坏动作，但在过度有利方向截断增益；反方向仍有梯度。学习：四分支手算、dual clip与数值clamp分开。追问：$\log ratio$限±20是不是clip0.2？

**Q22 [自拟] GRPO全同reward为何不是std除零爆炸？**

回答：中心化后优势为零，epsilon保护分母；但意味着无相对反馈。不同实现singleton处理需核，不能把G1当正常组比较。学习：组方差、有效组、dynamic sampling。追问：丢全零组会不会改变任务难度分布？

**Q23 [公开题型] DAPO改了什么？**

回答：不只clip upper，包含探索/clip asymmetry、dynamic sampling有效组、token级聚合和overlong处理等配方。学习：[DAPO精读](#paper=paper-dapo)。追问：哪个改动在你的任务起作用，补采成本计在哪里？

**Q24 [自拟] GSPO、SAPO、CISPO差别怎么讲？**

回答：分别看sequence ratio/聚合、平滑surrogate、裁剪detach importance系数但保留logprob梯度等；不要把所有名字当critic替换。学习：[统一对照](#q=rl-policy-optimization-map)。追问：clip后梯度是否仍存在，长响应权重怎样改变？

**Q25 [自拟] RLVR与GRPO是同一个概念吗？**

回答：RLVR指可验证reward来源，GRPO是一种优势/策略更新组织。可验证结果可以接PPO/RLOO等，GRPO也可以接其他reward。学习：反馈轴与算法轴。追问：计算答案EM和代码隐藏测试分别能漏掉哪些问题？

**Q26 [公开题型] Reward hacking如何处理？**

回答：先检查奖励是否代理了真正任务，区分格式套利、长度、测试漏洞与工具伪造；独立verifier、污染审计与真实任务评价配合。学习：目标错配、held-out tests。追问：训练reward升而真实成功降，首查哪三个日志？

**Q27 [公开题型] Agentic过程评分如何落到token？**

回答：先定义turn/transition，再决定给每个response token广播一个turn优势、估逐tokenreturn，还是用独立dense监督。观察不做actor loss；最终任务reward不自动成为每轮因果标签。学习：[GiGPO](#paper=paper-gigpo)与[turn框架](#q=agentic-turn-framework)。追问：哪些量必须等工具、哪些必须等终局？

## 6. 面试问答四 蒸馏与自己的工作

**Q28 [自拟] OPD更密集就一定更准确吗？**

回答：teacher概率不等任务真值，学生访问的坏状态、停止行为和支持偏差仍会影响。强SFT基线也可能接近。学习：[OPD基础](#q=opd-interview-foundations)、[新审计](#paper=paper-rethink-mopd)。追问：同prompt、独立调LR和同GPU-hour下你的增益还在吗？

**Q29 [自拟] Sampled RKL标量直接backward为什么错？**

回答：冻结已采样index后只对$\log p-\log q$求导，遗漏sampling score项，梯度期望可能0；full-vocab或正确PG surrogate需各自推导。学习：值估计与梯度估计区别。追问：固定prefix与完整state occupancy的导数一样吗？

**Q30 [自拟] Cross-tokenizer为何是两层对齐？**

回答：位置先找相同因果byte前缀，词表再定义公共概率事件；补齐logits维度不解决ID身份，span路径不总等完整事件marginal。学习：[CT题库](#q=opd-interview-cross-tokenizer)。追问：一个teacher token没完全生成时怎么监督？

**Q31 [自拟] ESCD的创新一句话怎么说？**

回答：在已自然访问的child状态监督合法completion集合总概率，不替teacher规定student分词路径之间的任意概率拆分；与root event entry互补。学习：[你的ESCD讲稿](#q=escd-project-interview)。追问：集合允许哪些token、loss是否把总概率拟合成teacher质量？

**Q32 [自拟] 为什么你的turn框架不叫普通PPO加critic？**

回答：当前actor是sampled OPD，独立critic调监督强度，没有另加task PPO actor loss；actor窗口和critic终局轨迹训练条件不同。学习：[critic创新与论证](#q=agentic-turn-innovation)。追问：正value差为何应强化蒸馏，是定理还是待检验假设？

## 7. 面试问答五 Infra、故障和证据

**Q33 [公开题型] 后训练显存通常浪费在哪？**

回答：权重/优化器、激活、KV、full logits、通信与同步临时buffer分开。RL/OPD多角色要按所有预留卡算成本。学习：[Infra账本](#q=infra-training-step)。追问：FSDP/ZeRO分片什么，TP/PP/CP改变什么？

**Q34 [公开题型] 大batch怎么影响off-policy？**

回答：更多in-flight和多epoch可能增加行为版本年龄与当前概率偏移；batch大小不是唯一变量，还看更新频率、队列与loss校正。学习：bounded queues、ratio、ESS与prefix staleness。追问：放大并发为何可能只增排队？

**Q35 [自拟] KV cache能直接用于训练省prefix backward吗？**

回答：推理缓存通常无训练梯度；detach prefix会改变目标。固定teacher可缓存条件一致的评分，actor精确prefix coalescing可在一个因果图里复用计算，二者不同。学习：[turn实现](#q=agentic-turn-framework)。追问：GDN缓存为何还受prefill分块和权重版本约束？

**Q36 [自拟] 梯度累积后每段取mean再mean有什么错？**

回答：变长turn/chunk的token数不同，平均的平均改变权重；应恢复未归一分子，再用完整窗口分母归一。学习：token/turn/episode三种reduction。追问：critic加权时分母是否也应含权重？

**Q37 [自拟] Resume为何不只是load checkpoint？**

回答：模型、optimizer、scheduler、随机性、版本、消费账本与环境状态各自恢复；不确定commit不能盲目重放一次update。学习：幂等identity、receipt和原子边界。追问：日志已有commit但没有checkpoint，能否恢复该模型？

**Q38 [自拟] 训练步数一致是否公平？**

回答：生成token、teacher评分、工具交互、重试、group大小、更新token与总GPU-hour可能不同；至少报匹配起点/协议下几种预算口径。学习：compute与sample efficiency。追问：只计算actor卡时漏了多少系统资源？

**Q39 [自拟] Pass@k、avg@k、mean score怎么区别？**

回答：pass@k看k次是否至少一次成功，avg@k平均单次成功，rubric mean是评分均值；采样温度、任务数与k都影响结论。学习：逐任务配对、置信区间、judge一致性。追问：PHYRD-40提升能否说成SWE pass@1提升？

**Q40 [自拟] 怎样回答“你的贡献不是调参吗”？**

回答：给问题、机制、与近邻差别、可证伪假设和控制实验；工程贡献用正确性/效率/可靠性证明，研究贡献还需消融与独立任务收益。学习：证据层级。追问：若随机权重同样有效，你的critic信息贡献还成立吗？

## 8. 训练回答与问题驱动学习

遇到答不清的题，先画输入/采样/评分/优势/loss/reduction/update流程，再读相应原文，最后手算一个两token或两turn例子。不要看答案后立即认定会了：遮住公式，先解释变量与梯度，再对照代码。

建议模拟：10分钟Q01–08数据，15分钟Q09–15偏好，20分钟Q16–27在线算法，15分钟Q28–32自己的工作，15分钟Q33–40系统与证据。每题先30秒直答，再接受一个具体追问。你自己的工作按[ESCD项目介绍](#q=escd-project-interview)与[turn框架](#q=agentic-turn-framework)展开，而不是从公司题解中借一段泛化话术。

本篇题源可见性与技术依据分开；原文性能、个人实验和CPU检查分别记录。已有长篇讲解不是要求一次背完，而是让每道题都能回到一个可运行/可手算/可核代码的知识点。
