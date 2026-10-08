---
id: infra-rl-pipeline
title: "05 · RL/OPD 训推如何形成闭环？从角色、数据契约到异步与排障"
category: systems
difficulty: 进阶
tags: ["P0", "Infra基础", "verl", "slime", "RL系统", "OPD", "异步"]
updated: 2026-10-08
summary: "用一条样本追踪rollout、评分、训练和权重发布，拆清behavior/current/reference/teacher概率、组屏障和有界队列，建立低利用率排查与验收方案。"
draft: false
---

## 1. 主问题与先修：框架把哪些依赖连接起来？

本篇接回 [verl/slime选择与使用](#q=training-inference-frameworks)：选好框架以后，怎样确认闭环是正确且高效的？前四篇分别准备了 [训练一步](#q=infra-training-step)、[GPU与通信](#q=infra-gpu-performance)、[多卡并行](#q=infra-distributed-parallelism)、[推理引擎](#q=infra-inference-engine)。现在把它们组织起来。

算法先修只需知道：SFT从固定标注学习；RL从策略执行得到奖励信号；OPD由学生生成前缀并取得teacher监督。更完整的目标推导见 [SFT/DPO/RL](#q=sft-dpo-rl)、[GRPO](#q=grpo-rlvr)、[OPD](#q=opd-vs-sft)。本篇围绕工程依赖，不重复全部算法历史。

核对日2026-10-05。架构、队列和指标案例为学习性设计，没有跑GPU训练。已有论文实测只作为作者条件内证据，具体配置与局限链接到精读报告。

## 2. 同步闭环：一次更新从哪里开始，到哪里结束？

一条典型在线训练路径是：取prompt → policy rollout → 环境/奖励或teacher评分 → 构造训练目标 → learner forward/backward/step → 保存/发布新权重 → 下一批rollout。不同算法会省去或新增角色，不要把全部模型都默认加载。

![本文示意：生成、评分、learner、权重发布与有界ready队列](./assets/infra/infra-rl-pipeline/rl-pipeline.svg)

**图解：** 实线为样本数据向learner流动，回边为新权重发布到rollout；评分角色按算法选择reward/reference/teacher。ready queue承接已完成数据，准入背压限制在途工作。工具/环境是rollout中的等待来源，可以不在GPU上。箭头表达依赖，不表示所有数据必须经过控制器，也不表示开启queue就自动得到异步训练。

| 角色 | 主要输入/输出 | 是否训练 |
|---|---|---|
| actor / policy learner | token、目标、概率 → 梯度、新权重 | 是 |
| rollout engine | prompt、采样参数、policy权重 → action/token与行为信息 | 通常只推理 |
| reference | 固定基线模型与token → logprob/KL所需量 | 通常冻结 |
| reward / verifier | 轨迹或答案 → 任务信号 | 本轮可冻结模型或程序 |
| critic | 状态/token → value，用于优势估计 | PPO类路径可训练 |
| teacher | 学生prefix/候选动作 → 蒸馏分布或分数 | OPD中通常冻结 |

reference用于约束或比较策略，teacher提供蒸馏目标，即使都“只前向”也不是同一职责。GRPO常由同prompt多个回答构造相对优势，不需要独立learned critic；具体KL/reference是否启用要看变体与配置。[DeepSeekMath](https://arxiv.org/abs/2402.03300v3)。

HybridFlow将算法控制流与各角色内部的分布式计算组织结合，用角色/资源的分离支持不同放置。选择verl/slime后，仍需要自己解释每个角色要哪些数据、能否并发、在哪张卡上。[HybridFlow](https://arxiv.org/abs/2409.19256v2)。

## 3. 数据契约：一条多轮样本必须带什么？

### Q：为什么“返回文本+reward”可能不够？

训练要还原模型实际看见的prefix、选过的action、使用的概率与最终目标。重新tokenize文本可能因模板、特殊token或截断改变序列。工具输出属于条件，不一定属于学生action，loss mask应明确。数据字段依框架有差异，下面是概念契约，不能直接当某个版本的类定义。

```text
sample_id, prompt_id, group_id / task_id
input_ids, attention_or_segment_metadata
response/action_positions, loss_mask
behavior_logprobs, sampling_config
policy_version_per_segment, tokenizer/template_revision
reward_or_teacher_scores, termination_reason
tool_observations, environment_snapshot_reference
produced_at, scored_at, consumed_at, retry/attempt_id
```

- `sample_id`与attempt分开：重试可能是同任务的新采样，避免重复记账或重复累积梯度。
- group ID表达算法依赖；task ID表达环境轨迹；一个task可含多个LLM请求，不能只用request ID。
- version若一条轨迹中途切换，需要记录位置段，单一“完成时版本”会丢掉真实行为信息。
- termination reason区分EOS、长度截断、工具失败、主动早停、系统abort，不要全部当正常完成。
- environment snapshot还可能含文件改动、工具会话、进程与外部状态；仅保存token不能保证coding任务可恢复。

### Q：prompt或工具文本loss=0，能从输入删掉吗？

不能。loss mask控制哪些位置被监督，attention条件仍依赖前面的prompt和observation。正确做法是保持真实条件，只在定义的action位置计算目标。sequence packing也必须明确不同样本是否隔离attention。见 [多轮梯度与mask](#q=agent-gradient)。

**最小正确性检查：** 给一条确定轨迹，逐位置打印token、角色、mask与logprob；确认tool observation没有被当成policy action，终止token与截断策略一致；用单样本loss与batch/packing loss比较。先证明这件事，再增加并发。

## 4. 概率一致性：behavior、old、current到底是谁？

behavior policy是真正生成action的分布，current policy是learner当前模型；算法中的old/proximal policy还可能是冻结用于限制更新的参考点。reference model则可长期固定。只有在特定同步流程下部分对象才重合；变量都叫`old_log_prob`不代表语义一样。

给定prefix s和生成action a，一个常见比值是：

$$
r(s,a)=\exp(\log\pi_{current}(a\mid s)-\log\pi_{behavior}(a\mid s))
$$

计算时先用logprob差，检查非有限值与极端ratio，必要的clip/mask必须按目标明确定义。把rollout logprob换成learner事后重算的“old”值，可能改变纠偏对象。

### Q：权重相同，训练与rollout概率为何也不完全相同？

可能是精度、attention kernel、模板、position IDs、padding/packing路径或采样变换。temperature、top-k/top-p作用后，“原模型softmax概率”和“实际采样分布”也可能不同。IS分母应该匹配算法所校正的behavior定义；不要假定API返回logprob已经采用你需要的归一化。

应在同权重、同token prefix与目标token上，比较两端logprob差的均值/分位数、ratio分布与有效样本量等，再核对训练使用的公式。verl有Rollout Correction文档讨论训练/生成不匹配及不同校正模式；本文不把其中任一模式当所有OPD目标的通用补丁。[verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)。

### Q：OPD为什么特别需要核对teacher缓存？

全词表teacher forward KL与student reverse KL的权重方向不同。异步时学生prefix和动作来自旧分布，缓存可能包含old student选择的top-k token、teacher概率及old logprob。仅重算当前学生已缓存动作的概率，并不自动补齐未缓存动作。

AsyncOPD研究旧采样、KL方向与缓存的关系，提出当前学生端重算部分reverse-KL信号、old-to-current IS和多动作MC来改善相应估计。它对动作层的校正，不能让旧学生访问的状态前缀自动变成当前学生分布。[AsyncOPD固定v1](https://arxiv.org/abs/2606.24143v1)、[详细精读](#paper=paper-asyncopd)。

工程上先列三栏：teacher固定可缓存量；behavior版本依赖量；current learner需重算量。再逐项验证loss梯度是否真的流向current模型、teacher是否冻结、mask是否相同。不要把PPO校正直接贴到蒸馏loss后就宣布保持on-policy。

## 5. 四个边界：ready、backward、step、publish

### Q：异步系统到底把哪条屏障移除了？

**Ready**表示某个目标可计算；**backward**产生梯度；**step**改变learner参数；**publish**改变后续rollout使用的权重。四个事件可分离。

| 改造 | 被提前的事件 | 算法/系统条件 |
|---|---|---|
| 固定参数先算microbatch梯度 | backward | 目标与全局归一化已确定，累积期间参数不变 |
| 完整GRPO组交付 | 数据ready/训练交付 | 该prompt所有组内统计已齐 |
| 生成与训练持续并行 | step与后续rollout重叠 | 旧数据、积压与behavior概率处理 |
| 一条轨迹途中同步权重 | publish影响未完成轨迹 | token段版本、混合行为与KV策略 |

FlexMARL、RolloutPipe等工作的区别见 [异步专题](#report=survey-async-training)。逐turn teacher反馈能够提前得到监督，不等于已证明可以独立提前backward：终局奖励、组内归一化或全局token分母仍可能未定。

### Q：为什么GRPO组内长尾是算法屏障？

若优势依赖同prompt一组回答的均值/标准差，单个回答完成不代表它的最终优势已知。引擎可以持续调度，却仍需等组内数据。将一个完整组作为交付单位，与“task内某个turn完成就训练”区别很大。改变为单rollout+critic会改变算法，不只是把queue实现换掉。

**例子：** global batch有8个prompt，每prompt4条回答。30条已完成、2条慢，受影响可能是2个组，不一定所有组都不能准备训练。但若既定优化器更新必须等完整global batch，则可以提前backward并累积，仍保留step屏障；如果目标未定，不能用临时均值先算后假设结果等价。

## 6. 有界队列、partial rollout与replay：pool存什么？

| 对象 | 存入/取出 | 必须记录 |
|---|---|---|
| ready queue | 已完成评分数据入，消费一次出 | sample ID、behavior版本、消费去重 |
| continuation buffer | 未完成任务保存，后续续跑 | prefix、版本段、环境状态 |
| teacher cache | 可复用监督量 | teacher版本、prefix、候选支持、score定义 |
| gradient buffer | fixed参数下已算梯度 | 参数版本、归一化、是否已累积 |
| replay pool | 完成样本重复抽样更新 | 抽样概率、重复次数、偏差处理 |

有queue不等于replay。APRIL保存未完成prefix、完成足够组后先训练；不表示直接用未完成片段训练。[APRIL](https://arxiv.org/abs/2509.18521v1)、[精读](#paper=paper-april)。AReaL通过投放控制、版本边界等管理异步积压，并在算法侧区分behavior与proximal。[AReaL v1](https://arxiv.org/abs/2505.24298v1)、[精读](#paper=paper-areal)。

### Q：为什么需要背压，而不是无限发任务？

背压是在生产者太快时限制新工作，控制在途与已就绪数据，避免内存、环境容量、过时和长尾恶化。教学例子：生产12样本/s、消费8样本/s，不限流每秒积压4个，5分钟多1200个。平均等待也会增加。

在稳定、长期平均且流入流出守恒的系统中，Little定律 $L=\lambda W$ 将队列内平均数量、吞吐和平均停留时间联系起来。ready队列均值80、稳定消费8/s，对应平均ready等待10s；这不包含rollout执行时间。若队列持续增长，上述稳定假设不成立，不能机械套式子。

限制“未消费工作”可以包括正在生成、等待评分和ready三类，不只是完成后按年龄丢弃。年龄秒数与版本差也是两维：learner更新变快，同一10秒可能跨更多step。

### 当前代码的边界：文档里的async参数不可望文生义

- [verl Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html)的`staleness_threshold`文档描述为允许过时样本比例相关的投放控制，不是简单“最多落后几个版本”；recipe还组合sync interval、require batches和partial rollout。应读公式与实际消费路径。
- [slime固定README](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/examples/fully_async/README.md)说明后台worker持续运行、完成组进入输出queue；含ABORTED的组重新入data buffer。该版本尚未接通ABORTED的partial-resume，会从头开始，不能写成已经实现环境续跑。

这些是固定资料的事实边界，不代表你的分支相同。接入时核对checkout commit与自定义generate返回字段。

## 7. 低利用率排障：以14.4%为问题线索，不当作本次测量

用户前述Generate 14.4%利用率是待解释的既有观察，本次未复测。第一步问采样方式和统计窗口：是GPU busy平均、SM指标、还是rollout active ratio？分母含工具/阶段等待吗？否则无法与他人的论文图直接比较。

建议记录一个任务的时间分解：environment setup、队列等待、prefill、decode、工具执行、teacher scoring、learner等待、训练、权重发布。请求级与task级分别计时，多轮返回后的排队会重复累积。

| 观察 | 判断所需证据 | 优先实验 |
|---|---|---|
| active agent多、ready请求少 | tool wait占比、ready队列曲线 | 扩展合法环境并发、异步工具、任务级调度 |
| ready请求多、GPU有空洞 | scheduler/CPU trace、engine budget | 调整调度、batch与提交开销 |
| KV满且频繁抢占 | cache占用、重算量、长度分布 | 准入token预算、并行副本/TP对照 |
| rollout等learner发布 | 版本窗口与sync耗时 | 有界队列、重分片与同步路线 |
| learner等数据 | ready产出、评分延迟、组屏障 | 先优化真实上游，再考虑角色配比 |
| token/s上升但任务收益没升 | 失败/截断/丢弃比例、任务难度 | 相同GPU-hour质量与覆盖对照 |

Heddle/TideRL提示多轮任务调度与ready资源匹配的重要性，见 [Heddle](#paper=paper-heddle)、[TideRL](#paper=paper-tiderl)。不必等全面异步化才能优化工具返回后的排队。另一方面，单纯扩大池可能加剧KV和环境竞争，需要一起看准入与尾部。

## 8. 最小baseline与验收：先证明正确，再比较收益

1. **同步闭环**：小数据、确定模板和reward；确认生成→评分→更新→发布及恢复。
2. **概率一致性**：相同权重/token分别算两端logprob，定位模板/精度/采样变换差异。
3. **固定版本提前backward**：保持既定step，验证梯度与同步分组误差；若loss依赖不满足则不做。
4. **有界async单次消费**：增加queue/背压，保持sample去重；记录版本跨度和消费延迟。
5. **版本与partial对照**：整轨迹固定版本 vs 中途更新；保存token与恢复环境分别验证。
6. **OPD目标对照**：缓存与current重算/IS分别比较，检查mask、支持与方差。

每轮固定模型/teacher、任务集合、GPU和时间预算、loss定义、长度与精度。输出：阶段时间线、ready/active数量、已生产/评分/消费/丢弃样本数、版本差分布、ratio/ESS/mask率、任务成功率与GPU-hour。不要只汇报step更快，也不要把最大输出长度调短当作无成本系统加速。

源码伴读路线（本次只核对文档与固定README，不宣称训练复现）：verl从控制器角色调用，追到rollout输出、compute logprob、advantage与update policy、weight sync；slime从train入口，追rollout function、Sample返回、Megatron训练与权重同步。完整文件位置以固定树核对：[verl代码](https://github.com/verl-project/verl/tree/8718ca30a3f002f93b7c4fd99b9b2506718681bc)、[slime代码](https://github.com/THUDM/slime/tree/8c17b676cb57af1d17ee4402e91e9209af84b60b)。

## 9. 面试问答：从概念答到方案评审

以下为本文结合论文、框架与作者系统面试讨论整理的练习；并非公司真题。[Chip Huyen的系统面试讨论](https://huyenchip.com/machine-learning-systems-design/research-vs-production.html)强调约束与生产行为，本篇将这种问法扩展到训推系统。

**Q01 [基础] actor、reference、teacher的区别？** actor被更新，reference用于约束比较，teacher给蒸馏信号；共享“前向服务”实现也不等于同职责。追问：GRPO一定加载critic吗？不一定，按算法变体核对。

**Q02 [机制] 为什么token IDs比纯文本更重要？** 要还原实际条件与action，重tokenize可能受模板和截断影响。追问：tool文本loss=0可删除吗？不可，仍是后续条件。

**Q03 [机制] rollout与learner同权重，ratio一定1吗？** 理想相同分布是，但数值/模板/采样变换可能不同；比较同prefix token的logprob。追问：纠偏分母是什么？实际定义的behavior分布。

**Q04 [设计] 异步是否就是提前optimizer step？** 不是，ready/backward/step/publish可独立变化。追问：保更新边界能先backward吗？目标、分母与组依赖已定且参数保持固定时可评估等价性。

**Q05 [设计] pool如何避免off-policy？** 先说明对象和单次/重复消费；背压控制积压但不证明分布完全on-policy。追问：age阈值与version阈值一样吗？不是，更新速度影响对应关系。

**Q06 [排障] 长coding轨迹保存prefix就能恢复吗？** 还需文件、进程、工具与环境状态；重放工具可能有副作用或不同结果。追问：怎么验收？相同snapshot续跑的观察一致性，区分重启新任务。

**Q07 [设计] GRPO组没齐可先训练吗？** 若组统计未定不可以假设目标已定；可准备其他完整组或改变算法。追问：OPD需要继承组屏障吗？仅当其实际目标有该依赖。

**Q08 [排障] 14.4%利用率首先调什么？** 首先核指标与时间窗口，再看ready而不是总active、工具等待、队列与抢占。追问：为何不先升TP？问题可能根本不是单次模型算力不足。

**Q09 [推导] 生产12/s、消费8/s会怎样？** 每秒净积压4，5分钟1200；不能用稳定Little定律假设队列已平稳。追问：限哪个量？明确生成中、评分中与ready共同容量。

**Q10 [评审] 更高rollout token/s能证明更好训练吗？** 不能，需相同GPU-hour任务收益、质量、覆盖与消费比例。追问：哪些日志证明没有省掉难任务？终止/丢弃原因、长度与任务分布、sample生命周期账本。

## 10. 复习练习与来源

**纸上练习：** 为一条两次工具调用轨迹标角色/token/mask/版本；画四个更新边界；分别定义三种buffer；按生产消费速率画队列。**实际工程练习待做：** 在你的checkout跑同步小闭环，并写一份数据契约与概率一致性报告，再逐项加异步机制。

| 来源 | 类型 | 本篇用途 |
|---|---|---|
| [HybridFlow](https://arxiv.org/abs/2409.19256v2) | 原论文 | 角色、控制流与资源 |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300v3) | 原论文 | GRPO组依赖 |
| [verl Performance](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) / [Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) | 官方教程 | 分阶段调优与概率对象 |
| [verl Fully Async](https://verl.readthedocs.io/en/latest/advance/fully_async.html) | 官方recipe | 过时与投放控制 |
| [slime固定README](https://github.com/THUDM/slime/blob/8c17b676cb57af1d17ee4402e91e9209af84b60b/examples/fully_async/README.md) | 固定官方源码说明 | 后台worker、组queue、resume缺口 |
| [AReaL v1](https://arxiv.org/abs/2505.24298v1) / [APRIL v1](https://arxiv.org/abs/2509.18521v1) | 原论文 | 有界async与continuation |
| [AsyncOPD v1](https://arxiv.org/abs/2606.24143v1) | 原论文 | KL、缓存、旧数据 |
| [Chip Huyen面试讨论](https://huyenchip.com/machine-learning-systems-design/research-vs-production.html) | 作者材料 | 指标与约束追问 |

[本篇来源清单](./research/infra-rl-pipeline-sources.json)。五篇到这里形成“状态→硬件→并行→推理→闭环”的基础复习链。下一轮可按实测缺口细化checkpoint恢复、MoE负载、低精度kernel或环境snapshot，分别成篇，不再堆到框架选型文内。

**数值练习下载：** [CPU验算脚本（仅Python标准库）](./assets/infra/infra-foundations-check.py) · [本次验算输出](./assets/infra/infra-foundations-check-results.json)。覆盖梯度有限差分、状态/KV字节、Roofline、TP与分块softmax；不包含GPU训练或测速。

环境侧精读：[DSec沙箱平台](#paper=paper-dsec)说明镜像/层组合、guest内存与CPU QoS，以及GPU trainer被抢占后保留agent loop和sandbox状态的路线；其系统实验与RL学习收益分别解读。
