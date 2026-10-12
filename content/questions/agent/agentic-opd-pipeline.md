---
id: agentic-opd-pipeline
title: "Agentic OPD执行流程、异步pipeline与Infra接线"
category: agent
difficulty: 进阶
tags: ["P0", "Agentic OPD", "Pipeline", "Async", "Infra", "Project"]
updated: "2026-10-12"
summary: "沿任务准入、sandbox、模型请求、轨迹重建、教师评分、learner和权重发布讲全链路；解释队列、背压、混合版本、KV与恢复及性能排查。"
draft: false
---

## 1. 问题背景 为什么有很多GPU仍然训练等rollout

多轮coding/search任务不是持续占用推理GPU：模型生成一段动作，调用工具，等待进程/网络/环境，观察返回后再排队生成。整批训练等待所有任务完成时，一个长轨迹会拖住learner；即使Generate阶段内，GPU也可能因为可运行请求不足、工具等待、重复prefill、长尾排队而空闲。总活跃agent数不等于可运行推理数。

本篇集中讲从agent执行到OPD更新的接线；基础KL与梯度先读[Agentic OPD](#q=agentic-opd-training)，基础PPO先读[训练一步](#q=rl-ppo-training)。算法loss正确和系统数据正确缺一不可。一个可以跑通的异步demo，仍可能因上下文重建不一致、旧行为概率覆盖、任务丢失而优化错误目标。

这里给的是可以用于项目设计和答辩的契约。公开ESCD源码用于跨tokenizer评分与loss实例，原始论文用于异步机制；不把某个设计记录中的配置、训练结果或研究假设写成已验证项目成绩。

## 2. 前置知识 角色、单位和三个时间

角色包括任务producer、agent controller、sandbox/tools、student serving、teacher scoring、learner、weight synchronizer。Actor可能指训练策略模型，也可能被工程代码用来指Ray actor进程，解释时应指明。Teacher是蒸馏目标，reference是RL约束目标，critic是价值估计；同一个服务器可能承担多角色，但数学职责不合并。

单位从小到大是token、一次LLM请求、turn、完整trajectory、同prompt group、优化minibatch、全局update。可以按请求切权重而不等trajectory；可以按完整trajectory交付而不等整批；但GRPO组优势可能仍等同prompt组完成。Token-level局部OPD未必有组等待，不能无故继承GRPO屏障。

三种时间必须独立：轨迹生成时间、排队居留时间、策略版本年龄。旧版本轨迹可能刚完成，墙钟很新；短任务也可能排队很久。版本差是更新次数，不是一个固定概率距离；频繁小更新与少数大更新相差明显。

## 3. 一张图串起执行链与训练链

![教学图 · 有界任务执行、评分、消费与权重发布](./assets/learning/agentic-opd-pipeline/pipeline.svg)

上方执行链按真实工具状态推进，已完成的可训练对象进入评分与ready queue；下方learner消费、更新、发布权重。回线不能倒灌未来观察进已保存的训练prompt。图为本文逻辑示意，不是厂商图或性能实测；每个箭头还需要下文的数据、版本和恢复契约。

| 阶段 | 输入 | 输出与必须保持的条件 |
| --- | --- | --- |
| 准入 | 唯一task ID、prompt、环境定义 | attempt ID、预算、队列permit；可审计任务覆盖 |
| 建环境 | 镜像/依赖、任务初始状态 | sandbox ID、工作区基线、工具会话 |
| 生成 | 当时模型真实输入 | 原始token、实际采样logprob、请求版本与停止原因 |
| 执行 | 解析后的动作 | 真实工具结果、状态变化、退出码/timeout |
| 重建 | 多次请求记录 | 可重算的逐调用样本或精确连续轨迹、mask和版本span |
| 评分 | 同条件student文本/token、teacher版本 | sampled score/分布/hidden payload、对齐有效性 |
| 消费 | 已就绪样本 | 去重、staleness检查、合法batch、全局loss分母 |
| 更新发布 | frozen数据、current模型 | backward→optimizer→新权重→serving确认 |

准入、生成、完成、评分、消费和训练成功都应有独立事件。只记“这道题结束”不能知道它是否真的贡献了梯度；sample/hour应以唯一训练消费数作分子，并另计被拒/重试的生成成本。

## 4. 一条coding轨迹如何成为训练样本

模型先读题与repo摘要，生成read_file；工具返回文件内容，模型再生成edit；编辑修改真实工作区，随后run_tests返回输出。TokenProxy或调用适配器应记录每次LLM实际prompt与response，以原始tokens/logprob为准。工具观察不会产生actor logprob，却影响下次条件。

有两种合法组织：把每次真实LLM调用作为独立样本，或重建可证明匹配的一条连续历史。不能先截断、总结或重排上下文，再对拼出的整条文字求概率，假装等于rollout时输入。每次调用若有不同system prompt、summary、工具schema或role模板，拼接不自动保持概率条件。共享模型无更新也可能因为前缀改变出现ratio≠1。

Loss mask只保留assistant生成token。工具输出、环境报错、注入提示、padding应分别明确；模型自生成的格式标签是否有loss与手动插入标签不同。Attention仍看实际观察。每个teacher score索引都必须对到相应student target token，而非相邻hidden row；自回归shift错误会把“看到答案后的hidden”当成预测该答案，形成泄漏。

```python
# Teaching contract: fields illustrate semantics, not project API.
record = {
    "task_id": task_id, "attempt_id": attempt_id, "request_id": request_id,
    "prompt_token_ids": actual_prompt_ids,
    "response_token_ids": sampled_ids,
    "behavior_logprobs": sampled_logprobs,
    "behavior_versions": version_spans,
    "assistant_loss_mask": action_mask,
    "stop_reason": stop_reason,
    "environment_snapshot_ref": snapshot_ref,
}
# Teacher/current-student scoring must use the recorded causal contexts.
```

保存文本便于人读，原始token用于计算，两者不应互相替代。模型目录、tokenizer revision、模板hash、精度与采样top-p/top-k/temperature也是概率契约；processed sampling logprob与raw模型logprob要明说，不能同名混用。

## 5. Teacher scoring如何独立服务

同词表sampled OPD只需教师对学生实际token的logprob，可以prefill整条响应一次评分，不需要教师自己重新生成答案。Full-vocab KL则需要每位置的概率分布，$[L,V_T]$张量传输和驻留很昂贵。跨tokenizer还要teacher重编码学生可见文本，保存双方位置映射。

一种公开代码实例来自[ESCD teacher接口](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/opd/cross_tokenizer/sglang_teacher.py)：teacher传送causal hidden rows，learner用冻结teacher LM head重建分布，分块处理词表/位置。它没有消除teacher forward或LM-head乘法，只改变传输与内存组织。Payload还校验request ID、格式、input长度与模型checkpoint，防止请求串线。

以**假设尺寸**$L=4096,V_T=150000,d_T=4096$、BF16为例：全logits约$4096\times150000\times2=1.2288$GB（十进制）；hidden约33.55MB，二者比值约36.62。只是通信规模估算，不是项目实测，也未包含teacher head驻留、额外数据或pipeline latency。公开例子是LM head线性重建，遇到额外归一层/量化头需保持原forward一致。

教师缓存key至少包括teacher权重、tokenizer/template、实际完整prefix、候选token/事件、temperature与score模式。同task不同turn不能共用一份评分。Frozen teacher可缓存同样条件的$q$；若某个advantage还含学生概率，学生更新后那部分不能当永久teacher cache。

## 6. 队列、pool和背压具体存什么

区分pending任务队列、active工具/推理任务、partial轨迹与环境快照、scoring队列、ready完整样本、learner batch、历史replay。Score buffer用于阈值，与训练replay也不同。最简单baseline是“生成一次、暂存、消费一次”的有界队列；无需立刻变成重复训练历史样本。

Producer投放前获取permit，消费者训练成功后释放。工作量上限要覆盖active、正在评分、ready以及worker持有但未入队的已完成对象，否则只限制ready queue仍会无限积压。Bytes/token预算比仅数轨迹更能控制变长任务内存。把sandbox并发限制、KV token容量和teacher scoring容量分别设置，再由统一准入决定是否投放。

稳定状态下Little’s law可帮助解释$N\approx\lambda W$：提高在途数$N$可能扩大排队和数据年龄，不一定提高持续服务率$\lambda$。它的条件是可稳定统计的系统与一致计数，不拿启动/最终drain区间套结论。应该同时量ready队列、可运行LLM请求、工具waiting、teacher waiting和learner idle。

First-ready消费消除批次长尾，却可能先训练容易/短任务。若把难任务丢弃后报吞吐，任务分布也改变了；应记录准入/训练唯一ID，限制等待或调度优先级，保留最终覆盖。重生成旧任务的所有尝试成本都计入账本，不能只统计最后成功那次。

## 7. 权重发布与请求边界

选择至少三类：整条trajectory固定一个版本；只在LLM请求边界切版本；生成中的token级切版本。DORA偏向多版本保持慢轨迹，PipelineRL/部分AReaL路径允许生成途中更新，各有资源与概率记录成本。不能把“trajectory内不换”与“请求内不换”合并描述。

请求边界发布可暂停新请求、等待已发请求真实完成、推送新权重、确认serving加载后恢复。工具执行可与发布重叠。Client timeout只说明调用方没等到结果，不证明GPU请求已停止；若在后台旧请求仍活跃时宣布drain成功，会产生版本标记错误。系统应以服务端完成/取消确认或明确故障状态处理。

权重分片与optimizer checkpoint不同，训练端FSDP/ZeRO/Megatron的布局也不等于serving所需TP格式。同步需要参数名、shape、dtype、分片映射、通信组一致；optimizer状态通常不发到推理服务。Serving权重更新后旧KV不能无条件复用，因为hidden/attention key-value依赖旧权重；重新prefill或框架明确的版本隔离需要计成本。

[AReaL异步文档](https://areal-ai.io/docs/en/algorithms/async.html)将max_head_offpolicyness写为版本控制并支持partial跨版本；[verl fully_async recipe](https://verl.readthedocs.io/en/latest/advance/fully_async.html)的staleness_threshold是旧样本比例/生产额度相关控制，不是“落后几个版本”。配置迁移时先查单位，不照抄数字。

## 8. Learner端哪些量冻结，哪些重算

冻结行为token/logprob、生成时context和版本；current student在同context重算；teacher若固定且条件完全相同可复用score。对于固定prefix的reverse KL，当前动作权重依赖当前学生，旧advantage可能需要重算：

$$
\rho(a,h)=\frac{p_\theta(a|h)}{p_{\rm beh}(a|h)},\quad
A_\theta=\mathrm{sg}(\log q(a|h)-\log p_\theta(a|h)).
$$

[AsyncOPD](https://arxiv.org/abs/2606.24143v1)区分action与prefix staleness。固定prefix下IS可以校正旧动作采样，要求支持集；只给token比值不能把旧策略已经进入的错误环境状态变成当前策略会到达的状态。Clipped IS进一步降低方差并引入偏差，需报告ESS、clip比例与质量。

Forward KL教师加权仍会受到访问prefix过时，只是局部动作权重不依赖学生。Old student top-k缓存漏掉新学生喜欢的候选，重新加权无法凭空补$q$。Teacher评分先完成不意味着所有loss目标已知；需要终局reward、组统计或advantage whitening时仍要等相应数据。

## 9. Partial rollout与环境恢复

Pause/resume至少保存token prefix、请求边界、采样状态、版本spans与工具状态。Coding还包括文件改动、进程、工具会话、工作目录、包状态；search可能包括浏览器cookie/session、检索索引和页面快照。仅保存token然后重新创建空sandbox，通常不叫语义等价续跑。

中断后继续生成与把未完成片段立即训练是两种操作。局部token蒸馏可能已有信号，可以设计早交付；终局RL优势通常未就绪。每turn局部loss若独立已知，提前backward累积可以保持既定optimizer边界，但共享参数必须不在相关forward/backward之间变动，microbatch权重与全局分母须一致。

NCCL collective或model-parallel backward仍可能需要rank同步。某个turn准备好不等于单独一个rank能任意进入collective；任务的ready控制应在数据与分布式执行契约两层处理。需要比较“评分提前”“backward提前但update固定”“update也异步”三个实验，避免归因混杂。

## 10. 性能账本与排查顺序

先画真实时间线：准入→sandbox准备→LLM排队/prefill/decode→工具执行→teacher→ready等待→student重算→backward/optimizer→权重drain/push→checkpoint。异步区间重叠，阶段时长相加不等于wall-clock，不把最早时间到最晚时间当一个角色实际busy总时长。

GPU utilization.gpu是设备活跃度，显存占用是容量，MFU需要实际模型FLOPs与理论峰值。利用率低时先看是否真的有足够可运行请求，再查工具并发、重排队、KV容量、TP/通信与prefill，不立刻加GPU。teacher瓶颈、learner饥饿和learner阻塞对应不同资源调整。

质量与性能同时报告：同任务集、起点、teacher、更新与总token预算；独立pass@1/SR、错误恢复、空patch、长度/轮数、OOD；唯一训练样本/hour、有效action tokens/GPU-hour、总卡时、重试浪费、版本年龄分布与telemetry覆盖。先同步正确性，逐步增加有界异步，必要时匹配校正-only对照。

## 11. 面试问答与反问训练

**Q01 [执行] 工具观察要不要算loss？** 默认不是actor动作，但保留真实上下文；手动标签与自生成标签分开。

**Q02 [数据] TokenProxy记录文本就够吗？** 不够，要原始token、概率、实际prompt、版本、停止原因。

**Q03 [队列] pool一定replay吗？** 暂存消费一次与反复抽样不同。

**Q04 [预算] Ready queue有界就全系统有界吗？** active、score与worker持有对象也须计入。

**Q05 [更新] Request boundary等于trajectory固定策略吗？** 一轨迹可有多个request版本，但每request固定。

**Q06 [取消] Timeout后能马上推权重吗？** 需确认backend请求已结束或安全进入故障状态。

**Q07 [KV] 重用prefix token等于重用旧KV吗？** token可保留，新权重下KV通常需要重建/版本隔离。

**Q08 [异步] IS能纠正错误工具状态吗？** 固定prefix动作校正不自动修正状态占用。

**Q09 [传输] Hidden返回省掉teacher forward了吗？** 没有，转移分布重建位置并减少某类通信。

**Q10 [早算] Teacher提前就能backward吗？** 核全部目标依赖、全局分母与collective；局部OPD和终局RL不同。

**Q11 [指标] 显存很满说明MFU高吗？** 不说明，容量与计算效率分开。

**Q12 [公平] 丢慢轨迹后吞吐高可报成功吗？** 必报覆盖与浪费、质量，不能以改变任务分布充当纯系统收益。

## 12. 来源、源码与复习入口

Primary依据为[AReaL](https://arxiv.org/abs/2505.24298v1)、[AsyncOPD](https://arxiv.org/abs/2606.24143v1)、上述官方文档与[ESCD固定源码](https://github.com/Tammytcl/ESCD/tree/befd0c0a582ec6efc055660467db95d522519591)。源码只静态核读；本文没有运行真实GPU pipeline或声明项目收益。

后续读[OPD面试基础](#q=opd-interview-foundations)、[跨tokenizer项目问答](#q=opd-interview-cross-tokenizer)和[项目展示与答辩](#q=opd-project-defense)。先画一条真实轨迹和完整版本时间线，再逐项解释为什么该样本可以被这个loss消费。
