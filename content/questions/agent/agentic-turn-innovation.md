---
id: agentic-turn-innovation
title: "Agentic Code OPD创新论证：在线Critic、turn权重与双更新时机"
category: agent
difficulty: 深入
tags: ["P0", "TRAIL", "Critic", "Turn OPD", "Research", "Interview"]
updated: "2026-10-13"
summary: "把当前TRAIL机制、标准value critic、监督收益critic和PPO备选方案分开；讲完整目标、版本、冷启动、共驻、12道创新追问与可证伪实验。"
draft: false
---

## 1. 问题背景 可以提前训练的turn，还需要同样强度的监督吗

Turn流水线减少等待，但各轮对学生任务能力的影响不同。Teacher likelihood给局部语言监督，不能直接告诉我们这一轮让任务更容易成功还是更难。你的当前TRAIL研究路线增加独立在线Critic，从真实终局结果学习状态成功率，再用同轮前后value差调整蒸馏强度。

这是一项需要实验支持的假设：预测进展可能帮助分配监督，但进展不是教师监督的边际收益。成功率下降的错误轮也可能更需要纠正，价值上升的熟练轮反而学习空间有限。面试应明确机制与待检验假设，不能仅凭“增加critic”或“权重有方差”宣称新算法已经更好。

本篇依据指定turn工作树的`critic_weighting.py`、`single_forward.py`与`docs/trail/reports/method-and-evidence.md`。代码身份沿[源码账本](./assets/learning/agentic-turn-framework/code-evidence.json)核；工作树含未提交内容。原始运行记录用于区分局部/短程和最终实验，不将旧文档中的运行状态写成2026-10-13实时状态。

## 2. 前置知识 四种Critic概念必须拆开

| 对象 | 预测/作用 | 当前路线关系 |
| --- | --- | --- |
| State value $V^\nu(h)$ | 固定参考策略从历史续跑的预期return | 当前在线成功率模型的理想参照 |
| Action value $Q^\nu(h,a)$ | 先执行动作、再续跑的return | 可选action_advantage要求相应边界训练 |
| PPO critic | 提供TD/GAE task advantage与value target | 项目保留教学/联合PPO设计，但当前Actor不加该task loss |
| Supervision-benefit critic | 增大某turn训练权重后，下游任务效能改变量 | 更严格、昂贵的备选研究设计，未由现有终局标签直接训练出来 |

Current critic拟合近期多个学生版本数据下的结果概率，不能说是当前Actor严格$V^\pi$。Teacher confidence尖锐度也不是任务value。三者都输出数值，不意味着可以在loss里直接互换。

## 3. 当前实现：双数据流，两个更新时机

![教学图 · 局部Actor更新与终局Critic监督](./assets/learning/agentic-turn-innovation/two-clocks.svg)

上方turn经teacher和同版本value pair获得权重，累计有效token更新Actor；下方完整任务经过有效verifier，累计新轨迹训练Critic。Teacher冻结、Actor/Critic各自更新，图不表示Critic在没有终局标签时已经完成训练。缺少下一状态或超Critic上下文回退与Actor正常蒸馏分开。

Actor与Critic backbone从原始Qwen3.5-9B出发，但参数与optimizer独立；Teacher为冻结Qwen3.5-27B。Critic增加标量head，value为sigmoid，initial head使输出0.5。模型不生成自然语言打分；终局反馈是监督标签，不拼进未发生时的模型输入。

当只有终局$R\in\{0,1\}$、gamma1时，理想value可解释为$\Pr_\nu(R=1|h)$。一般return、有折扣或多种reward不能直接叫成功概率。每个完整trajectory各prefix共享R，不说明每轮因果贡献等于R。

代码中head零初始化意味着初始head梯度可非零，backbone经零head传回的第一步梯度可能为零；随head更新后骨干才接到该路径信号。这不是backbone永久冻结。解释warmup时要核实际optimizer和训练步骤，不以“初始化0.5”推成模型一直等权。

## 4. Critic训练：终局数据、归一化与版本

当前online机制从正常rollout获得有效完整trajectory，累计32条新轨迹触发一次Critic更新；无效评测、缺失reward或未经验证截断不硬标失败。已消费trajectory ID去重，Critic own step与Actor commit count各自记录。

有prefix等权和trajectory等权两类聚合。后者在轨迹内平均，再批平均：

$$
L_V=\frac1B\sum_{i=1}^B\frac1{K_i}
\sum_{t\in\mathcal S_i}\mathrm{BCEWithLogits}(f_\phi(h_{i,t}),R_i),
\quad V_\phi=\sigma(f_\phi).
$$

不同$K_i$影响目标权重；同trajectory几十个prefix不作为几十个独立评测单位。选择最长合格历史一次因果forward读取多个边界，有助避免重复prefix，但必须保留完整训练图；ReturnCritic在train+use_cache时明确拒绝推理cache。

同一个Critic请求读取before/after两个位置，用**同一个checkpoint版本**计算差。若before用旧critic、after用新critic，模型更新带来的value漂移会伪装任务进展。服务回复核request key、prompt hash、positions和lineage；online latest也须记录实际served_version，不只保存“latest”字样。

Critic优先验证留出任务/仓库的Brier、AUROC、校准和基率常数baseline，不能仅看训练loss下降。绝对value预测有效也不保证同trajectory内微小delta排序有效，应独立做信息诊断。

## 5. 从状态差到有界权重

默认state_delta在工具后新请求的真实$h_{t+1}$上评分，且$h_{t+1}$必须以封存的$h_t+a_t$为token前缀。它需要反馈到达，当前raw stream甚至将feedback文件作为Teacher评分准入条件；不能说同turn teacher与工具已经完全解耦。

$$
\Delta_t=V_\phi(h_{t+1})-V_\phi(h_t),\quad
w_t=1+\alpha a_k\tanh(\Delta_t/s_t),\quad
a_k=\min(1,k/K_{\rm warm}).
$$

Current candidate采用$\alpha=.5,K_{\rm warm}=4$。第一次更新后raw范围[.875,1.125]，完全激活[.5,1.5]，这是理论范围不是实测直方图。$s_t$按同版本**过去**分差的RMS更新、有floor，当前sample不参与给自己定scale；这不是不确定性估计。

代码实际传输权重除公共$1+\alpha$以满足接口[0,1]限制。`critic_weighting.py`真实节选：

```python
delta = after - before
signed = delta if mode == 'positive' else -delta if mode == 'negative' else 0.
if mapping == 'tanh':
    signed = math.tanh(signed / scale)
return (1 + strength * activation * signed) / (1 + strength)
```

Common scaling在整个window weighted denominator中抵消。Negative模式仍是正权重，强调value下降turn，不把OPD梯度反向。单纯线性映射在delta很小时几乎等权；tanh扩大差别，但也可放大噪声，所以要同时检查scale、误差和质量。

State delta只有$\gamma=1,r_t=0$的非terminal形式才与简单TD残差数值相同；它不是天然$Q-V$或因果action advantage。可选action_advantage在response-end边界，代码要求服务声明训练过before_action/after_response，不能为省工具等待随便换索引。

## 6. Actor目标：teacher决定方向，Critic调整强度

当前single-forward sampled OPD，局部信号：

$$
A_{t,j}=\mathrm{sg}(\log q(y_{t,j}|c_{t,j})-\log p_{\theta_k}(y_{t,j}|c_{t,j})),\quad
\ell_{t,j}\text{的梯度等价于}-A_{t,j}\log p_\theta(y_{t,j}|c_{t,j}).
$$

Teacher给训练方向，Critic的$w_t$广播到该turn有效response token并detach。没有额外task PPO actor loss；Critic自己的监督loss不加到Actor目标，也不是ESCD dense FKL。

$$
L_{\rm actor}=\frac{\sum_{t\in\mathcal W}w_t\sum_jm_{t,j}\ell_{t,j}}
{\sum_{t\in\mathcal W}w_t n_t},\qquad n_t=\sum_jm_{t,j}.
$$

Current implementation窗口固定Actor参数、只访问各turn一次，old/proximal reference可取current logprob detach。异步行为版本仍旧，不能称当前目标无偏；通用PPO clip路径的复用不自动意味着在优化task return。

两个手算检查：公共乘$c>0$会抵消，不能把所有权重同时翻倍叫训练增强；只有一个turn时w也抵消，不能以单turn归一loss证明加强有效。若两个turn token数2/6，权重1.5/.5，分母6；有效系数按token均值归一变成2与2/3，可能超出raw[.5,1.5]，不能把raw界限套给最终系数。

相比RLOO/GRPO组优势，这里并没有把同prompt组reward改成baseline；相比PPO/value critic，这里没有task advantage单独改变actor动作概率目标。它是在线value引导的蒸馏分配机制。

## 7. 缺失值、上下文和共驻Infra

末轮无下一请求、或超出该Critic配置上下文，沿当前代码用中性raw权重1；不能把missing value设0得到强负delta。非terminal缺反馈需要等待或报错，不是静默回退机会。原始outbox保持不变，feedback/value/version/weight放sidecar。

Actor/Rollout的262144上限、Critic候选98304上限、Actor token更新预算262144分别定义。配置支持数值不代表这些长度都完成了正式性能验收。诊断必须报updated critic、initial equal、outside-context neutral、terminal neutral分别覆盖多少有效token，不能只看turn计数。

Teacher TP服务与Critic FSDP训练/评分共驻一组GPU：Teacher固定，可reuse cache；Critic更新会使其推理cache过时，需版本失效。FSDP参数/gradient/Adam状态、长prefix激活、teacher KV与峰值临时张量共同争容量；“显存还有空”不够证明共驻吞吐稳定。

服务可把train/score序列化或按明确机制调度，实际请求时间重叠不证明GPU kernel每时刻同时运行。TP的各rank共同执行teacher请求，FSDP训练又需collective，不能任意让其中一个rank独自切到训练。零value-head推理fastpath有明确初始化条件，不应沿用到训练后checkpoint。

## 8. 创新设计怎样与已有工作对照

| 近邻层 | 已有机制 | 你的可检验区别 |
| --- | --- | --- |
| AReaL / AsyncOPD | rollout/learner解耦、背压、版本和概率校正 | turn内部ready消费与终局critic独立更新条件 |
| FlexMARL / RolloutPipe | 提前backward/组就绪交付 | raw turn接口、final acceptance、token窗口与完整context |
| GiGPO | episode/anchor后续return局部相对优势 | 不额外anchor group，在线state value差调蒸馏 |
| SAGE-OPD / ATOD / TurnOPD | 轮级选择、监督强度或交互深度设计 | 价值随学生在线学习、版本一致delta与dual-clock机制 |
| 标准PPO critic | task advantage和value target | 当前只调OPD强度，不加task actor loss |
| 监督收益Critic备选 | 直接预测权重干预后的下游收益 | 当前value只是代理，不能混成已学真实supervision benefit |

系统层参考[AReaL](https://arxiv.org/abs/2505.24298v1)、[AsyncOPD](https://arxiv.org/abs/2606.24143v1)与既有精读。轮级近邻入口[SAGE-OPD](https://arxiv.org/abs/2606.19659)、[ATOD](https://arxiv.org/abs/2606.27814)、[TurnOPD](https://arxiv.org/abs/2607.05804)，本轮按问题定位，不把其全部实验当已独立复现。

适合面试的主张：“我们把局部蒸馏和终局价值学习放在不同更新时机，用在线状态价值变化控制有界监督强度，保持数据与版本可追溯。”不适合的主张：“首次提出turn加权”“有critic就解决信用分配”“无需实验已证明更强”。

## 9. 最小消融与证伪条件

| 要回答的疑问 | 控制实验与需要报告 |
| --- | --- |
| 有信息还是只输出基率 | 常数、长度/turn基线、训练/验证隔离、Brier/校准与trajectory内排序 |
| 权重有没有被消费 | Commit窗口实际weights、Critic versions、non-neutral token coverage |
| 对应关系有用吗 | Uniform、aligned、同trajectory/version shuffle；注意长度不同token直方图不保真 |
| 正方向成立吗 | Positive/negative/random-sign，固定scale/warmup与同监督覆盖 |
| 在线更新有用吗 | Frozen同一个已训checkpoint vs online，共同Actor起点 |
| 成本有效吗 | Uniform也运行Critic服务以匹配机制负载；另报无Critic整系统成本 |
| 真提升任务吗 | 固定终点checkpoint、独立SWE/Terminal任务、同budget/协议与重复seed |

Random-sign保留raw $|w-1|$而改方向，shuffle保留turn权重multiset但不同长度下不保留token-weight histogram，必须披露。只有单元素group的shuffle没有干预效应，报告实际可置换turn比例。

可证伪：若uniform或打乱对应同样有效，信息分配主张不成立；若value校准好而delta无信息，仍不能说进展weight有用；若质量相近但总卡时更大，需要说明效率取舍。短程接线与局部预测探针不等正式模型收益，当前文档仍有待完成主消融与最终benchmark。

## 10. 面试问答：12个创新追问

**Q01 Critic是否在线预测teacher监督收益？** 当前预测近期数据的任务成功率，delta是代理；收益Critic是另一个昂贵备选。

**Q02 正delta为什么要加强？** 这是主假设，不是定理；错误turn可能更需纠正，要比较方向。

**Q03 终局reward复制各prefix等于过程标签吗？** 不等于独立因果信用，但可学习条件结果概率。

**Q04 Value pair跨版本有什么风险？** 同状态模型漂移伪装进展；需同served checkpoint。

**Q05 Tanh产生更大方差证明有效吗？** 只证明数值变化，需对照随机/打乱与真实质量。

**Q06 Cold start为何等权？** 零head初始0.5，逐Critic update activation；理论范围不是实测覆盖。

**Q07 一个turn单独加权为何可能没变化？** Weighted denominator中该正标量抵消。

**Q08 整个窗口factor2会增强更新吗？** 同时乘分子分母抵消，主要是相对权重。

**Q09 Actor有PPO task loss吗？** 当前没有；备选文档不能当主实现。

**Q10 反馈未到能算state delta吗？** 不行；当前raw stream还门控Teacher，真正重叠有待独立改造。

**Q11 32卡验证证明64卡262K吗？** 不证明，需要该拓扑/长度/通信/质量独立验收。

**Q12 哪个结果能证明创新？** 对应关系和在线机制消融、同预算独立任务收益，不能只用系统跑通。

## 11. 面试讲稿：30秒与3分钟

30秒：“这个项目关注多轮Code OPD的监督等待和分配。我们用turn封存和固定参数窗口让局部蒸馏提前消费，同时独立在线Critic用终局结果学习学生状态价值。同版本前后value差映射到有界权重，Teacher决定方向、Critic调强度。关键是两条学习流与版本、分母契约；最终是否改善任务要由uniform、方向与打乱消融支持。”

3分钟依次讲：一次真实工具turn→原token封存/feedback→teacher与same-version value pair→current single-forward sampled信号→weighted FP32窗口→request-boundary发布→完整trajectory终局训练Critic。最后给当前已验证层和未完成的主结果，技术与研究假设分开。

个人贡献按你真实负责的协议、loss/backend、value模型/服务、调度或实验填写；不能只列栈名。更详细实现见[turn框架](#q=agentic-turn-framework)，两项目区别见[ESCD讲稿](#q=escd-project-interview)。
