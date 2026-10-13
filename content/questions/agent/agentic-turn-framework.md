---
id: agentic-turn-framework
title: "Agentic Code OPD turn框架：从请求到梯度、窗口更新与权重发布"
category: agent
difficulty: 深入
tags: ["P0", "Agentic Code OPD", "Turn", "Infra", "Megatron", "TRAIL", "Project"]
updated: "2026-10-13"
summary: "按指定turn工作树追真实入口、不可变数据、teacher因果评分、single-forward、FP32累积、prefix coalescing、版本gate与共驻部署；附16道实现追问。"
draft: false
---

## 1. 问题背景 为什么从完整trajectory改成turn交付

完整coding任务先运行很多工具轮次，结束后整批训练会延迟已经产生的局部蒸馏信号。Turn分支把一次模型请求的response封存后提前交付评分/learner，让任务执行与局部训练计算重叠。它不是只换队列名称，还要回答：上下文是否真实、参数何时可变、梯度分母如何累积、最终接受规则怎样影响已算梯度。

本篇专门核读用户指定的`MultiTeacher-OPD-Agent-turn`，不复用之前完整trajectory async目录的结论。2026-10-13工作树含大量未提交改动；HEAD e2e0bcc是共同基线，**不是以下代码的冻结版本**。本轮保存42个源码/文档的只读副本，逐文件SHA见[代码证据账本](./assets/learning/agentic-turn-framework/code-evidence.json)。没有修改该项目或运行GPU任务。

## 2. 前置知识 四类边界与四个角色

边界分别是request完成、工具反馈到达、完整任务验证、optimizer窗口提交。Request内参数固定不等于整trajectory策略固定；turn ready不等终局reward ready；backward完成不等optimizer已提交。

Rollout执行student已发布权重；Actor learner计算学生梯度；Teacher冻结提供token概率；可选独立Critic估状态成功率并调整OPD权重。Actor和Critic参数/optimizer独立，后者不是传统Actor共享value head；当前Actor不另加任务PPO loss。具体critic设计见[创新与论证](#q=agentic-turn-innovation)。

## 3. 主数据流与文件地图

![教学图 · turn封存、teacher评分、actor窗口与critic终局流](./assets/learning/agentic-turn-framework/turn-flow.svg)

图由本文依据实际接口绘制。Actor局部流和Critic终局流完成条件不同；默认state_delta还需下一请求的真实反馈才可算权重，不能画成teacher/工具在每turn已经完全并行。每个窗口固定Actor参数，窗口结束才发布；虚线是更新和反馈，不是工具可微。

| 路径，相对项目根 | 关键对象 | 阅读目的 |
| --- | --- | --- |
| `src/innovator_agent/turn_opd/train.py` | 显式模式/validate | 不误启动旧async入口，拒不兼容选项 |
| `protocol.py / store.py` | Turn、ScoredTurn、Receipt、digest | 数据身份、不可变tokens、冲突与去重 |
| `pipeline.py / async_driver.py` | TurnSync/TurnAsyncPipeline | 提前backward和真正update边界 |
| `teacher.py / teacher_routing.py` | payload、score、affinity | 完整条件、因果位置、队列与缓存 |
| `ray_learner.py / megatron_session.py` | 跨rankbackward/commit | 模型并行调度与单次step |
| `gradients.py` | TokenWeightedGradients | FP32恢复窗口全局均值 |
| `single_forward.py`及Slime `loss.py` | 当前logprob detach | 同forward构造局部OPD信号 |
| `coalescing.py / sequence_buckets.py` | 精确prefix链、shape buckets | 不改loss positions的计算优化 |
| `flow_control.py / version_gate.py` | Credits、EngineVersionGate | 背压、drain、发布、故障 |
| `critic_weighting.py / feedback.py` | value差、sidecar | 等工具后的评分和权重版本 |

文件名相同不代表与上游slime完全一样；你的vendored Slime接了turn session和loss改动。代码ledger只记录路径/hash，不包含任务名、服务地址、账号或原始运行日志。

## 4. 不可变Turn：为什么重试不能生成第二条身份

Turn字段包括run/trajectory/attempt/request ID、turn index、behavior version、prompt IDs、response IDs、loss mask、behavior logprobs、sampling config和logprob语义。构造时检查非空、对齐、mask二元/有有效token、finite probability与采样配置。

`protocol.py`真实节选：

```python
@property
def key(self) -> str:
    # Version/turn metadata is NOT part of identity: a retry claiming a
    # different version for the same request must cause a conflict.
    return json.dumps((self.run_id, self.trajectory_id, self.attempt_id, self.request_id))

@property
def digest(self) -> str:
    return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()
```

Key表示同一个逻辑请求，digest表示完整内容。相同key却改version、mask或token应冲突，而非拿不同版本当新样本再次训练。Receipt保存key/digest/计数等轻量字段，原始prefix仍在outbox，省的是控制器重复持有的内存，不是删除Actor训练条件。

工具输出影响后续input，默认不进入actor动作loss。真正verifier结果保存metadata，不能误拿OPD接口的占位零reward训练Critic。步骤上限有时是可保留的截断，普通EOF/工具失败/无有效token不同；匹配条件必须由真实执行协议决定。

## 5. Teacher评分：response-only不等于只输入response

`teacher.py::request_payload`仍发送`prompt_ids + response_ids`，`max_new_tokens=0`用于评分。Response-only设置只减少返回区间：

```python
# Official local excerpt from the frozen working-tree teacher.py.
"input_ids": list(turn.prompt_ids + turn.response_ids),
"return_logprob": True,
"logprob_start_len": len(turn.prompt_ids) - 1 if response_only else 0,
```

多包含一个predecessor，因为所用SGLang首次请求位置logprob可能为None；parse_teacher_reply核长度、token IDs、目标response、finite score和有效mask。不能仅按返回数组长度切片后假定对齐。

Teacher冻结，可按trajectory做affinity增加prefix reuse，但单个副本可能长尾堵塞，所以有可选steal、ready-order和endpoint selection。Teacher缓存不是Actor带梯度缓存，GDN状态还须按实现的prefill chunk边界复用；源文档记录4096对齐而不是任意最终状态续算。实际命中与score等价需以该模型与kernel版本核验。

## 6. turn-sync与turn-async是两个不同实验

| 维度 | turn-sync | turn-async |
| --- | --- | --- |
| 版本 | 原cohort固定 | trajectory跨请求可跨版本；单request固定 |
| 交付 | sealed turn提前评分/backward | 持续消费sealed turn |
| 接受 | 最终accepted cohort | 显式sealed-turn，未要求终局成功 |
| 更新 | 最终集合确认后一次commit | 有效response token累计到预算commit |
| 尾部 | 等cohort确认 | 流结束时显式drain最后部分窗口 |
| 过旧 | 固定窗口一致性 | 限lag并登记skip，不保证无偏 |

Sync若提前算了后来被drop的turn，或最终mask变化，代码abort清梯度、按最终集合重新score/backward，再commit。这样才能支持“提前反传、既定更新边界”而不偷偷改样本集合。只增task concurrency与这个机制不同。

Async `pipeline.py`的核心是：

```python
lag = version - turn.behavior_version
if lag > self.max_version_lag:
    await self.on_skip(turn, "version_lag")
    continue
if window.accept(turn, learner_version=version):
    await self.learner.backward(scored)
if window.active_tokens >= self.token_budget:
    await finish()
```

`finish`先seal、learner commit、成功receipt，再递增learner版本并publish。Token预算按整个turn纳入，可能越过阈值；不是把turn剪到剩余token数量。慢turn、skip、cancel成本须登记。旧样本logprob保留用于审计，不意味着当前loss一定做了IS。

## 7. Single-forward：ratio为1仍然有梯度

当前opt-in sampled路径，窗口内Actor固定，each accepted turn访问一次。Current训练forward直接产生logprob，不再另做独立student forward；优势和ratio reference由这个结果detach。

Slime `loss.py`约2610–2658行：

```python
if not args.use_rollout_logprobs and not old_log_probs:
    old_log_probs = [log_prob.detach() for log_prob in log_probs]
# In the validated single-forward branch:
detached_student = [value.detach() for value in log_probs]
advantages_list = [torch.zeros_like(value, dtype=torch.float32) for value in detached_student]
apply_opd_kl_to_advantages(args, dict(batch), advantages_list, detached_student)
advantages_list = [value.detach() for value in advantages_list]
```

Teacher/student采样token差构造$A=\mathrm{sg}(\log q-\log p_{\theta_k})$，局部梯度相当于$-A\nabla\log p_\theta$。Ratio reference是同次current结果detach，值1却有梯度；它**不是**旧rollout behavior denominator。默认validate拒use_rollout_logprobs、TIS、cross-tokenizer、额外KL/entropy、dropout与未验证自定义组合。

因此这一路不能说“single-forward已精确校正异步off-policy”：旧行为动作与prefix仍来自不同策略。它说明省forward后保持所定义的固定窗口sampled surrogate梯度，不证明最新策略RKL无偏。ESCD full-vocab目标是另一入口，也不把两项目的loss混成一套。

## 8. 梯度累计：先恢复分子，再除全窗口分母

Megatron schedule可能每chunk已做token normalization，继续在已归一main_grad上加下一chunk后再归一会错。`gradients.py`先捕获该chunk mean gradient乘有效token数，在独立FP32 buffer累积，窗口末恢复mean：

$$
g_{\mathcal W}=\frac{\sum_c n_c g_c}{\sum_c n_c}.
$$

真实关键语句：

```python
self.buffers[i].add_(gradient.detach().float(), alpha=active_tokens)
# At the single window commit:
mean = buffer / denominator
target.copy_(mean)
```

Capture前检查全部gradient finite，再修改buffer，防止半个chunk已累计、另一个参数NaN。每个chunk ID去重、version相同、global token计数对齐；commit统一gradient clip、optimizer、scheduler一次。Critic加权时分母另为$\sum_t w_tn_t$，不能仍除未加权token总数。

DP/CP的global counts、TP/PP执行与all-rank collective还要backend保证；纯累积模块测试不证明分布式等价。零loss DP transport padding不产生Turn身份或有效token；padding计算却仍付成本。

## 9. Prefix coalescing、shape与发布Infra

Coalescing把同一窗口、同trajectory/attempt、相同behavior/teacher与采样语义的**精确token前缀链**合成最长序列，保留全部原始有效loss位置和对应teacher/weight；masked工具text也逐token核prefix一致，不做summary。它在一个因果训练图复用历史计算，保留prefix梯度，不是detach KV cache。

Branch、重叠loss位置、不同版本或协议不合并。暂时coalesced request ID只用于训练例子，原outbox identity/credit/receipt不能替换成合并ID。Prefix buffer只重排accepted turn，不丢其他链；oldest chain与idle flush避免短批饥饿，chunk capacity还须小于credits容量，否则会等满chunk、producer却无额度而死锁。

Sequence buckets向右补齐少数长度，减少FLA/Triton shape编译；不截原生prefix、不把loss=0当无计算。Qwen GDN候选路径约束BSHD，不能擅自改THD动态packing。TP词表/矩阵、PP层、CP序列与DP样本分别承担不同维度；总rank数不能忽略CP再重复计卡。

权重resident Bridge从GPU导出，但detach共享pointer不是独立快照；必须serialize optimizer→export→load。发布gate先停新请求、drain真实backend、确认新版本并invalid旧cache再恢复。Client timeout不证明decoder停止，异常可能poison gate。Per-engine gate不自动让已有all-engine NCCL group支持subset更新。

## 10. 配卡与性能：不要把历史96卡叫当前唯一配置

| 配置层 | 可核记录 | 表述边界 |
| --- | --- | --- |
| 历史基础 | Actor64、Rollout24、Teacher8，共96；TP4/PP4/CP4/DP1 | 完整轨迹与turn历史对照配置，不是当前唯一方案 |
| v9候选 | Actor64可选TP4/PP2/CP4/DP2，prefix buffer/coalescing与resident source | 选项需显式开启，不推断全部运行都采用 |
| TRAIL短程候选 | 总32：Actor16、Rollout8、Teacher/Critic共8 | 来自当前方法记录，不是正式主结果 |
| 32卡并行 | Actor TP4/PP2/CP2/DP1；4个TP2 rollout；TeacherTP8、CriticFSDP8 | 共驻还需实测占用、评分/训练冲突 |
| token/长度 | Actor窗口有效token阈值262144；单上下文另限262144；Critic候选98304 | 相同数字不表示同一预算；配置不证明长输入性能验收 |

旧文档含DP1/no critic/fresh-run限制，新显式DP与Critic选项扩展了一部分范围；应从当前validate和实际args判断，不能把旧段落或文件名当运行事实。配置、冻结代码、CLI/profile override与dependency版本都需要runmanifest。本文只核工作树与文档，不读取生产服务或宣布当前在跑哪些任务。

指标按有效训练response token/GPU-hour、任务覆盖、score/ready/backward/commit/publish延迟、版本年龄、浪费和独立checkpoint质量组织。GPU utilization不等MFU，KV池分配不等实际KV使用，异步重叠时阶段时长不直接相加。

## 11. 面试问答：实现层16个追问

**Q01 为什么不是完整trajectory pool？** 交付和backward单位变成sealed turn，窗口update与任务终局分开。

**Q02 turn ready何时成立？** 普通评分可在生成后；默认critic state_delta需下一请求反馈，当前流对此还有等待。

**Q03 Sync提前backward怎样保持接受集合？** 最终集合/mask不符就abort/replay，再一次commit。

**Q04 Async token窗口必须精确等阈值吗？** 整turn纳入可越阈值，报告实际有效tokens。

**Q05 Teacher response-only看不到历史吗？** 全历史仍输入，只缩返回区间并保留predecessor。

**Q06 Key为何不含version？** 同请求改版本应作为冲突，被digest检出，不另建身份。

**Q07 单forward的old reference是什么？** Current logprob detach，不是rollout旧采样分母。

**Q08 Ratio1还能训吗？** 分母detach，current numerator有梯度。

**Q09 此路径用了TIS吗？** Validate拒该未验证组合，不能宣称精确行为校正。

**Q10 每chunkmean再mean可以吗？** 变长时改变权重，需token分子累计与全窗口分母。

**Q11 Coalescing是不是截历史？** 精确最长因果prefix、全部loss位置保留，不detach历史。

**Q12 把receipt变小省的是显存吗？** 主要控制器记账内存；真实Actor前缀计算仍在。

**Q13 Credits什么时候申请？** Engine lease前，避免满队列持有active请求让发布死锁。

**Q14 Resident export为何不能和optimizer重叠？** 共享pointer在写时不是冻结snapshot。

**Q15 账本能保证exactly-once optimizer恢复吗？** 单进程去重不等crash原子事务；未确认commit与checkpoint须处理。

**Q16 哪项结果证明创新？** 接线/局部验证不足，需要同预算质量与对应机制消融；见下一篇。

## 12. 代码证据与继续学习

源副本仅用于本轮静态核读，工作树hash见账本；snippet对应实际文件，省略周边代码处明确标注。项目没有统一公开commit可链接，不造GitHub源码URL。对照[AReaL](https://arxiv.org/abs/2505.24298v1)、[AsyncOPD](https://arxiv.org/abs/2606.24143v1)、[RolloutPipe](#paper=paper-rolloutpipe)、[FlexMARL](#paper=paper-flexmarl)认识已有系统近邻。

学习顺序：[后训练40题](#q=post-training-interview) → 本篇原始样本/窗口 → [Critic与创新论证](#q=agentic-turn-innovation) → [ESCD介绍](#q=escd-project-interview)对比两项目。本文不提交训练、修改集群、发布用户模型或把配置候选当结果。
