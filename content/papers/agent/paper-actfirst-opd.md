---
id: paper-actfirst-opd
title: "ActFirst-OPD：快动作推进环境，完整响应异步生成并训练"
paper_title: "Act First, Reason Later: Accelerating On-Policy Distillation for Multi-Turn Agents via Reference-Conditioned Inverse Dynamics"
authors: ["Zubin Zheng", "Jiahao Wu", "Shaofeng Zhang", "Zhirui Zhang", "Yew-Soon Ong", "Shengcai Liu"]
affiliations: ["Guangdong Provincial Key Laboratory of Brain-Inspired Intelligent Computation, Department of Computer Science and Engineering, Southern University of Science and Technology", "Zhongguancun Academy", "The Hong Kong Polytechnic University", "DeepCybo", "College of Computing and Data Science, Nanyang Technological University"]
author_affiliations: [[1], [3], [1, 2], [4], [5], [1]]
venue: "arXiv preprint (fixed v1)"
year: 2026
direction: agent
areas: [language]
tasks: [agents, reasoning, training-adaptation]
published: "2026-09-29"
paper_url: "https://arxiv.org/abs/2609.36608v1"
github_url: null
code_note: "固定v1未提供可核作者方法仓库；环境仓库不能替代方法源码，以下按正文/附录解释并标教学伪代码。"
method_figure: "./assets/papers/paper-actfirst-opd/method-source.png"
method_caption: "Figure 1 · 快动作执行与完整OPD响应的时间线"
evidence: 已核原文
note_ids: [agentic-opd-training, rl-ppo-training, agentic-rl-training]
tags: [ActFirst OPD, Agentic OPD, Async Rollout, Fast Action, Reference Guidance]
updated: "2026-10-12"
summary: "分开执行用的fast action和后台完整训练响应，以参考观察指导快动作、失配后回退；核mask、FIFO版本规则、预算和并发收益边界。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：环境推进被长思考串行阻塞

常规多轮OPD先让学生生成完整思考和动作，再执行动作获得新观察，随后重复。每个turn长响应位于环境推进关键路径：工具还没开始，GPU已经生成数百token；下一轮必须等上一轮执行。Teacher scoring可以很快，但完整学生rollout占主要时间。ActFirst尝试先用短动作推进环境，同时从已知真实历史后台生成完整响应用于训练，重叠多个turn的慢生成。

困难是快动作缺少长推理，容易偏离任务；若直接取消思考可能跑得快但轨迹质量差。方法使用离线参考观察指导快动作，在转移一致时继续沿参考，失配后切回独立快动作策略。它没有让教师在线执行每turn，也没有把参考观察直接塞进完整训练响应的未来上下文。

该设计必须讲清执行与学习的不同动作：环境实际执行fast action，后台完整response里的action可以不同，并未执行。State visitation因此来自辅助快策略，局部完整响应仍由学生采样；“学生生成响应”不自动等于整条轨迹完全匹配部署think-then-act策略。这是方法的收益来源，也是分布边界。

### 前置知识：关键路径、并发和策略版本

关键路径是必须顺序完成才能推进任务的步骤，不等于所有GPU工作量。短动作与完整生成都消耗算力，若共享serving并发小，额外请求可能排队，实际更慢。理想重叠公式只在充足并发、环境成本可忽略等条件成立。

异步生成训练还意味着完整响应可能来自不同学生版本，需要保存每response采样logprob和版本。FIFO暂存、只消费一次与replay反复训练不同；年龄、版本差和token级ratio也不是同一个控制量。快动作和完整响应如果共用参数，更新会影响两者，但loss未必同时覆盖两类数据。

### 已有工作：减少等待与减少生成

[GKD](https://arxiv.org/abs/2306.13649)、MiniLLM提供OPD基础；TCOD/TurnOPD组织教师前缀或turn监督；AsyncOPD解耦rollout/scoring/learner并处理旧数据，STRIDE主动早停/重启前缀。ActFirst重点改变环境推进关键路径，与直接剪掉长思考或只加serving并发不同；公平对照要包含无参考快动作和不同并发设置。

## 2. 方法与实现机制

### 三种生成模式如何共享一个学生

![Figure 1 · 环境链与完整响应生成链的交错](./assets/papers/paper-actfirst-opd/method-source.png)

**Figure 1解读。** 上方环境按fast action连续推进，下方从各turn已知上下文生成较长完整响应并进入评分/训练。短请求负责状态转移，长请求不再全部串在环境关键路径上。图中重叠是并发调度，不是完整响应末尾action必然等于执行action。原图为actfirst_opd_timeline.pdf、iclr2027_conference.tex中的fig:actfirst-framework；模板文件名不证明ICLR2027录用。

共同历史$c_t$保留真实执行actions与observations，不保留之前的长reasoning。指导fast action接收$c_t$及离线下一步参考观察$o_{\rm ref,next}$，快速生成$\tilde a_t$。执行后比较真实转移与参考；失配时关闭本episode后续指导，使用无参考的NAP fast action继续，而非任务终止。

并行的完整响应$y_t=(z_t,a_t)\sim\pi_\theta(\cdot|c_t)$只看真实历史$c_t$，不看未来参考观察。它包含学生思考$z_t$与动作$a_t$，teacher对该完整响应的实际token前缀评分。$a_t$可以不等于$\tilde a_t$，环境只执行后者。评价时回到无参考的正常think-then-act，验证训练所得学生能力。

### 转移一致性不是逐字动作一致

参考可能通过另一个有效动作达到相同观察，因此方法比较可观察转移、done与环境特定约束，不要求生成action等于参考字符串。附录按环境做观察归一化和动作合法性检查，也不是直接证明隐藏模拟器状态相同。未知状态、工具副作用或未来不可逆修改可能突破这个近似；coding迁移需要更严格的文件与进程快照检查。

若失配后继续给未来参考观察，可能提供与真实环境不相容的指导；NAP fallback使后续轨迹保持实际可执行。参考准备来自公开示例、模型rollout或oracle来源，质量和成本需要独立记录。不能把参考生成、教师RL和评价时间当不存在。

### 理想收益公式与反例

设$T$轮完整生成单轮耗时$\ell_{\rm full}$，快动作耗时$\ell_{\rm fast}$。常规关键路径近似$T\ell_{\rm full}$，足够重叠时：

$$
S_{\rm ideal}\approx
\frac{T\ell_{\rm full}}{(T-1)\ell_{\rm fast}+\ell_{\rm full}}
\le \min\left(T,\frac{\ell_{\rm full}}{\ell_{\rm fast}}\right).
$$

例如$T=10,\ell_{\rm full}=8,\ell_{\rm fast}=1$，理想80/17≈4.706倍。若两个请求完全串行执行，每轮反而付8+1，速度8/9<1。公式假定环境/排队等额外延迟可忽略、完整生成能并发；它不是无条件端到端上界。真实serving中prefill、KV、动态batch和多个轨迹长尾影响收益。

### 附录中的实际OPD surrogate

完整response从已完成rollout batch进入FIFO。更新只取当前或前一学生版本，更旧的在组batch时丢弃；已消费不复用。Teacher frozen，在相同token上下文评分。人工插入token和padding仍可见但mask=0，学生生成thought/action与格式token均可监督。

$$
A_{b,j}=\mathrm{sg}[\log q(x_{b,j}|c_{b,j})-\log p_{{\rm old},b}(x_{b,j}|c_{b,j})],
\quad
\rho=\exp(\mathrm{clip}(\log p_\theta-\log p_{\rm old},-20,20)).
$$

再用PPO宽度$\delta=0.2$与dual-clip $\kappa=3$。最小化loss先为$\ell_{\rm clip}=-\min(\rho A,\mathrm{clip}(\rho,1-\delta,1+\delta)A)$，负优势另取$\min(\ell_{\rm clip},-\kappa A)$；正优势不加dual分支。±20数值限幅与PPO ε剪切是两层不同机制。

归一化为seq-mean-token-mean：每个turn完整response按有效token平均，再按response行平均；这里seq不是整条episode。没有环境reward混合、advantage normalization、独立reference KL、entropy penalty，也没有fast action辅助loss。Teacher评分、behavior概率与advantages在更新中冻结，学生current forward有梯度。

### 关键代码应怎样阅读：本轮只有可核伪代码

固定v1未提供可核官方方法仓库，环境GitHub引用不是方法实现。以下是教学伪代码，按Algorithm/附录B拆职责，未执行，也不假设reference来自真实项目：

```python
# Teaching pseudocode, not author source.
history, guided = env.reset(task), True
pending = []
while not env.done:
    snapshot = freeze_actual_history(history)
    pending.append(generate_full_async(snapshot, student_version))
    fast = generate_fast(snapshot, reference_next if guided else None)
    observation, done = env.execute(fast.action)
    if guided and not transition_matches_reference(observation, done):
        guided = False          # no more reference guidance in this episode
    history = append_actual_action_and_observation(history, fast.action, observation)
responses = await_completed(pending)
# Score full responses at their saved contexts; fast actions have no extra loss.
# FIFO consume once; current/previous version only; dual-clipped sampled surrogate.
```

snapshot必须不可变，否则后台任务可能读取已经附加未来观察的history，导致泄漏。完整响应记录生成时模型版本，不假设整个episode只用一个版本；fast与full共享权重但数据角色不同。Teacher cache至少按真实prompt/token/模板/版本标识，不能按task ID复用成相同评分。

## 3. 实验设置与算力

教师为GiGPO任务训练Qwen3-8B，学生Qwen3-0.6B/1.7B/4B；三个环境ALFWorld/WebShop/ScienceWorld。每个设置保持teacher、student初始化和环境一致，比较vanilla、TCOD/TurnOPD、无指导快动作与ActFirst。

| 项目 | 固定v1 |
| --- | --- |
| 硬件 | 8 A100-SXM4-80GB |
| 分配 | 4rollout、2teacher scoring、2learner |
| 框架 | TrinityRFT，vLLM生成/评分，verl优化 |
| 训练预算 | 250updates，rollout16任务，优化64个turn响应 |
| 并行 | inference TP1，Ulysses SP2 |
| 精度/batch | BF16，动态16384tokens/GPU，显存比例0.7 |
| prompt/response | 10240 / 512训练上限 |
| thinking/action | 384 / 至多128token分阶段预算 |
| 训练解码 | temperature1、top-p1、不开top-k |
| rollout执行 | 每GPU1engine，每engine4runner，总16runner |
| prefix cache | 关闭 |
| turn预算 | ALF30、Web15、Science30 |
| 优化器 | AdamW，LR1e-6，constant无warmup，beta(0.9,0.999)，WD0.01，grad clip1 |
| loss | sampled logratio，PPO0.2，dual3，系数1，response内平均 |
| FIFO版本 | 当前/前一版本，消费一次，旧样本组batch时剔除；每update同步权重，优化1epoch |
| 测试任务 | ALF140seen/134unseen，Web100，Science150 |
| 测试协议 | 无reference，正常think-then-act；seed42/43/44 |

主训练时间只计250step训练，排除teacher RL、离线reference准备和evaluation。表中的mean±std来自同一checkpoint的三评测seed，不是三次独立训练。附录还列准备时间，要给自己的总项目成本应另外加入，不能只把主表hour乘卡数当全部投入。

效率微基准固定Qwen3-1.7B、16任务、fast16token，full128/512，serving并发16→256；其中虚拟turn与固定上下文用于剥离执行条件，不等于原环境真实训练。其并发收益只能解释关键路径机制，端到端主表另读。Teacher服务与learner采用独立GPU组，资源并未由这些表证明动态弹性分配。

老师、tokenizer、参考覆盖和环境变动都会影响结果。跨tokenizer需额外的概率对齐，本文没给跨词表验证。Optimizer/LR来自附录Table6，未开放方法代码的部分不拿其他框架默认值补齐；静态方法与CPU理想延迟例子也不代表模型复现。

## 4. 结果与图表解读

![Table 1 · ALFWorld准确率与训练时间](./assets/papers/paper-actfirst-opd/table-1-pdf.png)

**Table 1解读。** 0.6B vanilla SR60.46/time4.18h，ActFirst68.61/1.93h=2.17×；1.7B67.64/4.46→77.49/1.95=2.28×；4B79.93/4.56→82.60/1.93=2.37×。去ID的0.6B仅SR44.40，说明快动作本身能加速却损害质量。表内时间为该配置训练wall-clock，SR为独立学生测试，不是训练时指导成功率。[原表第7页](https://arxiv.org/pdf/2609.36608v1#page=7)。

![Table 2 · WebShop与ScienceWorld](./assets/papers/paper-actfirst-opd/table-2-pdf.png)

**Table 2解读。** 0.6B Science SR10→12.67，时间10.31→2.01h=5.12×；4B Science29.34→31.33、12.40→2.63=4.72×。并非所有格都提升：1.7B Web SR40.33→39.33下降1个百分点，4B Web score69.90→69.34也稍降。作者概括8/9设置SR至少相近/不低于vanilla，应保留这些例外。平均速度约ALF2.3/Web1.8/Science4.9，不能当每一个模型都是4.9×。[原表第7页](https://arxiv.org/pdf/2609.36608v1#page=7)。

![Table 3 · 指导、fallback与prefix replay消融](./assets/papers/paper-actfirst-opd/table-3-pdf.png)

**Table 3解读。** 表平均三学生规模。ALF默认SR76.23/1.94h，去指导58.92/2.45h，降17.31个百分点；去NAP fallback75.18/1.25h，虽快但质量略降。100%参考prefix replay ALF75.87/1.11h接近，但Web/Science更差，说明直接多用参考未保证全部环境效果。Rank是各规模排名再平均，非跨benchmark原始分数平均。[原表第8页](https://arxiv.org/pdf/2609.36608v1#page=8)。

![Figure 3 · serving并发、horizon和三GPU组利用](./assets/papers/paper-actfirst-opd/efficiency-source.png)

**Figure 3解读。** 左(a)并发16→256：128token全响应速度约0.80→1.49，512token约0.92→2.42，低并发确会低于1；(b)真实曲线明显低于理想虚线，收益随长turn改善但非无限线性。(c,d)展示rollout/scoring/update利用随时间，额外请求重叠提高环境steps/s。纵轴在2.5以上采用不同刻度，需读数而非直线延伸。微基准不能替代同budget学生SR证明。图源efficiency_analysis_combined.pdf、fig:efficiency-analysis。

结果说明关键路径重排有效，收益受并发、完整响应长度、参考可靠性与环境限制影响。论文并未证明执行fast动作与训练full动作的分布差距在任意agent都可忽略；需关注独立评测、fallback率、reference覆盖与失败恢复。

## 5. 局限、结论与后续阅读

ActFirst最有价值的系统思想是把环境推进与完整训练响应分开安排。它也改变访问状态：执行策略看参考未来观察，完整响应只看当前真实历史；局部学生采样不能消除整条状态分布偏移。Transition check依赖可观察表示，coding中的隐藏文件/进程状态和不可逆工具操作需要额外验证。

对照[AsyncOPD](#paper=paper-asyncopd)的概率缓存与旧数据处理，[UOPD](#paper=paper-uopd)的teacher执行，[STRIDE](#paper=paper-stride)的前缀重启。自己的项目应明确fast action是否有loss、full action是否执行、history何时freeze、buffer如何消费；四个问题比“全异步pipeline”名称更能说明实现。

**Q01 full响应里的action执行吗？** 训练时不执行，环境推进用fast action。

**Q02 full响应看未来reference吗？** 不看，保存真实当前history。

**Q03 转移检查要求字符串action相同吗？** 不要求，检查观察/done及环境条件。

**Q04 失配后任务终止吗？** 关闭后续reference指导，NAP继续执行。

**Q05 低并发一定加速吗？** 微基准出现<1，额外生成可能更慢。

**Q06 FIFO是replay吗？** 消费一次、版本受限，不是反复训练历史。

**Q07 sample是on-policy就等部署状态吗？** full局部由学生采样，访问状态来自fast辅助执行，仍有差距。

**Q08 主速度计全部准备成本吗？** 不计teacher RL、reference准备和评测，需另报。

**来源。** [固定v1全文及附录B/C/E](https://arxiv.org/abs/2609.36608v1)、原始图表；未取得作者方法代码，未模型复现，图版权归作者。
