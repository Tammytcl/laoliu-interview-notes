---
id: paper-stride
title: 'Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation'
paper_title: 'Know When to Stop, Where to Restart: Accelerating Multi-Turn Agentic On-Policy Distillation'
authors:
- Zhiyu Gui
- Kexin Huang
- Jia Guo
- Junkang Wu
- Zihao Wang
- Zhiqiang Zhang
- Jun Zhou
- Jiancan Wu
- Xiang Wang
affiliations:
- University of Science and Technology of China
- Ant Group
author_affiliations:
- - 1
  - 2
- - 1
- - 2
- - 1
- - 2
- - 2
- - 2
- - 1
- - 1
venue: arXiv preprint (fixed v1)
year: 2026
direction: infra
areas:
- language
tasks:
- training-adaptation
- reasoning
- agents
published: '2026-09-13'
method_figure: ./assets/papers/paper-stride/method-source.png
method_caption: Figure 1 · STRIDE early stopping and prefix buffer
paper_url: https://arxiv.org/abs/2609.14636v1
github_url: null
code_note: 本次未发现可核验的论文作者官方实现；不以同名项目或第三方复现替代。
evidence: 已核原文
note_ids:
- opd-vs-sft
tags:
- STRIDE
- OPD
- Async Training
- Rollout
updated: '2026-10-04'
summary: 在线 turn-level 教师信号决定早停，复用连续可信前缀并从最弱 turn 重启；分别核对速度、训练覆盖和错误恢复价值。
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

STRIDE 研究多轮 agentic on-policy distillation 的计算浪费：学生生成完整交互后才评分或裁剪，即使前面已经走入教师不认可的状态，后续环境与生成仍继续消耗。它在线读取 turn-level teacher logprob，决定什么时候停止，并把已掌握的前缀用作下次重启上下文，把生成预算集中到薄弱位置。

**它是不是异步 learner 系统？** 主要贡献是减少需要执行的 rollout 和逐步推进训练覆盖，并不是解除所有 rollout/teacher/learner 屏障。在线教师反馈与异步流水线有关，但本文没有证明逐 turn 梯度提前 backward、随后 global batch 统一更新。这是阅读用户表格时必须保留的边界：早拿到 teacher signal 和早计算梯度不等价。

**为什么 teacher 低概率可能有意义？** token-level OPD 以教师在学生访问前缀上的分布提供信号。若学生早期动作错误，之后访问的状态可能偏离教师熟悉区域，局部 token 奖励的可靠性下降。论文用多轮数据分析认为 turn-level mean logprob 的下降与首次错误相关，但这仍是特定教师、模板与任务环境的经验代理。低概率也可能代表不同但有效的解决方式、风格或错误恢复，不能作为通用正确性判定器。

只早停容易减少后面 turn 的训练覆盖。STRIDE 因此引入 prefix buffer：保存连续高教师认可的前缀，下一轮从其中最弱的 turn 重新生成，训练窗口随着学生改善向后推进。这与 APRIL 保存“尚未生成完”的 continuation 不同；STRIDE 保存“较可信、希望避免重复计算”的任务前缀，随后重新采样薄弱部分。

本文固定 [STRIDE v1](https://arxiv.org/abs/2609.14636v1)，首发 2026-09-13。核心 agent 实验使用 retail/telecom 环境而非 SWE-bench；论文也有单轮数学扩展。研究结果与长程 coding OPD 很相关，但环境状态恢复、失败修复的训练价值和多版本异步 cache 仍需单独验证。

## 2. 方法与实现机制

![Figure 1：早停、前缀保存与薄弱 turn 重启](./assets/papers/paper-stride/method-source.png)

Figure 1 左边概括教师信号的经验分析，中间用不同 epoch 显示重用与新生成 turn，右边给出质量/时间示例。蓝色框代表 reused prefix，绿色是本轮 generated + trained，橙色标最弱 turn。读图时尤其要看到重启点前一轮的 prefix 被当作上下文，而不是把同样已训练前缀反复计算梯度；论文想节省的是已掌握交互的重复生成成本。

**早停的量是什么？** 第 $k$ 轮学生动作的平均教师 logprob 记为 $\bar l_k$，区别于整个轨迹的总 token logprob。原文在累计 turn-level 信号低于负阈值时停止：

$$
K^*=\min\left\{K:\sum_{k=1}^{K}\bar l_k<\lambda\right\}.
$$

$\lambda$ 更接近 0 时更激进，更负时更保守。因为每个 turn 的平均 logprob 通常为负，累计和也会随轮次增加而下降；论文的“与错误相关”分析需要结合任务分布和长度效应理解，不能从这个公式推导长且有效轨迹永远不会被误停。不同域和生成单元应重新校准阈值。

触发早停的那一轮仍参加 OPD loss。低 teacher logprob 的错误动作提供负信号，如果连触发轮也删除，会丢掉纠错监督。只有之后未生成的后缀节省计算。这里不同于“先完整生成，事后按年龄丢样本”，因为后缀根本没有执行；也不同于普通长度截断，因为截止点依赖在线教师信号。

**有旧 prefix 时累计从哪里开始？** 从本轮新生成部分开始。若复用 $K_0$ 轮，早停只累计 $K_0+1$ 之后的信号。否则已掌握的历史负 logprob 会先耗尽阈值，使新部分过早终止。停止计数、teacher scoring mask 与训练 loss mask 必须对齐，不能只改 rollout 的最大 turn 数。

**哪些 prefix 能入 buffer？** 只有连续满足 $\bar l_k>\alpha$ 的前缀，得到最长连续合格范围 $1\ldots K_{max}$。不能跳过错误 turn，将后面偶尔高概率的片段拼成不存在的成功上下文。然后选择连续范围中教师认可最低的 turn：

$$
k^*=\arg\min_{1\le k\le K_{max}}\bar l_k.
$$

下次同任务只重用到 $k^*-1$，从 $k^*$ 重新采样。因此“最弱合格 turn”不应被错误地包含在冻结复用部分。随着早期动作改善，该位置向后移动，训练逐渐覆盖更远的交互，而不是固定人工 curriculum。

$\alpha$ 过低容易让不可靠 prefix 污染后续状态，过高又减少复用机会。agent 实验默认 −0.8；数学单轮里错误推理仍可能自我恢复，因此关闭质量过滤。这点尤其提醒 coding 应保留错误恢复价值：把低教师概率一律作为不值得学习，可能减少学生修复能力，实验必须按任务终态验证。

**它是否会重复训练同一历史样本？** prefix 可以多次作为上下文，但本轮生成与训练集中在新段，这不自动等同于已完成样本 replay。重复上下文与重复梯度是不同成本。需要记录 task ID、prefix 终点、resample 起点、新生成 mask 和最终 loss mask；只有这些记录才能准确统计省掉的是生成、教师评分还是梯度计算。

**如何恢复环境？** retail/telecom 的可重复环境与 user simulator 提供了相对明确的状态接口。coding agent 则需要重建文件系统、执行结果、工具进程和外部副作用，使 prefix 对应的状态真的成立。单纯把历史对话发给模型，不会恢复已编辑代码或后台进程。若通过重放工具调用恢复，重放成本和幂等性也应纳入墙钟，不能当成免费 prefix reuse。

本次未发现可核验的作者官方仓库，核读依据固定版 Algorithm、停止规则、buffer 更新和附录配置。没有将 verl 的通用 OPD 实现视为 STRIDE 代码，也未运行在线工具环境。用于异步 OPD 时，还需指定 teacher 是否固定、cache 版本、prefix 的学生版本和 rollout behavior logprob；STRIDE 的覆盖机制与 AsyncOPD 的旧动作分布校正解决不同问题。

若目标是验证“任务未结束先 backward”，STRIDE 提供的是信号可获得性的近邻。仍需证明每个 turn 的 loss 已经确定、global loss 的归一化可提前处理、梯度累积期间参数固定，以及后续任务失败是否改变 earlier turn 的目标。论文不会替代这项等价性分析。[方法原文](https://arxiv.org/pdf/2609.14636v1)

## 3. 实验设置与算力

| 场景 | 模型、预算与评估 |
|---|---|
| 单教师 agent | Qwen3-4B → Qwen3-30B-A3B-Thinking-2507；教师兼 user simulator；retail |
| 多教师 agent | 4B，另有 8B scaling；retail/telecom 各自 expert teacher，独立 simulator |
| 数学 | Qwen3-4B-Base 先用 OpenThought3-8B 做 500 steps SFT，teacher Qwen3-8B |
| 后端 | verl、vLLM、FSDP，参数和 optimizer state offload |
| Agent 配置 | AdamW、LR 1e-5、batch 24、prompt max 24K、response max 8K、alpha −0.8 |
| 数学配置 | LR 2e-6、batch 48、prompt max 1K、response max 30K、关闭 alpha 过滤 |
| GPU 分配 | agent 学生4＋教师/模拟器4；math 学生6＋教师2，型号未完整披露 |
| Steps | 单教师140、多教师120、math160；8B scaling 200 |
| 评估 | 单教师 mean@16/pass@16，多教师 mean@4，AIME24/25 mean@8；每20steps |

agent 评测使用任务的修正版 tau3 release，同时交互设定沿用 tau2-bench。训练任务是按作者引用的数据生成流程合成，不能直接理解成原 benchmark 任务全部用于训练。多教师配置中 simulator 不提供蒸馏监督，只有当前域 expert 评分；单教师同时负责模拟和评分，角色不同可能影响信号一致性。

墙钟统计只包含每步训练时间，排除 evaluation。Fast OPD 的五 turn 截断只用于训练，测试允许更长交互：单教师 50 turns，多教师 100。比较时应保持训练与测试预算分开，否则会将测试限制差异误当模型能力变化。各域报告预算内最佳 mean@4，因此不是所有表项都来自同一步 checkpoint。

论文主对照包含 full OPD、固定预算 Fast OPD、人工 temporal curriculum TCOD-F2B、turnOPD，以及关闭 prefix buffer 的 STRIDE。停止阈值在表中逐项给出；prefix quality threshold 按场景固定。最大的加速往往更激进，也可能损害 mean，不能只选最快一行而省掉质量。

公开超参数使方法可分析，但 GPU 型号、全部数据实例、环境重建细节和官方代码未充分可核验，仍不足以宣称完整复现。本文完成原文、附录和原图表核读，未执行八 GPU 蒸馏，也未测用户 coding pipeline。迁移时建议补 generation、teacher、train、environment restore 四段耗时账本，避免只看到 s/step 改善而漏掉前缀恢复成本。

## 4. 结果与图表解读

![Table 1：单教师 retail 的速度和质量](./assets/papers/paper-stride/table-1-pdf.png)

Table 1 列是 s/step、相对 full OPD 的倍率、mean@16、pass@16。full OPD 为 273.8 秒、0.477/0.850；仅早停、lambda −1 虽达到 61.8 秒、4.43×，mean 只有 0.368。完整 STRIDE 同阈值为 73.3 秒、3.73×、0.475/0.859，说明 buffer 补覆盖具有实证价值。最快并不是质量最高，不能把 4.43× 写成保质量加速。

lambda −2 的完整方案为 117.2 秒、2.34×、mean 0.483、pass 0.837。相对 baseline 的 mean 提升为 0.006，即 0.6 个百分点；相对比例约 1.3%，与“+1.3 个百分点”不同。pass 反而略低，说明两种指标不能互换。教师 mean 0.461、pass 0.847，学生在某些指标超过教师也不意味着全面超越教师能力。

![Figure 4：单教师表现随累计训练时间变化](./assets/papers/paper-stride/wall-clock-source.png)

Figure 4 横轴为累计训练时间（小时），纵轴为 mean@16。STRIDE 红线较早达到 full OPD 蓝线附近的质量；其他缩短 rollout 的方法速度不同但平台较低。虚线为 teacher 参考表现。这里的时间排除 evaluation，不能称为完整实验总墙钟，也没有包含硬件外部成本的统一 GPU-hour 对照。

表格展示多个阈值，曲线展示其中选定方案，两者不能混搭成“2.34× 方案的所有点都与 3.73× 一样”。曲线支持效率/覆盖折中，但少量评估点和非单调波动提醒我们保留误差与 seed 复测。最高一次 mean 不等于稳态平均。

多教师与数学是补充范围：作者在跨域训练仍报告约 4.5× 速度，在 AIME25/24 分别报告约 5.10×/3.08× 的特定对照。对应模型、初始化、阈值、评测采样和训练长度不同，不应排列成同一个统一“最高速度”数字。尤其数学质量过滤被关闭，机制不能直接照搬为 agent 的质量门槛。

**面试如何解释 prefix buffer 解决了什么？** 早停省掉后缀，但减少后期 turn 的学习；复用较可信前缀让新的生成预算推进到薄弱处，恢复覆盖。它并不保证历史 prefix 来自当前策略，也不自动修正 teacher 评分对分布偏移的敏感性。

**教师低概率一定是错误吗？** 论文在自己的环境中做首次错误附近的统计分析，提供代理信号的经验依据。实际 coding 中有效替代实现、长错误恢复或工具格式也可能低概率。迁移应比较教师信号与任务测试结果，并单独评估 recovery trajectories。上述风险不是否定方法，而是确定哪些结论可用、哪些需要新的证据。[结果原文](https://arxiv.org/pdf/2609.14636v1)

## 5. 局限、结论与后续阅读

STRIDE 的核心是在线判断停止位置，再通过可信 prefix 和薄弱 turn 重启保持后续覆盖。它减少的是需要做的交互，属于主动 rollout 选择；与持续生产消费、旧数据校正和提前梯度计算应分开归类。

限制包括 teacher endorsement 代理的可靠性、阈值跨域迁移、prefix 历史版本、环境恢复成本与错误恢复价值。retail/telecom 的证据不能直接代表长程 coding；没有官方实现也使 buffer 的状态保存细节尚不可代码核验。所有加速应带上质量指标与时间口径。

与 [AsyncOPD](#paper=paper-asyncopd) 对读，分别检查“少生成哪些状态”和“已生成旧数据如何训练”；与 [APRIL](#paper=paper-april) 比较高质量 prefix reuse 与未完成 continuation；与 [FlexMARL](#paper=paper-flexmarl) 比较在线反馈和提前 backward 的区别。若用于自己的 OPD baseline，先保留完整轨迹对照并测覆盖，再添加早停，最后打开 prefix buffer，才能定位速度与学习质量的变化。[固定版论文](https://arxiv.org/abs/2609.14636v1)
