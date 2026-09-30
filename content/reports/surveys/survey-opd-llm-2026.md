---
id: "survey-opd-llm-2026"
title: "专题调研：LLM On-Policy Distillation（OPD）— 多教师 MOPD 与跨 Tokenizer 路线"
type: "survey"
date: "2026-09-30"
directions: ["llm"]
paper_ids: ["paper-opd-2306-13649", "paper-opd-2306-08543", "paper-opd-2601-18734", "paper-opd-2602-12275", "paper-opd-2603-07079", "paper-opd-2603-11137", "paper-opd-2603-24472", "paper-opd-2604-00626", "paper-opd-2604-13016", "paper-opd-2604-07466", "paper-opd-2605-10889", "paper-opd-2605-12652", "paper-opd-2605-07711", "paper-opd-2606-01249", "paper-opd-2606-06021", "paper-opd-2606-09456", "paper-opd-2606-22793", "paper-opd-2606-30406", "paper-opd-2607-05184", "paper-opd-2607-13399", "paper-opd-2607-22334", "paper-opd-2608-06296", "paper-opd-2608-11829", "paper-opd-2608-24696", "paper-opd-2608-29662", "paper-opd-2609-10154", "paper-opd-2609-20511", "paper-opd-2609-22254", "paper-opd-2609-30837", "paper-opd-2609-34738", "paper-opd-2609-35347", "paper-opd-2609-35505", "paper-opd-2609-37326", "paper-opd-2609-37377", "paper-opd-2609-38025"]
tags: ["专题调研", "LLM", "On-Policy Distillation", "MOPD", "Cross-Tokenizer"]
updated: "2026-09-30"
summary: "截至 2026-09-30 的 LLM OPD 专题：整理 35 篇，其中 33 篇为 2026 年 arXiv 新作，聚焦 teacher/student 状态分布、训练稳定性、多教师 MOPD 与 cross-tokenizer 概率对齐。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

这份专题要回答一个实际的 post-training 问题：**让 student 在自己会走到的状态上学习 teacher 的知识，什么时候比 teacher 先写好数据再做 SFT 更合适？当 teacher/student tokenizer 不同，或能力来自多个专长 teacher 时，怎样保留稠密监督而不引入新的偏差？**

本文用 LLM 训练语境中的 **On-Policy Distillation**（OPD）作为主术语：student 按当前或近似当前策略采样输出，再在这些 student-induced prefixes 上获得 teacher/self-teacher 的监督。个别论文标题将类似过程称为 Online Policy Distillation；名称不能代替核对 state distribution。一个实用的纳入判断是：**训练状态是否实质来自 student 自身生成，而不仅是 teacher 写出的固定答案？**

直观地说，Off-policy SFT 像让学徒背诵专家已经写好的解答；OPD 则让学徒先自己尝试，在自己实际走到的每一步上请专家反馈。它可以修正学徒特有的错误路径，也会让 teacher 必须在低质量或未完成的 student prefix 上判断，因而出现 teacher/student mismatch、tokenizer 对不齐、过长回答和昂贵 teacher forward 等问题。OPD 不是单一 loss：有逐 token teacher 分布蒸馏、sampled-token log-ratio / policy-gradient 估计、representation-level 监督、verifier/outcome 混合和 peer-conditioned 等不同范式。

**适用场景**：小模型承接大模型推理、RL 专长模型合并、跨家族 teacher→student 迁移、降低数据生成与人工标注成本。若 teacher API 只给文本、没有概率或 logit，或者任务只有最终可验证 reward，就要先确认所用方法是否还保留 OPD 定义里的 student-state 反馈，而不是泛称“蒸馏”。

## 2. 检索范围、方法与覆盖边界

**检索截止日：2026-09-30（UTC）。** 候选通过 arXiv 官方 API / 论文页面检索与作者公开 HTML 核对；检索词覆盖 `on-policy distillation`、`on-policy self-distillation`、`cross-tokenizer`、`multi-teacher`、`MOPD` 等。先筛标题/摘要，再按是否直接涉及 LLM、student-generated state 和本专题的两个重点收录；同主题的一般图像/视频 diffusion、speech-only、多模态感知及只做通用 KD 的工作不计为核心条目。为理解技术边界，保留两篇早期基线与一篇 byte-interface 相邻工作。最终 **35 篇**：2 篇奠基工作 + 33 篇 2026 年论文或综述，低于 50 篇上限。

这是可更新的高相关专题集合，不声称穷尽所有 2026 预印本、会议论文、工业技术报告或尚未进入 arXiv 索引的工作。特别是 2026 年 9 月底提交的论文还很新；收录只表示它值得阅读，不等于同行评审通过或结论已经复现。所有数据配置和机构字段以对应论文页面为准；机构从作者公开 HTML 页面提取，未列出的条目明确标注未披露。每篇条目目前是统一格式的**资料整理卡片**，而不是宣称完成源码逐函数分析的精读报告。

## 3. 统一方法谱系：OPD 具体“on-policy”在哪里？

令提示为 $x$，student 策略为 $\pi_	heta$，teacher 为 $q_\phi$。学生生成 $y_{1:T}\sim\pi_	heta(\cdot\mid x)$。在 student 已走过的前缀 $y_{<t}$ 上，teacher 可以给完整下一 token 分布 $q_\phi(\cdot\mid x,y_{<t})$，或只给被采样 token 的 log probability、局部表示、偏好/回报。最简分布式目标可以写作：

$$
\mathcal L_{\mathrm{OPD}}=\mathbb E_{y\sim\pi_	heta}\left[\sum_{t=1}^{T} w_t\,D\!\left(q_\phi(\cdot\mid x,y_{<t}),\pi_	heta(\cdot\mid x,y_{<t})ight)ight].
$$

这里公式只表达共同骨架，**不规定 KL 方向、是否对所有词表求和、权重 $w_t$、序列归一化、temperature、stop token 或 rollout 策略**。这些选择会改变训练行为。另一路把 teacher/student 的 sampled-token log-ratio 视作 token reward，再做带 baseline/credit assignment 的 policy-gradient 更新；它与直接最小化完整分布 KL 在估计器、方差和支持集上都不等价。

| 路线 | 改动的环节 | 关键问题 | 代表收录 |
|---|---|---|---|
| 基线与状态分布 | 从静态 teacher 数据扩展到 student rollouts | 暴露偏差是否真的被缓解，student 的轨迹是否带来新监督 | GKD、MiniLLM、Self-Distilled Reasoner |
| 目标与稳定 | divergence、token 选择、归一、trust region、熵/梯度控制 | student prefix 偏离 teacher 训练分布时，局部监督会不会错导 | REOPOLD、TrOPD、Entropy-Aware、Demystifying、LSPD |
| 教师信息与反馈 | context/self teacher、verifier、representation、teacher continuation | teacher 的额外信息是否有益，稠密反馈如何与结果信号互补 | OPCD、OPRD、OPD with Verifiable Reward、AC-OPD |
| 多样能力合并 | 多 rollout 互相提供正负证据；多个领域 teacher 路由/校准 | 如何利用 peer signal，如何处理 teacher 间冲突和尺度不匹配 | Multi-Rollout MOPD、Multi-Teacher MOPD、MOPD-Router、Domain-Normalized MOPD |
| 跨 tokenizer | token/span/byte/prefix/event 对齐或跨家族桥接 | teacher 一个 token 对应多种 student token 时，概率质量放到哪里 | Breaking Tokenizer Barrier、SimCT、BPM、CompassOPD、ESCD |
| 能力边界与失效分析 | pass@K/avg@K、长度与 EOS、privileged contexts、数据组成 | 提升的是单次命中率、可解能力集合，还是只改变输出行为 | test-time scaling、EOS mismatch、Solving Without Stopping、Unmasking |

### MOPD 缩写冲突：必须按全名读

当前至少有两种机制差异很大的工作都叫 MOPD：

1. **Multi-Rollout On-Policy Distillation via Peer Successes and Failures**（2026-05，2605.12652）：一个 student 针对同一 prompt 采多条 rollout；teacher 在上下文里看到同伴的成功和失败解答，用正反案例构造更丰富的目标。它解决的是“同一次采样组里其他轨迹的信息被浪费”。
2. **MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training**（2026-06，2606.30406）：先分别对领域模型做 RL，形成多个专长 teacher，再通过 student 自身 rollout 蒸馏，解决多领域能力整合及更新冲突。摘要报告 Qwen3-30B-A3B 对照多种合并方案，并提到 MiMo-V2-Flash 后训练应用。

9 月出现的 **MOPD-Router** 和 **Domain-Normalized Multi-Teacher OPD** 属于第二条“多 teacher 融合”线的后续问题，分别强调 teacher 路由和跨领域信号归一；不要仅凭 acronym 把它们和 peer-rollout MOPD 合并。它们与多专家蒸馏的共同点是要利用异质能力，区别在 teacher 数量、teacher 是否按 prompt/领域选择、以及一条训练样本里出现哪些轨迹。

### Cross-tokenizer OPD：概率从 teacher 词表怎样送到 student 词表？

token ID 不是语言学意义上的共享动作。一个 teacher token 可能是多个字节/字符，student 可能把它拆成几个 token；反过来也可能合并。若仅用 teacher 生成的完整文字做 SFT，就丢掉 teacher 分布里的“还有哪些替代 token”信息；若硬把同位置 token 对齐，又可能把概率质量给了字节内容不匹配的学生动作。关键困难不仅是 tokenization 不同，还在于 student 已经输出了 teacher token 的**部分前缀**时，下一步可能存在多个合法补全。

本专题纳入的路线彼此不应视为换名字：

- **Byte-level interface** 将两边移到共享字节表示；先解决共同表示单位，但是否是严格 OPD还取决于训练状态是否由 student 采样。
- **SimCT / token mapping** 尝试恢复跨 tokenizer 中损失的 teacher 监督，重点审计概率守恒与映射语义。
- **BPM (Byte-Prefix Marginalization)** 将 teacher token 概率按字节前缀投到 student 可用 token 上，解决“概率给谁”的路由问题。
- **CompassOPD** 利用同家族 likelihood shift 作为跨模型族桥梁，是与显式字节映射不同的相对似然路线。
- **ESCD (Event-Set Completion Distillation)** 在 teacher token 事件已部分进入但未完成时，对多个 byte-compatible 下一步的**总概率质量**进行监督，而不武断要求 teacher 指定这些等价补全之间的拆分。摘要报告特定 tokenizer pair 下一步补全覆盖超过 99% 的兼容 teacher mass，并测 1T→35B MoE；这是该实验范围的观察，不是对任意词表的保证。

读这些方法时可用三个检查：①对齐在 token、字符串、UTF-8 bytes 还是 completion set 上？②student rollout 中间状态都有定义吗？③映射后的概率总量、条件归一和 teacher support 是否守恒？如果作者用一个短句说“align tokens”，应继续追算法框与 loss。

## 4. 实验与证据：横向比较要先对齐口径

OPD 论文常见 benchmark 包括数学推理、代码、通用指令和多轮工具任务；teacher/student 规模与架构、数据集、rollout 数、采样温度、max response length、teacher forward 次数、GPU 和训练 token 预算差异很大。目前本轮能从 2606.30406 摘要核实的模型例子是 **Qwen3-30B-A3B**，且作者提到用于 **MiMo-V2-Flash** post-training；2609.34738 摘要提到跨家族/跨 tokenizer，含 **1T teacher → 35B MoE student** 的实验。其他论文报告的硬件、optimizer、学习率、数据快照和 GPU-hours 需进对应全文/附录逐篇核对。

| 比较项 | 读每篇时需要记录 | 目前专题处理 |
|---|---|---|
| Teacher / student | checkpoint 全名、参数量、家族、是否 thinking/MoE、tokenizer | 只引用原文已明确的信息；不根据规模名称推断硬件 |
| 训练输入 | 预训练/冷启动数据、prompt 集、rollout 策略、每题样本数 | 区分 teacher 生成答案与 student 采样轨迹 |
| 目标与实现 | KL 方向、support、样本 token/全词表、mask、reduction、EOS | 逐项记录，避免把同名 OPD 当同目标 |
| benchmark | 数据集、split、提示、pass@k/avg@k、温度、长度 | 不拼接不可比的公开分数做排行榜 |
| 计算成本 | teacher calls、采样 token、训练 token、GPU 类型/数量、墙钟 | 未公开项写“原文未披露”；TPU 时间不臆换 GPU-hours |
| baseline | SFT、RLVR、常规 OPD、数据量/teacher预算 | 检查预算是否匹配，再接受效率结论 |

**当前证据强度判断。** 基础方法和问题定义已有多条相互引用的工作；但 2026 年新增方向密集且不少是 9 月 arXiv 初稿。诸如“OPD 扩大能力边界还是提升采样效率”“teacher 过强是否有害”“多 teacher 如何融合”等，当前更适合视为竞争中的研究假设。不同论文若用不同 pass@K、模型和训练预算，不能把摘要结论凑成确定共识。

## 5. 共识、分歧与工作假设

**较稳妥的共识**是 OPD 改变训练时遇到的 prefix/state 分布，并能给 student 自己生成的 token 位置提供更稠密反馈；因此它有机会针对自回归错误，而不是只复制专家答案。代价是 teacher 要在 student 生成的错误前缀上工作，teacher 质量、上下文兼容性、输出长度以及 student/teacher tokenizer 共同影响信号质量。

**仍在争论的解释**是它是否把新推理能力从 teacher 转移给 student，还是主要改善有限 sampling budget 下的解题概率。test-time scaling 研究用 avg@K 与 pass@K 分开衡量后报告不同趋势；机制研究则关注 teacher/student distributional distinguishability、高概率 token 重叠、prompt diversity 及 rollout 质量。这些指标回答的问题不同，应放在一起读而不是互相替代。

**本人的工作假设**：OPD 的关键不是“在线/离线”标签，而是训练更新实际在哪些状态上取样、teacher 在这些状态上能提供什么可识别的改进信号，以及优化器怎样把信号路由到 student 可执行的动作。判断算法前先画出 `prompt → student rollout → teacher observation → token/trajectory target → update` 数据流。

**常见误区**：

- 把 teacher-generated SFT 说成 OPD；它即使保存了 teacher logits，若训练状态仍由固定 teacher 文本给出，也不自动是 on-policy。
- 认为 reverse KL 的数学形式已经保证训练稳定；分布错位时估计梯度和 token support 仍可能失效。
- 将 MOPD 直接解释成一个唯一方法；目前至少有 peer-rollout 和 multi-teacher 两种同名论文。
- 因为 cross-tokenizer 工作做了文本对齐，就认为可以无损蒸馏概率；要查部分 token 前缀和合法补全的处理。
- 将 pass@1 上升等同于可解能力上限上升；高 K 评测可能给出不同答案。
- 把论文没披露的训练配置按常见集群经验补齐。GPU 算力应留空并标“未披露”，不从参数量猜卡数。

## 6. 场景化结论与选型

- **teacher/student tokenizer 相同，teacher 可本地算 logits**：先从 GKD/MiniLLM 的状态分布、KL方向入手，再比较 TrOPD、REOPOLD 与 entropy-aware 方法；把 token mask、response length、teacher compute 和 reverse-KL estimator 纳入监控。
- **多领域 RL teacher 要合并成一个模型**：优先看 multi-teacher MOPD 的 pipeline；再比较 MOPD-Router、domain normalization 和 reliability weighting。为每个 teacher 分别评测，再测冲突域和遗忘，不能只报告综合平均。
- **同一个 student 对同题采了多个答案**：peer-rollout MOPD 是更贴近问题的切口；要注意 teacher 看成功解答后是否泄漏答案，和额外上下文 token 成本。
- **teacher/student tokenizer 不同**：按数据流选 byte/token mapping、BPM、CompassOPD 或 ESCD；先对齐相同文本概率质量、 student rollout 和 teacher 调用预算，再比较任务结果。普通 cross-tokenizer KD 作为边界基线。
- **student thinking traces 很长或停不下来**：把 EOS / stop token 语义、thinking vs non-thinking、正确答案提交位置纳入诊断；不要先简单截断回答，否则会把内容和停止机制混在一起。
- **外部 teacher/标注不可用**：读 teacher-free self-distillation，但仔细界定所谓“无监督”还使用了什么 verifier、筛选或隐式先验。

这些是阅读路径，不是对尚未测试的训练配置给出的生产选型保证。

## 7. 空白与下一轮验证

建议把下一轮精读拆成三个可复现小问题，而不是再收 50 篇摘要：

1. **是否学到能力还是更容易采到？** 固定 base、训练 token 与题集，比较 base/OPD 的 avg@k、pass@k、solvability curve 和训练中间 checkpoint；报告温度、独立样本数与区间。
2. **不同 tokenizer 的监督是否守恒？** 选一对异构词表，逐字节验证 teacher 事件到 student token / completion set 的映射质量和条件化；统计漏掉/误分配概率，再测数学与代码，不只测 exact string imitation。
3. **多 teacher 的收益来自融合还是更多计算？** 固定 teacher rollout/token budget，对比 sequential RL、参数 merge、离线 SFT、multi-teacher OPD 与 router；逐领域报告保留、负迁移、teacher calls 和 GPU-hours。

每个精读报告再补论文图表解释与源码函数对照：图来自正式 arXiv PDF/源码而非网页缩略图；GitHub 只有作者原文/官方组织明确关联后才标“官方实现”。当前专题不声称 35 篇都完成了源码阅读，也没有从 HTML 抽图后拼成未经核对的卡片图。

## 8. 阅读顺序与自测

**建议顺序**：

1. GKD + MiniLLM：先把 exposure mismatch、KL 方向和自回归采样理解清楚。
2. 2026 survey 两篇：建立 teacher access / feedback / loss 粒度和公式路线图；将引用追到原文。
3. Rethinking + Unmasking + Demystifying + test-time scaling：理解何时有效、梯度质量、长度问题和能力边界争论。
4. Multi-Rollout MOPD vs Multi-Teacher MOPD：画两张不同的数据流图，不要只背同名简称。
5. Breaking Tokenizer Barrier → SimCT → BPM → CompassOPD → ESCD：画一个跨词表例子，比如 teacher token `"international"` 被 student 切成多个 subword 时，中间状态的 teacher 监督应该是什么。
6. 最后看 MOPD-Router、Domain-Normalized MOPD、EOS mismatch、AC-OPD 等 2026-09 新稿，判断它们各自在数据流上改了哪一步。

合上论文回答：①什么条件决定一个算法算 on-policy？②KL 两个方向对 student 自己低概率采到的 token 会有什么不同？③为什么 teacher/student 不同 tokenizer 时，合法补全的总概率可能比某个单 token 概率更有意义？④两个 MOPD 全名和监督单位分别是什么？⑤pass@1 变好而 pass@K 变差意味着什么？

## 9. 论文库收录清单与持续更新

以下每条都是论文库中的独立条目，可按“语言 → 训练与适配 / 推理 / Agent”筛选；此处按问题分组，库中仍按论文发表时间从近到远排序。每条卡片都保留英文原题、作者、arXiv 页面与可核机构，详情页标明资料整理的证据范围。

### 基础与定义

- [#paper-opd-2306-13649] — On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes（2023）
- [#paper-opd-2306-08543] — MiniLLM: On-Policy Distillation of Large Language Models（2023）
- [#paper-opd-2601-18734] — Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models（2026）
- [#paper-opd-2602-12275] — On-Policy Context Distillation for Language Models（2026）
### 目标函数、稳定性与训练动力学

- [#paper-opd-2603-07079] — Entropy-Aware On-Policy Distillation of Language Models（2026）
- [#paper-opd-2603-11137] — Scaling Reasoning Efficiently via Relaxed On-Policy Distillation（2026）
- [#paper-opd-2603-24472] — Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs?（2026）
- [#paper-opd-2604-13016] — Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe（2026）
- [#paper-opd-2605-10889] — Unmasking On-Policy Distillation: Where It Helps, Where It Hurts, and Why（2026）
- [#paper-opd-2606-01249] — Trust Region On-Policy Distillation（2026）
- [#paper-opd-2607-13399] — Demystifying On-Policy Distillation: Roles, Pathologies, and Regulations（2026）
- [#paper-opd-2608-06296] — On-Policy Self-Distillation without Any Supervision（2026）
- [#paper-opd-2608-11829] — Towards Understanding On-Policy Distillation through the Lens of Test-Time Scaling（2026）
- [#paper-opd-2608-24696] — On-policy Distillation with Verifiable Reward（2026）
- [#paper-opd-2609-20511] — When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation（2026）
- [#paper-opd-2609-22254] — Teacher Should Think Ahead: Adaptive Continuations for Reliable On-Policy Distillation（2026）
- [#paper-opd-2609-35505] — An RL View of OPD: Least Square Policy Distillation for Sample-Efficient LLM Reasoning（2026）
- [#paper-opd-2609-37326] — Solving Without Stopping: On-Policy Distillation at Small Scale（2026）
- [#paper-opd-2609-37377] — Beyond Prompt Count: How Data Shapes Transfer in On-Policy Distillation（2026）
- [#paper-opd-2609-38025] — Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models（2026）
### 多 rollout 与多 teacher（注意 MOPD 缩写重名）

- [#paper-opd-2605-12652] — Multi-Rollout On-Policy Distillation via Peer Successes and Failures（2026）
- [#paper-opd-2606-30406] — MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training（2026）
- [#paper-opd-2609-30837] — MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation（2026）
- [#paper-opd-2609-35347] — Beyond Teacher Assignment: Domain-Normalized Multi-Teacher On-Policy Distillation（2026）
### 跨 tokenizer / 跨模型族

- [#paper-opd-2604-07466] — Cross-Tokenizer LLM Distillation through a Byte-Level Interface（2026）
- [#paper-opd-2605-07711] — SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation（2026）
- [#paper-opd-2606-09456] — Breaking the Tokenizer Barrier: On-Policy Distillation across Model Families（2026）
- [#paper-opd-2607-22334] — Cross-Tokenizer On-Policy Distillation via Byte-Prefix Marginalization（2026）
- [#paper-opd-2608-29662] — ACTD: Anchor-Based Cross-Tokenizer Distillation with Residual Regularization（2026）
- [#paper-opd-2609-10154] — CompassOPD: Cross-Family On-Policy Distillation via Within-Family Likelihood Shifts（2026）
- [#paper-opd-2609-34738] — Beyond Token Alignment: Event Completion for Cross-Tokenizer On-Policy Distillation（2026）
### 综述入口

- [#paper-opd-2604-00626] — A Survey of On-Policy Distillation for Large Language Models（2026）
- [#paper-opd-2606-22793] — A Formula-Driven Survey and Research Agenda for On-Policy Distillation（2026）

**更新日志**

- 2026-09-30：建立 35 篇 OPD 专题集合；核对 arXiv 官方元数据/作者页面。加入 MOPD 双重命名说明与 tokenizer 路线分层。9 月底新稿保留为待精读候选。
- 下次更新：核实论文版本/会议状态和 GitHub 官方实现；逐篇补充模型、数据、benchmark、optimizer、硬件、训练成本与关键图；优先对 Multi-Teacher MOPD、ESCD、BPM 做完整精读。未披露项目将如实保留。
