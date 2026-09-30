# 论文库全量内容审查（2026-09-30）

**结论：目前 39 篇已发布论文并未全部达到 DDPM 精读水平。** 以 [DDPM 报告](../content/papers/diffusion/paper-ddpm.md)为对照，7 篇较完整报告通过修正后的自动结构检查，**32 篇 OPD 仍是短报告**。自动检查通过只表示格式、基本内容量和本地图片证据满足规则；没有逐项证明解释正确、实验信息已穷尽或源码确已复现。

整改采用统一的[图表证据与发布流程](./paper-evidence-workflow.md)。门槛现在核对方法图用途、原论文表格、PDF 裁剪出处与版本；`npm run papers:quality:strict` 会继续对 32 篇历史短报告报错，直到逐篇补齐。保留现有链接不等于批准这些报告的深度。

本次检查逐篇读取 frontmatter、五个正文模块、插图引用及本地 `figures.json`，对照仓库保留的 PDF 裁图和已抓取的 arXiv 原文结构，抽查 [GKD](https://arxiv.org/html/2306.13649)、[BPM](https://arxiv.org/html/2607.22334)、[DN-MOPD](https://arxiv.org/html/2609.35347)、[Beyond Prompt Count](https://arxiv.org/html/2609.37377)等官方原文。以下是**内容覆盖审查**，不是对 39 篇全部数值、代码与每张原图完成独立复现。`npm run papers:quality -- --summary` 可重新生成每篇的篇幅、图表与元数据清单。

## 全库可复核的缺口

| 范围 | 审查结果 | 对阅读的影响 |
| --- | --- | --- |
| OPD 35 篇 | 21 篇正文无图；11 篇只有 1 张图；只有 3 篇有 4–6 张图 | 31 篇在“结果与图表解读”里没有原图/原表，读者只能信转述的几个数字 |
| OPD 35 篇 | 30 篇实验模块没有 Markdown 配置表；这 30 篇正文也没有任何 Markdown 表格 | 模型、训练集、评测、超参、硬件和限制不易逐项核对 |
| 未升级 OPD 32 篇 | 每篇背景、方法、实验、结果四个模块都低于现有结构门槛；正文仅约 1.5–3.7 千字符 | 五个标题看似齐全，实际大多是提要，不足以从零学懂问题和机制 |
| 未升级 OPD 32 篇 | `paper_url` 均未固定到 `v1/v2…` | 图号、表格数字和后续修订版无法稳定对应 |
| 未升级 OPD 32 篇 | 9 篇单位仍写 `Institution not stated on the arXiv author page` | 元数据尚未核到论文首页，不能当完整书目信息 |
| 带旧图的 11 篇 OPD | 通常只有 `figure-method.png`，没有该论文的 `figures.json` 和固定 PDF 裁图记录 | 只能确认页面能加载图片，不能从仓库确认图号、版本、页码、裁剪完整性和结果表出处 |
| 所有报告 | `已核原文` 是来源标记，不等于 DDPM 水平；自动检查主要统计字数、公式/表格语法、图片个数与 hash | 这些检查不能判断段落是否讲清楚、数字是否取对、代码是否真的对应论文 |

前几轮发生浮动的直接原因是：批量收录时先产出五段式短报告，之后只把 DDPM 和 3 篇 OPD 标记为 `depth_standard: ddpm`。`content/metadata/paper-depth-legacy.json` 原先把其余 35 篇的旧文件 hash 冻结，使构建继续通过；其中 3 篇早期基础报告其实已有相当篇幅和图表，只是被过硬的规则误判。本次把这 3 篇纳入结构初筛并修正规则，冻结名单缩至 32 篇。冻结机制**阻止继续修改浅报告，却没有把旧报告补深**。因此“构建通过”“图片加载成功”和“全库达到 DDPM 标准”是三件不同的事。

自动门槛本身也不够准确。例如 ReAct 是流程型论文，没有必要硬凑数学公式，却因无 `$$` 被判失败；GQA、PagedAttention、ReAct 的原表后有实质解释，只因没有字面上的“解读”二字被判失败。这些误报已按流程型方法的原图/源码、表后实质段落修正。反过来，写足字数、放三张图并设置 `visuallyVerified: true`，脚本也无法证明解释正确。今后应把自动检查称为**结构/证据初筛**，再由人工核对原文、图表、配置和源码。

## 已有较完整报告：7 篇

| 报告 | 正文/插图 | 审查判断与仍需核对项 |
| --- | --- | --- |
| [DDPM](../content/papers/diffusion/paper-ddpm.md) | 约 7.7k 字符 / 5 图 | 当前参照样稿：背景、公式、源码、训练成本和主图表均成体系；没有声称覆盖所有附录图 |
| [GQA](../content/papers/llm/paper-gqa.md) | 约 10.6k / 7 图 | 方法、checkpoint 转换、实验与原表较完整；Table 1 后使用普通段落解释，旧规则误报已修正 |
| [PagedAttention](../content/papers/infra/paper-pagedattention.md) | 约 7.7k / 6 图 | 显存配置、系统对照及多张原图较完整；Table 1 后隔着一张 Markdown 表的解释现可被识别 |
| [ReAct](../content/papers/agent/paper-react.md) | 约 6.9k / 5 图 | 流程型论文的行为轨迹、实验表与代码已有解释；无需虚构公式，已按原图与源码对照检查 |
| [Multi-Rollout MOPD](../content/papers/llm/paper-opd-2605-12652.md) | 约 7.3k / 5 图 | 已补主结果、消融、信号质量和 8×H100 单步成本；其他任务的训练数据与完整 GPU-hours 仍需追原文/代码 |
| [Cross-Tokenizer OPD](../content/papers/llm/paper-opd-2606-09456.md) | 约 7.2k / 6 图 | 已补对齐/credit assignment、原表与配置；跨 tokenizer 特殊 token 边界的实现验证仍是静态核读 |
| [Multi-Teacher MOPD](../content/papers/llm/paper-opd-2606-30406.md) | 约 7.5k / 4 图 | 已补训练流水线、附录配置、主结果和动态曲线；作者未给完整官方仓库/总 GPU-hours，报告应持续保留此限制 |

这 7 篇不应被说成“已经独立验证论文全部结论”。它们是可继续精读的报告；另外 32 篇的首要问题是**根本没有到这一层级**。

## 无正文插图的 OPD：21 篇

表中“需补”指出最优先补的**解释与证据**；选择哪张最终插图仍要对照固定版本 PDF，不把 HTML 转换图直接当已核图片。

| 报告 | 正文约字 | 最优先补齐 |
| --- | ---: | --- |
| [MiniLLM](../content/papers/llm/paper-opd-2306-08543.md) | 2.1k | 序列级 reverse KL 与训练过程的推导、模型/数据配置、主结果原表和相关图 |
| [GKD](../content/papers/llm/paper-opd-2306-13649.md) | 2.3k | on-policy 比例与 JSD 的例子、T5 各任务配置、Figure 1/4/5 的原图与结果边界 |
| [Self-Distilled Reasoner](../content/papers/llm/paper-opd-2601-18734.md) | 2.4k | 自蒸馏训练/推理如何切换、基线预算、主结果与消融原表 |
| [On-Policy Context Distillation](../content/papers/llm/paper-opd-2602-12275.md) | 2.0k | 长上下文 teacher 与短上下文 student 的条件差异、方法图、主结果原表 |
| [Entropy-Aware OPD](../content/papers/llm/paper-opd-2603-07079.md) | 1.9k | entropy 权重/选择的数值例子、训练配置、按熵分组结果和消融图表 |
| [Relaxed OPD](../content/papers/llm/paper-opd-2603-11137.md) | 1.9k | relaxed 条件具体改变什么、目标函数与预算对照、主结果原表 |
| [Why Self-Distillation Degrades](../content/papers/llm/paper-opd-2603-24472.md) | 1.9k | 退化机制与替代解释、失败案例、核心对照/消融表，而非只复述现象 |
| [OPD Survey](../content/papers/llm/paper-opd-2604-00626.md) | 2.0k | 方法谱系图、原文比较表和未解决问题；综述没有自己的训练 GPU，不应硬填实验配置 |
| [Byte-Level Interface](../content/papers/llm/paper-opd-2604-07466.md) | 2.0k | byte 单元上的概率/边界例子、与 token 对齐方案的差别及对照表 |
| [Rethinking OPD](../content/papers/llm/paper-opd-2604-13016.md) | 2.0k | phenomenology→mechanism→recipe 的证据链、各实验条件和主图表 |
| [Unmasking OPD](../content/papers/llm/paper-opd-2605-10889.md) | 1.5k | 哪些条件有益/有害的定义、受控实验、失败案例与原图；当前是全库最短之一 |
| [OPRD](../content/papers/llm/paper-opd-2606-06021.md) | 1.9k | representation 蒸馏目标、张量/层级对应、训练细节和主对照图表 |
| [Formula-Driven Survey](../content/papers/llm/paper-opd-2606-22793.md) | 1.7k | 统一符号、方法比较表与研究议程；按综述证据要求审，不强制训练算力 |
| [Thinking Models Self-Distillation](../content/papers/llm/paper-opd-2607-05184.md) | 2.0k | thinking 轨迹与答案的监督边界、长度/预算控制、失败对照及原表 |
| [Demystifying OPD](../content/papers/llm/paper-opd-2607-13399.md) | 1.7k | teacher/student 各自作用、病理现象、调节机制和反例图表 |
| [Test-Time Scaling View](../content/papers/llm/paper-opd-2608-11829.md) | 1.7k | 推理预算与蒸馏收益的变量控制、曲线读法、训练/评测预算表 |
| [ACTD](../content/papers/llm/paper-opd-2608-29662.md) | 1.8k | anchor 映射与 residual regularization 逐步示例、跨词表实验与消融 |
| [DN-MOPD](../content/papers/llm/paper-opd-2609-35347.md) | 3.3k | 域标准差缩放已有公式，但缺 Figure 1/3、Table 1–4 原图、模型/长度/算力账本；单位仍占位 |
| [Least-Square Policy Distillation](../content/papers/llm/paper-opd-2609-35505.md) | 1.9k | RL/least-square 目标的推导、估计偏差、主结果/消融与单位核对 |
| [Solving Without Stopping](../content/papers/llm/paper-opd-2609-37326.md) | 1.7k | 小模型持续解题的训练流程、停止条件与对照曲线；单位仍占位 |
| [Beyond Prompt Count](../content/papers/llm/paper-opd-2609-37377.md) | 1.9k | prompt 数和 token/rollout 预算怎样匹配、来源迁移表与敏感性曲线；单位仍占位 |

## 只有一张正文插图的 OPD：11 篇

这些图多数是方法图，**没有附带任何主结果原表截图**；少数图实际在结果模块，方法图反而缺失。11 篇都还没有固定 PDF 图表清单。

| 报告 | 正文约字 | 最优先补齐 |
| --- | ---: | --- |
| [SimCT](../content/papers/llm/paper-opd-2605-07711.md) | 3.2k | 当前对齐图之外，补共同 token/文本单元的数值例子、主表与消融 |
| [Trust Region OPD](../content/papers/llm/paper-opd-2606-01249.md) | 2.3k | trust-region 约束的作用和失败条件、训练参数、主结果/稳定性表 |
| [BPM](../content/papers/llm/paper-opd-2607-22334.md) | 3.7k | byte-prefix marginalization 的逐步例子、算法成本与跨 tokenizer 主结果原表 |
| [Self-Distillation without Supervision](../content/papers/llm/paper-opd-2608-06296.md) | 2.1k | 无外部标签的反馈来源、训练闭环和可信度限制、主结果/消融 |
| [OPD with Verifiable Reward](../content/papers/llm/paper-opd-2608-24696.md) | 2.2k | reward 与 teacher 信号的组合方式、预算对齐和结果表；单位仍占位 |
| [CompassOPD](../content/papers/llm/paper-opd-2609-10154.md) | 2.2k | within-family shift 的概率解释、不同模型族条件和主结果表 |
| [EOS Disagreement](../content/papers/llm/paper-opd-2609-20511.md) | 2.2k | 当前唯一插图在结果模块；补 EOS 机制流程、长度分布/停止行为对照及配置，单位仍占位 |
| [Teacher Should Think Ahead](../content/papers/llm/paper-opd-2609-22254.md) | 2.4k | continuation 触发与截断规则、teacher 查询成本、主结果/消融原表；单位仍占位 |
| [MOPD-Router](../content/papers/llm/paper-opd-2609-30837.md) | 3.7k | router 决策的样例与路由代价、Table 1/2/4 原表；不能只引用表中的几个数字 |
| [ESCD](../content/papers/llm/paper-opd-2609-34738.md) | 3.3k | event completion 的条件概率例子、边界失败情况、主结果与 token 开销原表 |
| [Dr. OPD](../content/papers/llm/paper-opd-2609-38025.md) | 2.5k | 学什么/跟谁学的决策流程、Table 4/5 的模型与基线口径及原表 |

## 整改顺序与验收口径

先处理会影响整个专题理解的 GKD、MiniLLM、SimCT、BPM、MOPD-Router、DN-MOPD 和数据选择/失败机制论文，再覆盖其余记录。每篇先锁定 arXiv 版本并核作者单位；读正文和附录后写问题定义、代表前作与明确差异，沿一个具体输入或公式讲方法；再建模型/训练数据/评测/超参/硬件的表；最后从固定 PDF 提取能支持结论的原图/原表，逐张核对并解释。没有公开代码要写明，公开了则固定 commit 读核心函数。综述按方法谱系与比较矩阵验收，流程型论文用算法步骤和案例验收，不为了通过正则硬添公式。

整改时不应仅把文字扩到某个字数、随意插三张图或把 `visuallyVerified` 设为 true。每篇还需人工对照原文检查数值、图例、条件和结论边界。全部短报告补齐之前，**不能再宣称全库按 DDPM 标准整理完成**。
