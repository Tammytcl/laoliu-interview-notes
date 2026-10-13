---
id: reward-judge-interview
title: "Reward Model、PRM与LLM Judge：监督、校准和评测面试"
category: llm
difficulty: 进阶
tags: ["P0", "Reward Model", "PRM", "ORM", "LLM Judge", "Evaluation"]
updated: "2026-10-13"
summary: "补清单中结果/过程评分与judge知识，分清偏好模型、任务critic、verifier和生成式评委，给出公式、校准案例、16道问答与项目接线。"
draft: false
---

## 1. 问题背景 谁在打分，分数证明了什么

后训练中reward model决定优化方向，eval judge帮助评价结果，task critic预测续跑价值，OPD teacher提供语言概率。四者都可能输出一个score，却有不同标签与用途。把teacher confidence当task value，或把judge文本打分当无误verifier，会把整个学习目标解释错。

本篇合并[用户清单](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)80–87、100–107的知识缺口，原有[Agent评测](#q=agent-evaluation)、[RAG定位](#q=rag-vs-finetuning)继续复用。问题驱动学习的顺序是：先定目标，再问标签怎么来、损失怎么算、误差怎样进入训练，最后看独立效果。

## 2. 前置知识 六种评价对象

| 对象 | 输出 | 监督与作用 |
| --- | --- | --- |
| Preference reward model | 回答标量偏好score | 人/模型pair排序，用于偏好或RL |
| ORM | 完整输出的结果score | 最终正确性/偏好；ORM名称需具体定义 |
| PRM | 中间step/process score | 过程标注、错误位置或续跑反馈 |
| Rule verifier | 确定规则/测试反馈 | 答案检验、环境测试；覆盖不全仍会错 |
| LLM-as-Judge | rubric score/解释/选择 | 固定prompt的生成式评价，常有偏差 |
| Critic | 条件未来return/value | MC/TD或任务结果学习，不直接判语言偏好 |

模型结构可以复用，但标签和语义不能合并。即使任务reward=0/1，也需区分合法失败、缺失评分和基础设施故障。给一个默认0不能证明已取得失败真值。

## 3. Reward model如何训练

对同prompt的偏好pair$(y^+,y^-)$，常用Bradley–Terry：

$$
\Pr(y^+\succ y^-|x)=\sigma[r_\phi(x,y^+)-r_\phi(x,y^-)],
\quad L_{\rm RM}=-\log\sigma(r^+-r^-).
$$

语言模型backbone后接标量head是常见结构；读哪个token/怎么pooling、是否共享参数、是否训练全部backbone都要看实现。Pair约束差值，不唯一决定绝对score零点；加共同常数不改pair概率。不能直接把未校准score当成功概率。

高训练pair accuracy可能来自长度、风格、格式或judge偏爱，未必区分实际正确性。数据去重、同任务公平比较、近分pair、标注分歧和任务/领域覆盖要分开统计。RM与policy更新后访问的数据也可能变化，OOD打分失准会放大reward hacking。

经典[InstructGPT](https://arxiv.org/abs/2203.02155)中的SFT、偏好RM和RL提供一条实践链，不能据此说所有PPO必须固定四个等大模型；多任务规则reward或OPD可用其他角色组合。

## 4. ORM与PRM如何给出过程信号

ORM便宜地判断终局，却难定位前面哪步错；PRM可更细，但过程单位、标签成本、正确路径多样性和可恢复错误都更复杂。数学推理step、代码工具turn、一个token不能随意当同一单位。

PRM标签可来自人工、可靠verifier或固定prefix续跑估计等；后者依赖续跑策略，不自动成为普适步骤正确性。最终失败轨迹可能有正确中间步骤，最终成功也可能含绕路和自纠正。把终局标签复制每step，是一种结果value训练方式，不等细粒度过程真值。

训练可用step classification/ranking或明确的value target；使用时可能加reward、选最佳路径、截断或过滤。不同聚合如sum/min/product改变长度与错误惩罚，不能说“有PRM就直接用每步概率相乘”。看[Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)理解原任务定义，再决定agent迁移。

你turn项目的Critic是近期数据状态成功率估计，same-version delta用于weight，Actor仍蒸馏；不是该RM的BT偏好score，也不是已训练的监督收益oracle。详见[Critic论证](#q=agentic-turn-innovation)。

## 5. Judge设计：先让评分任务可复现

Rubric明确每维度的目标、证据、范围和判定规则；固定judge checkpoint/API版本、prompt、输入格式、采样、最大长度。可输出结构化score与reason，validate字段、范围、缺失与重试。解释文本帮助审阅，却不证明score一定可信。

绝对评分利于单样本多维度跟踪，但尺度可漂移；pairwise能降低部分尺度问题，却需固定排序/平局/不可比较，比较数也增加。Position bias用A/B交换与随机排序诊断；verbosity/style、自偏爱、参考答案泄漏与prompt injection各做控制。

不要在judge输入放“这是我们新方法”的身份标签；真实teacher答案只在有意reference-based rubric下使用，否则偷看gold会改变任务。模型生成引用不等引用支持，需要段落证据检查。安全judge还应配独立规则和必要人工复核，不靠“输出免责声明”判全部行为安全。

[MT-Bench/Chatbot Arena研究](https://arxiv.org/abs/2306.05685)讨论LLM评委偏差与人类比较，但单个judge在原任务的相关性不能转移成你项目所有场景可靠。不同benchmark的单轮/多轮、人偏好/答案规则、采样预算都需说明，不能直接把排行榜数字横比。

## 6. 可靠性、校准和统计例子

Accuracy/rank correlation检不同东西；二元预测可用Brier，pairwise看一致性/胜率，score看标注者一致与rubric误差。AUROC不评价概率校准，校准好也不保证turn delta有信息。只选easy balanced测试又可能掩盖部署长尾。

$$
\mathrm{Brier}=\frac1N\sum_i(p_i-y_i)^2.
$$

若成功基率0.1，全预测0.1可得到期望Brier0.09，却没有按state分辨能力；因此和基率常数、长度/任务简单模型一起比。为critic宣称信息，应再看留出排序、校准与同trajectory变化。构造人审子集时覆盖成功/失败、短/长、领域和边界，不只随机挑几个顺眼回答。

同一题多次采样、同trajectory多个turn或同repo相似issue不是独立测试对象。分train/dev/test按任务/仓库和用途明确；反复用验证诊断可合理迭代，但不能再把该集叫一次性最终test。报告paired outcome、重复seed/置信区间与基础设施失败。

PHYRD-40公开workflow主指标是rubric mean_score，threshold pass rate辅助；不把均值提升叫pass@1提升。ESCD数学avg@8与代码avg@2、pass@k分开，[项目讲稿](#q=escd-project-interview)提供具体数值口径。

## 7. 数据、Reward hacking与闭环

Reward hack不只有模型故意作弊，优化代理目标本来就可能利用误差：冗长、重复格式、测试漏洞、引用伪造或评委prompt注入。先保留原始生成与规则反馈，比较训练reward和独立质量，再定位是目标、模型、数据还是执行器。

闭环是失败分类→证据/标注→数据版本→离线验证→训练→独立评价→受控发布。用户反馈可提供候选，不直接作无误标签；区分真实用户风险、模型能力、接口/环境故障。偏好数据需有获取和用途边界，避免把测试回答混回训练。

PRM/ORM/Judge的成本都计入预算：评价延迟/调用、数据标注、重试、人工复核与模型训练。If judge只是评测环节，不自动把它加到Actor loss。若把score进reward，须说明尺度、KL和credit；若只用于筛选，须说明保留率及分布变化。

RAG评测分检索、证据利用、引用支持、无答案拒答和端到端成本。Recall@k高但证据位置/冗余影响生成，judge可以辅助诊断但不能用它的“信任分”替实际原始来源。

## 8. 面试问答16题

### 损失接线再核一次

二元成功预测宜用logit输入BCEWithLogits等稳定形式，先sigmoid再当logit输入会重复变换。TD target一般是实数return，不能因接口叫reward就全部按二元CE。BT pair loss输入两个score差，只有差值决定pair概率；它不要求每个score都介于0和1。这三个shape/output契约在代码审查中应各写一行，避免复用同一个value head就混了监督。

### 一套可以实际执行的评价方案

先固定一组独立任务，存每个response、source/rubric版本与模型配置；同题比较不同方法时保持judge匿名、交换顺序，并记录平局、无法判断和格式失败。人审子集事先抽样，涵盖长短、成功失败、缺证据和边界，而不从新方法最好样本里挑。

输出score schema与范围检查只是第一层；第二层核对应解释是否引用了真实证据；第三层把判定与人审/可验证结果比较，并按错误类型计误拒、误接受与分歧。评委不稳定时可重复评分或多judge，但共享模型/模板的误差相关，投票数量不能自动变成独立证据。

例如两个步骤得.9/.8，直接乘.72看似“过程成功概率”，只有明确条件语义/独立或条件概率分解才可这么解释。若它们只是同一个PRM的局部正确性score，乘积通常只是一种排序函数，并会惩罚更多steps。相同最终正确率下，长推理模型可能因此吃亏，应独立比较sum/min/product及长度控制。

MT-Bench偏多轮指定题的模型评价，AlpacaEval类协议偏单轮指令回答与基线pair比较，Arena类以交互人偏好排名为重要依据；各版本有任务、长度控制、baseline和评分变化，不把不同排行榜数字当一个metric。项目报告固定具体评测协议，而不是只写“用了大模型judge”。

BT reward对score加共同常数不变，BCE成功率输出则需绝对概率解释；RM与Critic可共享backbone技术，却不能共用未经重训的head语义。Teacher token logprob是语言概率，reference KL是距离约束，仍不是二元成功预测。把这一张对象图讲清，往往比背RM/PRM缩写更有用。

最后保存有效样本数、错误/timeout、judge retries和人工复核成本。固定evaluation sample上检查某一版criterion，与在test上反复调criterion后挑最好结果是不同强度的证据。相同题多次judge不能膨胀任务N，置信区间应反映真实独立单位。

**Q01 RM如何训练？** 同prompt pair标量差做BT损失，score零点不唯一。

**Q02 ORM与PRM差别？** 完整结果与中间过程，先定义过程单位和标签。

**Q03 PRM更细就一定更好？** 可能标注噪声、路径多样与错误恢复，需独立对照。

**Q04 Verifier一定比模型可靠吗？** 规则覆盖、答案解析与测试漏洞仍会错。

**Q05 RM高accuracy为何生成差？** 数据捷径、OOD和策略利用代理误差。

**Q06 DPO完全没reward假设吗？** 没显式RM训练，但含偏好概率与隐式reward关系。

**Q07 Judge用pair还是绝对score？** 比较需求、尺度、一致性、成本和可复现rubric。

**Q08 Swap能解决所有bias吗？** 仅缓解/诊断位置偏差，不能消除长度、自偏爱等。

**Q09 Judge解释可信就score可信吗？** 不保证，需人审/规则和一致性验证。

**Q10 AUROC高就校准好？** 排序和概率校准不同，报Brier与calibration。

**Q11 Brier优于0.25就有信息？** 先比真实基率常数baseline，不用平衡集常数套不平衡任务。

**Q12 同一任务50turn可当50测试题吗？** 非独立，统计单位按任务/仓库与重复结构。

**Q13 最终失败能给所有turn“错误”标签吗？** 不能当因果过程标签；value结果监督另定义。

**Q14 Judge与task critic互换吗？** 标签和目标不同，须重新训练/校准边界。

**Q15 安全评分如何更稳？** 独立规则、风险分层、人复核、保留证据与误拒/漏检统计。

**Q16 项目怎样报告效果？** 固定checkpoint和judge协议、独立test、逐任务分布、预算与失败分类。

## 9. 来源与后续入口

主要依据上述原始论文、[ESCD公开评价workflow](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/docs/phyrd40-evaluation.md)和项目实际value定义。本文的BT/Brier小例子是数学解释，不是用户模型成绩。

RAG工程与质量链读[已有RAG笔记](#q=rag-vs-finetuning)，发布/监控/敏感信息处理读[模型服务化](#q=model-service-interview)，算法目标读[后训练40题](#q=post-training-interview)。清单的每项归属可在[合并地图](#q=post-training-checklist-map)查，不用再建一份重复RM/OPD题库。
