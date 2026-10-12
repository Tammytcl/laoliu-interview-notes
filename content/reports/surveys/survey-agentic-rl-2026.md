---
id: survey-agentic-rl-2026
title: "Agentic RL专题：从Actor–critic到环境、信用、探索与训练系统"
type: survey
date: "2026-10-12"
directions: [agent, llm, infra]
paper_ids: [paper-ppo, paper-grpo, paper-dapo, paper-gdpo, paper-search-r1, paper-gigpo, paper-agent-lightning, paper-areal, paper-sao, paper-heddle, paper-tiderl]
tags: [专题调研, Agentic RL, Actor Critic, PPO, Credit Assignment, Pipeline]
updated: "2026-10-12"
summary: "按环境与反馈、信用分配、探索、数据接口和异步系统分层比较Agentic RL，接六篇教学主线与原文精读；不跨论文硬排成绩。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

Agentic RL优化的是与环境交互的策略，不只是单次回答。一个coding任务要真正修改文件并执行测试，一个search任务要多轮查询与读证据。研究问题分别是：如何定义可训练轨迹、如何把最终结果分到动作、如何探索有效分支、如何组织昂贵环境与模型计算。这些层可互补，不能把所有论文都理解成另一个PPO loss。

本专题用于授课、代码阅读、实验设计和面试。先学基础估计与优化，再比较agent工作；训推框架、GPU和模型格式仍在各自Infra文章，不重复扩成工具产品大全。

## 2. 范围、检索与覆盖边界

截至2026-10-12，核心精读为Search-R1 v5、GiGPO v3、Agent Lightning 2025 v1，结合已收录PPO/GRPO/DAPO及异步论文。补读ArCHer、RAGEN、ARPO、THUDM AgentRL、Practitioner’s Guide的原文相关方法，以及DeepSWE作者博客；补充导读不表示这些作品都完成同标准单篇精读。

固定代码与固定论文分别注明。Lightning当前2026代码不当2025实验源码；THUDM AgentRL与同名库区分；2026 Lightning v1.0不混为2025 arXiv v1。没有重新训练模型，不声称穷尽所有agentic工作，也不将摘要数字列成同预算排行榜。

## 3. 前置路线 从Actor–critic开始

[Actor–critic](#q=rl-actor-critic)先推策略梯度与baseline，[MC/TD/GAE](#q=rl-returns-gae)处理信用，[PPO完整训练](#q=rl-ppo-training)解释采样、冻结、current forward、clip与发布，[统一算法对照](#q=rl-policy-optimization-map)比较RLOO/GRPO/GSPO/DAPO/SAPO/CISPO等修改位置。

$$
J(\theta)=\mathbb E_\tau\sum_t\gamma^tr_t,\quad
\nabla J=\mathbb E_\tau\sum_t\gamma^tG_t\nabla\log\pi_\theta(a_t|h_t).
$$

这里环境不可微，学习通过已采样动作的score函数；不是对工具返回文字直接backward。Critic、reference、teacher、behavior各自有职责。时间单位可以token或turn，但reward、折扣、mask与概率分解要一致。终止与可bootstrap截断需分开，reset后trace不得跨episode。

## 4. 统一方法谱系

| 层 | 工作与原始入口 | 本轮核读的主要对象 | 不能自动推出 |
| --- | --- | --- | --- |
| 工具执行+规则反馈 | [Search-R1](#paper=paper-search-r1) | 搜索、观察、答案EM与loss mask | 互联网时效/引用真实、所有PPO优于GRPO |
| 轨迹与行为崩溃 | [RAGEN](https://arxiv.org/abs/2504.20073v2) | StarPO整轨迹优化、Echo Trap诊断 | 原文把PPO举为单轮用法，不意味着PPO理论只能单轮 |
| 两级相对信用 | [GiGPO](#paper=paper-gigpo) | episode组+重复anchor后续return | 相同观察必相同隐藏状态、step立即可训练 |
| 分层critic | [ArCHer](https://arxiv.org/abs/2402.19446v1) | utterance级Q/V与token级policy | “hierarchical”均指同一种算法 |
| 工具后探索 | [ARPO](https://arxiv.org/abs/2507.19849v1) | 观察后熵上升、global/partial采样与优势归因 | 高熵一定错误、复制branch只需token无需环境恢复 |
| 软件任务RL | [DeepSWE作者介绍](https://www.together.ai/blog/deepswe) | rLLM与Qwen3-32B的coding训练/评测路线 | pass@16可以当pass@1、全部收益来自一个算法 |
| 接入与trace转换 | [Agent Lightning](#paper=paper-agent-lightning) | 每调用样本、角色选择、训练执行解耦 | Trace自动具备全部行为概率、终局赋值等于细因果信用 |
| 多任务与系统 | [THUDM AgentRL](https://arxiv.org/abs/2510.04206v1) | function-call接口、异步、cross-policy采样与task归一 | 仅scheduler贡献或所有任务组合相同效果 |
| 配方审计 | [Practitioner’s Guide](https://arxiv.org/abs/2510.01132v2) | environment/reward/policy三轴、SFT:RL预算 | 一种最优比率普适 |
| 有界异步 | [AReaL](#paper=paper-areal) | behavior/proximal分离、背压与partial | PPO校正可无条件搬到OPD |
| 去组等待 | [SAO](#paper=paper-sao) | 单rollout、value优势、概率比token mask | 无criticGRPO任意改单样本后仍同目标 |
| 任务调度 | [Heddle](#paper=paper-heddle)、[TideRL](#paper=paper-tiderl) | 长任务排队、就绪队列与弹性资源 | rollout吞吐等最终任务质量或训练goodput |

ArCHer高层从在线交互replay训练Q/V，Bellman目标；低层token策略用高层估值，二者时间尺度不同。Agent Lightning原始v1基本分配终局return到调用，结构分层不等于同样的critic。GiGPO则利用同任务已采样重复观察，省新rollout但依赖future return。三者应按信用信号、数据复用、状态定义比较。

## 5. 一条真实轨迹怎样接到优化

执行记录包含task/attempt/request ID、原prompt、student tokens、behavior logprob与版本、tool input/output、done/truncation和最终验证。转换训练时保持actual input，不用重写后的摘要冒充原采样条件。观察可见但通常不计actor loss；query/action/thought的监督范围按算法明确。

先构造return/优势，再形成current/behavior ratio和surrogate，最后global reduction。共享reward重复广播给各turn后，逐turn均值、token均值、episode均值对长任务权重不同；组统计去重不自动解决loss权重。Lightning当前源码示例分别处理rollout代表行与token-count归一，正适合逐层检查。

环境reward有正确性、部分进展、成本和格式；任意加分不自动potential shaping。密集反馈可能加速也改变目标或被hack。Verifier与LLM judge需固定版本，工具报错有时是正常可恢复观察，系统timeout则可能是无效数据。RL训练可包含失败轨迹，删除所有失败会改变访问分布。

## 6. 实验与预算怎样统一

| 比较轴 | 必须固定或报告 |
| --- | --- |
| 学习起点 | checkpoint、SFT数据和预算、模型/模板/tokenizer |
| 任务 | 独立train/dev/test、任务难度、环境版本/镜像 |
| 行为预算 | rollout数、group、turn/token、工具并发、参考帮助 |
| 优化 | LR独立调优、advantage/reduction、epochs与有效batch |
| 系统 | 每角色卡数、TP/CP、总预留卡时、初始化/保存 |
| 评价 | pass@1/avg@k/pass@k、成功率/score、seed和模型选择 |
| 诊断 | 无效动作、失败类型、状态覆盖、长度、熵与reward方差 |

同step不等同生成token，也不同GPU-hour。GiGPO主表8组轨迹和PPO128独立环境拥有相同总环境数，但任务多样性不同；Search-R1的不同优化器checkpoint选择也有稳定性处理。不要用两张各自调配方的表认定某个loss本质更优。

Coding必须验证生成patch可执行与测试真实；pass@k不同于单次在线体验。QA EM不直接测证据忠实度；混合F1/format reward不等全任务成功率。评测集小则逐题outcome与置信区间比一个均值更重要。

## 7. Infra和Agentic OPD的连接

生成链受环境等待、KV、prefill、长尾影响，训练链受目标就绪、collective和全局batch影响。OPD将教师概率加入评分链，仍需保留真实环境状态。先建同步正确baseline，再改连续队列、teacher独立评分、请求边界权重同步与可解释校正。

详细数据契约见[Pipeline专题](#q=agentic-opd-pipeline)，蒸馏方法见[Agentic OPD综述](#report=survey-agentic-opd-2026)。增加并发、提前评分、提前backward累积、提前optimizer update是四个实验，不统称异步收益。有效训练样本/卡时与同预算独立任务收益应一起看。

## 8. 授课、面试和实验顺序

第一讲基础梯度与GAE；第二讲PPO batch与clip分支；第三讲action/observation真实轨迹；第四讲Search-R1与GiGPO；第五讲ArCHer/Lightning不同分层；第六讲探索、多任务与有界系统。共享[CPU练习](./assets/learning/agent-training-lab.py)核手算与梯度，未训练模型。

面试要能回答：工具后梯度为何还可训练、critic估什么、group优势何时就绪、观察与loss mask怎区分、长episode权重为何增加、吞吐是否改变任务分布。具体题目见[OPD基础](#q=opd-interview-foundations)与[项目答辩](#q=opd-project-defense)，不要把自拟题标成公司原题。

最小对照保持执行器、任务、起点和预算，先验证baseline，再比较信用方法，最后调系统。每次仅改一个可解释轴，失败案例和预算保留，不因调度复杂就跳过独立质量评测。

## 9. 来源、更新与待核范围

新核心三篇带固定原图、表格、源码与8道问答；补读工作主要用于定位方法轴，尚未全做单篇图表审计。性能不跨论文硬排，正式实验结论回各自原文。GSPO/SAPO按现有无原表规则保留完整草稿，不因本次扩充更改发布门槛。

2026-10-12接入六篇RL/OPD教学、三篇Agentic RL精读与系统专题，补独立方法比较和项目问答。后续优先做RAGEN/ArCHer/ARPO的独立精读；这不影响本轮已经核读并标范围的导读使用。
