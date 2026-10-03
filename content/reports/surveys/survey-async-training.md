---
id: survey-async-training
title: "专题调研：异步 RL / OPD 的交付粒度、版本与缓冲如何取舍？"
type: survey
date: '2026-10-04'
directions: [infra, agent, llm]
paper_ids: [paper-asyncopd, paper-heddle, paper-tiderl, paper-areal, paper-pipelinerl, paper-april, paper-sao, paper-stride, paper-flexmarl, paper-rolloutpipe]
tags: [专题调研, Async Training, OPD, Rollout]
updated: '2026-10-04'
summary: "围绕用户指定的 11 篇论文，比较就绪边界、权重版本、慢轨迹去向与 buffer；单独区分提前 backward、跨版本训练和主动早停。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

研究问题是：长程 agent 的 rollout 主导训练时，哪些等待可以在保持更新边界的情况下解除，哪些需要接受旧数据或混合版本，哪些应直接减少交互？本专题服务于异步 OPD 的 baseline 设计和论文/面试阅读。保持训练框架笔记集中介绍 verl/slime，本页只比较用户给出的异步相关工作。

统一记录四项：慢轨迹最后在哪里、策略版本何时变化、buffer 怎样进出、速度是否转化为相同 GPU-hour 的任务收益。用户提供的 Generate 14.4% 利用率仅作场景问题，未由本次实验复测。以下选型均为根据原文的推断，没有虚构 coding pipeline 实验。

## 2. 范围、检索与覆盖边界

2026-10-03 至 10-04 按给定题名核对 arXiv、固定 v1 PDF/LaTeX 与作者链接，补查官方仓库。收录 9 篇主列表及表格中的 FlexMARL、RolloutPipe，共 11 篇，不延伸成全部 RL 算法或推理系统综述。日期是首发日期，收录日期不等于发表日期；新版摘要不能覆盖旧版表格。

10 篇满足原有五模块、方法原图、两张结果证据及原表要求，发布单篇；[DORA v1](https://arxiv.org/abs/2604.26256v1) 已完成完整报告和原图核读，但原文没有表，依仓库原表要求暂保留草稿。本页仍明确纳入其机制与固定来源，不以无关表格补证。查到官方实现的 AsyncOPD/AReaL/PipelineRL/APRIL 固定当前 commit 核读，不能等同于重跑论文当年的代码。

## 3. 论文集合与方法谱系

| 瓶颈 | 工作 | 主要动作 |
|---|---|---|
| rollout 内排队与环境等待 | [Heddle](#paper=paper-heddle) | 预测剩余长度、推进长轨迹、放置与 KV 迁移 |
| ready 数据稀疏/突发、角色不平衡 | [TideRL](#paper=paper-tiderl) | token 准入、Ref–Actor 模式与 GPU 角色弹性 |
| 已完成数据被全局屏障挡住 | [FlexMARL](#paper=paper-flexmarl)、[RolloutPipe](#paper=paper-rolloutpipe) | 提前计算 microbatch 梯度 / 完整组交付 |
| 慢轨迹与连续生产消费 | [APRIL](#paper=paper-april)、[AReaL](#paper=paper-areal) | continuation buffer / 有界异步与校正 |
| 轨迹内部版本选择 | [DORA 原文](https://arxiv.org/abs/2604.26256v1)、[PipelineRL](#paper=paper-pipelinerl) | 多旧版本并存 / 在途权重更新 |
| OPD 旧数据与教师缓存 | [AsyncOPD](#paper=paper-asyncopd) | 当前学生重算 reverse-KL 信号、IS、多动作 MC |
| GRPO 组内算法屏障 | [SAO](#paper=paper-sao) | 单 rollout + critic + token masking |
| 完整交互后才裁剪的浪费 | [STRIDE](#paper=paper-stride) | 在线早停、可信 prefix 与薄弱 turn 重启 |

这些层次可以组合，但组合后是否仍保持原有 loss 语义需要验证。独立 teacher 服务并非新的系统贡献；逐 turn 提前 backward 若要成为差异点，需与 FlexMARL 的固定版本 GA 和 RolloutPipe 的完整组交付正面对照。

## 4. 统一比较表

| 工作 / 固定版 | 交付与更新边界 | 慢轨迹与缓冲 | 模型、资源与结果口径 | 主要代价 |
|---|---|---|---|---|
| AsyncOPD · 2026-06-23 | 三阶段持续流水线，允许 stale 学生数据 | 准入 semaphore 限未消费工作；记录版本段 | Qwen3 1.7/4/8B，8 B200；主表约1.6–3.3× consumed response token/s，附录特定设置约3.8× | action 校正不解决全部 prefix 偏移；旧 top-k teacher cache 缺支持 |
| Heddle · 2026-03-30 | 优化 rollout 内调度 | 长任务优先、迁移保留缓存 | Qwen3 8/14/32B，64 Hopper；相对不同框架1.1–2.5× rollout throughput | 预测、抢占、迁移和 MP 重配置成本 |
| TideRL · 2026-08-11 | ready microbatch、Ref–Actor 两模式 | paused task + ready buffer，按 token 准入 | 32H100+1024CPU环境；文本相对同步5.6×，多模态最高6.02× consumed token/s | Ref非OPD teacher；批量与角色切换影响；原表比例口径差异 |
| DORA · 2026-04-29 | 完成数据交付，轨迹内固定原版；窗口有界 | 多版本旧任务继续、同版本KV迁移 | 32B、64/128H800；dense总token/s相对同步1.65/2.12×；4096卡6.2×是rollout | 模型副本碎片、最旧版本窗口阻塞；原文无表、报告草稿 |
| AReaL · 2025-05-30 | rollout/learner解耦、行为/近端策略分离 | 投放背压、旧数据优先；中断后保prefix重算KV | R1蒸馏1.5–32B、128–384H800主运行；v1最高2.57× effective throughput | 边界定义与staleness计数；PPO校正不能无条件迁到OPD |
| PipelineRL · 2025-09-23 | 完成轨迹交付；生成途中更新权重 | 恒定并发，ring buffer限积压 | 7B、实验段落128H100；约2×学习速度，常规对照含时间折算 | mixed-policy早期token、KV近似、估计基线 |
| APRIL · 2025-09-23 | 足够完整组后训练，下一轮恢复 | 未完成prefix入continuation buffer | 4/8B、主图8MI300；最高44% rollout throughput改善 | 仍有组内等待，环境恢复与跨版本偏移 |
| SAO · 2026-07-08 | 单prompt单轨迹可交付，训练batch仍128 | 避免等同prompt兄弟样本 | Qwen3-30B-A3B、GPU预算未完整披露；SWE Verified29.8 vs GRPO+DIS27.0 | critic成本、mask选择偏差；无统一吞吐对照 |
| STRIDE · 2026-09-13 | turn教师信号决定停止；不等于提前backward | 可信prefix上下文，弱turn重新采样 | 4/8B、8GPU；retail特定阈值2.34–3.73× step速度、排除eval | 教师低概率代理、后续覆盖与环境重建 |
| FlexMARL · 2026-02-10 | fixed参数先backward，global batch齐后统一更新 | agent级Experience Store，ready字段与梯度缓存 | 工业NPU商业多agent；7.3×相对弱MAS-RL，MA相对MARTI约1.38× | 归一化/组目标依赖、swap、保密数据 |
| RolloutPipe · 2026-06-25 | 每U完整组optimizer，R组全消费才publish | FIFO与frontier组准入，保留整轮权重 | 1.7B，8RTX4090+2A100；主窗口减少30.7–42.3% | 组内与轮末发布屏障、顺序等价需trace |

统一表展示口径，不能作为跨硬件速度排行榜。训练 token、生成 token、包含 prompt 的总 token 和完成任务数都不同；耗时减少 40% 对应约1.67×速度，不是1.4×。具体原图、表注与配置见单篇。

## 5. 共识、分歧与证据强度

**“异步”究竟改了什么？** 至少有四个独立事件：数据就绪、梯度计算、optimizer step、推理权重发布。FlexMARL 提前梯度但保留global更新；RolloutPipe保持U组的既定更新并晚发布；PipelineRL改变在途权重；AsyncOPD允许数据旧并修改loss。只说“异步训练”无法表达算法风险。

**局部重要性采样能否解决长程状态偏移？** 在给定prefix上以 current/behavior ratio重加权动作，需要支持覆盖并可能高方差；它没有把旧行为访问的prefix变成当前学生访问的prefix。AsyncOPD的MC样本是同prefix多个next-token动作，不是重新执行多条完整coding任务。token masking进一步舍弃部分梯度，也不是无损校正。

**pool是什么？**

| 对象 | 进出规则 | 是否是 replay |
|---|---|---|
| ready queue | 生成/评分完成后入，learner消费一次出 | 通常不是 |
| continuation buffer | 未完成任务保存，之后续跑再完成 | 不是重复训练完成样本 |
| prefix buffer | 历史前缀作上下文，新段重新采样 | 是否复训旧段须看loss mask |
| gradient buffer | fixed参数下保存已计算梯度，更新后清空 | 不存经验样本 |
| multi-version model pool | 保存模型副本服务旧任务 | 与sample replay不同 |
| replay pool | 完成样本多次抽样参加更新 | 是；抽样策略与概率校正需单独实验 |

**证据强度怎样分？** 原文图表支持作者设置内的结果；固定代码支持当前实现路径；本文未执行训练。没有官方仓库的7篇不编造函数与链接。数学数据、retail/telecom和SWE任务支持不同迁移范围。任何“用于你们coding会更好”都应视为可验证假设。

## 6. 场景化结论与选型

若Generate内部低利用率，先读Heddle/TideRL并测ready requests、工具等待、反复prefill与排队。若已完成可训练数据被挡住，先做FlexMARL/RolloutPipe式保守交付，明确loss是否已确定。若需要持续生产消费，先做有界队列和APRIL/AReaL baseline，保持sample ID单次消费。

用于OPD时，先列loss依赖：teacher固定量、rollout学生量、learner当前学生量，按AsyncOPD决定哪些可缓存、哪些须重算。随后比较DORA固定版本和PipelineRL混合版本；最后再考虑STRIDE主动省交互。原OPD无组内奖励依赖时，不必额外继承GRPO group completion barrier。

## 7. 空白与下一轮验证

最小对照建议：同步完整任务；固定参数提前backward且末尾一次更新；有界async单次消费；整轨迹固定版本；允许中途换权重。固定模型、teacher、GPU、数据任务与loss，分别记录版本跨度、ratio/ESS、mask率、prefix恢复、队列等待和相同GPU-hour的任务成功率。以上实验尚未执行。

逐turn提前backward还需证明：目标已确定，global归一化可提前固定，GRPO组统计等依赖已齐，参数在累积期间不变，重试不会重复累积。若最终任务奖励会修改早期目标，单独turn ready不足以证明提前计算合法。编码环境需另验文件/工具/会话snapshot，而不是只保存token。

## 8. 阅读顺序与个人理解检查

建议顺序：Heddle/TideRL定位内部等待；FlexMARL/RolloutPipe读更新边界；APRIL/AReaL学保留慢任务与背压；AsyncOPD检查蒸馏；DORA/PipelineRL比较版本路线；SAO/STRIDE检验算法依赖与主动覆盖。不是替用户登记“已经读完”。

面试自测：为什么最长任务优先有时合理？为什么buffer不一定replay？为什么单prompt单rollout仍可global batch128？为什么真实behavior logprob不能由最终checkpoint替代？为什么保prefix不意味着旧KV跨权重精确可复用？答案均应包含目标、条件和代价，而不是只报论文名。

## 9. 持续更新记录与来源

2026-10-04：新增10篇发布报告与DORA完整草稿，保留11篇机制范围。修正RolloutPipe“U组更新/R组发布”的区别、STRIDE +1.3%相对mean与百分点口径、TideRL原表百分比差异、PipelineRL基线时间折算；未声称训练复现。

原文：[AsyncOPD](https://arxiv.org/abs/2606.24143v1)、[Heddle](https://arxiv.org/abs/2603.28101v1)、[TideRL](https://arxiv.org/abs/2608.10402v1)、[DORA](https://arxiv.org/abs/2604.26256v1)、[AReaL](https://arxiv.org/abs/2505.24298v1)、[PipelineRL](https://arxiv.org/abs/2509.19128v1)、[APRIL](https://arxiv.org/abs/2509.18521v1)、[SAO](https://arxiv.org/abs/2607.07508v1)、[STRIDE](https://arxiv.org/abs/2609.14636v1)、[FlexMARL](https://arxiv.org/abs/2602.09578v1)、[RolloutPipe](https://arxiv.org/abs/2606.26997v1)。原图出处、作者/单位、版本和官方代码commit见单篇证据清单。
