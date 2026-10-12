---
id: survey-agentic-opd-2026
title: "Agentic OPD专题：教师指导、状态访问、异步流水线与跨词表"
type: survey
date: "2026-10-12"
directions: [agent, llm, infra]
paper_ids: [paper-guided-opd, paper-uopd, paper-actfirst-opd, paper-asyncopd, paper-stride, paper-rethink-mopd, paper-opd-2306-13649, paper-opd-2306-08543, paper-opd-2609-34738, paper-opd-2607-22334, paper-opd-2608-24696]
tags: [专题调研, Agentic OPD, Async OPD, ESCD, Cross Tokenizer, Interview]
updated: "2026-10-12"
summary: "按proposal/execute/train三角色与动作/prefix两种过时对照Guided、UOPD、ActFirst、STRIDE、AsyncOPD，补强SFT审计和跨tokenizer项目接线。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

多轮蒸馏不仅要匹配下一token，还要处理错误动作导致的环境状态、长trajectory计算和旧策略。先问学生提案由谁生成、环境执行谁、训练监督哪段响应，再问teacher是否改变状态访问、是否提前减少工作、是否只是并行。相同名字OPD下可能是不同目标与访问分布。

本专题对应你的Agentic OPD与Cross-tokenizer OPD学习/展示路线，详细基础见[教学正文](#q=agentic-opd-training)，真实系统见[Pipeline](#q=agentic-opd-pipeline)。不把研究假设或本地设计记录写成已跑出的项目成绩。

## 2. 范围、检索与版本边界

核心新精读为Guided-OPD 2606.15912v2、UOPD 2609.34036v1、ActFirst 2609.36608v1、新多教师审计2610.04272v1，接已精读AsyncOPD/STRIDE及35篇OPD库。MAD-OPD原文作为补充定位，未完成独立全图表报告。

截止2026-10-12。Guided公开入口404，ActFirst无可确认方法仓库，新审计代码仍声明待公开，本轮按原文解释并标教学伪代码；UOPD/ESCD有固定公开commit，静态源码与模型复现分开。ESCD当前repo报告晚于arXiv v1，原图表继续固定v1，不偷换成最新PDF数值。

## 3. 前置知识 局部目标与状态分布

$$
D_F(q\|p_\theta)=\sum_aq(a|h)\log\frac{q(a|h)}{p_\theta(a|h)},\quad
D_R(p_\theta\|q)=\sum_ap_\theta(a|h)\log\frac{p_\theta(a|h)}{q(a|h)}.
$$

局部目标定义固定prefix上的动作分布；trajectory访问分布由真实执行过程决定。FKL教师加权、RKL学生加权，二者均受访问prefix影响。直接词表RKL需$p$保留梯度，sampled标量无偏值不自动给正确autograd梯度。教师SFT在$q$采样下对应forward cross-entropy条件期望，但有限样本、过滤与长度权重不同。

Teacher的概率不是任务真值，局部稠密反馈可能保留错误推理惯性。环境verifier是另一层证据，可以设计混合/门控，但需对应loss和尺度消融。

## 4. 一张机制比较表

| 工作 | 谁proposal/execute/train | 改哪层 | 首要边界 |
| --- | --- | --- | --- |
| [Guided-OPD](#paper=paper-guided-opd) | 每turn随机teacher或student，实际执行所选者；对应FKL/RKL | curriculum与访问分布 | 混合历史不等纯轨迹二选一，指数零端点需核 |
| [UOPD](#paper=paper-uopd) | student proposal；候选gate触发teacher执行/SFT，否则student执行/PG | 介入位置与双loss | proposal-dependent gate不等固定mixture，理论recoverability需条件 |
| [ActFirst](#paper=paper-actfirst-opd) | fast学生动作执行；full学生响应用于训练，不执行其末尾action | 关键路径与并发 | auxiliary状态分布、参考成本、低并发可能更慢 |
| [STRIDE](#paper=paper-stride) | 学生多轮、teacher turn信号选择停/前缀再启 | 主动减少交互与前缀复用 | teacher低分不等错误，恢复需环境状态 |
| [AsyncOPD](#paper=paper-asyncopd) | 旧学生持续rollout，teacher持续评分，current learner消费 | 三阶段stream与动作校正 | IS不自动纠正prefix状态占用，top-k漏支持 |
| [OPDVR](#paper=paper-opd-2608-24696) | student rollout，verifier与teacher信号门控 | 信号选择 | verifier错误和稀疏成功改变可用反馈 |
| [MAD-OPD](https://arxiv.org/abs/2605.01347v1) | 多agent讨论相关蒸馏设计 | 反馈/训练配方扩展 | 本轮仅补读定位，不泛化到任意工具agent |
| [新MOPD审计](#paper=paper-rethink-mopd) | 相同提示集比较SFT/soft/teacher-history/student-env | 基线、公平性与成本 | 主设置共同初始化/词表；复比较验证口径不同 |

Teacher参数相差也很大：Guided用Qwen3-30B-A3B基础teacher，UOPD用任务GiGPO/Search-R1 teacher，ActFirst用任务GiGPO Qwen3-8B。不同student、环境划分和训练资源使论文主表不能互相排名。

## 5. Teacher干预怎样改变实际执行

Guided独立选择role，$z_t\sim\mathrm{Bernoulli}(\beta)$，共享历史连续产生$d_{\mu}$，不是两个纯策略轨迹分布的线性混合。UOPD先采学生$a_S$再判$I(h,a_S)$，实际分布：

$$
\mu_G(a|h)=p_S(a|h)(1-I(h,a))+\mathbb E_{b\sim p_S}I(h,b)\,q_T(a|h).
$$

后者筛掉学生易触发的候选，连未干预样本也有选择偏差。对assisted rollout回报的条件分析不等学生SGD回报单调。Teacher执行改善状态，teacher SFT提供样本，两者要独立消融。

ActFirst用参考下一观察帮助fast动作，full响应只看真实历史，不泄漏未来参考；二者action可不同。失配后关闭指导，用NAP继续，转移一致性不是逐字动作相同。说明三个角色是避免误讲机制的最好方法。

## 6. Pipeline、buffer和权重版本

队列可能保存prompt、未完token+环境、已评分样本、batch或历史replay；存什么决定是否可续跑、是否重用梯度。APRIL保留未完成轨迹不自动训练未完成片段；STRIDE前缀重启需要相应环境恢复；ActFirst FIFO消费一次而非replay。

系统需保留actual prompt、tokens、behavior logprob与逐段version；teacher cache条件和candidate支持也记录。AsyncOPD在learner重算当前student-dependent信号、用old→current动作IS、多动作MC降低方差；这些修正针对固定prefix，不把已经做错的工具状态改成新策略状态。

权重选择可以trajectory固定、请求边界切换或in-flight token切换。DORA/PipelineRL/AReaL对应不同路线，不能说都是相同partial。发布gate要确认请求drain，KV随权重变化一般需重建；TP训练/推理分片映射和optimizer checkpoint不同。详细12道系统问答在[Pipeline](#q=agentic-opd-pipeline)。

## 7. 跨Tokenizer OPD怎样接这条链

工具历史先按真实可见内容固定，再对teacher重编码。位置对齐寻找共同byte prefix，词表对齐建立公共概率事件；不是token ID zip，也不是直接拼同维度logits。ESCD公开实现teacher causal hidden+frozen head重建分布、root FKL及自然访问child completion-set监督，候选full支持与生成top-k分开。

Shared/span/byte/event路线详见[已有OPD综述](#report=survey-opd-llm-2026)。Cross-tokenizer本身不解决多轮环境credit；异步又可能让student-dependent对齐/支持缓存过时，需按artifact与目标依赖核对。EOS没有普通bytes，要按stop语义，max-length截断不假装EOS。

项目源码题见[24道CT问答](#q=opd-interview-cross-tokenizer)。公开基线适配、历史checkpoint来源、CPU检查与多GPU验证各自标范围，不能用release模板充当每项原实验运行清单。

## 8. 同预算实验与面试展示

先同步真实学生OPD正确性，再逐项加continuous queue、提前teacher scoring、早backward固定update、optimizer异步；每次同时报告任务覆盖、年龄、重试、总GPU-hour与独立任务质量。对比较teacher介入的方法，再报实际率、teacher生成/评分次数、状态恢复与curriculum退出后的能力。

强SFT使用成功teacher语料、每assistant turn监督、独立LR搜索，验证/test与teacher训练分离。新审计的14.8–23.1×是full-run训练GPU-hours，排除一次语料生成、评测和其他搜索，不当项目总价。Cross-tokenizer再比较同teacher/student、同起点、相同事件支持与候选预算。

面试从[24道基础](#q=opd-interview-foundations)到[24道CT](#q=opd-interview-cross-tokenizer)再到[24道项目答辩](#q=opd-project-defense)，有30秒/3分钟/10分钟讲法与证据表。公开题源仅提供问题方向，自拟问题明确标注，个人项目结果按真实日志填。

## 9. 结论、来源与更新

把“谁执行改变状态”“谁提供监督”“怎样并行”“怎样减少工作”分开，是整理Agentic OPD的主线。原文加速可能指rollout、训练wall-clock或goodput，需要对应范围；任何流畅叙述都不能替代代码数据与实验条件。

2026-10-12新增三篇Agentic OPD精读、新多教师审计、系统接线和三份面试/展示文章。原图表版权归作者，CPU练习只核小张量与梯度；没有重新运行论文训练或给用户项目补造成绩。
