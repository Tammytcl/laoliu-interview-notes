---
id: escd-project-interview
title: "我的ESCD工作：面试介绍、方法推导、实验与关键追问"
category: behavior
difficulty: 深入
tags: ["P0", "ESCD", "Project", "Cross Tokenizer", "OPD", "Interview"]
updated: "2026-10-13"
summary: "以作者项目讲解组织30秒/3分钟/10分钟介绍，完整解释event entry/completion、root FKL、集合梯度、公开实现、主实验与诊断边界，附16道答辩。"
draft: false
---

## 1. 问题背景 这篇工作到底补了哪个缺口

论文是 Beyond Token Alignment: Event Completion for Cross-Tokenizer On-Policy Distillation，方法称Event-Set Completion Distillation，简称ESCD。面试不应只说“做了跨tokenizer对齐”，因为已有方法已经对齐位置、词表或span。更具体的问题是：学生只生成了教师token的一部分后，教师规定了剩余内容，却没有规定不同合法student补全token之间怎样分配概率。把分词差异硬编码为单个token目标，会引入教师并未提供的偏好。

本篇按你作为作者讲解自己的工作组织。技术、图表与成绩使用[公开repo固定报告](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/ESCD.pdf)和同commit代码；该报告45页，README注明晚于[arXiv v1](https://arxiv.org/abs/2609.34738v1)。既有[论文精读](#paper=paper-opd-2609-34738)仍保留v1图表，本篇不会把两版本方法措辞与附录编号混用。

公开报告标注你为共同一作及项目负责人；具体模块分工仍按实际贡献填写。下面用“我们”表达工作成果，不把每项团队实现和表格结果都归给单个人。

## 2. 前置知识 一分钟建立三个对象

Student从自己策略生成，teacher在对应上下文给监督。不同tokenizer下，位置不匹配和候选事件不匹配是两件事。**Root**是共享byte前缀处的下一动作，**child**是学生输出部分bytes后的真实访问状态，**completion set**是在child能一步完成代表残余bytes的合法student token集合。

Teacher概率按其原词表，student概率按其native完整词表；同字符串不等于相同ID。EOS是停止事件，不普通内容bytes。把词表概率重新归一到completion成员会令集合概率永远1，丢掉“能否完成”监督。

## 3. 可以直接练的三种项目介绍

### 30秒版本

“我的工作研究跨tokenizer OPD中的事件完成问题。学生生成教师token的一部分后，多个student token都可能完成剩余内容，但逐token监督可能强行规定它们的概率拆分。我们提出ESCD：root做公共事件的概率对齐，child只监督合法completion集合总概率，复用学生自然访问的状态，不增加额外rollout或修改词表。我们在数学、代码和大规模异构MoE上验证，并用局部梯度与可完成质量分析解释机制。”

### 3分钟版本

“背景是把不同家族教师的能力迁移给学生。OPD在学生prefix上学习，但词表不一样，直接逐ID KL没有共同事件。已有span/byte方法解决很多对齐问题，我关注的是学生进入一个事件但尚未完成时，怎样给出更合理监督。

例如教师候选Bread和Breads，学生先输出Br，剩余代表前缀ead。Student下一token可以是ead或eads，二者都满足这次残余前缀要求；教师未必要求二者均分概率。ESCD把相关teacher候选合组，保留其总概率作为权重，对student合法集合概率取负对数，同时保留root forward KL。这样学的是完成要求，而不是教师分词习惯。

实现上，我们用公共bytes、teacher full-vocab质量和自然访问child构建目标，冻结teacher，student保持梯度，logsumexp稳定计算；有EOS、空集、未支持质量和分片位置诊断。实验中固定student与teacher配对比较GOLD、X-Token、SimCT、BPM，同时披露适配与初始化条件。数学用avg@8/pass@8，代码avg@2/pass@2；大规模Kimi组依赖共同SFT起点。

局部分析把root、single-completion与集合监督在固定预测上比较，并量一步补全集合可用性。我们没有声称一步集合是任意tokenizer的精确完整文本分布，也不把这项方法当成解决所有OPD崩溃的保证。”

### 10分钟六页讲法

页1画不同分词和OPD状态；页2用Bread/Br解释root与child；页3合组、集合loss、梯度与Single分解；页4代码链与teacher hidden传输；页5主表、MoE同起点和COUF/覆盖诊断；页6局限、个人负责模块与下一步。

演示页2时先遮住公式问听众“teacher给的是ead还是eads的偏好？”再引出集合。页5分开任务结果与机制诊断，不能把局部COUF直接写成模型准确率。

## 4. 原图：event entry与event completion

![ESCD公开原图 · Root对齐、自然访问child与集合监督](./assets/learning/escd-project-interview/method-public.png)

**原图解读。** 左侧teacher Bread/Breads/Break的质量映到student Br，学生采Br后访问$x'=(x,\mathrm{Br})$。右上逐token条件目标会分别给ead/eads/eak概率，右下把Bread/Breads合组，监督ead与eads的总概率，Break组监督eak。$M_1=q_1+q_2$是root teacher质量，$P_1=p(\mathrm{ead}|x')+p(\mathrm{eads}|x')$是child student集合概率，二者职责不同。图源为[固定公开资产assets/method.png](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/assets/method.png)，对应报告Figure3，不是新绘实验图。

注意eads完成Bread这个共同约束，但后续语义并非永远与ead完全相同；方法有意放宽到代表前缀。图中“合法”是bytes事件要求，不保证整条任务正确。

## 5. 方法从集合定义推到loss

### Teacher合组：保留质量，放宽共同约束

公共实现使用完整teacher候选词表，取没有精确单student内容token counterpart的候选。按byte-prefix包含关系连接并形成components，代表$y_g$取最短成员，权重$M_g=\sum_{v\in V_g}q_T(v|x)$。

Bread/Breads组保留两者质量，却只要求共同Bread，不再要求额外s；这是一项可解释的约束coarsening，不是证明两个教师事件完全等价。冻结teacher质量，不对eligible groups重新归一。

若student采样$a$的bytes是$y_g$非空严格前缀，残余$r_g=y_g[|b_S(a)|:]$，合法一步集合定义为：

$$
C(r_g)=\{u\in V_S^{\rm cont}:r_g\preceq b_S(u)\},\qquad
P_g=\sum_{u\in C(r_g)}p_\theta(u|x,a).
$$

允许token恰好完成或跨过残余，native概率不在集合内重归一。空集合不产生child项，仍需统计未覆盖质量。不同eligible component的代表前缀互不包含，相应completion集合按定义分开。

### Root FKL与child事件负对数

Root复用BPM的byte alignment/projection：映射质量到student cells与complement，两侧剩余质量都保留，计算FKL。当前报告明确不保留BPM完整interior/spanning supervision，不能简单称“原BPM所有位置loss再加child”。

$$
\ell_{\rm root}=\mathrm{KL}(\bar q_x\|\bar p_{\theta,x}),\qquad
\ell_{\rm child}=-\sum_{g\in G(x,a)}M_g\log P_g,\qquad
L_{\rm ESCD}=L_{\rm root}+L_{\rm child}.
$$

Root与child放到各自预测位置，沿统一有效response token reduction。Teacher mass是child权重，不是令$P_g=M_g$的MSE目标；没有“把teacher概率均分给每个合法ID”。Full student词表softmax的分母参与梯度，集合外概率因此也被影响。

### 为什么保留集合优于指定一个member：分解而非口号

若指定$u^\star\in C_g$，有：

$$
-M_g\log p(u^\star)
=-M_g\log P_g-M_g\log\frac{p(u^\star)}{P_g}.
$$

第一项提高集合总质量；第二项要求集合内部更偏爱指定member。ESCD只保留第一项，去掉缺乏teacher依据的tokenizer-dependent分配偏好。这是核心讲解，不能把实验提升全归“覆盖更多位置”。

单事件对student logit$z_j$的梯度为：

$$
\partial L_C/\partial z_j=M_g[p_j-\mathbf1(j\in C)p_j/P_C].
$$

例如$p=(.1,.2,.4,.3)$、$C=\{1,2\}$零基、$M=.7$，$P_C=.6$，梯度$(.07,-.09333,-.18667,.21)$。集合内增概率、集合外减少；不会强制.2与.4均分。已有[CPU练习](./assets/learning/agent-training-lab.py)核了这组梯度，不是GPU训练复现。

## 6. 真实代码和Infra链

固定公开commit为befd0c0a582ec6efc055660467db95d522519591。入口不是根据通用sampled字段猜loss，而是后端compute_cross_tokenizer_opd_token_losses分派`escd_full_vocab`：

| 代码位置 | 应解释什么 |
| --- | --- |
| `tokenizer_bytes.py` | 原生bytes、特殊token与artifact；不是单token decode猜偏移 |
| `escd_core.py` | prefix components、残余与completion索引 |
| `escd_full_vocab.py` | full teacher probabilities、质量校验、root projection、global/local child |
| `escd.py::_score_child_targets` | selected student概率、分桶、事件logsumexp |
| `sglang_teacher.py` / `teacher_head.py` | causal hidden传输、冻结LM head重建分布与请求身份 |

核心原代码节选：

```python
# Official excerpt: escd.py::_score_child_targets, per-event loop
event_log_probability = torch.logsumexp(
    log_probs[row_index].index_select(0, selected), dim=0
)
row_loss = row_loss - float(mass) * event_log_probability
```

见[固定源码](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/opd/cross_tokenizer/escd.py#L320)。Teacher不反传，student的selected logprob来自完整归一化。索引用startswith寻找完成/跨过残余的全部IDs，不只完全相等字面；不同ID同bytes也不能漏。

Teacher causal hidden+冻结LM head让learner分块重建分布，减少全$[L,V]$传输，但仍付teacher forward/head计算和驻留成本。自回归shift、teacher重编码长度、request ID、模型版本、dtype与CP/TP全局位置是数据契约；CPU玩具不能证明全部多rank正确。

生成top-k、teacher candidate_top_k、X-Token projection_top_k分别控制采样、监督支持和投影，公共candidate=0表示full；不能统一设同数值当公平。EOS映射停止事件，truncation不伪装EOS；whitespace mask用于避免某些切分信号误导，实际空白仍保留在模型上下文。

公开release不包含大规模/Agentic训练全部recipe与历史checkpoint来源。代码和方法适配见[methods](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/docs/methods.md)，复现范围见[reproduction](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/docs/reproduction.md)。

## 7. 成绩怎么讲，机制分析怎么讲

### 配置与计算账本：按同一列读

公开报告Table6小规模是Slime/Megatron-LM/SGLang，20K DAPO-Math+TACO prompts，global rollout/train batch16、每prompt1响应、microbatch1；prompt5120、response27648、sequence33280，T1/top-p.95/top-k20，LR5e-7、WD.1，训练TP2/PP2/CP8。Teacher16卡、student训练32、rollout16，总64，独立评价卡不含在此总数。

两组大规模用17577条SciDeriv，global batch仍16、LR1e-6；397B组prompt4096/response27648/sequence33280，1T组response80000/sequence84112。Teacher32、student训练64、rollout32，总128；大规模recipe不在公开release完整覆盖内。模型总参数/active参数、三角色卡数、trainable states与实际GPU-hour分别说，不能仅用“1T→35B”推导费用或速度。

![Table1公开原表 · 两teacher配对的数学和代码结果](./assets/learning/escd-project-interview/table1-public.png)

**原表解读。** Student均为Qwen3.5-2B，上半Qwen3-32B teacher，下半GLM-Z1-9B。数学k8、代码k2，avg与pass分列；应比较同teacher组。Qwen组LCB avg@2 BPM15.4/ESCD31.0，差15.6个百分点；但pass@2的最佳baseline是X-Token18.1，ESCD42.9差24.8，不能拿BPM15.9算另一种“相对最佳”差值。图源[固定assets/table1.png](https://github.com/Tammytcl/ESCD/blob/befd0c0a582ec6efc055660467db95d522519591/assets/table1.png)，为论文报告结果，非本轮独立复现。

可以选三组成绩讲而不背全表：Qwen组AIME24 avg@8 38.8→46.7（与该列最佳BPM比）；LCB avg@2 15.4→31.0；GLM组LCB avg@2 22.0→28.3。按“模型对+指标+k+baseline+差值”完整说。某些pass列持平，不能概括所有格严格更高。

大MoE表：397B teacher→30B-A3B student在FrontierScience BPM48/ESCD51、PHYRD46.6/49.6；1T Kimi→35B-A3B使用共同SFT起点，FrontierScience SFT68/BPM69/ESCD72，PHYRD66.2/69.2/74.3。应说ESCD对BPM的同起点差，不把SFT起点提升也全归ESCD；总参数规模不是active计算量。

**局部诊断不是又一轮模型训练。** 报告固定student predictions、teacher events/mass和root loss，比Root、Single、Event与指定byte-event reference的梯度。COUF为$1-\sum_x\|z_x-z_x^\star\|^2/\sum_x\|z_x^\star\|^2$，Qwen组.7383/.4323/.8495，GLM组.8848/.7493/.9341。它结合方向与幅度，可能为负；不能叫纯cosine或任务成功率，也不证明所有真实更新改善任务。

**覆盖的两个分母。** 观察到可分支teacher mass中自然访问质量约81.83/83.29%；在相应visited compatible mass里，一步完成为99.01/99.43%，deeper .99/.57%。“超过99%”有条件分母，不等全词表、全部student token或任意语言都99%。这支持depth-one在研究tokenizer pair的可用性，不是无限事件概率精确求解。

## 8. 面试问答与反驳练习

**Q01 为什么不是另一个token对齐？** 已有对齐解决entry，本文研究部分进入后的completion约束与集合内部任意拆分。

**Q02 为什么用集合而非均匀soft label？** Teacher没有提供集合内部均分偏好；均匀标签加了方法不需要的约束。

**Q03 Child loss把概率拟合到M吗？** M为NLL权重，$-\!M\log P$不是$(P-M)^2$。

**Q04 Bread/Breads合组损失什么？** 放宽到共同Bread，保留质量但不再要求额外s；需解释coarsening。

**Q05 集合包含eads安全吗？** 满足代表残余bytes前缀，不保证完整任务语义等价；后续策略和反馈仍重要。

**Q06 Native full softmax为什么重要？** 集合内重归一会令P=1、丢失集合外竞争和completion监督。

**Q07 需要生成额外child吗？** 利用student自然访问状态，查询其他token概率不等额外rollout。

**Q08 空集说明模型不会吗？** 只说明该一步词表没有定义的合法completion，不能当完整多步能力判断。

**Q09 Root就是全部BPM loss吗？** 当前报告复用root，不保留BPM完整interior/spanning，因此要披露目标差异。

**Q10 99%是哪部分？** Visited compatible mass中的一步覆盖，非所有tokens、所有bytes。

**Q11 COUF证明accuracy吗？** 固定局部参考梯度保真诊断，需主训练/测试另支持。

**Q12 1T组是不是直接OPD稳定？** 报告Kimi组共同SFT初始化，方法不保证解决所有崩溃。

**Q13 X-Token原法完全复现了吗？** 公共H-KL、固定projection与alignment适配，不能泛称原P-KL全复刻。

**Q14 Hidden transport省掉了什么？** 减少某类分布传输与中间驻留；没有省所有teacher/head计算。

**Q15 如何证明个人贡献？** 公开作者/负责人身份之外，再说具体问题提出、目标推导、实现或实验分工，附真实证据。

**Q16 最大局限和下一步？** 一步完成近似、代表约束coarsening、tokenizer假设、EOS与初始化；进一步做不同语言/分词距离、同预算与多步事件控制实验。

## 9. 个人介绍的证据卡

可确认的团队工作：ESCD方法、公开代码、原报告表格、SciDeriv/PHYRD-40及科学推理扩展。个人负责的模块、关键决策、失败debug和实测预算请按你的分工补一行，不从“项目负责人”推成全部独立完成。面试可先使用本篇完整技术讲稿，再接一个自己的真实例子。

补充路线：[CT 24道代码题](#q=opd-interview-cross-tokenizer)、[后训练40题](#q=post-training-interview)、[turn实现](#q=agentic-turn-framework)。本轮只静态核读公开代码、报告与原资产，没有启动模型训练；图表与成果按公开报告署名。
