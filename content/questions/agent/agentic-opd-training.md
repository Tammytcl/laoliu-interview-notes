---
id: agentic-opd-training
title: "Agentic OPD教师干预状态分布与计算复用"
category: agent
difficulty: 深入
tags: ["P1", "Agentic OPD", "Distillation", "Guided-OPD", "UOPD", "ActFirst-OPD", "STRIDE"]
updated: "2026-10-12"
summary: "推导条件KL与sampled梯度，比较随机指导、uncertainty干预、早停重启、动作优先和异步蒸馏，解释真实执行与训练样本。"
draft: false
---

## 1. 问题背景 教师知道答案为何仍会教不稳多轮学生

学生在环境里走自己的路径，教师对这些路径提供概率信号，这种OPD能覆盖学生实际犯错的前缀。但多轮中错误动作改变后续环境；几步偏离后，教师面对的状态也可能陌生，或者学生总在低价值循环里反复得到局部监督。只在最后裁剪一条完整轨迹，还可能浪费前面已付出的生成和工具计算。

Agentic OPD重点研究状态访问、教师信号与多轮执行组织。它与[Agentic RL](#q=agentic-rl-training)共享环境与policy训练，但纯OPD不必依赖任务reward，也不必有critic。教师token概率不是最终任务成功率，KD loss下降不能替代独立评测。

## 2. 前置知识 三种对象和两条分布

**Proposal** 是模型提议；**executed action** 真正改变环境；**training response** 是被教师评分并进入loss的文本。传统OPD里三者常接近，教师干预和ActFirst路线会让它们分开。保存一份“模型输出”字段不够描述实际轨迹。

**Action distribution** 是给定当前history的行为概率；**state visitation distribution** 描述整个策略/环境运行会访问哪些history。局部动作校正不自动修复后者。所谓on-policy可指学生生成的token，也可指完整状态来自纯学生；具体工作可能只满足其中一层。

令$p_\theta(\cdot\mid h)$为学生，$q(\cdot\mid h)$为冻结教师，二者在同一真实prefix上、相同tokenizer/词表和明确采样规则下比较。Token mask决定哪些生成位置训练，工具观察作为条件。跨tokenizer、不同system prompt或teacher privileged context需要另定义目标，不可直接按token ID相减。

## 3. Forward Reverse KL与正确梯度

固定prefix上：

$$
L_{\mathrm{rKL}}=\sum_vp_\theta(v)\log\frac{p_\theta(v)}{q(v)},\qquad
L_{\mathrm{fKL}}=\sum_vq(v)\log\frac{q(v)}{p_\theta(v)}.
$$

Reverse KL按学生概率加权，倾向避免学生选择教师弱支持的区域；forward KL按教师概率加权，要求学生覆盖教师给质量的行为。Mode-seeking/covering是常见趋势，不保证每个离散任务都表现相同，也不能脱离支持和模型容量讨论。

### Sampled scalar无偏不等于直接autograd无偏

对当前学生sample $v\sim p_\theta$，$\log p_\theta(v)-\log q(v)$的期望是reverse KL；但采样分布也依赖theta。只对这个采样后的标量求导，遗漏score-function部分。在固定prefix下：

$$
\nabla_\theta L_{\mathrm{rKL}}
=\mathbb E_{v\sim p_\theta}
[(\log p_\theta(v)-\log q(v))\nabla_\theta\log p_\theta(v)].
$$

原导数中的+1项因score期望0消失。可将$A^{\mathrm{KD}}=\log q-\log p_\theta$作为stop-grad系数，构造负policy-gradient形式。多epoch/旧样本还要实际behavior和current信号，不能把任意logprob差直接叫完整OPD loss。

```python title="教学实现 固定prefix的reverse KL梯度"
kd_adv = (teacher_logprob - student_logprob).detach()
loss = -(kd_adv * student_logprob)
```

这个单样本式只针对给定prefix、当前学生采样；不自动包含访问prefix分布的导数。若要优化整条轨迹的KL或任务return，还需相应未来credit与状态占用处理。常见OPD可采用条件蒸馏的semi-gradient方式，应把实际目标说清。

Teacher sample $v\sim q$时，SFT $-\log p_\theta(v)$的期望为$H(q)+D_{\mathrm{KL}}(q\|p_\theta)$，教师熵对theta固定，所以SFT和forward KL具有相同条件期望梯度。反过来，把student sample-logratio直接替换成teacher样本，其标量期望变成**负forward KL**，方向不能随便沿用。

## 4. Guided-OPD 随机按turn指导与课程退出

[Guided-OPD精读](#paper=paper-guided-opd)在每个turn按beta随机选择teacher或student产生完整response，再执行并追加实际观察。Teacher turn使用forward KL，student turn使用reverse KL；beta随训练progress下降，默认cosine从1到0，前80%训练作curriculum，最后保持纯学生OPD。

不是在一个语法不完整的action内部任意切换token角色。Turn边界更贴近环境转移；teacher与student共享真实history，角色应作为训练数据记录。教师生成的turn不是“学生on-policy样本”，使用合适的示范监督比伪造student behavior更清楚。

给定history，独立角色抽样的执行分布为：

$$
\mu_\beta(a\mid h)=(1-\beta)p_\theta(a\mid h)+\beta q(a\mid h).
$$

**混合的是每turn，不是整条路径。** 中间beta时history来自整条mixed rollout，不能简单当作$(1-\beta)$条纯学生trajectory加$\beta$条纯教师trajectory。蒸馏权重和所访问状态必须分别说明，单篇采用原Eq9的角色指示形式解释。

该工作使用Qwen3-30B-A3B教师与0.6B/1.7B/4B学生；UOPD/ActFirst的teacher配置不同，不能看ALFWorld绝对成绩就横向排名。Teacher PPL可反映支持程度，但不是已经测得完整teacher state density或任务真值。

## 5. UOPD 在执行前按uncertainty换动作

[UOPD精读](#paper=paper-uopd)先让student提出response，teacher在当前history上计算uncertainty；高于阈值时再生成teacher correction并执行，其他turn执行student。阈值由近期分数buffer的分位数与target intervention rate决定，并在一次episode内固定。

uncertainty可以由teacher在student输出上的低confidence或student-teacher gap实例化；它不必等于teacher整词表Shannon entropy。Target rate控制“预期介入多少”，实际分数漂移、ties和buffer状态使实现率未必恰好等于目标。

### 为什么不是一个固定mixture

若$I(h,a)$是student proposal是否触发的指示，则由执行规则可推得：

$$
\mu_G(a\mid h)
=p_\theta(a\mid h)(1-I(h,a))
+\left[\sum_up_\theta(u\mid h)I(h,u)\right]q(a\mid h).
$$

这是本篇对规则的固定history推导。未介入student动作已经经过筛选，不能把它们当作未筛选p的完整样本；也不能简单用target rate代替该式中的实际触发概率。条件蒸馏损失、辅助执行policy与完整student状态分布应区分。

### 两种loss的源码接线

固定[UOPD policy loss](https://github.com/onepounchman/UOPD/blob/61f90c5c3f2c75b0f4be2742682baab63b8a8672/trinity/algorithm/policy_loss_fn/uopd_policy_loss.py#L48-L57)核心节选：

```python title="UOPD真实节选 按实际角色选择目标"
ratio = torch.exp(log_ratio)
pg_tok = -advantages * ratio
sft_tok = self.sft_coef * (-logprob)
is_trig = is_triggered.bool().view(-1, 1)  # [B, 1]
per_tok = torch.where(is_trig, sft_tok, pg_tok)
loss = aggregate_loss(per_tok, action_mask, loss_agg_mode=self.loss_agg_mode)
```

log_ratio在前面限幅，student branch使用OPD PG，teacher branch使用plain SFT；不是每token都做同一种KL。这里是省略空行/注释的节选，完整函数的clamp、metric和参数见链接。触发teacher时，训练文本和mask对应执行的teacher correction，不能把被拒student proposal混入其监督。

原文理论的return改善依赖选定教师动作平均有正优势的recoverability条件；低confidence本身不证明这个条件。更重要的是，辅助policy当场走得更好，不等于学生经过某一次SGD一定变好。原表也显示更多干预或更大SFT权重并非普遍最好。

## 6. STRIDE 早停和前缀重启在省哪部分

[STRIDE精读](#paper=paper-stride)根据turn-level教师信号主动决定继续/停止，保留适合复用的prefix，再从选定较薄弱位置重启。它改变如何分配交互预算，与执行前直接teacher接管不同。

Prefix buffer保存可从哪里继续，不必等于replay旧样本多次训练。复用前缀减少重复交互，但应记录旧段是否进入loss、后续新段由哪版学生采样。Coding agent还要重建文件、进程、工具session和外部状态；只存token不足以真正恢复。

该论文主要在tau-squared bench的retail/telecom环境验证，不能把“教师认可低”直接当coding动作错误，也不能把其速度与另一硬件下训练TPS比较。低confidence的恢复动作可能对任务有价值，停在哪里影响后期turn覆盖。

## 7. ActFirst-OPD 先推进环境再补完整训练响应

![本文示意 三种文本对象与蒸馏路径](./assets/learning/agentic-opd-training/opd-action-state.svg)

**图解。** 传统student response常直接参与环境；teacher intervention替换执行对象并改变loss；ActFirst的fast action推进环境，完整student response另行生成供teacher评分。所有训练response都要标自己的真实input和behavior，不能把另一个action的logprob借给它。

[ActFirst-OPD精读](#paper=paper-actfirst-opd)用离线reference的下一observation作为transition target，让同一个学生用action-only prompt推断fast action。若实际transition不再匹配reference，后续改为自主next-action prediction，继续从真实状态采样，而不是强行执行reference后缀。

与此同时，从每个已到达context $c_t$异步生成标准think-then-act完整response $y_t=(z_t,a_t)$，它不看到reference next observation。这个$a_t$不必等于执行的$\tilde a_t$，也不送到环境；teacher只给完整response蒸馏监督。

**为什么纯OPD能这样做？** 局部teacher supervision可在给定真实context上训练student response，不需要把未执行response的action当作取得任务reward的动作。但这些context由fast/reference-assisted policy访问，与部署时完整think-then-act学生的state distribution不同。不能把conditional student sampling说成完全消除了rollout分布差异。

理想化T-turn、fast/full延迟分别为$\ell_f,\ell_y$，足够并发且忽略环境/评分成本时：

$$
S_{\mathrm{ideal}}
=\frac{T\ell_y}{(T-1)\ell_f+\ell_y}
\le\min(T,\ell_y/\ell_f).
$$

它增加额外fast生成工作，只是将full response移出环境依赖链；并发不足可变慢。原实验低并发下有低于1的rollout speedup，完整训练平均加速还排除了teacher训练/reference准备成本。单篇保留硬件、更新步数、时间口径和质量下降格子。

## 8. AsyncOPD 缓存和旧策略在哪影响目标

[AsyncOPD精读](#paper=paper-asyncopd)将rollout、teacher scoring、learner持续流水化，区分KL方向与缓存量。固定prefix上，teacher-weighted forward KL和student-weighted reverse KL对旧student采样/分数的依赖不同；learner重算current student信号、importance sampling和多个MC动作样本有各自条件。

至少区分三类缓存：teacher固定模型在固定prefix的logprob；rollout时student的behavior logprob；learner当前student logprob。第一类在条件完全相同时可复用，第二类应保存历史事实，第三类随参数变化不能用旧值冒充。

给定prefix的动作ratio校正不会自动把旧history变成current student会访问的history。数学推理实验中的吞吐/准确率证据不能直接证明长程coding state drift已解决。控制积压与版本差，可结合[AReaL](#paper=paper-areal)、[APRIL](#paper=paper-april)及异步专题。

## 9. 其他路线和反馈组合

TCOD、TurnOPD等从参与监督的turn范围、长度/权重入手；UOPD对这些原始方法的引用不能代替逐篇实验核查。当前已核核心加速/干预机制与候选阅读在[Agentic OPD专题](#report=survey-agentic-opd-2026)区分，不用二手摘要填“同预算最好”。

若teacher是黑盒，只提供文字而无token分布，需要示范、偏好、rubric或其他对齐设计；不能把它伪装成全词表KL。多teacher与跨tokenizer的集合见[既有OPD专题](#report=survey-opd-llm-2026)，本篇只在明确接口处连接，不重写全部路由方法。

可验证任务reward可与KD共同使用，例如 $L=L_{\mathrm{RL}}+\lambda_{\mathrm{KD}}L_{\mathrm{KD}}$，但teacher偏好与任务真值可能冲突。记录成功率、分布差异、teacher调用、动作/turn数、状态覆盖和wall time/GPU-hour，不能只凭loss较低或teacher喜欢判断进步。

## 10. 统一比较与实验顺序

| 路线 | 执行由谁驱动 | 核心学习信号/变化 | 必须保存 |
| --- | --- | --- | --- |
| Vanilla OPD | student | 条件teacher蒸馏 | 实际prefix、学生版本/概率 |
| Guided-OPD | 随机turn混合 | student rKL、teacher fKL、beta课程 | role、shared history、schedule |
| UOPD | student proposal经uncertainty选择 | 非触发PG、触发teacher SFT | proposal/执行、threshold/buffer |
| STRIDE | student加prefix重启 | 自适应长度与计算复用 | prefix状态、重启位置、新段mask |
| ActFirst-OPD | reference-conditioned fast student | context上的完整student response蒸馏 | fast执行、reference对齐、完整sample |
| AsyncOPD | 近期student持续生产 | KL/缓存/版本校正 | behavior/current、队列、MC条件 |

先建立同teacher/student/任务的vanilla baseline并检查概率梯度，再对照指导/不确定性干预，最后添加恢复与流水线。将“teacher执行带来当场收益”和“student独立部署提升”分开测；将same updates和same time/GPU-hour分开测。

## 11. 面试问答

**Q01 [定义] OPD为什么不必有critic？** 反馈可来自teacher条件分布，不是必须估计未来任务return；不同loss仍要正确梯度。

**Q02 [梯度] Sample logprob差无偏为何backward仍可能错？** 采样分布依赖theta，直接求导遗漏score-function；固定prefix的解析/期望梯度可检验。

**Q03 [方向] Teacher样本能直接沿用student rKL项吗？** 不自动。Raw logratio期望变负fKL，SFT才给相应forward目标梯度。

**Q04 [分布] Guided beta0.5是半条teacher轨迹吗？** 每turn独立混合，后续history受前面两种角色影响，不是两类完整路径简单拼比例。

**Q05 [选择] UOPD的target rate等于真实teacher比例吗？** 不保证，buffer、ties与漂移影响实现率，需同时统计目标与实际。

**Q06 [保证] Teacher低confidence证明替换更好吗？** 不证明，理论依赖被选教师动作有平均正优势等条件，任务评测仍需独立做。

**Q07 [恢复] STRIDE的prefix能直接用于coding吗？** 还要实际环境、文件、进程和session恢复，旧段mask/版本也要记录。

**Q08 [对象] ActFirst完整response的action执行了吗？** 不必执行，环境由fast action驱动，完整response另行生成用于蒸馏。

**Q09 [概率] 能给完整response用fast action的logprob吗？** 不能，输入和输出不同；训练样本应保存自己的真实条件与概率。

**Q10 [效率] 多并发为何可能使ActFirst更快？** Full生成脱离transition链可重叠，但新增fast工作、prefill和资源竞争要付费；低并发可变慢。

**Q11 [异步] Importance ratio修复全部state drift吗？** 固定prefix动作校正不等于state occupancy校正，旧teacher/student缓存也分对象。

**Q12 [评测] 最后应该报告哪些收益？** Student独立任务质量、teacher辅助比例/成本、有效turn、训练时间/GPU-hour和状态覆盖，区分原文测量与自己的实验。

## 12. 来源与CPU验证

[来源清单](./research/agentic-opd-training-sources.json)登记固定原文与源码。[CPU脚本](./assets/learning/agent-training-lab.py)和[结果](./assets/learning/agent-training-lab-results.json)核fKL/rKL值与梯度、错误naive梯度、mixture与gate执行分布、理想speedup等，均只针对明确的小模型/条件。

导入阅读可先看[DAgger原始工作](https://proceedings.mlr.press/v15/ross11a.html)理解learner访问状态上的expert指导，再对照本篇的角色、目标与state变化。它提供前作关系，不是这些LLM方法的完整实现或无条件保证。
