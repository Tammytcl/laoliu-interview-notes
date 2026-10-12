---
id: opd-interview-foundations
title: "OPD面试基础：目标、梯度、训练代码与强基线"
category: agent
difficulty: 进阶
tags: ["P0", "OPD", "Interview", "KL", "Policy Gradient", "Distillation"]
updated: "2026-10-12"
summary: "24道分层问答，从OPD定义到sampled KL梯度、teacher API、mask、异步、失败轨迹和同预算实验；公开题源与自拟追问分别登记。"
draft: false
---

## 1. 问题背景 准备OPD项目面试要能解释哪四件事

面试不会只问OPD全称，通常会从“为什么选它”追到“loss真的怎么计算、系统怎么运行、结果怎么证明”。本篇按定义→数学→实现→实验组织24道问题，每题给回答要点与可继续追问的方向。先读[Actor–critic](#q=rl-actor-critic)、[PPO训练一步](#q=rl-ppo-training)、[Agentic OPD](#q=agentic-opd-training)，再用本篇闭卷答。

本文不声称收集“网上全部”题目，也不根据一个匿名帖子宣布公司高频题。公开题解、面经整理、个人学习页和项目自拟问题具有不同证据等级。答案回到原始论文/作者代码，公开题解若有过度简化不直接继承。你的具体模型、运行配置与实验成绩只填真实记录，不写预期收益当结果。

## 2. 前置知识与题源账本

| 来源 | 能确认什么 | 本篇怎样使用 |
| --- | --- | --- |
| [牛客AI Infra面经题解](https://www.nowcoder.com/discuss/933731201871806464?sourceSSR=post) | 页面讨论OPD动机、错误轨迹筛选与项目表达；系公开整理，无法独立核公司真实性 | Q01/Q18的问题方向参考，答案独立推导 |
| [九坤Agent题解章](https://ricardokevins.github.io/notes/llm-interview-question-bank/chapters/091.html) | 整理OPD流程、与SFT区别、API概率可得性 | Q02/Q12题型参考；不宣称公司原题认证 |
| [个人Interview知识页](https://lpq29743.github.io/wiki/interview/) | 公开复习页面含后训练/OPD | 阅读路线补充，不用作技术事实或频次证据 |
| [Thinking Machines作者博客](https://thinkingmachines.ai/blog/on-policy-distillation/) | 介绍学生rollout、教师sampled评分与训练流程 | 教学思路参考，不是面经 |
| [GKD](https://arxiv.org/abs/2306.13649)、[MiniLLM](https://arxiv.org/abs/2306.08543) | 目标与蒸馏原始方法 | 数学与机制依据 |
| [新MOPD审计](https://arxiv.org/abs/2610.04272v1) | 强SFT配方、独立调参与成本边界 | Q22–24为围绕新论文自拟答辩问题 |

以上网页于2026-10-12核对。Q01/Q02/Q12/Q18是从公开题型改写，其余为根据原文、代码和你的项目方向自拟，包含推导题、白板题、排障题；不是照抄一份不明来源的“真题大全”。

约定：$p_\theta$学生、$q$冻结教师、$p_b$行为策略；$h$为固定当前prefix。局部词表KL和完整轨迹目标分开讨论，避免把访问分布导数偷偷丢掉又称严格总梯度。

## 3. 第一层 定义与方法选择的面试问答

**Q01 [公开题型·动机] 什么是OPD，为什么不直接做SFT？**

先说学生生成自己的轨迹，教师在学生实际prefix上提供监督，learner更新，循环再采样。Teacher SFT在教师记录历史上拟合输出；OPD覆盖学生自己犯错后的前缀，适合分析暴露偏差与恢复。随后限定条件：教师在学生状态上要可用，学生需具有足够起始能力，在线生成和环境成本也要值得。不能回答“OPD必然比SFT强”，新MOPD审计正要求认真筛选与调优SFT。

追问：你项目的增益发生在初始正确率、错误恢复、OOD还是训练效率？哪组消融能定位？

**Q02 [公开题型·比较] OPD、RL、DPO和普通KD的区别？**

至少拆数据来源、反馈、目标三轴。普通离线KD用固定上下文；OPD用学生状态。RL可用环境终局reward，也可有稠密过程反馈，不能定义成永远稀疏；OPD通常用教师概率/动作指导。DPO通常训练固定chosen/rejected偏好pair与reference相对概率，不等于在线学生rollout。OPD可以用PG surrogate实现，也可直接词表KL，所以“用PG代码”不自动证明优化的是任务回报。

**Q03 [自拟·命名] On-policy究竟指哪些分布？**

说明生成token由谁采样、历史状态由谁执行、版本是否落后。Teacher-prefix局部学生生成，token内前缀来自学生但跨turn历史来自教师。ActFirst的full响应来自学生，环境却由fast策略推进；异步数据来自旧学生。不能只说“student生成了”就把所有层都称严格当前部署on-policy。

**Q04 [自拟·教师] Teacher一定要参数更多吗？**

重要的是目标任务能力、学生状态兼容性和反馈质量，不是参数量标签。小任务RL专家可能胜过大未适配基础模型。MoE总参数与active参数影响内存/计算不同。多教师要说明领域路由、概率校准、冲突与各域训练权重；同一缩写MOPD也可能指multi-rollout而非multi-teacher。

## 4. 第二层 概率、梯度与白板题

**Q05 [自拟·公式] 写出FKL/RKL，并说明哪个分布加权。**

$$
D_F(q\|p)=\sum_aq(a)\log\frac{q(a)}{p(a)},\quad
D_R(p\|q)=\sum_ap(a)\log\frac{p(a)}{q(a)}.
$$

FKL按教师概率加权，RKL按学生概率加权；它们都需要同一条件和公共样本空间。Mode-covering/seeking是有用直觉，不是“RKL保证学生只学一个正确答案”的定理。$q(a)=0,p(a)>0$使RKL无穷；模型softmax理论正概率，实际top-k截断/量化/下溢仍会造支持问题。

**Q06 [自拟·梯度] FKL对student logits的梯度是什么？**

固定$q$且词表完整归一时为$p-q$。教师熵与学生参数无关，FKL与教师soft-label cross-entropy差常数。教师token按$q$采样的SFT梯度条件期望也为$p-q$；单样本有方差，成功过滤后采样分布变为$q_{\rm selected}$，不能再说无条件仍等于原$q$监督。

**Q07 [自拟·推导] RKL对参数的梯度怎样写？**

固定prefix与教师：

$$
\nabla D_R=\sum_ap(a)(\log p(a)-\log q(a)+1)\nabla\log p(a)
=\mathbb E_{a\sim p}[(\log p-\log q)\nabla\log p].
$$

最后一步用$\mathbb E_p\nabla\log p=0$去掉常数。它是当前prefix局部梯度；若目标对未来状态占用也求导，还需相应score/return项，不把二者混用。

**Q08 [自拟·陷阱] Sampled KL是无偏值估计，直接backward就对吗？**

不对。采样$a\sim p$后将$a$当固定索引，对$\log p(a)-\log q(a)$直接autograd，梯度期望是$\mathbb E_p\nabla\log p=0$，遗漏采样分布导数。Full-vocab RKL中$p$权重必须有梯度；PG实现可用detach的$(\log q-\log p)$作为优势，配正确score梯度。值估计无偏与自动求导梯度无偏是不同命题。

**Q09 [自拟·代码] 写一个能自动求导的固定prefix full-vocab loss。**

```python
# Teaching code, [B,L,V], same vocabulary and fixed histories.
logp = student_logits.float().log_softmax(-1)
with torch.no_grad():
    logq = teacher_logits.float().log_softmax(-1)
rkl = (logp.exp() * (logp - logq)).sum(-1)  # do NOT detach p
loss = (rkl * assistant_mask).sum() / assistant_mask.sum().clamp_min(1)
loss.backward()  # only student receives gradients
```

应解释形状、teacher no_grad、FP32稳定计算、shift和有效token分母。这个例子是token-mean，若论文用seq-mean或group-mean要改聚合，不能只说“都取mean一样”。拿两/三token词表检查数值与梯度，再检teacher.grad为空。

**Q10 [自拟·PG] Sampled OPD的一步如何写？**

保存行为logprob$\log p_b$；current forward得$\log p_\theta$；教师同条件评分$\log q$。固定prefix的当前RKL示意采用$A_\theta=\mathrm{sg}(\log q-\log p_\theta)$与$\rho=p_\theta/p_b$，最小化$-\rho A_\theta$。若复用$A_b=\log q-\log p_b$、PPO clip或其他约束，则是具体surrogate，需说明何时近似或改变梯度。不要把所有配置叫同一个严格目标。

**Q11 [自拟·温度] Teacher/student temperature和top-k是一个参数吗？**

不是。Rollout温度/截断控制实际behavior分布，teacher soft distribution温度控制监督，候选top-k控制存储/近似，投影top-k控制映射。$p_{\rm raw}$与top-p重归一后的$p_{\rm sample}$不同。若用raw概率当实际采样denominator要解释偏差/修正；截断支持缺失时普通IS无法恢复未采动作。温度KD的$T^2$缩放与特定目标相关，不能给所有OPD默认机械乘。

## 5. 第三层 教师接口与训练数据

**Q12 [公开题型·API] Teacher API不返回完整logits，还能做OPD吗？**

要问能返回哪些信息。支持给定学生token的logprob，可实现对应sampled目标；只返回top-k，需处理采样token缺失与残余质量；只返回文本，无法直接计算原定义full-vocab KL，但可设计教师在学生状态上生成动作进行SFT、偏好或评价。黑盒反馈可能仍是广义学生状态监督，不能把它说成标准词表KL已被计算。

**Q13 [自拟·shape] Logits和labels为什么要shift？**

因果LM位置$t$的hidden预测下一个token，不是已经看到的当前位置。Teacher重编码后长度不同，不能直接使用student位置索引teacher row。Prompt最后一个hidden预测首个response token；padding/EOS与人工插入token还需mask。最小例子用可打印三token序列检查每个target和条件prefix，防止future token泄漏。

**Q14 [自拟·角色] OPD teacher就是PPO reference吗？**

Teacher提供希望学生学习的分布，reference限制策略变化；critic估值，behavior记录采样来源，proximal是更新约束锚点。可以复用模型或把某个KL正则替成蒸馏目标，但需要重新解释目标职责。说“都有logprob所以一样”会混淆KL方向和ratio分母。

**Q15 [自拟·缓存] Teacher frozen意味着所有缓存可永久复用吗？**

只有相同模型、tokenizer/template、actual prefix、候选/事件与温度条件的教师分数可以复用。Advantage含学生项、候选支持来自旧学生top-k，更新后都可能变。Cache key只用prompt ID会把不同student continuation错合并。缓存命中率应和概率等价性、bytes与前向成本同时报告。

**Q16 [自拟·mask] 工具文本、思考和动作谁有loss？**

取决目标，通常模型生成thought/action可训练、外部观察不训练但可见。手动插入标签不属于采样动作，自生成标签则可能属于。Loss mask、attention mask、padding与GAE边界mask职责不同。应给一次真实工具调用的逐token表，而不是回答“assistant都乘1”后忽略重建方式。

**Q17 [自拟·数据] 为什么不能用修改后的prompt重算旧logprob？**

行为概率条件是当时真实prefix。换summary、删除观察、修改工具schema或重排role会换条件，即使权重不变ratio也可能≠1。重算可用于current/prox模型，但必须在同样已记录的输入上；旧分母要保留实际behavior概率或经严格同版本同输入验证的重算值。

## 6. 第四层 失败轨迹、异步与性能

**Q18 [公开题型·过滤] 错误学生轨迹为什么不全部丢？**

OPD正希望在学生错误状态提供恢复监督，全部删失败会变成只学成功访问分布。系统故障、无合法输出、数据污染和任务失败分开；可恢复错误与正确前缀可能有价值，教师完全不可靠的状态则需gate/降权/截断。筛选改变分布，必须报告保留率和任务/难度覆盖，而不是“清洗越严格越好”。

**Q19 [自拟·异步] 数据落后两版本就安全吗？**

版本次数不等概率距离，跨turn混合版本还需每段标注。控制在途工作、FIFO/就绪消费、版本限制与局部IS只是不同层的处理；[AsyncOPD](https://arxiv.org/abs/2606.24143v1)明确旧动作与旧prefix不同。要看ratio、ESS、clip fraction、年龄、长任务覆盖和独立质量，不把bounded staleness当无偏保证。

**Q20 [自拟·性能] Teacher更大为什么rollout仍可能最慢？**

Teacher可一次prefill评分，student生成串行多token、多turn，工具等待和重复prefill进一步延迟。Full-vocab输出、跨词表对齐和teacher KV也可能形成新瓶颈，需真实profile。GPU utilization高不等MFU高，卡时按全部预留GPU计；只算learner卡会漏rollout与teacher成本。

**Q21 [自拟·早算] 每turn有teacher信号就能边执行边训练吗？**

要先区分score提前、backward提前累积、optimizer提前更新。Local token OPD信号或已知，终局RL回报/GRPO组优势可能未就绪。提前backward还需完整loss分母、同参数版本与collective顺序；更新提前会改变未来行为分布。对照实验应把三种改动分别隔离，而非统称pipeline。

## 7. 第五层 强基线、证据与展示

**Q22 [自拟·新论文] 新MOPD论文否定了OPD吗？**

没有。它在四个1.7B–8B模型、两种专家组成与相同初始化/词表条件下，独立调优与成功教师语料使离线方法接近MOPD的ID恢复。它还指出OOD保留和成本不同；不能外推跨tokenizer、所有长程coding或任意更大teacher。读[精读](#paper=paper-rethink-mopd)时说明主实验与旧论文复比较的验证口径不同。

**Q23 [自拟·实验] 你的对照至少有哪些？**

相同起点、prompt、teacher与独立测试，充分调LR的teacher-success SFT、soft KD、局部teacher-history蒸馏、真实学生rollout OPD。另按同生成token、同teacher调用、同GPU-hour各比较；报告工具、重试、评价和语料生成的成本，保留OOD与长度。方法组合需拆loss-only、scheduler-only/必要校正、联合处理，不能一个混合方案就称单因果结论。

**Q24 [自拟·结论] 怎样回答“你的贡献是不是只是工程实现”？**

先明确自己实现了什么，不以“加pool”自动称研究创新。问题定义、现有机制不足、可证伪假设、区别于近邻的设计和对应消融形成研究证据；系统工程贡献可以用正确性契约、可靠性、效率与可复用性证明。只跑短训练不能证明收敛或统计非劣，公开论文成绩也不能冒充个人复现结果。

## 8. 自测顺序与来源

用15分钟讲Q01–04建立对象，20分钟白板Q05–11，20分钟沿真实batch字段答Q12–17，最后20分钟答Q18–24和项目质疑。每题按“结论→关键式/数据→边界→项目证据”回答，不背一段通用口号。更多系统追问读[Pipeline](#q=agentic-opd-pipeline)，项目展示读[答辩稿](#q=opd-project-defense)，异词表读[Cross-tokenizer问答](#q=opd-interview-cross-tokenizer)。

数值与梯度可运行[共享CPU课堂脚本](./assets/learning/agent-training-lab.py)，[记录结果](./assets/learning/agent-training-lab-results.json)。它只验证小张量，不是模型训练。题源与原始依据已在前文登记，新增问题会沿相同证据规则维护，不自动标注“某公司真题”。
