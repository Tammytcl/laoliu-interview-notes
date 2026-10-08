---
id: survey-policy-optimization
title: "PPO DPO GRPO GSPO GDPO DAPO SAPO 的机制比较与面试复习"
type: survey
date: '2026-10-08'
directions: [llm, agent, infra]
paper_ids: [paper-ppo, paper-dpo, paper-grpo, paper-dapo, paper-gdpo]
tags: [专题调研, Policy Optimization, RLHF, RLVR, Interview]
updated: '2026-10-08'
summary: "把七种方法放入同一训练流程，比较数据来源、优势、概率比、裁剪、reward组织和长度权重；附推导入口、手算例子与场景化面试追问。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

这七种方法分别改变训练的哪一个环节，遇到什么问题才需要它？本专题围绕后训练优化机制，服务于论文精读、工程排查和面试表达。重点是公式与实际数据流，框架选型、GPU架构和checkpoint格式留在各自Infra笔记中，不扩成后训练大杂烩。

主线是：PPO让旧策略生成的数据能被较稳健地多次更新；DPO从偏好pair直接拟合策略；GRPO用同题组替代valuebaseline；GSPO把概率比和clip改成sequence级；DAPO改探索、有效组、token权重与超长反馈；GDPO改多reward归一化顺序；SAPO改硬clip为可微softgate。这些修改不全处于同一层，也不一定互斥。

先给一个面试开场：**先分离离线偏好拟合与在线策略优化，再沿优势估计、概率比、更新约束、奖励组织和reduction比较。** 回答“哪个更好”前需要模型、数据、反馈、长度、预算与quality指标，不能只背缩写年代。

## 2. 范围、检索与覆盖边界

2026-10-08按用户列出的七种算法检索原文、arXiv固定版本、LaTeX/PDF及公开实现；补读OpenAI SpinningUp、ICLR PPO实现博客、HuggingFace DPO教程、Qwen GSPO与DAPO/GDPO作者项目。网络检索接口不可用时直接读取原站点和公开仓库，不用搜索摘要代替原文。

每篇正文约9.4k—11.4k字符，包含五模块、原图、推导、手算、源码、配置和问答。PPO/DPO/GRPO/DAPO/GDPO具备原表并发布；GSPO/SAPO全文与原图已完成，但原文没有结果或实验表，依仓库现有规则保留单篇草稿。下文仍比较其方法并链接固定原文，未以别篇表格补证。未执行七种方法的重新训练，性能都标明论文条件。

| 名称 | 固定原文 | 单篇入口与状态 |
| --- | --- | --- |
| PPO | [1707.06347v2](https://arxiv.org/abs/1707.06347v2) | [PPO精读](#paper=paper-ppo) |
| DPO | [2305.18290v3](https://arxiv.org/abs/2305.18290v3) | [DPO精读](#paper=paper-dpo) |
| GRPO | [DeepSeekMath 2402.03300v3](https://arxiv.org/abs/2402.03300v3) | [DeepSeekMath精读](#paper=paper-grpo) |
| GSPO | [2507.18071v2](https://arxiv.org/abs/2507.18071v2) | 完整本地草稿；原文无表 |
| GDPO | [2601.05242v1](https://arxiv.org/abs/2601.05242v1) | [GDPO精读](#paper=paper-gdpo) |
| DAPO | [2503.14476v2](https://arxiv.org/abs/2503.14476v2) | [DAPO精读](#paper=paper-dapo) |
| SAPO | [Soft Adaptive 2511.20347v2](https://arxiv.org/abs/2511.20347v2) | 完整本地草稿；原文无表 |

GDPO排除同名GFlowNet和image-superresolution工作；SAPO排除single-rollout、self-adaptiveprocess等其他缩写。GRPO没有伪造一篇独立题名“Group Relative Policy Optimization”的原论文，而以真实DeepSeekMath全文收录。

## 3. 论文集合与方法谱系

### 先建立共同符号与四个模型职责

| 对象 | 输入与输出 | 更新周期 | 容易混淆的对象 |
| --- | --- | --- | --- |
| currentpolicy $\pi_\theta$ | prefix→nexttoken概率 | optimizerstep变化 | actor不是所有阶段都同一版本 |
| behavior/old $\pi_b$ | 实际生成回答的策略/概率 | 每轮rollout或异步版本段 | 与reference职责不同 |
| reference $\pi_{ref}$ | 原行为分布锚点 | 可固定，也可outeriteration刷新 | 不是GAE的baseline |
| reward $R$ | 完整回答/过程/环境→反馈 | 规则固定或模型迭代 | 不必可微，不等于critic |
| value $V$ | prefix/state→未来预期回报 | critic训练更新 | 与rewardmodel学习目标不同 |

old有时只需保存logprob，不必常驻一套可训练参数；reference可能缓存；actor/critic可共享backbone。这里是逻辑职责，不是“五套同大小模型的峰值显存公式”。

在线组方法的流程是prompt→同题G条rollout→评分→组优势→重算logprob→ratio→surrogate→reduction→optimizer→权重发布。GRPO/GDPO改优势，GSPO改ratio和clip单位，SAPO改surrogate，DAPO同时作用于采样、clip、reduction、reward。训推异步主要改变这些量何时就绪和属于哪个版本，数学含义仍需核对。

DPO流程是固定prompt/chosen/rejected→policy/reference四组logprob→pairmargin→logsigmoidloss→optimizer。它不在每个更新步做rollout或学习value，但生成偏好数据的成本仍存在；onlineDPO等后续路线会改变这一点。

### 最小公式卡片

用$r_{i,t}=\exp(\ell_\theta-\ell_b)$表示tokenratio，$s_i=\exp(\operatorname{mean}_t\log r_{i,t})$表示几何平均sequence比。忽略非核心项后：

$$
\begin{aligned}
\text{PPO: }&\;A_t=\operatorname{GAE}(R,V),\quad J=\mathbb E\min(r_tA_t,\operatorname{clip}(r_t)A_t).\\
\text{GRPO: }&\;A_i=\frac{R_i-\mu_q}{\sigma_q},\quad J=\mathbb E\operatorname{mean}_{i,t}\min(r_{i,t}A_i,\operatorname{clip}(r_{i,t})A_i).\\
\text{GSPO: }&\;J=\mathbb E\operatorname{mean}_i\min(s_iA_i,\operatorname{clip}(s_i)A_i).\\
\text{GDPO: }&\;A_i=\operatorname{Whiten}_B\left(\sum_kw_k\operatorname{ZScore}_q(R_{i,k})\right).\\
\text{SAPO: }&\;J=\mathbb E\operatorname{mean}_{i,t}\left[\frac4{\tau_i}\sigma(\tau_i(r_{i,t}-1))A_i\right].
\end{aligned}
$$

上式的mean只是谱系示意，GRPO与DAPO的sequence/tokenreduction不同；原始GRPO还含referenceKL，DAPO最终配方移除KL，不能以此卡片取代各篇完整loss。

DPO令$m=\beta[(\ell_{\theta,w}-\ell_{ref,w})-(\ell_{\theta,l}-\ell_{ref,l})]$，$L=-\log\sigma(m)$。分母是reference不是behavior；其pairmargin不是在线GRPO的组优势。

## 4. 统一比较表

| 方法 | 数据与反馈 | value/优势 | ratio与更新约束 | 长度或多目标改动 | 主要成本与风险 |
| --- | --- | --- | --- | --- | --- |
| PPO | 在线轨迹，任意合适reward | 学习V，GAE常见 | token/actioncurrentold比，clip或KLpenalty | 取决应用实现 | critic、优势误差、旧数据偏离 |
| DPO | 原始为离线偏好pair | 无value，implicitrewardmargin | policy/ref比进入分类logit；无PPOclip | 原始sequence logprob求和 | 偏好覆盖、噪声、reference前向 |
| GRPO | 在线同题G回答 | 组mean/std；无需独立critic | tokenratio硬clip，原文referenceKL | 原始sequence均值；outcome/process | G次rollout、零方差、组等待 |
| GSPO | 同题组与sequence reward | 继承组优势 | 几何平均ratio，sequence硬clip | 长度归一化ratio、GSPO-token | 局部异常被平均，整组信号与配置缺口 |
| DAPO | 数学可验证reward，过滤补采 | 组优势 | tokenratio非对称clip，最终无KL | globaltokenmean与超长shaping | 超采成本、覆盖偏差、verifier漏洞 |
| GDPO | 在线多reward | 每reward组归一化再batchwhiten | 可沿用GRPO/DAPOsurrogate | rewardweight在归一化后、条件reward | 冲突抵消、尺度与whitening统计 |
| SAPO | 在线组reward | 继承组优势 | tokenratio平滑sigmoid目标 | 正负优势不同tau；sequence均值 | 饱和、温度、近似条件与旧数据 |

### 实验数字为什么不能直接拼排名

| 工作 | 代表证据 | 需要保留的条件 |
| --- | --- | --- |
| PPO | MuJoCosweepclip0.2归一化score0.82 | 2017控制任务，不是LLM准确率82% |
| DPO | TL;DR约61%对PPO约57% | GPT-J、judge、解码、参考摘要；不是所有任务 |
| GRPO | MATH46.8%→51.7% | DeepSeekMathInstruct→RL7B；含reward与训练配方 |
| GSPO | 相同compute曲线更早达到相近quality | Qwen3MoE、GRPO已routingreplay、无绝对GPU-hour |
| DAPO | AIMEavg@32从朴素GRPO30到50 | progressive配方、base32B、更多采样可能 |
| GDPO | 1.5B BFCL30.18%→32.81% | 多reward、5runsmean、format另看 |
| SAPO | 更久保持训练与验证改善 | AIME25等、temperature消融、非永不崩溃 |

DPO原论文并未在全部任务重新训练同规模PPO；GSPO/SAPO未给完整资源表；DAPOsteps减半不等于GPU-hour减半；GDPO部分accuracy下降换取长度改善。保留这些条件才能比较“证据支持什么”，而非凭更新论文的分数越来越高假装算法进步量可直接相减。

## 5. 共识、分歧与证据强度

### clip三连问

“越界是否零梯度？”先看advantage符号。正优势上越界、负优势下越界进入平台，反方向仍纠正。共享参数或KL/value项还可能改变其概率。“clip是否严格trustregion？”它约束某个surrogate收益，不保证完整KL或参数距离硬边界。“clipfraction是否越低越好？”GSPO高于GRPO两数量级却更有效，不能拿一个日志当quality。

### importance sampling与序列比

完整trajectoryratio可做标准换测度，但长序列方差很高；token局部ratio只校正给定prefix的action分布，不能替换整条prefix访问分布。GSPO用完整ratio的长度次方根改善尺度，同时一般不再满足严格IS恒等式。SAPO进一步平滑抑制，裁剪/门控都不是无损校正。

作者对“tokenratio失效”的诊断应与一般概率论事实分开：单样本IS可以无偏但高方差，样本数1本身不让densityratio非法。实际surrogate、状态分布偏移、clip和估计误差需要一起讨论；面试中直接说“GSPO理论正确，GRPO理论错误”过度简化。

### normalization不是纯工程细节

std、groupmean、leave-one-out、batchwhitening和tokenreduction决定梯度尺度与数据权重。两个rollout时z-score消掉总reward差幅；多个reward分别标准化后可能保留部分组成，但等权互补仍会抵消。分布式局部tokenmean与全局tokenmean不等价，microbatch重排也需UID保证同题统计。

### actorloss不是优化一切的通用药

reward错误时更有效优化可能更快rewardhack；只保存token不等于环境可恢复；group优势仍可能等待同题尾部；softgate不等于无限stale轨迹可用。系统吞吐改进应落到相同GPU-hour下任务收益，算法替换也不自动带来GPU高利用率。

## 6. 场景化结论与选型

以下是依据机制提出的工程判断，不是个人训练验证或跨论文排行榜。

| 已观察问题 | 优先做的对照 | 要验证什么 |
| --- | --- | --- |
| 有高质量固定偏好pair，暂不需新探索 | SFT与DPO | held-outpair、KL、length与真实任务quality |
| 在线reward与较长时序credit，critic可承担 | PPO/GAEbaseline | value质量、bootstrap、KL与sampling成本 |
| 不想训独立critic、同题能生成多回答 | GRPO | G、zero-variance、groupbarrier与reduction |
| 长CoT熵快速下降且有效组减少 | DAPO各项消融 | clip上下界、acceptance、总生成tokens |
| MoEtokenratio波动、routing开销大 | GSPO与routingreplay对照 | ratio定义、clipscale、相同computequality |
| 正确性、长度、格式reward互相挤压 | GDPO与去std/条件reward | 每维reward、权重、tailconstraint与accuracy |
| 硬clip丢信号且有tokenoutlier | SAPOtemperature对照 | gate分布、sign不对称、后期崩溃时间 |
| rollout主导且工具等待严重 | 先系统profiling，再对应算法 | readyrequests、环境时间、版本滞后与goodput |

概念上GDPO优势可接GSPO/SAPO loss，DAPO过滤可接别的surrogate，但组合会改变有效batch、尺度与探索。推荐先建立一条可核对的baseline，一次改变一个轴，再测试联合方案；“所有开关都打开”很难解释失败。

## 7. 空白与下一轮验证

七篇已完成固定来源和核心机制精读，仍有明确验证空白：GSPO/SAPO缺完整实验表和硬件参数；DeepSeekMath原GRPO训练代码未公开；各论文不同任务、模型和reward无法直接互证；本次未运行训练。应从原作者/维护者配方补充，不用常见超参填空。

若做一套真正可比较实验，可固定一个小模型与同一math/coding数据，冻结verifier、初始化、最大长度与采样预算。记录每promptG、每rollout更新数、有效组比例、每rank全局token分母、生成/评分/训练耗时、GPU-hour、held-outPass@1与尾部长度。然后分三组比较：优势（GRPO/GDPO）、ratio（GRPO/GSPO）、surrogate（hardclip/SAPO）；DAPO四项再独立消融。此为建议方案，尚未执行。

需要更完整的理论对照时，后续可专门读GAE、RLOO、Dr.GRPO、REINFORCE++、CISPO等。本轮仅将它们作为边界说明，不追加浅摘要扩大论文集合。

## 8. 阅读顺序与个人理解检查

建议先PPO Figure1和GAE，接DPO最优策略与pairloss，再读GRPO Figure4。随后DAPO Algorithm1/Table1建立“诊断→改动”，用GDPO Figure2手算多reward，再读GSPO原式与SAPO Figure1梯度gate。GSPO/SAPO无表的单篇草稿不影响从固定原文学习。

### 面试复习主问题

1. **不用缩写讲清一轮在线RL训练。** 说出生成策略、reward、advantages、currentlogprob、ratio、surrogate、mask/reduction、optimizer和下一轮权重；不要只说rollout再backward。
2. **为什么reward、critic、reference、oldpolicy不同？** reward评价行为，critic估计状态未来回报，reference锚定行为，old描述实际数据来源；它们的更新周期各异。
3. **DPO到底省掉哪些环节？** 原始训练步不需onlinegeneration、显式RM训练和value估计，仍需偏好pair与reference分数。
4. **GRPO去critic的代价是什么？** 同题G次rollout、统计尺度、零方差、粗credit与组完成依赖。
5. **用正负A四种情况解释clip。** 正上平台、负下平台；另一侧保留纠错梯度。
6. **GSPO的ratio如何计算？** 对logprob差取有效token均值再exp；是几何平均，不是算术平均或完整无偏densityratio。
7. **GSPO-token为何前向为1的项还有梯度？** 分子可微、分母detach；前向值与导数不能混为一谈。
8. **DAPO为何仍使用GRPOadvantage？** 它主要改采样、clip、reduction与超长reward，adv_estimator标签只说明一个环节。
9. **GDPO权重应该在哪一步乘？** 逐维z-score之后；归一化前整体乘正数通常被抵消。
10. **SAPO公式中的f和w分别是什么？** f是sigmoid代理目标，w是导数4p(1-p)，真实logpolicygradient还乘r。
11. **提高tau为什么更快衰减？** 因为tau乘在sigmoid输入上；不要用除temperature的softmax习惯。
12. **如何诊断相同reward但不同梯度？** 检查groupstd、batchwhiten、advsign、ratio与tokenreduction，再排查mask和版本。
13. **解释avg@32、pass@32、cons@32。** 平均单次成功率、至少一次成功、候选投票，是三个不同预算/选择问题。
14. **新算法是否天然适合异步？** 需behaviorlogprob、版本段、组依赖、队列背压；门控/clip不解决所有状态分布漂移。
15. **如何公平比较训练效率？** 同模型、数据、reward、长度和GPU-hour，报有效训练tokens与taskquality，不只steps/s。
16. **如何保证这不是背论文结论？** 手算、推导梯度、说明一条源码分支，再给一个失效条件或反例。

### 五个手算自检

| 题目 | 应得到的结论 |
| --- | --- |
| epsilon0.2，A=±2，r=0.7/1.3 | 四种clip分支不同，不是越界统一零梯度 |
| rewards[0,0,1,1] | populationadv±1，samplestd约±0.866；要注明定义 |
| tokenratios[2,0.5] | GSPO几何平均1，算术平均1.25 |
| 两reward互补[1,1,0]与[0,0,1] | 总分全1；等权GDPO也可抵消为0 |
| SAPOtau1，r=1/2/5 | w约1/0.7864/0.0707；wr与w不同 |

题目由本文根据原论文整理，不声称来自具体公司的真实面试题库。单篇另外提供共56道机制问答及追问；用途是检查理解，不以未经验证的面试频率作为选题依据。

## 9. 持续更新记录与来源

2026-10-08：完成七篇固定版本正文、原图、公开源码伴读与本专题；五篇发布，两篇因原文无表保留完整草稿。新增概率比/优势/reduction对照、GPU-hour口径、56道单篇问答与16道综合复习问题。未执行训练复现。

参考讲解的贡献限于教学入口： [SpinningUp PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)用于正负优势，[ICLR PPO实现细节](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)用于版本和边界，[HF DPO/TRL](https://huggingface.co/blog/dpo-trl)用于pair数据流，[Qwen GSPO](https://qwenlm.github.io/blog/gspo/)用于sequence单位，[DAPO项目](https://dapo-sia.github.io/)用于训练症状，[GDPO项目](https://nvlabs.github.io/GDPO/)用于多reward手算。各篇数字、曲线与公式以固定原文为依据。

来源版本、源码commit、图表label/hash与人工检查记录在单篇和图表清单；下一轮若更换原文版本，先更新证据和实验账本再改正文，不覆盖旧数字而不记录条件变化。
