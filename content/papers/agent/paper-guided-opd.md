---
id: paper-guided-opd
title: "Guided-OPD：按turn混合教师与学生，再逐渐退出指导"
paper_title: "On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents"
authors: ["Gengsheng Li", "Mao Zheng", "Mingyang Song", "Ruiqi Liu", "Tianyu Yang", "Jie Sun", "Qiyong Zhong", "Haiyun Guo", "Junfeng Fang", "Dan Zhang", "Jinqiao Wang"]
affiliations: ["Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences", "School of Artificial Intelligence, University of Chinese Academy of Sciences", "Large Language Model Department, Tencent", "University of Science and Technology of China", "Zhejiang University", "National University of Singapore", "Wuhan AI Research"]
author_affiliations: [[1, 2], [3], [3], [2], [1, 2], [4], [5], [1, 2], [6], [6], [1, 2, 7]]
venue: "arXiv preprint / Working in Progress (fixed v2)"
year: 2026
direction: agent
areas: [language]
tasks: [agents, reasoning, training-adaptation]
published: "2026-06-14"
paper_url: "https://arxiv.org/abs/2606.15912v2"
github_url: null
code_note: "论文列出Zzzz-166/Guided-OPD；公开README与API入口本轮404，未取得可固定作者实现。"
method_figure: "./assets/papers/paper-guided-opd/method-source.png"
method_caption: "Figure 2 · turn级指导、两种KL和课程退出"
evidence: 已核原文
note_ids: [agentic-opd-training, rl-ppo-training]
tags: [OPD, Guided OPD, Curriculum, Agentic OPD, Forward KL, Reverse KL]
updated: "2026-10-12"
summary: "随机按turn决定执行者，在学生turn用reverse KL、教师turn用forward KL，并衰减指导；详解真实混合历史、配置、消融和公式边界。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：学生到达的错误状态，教师也未必教得好

OPD让学生自己生成，再请教师监督这些学生前缀，缓解离线教师轨迹与学生部署分布的差距。但多轮agent的错误不仅是一个token：错误查询、无效操作或偏离任务的工具路径会改变后续环境状态。教师在很远的学生历史上也可能给不可靠局部信号，长期只做学生rollout可能难以恢复到可学习区域。

Guided-OPD用教师偶尔接管整个turn，将轨迹带回更熟悉区域，再逐渐减少指导，让学生后期独立执行。它不是只给正确答案让学生SFT，也不是在每个学生token上混合教师概率后继续把它叫纯学生策略；执行者真的改变，环境也由所选动作推进。

需要区分两种改善：教师接管使训练轨迹更顺，和更新后学生无教师时更强。前者并不自动推出后者。课程退出和独立学生评测是必要组成；教师PPL变化只是诊断代理，不是对隐藏状态分布或任务正确性的直接证明。

### 前置知识：一个turn、两种KL和课程

Turn是一轮完整思考与动作，环境观察附加到历史但不是生成者输出。Teacher turn与student turn监督方向不同：在学生采样下reverse KL偏向避免教师低概率输出，在教师采样下forward KL传递教师覆盖。Teacher行动改变下一轮历史，因此监督方向和状态访问是两个轴。

课程参数$\beta$控制本次训练step中每turn选择教师的概率，整条trajectory内随机选择多个执行者。不同turn的独立角色选择不意味着整个episode由教师或学生二选一。衰减结束后$\beta=0$，末期回到纯学生rollout；教师模型仍用于评分。

### 已有工作：OPD、教师帮助与混合历史

[GKD](https://arxiv.org/abs/2306.13649)奠定学生上下文监督，[MiniLLM](https://arxiv.org/abs/2306.08543)研究reverse KL。TCOD等以阶段/前缀组织教师帮助，Guided-OPD侧重每turn随机选择与课程退出。与[UOPD](#paper=paper-uopd)的“先学生proposal，再按不确定性接管”不同，这里角色选择不先依赖某个候选动作的分数；二者实际执行分布也因此不同。

## 2. 方法与实现机制

### 执行者选择不是训练mask选择

![Figure 2 · 混合执行、非对称KL与课程衰减](./assets/papers/paper-guided-opd/method-source.png)

**Figure 2解读。** 每turn从共享历史出发，随机选择教师或学生生成，执行其动作得到新观察；后续模型看到的是已经实际执行的混合历史。对应分支使用不同KL，课程降低教师选择概率。图不是“教师只在loss侧给建议”：教师turn直接影响环境。图源固定v2的figures/figure2.pdf、acl_latex.tex中的fig:method，对应原文Figure2。

训练step$k$先确定$\beta_k$，每turn采$z_t\sim\mathrm{Bernoulli}(\beta_k)$，$z_t=1$用教师$q$，否则学生$p$。若$a_t$是完整turn响应，执行策略为：

$$
\mu_k(a_t|h_t)=\beta_k q(a_t|h_t)+(1-\beta_k)p(a_t|h_t).
$$

$h_{t+1}$由实际$a_t$与环境观察更新，包含过去各turn的随机角色。它的占用分布是混合策略递推得到的$d_{\mu_k}$，一般**不是**$(1-\beta)d_p+\beta d_q$两个纯策略整条轨迹分布的线性混合。例如两turn角色可以SS/ST/TS/TT四种路径；只混合纯学生与纯教师轨迹会遗漏ST/TS。

### 两种loss及梯度边界

学生turn沿学生生成token前缀，在局部词表上reverse KL；教师turn在教师生成的响应前缀上forward KL：

$$
\ell_s(h)=D_{\rm KL}(p_\theta(\cdot|h)\|q(\cdot|h)),\qquad
\ell_t(h)=D_{\rm KL}(q(\cdot|h)\|p_\theta(\cdot|h)).
$$

按角色加权后整体训练使用混合历史上的两类局部监督。教师冻结，历史通常视作已采样数据；如果通过sampled policy-gradient surrogate实现，需保留行为概率与detach的teacher信号。论文公式阐明方向，但本轮未取得公开作者代码，不能擅自宣称一定做全词表FKL或以某个SFT函数替代。

在教师token按$q$采样时，$-\log p(a)$的条件期望梯度对应forward cross-entropy，教师熵与学生参数无关；这可以帮助理解teacher turn。但一条SFT样本不是完整词表监督，其方差、过滤与归一化仍不同。学生sampled scalar RKL更不能简单autograd当作完整正确梯度，详见[Agentic OPD](#q=agentic-opd-training)。

### 课程公式与零端点

默认余弦衰减从1到0，前$\rho K$个训练step完成退出：

$$
\beta_k=\frac{1+\cos(\pi\min(k/(\rho K),1))}{2},\qquad
K=250,\quad \rho=0.8.
$$

因此$k=0/100/200/250$对应1/0.5/0/0。前期执行更依赖教师，后50step完全学生执行，但继续请求教师监督。这是控制rollout状态分布，不是同时调学习率或蒸馏temperature。

论文另列指数课程，涉及正端点的对数/比值表达。若末端直接设$\beta_{\rm end}=0$，原表达不能直接代入；指数消融的floor/clamp等实现需要作者代码或进一步说明，不能以教学自选epsilon当成论文真实配置。默认余弦本身无此问题。

### 一个两轮任务怎样理解指导

学生第一轮可能打开错误页面。随机教师第二轮接管后重新定位目标，第三轮学生再继续。这条历史既不同于纯教师成功轨迹，也不同于学生始终犯错的轨迹。训练希望学生在这些中间可恢复状态学到下一步，而非只背完整教师解答。$\beta$降低后，模型必须自己完成修复；评测不应继续拿教师动作帮忙。

若只在教师turn打一个训练role字段，却仍执行学生原动作，状态分布并未实现论文方法。若把teacher response写进文本但sandbox实际文件状态还是学生改动，也会产生不可实现的历史。Coding迁移需要同时切换动作执行与可恢复环境状态，不只换自然语言trace。

### 教学伪代码与未公开代码状态

```python
# Teaching pseudocode based on Algorithm 1, not author source.
beta = cosine_schedule(step, total=250, decay_fraction=0.8)
history = env.reset(task)
while not env.done:
    role = "teacher" if bernoulli(beta) else "student"
    response = generate(role, history)
    record(history, response, role, policy_version)
    observation = env.execute(parse_action(response))
    history = append(history, response, observation)
# Build teacher-turn FKL and student-turn RKL with valid assistant masks.
# Freeze teacher; update student; next batch uses the new student.
```

record至少包含调用实际prompt、tokens、role、行为logprob、版本与终止原因；不同role的采样分布不同，不能全部写student old logprob。response含thought/action而观察不计学生生成；role选择作用在turn而非token。训练究竟采用怎样的surrogate、缓存和mask需核具体实现。

原文给出[Zzzz-166/Guided-OPD](https://github.com/Zzzz-166/Guided-OPD)，本轮公开API及README返回404，未取得可固定作者实现。只能记录此公开入口状态，不据404断言项目必然私有或永远未发布，不伪造源码函数。

## 3. 实验设置与算力

教师是Qwen3-30B-A3B基础模型，学生0.6B/1.7B/4B；不是UOPD和ActFirst中的任务RL教师。三个环境ALFWorld、ScienceWorld、WebShop，ALF含IID/OOD列；统一均值的构成也不同：Avg Score只平均Science/Web，Avg SR与round平均四个测试列。

| 配置项 | 固定v2 |
| --- | --- |
| 教师/学生 | Qwen3-30B-A3B / Qwen3-0.6B、1.7B、4B |
| 框架 | TrinityRFT |
| GPU | 一节点8H20，96GB |
| 分配 | 4学生rollout、2教师评分、2learner |
| 并行 | TP2，Ulysses SP2 |
| 精度 | BF16 |
| 动态batch | 每GPU 16384 tokens，推理显存比例0.7 |
| 优化器 | AdamW，LR1e-6，grad clip1 |
| loss | KL系数1，更新重复1 |
| 异步控制 | 最大版本差2，不能脱离框架语义改作样本百分比 |
| step | 250，rollout batch16，train batch64 |
| 长度 | prompt10240，response512 |
| 训练/评测 | 温度1.0/0.4，seed42，eval每5step |
| turn上限 | ALF30、Science30、Web15 |
| 默认课程 | cosine，beta1→0，rho0.8 |

Teacher model是否具有任务适配，是与其他Agentic OPD论文比较必须记录的因素。模型更大不代表在任意学生错误状态都可靠；PPL曲线作为分析依据需和实际任务成功一起读。ALFWorld score未单列，因此不能用“所有任务平均score”解释该论文Avg。

八H20配置与同卡时成绩并非同一问题：论文主表关注各模型任务score/SR/round，不能把它与八A100 ActFirst的训练时间、八Blackwell UOPD的吞吐放进一列统一比。共享名称OPD不意味着teacher、任务划分、版本规则和解码预算一致。未披露项不以其他工作默认值补齐。

本报告只手算课程与KL梯度，并核论文、图表；未训练模型。课堂CPU脚本给出beta关键step和role-dependent执行分布例子，但不代表原环境的学习结果或吞吐。

Qwen3-30B-A3B的A3B表示每token激活参数量，不代表只加载3B权重；教师驻留显存、TP与评分batch仍按完整模型结构配置。算力账本要区分teacher评分和teacher生成，课程退出减少的是教师执行比例，不一定消除评分成本。

## 4. 结果与图表解读

![Table 1 · 三环境与三学生规模](./assets/papers/paper-guided-opd/table-1-pdf.png)

**Table 1解读。** 0.6B vanilla Avg Score25.16、SR11.30、round21.68，Guided34.04、15.01、19.96，分别约35.3%/32.8%相对提升、7.9%轮数下降；不是提高35.3/32.8个百分点。1.7B Score32.27→37.94，4B38.66→42.64。4B Avg Score超过教师41.40，只在这些设置和指标成立，不能推成学生所有能力超教师。各环境SR与round需分列读，轮数短也可能来自提前失败。[原表PDF第7页](https://arxiv.org/pdf/2606.15912v2#page=7)。

![Table 2 · KL方向、指导概率与课程形状消融](./assets/papers/paper-guided-opd/table-2-pdf.png)

**Table 2解读。** 固定1.7B，默认37.94/19.93/18.73。交换KL角色降到33.32/15.47/23.30；两分支都RKL为36.58/17.83，两分支FKL为34.56/17.02。固定beta0.25/0.5/0.75都低于默认，linear36.12、exp34.44。表支持“非对称方向+退出课程”的联合设计，不能从一个固定beta失败推断所有固定策略都无用。Exp实现零端点问题另列复现缺口。原表裁图完整保留左栏caption、末行与数值。[原表第7页](https://arxiv.org/pdf/2606.15912v2#page=7)。

![Figure 1 · 教师PPL随turn与指导状态示意](./assets/papers/paper-guided-opd/diagnostic-source.png)

**Figure 1解读。** 上半纵轴teacher PPL越低越好，横轴交互turn；step0及vanilla OPD后期的曲线随turn升高，Guided后期较低。下半是作者对熟悉状态区域的示意，黑段学生、红段教师。PPL是给定文本上的概率诊断，不是完整环境状态密度，也不等于task reward；示意的蓝色区域不能当实测高维state投影。图支持作者的教学直觉，因果机制还需任务级消融和状态诊断。图源figure1.png、fig:teaser。

三类证据应分开：主表回答无指导学生的任务表现，消融回答配方选择，PPL回答代理诊断。Teacher介入成本与系统GPU占用未以同一总预算覆盖全部比较，因此新项目还需报告教师动作生成、评分调用和rollout长度对成本的影响。

## 5. 局限、结论与后续阅读

Guided-OPD主动改变训练时的状态访问，把随机teacher turn与student turn配不同KL，再逐渐退出。它的理论直觉并不保证teacher在错误状态永远可靠；转到coding还需环境可恢复性与状态同步。词表共享方法不能直接用于跨tokenizer，不同role的概率和loss更不能混为一套behavior字段。

可对照[UOPD](#paper=paper-uopd)的proposal-dependent gate、[ActFirst](#paper=paper-actfirst-opd)的执行与训练响应分离，以及[Agentic OPD教学](#q=agentic-opd-training)的梯度检查。当前缺官方源码与指数零端点说明，复现需要保留这些证据缺口。

**Q01 指导发生在哪里？** 整turn执行者选择，真实改变环境历史。

**Q02 混合历史等于纯教师/纯学生轨迹混合吗？** 不等于，可出现ST/TS等混合路径。

**Q03 两分支KL是什么？** 学生turn RKL，教师turn FKL。

**Q04 教师turn SFT一定是作者实现吗？** 条件期望梯度有联系，公开代码不可核，不能断言。

**Q05 beta何时退出？** 默认250step前200完成1→0，最后50学生执行。

**Q06 PPL低表示任务必对吗？** 不表示，它是代理诊断。

**Q07 Avg Score平均几个测试列？** 两任务score；SR/round则四列。

**Q08 指数末端0能直接代公式吗？** 不能，需floor/clamp或其他实现说明。

**来源。** [固定v2全文与附录](https://arxiv.org/abs/2606.15912v2)、原图表与上述仓库状态核查；未进行模型复现，原图版权归作者。
