---
title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
authors: ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"]
affiliations: ["Stanford University", "CZ Biohub"]
author_affiliations: [[1], [1], [1], [1, 2], [1], [1]]
venue: "NeurIPS 2023"
year: 2023
published: "2023-05-29"
summary: "由KL正则化最优策略和Bradley–Terry偏好模型推导DPO，解释隐式奖励、beta、长度与mask，分开离线分类拟合和在线探索。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-09"
template_version: 5
depth_standard: "ddpm"
draft: false
id: "paper-dpo"
paper_title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
paper_url: "https://arxiv.org/abs/2305.18290v3"
method_figure: "./assets/papers/paper-dpo/figure-1-source.png"
method_caption: "Figure 1 显式RLHF与直接偏好优化流程"
github_url: "https://github.com/eric-mitchell/direct-preference-optimization"
tags: ["DPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
---

## 1. 背景与已有工作

### 问题背景

已有同一prompt下的偏好pair，传统RLHF仍要先学reward，再生成新回答、评分并在线更新策略。奖励拟合与策略优化会分别引入误差，系统还需协调采样、critic、KL和权重版本。DPO问的是：能否把这份已收集的比较数据直接变成策略训练目标，减少中间训练阶段。

这份问题不等同于“如何探索从未出现过的解题路径”。原始DPO使用固定偏好数据，主要目标是把已有相对偏好转成行为。数据生成与人类标注成本仍存在，噪声与覆盖不足也不会因为loss简单而消失。

### 前置知识

**偏好样本。** 一条记录是$(x,y_w,y_l)$：同一prompt、较受偏好的回答、较不受偏好的回答。winner只是相对更好，并非每token都正确；两答案可能都有错误。pair分类与只拟合chosen的SFT不是同一目标。

**自回归似然与teacher forcing。** 完整回答logprob是有效回答token的logprob求和，$\log\pi(y\mid x)=\sum_t\log\pi(y_t\mid x,y_{<t})$。teacher forcing把已知前缀送入模型，计算下个真实token的概率，不需要重新采样整条回答。prompt和padding是条件或补齐，不计入answer loss；一位移位与EOS约定须一致。

**参考分布与KL。** reference是初始行为锚点，通常冻结；当前policy参与梯度。$D_{KL}(\pi\Vert\pi_{ref})$度量当前分布偏离reference的程度，方向不能互换。这里的policy/reference分数不是PPO当前/behavior比，reference没有承担“记录本轮真实采样分布”的职责。

**sigmoid与二元交叉熵。** $\sigma(z)=1/(1+e^{-z})$把分数差转成概率，$-\log\sigma(z)$鼓励winner分数高于loser。错误排序时梯度较强，已获得大正margin时梯度减小。logsigmoid是稳定计算形式，不应先sigmoid再log造成下溢。

**奖励差值与归一化常数。** 同一prompt下所有回答reward加同一个$c(x)$不改变偏好。后面推导中的$Z(x)$负责把完整回答概率归一化，比较同prompt两个回答时会抵消；这一步是DPO成立的关键，不是直接忽略一个难算项。

### 已有工作与本文位置

[InstructGPT](https://arxiv.org/abs/2203.02155)等显式RLHF先学偏好reward再用PPO；Preferred-FT只学chosen，朴素unlikelihood同时增强winner与压低loser。DPO借KL正则化最优策略和Bradley–Terry模型重参数化reward，形成有自适应权重的pair loss。[固定v3 §3—§5](https://arxiv.org/pdf/2305.18290v3)。

| 路线 | 关键依赖 | 与DPO比较时的重点 |
| --- | --- | --- |
| SFT / Preferred-FT | 正例答案 | 缺少相对reference的负例margin |
| 显式RLHF / PPO | reward拟合与在线采样 | 在线探索、critic和优化误差 |
| DPO | 偏好pair与reference分数 | 简化训练，但依赖离线覆盖 |
| Best of N | 推理时采样和scorer | 选择成本不等于策略训练成本 |

题名的“隐式reward model”指策略可以表达一个奖励等价类，不意味着已经得到可靠的通用评分器；跨prompt、跨reference的reward数值不能直接排序。

## 2. 方法与实现机制

### 从 KL 正则化目标得到最优策略

给定 prompt $x$、回答 $y$、参考策略 $\pi_{\mathrm{ref}}$，传统目标可以写成：

$$
\max_\pi\;\mathbb E_{x,y\sim\pi}[r(x,y)]
-\beta\mathbb E_xD_{\mathrm{KL}}\bigl(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\bigr).
$$

这里 KL 方向是当前策略到 reference，$\beta>0$ 在此定义中是约束强度。对完整回答分布求解，最优策略为：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right),\qquad
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}.
$$

直观上，参考模型提供初始支持，奖励通过指数倾斜改变回答概率。较大的 beta 在这个固定 reward 的理想问题中使倾斜更温和。由于所有回答空间极大，直接计算 $Z(x)$ 不现实，但 DPO 接下来利用同 prompt 两回答的比较让它消失。

将式子取对数并解出奖励：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

给某个 prompt 下所有回答的 reward 都加同一个 $c(x)$ 不改变排序，也不改变该目标的最优策略。这就是奖励的等价类：偏好数据识别的是差值，无法唯一识别绝对平移。DPO 用可以归一化成策略的代表奖励表达这个等价类；不能用该平移自由度宣称任何 noisy 数据都能精确拟合。

![Figure 1 · 先奖励建模再RL与直接偏好优化的原始流程对照](./assets/papers/paper-dpo/figure-1-source.png)

**Figure 1 解读。** 左侧偏好数据先拟合 reward model，语言模型持续产生新 completion，reward model 为这些 completion 打分并驱动 RL；右侧同样需要偏好数据，但直接更新最终 LM。图只概括偏好优化阶段，未画出所有 SFT、reference、mask 和优化器细节。右侧没有循环不等于生成数据或推理不需要采样，也不等于 DPO 不需要参考分布。[图源：v3 Figure 1、源码 `figures/diagrams/teaser.png`](https://arxiv.org/src/2305.18290v3)。

### Bradley–Terry 如何把奖励变成二元标签

对同一 prompt 的 winner $y_w$ 和 loser $y_l$，Bradley–Terry 假设偏好概率为：

$$
P(y_w\succ y_l\mid x)=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr).
$$

把重参数化 reward 代入，两项 $\beta\log Z(x)$ 抵消。再把未知最优策略替换成参数模型，得到 DPO loss：

$$
L_{\mathrm{DPO}}(\theta)=-\mathbb E_{(x,y_w,y_l)\sim D}
\log\sigma\left[\beta\left(
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right].
$$

需要的量是 policy 的 chosen/rejected logprob 和 reference 的 chosen/rejected logprob，共四个序列分数。reference 冻结，策略前向 teacher forcing；不用在每个更新步重新生成回答，也不需要学习 value。这里的 policy/reference 比不是 PPO 的 current/behavior 比，原始 DPO 不通过重要性采样校正在线轨迹分布。

完整回答的 logprob 是有效回答 token 的 logprob **求和**：$\log\pi(y\mid x)=\sum_t\log\pi(y_t\mid x,y_{<t})$。prompt token 只是条件，padding 不参与分数，EOS 是否计入必须一致。若擅自改成长度平均，相当于改变原始目标，不能说只是实现更稳定而数学完全一样。序列较长时分数更负并不意味着模型更不喜欢它；原目标同时比较 policy 与 reference 的相对变化，仍可能受到长度和数据选择偏差影响。

### 梯度、隐式 reward 与 beta 的两个角色

定义 $\widehat r_\theta(x,y)=\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$、$m=\widehat r_w-\widehat r_l$，有：

$$
\nabla_\theta L=-\beta\sigma(-m)
\left[\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\right].
$$

错误排序时 $m<0$，权重较大；已建立大正 margin 时权重较小。这个权重解释 DPO 为什么不同于无条件把 positive likelihood 加上 negative unlikelihood。chosen 的梯度通常增加其相对倾向，rejected 的梯度降低其相对倾向，但由于参数共享与归一化，不能保证每一步 chosen 的绝对概率都上升，更不能保证所有未见回答的质量改善。

手算：policy 的两个 logprob 为 $[-2,-3]$，reference 为 $[-2.5,-2.5]$，beta 为 0.1，则相对 logprob 为 $[0.5,-0.5]$，隐式 reward 为 $[0.05,-0.05]$，margin 为 0.1，pair loss 约 0.6444。若两模型初始化相同，margin=0、loss=$\log2$，**梯度仍非零**，因为 policy 的差值可微。若固定所有差值后令 beta 趋近于零，梯度也趋近零；所以“beta 越小就无限强化偏好”是错误理解。

beta 在最初 KL 目标中控制约束，在实际 classification loss 中又缩放 logit 与梯度。固定 reward 的理论最优策略随 beta 的趋势，不等于有限数据、有限步数训练的所有行为都单调。调 beta 应一起观察 held-out preference、KL、回答长度和下游质量，不只看 rewards/margins 越大越好。训练 margin 可被过拟合放大。

### 实际数据流与最容易写错的位置

```python title="DPO 教学重述 · 流程示意"
# 原始DPO的教学重述；每条序列只统计answer部分
pi_w = sum_answer_logp(policy, prompt, chosen)
pi_l = sum_answer_logp(policy, prompt, rejected)
with no_grad():
    ref_w = sum_answer_logp(reference, prompt, chosen)
    ref_l = sum_answer_logp(reference, prompt, rejected)
margin = beta * ((pi_w - pi_l) - (ref_w - ref_l))
loss = -logsigmoid(margin).mean()
loss.backward()
```

优先用 `logsigmoid`，不要先 sigmoid 再 log 导致极端 logit 下溢。偏好 pair 必须共用相同 prompt；tokenizer、chat template、截断策略和 response mask 在 policy/reference 间一致。如果 reference logprob 缓存，应绑定 reference checkpoint 与预处理版本，不能更新模板后继续用旧缓存。chosen 与 rejected 拼接前向可以减少 FSDP all-gather 次数，但显存仍随有效 token 和激活增长。

离线 reference 缓存或 PEFT 共享冻结 base 可以降低资源；它们是部署实现策略，原算法仍需要 reference 分数。reference-free、label smoothing、IPO、length-normalized DPO 和 online DPO 是不同配置或后续变体，不要读到同一个 trainer 支持它们就将其全部写进 2023 原论文。

### 官方源码与公式对照

固定 [作者仓库](https://github.com/eric-mitchell/direct-preference-optimization)提交 `f8b8c0f49dc92a430bae41585f9d467d3618fe2f`，核心文件为 [trainers.py](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py)。这个快照已含后续选项，核对原始 DPO 时应使用 `label_smoothing=0, ipo=False, reference_free=False`。

`_get_batch_logps` 接收 `[B,L,V]` logits 和 `[B,L]` labels，错开一位实现自回归预测，忽略 -100 标签并 gather 目标 token logprob；默认 sum，`average_log_prob=True` 则改变 reduction。`concatenated_forward` 先把 chosen/rejected 补齐到同一长度，拼成一次前向，再拆回 `[B]` 的两组序列分数。`preference_loss` 计算两组 policy 差与 reference 差，再通过 `-logsigmoid(beta*logits)` 得到逐 pair loss；输出的 implicit rewards 使用 detach 用于日志，而不是额外训练 reward head。

`get_batch_metrics` 在 reference 的 no-grad 前向后调用这个 loss，统计 chosen/rejected rewards、margin 和排序准确率。这些是偏好拟合诊断，不能与 GPT-4 win rate 或数学 Pass@1 混为一谈。本文静态核读，不声称执行训练。[Hugging Face 的 DPO/TRL 讲解](https://huggingface.co/blog/dpo-trl)采用 prompt/chosen/rejected 的组织方式，本文沿用这个易懂的数据入口；其中 Llama 2 + QLoRA 是教程案例，不是 DPO 原论文的实验模型或超参。

### 课堂源码拆解：从token概率到pair分类

先画两条回答输入，每条都是prompt+answer；再画同样的reference前向。训练最终得到四个 `[B]` 序列分数，loss不接收新的rollout，也不接收value预测。先让学生理解teacher forcing与mask，再推导margin，否则很容易把DPO误写成逐tokenchosen/rejected比较。

#### 有效回答如何得到一个logprob

**真实源码节选：[DPO 作者实现 · 取回答logprob](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L103-L115)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DPO 作者实现 · 取回答logprob"
labels = labels[:, 1:].clone()
logits = logits[:, :-1, :]
loss_mask = (labels != -100)

# dummy token; we'll ignore the losses on these tokens later
labels[labels == -100] = 0

per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

if average_log_prob:
    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
else:
    return (per_token_logps * loss_mask).sum(-1)
```



输入logits形状 `[2B,L,V]`，labels是 `[2B,L]`。labels去掉首位、logits去掉末位，实现“位置t预测t+1”；`-100`标记prompt或padding。先把-100替成合法gather索引0，随后用mask去除，既避免索引错误，也不训练这些位置。default分支求和给完整answer的logprob；average分支并非原始DPO调用选择。

例如answer只有两个有效token，其logprob为-0.3和-0.7，返回-1.0；前面的prompt概率不能被加进来。chosen/rejected有不同长度也必须使用一致tokenizer、EOS和截断规则。reference缓存还要绑定这些规则，不能换模板继续复用旧分数。

#### 四个分数怎样变成偏好目标

**真实源码节选：[DPO 作者实现 · 偏好margin与loss](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L70-L87)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DPO 作者实现 · 偏好margin与loss"
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = reference_chosen_logps - reference_rejected_logps

if reference_free:
    ref_logratios = 0

logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

if ipo:
    losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
else:
    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

return losses, chosen_rewards, rejected_rewards
```



原始DPO对应`reference_free=False, ipo=False, label_smoothing=0`。这三个现代分支与原算法分开；不能在课上把仓库支持的全部选项说成同一条推导。

| 运算 | 读法 | 为什么必要 |
| --- | --- | --- |
| policy chosen−rejected | 当前两答案的相对logprob | 只提升chosen不是相对偏好目标 |
| reference chosen−rejected | 原分布对两答案的相对偏好 | 校准原有概率差异 |
| 两者再相减 | 策略比原分布更偏向winner多少 | 对应隐式reward差 |
| beta与logsigmoid | 将margin映射成偏好概率 | 错排序强更新，正确大margin弱更新 |
| rewards的detach | 只用于日志的隐式分数 | 不额外训练reward head |

把四分数设为policy[-2,-3]、reference[-2.5,-2.5]，beta0.1，手算margin0.1、loss约0.6444。再让两模型分数相同：loss为log2但actor梯度非零。最后问“chosen绝对概率是否一定上升”：目标只要求relative margin，参数共享使绝对量不必逐步单调。

#### 讲解的收束

从KL最优策略、reward重参数化、同prompt常数抵消、pair logistic loss，一直走到shift/mask/sum与四分数。这条链讲完整，才解释了DPO为何简单，也解释了为什么离线覆盖、reference和偏好质量仍不可省略。


## 3. 实验设置与算力

原论文分别研究可控情感、摘要、单轮对话。前者有已知 scorer，可画 reward/KL 前沿；后两者主要由 GPT-4 比较模型回答与参考回答。三类指标不是同一单位，模型也不同。作者没有用一个模型、一个共同预算给出所有任务上的无条件 DPO/PPO 优劣。

| 项目 | v3 配置或披露 | 出处与解释 |
| --- | --- | --- |
| IMDb sentiment | GPT-2-large；前缀 2—8 tokens | 附录 C；情感 classifier 作 ground-truth reward |
| 情感偏好数据 | 25k prefixes，每个 4 completions、6 个 pair | 约 150k 比较，本文算数；不同 pair 非独立 |
| sentiment reference | 在 IMDb 上 SFT 的模型 | 同一生成初始分布用于比较 |
| TL;DR 摘要 | GPT-J 6B SFT checkpoint | Reddit TL;DR 人类偏好；不是 Llama 2 |
| 单轮对话 | Pythia-2.8B，先 Preferred-FT 再 DPO | Anthropic HH 的单轮子集 |
| beta | 默认 0.1；摘要用 0.5 | 附录 B，不能套教程 0.1 到全部任务 |
| 优化器 / LR | RMSprop，$10^{-6}$ | 原论文默认，现代 trainer 常用 AdamW |
| batch / warmup | batch 64；150 steps 线性 warmup | 附录 B |
| 比较方法 | SFT、Preferred-FT、Unlikelihood、PPO、PPO-GT、Best of N | 各任务参与的 baseline 不完全相同 |
| 自动评测 | GPT-4-0314；随机化 A/B 顺序 | judge prompt 会影响结果 |
| 摘要 / 对话解码 | 扫描不同 temperature | 不能只记每条曲线的最优点 |
| GPU 型号、数量、precision | 未完整披露逐实验可复算账本 | 不补写 A100 卡数或 BF16 默认 |
| 墙钟 / GPU-hours | 未完整披露 | 原论文简化工程不等于精确总算力倍数 |
| 作者代码 | 固定 commit，见方法模块 | 本文未重新训练或重新评测 GPT-4 |

情感数据由 classifier 对同模型回答排序，属于受控代理反馈，不是全部人类标注。摘要偏好由此前工作收集，其生成模型与本文初始化模型并不完全相同；对话把 chosen 用于 Preferred-FT 是为了先把数据带入参考模型支持范围。DPO 的便利依赖已有覆盖合适回答的 pair，不应该隐去数据生成和筛选成本。

摘要与对话评测使用不同 baseline：摘要比较 human-written summary，对话比较 HH 测试集 chosen response。胜率 61% 意味着评测比较中被 judge 选中的比例，不是 61% 题目“正确”。对话 baseline 的 PPO 模型来自外部 6B checkpoint，作者未找到胜过 2.8B base 的合适设置，因此用了 Best of 128 作粗略性能参照；这不是完整公平的同规模 PPO 重训对照。

人工研究有 25 名志愿者，其中 1 名结果未进入分析，每人原定 25 判断。Table 2 的 N respondents 行给出各比较的判断计数，不能误写成分别有 272、122、199 名独立参与者。每次显示回答顺序随机化，GPT-4 的两个 prompt 分别偏向一般摘要质量与准确简洁；这些选择对 win rate 有实际影响。

## 4. 结果与图表解读

![Figure 2 · 情感任务的reward与KL前沿以及摘要胜率](./assets/papers/paper-dpo/figure-2-source.png)

**Figure 2 解读。** 左图横轴是相对 reference 的 KL，纵轴是 sentiment scorer reward；黄色 DPO 在给出的前沿上优于多种 PPO 配置，比较包括可直接访问 ground-truth scorer 的 PPO-GT。右图横轴是 sampling temperature，纵轴是相对参考摘要的 GPT-4 win rate；temperature=0 时 DPO 约 61%，PPO 约 57%，差约 4 个百分点。两图分别回答优化前沿与实际摘要质量，不能把左图奖励当成数学正确率。[图源：v3 Figure 2，两个源码面板完整保留](https://arxiv.org/src/2305.18290v3)。

右图说明 DPO 在本任务上对温度更稳健，但其高温表现也会下降。PPO 的高温曲线退化较明显，这反映这套模型、训练和解码组合；不能推出所有后续 PPO 在非零 temperature 下必然失败。Best of 128 的曲线还包含很高推理采样成本，与生成一条 DPO 回答的部署成本不同。

![Figure 3 · 单轮对话胜率及其随训练步数的变化](./assets/papers/paper-dpo/figure-3-source.png)

**Figure 3 解读。** 左图横轴是 temperature，纵轴是对 HH chosen 的 win rate，DPO 在合适温度下接近或超过 Best of 128；右图横轴是 fine-tuning step，两条线对应 1.0 与 0.7 温度，早期快速改善后波动。原图右面板是训练步数，不能因为 caption 表述而误读为另一张温度曲线。图支持有效的离线偏好拟合，并提示训练更多步不保证质量持续提高。[图源：v3 Figure 3](https://arxiv.org/src/2305.18290v3)。

![Table 1 · 从Reddit摘要迁移到CNN DailyMail新闻的评测](./assets/papers/paper-dpo/table-1-pdf.png)

**Table 1 解读。** 行为 DPO/PPO，列为 0 与 0.25 温度；相对新闻参考摘要的 GPT-4 胜率分别是 DPO 0.36/0.31、PPO 0.26/0.23。贪心差为 10 个百分点，仍都低于 50%，所以“比 PPO 更好”不等于“新闻摘要已胜过参考答案”。这是一个输入分布迁移实验，不是普遍 OOD 保证。[表源：v3 第 9 页](https://arxiv.org/pdf/2305.18290v3#page=9)。

![Table 2 · GPT4与人类偏好判断的对照](./assets/papers/paper-dpo/table-2-pdf.png)

**Table 2 解读。** 每列是某生成方法与 greedy PPO 的比较，三类 win rate 分别来自两种 GPT-4 prompt 和人类判断。DPO 人类胜率 58%，GPT-4(C) 为 54%，GPT-4(S) 为 47%，说明 judge prompt 足以改变结论强度；DPO 列的人类与人类 agreement 为 65%，与 GPT-4(C) 为 67%。这支持把 GPT-4 作为该实验中的有用代理，但不证明 judge 没有长度、风格或事实判断偏差。[表源：v3 第 10 页](https://arxiv.org/pdf/2305.18290v3#page=10)。

结果支持的是“这些偏好数据和中小规模模型上，简单分类路线足够有效”。受控 scorer 上的前沿优越、摘要 win rate、对话与 Best of N 的比较，证据类型不同。论文没有展示 coding agent 的在线环境探索、长期工具调用或当代百亿到万亿规模全量预算对照；把这种范围差异写清楚，才不会让面试答案变成 DPO 无条件替代 PPO。

## 5. 局限、结论与后续阅读

DPO 将偏好 reward 的重参数化与策略学习连起来，降低在线 RL 系统的复杂度。理论闭式解涉及可表达的完整策略分布、正的 reference 支持、合适的偏好模型等假设；现实中模型有限、标签有噪声、pair 分布不覆盖新策略。优化分类 loss 与得到全局理想策略之间还存在统计和建模误差。

### 面试问答与追问

**Q1：DPO 为什么不需要显式 reward model？** 由 KL 正则化最优策略把 reward 写成 beta 倍 policy/reference log-ratio，代入 Bradley–Terry 后同 prompt 的配分函数抵消。reward 并没有消失，而是隐式参数化在策略中。追问应能从最优策略推到 pair logit。

**Q2：DPO 是 SFT 吗？** 实现可以使用 teacher forcing 和监督式 minibatch，但目标同时依赖 chosen、rejected 与 reference，学的是偏好差值；原始 SFT 只拟合目标回答 logprob。把两者都叫“离线训练”不足以说明区别。

**Q3：beta 越小，模型变化一定越大吗？** 固定真实 reward 的理论解中 beta 越小倾斜越强；实际 DPO 的 beta 同时缩放 logit 与梯度，有限训练不保证单调，beta=0 的原始 loss 甚至无有效策略梯度。要区分两个问题中的量和假设。

**Q4：chosen 概率下降是否一定写错了？** 不一定。目标约束的是相对 reference 的 pair margin，参数共享也会影响两回答。应检查 margin、mask、绝对 logprob 和实际质量；不能只凭单个 chosen logprob 判定梯度符号错误。

**Q5：为什么仍要 reference？** 它提供奖励参数化的锚点，校准回答原有概率。reference logprob 可以预计算但仍属于目标。reference-free 是另一个假设，不应把节省一次前向说成原始算法天然不需要 reference。

**Q6：固定偏好数据有什么限制？** 无法自然探索数据中未覆盖的新行为，容易继承生成模型与标注分布偏差。online DPO 或迭代偏好生成可以改变数据来源，但它们增加了新采样与标注环节，不能直接算入原论文成本优势。

**Q7：训练 preference accuracy 达到100%是否完成对齐？** 只说明这些 pair 的 implicit score 排序匹配标签。它不保证 factuality、工具成功率、OOD 表现或避免 reward hacking。应使用独立任务评测及长度、KL、judge 偏差诊断。

**Q8：什么时候优先选 DPO？** 已有较干净、覆盖目标行为的偏好 pair，想用相对简单的离线训练改进风格、帮助性或领域偏好时，适合作为 baseline；若主要缺口是在线探索新解题路径，应同时考虑 PPO/GRPO 路线。此为基于数据需求的工程判断，不是跨论文成绩排名。

参考讲解包括 [Hugging Face DPO with TRL](https://huggingface.co/blog/dpo-trl)的数据组织、[作者实现](https://github.com/eric-mitchell/direct-preference-optimization)的四 logprob 接线；实验数字和插图均回到固定 v3。后续对照 [PPO](#paper=paper-ppo)、[GRPO原文](#paper=paper-grpo)和[统一比较](#report=survey-policy-optimization)。2026-10-08 核读正文、理论附录、实验细节与源码，未进行个人训练复现。




**2026-10-09更新。** 背景拆为问题背景、前置知识、已有工作；补固定源码节选、逐段形状/梯度讲解与课堂检查。源码节选不是完整可运行训练程序；课堂张量练习见[CPU演示脚本](./assets/learning/policy-optimization-lab.py)，不下载模型且不执行真实RL训练。
