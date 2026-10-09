---
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
authors: ["Zhihong Shao", "Peiyi Wang", "Qihao Zhu", "Runxin Xu", "Junxiao Song", "Xiao Bi", "Haowei Zhang", "Mingchuan Zhang", "Y. K. Li", "Y. Wu", "Daya Guo"]
affiliations: ["DeepSeek-AI", "Tsinghua University", "Peking University"]
author_affiliations: [[1, 2], [1, 3], [1, 3], [1], [1], [1], [1], [1], [1], [1], [1]]
venue: "arXiv preprint"
year: 2024
published: "2024-02-05"
summary: "以DeepSeekMath原论文解释GRPO如何用同题多回答替代value模型，覆盖outcome/process监督、KL估计、组内零方差和原始奖励模型实验。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-09"
template_version: 5
depth_standard: "ddpm"
draft: false
id: "paper-grpo"
paper_title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
paper_url: "https://arxiv.org/abs/2402.03300v3"
method_figure: "./assets/papers/paper-grpo/figure-4-source.png"
method_caption: "Figure 4 PPO与GRPO的模型和优势计算路径"
github_url: "https://github.com/deepseek-ai/DeepSeek-Math"
tags: ["GRPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
---

## 1. 背景与已有工作

### 问题背景

语言模型PPO常需一个与actor规模相近的value模型。长回答的终局反馈稀疏，每个前缀的未来回报又难估计，critic的显存和前反向开销因此未必带来可靠优势。DeepSeekMath提出：对同一问题多采几条回答，直接用组内表现构建baseline，能否省去独立critic。

本文的最终模型还经历数学continued pretraining和SFT。GRPO负责后续RL优化，而不是从无知识状态创造全部数学能力。报告以真实DeepSeekMath题名收录，重点在§4强化学习和§5.2分析，不把最终MATH成绩全部归因于去critic。

### 前置知识

**策略、终局reward与credit assignment。** 模型的动作是下一token，回答结束后scorer给出一个分数；需要决定这个分数怎样作用到前面的动作。outcome supervision只知道完整结果，process supervision额外给出中间步骤反馈，两者的信息粒度不同。

**value baseline与相对优势。** critic估计$V(s)$，PPO常用GAE比较动作回报和预期。组方法则比较同题的多个完整回答；题A平均0.9与题B平均0.1具有不同baseline。组均值不是逐前缀value，省掉critic也不自动保留GAE全部credit能力。

**分组、标准差与广播。** $G$是同一prompt的回答数，不是batch内所有回答数。UID保证重排后仍找到兄弟样本。组内减均值再除std产生scalar advantage，再广播到有效回答token。总体std与样本std的有限组数值不同；全同reward分子为零，epsilon不能创造区分信息。

**三类模型分布。** current参与本次优化，old描述这组真实生成的概率，reference约束相对锚点的行为变化。old/reference可有不同刷新周期。KL、reward、组优势各在哪一步计算，会改变最终loss，不能只凭名称GRPO判定配方。

**reduction与mask。** 原GRPO先每回答平均token，再平均回答；这与全batch token平均不同。prompt、padding、工具observation不属于actor动作，mask要排除它们。把同一scalar advantage复制到所有token只是一种outcome估计，不等于每个token都得到独立正确性标签。

### 已有工作与本文位置

[PPO](#paper=paper-ppo)给出概率比与clip，REINFORCE提供logprob梯度，[DPO](#paper=paper-dpo)直接拟合离线偏好。GRPO改变的是在线策略梯度的baseline，保留token ratio并在原文中直接加入reference KL。[固定v3 §4](https://arxiv.org/pdf/2402.03300v3#page=13)。

| 机制 | PPO常见做法 | 原GRPO |
| --- | --- | --- |
| baseline | 学习value、GAE | 同题组均值与std |
| outcome credit | 回报与逐状态value | 每回答共享组优势 |
| 更新约束 | current/old ratio与clip | 保留token ratio与clip |
| 新开销 | critic前反向 | G次生成与组完成等待 |

RLVR说明reward可被规则验证，GRPO说明怎样优化。原DeepSeekMath使用学习reward模型；后来的规则判题与GRPO组合，不意味着二者同义。

## 2. 方法与实现机制

### 同题组如何产生优势

对问题 $q$ 用本轮旧策略采样 $G$ 条回答 $o_i$，记录奖励 $R_i$、有效 token mask 和 behavior logprob。outcome supervision 的优势是：

$$
\widehat A_i=\frac{R_i-\mu_q}{\sigma_q+\varepsilon_{\mathrm{num}}},\qquad
\mu_q=\frac1G\sum_{j=1}^G R_j,\qquad
\widehat A_{i,t}=\widehat A_i.
$$

$\varepsilon_{\mathrm{num}}$ 是教学实现中防止零除的数值项，与 PPO clip 的 epsilon 不是同一个参数。组标准差的 population/sample 定义会改变有限组数值，论文简式不规定所有框架细节；手算和代码必须注明 correction。组大小越大通常能提供更丰富比较，但 rollout 成本更高，也不能保证线性降低估计误差。

![Figure 4 · PPO的value路径与GRPO的组内比较路径](./assets/papers/paper-grpo/figure-4-source.png)

**Figure 4 解读。** 上半部分 PPO 从单个回答获得 reward 与 value，经 GAE 计算优势；下半部分 GRPO 一次生成多个回答，reward model 分别评分，再通过 group computation 输出优势。黄色表示训练中的模型，蓝色表示该阶段冻结的模型。GRPO 去掉的是 value 路径，图中 reference 与 reward 仍存在；这不是“只要一个模型就能完成所有训练”的证据。[图源：v3 Figure 4，源码 `figures/GRPO.pdf`](https://arxiv.org/src/2402.03300v3)。

用总体标准差教学手算，奖励 $[0,0,1,1]$ 的均值为 0.5、标准差为 0.5，优势为 $[-1,-1,1,1]$。若采用样本标准差则为约 $[-0.866,-0.866,0.866,0.866]$。两者排序相同，但尺度不同。奖励 $[1,1,1,1]$ 或 $[0,0,0,0]$ 中分子全为零，没有同组区分信号；防零 epsilon 不能把相同 reward 变成学习信息。

这个基线包括当前样本本身，有限组下不同于独立无偏基线或 leave-one-out。忽略 std 时，减去含自身的 group mean 会在理想独立采样条件下引入 $(G-1)/G$ 的梯度尺度因子；加入随机标准差后情况更复杂。不能把去 critic 简写成无偏、无方差代价的免费替换。

### 完整目标与三类策略对象

令 $r_{i,t}=\pi_\theta(o_{i,t}\mid q,o_{i,<t})/\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})$，原论文的目标为：

$$
J_{\mathrm{GRPO}}=\mathbb E\left[\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_t
\left\{\min\left(r_{i,t}\widehat A_{i,t},
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\widehat A_{i,t}\right)
-\beta k_{i,t}\right\}\right].
$$

current 是正在优化的策略；old 是本轮实际采样策略；reference 是对齐约束锚点。current/old 描述经验复用造成的变化，current/reference 描述相对锚点的偏离。二者可以不同步刷新。原文 iterative GRPO 在外层迭代更新 reference，内层每次探索更新 old；不能把 reference 永久固定或每个 minibatch 都更新说成唯一标准。

原文在采样 token 上使用的 KL 形式可写为：

$$
k_{i,t}=u_{i,t}-\log u_{i,t}-1,\qquad
u_{i,t}=\frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})}{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}.
$$

$u-\log u-1\ge0$，$u=1$ 时为零。若动作从当前策略分布采样，其期望对应当前到 reference 的 KL；实际回答由 old 采样、之后反复更新，不能无条件把这个单样本式子称为对当前完整分布始终无偏。是否进行分布校正、将哪一部分 detach、是否使用 full-vocabulary KL，都影响实际梯度与估计口径。

原文把 KL 直接放入目标，使组优势只基于 task reward，避免 reference 惩罚改变组比较。若把 KL 先加进 reward 再组内标准化，就得到不同 advantage；即使都叫 GRPO，信号也不相同。组优势为零时若仍有 KL 项，整体 loss 仍可能有梯度；只有相关项也为零或关闭时才能说这一组没有 actor 训练信号。

### Outcome与process supervision

Outcome reward 在完整回答末尾评分，把标准化后的单个分数广播到所有有效回答 token。这是低成本但粗粒度的 credit assignment：一次错误最终答案会让整条回答得到负优势，其中可能包含正确且有用的中间推理。它提高整条行为的相对概率，不直接证明每个 token 都应受同方向监督。

Process reward 对每个 reasoning step 结束位置评分，原论文在组内所有步骤分数上标准化，再对 token 取后续 step 的归一化奖励之和：

$$
\widehat A_{i,t}=\sum_{j:\,\operatorname{end}(j)\ge t}\widetilde R_{i,j}.
$$

因此前面 token 可以影响更多后续步骤，其优势不再全序列相同。step 边界、process scorer 校准与奖励可被利用的问题成为新依赖。不能把 process GRPO 理解成把同一个 outcome advantage 复制到每个 turn，也不要把它与 value model 的状态回报估计混同。

Iterative RL 进一步根据当前策略输出重建 reward-model 数据，用约 10% 历史数据 replay 持续训练 reward model，再继续策略优化。这里 replay 的对象是 reward-model 训练数据；它不证明 actor 无限复用历史轨迹也安全。长程 coding agent 若保存 partial rollout，还需保留环境状态与行为版本，这不由组优势公式自动解决。

### 长度、难度与组完成依赖

原目标先在每条回答内平均 token loss，再在组内平均回答。因此 100-token 和 1000-token 回答在“总序列权重”上接近，每个短回答 token 权重更大；这与把全 batch token 一次求平均不同。std 标准化又使困难或容易问题的有效奖励尺度与成功率相关；很多后续变体正针对这种 difficulty/length bias 调整 reduction 或归一化。

在线组方法需要同题多个 reward 才能构建 advantage。异步系统即使取消全局 batch barrier，也可能留下 group-completion barrier：最快完成的答案仍要等待同题慢答案。DAPO 过滤全同组解决信号稀疏，不直接消除这种时序依赖。若改成部分组、running baseline 或单 rollout，应重新审视估计器，不能只说系统“流式”而保持数学目标完全不变。

```python title="GRPO 教学重述 · 流程示意"
# outcome GRPO教学重述；std correction需与实际框架匹配
responses, old_logp = sample_group(old_policy, prompt, G)
reward = score(responses)
adv = normalize_within_prompt(reward).detach()
logp = policy.answer_logprob(prompt, responses)
ratio = exp(logp - old_logp.detach())
pg = minimum(ratio * adv[:, None],
             clamp(ratio, 1-eps, 1+eps) * adv[:, None])
loss = mean_of_sequence_means(-pg + beta * sampled_kl, response_mask)
```

工具 observation、prompt、padding、无效截断片段需要明确 mask。behavior logprob 应与真实采样机制对应；temperature、top-p 与推理后端精度差异都可能使“保存的概率”不同于实际动作分布。概率比不是只要两次 forward 形状一样就正确。

### 原始公开资料与现代源码伴读

[DeepSeek-Math 官方仓库](https://github.com/deepseek-ai/DeepSeek-Math)提供模型权重、推理与评测说明，未提供完整原始 GRPO 训练管线。不能用 README 的 `model.generate` 当作官方算法实现。此处核读现代维护者实现，固定 verl commit `75879f7f475fd6b64c779f7d9212e45503f58b8f`，与论文实验版本区分。[核心文件](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py)。

`compute_grpo_outcome_advantage` 接收 `[B,L]` token rewards 和 response mask，用末维求和得到 `[B]` outcome score；按 prompt UID 聚集均值、样本 std，在 no-grad 中得到 scalar advantage 后广播回 `[B,L]`。`norm_adv_by_std_in_grpo=False` 是可选的去 std 变体，不能把开关两边都说成原始 GRPO。单元素组代码使用特殊 mean/std 回退，而不是一个有效的同题多样本相对基线。

`compute_policy_loss_vanilla` 通过新旧 logprob 差取 exp，构造负号版本的两支 clip loss，使用 maximum 与可配置 clip 上下界。`agg_loss` 的 `seq-mean-token-mean` 对应原论文的回答内平均；`token-mean` 对应另一种 token 权重，且分布式使用全局 batch token 总数而非各卡各自平均。三个函数分别实现优势、surrogate 和 reduction，单看函数名 GRPO 不足以判断完整目标。

### 课堂源码拆解：同题统计量不是全batch统计量

取两个prompt，每题三回答，得到B=6条回答；每条有效长度不同，但补齐为 `[6,L]`。先由token reward求和得到 `[6]` 分数，再用UID分成两个G=3组。**先分组、后标准化、最后广播**，不能先在整个batch上求std冒充GRPO。

#### 一道题自己的基线

**真实源码节选：[GRPO 现代verl · 按prompt计算统计量](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L314-L323)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="GRPO 现代verl · 按prompt计算统计量"
for idx in id2score:
    if len(id2score[idx]) == 1:
        id2mean[idx] = torch.tensor(0.0)
        id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)
        id2std[idx] = torch.std(scores_tensor)
    else:
        raise ValueError(f"no score in prompt index: {idx}")
```



字典键是prompt UID，`scores_tensor`是一道题的G个分数。`torch.std`在这个固定实现默认使用样本标准差，G=3时分母是G−1；与教学用population std的尺度不同。G=1分支设置mean0/std1是实现回退，不是凭单回答获得了可靠组相对基线。正常GRPO需至少两个兄弟样本，仍可能遇到全同分数。

#### 标量优势怎样作用到token

**真实源码节选：[GRPO 现代verl · 标准化并广播优势](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L324-L329)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="GRPO 现代verl · 标准化并广播优势"
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask
```



此段先逐回答更新分数，`i`属于哪个prompt由`index[i]`决定；退出循环后再统一广播。每条回答只生成一个scalar advantage，`unsqueeze(-1)`从 `[B]`变 `[B,1]`，乘mask广播为 `[B,L]`。返回两个相同tensor只是outcome接口兼容，不是学出了每token value。

奖励[0,0,1]的样本std优势约[-0.577,-0.577,1.155]；全[1,1,1]为零。换成另一道题[2,2,3]优势相同，说明group baseline去掉题间平移，但并没有保留绝对分数。再看`norm_adv_by_std_in_grpo=False`，尺度不同，应作为去std变体讨论。

#### 接到policy loss之前应检查什么

UID是否在DP重排后保留，advantage是否no-grad，prompt/tool/padding mask是否正确，old logprob是不是实际behavior版本，reduction究竟sequence-mean还是token-mean。组统计、surrogate、KL与reduction是四个独立步骤；写对advantage函数并不代表整套GRPO配方正确。

课堂上最后问“最快的一条回答能否立即训练”：如果其同题兄弟尚未评分，group baseline还不齐。由这个依赖再进入异步系统，才能说明取消全局batch等待不自动取消group-completion barrier。


## 3. 实验设置与算力

原文包含数学数据构建、continued pretraining、SFT 和 RL，下面只用前两阶段建立初始化背景，重点列出 §4.2 的 RL 配置。120B 是数学 corpus 规模；500B 是 continued-pretraining 消费的混合 token 总量，涉及多轮数据使用和 code/natural-language 混合，不能将二者都解释成 RL rollout tokens。

| 项目 | 原文 v3 配置 | 出处与边界 |
| --- | --- | --- |
| 初始预训练模型 | DeepSeek-Coder-Base-v1.5 7B | §3.1；continued training 前的初始化 |
| 数学 corpus | 约 120B tokens，筛选 Common Crawl | 与 RL 题目数量不同 |
| continued pretraining | 500B 混合 tokens | 56% math corpus、4% AlgebraicStack、10% arXiv、20% code、10%自然语言 |
| SFT / RL actor | DeepSeekMath-Instruct 7B | RL 接续已完成数学 SFT 的模型 |
| RL 数据 | SFT中约 144K 条 GSM8K/MATH CoT 问题 | 不包含所有其他 SFT 任务 |
| reward model | 初始化自 DeepSeekMath-Base 7B | 初始 LR $2\times10^{-5}$ |
| policy LR | $10^{-6}$ | §4.2 |
| KL coefficient | 0.04 | 原始配方保留 KL |
| 同题采样数 | 64 | 不是推理时多数投票候选数的唯一含义 |
| 最大输出长度 | 1024 | RL阶段配置；模型上下文4K不是同一量 |
| training batch | 1024；每次 exploration 后单次 policy update | 原文未细化所有 distributed microbatch 口径 |
| 测试集 | GSM8K、MATH、MGSM-zh、CMATH及工具推理 | CoT 与 tool-enabled 分开 |
| 分析模型 | 1.3B 对照与 7B iterative / Maj@K 分析 | 不把不同规模曲线合成同实验 |
| GPU / precision / parallelism | 未完整披露 RL 阶段设备型号数量与精度 | 不能从模型7B反推原始卡数 |
| 时间 / GPU-hours | 未给出可拆分全部阶段的 RL 预算 | 不编造省显存百分比或速度倍数 |
| 源码状态 | 原训练实现未公开；现代 verl 伴读 | 静态核读，不声称复现 |

评测中的 in-domain 是 RL 训练使用的 GSM8K/MATH CoT，其他 task 用于跨设置迁移。tool-integrated reasoning 允许生成代码并通过工具辅助，与不允许工具的 CoT 成绩不同。不同 baseline 有些引用原作者报告、有些含 majority vote，Table 5 用灰色标注这类差异；不是所有行都同解码、同预算重训。

组内 64 条回答意味着同 prompt 需要大量解码与评分，去 critic 后总成本仍可能由 rollout 主导。如果估算自己任务的资源，应记录 prompt 数、G、实际输出长度、actor前反向、reward前向、reference前向、重算logprob及并行策略；原文配置给不出当代消费 GPU 的“最低卡数”。

本文采用该版原文的 reward-model 描述，而不是把后来 DeepSeek-R1 的 rule-based reward 配方移入本表。想对比可验证奖励的现代 GRPO 应单独登记 verifier、判题容错、答案提取、测试覆盖率与错误反馈，不把它们藏在统一的“reward=正确率”简式中。

## 4. 结果与图表解读

![Table 5 · CoT与工具辅助推理的模型对照](./assets/papers/paper-grpo/table-5-pdf.png)

**Table 5 解读。** 上半块不允许工具，下半块允许工具；列分别为英语 GSM8K/MATH 与中文 MGSM-zh/CMATH。最可归因于这轮 RL 的同模型对照是 Instruct→RL：CoT GSM8K 82.9%→88.2%，MATH 46.8%→51.7%，分别增加 5.3 与 4.9 个百分点；工具 MATH 57.4%→58.8%，增加 1.4 个百分点。灰色的投票成绩、闭源模型与不同参数量只作背景，不能当作严格同预算算法消融。[表源：v3 第 12 页](https://arxiv.org/pdf/2402.03300v3#page=12)。

RL 的中文指标也提升，说明在受限英语 RL 数据上存在迁移收益；但这不能证明任意数学数据都带来同样跨语言效果，更不能证明 coding agent 的完整任务成功率。Table 5 验证整条 RL 配方有效，并没有把“去 critic”单独与每项数据、reward-model 更新和 KL选择逐因素隔离。

![Figure 5 · RFT Online RFT与两种GRPO监督的1.3B对照](./assets/papers/paper-grpo/figure-5-source.png)

**Figure 5 解读。** 横轴训练 steps，左右为 GSM8K/MATH accuracy；紫色 RFT、绿色 Online RFT、橙色 outcome GRPO、蓝色 process GRPO。后期 online 方法明显好于固定采样的 RFT，process 曲线通常又好于 outcome，支持数据刷新与更细粒度 credit 信号的作用。曲线的模型是 1.3B，不是 Table 5 的主模型7B；图里没有完整 PPO/DPO 曲线，不能据此宣称直接击败这两者。[图源：v3 Figure 5，源码独立原图](https://arxiv.org/src/2402.03300v3)。

![Figure 7 · 单样本质量 多数投票与候选覆盖的区别](./assets/papers/paper-grpo/figure-7-source.png)

**Figure 7 解读。** 横轴候选数 K，左右为 GSM8K/MATH；紫/橙为 Instruct/RL 的 Maj@K，绿/蓝为 Pass@K，温度0.7。RL 的多数投票明显改善，但 Pass@K 没有同等提升，部分 GSM8K 区间还较低。作者据此观察 RL更像提高正确输出的概率质量，而未必扩张“采样K次至少能找到一个正确答案”的覆盖上限。这是该模型与评测的经验观察，不是 RL永远不能创造新能力的证明。[图源：v3 Figure 7](https://arxiv.org/src/2402.03300v3)。

Maj@K 需要聚合答案，Pass@K 是至少成功一次，Pass@1或average@K则是单次成功率的多样本估计。三种数值对应不同推理预算和选择信息，不可替换。原摘要的MATH 51.7%与64样本self-consistency 60.9%也不能讲成同一次调用的性能。

这几张证据共同说明：GRPO 配方提升了已具备数学基础的模型，online sampling、positive/negative信号与process监督有可见作用；更高采样预算下的coverage结论要另外读。源码只提供现代实现映射，没有个人独立训练结果或完整原始GPU开销测量。

## 5. 局限、结论与后续阅读

GRPO 用组相对信号代替独立 value baseline，降低 critic 的训练开销，但需要同题多样本且优势只表达相对表现。全同奖励、过小 G、低质量 scorer、长度权重和跨题难度缩放都会影响训练。省下 value model 不意味着消除了方差、on/off-policy问题或credit assignment。

### 面试问答与追问

**Q1：GRPO 为什么不需要critic？** 它用同题多个reward的组均值作baseline，并归一化构造优势；outcome版本所有回答token共享该优势。追问：省掉的是哪个模型的哪些开销，新增G次采样如何影响总GPU-hour？

**Q2：组内全对或全错怎样？** outcome任务优势为零；若保留KL，整体仍可能有梯度。DAPO会过滤零方差组并继续采样，但无法自动使模型探索到正确解。追问：全错过滤是否丢掉最难问题的训练覆盖？

**Q3：GRPO就是RLVR吗？** 不是。前者是优化/优势估计，后者是奖励可验证性。原论文有学习reward model，二者可以组合也可以分开；测试漏洞也可能使RLVR产生被利用的错误高分。

**Q4：同一回答为什么所有token都收到一样的优势？** outcome评分只知道整条结果，将其广播是粗粒度估计；通过每个token的logprob影响整序列概率。process supervision则根据后续step reward求和，可以区分token，但引入过程标签质量和边界定义。

**Q5：标准差归一化只是数值稳定吗？** 不只是。它改变不同问题和不同reward组合的相对梯度尺度。总体std与样本std、是否去std、epsilon大小均应固定；多奖励先求和会压缩差值，正是GDPO研究的对象。

**Q6：G越大效果一定越好？** 更多rollout可能稳定统计、提高遇到正样本的机会，也增加生成、评分和尾部等待；强相关或全同输出让新增样本价值降低。应按相同GPU-hour比较任务收益，而不是只比较更新数。

**Q7：长度偏差从哪里来？** 回答内token平均让短序列每个token的权重更大；reward、截断与组std又有各自偏差。DAPO改变reduction，GSPO改变ratio单元，GDPO改变多reward归一化，不能将三个改动混成同一件事。

**Q8：部署异步GRPO最难的算法依赖是什么？** 除版本滞后，还要等同题多个reward构造组优势。partial buffer保留未完成回答不等于可以即时对半组训练；应明确组完成、策略版本、mask、teacher/reward数据何时可用。

后续读[DAPO](#paper=paper-dapo)的零方差过滤与token权重，[GDPO](#paper=paper-gdpo)的多目标信号，[统一专题](#report=survey-policy-optimization)的GSPO/SAPO对照。固定原文2402.03300v3；首次公开2024-02-05。2026-10-08核读全文相关背景、RL机制、原表与分析图，以及官方公开资料和现代verl源码；没有执行训练复现。




**2026-10-09更新。** 背景拆为问题背景、前置知识、已有工作；补固定源码节选、逐段形状/梯度讲解与课堂检查。源码节选不是完整可运行训练程序；课堂张量练习见[CPU演示脚本](./assets/learning/policy-optimization-lab.py)，不下载模型且不执行真实RL训练。
