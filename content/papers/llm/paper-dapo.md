---
title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
authors: ["Qiying Yu", "Zheng Zhang", "Ruofei Zhu", "Yufeng Yuan", "Xiaochen Zuo", "Yu Yue", "Weinan Dai", "Tiantian Fan", "Gaohong Liu", "Juncai Liu", "Lingjun Liu", "Xin Liu", "Haibin Lin", "Zhiqi Lin", "Bole Ma", "Guangming Sheng", "Yuxuan Tong", "Chi Zhang", "Mofan Zhang", "Ru Zhang", "Wang Zhang", "Hang Zhu", "Jinhua Zhu", "Jiaze Chen", "Jiangjie Chen", "Chengyi Wang", "Hongli Yu", "Yuxuan Song", "Xiangpeng Wei", "Hao Zhou", "Jingjing Liu", "Wei-Ying Ma", "Ya-Qin Zhang", "Lin Yan", "Mu Qiao", "Yonghui Wu", "Mingxuan Wang"]
affiliations: ["ByteDance Seed", "Institute for AI Industry Research (AIR), Tsinghua University", "The University of Hong Kong", "SIA-Lab of Tsinghua AIR and ByteDance Seed"]
venue: "arXiv preprint"
year: 2025
published: "2025-03-18"
summary: "沿Clip-Higher、动态采样、token加权和超长奖励四个机制解释DAPO，拆开AIME avg@32与pass@32、更新步数与实际rollout预算。"
direction: "llm"
areas: ["language"]
tasks: ["training-adaptation", "reasoning"]
evidence: "已核原文"
note_ids: ["sft-dpo-rl", "grpo-rlvr", "infra-rl-pipeline"]
updated: "2026-10-09"
template_version: 5
depth_standard: "ddpm"
draft: false
id: "paper-dapo"
paper_title: "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
paper_url: "https://arxiv.org/abs/2503.14476v2"
method_figure: "./assets/papers/paper-dapo/algorithm-1-pdf.png"
method_caption: "Algorithm 1 动态采样缓冲与策略更新"
github_url: "https://github.com/BytedTsinghua-SIA/DAPO"
tags: ["DAPO", "Policy Optimization", "Paper Reading", "Interview"]
code_note: "正文区分论文原实现与固定commit的现代框架伴读；仅静态核读，未登记训练复现。"
author_affiliations: [[1, 2, 4], [1], [1], [1], [1], [1], [1, 2, 4], [1], [1], [1], [1], [1], [1], [1], [1], [1, 3], [1, 2, 4], [1], [1], [1], [1], [1], [1], [1], [1, 4], [1], [1, 2, 4], [1, 2, 4], [1], [2, 4], [2, 4], [2, 4], [2, 4], [1, 4], [1, 4], [1], [1, 4]]
---

## 1. 背景与已有工作

### 问题背景

从base模型开始长CoT数学RL，朴素GRPO经常只能得到有限收益：entropy快速降低、同题回答趋同、有效奖励组变少、长回答更新权重不合理、截断反馈混入噪声。DAPO把这些具体症状拆开处理，公开一套训练配方，而不是只提出一个新的loss符号。

Decoupled Clip拆开概率比上下界；Dynamic Sampling过滤缺少相对信号的prompt组后补采。它们不等于多reward归一化，也不等于推理continuous batching。配方仍用组优势，数学主实验使用规则验证并移除reference KL，和DeepSeekMath原GRPO的reward模型与短输出设置不同。

### 前置知识

**组优势与有效组。** 同题G回答全对或全错时，binary reward的std为零，任务优势缺乏区分。有效训练batch应区分候选prompt、保留prompt和回答条数；过滤后不补采会减少每步实际信号。

**概率比与PPO单侧平台。** ratio衡量当前对旧动作的相对概率，正优势超过upper才进入有利方向平台，负优势低于lower才进入平台。0.01概率token在ratio1.2处只增到0.012，0.9概率token则对应1.08；surrogate门槛不是合法概率的硬限制。

**entropy与探索。** entropy衡量动作分布的不确定性，过低可能缺少探索，过高也可能是乱码或退化。reward、entropy、验证分数和长度必须一起看，不能以entropy单调升高作为训练完成条件。

**token平均与sequence平均。** 两条长度100/1000的回答，sequence平均给予相近总权重，token平均给予约1:10总权重。二者都通过token logprob更新，区别是分母与样本权重；global token分母还要跨microbatch与DP保持一致。

**截断、正确性与长度reward。** 长度上限可能使本来有用的推理没有完成。loss mask可以不训练某些截断片段，soft length penalty则改变reward；这些操作不等于认定所有长答案逻辑错误。惩罚buffer是长度区间，不是动态采样数据buffer。

### 已有工作与本文位置

[PPO](#paper=paper-ppo)提供clip，[GRPO](#paper=paper-grpo)提供组优势。DAPO组合Clip-Higher、补采、token平均和超长reward shaping；最终配方不含reference KL。[固定v2 §3—§4](https://arxiv.org/pdf/2503.14476v2)。

| 症状 | 修改 | 必须付出的代价 |
| --- | --- | --- |
| 过早确定化 | 提高upper clip | 更宽有利更新区间需监控 |
| 零方差组增多 | 过滤并补采 | 生成成本、难度覆盖变化 |
| 长回答token弱权重 | token-mean reduction | 长回答获得更高总权重 |
| 截断奖励噪声 | mask与soft penalty | 明确长度目标和verifier |

原主表是按顺序累加的配方实验，不能把最后一项增益当成与其他机制独立的因果贡献，更不能把少一半update steps解释成少一半GPU-hour。

## 2. 方法与实现机制

### 共同骨架与完整目标

对prompt $q$采样$G$个回答，得到组优势$\widehat A_i$。令$r_{i,t}$为token current/old概率比，$L_i$为有效回答长度。DAPO用非对称clip并改变reduction：

$$
J_{\mathrm{DAPO}}=\mathbb E\left[
\frac{1}{\sum_iL_i}\sum_i\sum_t
\min\left(r_{i,t}\widehat A_i,
\operatorname{clip}(r_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})\widehat A_i\right)
\right],\qquad 0<N_{\mathrm{correct}}(q)<G.
$$

这个条件表示纳入的binary correctness组既非全对也非全错。实际框架可根据acc、seq_reward或其他指标过滤，不同指标在叠加长度reward后未必等价；应明确过滤作用在哪个分数上。reference KL未出现在本文最终配方里，不能将通用GRPO KL项直接照搬到完整DAPO公式。

![Algorithm 1 · 同题采样 过滤补采与策略更新](./assets/papers/paper-dapo/algorithm-1-pdf.png)

**Algorithm 1解读。** 输入是初始策略、reward、prompt和上下clip；第4—6行采样评分并过滤，第7—8行检查buffer是否达到有效训练量，不足时继续采样；第9—11行在有效数据上构造优势并更新。buffer存的是已完成、可训练的组，不是任意历史replay或未完成partial prefix。它展示实际算法控制流，不是原文另有一张系统架构图。[原算法：v2第7页，内联LaTeX表格局部裁图](https://arxiv.org/pdf/2503.14476v2#page=7)。

### Clip-Higher 为什么有利于低概率探索

PPO上clip为1.2时，一个旧概率0.01的正优势token最多在本项中被奖励到0.012，而旧概率0.9的token对应上限1.08，已经超过合法概率1，实际上不构成同样强的有利方向限制。这里是surrogate平台阈值，不是硬约束概率不能继续变化；其他样本仍可通过共享参数改变它。

DAPO保持$\epsilon_{\mathrm{low}}=0.2$，增大$\epsilon_{\mathrm{high}}=0.28$。对0.01 token，上平台阈值变成0.0128，相对更宽容；不是直接给它更高reward，也不是增加一个entropy bonus。原论文还观察up-clipped token的平均概率较低，支持对探索动作过早抑制的诊断。

非对称clip与SAPO非对称temperature区别在于：前者改变硬平台的位置，后者改变平滑梯度衰减速度。不能简单用“正样本大胆、负样本保守”代替数学说明，因为PPO的min分支取决于advantage符号，而probability ratio过大的负优势项仍有纠正梯度。

### Dynamic Sampling 为什么必须补采

若一个prompt的G条回答reward相同，优势分子全为零。随着模型改善，全对组可能越来越多；如果仍按固定prompt数量做actor更新，每个batch实际有效任务信号会减少。DAPO把这类组滤掉，继续生成直到攒够目标有效prompt数，再按原设定进行更新。

例如目标512个有效prompt，某轮保留率50%，平均需约1024个候选prompt才能补足；若保留率10%，候选数量可近似增至5120。这个计算是教学期望，不是原论文实际开销。多次补采时要保证当前更新前的behavior版本固定，否则buffer内概率与old策略定义不再一致。

过滤并没有让全错题自动变得可学习。它在当前奖励条件下选择有组差异的题，可能集中中等难度并减少最难题或已掌握题的覆盖。要监控acceptance rate、重采轮数、有效prompt多样性与最长等待；对于工具任务还要计算环境执行与失败开销。它与异步持续采样可以组合，但不是异步方法本身。

### Token-level loss改变了谁的权重

原始GRPO形式是每回答先平均，再平均回答：

$$
J_{\mathrm{seqmean}}=\frac1G\sum_i\frac1{L_i}\sum_t z_{i,t}.
$$

DAPO是整体有效token平均：$J_{\mathrm{tokenmean}}=(\sum_i\sum_tz_{i,t})/(\sum_iL_i)$。两条长度100/1000的回答，前者各占一半总权重；后者分别占约1/11和10/11。它提高长回答的整体贡献，使同等token信号不因所在回答更长而被额外缩小。

这不是“原GRPO不是token级策略梯度”：原GRPO同样通过token logprob更新，只是reduction不同。也不是完全消除长度偏差，长序列会得到更多总权重，reward与截断又各有长度相关效应。在microbatch和DP上应按全局有效token总数归一化，否则各rank或各microbatch独立平均会悄悄改变目标。

### Overlong filtering与soft punishment

直接把所有超过长度上限的回答标成错，会把“推理可能正确但被预算截断”与“逻辑错误”混在一起。作者先做overlong filtering：屏蔽截断回答的loss，观察训练更稳；随后增加soft overlong punishment，让接近上限的回答逐步获得负长度reward，而不只有突然的一刀切惩罚。

按原公式记$L_{\max}$为硬上限、$L_{\mathrm{cache}}$为惩罚区间：

$$
R_{\mathrm{len}}(y)=
\begin{cases}
0,&L\le L_{\max}-L_{\mathrm{cache}},\\
\frac{L_{\max}-L_{\mathrm{cache}}-L}{L_{\mathrm{cache}}},&L_{\max}-L_{\mathrm{cache}}<L\le L_{\max},\\
-1,&L>L_{\max}.
\end{cases}
$$

原实验期望长度16384，额外惩罚区间4096，因此硬生成上限20480。在18384长度时，教学惩罚约-0.4883；到20480为-1。长度reward叠加correctness后再进入优势，不能把它称为task correctness本身。截断loss mask与渐进reward是两个不同操作，原论文逐步消融也把它们分开。

### 公开源码中对应的三个入口

作者项目为[BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)，项目README固定在`33fe3176f0bb212588e84fc8ccf50dd554975144`。实现使用verl；核读[维护者公开复现提交](https://github.com/verl-project/verl/tree/4f80e465c2ec79ab9c3c30ec74b9745de61d0490)，该版本与原论文实验和项目早期分支区分。

[RayDAPOTrainer.fit](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/dapo_ray_trainer.py)以prompt UID聚集reward，计算每组std，保留std>0的组，过滤所有对应trajectory后拼接buffer。若有效prompt不足，`continue`跳到下一generation batch；达到`max_num_gen_batches`还不足就抛错。它同时支持不同filter metric，意味着不能只看`adv_estimator=grpo`就判断实际配方。

[DAPORewardManager.__call__](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/workers/reward_manager/dapo.py)取有效回答长度，调用rule-based scorer，将超过expected length的线性负项加入reward，再放到最后有效token位置。这个`buffer`是长度惩罚区间，与采样数据buffer不同，变量名相似但职责不同。

[compute_policy_loss与agg_loss](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/trainer/ppo/core_algos.py)接收 `[B,L]` 新旧logprob和mask，支持分别设置cliprange_low/high，最后`token-mean`求标量。复现脚本还配置dual clip等额外项，其行为不能冒充论文核心公式的唯一实现。本文核对上述输入、分支与reduction，未运行下载的训练脚本。

[作者项目讲解](https://dapo-sia.github.io/)按训练症状解释四个机制；[verl DAPO文档](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/docs/algo/dapo.md)提供版本与公开复现记录。教程52分、早期44分和论文50分是不同配置/版本，不混写为一个成绩。

### 课堂源码拆解：筛选条件必须改变控制流

先在白板上分开候选prompt数、保留prompt数和trajectory数。目标512prompt、G16意味着8192条有效回答；它不意味着只生成8192条。把过滤后继续采样画成循环，学生才能理解算法与系统成本。

#### 按prompt整组过滤

**真实源码节选：[DAPO 公开复现 · 过滤零方差组](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/dapo_ray_trainer.py#L192-L200)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DAPO 公开复现 · 过滤零方差组"
prompt_uid2metric_std = {}
for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
    prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

kept_prompt_uids = [
    uid for uid, std in prompt_uid2metric_std.items()
    if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
]
num_prompt_in_batch += len(kept_prompt_uids)
```



metric可以是accuracy或reward，`np.std`在该实现是population std；这里只用它是否大于零，正常非退化组的数值尺度不影响筛选真假。保留UID之后必须保留该UID对应的全部回答，不能只留下成功样本再算组优势。单回答回退是实现兼容，正常G16配方不是单rollout。

#### 不足时没有optimizer update

**真实源码节选：[DAPO 公开复现 · 补采与训练量对齐](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/recipe/dapo/src/dapo_ray_trainer.py#L213-L227)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DAPO 公开复现 · 补采与训练量对齐"
prompt_bsz = self.config.data.train_batch_size
if num_prompt_in_batch < prompt_bsz:
    print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
    max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
    if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
        print(f'{num_gen_batches=}. Keep generating...')
        continue
    else:
        raise ValueError(
            f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
        )
else:
    # Align the batch
    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
    batch = batch[:traj_bsz]
```



`continue`回到生成循环，尚未执行actor update。`max_num_gen_batches`防止一直补不齐；达到目标后按promptbatch×G裁到训练量。接受率低时成本上升，group diversity与最长等待要监控。代码里的buffer是已完成数据，不能拿它当未完成prefix的partial resume。

#### 长度区间与token权重

**真实源码节选：[DAPO 公开复现 · Soft Overlong Reward](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/workers/reward_manager/dapo.py#L103-L108)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DAPO 公开复现 · Soft Overlong Reward"
overlong_buffer_len = self.overlong_buffer_cfg.len
expected_len = self.max_resp_len - overlong_buffer_len
exceed_len = valid_response_length - expected_len
overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
reward += overlong_reward
```



`max_resp_len-buffer_len`是开始惩罚的expected长度，超出量除buffer长度得到线性负项，小于阈值时`min(...,0)`为0。这与正确性reward相加，不是替代正确性。代码允许`penalty_factor`缩放，原文核心示例设1；硬上限与有效response length控制实际取值范围。

**真实源码节选：[DAPO 公开复现 · token mean分支](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/trainer/ppo/core_algos.py#L283-L284)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DAPO 公开复现 · token mean分支"
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
```



短分支虽然只有两行，却决定所有有效token用同一分母。让两回答长度2/4且每token值分别1/3，sequence平均是2，token平均是7/3。actor policy kernel仍是clip形式；函数名GRPO不能掩盖DAPO的过滤、reward与reduction改动。

最后拿接受率50%与10%算候选生成量，让学生解释为何更少update steps不直接证明更低GPU-hour；再用progressive主表解释单项收益为何依赖前置机制。


## 3. 实验设置与算力

主实验从Qwen2.5-32B-Base做数学RL，而不是从DeepSeekMath-Instruct接续。数据、上下文和验证方式均影响最终成绩。原文的training step指梯度更新，rollout step指一批生成，两者相差16次更新；dynamic sampling还可能一个有效batch经过多次生成。

| 项目 | 原文或明确标注的公开实现 | 证据与口径 |
| --- | --- | --- |
| actor初始化 | Qwen2.5-32B-Base | 不是随机初始化32B参数 |
| 训练数据 | DAPO-Math-17k，数学题与可校验答案 | dataset transformation与项目公开数据 |
| reward | 规则答案匹配、长度shaping | 无独立学习reward model的主配方 |
| KL | 最终配方移除reference KL | 与DeepSeekMath原始GRPO不同 |
| 优化器 / LR | AdamW，固定$10^{-6}$ | §4.1 |
| warmup | 20 rollout steps | 不应误记成20个optimizer step |
| prompt batch | 512；每prompt16回答 | 8192条有效回答/rollout batch，本文乘算 |
| training mini-batch | 512回答；每rollout16次梯度更新 | 更新与采样预算分开 |
| clip | low0.2、high0.28 | 非对称token ratio |
| 长度 | expected16384，buffer4096，hard20480 | §4.1 |
| AIME2024评测 | 32重复，报告avg@32 | temperature1.0，top-p0.7 |
| 项目硬件说明 | 作者网站公开128 H20 GPU的开源实验 | 项目说明，不是论文逐消融完整资源表 |
| 后续维护者复现 | 文档完整/去dynamic配置使用128 H800；早期配置128 H20 | 与原50分数据分列，不合并预算 |
| precision / parallelism | 原文未完整给出逐实验账本 | 实现脚本SP、TP与offload只是该快照配置 |
| 时间 / GPU-hours | 无完整逐消融绝对预算 | 不能以50%更新数直接宣称GPU-hour减半 |
| 代码 | 项目与verl复现各固定commit | 仅静态伴读 |

官方实现脚本使用更大的generation prompt batch来超采，再过滤到目标训练prompt数。这个候选batch与论文有效batch512不同；不能把所有生成请求都当成最终训练样本。脚本的microbatch、SP、TP、offload和最大token预算适配特定系统，不是换GPU型号就一定能直接跑通的抽象参数。

AIME2024只有30题，对每题重复32次生成取平均能够降低采样噪声；avg@32仍对应单次解题成功率的多次估计。pass@32是至少成功一次，cons@32是投票聚合，它们给出不同推理预算下的能力。准确率50与pass@32接近80不能相互替换。

主表是progressive additions，并非全因子实验或所有机制的leave-one-out。后加机制的增益取决于已加入哪些前置机制；尤其生成、entropy和长度改变会影响每个step成本。应把这个公开配方作为可复现baseline，再以固定GPU-hour和相同verifier检验迁移到coding/agent是否有效。

## 4. 结果与图表解读

![Table 1 · 从朴素GRPO到完整DAPO的逐步配方](./assets/papers/paper-dapo/table-1-pdf.png)

**Table1解读。** 所有技术按表中顺序累加，AIME24 avg@32从30，依次到36、38、41、42、50；比较的DeepSeek-R1-Zero-Qwen-32B为47。最后dynamic sampling这一阶段增加8个百分点，但不是无条件独立贡献8点，因为它依赖前面配方。最后50分也不等于从随机初始化模型训练出了50%正确率，初始化是已有32B base。[表源：v2第9页](https://arxiv.org/pdf/2503.14476v2#page=9)。

![Figure 1 · Avg Pass与Consistency三种AIME指标](./assets/papers/paper-dapo/figure-1-source.png)

**Figure1解读。** 横轴梯度update steps，纵轴AIME2024 accuracy；紫色avg@32终点50，浅青pass@32与蓝色cons@32明显更高，横虚线为对照47分。原文强调达到相近/更高水平所需更新数约一半，但超采数量、回答长度和每更新设备时间并未由这张图统一。它不能直接支持总tokens、GPU-hours或墙钟也减半。[图源：v2 Figure1](https://arxiv.org/src/2503.14476v2)。

![Figure 2 · Clip-Higher对验证质量与entropy的对照](./assets/papers/paper-dapo/figure-2-source.png)

**Figure2解读。** 左侧AIME avg@32，右侧generation entropy，横轴为steps；紫色有Clip-Higher，浅青无。无此机制的entropy接近零，验证分数后期停滞；有此机制保留更高entropy并继续改善。此图支持防止过早确定化的诊断，不证明把entropy无限增大或epsilon上界无限放宽都好。[图源：v2 Figure2，两面板完整提取](https://arxiv.org/src/2503.14476v2)。

![Figure 6 · 动态采样在相近分数处需要的更新步数](./assets/papers/paper-dapo/figure-6-source.png)

**Figure6解读。** 横轴仍是step，不是墙钟，纵轴AIME avg@32；紫色dynamic sampling更早到达横线附近，浅青无dynamic需要更多更新。曲线验证有效训练组能加快按更新数衡量的收敛；正文声称其额外采样未显著增加总时间，但图自身没有时间轴、设备计时或统一GPU-hour，所以不从这张图反推固定系统加速倍数。[图源：v2 Figure6](https://arxiv.org/src/2503.14476v2)。

这些结果把算法症状与验证quality联系起来，而不是只展示training reward上升。仍需同时看到截断率、acceptance、平均/尾部长度和entropy；作者后续Figure7还展示相关训练监控。省下无效actor update可能被更多rollout成本抵消，是否改善自己任务的goodput需要实测。

## 5. 局限、结论与后续阅读

DAPO是一套针对长CoT数学训练验证过的配方。它解决的是探索、有效组、reduction和截断反馈之间的相互影响，不承诺所有任务都不需要KL或相同上下clip。规则reward可被答案提取漏洞利用，动态过滤改变题目分布，token平均增加长回答权重；这些代价应在迁移时分别测量。

### 面试问答与追问

**Q1：DAPO比GRPO改了什么？** 四项：扩大有利方向上clip、过滤零差异组并补采、整体token平均、处理超长反馈；最终配方还移除KL。追问应能把每项与对应失败现象连接。

**Q2：为何只提高upper clip？** 正优势的低概率token相对概率提升受上平台限制，较高upper允许更多探索变化，lower仍约束负优势的继续压低。它不是对所有token施加同一绝对概率增幅。

**Q3：动态采样为什么不是把全错样本reward改成非零？** 它保持reward定义，选择有组差异的prompt补足有效batch；全错题仍可能没有学习信号，并有覆盖偏差和超采成本。

**Q4：token-level loss和GRPO有什么区别？** 都通过tokenlogprob求导，区别是先每sequence平均还是全batchtoken平均。用100/1000长度手算各回答的总权重，比“长序列更稳定”更准确。

**Q5：soft overlong reward是不是对错误答案更宽容？** 它区分生成长度预算与任务正确性，避免截断一律引入错误信号；本身不证明推理正确。截断mask、答案提取和soft长度负项必须分开。

**Q6：少一半steps是不是训练算力减半？** 不成立。每step采样量、回答长度、筛选率与backward不同，需计算totalrollout tokens、有效tokens、GPU-hour与同预算quality。

**Q7：为何移除KL？** 作者长推理RL希望模型大幅改变base生成分布，实验中KL会阻碍这种变化。这是任务选择，不能推广到所有安全、风格或人类偏好对齐场景。

**Q8：迁移到coding agent先查什么？** verifier覆盖、工具输出mask、失败环境开销、同题组尾部等待、超长是否对应任务复杂度以及过滤率。优化配方不自动处理sandbox状态保存和异步策略版本。

参考[作者项目](https://dapo-sia.github.io/)、[项目源码说明](https://github.com/BytedTsinghua-SIA/DAPO)、固定verl实现与[统一专题](#report=survey-policy-optimization)。原文固定2503.14476v2，首次公开2025-03-18，PDF封面的项目日期不同。该版贡献页列37位去重贡献者，包含abs元数据未列出的Juncai Liu、Ru Zhang，本记录按正文完整列表和单位映射收录。2026-10-08核读原文、原图与公开代码；未执行训练复现。




**2026-10-09更新。** 背景拆为问题背景、前置知识、已有工作；补固定源码节选、逐段形状/梯度讲解与课堂检查。源码节选不是完整可运行训练程序；课堂张量练习见[CPU演示脚本](./assets/learning/policy-optimization-lab.py)，不下载模型且不执行真实RL训练。
