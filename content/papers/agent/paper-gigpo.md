---
id: paper-gigpo
title: "GiGPO：episode组与anchor state组的两层信用分配"
paper_title: "Group-in-Group Policy Optimization for LLM Agent Training"
authors: ["Lang Feng", "Zhenghai Xue", "Tingcong Liu", "Bo An"]
affiliations: ["Nanyang Technological University, Singapore", "Skywork AI, Singapore"]
author_affiliations: [[1], [1], [1], [1, 2]]
venue: "arXiv preprint (fixed v3)"
year: 2025
direction: agent
areas: [language]
tasks: [agents, retrieval, reasoning, training-adaptation]
published: "2025-05-16"
paper_url: "https://arxiv.org/abs/2505.10978v3"
github_url: "https://github.com/langfengQ/verl-agent"
code_note: "静态核读20bd331bdbc9026a5668e11362178e10ab7400c8；当前默认配置与论文精确定义分开说明。"
method_figure: "./assets/papers/paper-gigpo/method-source.png"
method_caption: "Figure 2 · 全轨迹组和重复anchor state内的动作组"
evidence: 已核原文
note_ids: [agentic-rl-training, rl-policy-optimization-map, rl-returns-gae]
tags: [GiGPO, Agentic RL, Credit Assignment, GRPO, Anchor State]
updated: "2026-10-12"
summary: "在既有rollout中寻找重复观察，以后续折扣回报做step优势，再结合episode优势；详解条件、源码、配置与计算开销算术纠正。"
template_version: 5
depth_standard: ddpm
draft: false
---

## 1. 背景与已有工作

### 问题背景：最后失败，不代表每一步都该受同样惩罚

长程agent可能前几步找对工具、后几步选错对象。用最终reward给整条轨迹一个组内优势，训练器很难区别哪些局部决策真正影响后续结果。独立critic可以估计价值，但增加模型、显存与估计误差。GiGPO提出不额外采rollout的两层相对比较：先比较同任务的整个episode，再从这些已收集轨迹中找到重复观察，在相同anchor处比较后续回报。

例如两条WebShop轨迹都到达同商品列表，第一条选中目标商品并最终成功，第二条翻页后失败。在这个共同列表上，局部动作值得区别；同一成功轨迹在列表重复出现时，早期绕路与后期有效选择还可以通过折扣后续回报区分。关键是利用已有重复状态，不额外从每一个anchor重新生成新分支。

它解决信用分配，未直接解决异步权重同步、sandbox恢复或环境等待。Step优势依赖未来回报，通常仍需完成轨迹，不能因为名称中有step就解释成每轮立即backward。对于观察几乎不重复的coding任务，额外局部信号的覆盖也可能很低。

### 前置知识：组、回报和可观察状态

Episode组由同任务和相同初始条件的多条轨迹组成；step组把同任务中相同anchor observation对应的动作放在一起。回报$R_t$是从该动作开始的折扣未来反馈，和即时$r_t$不同。组均值作为baseline，标准差是否加入分母是独立选择，不用std不等于严格leave-one-out；其中仍可能包含当前样本。

“相同观察”不总等于相同完整MDP状态。用户历史、隐藏环境变量、已经修改的文件或剩余预算可能不同，模型也可能看到不同历史。因此anchor匹配是实际表示上的分组假设，必须审查其是否足以支持公平比较。哈希碰撞和过度相似度匹配也会引入错误组。

### 已有工作：从GRPO到局部相对优势

[PPO](https://arxiv.org/abs/1707.06347)常用critic/GAE；[GRPO所在DeepSeekMath](https://arxiv.org/abs/2402.03300)通过同prompt组结果去掉critic；RLOO采用leave-one-out baseline。GiGPO保留整条轨迹相对效果，又加重复anchor的后续回报比较。它不是学一个teacher、也不是新增过程reward model。与Search-R1互补：Search-R1给工具交互闭环，GiGPO在这种闭环里改善局部信用。

## 2. 方法与实现机制

### 两级组的输入和输出

![Figure 2 · 既有轨迹中的episode与anchor分组](./assets/papers/paper-gigpo/method-source.png)

**Figure 2解读。** 输入为同任务多条完整轨迹，episode部分比较总回报；step部分从已出现的观察中聚合动作和后续回报，组合优势再进入策略优化。图没有要求在每个anchor额外启动一批rollout；这一点解释“无额外LLM生成”的含义。图源为固定v3的figure/workflow.pdf、body/method.tex中的fig:workflow。箭头表示数据依赖，不是所有局部优势都可在动作刚执行时立即得到。[固定原文](https://arxiv.org/abs/2505.10978v3)。

Episode级优势可写为$A_E^{(i)}=(R^{(i)}-\bar R)/F_E$，$F_E$为组std或1。对anchor $\tilde s$，收集已访问该观察的$(a_t^{(i)},R_t^{(i)})$，计算：

$$
R_t^{(i)}=\sum_{k=t}^{T_i-1}\gamma^{k-t}r_k^{(i)},\qquad
A_S^{(i,t)}=\frac{R_t^{(i)}-\mathrm{mean}_{G_S(\tilde s)}R}
{F_S},\qquad A^{(i,t)}=A_E^{(i)}+\omega A_S^{(i,t)}.
$$

它把全局“这条轨迹比同任务其他轨迹怎样”与局部“这个动作比同观察其他动作怎样”相加。$\omega$控制局部信号权重；$F=1$保留reward尺度，$F=\mathrm{std}$改变任务和组间权重。单例组没有可靠相对比较，原理上应观察覆盖并明确实现处理；不能自动把一个样本的归一化优势视为统计可信。

### 一个能手算的两轨迹例子

两条轨迹从同观察$S_0$开始，均又到$S_1$。取$\gamma=1$，第一条reward为$(0,1)$，第二条为$(1,-1)$，总回报分别1/0。Episode中心优势为$+0.5/-0.5$。在$S_0$，后续回报也是1/0，step优势$+0.5/-0.5$；在$S_1$，后续回报1/−1，step优势$+1/-1$。$\omega=1$且不除std时，组合在$S_0$为$+1/-1$，在$S_1$为$+1.5/-1.5$。同一轨迹不同动作因后续回报比较得到不同信用。

这个例子不证明所有anchor组都有反事实因果含义；样本间可能相关，历史可能不等价，观察次数影响统计。它只演示给定分组与return后的计算，共享CPU课堂脚本对此做了数值检查。

![Figure 3 · WebShop重复anchor下不同选择与后续结果](./assets/papers/paper-gigpo/anchor-source.png)

**Figure 3解读。** 两条轨迹在类似商品列表出现，第一条经过选择最终成功，第二条随后离开并失败。红色高亮表示可以匹配的anchor，绿色勾与红叉表示最终结果，时间标记显示同观察可在不同step再出现。折扣让成功路径较早的重复绕路收到较小后续return。这是状态分组示意，不是通过截图就验证了完整隐藏状态相同。图源figure/anchor_state.pdf、fig:how_work，对应固定v3方法部分。

### 真实源码：return、组匹配、组合与token接线

固定commit为20bd331bdbc9026a5668e11362178e10ab7400c8。[core_gigpo.py](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py#L87)的compute_step_discounted_returns逆序遍历同轨迹reward，按gamma计算running return；轨迹边界必须由索引维护，不能跨任务累加。compute_gigpo_outcome_advantage再分别构造episode与step优势。

下列三行是该函数的关键组合，省略参数与中间日志；应沿固定源码138–171行对照：

```python
# Official excerpt, combination after group construction.
step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)
scores = episode_advantages + step_advantage_w * step_advantages
return scores, scores
```

两个返回值在该训练器的结果优势接口里传下去，不意味着传统actor–critic里的value target一定等于advantage。没有critic时常用相同对象占接口位；迁移算法不能机械照抄接口名。相关rollout_loop记录每turn的anchor观察和轨迹索引，loss再广播到该动作的模型token，并使用有效动作mask。

build_step_group按任务index与观察hash聚合，支持可选相似度匹配；当前代码默认enable_similarity=False，threshold0.95只在开启时有效。这是当前静态仓库能力，不应说成论文主实验必然使用模糊state相似度。[分组源码](https://github.com/langfengQ/verl-agent/blob/20bd331bdbc9026a5668e11362178e10ab7400c8/gigpo/core_gigpo.py#L243)。

还要注意episode_norm_reward当前默认compute_mean_std_cross_steps=True，可能按turn行参与统计；关闭时按(index,traj_index)去重。长轨迹含更多行时，这与纸面“每episode一个总回报、等权组统计”可能不同。必须检查实际配置，而不能仅凭函数名认定实现与数学权重完全一致。Singleton分支与torch.std默认自由度也需查，不能用教学人口std替代源码样本std。

### 为什么需要完整轨迹，以及何时不能用anchor

若最终反馈尚未返回，$R_t$还不完整；训练器可以提前做forward、teacher scoring等，但该优势不能当作已知常数提前backward，除非另有合法bootstrap或重新计算机制。编码出错、观察截断、不同文件快照共用同一hash，都会让局部比较失真。对coding agent建议先记录精确环境快照/工作区版本、历史长度、组大小、重复覆盖、单例比例，再决定是否用相似组。

算法新增分组不增加LLM生成调用，但需CPU hash、组聚合与额外优势计算；“零额外rollout”不等于零系统开销。相关原图计算轴有断轴，也应按数值而非柱高评估。

## 3. 实验设置与算力

主交互环境为ALFWorld/WebShop，模型Qwen2.5-1.5B/7B-Instruct；搜索QA为3B/7B。论文表给三random seeds，不能把单次最佳结果与其均值混用。搜索任务训练NQ+HotpotQA，七QA评测与Search-R1类似，但配方与版本不同，跨论文成绩不能自动按同预算排名。

| 配置 | ALFWorld | WebShop | Search QA |
| --- | --- | --- | --- |
| prompt/response | 2048/512 | 4096/512 | 4096/512 |
| 最大turn | 50 | 15 | 4 |
| actor LR | 1e-6 | 1e-6 | 1e-6 |
| PPO critic LR | 1e-5，仅PPO | 1e-5，仅PPO | 不作为GiGPO模块 |
| 任务reward | 成功10、失败0 | 成功10、失败0 | 正确1、错误0 |
| 无效动作 | −0.1 | −0.1 | −0.01 |
| group | 8，同批16任务组，共128环境 | 同左 | group5，训练batch项256 |
| minibatch | 256 | 64 | 512 |
| rollout/eval温度 | 1.0/0.4 | 1.0/0.4 | 1.0/0.0 |
| KL系数 | 0.01 | 0.01 | 0.001 |
| gamma / omega | 0.95 / 1 | 0.95 / 1 | 0.95 / 1 |
| GPU和迭代 | 1.5B 2H100、7B 4H100，150迭代 | 同左 | 3B 4H100、7B 8H100，200迭代 |

PPO交互设置用128独立环境；组算法用16任务×8rollout，总环境数一样但prompt多样性和组依赖不同。相同超参是一种控制方法，却不是所有算法独立调到最优的证明；不同长度、anchor密度与reward稀疏度还会改变有效训练量。Fnorm std与1分别作为GiGPO两变体，QA主设置用std，不随意把WebShop更好变体套到全部任务。

评测ALFWorld按任务子类与overall成功率；WebShop同时报平均score和success rate。二者不能互换，score提高不保证所有完全成功任务更多。计算开销实验拆rollout、旧概率、reference概率、update和两级优势，来自特定设置；没有由单次step耗时推导完整训练GPU-hour或跨GPU利用率。本文未给自己的训练复现，以下成本比例是整理者对原始数字的算术核查。

## 4. 结果与图表解读

![Table 1 · 两个交互环境与不同模型规模](./assets/papers/paper-gigpo/table-1-pdf.png)

**Table 1解读。** 1.5B ALF overall：GRPO72.8，GiGPO/std86.7，GiGPO/no-std86.1；WebShop score75.8→83.1/83.5、SR56.8→65.0/67.4。7B ALF GRPO77.6对GiGPO/std90.8/no-std90.2，WebShop no-std score86.2、SR75.2。不能概括“去std处处更好”，也不把14点差写成14%相对提升。每个子任务差异仍需读完整原表。[表源PDF第8页](https://arxiv.org/pdf/2505.10978v3#page=8)。

![Table 2 · 搜索问答的跨任务结果](./assets/papers/paper-gigpo/table-2-pdf.png)

**Table 2解读。** 主设置Fnorm=std，3B/7B平均42.1/47.2，七列有ID/OOD标记。它显示局部组思想可迁移到搜索，但各baseline的模型、训练和固定版本需要另外对齐；不能用此表与另一论文均值直接判同GPU-hour优势。StepSearch缺失部分列记为横线，不能当零分或替作者补平均。[表源PDF第8页](https://arxiv.org/pdf/2505.10978v3#page=8)。

![Figure 4 · 全局与局部优势、std选择的消融](./assets/papers/paper-gigpo/ablation-source.png)

**Figure 4解读。** 雷达轴为六类ALF任务和WebShop，四条路线对应去AE、去AS以及两种完整归一化。去任何一个优势层普遍降低结果，完整两层更强；std与不std在各轴各有高低。面积不是线性平均指标，重叠也会遮挡，不根据雷达面积宣布统计显著。图源figure/ablation.pdf、body/experiment.tex的fig:ablation，是原始结果资产。

![Figure 6 · 分组与step优势的计算成本](./assets/papers/paper-gigpo/compute-source.png)

**Figure 6解读。** 秒数分别rollout221.97、old prob24.99、ref prob25.96、update89.86、AE0.05、grouping0.01、AS0.53。纵轴在1秒附近断开，不能用视觉柱高比值。四个主阶段和AE合计362.83秒，新增0.54秒，整理者计算$0.54/362.83\times100\approx0.14883\%$；原文写“<0.002%”与图内数字不一致，本报告明确保留这个算术差异，而非重复错值。仍可说该设置额外优势成本很小，不能说严格无开销。原图源figure/compute.pdf，标签fig:budget。

作者在特定ALFWorld、N=8设置观察到较多重复状态，这是局部分组可用的机制证据，不能外推所有真实软件仓库都具有同样重复率。Anchor覆盖、组单例、隐藏状态等价与样本相关性应该是自己项目必报的诊断，而不能只看最后SR。

## 5. 局限、结论与后续阅读

GiGPO值得学习的是：已有trajectory能提供比单一终局组优势更细的比较，不一定要新增critic或过程reward模型。但优势依赖return，anchor定义也依赖任务结构。精确观察复用不足、历史不等价、随机工具状态与长程代码修改都可能降低信号可靠性；相似度分组扩大覆盖会引入额外假设。

后续比较[Search-R1](#paper=paper-search-r1)的动作/观察mask、[Agent Lightning](#paper=paper-agent-lightning)的transition适配和[Agentic RL](#q=agentic-rl-training)统一轨迹契约。迁移时先复核当前代码的episode统计是否按turn加权，再做AE/AS/std/anchor匹配的独立消融。

**Q01 加了什么？** 已有数据上加anchor组与step后续return优势。

**Q02 需要额外rollout吗？** 主设计不需要，但分组仍有CPU与统计开销。

**Q03 无critic是否等于无信用分配？** 不等于，两级组统计提供信用。

**Q04 相同观察是否必为相同状态？** 未必，需检查隐藏状态和历史。

**Q05 可以每turn立即算完整AS吗？** 最终回报未回来时通常不行。

**Q06 不除std就是LOO吗？** 不是，组均值可能含当前样本，尺度也不同。

**Q07 当前源码与公式哪里需核？** episode默认跨step统计、singleton与相似匹配开关。

**Q08 新增开销小于0.002%吗？** 图内数值计算约0.149%，应注明原文算术不一致。

**来源。** [固定v3正文和附录D/E](https://arxiv.org/abs/2505.10978v3)、原始图表、上述固定commit。只静态核读与CPU手算，未训练模型；图版权归作者。
