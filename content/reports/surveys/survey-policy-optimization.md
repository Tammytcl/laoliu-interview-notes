---
id: survey-policy-optimization
title: "七篇策略优化论文的机制 代码与实验证据对照"
type: survey
date: '2026-10-08'
directions: [llm, agent, infra]
paper_ids: [paper-ppo, paper-dpo, paper-grpo, paper-dapo, paper-gdpo]
tags: [专题调研, Policy Optimization, RLHF, RLVR, Interview]
updated: "2026-10-15"
summary: "连接RL基础主线与七篇精读，逐段对应真实源码、原图和实验条件，区分算法修改位置与可比较证据。"
template_version: 1
draft: false
---

## 1. 研究问题与使用场景

本文集中回答：**论文改了哪一层，真实代码怎样实现，原实验支持多强的结论。** 从零学习先看[RL算法总览](#q=rl-policy-optimization-map)，沿[Actor Critic](#q=rl-actor-critic) → [GAE](#q=rl-returns-gae) → [PPO训练一步](#q=rl-ppo-training) → [GRPO](#q=grpo-rlvr)建立一条数据流，再回本文核论文。

学习专题与论文精读的分工如下。基础专题负责定义、推导和统一小例子；本报告负责横向比较和源码定位；单篇保留作者原图、完整方法、模型/数据/算力、结果与局限。遇到同一知识点优先回主文，不需要在三个入口重复学习。

| 要解决的问题 | 主入口 |
| --- | --- |
| 策略梯度、baseline、V/Q/A为何成立 | [Actor Critic](#q=rl-actor-critic) |
| 回报、TD、GAE、终止与截断怎么算 | [GAE](#q=rl-returns-gae) |
| 张量怎样经过loss、backward、step | [PPO训练一步](#q=rl-ppo-training) |
| 不同算法如何共用一套接口 | [算法总览](#q=rl-policy-optimization-map) |
| 原论文的方法图、实验数字、配置与代码身份 | 本文及下面七篇精读 |
| 工具轨迹的credit、状态和系统 | [Agentic RL](#q=agentic-rl-training) |

![本文教学图 · 离线偏好与在线训练的修改位置](./assets/learning/policy-optimization-flow.svg)

**教学图解读。** 上行DPO从固定pair得到四个序列分数并做分类更新；下行在线方法从实际behavior采样，再经过reward、优势、ratio、surrogate和reduction。标签标出每种算法修改的位置，回线是下一轮权重发布，不是reference更新。图由本文为讲课绘制，展示逻辑依赖，不是作者方法原图或性能实验；GPU放置、工具时延和通信均未画入。

先给一个面试开场：**先分离离线偏好拟合与在线策略优化，再沿优势估计、概率比、更新约束、奖励组织和reduction比较。** 回答“哪个更好”前需要模型、数据、反馈、长度、预算与quality指标，不能只背缩写年代。

## 2. 范围、检索与覆盖边界

2026-10-08按用户列出的七种算法检索原文、arXiv固定版本、LaTeX/PDF及公开实现；补读OpenAI SpinningUp、ICLR PPO实现博客、HuggingFace DPO教程、Qwen GSPO与DAPO/GDPO作者项目。网络检索接口不可用时直接读取原站点和公开仓库，不用搜索摘要代替原文。

七篇保留五模块、原图、推导、手算、配置和问答，本轮进一步将问题背景与前置知识分开，加入固定源码代码块与逐段课堂讲解。PPO/DPO/GRPO/DAPO/GDPO具备原表并发布；GSPO/SAPO全文与原图已完成，但原文没有结果或实验表，依仓库现有规则保留单篇草稿。下文仍比较其方法并链接固定原文，未以别篇表格补证。未执行七种方法的重新训练，性能都标明论文条件。

| 名称 | 固定原文 | 单篇入口与状态 |
| --- | --- | --- |
| PPO | [1707.06347v2](https://arxiv.org/abs/1707.06347v2) | [PPO精读](#paper=paper-ppo) |
| DPO | [2305.18290v3](https://arxiv.org/abs/2305.18290v3) | [DPO精读](#paper=paper-dpo) |
| GRPO | [DeepSeekMath 2402.03300v3](https://arxiv.org/abs/2402.03300v3) | [DeepSeekMath精读](#paper=paper-grpo) |
| GSPO | [2507.18071v2](https://arxiv.org/abs/2507.18071v2) | [完整草稿](./paper-drafts/paper-gspo.md)；原文无表 |
| GDPO | [2601.05242v1](https://arxiv.org/abs/2601.05242v1) | [GDPO精读](#paper=paper-gdpo) |
| DAPO | [2503.14476v2](https://arxiv.org/abs/2503.14476v2) | [DAPO精读](#paper=paper-dapo) |
| SAPO | [Soft Adaptive 2511.20347v2](https://arxiv.org/abs/2511.20347v2) | [完整草稿](./paper-drafts/paper-sapo.md)；原文无表 |

GDPO排除同名GFlowNet和image-superresolution工作；SAPO排除single-rollout、self-adaptiveprocess等其他缩写。GRPO没有伪造一篇独立题名“Group Relative Policy Optimization”的原论文，而以真实DeepSeekMath全文收录。

## 3. 论文集合与方法谱系

### 用统一接口定位每篇修改

共同符号、公式推导和教学实现已集中到[算法总览](#q=rl-policy-optimization-map)，下面直接对照真实源码。阅读一个函数时按“输入形状 → 哪一量改变 → 是否可微 → 输出交给谁”四步解释。

| 论文 | 首先定位的代码 | 改变的对象 | 继续追踪 |
| --- | --- | --- | --- |
| PPO | GAE与`surr1/surr2` | 固定优势下的悲观更新目标 | old拷贝时机、minibatch与optimizer |
| DPO | 四个回答分数与`logsigmoid` | 偏好分类目标 | 回答mask、sum、reference缓存 |
| GRPO | prompt索引与mean/std | 回答级优势 | 广播后进入哪种policy loss |
| GSPO | mean log-ratio与detach | sequence权重、token梯度路径 | sequence归约和clip尺度 |
| GDPO | 分reward调用归一化 | 多奖励优势 | 最后whitening的统计单位 |
| DAPO | filter/refill、agg loss | 数据准入、clip、长度权重 | 超长reward与实际补采成本 |
| SAPO | sigmoid代理函数 | 平滑surrogate | 正负tau和对logprob的梯度 |

[统一课堂脚本](./assets/learning/rl-algorithms-unified-lab.py)使用可独立运行的小张量接口；下列原源码依赖各自框架。两者分开，便于先理解数学，再读部署系统。

### 关键源码对照：先找变量在哪一层改变

以下直接节选固定源码，统一去除共同缩进与非语义行末空白，保留语句与分支。它们依赖原函数的输入与外层状态；**可复制用于逐行讲解，不是七个可独立运行的trainer。** 原始PPO/DPO、作者GDPO、维护者现代verl与DAPO公开复现的身份各自标明。

#### PPO：固定优势之后的悲观更新

**真实源码节选：[PPO 原实现 · 裁剪策略目标](https://github.com/openai/baselines/blob/da997060461e3cbf54ca4dc7a67081a731fb6b3b/baselines/pposgd/pposgd_simple.py#L110-L113)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="PPO 原实现 · 裁剪策略目标"
ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
surr1 = ratio * atarg # surrogate from conservative policy iteration
surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
```



此段只定义actor surrogate。GAE需要在它之前得到固定优势，value/entropy还在别处；先用A正负四个例子判断minimum分支，再将符号转换成最小化loss。clip不直接裁参数、不直接裁A，也不把所有越界样本一起删除。

#### DPO：参考校准后的pair margin

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



输入是四个 `[B]`完整answer logprob，不是 `[B,L]`token advantage。原DPO关闭reference-free、IPO、label smoothing三个后续选项。policy差减reference差得到当前相对偏好变化；`logsigmoid`是训练目标，输出detached rewards只作日志。

#### GRPO：一组的scalar再广播

**真实源码节选：[GRPO 现代verl · 标准化并广播优势](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L324-L329)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="GRPO 现代verl · 标准化并广播优势"
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask
```



这一段位于no-grad与逐样本循环中；UID决定取哪个mean/std。输出 `[B,L]`是同一scalar沿有效token广播，不能解释成每token独立估计reward。正负优势、group std、reduction仍是不同环节。

#### GSPO：共享前向值但保留梯度

**真实源码节选：[GSPO 现代verl · 平均logratio与detach](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L1583-L1593)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="GSPO 现代verl · 平均logratio与detach"
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

# Combined ratio at token level:
# s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
# In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

# finaly exp() to remove log
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```



token差值先按mask取均值；`log_prob-log_prob.detach()`前向零、反向非零。最终ratio在sequence内相同，clip阈值也应按sequence尺度调。把两处detach全部删掉或全部保留，会改变梯度，不是无关的性能调整。

#### GDPO：reward维先保留，再合并

**真实源码节选：[GDPO 作者verl分支 · 拆分奖励再合并](https://github.com/NVlabs/GDPO/blob/4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115/verl-GDPO/verl/trainer/ppo/ray_trainer.py#L187-L202)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="GDPO 作者verl分支 · 拆分奖励再合并"
## handle correctness first
correctness_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_correctness,
                                                                eos_mask=response_mask,
                                                                index=index)

## handle format now
format_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_scores_format,
                                                                eos_mask=response_mask,
                                                                index=index)

new_advantage = correctness_normalized_score + format_normalized_score

advantages = masked_whiten(new_advantage, response_mask) * response_mask

data.batch['advantages'] = advantages
data.batch['returns'] = advantages
```



原分支只处理correctness/format两个维，逐维调用组归一化函数，再合并并whiten；不改变current/old ratio。按token mask做whitening时，长度还影响统计权重，不能与逐回答统计无条件等同。

#### DAPO：一个很短却改变样本权重的分支

**真实源码节选：[DAPO 公开复现 · token mean分支](https://github.com/verl-project/verl/blob/4f80e465c2ec79ab9c3c30ec74b9745de61d0490/verl/trainer/ppo/core_algos.py#L283-L284)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="DAPO 公开复现 · token mean分支"
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
```



它只展示token reduction，完整DAPO还包括filter/refill、upper clip和长度reward，详见单篇四段源码。两条回答长度2/4，每token值1/3，原sequence均值2，token均值7/3；两者都“训练token”，改变的是分母。漏掉补采控制流会把过滤写成更小的有效batch。

#### SAPO：用目标产生gate，不能把gate再当目标

**真实源码节选：[SAPO 现代verl · 可微代理函数](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L1657-L1659)。** 以下保留原始语句，仅去除共同缩进与非语义行末空白；变量初始化和未展示分支见原函数。

```python title="SAPO 现代verl · 可微代理函数"
def gate_function(x, tau):
    """The gating function used in SAPO"""
    return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)
```



返回的是f(r)，自动求导得到w(r)=4p(1-p)，对logprob的权重还乘r。命名gate_function不表示这里直接返回w。ratio=2、tau=1时w约0.7864、wr约1.5729；手工优化w(r)r则多出w′项，和SAPO不等价。


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

## 7. 控制实验与证据空白

七篇已保留固定原文、原图和机制解释，仍要区分原论文证据与新组合假设。GSPO/SAPO没有原文实验表，相关硬件和部分训练参数也未完整公开；DeepSeekMath没有公开原始GRPO完整训练代码，因此维护者实现只能作为伴读，不能冒充原实验程序。

若要做可比较实验，固定初始化、任务、verifier、采样预算、长度与训练资源。分别比较优势估计（critic/组baseline/逐reward）、ratio（token/sequence）、surrogate（hard clip/soft sigmoid/stop-grad weight），一次改变一层。DAPO应拆开过滤补采、clip、归约与超长处理，避免把总收益全部归给一个loss。

记录生成/评分/训练耗时、实际总生成tokens、环境调用、全局有效token数、G、过滤率、KL、ratio尾部、clip/gate分布、held-out任务成功与GPU-hour。组合不同层在接口上可行，只说明可以实现；收益是否叠加仍需独立对照。

## 8. 理论到代码的课堂顺序

| 顺序 | 讲解内容 | 当场完成的动作 |
| --- | --- | --- |
| 1 | 两bit任务、策略梯度与价值 | 列出四条路径，算真实V=[0.25,0,0.5] |
| 2 | GAE与固定标签 | 用不完美critic算两步advantages和returns |
| 3 | PPO一次更新与同批复用 | 核首轮ratio1但actor梯度非零，第二轮old不变 |
| 4 | 同题组反馈 | 用[0,0,1]与[0,1,1]算GRPO，再比较LOO |
| 5 | GSPO和SAPO | 解释detach的前向/后向，区分f、f′、rf′ |
| 6 | DAPO与GDPO | 指出collector、reward轴和reduction的位置 |
| 7 | 真实代码与原表 | 从上方节选进入单篇，说明版本与实验条件 |

### 两种代码怎样一起读

先运行[统一脚本](./assets/learning/rl-algorithms-unified-lab.py)，看[完整结果](./assets/learning/rl-algorithms-unified-lab-results.json)。脚本用一个可枚举MDP执行两次PPO step，再用同一六回答数据比较五种局部目标；mask位置必须没有actor梯度，GSPO两种接线只在指定条件下等价，global分母的拆批梯度应一致。

再去读本报告的真实节选：找出相同数学对象在原代码中叫什么，原实现增加了哪些配置或系统分支。例如GSPO现代分支有数值clamp和额外rollout权重接口；GDPO作者分支按token mask whiten；DAPO动态采样需要回到collector控制流。这些差异解释了为什么一个20行函数不能宣称复现完整论文。

原[局部策略演示](./assets/learning/policy-optimization-lab.py)继续用于PPO四分支、DPO四分数、GSPO detach和SAPO错误目标反例；[基础演示](./assets/learning/agent-training-lab.py)负责baseline方差与GAE边界。它们都是CPU教学验证，不包含模型benchmark。

### 面试如何回答算法比较题

按四句话组织：**共同骨架是什么；它改了哪一层；给一个公式、手算或代码分支；这个改动有什么代价和条件。** 例如GSPO与GRPO，共享组优势和近策略训练，主要改token ratio为几何平均sequence ratio与共享clip，随后说明长度归约和局部异常被平均的取舍。

分层练习题统一到[总览24问](#q=rl-policy-optimization-map)，PPO/GAE/GRPO主文继续保留自己的机制问答。涉及原论文具体结果，必须同时说模型、任务、采样预算和指标，不能只背一个提升数字。

## 9. 持续更新记录与来源

2026-10-15：将基础推导与教学公式集中到统一主线，本报告聚焦七篇真实源码与实验证据，保留原图表入口和实验条件。加入可枚举MDP到optimizer的CPU示例及同批梯度对照。


2026-10-09：将七篇背景区分为问题、前置知识、已有工作；添加16段固定源码与形状/梯度讲解，专题增加7段源码对照、教学数据流图、90分钟讲授顺序与CPU小张量脚本。脚本已在本地CPU运行通过，不是论文训练复现；GSPO/SAPO保持原草稿状态。

2026-10-08：完成七篇固定版本正文、原图、公开源码伴读与本专题；五篇发布，两篇因原文无表保留完整草稿。新增概率比/优势/reduction对照、GPU-hour口径、56道单篇问答与16道综合复习问题。未执行训练复现。

参考讲解的贡献限于教学入口： [SpinningUp PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)用于正负优势，[ICLR PPO实现细节](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)用于版本和边界，[HF DPO/TRL](https://huggingface.co/blog/dpo-trl)用于pair数据流，[Qwen GSPO](https://qwenlm.github.io/blog/gspo/)用于sequence单位，[DAPO项目](https://dapo-sia.github.io/)用于训练症状，[GDPO项目](https://nvlabs.github.io/GDPO/)用于多reward手算。各篇数字、曲线与公式以固定原文为依据。

来源版本、源码commit、图表label/hash与人工检查记录在单篇和图表清单；下一轮若更换原文版本，先更新证据和实验账本再改正文，不覆盖旧数字而不记录条件变化。
