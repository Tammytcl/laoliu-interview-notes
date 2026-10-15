---
id: grpo-rlvr
title: "GRPO从组奖励到token梯度的完整流程"
category: agent
difficulty: 进阶
tags: ["P0", "GRPO", "RLVR", "Policy Gradient", "Implementation"]
updated: "2026-10-15"
summary: "区分可验证反馈与优化算法，按分组评分、优势广播、概率比、clip和reduction讲清GRPO，配真实源码入口与手算问答。"
draft: false
---

## 1. 问题背景 为什么想去掉critic

常见PPO训练中，actor负责生成，critic学习从各状态出发能得多少回报。长推理与大模型使critic的训练、显存和部署成为成本；而同一个数学题往往可以生成多份答案，并由判题器给出成功或失败。GRPO利用这些同题答案的相对表现构造baseline，省去学习V的模型。

**RLVR与GRPO不在同一层。** RLVR说明反馈可以被规则或环境验证，例如数学答案匹配、代码测试、约束检查；GRPO说明怎样将这些反馈变成策略更新。RLVR可以配PPO、RLOO等方法，GRPO也可以使用学习到的评分模型。先区分反馈与优化，再讨论组合。

先修按[Actor Critic](#q=rl-actor-critic) → [PPO完整一步](#q=rl-ppo-training)阅读。原论文图表与实验在[DeepSeekMath精读](#paper=paper-grpo)；后续变体统一放在[算法总览](#q=rl-policy-optimization-map)，本篇只把outcome GRPO完整讲清。

## 2. 前置知识 一组回答和三个策略

一个prompt采$G$条回答，每条回答拿到一个结果reward $R_i$。$G$是同题回答数，不是batch大小、回答token数或GPU数。若两个prompt各采3次，训练展平后有6条回答，仍必须知道每条属于哪个prompt。

$\pi_{old}$在这一轮固定并提供比较概率；$\pi_\theta$是learner当前可微策略；$\pi_{ref}$提供长期行为锚点。同步简化场景中old也生成数据；行为采样经过temperature/top-p或异步版本变化时要另外核对采样契约。Reference参与KL，不用它来计算PPO式current/old更新比。

Outcome reward只有一个回答级标量；token logprob则是每个动作token一个数。**把回答级优势广播到token，不等于已经获得细粒度过程奖励。** DeepSeekMath同时研究process supervision，不能将outcome例子当成所有GRPO形式的定义。

## 3. 从reward到优势 先手算再看代码

同prompt组均值与标准差为$\bar R,\sigma_R$：

$$
A_i=\frac{R_i-\bar R}{\sigma_R+\epsilon}.
$$

例如$G=3$，reward为[0,0,1]，组均值1/3。采用样本std得到$\sigma=\sqrt{1/3}\approx0.57735$，优势约[-0.57735,-0.57735,1.15470]。前两条比同组通常表现差，第三条比通常表现好；即使reward为0，也不是“标签缺失”，而是可以相对抑制的反馈。

若另一组是[0,1,1]，优势约[-1.15470,0.57735,0.57735]。不能将两组混在一起求全batch均值再称为同prompt baseline，否则不同题目的难度分布会相互影响。

```python title="教学实现 完整同题组的优势"
# rewards: [Q,G]，此例G>=2，显式采用样本std
with torch.no_grad():
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, correction=1)
    answer_adv = centered / (std + 1e-6)
# answer_adv: [Q,G]；随后按同一顺序展平为[B]
```

epsilon只处理分母数值问题，不产生新监督。全组[1,1,1]或[0,0,0]的centered都是0，组相对reward项失去区分能力；如果有独立KL项，整体训练仍可能更新。

### 组std实际改变了什么

不除std时，reward差距越大通常带来更大的更新系数；除std后，不同prompt的组内尺度被拉近。因此局部std不只是防止梯度过大，也会改变题目之间的相对权重。有限组统计受到成功概率、G与reward设计影响，“组标准化”不能直接翻译为“严格无偏优势”。

与RLOO比较，减组mean还包含了当前样本自身。Leave-one-out优势为$A_i^{LOO}=G(R_i-\bar R)/(G-1)$；它排除自身，支持给定prompt下独立baseline的证明。关闭GRPO的std仍差这个倍率，完整讨论见[算法总览](#q=rl-policy-optimization-map)。

## 4. 从回答优势到每个token的更新

把回答$i$的$A_i$沿有效生成token广播：

$$
r_{i,t}=\exp(\ell_{\theta,i,t}-\ell_{old,i,t}),\qquad
U_{i,t}=\min(r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon_l,1+\epsilon_h)A_i).
$$

梯度经过当前logprob $\ell_\theta$。旧logprob和优势固定，工具返回、reward计算和组统计不需要可微。

```python title="教学实现 从[B]优势到[B,L]loss"
# logp/old_logp/action_mask: [B,L]，answer_adv: [B]
adv = answer_adv[:, None].detach()
ratio = (logp - old_logp.detach()).exp()
plain = ratio * adv
bounded = ratio.clamp(1 - eps_low, 1 + eps_high) * adv
loss_items = -torch.minimum(plain, bounded)
lengths = action_mask.sum(-1)
valid = lengths > 0
actor_loss = (
    (loss_items * action_mask).sum(-1) / lengths.clamp(min=1)
)[valid].mean()
```

这是回答内平均、再回答间平均的教学核心，不含KL等额外项。若全batch没有有效动作应跳过或报错；真实框架还要处理跨卡global分母。将最后几行换成全部有效token mean，长回答的总权重会增大，因此不能把两种reduction视为完全同一目标。

### 同一优势不代表每token相同梯度

假设一条成功回答$A=1$，两个token ratio为[2,0.5]，epsilon0.2。第一个已经处于正优势上平台，PPO式这一项梯度为0；第二个仍有提高概率的梯度。再加上不同prefix的网络导数，各token梯度本来就不同。

这也解释了GSPO为什么另改ratio与clip的共享粒度：GRPO提供回答级优势，不自动使token级更新权重一致。更细粒度的credit则需额外的process reward、状态比较或其他估计。

## 5. Reference KL在哪里 梯度是什么

原GRPO包含reference KL正则。一个常见采样形式令$d=\ell_{ref}-\ell_\theta$，使用：

$$
\widehat D=e^d-d-1.
$$

其逐样本值非负，若动作确实来自当前策略，在固定prefix下对动作取期望可得到$D_{KL}(\pi_\theta\|\pi_{ref})$。这说明数值估计的方向；若轨迹来自old策略，或优化时把采样当固定，期望与梯度等价还需要检查，不能仅见这个公式就宣布完整KL的无偏梯度。

```python title="教学接线 采样KL作为独立loss"
d = ref_logp.detach() - logp
kl_items = d.exp() - d - 1
total_loss = actor_loss + beta * reduce_with_same_contract(kl_items, action_mask)
```

这段只表达一种实现路径。KL放入reward再参与优势，和learner直接加入可微loss的传播方式不同。某些后续配方不保留KL，也仍被框架放在GRPO系列入口；要读真实配置，而非凭函数名猜目标。原式、三策略与实现边界见[GRPO精读](#paper=paper-grpo)。

## 6. 真实verl源码怎样对应上面流程

固定commit的[组优势函数](https://github.com/verl-project/verl/blob/75879f7f475fd6b64c779f7d9212e45503f58b8f/verl/trainer/ppo/core_algos.py#L268-L331)先将token reward求和为每回答分数，再按`index`收集同prompt组，计算mean/std，最后：

```python title="真实verl节选 组统计后广播mask"
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
scores = scores.unsqueeze(-1) * response_mask
```

`scores`由`[B]`变成`[B,L]`，不是给每个token重新评分。`norm_adv_by_std_in_grpo=False`只去std，不替换下游ratio或归约。该版本对singleton组另设mean0/std1；这是其实现分支，不是G=1自动具有组相对统计。

阅读顺序应是：调用侧如何分配prompt UID → reward放在哪些token → 本函数怎样分组 → policy loss如何使用advantages → aggregation采用哪个分母。只复制中间归一化六行，会漏掉算法实际使用的数据语义。

## 7. 代价与Agent场景中的边界

省掉critic后，主要代价转移为同题G次rollout、环境执行与评分。G越大，通常能获得更充分的组比较，也可能增加长尾和工具成本；它不保证单位GPU-hour收益提高。同组要等到足够回答和评分完成才能确定组统计，形成group-completion依赖。

对多轮code agent，环境测试是否可信、真实输入能否重现、工具文本是否误进loss、终止是否被截断，都比换一个算法名更基础。终局结果相同的两条轨迹可能包含完全不同的中间决策，outcome组优势无法自动定位哪一轮最关键。[Agentic RL](#q=agentic-rl-training)继续讨论信用分配与实际环境研究。

比较GRPO与PPO需要固定任务、初始化、reward、生成预算、最大长度、训练资源与独立评测。PPO的critic成本、GRPO多采样成本以及过滤补采的所有失败尝试都应计入；只比较learner step/s会遗漏主要开销。

## 8. 面试问答

**Q01 [分类] RLVR就是GRPO吗？** RLVR是反馈来源，GRPO是优化方法；可验证reward可接其他算法。追问：verifier有漏洞会怎样？

**Q02 [baseline] 为什么省掉critic？** 用同prompt回答的reward统计构造相对系数，无需学习V。追问：rollout成本是否反而增加？

**Q03 [手算] [0,0,1]的优势是多少？** 样本std下约[-0.577,-0.577,1.155]。追问：population std会改变多少？

**Q04 [零方差] 全组成功还能学吗？** 组相对reward项通常为0，KL等独立项仍可作用。追问：动态过滤补采改了哪个环节？

**Q05 [粒度] 一条回答一个优势等于每token同梯度吗？** 不同ratio、clip分支与网络导数都会改变token梯度。追问：GSPO怎样改共享粒度？

**Q06 [概率] old与reference能互换吗？** old用于更新比较，reference用于行为锚定。追问：谁随每轮采样刷新？

**Q07 [组别] 为什么必须存prompt UID？** 数据打乱和多卡重排后仍须恢复正确同题组。追问：可变G如何处理？

**Q08 [估计] 去std就严格等价RLOO吗？** 还差G/(G-1)，也须明确独立采样和更新目标。追问：singleton分支有什么语义？

**Q09 [长度] 两条回答平均和全部token平均相同吗？** 只有等长等条件下才可能一致，否则长短回答总权重不同。追问：DDP局部mean有什么问题？

**Q10 [KL] exp(d)−d−1就是完整KL梯度吗？** 数值期望、采样分布与固定样本的求导路径需分别分析。追问：在reward和loss里同时扣是否重复？

**Q11 [Agent] GRPO解决中间轮credit了吗？** Outcome形式通常仍用轨迹级系数，可能需要过程反馈或状态级估计。追问：正确state应包含哪些工具信息？

**Q12 [排障] 任务reward高但训练没区分信号怎么办？** 检查组内方差、题目难度、G、反馈和KL，再考虑采样/归一化修改。追问：是否只剩全成功或全失败题？

## 9. 来源与可运行练习

原始[DeepSeekMath v3](https://arxiv.org/abs/2402.03300v3)给出GRPO及实验；[论文精读](#paper=paper-grpo)保留原图、原表和代码身份。[来源清单](./research/grpo-rlvr-sources.json)固定本篇源码版本。

[统一课堂脚本](./assets/learning/rl-algorithms-unified-lab.py)在两prompt各三回答的同批样本上比较GRPO与变体，[结果](./assets/learning/rl-algorithms-unified-lab-results.json)包含所有mask和logit梯度。先预测哪些token进入clip平台，再看输出。继续在[算法总览](#q=rl-policy-optimization-map)逐层比较GSPO、DAPO、GDPO、SAPO；实验与真实源码回[七篇对照报告](#report=survey-policy-optimization)。
