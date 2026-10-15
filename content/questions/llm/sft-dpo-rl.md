---
id: sft-dpo-rl
title: "SFT、DPO 和在线 RL 分别在优化什么？"
category: llm
difficulty: 进阶
tags: ["P0", "后训练", "DPO"]
updated: "2026-10-15"
summary: "按数据来源、目标与反馈形式比较方法，避免把偏好优化等同于推理能力。"
draft: false
---

## 从问题出发

模型能够续写，不意味着会按要求完成任务。后训练用示范、偏好或结果反馈改变输出分布，三种信号解决的问题不同。

## 方法对比

| 方法 | 数据与反馈 | 直接优化量 | 主要风险 |
|---|---|---|---|
| SFT | 示范回答 | 示范 token 的负 logprob | 示范覆盖不足 |
| 标准离线 DPO | chosen/rejected 偏好对与参考策略 | 相对 logprob 的偏好损失 | 数据偏好偏差与分布错位 |
| 在线 RL | 当前/近期策略 rollout 与奖励 | 奖励导出的策略目标 | 奖励投机、方差和采样成本 |

固定离线数据和在线采样是常见设置，不是绝对边界；DPO 也有在线变体，SFT 也可持续更新数据。

## DPO 的一个标准公式

### 问题背景与理论假设分别是什么

问题背景是：有chosen/rejected偏好对，却不想先学一个显式reward model、再运行完整在线PPO。前置知识是KL正则最优化、Bradley–Terry偏好模型和自回归序列logprob。二者分开后，DPO可以理解为“利用特定模型假设，把偏好学习直接写成策略分类目标”，而不是“把两个答案做一次普通SFT”。

对固定prompt，考虑期望reward减reference KL的目标，其最优策略形式为：

$$
\pi^*(y\mid x)\propto\pi_{ref}(y\mid x)\exp(r(x,y)/\beta).
$$

取log后，reward可以写成$\beta\log(\pi^*/\pi_{ref})+\beta\log Z(x)$。同prompt两回答做差时，未知的$\log Z(x)$抵消；再代入Bradley–Terry的偏好概率$\sigma(r_w-r_l)$，得到下面的DPO loss。这条推导解释了为什么必须比较同prompt pair，以及reference为什么出现在公式里。

同一 prompt x 下，偏好回答 y_w 与 y_l，固定参考策略 pi_ref：

```text
delta = [log pi_theta(y_w|x)-log pi_ref(y_w|x)]
      - [log pi_theta(y_l|x)-log pi_ref(y_l|x)]
L_DPO = -log sigmoid(beta*delta)
```

标准序列 logprob 是有效回答 token logprob 的和；长度归一化等改动会改变目标。beta 不是学习率。DPO 在特定 KL 正则奖励建模假设下推导，不需要先显式训练奖励模型，但并不意味着任何偏好数据都对应真实任务奖励。

### 四个序列分数怎样落到代码

对chosen与rejected分别做causal forward，先按有效回答mask求token logprob，再沿token维求和得到两个`[B]`分数；reference以相同文本、模板和mask产生另外两个固定分数。

```python title="教学实现 DPO不是逐token比较chosen与rejected"
policy_margin = chosen_logp - rejected_logp
reference_margin = (ref_chosen_logp - ref_rejected_logp).detach()
preference_logit = beta * (policy_margin - reference_margin)
loss = -F.logsigmoid(preference_logit).mean()
```

Chosen与rejected长度可以不同，不需要逐token对齐。原目标使用回答logprob求和；改成长度平均会改变偏好模型对应的目标。Reference可以缓存，不意味着可以混用另一套template、tokenizer或mask生成的旧分数。

例如chosen=-2、rejected=-3，reference均=-2.5，beta0.1，得到margin0.1、loss约0.6444。Loss对chosen分数的导数为负、对rejected为正，鼓励相对偏好差；模型参数共享后，两个回答的绝对概率可能同时下降，所以margin提高不保证chosen概率每步上升。

## 为什么不能只比 loss？

SFT 的低 loss 可能来自短且重复的回答；DPO 的偏好准确率可能提高而事实正确性下降；RL 奖励可能被漏洞绕过。要用同一独立任务集、相同推理预算评测，并拆分能力、风格和长度变化。

## 小实验与自检

给同一个知识问答构造“简短正确”和“冗长错误”的偏好对，讨论标注者会不会误偏好后者。手算四个 logprob 的 delta，判断一次更新会鼓励什么相对变化。

自检：DPO 是否保证 chosen 的绝对概率一定升高？为什么相对 margin 变大与绝对概率变化不能混为一谈？

## 面试表达

先说反馈从哪里来，再说优化目标和独立评测。在线 RL 的优势构造继续看 [GRPO](#q=grpo-rlvr)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [DPO](https://arxiv.org/abs/2305.18290)：核对偏好目标与参考策略的作用。
- [InstructGPT](https://arxiv.org/abs/2203.02155)：理解示范、奖励模型与 RLHF 的阶段关系。


## 原论文精读与机制比较

[DPO原论文报告](#paper=paper-dpo)继续讲最优策略、梯度、官方代码与原实验；[RL算法总览](#q=rl-policy-optimization-map)按反馈、优势、ratio、目标和归约比较各方法。[统一CPU脚本](./assets/learning/rl-algorithms-unified-lab.py)的`dpo_loss`只接收四个回答分数，可和在线loss接口直接对照；[结果](./assets/learning/rl-algorithms-unified-lab-results.json)核对上述margin与梯度。
