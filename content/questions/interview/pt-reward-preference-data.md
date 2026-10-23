---
id: "pt-reward-preference-data"
title: "Q04 · 奖励模型和偏好数据怎么做，如何验证标注真的可靠？"
category: "interview"
difficulty: "进阶"
tags: ["后训练面试", "SFT 与数据", "阿里"]
updated: "2026-10-23"
summary: "把偏好标签的可信度、reward model 的目标和独立评测区分开，避免模型在自己的评分闭环里自我确认。"
draft: false
---

## 60–90 秒回答

我会先定义什么叫好答案，再确定监督是结果正确性、相对偏好还是逐步 rubric。成对偏好必须来自同一个 prompt，标签应尽量反映目标能力，而非长度、位置和文风。奖励模型可以用 chosen 与 rejected 的分数差拟合偏好；即便用强模型打标签，也要做位置交换、重复评分和人工抽样校准。训练、标注验证和最终测试需要独立切分，保存 judge 版本与提示词。评价不能只看奖励模型训练准确率，还要检查它在学生新生成分布上的排序是否可靠，以及学生是否学会利用其漏洞。

## 白板与例子

以 Bradley–Terry 模型为例：

$$
P(y^+\succ y^-\mid x)=\sigma(r_\phi(x,y^+)-r_\phi(x,y^-)),
\quad L_{RM}=-\log\sigma(r^+-r^-).
$$

加同一个 prompt 相关常数不改变 pair 排序，因此 reward 的绝对数值并不天然可跨 prompt 比较。可验证结果、偏好 reward 与 critic 的 future return 也不是同一对象。

最小质量审计：同 prompt 的短正确/长错误答案做成一对，再交换展示顺序。若 judge 结论频繁翻转，先修评分可靠性；不能因为它“规模大”就自动相信。

## 追问怎么接

**有参考解就不需要 judge 校准了？** 复杂推导仍可能存在参考错误、等价解、遗漏条件和裁判能力不足。参考和 rubric 提供约束，不保证自动正确。

**多个奖励维度怎么合？** 先区分硬约束和可交换目标，再明确尺度与权重。先验证安全/格式有效等必要条件，再讨论质量；无条件加权可能让一个高分掩盖关键失败。

**DPO pair 的负样本越差越好吗？** 太容易区分的 pair 可能主要教风格；过难又可能标签不可靠。关注可判别的能力差异、难度覆盖与标签一致性，而非单追 margin。

## 项目怎么联系

PHYRD-40 可以作为例子：候选模型只见题面，judge 才见参考和 rubric；正式指标为 40 题分数均值，阈值通过率是另一个指标。可展示公开评分校验与环境变量配置，但不把换了 judge 后的分数当作论文原设置的复现。

## 容易答错的点

- 用同一 judge 生成偏好、优化学生并唯一验证效果。
- 混淆 reward model、LLM judge、reference 和 critic。
- 用平均奖励上涨代替独立任务质量改善。

## 闭卷自检

设计一张含位置偏差、长度偏差、标签冲突和学生新分布四项检查的审计表，并说清每项失败后改变数据、评分器还是训练目标。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [阿里国际「国际一面」（2025）](https://www.nowcoder.com/feed/main/detail/166d1f3bb6b84624b912dcc3997f8081) | 经历自述，招聘真实性未独立核验；检索可见正文；直链读取返回简页；仅作题型依据 |
| [阿里算法岗公开题型汇总](https://www.nowcoder.com/discuss/923309531445071872) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 7 月；仅作题型依据 |
| [TRL DPO 官方文档](https://huggingface.co/docs/trl/en/dpo_trainer) | 偏好格式、reference log-prob 和训练实现 |
| [LLM-as-a-Judge / MT-Bench](https://arxiv.org/abs/2306.05685v4) | 位置、长度及自我偏好等裁判偏差 |

**原理只在主文深入维护：** [Reward Model、PRM与LLM Judge：监督、校准和评测面试](#q=reward-judge-interview) · [SFT、DPO 和在线 RL 分别在优化什么？](#q=sft-dpo-rl)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-lora-budget) · [下一题](#q=pt-ppo-four-models)

[//]: # (interview-generated-end)
