---
id: "pt-rollout-staleness"
title: "Q15 · Rollout 为什么拖慢训练，异步之后如何控制策略滞后？"
category: "interview"
difficulty: "深入"
tags: ["后训练面试", "系统与 Agent", "字节", "阿里"]
updated: "2026-10-23"
summary: "将串行生成、长尾等待、数据交接与策略滞后一起分析，说明异步提吞吐的代价和可测边界。"
draft: false
---

## 60–90 秒回答

我会先 profile rollout、teacher/reward、learner、工具执行和权重同步的时间线。大教师评分可批量 prefill，学生生成却按 token 串行推进，多轮工具等待和少数长任务还会使整批交接出现气泡。优化可从连续批处理、长度分桶、部分就绪消费、资源配比和轮级异步入手，但异步会引入旧策略数据。每段响应必须保存真实 history、behavior log-prob 和版本；控制在途量、队列年龄、ratio 尾部与任务覆盖，并在适用条件下做校正。仅记录“落后两版”不等于概率偏差小，也不能声称局部 IS 消除了旧状态分布问题。

## 白板与例子

```text
同步整轨迹：短任务完成 ─────────── 等最长任务 ── 批量评分 ── 更新
轮级流水线：turn 完成 → 可评分 → 数据就绪 → 梯度累积 → 更新
                      └─ 环境工具执行与其他任务可交叠
```

这些是调度示意，不是实测时间比例。当前 turn 的监督何时可用，要看它是否依赖工具反馈或终局结果，不能把所有阶段都提前。

在固定已记录状态 $h$ 上，$\rho=\pi_\theta(a|h)/\pi_b(a|h)$ 可用于动作层的相应 IS 估计；但旧轨迹产生的状态占用仍来自旧策略。再加裁剪、截断、改变模板或 top-p 支持后，还需分别讨论偏差。

## 追问怎么接

**Rollout 慢是不是直接多加卡？** 先看 decode 计算、KV、prefill 重复、通信还是工具等待。卡空闲可能是请求不足或整批屏障；增加同类资源未必改善关键路径。

**怎么量化 staleness？** 同时看版本距离、墙钟年龄、实际条件下 log-ratio/ESS/clip fraction，并按长短任务拆分。ESS 是诊断，不是状态分布已匹配的证明。

**长任务慢，直接丢弃行吗？** 会改变难度和长度分布。应记录超时原因和保留率，在独立长任务集检查覆盖；公平吞吐应计所有预留资源与失败生成。

## 项目怎么联系

你的 Agentic OPD 可以讲响应逐轮进入教师评分、满足反馈条件后累积有效 token，actor 达到预算再更新，后续生成请求边界同步策略。另用已经结束且结果有效的轨迹更新 critic。说明两个就绪条件，别把“轮级”说成“每 token 即刻更新”。

## 容易答错的点

- 用异步后的每秒样本数单独宣布训练更有效。
- 用修改后的 summary/history 重算并冒充原始 behavior log-prob。
- 将环境错误重试、策略更新和 teacher 版本混为一个版本号。

## 闭卷自检

画出一条跨两次 actor 更新的三轮轨迹，为每轮标 policy version、请求边界、可用监督和最终回报到达时刻；指出哪些缓存可以安全复用。

[//]: # (interview-generated-start)

## 题源与延伸

**题目归类：公开题型改写。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。

| 来源 | 使用范围 |
| --- | --- |
| [字节算法岗公开题型汇总](https://www.nowcoder.com/discuss/922308546966847488) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 4 月；仅作题型依据 |
| [阿里算法岗公开题型汇总](https://www.nowcoder.com/discuss/923309531445071872) | 二次汇总，不按公司认证真题处理；公开正文；所选条目标注 2026 年 7 月；仅作题型依据 |
| [IMPALA / V-trace](https://arxiv.org/abs/1802.01561v3) | actor–learner 解耦和 off-policy 校正的经典依据 |
| [verl 性能调优文档](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) | rollout、batch、packing 和训练/推理内存取舍；参数支持随版本核对 |

**原理只在主文深入维护：** [05 · RL/OPD 训推如何形成闭环？从角色、数据契约到异步与排障](#q=infra-rl-pipeline) · [Agentic OPD执行流程、异步pipeline与Infra接线](#q=agentic-opd-pipeline)

[返回 20 题目录](#q=post-training-interview-practice) · [上一题](#q=pt-memory-parallel) · [下一题](#q=pt-agent-credit-mask)

[//]: # (interview-generated-end)
