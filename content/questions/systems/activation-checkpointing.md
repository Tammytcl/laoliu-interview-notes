---
id: activation-checkpointing
title: "Activation checkpointing 为什么能省显存？重算有哪些正确性陷阱？"
category: systems
difficulty: 进阶
tags: ["P0", "重计算", "训练显存"]
updated: 2026-09-30
summary: "把保存激活与重新前向的成本联系起来，定位随机性和状态副作用。"
draft: false
---

## 从问题出发

训练反向需要前向中间量。Checkpointing 不保存全部中间量，而是保存少量边界，在 backward 时重新计算需要的部分。

## 内存与算力交换

```text
普通：forward 保存大量激活 → backward 读取
重算：forward 保留区段输入 → backward 重跑区段 forward → 求梯度
```

它减少的是被 checkpoint 区段内保存的激活，不会自动减少权重、Adam 状态或长期 KV。粒度越细/覆盖越多不一定越好，重算时间、边界输入和框架调度都影响收益。

## 正确性边界

重算应对应原 forward 的数学行为。Dropout 需要合适 RNG 状态；改变全局变量、在 forward 中修改缓存、依赖外部状态或重复副作用，都可能破坏一致性。混合精度和自定义算子也要按实际实现验证。

PyTorch 存在不同 checkpoint 实现选项，其梯度支持和行为不同；写代码前按已安装版本查看官方文档，不盲目复制旧示例。

## 与并行训练的关系

FSDP/TP 改变参数和算子分布；重算改变激活生命周期。组合可能降低峰值，也可能改变通信时机。节省显存后增大 microbatch，吞吐是否改善必须实测，而不是只看额外 forward 次数。

## 小实验与自检

固定小模型输入、权重、seed 和 dropout，比较无重算与有重算的 loss/梯度在容差内是否一致，同时测峰值内存和 step 时间。再给 forward 加一个计数副作用，观察它为何会执行更多次。

自检：训练状态是最大项时，checkpointing 为什么帮不了太多？为何同一函数重算可能不等于原 forward？

## 面试表达

用保存与重算的生命周期解释原理，指出它以计算换激活内存，再说明 RNG 和副作用的正确性约束。

关联：[显存账本](#q=memory-budget)、[分布式训练](#q=distributed-training)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [PyTorch checkpoint](https://docs.pytorch.org/docs/2.14/checkpoint.html)：核对 RNG 保存、实现选项及重算限制。
