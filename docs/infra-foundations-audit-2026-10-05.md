# Infra基础复习五篇 · 调研与审阅记录

本轮目标：在已有verl/slime框架文章之外，新增五篇详细且可顺序复习的Infra内容，从训练基础到训推闭环；结合论文、官方教程、作者博客/分享与面试讨论，提供知识解释、算例、区别、问答和练习。五篇皆为独立已发布学习笔记，不混入论文库，也未修改DORA草稿状态。

| 笔记 | 主问题与输出 | 面试整理题 | 插图 |
|---|---|---:|---:|
| [训练一步](../content/questions/systems/infra-training-step.md) | 参数/梯度/optimizer、batch、dtype、显存与OOM | 8 | 1张本文生命周期图 |
| [GPU性能](../content/questions/systems/infra-gpu-performance.md) | 算力/字节下界、collective、计时与profile | 8 | 1张本文Roofline示意 |
| [分布式并行](../content/questions/systems/infra-distributed-parallelism.md) | 状态、矩阵、层与序列切分、16卡布局 | 8 | 1张本文切分示意 |
| [推理引擎](../content/questions/systems/infra-inference-engine.md) | 请求、KV、kernel、batch/cache、调参验收 | 8 | 1张本文请求图 + PagedAttention v1 Figure6 |
| [RL/OPD闭环](../content/questions/systems/infra-rl-pipeline.md) | 数据契约、概率、四边界、queue、版本与排障 | 10 | 1张本文流水线图 |

## 资料与事实边界

逐篇sources manifest共54条登记，URL去重后为43个来源。主要为PyTorch、NVIDIA/NCCL、DeepSpeed、verl/SGLang/vLLM文档，Adam/ZeRO/Megatron/GPipe/PipeDream/FlashAttention/PagedAttention/Orca/SGLang/Speculative/HybridFlow/GRPO及已收录异步论文；博客/分享包括PyTorch重算、vLLM架构、How To Scale Your Model与USENIX作者页面。作者面试讨论采用Chip Huyen资料，所有42道问题是本文整理，不冒充具体公司真题。

arXiv引用固定当前核对版本，PagedAttention图固定v1。代码入口固定：verl `8718ca30a3f002f93b7c4fd99b9b2506718681bc`、slime `8c17b676cb57af1d17ee4402e91e9209af84b60b`、Megatron `ef8cd10b8e34a94c9ad7452e7851f58337f353a0`。本次核对文档、代码树与slime异步README，未声称新完成全面源码分析或GPU复现。

`topics:watch`扩展到六篇，新增三仓库入口与slime README基线。框架样稿原有Megatron来源观察出现候选变化，保留待核对状态；新笔记使用当前固定入口。无watch项的教程/博客在后续调研人工检查，不把0 observed解释成内容永久不变。

## 两轮内容审阅

第一轮：按读者先修顺序审视独立主问题；检查Adam、梯度平均、BF16、模型状态与激活、Roofline、ring假设、TP/PP/CP/EP、推理指标与队列稳定性。保留参数/版本/计时口径，GPU片段明确为教学待实施。

第二轮：独立CPU复算并有限差分检查梯度；分块softmax含极端score，与整块输出一致；TP切分后逐项比对MLP输出。逐图查看后发现PagedAttention Figure4不对应拟写映射说明，改用Figure6并按A0/1/2→物理7/1/3，B0/1→物理5/2重写图解。5张自绘示意标为概念图，不能当实测性能证据。

核对固定slime README：ABORTED组重新入buffer，但partial resume尚未接通，不能写“已经续跑环境”。verl `staleness_threshold`按文档比例/投放公式解释，不当版本差整数阈值。保留action/prefix staleness区别、工具状态恢复和完整组交付的边界。

## 数值证据

[标准库CPU脚本](../assets/infra/infra-foundations-check.py)与[本次输出](../assets/infra/infra-foundations-check-results.json)：

- 整体与加权分块梯度=-22.5；平均局部平均=-31；有限差分=-22.500000000036376。
- 特定7B状态=112GB=104.308GiB；D8状态分片=[112,38.5,26.25,14]GB；指定GQA KV=8GiB。
- B1/B128强度约0.9995/120.4706；假想设备下界已复算，不对应实际硬件性能。
- ring D8发送1.75GiB，25GB/s假设下传输75.162ms。
- 分块softmax与整块值均6.510162675192582；两份TP结果与完整MLP一致。
- P4、m4/m16的简化bubble为3/7与3/19；无背压生产12消费8，300s净积压1200。

## 工作流验收

注册五篇来源/图表/必要章节与深度约束，构建自动发布新来源清单。线上检查由固定单篇扩成注册表全量：逐篇核对已部署正文、来源JSON内容、图文件hash、公式、问答、比例、手机溢出、目录与刷新。浏览器读者流程还检查框架→第一篇→第二篇及CPU练习下载。

实际执行结果在交付前核验，不以本文文字替代命令成功证据。未实施的项目：GPU训练/多卡数值、服务benchmark、真实coding环境snapshot、无人值守LLM写作。未来迭代从具体实验缺口与上游变更账本进入，沿用现有推送与Pages部署流程。

本地验收已通过：专题质量检查、29项单元测试、15项浏览器测试；构建包含36篇学习笔记。六篇专题在1440/390两种宽度核对正文、来源与图hash、公式、问答、目录与刷新，8个旧URL跳转通过。提交后的线上验收由Pages工作流重复执行，以Actions运行结果为准。
