# 权重/分片、ZeRO与训练后端 · 2026-10-05核查

用户补充范围包括ckpt/safetensors、分片格式、ZeRO、DeepSpeed/Lightning/FSDP，以及参数量和磁盘/推理显存的关系、1T算例。按三个独立主问题发布，延伸原基础链，不把全部细节塞回verl/slime主文。

| 新笔记 | 覆盖 | 问答 |
|---|---|---:|
| [06 权重与容量](../content/questions/systems/infra-checkpoint-formats.md) | 编码/内容/文件分卷/张量切分；共享storage；DCP/ZeRO恢复与导出；1T精度、checkpoint磁盘和KV显存 | 10 |
| [07 ZeRO](../content/questions/systems/infra-zero-deepspeed.md) | stage1/2/3所有权与一次更新；16P布局；通信假设；bucket、预取、持久参数；CPU/NVMe；配置与保存 | 8 |
| [08 后端](../content/questions/systems/infra-training-backends.md) | 原生循环/FSDP1/2、DeepSpeed engine、Lightning/Fabric/strategy；初始化、step、精度、保存与选型 | 10 |

## 原始资料与两轮审阅

第一轮核对PyTorch序列化/torch.load、DCP/fully_shard教程，Safetensors README/API/shared tensors，HF文件分卷与量化，DeepSpeed schema、ZeRO/Offload/Infinity、engine保存/转换。三篇合计33条来源登记，重复URL合并后27个来源。3张概念图均逐张查看，标为本文示意。

Lightning页面抓取返回JS壳，改读固定官方RST及strategy源码，不把搜索摘要当完整文档。固定commit：Safetensors `e246a2560645b7525f5775669ed816eb57c5bcc8`、DeepSpeed `a77aeb676507beb9fe0604bc31c2bf075daf7c46`、Lightning `84df182f50ab34301aabb3c0eb4031815bfb413d`。源码证明该Lightning FSDPStrategy调用FullyShardedDataParallel；ModelParallelStrategy/fully_shard组合另按固定教程说明。

第二轮独立核算1T各精度byte/TB/TiB、分组量化metadata、80GB/64GB的权重下界、128GiB KV例子和训练checkpoint12/14TB假设。明确常规驻留存MoE全部expert；激活参数不代表磁盘参数。HF分卷index通常映射完整tensor，不等于TP切片。更新边界checkpoint通常不含梯度/激活，不能把16P直接当磁盘常数。

ZeRO通信1.5倍仅在相同2字节参数/梯度及两次参数聚合一次梯度规约的教学模型成立。stage3聚合与offload不消除物理数据和峰值；1T显存/卡数只作明确单位与均匀分片下界，不写可部署保证。

## CPU实验实际证据

[脚本](../assets/infra/infra-checkpoint-check.py) / [结果](../assets/infra/infra-checkpoint-check-results.json)，PyTorch2.11.0+cu129、safetensors0.7.0，全程CPU；只创建临时小文件，结束后清理：

- 8参数FP32模型payload32字节；safetensors160=8+120header+32；pt1893；教学完整ckpt9048（含Adam、RNG等）。小文件倍数不推广到大模型。
- PyTorch→safetensors→张量、两文件whole-tensor index恢复均相等。
- 恢复Adam后的下一步loss0.20318153500556946，参数最大误差0，step/exp_avg/exp_avg_sq相等。
- CPU Gloo/DTensor两rank保存，一rank加载4×4张量，最大误差0；文件为.metadata和两个.distcp。
- 1T FP32/BF16/8bit/packed4bit数据4/2/1/0.5TB；group128、scale+zero4byte示例531.25GB。
- KV教学配置128GiB；2TB权重+该KV、每卡64GB均匀容量下界34卡；未计其他项与实际切分约束。

未验证项目：DeepSpeed/Lightning多GPU训练或恢复、FSDP多GPU数值、真实1T模型、offload吞吐与GPU服务延迟。DCP小张量实验不证明所有optimizer/版本/拓扑组合都可恢复。

## 发布与迭代

新三篇登记topics/source/figure manifests，来源观察覆盖固定官方仓库入口。学习路线、框架主文、并行/推理篇新增内部导航。线上检查依据注册表覆盖九篇，额外从正文发现练习下载并逐字节核对脚本/结果；原有来源/图片hash/公式/手机/目录检查继续保留。

执行质量、单元、build、浏览器与线上验收，以实际命令和Actions结果确认。正文与资料状态分开，观察差异不会自动改写已核版本；后续从具体实验/源码变化修订对应专题。

本地验收已通过：专题来源/图表/公式检查、29项单元测试、16项浏览器测试、39篇笔记构建；九篇专题在1440/390两种宽度核对正文/来源/图hash、练习脚本与结果字节、问答、目录、直接刷新及8个旧URL跳转。提交后Pages工作流还会重复线上验收。
