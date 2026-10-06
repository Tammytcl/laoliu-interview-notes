# AI算力卡专题 · 2026-10-06核查

用户要求新增A/H/B显卡、其他AI算力卡、架构与训练区别。按NVIDIA A100/H100/H200/B200理解主线，并明确口语“A卡”可能指AMD；AMD单列。正文一篇独立专题，连接原GPU/显存/并行/框架文章。

## 范围与资料

[专题正文](../content/questions/systems/infra-ai-accelerators.md)约1.8万字符、12道本文整理问答、1张本文约束图与1张NVIDIA GH100原图。覆盖Ampere/Hopper/Blackwell、B300/GB200系统形态、AMD CDNA3/4、Gaudi、TPU、Trainium、昇腾，及L40S/RTX与训练角色的边界。来源清单34个官方/作者资料；没有跨卡GPU测速、采购价格或云端供给排名。

第一轮核对产品/架构/层次和软件栈，第二轮复查具体列、脚注、单位、数字与图源。架构机制回到作者技术文档，兼容性回到CUDA CC/Blackwell指南、ROCm/PyTorch HIP与各厂商训练入口。

## 规格口径的关键修正

- A100固定80GB SXM，HBM2e2.039TB/s；PCIe1.935另说明。
- H100 SXM80GB3.35与NVL94GB3.9分别看；H100/H200网页1979TFLOPS带稀疏脚注，dense约一半。
- Blackwell固定PDF第8页HGX B200单GPU180GB/7.7TB/s，稀疏BF16=4.5PF，dense=2.25PF；GB200列186GB/8TB/s、dense2.5PF。
- Ultra固定PDF第5页HGX B300单GPU270GB/7.7TB/s、denseBF16=2.25PF；GB300列279GB/8TB/s。未用发布期192/288数字替代，不推断差异原因。
- Neuron Trainium2架构文档标每chip96GiB；EC2实例页系统标签1.5TB保留原口径，不混算或当每chip容量。
- GH100图是2022年完整144SM示意，启用SKU配置另看；GPU、Superchip、HGX与NVL72不能当同一层。
- TPU7x当前官方有训练recipe，不据早期推理定位断言不能训练。Gaudi/CANN资料只支持对应接入，不把forward可跑等同于完整训练链路。

## 固定证据

[来源清单](../content/research/infra-ai-accelerators-sources.json)登记全部引用与核对日。两份PDF保存于gitignored缓存，登记源URL/hash并监控hash变化：

| 证据 | SHA256 | 核查位置 |
|---|---|---|
| Blackwell datasheet | `ad8ed65e64974278670d8e6904c0e3020391027925e339bf28e75fdfe2d307d7` | PDF第8页，具体产品列和稀疏脚注 |
| Blackwell Ultra datasheet | `ccdffc064731cc160016f4c7a77ecfa37983038bcaf60e39190ba4175b39c7c8` | PDF第5页，HGX/GB300列 |
| NVIDIA GH100原图 | `7df1ff29256dca70ab93eb2fce75b9a9b734e5b47bc62c2e5d99905641fabd20` | 从Hopper作者文章原资源下载，逐节点核对 |

两张插图metadata记录出处、hash、归属、视觉核查和解读范围。固定PDF可仍被上游同URL替换，观察不会覆盖正文已核基线；其他动态产品页下轮人工核查。

## CPU验算与性能边界

[脚本](../assets/infra/infra-accelerator-check.py) / [输出](../assets/infra/infra-accelerator-check-results.json)，仅Python标准库：

- 70B、16字节/参数布局1120GB总状态；D8/D16每rank140/70GB。
- 假想F100TFLOPs、Q400GB、dense989.5TFLOP/s：H100理想下界119.4ms、H200101.1ms，约1.18倍；带宽比1.43不直接等于速度比。
- 1T BF16纯权重2TB，每设备80%名义容量，80/180/270GB容量设备的均匀分片下界32/14/10。
- 400Gb/s=50GB/s；各产品Roofline转折点按dense FLOP/s与HBM byte/s复算。

这些不是真实kernel、集群或训练收敛实验；未计峰值/切分约束的设备数不是可部署机器清单。没有把其他厂商TOPS与NVIDIA稀疏FP8峰值组成速度总榜。

## 发布检查

注册新专题，原Infra列表成为十篇、全站学习笔记40篇；更新路线、框架和GPU基础入口。既有质量/来源/图hash/公式门槛继续执行，新增浏览器场景检查SKU、Trainium单位、原图和CPU结果。线上验收由注册表覆盖十篇，不仅检查新增标题。

本地实际验收通过：专题质量、29项单元测试、17项浏览器测试、40篇学习笔记构建；十篇专题×1440/390宽度核查正文、来源JSON、图片hash/显示比例、公式、问答、脚本/结果字节、目录和直接刷新，8个旧URL跳转通过。两份PDF自动hash观察成功，与人工基线一致。其他原有专题上游commit变化保留为候选，不因此改写已核正文。
