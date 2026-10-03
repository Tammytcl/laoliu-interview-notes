# 异步 RL / OPD 指定论文收录核查 · 2026-10-04

本批覆盖用户指定 11 篇：10 篇发布，DORA 完整草稿。固定首版便于核对用户引用的结果，未来升级版本应连同作者、图表和数字重审。未执行 GPU 训练。所有原图和原表逐张核对；证据清单保存资产/TeX/PDF hash，源包在 gitignored 缓存。

| 记录 | 固定版 / PDF 页数 | PDF SHA256 | 原图表文件 | 状态 |
|---|---|---|---|---|
| [asyncopd](../content/papers/infra/paper-asyncopd.md) | [2606.24143v1](https://arxiv.org/abs/2606.24143v1) / 18 | `33243254ecb041a8c0ae9557fb8d60912fba08752aa32ccce981d59cd3055a2a` | method-source.png, kl-comparison-source.png, table-2-pdf.png, table-7-pdf.png | 发布；结构与图源核对 |
| [heddle](../content/papers/infra/paper-heddle.md) | [2603.28101v1](https://arxiv.org/abs/2603.28101v1) / 14 | `3343346d366edb9abea3448d93a5430127cbb28cd2449a2521675c10216735d0` | method-source.png, throughput-source.png, table-1-pdf.png | 发布；结构与图源核对 |
| [tiderl](../content/papers/infra/paper-tiderl.md) | [2608.10402v1](https://arxiv.org/abs/2608.10402v1) / 19 | `71e5b879bd6c3f0b22dcea152db6b2704d84aab8b22183358c0c6c37b47a5369` | method-source.png, text-throughput-source.png, multimodal-throughput-source.png, table-2-pdf.png | 发布；结构与图源核对 |
| [dora](../content/papers/infra/paper-dora.md) | [2604.26256v1](https://arxiv.org/abs/2604.26256v1) / 15 | `b288abb5a87514c1063260a6b8ae22051ddb8017987f1fabf7fa569dd64f1e07` | method-source.png, throughput-source.png, convergence-source.png | 完整草稿：作者原文无表 |
| [areal](../content/papers/infra/paper-areal.md) | [2505.24298v1](https://arxiv.org/abs/2505.24298v1) / 18 | `943c1fe1e37ef360f400505757835cbb69d28308f8087b208fd4b9ed0c0bd5c2` | method-source.png, scaling-source.png, table-1-pdf.png | 发布；结构与图源核对 |
| [pipelinerl](../content/papers/infra/paper-pipelinerl.md) | [2509.19128v1](https://arxiv.org/abs/2509.19128v1) / 16 | `80a8453aa2bd1a52d1344cab3a20c213cd2a1b5e787f7811ea7b83a22089ff56` | method-source.png, training-curves-source.png, table-1-pdf.png | 发布；结构与图源核对 |
| [april](../content/papers/infra/paper-april.md) | [2509.18521v1](https://arxiv.org/abs/2509.18521v1) / 18 | `1c0e68ee6d0ed02f4da57295e6122235d4e56739950ce1c27e2d09d1ccb3d426` | method-source.png, throughput-source.png, off-policy-source.png, table-1-pdf.png | 发布；结构与图源核对 |
| [sao](../content/papers/infra/paper-sao.md) | [2607.07508v1](https://arxiv.org/abs/2607.07508v1) / 14 | `44c695be0428c666d06c914ba76c037e3ac77eeb5db0a81bbe239719c21bda48` | method-source.png, training-curves-source.png, table-1-pdf.png, table-2-pdf.png | 发布；结构与图源核对 |
| [stride](../content/papers/infra/paper-stride.md) | [2609.14636v1](https://arxiv.org/abs/2609.14636v1) / 14 | `1a145ddd0307ae642e91ae64766e6e70ae0aab27acee6c1919f9c71fad378d59` | method-source.png, wall-clock-source.png, table-1-pdf.png | 发布；结构与图源核对 |
| [flexmarl](../content/papers/infra/paper-flexmarl.md) | [2602.09578v1](https://arxiv.org/abs/2602.09578v1) / 14 | `a934d80837f7026e45d516f113c107d540dd7d6d05021e10503d0909821521b5` | method-source.png, microbatch-source.png, table-2-pdf.png, table-3-pdf.png | 发布；结构与图源核对 |
| [rolloutpipe](../content/papers/infra/paper-rolloutpipe.md) | [2606.26997v1](https://arxiv.org/abs/2606.26997v1) / 15 | `ae96789b43b4ff8f01cf64deec2e98ce2bc8f16acd76b7caed40c7c6a97e8d9b` | method-source.png, timeline-source.png, training-time-source.png, table-1-pdf.png | 发布；结构与图源核对 |

## 代码与证据边界

官方代码核读固定 AsyncOPD `e3f79e916aad9dba886256d202bab96f8504a447`、AReaL `2fad2d0e308fe631e70e971b97188ad5c5cc03cb`、PipelineRL `58d393458625ad63ed539f2dcd072c85700c557f`、APRIL `adbb0175f010d66cdae215df8ffaca76aa142de6`。当前实现与原论文实验 commit 不等价；其余七篇未发现可核验官方实现，未补第三方仓库冒充。

DORA 无原表，遵循现有证据工作流保留 draft；其方法、实验与3张原图已整理，专题引用固定原文。替代证据尚待用户明确选择，未放宽质量脚本或添加无关表。

事实修正：FlexMARL方法图为Figure2、GA为Figure4；RolloutPipe架构/时间线为Fig2/3。RolloutPipe每U组更新、每R组消费完发布；TideRL消融百分比与吞吐相除不完全一致；STRIDE +1.3%是特定mean相对提升；PipelineRL常规基线时间含估算。

## 下一批可复用流程

1. 在Daily列范围、固定题名/版本/作者单位，先核查原文是否有足够证据。
2. 下载固定PDF/LaTeX，清除注释后定位有效Figure/Table；登记crop计划与hash，优先提取独立资产。
3. 写五模块与配置账本；有官方实现就固定commit并追2–4条关键路径。
4. 按每张图的坐标、图例、全部子图、表头末行核读，写紧邻解读后才标verified。
5. 用专题按问题比较交付、backward、optimizer、publish、buffer；不拼不同口径加速排名。
6. 单篇质量、全库strict、单元测试、build与浏览器检查通过后提交推送，再核验线上图片与发布运行。

本批没有新增自动定时检索任务；这是本次收录与可重复的仓库更新流程，不宣称无人值守事实审核。
