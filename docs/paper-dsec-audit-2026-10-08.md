# DSec精读证据账本 · 2026-10-08

用户指定DeepSeek DSec，固定首版2609.22978v1（abs页当前仅v1），31页。按DDPM五模块/证据工作流精读，不作为新的通用Infra知识篇，也不替代已有框架选型或异步综述。

## 固定来源与作者

| 资源 | SHA256 |
|---|---|
| PDF | `649edc87e70b8e57f9515360b15ed5576c6c8afb3c46a3763e83c22a85a660e2` |
| LaTeX tar | `7b675c884d64ac45537c5e0ed79d0a29d3506da9245c6a94e149b045a1feba27` |
| HTML | `3f3e50bc0bca24baf70af82204fab695ac9a3eaa1b9db33892d61b69a45beb91` |

[完整报告](../content/papers/infra/paper-dsec.md)保留131位作者。单位为DeepSeek-AI、Tsinghua University；Jialiang Huang为1/2，Mingxing Zhang为2，其余按原byline归属1。dagger是项目开发者标记，不当作单位编号；没有将作者原始缩写猜成完整姓名。

原文§7指明共享Rust OverlayBD组件：AgentENV `34cdc8098096726646853a18cec7ae143995dcef`，静态核读open_remote_blob_with_size、ImageFile read/write、FileCacheBackend do_preadv2_generic，正文永久链接对应具体文件。当前AgentENV主库不是DSec完整系统，未找全平台/libdsec公开实现，不引用同名CLI或第三方仿制冒充作者代码。

## 原图表账本

| 图表 | 主张 / 读法 | 源码 / PDF | 报告模块 |
|---|---|---|---|
| Figure1 | 管理/数据路径、Node Runtime、FnCall特殊路径 | figs/arch.pdf；PDF6 | 方法 |
| Figure4 | Toolkit更新与独立层；保持左右两面板顺序 | composable-layer-A/B.pdf；PDF11 | 方法 |
| Figure7 | 时间分钟log轴、CDF、p50/p99 | lifetime.pdf；PDF12 | 背景 |
| Figure9 | Compose/CPU QoS/memory/on-demand组合 | figs/dsec-arch.pdf；PDF14 | 方法 |
| Table2 | 活动环境资产，82.8+50.9=133.7TB | PDF10表体紧裁 | 实验 |
| Table3 | 各语言访问比例与镜像大小 | PDF13表体紧裁 | 实验 |
| Figure10 | 容器宿主VM活跃数；实线IOPS/虚线写总GB | docker_pull_comparison.pdf；PDF22 | 结果 |
| Figure11 | CPU%与disk MB/s；Tar/EROFS完成时间 | tar_cpu_diskio_ab.pdf；PDF22 | 结果 |
| Figure12 | peak vs time-integrated memory；CPU断轴 | uvm_mem.pdf；PDF23 | 结果 |
| Figure13 | BE负载%与chess agent time秒、无background参照 | qos_effect.pdf；PDF24 | 结果 |

提取计划：[paper-dsec.json](../scripts/figure-crops/paper-dsec.json)；[资产清单](../assets/papers/paper-dsec/figures.json)。方法/曲线来自固定LaTeX独立PDF，不执行TeX；原表裁掉邻接正文/页眉，表头、最后一行与边界逐张放大核查后标verified。未展示Figure2/3/5/6/8和Table1的用途/关键描述保留在正文，不宣称全部原图都贴出。

## 两轮审阅与关键口径

第一轮读正文/源码/实验，第二轮逐图对照PDF与数字、变量/单位/基线。修正了将未使用的3PB LaTeX宏当原文事实的风险：编译正文仅称PB级环境语料，本报告不引用未使用宏。Table2是一周活动工作集，不是全生产语料或每节点容量。

1. 生产scale unit160节点/30K cores/250TB与10节点CPU受控实验分开；380K concurrency、3M/day、5000/s不是§8实验负载。
2. §8只评估四类基础设施机制，明确不覆盖§6RL协同；未产生GPU训练吞吐、SWE任务质量或收敛收益。
3. hosts Linux7.0、guest Linux6.1按原文，不用常见版本覆盖；container实验QEMU512GB，microVM实验baremetal1.5TB，预算不相同。
4. Figure10约35min vs超过60min、作者1.71×slowdown；IOPS与总GB分开，57%采用作者统计，粗1600/700只给近似解释。
5. Figure11确定性预录工具调用替代LLM；79/45≈1.76；总写量5.5×、峰值带宽3.4×分开。高CPU可因更多任务更早运行，不等于setup更重。
6. Figure12 pmem峰值内存-40.2%，FPR时间积分-21.2%；不相加。CPU26.5→41.4是+14.9pp，相对约56.2%。图右横轴不均匀，不能按像素面积积分。
7. Figure13 baseline膨胀45.2%→idle+core17.3%是27.9pp；在同无负载参照下延迟相对降低约19.2%，不是27.9%。未把误差条/阴影补成未披露的统计定义。
8. FnCall没有Aether/Chronus；containers/FnCall外有QEMU VM；pack_diff磁盘状态、微VM内存snapshot、agent loop与trainer checkpoint各自分开。
9. 原文共享组件开源不等于全平台开源，当前组件分支/可配置块大小不等于生产实验实现；未运行仓库安装脚本、SDK、部署或kernel调参。

## 验收范围

新报告用v5/ddpm门槛，五模块长度1883/7101/2533/3112/1914，10张证据图表、2张实验原表。单篇draft阶段直接checkPaperQuality为零错误，人工核查后纳入发布检查，另完成同日Daily。静态源码核读与算术重算不标为平台/训练复现。

提交前检查：全库strict、topics质量、单元测试、build、真实浏览器；实际渲染桌面与手机原图比例、目录、131作者与长表布局，再推送并等线上工作流核验。检查记录支持结构/显示，不能替代论文事实阅读。

本地验收已完成：50/50全库strict、topics质量、29项单元测试、18项浏览器测试；构建40篇笔记/50篇论文/6篇研究报告。全库50篇×两种宽度共422次图表检查通过，十篇专题/下载/旧链接检查通过。另逐段看桌面/手机作者、方法图、长表和结果曲线；解码/重绘后确认Figure12实际显示正常。提交后仍以Pages部署及线上验收结果确认发布。
