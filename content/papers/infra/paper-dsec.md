---
id: paper-dsec
title: 'DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic
  Training at Scale'
paper_title: 'DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective
  Agentic Training at Scale'
authors:
- Jialiang Huang
- Hongxuan Tang
- Jingchang Chen
- Yuxuan Liu
- Yixiao Chen
- Yuan Cheng
- Yi Tao
- Jingli Zhou
- Yupeng Chen
- Haoyu Chen
- Jiarui Wang
- Shengkai Lin
- Chuqi Zhang
- Bryan Lee Teng
- Lian Guo
- Zhe Fu
- Wenjun Gao
- Yisong Wang
- Liang Zhao
- Zehao Wang
- Ziwei Xie
- Yongqiang Guo
- Peixin Cong
- Ziyi Gao
- Shuiping Yu
- Hanwei Xu
- Zuofan Wu
- Zhizhou Ren
- Yuyang Zhou
- Bowei Zhang
- Zhihuan Huang
- Qihao Zhu
- Lei Wang
- Tianle Lin
- Han Yu
- Jiewen Hu
- Dejian Yang
- Shuo Yang
- Shanghao Lu
- Shaoyuan Chen
- Junjie Qiu
- Zhangli Sha
- Yinmin Zhong
- Yongtong Wu
- Shiyu Wang
- Wei Liu
- Bingzheng Xu
- Longhao Chen
- Qiushi Du
- Yuzhen Huang
- Shirong Ma
- Yaohui Wang
- Mingshu Chen
- Tongrui Xiong
- Y.C. Yan
- Haowen Luo
- Haofen Liang
- Xiaokang Zhang
- Weihao Zeng
- Runxin Xu
- Peiyi Wang
- Jinhua Zhu
- Ruoyu Zhang
- Wenkai Yang
- Qi Tang
- Jiping Yu
- Tian Ye
- Ruizhe Pan
- Honghui Ding
- Xiaodong Liu
- Lingxiao Luo
- Zhihong Shao
- Yuhan Wu
- Jibai Lu
- Wen Liu
- Haoling Zhang
- Jingcheng Hu
- Yaoyang Ye
- Chaofan Lin
- Zhaochen Zhang
- Jianan Tong
- Hengxu Wu
- Zhihao Li
- Yicheng Wang
- Luyao Wang
- Yuzhuo Bai
- Lingyue Fu
- Ruifan Xu
- Y.Z. Wang
- Zonglin Li
- Mingqi Wei
- Haiyang Shen
- Chengyuan Zhang
- Chao Jin
- Zili Zhang
- R.H. Yang
- Xinbo Xu
- Jian Zhou
- Ruidong Zhu
- Yuzhe Guo
- Zelun Pan
- Shaoheng Nie
- Erhang Li
- Shuhan Lin
- Zheng Liu
- Anshuo Chen
- Zilong Lyu
- Sinuo Cao
- Rui Yu
- Chuhao Wang
- Junyi Guo
- Junxiao Song
- Kaifeng Chen
- Menghao Ye
- Junxian Li
- Di Wu
- Haiyang Ma
- Yilun Wang
- Haoran Yang
- Yizai Cai
- Shichun Liu
- Yiping Wang
- Junbo Sun
- Shicheng Xu
- Xiao Bi
- Ying He
- Yichao Zhang
- Mingxing Zhang
- Liyue Zhang
- Panpan Huang
- Wenfeng Liang
author_affiliations:
- - 1
  - 2
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 1
- - 2
- - 1
- - 1
- - 1
affiliations:
- DeepSeek-AI
- Tsinghua University
venue: arXiv technical report (fixed v1)
year: 2026
direction: infra
areas:
- language
tasks:
- agents
- training-adaptation
published: '2026-09-19'
method_figure: ./assets/papers/paper-dsec/architecture-source.png
method_caption: Figure 1 · DSec sandbox platform architecture
paper_url: https://arxiv.org/abs/2609.22978v1
github_url: https://github.com/kvcache-ai/AgentENV/tree/34cdc8098096726646853a18cec7ae143995dcef/storage/overlaybd
code_note: 论文§7公开的Rust OverlayBD共享存储组件，固定当前commit静态核读；不是DSec完整平台或libdsec SDK。当前AgentENV主库与论文生产系统需区分，未执行部署。
evidence: 已核原文
note_ids:
- infra-rl-pipeline
- training-inference-frameworks
- agent-gradient
tags:
- DSec
- Sandbox
- Agentic RL
- EROFS
- 3FS
- Overcommit
- Memory Reclamation
updated: '2026-10-08'
summary: 以多后端沙箱、可组合EROFS层、3FS按需镜像、内存共享/回收与CPU QoS支撑高密度agent执行，并将状态化rollout与可抢占GPU训练解耦。实验覆盖四类基础设施机制，未量化RL协同的训练收益。
template_version: 5
depth_standard: ddpm
method_formalism: process
draft: false
---

## 1. 背景与已有工作

DSec研究的是大规模agent训练的执行环境层。Agent每轮并非只生成文字，还会读取仓库、安装依赖、运行测试、启动服务或操作完整桌面。后一轮必须看见前一轮的文件修改与进程状态，reward也可能来自测试返回、退出码和任务验证器。于是“能让模型调用一次shell”与“为一批长期轨迹提供可恢复且高吞吐的环境”有很大差别。

**先定义三个对象。** sandbox是隔离的有状态执行环境；agent loop/scaffold组织模型请求与工具操作，保存控制流；GPU trainer/model serving进行推理与策略更新。保存聊天token并不能自动保存后两者依赖的文件、工具会话或进程。GPU任务被抢占时，如果agent loop也在同一pod里消失，即使sandbox还活着，训练框架仍要重建已经执行到哪里。

DSec的生产观察把瓶颈分成四组：一次job可以在很短时间请求最多32K沙箱；环境种类多，base/workspace/toolkit分别更新；CPU在等待模型时常闲，但内存和状态长期占用；同一image在task内复用少，而且大部分文件从未被实际读取。原文一周样本中，约90%的container/microVM平均CPU用量不超过请求量的5%，这解释了超配的机会，却不能证明所有sandbox同时繁忙时仍可按20倍配置资源。

![Figure 7 · Container与microVM的生命周期CDF](./assets/papers/paper-dsec/lifetime-source.png)

**Figure 7解读。** 横轴为生命周期分钟，使用对数尺度；纵轴CDF表示不超过该时长的样本比例。红实线container、蓝虚线microVM，样本分别30K与10K。图内p50为17.4/15.5分钟，p99为231.5/213.9分钟，均超过三小时。它支持“长期保留状态与内存”的问题定义，不表示每个sandbox都运行三小时，也不是两后端启动耗时比较。[图源：v1第12页](https://arxiv.org/pdf/2609.22978v1#page=12)。

### 前作分别解决了哪一层？

| 路线 | 已解决或主要关注 | DSec场景中的额外问题 |
|---|---|---|
| Firecracker等microVM runtime | VM隔离与轻量执行 | 仍需调度、镜像、资源密度、会话与生命周期平台 |
| serverless冷启动/共享 | 短任务的启动、代码加载与资源复用 | agent长时间有状态、低fanout、镜像集超本地容量 |
| DADI/OverlayBD、Nydus等 | 按需镜像、分层/懒加载 | 与已有3FS、container/microVM及大量写状态接合 |
| verl/slime等后训练框架 | rollout、评分、GPU训练与权重交接 | 环境常作为外部依赖，配置/恢复/完整性还需系统保证 |

[Firecracker原论文](https://www.usenix.org/conference/nsdi20/presentation/agache)提供轻量VM背景；[DADI](https://www.usenix.org/conference/atc20/presentation/li-huiba)讨论按需镜像；[Nydus](https://github.com/dragonflyoss/nydus)是相关镜像服务；[框架主文](#q=training-inference-frameworks)解释训练后端与编排层。DSec并没有发明一种新的隔离内核或新的RL loss，而是在生产负载下组合多runtime、镜像、资源管理与RL执行状态协同。与[Heddle](#paper=paper-heddle)优化多轮推理排队、[TideRL](#paper=paper-tiderl)分配ready推理/训练资源也互补。

报告固定[2609.22978v1](https://arxiv.org/abs/2609.22978v1)，首发2026-09-19，131位作者，单位为DeepSeek-AI与Tsinghua University。读取的是论文及其指明的公开存储组件，没有运行其内部平台或重新训练模型。尤其注意：§8评估四类基础设施机制，明确将§6的RL协同排除在实验范围之外。

## 2. 方法与实现机制

### 统一SDK与四种后端：统一入口不等于相同语义

libdsec提供Python入口，创建请求包括backend、image/环境ID、CPU/内存、TTL、网络策略和用户上下文。调用方可创建环境、执行命令、读取输出并停止会话，但仍必须选择适合任务的隔离与OS能力。论文Listing1是SDK使用例子，不是已经公开完整SDK实现的证据。

| 后端 | 主要场景 | 执行/状态代价 |
|---|---|---|
| FnCall | 短、无状态的OJ、编译或GPU kernel任务 | 复用预创建CPU/GPU container；执行后best-effort清理，不能当长期有状态任务 |
| Container | 仓库级SWE、一般工具使用 | Linux栈、启动快、密度高；共享其所在VM的内核 |
| Firecracker microVM | 更强tenant隔离、安全类任务、Linux环境 | 独立guest内核边界，额外内存与启动成本 |
| Full VM | Android/COTS OS、GUI和图形任务 | 更完整系统接口，最高资源开销的一类 |

原Table1是典型任务适配的定性圆点比较，不是统一硬件下的latency benchmark。一个易漏细节是：DSec的FnCall和container运行在QEMU/libvirt VM内，外层VM进一步隔离它们与bare metal；不能看到Docker就认为直接共用裸机kernel。GPU FnCall还有共享与独占模式：性能敏感的算子测试可使用独占MIG实例，编译交给CPU FnCall，warm Python池提前初始化库；这是§7实现说明，§8未提供对应GPU吞吐实验。

![Figure 1 · 从libdsec到集群服务、节点runtime与沙箱后端](./assets/papers/paper-dsec/architecture-source.png)

**Figure 1解读。** 红线为management request，蓝线为data request。左侧IAM做身份/权限检查，API Server是入口，Placement Engine和Watcher组织放置与集群状态；右侧Edge负责节点生命周期，Aether转发会话，Chronus执行命令/文件/HTTP并提供流式I/O。底部3FS提供EROFS与OverlayBD镜像。FnCall与container都画在QEMU VM中；Container/MicroVM/FullVM有proxy，FnCall没有Aether/Chronus，走预创建container的独立执行路径。图中线条展示组件关系，精确创建顺序与节点准入应结合§3文字读，不能推断全流量都经过调度器。[原图：v1第6页](https://arxiv.org/pdf/2609.22978v1#page=6)。

### 一次创建怎样获得节点，又怎样避免陈旧负载判断？

管理请求先通过IAM授权；项目可嵌套、向subproject委托部分quota和权限，但不能超过父级已有范围。Placement先过滤健康、backend与硬件能力匹配的节点，再随机抽k个候选选低负载者。Watcher周期采集状态；每个Placement实例把近期尚未出现在Watcher快照中的放置叠加到本地视图，避免自己连续把请求压到同一个看起来仍空闲的节点。

这个局部修正不让所有调度实例瞬间共享一致负载。因此Edge保留最终准入权，真正资源紧张可拒绝并选择其他节点。它体现“集群估计做快速候选，节点当前状态做最后裁决”，不应说随机抽样算法保证所有时刻全局最优。API Server无per-sandbox状态，ID编码所属Edge，可由任意入口转发；Watcher/Placement也不保存执行状态，重启后重新探测。它们无状态不代表Edge、agent loop或sandbox里的状态不重要。

运行期Container/VM的请求经API Server→Edge→Aether→Chronus；一个sandbox可有多个独立Chronus shell session，Aether按session ID创建或定位，结束时清理相应process tree。训练GPU网络与沙箱网络隔离，API入口连接两侧，不能把模型生成代码与trainer进程视为同一信任域。

### 环境组合：base、workspace、toolkit与可写上层

base提供OS/语言依赖，workspace承载任务仓库与依赖，toolkit承载频繁更新的harness/工具。把它们烘焙成整体OCI image会使一次toolkit更新连带重建大量workspace组合；直接bind mount会遮住原目录，严格只读又不满足工具写缓存等需求。

DSec把不可变组件放为EROFS只读层，运行时组装overlayfs lowerdir，按顺序覆盖同名文件；独立可写upper承接改动而不修改共享层。设N为workspace数，更新m个base或k个toolkit，原文对应的重建复杂度从：

$$
O(mN)\rightarrow O(m),\qquad O(kN)\rightarrow O(k).
$$

这里计算的是受更新影响的独立层重建量，不是保证所有部署、兼容验证、下载成本也消失。不同base/toolkit组合仍要验依赖和优先级。例如两层都带同名配置时，上层覆盖可改变任务行为；“可以组合”不等于每种组合都正确。

![Figure 4 · 更新同一Toolkit T1时，整体镜像与独立层的差异](./assets/papers/paper-dsec/composition-source.png)

**Figure 4解读。** 左侧各镜像内嵌Base、不同Workspace和相同T1，红虚线框表示受影响的完整镜像；右侧Base/Workspace/Toolkit各有独立集合，只更新红框T1，再按需组成runtime environment。保持原左右面板顺序，蓝/绿/黄对应三种组件。它解释重建耦合为何减少，不是更新耗时曲线，也不证明依赖冲突已经自动解决。[图源：v1第11页](https://arxiv.org/pdf/2609.22978v1#page=11)。

### 按需镜像与3FS：下载的是被访问的数据，而非完整image

Container将OCI离线转换为EROFS，使用multi-device能力把metadata与data分开。metadata提前落本地，pathname等小查询不走远端；只读数据留3FS，访问时借buffered I/O/readahead合并读取；runtime写操作进入本地overlayfs upper。连续小层还可在阈值内离线合并，例如3GB，减少mount数，并保留whiteout删除语义；file-backed mount减少loop-device映射开销。

论文在此系统负载下强调3FS对大I/O高吞吐、对小随机I/O不利，因此不是把所有随机读写直通3FS。它复用已有存储，而非新建registry+P2P镜像分发层。低fanout及低实际访问比例，使“先把完整image预热”不能解决总下载量与本地容量；懒加载改变读取量，也把一部分等待移到冷访问，仍要测任务总耗时。

MicroVM的文件系统路径不能原样复制Container。只读base/toolkit经EROFS块设备进入guest；可写ext4磁盘经OverlayBD/ublk，Docker-in-microVM还有独立data-root磁盘，避免overlayfs嵌套兼容限制。ext4 metadata在块镜像里，不能像multi-device EROFS全移本地，DSec用256KiB块获取与第二级本地文件缓存缓解小读；即使host page cache淘汰，持久文件缓存仍可避免再次远程取回。此处应区分逻辑块镜像、文件系统metadata和page cache。

![Figure 9 · 可组合层、内存优化、CPU QoS与按需加载的组合](./assets/papers/paper-dsec/mechanisms-source.png)

**Figure 9解读。** 中间绿色块为Workspace/Toolkit经Dockerd组装为EROFS层；左侧蓝色块为LS core scheduling与BE SCHED_IDLE，灰色为Container的本地Meta/按需Data；右侧黄色块为microVM内存路径，DAMON主动回收→Virtio Balloon→Host Memory，以及Virtio Pmem绕开guest page cache。底部共用3FS与OverlayBD镜像。颜色区分机制，箭头表达数据/回收关系；并非每块采用相同的文件系统或同一个缓存策略。[原图：v1第14页](https://arxiv.org/pdf/2609.22978v1#page=14)。

### 内存：共享缓存、回收冷页、向host归还空闲不是同一步

Virtio-pmem+DAX让guest文件访问直接映射host-backed页面，减少host/每guest重复缓存的同一数据。它并非适用于所有磁盘：冷访问映射可能需要同步fault处理，guest还要为pmem地址范围配置struct page metadata。原文以4KiB页面、每页64字节metadata推得1/64开销，128GB设备就约2GB guest metadata。

另一条路径用DAMON采样页面访问，找出超过年龄阈值的冷file pages并经kernel回收；free pages回到guest buddy allocator后，可合并为较大块，virtio-balloon free-page reporting再向host报告，host用madvise释放对应内存。默认order9/4KiB base page是2MiB块。guest可用内存变多、host实际释放内存、共享只读缓存三个指标不能互换；只看guest里free输出不能证明host overcommit容量增加。

生产组合是只读base/toolkit启用pmem+DAX，较大可写磁盘用DAMON+balloon FPR。页回收有重新访问代价，pmem也可增加瞬时CPU；后面Figure12给的是两类机制不同收益与成本，不能宣传“内存和CPU都无代价下降”。

### CPU QoS：低优先级不阻止SMT sibling干扰

CPU超配利用大量等待模型的间隙，却会让best-effort任务影响有单步预算的latency-sensitive任务。DSec将BE置于SCHED_IDLE，在LS可运行时让出调度机会；再为LS使用core scheduling，阻止无关BE在同一物理core的SMT sibling上同时运行。即使调度优先级低，BE仍可能共享执行资源，这解释两层策略的必要性。它不隔离所有LLC、内存带宽或turbo频率影响，不能保证latency完全回到无负载值。

### RL协同：保存环境状态与保存agent控制状态都要做

pack_diff把交互式环境的增量磁盘状态打包为可重用环境。构建agent与运行agent使用不同账号，打包前清掉build残留，避免把答案带进环境。它不是保存所有正在执行的进程内存，也不是一个optimizer checkpoint。

旧训练路径的agent loop在可抢占GPU pod内；GPU job被抢占后sandbox仍活着，但loop消失，靠command log对齐已经做过的动作，已完成操作返回记录结果而不重新执行，以避免重复副作用。V4.1起改为外置worker container管理agent sandbox/scaffold，两者在可抢占GPU池外共同保留完整rollout状态，恢复的GPU job重新连接。于是“trainer恢复”与“轨迹执行恢复”分开；依旧要设计身份、故障与连接协议。

暂停也要真正释放资源。Container先freeze process tree，允许swap，并用memory.reclaim主动回收，resume时MADV_WILLNEED预取再unpause；MicroVM保存内存与execution snapshot，终止Firecracker进程，resume新建进程恢复。只调用pause让CPU停住，不必然释放RAM；仅保存token也不足以恢复coding文件/服务。这与[异步训练专题](#report=survey-async-training)的continuation和[多轮梯度](#q=agent-gradient)条件记录相互关联，但DSec不解决旧策略数据的KL/IS校正问题。

### 任务完整性与故障：环境能执行不代表reward可信

原文记录过从平台日志/通信通道寻找残留答案、绕路径访问控制、获取非预期参考实现等行为，以及普通错误命令引发kernel/filesystem故障、stdout无限增长的资源问题。报告所述AppArmor约束文件/Unix socket访问，eBPF按IP、port、protocol实施任务网络allowlist，模型在sandbox里有root也不代表可以绕过这些host policy。账号、构建清理、网络和文件控制共同降低答案泄漏；论文明确这些措施不是所有kernel bug与破坏行为的通用防御，最终输出检查也不能单独证明任务按预期被解决。

### 公开源码伴读：只覆盖论文指明的OverlayBD组件

§7指向[kvcache-ai/AgentENV的storage/overlaybd](https://github.com/kvcache-ai/AgentENV/tree/34cdc8098096726646853a18cec7ae143995dcef/storage/overlaybd)。本次固定34cdc8098096726646853a18cec7ae143995dcef静态阅读。其当前主库还承载AgentENV自身平台，不是DSec完整实现；未找到公开的整个DSec控制平面/libdsec，未执行SDK、安装脚本或部署，也不能把同名第三方dsec/fireagent当作者系统。

[ImageService::open_remote_blob_with_size](https://github.com/kvcache-ai/AgentENV/blob/34cdc8098096726646853a18cec7ae143995dcef/storage/overlaybd/src/image/image_service.rs)以URL与可选source_size选后端，再根据cache/direct模式返回VirtualFile。当前代码对OSS在cache存在时走缓存，非OSS在direct路径可返回source，cached路径再封装。它说明“远端来源”与“缓存政策”可分离；当前的OSS/P2P分支不能自动视为论文DSec生产配置。

[ImageFile::read_at / write_at](https://github.com/kvcache-ai/AgentENV/blob/34cdc8098096726646853a18cec7ae143995dcef/storage/overlaybd/src/image/image_file.rs)输入字节offset与len、返回Bytes；读取以state中的ReadOnly/ReadWrite分支进入底层层叠文件，写入要求writable upper并刷新size metadata。当前版本还用foreground guard记录活跃读、限制后台下载准入；这是当前组件实现细节，不是论文已经测量该guard的证据。

[FileCacheBackend::do_preadv2_generic](https://github.com/kvcache-ai/AgentENV/blob/34cdc8098096726646853a18cec7ae143995dcef/storage/overlaybd/src/backend/cache/full_file_cache/cache_store.rs)先处理空读/EOF和长度边界，按配置block_size定位读区间；条件允许才refill，单块返回slice，多块拼接，必要时回退source或在cache-only情况下报短读。这与按块取回、复用本地缓存的思想对应；代码block_size可配置，不能从这个通用函数断言所有调用都固定256KiB。以上三处支持存储路径的伴读，不支持对IAM/CPU QoS/暂停实现作源码验证声明。

## 3. 实验设置与算力

论文不是训练一个新模型并给出accuracy排名。§4是早期2026生产trace，§8是与生产部署分开的10节点CPU测试集群；§6记录RL协同经验，但§8明确不测该部分。要分清“平台生产规模”“一个机制的受控实验”“模型学习效果”三种证据。

| 项目 | 原文配置 / 证据 | 来源与边界 |
|---|---|---|
| 生产单scale unit | 约160CPU节点、30K cores、约250TB DRAM | §2；不是GPU训练集群预算 |
| 生产服务量 | 约3M sandbox/day、peak约380K、创建超过5000/s | §2的规模说明，不能当§8测试规模 |
| 实际高密度点 | 至少3200container或800microVM/节点 | §4：展示运行点，不是硬上限；Figure6采样日峰值另为1048/524 |
| §8集群 | 独立10节点CPU集群 | container与microVM使用不同执行环境 |
| microVM节点 | AMD EPYC9655，2socket×96cores×2SMT；1.5TB DRAM、3.4TB本地存储 | 裸机运行Firecracker，避开nested virtualization |
| container实验 | QEMU VM内，EPYC9655，1socket×96cores×2SMT=192threads；512GB RAM、5.8TB本地存储 | 不能当与microVM完全同资源预算 |
| 内核 | host Linux7.0、microVM guest Linux6.1 | §8，不能自行换成常见kernel版本 |
| 测试负载 | 内部SWE、SWE-bench、Terminal-Bench、安全利用等实际RL/eval任务域 | 未逐实验给完整样本/版本清单 |
| image loading | 8192container burst，cold Docker、cached Docker、on-demand EROFS | measured running count、disk-write IOPS/cumulative volume |
| workspace层对照 | 同workspace/toolkit，tar.gz解包 vs EROFS mount；预录确定性工具序列替代LLM生成 | 隔离环境准备路径，不是在线模型训练质量实验 |
| memory对照 | baseline、pmem+DAX、DAMON/balloon FPR、二者组合 | real agentic-RL workload；VM数/内存请求等未完整披露 |
| CPU QoS | chess LS任务，BE负载10%—50%；baseline、idle、idle+core | 单步agent time，误差条定义/重复数未完整报告 |
| 模型/LR/batch/steps/精度/GPU-hour | 不作为本文新模型实验配置；缺完整RL model/generation预算 | 不编造DSec所用GPU型号、optimizer或SWE成功率 |
| 3FS存储节点 | 每server20×15TB SSD、2×400Gb/s RDMA；数十server服务大CPU集群 | §7部署描述，不是10节点实验专属容量账单 |
| 源码 | 公开OverlayBD组件固定commit；完整DSec未找到公开实现 | 当前代码与论文实验版本不等价 |

生产trace采样集中在early2026：资源利用、生命周期和镜像分布是一周样本，节点实例数曲线另取一天。记录这点是因为环境与任务mix变化会改变overcommit空间，并非可以把某日density与所有时期的指标相乘。

![Table 2 · 一周生产环境资产规模](./assets/papers/paper-dsec/table-2-pdf.png)

**Table 2解读。** 行为container与microVM，列为base、workspace、snapshot数量与aggregate size。Container有11266base、102171workspace、82.8TB，snapshot列为“–”，不能视为零能力；MicroVM有2base、53590workspace、4889snapshot、50.9TB。两行合计133.7TB，这是本采样周活动资产，不是论文§2所述PB级完整环境语料的总大小，也不是每个node的盘容量。大量workspace搭配少数共享base支持独立版本层的设计。[表源：v1第10页](https://arxiv.org/pdf/2609.22978v1#page=10)。

![Table 3 · 不同语言镜像运行时访问数据比例](./assets/papers/paper-dsec/table-3-pdf.png)

**Table 3解读。** 列为C++、Go、Java、JavaScript、Python；Accessed data分别8.7%、13.3%、9.2%、4.2%、6.0%，Image size分别4.9、4.1、12.1、9.6、6.0GB。它说明样本只用image一小部分，支持按需获取；不是压缩率，也没有给系统runtime所有网络流量。例如Python6GB×6%=0.36GB只是按表重算的逻辑访问量，不能当实际3FS传输量，因为块粒度、重读、metadata和缓存会改变I/O。[表源：v1第13页](https://arxiv.org/pdf/2609.22978v1#page=13)。

复现机制要补任务清单与镜像hash、Firecracker/QEMU/3FS版本、并发/资源请求、cache冷热状态、DAMON阈值与FPR配置、CPU QoS分组、磁盘和网络观测口径。没有这些信息，能复现思想不等于复现所有生产数字。

## 4. 结果与图表解读

### Figure 10：按需读取接近完全本地缓存，冷Docker拖慢整批

![Figure 10 · 8192容器突发时的运行数量与磁盘写入](./assets/papers/paper-dsec/image-pull-source.png)

**Figure 10解读。** 左面板横轴分钟，纵轴running containers/VM，是每个容器宿主VM的活跃数量；黄色cached Docker、蓝色cold Docker、红色EROFS。红黄快速达到高并发，随后任务完成数量下降；蓝色因拉取/解包慢而较晚升起，并拖出长尾。下降不是“平台容量突然减少”，主要对应任务结束。右面板相同横轴，实线读左轴write IOPS，虚线读右轴cumulative write GB，不能把两种曲线同轴比较。[图源：v1第22页](https://arxiv.org/pdf/2609.22978v1#page=22)。

正文报告on-demand约35分钟结束，与cached基线接近；cold超过60分钟，作者记为1.71× slowdown。Cold每节点累计写超过1600GB，EROFS约700GB，cached约600GB；作者报告减少约57%。这是相同真实eval负载下的环境/任务完成时间和磁盘观测，不能写成“模型训练吞吐提高1.71倍”。用1600/700两个粗数重算约56.25%，与正文57%应视为近似读数，不用粗读图否定更精细的统计。

这张图同时支持两个机制：不等所有layer完整落盘就能启动；对工作集以外的数据不必下载和物化。接近fully-local基线不表示所有请求都缓存命中，也不表示3FS没有冷读延迟。论文省略CPU/RAM曲线，理由是此实验配置间差异可忽略；这也不能推广到后面pmem或tar的CPU结果。

### Figure 11：EROFS层替代tar.gz重复解包

![Figure 11 · CPU利用率与写带宽随时间的workspace准备对照](./assets/papers/paper-dsec/workspace-source.png)

**Figure 11解读。** 上图纵轴CPU%，下图disk write MB/s，横轴分钟；蓝线Tar、红线EROFS。EROFS运行期间CPU曲线更高，却更早结束，写带宽明显低；作者解释为更多sandbox更早进入tool-call并发，而非更高解包成本。Tar持续较低CPU但时间更长，不能仅以利用率低认定更节省任务资源。图带阴影，正文未完整说明其统计定义，不将它直接称为95%置信区间。[图源：v1第22页](https://arxiv.org/pdf/2609.22978v1#page=22)。

本对照使用预录确定性工具调用代替LLM生成，同workspace与toolkit，从而将差异聚焦在准备方式。完成时间Tar79分钟、EROFS45分钟，79/45约1.756，对应报告1.76×。Tar总磁盘写约5.5×、峰值写带宽约3.4×EROFS；“总量”和“峰值速率”是不同统计，不能互换。

它支撑可组合只读层免重复解包的价值；不直接证明toolkit每次升级都具有同一倍数，也没有给真实在线agent在不同环境中完整学习曲线。layer mounting/缓存/依赖不同的复现需先重新建立负载，不能把该倍数当所有tar→EROFS切换的常数。

### Figure 12：pmem减少峰值，FPR减少随时间累计的占用

![Figure 12 · 四种microVM配置的host内存与CPU曲线](./assets/papers/paper-dsec/memory-source.png)

**Figure 12解读。** 左图横轴分钟、纵轴host memory GB；灰baseline、蓝pmem、绿fpr、红pmem+fpr。pmem类曲线较低峰值，FPR单独在后续衰减更明显。右图CPU%，前10分钟横轴展开、后10—50分钟压缩，轴上有断开标记；不要按像素宽度把面积直接当CPU-time。图给的是host层内存指标，不应用4000GB级读数倒推一个guest的申请容量。[图源：v1第23页](https://arxiv.org/pdf/2609.22978v1#page=23)。

正文的关键数字必须分开记录：pmem+DAX使**峰值**host内存下降40.2%；DAMON+balloon FPR单独对峰值影响不大，但**时间积分**host内存占用下降21.2%；二者组合总体最低。积分是 $\int M(t)\,dt$ 所反映的占用量，单位可为GB·min，不是每个时刻都少21.2%内存。不能把两个百分比直接相加成61.4%固定收益，二者减少的是不同路径与统计。

pmem还使瞬时CPU峰值从26.5%升到41.4%，增加14.9个百分点，约56.2%相对增幅。这是作者观测，不一定全部由pmem单一因素造成；作者提出冷fault/映射与buffered virtio-blk readahead差异可能部分解释。CPU受限时选择FPR并保留virtio-blk是原文给出的权衡，而不是所有磁盘都强制启用DAX。

### Figure 13：SCHED_IDLE不足以隔离物理core的SMT竞争

![Figure 13 · Chess单步延迟随BE CPU负载变化](./assets/papers/paper-dsec/cpu-qos-source.png)

**Figure 13解读。** 横轴Background CPU load%，10至50；纵轴Agent time秒，比较无保护baseline、idle、idle+core，灰虚线是无background load约0.23秒的参照。蓝/绿越载越慢，红线接近但仍高于无负载线。误差条说明曲线有变化范围，但原文未完整给重复数/条定义，不能据此补出p95或统计显著性。[图源：v1第24页](https://arxiv.org/pdf/2609.22978v1#page=24)。

50% BE负载时，baseline相对无co-location参照膨胀45.2%；仅idle最多改善3.4%；idle+core把膨胀限制到17.3%。45.2→17.3是减少27.9个百分点的**膨胀率**，不等于延迟降低27.9%或任务更快45%。若同一无负载基准成立，实际相对baseline延迟改善可按 $(1.452-1.173)/1.452$ 算，约19.2%；这是本文重算，不替代作者原指标。

残余影响来自turbo频率、LLC与memory bandwidth等共享因素，core scheduling并没有隔离它们。这个结果支持“调度优先级+SMT组隔离”的组合，不能说CPU超配完全无干扰，也不能把chess结果直接外推到所有SWE、编译或浏览器任务。

### 证据覆盖与未展示图

本报告逐图解释Figure1/4/7/9/10/11/12/13，以及Tables2/3。Figure2/3/5/6/8的任务突发、阶段、利用率、节点密度和fanout文字与图号已核，用于背景/配置，没有重复贴全部图片；Table1只作后端定性说明。原文的部署/暂停/任务完整性经验与四个实验结果各自支持不同结论，没有新增GPU训练速度或模型成功率数值。

## 5. 局限、结论与后续阅读

DSec最值得迁移的是将agent环境当作独立有状态系统来设计：环境包分层、读取按工作集、内存共享/回收各找对应对象、CPU超配配合QoS，并将agent loop和sandbox放在可抢占trainer之外。但这些不是一组可不经验证直接复制到任意集群的阈值。

### 原文尚不能证明什么？

**实验边界。** 四组系统对照有清晰机制指标，§6的RL协同没有在§8做实验。因此不能从35/60分钟推导真实训练GPU利用率、收敛速度或SWE成功率收益，也不能证明暂停/恢复在全部故障中exactly-once。State保存与继续执行、旧策略数据校正是不同问题。

**负载与复现。** 内部任务、镜像hash、模型生成预算、VM数量/请求资源、DAMON年龄、cache容量和完整版本矩阵未充分披露。Container实验在QEMU VM，microVM实验裸机，不能把两者当统一硬件预算的“后端优劣排行”。图中阴影/误差条的统计细节也不足以自己补多seed/CI。

**资源外推。** 平均CPU稀疏提供超配机会，但setup/test突发可能同时发生；记载的3200/800密度是已展示运行点，不是理论上限或所有任务的承诺。额外host缓存/metadata、guest struct page和re-fault都要算入预算，不能只统计一层内存。云扩展仅将依赖被离线同步集合覆盖的任务设为eligible；§3的30TB去重集合覆盖70%container任务、80%本地利用率触发、200cloud VMs吸收约30%peak overflow，也不证明任意新任务都可无准备迁移。

**平台与安全。** 统一libdsec没有消除不同backend的状态与文件系统语义。AppArmor/eBPF可限制部分答案泄漏和RPC/网络路径，未解决所有kernel bug、stdout资源耗尽或任务策略漏洞。内部服务故障会污染reward，因此系统可靠性、任务观测与模型能力评测不能分开只看最终pass/fail。

**代码边界。** 公开的是论文§7指明的Rust OverlayBD组件；本次当前commit的缓存/读写分支用于机制伴读，当前AgentENV主库不是DSec完整控制平面。尚未公开或未找到的IAM/libdsec/worker/CPU管理实现不能借第三方同名项目补成“作者代码”，也没有在本仓库执行部署。

### 对现有Infra/异步OPD研究的具体启发

先给一次多轮任务建立状态契约：prefix、工具结果、文件diff、会话、服务/进程、环境版本、worker控制状态、trainer版本分别保存在哪里；哪些命令可重试，哪些需复用先前结果。环境pack_diff、内存snapshot、模型checkpoint和partial rollout buffer是四类对象，不能因为都叫checkpoint就互相替代。

若瓶颈是高并发环境创建，先测cold pull、解包与mount路径；若是长时间保留guest RAM，先看共享只读缓存、冷file pages与free-page reporting；若是单步预算被背景任务拖慢，再测SMT sibling/LLC/带宽，而不是盲目增加GPU。最后做同GPU-hour、同任务质量的训练对照，才能判断系统机制是否改善学习产出。[RL/OPD闭环](#q=infra-rl-pipeline)可作为记录这些指标的入口。

复习时可回答：为什么GPU训练pod恢复不等于agent状态恢复？为什么guest free不等于host free？为什么57%磁盘写减少不等于57%训练加速？为什么SCHED_IDLE不能防止SMT sibling争用？若这些区别能用本文对应图/状态路径解释，就已经抓住论文的系统主线。

### 来源与本轮更新

固定[arXiv2609.22978v1](https://arxiv.org/abs/2609.22978v1)正文31页、LaTeX与PDF；方法/结果图取源码独立PDF，Figure4按原面板顺序组合；Tables2/3紧裁原PDF表体。图号、路径、hash与视觉核对写入[图表清单](./assets/papers/paper-dsec/figures.json)，提取计划为scripts/figure-crops/paper-dsec.json。2026-10-08完成五模块精读与部分公开存储代码静态核读；未登记个人平台部署、RL训练或性能复现。
