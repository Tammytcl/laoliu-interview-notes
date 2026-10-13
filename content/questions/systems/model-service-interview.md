---
id: model-service-interview
title: "模型服务化面试：性能、路由、观测、灰度与安全闭环"
category: systems
difficulty: 进阶
tags: ["P1", "Serving", "Deployment", "Checklist", "Interview", "Observability"]
updated: "2026-10-13"
summary: "按请求链合并部署、动态adapter、性能压测、负载、日志、灰度、熔断与应用风险题；既有推理/并行原理复用，附16道场景问答。"
draft: false
---

## 1. 问题背景 从训练完成到可靠服务还缺什么

Checkpoint保存不意味着用户拿到同一个模型，API返回200也不意味着答得正确。服务需要正确导出/加载、版本与协议、路由和容量、真实请求指标、回滚与故障处理。面试的主线应是一条用户请求，而不是先列Nginx、vLLM、Prometheus、Grafana。

本篇补[清单](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)153–174、194–198、204–222的上线知识，已有[推理引擎](#q=infra-inference-engine)、[分布式](#q=infra-distributed-parallelism)、[GPU性能](#q=infra-gpu-performance)继续讲内部机制。本篇不把外部作者医疗场景或具体Qwen部署配置当你的项目经历。

## 2. 前置知识 Runtime、Artifact与业务成功

Artifact是权重/tokenizer/template/adapter/config等版本集合；runtime是加载、请求调度和执行环境；业务成功是任务正确、可靠、延迟与成本满足约束。单独tokens/s不够，质量下降、请求timeout或输出变长都可能“看起来吞吐高”。

Replica是独立请求副本，TP是一个模型的矩阵/词表协同rank；更多卡既可扩replica也可扩TP，取舍看模型是否放得下、通信与请求规模。KV、prefix cache、answer cache和用户session又是不同状态，不共用一套模糊的cache key。

## 3. 请求链与API契约

客户端→鉴权/限流→路由→可选检索→模型队列→prefill/decode→输出校验/验证→流式结束→事件/metrics。同步适合简单完成响应，异步job适合长任务和恢复，streaming提前显示片段；HTTP、gRPC、WebSocket/SSE按调用生态、双向需求、取消和可靠性选择，不能只说哪个“更快”。

请求至少有request ID、tenant/权限范围、模型/adapter revision、messages/schema、生成上限与deadline；响应记录resolved model version、finish_reason、usage与错误类别。固定alias映到哪个artifact需要可追溯，不能日志只写“latest”。

幂等与重试按动作语义定义。纯生成重复可能只是多算钱，工具写文件、购买、更新数据库则可能重复副作用。Client timeout不证明backend停止；要请求取消确认、进度查询或可审计未知状态，不把重发全部当无害操作。

示意契约（教学，不绑定某厂商API）：

```json
{
  "request_id": "stable-id",
  "model_revision": "immutable-revision",
  "adapter_revision": "optional-revision",
  "knowledge_revision": "optional-index-version",
  "deadline_ms": 8000,
  "max_output_tokens": 1024
}
```

具体兼容参数回对应产品固定版本文档核验，教学JSON不是当前vLLM/SGLang完整请求schema。模型服务地址和真实凭据不属于公开面试笔记内容。

## 4. Artifact发布：DCP、HF与模板一致

训练参数分片、optimizer checkpoint、serving TP布局不是同一种格式。导出时核完整键/shape/dtype、模型config/tokenizer与revision，必要数值/输出对照和真实canary；checkpoint元数据存在不证明所有rank shard完整。

Model本体、LoRA/量化、generation/stop config、chat template、推理kernel版本一起登记。你的turn项目独立checkpoint导出与评测流程应固定迭代selector；不能复用一个旧转换成功文件，就把它当当前request的模型。

CI可包括metadata/格式、单样本加载、输出协议、任务质量、行为边界与性能回归；自动测试通过后再按既定发布流程切流。Static LoRA merge和dynamic adapter有不同cache与资源契约；热切换能力依backend版本，不宣称所有rank/量化组合都无需重启。

Knowledge index与模型可独立更新，但引用、权限和质量也会改变。需要版本bundle或明确兼容矩阵，回滚时不只回weights而留下新template/index，使“恢复旧版本”实际成为从没评测过的组合。

## 5. 性能压测与容量

TTFT从定义的请求起点到首个输出token，含网络/路由/queue与prefill多少需明确；TPOT/ITL量后续生成间隔，整请求latency另报。Streaming可改善感知首响应，但不自动减少GPU总计算或完整结果时间。

Tokens/s说明吞吐，tasks/s与goodput说明有效任务在SLO内完成；input/output长度、cached tokens、batch、concurrency和错误率都应一起列。Prefill与decode运行特征不同，FlashAttention与PagedAttention分别优化计算IO和KV管理，详见原Infra文章。

P50/p95/p99按单请求样本/窗口统计，不用每node平均p99再平均假装全系统p99。包含timeout和失败样本的规则预先固定；只保成功低延迟请求会虚报。稳态、启动、warmup、最后drain分开，不把所有窗口混进一个均值。

负载测试固定或分桶真实prompt/output分布与到达过程，测queue length、pending、batch、KV used/capacity、GPU activity/显存、CPU/network与cancel率。闭环压测客户端等上一请求后再发，可能低估开放到达过程的排队；并发数增加到服务率饱和后通常只放大尾延迟。

Page/block、context cap、显存utilization与max-running等配置影响容量、fragmentation和prefill/缓存；不是越大越好。只能根据对应backend版本与实测选，不把另一SKU上的最优值带入。模型纯权重下界不等服务最低显存，KV、激活和临时工作区也算。

## 6. 负载均衡、路由与分布式Serving

多replica可round-robin、least-load、cache affinity等；连接数不完全代表模型token工作量，长请求和tool等待也可能失真。一致性hash提高prefix/session缓存亲和，但hot key与长轨迹会造成负载不均，需要可控steal/重路由和状态契约。

一条TP请求需要其group各rank协同；不能每rank独立收到不同prompt而当更多副本。Replica健康包括进程存活、模型ready、资源可用、实际推理canary；liveness与readiness分开，启动下载/加载阶段不直接接流。

反向代理维护deadline、流式不缓冲过久、连接/取消与错误传播。Nginx开源/商业或其他代理的具体主动health-check能力依产品版本，不以一个示意图宣称全部配置都支持。

动态路由依据请求复杂度、证据/工具需求、风险、模型能力与资源走fast/full path。路由模型本身可能错，记录decision与质量/latency，并有fallback。高风险请求所需验证不能因拥塞任意关闭；优先级也需防普通请求无限饥饿。

## 7. 可观察性、日志与告警

Metrics用于聚合，trace用于请求路径，logs用于事件内容；request ID跨检索/模型/工具/judge串起来。记录resolved版本、长度、阶段时间、cache状态、queue、finish/error、重试与验证结果，敏感原始输入是否保存按用途与授权控制。

Prometheus/Grafana是采集/呈现工具，不能自动产生可靠业务指标。Export常聚合counter/histogram、label避免request ID这种高基数；trace可存细ID，metrics按可控tenant/模型/阶段分类。告警阈值结合SLO、风险和持续窗口，P0/P1/P2不能所有团队照同一常数。

区分设备activity与MFU、allocated memory与used KV、reserved GPU-hour与仅活跃卡时。训练与tool工作重叠时阶段时间相加不等wall-clock；观测缺口不填零。权重commit、publish与checkpoint completion分别发事件。

输出审计可存decision/证据出处/模型版本/必要脱敏片段与hash，不能为了debug默认全量跨用户保存私密prompt。Hash不是彻底匿名化，也不替访问控制或删除策略。

## 8. 灰度、回滚、熔断与降级

Canary先固定小流量、观察质量/错误/SLO，再分阶段扩大；assignment和user/session粘性需定义，别每turn随机换模型导致交互混版本。Blue-green准备两环境后切换，回滚较清晰但资源重复；canary渐进观察但多版本与统计复杂，各有条件。

回滚触发和观察窗口预先定义，量质量、版本、错误与成本，避开事后挑有利样本。Weights、template、adapter、index/routing规则可能需要协调回滚；滚回模型不自动撤销已执行工具副作用。

Circuit breaker按依赖错误/timeout/容量触发open、试探half-open和恢复策略，避免请求风暴；重试有budget、backoff和幂等条件。降级可选择安全的可验证静态信息、减少非必要增强、明确不可用或人工接入，但不能输出假成功替缺模型。

上线反馈回到候选样本、标注/去重/权限、版本化数据、离线回归与受控训练发布。用户低评分不等无误偏好标签，judge高分不等上线安全。模型服务成功要端到端证明，非一个deployment状态。

## 9. 清单中的医疗/安全题怎样转成系统题

本篇处理风险路由、隐私、引用与人工升级，不给病情判断或处方方案。原清单“医疗安全三个原则”不是统一算法定理，按实际用途、专家审定需求和监管边界定义。参考[WHO健康领域LMM治理指南](https://www.who.int/publications/i/item/9789240084759)的责任、透明与风险治理方向，具体医疗知识交由可验证来源和领域专业人员。

识别敏感信息不只是正则删名字：上下文中的身份、记录、关联关系、query cache和可重识别风险都要控制。Tenant/permission、个性化特征和数据版本进入cache key；TTL不能授权跨患者复用答案。脱敏、加密、权限、审计和保留/删除是不同层。

危险输出检查需要定义语义行为、证据和误拒/漏检，而不只是否写“我不是医生”。高风险路由/专家处理在服务层实现可靠priority/可审计升级，不能把关键词分类器当临床真值。出处冲突保版本和适用范围，不能以embedding相似度最高代替专业判断。

## 10. 面试问答16题

**Q01 上线前最少验证什么？** Artifact/模板、加载输出、任务quality、协议/性能回归和版本。

**Q02 Sync/async/stream怎样选？** 任务时长、恢复和用户交互，取消/未知状态契约。

**Q03 Timeout后直接重发？** 核backend完成、幂等和副作用，不能假定停止。

**Q04 TP和replica区别？** 同一请求协同与独立请求扩容。

**Q05 动态LoRA等静态merge？** Adapter/cache/资源身份不同，能力看版本。

**Q06 TTFT/TPOT怎么解释？** 首token和后续间隔，起点/窗口定义。

**Q07 p99可以平均各nodep99？** 不可当全体分位数，应按样本或正确histogram合并。

**Q08 高tokens/s一定好？** 长输出、失败和SLO可能改变goodput。

**Q09 Readiness等liveness？** 活着不等模型已加载并可接流。

**Q10 Affinity总比least-load好？** 缓存收益与hot key长尾取舍。

**Q11 Log全部prompt最方便？** 需必要性、权限、脱敏/保留和可追踪证据。

**Q12 Canary关键是什么？** 固定assignment、指标、窗口与回滚条件。

**Q13 Blue-green更便宜？** 可回滚便利与双环境资源取舍。

**Q14 Circuit breaker和retry区别？** 控依赖故障流量与重试单请求，须协调budget。

**Q15 安全拒答率越高越好？** 误拒、漏检、任务效用与风险分别量。

**Q16 用户反馈怎么训练闭环？** 候选不直接真值，经标注/去重/验证/版本与发布。

## 11. 来源、合并与项目关联

权重/显存/框架/推理核心机制复用既有Infra原文；请求契约、SLO分解和发布对照是本文教学设计，不登记当前商业产品报价或万能最优参数。源清单只提供问题范围，不证明所列医疗示例在你项目做过。

你的turn项目看[实现](#q=agentic-turn-framework)和[Critic/创新](#q=agentic-turn-innovation)，ESCD看[项目介绍](#q=escd-project-interview)，RAG看[原入口](#q=rag-vs-finetuning)。[清单合并地图](#q=post-training-checklist-map)将重复vLLM/分布式/项目题指向同一正文，避免再造多个八股版本。
