---
id: rag-vs-finetuning
title: "知识问答该用 RAG 还是微调？怎样判断失败在检索还是生成？"
category: llm
difficulty: 基础
tags: ["P0", "RAG", "评测"]
updated: 2026-10-13
summary: "合并RAG清单与延迟/并行/缓存题，补chunk、融合、rerank、grounding、版本与16道面试；保留人工证据定位矩阵。"
draft: false
---

## 问题背景

模型回答不了最新内部文档，直接微调不一定是最有效办法。先判断缺知识、缺检索还是不会利用证据。

本轮按[用户清单](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)88–99与175–198合并，继续用已有入口。医疗知识库等原项目例子在本页作为通用版本/权限/证据问题处理，不填成你的真实项目经历。

## 前置知识

知识供给、检索排序、生成行为、证据核验、路由和系统SLO不同。Embedding相似度不是事实可信概率；top-k不是生成采样top-k；检索是否命中正确证据也不同于是否读懂它。模型、索引、corpus、chunker、embedding与reranker都应版本化。

## 两种机制

RAG 把外部检索结果作为当前输入的证据，方便更新、溯源和权限控制；微调改变参数，更适合学习任务格式、领域行为和稳定模式。二者可组合，没有万能二选一。

```text
文档清洗 → 切块与索引
用户问题 → query 处理 → 检索 → 可选 rerank → 证据上下文 → 生成 → 引用核验
```

这是实践拆解，并非要求每一步都必需。原始 RAG 论文研究检索器与生成器结合，今天的工程管线可能使用不同组件。

## 失败定位矩阵

| 对照 | 观察 | 更可能的问题 |
|---|---|---|
| 人工正确证据 + 当前模型 | 仍答错 | 证据利用、指令或模型能力 |
| 正常检索 + 当前模型 | 证据缺失 | query、切块、召回、索引 |
| 检索含正确证据但答错 | 受无关内容干扰 | 排序、上下文组织、生成 |
| 无答案问题 | 仍编造 | 拒答和证据约束不可靠 |

切块越小不一定越好：可能丢失定义上下文；越大也可能稀释检索语义、增加 token 成本。先用一小组已知证据的题定位，再调参数。

## 指标要分层

检索看 Recall@k 和证据覆盖，生成看正确性、引用是否真实支持结论、无答案时的行为。最终看端到端成功率与延迟。Recall 高不表示回答必然正确；把答案所在文档泄漏进测试构造也会制造虚高效果。

## 小实验与自检

选 20 个可公开验证的问题，标注支持段落。逐题运行无检索、正常检索、人工正确证据三组，记录错误归因。这是建议实验，不是站内已完成结果。

自检：为何微调不提供实时知识更新保证？引用存在和引用正确有什么区别？

## 面试表达

从知识更新频率、可溯源要求和行为学习需求出发选方法，再讲检索与生成的分阶段评测。

关联：[Agent 评测](#q=agent-evaluation)、[上下文管理](#q=agent-context)。

## 原始资料与阅读提示

调研日期：2026-09-30。以下为论文或官方文档；正文是学习性整理，小实验是建议练习，未声称已复现。

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)：理解参数知识与非参数检索记忆的结合；工程对照是本文建议。

## Chunk、Embedding、Hybrid与Rerank

切块先考虑结构：标题/段落/表格、出处、时间与权限，不只每512tokens硬切。小块可语义集中却丢定义，大块可保上下文却稀释表示/增加生成成本。Overlap能保边界信息，也增加重复/存储，不能无条件越大越好。索引保source/document/chunk版本与稳定ID，重切分后旧引用需可追踪。

Embedding选择先用实际语言/领域和短长query的检索集，量Recall@k、MRR/nDCG、成本和embedding latency；向量维度大不保证更强。Cosine、dot product与归一化需匹配训练/索引设定，不任意改distance function后叫同模型。

Dense更善语义近邻，sparse如词项检索更善精确名称/编号；hybrid可并行召回并融合。RRF用排名避免不同score尺度直接相加；若学加权fusion，用独立数据调权与验证。去重不能误合并不同日期/条件冲突文档。

Reranker对query–candidate联合打分，可能更精准但加prefill/模型请求。先比较无rerank、固定候选数/长度的rerank与oracle证据；若已召回缺证据，rerank不能凭空找出缺失文档。跳过rerank的fast path要以质量/风险评估，而非固定“简单问题都不需要”。

“Naive/Advanced/Modular RAG”常是工程组织术语，不是一套全球固定loss；解释增加了query改写、hybrid、rerank、证据验证或可组合接口哪一步。Safety-RAG也需具体定义风险/验证模块，不能仅多加一段安全prompt就宣称保证。

## Lost in the Middle、Grounding和无答案

长context可接受不等证据能可靠利用。证据排序、去冗余、摘要保出处和针对性query可帮助，但要用固定集量不同证据位置、冲突和多跳。不能看到生成可引用就判断支持正确：逐claim映到具体片段，检查是否真蕴含、是否过时、多个源是否独立。

缺证据可澄清、改写query、补检索或明确无充分依据；不应以回答流畅作为成功。学拒答还需区分answerable与unanswerable、错误拒答与幻觉，安全系统不能只把拒答率最大化。

对冲突来源，保时间、版本、适用范围、权威性与原文，不强行以相似度最高决定事实；重要应用的冲突处理规则由领域要求定义。Search agent可以主动查证，而简单RAG不一定有多轮动作，两者详见[Search组件](#q=agent-search-components)。

## 延迟、并行和Batch的真实关键路径

一次请求可拆query预处理、embedding、dense/sparse召回、融合/去重、rerank、context组织、LLM排队/prefill/decode、引用验证。没有普适“每阶段典型百分比”；收真实trace和p50/p95/p99分布，同窗口预算和网络。

独立dense/sparse可以并行，依赖排序结果的最终grounded回答不能在证据未到时假装已完成。所谓检索/生成重叠可以做无证据中间操作或明确的speculation，最终答案仍须满足证据条件。部分检索失败时明确degraded policy、timeout/cancel、是否允许少源回答。

```python
# Teaching sketch: not a complete production retriever.
async def retrieve(query):
    jobs = [dense(query), sparse(query)]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    usable = [r for r in results if not isinstance(r, Exception)]
    if not usable:
        raise RetrievalUnavailable()
    return fuse_and_deduplicate(usable)
```

这里parallel不自动解决deadline、限流、任务取消与异常分类，生产还需bounded concurrency和trace。FAISS IVF/HNSW等近似检索参数影响召回/内存/latency，不把索引类型当万能优化。Batch embedding减少部分RPC和提升设备效率，但等batch会增加尾延迟；区分离线索引与在线交互。

P99可由排队、cache miss、长文、rerank候选爆炸、索引锁、慢依赖和长输出造成。增加concurrency若已饱和只增排队；先查实际可运行请求与每阶段队列，再调batch和资源。Token吞吐高可能来自输出变长，与任务goodput/SLO不同。

## Cache、TTL、权限与动态路由

分层缓存query规范化结果、embedding、retrieval candidates、rerank、prefix KV或最终回答。每层对象和key不同；semantic answer cache尤其不能仅靠相似向量跨用户reuse个性化答案。

Key至少含原始/规范query、corpus/index/embedding/reranker版本、权限/租户、过滤条件；答案还受model/adapter/template和解码影响。TTL只是时间策略，知识源/权限改变需主动失效。相同文本embedding可按内容hash与model revision重用，source出处关系仍保留。

Prefix caching复用推理模型相同prefix的KV，不等最终答案缓存；model/adapter变化或不同prompt字节会改变条件。共享前缀也不能跨租户越权暴露内容。服务安全与版本见[服务化专题](#q=model-service-interview)。

Dynamic route可以按证据可得性、复杂度、风险和资源走fast/full path，需明确fallback与decision日志。高风险并非任意关键词就能可靠判定，规则/模型route需专家规则与误漏检评测；不要为降低延迟关掉必要验证。全path效果、路由后效果和路由误差分开报告。

## 面试问答16题

性能优化后重新评估同一题集的证据完整性和无答案行为。减少top-k、缩短context或跳rerank会改变质量条件；若主动改变协议，应与纯系统提速分开记录，不能把牺牲任务覆盖后的低延迟称为同条件优化。

### 一次检索失败怎样形成可学习的对照

固定题集记录gold supporting spans和无答案标注。A组正常检索，B组人工正确证据，C组无证据，D组保留正确证据但注入同主题无关片段。A差B好指向召回/排序，B仍错指向利用或模型，C/D自信编造揭示证据约束或鲁棒性。具体分数来自实验，不把这个定位逻辑写成所有错误的唯一因果证明。

Retrieval Recall@k要求定义正确证据集合，文档命中与完整支持多个claim不同。多跳问题只找到第一跳可能计文档recall不错却不足回答，需coverage或各跳支持。MRR侧重首个相关项，nDCG可用相关等级，但grade定义来自标签；不同指标权重不等端到端任务收益。

排名融合可用$\mathrm{RRF}(d)=\sum_j1/(c+\mathrm{rank}_j(d))$，c为平滑常数，缺失候选按既定规则处理。它避免直接相加不同分布的dense/sparse原始score，却不使所有rank变成可校准正确概率。选择c与候选预算仍要在独立验证集，不在test上反复挑最优。

Cache测命中率之外还测正确命中：同query但权限/知识日期/个人条件不同，不应返回另一请求答案。公共embedding可在许可对象内重用，retrieval与最终response key必须覆盖实际过滤与版本。记录cache miss成本、stale reject与主动失效，并观察尾延迟，而不只平均cache-hit latency。

问题路由的总质量应覆盖fast/full/degraded与route错分：把所有难题直接送人工可降低模型错误，却改变自动完成率与成本。汇报自动成功、人工比例和总延迟，不拿被筛剩的容易请求当全流量。对敏感应用，具体risk规则由领域需求定义，本文不把简单关键词检索当真实临床判定。

**Q01 RAG还是微调？** 知识更新/权限/溯源与行为学习分开，二者可结合。

**Q02 Chunk怎么选？** 结构/任务证据、质量与token预算对照，不只调固定长度。

**Q03 Overlap越大越好？** 保边界与重复/成本有取舍。

**Q04 Embedding用什么指标？** 实际领域Recall/MRR/nDCG、延迟成本，不只向量维度。

**Q05 Dense+sparse score直接加吗？** 尺度未校准可用rank fusion或独立调权。

**Q06 Reranker总是必要吗？** 独立quality/成本对照，缺召回不能靠重排补。

**Q07 Lost-in-middle只靠加context解决？** 可接受长度和证据利用不同，做位置/冲突测试。

**Q08 引用存在就grounded？** 要核claim与原片段支持、时效和独立性。

**Q09 Recall高为什么还错？** 排序/上下文/生成/评价等层，使用原定位矩阵。

**Q10 Retrieval慢怎么拆？** Query/embedding/索引/融合/rerank/网络各量，不造固定占比。

**Q11 asyncio.gather就完整容错？** 还需deadline、bounded requests、cancel与degraded policy。

**Q12 Batch对实时总更快？** 设备吞吐与等待尾延迟分别量。

**Q13 Semantic cache如何避免错复用？** 权限/版本/个性化key与相似阈值、有效性验证。

**Q14 TTL足够吗？** 版本/权限改变还需主动失效。

**Q15 Fast path依据什么？** 质量、风险、资源并评估route误差，不绕必要验证。

**Q16 项目怎么证明收益？** 检索/生成/grounding与端到端quality/latency，同集合对照。

## 合并来源与学习路线

旧版RAG/微调定位矩阵保留，本轮在原文内新增清单所缺工程部分，没有复制一个同名RAG专题。Judge与RM理论见[Reward/Judge](#q=reward-judge-interview)，推理prefill/KV与TP见[Infra引擎](#q=infra-inference-engine)。[题号地图](#q=post-training-checklist-map)只记录知识对应与正文入口，不声称你已做过外部作者的医疗项目。
