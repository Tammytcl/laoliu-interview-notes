# 后训练清单与个人项目审查 · 2026-10-13

本轮新增8笔记，原LoRA/RAG保留ID扩写，新增1survey/1Daily，连接原项目hub、通用pipeline与roadmap。最终66笔记、63论文、14报告（6Daily/8survey）、38注册专题。本轮正文编号问答172，外部清单另222个sourceNumber完整映射。

## 范围与许可

原URL100问、导航152问、正文222问，HTML表实际唯一编号1–222。来源只作题型范围；未找到可确认开放license，不转载整份原题。Mapping以本文重述的简短知识标签关联入口，JSON不包含sourceQuestion，旧个人医学项目模型/数据/指标不移植为用户经历。

旧算法与Infra复用，LoRA/RAG原文合并，缺口新增基础分词、Reward/Judge、服务化。医学相关只讲风险路由/隐私/责任与证据，不提供诊断处方。WHO健康LMM治理资料为primary依据，具体领域规则需按应用定义。

## 第一轮：方法、代码与项目角色

ESCD用公开befd0c0a582ec6efc055660467db95d522519591及repo45页PDF。Root与child、teacher mass权重、native完整分布、集合NLL、Single分解、覆盖两个分母与COUF分别解释。固定原报告与arXiv v1不混；公开成绩不当本轮复现，作者/项目负责人身份不替具体个人模块分工。

Turn工作树在用户指定目录，只读冻结42源码/文档sha。HEAD e2e0bcc是基线，不是未提交内容的公开commit。确认single-forward current detach reference与rollout behavior不同，TIS路径未开启，Actor只有weighted sampled OPD，PPO联合方案与监督收益Critic是备选。Same-version value pair、过去RMS、warmup、weighted denominator及中性回退按真实实现。

对基础题修正：WordPiece/BPE算法与SentencePiece工具不同，中文效率依语料/单位，QLoRA按LoRA+三关键组件不凭空第四技术，BF16不保证永不非有限值，DAPO不只长尾preference。LoRA零B初始A/B梯度、merge理想等价与全参结果不同，RAG缓存/权限/证据与服务的版本分开。

## 第二轮：数值、原资产和呈现

ESCD两张公开原资产实际查看，与45页报告Figure3/Table1对应；没有换旧v1报告原表。三张新教学SVG在浏览器逐张查看，双更新图把dashed版本回线移出说明文字；原资产/图解/provenance/hash登记各figures.json。

CPU白板实际运行：causal MHA未来扰动对先前输出差0；LoRA首步A梯度0、B非零，理想merge差约8.88e-16；DPO初始loss log2、chosen/rejected梯度−0.1/+0.1；共同缩放权重不改window梯度−3.2；ratio1仍有logprob梯度−0.7；BPE非重叠与critic跨版本漂移例子。Scope为toy，不是GPU/真实tokenizer/环境或模型benchmark。

## 检查记录

38专题来源/公式/编号/链接门槛通过，63/63论文结构证据门槛通过，32单元测试通过；新增测试核222映射与42工作树hash、不转载原题及项目手机入口。Build/浏览器/线上检查按实际完成结果归入最终交付与Actions日志，不以push替部署确认。

本轮未改用户项目代码、提交训练、查询/修改云任务、导出用户模型或给未完成消融造结果。源观察候选与正文已核版本分开；外部变化需下轮语义核查。

本地38专题双视口与8旧路由、来源/资产/CPU字节验收无错误；24浏览器用例通过，ESCD原图和turn源码块实际查看。地图链接标签由内部ID改成中文专题名，保留同一222映射与目标；最终部署检查仍按新正文执行。
