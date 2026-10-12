# RL、Agentic OPD与项目面试审查 · 2026-10-12

范围：十篇新增笔记、七篇新精读、两份专题、一份Daily；补旧PPO先修链与ESCD公开代码状态。内容为58笔记、63已发布论文、12报告（5Daily/7survey）、28注册专题。GSPO/SAPO与DORA既有草稿状态保留。

## 第一轮：对象、公式和源码

六篇基础先区分问题背景、前置知识，再推策略梯度、baseline、TD/GAE、PPO ratio/clip、group与归一。两动作例子、bootstrap与trace两个mask、工具观察可见但不参与action loss逐项检查。

Agentic沿proposal/execute/train三身份和真实history比较。Guided混合历史不是纯轨迹分布线性混合；UOPD候选gate不是独立mixture；ActFirst full action不执行、fast推进状态，未来reference不进full；异步动作校正不自动修正state occupancy。

固定源码包括历史Baselines、verl、Search-R1、GiGPO、微软Lightning、UOPD与ESCD。现代实现不混为早期实验；真实节选和教学伪代码分开。Guided404、ActFirst无方法仓库、新MOPD legal-review状态均保留。ESCD现公开，arXiv v1图表固定，repo新PDF不代换原成绩。

## 第二轮：原图、配置与结果账本

| 报告 | 固定版本/原始证据 | 核对与修正 |
| --- | --- | --- |
| Search-R1 | 2503.09516v5；Figure1、Table2/3/4 | 0.431−0.348=8.3pp、约23.9%relative；PPO并非全设置领先；mask不删观察 |
| GiGPO | 2505.10978v3；Figure2/3/4/6、Table1/2 | 1.5B/std ALF86.7；0.54/362.83约0.149%，原文<0.002%不一致；当前episode跨step统计另核 |
| Lightning | 2508.03680v1；Figure3/4/5/6/7、Table1 | 图号、角色共享权重、终局信用、曲线无统一算力/多seed证据 |
| Guided | 2606.15912v2；Figure1/2、Table1/2 | Score两列与SR/round四列；250前200退出，exp零端点缺口 |
| UOPD | 2609.34036v1；Figure1/2/3、Table1/2 | 主计时八RTX PRO6000 Blackwell；评测seed非训练seed；±20非epsilon clip |
| ActFirst | 2609.36608v1；Figure1/3、Table1/2/3 | 真论文全名；八A100、FIFO一次；低并发<1、Web1.7 SR降1pp、准备成本排除 |
| 新MOPD审计 | 2610.04272v1；Figure2/3/4、Table1/2/4/19 | 实际fig_cost而非旧fig_curves；full-run非总成本；长度概括与柱值不一致；reporting集调参边界 |

37张新原始证据图逐张查看，对照固定PDF/TeX图号。Search/GiGPO/ActFirst/Guided表注边界补padding，不截整页、不重画结果；内联TikZ方法图记录PDF理由。Archive/TeX/asset/PDF/page/crop/hash保留figures.json。七张教学SVG实际浏览器渲染逐张查看，Pipeline回线移到框外。

## 题源和项目证据

新笔记156题：基础72、系统12、三篇面试72；新论文另56题。牛客/题解/个人页只提供公开题型，未独立验证公司/频次，其余明确自拟；答案回原文与公开代码。

ESCD固定befd0c0a582ec6efc055660467db95d522519591，核teacher causal hidden、root FKL、completion logsumexp、三top-k、EOS与基线适配。Release不保证全部历史checkpoint provenance。Agentic工作树只读用于设计问题，未启动实验，不公开内部任务/地址/原成绩；个人分工与未独立核成绩按真实记录填写。

## 数值与网页验收

共享CPU脚本只用标准库/torch CPU，核baseline、GAE边界、clip/CISPO、组归一、RKL/FKL、gate、假设latency、GiGPO算术、改context伪ratio和completion事件梯度。结果scope明确，非模型或真实CP/TP验证。

结构/来源/公式/坏链接门槛通过，31单元测试通过。浏览器首次启动遇60s server超时后重试，不跳过验收。后续实际部署/线上验收状态另补，push不等部署成功。新专题、图片hash、练习下载和刷新均沿通用检查器。

本地验收完成：63/63论文严格结构证据门槛、28专题来源门槛、31单元测试、23浏览器测试通过；全库双视口562次原图检查及28专题×2视口、8旧路由、source/练习字节检查无错误。CPFS构建超过自动server 60s窗口，改为先完成build、手工启动相同两个preview地址后运行原完整浏览器测试集，未跳过用例。新表格放大图等待实际加载后人工查看，桌面项目页与手机pipeline标题/目录核对。
