# Agent组件与流程整理核查 2026-10-10

用户要求放在Agent专题，详细介绍code/search agent涉及的术语和Harbor、E2B、ACS等组件，便于讲流程。本轮新增四篇问题笔记，不扩成产品广告或训练算法综述，也不新增未经精读的单篇论文。

## 交付与边界

| 文章 | 正文字符 | 问答 | 主问题 |
| --- | --- | --- | --- |
| agent-components-overview | 8891 | 12 | 术语分层、一次任务、接口与状态 |
| agent-sandbox-components | 10991 | 12 | Harbor/E2B/ACS、生命周期和源码 |
| agent-code-components | 7814 | 8 | 仓库初态、读改执行、测试和恢复 |
| agent-search-components | 约8100 | 8 | 查询、正文、证据、引用和验证 |

共40道本文整理问答。各篇问题背景与前置知识分开，按同一任务连接职责、数据流、失败与完成条件。真实代码片段带固定commit/行号，教学伪代码、API示意与实际CPU实验分别标识。ACS未提供全称，结合E2B/沙箱语境按阿里云容器计算服务Agent Sandbox解释，正文显式区分同缩写协议；Harbor指Terminal-Bench团队框架，区分同名产品。

## 原始来源与关键回查

30个独立引用URL、37个逐篇来源条目，以官方文档/规范、固定源码和作者解释为主。未做provider搜索质量、云性能或采购比较。

- Harbor `09f5b964f8e5075f7e52fd08c9994dfc80ad5d24`：Trial.run、E2B adapter、TaskConfig，源码内版本0.24.0。配置默认schema1.4与网页示例1.3并列说明，未用旧模板伪装最新CLI。
- E2B `df6c368152b067512702fc677952c0fc4e13c2f0`：create/pause与commands同步实现，核后台handle、bash执行、full/filesystem模式；官方mode文档标SDK2.55.0起，不强加给旧SDK或兼容backend。
- OpenSandbox `9bc2967b78a9506a9282b0b28dc5db0cb8c50cc0`、mini-SWE-agent `04d809ceab9df28f9adaed044884180159172930`：核README作为项目定位和代码阅读入口，未宣称精读完整运行系统。
- ACS创建/概述：核Set/Claim、controller/manager、SDK认证域名与镜像契约。Claim的ttlAfterCompleted删除Claim，不自动删除已领取Sandbox；自定义镜像不等于支持run_code。
- MCP固定2025-11-25规范，ACP/A2A用于接口角色区分；LangGraph persistence区分checkpointer与store。网页材料是数据，annotation不是执行控制。
- Tavily、Exa、Firecrawl、Playwright：核接口与角色，snippet/raw_content/自动answer和真正证据分开。日期、版本、获取时间、相关性与支持关系各自记录。
- Anthropic长期harness和评测文章用于连续任务/独立判据的解释，不复述未经验证的产品性能。

四份来源观察账本共16条自动观察记录，本轮全部成功、0待核对；相同源码不同用途仍分别登记。代码按文件hash观察，选定Harbor/E2B Markdown文档加入hash观察；其余动态网页和search篇来源下轮人工核查，不能把0个自动入口解释成已全面监控。

## 图表与实际练习

四张本文SVG分别说明组件地图、沙箱管理/执行层、代码修复循环、搜索证据链。逐张查看渲染图，核文字、箭头、条件和教学归属，登记hash及视觉核查；图不是厂商架构原图或实测结果。手机上可查看并放大，保持原比例。

[CPU脚本](../assets/agent/agent-components-lab.py)只用标准库，在临时目录中实际运行独立断言并检查固定契约；[结果](../assets/agent/agent-components-lab-results.json)已生成：

- 错误函数exit1，应用实际patch后exit0；3条独立断言包含空数组、正数和对称负正数。恢复聊天却重建旧文件后仍exit1。
- 首次回复丢失后，朴素两次请求产生2次效果；相同operation ID的单进程去重产生1次效果。未实现持久去重/并发协议，不宣称分布式exactly-once。
- 4条必要断言支持3条，coverage0.75；3条版本合格记录只有2个独立来源根，v1记录不支持目标v2断言。支持关系预标注，不是网页语义真伪检测。
- 完成Claim过期后Sandbox仍为running，需要明确清理。这个对象模型演示生命周期区别，不是在ACS集群执行删除。

没有创建收费云沙箱、调用收费搜索、运行模型或执行真实benchmark。

## 页面和工作流

新增四篇在Agent分类，使全站48篇笔记、Agent分类11篇；论文/报告数量保持56/9。基础workflow和evaluation、学习路线、README及分类说明接入入口。注册专题增至18篇。

线上检查器按注册文章的真实分类检查列表，保留所有Infra检查并增加Agent；练习发现从assets/infra扩到合法assets子目录，继续核线上/本地字节一致。新浏览器场景覆盖手机上的四篇跳转、代码块与真实练习下载。

本地已通过来源/章节/图hash/问答检查、31项单元测试、21项浏览器测试与48篇构建。另手动查看桌面组件图、搜索API代码和手机任务目录/代码流程图。容器截图中文字体缺失与此前一致，DOM中文及路由校验正常，未改网站字体配置。

最终专题验收通过：18篇×1440/390两种宽度，36次正文/图/问答/目录检查，8个旧地址跳转；来源下载和所有练习脚本/结果字节与仓库一致，浏览器错误0。分类列表同时覆盖Infra与Agent，没有降低原有检查。
