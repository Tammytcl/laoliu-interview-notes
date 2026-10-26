# opd-interview-foundations 来源观察记录

自动检测时间：2026-10-26T07:48:07.880Z

这是来源变化的候选账本。正文核对日期、固定版本和结论不会由检测脚本覆盖。仓库 commit 更新可能与本文无关；需要阅读差异后再判断。

观察 2 个来源条目；待核对 2 项；本轮失败 0 项。未观察的网页仍需在整理任务中核查。

## 待核对变化

- [Tinker OPD训练源码](https://github.com/thinking-machines-lab/tinker-cookbook/compare/54b3cb293038ccf2e49952e46ed5abac7119fdf8...5624af2cd982fba6e5861e08dc142f0e2ae8e0a2)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [verl OPD官方文档](https://github.com/verl-project/verl/compare/3c6d9b43b67c74ec343d77874cd8fa6e841523da...3dd97a0e1b4b2cdf84c302c53431efaad441ae9b)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。

## 抓取失败

本轮观察请求均成功。

## 完成核对的方法

按 docs/topic-workflow.md 读取来源差异，修订正文、引用、图解与题目，人工核对后更新 manifest 的 reviewedAt / revision / reviewedSha256。运行 topics:quality、测试、构建和浏览器验收，再提交推送；候选状态将在下一次观察中重新计算。
