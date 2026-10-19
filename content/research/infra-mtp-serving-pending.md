# infra-mtp-serving 来源观察记录

自动检测时间：2026-10-19T07:46:02.830Z

这是来源变化的候选账本。正文核对日期、固定版本和结论不会由检测脚本覆盖。仓库 commit 更新可能与本文无关；需要阅读差异后再判断。

观察 9 个来源条目；待核对 2 项；本轮失败 0 项。未观察的网页仍需在整理任务中核查。

## 待核对变化

- [固定MTP文档](https://github.com/vllm-project/vllm/compare/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5...bbf59fbad49243d12dc966f13b037ffee39f7317)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [EagleVerifyInput](https://github.com/sgl-project/sglang/compare/23f890613e1798a99e9eee4453350f67751b1637...af9bb2a32d4a313d20a629ac1b2cb8e6ef245b95)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。

## 抓取失败

本轮观察请求均成功。

## 完成核对的方法

按 docs/topic-workflow.md 读取来源差异，修订正文、引用、图解与题目，人工核对后更新 manifest 的 reviewedAt / revision / reviewedSha256。运行 topics:quality、测试、构建和浏览器验收，再提交推送；候选状态将在下一次观察中重新计算。
