# agent-sandbox-components 来源观察记录

自动检测时间：2026-10-26T07:48:03.170Z

这是来源变化的候选账本。正文核对日期、固定版本和结论不会由检测脚本覆盖。仓库 commit 更新可能与本文无关；需要阅读差异后再判断。

观察 12 个来源条目；待核对 12 项；本轮失败 0 项。未观察的网页仍需在整理任务中核查。

## 待核对变化

- [Harbor核心对象](https://docs.harborframework.com/core-concepts/index)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [Harbor ATIF](https://docs.harborframework.com/agents/atif)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [harbor-framework/harbor src/harbor/trial/trial.py](https://github.com/harbor-framework/harbor/compare/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24...872da9479f4a2078142d22c49c9ffae0f09e2e63)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [harbor-framework/harbor src/harbor/models/task/config.py](https://github.com/harbor-framework/harbor/compare/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24...872da9479f4a2078142d22c49c9ffae0f09e2e63)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [Harbor任务配置网页](https://docs.harborframework.com/tasks/configuration)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [Harbor验证与产物](https://docs.harborframework.com/tasks/verifier)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [e2b-dev/E2B packages/python-sdk/e2b/sandbox_sync/commands/command.py](https://github.com/e2b-dev/E2B/compare/df6c368152b067512702fc677952c0fc4e13c2f0...a2b2caea4498766aee8fd33d3280b56744070e99)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [E2B full持久化](https://docs.e2b.dev/sandbox/persistence)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [E2B filesystem-only](https://docs.e2b.dev/sandbox/filesystem-only-snapshots)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [E2B fork](https://docs.e2b.dev/sandbox/fork)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [alibaba/OpenSandbox README.md](https://github.com/alibaba/OpenSandbox/compare/9bc2967b78a9506a9282b0b28dc5db0cb8c50cc0...4423a6ad8b6e91ce369422feb6c97605636303bf)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [harbor-framework/harbor src/harbor/environments/e2b.py](https://github.com/harbor-framework/harbor/compare/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24...872da9479f4a2078142d22c49c9ffae0f09e2e63)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。

## 抓取失败

本轮观察请求均成功。

## 完成核对的方法

按 docs/topic-workflow.md 读取来源差异，修订正文、引用、图解与题目，人工核对后更新 manifest 的 reviewedAt / revision / reviewedSha256。运行 topics:quality、测试、构建和浏览器验收，再提交推送；候选状态将在下一次观察中重新计算。
