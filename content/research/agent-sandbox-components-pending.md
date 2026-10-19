# agent-sandbox-components 来源观察记录

自动检测时间：2026-10-19T07:46:04.999Z

这是来源变化的候选账本。正文核对日期、固定版本和结论不会由检测脚本覆盖。仓库 commit 更新可能与本文无关；需要阅读差异后再判断。

观察 12 个来源条目；待核对 6 项；本轮失败 0 项。未观察的网页仍需在整理任务中核查。

## 待核对变化

- [harbor-framework/harbor src/harbor/trial/trial.py](https://github.com/harbor-framework/harbor/compare/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24...357c5e74020fc0890d69fb3a7db610896b9ce57d)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [harbor-framework/harbor src/harbor/models/task/config.py](https://github.com/harbor-framework/harbor/compare/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24...357c5e74020fc0890d69fb3a7db610896b9ce57d)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [Harbor任务配置网页](https://docs.harborframework.com/tasks/configuration)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [E2B full持久化](https://docs.e2b.dev/sandbox/persistence)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [alibaba/OpenSandbox README.md](https://github.com/alibaba/OpenSandbox/compare/9bc2967b78a9506a9282b0b28dc5db0cb8c50cc0...834c839dcf587784c16449aed56c21a5cf71b6d1)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [harbor-framework/harbor src/harbor/environments/e2b.py](https://github.com/harbor-framework/harbor/compare/09f5b964f8e5075f7e52fd08c9994dfc80ad5d24...357c5e74020fc0890d69fb3a7db610896b9ce57d)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。

## 抓取失败

本轮观察请求均成功。

## 完成核对的方法

按 docs/topic-workflow.md 读取来源差异，修订正文、引用、图解与题目，人工核对后更新 manifest 的 reviewedAt / revision / reviewedSha256。运行 topics:quality、测试、构建和浏览器验收，再提交推送；候选状态将在下一次观察中重新计算。
