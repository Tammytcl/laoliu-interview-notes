# training-inference-frameworks 来源观察记录

自动检测时间：2026-10-19T07:45:59.272Z

这是来源变化的候选账本。正文核对日期、固定版本和结论不会由检测脚本覆盖。仓库 commit 更新可能与本文无关；需要阅读差异后再判断。

观察 12 个来源条目；待核对 9 项；本轮失败 0 项。未观察的网页仍需在整理任务中核查。

## 待核对变化

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/compare/b1a6f076993662664f824d1cb8e81e51cf572b8b...ce852bfb93f364f64a556602fc71a5d9ed2c377e)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [verl 官方仓库](https://github.com/verl-project/verl/compare/8718ca30a3f002f93b7c4fd99b9b2506718681bc...263653e749306287a5032140cce3d7833c1959d0)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [slime 官方仓库](https://github.com/THUDM/slime/compare/8c17b676cb57af1d17ee4402e91e9209af84b60b...4f204660abcd531882dd4918ee301f107a952eb4)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [示例目录](https://github.com/verl-project/verl/compare/8718ca30a3f002f93b7c4fd99b9b2506718681bc...263653e749306287a5032140cce3d7833c1959d0)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [Qwen 数学 GRPO 示例](https://github.com/verl-project/verl/compare/8718ca30a3f002f93b7c4fd99b9b2506718681bc...263653e749306287a5032140cce3d7833c1959d0)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [Qwen3-4B 启动脚本](https://github.com/THUDM/slime/compare/8c17b676cb57af1d17ee4402e91e9209af84b60b...4f204660abcd531882dd4918ee301f107a952eb4)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [AReaL](https://github.com/areal-project/AReaL/compare/2fad2d0e308fe631e70e971b97188ad5c5cc03cb...1683e8e8fba70e296b4238aa3c2300defd192f17)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [slime fully_async 示例](https://github.com/THUDM/slime/compare/8c17b676cb57af1d17ee4402e91e9209af84b60b...4f204660abcd531882dd4918ee301f107a952eb4)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。
- [slime-agent-guide](https://github.com/THUDM/slime/compare/8c17b676cb57af1d17ee4402e91e9209af84b60b...4f204660abcd531882dd4918ee301f107a952eb4)：与正文核对基线有差异；核查受影响段落、支持组合、源码入口和示例。

## 抓取失败

本轮观察请求均成功。

## 完成核对的方法

按 docs/topic-workflow.md 读取来源差异，修订正文、引用、图解与题目，人工核对后更新 manifest 的 reviewedAt / revision / reviewedSha256。运行 topics:quality、测试、构建和浏览器验收，再提交推送；候选状态将在下一次观察中重新计算。
