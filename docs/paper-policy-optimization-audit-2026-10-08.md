# PPO与偏好及组优化论文批次核读记录

2026-10-08。按既有五模块精读与图表证据流程完成七篇；五篇发布，GSPO/SAPO因原文无表保留完整草稿。没有修改原表门槛，也没有执行训练复现。

## 固定版本与证据

| 报告 | 版本 | PDF SHA256 | 正文字符 | 原图证据 | 问答 | 状态 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| PPO | 1707.06347v2 | `e78feadadbdbb0b601b3c2bcc81404722cd431a489b307545f9b7bea1e8c4f5b` | 11268 | 4 | 8 | 发布 |
| DPO | 2305.18290v3 | `92cb3a2b71362acda98a789b03d88688fd33cf5fcf13f81d2b1de30ee7d3b67a` | 11410 | 5 | 8 | 发布 |
| GRPO | 2402.03300v3 | `6cc20b3c5b8d25b8b53868fc4ec1792c144f07d67bdf7395138efd4422197e7b` | 10993 | 4 | 8 | 发布 |
| GSPO | 2507.18071v2 | `67221ec313f42d8b3029686b604b2621a311ae0d74e38524092e7c7a94613bd0` | 9696 | 4 | 8 | 草稿：原文无表 |
| DAPO | 2503.14476v2 | `f01e4fd347530cadd68e5c36b1998532a6d1adb272c817e73b927453c26e9d79` | 9722 | 5 | 8 | 发布 |
| GDPO | 2601.05242v1 | `77a9412a04c3ea358d1305e7815ed3af82afb783261b25966f93f9ff8f922193` | 9962 | 8 | 8 | 发布 |
| SAPO | 2511.20347v2 | `a413610718e93f3519de9ce3cf31511ec07a283784d8fb68b3d61b9891efdd0f` | 9415 | 4 | 8 | 草稿：原文无表 |

## 原图与手算审查

34张图/表/算法原式全部经view_image逐张查看。LaTeX原资产优先，多面板保留全部轴和图例；原表用PDF紧裁。PPO源码包只有编译PDF，因此记录不存在独立资产的原因；GSPO没有方法图，展示原文公式5—7并如实命名。PPO Table1修复窄裁导致caption截断，Figure3去除相邻正文；GDPO Table5第一次预览缺后面任务，最终补齐Apps/Codecontests/Codeforces/Taco与底线后复核。计划与hash在scripts/figure-crops和assets/papers各清单。

手算在本地CPU用数学函数核对：clip正负分支、GRPO标准差、GSPO几何平均、GDPO等权互补抵消、DPO sigmoid loss、SAPO gate与wr、DAPO长度惩罚和表格百分点差。没有调用下载的训练脚本。

## 来源与版本差异

- PPO原始公开commit `da997060461e3cbf54ca4dc7a67081a731fb6b3b`：traj_segment_generator、add_vtarg_and_adv、learn；与后续ppo2/SpinningUp区分。原Table2最后100episode胜出PPO19/A2C1/ACER28，另有1tie，已据原表核正。
- DPO官方commit `f8b8c0f49dc92a430bae41585f9d467d3618fe2f`：_get_batch_logps、concatenated_forward、preference_loss、get_batch_metrics；原始DPO关闭后续IPO/reference-free/label-smoothing分支。
- DeepSeekMath官方只给公开模型推理与评测说明，没有完整原始GRPO训练实现；现代verl commit `75879f7f475fd6b64c779f7d9212e45503f58b8f`核读优势、surrogate和reduction，GSPO/SAPO也用该维护者版本伴读。
- DAPO项目README commit `33fe3176f0bb212588e84fc8ccf50dd554975144`；公开复现verl commit `4f80e465c2ec79ab9c3c30ec74b9745de61d0490`核读fit过滤补采、reward长度项、loss/reduction。旧命名分支已不存在，改读可固定历史commit。128H20项目说明与后续128H800复现记录分列。
- GDPO官方commit `4ad86b4fbfc5db594f3a2750ff9c39fdc8ee6115`：compute_advantage、compute_grpo_outcome_advantage、masked_whiten；具体verl分支只处理两个reward，whitening按tokenmask统计，不能冒充任意n和逐sequence完全等价。
- DAPO完整贡献页37位去重人员，abs作者元数据35位，本报告按正文映射。GDPO的HKUST脚注、DPO的CZBiohub和DeepSeekMath的跨校单位按固定原文。

## 学习性解释的边界

GSPO长度根ratio不是标准完整densityratio；单样本IS不因样本数1就非法。SAPO平均gate误差界不等于全梯度误差界，soft目标可能饱和，clipfraction日志0不表示未衰减。GDPO两个reward互补等权仍能抵消；正确率与长度/格式的trade-off完整保留。DPO对话外部6B PPO不是公平2.8B重新训练对照。DAPO progressive表不是独立单因素消融，steps不直接等于GPU-hour。

## 讲解与发布工作流

参考OpenAI SpinningUp、ICLR PPO Implementation Details、HF DPO-TRL、Qwen GSPO和DAPO/GDPO项目页，来源融入正文且结果回到原文。外部搜索工具本轮返回429，使用原站点HTTP读取及GitHub接口获取可核对来源。未伪造面试题频、用户阅读时间或个人训练成绩。

七篇各8题，共56题；专题另16综合问题和5手算练习。既有同日Daily追加批次并保留DSec，sft-dpo-rl/grpo-rlvr补精读入口；专题按七种机制比较，没有扩写框架或硬件知识。GSPO/SAPO缺表问题已通过异步问题征询用户；未收到明确放行时不改变门槛。

## 验证记录

已通过55/55发布报告结构/证据检查、29项单元测试、18项Playwright浏览器测试与构建；本地全库55篇在1440/390两宽度完成474次图表检查，错误0；10个Infra专题、8个旧入口通过来源一致性与双宽度检查。额外新报告gallery与专题backlink手机核对通过，专题9模块全部七种方法、公式与全页宽度正常。截图已人工查看PPO/DPO/GDPO方法图放大布局。在线部署结果在完成后记录于GitHubActions，不能用本地成功替代上线成功。
