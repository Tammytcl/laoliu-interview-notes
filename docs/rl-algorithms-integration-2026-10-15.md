# RL算法与论文内容整合记录

日期：2026-10-15。原有笔记与论文ID全部保留，未改变原文图表、实验数字或论文发布状态。站点仍有66篇笔记、63篇已发布论文和14份报告；GRPO主文加入来源观察与线上验收后，注册专题为39个。

## 阅读结构

- `rl-policy-optimization-map`：统一入口，先修路线、同一符号、五层比较、24道综合问答。
- `rl-actor-critic`：状态、轨迹概率、策略梯度、baseline、V/Q/A与softmax局部导数。
- `rl-returns-gae`：MC、TD、n-step、lambda-return、GAE、两个边界mask与target单位。
- `rl-ppo-training`：冻结数据、surrogate、clip、causal logprob移位、联合更新与同批复用。
- `grpo-rlvr`：同题分组、优势广播、token梯度、KL与采样成本。
- `sft-dpo-rl`：偏好目标推导、四个回答分数和分类实现。
- `survey-policy-optimization`：七篇真实源码、原实验比较、证据条件和课堂串讲。重复的公式卡片与题库回链到主文。
- 七篇论文：方法模块增加专属变量对应表，`note_ids`接回基础与总览，原五模块和专属证据继续保留。

## 理论与代码核对

重新读取固定Baselines与verl源码，纠正PPO主文将等价TensorFlow重述标为真实节选的两行，改回`U.clip`与`U.mean`原句。源码commit、文件字节hash与作者项目页面记录在`assets/learning/rl-algorithms-source-evidence.json`，固定版本不冒充上游当前HEAD。

自有`rl-algorithms-unified-lab.py`使用CPU float64，两步MDP枚举冻结均匀策略的四条等概率路径。初始化不完美critic，执行两轮真实forward/backward/SGD；检查old logprob、advantages与returns未被覆盖。另用同一六回答数据比较局部目标的梯度，明确DAPO局部函数不包含filter/refill与超长reward，GDPO课堂最终whitening按回答而非作者token-mask统计。

脚本检查PPO四分支、GSPO等优势时两种接线的值与梯度等价、不同token优势时的梯度差异、padding位置无actor梯度、global分母拆批一致、终止与截断、LOO缩放、零方差组、互补reward抵消和DPO梯度。JSON是该脚本的实际输出，不是模型训练、框架集成或论文复现。

## 验收范围

验收沿用专题来源与深度检查、全库论文严格证据检查、单元测试、构建、浏览器阅读链与桌面/手机布局。新增浏览器用例从统一入口进入基础与GRPO，再进论文报告并回到主文，同时核对课堂下载与结果字段。线上验收继续核对39个注册专题的正文、资源和旧地址；GitHub Pages部署还覆盖已发布论文的原图与布局。

源码观察仅记录上游变化，不自动将正文固定commit替换成新HEAD。GSPO/SAPO完整草稿继续可下载，其原始证据门槛沿用现有规则。
