---
id: "infra-mtp-serving"
title: "MTP 工程：怎样接入 vLLM SGLang，并判断是否真正加速？"
category: "systems"
difficulty: "深入"
tags: ["P1", "MTP", "Multi-Token Prediction", "Speculative Decoding", "Infra"]
updated: "2026-10-09"
summary: "核固定后端源码、权重完整性、参数入口、接受长度与显存账本，设计同负载A/B验收。"
draft: false
---

## 1. 问题背景：开关打开不等于加速链条成立

MTP服务要同时满足三个条件：模型确有可用辅助权重，后端知道怎样消费hidden/token与验证状态，真实负载下节省的target调用足以抵消draft和verification成本。只在配置中写mtp，既不能为普通NTP checkpoint凭空训练辅助模块，也不能证明模型文件中的额外权重真正被加载。

这篇按权重→模型接线→服务参数→测量组织。版本固定在2026-10-09核读的vLLM/SGLang commit；例子用于解释官方入口，不声称已启动GPU服务。基础原理见[MTP总览](#q=infra-mtp-overview)，采样正确性见[投机解码](#q=infra-mtp-speculative)。

## 2. 前置知识：四种长度与三种模型成本

**训练深度D**描述学了多少额外预测模块；**draft steps**描述运行时调用模块几次；**tree node预算**描述验证多少候选节点；**accepted prefix length**描述最后沿一条路径实际接受多少token。它们不相等，一个训练模块可以循环调用，增加节点也可能没有增加有效输出。

**总参数**影响权重存储，**激活参数**影响每token被使用的计算，**共享参数**决定是否重复存储。DeepSeek MTP的block包含MoE，不是一个几MB输出头；不能用激活参数量当磁盘或显存下界。逻辑共享也不保证在两个worker/process/parallel placement中物理只存一份。

**Round cost**包括draft、target verify、调度、同步与提交；每个有效输出token的时间要除以真正的产出量。KV还要区分target缓存、MTP/draft缓存、候选tree临时位置与pending token。先定义账本，再看“接受率高”有没有价值。

![本文示意 · 每轮开销与每个真实输出token的成本](./assets/infra/infra-mtp-serving/mtp-cost-budgets.svg)

**图解。** 第一排把总round cost拆成draft、verification和其他开销；第二排分别提醒接受前缀、内存峰值与同负载比较。公式使用平均round time除平均emitted tokens，不是只比较target调用次数。该图是成本分类，不包含GPU实测；权重、KV、临时buffer不能由一个接受率替代。

## 3. Checkpoint检查：先看config和权重键，再看模型名称

[DeepSeek-V3固定权重说明](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md)区分61层main与1层MTP：main为`model.layers.0`到`.60`，辅助层接在`.61`。`num_nextn_predict_layers=1`描述模块数，不是需要额外起一个完整独立语言模型。

```json title="官方权重说明的关键字段 · 结构摘录"
{
  "model_type": "deepseek_v3",
  "num_hidden_layers": 61,
  "num_nextn_predict_layers": 1
}
```

该JSON是解释字段关系的摘录，不是完整可用config。norm、projection、共享embedding/output、attention/MoE权重和量化scale都须检查；模型主网络可以正常NTP运行，也可能恰好缺失MTP层。

当前固定官方说明写MTP为**11.5B unique parameters**，不含共享0.9B embedding和0.9B output head；激活参数2.4B包含共享部分。不能把传播中的某个“14B draft model”数字不加口径地当成独立常驻模型。不同checkpoint与导出版本应逐份核对字段和权重键。

以11.5B全部按BF16存储的假设计算，独有权重约23GB、21.42GiB；这不是实际混合精度checkpoint的完整显存。FP8还需scale，norm与projection也未必都是FP8。共享参数若跨角色被复制、再加KV、verification logits与allocator reserve，实际峰值会更高。[权重与分片基础](#q=infra-checkpoint-formats)解释怎样分别检查存储对象和dtype。

### 导出、量化与恢复中的常见缺口

转换工具可能只遍历main的`num_hidden_layers`，把额外层删除；量化脚本可能没有处理其权重或scale；SFT/RL更新main后，MTP仍是旧版本。前两者会无法运行或错误加载，后者可能降低proposal接受率。不能以NTP能生成判断MTP完整，也不能把关闭完整性检查当成修复缺层。

[vLLM加载检查](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/vllm/model_executor/models/deepseek_mtp.py#L515-L532)扫描加载到的MTP层，在缺少预期层时抛错。它检查层的存在，不等于校验该层与main共同训练或全部tensor语义正确；还要保留完整来源revision和实际load日志。

## 4. 关键源码：模块复用不等于新增训练head

固定vLLM commit `18c19eaf239f09bc2543637c73f3b2d05f9bf5c5`的[DeepSeekMultiTokenPredictor.forward](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/vllm/model_executor/models/deepseek_mtp.py#L214-L224)：

```python title="vLLM真实节选 · 按运行步选择已有模块"
if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
current_step_idx = spec_step_idx % self.num_mtp_layers
return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
    input_ids,
    positions,
    previous_hidden_states,
    inputs_embeds,
    current_step_idx,
)
```

当num_mtp_layers=1时，不论spec_step_idx增至几，都仍选择同一模块；输入candidate token与previous hidden逐步变化。增加runtime depth扩展起草跨度，不创造新训练层，误差与cost会累积。norm、projection和pre/post-final-norm状态细节见[训练篇源码](#q=infra-mtp-training)。

固定SGLang commit `23f890613e1798a99e9eee4453350f67751b1637`的[DeepseekModelNextN.forward](https://github.com/sgl-project/sglang/blob/23f890613e1798a99e9eee4453350f67751b1637/python/sglang/srt/models/deepseek_nextn.py#L193-L221)从`forward_batch.spec_info.hidden_states`取此前表示，将embedding与hidden分别norm后融合；CUDA路径使用fused_eh_norm，其他路径显示torch.cat语义。这是维护者的推理实现，不是公开DeepSeek原始MTP训练loop。

服务必须知道hidden对应哪个token位置、哪个权重版本、pre/post norm哪一版；只缓存一个[N,H]tensor而没有这些契约，无法保证下一步条件正确。MTP与main共享架构权重的概念，不自动免掉draft角色的状态和并行通信。

## 5. vLLM与SGLang：相同目标，不同参数计数

### vLLM官方MTP入口

[固定MTP文档](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/docs/features/speculative_decoding/mtp.md)以原生MTP模型为前提，在线例子：

```bash title="vLLM官方配置入口 · 未在本轮启动GPU服务"
vllm serve XiaomiMiMo/MiMo-7B-Base \
  --tensor-parallel-size 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

`method`选择MTP执行，`num_speculative_tokens`是runtime草稿深度，不是训练模块数。模型家族要在该版本实现，checkpoint要含所需权重。该例不表示所有7B checkpoint都支持MTP；完整模型、采样和内存配置另外检查。

TP主模型配置不应误塞进speculative_config的`tensor_parallel_size`键；draft角色有不同的配置字段。文档main快照也不等于你安装的release，必须登记包版本与实际commit、模型revision和启动参数，再比较行为。

### SGLang官方MTP入口

[固定speculative文档](https://github.com/sgl-project/sglang/blob/23f890613e1798a99e9eee4453350f67751b1637/docs/docs/advanced_features/speculative_decoding.mdx#multi-token-prediction)将MTP接入EAGLE工作流。以下以官方MiMo-7B-RL例子为基础，将host限制到本地，保留机制相关参数：

```bash title="SGLang配置摘录 · 参数计数与vLLM不同"
python3 -m sglang.launch_server \
  --model XiaomiMiMo/MiMo-7B-RL \
  --host 127.0.0.1 \
  --trust-remote-code \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2
```

这里使用EAGLE执行组织，不表示DeepSeek的MTP训练目标变成EAGLE训练。[EagleVerifyInput](https://github.com/sgl-project/sglang/blob/23f890613e1798a99e9eee4453350f67751b1637/python/sglang/srt/speculative/eagle_info.py#L49-L61)的verify链长度含root，node budget和draft steps各有职责；不要把vLLM值1直接复制到每个SGLang字段。tree top-k增大还会增加验证节点、临时KV与调度开销。

DeepSeek具体入口见[固定V3 cookbook](https://github.com/sgl-project/sglang/blob/23f890613e1798a99e9eee4453350f67751b1637/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V3.mdx)：文档有H200 TP8下的服务结果，属于该模型/负载的官方测量，不是本站实验。支持矩阵、overlap scheduler、PP/DP/量化组合必须核所用版本；不能把旧版不兼容警告或新的main功能当作所有release的永久事实。

## 6. 性能模型：接受率高，为什么仍可能变慢？

对固定长度gamma的线性草稿，设第j个token在前面全部接受条件下的接受率为alpha_j。忽略EOS与输出上限，平均产出：

$$
E[N_{emit}]=1+\sum_{j=1}^{\gamma}\prod_{m=1}^{j}\alpha_m.
$$

不能用“每头准确率都90%”直接替换所有alpha_j，它们是实际生成路径的条件概率；也不能把所有节点accepted/proposed的aggregate rate当作这个条件序列。若都为0.9、gamma=3，E[N]=3.439，低于4。

以普通decode一token耗时t_0为单位，稳态假设速度比为：

$$
S\approx\frac{E[N_{emit}]t_0}{E[t_{draft}+t_{verify}+t_{other}]}.
$$

| 本文假设例 | 每轮成本 / t0 | 平均产出 | 理论速度比 |
| --- | --- | --- | --- |
| 较便宜draft与verify | 1.6 | 3.439 | 2.149 |
| 原负载已很饱和、验证更贵 | 4.2 | 3.439 | 0.819 |

这张表是CPU算术例，不是GPU测量。同样接受长度，成本不同可从加速变成减速。draft输出projection、target多个query位置、通信、scheduler与cache commit都可能增长；高batch已通过并行请求摊薄权重读取时，MTP新增工作未必划算。

TTFT常仍受prefill主导；decode提升不能直接宣称首token也快同样倍率。工具等待、排队和环境执行占主要墙钟时，decode局部收益还要乘实际占比。低利用率若来自没有ready请求，应先排查供给，而不是只增加起草深度。

## 7. 接受计数与显存：先写对象，再解释指标

[vLLM当前固定acceptance指标](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/docs/features/speculative_decoding/acceptance_metrics.md)将mean_acceptance_length定义为含bonus的每步平均产出；draft_acceptance_rate只计accepted draft/proposed draft。三步各提3个草稿，实际接受[0,1,3]，产出[1,2,4]：aggregate rate=4/9，平均产出=7/3，不能报成同一个“接受率”。该API是实验字段，n>1、streaming与有效draft约束还有明确口径。

```text title="计数账本 · 先排除bonus混用"
verification rounds = 3
drafted candidates  = 9
accepted draft      = 4
emitted tokens      = 7
acceptance ratio    = 4 / 9
mean emitted        = 7 / 3 = 1 + accepted_draft / rounds
```

| 显存对象 | MTP为何可能增加 | 需要确认 |
| --- | --- | --- |
| main权重 | 通常仍需完整target | 精度、分片、加载副本 |
| 独有MTP参数 | 额外block/norm/projection | unique与shared、量化scale |
| main KV | 验证候选占临时位置 | accepted path与rollback |
| draft/MTP KV | 串行起草也有状态 | 生命周期、prefix复用与版本 |
| hidden与logits临时buffer | 多位置或tree节点共同验证 | 峰值、输出词表大小与dtype |
| graph/allocator reserve | shape捕获与可用KV容量变化 | 不只看参数总和 |

共享embedding/output在架构中只数一次，但loader/worker可能分开存储；KV临时节点与已提交路径也不能都按完整context重算或都忽略。先核真实tensor布局与内存峰值，再估算模型能放多少请求。

## 8. 同负载A/B验收：一个最小实验应该记录什么？

baseline关闭投机，实验组开启最小MTP深度，固定同一target checkpoint、dtype、tokenizer、采样策略、长度与并发。先做greedy一致性/状态测试，再做随机采样与任务质量，最后计时。相同seed文本完全相同不是所有stochastic后端的保证，见[算法篇](#q=infra-mtp-speculative)。

warmup后分别测prefill与decode；既报每请求ITL/P50/P95，也报总有效输出tokens/s与达标请求数。drafted或verified tokens不是用户可见tokens，重复尝试不能加到有效吞吐。记录平均连续接受长度、各深度条件接受、拒绝位置分布、round次数、main/draft/verify耗时、KV/显存和请求长度分布。

出现OOM先看MTP权重加载、graph reserve与candidate budget，再看KV；接受率低先核main/MTP版本、tokenizer、prompt模板、norm状态和sampling q，再考虑深度；接受率高但速度低检查验证query数、输出head成本、并行通信和scheduler。因果顺序比“把steps调大”更可靠。

[AMD官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/v16.0/notebooks/inference/mtp.html)展示了MTP与SGLang在具体平台的服务路径，作为读profile和工作流的补充；不将其结果与H200或Meta7B直接排名。本文未运行GPU A/B，CPU只验证算术与状态示例。

## 9. 面试问答

**Q01 [权重] main能生成为什么MTP还会失败？** 主网络加载完整不代表额外层存在；导出/量化常可能遗漏MTP。查config、键、scale与load日志。

**Q02 [计数] D=1为什么num_speculative_tokens能设3？** 服务循环复用已有模块多步，不增加训练层；后面接受与计算成本要测。

**Q03 [对比] SGLang用EAGLE说明模型就是EAGLE训练吗？** 不说明，它借用执行/验证工作流；候选模块训练目标仍取决checkpoint。

**Q04 [资源] MTP只有2.4B激活，能按2.4B算存储吗？** 不能，存储看总unique参数与共享/副本，激活参数是每步计算口径。

**Q05 [指标] mean_acceptance_length与acceptance rate怎样换？** 在固定gamma且每round有一个额外产出的条件下，mean=1+gamma×aggregate acceptance；变量长度、EOS与无效draft要按实际计数。

**Q06 [性能] 接受率90%为什么还慢？** 节省的target轮数必须抵过draft、验证多位置、通信、调度和内存开销；高batch常需要重新测break-even。

**Q07 [训练] MTP加速rollout是否可直接复用draft logprob？** 不可。行为分布应对应最终target采样，API还可能返回raw/processed不同概率，需核数据契约。

**Q08 [排障] 首先看哪个参数？** 首先确认权重与语义正确，然后最小深度做同负载baseline；把shape预算、接受与分段耗时联系起来，不先盲目加steps或top-k。

## 10. 来源、练习与更新

核心实现固定vLLM/SGLang及官方DeepSeek权重说明。本文代码入口是静态核读，未启动模型或GPU服务；[CPU脚本](./assets/infra/mtp-lab.py)与[结果](./assets/infra/mtp-lab-results.json)只验证指标、假设speedup和logit容量算术。图为本文成本示意。

[来源清单](./research/infra-mtp-serving-sources.json)含固定revision、文档hash和核读用途，选定官方入口进入后续来源观察。更新日期2026-10-09。继续从[系列总览](#q=infra-mtp-overview)检查自己能否分开训练收益、candidate质量、验证正确性和服务成本。
