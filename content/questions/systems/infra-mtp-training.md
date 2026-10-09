---
id: "infra-mtp-training"
title: "MTP 训练：Meta 多头与 DeepSeek 串行模块有什么区别？"
category: "systems"
difficulty: "深入"
tags: ["P1", "MTP", "Multi-Token Prediction", "Speculative Decoding", "Infra"]
updated: "2026-10-09"
summary: "推导未来标签、因果条件、共享参数与逐头反传，核原图和消融并对照关键源码。"
draft: false
---

## 1. 问题背景：未来监督为什么可能改变主干表示？

普通NTP在位置i直接预测下一token。若很多局部过渡很容易，损失可能主要奖励短距离模式；对于真正决定后续程序、推理或语义的选择，增加几个未来目标可能迫使同一个前缀表示保留更远信息。MTP研究的是这种训练信号与表示能力的关系，而不是已经保证模型能“预知未来”。

Meta的方案从头训练共享trunk与独立未来heads；DeepSeek-V3在主目标之外加入串行辅助模块。两者的条件信息、参数组织、预算比较和loss权重不同。这里先定义读懂它们的知识，再沿标签、计算图与原图解释方法，最后核对质量和成本证据。

## 2. 前置知识：位置、因果条件、标签和梯度

设输入$x_0\ldots x_{T-1}$，主干$h_i=f(x_{\le i})$。teacher forcing让整段所有位置一次并行计算，但h_i只允许访问历史。offset为j的未来目标是$x_{i+j}$；目标越远，有效位置越少。未来目标不等于把真实未来塞进主干，因果attention和标签mask需要分别检查。

logits形状通常为[B,T,V]，hidden为[B,T,H]；V常远大于H，因此多个head同时物化词表logits可能很贵。交叉熵将目标token索引与对应logits比较，padding、超出序列尾部或跨文档目标要按数据契约屏蔽。一个head没有任何有效标签时，不能对空集合mean得到NaN。

参数共享与表示共享也不同：多个head可以共享unembedding矩阵，其参数梯度来自多个head；同一个h接多个loss，trunk梯度是这些路径的和。detach切断计算图，不是无条件节省显存的等价操作；只有收集到同样的hidden梯度再接回trunk，才可能保留原目标梯度。

先修：[训练一步与反传](#q=infra-training-step)、[FSDP等训练后端](#q=infra-training-backends)。这里不展开完整分布式通信教程，只在逐头backward影响通信重叠时回到相应基础。

## 3. Meta并行头：同一个前缀，多种未来边际目标

[Meta MTP v1](https://arxiv.org/abs/2404.19737v1)使用共享trunk、n个独立head与共享unembedding。每个head包含额外Transformer层，而不只是独立Linear分类器。h_i相同，head j预测$x_{i+j}$；n=4包含普通下一token head和三个更远head。

$$
q_j(x_{i+j}\mid x_{\le i})=\operatorname{softmax}(W_u f_j(h_i)),\qquad
L=\sum_{j=1}^{n}w_j\sum_i m_{i,j}\operatorname{CE}(q_j,x_{i+j}).
$$

这里m与w是便于说明实际mask和权重的写法，具体reduction需按训练配方固定。原论文直接将多未来CE相加；改成各head独立mean、按有效token整体mean或调不同w都会改变有限batch的尺度。教学脚本选择各head有效tokenmean，仅在两种调度中保持同一选择以验证梯度等价。

独立预测的是给定同一prefix的边际分布，第二head没有显式接收第一head的候选token。因此未来之间仍可能缺少条件一致性；它增加监督范围，不自动把多个输出变成正确joint continuation。普通推理可以仅用第一个head，其他head也可在[投机篇](#q=infra-mtp-speculative)中提供候选。

## 4. DeepSeek串行模块：每层增加一位已条件化的token

[DeepSeek-V3 v2 §2.2](https://arxiv.org/abs/2412.19437v2)将D定义为额外MTP深度，和Meta包含主head的n计数不同。主模型h_i^0预测$x_{i+1}$；第k个模块融合此前深度表示与$x_{i+k}$的embedding，再预测$x_{i+k+1}$。

$$
h_i^{\prime k}=M_k[\operatorname{RMSNorm}(h_i^{k-1});\operatorname{RMSNorm}(E(x_{i+k}))],\quad
h^k=\operatorname{TRM}_k(h^{\prime k}),\quad
q^k_{i+k+1}=\operatorname{OutHead}(h_i^k).
$$

![DeepSeek-V3 Figure 3 · 逐深度融合与共享embedding输出](./assets/infra/infra-mtp-training/deepseek-mtp-source.png)

**原图解读。** 左边main预测t2,t3…；中间MTP1输入提前一位的token embedding与main hidden，预测再后一位；右边MTP2继续递推。绿色虚线表示embedding与output共享，黄色projection/block属于辅助模块。图展示一般D=2链，最终V3训练设置实际D=1，不能看图就说发布模型有两个MTP模块。[图源：固定v2 Figure3](https://arxiv.org/pdf/2412.19437v2#page=10)。

### 为什么提前一位真值不是标签泄露？

在位置i，MTP1目标是$x_{i+2}$，输入额外看到$x_{i+1}$是允许的条件，而不是目标本身。它并未让main的$x_{i+1}$预测看到答案。辅助TRM仍需因果mask，任何连接到$x_{i+2}$或更后位置的错误路径才是泄露。

```text title="教学对齐 · 一个位置的因果条件"
主干 h_i:       已读 x[0:i+1]       → 主head目标 x[i+1]
MTP1: h_i + E(x[i+1])              → 额外目标   x[i+2]
MTP2: h_i^1 + E(x[i+2])            → 额外目标   x[i+3]
训练: 额外embedding通常用真值
服务: 用已确认或实际草稿token，不能获得真实未来
```

这也产生teacher-forcing与draft rollout差异：训练输入正确中间token，服务可能输入错误candidate；模块复用多步时误差会累积。接受率必须在真实采样轨迹上测量，不由训练CE直接保证。

### 训练深度与辅助权重

原文最终D=1，即main加一个额外预测；辅助loss为$\lambda/D$乘各额外深度CE平均，再加main loss。前10T训练tokens取lambda0.3，余4.8T取0.1。这是V3原训练配方，不是所有MTP都默认0.3，也不是服务的draft step数。

## 5. 标签与packing：shift不只是切一个数组

两文档打包为[A,B,C,D,E,F]，doc IDs=[0,0,0,1,1,1]。同一prefix的offset1标签为[B,C,忽略,E,F,忽略]；offset2是[C,忽略,忽略,F,忽略,忽略]。offset3没有同文档有效标签。不能让C的未来head把D当成同一文本续写，也不能只mask序列最后一个token。

**本文CPU脚本的标签函数：**

```python title="教学实现 · 按offset生成未来标签"
labels = torch.full_like(tokens, -100)
if offset < tokens.shape[1]:
    valid = documents[:, :-offset] == documents[:, offset:]
    labels[:, :-offset] = torch.where(valid, tokens[:, offset:], -100)
```

这段假设doc IDs在pack内唯一且连续；真实packing还需要block-diagonal causal attention、position IDs和EOS契约，标签屏蔽不代替attention隔离。对于SFT或工具轨迹还需决定未来窗口是否跨response/observation边界，不能只按“都是token”加loss。当前系列不替具体训练框架设定这些数据规则。

## 6. 内存高效反传：逐头释放logits，最后接回主干

![Meta Figure 2 · 逐头反传与hidden梯度汇总](./assets/papers/paper-mtp-meta/figure-2-source.png)

**原图解读。** 左边编号表示trunk先forward，再对head1/head2分别forward/backward，最后把合并hidden gradient传回trunk；右边是作者原图中的Python示意。d是与z数值相同的新leaf，d.grad收集各头贡献；末行z.backward重新连接主干。若省掉末行，训练只更新heads，不能称为等价MTP训练。[图源：固定Meta v1 Figure2](https://arxiv.org/pdf/2404.19737v1#page=3)。

```python title="教学实现 · 与整体loss梯度相同的分阶段调度"
z = model.shared(tokens)                 # [B,T,H], 保留trunk计算图
leaf = z.detach().requires_grad_(True)    # 新leaf，数值不变
for head_index in range(number_of_heads):
    loss = model.head_loss(leaf, labels[head_index])
    loss.backward()                     # 累加leaf.grad及共享输出矩阵grad
z.backward(leaf.grad)                    # trunk只反传一次
# 所有head完成后才optimizer.step；不能每个head都更新参数。
```

该段是教学重述，具体可运行版本在[CPU脚本](./assets/infra/mtp-lab.py)。本轮在float64小模型中核对，整体loss与分阶段方案的最大参数梯度差为约$2.78\times10^{-17}$。它验证所列计算图，不是Meta训练或GPU显存复现；toy head是Linear，真实论文使用Transformer层。

峰值logits从近似O(nBTV)降到O(BTV)，hidden与梯度约O(BTH)。head参数和optimizer states仍存在，共享unembedding梯度也要累加。AMP的loss scaling、FSDP all-gather与梯度同步会改变实际调度，不能机械在每个head清零grad，也不能把已经scaled的leaf.grad再次scale。Meta附录真实训练时间出现1.07—1.22倍开销，作者归因于FSDP通信计算重叠不足。

## 7. 关键源码：符号公式怎样接到服务实现？

以下是维护者vLLM实现，固定commit `18c19eaf239f09bc2543637c73f3b2d05f9bf5c5`，不是DeepSeek原始训练loop。[DeepSeekMultiTokenPredictorLayer.forward](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/vllm/model_executor/models/deepseek_mtp.py#L124-L136)核心节选：

```python title="vLLM真实节选 · 两路norm与projection"
inputs_embeds = self.enorm(inputs_embeds)
previous_hidden_states = self.hnorm(previous_hidden_states)
hidden_states = self.eh_proj(
    torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
)
hidden_states, residual = self.mtp_block(
    positions=positions,
    hidden_states=hidden_states,
    residual=None,
)
hidden_states = residual + hidden_states
```

服务将有效位置打平，输入常为[N,H]而非教学[B,T,H]；concat变[N,2H]，projection回[N,H]，block继续产生新hidden。公式中[h;embedding]与源码[embedding;h]的排列约定不能机械交换，投影权重列必须匹配实际checkpoint。此节选省略原末行注释，未改运算，完整分支见链接。

同一文件后续区分pre-final-norm logits hidden与post-final-norm recycled hidden，避免重复final norm；next draft step使用哪一版hidden属于模型/后端契约。共享embedding/output是架构关系，跨worker是否物理共用仍要看加载与并行放置，见[服务篇](#q=infra-mtp-serving)。

## 8. 质量证据：去掉模块后的收益和保留模块的成本分开

![DeepSeek Table 4 · 两个MoE尺度的MTP消融](./assets/infra/infra-mtp-training/deepseek-table-4-pdf.png)

**原表解读。** 小模型15.7B总参数、2.4B激活，1.33T训练tokens；大模型228.7B总参数、20.9B激活，540B tokens，均额外训练1-depth MTP。推理时移除辅助模块，所以表中相同推理参数量是在衡量主模型质量。HumanEval小尺度20.7→26.8、大尺度44.5→53.7，分别增加6.1与9.2个百分点；大尺度MMLU67.5→66.6、小尺度NQ22.7→22.3也有下降，不能说全部格子胜出。[表源：固定v2 Table4](https://arxiv.org/pdf/2412.19437v2#page=26)。

这个消融不是671B最终模型的MTP开关测速。Meta又使用参数匹配预算：增加未来head层时减少trunk层，比较模型总参数相同。DeepSeek附加模块与Meta参数匹配实验不构成同一种成本对照；训练收益、候选准确与decode倍率需独立记录。

## 9. 面试问答

**Q01 [基础] MTP的标签为什么offset不同？** 同一h_i负责不同距离的$x_{i+j}$，每head可训练位置不同；序列尾、文档边界与SFT mask都需重新对齐。

**Q02 [区别] Meta的n=4与DeepSeek的D=1怎么比较？** n包含主下一token头，D只数额外模块；后者main+一个extra对应两个目标，不能直接说1头对4头。

**Q03 [因果] DeepSeek看到提前真值是否泄露？** 当前模块预测的是更后一位，提前token是条件。要分别检查main与auxiliary的attention路径，不能只看整体输入存在未来token。

**Q04 [梯度] detach后主干为什么还有更新？** 先在leaf上累加所有head梯度，再用z.backward(leaf.grad)接回trunk。只detach而不接回会改目标。

**Q05 [内存] 逐头反传省掉什么？** 同时驻留的词表logits及其局部图，未省掉全部参数、optimizer、trunk激活或分布式通信。

**Q06 [预算] MTP训练一定没有开销吗？** 原附录实际n=4有1.07—1.22倍耗时；不同FSDP/融合/共享实现需实测。loss密度更多不是吞吐免费。

**Q07 [实现] norm或concat顺序可以随便换吗？** 不可以。input/hidden语义、projection列顺序和pre/post-final-norm状态都绑定checkpoint与后端；符号上的等价要配合权重变换。

**Q08 [证据] 去掉辅助模块仍提升为什么重要？** 它隔离主模型表示收益，不依赖推理候选验证。反过来，推理保留模块测速不能替代这种质量消融。

## 10. 来源与课堂实验

原图来自Meta v1源码与DeepSeek v2源码，原表紧裁固定PDF，作者归属保留。官方DeepSeek模型权重结构另见[固定权重说明](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md)；主模型推理示例不等于完整MTP训练实现。未取得Meta访问受限模型源码，本轮不声称核过原训练代码。

[CPU脚本](./assets/infra/mtp-lab.py)、[实际CPU结果](./assets/infra/mtp-lab-results.json)包含packing标签和梯度等价验证；它不验证真实attention隔离、GPU性能或训练收敛。来源/hash见[登记清单](./research/infra-mtp-training-sources.json)。调研日期2026-10-09。继续读[投机解码](#q=infra-mtp-speculative)，把预测结果变成可正确提交的token。
