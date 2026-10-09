---
id: "infra-mtp-speculative"
title: "MTP 投机解码：草稿怎样验证，为什么仍能保持目标分布？"
category: "systems"
difficulty: "深入"
tags: ["P1", "MTP", "Multi-Token Prediction", "Speculative Decoding", "Infra"]
updated: "2026-10-09"
summary: "从greedy前缀验证到p/q接受与残差采样，解释bonus、KV回滚和proposal分布。"
draft: false
---

## 1. 问题背景：草稿不是输出，验证也不只是打分

MTP模块可以便宜地提出未来token，但它不是定义最终输出分布的target。如果直接把多个head的argmax或sample拼成答案，会改变模型行为。投机解码的目标是在合适条件下减少target串行调用，同时保留target原本的greedy结果或采样分布。

本篇只讲与MTP直接相关的draft→verify→accept/recover→commit。先推导线性草稿链，之后解释候选树与工程实现的额外约束。不能把“有target验证”当成任何接受阈值都保真，也不能把头预测准确率等同于完整连续前缀接受。

## 2. 前置知识：p、q、条件前缀与两种解码口径

p是target在**当前已经接受的前缀**下的动作分布；q是proposal真实生成token的分布。如果draft从head softmax随机采样，q可与其采样处理后的分布对应；如果draft取argmax，q是该token上的delta分布，不是原softmax。top-k、top-p、temperature或candidate选择改变q，必须一起记录。

greedy以target argmax作为确定性规则；stochastic decoding希望每步符合target采样分布。相同分布不保证相同seed下逐字相同，随机数消费顺序和浮点kernel都可能不同；greedy相等性测试、随机分布统计和模型质量评测要分别进行。

线性长度gamma表示本轮草稿token数，最多再输出一个补偿或bonus，所以常见上界gamma+1。Meta的“4 heads”包含主head，三个额外候选；DeepSeek的训练深度D只数辅助模块。各后端参数又可能计root或验证节点，不能统一按变量名解释。

## 3. Greedy验证：第一次不匹配后必须停

假设draft为[A,X,C]。target在prefix下的下一token是A，接受；在prefix+A下应输出B，与X不匹配。因此提交A和target纠正B，丢弃剩余草稿。C是在包含X的错误前缀下提出或验证，不能跳过X后接着当成prefix+A+B的合法续写。

![本文示意 · 输出、已完成KV与pending纠正token](./assets/infra/infra-mtp-speculative/verification-state.svg)

**图解。** 第一排显示候选、验证结果与真正输出，第二排分别保留A的有效KV、撤销X/C对应状态、让B进入下一轮计算。纠正token已经对用户可见，不意味着它的KV已经由本轮verification forward生成；验证输入中对应的是X而非B，B可能还是pending input。图是简化状态例，实际target与draft cache位置由后端契约管理。

**vLLM固定源码节选：** [greedy kernel](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/vllm/v1/sample/rejection_sampler.py#L767-L773)的普通分支：

```python title="vLLM真实节选 · greedy验证的首个拒绝"
token_id = target_argmax_id
rejected = draft_token_id != target_argmax_id
tl.store(
    output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
    token_id,
)
```

此段在`if not rejected`循环内，节选省略前面的synthetic test分支；普通模式写入target token，不匹配后不再继续写后续草稿。所有候选都匹配时，末尾再写bonus token。它是Triton内核片段，不能当作纯Python函数单独运行。

## 4. 随机采样：接受概率与残差为什么缺一不可？

经典[speculative decoding](https://arxiv.org/abs/2211.17192v2)在同一前缀提出y~q，接受概率：

$$
a(y)=\min\left(1,\frac{p(y)}{q(y)}\right).
$$

实际被接受的概率质量是$q(y)a(y)=\min(p(y),q(y))$。draft过度提供的质量被拒绝，target仍欠的质量来自$(p-q)_+$。拒绝之后必须用归一化残差采样：

$$
r(v)=\frac{\max(p(v)-q(v),0)}{\sum_u\max(p(u)-q(u),0)}.
$$

设$\alpha=\sum_v\min(p(v),q(v))$，最终单步质量为：

$$
\Pr(Y=v)=\min(p(v),q(v))+(1-\alpha)r(v)=p(v).
$$

这就是精确保留target分布的原因。若p=q，接受率1，残差分母0的分支不会实际发生；代码仍应避免不必要的除零。提出的token若q(y)=0，是proposal/probability契约矛盾，不能继续按正常概率比解释。

### 三token词表的手算

令p=[0.2,0.5,0.3]，q=[0.5,0.3,0.2]。接受质量是[0.2,0.3,0.2]，总接受率0.7；残差为[0,2/3,1/3]。拒绝率0.3乘残差补回[0,0.2,0.1]，最终正好p。

如果拒绝后直接从p重采，最终变成[0.26,0.45,0.29]，虽然没有跳过target，却仍有偏。这个反例很适合讲课：**有验证、有补采，并不自动等于分布校正正确。**

```python title="教学重述 · 单个固定前缀的精确接受与补偿"
y = sample(actual_proposal_q)
if uniform_0_1() <= min(1, p[y] / q[y]):
    output = y
else:
    residual = maximum(p - q, 0)
    output = sample(residual / residual.sum())
```

q必须是actual proposal，不能先greedy取y，再把head softmax数值假装成它的生成概率。对greedy draft的delta q，拒绝后的残差就是把该候选token的target质量去掉，再在其余token间按target比例抽样；这仍可与随机target结合。

### 与真实残差kernel对照

[vLLM residual kernel](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/vllm/v1/sample/rejection_sampler.py#L1051-L1061)在具有draft probabilities的分支计算：

```python title="vLLM真实节选 · 剩余概率质量"
draft_prob = tl.load(
    draft_probs_ptr + token_idx * vocab_size + vocab_offset,
    mask=vocab_mask,
    other=0.0,
)
target_prob = tl.load(
    target_probs_ptr + token_idx * vocab_size + vocab_offset,
    mask=vocab_mask,
    other=0.0,
)
prob = tl.maximum(target_prob - draft_prob, 0.0)
```

后续使用exponential-race/Gumbel类选择机制，不必显式物化归一化概率；“没有除sum”不是忘记概率校正，因为统一正比例缩放不会改变该采样选择。源码NO_DRAFT_PROBS分支则对应deterministic proposal，不能将两者混成同一softmax q。采样逻辑只有在其输入p/q与真实动作分布一致时才成立。

## 5. 多步与bonus：保真的对象是每个已接受前缀

target可以对prefix+全部draft做一次因果verification，从而得到每个draft位置及末尾的条件分布。计算位置可以并行，但每个位置只看其之前的draft路径，不是所有token共享一个p。逐位置按顺序验证，首次拒绝后用该处残差补偿并停止本轮，避免使用错误prefix下的后续概率。

全接受时从末尾target分布取bonus。拒绝时补偿token承担本轮至少推进一位的作用；没有草稿被接受时也不必空转。EOS、最大输出长度或约束终止可能缩短实际产出，不能把gamma+1上界当作每轮保证。

对Meta并行heads，q_j可以只依赖原prefix而缺少已拟议中间token信息；它仍能成为proposal，但更可能与target条件分布不匹配。DeepSeek串行模块显式条件化中间token，有望提高候选质量，是否改善连续接受长度需实测而非只比较head CE。

## 6. 候选树与KV状态：不要把节点数当路径长度

Medusa/EAGLE类实现可能为多个位置各取top candidates，组成树并以tree attention同时验证。branch factor=2、depth=2时可以有2+4=6个非root节点，但最终接受的是其中一条最长合法路径，长度最多2，而不是全部6个节点。

tree mask必须让一个节点只看真正祖先，不能看到兄弟token；普通一维causal mask未必表达这个关系。prefix cache、position IDs、target槽位、draft槽位和accepted indices需要一致选择，未被采用的支路状态必须回收。复制所有候选KV后只删输出token会污染下一轮。

经典线性p/q证明不能自动覆盖任意tree candidate selection、typical acceptance或放松阈值策略。要保真，必须按实际proposal与选择过程构造对应校正；若接受规则有意近似，应单独记录质量/分布变化，不能沿用“speculative天然lossless”的口号。[Medusa v3](https://arxiv.org/abs/2401.10774v3)与[EAGLE v3](https://arxiv.org/abs/2401.15077v3)有各自的训练和验证定义。

## 7. RL rollout的logprob应该是谁的？

如果采样器精确保留target的processed distribution，那么最终行为对应target p，而不是便宜MTP draft q。PPO/GRPO/OPD所记录的behavior概率应与最终真正采样和接受的分布契约一致，不能把被接受草稿的head logprob直接当作actor logprob。

[vLLM accepted logprob路径](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/vllm/v1/sample/rejection_sampler.py#L254-L264)从对应位置的target logits取accepted tokens分数。但API可能返回raw-model或经某种处理的logprob，temperature、top-p、penalty、grammar等又影响实际sampling distribution；是否用于训练必须逐项核对，不能只看字段名logprob。

浮点精度、batch kernel与随机数顺序也影响重复运行。[vLLM保真说明](https://github.com/vllm-project/vllm/blob/18c19eaf239f09bc2543637c73f3b2d05f9bf5c5/docs/features/speculative_decoding/README.md)区分理论分布、算法测试与logprob稳定性。长程agent还要在checkpoint更新后使main/MTP/draft状态属于同一版本，MTP不自动解决异步staleness。

## 8. 面试问答

**Q01 [基础] MTP预测四个token为什么仍要验证？** 候选来自辅助预测，不等于target条件分布的真实输出。验证保留目标行为，成本也必须计入。

**Q02 [算法] greedy第一次拒绝后为何不能接着检查C？** C的前缀包含已拒绝的X，target在prefix+A+B下可能有不同概率；要重新起草或验证。

**Q03 [推导] p/q接受后为什么还需要残差？** 接受质量是min(p,q)，欠的质量是(p−q)+。直接从p重采会重复已经接受过的质量，三token例子给出明确偏差。

**Q04 [概率] draft取argmax时q是什么？** delta分布，而不是网络原softmax；NO_DRAFT_PROBS等实现分支常处理这一情况。追问：greedy draft能否配随机target？可以，用相应校正。

**Q05 [状态] 输出B之后它的KV一定已经存在吗？** 不一定。verification输入可能是X，纠正B只是由logits选出，可能作为pending token在下一次forward生成KV。

**Q06 [结构] tree节点越多是否接受token越多？** 更多候选可提高找到路径的机会，但产出沿单路径，验证计算与临时KV按节点增长，不能把节点数当接受长度。

**Q07 [复现] 同seed不同文本说明算法不保真吗？** 不自动说明；分布等价不要求随机数流相同，浮点/批量计算也可能改变结果。用greedy equality与随机统计分开验证。

**Q08 [RL] 用MTP head logprob训练actor有什么问题？** head q并非最终behavior p；即使candidate被接受，也不能混换概率。还要明确API raw与processed logprob的区别。

## 9. CPU验证与来源

[CPU脚本](./assets/infra/mtp-lab.py)在固定prefix的三token词表上解析核对残差，并用100000次采样得到约[0.19884,0.49939,0.30177]，接近target[0.2,0.5,0.3]；朴素拒绝后从p重采的解析结果确实不同。[已运行结果](./assets/infra/mtp-lab-results.json)同时记录delta proposal、接受计数和pending KV示例。这不是对真实tree kernel、GPU性能或模型质量的验证。

来源均固定到论文版本或代码commit；[登记清单](./research/infra-mtp-speculative-sources.json)保留出处与用途。图为本文示意，不是论文实测。调研2026-10-09。下一篇[MTP服务落地](#q=infra-mtp-serving)将算法条件变成checkpoint检查、参数与性能验收。
