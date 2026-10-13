---
id: llm-foundations-interview
title: "LLM基础与Tokenizer面试：架构、缩放、前缀和训练协议"
category: llm
difficulty: 基础
tags: ["P0", "Transformer", "Tokenizer", "Checklist", "Interview"]
updated: "2026-10-13"
summary: "合并清单中的基础、Transformer与分词知识缺口，建立公式、具体例子与16道追问；已有Attention/RoPE/手撕页面继续作为详细实现入口。"
draft: false
---

## 1. 问题背景 基础题为什么会追到后训练实现

面试问“为什么除sqrt(d)”不只是在考公式。它会连到attention稳定性、混合精度和OOM；“能不能换tokenizer”会连到embedding、概率空间和ESCD；“Decoder-only为什么适合生成”会连到因果mask、response loss与KV。基础对象定义错，后训练ratio/teacher评分也可能错。

本篇补[用户清单](https://tcsnyy.github.io/llm-interview-notes/22_100%E9%97%AEChecklist/)1–28的学习缺口，保留已有[Attention](#q=attention-complexity)、[MHA/GQA/MQA](#q=mha-gqa-mqa)、[RoPE](#q=rope-position)、[因果LM训练](#q=causal-lm-training)和[手写attention](#q=causal-attention-coding)入口。这里不把公开资料未披露的GPT架构当标准答案，也不把“中文必然低效”等题面前提当普遍事实。

## 2. 前置知识 架构、参数和数据不是同一个规模

Parameter count是可存储权重数量，training tokens是数据/重复暴露预算，FLOPs是计算量，context length是单序列范围。MoE total与active参数不同；1T总权重并不意味着每token计算1T。增加参数、数据、上下文和生成次数影响的成本不同。

一个典型decoder block输入$[B,L,d]$，预归一、causal attention、残差、预归一、FFN与残差；不同模型可能换norm、位置编码、激活、attention结构，不能把这一图当所有模型强制实现。输出head给$[B,L,V]$ logits，位置$t$预测$t+1$。

## 3. 架构、Scaling与能力的讲法

Encoder-only常适合表示/理解，encoder–decoder用输入编码与输出解码，decoder-only适合统一因果续写。三者都有适用场景，不用“现在主流”证明其理论最优。多模态decoder可以引入cross-attention或其他融合，不能回答“decoder永远不允许cross-attention”。

Scaling law通常是在某类模型、数据和预算下拟合loss随规模的经验关系，不是任意能力都会平滑增长的定理。固定计算时参数/训练token配比也重要；只增参数可能数据不足，只重复数据可能泛化有限。看[Chinchilla](https://arxiv.org/abs/2203.15556)的计算最优讨论，不能未经验证移植所有常数到不同架构。

GPT公开演进可以讲预训练迁移、扩大数据/参数、few-shot/ICL、指令对齐与后训练，不背未知内部层数、MoE专家或数据配比。所谓涌现还受评测尺度、阈值与metric定义影响；某任务突然过阈值，不证明参数发生神秘能力相变。

ICL在推理上下文提供例子，一般不做optimizer更新；SFT改变参数。CoT改变生成组织与推理预算，可能帮助复杂任务，也可能产生看似合理的错误推理；需要独立正确性验证。Loss下降、benchmark上升与真实部署行为不是同一对象。

## 4. Attention的最小推导

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(QK^\top/\sqrt{d_k}+M\right)V.
$$

Q检索query、K提供匹配keys、V提供被汇总内容。若各维独立、零均值、方差约1，未缩放点积方差约$d_k$，除$\sqrt{d_k}$把量级稳定在约1；这是初始化量级动机，不保证任意训练后分布严格满足假设。

Mask使未来不可见，不能用全零attention weight假装跳过softmax问题。若一行全部masked，需要实现定义有效样本/边界，防止NaN；response loss mask与causal attention mask又不同。Multi-head投影提供多个子空间，但不能预先规定每个head只学语法/实体某一种固定含义。

Pre-Norm先normalize子层输入再残差，Post-Norm在残差后normalize；梯度路径与初始化/深度稳定性不同。RMSNorm按均方幅度缩放、不像LayerNorm减均值；并非所有应用都更优。SwiGLU可写$\mathrm{SiLU}(xW_g)\odot(xW_u)$后投影，门控非线性改变表达和计算，不能只把它叫“换了一个更好激活”。

FlashAttention用分块、在线softmax与IO优化避免显式$L^2$矩阵驻留，精确attention在相应数值范围内实现；不是删token或把复杂度无条件改$O(L)$。PagedAttention管理推理KV物理blocks，解决的是另一层。RoPE数学与长上下文外推看既有文章，不从旋转可计算推导模型能无限泛化。

## 5. 分词、训练样本与特殊协议

BPE先有基础符号，迭代统计相邻pair并合并，最后按学得规则编码；byte-level起点利于覆盖bytes，仍需normalizer和pre-tokenizer。WordPiece有不同的建词/编码规则，不能只根据token前缀符号认算法。SentencePiece是工具/框架，支持BPE和Unigram等，不等于一个与所有BPE互斥的单算法；Hugging Face Tokenizers也可实现多种算法。

Vocabulary大可能缩短序列，却增加embedding/head与softmax成本；小词表增加token数与上下文成本。最佳大小取决语言、数据与模型，不是固定越大越好。中文每字token数与英文每词token数也不是同单位比较；应统计实际语料的chars/bytes/tokens、coverage和任务效果。

Tokenizer通常由normalization、pre-tokenization、模型切分、postprocessing/特殊token、decoder组成。Special tokens既含BOS/EOS/padding，也可能有role/tool/control；是否停止要读generation config/stop sets，而不是所有“看着像结束”的文本都停止。

Chat template把messages转成模型约定token序列，包含role/分隔/可能的thinking prefix。Training与serving模板不一致会改条件分布，OPD比较会混入协议偏差。模型生成的think标记与prompt控制标记分别处理；不要强行给截断response补结束符再当真实行为。

换tokenizer会同时改变ID语义、embedding rows、output head、词频、序列长度和模板。加载旧embedding再只改tokenizer通常不兼容；可做训练迁移/初始化适配/蒸馏，但需明确过程。ESCD保持各模型原生词表，在共同bytes事件上建立监督，没有声称可以随意替换student词表。

## 6. 一个字节对齐例子

Text“中a”有两个Unicode字符、UTF-8四bytes；byte-level token可在字符内部切。两个tokenizer第2个token不一定有相同prefix，单token decode可能出现replacement character，不能把decode后的字符串长度当无损byte offset。

Teacher把120切1/2/0，student一个120时，teacher三个条件logprob相加是一条span路径概率；如何给多个student位置分配credit还需方案，不是索引对应就结束。Root相同prefix、child部分进入事件、EOS停止事件分开，详见[ESCD项目](#q=escd-project-interview)。

建议最小验证：causal mask排未来、RoPE点积恒等式、不同模板token差异、native bytes roundtrip、BPE合并、EOS与padding。它们验证某个契约，不替代真实长上下文模型quality。

## 7. 面试问答16题

### 白板补课：从形状到可运行代码

以B2、L4、d8、H2为例，Q/K/V各[B,H,L,dk]=[2,2,4,4]，scores[B,H,L,L]，mask在后两维广播，attention输出转回[B,L,d]再out projection。Transpose后的reshape需按真实layout处理，不能把内存连续性和语义维度混在一起。除sqrt(dk)的dk是head维度，不是总hidden d；softmax沿key位置，不是batch/head。

Loss与attention再分开：输入ids为[B,L]，logits[:, :-1]预测ids[:,1:]，有效target mask也要shift。如果整个样本无有效target，要skip或定义零贡献，不能无检查调用全ignore mean CE得到NaN。训练多样本packing需阻断跨样本attention，padding即使loss0也可能参加算子和显存。

LoRA改变weight增量，RMSNorm缩幅度，RoPE转Q/K，chat template改变离散输入；这些操作作用对象不同，排错先把所在层画清。KV cache存依赖权重与上下文的中间状态，不能跨不同adapter/version随意共享。

BPE最小合并函数只解决已给定pair的一次替换，完整训练还需词频、候选计数、tie、边界和normalization。合并必须从左到右避免重叠，例如a/a/a合并(a,a)得到aa/a，而不是同时产生两个aa。`merge_count`、vocab size和训练语料预算不同，空白/字节fallback又属于tokenizer其他层。

```python
# Teaching sketch: one non-overlapping merge, not a full tokenizer.
def merge_pair(tokens, pair):
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tuple(tokens[i:i+2]) == pair:
            out.append(tokens[i] + tokens[i+1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out
```

[CPU白板练习](./assets/learning/post-training-project-lab.py)含multi-head causal forward、BPE非重叠merge、DPO梯度、LoRA与加权窗口。[结果](./assets/learning/post-training-project-lab-results.json)显示改变未来输入不影响前面位置，验证的是玩具因果算子，不是模型长期生成能力。实际tokenizer/schema还需独立样例核native bytes、special tokens与模板。

**Q01 大在哪里？** 参数/训练tokens/计算/上下文分别定义，MoE总与active不混。

**Q02 Scaling能解释全部能力吗？** 经验loss规律有配置范围，阈值能力还受metric影响。

**Q03 GPT演进怎么讲？** 讲公开训练范式和能力，不填未披露内部架构常数。

**Q04 为什么decoder-only？** 统一因果生成与训练/推理便利，其他架构仍有用途。

**Q05 ICL等于微调吗？** 通常只换上下文不更新参数；SFT需要梯度step。

**Q06 CoT保证正确吗？** 推理组织与预算可能有帮助，也可能放大错误，需verifier。

**Q07 为什么sqrt(d)？** 点积方差量级在特定初始化假设下随d增长。

**Q08 QKV能删K直接V吗？** 匹配与内容职责不同；修改结构需重新定义权重与学习。

**Q09 Norm哪个普遍更好？** 取决深度、初始化与训练；说清Pre/Post梯度路径。

**Q10 Flash是稀疏attention吗？** 主机制IO与在线softmax，不必删连接。

**Q11 SentencePiece和BPE能并列当互斥算法吗？** 不能，工具可含多种切分算法。

**Q12 中文必比英文token低效吗？** 需同语料/单位/词表统计，不接受无条件题面。

**Q13 Vocab越大越省成本吗？** Token变少与head/embedding变大有取舍。

**Q14 EOS等于padding吗？** 停止事件与布局空位不同；有些配置共ID但mask语义仍分。

**Q15 换template会影响teacher评分吗？** 改了条件和特殊协议，可能造成伪KL/ratio。

**Q16 能只换tokenizer不训吗？** 旧ID/embedding语义失配，需适配或公共事件蒸馏方案。

## 8. 来源与学习入口

[Attention原论文](https://arxiv.org/abs/1706.03762)、[RoFormer](https://arxiv.org/abs/2104.09864)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[SentencePiece](https://arxiv.org/abs/1808.06226)为基础依据；具体模型源码、kernel和API版本按对应Infra专题固定。本篇不登记GPT未知内部配置或benchmark成绩。

顺序：本篇对象 → 手写causal attention → 模板/bytes → [后训练40题](#q=post-training-interview) → ESCD/turn项目。详细的数学/代码在原有页面复用，清单编号到入口的对应见[专题合并地图](#q=post-training-checklist-map)。
