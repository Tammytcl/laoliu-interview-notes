---
id: paper-id
title: "Full English Paper Title"
paper_title: "Full English Paper Title"
authors: ["Full Author Name"]
affiliations: ["Institution Name in English"]
author_affiliations: [[1]]
venue: "Conference / Journal / Preprint"
year: 2026
direction: llm
areas: [language]
tasks: [text-generation]
published: 2026-09-30
method_figure: null
method_caption: "完成方法图核对后填写"
paper_url: "https://arxiv.org/abs/0000.00000v1"
github_url: null
code_note: "未发现代码时保留 null；找到后说明官方实现、框架或第三方复现。"
evidence: 资料整理
note_ids: []
tags: [Method, Task]
updated: 2026-09-30
summary: "用一段准确陈述概括研究问题、关键机制与适用范围。"
template_version: 5
depth_standard: ddpm
draft: true
---

## 1. 背景与已有工作

写作前打开 [DDPM 精读范例](../content/papers/diffusion/paper-ddpm.md)：本模板的五个标题只是目录，不是完成标准。发布前应达到它的解释密度、原图覆盖与实验可追溯程度，并通过 `npm run papers:quality`。

写成连贯的解释：实际任务和瓶颈是什么？为什么需要解决？此前谁发现过、怎样处理？代表工作的具体差异与出处是什么？本文补上了哪个尚未解决的环节？解释必要概念，不使用只有一句话的提纲代替报告。

可用一张方法谱系表比较前作的目标、设计、代价和剩余限制。相关论文使用原始引用；已有收录记录使用稳定 ID 链接。

## 2. 方法与实现机制

沿输入 → 关键机制 → 训练目标 / 系统执行 → 输出组织，不需要照搬原论文 section。说明重要变量、张量形状、数据流、算法选择及为什么有效。必要时使用 $H$ 等行内公式，或独立公式：

$$
O=\operatorname{softmax}(QK^{\mathsf T}/\sqrt d)V.
$$

公式要有变量解释与适用条件；伪代码标明是原文算法还是教学重述。插入真正需要解释机制的原图，使用以下格式（完成图表提取后再填写真实文件路径）：

```markdown
![Figure N · 简洁的中文说明](./assets/papers/paper-id/figure-N.png)

**Figure N 解读。** 图在回答什么？模块与连线 / 横纵轴表示什么？
观察支持哪项设计？哪些结论不能仅凭该图推出？[图源](固定版本原文#图编号)。
```

### 核心源码与方法对照

开源时固定官方仓库 commit 和文件永久链接。选择 2–4 个真正实现核心思想的函数，沿调用链用段落解释：谁调用它、输入输出和张量形状、关键分支或状态、对应的公式 / 图、为什么这样写。可添加简短教学伪代码，但不能用它代替真实源码阅读。注明历史版本、现代实现或第三方复现的区别；仅静态阅读时不能声称已运行复现。未开源则明确说明，并用原文算法解释。

## 3. 实验设置与算力

| 项目 | 原文配置 | 出处 / 披露状态 |
| --- | --- | --- |
| 模型 / checkpoint / 参数规模 | | |
| 训练阶段与训练数据 / 数据快照 | | |
| 评测 benchmark / split / 样本数 | | |
| 指标 / 生成或解码设置 | | |
| 优化器 / LR / batch / steps / seed | | |
| 精度 / sequence length / 并行配置 | | |
| GPU / TPU 型号、数量、显存 | | |
| 墙钟时间 / GPU-hours / chip-days | | |
| Baselines 与预算是否可比 | | |
| 代码 commit / 模型 revision | | |

明确区分预训练、微调、prompt 示例、评测数据与系统负载。系统论文可以“未训练新模型”；prompting 实验可以“使用冻结模型”。原文未披露时写明缺口与影响，不把常见配置当成事实，不将 TPU 随意换算为 GPU，不声称最低复现算力。

## 4. 结果与图表解读

把主要结果与实验口径一起解释：相对哪个 baseline、哪个模型、哪个任务 / 负载、哪个指标取得什么效果？数字可追溯到具体表格。单位、dev/test、百分比 / 百分点、best / mean 都要核对。

插入关键结果图、消融图与原表截图；每张图写明比较问题、轴 / 图例、观察、所支持结论和边界。多子图不能只截第一幅。若有很多补充图，可用图表清单解释剩余部分，但必须明确本轮已解读 / 待补范围。

| 图表 | 研究问题 | 核心观察 | 解释位置 / 待补原因 |
| --- | --- | --- | --- |
| Figure / Table N | | | |

## 5. 局限、结论与后续阅读

总结得到证据支持的结论，区分作者实测与整理者判断。讨论失败条件、外推限制、缺少的对照、工程成本。连接相关论文、笔记或专题，并给出下一步值得核对的问题。

结尾记录原文版本、核读范围、插图出处、更新日期与具体修改。复现记录只有真正执行后才能填写；正文不使用“待填”占位作为发布完成状态。

### 参考讲解与代码出处

记录实际阅读的作者讲解、技术文章与固定 commit 的核心文件链接。说明吸收了什么解释角度；实验数字仍以原论文为准。参考观点融入正文，不追加多个摘要拼接段。
