---
id: survey-post-training-interview
title: "后训练面试专题：清单合并、问题驱动学习与两项目介绍"
type: survey
date: "2026-10-13"
directions: [llm, agent, infra]
paper_ids: [paper-ppo, paper-dpo, paper-grpo, paper-dapo, paper-opd-2609-34738, paper-opd-2607-22334, paper-asyncopd, paper-rethink-mopd]
tags: [专题调研, Interview, Post Training, ESCD, Turn OPD]
updated: "2026-10-13"
summary: "把外部实际222项清单映射到已有/新增专题，保留LoRA和RAG原入口扩写，分别组织后训练答题、ESCD作者讲稿和turn项目代码/创新论证。"
template_version: 1
draft: false
---

## 1. 本轮研究问题

面对后训练面试，怎样从问题补知识并用自己的工作回答，而不只背方法名？用户这轮明确了ESCD是自己的工作、当前Agentic Code OPD使用turn目录。介绍需回答算法目标、训练数据、实际Infra接线、个人贡献和证据；通用知识与项目实现分开。

## 2. 检索与版本

2026-10-13读取用户清单，正文实际222项、29组，URL100/导航152是历史标题。来源作为题型，未找到可确认开放内容license，本站不转载整份原题；逐编号用自写知识标签和入口映射。面经汇总与自述不当公司认证或频次统计。

ESCD用公开commit befd0c0a582ec6efc055660467db95d522519591和repo45页报告；arXiv v1精读继续固定旧图表。Turn工作树含未提交变化，基线HEAD不能当全部实现revision，本轮保存42个源文件hash作为静态证据，不修改项目、不运行GPU。

## 3. 清单按专题合并

[222项地图](#q=post-training-checklist-map)指向已有算法、Infra与代码入口；LoRA/QLoRA和RAG在原ID扩写。缺口补基础分词、Reward/Judge、服务化；作者个人医疗项目参数与成绩不迁移给用户。链接是学习路线，不是用户已掌握状态。

## 4. 问题驱动回答

[40道后训练题](#q=post-training-interview)按数据→偏好→RL→蒸馏→Infra/证据，每题有直答、知识点和追问。[基础/Tokenizer](#q=llm-foundations-interview)、[LoRA](#q=lora-finetuning)、[Reward/Judge](#q=reward-judge-interview)、[RAG](#q=rag-vs-finetuning)、[服务化](#q=model-service-interview)各有16道题。已有PPO/DPO/GRPO精读与RL六篇复用，不重造缩略版。

## 5. ESCD的项目介绍

[作者讲稿](#q=escd-project-interview)含30秒/3分钟/10分钟、原method与Table1、root FKL与child事件NLL、Single分解、集合梯度、teacher hidden/LM head、三种top-k、EOS和公开配置。主任务成绩、COUF局部梯度诊断、visited compatible mass一步覆盖的分母分别说，个人模块分工按真实记录补。

## 6. Turn框架与Infra

[实现专题](#q=agentic-turn-framework)从Turn identity/digest、完整teacher前缀评分、final cohort接受与sealed-turn异步开始，再讲single-forward reference、FP32 weighted gradient、coalescing、bucket、credits/gate与resident export。单forward省计算不意味着旧行为概率IS已实现，receipt去重不保证crash原子optimizer恢复。

## 7. 创新与证伪

[Critic专题](#q=agentic-turn-innovation)说明当前Actor只有加权sampled OPD，task critic只调强度；PPO联合loss和真实监督收益critic是不同备选。Same-version value pair、终局标签、双更新时机、cold start与past-only scale逐项核。正delta促进蒸馏只是待检验假设，需uniform、方向、shuffle/random-sign、frozen/online及同预算质量对照。

## 8. 验证与课堂代码

新增CPU练习检causal MHA、BPE非重叠合并、LoRA初始化/merge、DPO初始梯度、weighted window共同比例抵消、ratio1梯度与critic跨版本伪delta。它不加载模型、不验证真实kernel/CP/TP/云端服务，不给用户项目造benchmark。

来源/公式/链接、原图与SVG、手写代码和手机路由沿现有工作流检查。外部资料有不完整或简化题面时回到primary依据修正，而不是复制错前提。

## 9. 下一次复习

先从地图选与你岗位有关的题，闭卷30秒答→画shape/数据流→手算→读固定代码→补项目证据。ESCD讲公共事件、turn讲局部与终局两条流，最后解释为什么证据支持某个贡献以及什么尚未证实。用户真实个人分工/实验确认后替换具体讲稿中的待填写记录。
