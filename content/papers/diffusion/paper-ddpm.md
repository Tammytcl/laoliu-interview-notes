---
id: paper-ddpm
title: "Denoising Diffusion Probabilistic Models"
paper_title: "Denoising Diffusion Probabilistic Models"
authors: ["Jonathan Ho", "Ajay Jain", "Pieter Abbeel"]
affiliations: ["UC Berkeley"]
author_affiliations: [[1], [1], [1]]
venue: "NeurIPS 2020"
year: 2020
research_categories: [generative-modeling]
method_figure: "./assets/papers/paper-ddpm/figure-2.png"
method_caption: "Figure 2 · 正向加噪与反向生成"
direction: diffusion
paper_url: "https://arxiv.org/abs/2006.11239v2"
github_url: "https://github.com/hojonathanho/diffusion"
code_note: "Official TensorFlow implementation; the original experiments used TPU v3-8."
evidence: 已核原文
note_ids: [ddpm-denoising, diffusion-parameterization]
tags: [Diffusion, DDPM, Generative Modeling, Noise Prediction, U-Net]
updated: 2026-09-30
summary: "把逐步加噪的扩散过程与可学习的反向去噪链结合，通过噪声预测目标实现高质量图像生成。"
template_version: 3
draft: false
---

## 1. 背景与已有工作

读这篇论文，先把“生成”理解为学习一个抽样规则：给模型随机数，希望它输出像训练集、却不是简单复制训练图片的新样本。我们通常只拿得到有限张真实图片，不知道所有可能图片的概率分布，因此不能直接从这个未知分布抽样。DDPM 的办法是人为建立一条容易计算的桥：真实图片逐步变成接近标准高斯的噪声；桥的终点容易抽样，学习好反方向的每一步后，就能从随机噪声走回图像分布。

这里“去噪”也需要准确理解。一张严重受污染的图可以对应许多干净图，网络并不知道唯一正确的原图；它学习的是给定噪声状态后，往更清晰的数据分布移动的统计规律。训练时知道加进去的噪声，所以有监督目标；生成时没有原图，需要把学习到的规律反复应用。理解这个训练与生成的信息差，才能理解后面为什么训练只抽一个时间步，而生成需要循环很多步。

图像生成模型既要覆盖复杂数据分布，又要能稳定训练、产生清晰样本。GAN 已能生成高质量图片，但优化是对抗式的；自回归模型与 flow 能提供不同的似然 / 结构优势，也各有生成和架构限制。DDPM 研究另一条路线：先定义把真实数据逐渐扰动成简单高斯分布的过程，再学习逐步逆转这些扰动。

这个方向不是 2020 年才出现。Sohl-Dickstein 等人的 [Deep Unsupervised Learning using Nonequilibrium Thermodynamics（2015）](https://arxiv.org/abs/1503.03585)已经建立扩散式生成与变分训练框架；Song 与 Ermon 的 [Generative Modeling by Estimating Gradients of the Data Distribution（2019）](https://arxiv.org/abs/1907.05600)用多噪声等级 score matching 和退火 Langevin 采样生成图像。本文连接这两类思想，重点证明：适当参数化反向过程与调整训练权重后，扩散模型也能得到高质量样本。

因此，核心问题不是“能不能给图片加噪声”，而是：**怎样定义可训练的反向高斯链，怎样选择网络预测目标，为什么优化噪声预测可以改善最终样本？** 论文的高质量样本证据与其压缩 / 潜变量解释，应分开理解。本文是像素空间无条件生成，不是后来 Stable Diffusion 的潜空间文本条件训练。

## 2. 方法与实现机制

### 正向扩散与可直接采样的训练输入

正向过程固定而不训练。每步将图像缩放后加入少量高斯噪声：

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I),\qquad \bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s).
$$

由高斯递推可以直接得到任意时刻的训练输入，不需要为每张训练图片跑完前面所有步：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

![Figure 2 · 固定正向扩散与可学习反向链的图模型](./assets/papers/paper-ddpm/figure-2.png)

**Figure 2 解读。** 一条方向对应 $q$ 的加噪，反方向对应 $p_\theta$ 的生成。节点是与原图同维度的潜变量，不是缩小到低维空间的 VAE latent。箭头体现 Markov 依赖；训练时可以直接构造 $x_t$，生成时则需要按照反向链逐步执行。[图源：原文 Figure 2](https://arxiv.org/html/2006.11239v2#S2.F2)。

### 反向去噪与噪声预测

网络 $\epsilon_\theta(x_t,t)$ 输入当前噪声图及时间步，预测此次正向构造中的噪声。作者用这一预测参数化反向高斯分布的均值：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

简化训练目标是：

$$
L_{\mathrm{simple}}=\mathbb E_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\|^2\right].
$$

这不是原始变分下界逐项权重完全不变的重写，而是作者选择的简化 / 重加权目标。某些目标可能得到更好的 likelihood，却不一定有更好的 FID；需要用实验判断优化目的。

```python
# 教学重述：训练一步，不包含网络、数据管线和 EMA 实现
x0 = next_batch()
t = uniform_integer(1, T, size=batch_size)
eps = standard_normal_like(x0)
xt = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * eps
loss = mean_squared_error(model(xt, t), eps)
optimizer_step(loss)
# 生成从 x_T ~ N(0, I) 开始，依次执行反向高斯步骤。
```

生成时使用预测均值，加上对应方差的随机噪声，最后一步不再加入新的采样噪声。网络并不是“一次预测就把纯噪声变为完整图片”。[DDPM 基础笔记](#q=ddpm-denoising)和 [预测目标](#q=diffusion-parameterization)可继续推导 noise、$x_0$ 与其他参数化的关系。

### 核心思想：为什么预测噪声能学会生成

网络看到的是混合后的 $x_t$ 和噪声等级 $t$，不能直接读取训练时生成的随机 $\epsilon$。若模型只看 $t$ 或只输出零，就无法解释图中与数据结构相关的偏移。最小化平方误差会让网络学习给定 $x_t,t$ 时噪声的条件平均估计；这个估计又能转换成对干净样本的估计以及反向分布的均值。因此噪声预测是反向生成机制的参数化，不是一项与生成无关的辅助任务。

同一网络覆盖很多噪声等级，需要时间编码告诉它当前是“几乎干净的细节修复”还是“几乎纯噪声的结构形成”。U-Net 的下采样路径提供更大感受野，上采样与跳连保留空间定位。时间条件进入网络，使它在不同 $t$ 下使用不同的去噪规则。每个训练 batch 随机抽样时间步就能训练这些规则；但采样必须把一个状态的输出送入下一个状态，不能把彼此无关的训练样本拼起来当生成轨迹。

### 核心源码：从一次监督到完整采样

官方 TensorFlow 实现的 `GaussianDiffusion.q_sample` 接收 `[B,H,W,C]` 的 `x_start` 和 `[B]` 的时间索引。`_extract` 为 batch 中各样本取出对应系数，再广播到图像维度；返回值恰好对应上面的闭式 $x_t$。源码的 `t=0` 表示已经加噪一步，而论文的干净图写作 $x_0$，阅读数组索引时要错开这一个位置。

`p_losses` 调用 `q_sample` 后把带噪图和时间传给 `denoise_fn`，要求预测与图片同形状。在 `noisepred` 分支中，它比较预测和实际采样噪声，并在空间与通道维度求均值，输出每个样本一个 loss。这解释了为什么训练数据不需要人工标注噪声，也提醒我们变量名 `x_recon` 在该分支其实是噪声预测，不能看到名字就认定它是干净图。

推理沿 `p_sample_loop → p_sample → p_mean_variance` 执行。最后一个函数先由噪声预测反推出干净图估计，可将其裁剪到归一化区间 `[-1,1]`，再借 `q_posterior` 计算反向均值与方差；`p_sample` 加入相应尺度的随机噪声，在索引零的最后一步关闭该随机项。外层循环从标准高斯初始化，按时间倒序更新同一张图。这是带随机性的祖先采样，不能简单把网络调用一次、直接输出它预测的噪声当图片。本文只静态核读这一实现，未重新训练。

## 3. 实验设置与算力

读实验前先明确评价对象：这里生成的是无条件图片，模型没有收到文字提示。CIFAR-10 是低分辨率图像数据集，LSUN 则包含更大尺寸的特定场景图像。FID 比较真实与生成样本在特征空间中的统计距离，通常越低越好，但它依赖特征提取、样本数和预处理；Inception Score 越高通常表示分类预测更明确、类别分布更丰富，却不直接检验与真实分布的距离。似然指标关注数据概率建模，和视觉样本质量不是同一目标。因此不能把不同分辨率或不同取样数量的 FID 放在一起认定谁更好。

设置依据原文 §4 与附录 B；训练数据同时是各无条件生成任务的图像来源，评价以生成样本的分布统计为主。

| 项目 | CIFAR10 | LSUN / CelebA-HQ 256×256 |
| --- | --- | --- |
| 网络 | U-Net / Wide ResNet backbone，35.7M 参数 | 常规 114M；LSUN Bedroom 大模型约 256M |
| 分辨率层级 | 32×32 至 4×4，共四个层级 | 六个分辨率层级 |
| Block | 每层级两个卷积 residual blocks，16×16 处 self-attention | 同类结构、不同宽度 / 层级 |
| 时间条件 | sinusoidal embedding 加入 residual blocks | 同上 |
| Diffusion | 1000 步，线性 beta 从 1e-4 到 0.02 | 同上 |
| 优化器 / LR | Adam，2e-4 | Adam，2e-5 |
| Batch | 128 | 64 |
| Dropout | 0.1 | 0 |
| EMA | decay 0.9999 | decay 0.9999 |
| 数据增强 | 随机水平翻转 | 除 LSUN Bedroom 外使用水平翻转 |

CIFAR10 和 CelebA-HQ 来自 TensorFlow Datasets，LSUN 按 StyleGAN 数据准备方式处理。作者先主要在 CIFAR10 上选择超参，再迁移设置到其他数据集；不能说每个数据集都进行了同样规模的独立 sweep。

| 训练 / 采样成本 | 原文披露 |
| --- | --- |
| 硬件 | 所有实验使用 TPU v3-8；作者将它粗略类比 8 V100，但并非实际 GPU 实验 |
| CIFAR10 训练 | 21 steps/s，800k steps，约 10.6 小时 |
| CIFAR10 采样 | batch 256 图片约 17 秒 |
| 256×256 常规模型 | 2.2 steps/s；batch 128 采样约 300 秒 |
| 256×256 训练步数 | CelebA-HQ 0.5M；LSUN Bedroom 2.4M；Cat 1.8M；Church 1.2M |
| Bedroom 大模型 | 1.15M steps；不能将常规模型吞吐未经验证套用到大模型 |

论文没有给出今天消费级 GPU 的最低复现卡数。上述 TPU 时间只在原配置下成立；逐步生成的采样成本也不能与后来的 DDIM、蒸馏或 latent diffusion 混用。

主要指标为 FID（越低越好）、Inception Score（越高越好）、NLL bits/dim（越低越好）。CIFAR10 的 FID / IS 基于 50,000 个生成样本，LSUN FID 也基于 50,000 个；评价实现分别来自当时指定仓库。作者报告训练过程中最低 FID 对应的结果，最终实验训练一次，没有把多 seed 均值与方差完整报告出来。复现应记录同样的特征提取器、参考统计与模型选择口径。

## 4. 结果与图表解读

![Table 1 · CIFAR10 的样本质量与似然比较](./assets/papers/paper-ddpm/table-1.png)

**Table 1 解读。** 本文 $L_{simple}$ 模型达到 IS 9.46、FID 3.17，是论文重点的无条件 CIFAR10 样本质量结果。各行同时列不同类型生成模型，需要分别看 FID、IS 与 NLL，不能把“某项最优”说成所有目标都最好。尤其本文强调样本质量，likelihood 并非全面领先。[表源：原文 Table 1](https://arxiv.org/html/2006.11239v2#S4.T1)。

![Table 2 · 反向均值参数化与目标函数消融](./assets/papers/paper-ddpm/table-2.png)

**Table 2 解读。** 行列组合对照均值 / noise 预测参数化与训练目标。它回答的不是“加噪步数越多越好”，而是网络预测什么、训练给各项怎样的权重影响生成。空白格来自训练不稳定、样本分数超出范围，不能当作零分，也不能只挑最佳格忽略失败。该实验支撑选择 noise prediction 与简化目标。[表源：原文 Table 2](https://arxiv.org/html/2006.11239v2#S4.T2)。

![Figure 5 · CIFAR10 的 rate-distortion 与反向过程时间](./assets/papers/paper-ddpm/figure-5.png)

**Figure 5 解读。** 曲线把变分项解释为随反向过程逐步补充信息的代价，distortion 使用 [0,255] 图像尺度上的 RMSE。它展示高层结构与细节在信息量和视觉误差上的不同作用；它不是实际文件压缩器的端到端 benchmark。原文明确 compression 只是 proof of concept，所需高维随机编码过程并不实用。[图源](https://arxiv.org/html/2006.11239v2#S4.F5)。

![Figure 6 · 从中间噪声状态预测的图像逐渐细化](./assets/papers/paper-ddpm/figure-6.png)

**Figure 6 解读。** 从左到右观察估计的 $\hat x_0$，先形成粗略结构，再补细节。展示的是不同噪声阶段下的原图估计，不能误认成所有格子都是直接显示原始 $x_t$。这张可视化支持 coarse-to-fine 的解释，但不单独证明 FID 改善。[图源](https://arxiv.org/html/2006.11239v2#S4.F6)。

在 LSUN，常规 Bedroom FID 为 6.36，大模型为 4.90，Church 为 7.89，Cat 为 19.75；不同类别的效果差异说明“能生成高质量图像”不等于所有数据集都优于所有 GAN。此轮重点解释 Figure 2/5/6 和 Tables 1/2，其余样本、插值及邻居检查图保留为后续核读范围。

## 5. 局限、结论与后续阅读

DDPM 的贡献是明确连接反向高斯链、噪声预测和去噪 score matching，并用样本质量实验说明这一路线可行。它的采样需要多次网络执行；优化 likelihood、感知质量与采样时间并非同一个目标。像素空间的无条件结果也不能直接外推到文本生成、条件编辑或今天的大规模图文训练。

下一步可对照 DDIM 怎样改变采样路径、latent diffusion 怎样降低状态维度、不同 prediction targets 怎样改变 loss 与数值行为。若做自己的小规模验证，先测试加噪公式、时间索引和采样最后一步，再记录实际模型、训练 / 测试划分、seed、FID 实现和硬件，避免只看一组漂亮样本。

**来源与更新。** 核对 [arXiv v2 正文与附录 B](https://arxiv.org/html/2006.11239v2)、原图表与 [作者代码](https://github.com/hojonathanho/diffusion)。2026-09-30 更新为五模块报告，补充网络配置、训练 / 采样算力与原图解读。图表与原论文成果归作者；本报告没有登记个人复现成绩。

### 参考讲解与源码版本

本报告参考 [Niels Rogge 与 Kashif Rasul 的 The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)，吸收训练一步与完整采样分开的讲解顺序；其 PyTorch 教学实现并非原论文全部配置。解释已融入问题与方法部分；数字、图表和实验口径回到固定版本原文核对。

源码静态核读固定于 `1e0dceb3b3495bbe19116a5e1b3596cd0706c543`。核心文件：[diffusion_tf/diffusion_utils.py](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py)。没有执行代码或重新训练。
