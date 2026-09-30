# 牢留的实习面试笔记 · Laoliu’s Interview Notebook

[在线阅读](https://tammytcl.github.io/laoliu-interview-notes/) · [GitHub 仓库](https://github.com/Tammytcl/laoliu-interview-notes)

一个内容与界面分离的中文面试准备站。浅色纸张风格、主题导航、全文搜索、题目目录、收藏与复习状态。没有后端、数据库、追踪统计或运行时 CDN 依赖，适合 GitHub Pages。

> 当前有 38 篇学习笔记、4 篇基础论文精读示例、1 份 Daily 和 1 份专题调研。论文收录独立于面试笔记；基础报告是资料整理，个人阅读状态初始为“待精读”。旧项目表达题保留在综合面试中。

## 从这里开始学习

- [四条学习路线](content/questions/guide/learning-roadmap.md)：按先修关系阅读，P0 主干 / P1 深入不是面试频率统计。
- [知识与项目架构](content/questions/guide/learning-architecture.md)：针对四方向并行学习设计；缺少个人项目与岗位信息的部分明确保留为建议。
- [原始资料索引](content/questions/guide/research-sources.md)：论文、官方文档与对应笔记。
- [来源记录](content/research/sources.json)：2026-09-30 调研访问的来源和使用位置。

内容通常包含问题背景、机制/推导、例子、误区、小实验、自检及原始资料。实验是建议练习，尚未完成模型复现；正文算例不冒充硬件实测。数学式使用可直接阅读的文本/代码块。

算法只增加一个与模型紧密相关的 causal attention 手撕练习，另保留 LRU；项目论文和综合面试暂不扩量。

## 论文收录：单篇精读 + 两类积累报告

[进入论文收录](https://tammytcl.github.io/laoliu-interview-notes/#view=papers) · [Daily](https://tammytcl.github.io/laoliu-interview-notes/#view=papers&tab=daily) · [专题调研](https://tammytcl.github.io/laoliu-interview-notes/#view=papers&tab=survey)

- **论文库**：按 Diffusion / LLM / Agent / Infra 筛选，一篇论文一条稳定记录，正文是统一格式的精读报告。可按个人阅读状态、收藏和全文检索。
- **Daily 报告**：每天开始工作时手动更新，记录候选筛选、收录理由、主阅读、个人理解变化及下一步。不是自动新论文抓取或定时任务。
- **专题调研**：围绕研究问题整理范围、方法谱系、比较条件、共识/分歧、空白与持续更新。引用论文库，不重复复制单篇全文。

三份可下载的模板：[单篇精读](templates/paper.md)、[Daily](templates/daily-report.md)、[专题调研](templates/survey-report.md)。[整理流程与 Agent 使用约定](docs/paper-workflow.md) 可用作后续整理任务的统一要求。

```bash
npm run new:paper -- paper-my-topic llm
npm run new:daily                       # UTC 当天，已存在则提示继续更新
npm run new:daily -- 2026-10-01          # 也可明确指定报告日期
npm run new:survey -- survey-my-topic infra
```

命令生成 `draft: true` 的草稿。填写原文信息和正文，完成后改为 `false`，运行构建、提交并推送。新建论文/专题拒绝覆盖；同日 Daily 继续维护一个文件。重复 arXiv 论文（包括不同版本）、同日期重复 Daily、未发布或不存在的关联论文会让构建失败。

单篇必需元数据包含 `paper_title`、`authors`、`year`、`direction`、`paper_url`、`evidence`；报告包含 `type: daily/survey`、`date`、`directions`、`paper_ids`。公共字段为 `id/title/summary/tags/updated/draft`，模板版本为 `template_version: 1`。论文用 `#paper=id` 链接、报告用 `#report=id`，学习笔记仍用 `#q=id`。日期和内容状态请按真实情况填写。

**资料核验程度与个人阅读进度分开。** 报告的 `evidence` 为“资料整理 / 已核原文 / 已复现”；个人状态为“待精读 / 阅读中 / 已复述”。Agent 生成报告不自动推进你的阅读进度。页面支持读前问题、合上正文后复述、连接/待验证与复习日期；没有自己的复述文字时不能标记“已复述”。到期论文会在列表中提示。

**个人阅读草稿仅保存在当前浏览器。** 用论文模块中的“导出/导入阅读记录”备份和迁移，与侧栏的面试复习进度备份相互独立；不会自动写回 Markdown。需要长期积累或公开的结论，应整理回论文正文或 Daily。它们保存在 `content/papers/` 和 `content/reports/`，通过 Git 维护。已有的复习记录和旧问题链接保留。

## 内容与排版分开

```text
content/questions/**/*.md  学习笔记
content/papers/**/*.md     按方向存放单篇精读
content/reports/**/*.md    Daily 与专题报告
          ↓ 统一解析 YAML 元数据 + Markdown
scripts/build.mjs         校验字段、渲染 HTML、生成搜索数据与目录
          ↓
dist/data.json            构建产物，不手工编辑
          ↓
web/index.html + style.css + app.js
                          统一布局、视觉、搜索、阅读和复习交互
```

不用为每道题写 HTML，不用维护目录列表，也不用每次新增题目就改 JS。Markdown 标题、列表、代码块、表格和引用采用同一套样式。

页面实际加载带内容指纹的 `data-*.json`、`app-*.js`、`papers-*.js` 和 `style-*.css`。内容变化会生成新地址，避免旧数据缓存导致题目数量未更新；`data.json` 保留给外部读取。GitHub Pages 首页仍可能缓存约十分钟，发布后需要立即查看时可在网址加一个新的 `?v=版本号`，无需清除复习进度。

这里选择**发布时解析**，而不是让每个浏览器重新下载并解析所有 Markdown：访问时读取已经生成的 `data.json` 即可。因此只改 Markdown 后需重新构建；GitHub Actions 会自动完成。

## 本地运行

需要 Node.js 22+。

```bash
git clone https://github.com/Tammytcl/laoliu-interview-notes.git
cd laoliu-interview-notes
npm ci
npm test
npm run build
npm run preview
```

访问 `http://127.0.0.1:4173`。服务器只监听本机；远程机器请通过 SSH 或开发环境端口转发访问，不要直接双击 HTML。修改内容或前端后重新运行 `npm run build` 并刷新；当前预览不是自动热更新。

## 日常：只添加 Markdown

```bash
npm run new -- attention-mask llm
```

命令会创建 `content/questions/llm/attention-mask.md`，同名文件已存在时拒绝覆盖。也可以直接复制 `templates/question.md` 到任意 `content/questions/` 子目录。

```markdown
---
id: attention-mask
title: "Attention mask 有哪些作用？"
category: llm
difficulty: 基础
tags: [Transformer, Attention]
updated: 2026-09-30
summary: "理解 padding mask 和 causal mask 的区别。"
draft: false
---

## 一句话回答

在这里写你的回答。

## 核心思路

正常使用 Markdown，无需写 HTML 或 CSS。

## 面试追问

- 哪些条件下会有不同的实现？
```

- `id`：唯一、稳定的小写英文标识。用于题目链接和复习状态，重命名文件不受影响；不要随意修改 id。
- `category`：使用下方已配置的分类 id。
- `difficulty`：`基础` / `进阶` / `深入`。
- `tags`：字符串数组，自动加入筛选器。
- `updated`：有效 `YYYY-MM-DD` 日期，决定默认排序。
- `summary`：列表摘要，详细内容写在正文。
- `draft: true`：不进入网站和搜索数据；准备好后改成 `false`。草稿源文件仍会存在于 Git 仓库，**不是保密机制**。
- 正文结构不强制；所有 Markdown 标题自动生成题目内目录。

提交后，GitHub Actions 会重新构建并发布。字段错误或重复 id 会导致构建明确失败，避免静默漏题。

## 默认分类

| id | 名称 |
|---|---|
| `guide` | 学习路线与架构 |
| `diffusion` | Diffusion · 生成模型 |
| `llm` | LLM · 模型与后训练 |
| `agent` | Agent · 交互与学习 |
| `systems` | Infra · 训练与推理 |
| `coding` | 算法与编程 |
| `behavior` | 综合面试 |

论文收录是独立组件，入口由 `paperLibrary` 配置；不作为题目分类。新增学习笔记分类只需修改 `site.config.json` 中的 `categories`，无需修改页面逻辑。站名、副标题、作者也在这里配置。跨题链接可写 `[查看相关题](#q=kv-cache)`；外部资料使用完整 HTTPS 链接。当前不处理相对 Markdown 文件链接，也不自动复制内容目录里的图片；如需本地图片支持，可后续统一增加 assets 管线。

## 复习功能与数据边界

- 全文搜索覆盖标题、摘要、标签和 Markdown 正文。
- 主题、标签、难度、复习状态可以组合筛选。
- 点击随机复习，从当前筛选结果中选题。
- 题目详情可标记「未开始 / 复习中 / 已掌握」，也可收藏。
- `/` 快捷键聚焦搜索；详情页按 `Esc` 返回列表。
- 进度保存在 `localStorage`，**不会写回 Markdown，也不会自动同步到 GitHub或其他设备**。
- 使用侧栏「导出进度 / 导入进度」迁移。导入会合并，冲突 id 以备份为准，并先征求确认。
- 清理浏览器数据、换域名或更换项目 URL 都可能看不到原进度，请先导出。
- 此站没有登录或访问控制；不要把公司机密、账号密钥、未授权面试材料放到公开仓库或网页。

## 发布到 GitHub Pages

1. 在 GitHub 创建空仓库，例如 `interview-notes`，不要勾选自动初始化 README。
2. 在本地项目执行（将用户名改为自己的）：

   ```bash
   git init -b main
   git add .
   git commit -m "Build interview notebook"
   git remote add origin https://github.com/YOUR_USERNAME/interview-notes.git
   git push -u origin main
   ```

3. 仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
4. 在 Actions 中手动运行 `Deploy interview notes`，或再推送一次提交。
5. 部署完成后在 Pages 设置或 workflow 中查看实际地址，通常为 `https://YOUR_USERNAME.github.io/interview-notes/`。

前端资源均采用相对地址，题目使用 hash 路由，适配仓库子路径，刷新题目详情不会请求不存在的服务器路由。PR 只构建测试，不部署；`main` 推送才发布。

部署流程采用 [GitHub Pages 官方自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) 的构建产物上传与发布机制。

## 文件结构

```text
interview-notes/
├── content/questions/       学习路线与各方向笔记
├── content/papers/          各方向论文精读
├── content/reports/         Daily 与专题报告
├── content/research/        来源记录与调研边界
├── docs/paper-workflow.md   后续整理与 Agent 使用约定
├── templates/question.md    新题模板
├── site.config.json         站点与分类配置
├── web/                     统一的页面、样式和交互
├── scripts/                 构建、新建题目、本地预览
├── tests/                   内容解析与字段校验测试
├── .github/workflows/       GitHub Pages 自动部署
└── dist/                    生成产物（不提交 Git）
```

## 后续可扩展

开发页面时可额外运行真实浏览器测试（首次需下载 Chromium）：

```bash
npx playwright install chromium
npm run test:browser
```

测试覆盖搜索筛选、详情刷新、收藏与状态持久化、进度导入导出、手机宽度和 GitHub Pages 子路径。服务器若缺少中文字体，自动截图可能显示方框；用户浏览器使用系统中文字体，不依赖在线字体服务。

先维护内容，不急着增加复杂基础设施。以后可以统一增加数学公式渲染、代码高亮、个人笔记、间隔复习、面试路线图和本地图片复制；不需要改变「Markdown 内容 + 统一页面」这个基本结构。当前数学公式可用文本或代码块表达，尚未接入 LaTeX 渲染。
