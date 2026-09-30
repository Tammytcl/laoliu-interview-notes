# 论文收录与精读工作流 v5

这套组件管理的是持续积累的报告，不是自动解析论文的产品。抓取与图表工具只准备资料；报告由人工或 Agent 核对原文后按统一结构整理，再经过内容和构建检查发布。

**单篇精读的正式验收以 [DDPM 深度标准](./paper-depth-standard.md) 为准。** 五个标题仅是结构，不能代替背景铺垫、核心机制、配置账本、源码伴读和逐图解读。新论文模板为 v5，发布前必须通过 `npm run papers:quality`；旧版短报告可继续访问，待按同一标准逐篇升级。当前 Multi-Rollout MOPD、Multi-Teacher MOPD 与 Breaking the Tokenizer Barrier 已作为 OPD 专题的升级样例。

## 内容与页面设计

论文库每条记录展示英文全名、完整作者、发表时的英文单位、arXiv 固定版本、GitHub 代码出处及 tags。单篇报告以英文标题作为唯一主标题，作者和单位分行；摘要用中文解释贡献。资源用带文字的 SVG 图标入口呈现，避免纯图标难以识别。正文五个主模块，右侧只放目录；手机上目录移到正文前。图表点击放大，可打印为 PDF。

三类记录：单篇精读是可更新的知识记录；Daily 是检索、纳入与修改日志；专题报告比较多篇论文、更新研究判断。Daily / 专题引用稳定 paper ID，不重复复制精读报告。

## 五个模块的写作标准

| 主模块 | 必须回答 | 可用的证据 |
| --- | --- | --- |
| 背景与已有工作 | 什么问题、谁已发现、前作怎么做、为什么仍不足 | 原论文引言 / related work、相关原始论文 |
| 方法与实现机制 | 输入、核心设计、算法 / 训练目标、执行过程、输出和设计理由 | 方法图、公式、变量、伪代码、作者代码 |
| 实验设置与算力 | 哪个模型、哪些训练数据与 benchmark、什么配置和硬件 | 正文实验、附录、代码配置、checkpoint 说明 |
| 结果与图表解读 | 在什么条件下改善什么、每张图表支持什么 | 主表、结果图、消融、负例、可追溯数值 |
| 局限、结论与后续阅读 | 哪些结论成立、哪里不能外推、还需查什么 | limitations、缺少对照、自己的明确推导 |

正文写成能独立读懂的报告，不把原 section 标题机械翻译成十二个短条目。通常只有五个 H2 主标题；方法 / 实验 / 结果内部按解释需要设 H3。不要用“先写问题再读正文”的大段页面提示代替内容。

## 元数据约定

`templates/paper.md` 为 v5。`title` 和 `paper_title` 使用原文英文全名，`authors` 按论文顺序列全名，`affiliations` 使用论文发表时单位，不能用今天网页里的任职代替。`author_affiliations` 逐作者记录从 1 开始的单位编号，可表达多单位。`venue` 是原始发表信息；arXiv 首次年份与会议年份不同时在正文说明。

`paper_url` 固定 arXiv 版本。`github_url` 找不到时为 null；`code_note` 说明作者官方实现、论文引用框架或第三方复现。框架仓库不等于完整复现包，公开仓库的当前 main 不等于原实验 commit。不要编造代码链接。

`evidence` 仍采用：资料整理、已核原文、已复现。已核原文用于报告来源说明，不在页面上作为个人阅读状态显示，也不表示做过实验。已复现必须有实际执行配置、日志和结果。

## 完整整理流程

1. **检索与筛选。** Daily 记检索方向、来源、时间、候选与纳入 / 排除理由。先说明为什么与当前问题有关，不靠流行度直接收录。
2. **建稳定记录。** 执行 `npm run new:paper -- paper-my-topic llm`，选择 diffusion / llm / agent / infra。默认 draft，不覆盖同名记录。
3. **保存固定版本资料。** `npm run papers:fetch -- paper-my-topic 2305.13245v3` 下载 LaTeX、PDF、HTML。获取失败写入 manifest，不能暗中替换另一个版本。查看主 `.tex`、图文件、`.bib` 和附录；不自动执行不可信 LaTeX 命令。
4. **提取证据资产。** 先对照固定版本 PDF，人工登记 `scripts/figure-crops/<id>.json` 的页码和裁剪区域，再执行 `npm run papers:pdf-figures -- paper-my-topic`。长期资产放在 `assets/papers/paper-my-topic/`，逐张核对图例、坐标、子图、表头及末行。源码中独立原图也可直接使用并登记出处；`papers:figures` 的 HTML 截图仅作辅助，必须与 PDF 对照后才能采用。工具不生成精读结论，不提交整篇 PDF。
5. **建立问题和方法解释。** 先写主要瓶颈、代表前作与差异，再沿输入输出解释机制。公式注明变量与假设，代码注明原始算法或教学重述；源代码的 tensor layout 必须另核。
6. **核对实验账本。** 把预训练、微调、prompt 示例、benchmark 和负载数据分开。逐项核对模型、数据、评测 split、指标、超参、生成设置、硬件、训练时间与代码版本。缺项写“原文未披露”，并说明对复现的影响。没有训练环节就直接写“未训练新模型”。
7. **逐图理解。** 每张主图 / 表登记研究问题、轴 / 图例、对照、主要观察、支持结论与限制；图的插入位置服务于解释。较多附录图可以用清单登记未完成范围，不声称全部解读。完成后把 `figures.json` 对应图的 `explanation` 改成 `complete`，补 `reportSection` 与说明。
8. **复核与发布。** 对照原文逐个确认数字、单位、图号和引用。写清作者实测、整理者计算与未验证项。先对照 [DDPM 深度标准](./paper-depth-standard.md) 并运行 `npm run papers:quality`，再运行 `npm test`、`npm run build`；页面核查公式、插图和窄屏布局，通过后才改 draft、提交与推送。链接可通过 `#paper=paper-id` 稳定访问。
9. **建立积累。** Daily 只写本次新增 / 修改与判断变化；方向专题比较实验条件后再更新结论。同一论文新版本继续更新原 ID，并说明哪些结果受影响。新增图文件用新名称，不悄悄替换旧版本的图源。

## 图表与存储分层

| 路径 | 内容 | Git / 发布 | 清理策略 |
| --- | --- | --- | --- |
| `content/papers/` | 精读 Markdown | 保留并发布 | 不由缓存清理删除 |
| `content/reports/` | Daily 与专题 | 保留并发布 | 不由缓存清理删除 |
| `assets/papers/<id>/` | 长期插图、原表截图、图表清单 | 保留并发布 | 单独审核，不自动删除 |
| `.paper-cache/<id>/` | source archive、解压 LaTeX、PDF、HTML、HTML 图片与下载清单 | gitignore，不发布 | 空间不足时一键清理 |

图表清单保存原文版本、原图编号、捕获方式、文件 hash 与解释状态。只复制需要公开阅读的证据图，保留原作者归属和原文链接；不要把整篇论文逐页截图发布。

`papers:fetch` 需要 Python 3.12+（安全 tar 解压）；`papers:figures` 使用项目已有 Playwright / Chromium。安装仓库依赖后可按需执行 `npx playwright install chromium`。这两个工具不参加网站日常构建，CI 不需要下载原文或安装 Python 图像库。

## 一键清理

执行 `npm run papers:workbench`，打开显示的本地 URL → 论文收录 → 收录指南。缓存管理显示文件数量与逻辑文件大小，按钮确认后只清理固定的 `.paper-cache/`。报告和已保存插图不受影响。也可直接执行 `npm run papers:clean`。

普通 preview 不开放写接口；GitHub Pages 是静态网站，无法删除这台机器的源码。线上按钮保持禁用，并提示打开本地工作台。工作台仅监听 127.0.0.1，清理接口校验同源与会话 token，不接受任意路径。不要将工作台代理为公共写服务。

本轮默认保留下载的源码，未替用户执行清理。资料重新下载时要沿用相同固定版本；缓存已有不同版本时会拒绝混用。

## 给 Agent 的统一整理约定

> 为指定论文更新稳定的精读记录。先通读 DDPM 样稿和 `docs/paper-depth-standard.md`，再核对固定版本原文、LaTeX、图表和作者代码来源，按 v5 五个主模块写成完整报告。页首英文全名、作者、单位必须准确；代码若只是框架要注明。背景回答前作是否已发现问题、如何解决和剩余缺口。方法讲变量、形状、机制、训练与推理流程。实验逐项记录模型、数据、benchmark、超参、解码和算力，未披露不得填猜测。主图逐张说明轴 / 对照、观察、证据和边界，将图表复制到长期资产目录并记录出处。结论区分作者实测、自己的计算与待验证；不编造用户理解、阅读进度或复现。完成后检查数字、单位、原图多子图、公式和坏链接，更新图表解释清单与 Daily / 专题引用。源码先保留。`papers:quality` 与人工逐图核对全部通过才允许发布。

## 当前报告范围

GQA 已按 v3 源码与原表核对，正文解释全部 6 张主图和 Table 1。其余基础论文按相同元数据与五模块结构更新，重点图表在正文解读；未完成的补充图仍记录在各自清单中。它们均不声称本人跑过论文实验。

页面不提供收藏、待精读、复习状态或进度操作，题目与论文均按内容组织与检索。

## V4：视觉 / 语言检索、源码伴读与讲解融合

论文库一级使用 `vision` 视觉 / `language` 语言，第二行按所选领域列出任务。视觉覆盖理解、识别、检测、跟踪、分割、图像生成、视频生成；语言覆盖理解、生成、推理、Agent、检索问答、训练适配。`areas` 与 `tasks` 允许多个值，多模态论文可以同时归入两个领域。Diffusion、LLM、Attention、Infra、KV Cache 等技术或模型名称作为 tags，不再使用四个贡献类别。旧 direction 只用于文件路径兼容。Daily 和专题的筛选来自关联论文。

`published` 必须写论文首次公开日期，arXiv 论文取 v1 日期；会议名称和年份独立记录。论文库按 published 降序，报告按报告日期降序，updated 只表示报告整理时间。

`method_figure` 指向已核对、已保存的原论文方法图，`method_caption` 写图号与内容。方法图用于库中预览，正文仍需详细解释图；没有合适方法图时可以留 null，不生成假图。正式报告采用 template_version: 4。

方法阅读同时做源码伴读。先确定官方代码是否真的发布了论文方法，再固定 commit。选择 2–4 个核心函数，沿真实调用链解释输入输出、张量形状或系统状态、重要分支、对应公式与为何有效。明确原始实现、历史 release、现代框架端点或第三方复现；一个框架链接不等于完整复现包。示例伪代码标为教学重述，不能代替核心函数链接。静态核读不写成已运行验证。下载源码可放在 .paper-cache/<id>/code 中，清理缓存不会删除报告里的解释和永久链接。

写背景时先定义任务和最小必要概念，再解释具体瓶颈、已有方案与本文补上的环节。方法先用连贯段落解释核心思想，再给公式、图和代码；实验前补清 benchmark 的任务及指标含义，数字必须伴随条件。不要靠增加小标题或一行要点制造篇幅。保留五个主模块，子标题只服务真实阅读转折。

经典论文额外检索作者博客和优质精读。登记 URL、作者、发表时间和实际吸收的解释角度；技术事实以论文和核心实现交叉核对。不同版本、不同模型或不同 baseline 的数字不能混合。把有帮助的论证顺序、例子和观点融进自己的解释，参考资料集中放第五模块末尾；不复制长段落、图或拼贴摘要。验收时检查：新读者能否说清任务、核心机制、训练/推理差异、代码如何落地、实验究竟证明什么。


## 引用量与自动更新

`npm run papers:citations` 更新 `content/metadata/citations.json`。每周一 UTC 02:17 GitHub Actions 自动尝试更新、提交快照并发布；手动推送也检查七天内是否更新过。请求失败不覆盖上次成功计数和日期，不把缺失值写成零。卡片及报告展示数据库来源、统计日期和 Google Scholar 入口。

默认使用 OpenAlex。可在论文 frontmatter 填入经过核对的 `openalex_id` 固定正式发表记录，避免预印本与会议版本分散；不相加可能重复的版本引用。程序校验完整标题与第一作者姓氏，匹配失败时不显示数字。不同数据库覆盖不同，OpenAlex 计数不是 Google Scholar 计数，也不自动据此给论文贴“经典”标签。

需要 Google Scholar 统计时，在 GitHub 仓库 Settings → Secrets and variables → Actions 配置 `SERPAPI_API_KEY`；本地则使用环境变量。程序调用 SerpApi Google Scholar 搜索，以完整标题和作者核对唯一匹配，读取 cited_by.total，单独保存 Google Scholar 指标。密钥只进入服务端/Actions，不写入仓库和前端，不打印请求 URL。没有密钥时不调用该收费接口。OpenAlex 限流时可配置免费的 `OPENALEX_API_KEY`，也允许无密钥的基础查询。依据：[Google Scholar 自动访问说明](https://scholar.google.com/intl/en/scholar/help.html)、[SerpApi 接口](https://serpapi.com/google-scholar-api)、[OpenAlex 认证说明](https://help.openalex.org/api/authentication/)。

## PDF 图表核对与发布后检查

过去的 HTML 截图可留作历史资产，但正文现在引用核对过的 `*-pdf.png`。arXiv HTML 是实验性转换，可能改变子图布局、表格、LaTeX 条件选择或丢失文字；加载成功并不代表图内容正确。优先采用固定版本 PDF 的原始排版裁剪，LaTeX 包用于确认图文件、编号和上下文，不要求重新执行编译。

安装可选裁剪依赖：`python -m pip install -r scripts/requirements-paper-images.txt`。在 `scripts/figure-crops/<id>.json` 人工登记 PDF 页码与从左上角起的 point 坐标，执行 `npm run papers:pdf-figures -- <id>`。工具校验版本、边界与空白图，记录 PDF SHA256、图 SHA256、页码和矩形；已有文件需核对后使用 --refresh。裁剪后逐张与原页对照，确认图例、坐标、所有子图、表头及末行完整，才标记 visuallyVerified。报告及卡片引用新文件名，防止缓存旧图。

每次部署后 Actions 执行 `papers:check-live`，实际打开线上页面，检查桌面和手机两种宽度、所有报告图片加载与非空白、纵横比、目录、图片放大关闭与横向溢出，保存截图为 `paper-live-check` artifact。这个自动检查辅助发现排版问题，不能替代逐图人工核对。也可手动运行 `npm run papers:check-live -- https://tammytcl.github.io/laoliu-interview-notes/`。
