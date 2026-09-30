const $ = selector => document.querySelector(selector);
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const tabs = { library: '论文库', daily: 'Daily 报告', survey: '专题调研', guide: '收录指南' };
let papers = [], reports = [], directions = [], notify;
let libraryName = '论文收录';
let query = '', direction = '';
const directionName = id => directions.find(d => d.id === id)?.name || id;
export const isPaperRoute = params => params.get('view') === 'papers' || params.has('paper') || params.has('report');
function hub(tab = 'library') { return `#view=papers&tab=${tab}`; }
function relatedPaperLinks(ids) {
  return ids.map(id => {
    const p = papers.find(x => x.id === id);
    return p ? `<a class="button small" href="#paper=${esc(id)}">${esc(p.title)} ↗</a>` : '';
  }).join('');
}
const icons = {
  arxiv: '<path d="m5 4 14 16M19 4 5 20M8 4h11v11"/>',
  github: '<path d="M9 19c-4 1-4-2-6-2m12 4v-3.5c0-1 .3-1.5.7-2-3 .3-6.7-1.5-6.7-5.5 0-1.3.5-2.4 1.3-3.2-.2-.8-.2-2 .3-3.3 0 0 1.2-.3 3.4 1.3a12 12 0 0 1 6 0C22 3.2 23 3.5 23 3.5c.5 1.3.5 2.5.3 3.3.8.8 1.3 1.9 1.3 3.2 0 4-3.7 5.8-6.7 5.5.4.5.7 1 .7 2V21" transform="translate(-3 0) scale(.9 1)"/>',
  file: '<path d="M14 3H5v18h14V8l-5-5ZM14 3v5h5M8 12h8M8 16h8"/>',
  print: '<path d="M7 8V3h10v5M7 17H3V9h18v8h-4M7 14h10v7H7zM17 11h1"/>',
  cache: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>'
};
function icon(name) { return `<svg class="paper-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.file}</svg>`; }
function resourceLinks(p) {
  return `<a class="button small icon-button" href="${esc(p.paperUrl)}" target="_blank" rel="noopener noreferrer">${icon('arxiv')}arXiv</a>${p.githubUrl ? `<a class="button small icon-button" href="${esc(p.githubUrl)}" target="_blank" rel="noopener noreferrer" ${p.codeNote ? `title="${esc(p.codeNote)}"` : ''}>${icon('github')}GitHub</a>` : '<span class="code-unavailable">Code: not released / 未核实</span>'}`;
}
function categoryIds(item) {
  return item.categories || [...new Set((item.paperIds || []).flatMap(id => papers.find(p => p.id === id)?.categories || []))];
}
function categoryBadges(item) { return `<div class="paper-category-badges">${categoryIds(item).map(id => `<span class="paper-category" data-category="${esc(id)}">${esc(directionName(id))}</span>`).join('')}</div>`; }
function renderCards(tab) {
  const source = tab === 'library' ? papers : reports.filter(r => r.type === tab);
  const items = source.filter(item => (!direction || categoryIds(item).includes(direction))
    && (!query || `${item.title} ${item.summary} ${item.tags.join(' ')} ${item.body} ${item.paperTitle || ''} ${(item.authors || []).join(' ')} ${(item.affiliations || []).join(' ')}`.toLocaleLowerCase().includes(query.toLocaleLowerCase())));
  $('#paper-result-count').textContent = `${items.length} 篇`;
  $('#paper-results').innerHTML = items.length ? items.map(item => {
    const paper = tab === 'library';
    const meta = paper ? `${item.year}${item.venue ? ` · ${item.venue}` : ''}` : `${tabs[tab]} · ${item.date} · ${categoryIds(item).map(directionName).join(' / ')}`;
    return `<article class="paper-card" data-category="${esc(categoryIds(item)[0] || '')}"><div class="question-copy">${categoryBadges(item)}<p class="paper-meta">${esc(meta)}</p><h3><a href="#${paper ? 'paper' : 'report'}=${esc(item.id)}">${esc(paper ? item.paperTitle : item.title)}</a></h3>${paper ? `<p class="paper-card-authors" lang="en">${esc(item.authors.join(' · '))}</p><p class="paper-card-affiliations" lang="en">${esc((item.affiliations || []).join(' / '))}</p><div class="paper-card-resources">${resourceLinks(item)}</div>` : ''}<p>${esc(item.summary)}</p><div class="tags">${item.tags.map(t => `<span># ${esc(t)}</span>`).join('')}</div><div class="paper-card-footer"><span>更新 ${esc(item.updated)}${paper ? '' : ` · 关联 ${item.paperIds.length} 篇论文`}</span><a href="#${paper ? 'paper' : 'report'}=${esc(item.id)}">${paper ? '打开精读报告' : '打开报告'} ↗</a></div></div>${paper && item.methodFigure ? `<a class="paper-method-preview" href="#paper=${esc(item.id)}" aria-label="查看方法图与报告：${esc(item.paperTitle)}"><img src="${esc(item.methodFigure)}" alt="${esc(item.methodCaption)}" loading="lazy"><span>${esc(item.methodCaption)}</span></a>` : ''}</article>`;
  }).join('') : `<div class="empty"><h3>这里暂时没有匹配的${tab === 'library' ? '论文' : '报告'}</h3><p>可以清除筛选，或按统一模板添加记录。</p><button class="button" data-paper-reset>清除筛选</button><a class="button" href="${hub('guide')}">查看收录指南</a></div>`;
}
function renderGuide() {
  return `<article class="prose paper-guide"><p class="eyebrow">COLLECTION & RESEARCH WORKFLOW / V3</p><h2>从原文证据到可持续更新的精读报告</h2><p>单篇报告是长期记录，Daily 是检索与更新日志，专题报告是跨论文的研究结论。一次收录沿着下列流程完成，正文不照搬论文 section，也不由下载工具自动生成。</p>
    <div class="template-grid"><a class="paper-template" href="./templates/paper.md" download><strong>精读报告模板 ↓</strong><span>五个主模块 · 原文图表 · 实验与算力</span></a><a class="paper-template" href="./templates/daily-report.md" download><strong>Daily 模板 ↓</strong><span>检索候选 · 纳入理由 · 当日更新</span></a><a class="paper-template" href="./templates/survey-report.md" download><strong>专题调研模板 ↓</strong><span>方法谱系 · 条件对齐 · 证据与判断</span></a></div>
    <ol class="workflow-steps"><li><strong>登记与筛选</strong><p>记录英文全名、完整作者、论文发表时的英文单位、arXiv 版本与代码出处。说明为什么值得读；Daily 先记候选，确定精读后再建稳定论文 ID。</p></li><li><strong>保存原文与图表</strong><p>下载固定版本的 LaTeX、PDF 和 HTML 到本地缓存。核对主文件、参考文献与图表编号；从源码取图或截取原文图表，保存在报告资产目录，并记录出处。源码中的命令不自动执行。</p></li><li><strong>建立问题与方法解释</strong><p>先回答谁已发现这个问题、以前如何处理、为什么仍不足。再沿输入、关键机制、训练与推理路径讲清本文方法，用必要的公式、维度和伪代码支撑解释。开源论文还要固定代码 commit，沿调用链解读核心函数，说明输入输出、关键状态与论文公式的对应；检索作者及优质讲解，核对后融合进正文并在末尾引用。</p></li><li><strong>核对实验与逐图证据</strong><p>逐项登记模型/checkpoint、训练数据、benchmark、指标、超参、硬件数量和计算成本。每张图写清坐标/对照、观察、结论与不能推出的内容；每张表说明实验口径。未披露的信息明确注明，TPU 不换写成 GPU。</p></li><li><strong>复核、关联与发布</strong><p>检查数字、公式、图号与引用；区分作者实测、原文事实、整理者推导和未验证项。构建通过后发布，Daily 链接本次改动，专题报告更新受影响的判断。源码先保留，空间紧张时单独清理。</p></li></ol>
    <h3>本地执行</h3><pre>npm run new:paper -- paper-my-topic llm
npm run papers:fetch -- paper-my-topic 2305.13245v3
npm run papers:figures -- paper-my-topic
# 核读原文，填写报告，逐张图写解释
npm test && npm run build
npm run papers:workbench</pre><p>图表获取失败时保留错误信息，改用源码图片或 PDF 手动裁剪。获取工具只准备资料，不替你写结论。完整的字段说明、验收标准与给 Agent 的整理约定见 <a href="https://github.com/Tammytcl/laoliu-interview-notes/blob/main/docs/paper-workflow.md" target="_blank" rel="noopener noreferrer">工作流文档 ↗</a>。</p>
    <section class="paper-cache-panel" aria-label="源码缓存管理"><h3>${icon('cache')}源码缓存管理</h3><p>清理本地下载的源码、PDF、HTML 与中间文件。已发布的 Markdown 报告和 assets/papers 中的插图保留。</p><p id="paper-cache-status" role="status">检查本地工作台…</p><button class="button icon-button" data-cache-clean disabled>${icon('cache')}一键清理源码缓存</button><p class="cache-command">本地入口：<code>npm run papers:workbench</code>；命令行清理：<code>npm run papers:clean</code>。</p></section>
    <h3>更新约定</h3><p>同一论文不同版本更新同一记录，同一天 Daily 继续写同一文件。新增图表使用新文件名并注明原文版本，旧结论被推翻时写清证据变化。精读正文由仓库版本管理。</p></article>`;
}
let cacheToken = '';
async function loadPaperCache() {
  const status = $('#paper-cache-status'); if (!status) return;
  if (!['localhost', '127.0.0.1', '[::1]'].includes(location.hostname)) { status.textContent = 'GitHub Pages 展示报告；清理本机源码请打开本地工作台。'; return; }
  try {
    const response = await fetch('./api/paper-cache', { cache: 'no-store' });
    if (!response.ok) throw new Error('not-workbench');
    const data = await response.json(); cacheToken = data.token;
    if (!$('#paper-cache-status')) return;
    $('#paper-cache-status').textContent = `${data.files} 个缓存文件 · ${(data.bytes / 1024 / 1024).toFixed(2)} MB`;
    $('[data-cache-clean]').disabled = data.files === 0;
  } catch { if ($('#paper-cache-status')) $('#paper-cache-status').textContent = '当前是普通预览；使用 npm run papers:workbench 启用本地清理。'; }
}
async function cleanPaperCache() {
  if (!cacheToken || !confirm('清理全部本地论文源码缓存？报告正文和已保存插图会保留；源码可重新下载。')) return;
  const button = $('[data-cache-clean]'); button.disabled = true;
  try {
    const response = await fetch('./api/paper-cache/clean', { method: 'POST', headers: { 'X-Paper-Cache-Token': cacheToken } });
    if (!response.ok) throw new Error('本地服务拒绝请求');
    const data = await response.json(); notify(`已清理 ${data.deletedFiles} 个缓存文件，报告与插图已保留。`); await loadPaperCache();
  } catch (e) { notify(`清理失败：${e.message}`); button.disabled = false; }
}
function renderHub(params) {
  const tab = Object.hasOwn(tabs, params.get('tab')) ? params.get('tab') : 'library';
  const selectedDirection = params.get('direction');
  if (selectedDirection && directions.some(d => d.id === selectedDirection)) direction = selectedDirection;
  $('#paper-space').innerHTML = `<header class="paper-hub-header"><p class="eyebrow">PAPER LIBRARY · RESEARCH REPORTS</p><h1>${esc(libraryName)}</h1><p>按主要研究贡献归档论文精读，以 Daily 与专题调研记录持续更新。</p><div class="paper-hub-summary"><span>${papers.length} 篇论文</span><span>${reports.filter(r => r.type === 'daily').length} 篇 Daily</span><span>${reports.filter(r => r.type === 'survey').length} 篇专题</span></div></header><nav class="paper-tabs" aria-label="论文与报告模块">${Object.entries(tabs).map(([id, name]) => `<a class="button ${tab === id ? 'primary' : ''}" href="${hub(id)}" ${tab === id ? 'aria-current="page"' : ''}>${name}</a>`).join('')}</nav>${tab === 'guide' ? renderGuide() : `<section id="paper-report-zone" aria-label="${tabs[tab]}"><div class="section-heading"><h2>${tabs[tab]} <small id="paper-result-count"></small></h2><label class="search"><input type="search" id="paper-search" aria-label="搜索论文和报告" placeholder="搜索标题、方法、作者或报告正文…" value="${esc(query)}"></label></div><p class="paper-zone-intro">${tab === 'library' ? '类别表示主要研究贡献，允许交叉归属；Diffusion、LLM 等应用领域保留在标签中。右侧原图展示方法结构，完整解读见报告。' : tab === 'daily' ? '按日期记录候选、收录理由与阅读更新。关联稳定论文记录，不重复复制精读正文。' : '围绕方向或问题系统调研，比较方法与证据，持续记录结论怎样变化。'}</p><div class="paper-filters"><select id="paper-direction" aria-label="论文贡献类别"><option value="">全部贡献类别</option>${directions.map(d => `<option value="${d.id}" ${direction === d.id ? 'selected' : ''}>${esc(d.name)}</option>`).join('')}</select><a class="button small" href="${hub('guide')}">模板与更新方式 ↗</a></div><div id="paper-results" class="paper-results" aria-live="polite"></div></section>`}`;
  if (tab === 'guide') loadPaperCache();
  if (tab !== 'guide') {
    renderCards(tab);
    $('#paper-search').addEventListener('input', e => { query = e.target.value; renderCards(tab); });
    $('#paper-direction').addEventListener('change', e => {
      direction = e.target.value;
      const next = new URLSearchParams(location.hash.slice(1));
      if (direction) next.set('direction', direction); else next.delete('direction');
      location.hash = next.toString(); renderCards(tab);
    });
  }
}
function renderDetail(params) {
  const paper = params.has('paper'); const id = params.get(paper ? 'paper' : 'report');
  const item = (paper ? papers : reports).find(x => x.id === id);
  if (!item) { $('#paper-space').innerHTML = `<div class="empty"><h1>这篇${paper ? '论文' : '报告'}暂时不存在</h1><a class="button" href="${hub()}">返回论文收录</a></div>`; return; }
  document.title = `${paper ? item.paperTitle : item.title} · 论文收录`;
  const relatedReports = paper ? reports.filter(r => r.paperIds.includes(id)) : [];
  const authorLine = paper ? item.authors.map((author, i) => `<span>${esc(author)}${item.authorAffiliations?.[i]?.length ? `<sup>${item.authorAffiliations[i].join(',')}</sup>` : ''}</span>`).join('<span class="author-separator">·</span>') : '';
  const sourceLink = `https://github.com/Tammytcl/laoliu-interview-notes/blob/main/${item.source}`;
  $('#paper-space').innerHTML = `<a class="back-link" href="${hub(paper ? 'library' : item.type)}">← 返回${paper ? '论文库' : tabs[item.type]}</a>
    <header class="reader-header publication-header">
      <div class="publication-kicker"><span>${paper ? 'PAPER READING REPORT' : 'RESEARCH REPORT'}</span><span>${paper ? `${esc(item.venue || directionName(item.direction))} · ${item.year}` : `${tabs[item.type]} · ${item.date}`}</span></div>
      <h1 tabindex="-1" ${paper ? 'lang="en"' : ''}>${esc(paper ? item.paperTitle : item.title)}</h1>
      ${paper ? `<div class="publication-authors" lang="en">${authorLine}</div><div class="publication-affiliations" lang="en">${(item.affiliations || []).map((name, i) => `<span>${item.affiliations.length > 1 ? `<sup>${i + 1}</sup>` : ''}${esc(name)}</span>`).join('')}</div>` : ''}
      ${paper ? categoryBadges(item) : ''}<p class="publication-summary">${esc(item.summary)}</p>
      <div class="tags publication-tags">${item.tags.map(tag => `<span>${esc(tag)}</span>`).join('')}</div>
      <div class="publication-actions">${paper ? resourceLinks(item) : ''}<a class="button small icon-button" href="${esc(sourceLink)}" target="_blank" rel="noopener noreferrer">${icon('file')}报告 Markdown</a><button class="button small icon-button" data-print-paper>${icon('print')}打印 / PDF</button></div>
      <div class="publication-revision"><span>报告更新 ${esc(item.updated)}</span><span>正文 ${item.minutes} 分钟</span></div>
    </header>
    ${!paper ? `<section class="paper-related"><h2>本报告关联的论文</h2><div>${relatedPaperLinks(item.paperIds) || '<p>本次没有新增论文。</p>'}</div></section>` : ''}
    <div class="reader-grid paper-reader-grid publication-grid"><div><article class="prose paper-body publication-body" id="paper-body">${item.html}</article>${paper && relatedReports.length ? `<section class="paper-related"><h2>相关收录与调研</h2>${relatedReports.map(r => `<p><a href="#report=${r.id}">${tabs[r.type]} · ${r.date} · ${esc(r.title)} ↗</a></p>`).join('')}</section>` : ''}</div><aside class="reader-aside"><nav class="toc publication-toc" aria-label="报告目录"><span class="eyebrow">CONTENTS / 目录</span>${item.toc.filter(t => t.level === 2).map((t, i) => `<button data-section="${t.id}" class="level-2"><span>${String(i + 1).padStart(2, '0')}</span>${esc(t.title.replace(/^\d+[.、]\s*/, ''))}</button>`).join('')}</nav></aside></div>
    <dialog id="paper-figure-dialog" class="figure-dialog"><div class="figure-dialog-toolbar"><p id="figure-caption"></p><button class="button" data-close-figure aria-label="关闭图片">关闭 ×</button></div><img id="figure-full" alt=""><a id="figure-original" class="button small" target="_blank" rel="noopener noreferrer">打开原尺寸 ↗</a></dialog>`;
  for (const image of document.querySelectorAll('.publication-body img')) {
    const button = document.createElement('button'); button.className = 'paper-figure-button'; button.type = 'button'; button.dataset.figure = 'true'; button.setAttribute('aria-label', `放大图片：${image.alt}`);
    image.replaceWith(button); button.append(image);
    const caption = document.createElement('span'); caption.className = 'paper-figure-caption'; caption.textContent = image.alt; button.append(caption);
  }
}
export function renderPaperLibrary(params) {
  if (!isPaperRoute(params)) return false;
  $('#paper-space').hidden = false;
  $('#breadcrumb').textContent = libraryName;
  if (params.has('paper') || params.has('report')) renderDetail(params); else renderHub(params);
  return true;
}
export function initPaperLibrary(data, toast) {
  libraryName = data.config?.paperLibrary?.name || '论文收录';
  papers = data.papers || []; reports = data.reports || []; directions = data.paperCategories || []; notify = toast;
  $('#paper-space').addEventListener('click', e => {
    if (e.target.closest('[data-print-paper]')) window.print();
    const figure = e.target.closest('[data-figure]');
    if (figure) {
      const image = figure.querySelector('img'); $('#figure-full').src = image.src; $('#figure-full').alt = image.alt;
      $('#figure-caption').textContent = image.alt; $('#figure-original').href = image.src; $('#paper-figure-dialog').showModal();
    }
    if (e.target.closest('[data-close-figure]')) $('#paper-figure-dialog')?.close();
    if (e.target.closest('[data-cache-clean]')) cleanPaperCache();
    if (e.target.closest('[data-paper-reset]')) { query = ''; direction = ''; const params = new URLSearchParams(location.hash.slice(1)); params.delete('direction'); renderPaperLibrary(params); }
  });
}
