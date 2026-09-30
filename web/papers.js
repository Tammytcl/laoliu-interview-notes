const $ = selector => document.querySelector(selector);
const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const stages = { queued: '待精读', reading: '阅读中', recalled: '已复述' };
const tabs = { library: '论文库', daily: 'Daily 报告', survey: '专题调研', guide: '收录指南' };
const key = `paper-reading:v1:${location.pathname}`;
let papers = [], reports = [], directions = [], reading = {}, notify;
let libraryName = '论文收录';
let query = '', direction = '', stage = '', starredOnly = false;
const blank = () => ({ stage: 'queued', starred: false, question: '', recall: '', connection: '', nextReview: '' });
const state = id => Object.hasOwn(reading, id) ? reading[id] : blank();
const directionName = id => directions.find(d => d.id === id)?.name || id;
const dateValid = value => !value || (/^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value);
export function validateReading(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('阅读记录格式无效');
  const result = {};
  for (const [id, item] of Object.entries(value)) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id) || !item || !Object.hasOwn(stages, item.stage) || typeof item.starred !== 'boolean') throw new Error('阅读状态格式无效');
    for (const field of ['question', 'recall', 'connection', 'nextReview']) if (typeof item[field] !== 'string' || item[field].length > 50000) throw new Error('阅读笔记格式无效');
    if (!dateValid(item.nextReview)) throw new Error('复习日期无效');
    if (item.stage === 'recalled' && !item.recall.trim()) throw new Error('已复述记录缺少个人复述');
    result[id] = Object.fromEntries(Object.keys(blank()).map(field => [field, item[field]]));
  }
  return result;
}
function save(id, patch) {
  reading[id] = { ...state(id), ...patch };
  if (reading[id].stage === 'recalled' && !reading[id].recall.trim()) reading[id].stage = 'reading';
  try { localStorage.setItem(key, JSON.stringify(reading)); }
  catch { notify('阅读记录无法保存，请立即导出备份。'); }
}
function refreshDetailState() {
  const open = $('#paper-body')?.open;
  renderPaperLibrary(new URLSearchParams(location.hash.slice(1)));
  if (typeof open === 'boolean' && $('#paper-body')) $('#paper-body').open = open;
}
export const isPaperRoute = params => params.get('view') === 'papers' || params.has('paper') || params.has('report');
function hub(tab = 'library') { return `#view=papers&tab=${tab}`; }
function relatedPaperLinks(ids) {
  return ids.map(id => {
    const p = papers.find(x => x.id === id);
    return p ? `<a class="button small" href="#paper=${esc(id)}">${esc(p.title)} ↗</a>` : '';
  }).join('');
}
function renderCards(tab) {
  const today = new Date().toLocaleDateString('en-CA');
  const source = tab === 'library' ? papers : reports.filter(r => r.type === tab);
  const items = source.filter(item => (!direction || (item.direction ? item.direction === direction : item.directions.includes(direction)))
    && (!query || `${item.title} ${item.summary} ${item.tags.join(' ')} ${item.body} ${item.paperTitle || ''} ${(item.authors || []).join(' ')}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()))
    && (tab !== 'library' || ((!stage || state(item.id).stage === stage) && (!starredOnly || state(item.id).starred))));
  $('#paper-result-count').textContent = `${items.length} 篇`;
  $('#paper-results').innerHTML = items.length ? items.map(item => {
    const paper = tab === 'library'; const entry = state(item.id);
    const meta = paper ? `${directionName(item.direction)} · ${item.year} · ${item.evidence}` : `${tabs[tab]} · ${item.date} · ${item.directions.map(directionName).join(' / ')}`;
    const due = paper && entry.nextReview && entry.nextReview <= today;
    return `<article class="paper-card"><div class="question-copy"><p class="paper-meta">${esc(meta)}</p><h3><a href="#${paper ? 'paper' : 'report'}=${esc(item.id)}">${esc(item.title)}</a></h3>${paper ? `<p class="paper-original-title">${esc(item.paperTitle)}</p>` : ''}<p>${esc(item.summary)}</p><div class="tags">${item.tags.map(t => `<span># ${esc(t)}</span>`).join('')}</div><div class="paper-card-footer"><span>更新 ${esc(item.updated)}${paper ? ` · ${stages[entry.stage]}${due ? ' · 到期复习' : ''}` : ` · 关联 ${item.paperIds.length} 篇论文`}</span><a href="#${paper ? 'paper' : 'report'}=${esc(item.id)}">${paper ? '打开精读报告' : '打开报告'} ↗</a></div></div>${paper ? `<button class="star ${entry.starred ? 'selected' : ''}" data-paper-star="${esc(item.id)}" aria-pressed="${entry.starred}" aria-label="收藏论文：${esc(item.title)}">${entry.starred ? '★' : '☆'}</button>` : ''}</article>`;
  }).join('') : `<div class="empty"><h3>这里暂时没有匹配的${tab === 'library' ? '论文' : '报告'}</h3><p>可以清除筛选，或按统一模板添加记录。</p><button class="button" data-paper-reset>清除筛选</button><a class="button" href="${hub('guide')}">查看收录指南</a></div>`;
}
function renderGuide() {
  return `<article class="prose paper-guide"><h2>让每次阅读都留下一点自己的理解</h2><p>论文只建一条稳定记录，Daily 和专题报告链接回它。资料整理程度和个人阅读程度分开：Agent 准备了报告，不代表你已经读懂。</p><h3>三种统一模板</h3><div class="template-grid"><a class="paper-template" href="./templates/paper.md" download><strong>单篇精读模板 ↓</strong><span>问题 → 方法 → 证据 → 局限 → 复述与验证</span></a><a class="paper-template" href="./templates/daily-report.md" download><strong>Daily 模板 ↓</strong><span>候选筛选 → 今日重点 → 阅读更新 → 下一步</span></a><a class="paper-template" href="./templates/survey-report.md" download><strong>专题调研模板 ↓</strong><span>研究问题 → 范围 → 方法谱系 → 比较 → 结论与空白</span></a></div><h3>每天开始工作时：10–15 分钟的手动流程</h3><ol><li>打开 Daily，看上次留下的待核问题与阅读进展。</li><li>收集候选，写清为什么值得读、先读哪里；没有合适论文也可以记录“今日不新增”。</li><li>选一篇作为主阅读，打开原文前写下一个想解答的问题。</li><li>阅读后合上报告，用自己的话写两三句，再设下一次复习日期。</li><li>把本次理解变化写回 Daily；有跨论文问题时补进专题报告。</li></ol><h3>单篇精读要留下什么</h3><p>原文、作者和版本；收录动机；问题与先修；方法及推导；关键图表和实验条件；作者结论与自己的判断；局限与反例；和已有知识的连接；不看报告的复述；待验证动作和更新记录。</p><h3>专题报告怎样积累</h3><p>先定义一个可回答的研究问题，再写检索范围、纳入/排除规则、分类与比较维度。引用单篇记录，对不兼容的实验条件作说明。每次更新写清“新增证据改变了哪个判断”，避免变成摘要合集。</p><details><summary>在本地追加内容并发布</summary><pre>npm run new:paper -- paper-my-topic llm
npm run new:daily
npm run new:survey -- survey-my-topic infra</pre><p>默认创建草稿，不会覆盖已有文件。Daily 同日再次执行会提示打开当天已有报告。填写内容和原文信息后将 draft 改为 false，再构建、提交并推送。页面里的“原文件”入口可定位 Markdown。</p></details><p class="notice">右侧/下方的个人阅读记录自动保存在当前浏览器，使用「导出阅读记录」备份；它不会自动写回公开精读报告。下载的 Markdown 模板与仓库中的报告才是长期积累的正文。当前 Daily 由你手动更新。</p></article>`;
}
function renderHub(params) {
  const tab = Object.hasOwn(tabs, params.get('tab')) ? params.get('tab') : 'library';
  const selectedDirection = params.get('direction');
  if (selectedDirection && directions.some(d => d.id === selectedDirection)) direction = selectedDirection;
  const recalled = papers.filter(p => state(p.id).stage === 'recalled').length;
  $('#paper-space').innerHTML = `<header class="paper-hub-header"><p class="eyebrow">READ · CONNECT · REMEMBER</p><h1>${esc(libraryName)}</h1><p>按方向读透单篇论文，用 Daily 和专题调研串起长期积累。</p><div class="paper-hub-summary"><span>${papers.length} 篇论文</span><span>${reports.filter(r => r.type === 'daily').length} 篇 Daily</span><span>${reports.filter(r => r.type === 'survey').length} 篇专题</span><span>${recalled} 篇已复述</span></div></header><nav class="paper-tabs" aria-label="论文与报告模块">${Object.entries(tabs).map(([id, name]) => `<a class="button ${tab === id ? 'primary' : ''}" href="${hub(id)}" ${tab === id ? 'aria-current="page"' : ''}>${name}</a>`).join('')}</nav><div class="paper-backup"><button class="button small" data-reading-export>导出阅读记录</button><button class="button small" data-reading-import>导入阅读记录</button><input type="file" id="reading-import-file" accept="application/json,.json" hidden></div>${tab === 'guide' ? renderGuide() : `<section id="paper-report-zone" aria-label="${tabs[tab]}"><div class="section-heading"><h2>${tabs[tab]} <small id="paper-result-count"></small></h2><label class="search"><input type="search" id="paper-search" aria-label="搜索论文和报告" placeholder="搜索标题、方法、作者或报告正文…" value="${esc(query)}"></label></div><p class="paper-zone-intro">${tab === 'library' ? '每条是一篇论文的精读记录。收录 ≠ 读懂；资料状态由报告标注，阅读状态由你维护。' : tab === 'daily' ? '按日期记录候选、收录理由与阅读更新。关联稳定论文记录，不重复复制精读正文。' : '围绕方向或问题系统调研，比较方法与证据，持续记录结论怎样变化。'}</p><div class="paper-filters"><select id="paper-direction" aria-label="论文方向"><option value="">全部方向</option>${directions.map(d => `<option value="${d.id}" ${direction === d.id ? 'selected' : ''}>${esc(d.name)}</option>`).join('')}</select>${tab === 'library' ? `<select id="paper-stage" aria-label="论文阅读状态"><option value="">全部阅读状态</option>${Object.entries(stages).map(([id, label]) => `<option value="${id}" ${stage === id ? 'selected' : ''}>${label}</option>`).join('')}</select><label><input id="paper-starred" type="checkbox" ${starredOnly ? 'checked' : ''}> 只看收藏</label>` : ''}<a class="button small" href="${hub('guide')}">模板与更新方式 ↗</a></div><div id="paper-results" class="paper-results" aria-live="polite"></div></section>`}`;
  if (tab !== 'guide') {
    renderCards(tab);
    $('#paper-search').addEventListener('input', e => { query = e.target.value; renderCards(tab); });
    $('#paper-direction').addEventListener('change', e => {
      direction = e.target.value;
      const next = new URLSearchParams(location.hash.slice(1));
      if (direction) next.set('direction', direction); else next.delete('direction');
      location.hash = next.toString(); renderCards(tab);
    });
    $('#paper-stage')?.addEventListener('change', e => { stage = e.target.value; renderCards(tab); });
    $('#paper-starred')?.addEventListener('change', e => { starredOnly = e.target.checked; renderCards(tab); });
  }
}
function readingWorkbench(p) {
  const entry = state(p.id);
  return `<section class="reading-workbench" aria-label="个人阅读记录"><p class="eyebrow">MY READING</p><h2>这篇留下了什么？</h2><div class="reading-stages">${Object.entries(stages).map(([id, label]) => `<button class="button small" data-paper-stage="${id}" data-paper-id="${p.id}" aria-pressed="${entry.stage === id}">${label}</button>`).join('')}</div><label for="reading-question">读前：我想解答的问题</label><textarea maxlength="50000" id="reading-question" data-reading-field="question" data-paper-id="${p.id}" placeholder="先写一个问题，再打开原文或精读正文。">${esc(entry.question)}</textarea><button class="button" data-close-paper>合上报告，开始复述</button><label for="reading-recall">读后：不用报告，复述核心机制</label><textarea maxlength="50000" id="reading-recall" data-reading-field="recall" data-paper-id="${p.id}" placeholder="问题是什么？关键设计为何有效？哪些条件下不成立？">${esc(entry.recall)}</textarea><label for="reading-connection">连接与待验证</label><textarea maxlength="50000" id="reading-connection" data-reading-field="connection" data-paper-id="${p.id}" placeholder="和已有知识的联系、仍不理解的点、下一次要做的实验。">${esc(entry.connection)}</textarea><label for="reading-next-review">下次复习日期</label><input id="reading-next-review" type="date" data-reading-field="nextReview" data-paper-id="${p.id}" value="${esc(entry.nextReview)}"><p class="reading-save-note">输入自动保存到当前浏览器。换设备前请导出。</p><button class="button small" data-reading-export>导出阅读记录</button></section>`;
}
function renderDetail(params) {
  const paper = params.has('paper'); const id = params.get(paper ? 'paper' : 'report');
  const item = (paper ? papers : reports).find(x => x.id === id);
  if (!item) { $('#paper-space').innerHTML = `<div class="empty"><h1>这篇${paper ? '论文' : '报告'}暂时不存在</h1><a class="button" href="${hub()}">返回论文收录</a></div>`; return; }
  document.title = `${item.title} · 论文收录`;
  const relatedReports = paper ? reports.filter(r => r.paperIds.includes(id)) : [];
  const entry = state(id);
  $('#paper-space').innerHTML = `<a class="back-link" href="${hub(paper ? 'library' : item.type)}">← 返回${paper ? '论文库' : tabs[item.type]}</a><header class="reader-header"><p class="eyebrow">${paper ? `${esc(directionName(item.direction))} / ${item.year} / ${esc(item.evidence)}` : `${tabs[item.type]} / ${item.date}`}</p><h1 tabindex="-1">${esc(item.title)}</h1><p>${esc(item.summary)}</p>${paper ? `<p class="paper-original-title">${esc(item.paperTitle)}<br>${esc(item.authors.join(' · '))}</p>` : ''}<div class="reader-meta"><span>更新 ${esc(item.updated)} · 正文约 ${item.minutes} 分钟（不含原文精读）</span>${paper ? `<a class="button small" href="${esc(item.paperUrl)}" target="_blank" rel="noopener noreferrer">打开论文原文 ↗</a><button class="button small" data-paper-star="${id}" aria-pressed="${entry.starred}">${entry.starred ? '★ 已收藏' : '☆ 收藏论文'}</button>` : ''}<a class="button small" href="https://github.com/Tammytcl/laoliu-interview-notes/blob/main/${esc(item.source)}" target="_blank" rel="noopener noreferrer">原文件 ↗</a></div></header>${paper ? '<p class="notice">这是资料整理的精读报告，不代表你已读完原文。先写读前问题，再读报告；作者结论、整理者判断和未验证内容在正文中分开。</p><button class="button" data-start-reading>先写读前问题，再读报告</button>' : `<section class="paper-related"><h2>本报告关联的论文</h2><div>${relatedPaperLinks(item.paperIds) || '<p>本次没有新增论文，可在正文记录原因与已有阅读进展。</p>'}</div></section>`}<div class="reader-grid paper-reader-grid"><div><details class="paper-body" id="paper-body" open><summary>${paper ? '精读报告正文' : '报告正文'} · 可收起后自行复述</summary><article class="prose">${item.html}</article></details>${paper && relatedReports.length ? `<section class="paper-related"><h2>收录与调研记录</h2>${relatedReports.map(r => `<p><a href="#report=${r.id}">${tabs[r.type]} · ${r.date} · ${esc(r.title)} ↗</a></p>`).join('')}</section>` : ''}</div><aside class="reader-aside">${paper ? readingWorkbench(item) : ''}<nav class="toc" aria-label="报告目录"><span class="eyebrow">本文目录</span>${item.toc.map(t => `<button data-section="${t.id}" class="level-${t.level}">${esc(t.title)}</button>`).join('')}</nav></aside></div>`;
}
export function renderPaperLibrary(params) {
  if (!isPaperRoute(params)) return false;
  $('#paper-space').hidden = false;
  $('#breadcrumb').textContent = libraryName;
  if (params.has('paper') || params.has('report')) renderDetail(params); else renderHub(params);
  return true;
}
function exportReading() {
  const blob = new Blob([JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), reading }, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `paper-reading-${new Date().toISOString().slice(0, 10)}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function initPaperLibrary(data, toast) {
  libraryName = data.config?.paperLibrary?.name || '论文收录';
  papers = data.papers || []; reports = data.reports || []; directions = data.paperDirections || []; notify = toast;
  try { reading = validateReading(JSON.parse(localStorage.getItem(key) || '{}')); }
  catch { notify('无法读取阅读记录；可以从导出备份恢复。'); }
  $('#paper-space').addEventListener('input', e => {
    const input = e.target.closest('[data-reading-field]'); if (!input) return;
    if (input.dataset.readingField === 'nextReview' && !dateValid(input.value)) return;
    save(input.dataset.paperId, { [input.dataset.readingField]: input.value });
    for (const button of document.querySelectorAll('[data-paper-stage]')) button.setAttribute('aria-pressed', String(button.dataset.paperStage === state(input.dataset.paperId).stage));
  });
  $('#paper-space').addEventListener('click', e => {
    const star = e.target.closest('[data-paper-star]');
    if (star) { const id = star.dataset.paperStar; save(id, { starred: !state(id).starred }); refreshDetailState(); }
    const button = e.target.closest('[data-paper-stage]');
    if (button) {
      const id = button.dataset.paperId;
      if (button.dataset.paperStage === 'recalled' && !state(id).recall.trim()) { notify('先写一段自己的复述，再标记已复述。'); return; }
      save(id, { stage: button.dataset.paperStage }); refreshDetailState();
    }
    if (e.target.closest('[data-start-reading]')) { $('#paper-body').open = false; $('#reading-question')?.focus(); }
    if (e.target.closest('[data-close-paper]')) { $('#paper-body').open = false; $('#reading-recall')?.focus(); }
    if (e.target.closest('[data-reading-export]')) exportReading();
    if (e.target.closest('[data-reading-import]')) $('#reading-import-file')?.click();
    if (e.target.closest('[data-paper-reset]')) { query = ''; direction = ''; stage = ''; starredOnly = false; const params = new URLSearchParams(location.hash.slice(1)); params.delete('direction'); renderPaperLibrary(params); }
  });
  $('#paper-space').addEventListener('change', async e => {
    if (e.target.id !== 'reading-import-file') return;
    const file = e.target.files[0]; if (!file) return;
    try {
      if (file.size > 10_000_000) throw new Error('阅读备份过大');
      const value = JSON.parse(await file.text()); if (value.version !== 1) throw new Error('不支持的备份版本');
      const incoming = validateReading(value.reading);
      if (!confirm(`导入 ${Object.keys(incoming).length} 条阅读记录？同 id 的记录将以备份为准。`)) return;
      reading = { ...reading, ...incoming }; localStorage.setItem(key, JSON.stringify(reading)); renderPaperLibrary(new URLSearchParams(location.hash.slice(1))); notify('阅读记录已合并');
    } catch (error) { notify(`导入失败：${error.message}`); }
    finally { e.target.value = ''; }
  });
}
