const $ = selector => document.querySelector(selector);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
let config, questions = [], category = '', toastTimer, paperCount = 0, reportCount = 0;
function toast(message) {
  $('#toast').textContent = message; $('#toast').classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('show'), 3500);
}
const categoryName = id => config.categories.find(c => c.id === id)?.name || id;
function renderSidebar() {
  const inPapers = isPaperRoute(new URLSearchParams(location.hash.slice(1)));
  $('#categories').innerHTML = `<button data-category="" class="nav-item ${!inPapers && category === '' ? 'active' : ''}" aria-pressed="${!inPapers && category === ''}"><span class="nav-icon">▦</span><span>全部笔记</span><small>${questions.length}</small></button>` + config.categories.map((c, i) => `<button data-category="${esc(c.id)}" class="nav-item ${!inPapers && category === c.id ? 'active' : ''}" aria-pressed="${!inPapers && category === c.id}"><span class="nav-icon">${String(i + 1).padStart(2, '0')}</span><span>${esc(c.name)}</span><small>${questions.filter(q => q.category === c.id).length}</small></button>`).join('') + `<a href="#view=papers" data-paper-nav class="nav-item ${inPapers ? 'active' : ''}" ${inPapers ? 'aria-current="page"' : ''}><span class="nav-icon">▤</span><span>${esc(config.paperLibrary?.name || '论文收录')}</span><small>${paperCount}</small></a>`;
}
function renderStats() {
  $('#stat-total').textContent = String(questions.length).padStart(2, '0');
  $('#stat-categories').textContent = String(config.categories.length).padStart(2, '0');
  $('#stat-papers').textContent = String(paperCount).padStart(2, '0');
  $('#stat-reports').textContent = String(reportCount).padStart(2, '0');
}
function filteredQuestions() {
  const query = $('#search').value.trim().toLocaleLowerCase();
  return questions.filter(q => (!category || q.category === category)
    && (!$('#difficulty').value || q.difficulty === $('#difficulty').value)
    && (!$('#tag').value || q.tags.includes($('#tag').value))
    && (!query || `${q.title} ${q.summary} ${q.tags.join(' ')} ${q.body}`.toLocaleLowerCase().includes(query)));
}
function renderList() {
  const result = filteredQuestions();
  $('#list-title').innerHTML = `${esc(category ? categoryName(category) : '全部笔记')} <span id="result-count">${result.length} 篇</span>`;
  $('#breadcrumb').textContent = category ? categoryName(category) : '全部笔记';
  $('#question-list').innerHTML = result.length ? result.map((q, i) => {
    return `<article class="question-card"><span class="question-number">${String(i + 1).padStart(2, '0')}</span><div class="question-copy"><div class="question-meta"><span>${esc(categoryName(q.category))}</span><i>·</i><span class="difficulty">${esc(q.difficulty)}</span></div><h3><a href="#q=${q.id}">${esc(q.title)}</a></h3><p>${esc(q.summary)}</p><div class="question-bottom"><div class="tags">${q.tags.map(t => `<span># ${esc(t)}</span>`).join('')}</div><span class="updated">${esc(q.updated)} · ${q.minutes} 分钟</span></div></div><div class="card-actions"><a class="read-arrow" href="#q=${q.id}" aria-label="阅读：${esc(q.title)}">↗</a></div></article>`;
  }).join('') : '<div class="empty"><span>⌕</span><h3>这里暂时没有匹配的问题</h3><p>试试其他关键词，或清除筛选后继续浏览。</p><button class="button" id="reset-filters">清除筛选</button></div>';
  $('#reset-filters')?.addEventListener('click', resetFilters);

}
function resetFilters() {
  category = ''; $('#search').value = ''; $('#difficulty').value = ''; $('#tag').value = '';
  renderSidebar(); renderList();
}
function route() {
  const params = new URLSearchParams(location.hash.slice(1));
  renderSidebar();
  if (isPaperRoute(params)) {
    $('#library').hidden = true; $('#reader').hidden = true; $('#reader').innerHTML = '';
    document.title = `论文收录 · ${config.title}`;
    renderPaperLibrary(params); return;
  }
  $('#paper-space').hidden = true; $('#paper-space').innerHTML = '';
  const id = params.get('q');
  $('#library').hidden = !!id; $('#reader').hidden = !id;
  if (!id) { document.title = config.title; renderList(); return; }
  const q = questions.find(q => q.id === id);
  if (!q) { $('#reader').innerHTML = '<div class="empty"><h1>这道题暂时不存在</h1><p>它可能仍是草稿，或链接中的 id 已发生变化。</p><a href="#" class="button">返回题库</a></div>'; return; }
  document.title = `${q.title} · ${config.title}`; $('#breadcrumb').textContent = categoryName(q.category);
  $('#reader').innerHTML = `<a href="#" class="back-link">← 返回问题列表</a><header class="reader-header"><p class="eyebrow">${esc(categoryName(q.category))} / ${esc(q.difficulty)}</p><h1 tabindex="-1">${esc(q.title)}</h1><p>${esc(q.summary)}</p><div class="reader-meta"><span>更新于 ${esc(q.updated)} · 约 ${q.minutes} 分钟</span></div></header><div class="reader-grid"><article class="prose">${q.html}</article><aside class="reader-aside"><nav class="toc" aria-label="文章目录"><span class="eyebrow">本文目录</span>${q.toc.map(t => `<button data-section="${t.id}" class="level-${t.level}">${esc(t.title)}</button>`).join('')}</nav></aside></div><div class="reader-end"><span>能不用看笔记，再讲一遍吗？</span><button id="next-question" class="button">下一道问题 →</button></div>`;
  $('#next-question').addEventListener('click', () => {
    const pool = filteredQuestions(); const list = pool.length ? pool : questions;
    const current = list.findIndex(x => x.id === id); location.hash = `q=${list[(current + 1) % list.length].id}`;
  });
}
function openGuide() {
  if (isPaperRoute(new URLSearchParams(location.hash.slice(1)))) location.hash = 'view=papers&tab=guide';
  else $('#guide').showModal();
}
try {
  const response = await fetch('./__DATA_FILE__');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  ({ config, questions } = data);
  paperCount = data.papers?.length || 0; reportCount = data.reports?.length || 0;
  initPaperLibrary(data, toast);
  $('#brand-title').textContent = config.title; $('#footer-title').textContent = `${config.owner} / ${config.title}`;
  $('.side-label span').textContent = `01 — ${String(config.categories.length + 1).padStart(2, '0')}`;
  $('#subtitle').textContent = config.subtitle; $('#category-help').textContent = config.categories.map(c => c.id).join('、');
  $('#tag').innerHTML += [...new Set(questions.flatMap(q => q.tags))].sort((a, b) => a.localeCompare(b, 'zh-CN')).map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
  renderSidebar(); renderStats(); route();
  $('#categories').addEventListener('click', e => {
    const button = e.target.closest('[data-category]'); if (!button) return;
    category = button.dataset.category; renderSidebar();
    if (location.hash) location.hash = ''; else renderList();
  });
  for (const id of ['search', 'difficulty', 'tag']) $(`#${id}`).addEventListener(id === 'search' ? 'input' : 'change', renderList);
  document.addEventListener('click', e => {
    const toc = e.target.closest('[data-section]');
    if (toc) document.getElementById(toc.dataset.section)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  window.addEventListener('hashchange', () => { route(); window.scrollTo(0, 0); ($('#reader h1') || $('#paper-space h1'))?.focus({ preventScroll: true }); });
  $('#random').addEventListener('click', () => {
    const pool = filteredQuestions(); if (!pool.length) { toast('当前筛选没有题目，清除筛选后再试。'); return; }
    location.hash = `q=${pool[Math.floor(Math.random() * pool.length)].id}`;
  });
  $('#add').addEventListener('click', openGuide); $('#writing-guide').addEventListener('click', openGuide);
  $('#close-guide').addEventListener('click', () => $('#guide').close());
  $('#guide').addEventListener('click', e => { if (e.target === $('#guide')) { const r = $('#guide').getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) $('#guide').close(); } });
  document.addEventListener('keydown', e => {
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) && !$('#guide').open && isPaperRoute(new URLSearchParams(location.hash.slice(1)))) {
      e.preventDefault();
      if (!$('#paper-search')) location.hash = 'view=papers';
      setTimeout(() => $('#paper-search')?.focus(), 0); return;
    }
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) && !$('#guide').open) { e.preventDefault(); if (location.hash) location.hash = ''; setTimeout(() => $('#search').focus(), 0); }
    if (e.key === 'Escape' && !$('#guide').open && !$('#paper-figure-dialog')?.open && location.hash) location.hash = $('#paper-space .back-link')?.getAttribute('href') || '';
  });

} catch (error) {
  $('#question-list').innerHTML = `<div class="empty"><h3>知识库暂时无法加载</h3><p>${esc(error.message)}</p><p>请运行 npm run build 和 npm run preview；不要直接双击 HTML 文件。</p><button class="button" id="retry">重新加载</button></div>`;
  $('#retry').addEventListener('click', () => location.reload());
}
import { initPaperLibrary, renderPaperLibrary, isPaperRoute } from './__PAPERS_MODULE__';
