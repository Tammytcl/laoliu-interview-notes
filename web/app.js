const $ = selector => document.querySelector(selector);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const labels = { new: '未开始', review: '复习中', mastered: '已掌握' };
const storageKey = `interview-notes:v1:${location.pathname}`;
let config, questions = [], progress = {}, category = '', filter = 'all', toastTimer;
let storageAvailable = true;
function toast(message) {
  $('#toast').textContent = message; $('#toast').classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('show'), 3500);
}
function validProgress(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('无效的进度对象');
  const clean = {};
  for (const [id, item] of Object.entries(value)) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id) || !item || typeof item !== 'object' || !Object.hasOwn(labels, item.status) || typeof item.starred !== 'boolean') throw new Error('进度文件格式不正确');
    clean[id] = { status: item.status, starred: item.starred };
  }
  return clean;
}
try { progress = validProgress(JSON.parse(localStorage.getItem(storageKey) || '{}')); }
catch { storageAvailable = false; }
const stateOf = id => progress[id] || { status: 'new', starred: false };
function save(id, patch) {
  progress[id] = { ...stateOf(id), ...patch };
  try { localStorage.setItem(storageKey, JSON.stringify(progress)); }
  catch { toast('浏览器无法保存进度，请用“导出进度”备份。'); }
  renderSidebar(); renderStats();
}
const categoryName = id => config.categories.find(c => c.id === id)?.name || id;
function renderSidebar() {
  $('#categories').innerHTML = `<button data-category="" class="nav-item ${category === '' ? 'active' : ''}" aria-pressed="${category === ''}"><span class="nav-icon">▦</span><span>全部笔记</span><small>${questions.length}</small></button>` + config.categories.map((c, i) => `<button data-category="${esc(c.id)}" class="nav-item ${category === c.id ? 'active' : ''}" aria-pressed="${category === c.id}"><span class="nav-icon">${String(i + 1).padStart(2, '0')}</span><span>${esc(c.name)}</span><small>${questions.filter(q => q.category === c.id).length}</small></button>`).join('');
  const count = questions.filter(q => stateOf(q.id).status === 'mastered').length;
  $('#progress').max = Math.max(questions.length, 1); $('#progress').value = count;
  $('#progress-count').textContent = `${count} / ${questions.length}`;
}
function renderStats() {
  $('#stat-total').textContent = String(questions.length).padStart(2, '0');
  $('#stat-review').textContent = String(questions.filter(q => stateOf(q.id).status === 'review').length).padStart(2, '0');
  $('#stat-mastered').textContent = String(questions.filter(q => stateOf(q.id).status === 'mastered').length).padStart(2, '0');
  $('#stat-starred').textContent = String(questions.filter(q => stateOf(q.id).starred).length).padStart(2, '0');
}
function filteredQuestions() {
  const query = $('#search').value.trim().toLocaleLowerCase();
  return questions.filter(q => (!category || q.category === category)
    && (filter === 'all' || (filter === 'starred' ? stateOf(q.id).starred : stateOf(q.id).status === filter))
    && (!$('#difficulty').value || q.difficulty === $('#difficulty').value)
    && (!$('#tag').value || q.tags.includes($('#tag').value))
    && (!query || `${q.title} ${q.summary} ${q.tags.join(' ')} ${q.body}`.toLocaleLowerCase().includes(query)));
}
function renderList() {
  const result = filteredQuestions();
  $('#list-title').innerHTML = `${esc(category ? categoryName(category) : '全部笔记')} <span id="result-count">${result.length} 篇</span>`;
  $('#breadcrumb').textContent = category ? categoryName(category) : '全部笔记';
  $('#question-list').innerHTML = result.length ? result.map((q, i) => {
    const state = stateOf(q.id);
    return `<article class="question-card"><span class="question-number">${String(i + 1).padStart(2, '0')}</span><div class="question-copy"><div class="question-meta"><span>${esc(categoryName(q.category))}</span><i>·</i><span class="difficulty">${esc(q.difficulty)}</span></div><h3><a href="#q=${q.id}">${esc(q.title)}</a></h3><p>${esc(q.summary)}</p><div class="question-bottom"><div class="tags">${q.tags.map(t => `<span># ${esc(t)}</span>`).join('')}</div><span class="updated">${esc(q.updated)} · ${q.minutes} 分钟</span></div></div><div class="card-actions"><button class="star ${state.starred ? 'selected' : ''}" data-star="${q.id}" aria-label="${state.starred ? '取消收藏' : '收藏'}：${esc(q.title)}" aria-pressed="${state.starred}">${state.starred ? '★' : '☆'}</button><span class="status ${state.status}">${labels[state.status]}</span><a class="read-arrow" href="#q=${q.id}" aria-label="阅读：${esc(q.title)}">↗</a></div></article>`;
  }).join('') : '<div class="empty"><span>⌕</span><h3>这里暂时没有匹配的问题</h3><p>试试其他关键词，或清除筛选后继续浏览。</p><button class="button" id="reset-filters">清除筛选</button></div>';
  $('#reset-filters')?.addEventListener('click', resetFilters);
  for (const button of document.querySelectorAll('[data-filter]')) {
    button.classList.toggle('active', button.dataset.filter === filter);
    button.setAttribute('aria-pressed', String(button.dataset.filter === filter));
  }
}
function resetFilters() {
  category = ''; filter = 'all'; $('#search').value = ''; $('#difficulty').value = ''; $('#tag').value = '';
  renderSidebar(); renderList();
}
function route() {
  const id = new URLSearchParams(location.hash.slice(1)).get('q');
  $('#library').hidden = !!id; $('#reader').hidden = !id;
  if (!id) { document.title = config.title; renderList(); return; }
  const q = questions.find(q => q.id === id);
  if (!q) { $('#reader').innerHTML = '<div class="empty"><h1>这道题暂时不存在</h1><p>它可能仍是草稿，或链接中的 id 已发生变化。</p><a href="#" class="button">返回题库</a></div>'; return; }
  const state = stateOf(id);
  document.title = `${q.title} · ${config.title}`; $('#breadcrumb').textContent = categoryName(q.category);
  $('#reader').innerHTML = `<a href="#" class="back-link">← 返回问题列表</a><header class="reader-header"><p class="eyebrow">${esc(categoryName(q.category))} / ${esc(q.difficulty)}</p><h1 tabindex="-1">${esc(q.title)}</h1><p>${esc(q.summary)}</p><div class="reader-meta"><span>更新于 ${esc(q.updated)} · 约 ${q.minutes} 分钟</span><button data-star="${q.id}" class="button small" aria-pressed="${state.starred}">${state.starred ? '★ 已收藏' : '☆ 收藏问题'}</button></div></header><div class="reader-grid"><article class="prose">${q.html}</article><aside class="reader-aside"><div class="review-box"><span class="eyebrow">MY PROGRESS</span><h3>这个问题，我…</h3>${Object.entries(labels).map(([key, label]) => `<button data-status="${key}" data-id="${id}" class="review-option ${state.status === key ? 'selected' : ''}" aria-pressed="${state.status === key}"><span>${key === 'mastered' ? '✓' : key === 'review' ? '↻' : '○'}</span>${label}</button>`).join('')}<p>状态仅保存在本浏览器。</p></div><nav class="toc" aria-label="文章目录"><span class="eyebrow">本文目录</span>${q.toc.map(t => `<button data-section="${t.id}" class="level-${t.level}">${esc(t.title)}</button>`).join('')}</nav></aside></div><div class="reader-end"><span>能不用看笔记，再讲一遍吗？</span><button id="next-question" class="button">下一道问题 →</button></div>`;
  $('#next-question').addEventListener('click', () => {
    const pool = filteredQuestions(); const list = pool.length ? pool : questions;
    const current = list.findIndex(x => x.id === id); location.hash = `q=${list[(current + 1) % list.length].id}`;
  });
}
function openGuide() { $('#guide').showModal(); }
function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
try {
  const response = await fetch('./data.json');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  ({ config, questions } = await response.json());
  $('#brand-title').textContent = config.title; $('#footer-title').textContent = `${config.owner} / ${config.title}`;
  $('.side-label span').textContent = `01 — ${String(config.categories.length).padStart(2, '0')}`;
  $('#subtitle').textContent = config.subtitle; $('#category-help').textContent = config.categories.map(c => c.id).join('、');
  $('#tag').innerHTML += [...new Set(questions.flatMap(q => q.tags))].sort((a, b) => a.localeCompare(b, 'zh-CN')).map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
  renderSidebar(); renderStats(); route();
  if (!storageAvailable) toast('未能读取本地进度；请检查浏览器存储设置或导入备份。');
  $('#categories').addEventListener('click', e => {
    const button = e.target.closest('[data-category]'); if (!button) return;
    category = button.dataset.category; renderSidebar();
    if (location.hash) location.hash = ''; else renderList();
  });
  $('#status-tabs').addEventListener('click', e => {
    const button = e.target.closest('[data-filter]'); if (button) { filter = button.dataset.filter; renderList(); }
  });
  for (const id of ['search', 'difficulty', 'tag']) $(`#${id}`).addEventListener(id === 'search' ? 'input' : 'change', renderList);
  document.addEventListener('click', e => {
    const star = e.target.closest('[data-star]');
    if (star) { const id = star.dataset.star; save(id, { starred: !stateOf(id).starred }); route(); }
    const status = e.target.closest('[data-status]');
    if (status) { save(status.dataset.id, { status: status.dataset.status }); route(); toast('复习状态已更新'); }
    const toc = e.target.closest('[data-section]');
    if (toc) document.getElementById(toc.dataset.section)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  window.addEventListener('hashchange', () => { route(); window.scrollTo(0, 0); $('#reader h1')?.focus({ preventScroll: true }); });
  $('#random').addEventListener('click', () => {
    const pool = filteredQuestions(); if (!pool.length) { toast('当前筛选没有题目，清除筛选后再试。'); return; }
    location.hash = `q=${pool[Math.floor(Math.random() * pool.length)].id}`;
  });
  $('#add').addEventListener('click', openGuide); $('#writing-guide').addEventListener('click', openGuide);
  $('#close-guide').addEventListener('click', () => $('#guide').close());
  $('#guide').addEventListener('click', e => { if (e.target === $('#guide')) { const r = $('#guide').getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) $('#guide').close(); } });
  document.addEventListener('keydown', e => {
    if (e.key === '/' && !e.ctrlKey && !e.metaKey && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName) && !$('#guide').open) { e.preventDefault(); if (location.hash) location.hash = ''; setTimeout(() => $('#search').focus(), 0); }
    if (e.key === 'Escape' && !$('#guide').open && location.hash) location.hash = '';
  });
  $('#export').addEventListener('click', () => download(`interview-progress-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), progress }, null, 2)));
  $('#import').addEventListener('click', () => $('#import-file').click());
  $('#import-file').addEventListener('change', async e => {
    const file = e.target.files[0]; if (!file) return;
    try {
      if (file.size > 2_000_000) throw new Error('进度文件过大');
      const value = JSON.parse(await file.text()); if (value.version !== 1) throw new Error('不支持的备份版本');
      const incoming = validProgress(value.progress);
      if (!confirm(`导入 ${Object.keys(incoming).length} 条记录？同 id 的本地状态会被备份覆盖。`)) return;
      progress = { ...progress, ...incoming }; localStorage.setItem(storageKey, JSON.stringify(progress));
      renderSidebar(); renderStats(); route(); toast('进度已合并导入');
    } catch (error) { toast(`导入失败：${error.message}`); }
    finally { e.target.value = ''; }
  });
} catch (error) {
  $('#question-list').innerHTML = `<div class="empty"><h3>知识库暂时无法加载</h3><p>${esc(error.message)}</p><p>请运行 npm run build 和 npm run preview；不要直接双击 HTML 文件。</p><button class="button" id="retry">重新加载</button></div>`;
  $('#retry').addEventListener('click', () => location.reload());
}
