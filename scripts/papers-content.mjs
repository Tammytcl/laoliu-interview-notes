import { parse } from 'yaml';
import { renderMarkdown } from './content.mjs';

export const paperDirections = [
  { id: 'diffusion', name: 'Diffusion' }, { id: 'llm', name: 'LLM' },
  { id: 'agent', name: 'Agent' }, { id: 'infra', name: 'Infra' }
];
export const evidenceLevels = ['资料整理', '已核原文', '已复现'];
const validId = value => typeof value === 'string' && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value);
const validDate = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)
  && Number.isFinite(Date.parse(value)) && new Date(value).toISOString().slice(0, 10) === value;

export function parsePaperContent(source, file, collection) {
  const fail = message => { throw new Error(`${file}: ${message}`); };
  const match = source.replace(/\r\n/g, '\n').match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) fail('缺少 YAML frontmatter');
  const data = parse(match[1]);
  if (!data || typeof data !== 'object' || Array.isArray(data)) fail('元数据必须为对象');
  for (const key of ['id', 'title', 'summary']) {
    if (typeof data[key] !== 'string' || !data[key].trim()) fail(`${key} 必须为非空字符串`);
  }
  if (!validId(data.id)) fail('id 请使用小写英文、数字和连字符');
  if (!validDate(data.updated)) fail('updated 应为有效 YYYY-MM-DD 日期');
  if (data.draft !== undefined && typeof data.draft !== 'boolean') fail('draft 应为布尔值');
  if (!Array.isArray(data.tags) || data.tags.some(t => typeof t !== 'string' || !t.trim())) fail('tags 应为字符串数组');
  if (data.template_version !== undefined && ![1, 2].includes(data.template_version)) fail('暂不支持此 template_version');
  const body = match[2].trim();
  if (!body) fail('正文不能为空');
  const result = { id: data.id, title: data.title, summary: data.summary, updated: data.updated,
    tags: [...new Set(data.tags)], draft: data.draft === true, body, ...renderMarkdown(body), source: file,
    minutes: Math.max(1, Math.ceil(body.length / 450)), templateVersion: data.template_version ?? 1 };
  if (collection === 'papers') {
    if (!paperDirections.some(d => d.id === data.direction)) fail('direction 应为 diffusion / llm / agent / infra');
    if (typeof data.paper_title !== 'string' || !data.paper_title.trim()) fail('paper_title 必须为原论文标题');
    if (!Array.isArray(data.authors) || !data.authors.length || data.authors.some(a => typeof a !== 'string' || !a.trim())) fail('authors 应为非空作者数组');
    if (!Number.isInteger(data.year) || data.year < 1900 || data.year > 2100) fail('year 应为合理年份整数');
    let url;
    try { url = new URL(data.paper_url); } catch { fail('paper_url 应为 HTTPS 原文链接'); }
    if (url.protocol !== 'https:' || url.username || url.password) fail('paper_url 应为 HTTPS 原文链接');
    if (!evidenceLevels.includes(data.evidence)) fail('evidence 应为资料整理 / 已核原文 / 已复现');
    const noteIds = data.note_ids ?? [];
    if (!Array.isArray(noteIds) || noteIds.some(id => !validId(id))) fail('note_ids 应为笔记 id 数组');
    const affiliations = data.affiliations ?? [];
    if (!Array.isArray(affiliations) || affiliations.some(a => typeof a !== 'string' || !a.trim())) fail('affiliations 应为英文单位数组');
    const authorAffiliations = data.author_affiliations ?? [];
    if (!Array.isArray(authorAffiliations) || authorAffiliations.some(a => !Array.isArray(a) || a.some(i => !Number.isInteger(i) || i < 1 || i > affiliations.length)) || (authorAffiliations.length && authorAffiliations.length !== data.authors.length)) fail('author_affiliations 应逐作者列出单位编号');
    let githubUrl = null;
    if (data.github_url) {
      try { githubUrl = new URL(data.github_url); } catch { fail('github_url 无效'); }
      if (githubUrl.protocol !== 'https:' || githubUrl.hostname !== 'github.com' || githubUrl.username || githubUrl.password) fail('github_url 应为 GitHub HTTPS 链接');
    }
    if (data.template_version === 2 && !affiliations.length) fail('v2 报告必须注明 affiliations');
    const venue = data.venue ?? '';
    const codeNote = data.code_note ?? '';
    if (typeof venue !== 'string' || typeof codeNote !== 'string') fail('venue / code_note 应为字符串');
    return { ...result, direction: data.direction, paperTitle: data.paper_title, authors: data.authors,
      affiliations, authorAffiliations, githubUrl: githubUrl?.href ?? null, venue, codeNote,
      year: data.year, paperUrl: url.href, evidence: data.evidence, noteIds: [...new Set(noteIds)] };
  }
  if (collection !== 'reports') fail('未知内容集合');
  if (!['daily', 'survey'].includes(data.type)) fail('type 应为 daily / survey');
  if (!validDate(data.date)) fail('date 应为有效报告日期');
  if (!Array.isArray(data.directions) || !data.directions.length || data.directions.some(id => !paperDirections.some(d => d.id === id))) fail('directions 应为非空方向数组');
  if (!Array.isArray(data.paper_ids) || data.paper_ids.some(id => !validId(id))) fail('paper_ids 应为论文 id 数组');
  return { ...result, type: data.type, date: data.date, directions: [...new Set(data.directions)], paperIds: [...new Set(data.paper_ids)] };
}

export function validatePaperLinks(questions, papers, reports) {
  const ids = new Set();
  for (const item of [...questions, ...papers, ...reports]) {
    if (ids.has(item.id)) throw new Error(`重复内容 id: ${item.id}`);
    ids.add(item.id);
  }
  const noteIds = new Set(questions.filter(q => !q.draft).map(q => q.id));
  const paperIds = new Set(papers.filter(p => !p.draft).map(p => p.id));
  const reportIds = new Set(reports.filter(r => !r.draft).map(r => r.id));
  const originals = new Set();
  for (const paper of papers.filter(p => !p.draft)) {
    const url = new URL(paper.paperUrl);
    const arxiv = /^(?:www\.|export\.)?arxiv\.org$/.test(url.hostname);
    const original = arxiv
      ? `arxiv:${url.pathname.replace(/^\/(?:abs|pdf|html)\//, '').replace(/\.pdf$/, '').replace(/v\d+$/, '')}`
      : `${url.origin}${url.pathname}${url.search}`;
    if (originals.has(original)) throw new Error(`${paper.source}: 同一论文已收录，请更新已有记录 ${original}`);
    originals.add(original);
  }
  const dailyDates = new Set();
  for (const report of reports.filter(r => !r.draft && r.type === 'daily')) {
    if (dailyDates.has(report.date)) throw new Error(`${report.source}: 当天 Daily 已存在 ${report.date}`);
    dailyDates.add(report.date);
  }
  for (const item of [...papers, ...reports].filter(x => !x.draft)) {
    for (const id of item.noteIds ?? []) if (!noteIds.has(id)) throw new Error(`${item.source}: 笔记未发布或不存在 ${id}`);
    for (const id of item.paperIds ?? []) if (!paperIds.has(id)) throw new Error(`${item.source}: 论文未发布或不存在 ${id}`);
    for (const [, type, id] of item.body.matchAll(/\]\(#(q|paper|report)=([a-z0-9-]+)\)/g)) {
      const target = type === 'q' ? noteIds : type === 'paper' ? paperIds : reportIds;
      if (!target.has(id)) throw new Error(`${item.source}: 内容链接未发布或不存在 ${type}=${id}`);
    }
  }
}
