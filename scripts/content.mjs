import MarkdownIt from 'markdown-it';
import { parse } from 'yaml';
import katex from 'katex';

const md = new MarkdownIt({ html: false, linkify: true, typographer: false });
const mathHtml = (source, displayMode) => katex.renderToString(source, { displayMode, throwOnError: false, trust: false, strict: 'ignore', maxExpand: 1000 });
md.inline.ruler.before('escape', 'math_inline', (state, silent) => {
  const start = state.pos;
  if (state.src[start] !== '$' || state.src[start + 1] === '$' || /\s/.test(state.src[start + 1] || ' ')) return false;
  let end = start + 1;
  while ((end = state.src.indexOf('$', end)) !== -1 && state.src[end - 1] === '\\') end++;
  if (end < 0 || /\s/.test(state.src[end - 1]) || state.src.slice(start, end).includes('\n')) return false;
  if (!silent) { const token = state.push('math_inline', '', 0); token.content = state.src.slice(start + 1, end); }
  state.pos = end + 1; return true;
});
md.block.ruler.before('fence', 'math_block', (state, start, end, silent) => {
  const line = state.src.slice(state.bMarks[start] + state.tShift[start], state.eMarks[start]).trim();
  if (!line.startsWith('$$')) return false;
  if (line.length > 4 && line.endsWith('$$')) {
    if (silent) return true;
    const token = state.push('math_block', '', 0); token.block = true;
    token.content = line.slice(2, -2).trim(); token.map = [start, start + 1];
    state.line = start + 1; return true;
  }
  if (line !== '$$') return false;
  let next = start + 1;
  while (next < end && state.src.slice(state.bMarks[next] + state.tShift[next], state.eMarks[next]).trim() !== '$$') next++;
  if (next === end) return false;
  if (silent) return true;
  const token = state.push('math_block', '', 0); token.block = true;
  token.content = state.getLines(start + 1, next, 0, false); token.map = [start, next + 1];
  state.line = next + 1; return true;
});
md.renderer.rules.math_inline = (tokens, i) => mathHtml(tokens[i].content, false);
md.renderer.rules.math_block = (tokens, i) => `<div class="math-block">${mathHtml(tokens[i].content, true)}</div>\n`;
const imageRule = md.renderer.rules.image;
md.renderer.rules.image = (tokens, i, options, env, renderer) => {
  tokens[i].attrSet('loading', 'lazy'); tokens[i].attrSet('decoding', 'async');
  return imageRule(tokens, i, options, env, renderer);
};
export const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

export function renderMarkdown(body) {
  const tokens = md.parse(body, {});
  const toc = [];
  tokens.forEach((token, i) => {
    if (token.type === 'heading_open') {
      const id = `section-${toc.length + 1}`;
      token.attrSet('id', id);
      toc.push({ id, title: tokens[i + 1].content, level: Number(token.tag.slice(1)) });
    }
  });
  return { html: md.renderer.render(tokens, md.options, {}), toc };
}

export function parseQuestion(source, file, categories) {
  const match = source.replace(/\r\n/g, '\n').match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) throw new Error(`${file}: 缺少 YAML frontmatter`);
  const data = parse(match[1]);
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error(`${file}: 元数据必须为对象`);
  for (const key of ['id', 'title', 'category', 'difficulty', 'updated', 'summary']) {
    if (typeof data[key] !== 'string' || !data[key].trim()) throw new Error(`${file}: ${key} 必须为非空字符串`);
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(data.id)) throw new Error(`${file}: id 请使用小写英文、数字和连字符`);
  if (!categories.some(c => c.id === data.category)) throw new Error(`${file}: 未定义的分类 ${data.category}`);
  if (!['基础', '进阶', '深入'].includes(data.difficulty)) throw new Error(`${file}: difficulty 应为基础 / 进阶 / 深入`);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(data.updated) || !Number.isFinite(Date.parse(data.updated)) || new Date(data.updated).toISOString().slice(0, 10) !== data.updated) throw new Error(`${file}: updated 应为有效 YYYY-MM-DD 日期`);
  if (data.draft !== undefined && typeof data.draft !== 'boolean') throw new Error(`${file}: draft 应为布尔值`);
  if (!Array.isArray(data.tags) || data.tags.some(t => typeof t !== 'string' || !t.trim())) throw new Error(`${file}: tags 应为字符串数组`);
  const body = match[2].trim();
  if (!body) throw new Error(`${file}: 正文不能为空`);
  const rendered = renderMarkdown(body);
  return {
    id: data.id, title: data.title, category: data.category, difficulty: data.difficulty,
    updated: data.updated, summary: data.summary, tags: [...new Set(data.tags)],
    draft: data.draft === true, body, ...rendered, source: file,
    minutes: Math.max(1, Math.ceil(body.length / 450))
  };
}
