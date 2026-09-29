import MarkdownIt from 'markdown-it';
import { parse } from 'yaml';

const md = new MarkdownIt({ html: false, linkify: true, typographer: false });
export const escapeHtml = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

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
  const tokens = md.parse(body, {});
  const toc = [];
  tokens.forEach((token, i) => {
    if (token.type === 'heading_open') {
      const id = `section-${toc.length + 1}`;
      token.attrSet('id', id);
      toc.push({ id, title: tokens[i + 1].content, level: Number(token.tag.slice(1)) });
    }
  });
  return {
    id: data.id, title: data.title, category: data.category, difficulty: data.difficulty,
    updated: data.updated, summary: data.summary, tags: [...new Set(data.tags)],
    draft: data.draft === true, body, html: md.renderer.render(tokens, md.options, {}), toc, source: file,
    minutes: Math.max(1, Math.ceil(body.length / 450))
  };
}
