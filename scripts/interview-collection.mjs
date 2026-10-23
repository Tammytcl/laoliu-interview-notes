import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseQuestion } from './content.mjs';

export const collectionPath = 'content/research/post-training-interview-practice-sources.json';
const start = '[//]: # (interview-generated-start)';
const end = '[//]: # (interview-generated-end)';
const cell = text => String(text).replace(/\|/g, '\\|').replace(/[\r\n]/g, ' ');
const label = q => q.title.replace(/^Q\d+\s*[·：:]\s*/, '');

export function validateInterviewCollection(manifest, questions) {
  const published = new Map(questions.filter(q => !q.draft).map(q => [q.id, q]));
  const sources = new Map(), seen = new Set(), concepts = new Set(), numbers = new Set();
  if (!published.has(manifest.topicId)) throw new Error('Interview collection index missing');
  if (!Array.isArray(manifest.questions) || !manifest.questions.length) throw new Error('Empty interview collection');
  for (const source of manifest.sources) {
    if (sources.has(source.id)) throw new Error(`Duplicate interview source: ${source.id}`);
    const url = new URL(source.url);
    if (url.protocol !== 'https:' || url.username || url.password || !source.title || !source.purpose || !source.reviewedAt) throw new Error(`Invalid interview source: ${source.id}`);
    sources.set(source.id, source);
  }
  const groups = new Set(manifest.groups.map(g => g.id));
  if (groups.size !== manifest.groups.length) throw new Error('Duplicate interview group');
  for (const item of manifest.questions) {
    const q = published.get(item.id);
    if (!q || q.category !== 'interview') throw new Error(`Missing published interview question: ${item.id}`);
    if (seen.has(item.id) || concepts.has(item.concept) || numbers.has(item.number)) throw new Error(`Duplicate interview question, number or concept: ${item.id}`);
    if (!/^[a-z0-9-]+$/.test(item.concept || '') || !Number.isInteger(item.number) || item.number < 1 || !groups.has(item.group)) throw new Error(`Invalid interview metadata: ${item.id}`);
    if (!['公开题型改写', '项目自拟'].includes(item.origin)) throw new Error(`Invalid question origin: ${item.id}`);
    if (!item.sourceIds?.length || item.sourceIds.some(id => !sources.has(id))) throw new Error(`Unresolved interview source: ${item.id}`);
    if (item.origin === '公开题型改写' && !item.sourceIds.some(id => sources.get(id).kind === 'community')) throw new Error(`Public question lacks interview evidence: ${item.id}`);
    if (!item.theoryIds?.length || item.theoryIds.some(id => !published.has(id))) throw new Error(`Unresolved interview theory link: ${item.id}`);
    if (!q.title.startsWith(`Q${String(item.number).padStart(2, '0')} · `)) throw new Error(`Question title/number differs: ${item.id}`);
    for (const section of ['60–90 秒回答', '白板与例子', '追问怎么接', '项目怎么联系', '容易答错的点', '原理拆解']) {
      if (!q.toc.some(entry => entry.title === section)) throw new Error(`${item.id}: missing answer section ${section}`);
    }
    if (q.html.includes('katex-error')) throw new Error(`Invalid interview formula: ${item.id}`);
    seen.add(item.id); concepts.add(item.concept); numbers.add(item.number);
  }
  if (questions.some(q => !q.draft && q.category === 'interview' && !seen.has(q.id))) throw new Error('Unregistered interview question');
  return { published, sources };
}

export function generatedInterviewBlocks(manifest, questions) {
  const { published, sources } = validateInterviewCollection(manifest, questions);
  const sorted = [...manifest.questions].sort((a, b) => a.number - b.number);
  const blocks = new Map();
  const rows = [`**当前收录：${sorted.length} 道主问题。** 追问属于同一主问题，不重复计数。`, ''];
  for (const group of manifest.groups) {
    rows.push(`### ${group.title}`, '', '| 题号 | 主问题 | 题源类型 |', '| --- | --- | --- |');
    for (const item of sorted.filter(i => i.group === group.id)) {
      const companies = [...new Set(item.sourceIds.map(id => sources.get(id).company).filter(Boolean))];
      rows.push(`| ${String(item.number).padStart(2, '0')} | [${cell(label(published.get(item.id)))}](#q=${item.id}) | ${cell(companies.length ? companies.join(' / ') + ' · 改写' : item.origin)} |`);
    }
    rows.push('');
  }
  rows.push('## 公司题型怎么查', '', '以下是所选公开帖中的题型映射，不代表某公司的完整题库或出现频率。原帖中的错误技术表述没有沿用。', '', '| 帖子标注的公司 | 公开记录涉及的方向 | 对应练习 | 证据范围 |', '| --- | --- | --- | --- |');
  for (const source of manifest.sources.filter(s => s.kind === 'community')) {
    const items = sorted.filter(q => q.sourceIds.includes(source.id));
    rows.push(`| [${cell(source.company)} · ${cell(source.title)}](${source.url}) | ${cell(source.purpose)} | ${items.map(q => `[Q${String(q.number).padStart(2, '0')}](#q=${q.id})`).join('、')} | ${cell(source.evidence)}；${cell(source.access)} |`);
  }
  rows.push('', '## 来源账本', '', `本轮核对：${manifest.reviewedAt}。面经只支持题型归类；技术解释依据原论文、作者资料与独立推导。未确认年份的日期不自行补全年份。`, '', '| 资料 | 用途 |', '| --- | --- |');
  for (const source of manifest.sources.filter(s => s.kind !== 'community')) rows.push(`| [${cell(source.title)}](${source.url}) | ${cell(source.purpose)} |`);
  blocks.set(manifest.topicId, rows.join('\n'));
  sorted.forEach((item, i) => {
    const selected = item.sourceIds.map(id => sources.get(id));
    const lines = ['## 题源与延伸', '', `**题目归类：${item.origin}。** 下面的追问、算例和口述稿由本站独立整理；不是逐字面试记录。`, '', '| 来源 | 使用范围 |', '| --- | --- |'];
    for (const s of selected) lines.push(`| [${cell(s.title)}](${s.url}) | ${cell(s.kind === 'community' ? s.evidence + '；' + s.access + '；仅作题型依据' : s.purpose)} |`);
    lines.push('', '**相关专题详解：** ' + item.theoryIds.map(id => `[${cell(published.get(id).title)}](#q=${id})`).join(' · '), '', `[返回 ${sorted.length} 题目录](#q=${manifest.topicId})` + (i ? ` · [上一题](#q=${sorted[i - 1].id})` : '') + (i < sorted.length - 1 ? ` · [下一题](#q=${sorted[i + 1].id})` : ''));
    blocks.set(item.id, lines.join('\n'));
  });
  return blocks;
}

export async function checkInterviewCollection(root, questions, { write = false } = {}) {
  let manifest;
  try { manifest = JSON.parse(await readFile(join(root, collectionPath), 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return null; throw error; }
  const blocks = generatedInterviewBlocks(manifest, questions);
  for (const [id, block] of blocks) {
    const q = questions.find(q => q.id === id);
    const file = join(root, q.source), source = await readFile(file, 'utf8');
    const a = source.indexOf(start), b = source.indexOf(end);
    if (a < 0 || b <= a || source.indexOf(start, a + start.length) >= 0) throw new Error(`${id}: missing unique generated block`);
    const replacement = `${start}\n\n${block.trim()}\n\n${end}`;
    const next = source.slice(0, a) + replacement + source.slice(b + end.length);
    if (next !== source) {
      if (write) await writeFile(file, next);
      else throw new Error(`${id}: interview index/source links changed; run npm run interviews:index`);
    }
  }
  return manifest;
}

async function main() {
  const root = fileURLToPath(new URL('../', import.meta.url));
  const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'));
  const questions = [];
  async function walk(path) {
    for (const entry of await readdir(path, { withFileTypes: true })) {
      const file = join(path, entry.name);
      if (entry.isDirectory()) await walk(file);
      else if (entry.name.endsWith('.md')) questions.push(parseQuestion(await readFile(file, 'utf8'), relative(root, file), config.categories));
    }
  }
  await walk(join(root, 'content/questions'));
  const result = await checkInterviewCollection(root, questions, { write: process.argv.includes('--write') });
  console.log(`✓ ${result?.questions.length || 0} 道面试主问题：题号、概念、来源、目录和关联链接一致`);
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) await main();
