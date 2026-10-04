import { readFile, readdir } from 'node:fs/promises';
import { resolve, join, relative, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { parseQuestion } from './content.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
const securePath = (root, path) => {
  const absolute = resolve(root, path), local = relative(root, absolute);
  if (local === '..' || local.startsWith(`..${sep}`) || local.startsWith(sep)) throw new Error(`Unsafe topic path: ${path}`);
  return absolute;
};
export async function auditTopics(root, questions) {
  let topics;
  try { topics = JSON.parse(await readFile(join(root, 'content/research/topics.json'), 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  const errors = [], byId = new Map(questions.filter(q => !q.draft).map(q => [q.id, q]));
  for (const topic of topics) {
    const q = byId.get(topic.id);
    if (!q) { errors.push(`${topic.id}: published topic missing`); continue; }
    const manifest = JSON.parse(await readFile(securePath(root, topic.sources), 'utf8'));
    if (manifest.topicId !== q.id) errors.push(`${q.id}: source manifest topic differs`);
    const ids = new Set();
    const types = new Set(['official-code', 'official-docs', 'author-blog', 'paper', 'community']);
    for (const s of manifest.sources) {
      if (ids.has(s.id)) errors.push(`${q.id}: duplicate source ${s.id}`);
      ids.add(s.id);
      if (!types.has(s.kind) || !/^\d{4}-\d{2}-\d{2}$/.test(s.reviewedAt || '') || !s.purpose || !s.title) errors.push(`${q.id}: incomplete source ${s.id}`);
      if (new URL(s.url).protocol !== 'https:') errors.push(`${q.id}: source must use HTTPS`);
      if (s.repository && (!/^[a-f0-9]{40}$/.test(s.revision || '') || !s.pinnedUrl?.includes(s.revision))) errors.push(`${q.id}: unpinned repository ${s.id}`);
      if (s.watchUrl && !/^[a-f0-9]{64}$/.test(s.reviewedSha256 || '')) errors.push(`${q.id}: missing reviewed document hash ${s.id}`);
    }
    const sources = new Set(manifest.sources.map(s => s.url));
    for (const [, url] of q.body.matchAll(/\]\((https:\/\/[^\s)]+)\)/g)) if (!sources.has(url)) errors.push(`${q.id}: citation not registered: ${url}`);
    for (const section of topic.requiredSections || []) if (!q.toc.some(s => s.title.includes(section))) errors.push(`${q.id}: missing section ${section}`);
    if (topic.minimumCharacters && q.body.length < topic.minimumCharacters) errors.push(`${q.id}: incomplete topic depth`);
    const numbers = [...q.body.matchAll(/\*\*Q(\d{2}) /g)].map(m => m[1]);
    if (topic.minimumQuestions && (numbers.length < topic.minimumQuestions || new Set(numbers).size !== numbers.length)) errors.push(`${q.id}: missing or duplicate numbered questions`);
    if (q.html.includes('katex-error')) errors.push(`${q.id}: invalid formula`);
    const figures = JSON.parse(await readFile(securePath(root, topic.figures), 'utf8')).figures;
    for (const [, alt, path] of q.body.matchAll(/!\[([^\]]*)\]\((\.\/assets\/[^\s)]+)\)/g)) {
      const entry = figures.find(f => path.endsWith('/' + f.file));
      if (!alt || !entry?.sourcePage || !entry?.credit || !entry?.visuallyVerified || entry.explanation !== 'complete') errors.push(`${q.id}: incomplete figure evidence: ${path}`);
      try {
        const bytes = await readFile(securePath(root, path));
        if (!entry || entry.sha256 !== createHash('sha256').update(bytes).digest('hex')) errors.push(`${q.id}: figure hash differs: ${path}`);
      } catch { errors.push(`${q.id}: missing figure: ${path}`); }
    }
  }
  // Check every published note link, including links from unrelated learning paths.
  for (const q of questions.filter(q => !q.draft)) {
    for (const [, id] of q.body.matchAll(/\]\(#q=([a-z0-9-]+)(?:&[^)]*)?\)/g)) if (!byId.has(id)) errors.push(`${q.id}: unresolved note link ${id}`);
  }
  return errors;
}
async function main() {
  const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'));
  const questions = [];
  async function walk(path) {
    for (const file of await readdir(path, { withFileTypes: true })) {
      if (file.isDirectory()) await walk(join(path, file.name));
      else if (file.name.endsWith('.md')) {
        const p = join(path, file.name);
        questions.push(parseQuestion(await readFile(p, 'utf8'), relative(root, p), config.categories));
      }
    }
  }
  await walk(join(root, 'content/questions'));
  const errors = await auditTopics(root, questions);
  if (errors.length) throw new Error(errors.join('\n'));
  console.log('✓ 专题来源、固定版本、图表、问答与笔记链接检查通过（事实正确性仍需阅读核对）');
}
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) await main();
