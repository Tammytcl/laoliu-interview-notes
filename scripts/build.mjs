import { readFile, readdir, mkdir, copyFile, writeFile, cp, access } from 'node:fs/promises';
import { resolve, relative, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { parseQuestion, escapeHtml } from './content.mjs';
import { parsePaperContent, paperDirections, paperCategories, paperTaxonomy, validatePaperLinks } from './papers-content.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'));
if (!Array.isArray(config.categories) || new Set(config.categories.map(c => c.id)).size !== config.categories.length) throw new Error('分类配置无效或重复');
for (const c of config.categories) {
  if (!/^[a-z0-9-]+$/.test(c.id) || !c.name) throw new Error('分类需要有效 id 和 name');
}
async function walk(dir) {
  const files = [];
  let entries;
  try { entries = await readdir(dir, { withFileTypes: true }); }
  catch (error) { if (error.code === 'ENOENT') return files; throw error; }
  for (const entry of entries) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(p));
    else if (entry.isFile() && entry.name.endsWith('.md')) files.push(p);
  }
  return files.sort();
}
const questions = [], allQuestions = [], seen = new Set();
for (const p of await walk(join(root, 'content/questions'))) {
  const question = parseQuestion(await readFile(p, 'utf8'), relative(root, p).split('\\').join('/'), config.categories);
  if (seen.has(question.id)) throw new Error(`重复题目 id: ${question.id}`);
  seen.add(question.id);
  allQuestions.push(question);
  if (!question.draft) questions.push(question);
}
questions.sort((a, b) => b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title, 'zh-CN'));
const collections = {};
for (const name of ['papers', 'reports']) {
  collections[name] = [];
  for (const p of await walk(join(root, 'content', name))) {
    collections[name].push(parsePaperContent(await readFile(p, 'utf8'), relative(root, p).split('\\').join('/'), name));
  }
}
validatePaperLinks(allQuestions, collections.papers, collections.reports);
const papers = collections.papers.filter(p => !p.draft).sort((a, b) => (b.published || `${b.year}-01-01`).localeCompare(a.published || `${a.year}-01-01`) || a.id.localeCompare(b.id));
const reports = collections.reports.filter(r => !r.draft).sort((a, b) => b.date.localeCompare(a.date) || b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title, 'zh-CN'));
const dist = resolve(root, 'dist');
// 图表是报告的长期资产；缓存源码不会进入发布目录。
for (const paper of papers) {
  if (paper.methodFigure) await access(join(root, paper.methodFigure));
  for (const [, path] of paper.body.matchAll(/!\[[^\]]*\]\((\.\/assets\/papers\/[^\s)]+)(?:\s+"[^"]*")?\)/g)) {
    if (path.includes('..')) throw new Error(`${paper.source}: 图表路径不得越界`);
    await access(join(root, path));
  }
}
await mkdir(dist, { recursive: true });
try { await cp(join(root, 'assets'), join(dist, 'assets'), { recursive: true }); } catch (error) { if (error.code !== 'ENOENT') throw error; }
const katexVersion = JSON.parse(await readFile(join(root, 'node_modules/katex/package.json'), 'utf8')).version;
const mathPath = `vendor/katex-${katexVersion}`;
await cp(join(root, 'node_modules/katex/dist'), join(dist, mathPath), { recursive: true });
const fingerprint = text => createHash('sha256').update(text).digest('hex').slice(0, 16);
let citations = {};
try { citations = JSON.parse(await readFile(join(root, 'content/metadata/citations.json'), 'utf8')); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
for (const paper of papers) {
  const entry = citations[paper.id];
  if (entry) {
    for (const provider of ['openalex', 'google-scholar']) {
      const metric = entry[provider];
      if (!metric) continue;
      if (!Number.isInteger(metric.count) || metric.count < 0 || !/^\d{4}-\d{2}-\d{2}T/.test(metric.fetchedAt)) throw new Error(`Invalid citation metric: ${paper.id}`);
      const url = new URL(metric.url);
      if (url.protocol !== 'https:' || !['openalex.org','scholar.google.com'].includes(url.hostname) || url.username || url.password) throw new Error('Invalid citation link');
    }
    paper.citations = entry;
  }
}
const data = JSON.stringify({ config, questions, papers, reports, paperDirections, paperCategories, paperTaxonomy });
const dataFile = `data-${fingerprint(data)}.json`;
const papersModule = await readFile(join(root, 'web/papers.js'), 'utf8');
const papersModuleFile = `papers-${fingerprint(papersModule)}.js`;
await writeFile(join(dist, papersModuleFile), papersModule);
const app = (await readFile(join(root, 'web/app.js'), 'utf8')).replaceAll('__DATA_FILE__', dataFile).replaceAll('__PAPERS_MODULE__', papersModuleFile);
const appFile = `app-${fingerprint(app)}.js`;
const style = await readFile(join(root, 'web/style.css'), 'utf8');
const styleFile = `style-${fingerprint(style)}.css`;
await writeFile(join(dist, dataFile), data);
await writeFile(join(dist, appFile), app);
await writeFile(join(dist, styleFile), style);
await copyFile(join(root, 'web/favicon.svg'), join(dist, 'favicon.svg'));
await mkdir(join(dist, 'templates'), { recursive: true });
for (const file of ['paper.md', 'daily-report.md', 'survey-report.md']) {
  await copyFile(join(root, 'templates', file), join(dist, 'templates', file));
}
const template = await readFile(join(root, 'web/index.html'), 'utf8');
await writeFile(join(dist, 'index.html'), template.replaceAll('__TITLE__', escapeHtml(config.title)).replaceAll('__DESCRIPTION__', escapeHtml(config.description)).replaceAll('__MATH_CSS__', `./${mathPath}/katex.min.css`).replace('./app.js', `./${appFile}`).replace('./style.css', `./${styleFile}`));
// 保留固定地址供外部读取；页面读取带内容指纹的文件，避免命中上一版缓存。
await writeFile(join(dist, 'data.json'), data);
await writeFile(join(dist, '.nojekyll'), '');
console.log(`✓ ${questions.length} 篇笔记 · ${papers.length} 篇论文 · ${reports.length} 篇报告 → dist/`);
