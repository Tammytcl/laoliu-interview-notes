import { readFile, readdir, mkdir, copyFile, writeFile } from 'node:fs/promises';
import { resolve, relative, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseQuestion, escapeHtml } from './content.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'));
if (!Array.isArray(config.categories) || new Set(config.categories.map(c => c.id)).size !== config.categories.length) throw new Error('分类配置无效或重复');
for (const c of config.categories) {
  if (!/^[a-z0-9-]+$/.test(c.id) || !c.name) throw new Error('分类需要有效 id 和 name');
}
async function walk(dir) {
  const files = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(p));
    else if (entry.isFile() && entry.name.endsWith('.md')) files.push(p);
  }
  return files.sort();
}
const questions = [], seen = new Set();
for (const p of await walk(join(root, 'content/questions'))) {
  const question = parseQuestion(await readFile(p, 'utf8'), relative(root, p).split('\\').join('/'), config.categories);
  if (seen.has(question.id)) throw new Error(`重复题目 id: ${question.id}`);
  seen.add(question.id);
  if (!question.draft) questions.push(question);
}
questions.sort((a, b) => b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title, 'zh-CN'));
const dist = resolve(root, 'dist');
await mkdir(dist, { recursive: true });
for (const file of ['app.js', 'style.css', 'favicon.svg']) await copyFile(join(root, 'web', file), join(dist, file));
const template = await readFile(join(root, 'web/index.html'), 'utf8');
await writeFile(join(dist, 'index.html'), template.replaceAll('__TITLE__', escapeHtml(config.title)).replaceAll('__DESCRIPTION__', escapeHtml(config.description)));
await writeFile(join(dist, 'data.json'), JSON.stringify({ config, questions }));
await writeFile(join(dist, '.nojekyll'), '');
console.log(`✓ ${questions.length} 道题目 · ${config.categories.length} 个主题 → dist/`);
