import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
const root = fileURLToPath(new URL('../', import.meta.url));
const [id, category = 'llm'] = process.argv.slice(2);
if (!id || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) throw new Error('用法：npm run new -- question-id llm（小写英文 id）');
const config = JSON.parse(await readFile(join(root, 'site.config.json'), 'utf8'));
if (!config.categories.some(c => c.id === category)) throw new Error(`未知分类：${category}`);
let template = await readFile(join(root, category === 'interview' ? 'templates/interview-question.md' : 'templates/question.md'), 'utf8');
if (category === 'interview') {
  const collection = JSON.parse(await readFile(join(root, 'content/research/post-training-interview-practice-sources.json'), 'utf8'));
  const nextNumber = Math.max(0, ...collection.questions.map(q => q.number)) + 1;
  template = template.replace('__NUMBER__', String(nextNumber).padStart(2, '0'));
}
template = template.replace('id: question-id', `id: ${id}`).replace('category: llm', `category: ${category}`).replace('updated: 2026-09-30', `updated: ${new Date().toISOString().slice(0, 10)}`);
const dir = join(root, 'content/questions', category);
await mkdir(dir, { recursive: true });
const path = join(dir, `${id}.md`);
await writeFile(path, template, { flag: 'wx' });
console.log(`✓ 新建草稿：${path}\n完成内容后，将 draft 改为 false，再运行 npm run build。`);
if (category === 'interview') console.log('发布前登记 concept、题号、来源与关联笔记，再运行 npm run interviews:index；参见 docs/interview-workflow.md。');
