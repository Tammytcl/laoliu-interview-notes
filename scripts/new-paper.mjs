import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { paperDirections } from './papers-content.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));
const [type, suppliedId, direction = 'llm'] = process.argv.slice(2);
const today = new Date().toISOString().slice(0, 10);
const specs = {
  paper: { template: 'paper.md', collection: 'papers', marker: 'paper-id' },
  daily: { template: 'daily-report.md', collection: 'reports/daily', marker: 'daily-date' },
  survey: { template: 'survey-report.md', collection: 'reports/surveys', marker: 'survey-id' }
};
if (!Object.hasOwn(specs, type)) throw new Error('使用 new:paper / new:daily / new:survey');
let reportDate = today;
if (type === 'daily' && suppliedId) reportDate = suppliedId;
if (!/^\d{4}-\d{2}-\d{2}$/.test(reportDate) || !Number.isFinite(Date.parse(reportDate)) || new Date(reportDate).toISOString().slice(0, 10) !== reportDate) throw new Error('Daily 日期应为有效 YYYY-MM-DD');
const id = type === 'daily' ? `daily-${reportDate}` : suppliedId;
if (!id || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) throw new Error('请提供稳定的小写英文 id，例如 paper-my-topic 或 survey-my-topic');
if (!paperDirections.some(d => d.id === direction)) throw new Error('方向应为 diffusion / llm / agent / infra');
const spec = specs[type];
const dir = join(root, 'content', spec.collection, ...(type === 'paper' ? [direction] : []));
await mkdir(dir, { recursive: true });
const file = join(dir, `${id}.md`);
if (type === 'daily') {
  try { await access(file); console.log(`当天报告已存在，请继续更新：${file}`); process.exit(0); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
let template = await readFile(join(root, 'templates', spec.template), 'utf8');
template = template.replace(`id: ${spec.marker}`, `id: ${id}`).replace('updated: 2026-09-30', `updated: ${today}`);
if (type === 'paper') template = template.replace('direction: llm', `direction: ${direction}`).replace('published: 2026-09-30', `published: ${today}`).replace('year: 2026', `year: ${new Date().getUTCFullYear()}`);
if (type === 'paper' && direction === 'diffusion') template = template.replace('areas: [language]', 'areas: [vision]').replace('tasks: [text-generation]', 'tasks: [image-generation]');
if (type === 'paper' && direction === 'agent') template = template.replace('tasks: [text-generation]', 'tasks: [agents]');
if (type === 'survey') template = template.replace('directions: [llm]', `directions: [${direction}]`).replace('date: 2026-09-30', `date: ${today}`);
if (type === 'daily') template = template.replace('date: 2026-09-30', `date: ${reportDate}`).replace('YYYY-MM-DD', reportDate);
await writeFile(file, template, { flag: 'wx' });
console.log(`✓ 新建草稿：${file}\n填写内容后将 draft 改为 false；paper_ids 关联稳定论文 id。`);
