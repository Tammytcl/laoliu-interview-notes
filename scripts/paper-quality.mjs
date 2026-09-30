import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, relative, basename } from 'node:path';
import { createHash } from 'node:crypto';
import { parse } from 'yaml';

const root = fileURLToPath(new URL('../', import.meta.url));
const titles = ['背景与已有工作', '方法与实现机制', '实验设置与算力', '结果与图表解读', '局限、结论与后续阅读'];
const floors = [750, 1800, 1100, 1100, 500];

export async function checkPaperQuality(file, base = root) {
  const raw = await readFile(file, 'utf8');
  const sourceSha256 = createHash('sha256').update(raw).digest('hex');
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { file: relative(base, file), errors: ['缺少 frontmatter'] };
  const meta = parse(match[1]);
  const body = match[2];
  const errors = [];
  const sections = [...body.matchAll(/^## (.+)$/gm)];
  const parts = sections.map((entry, index) => body.slice(entry.index, sections[index + 1]?.index ?? body.length));
  if (sections.length !== 5 || sections.some((s, i) => !s[1].includes(titles[i]))) errors.push('正文需按 DDPM 范例保留五个主模块');
  parts.forEach((part, i) => { if (part.length < floors[i]) errors.push(`${titles[i]} 太薄：${part.length} < ${floors[i]} 字符`); });
  if (body.length < 6900) errors.push(`正文太短：${body.length} < 6900 字符`);
  if (!/^https:\/\/arxiv\.org\/abs\/\d{4}\.\d{4,5}v\d+$/.test(meta.paper_url ?? '')) errors.push('paper_url 必须固定 arXiv 版本');
  if (!/\$\$[\s\S]+?\$\$/.test(parts[1] ?? '')) errors.push('方法缺少可追溯的关键公式');
  if (!/\|.+\|/.test(parts[2] ?? '')) errors.push('实验缺少配置账本');
  if (!/GPU|TPU|算力|计算资源/.test(parts[2] ?? '')) errors.push('实验未交代算力或未披露情况');
  if (!/github\.com\//.test(body) && meta.github_url) errors.push('已开源但正文没有核心源码出处');
  if (!meta.github_url && !/未.*(?:代码|开源|仓库)/.test(body)) errors.push('未开源时应明确说明代码状态');
  const images = [...body.matchAll(/!\[([^\]]+)\]\((\.\/assets\/papers\/[^)]+)\)/g)];
  if (images.length < 3) errors.push('正文需至少有方法图与两张结果证据图表；无图论文须人工审定例外');
  if (!(parts[1] ?? '').includes('![')) errors.push('方法模块缺少原图');
  if ([...((parts[3] ?? '').matchAll(/!\[/g))].length < 2) errors.push('结果模块缺少主结果与对照/消融图表');
  if (!meta.method_figure || !images.some(x => x[2] === meta.method_figure)) errors.push('论文库方法图应出现在正文方法解释中');
  let manifest;
  try { manifest = JSON.parse(await readFile(join(base, 'assets/papers', meta.id, 'figures.json'), 'utf8')); }
  catch { errors.push('缺少图表证据清单 figures.json'); }
  for (const img of images) {
    const path = join(base, img[2]);
    let bytes;
    try { bytes = await readFile(path); } catch { errors.push(`插图不存在：${img[2]}`); continue; }
    const entry = manifest?.figures?.find(f => f.file === basename(path));
    if (!entry) { errors.push(`插图未登记出处：${img[2]}`); continue; }
    if (entry.method !== 'pinned-original-pdf-crop' || !entry.visuallyVerified || entry.explanation !== 'complete') errors.push(`插图未按固定 PDF 逐张核对：${img[2]}`);
    if (entry.sha256 !== createHash('sha256').update(bytes).digest('hex')) errors.push(`插图 hash 与清单不符：${img[2]}`);
    if (!body.slice(img.index + img[0].length, img.index + img[0].length + 350).includes('解读')) errors.push(`插图旁缺少逐图解读：${img[2]}`);
  }
  if (manifest && !manifest.arxivVersion) errors.push('图表清单缺少 arXiv 固定版本');
  return { file: relative(base, file), id: meta.id, marked: meta.depth_standard === 'ddpm' || meta.template_version >= 5, draft: !!meta.draft, images: images.length, characters: body.length, sourceSha256, errors };
}

async function collect(dir) {
  const out = [];
  let items;
  try { items = await readdir(dir, { withFileTypes: true }); }
  catch (error) { if (error.code === 'ENOENT') return out; throw error; }
  for (const item of items) {
    const path = join(dir, item.name);
    if (item.isDirectory()) out.push(...await collect(path));
    else if (item.name.endsWith('.md')) out.push(path);
  }
  return out;
}

export async function auditPapers(base = root) {
  const files = await collect(join(base, 'content/papers'));
  const results = await Promise.all(files.map(file => checkPaperQuality(file, base)));
  let snapshot = {};
  try { snapshot = JSON.parse(await readFile(join(base, 'content/metadata/paper-depth-legacy.json'), 'utf8')).sha256 ?? {}; }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  for (const result of results) {
    if (!result.draft && !result.marked && snapshot[result.id] !== result.sourceSha256) result.enforcementError = '新增或修改的旧版报告必须升级到 DDPM 深度标准';
  }
  return results;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const all = await auditPapers();
  const auditAll = process.argv.includes('--audit-all');
  const scoped = auditAll ? all.filter(x => !x.draft) : all.filter(x => !x.draft && (x.marked || x.enforcementError));
  for (const p of scoped) {
    const problems = [...p.errors, ...(p.enforcementError ? [p.enforcementError] : [])];
    console.log(`${problems.length ? 'FAIL' : 'PASS'} ${p.id}: ${p.characters} chars, ${p.images} images${problems.length ? '\n  - ' + problems.join('\n  - ') : ''}`);
  }
  const failures = scoped.filter(p => p.errors.length || p.enforcementError);
  console.log(`${scoped.length - failures.length}/${scoped.length} reports meet the DDPM depth gate; ${all.filter(x => !x.draft && !x.marked).length} frozen legacy reports remain to upgrade.`);
  if (failures.length && !auditAll) process.exitCode = 1;
}
