import { readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, relative, basename } from 'node:path';
import { createHash } from 'node:crypto';
import { parse } from 'yaml';
import { renderMarkdown, markdownHeadings } from './content.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const titles = ['背景与已有工作', '方法与实现机制', '实验设置与算力', '结果与图表解读', '局限、结论与后续阅读'];
const floors = [750, 1800, 1100, 1100, 500];
const figureRoles = new Set(['method', 'result-table', 'result-figure', 'experiment-table']);

export async function checkPaperQuality(file, base = root) {
  const raw = await readFile(file, 'utf8');
  const sourceSha256 = createHash('sha256').update(raw).digest('hex');
  const match = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { file: relative(base, file), errors: ['缺少 frontmatter'] };
  const meta = parse(match[1]);
  const body = match[2];
  const errors = [];
  const sections = markdownHeadings(body);
  const parts = sections.map((entry, index) => body.slice(entry.index, sections[index + 1]?.index ?? body.length));
  const sectionCharacters = parts.map(part => part.length);
  const imageSections = parts.map(part => [...part.matchAll(/!\[/g)].length);
  const experimentRows = [...((parts[2] ?? '').matchAll(/^\|.*\|$/gm))].length;
  const pinnedSource = /^https:\/\/arxiv\.org\/abs\/\d{4}\.\d{4,5}v\d+$/.test(meta.paper_url ?? '');
  const affiliationPlaceholder = (meta.affiliations ?? []).some(value => /not stated|待核|未知/i.test(value));
  if (sections.length !== 5 || sections.some((s, i) => !s.title.includes(titles[i]))) errors.push('正文需按 DDPM 范例保留五个主模块');
  parts.forEach((part, i) => { if (part.length < floors[i]) errors.push(`${titles[i]} 太薄：${part.length} < ${floors[i]} 字符`); });
  if (!pinnedSource) errors.push('paper_url 必须固定 arXiv 版本');
  if (!/\$\$[\s\S]+?\$\$/.test(parts[1] ?? '') && !(meta.method_formalism === 'process' && /github\.com\//.test(parts[1] ?? '') && /!\[/.test(parts[1] ?? ''))) errors.push('方法缺少关键公式，或已说明为流程型方法并提供原图与源码对照');
  if (!/\|.+\|/.test(parts[2] ?? '')) errors.push('实验缺少配置账本');
  if (!/GPU|TPU|算力|计算资源/.test(parts[2] ?? '')) errors.push('实验未交代算力或未披露情况');
  if (!/github\.com\//.test(body) && meta.github_url) errors.push('已开源但正文没有核心源码出处');
  if (!meta.github_url && !/未.*(?:代码|开源|仓库)/.test(body)) errors.push('未开源时应明确说明代码状态');
  const images = [...body.matchAll(/!\[([^\]]+)\]\((\.\/assets\/papers\/[^)]+)\)/g)];
  const methodImages = images.filter(img => img.index >= (sections[1]?.index ?? Infinity) && img.index < (sections[2]?.index ?? Infinity));
  const resultImages = images.filter(img => img.index >= (sections[3]?.index ?? Infinity) && img.index < (sections[4]?.index ?? Infinity));
  if (images.length < 3) errors.push('正文需至少有方法图与两张结果证据图表；证据不足时先保持草稿');
  if (!methodImages.length) errors.push('方法模块缺少原图');
  if (resultImages.length < 2) errors.push('结果模块缺少主结果与对照/消融图表');
  if (!meta.method_figure || !methodImages.some(x => x[2] === meta.method_figure)) errors.push('论文库方法图应出现在正文方法解释中');
  let manifest;
  try { manifest = JSON.parse(await readFile(join(base, 'assets/papers', meta.id, 'figures.json'), 'utf8')); }
  catch { errors.push('缺少图表证据清单 figures.json'); }
  const entryFor = path => manifest?.figures?.find(f => f.file === basename(path));
  const methodRoleOk = !!meta.method_figure && methodImages.some(img => img[2] === meta.method_figure) && entryFor(meta.method_figure)?.role === 'method';
  if (meta.method_figure && !methodRoleOk) errors.push('论文库预览图必须在方法模块，且在证据清单中标为 method；不能以结果图冒充方法图');
  const evidenceTables = images.filter(img => img.index >= (sections[2]?.index ?? Infinity) && img.index < (sections[4]?.index ?? Infinity) && ['experiment-table', 'result-table'].includes(entryFor(img[2])?.role));
  if (!evidenceTables.length) errors.push('实验或结果模块缺少登记用途的原论文表格截图');
  if (resultImages.filter(img => ['result-table', 'result-figure'].includes(entryFor(img[2])?.role)).length < 2) errors.push('结果模块至少需要两张登记为结果证据的图表');
  for (const img of images) {
    const path = join(base, img[2]);
    let bytes;
    try { bytes = await readFile(path); } catch { errors.push(`插图不存在：${img[2]}`); continue; }
    const entry = entryFor(img[2]);
    if (!entry) { errors.push(`插图未登记出处：${img[2]}`); continue; }
    if (!figureRoles.has(entry.role)) errors.push(`插图缺少图表用途 role：${img[2]}`);
    if (!['pinned-original-pdf-crop', 'pinned-latex-asset'].includes(entry.method) || !entry.visuallyVerified || entry.explanation !== 'complete') errors.push(`插图未按固定来源逐张核对：${img[2]}`);
    if (entry.method === 'pinned-latex-asset') {
      const safe = value => typeof value === 'string' && value.length > 0 && !value.startsWith('/') && !value.split(/[\\/]/).includes('..');
      if (!/^[a-f0-9]{64}$/.test(entry.sourceArchiveSha256 ?? '') || !safe(entry.texFile) || !entry.figureLabel || !/^[a-f0-9]{64}$/.test(entry.texSha256 ?? '') || !entry.sourceFiles?.length || !entry.sourceFiles.every(f => safe(f.path) && /^[a-f0-9]{64}$/.test(f.sha256 ?? ''))) errors.push(`插图缺少 LaTeX 压缩包、原资产、TeX 引用或校验值：${img[2]}`);
      if (entry.original !== `https://arxiv.org/src/${manifest.arxivVersion}`) errors.push(`LaTeX 图源版本不一致：${img[2]}`);
    } else {
      if (!Number.isInteger(entry.pdfPage) || !Array.isArray(entry.cropBox) || entry.cropBox.length !== 4 || !/^[a-f0-9]{64}$/.test(entry.pdfSha256 ?? '')) errors.push(`插图缺少 PDF 页码、裁剪区域或原 PDF 校验值：${img[2]}`);
      const [x0, y0, x1, y1] = entry.cropBox ?? [];
      const [width, height] = entry.pdfPageSize ?? [];
      if (!(width > 0 && height > 0 && x0 >= 0 && y0 >= 0 && x1 > x0 && y1 > y0 && x1 <= width && y1 <= height)) errors.push(`缺少有效 PDF 页尺寸和裁剪边界：${img[2]}`);
      else if ((x1 - x0) * (y1 - y0) / (width * height) > .65) errors.push(`禁止整页/近整页截图：${img[2]}`);
      if (['method', 'result-figure'].includes(entry.role) && !entry.sourceUnavailableReason) errors.push(`应优先使用 LaTeX 原资产；PDF 图需记录没有独立图片的原因：${img[2]}`);
    }
    if (entry.sha256 !== createHash('sha256').update(bytes).digest('hex')) errors.push(`插图 hash 与清单不符：${img[2]}`);
    const remainder = body.slice(img.index + img[0].length);
    const stop = [remainder.indexOf('!['), remainder.search(/^## /m)].filter(index => index >= 0).sort((a, b) => a - b)[0] ?? remainder.length;
    const nearby = remainder.slice(0, Math.min(stop, 1200));
    const prose = nearby.replace(/^\|.*$/gm, '').replace(/\[[^\]]+\]\([^)]+\)/g, '').replace(/[\s*#>|-]/g, '');
    if (prose.length < 80 || !/(解读|图源|表源|原图|原表|原文|PDF|Figure|Table)/.test(nearby)) errors.push(`插图旁缺少充分的证据解读：${img[2]}`);
  }
  if (renderMarkdown(body).html.includes('katex-error')) errors.push('公式含 KaTeX 解析错误');
  if (/方法证据页|主结果证据页|整页保留公式|L\(theta\)=E_/.test(body)) errors.push('残留整页占位说明或伪 LaTeX 公式');
  if (manifest && manifest.arxivVersion !== (meta.paper_url ?? '').match(/(\d{4}\.\d{4,5}v\d+)$/)?.[1]) errors.push('图表清单版本与 paper_url 的 arXiv 固定版本不一致');
  return { file: relative(base, file), id: meta.id, marked: meta.depth_standard === 'ddpm' || meta.template_version >= 5, draft: !!meta.draft, images: images.length, imageSections, sectionCharacters, experimentRows, methodRoleOk, evidenceTables: evidenceTables.length, pinnedSource, affiliationPlaceholder, characters: body.length, sourceSha256, errors };
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
  if (process.argv.includes('--summary')) {
    console.log('| Paper | 正文字符 | 背景/方法/实验/结果 | 方法图/结果图 | 已核方法预览 | 原论文表格 | 实验表格行 | 固定版本 | 单位占位 | 自动检查 |');
    console.log('| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |');
    for (const p of all.filter(x => !x.draft)) {
      console.log(`| ${p.id} | ${p.characters} | ${p.sectionCharacters.slice(0, 4).join('/')} | ${p.imageSections[1] ?? 0}/${p.imageSections[3] ?? 0} | ${p.methodRoleOk ? '是' : '否'} | ${p.evidenceTables} | ${p.experimentRows} | ${p.pinnedSource ? '是' : '否'} | ${p.affiliationPlaceholder ? '是' : '否'} | ${p.errors.length ? '缺口' : '结构通过'} |`);
    }
    process.exit(0);
  }
  const strict = process.argv.includes('--strict');
  const auditAll = strict || process.argv.includes('--audit-all');
  const scoped = auditAll ? all.filter(x => !x.draft) : all.filter(x => !x.draft && (x.marked || x.enforcementError));
  for (const p of scoped) {
    const problems = [...p.errors, ...(p.enforcementError ? [p.enforcementError] : [])];
    console.log(`${problems.length ? 'FAIL' : 'PASS'} ${p.id}: ${p.characters} chars, ${p.images} images${problems.length ? '\n  - ' + problems.join('\n  - ') : ''}`);
  }
  const failures = scoped.filter(p => p.errors.length || p.enforcementError);
  console.log(`${scoped.length - failures.length}/${scoped.length} reports pass the automatic structure/evidence checks; ${all.filter(x => !x.draft && !x.marked).length} frozen legacy reports remain to upgrade. Passing does not verify explanation quality or factual accuracy.`);
  if (failures.length && (strict || !auditAll)) process.exitCode = 1;
}
