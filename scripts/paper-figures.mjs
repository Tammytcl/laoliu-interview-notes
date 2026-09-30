// 原论文 HTML 图表截图；只准备证据资产，正文由人工 / Agent 核读后整理。
import { chromium } from '@playwright/test';
import { readFile, mkdir, writeFile, access } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { join, basename } from 'node:path';
import { createHash } from 'node:crypto';
const root = fileURLToPath(new URL('../', import.meta.url));
const [id] = process.argv.slice(2);
if (!id || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id)) throw new Error('npm run papers:figures -- paper-id');
const cached = JSON.parse(await readFile(join(root, '.paper-cache', id, 'manifest.json'), 'utf8'));
const url = `https://arxiv.org/html/${cached.arxivVersion}`;
const destination = join(root, 'assets/papers', id); await mkdir(destination, { recursive: true });
const manifestFile = join(destination, 'figures.json');
try { if (process.argv.includes('--refresh')) { const previous = JSON.parse(await readFile(manifestFile, 'utf8')); if (previous.arxivVersion !== cached.arxivVersion) throw new Error('图表已有不同原文版本，请先归档旧图并使用新资产路径。'); throw Object.assign(new Error(), { code: 'ENOENT' }); } await access(manifestFile); throw new Error('已有图表清单；请先核对已有资产，避免覆盖报告插图。'); } catch (e) { if (e.code !== 'ENOENT') throw e; }
const browser = await chromium.launch({ headless: true });
const entries = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });
  const cache = join(root, '.paper-cache', id);
  let html = await readFile(join(cache, 'paper.html'), 'utf8');
  // 用已缓存正文准备图表；禁止原文脚本及外部网络请求执行。
  html = html.replace(/<script\b[\s\S]*?<\/script>/gi, '').replace(/<head\b[\s\S]*?<\/head>/i, '<head><meta charset="utf-8"><style>body{margin:24px;font:16px Georgia,serif;background:white;color:#000}.ltx_flex_figure{display:flex;flex-wrap:wrap;max-width:1100px;gap:16px}.ltx_flex_break{flex-basis:100%}.ltx_figure{width:max-content;max-width:1380px; margin:20px 0}.ltx_table{width:max-content;max-width:1380px}.ltx_table table{border-collapse:collapse}.ltx_table th,.ltx_table td{padding:6px 9px}.ltx_border_tt,.ltx_border_t{border-top:1px solid black}.ltx_border_bb,.ltx_border_b{border-bottom:1px solid black}.ltx_font_bold{font-weight:bold}.ltx_font_italic{font-style:italic}.ltx_align_center{text-align:center}.ltx_align_right{text-align:right}svg{display:inline-block}figure>figcaption{display:none}figure figure{display:inline-block;margin:0}</style></head>');
  html = html.replace(/<object\b([^>]+)>[\s\S]*?<\/object>/gi, (_, attrs) => `<img ${attrs.replace(/\bdata=/, 'src=')}>`);
  const metadata = cached.files['html-images'] || {};
  for (const [source, file] of Object.entries(metadata)) {
    const bytes = await readFile(join(cache, 'html-images', file));
    const ext = file.split('.').at(-1); const mime = ext === 'svg' ? 'image/svg+xml' : ext === 'jpg' ? 'image/jpeg' : `image/${ext}`;
    html = html.split(`src="${source}"`).join(`src="data:${mime};base64,${bytes.toString('base64')}"`);
  }
  await page.route('**/*', route => route.abort());
  await page.setContent(html, { waitUntil: 'domcontentloaded', timeout: 30000 });
  await page.evaluate(async () => {
    document.querySelectorAll('img').forEach(image => image.loading = 'eager');
    await document.fonts.ready;
    await Promise.all([...document.images].map(image => image.decode().catch(() => {})));
  });
  await page.evaluate(() => {
    for (const table of document.querySelectorAll('figure.ltx_table')) {
      for (const wrapper of table.querySelectorAll('.ltx_transformed_outer, .ltx_transformed_inner, .ltx_minipage')) {
        wrapper.style.transform = 'none'; wrapper.style.width = 'auto'; wrapper.style.height = 'auto'; wrapper.style.display = 'block'; wrapper.style.position = 'static';
      }
    }
  });
  const figures = page.locator('figure.ltx_figure, figure.ltx_table');
  if (!await figures.count()) throw new Error('此论文无可用 HTML 图表，请使用源码图片或手动 PDF 裁剪。');
  for (const figure of await figures.all()) {
    const sourceId = await figure.getAttribute('id');
    let number = sourceId?.match(/\.(F|T)(\d+)$/);
    if (!number) { const caption = await figure.locator(':scope > figcaption').textContent().catch(() => ''); const label = caption?.match(/Figure\s+(\d+)/); if (label) number = ['', 'F', label[1]]; }
    if (!number) continue;
    if (number[1] === 'T' && sourceId.startsWith('A') && !process.argv.includes('--include-appendix')) continue;
    const name = `${number[1] === 'F' ? 'figure' : 'table'}-${number[2]}.png`;
    const content = figure.locator('svg.ltx_picture, img.ltx_graphics, table.ltx_tabular');
    if (!await content.count()) { entries.push({ sourceId, status: 'manual-capture-required' }); continue; }
    const path = join(destination, name);
    const failedImages = await figure.locator('img.ltx_graphics').evaluateAll(images => images.some(image => !image.complete || !image.naturalWidth));
    if (failedImages) throw new Error(`${sourceId}: 图像资源未完整下载`);
    const target = figure;
    await target.screenshot({ path, timeout: 30000 });
    const bytes = await readFile(path);
    entries.push({ sourceId, file: name, sha256: createHash('sha256').update(bytes).digest('hex'), original: `${url}#${sourceId}`, method: 'cached-original-html-screenshot', explanation: 'pending' });
    console.log(`✓ ${name} ← ${sourceId}`);
  }
  await writeFile(manifestFile, JSON.stringify({ paperId: id, arxivVersion: cached.arxivVersion, originalUrl: url, attribution: 'Original figures and tables by the paper authors. Captured for scholarly reading; explanations are in the report.', figures: [...new Map(entries.map(entry => [entry.file || entry.sourceId, entry])).values()] }, null, 2));
} finally { await browser.close(); }
