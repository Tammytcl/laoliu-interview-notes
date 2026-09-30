import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile, readFile, rm, cp, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { paperCacheStatus, cleanPaperCache } from '../scripts/paper-cache.mjs';
import { renderMarkdown } from '../scripts/content.mjs';
import { parsePaperContent } from '../scripts/papers-content.mjs';
const root = fileURLToPath(new URL('../', import.meta.url));

test('正式报告保留单位与代码来源，五模块和全部 GQA 证据资产可追溯', async () => {
  for (const name of ['llm/paper-gqa', 'agent/paper-react', 'infra/paper-pagedattention', 'diffusion/paper-ddpm']) {
    const p = parsePaperContent(await readFile(join(root, 'content/papers', name + '.md'), 'utf8'), name, 'papers');
    assert.equal(p.title, p.paperTitle); assert.equal(p.templateVersion, 3); assert.ok(p.affiliations.length);
    assert.equal(p.authorAffiliations.length, p.authors.length); assert.match(p.githubUrl, /^https:\/\/github.com\//);
    assert.ok(p.categories.length); assert.ok((await readFile(join(root, p.methodFigure))).length > 100);
    assert.match(p.body, /核心源码|源码对照/); assert.match(p.body, /github\.com\/.+\/blob\/[a-f0-9]{40}\//);
    assert.equal(p.toc.filter(h => h.level === 2).length, 5);
    assert.doesNotMatch(p.body, /待填|这里不替你填|基础精读示例/);
    for (const [, file] of p.body.matchAll(/!\[[^\]]*\]\((\.\/assets\/papers\/[^)]+)\)/g)) assert.ok((await readFile(join(root, file))).length > 100);
  }
  const manifest = JSON.parse(await readFile(join(root, 'assets/papers/paper-gqa/figures.json'), 'utf8'));
  assert.equal(manifest.figures.length, 7); assert.ok(manifest.figures.every(f => f.explanation === 'complete'));
});

test('公式渲染包含数学语义并拒绝可信 HTML / URL 注入', () => {
  const rendered = renderMarkdown('## 方法\n\n$H/G$\n\n$$\nO=QK^{\\mathsf T}\n$$\n\n<script>bad()</script>\n\n$\\href{javascript:alert(1)}{x}$');
  assert.match(rendered.html, /class="katex"/); assert.match(rendered.html, /<math/);
  assert.match(rendered.html, /math-block/); assert.equal(rendered.toc.length, 1);
  assert.doesNotMatch(rendered.html, /<script>|href="javascript:/);
});

test('源码清理只删除缓存，报告 / 插图保留且拒绝符号链接', async t => {
  const fixture = await mkdtemp(join(tmpdir(), 'paper-cache-'));
  t.after(() => rm(fixture, { recursive: true, force: true }));
  for (const folder of ['.paper-cache/id', 'content/papers', 'assets/papers']) await mkdir(join(fixture, folder), { recursive: true });
  await writeFile(join(fixture, '.paper-cache/id/main.tex'), '123456');
  await writeFile(join(fixture, 'content/papers/report.md'), 'report'); await writeFile(join(fixture, 'assets/papers/chart.png'), 'image');
  assert.deepEqual((await paperCacheStatus(fixture)).files, 1);
  const result = await cleanPaperCache(fixture); assert.equal(result.freedBytes, 6);
  assert.equal(await readFile(join(fixture, 'content/papers/report.md'), 'utf8'), 'report');
  assert.equal(await readFile(join(fixture, 'assets/papers/chart.png'), 'utf8'), 'image');
  await symlink(join(fixture, 'content'), join(fixture, '.paper-cache/unsafe'));
  await assert.rejects(cleanPaperCache(fixture), /符号链接/);
  assert.equal(await readFile(join(fixture, 'content/papers/report.md'), 'utf8'), 'report');
});

test('本地清理 API 要求同源与 token，不接受任意删除路径', async t => {
  const fixture = await mkdtemp(join(tmpdir(), 'paper-workbench-'));
  await cp(join(root, 'scripts'), join(fixture, 'scripts'), { recursive: true });
  await mkdir(join(fixture, '.paper-cache/id'), { recursive: true }); await mkdir(join(fixture, 'dist'), { recursive: true });
  await writeFile(join(fixture, '.paper-cache/id/main.tex'), 'cache');
  const server = spawn(process.execPath, ['scripts/serve.mjs'], { cwd: fixture, env: { ...process.env, PORT: '0', PAPER_WORKBENCH: '1' } });
  t.after(async () => { server.kill(); await rm(fixture, { recursive: true, force: true }); });
  const base = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { server.kill(); reject(new Error('server timeout')); }, 10000);
    server.stdout.on('data', data => { const match = String(data).match(/http:\/\/127\.0\.0\.1:\d+/); if (match) { clearTimeout(timer); resolve(match[0]); } });
    server.on('error', reject);
  });
  const status = await (await fetch(base + '/api/paper-cache')).json(); assert.equal(status.files, 1);
  assert.equal((await fetch(base + '/api/paper-cache/clean', { method: 'POST' })).status, 403);
  assert.equal((await fetch(base + '/api/paper-cache/clean', { method: 'POST', headers: { Origin: 'https://evil.example', 'X-Paper-Cache-Token': status.token } })).status, 403);
  assert.equal(await readFile(join(fixture, '.paper-cache/id/main.tex'), 'utf8'), 'cache');
  const cleaned = await fetch(base + '/api/paper-cache/clean', { method: 'POST', headers: { Origin: base, 'X-Paper-Cache-Token': status.token } });
  assert.equal(cleaned.status, 200); assert.equal((await cleaned.json()).deletedFiles, 1);
  assert.equal((await fetch(base + '/api/paper-cache/clean/../../content', { method: 'POST', headers: { 'X-Paper-Cache-Token': status.token } })).status, 405);
});
