import { chromium, expect } from '@playwright/test';
import { mkdir, writeFile, readFile, readdir } from 'node:fs/promises';
import { parseQuestion } from './content.mjs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const base = new URL(process.argv[2] || 'http://127.0.0.1:4273/');
assert.ok(['http:', 'https:'].includes(base.protocol));
const config = JSON.parse(await readFile(new URL('../site.config.json', import.meta.url), 'utf8'));
const registry = JSON.parse(await readFile(new URL('../content/research/topics.json', import.meta.url), 'utf8'));
const sourceQuestions = [];
async function walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = new URL(entry.name + (entry.isDirectory() ? '/' : ''), dir);
    if (entry.isDirectory()) await walk(path);
    else if (entry.name.endsWith('.md')) sourceQuestions.push(parseQuestion(await readFile(path, 'utf8'), entry.name, config.categories));
  }
}
await walk(new URL('../content/questions/', import.meta.url));
const expectedTopics = registry.map(item => {
  const q = sourceQuestions.find(q => q.id === item.id && !q.draft);
  assert.ok(q, `Missing registered topic ${item.id}`);
  return { ...q, registration: item };
});
const artifacts = 'test-results/topic-live';
await mkdir(artifacts, { recursive: true });
const browser = await chromium.launch();
const errors = [], results = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', error => errors.push(error.message));
  const url = new URL(base); url.searchParams.set('check', Date.now());
  const dataResponsePromise = page.waitForResponse(response => /\/data-[a-f0-9]+\.json$/.test(new URL(response.url()).pathname));
  await page.goto(url.href);
  await page.locator('.question-card').first().waitFor();
  const dataResponse = await dataResponsePromise;
  assert.equal(dataResponse.status(), 200, 'Versioned data response failed');
  const data = await dataResponse.json();
  const framework = data.questions.find(q => q.id === 'training-inference-frameworks');
  assert.ok(framework);
  if (process.env.EXPECTED_TOPIC_UPDATED) assert.equal(framework.updated, process.env.EXPECTED_TOPIC_UPDATED);
  const downloads = new Set(['research/topics.json', 'research/framework-sources.json', 'research/training-inference-frameworks-observations.json', 'research/training-inference-frameworks-pending.md', 'docs/topic-workflow.md', 'templates/topic.md']);
  for (const topic of expectedTopics) {
    downloads.add(`research/${topic.registration.sources.split('/').at(-1)}`);
    const manifest = JSON.parse(await readFile(new URL('../' + topic.registration.figures, import.meta.url), 'utf8'));
    const folder = topic.registration.figures.slice(0, topic.registration.figures.lastIndexOf('/') + 1);
    for (const figure of manifest.figures) {
      const response = await page.request.get(new URL(folder + figure.file, base).href);
      assert.equal(response.status(), 200, `Missing figure: ${topic.id}/${figure.file}`);
      assert.equal(createHash('sha256').update(await response.body()).digest('hex'), figure.sha256, `Stale figure: ${topic.id}/${figure.file}`);
    }
  }
  for (const topic of expectedTopics) {
    for (const [, path] of topic.body.matchAll(/\]\(\.\/(assets\/infra\/[^\s)]+\.(?:py|json))\)/g)) downloads.add(path);
  }
  for (const file of ['infra-foundations-check.py', 'infra-foundations-check-results.json']) downloads.add(`assets/infra/${file}`);
  for (const path of downloads) {
    const response = await page.request.get(new URL(path, base).href);
    assert.equal(response.status(), 200, `Missing download: ${path}`);
    if (path.endsWith('-sources.json')) {
      const source = await readFile(new URL('../content/' + path, import.meta.url), 'utf8');
      assert.deepEqual(await response.json(), JSON.parse(source), `Stale source download: ${path}`);
    }
    if (path.startsWith('assets/infra/')) assert.deepEqual(await response.body(), await readFile(new URL('../' + path, import.meta.url)), `Stale exercise download: ${path}`);
  }
  const publishedSystems = sourceQuestions.filter(q => !q.draft && q.category === 'systems').map(q => q.id).sort();
  assert.deepEqual(data.questions.filter(q => q.category === 'systems').map(q => q.id).sort(), publishedSystems);
  url.hash = '';
  await page.goto(url.href);
  await page.locator('[data-category="systems"]').click();
  await expect(page.locator('.question-card')).toHaveCount(publishedSystems.length);
  for (const topic of expectedTopics) await expect(page.locator('.question-card a').filter({ hasText: topic.title })).toHaveCount(1);
  const redirects = JSON.parse(await readFile(new URL('../content/metadata/note-redirects.json', import.meta.url), 'utf8'));
  assert.deepEqual(data.noteRedirects, redirects);
  for (const [oldId, target] of Object.entries(redirects)) {
    url.hash = `q=${oldId}`;
    await page.goto(url.href);
    await page.locator('.prose').waitFor();
    assert.ok(page.url().includes(`q=${target}`), `Broken legacy redirect: ${oldId}`);
  }
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const topic of expectedTopics) {
      const deployed = data.questions.find(q => q.id === topic.id);
      assert.ok(deployed, `Missing topic: ${topic.id}`);
      assert.equal(deployed.updated, topic.updated);
      assert.equal(deployed.body, topic.body, `Deployed topic differs: ${topic.id}`);
      const expectedQuestions = new Set([...topic.body.matchAll(/\*\*Q(\d{2})[ ：]/g)].map(m => m[1])).size;
      const expectedImages = [...topic.body.matchAll(/!\[/g)].length;
      url.hash = `q=${topic.id}`;
      await page.goto(url.href);
      await page.locator('.prose').waitFor();
      await page.locator('.prose img').evaluateAll(images => images.forEach(i => i.loading = 'eager'));
      await page.waitForFunction(() => [...document.querySelectorAll('.prose img')].every(i => i.complete && i.naturalWidth > 0));
      const text = await page.locator('.prose').innerText();
      assert.equal(new Set([...text.matchAll(/Q(\d{2})[ ：]/g)].map(m => m[1])).size, expectedQuestions);
      assert.equal(await page.locator('.prose img').count(), expectedImages);
      assert.equal(await page.locator('.prose .katex-error').count(), 0, topic.id);
      assert.ok(!text.includes('$$'), `Unrendered formula: ${topic.id}`);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Horizontal overflow: ${topic.id}`);
      assert.equal(await page.locator('.prose img').evaluateAll(images => images.filter(i => {
        const r = i.getBoundingClientRect(); return Math.abs(r.width / r.height - i.naturalWidth / i.naturalHeight) > 0.02;
      }).length), 0, `Distorted images: ${topic.id}`);
      await page.screenshot({ path: `${artifacts}/${topic.id}-${viewport.width}.png`, fullPage: true });
      const section = topic.toc.find(s => s.level === 2 && s.title.includes('面试')) || topic.toc.find(s => s.title.includes('横向对比'));
      assert.ok(section, `Missing revision section: ${topic.id}`);
      await page.locator(`.toc button[data-section="${section.id}"]`).click();
      await expect(page.locator(`#${section.id}`)).toBeInViewport();
      await page.reload();
      await page.locator('.prose').waitFor();
      assert.ok((await page.locator('.reader-header h1').innerText()).includes(topic.title));
      results.push({ topic: topic.id, updated: topic.updated, viewport: viewport.width, questions: expectedQuestions, images: expectedImages });
    }
  }
  assert.deepEqual(errors, []);
  await writeFile(`${artifacts}/result.json`, JSON.stringify({ site: base.href, checkedAt: new Date().toISOString(), topics: expectedTopics.length, results, errors }, null, 2));
  console.log(JSON.stringify({ topics: expectedTopics.length, results, redirects: Object.keys(redirects).length, errors }));
} finally { await browser.close(); }
