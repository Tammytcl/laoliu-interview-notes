import { chromium, expect } from '@playwright/test';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { parseQuestion } from './content.mjs';
import assert from 'node:assert/strict';
const base = new URL(process.argv[2] || 'http://127.0.0.1:4273/');
const config = JSON.parse(await readFile(new URL('../site.config.json', import.meta.url), 'utf8'));
const expectedTopic = parseQuestion(await readFile(new URL('../content/questions/systems/training-inference-frameworks.md', import.meta.url), 'utf8'), 'topic', config.categories);
const expected = process.env.EXPECTED_TOPIC_UPDATED || expectedTopic.updated;
const expectedQuestions = new Set([...expectedTopic.body.matchAll(/\*\*Q(\d{2})[ ：]/g)].map(m => m[1])).size;
const expectedImages = [...expectedTopic.body.matchAll(/!\[/g)].length;
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
  const topic = data.questions.find(q => q.id === 'training-inference-frameworks');
  assert.ok(topic, 'Topic has not reached the deployed site');
  assert.equal(topic.updated, expected);
  assert.equal(topic.body, expectedTopic.body, 'Deployed topic differs from the checked-out source');
  for (const path of ['research/framework-sources.json', 'research/training-inference-frameworks-observations.json', 'research/training-inference-frameworks-pending.md', 'docs/topic-workflow.md', 'templates/topic.md']) {
    assert.equal((await page.request.get(new URL(path, base).href)).status(), 200, `Missing download: ${path}`);
  }
  assert.deepEqual(data.questions.filter(q => q.category === 'systems').map(q => q.id), [topic.id]);
  assert.equal(Object.keys(data.noteRedirects).length, 8);
  for (const oldId of Object.keys(data.noteRedirects)) {
    url.hash = `q=${oldId}`;
    await page.goto(url.href);
    await page.locator('.prose').waitFor();
    assert.ok(page.url().includes(`q=${topic.id}`), `Broken legacy redirect: ${oldId}`);
  }
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    url.hash = `q=${topic.id}`;
    await page.goto(url.href);
    await page.locator('.prose').waitFor();
    await page.locator('.prose img').evaluateAll(images => images.forEach(i => i.loading = 'eager'));
    await page.waitForFunction(() => [...document.querySelectorAll('.prose img')].every(i => i.complete && i.naturalWidth > 0));
    const text = await page.locator('.prose').innerText();
    assert.equal(new Set([...text.matchAll(/Q(\d{2})[ ：]/g)].map(m => m[1])).size, expectedQuestions);
    assert.equal(await page.locator('.prose img').count(), expectedImages);
    assert.equal(await page.locator('.prose .katex-error').count(), 0);
    assert.ok(!text.includes('$$'), 'Unrendered formula');
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Horizontal overflow');
    const imageErrors = await page.locator('.prose img').evaluateAll(images => images.filter(i => {
      const r = i.getBoundingClientRect(); return Math.abs(r.width / r.height - i.naturalWidth / i.naturalHeight) > 0.02;
    }).length);
    assert.equal(imageErrors, 0, 'Distorted images');
    await page.screenshot({ path: `${artifacts}/article-${viewport.width}.png`, fullPage: true });
    await page.locator('.toc button').filter({ hasText: '横向对比与选型' }).click();
    await expect(page.locator('h2').filter({ hasText: '横向对比与选型' })).toBeInViewport();
    await page.screenshot({ path: `${artifacts}/comparison-${viewport.width}.png` });
    results.push({ viewport: viewport.width, questions: expectedQuestions, images: expectedImages, redirects: 8 });
  }
  assert.deepEqual(errors, []);
  await writeFile(`${artifacts}/result.json`, JSON.stringify({ site: base.href, checkedAt: new Date().toISOString(), updated: topic.updated, results, errors }, null, 2));
  console.log(JSON.stringify({ topic: topic.id, updated: topic.updated, results, errors }));
} finally { await browser.close(); }
