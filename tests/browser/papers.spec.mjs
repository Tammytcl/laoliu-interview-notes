import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('论文库、方向筛选、Daily 与专题报告相互链接', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/'); await page.locator('[data-paper-nav]').click();
  const data = await (await page.request.get('/data.json')).json();
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.length);
  await expect(page.locator('[data-paper-nav]')).toHaveAttribute('aria-current', 'page');
  await page.locator('#paper-direction').selectOption('llm');
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.filter(p => p.direction === 'llm').length);
  await page.locator('#paper-search').fill('no-match-paper-123');
  await expect(page.locator('[data-paper-reset]')).toBeVisible();
  await page.locator('[data-paper-reset]').click();
  await expect(page.locator('.paper-card')).toHaveCount(data.papers.length);
  await page.getByRole('link', { name: 'Daily 报告', exact: true }).click();
  await expect(page.locator('.paper-card')).toHaveCount(data.reports.filter(r => r.type === 'daily').length);
  await page.locator('.paper-card h3 a[href="#report=daily-2026-09-30"]').click();
  await expect(page.locator('.paper-related a[href^="#paper="]')).toHaveCount(data.reports.find(r => r.id === 'daily-2026-09-30').paperIds.length);
  await page.locator('.paper-related a[href="#paper=paper-gqa"]').click();
  await expect(page.locator('#paper-space h1')).toContainText('GQA');
  await expect(page.locator('.paper-related a[href="#report=survey-inference-memory"]')).toBeVisible();
  await page.locator('.paper-related a[href="#report=survey-inference-memory"]').click();
  await page.reload(); await expect(page.locator('#paper-space h1')).toContainText('专题调研');
  await page.locator('.back-link').click();
  await expect(page.locator('.paper-tabs [aria-current="page"]')).toHaveText('专题调研');
  expect(errors).toEqual([]);
});

test('个人复述、复习日期、收藏持久化，并可导出导入', async ({ page }) => {
  await page.goto('/#paper=paper-gqa');
  await page.locator('[data-paper-stage="recalled"]').click();
  await expect(page.locator('#toast')).toContainText('先写一段自己的复述');
  await expect(page.locator('[data-paper-stage="recalled"]')).toHaveAttribute('aria-pressed', 'false');
  await page.locator('#reading-question').fill('共享 KV 是否减少 query 头？');
  await page.locator('[data-close-paper]').click();
  await expect(page.locator('#paper-body')).not.toHaveAttribute('open', '');
  await page.locator('#reading-recall').fill('query 仍独立，组内共享 KV；容量减少不等于端到端速度等比例提升。');
  await page.locator('#reading-connection').fill('连接 PagedAttention 的放置策略。');
  await page.locator('#reading-next-review').fill('2020-01-01');
  await page.locator('[data-paper-stage="recalled"]').click();
  await page.locator('[data-paper-star]').click();
  await expect(page.locator('#paper-body')).not.toHaveAttribute('open', '');
  await page.reload();
  await expect(page.locator('#reading-question')).toHaveValue('共享 KV 是否减少 query 头？');
  await expect(page.locator('[data-paper-stage="recalled"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('[data-paper-star]')).toHaveAttribute('aria-pressed', 'true');
  const downloading = page.waitForEvent('download'); await page.locator('[data-reading-export]').click();
  const download = await downloading;
  const backup = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(backup.reading['paper-gqa'].stage).toBe('recalled');
  expect(backup.reading['paper-gqa'].recall).toContain('query');
  await page.evaluate(() => localStorage.clear()); await page.goto('/#view=papers');
  page.on('dialog', dialog => dialog.accept());
  await page.locator('#reading-import-file').setInputFiles({ name: 'reading.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(backup)) });
  await page.locator('#paper-stage').selectOption('recalled');
  await expect(page.locator('.paper-card')).toHaveCount(1);
  await expect(page.locator('.paper-card')).toContainText('到期复习');
  await page.locator('.paper-card h3 a').click();
  await expect(page.locator('#reading-connection')).toHaveValue('连接 PagedAttention 的放置策略。');
});

test('三份模板可下载，手机与仓库子路径的论文路由可直接刷新', async ({ page }) => {
  await page.goto('/#view=papers&tab=guide');
  await expect(page.locator('.paper-template')).toHaveCount(3);
  for (const file of ['paper.md', 'daily-report.md', 'survey-report.md']) {
    const response = await page.request.get(`/templates/${file}`);
    expect(response.ok()).toBe(true); expect(await response.text()).toContain('draft: true');
  }
  await page.setViewportSize({ width: 390, height: 844 });
  for (const route of ['view=papers', 'view=papers&tab=daily', 'view=papers&tab=survey', 'view=papers&tab=guide', 'paper=paper-gqa', 'report=survey-inference-memory']) {
    await page.goto(`http://127.0.0.1:4176/interview-notes/#${route}`);
    await page.reload(); await expect(page.locator('#paper-space h1')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await page.screenshot({ path: 'test-results/paper-mobile.png', fullPage: true });
});
