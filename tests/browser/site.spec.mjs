import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test('加载、全文搜索、筛选和空结果', async ({ page }) => {
  const errors = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/'); await expect(page.locator('.question-card').first()).toBeVisible();
  const data = await (await page.request.get('/data.json')).json();
  await expect(page.locator('.question-card')).toHaveCount(data.questions.length);
  await page.locator('#search').fill(data.questions[0].title);
  await expect(page.locator('.question-card').first()).toContainText(data.questions[0].title);
  await page.locator('#search').fill('no-match-zzzz-123456');
  await expect(page.locator('#reset-filters')).toBeVisible();
  await page.locator('#reset-filters').click();
  await page.locator(`[data-category="${data.questions[0].category}"]`).click();
  await expect(page.locator('.question-card')).toHaveCount(data.questions.filter(q => q.category === data.questions[0].category).length);
  expect(errors).toEqual([]);
});
test('详情路由、目录、收藏和复习状态持久化', async ({ page }) => {
  await page.goto('/'); await page.locator('.question-copy h3 a').first().click();
  await expect(page.locator('.prose')).toBeVisible();
  await page.locator('[data-status="mastered"]').click();
  await page.locator('#reader [data-star]').click(); await page.reload();
  await expect(page.locator('[data-status="mastered"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#reader [data-star]')).toHaveAttribute('aria-pressed', 'true');
  await page.locator('.toc button').first().click();
  await expect(page.locator('#reader')).toBeVisible();
  await page.locator('.back-link').click(); await page.locator('[data-filter="mastered"]').click();
  await expect(page.locator('.question-card')).toHaveCount(1);
});
test('进度导出与导入', async ({ page }) => {
  await page.goto('/'); await page.locator('.question-copy h3 a').first().click();
  await page.locator('[data-status="review"]').click();
  const download = page.waitForEvent('download'); await page.locator('#export').click();
  const file = await download; const backup = JSON.parse(await readFile(await file.path(), 'utf8'));
  expect(backup.version).toBe(1); expect(Object.keys(backup.progress)).toHaveLength(1);
  await page.evaluate(() => localStorage.clear()); await page.reload();
  page.on('dialog', dialog => dialog.accept());
  await page.locator('#import-file').setInputFiles({ name: 'progress.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(backup)) });
  await expect(page.locator('[data-status="review"]')).toHaveAttribute('aria-pressed', 'true');
});
test('新增内容说明和未知链接', async ({ page }) => {
  await page.goto('/'); await expect(page.locator('.question-card').first()).toBeVisible();
  await page.locator('#add').click(); await expect(page.locator('#guide')).toBeVisible();
  await page.keyboard.press('Escape'); await expect(page.locator('#guide')).not.toBeVisible();
  await page.goto('/#q=not-a-real-question'); await expect(page.locator('#reader')).toContainText('这道题暂时不存在');
});
test('手机布局无横向溢出', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 }); await page.goto('/');
  await expect(page.locator('.question-card').first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
  await page.locator('.question-copy h3 a').first().click();
  await expect(page.locator('.prose')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
test('GitHub Pages 仓库子路径与直接刷新', async ({ page }) => {
  await page.goto('http://127.0.0.1:4176/interview-notes/');
  await expect(page.locator('.question-card').first()).toBeVisible();
  await page.screenshot({ path: 'test-results/desktop.png', fullPage: true });
  await page.locator('.question-copy h3 a').first().click(); await page.reload();
  await expect(page.locator('.prose')).toBeVisible();
  await page.screenshot({ path: 'test-results/reader.png', fullPage: true });
});
