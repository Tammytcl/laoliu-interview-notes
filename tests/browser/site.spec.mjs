import { test, expect } from '@playwright/test';

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
test('详情路由与目录保留，题目不再包含收藏或进度操作', async ({ page }) => {
  await page.goto('/'); await page.locator('.question-copy h3 a').first().click();
  await expect(page.locator('.prose')).toBeVisible();
  await expect(page.locator('[data-status], [data-star], #progress, #export, #import, #status-tabs')).toHaveCount(0);
  await page.reload(); await expect(page.locator('.prose')).toBeVisible();
  await page.locator('.toc button').first().click(); await expect(page.locator('#reader')).toBeVisible();
  await page.locator('.back-link').click(); await expect(page.locator('.question-card').first()).toBeVisible();
});
test('旧浏览器进度不影响内容检索，页面只显示内容统计', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => localStorage.setItem(`interview-notes:v1:${location.pathname}`, JSON.stringify({ 'ddpm-denoising': { status: 'mastered', starred: true } })));
  await page.reload();
  const data = await (await page.request.get('/data.json')).json();
  await expect(page.locator('.question-card')).toHaveCount(data.questions.length);
  await expect(page.locator('#stat-total')).toHaveText(String(data.questions.length).padStart(2, '0'));
  await expect(page.locator('#stat-papers')).toHaveText(String(data.papers.length).padStart(2, '0'));
  await expect(page.locator('[data-star], [data-status]')).toHaveCount(0);
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
