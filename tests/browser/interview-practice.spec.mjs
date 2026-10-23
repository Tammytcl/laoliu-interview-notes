import { test, expect } from '@playwright/test';

test('后训练面试入口、20个独立题页、来源和前后题导航', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: /后训练面试 · \d+ 题/ }).click();
  await expect(page.locator('.reader-header h1')).toHaveText('后训练面试题库：公司题型、回答与项目追问');
  const data = await (await page.request.get('/data.json')).json();
  const items = data.questions.filter(q => q.category === 'interview');
  expect(items.length).toBeGreaterThanOrEqual(20);
  for (const q of items) await expect(page.locator(`.prose a[href="#q=${q.id}"]`).first()).toBeVisible();
  await page.locator('.prose a[href="#q=pt-method-selection"]').first().click();
  await expect(page.locator('.reader-header h1')).toContainText('Q01');
  await expect(page.getByRole('heading', { name: '60–90 秒回答', exact: true })).toBeVisible();
  await page.getByRole('link', { name: '下一题', exact: true }).click();
  await expect(page.locator('.reader-header h1')).toContainText('Q02');
  await page.reload();
  await expect(page.getByRole('heading', { name: '题源与延伸', exact: true })).toBeVisible();
  await expect(page.locator('.katex-error')).toHaveCount(0);
});

test('按公司筛选练习，手机阅读公式和宽表没有横向溢出', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.locator('[data-category="interview"]').click();
  await page.locator('#tag').selectOption('字节');
  const data = await (await page.request.get('/data.json')).json();
  const expected = data.questions.filter(q => q.category === 'interview' && q.tags.includes('字节'));
  await expect(page.locator('.question-card')).toHaveCount(expected.length);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  for (const id of ['post-training-interview-practice', 'pt-clip-kl', 'pt-memory-parallel', 'pt-escd-events']) {
    await page.goto(`/#q=${id}`);
    await expect(page.locator('.prose')).toBeVisible();
    await expect(page.locator('.katex-error')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id).toBe(true);
  }
  await page.screenshot({ path: 'test-results/interview-mobile.png', fullPage: true });
});
