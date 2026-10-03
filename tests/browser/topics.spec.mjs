import { test, expect } from '@playwright/test';
test('训练框架专题范围、旧地址与手机布局', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-category="systems"]').click();
  await expect(page.locator('.question-card')).toHaveCount(1);
  await expect(page.locator('.question-card')).toContainText('训练框架怎么选、怎么用');
  await page.goto('/#q=rollout-systems');
  await expect(page.locator('.prose')).toBeVisible();
  await expect(page).toHaveURL(/q=training-inference-frameworks/);
  await expect(page.locator('.prose')).toContainText('Q08');
  await expect(page.locator('.prose')).not.toContainText('Q60');
  await expect(page.locator('.prose h2')).toHaveCount(9);
  await expect(page.locator('.prose img')).toHaveCount(2);
  await expect(page.locator('.prose .katex-error')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.toc button').filter({ hasText: '横向对比与选型' }).click();
  await expect(page.locator('h2').filter({ hasText: '横向对比与选型' })).toBeInViewport();
});
