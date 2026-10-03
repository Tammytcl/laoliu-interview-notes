import { test, expect } from '@playwright/test';
test('Infra 列表只显示总览，旧地址跳转，图表与60题在手机可读', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-category="systems"]').click();
  await expect(page.locator('.question-card')).toHaveCount(1);
  await expect(page.locator('.question-card')).toContainText('训推框架梳理');
  await page.goto('/#q=rollout-systems');
  await expect(page.locator('.prose')).toBeVisible();
  await expect(page).toHaveURL(/q=training-inference-frameworks/);
  await expect(page.locator('.prose')).toContainText('Q60');
  await expect(page.locator('.prose img')).toHaveCount(3);
  await expect(page.locator('.prose .katex-error')).toHaveCount(0);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.toc button').filter({ hasText: '横向对比与选型' }).click();
  await expect(page.locator('h2').filter({ hasText: '横向对比与选型' })).toBeInViewport();
});
