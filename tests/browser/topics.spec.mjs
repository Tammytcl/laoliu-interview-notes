import { test, expect } from '@playwright/test';
test('训练框架专题范围、旧地址与手机布局', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-category="systems"]').click();
  const data = await (await page.request.get('/data.json')).json();
  await expect(page.locator('.question-card')).toHaveCount(data.questions.filter(q => q.category === 'systems').length);
  await expect(page.locator('.question-card').filter({ hasText: '训练框架怎么选、怎么用' })).toHaveCount(1);
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

test('框架入口连接五篇基础复习，来源与CPU练习可读取', async ({ page }) => {
  await page.goto('/#q=training-inference-frameworks');
  await page.locator('.prose a[href="#q=infra-training-step"]').click();
  await expect(page.locator('.reader-header h1')).toContainText('01 · 训练一步');
  const response = await page.request.get('/research/infra-training-step-sources.json');
  expect(response.status()).toBe(200);
  expect((await response.json()).topicId).toBe('infra-training-step');
  const arithmetic = await page.request.get('/assets/infra/infra-foundations-check-results.json');
  expect((await arithmetic.json()).gradient.full).toBe(-22.5);
  await page.locator('.prose a[href="#q=infra-gpu-performance"]').click();
  await expect(page.locator('.reader-header h1')).toContainText('02 · GPU');
});

test('五篇基础复习在手机上保留公式、问答、图与独立路由', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const id of ['infra-training-step', 'infra-gpu-performance', 'infra-distributed-parallelism', 'infra-inference-engine', 'infra-rl-pipeline']) {
    await page.goto(`/#q=${id}`);
    await expect(page.locator('.prose')).toContainText('Q08');
    await expect(page.locator('.prose .katex').first()).toBeVisible();
    await expect(page.locator('.prose .katex-error')).toHaveCount(0);
    await expect(page.locator('.prose img').first()).toHaveAttribute('alt', /示意/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id).toBe(true);
    await page.reload();
    await expect(page.locator('.prose')).toContainText('CPU验算脚本');
  }
});

test('权重、ZeRO和训练后端三篇在手机上展示容量与独立问答', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  for (const id of ['infra-checkpoint-formats', 'infra-zero-deepspeed', 'infra-training-backends']) {
    await page.goto(`/#q=${id}`);
    await expect(page.locator('.prose')).toContainText('Q08');
    await expect(page.locator('.prose .katex-error')).toHaveCount(0);
    await expect(page.locator('.prose img').first()).toHaveAttribute('alt', /示意/);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id).toBe(true);
    await page.reload();
    await expect(page.locator('.reader-header h1')).toBeVisible();
  }
  const response = await page.request.get('/assets/infra/infra-checkpoint-check-results.json');
  const result = await response.json();
  expect(result.dcp_reshard.max_error).toBe(0);
  expect(result.capacity_1T.find(item => item.encoding === 'BF16/FP16').TB).toBe(2);
});
