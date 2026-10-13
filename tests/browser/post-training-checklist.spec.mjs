import { test, expect } from '@playwright/test';
test('222项地图连接合并原入口和具体项目，下载保持完整', async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/#q=post-training-checklist-map');
  await expect(page.locator('.reader-header h1')).toContainText('222');
  await expect(page.locator('.prose')).toContainText('Q08');
  await page.locator('.prose a[href="#q=lora-finetuning"]').first().click();
  await expect(page.locator('.prose')).toContainText('QLoRA');
  await expect(page.locator('.prose')).toContainText('Q16');
  const mapping = await (await page.request.get('/assets/learning/post-training-checklist-map/mapping.json')).json();
  expect(mapping.items).toHaveLength(222);
  for (const id of ['escd-project-interview','agentic-turn-framework','agentic-turn-innovation']) {
    await page.goto('/#q=' + id);
    await expect(page.locator('.prose')).toContainText('Q12');
    await expect(page.locator('.prose .katex-error')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), id).toBe(true);
  }
  const result = await (await page.request.get('/assets/learning/post-training-project-lab-results.json')).json();
  expect(result.single_forward_ratio).toBe(1);
  expect(result.single_forward_logprob_gradient).toBeCloseTo(-.7,12);
});
